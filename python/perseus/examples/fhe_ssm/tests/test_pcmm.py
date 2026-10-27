# SPDX-License-Identifier: MIT
"""The PCMM linears (perseus.impl.pcmm's fake path) on the port's layout, at the shape PCMM is cut for
(d = 1,024 at t = 32: one port ciphertext is one PCMM block): every tile shape (1 -> 1, 1 -> 4, 4 -> 1)
against ``x @ W_q`` (the integer weights: bitwise) and ``x @ W`` (the rounding), one block of the
model under pcmm-trunk and pcmm against the quantized mirror (the layout alone) and the closed form
(the rounding), the external refresh count and the rotation keys."""

import numpy as np
import pytest

from perseus.examples.fhe_ssm import fakes, filling, geometry, ref
from perseus.examples.fhe_ssm.model import FheSsmModel
from perseus.examples.fhe_ssm.synthetic import synthetic
from perseus.impl import fake, pcmm
from perseus.impl.layout import Dims
from perseus.impl.ops import FheOps
from perseus.impl.pcmm import PcmmLinear, Shape, from_planes
from perseus.impl.rt import Rt

N, D, E, T = 32768, 1024, 4096, 32

pytestmark = pytest.mark.slow_shape


def session():
    steps = filling.rotation_steps(N, D) | set(pcmm.rot_steps())
    fhe, inf = fakes.make_runtime_fake(
        N=N, hid=D, dim=D, H=1, H_real=1, E=E, E_real=E, rot_steps=steps | {N - s for s in steps},
        complex_payload=True,
    )  # fmt: skip
    ops = FheOps(inf, fake.core, fhe.unit)
    return fhe, inf, ops, Rt(ops, Dims.from_inf(inf))


def integer_weights(lin, sh):
    """The integer matrix (n_in B, n_out B) PCMM multiplies by, in channel order."""
    inv = np.argsort(sh.ch_of_row())
    Wi = from_planes(lin.Wd)
    nch = sh.nch
    out = np.empty_like(Wi)
    for o in range(lin.n_out):
        for c in range(lin.n_in):
            blk = Wi[o * nch : (o + 1) * nch, c * nch : (c + 1) * nch]
            out[o * nch : (o + 1) * nch, c * nch : (c + 1) * nch] = blk[inv[:, None], inv[None, :]]
    return out.T.astype(np.float64)


@pytest.mark.parametrize(("n_in", "n_out", "wbits"), [(1, 1, 16), (1, 4, 16), (4, 1, 16), (1, 1, 24)])
def test_every_tile_shape_is_exact_on_the_integer_weights(n_in, n_out, wbits):
    fhe, _inf, _ops, rt = session()
    rng = np.random.default_rng(n_in * 10 + n_out)
    W = rng.standard_normal((n_in * D, n_out * D)) / np.sqrt(n_in * D)
    X = rng.standard_normal((2 * T, n_in * D))
    xs = [fake.FakeCt(filling.pack_pairs(X[:, c * D : (c + 1) * D], N), 34, 2) for c in range(n_in)]
    lin = PcmmLinear.encode(W, T, 1.0, name="t.w", wbits=wbits)
    Wq = integer_weights(lin, Shape(T)) / lin.w_scale
    np.testing.assert_array_equal(Wq, geometry.pcmm_quantize(W, wbits))  # the mirror's rounding rule
    b0 = fhe.n_bootstraps
    outs = pcmm.run(rt, xs, lin)
    got = np.concatenate([filling.unpack_pairs(y.vec, D, 2 * T) for y in outs], axis=1)
    if n_in == 1:
        np.testing.assert_array_equal(got, X @ Wq)  # one block: the same products in the same order
    else:
        assert ref.rel_err_rms(got, X @ Wq) < 1e-14  # four blocks summed: float64 order (the record: 2.2e-15)
    err = ref.rel_err_rms(got, X @ W)
    assert err < (1e-3 if wbits == 16 else 1e-5)
    assert fhe.n_bootstraps - b0 == n_out  # one external refresh per output block
    assert all((y.level, y.noise_deg) == (26, 1) for y in outs)  # the fake's landing (pcmm._N32_LAND)
    rots = [c for c in fhe.calls if c.startswith("rotate")]
    assert rots == [f"rotate{-1}"] * n_in + ["rotate1"] * n_out


@pytest.mark.parametrize(
    ("linear", "wbits", "external", "diag_tiles"), [("pcmm-trunk", 24, 3, 2), ("pcmm", 16, 11, 0)]
)
def test_one_block_under_pcmm_matches_the_quantized_mirror(linear, wbits, external, diag_tiles):
    params, *_ = synthetic(L=1, d=D, dff=E, seed=13)
    fhe, inf = fakes.make_runtime_fake(
        N=N, hid=D, dim=D, H=1, H_real=1, E=E, E_real=E, rot_steps=(), complex_payload=True
    )
    model = FheSsmModel(
        inf, params, core=fake.core, unit=2, resident_weights=False, linear=linear, pcmm_wbits=wbits
    )
    fhe.loaded_rot_steps = sorted(model.geo.rotation_steps())
    fhe._rot = set(fhe.loaded_rot_steps)
    assert {1, N - 1} <= set(fhe.loaded_rot_steps)
    assert model.rotation_audit() == []
    mirror = ref.MirrorModel(params, 64, linear=linear, pcmm_wbits=wbits)
    plain = ref.MirrorModel(params, 64)
    exact = ref.ExactModel(params, 64)
    model.start()
    rng = np.random.default_rng(14)
    for pos in range(2):
        rows = params.emb[rng.integers(0, params.vocab, 64)].astype(float)
        c0 = len(fhe.calls)
        got = model.step(rows, pos)["out"]
        calls = fhe.calls[c0:]
        assert ref.rel_err_rms(got, mirror.step(rows)) < 1e-9  # the layout alone
        q = ref.rel_err_rms(got, plain.step(rows))
        assert 1e-9 < q < (1e-2 if wbits == 16 else 1e-4)  # the rounding, visible against float64 weights
        assert ref.rel_err_rms(got, exact.step(rows)) < 1e-2
        assert model.filled is not None
        assert model.filled.pcmm_stats["outputs"] == external * (pos + 1)
        assert calls.count("mult_pt") == diag_tiles * 4 * D + 6 + 2 * 22 + 1
        assert model.levels() == [(34, 34)]
    assert model.axes()["pcmm_wbits"] == wbits
    assert model.axes()["linear"] == linear
