# SPDX-License-Identifier: MIT
"""The dense token layout on the numpy fake: at the synthetic shape the lazy diagonal table against
``DiagLinear.slot_vectors`` and ``apply_diag`` against upstream's ``diag_apply`` (op stream and values);
at the real shape (2^16 ring: N = 32,768 slots, d = 1,024) 32 conversations per ciphertext (real) and
64 as real/imaginary pairs, through one 1024 x 1024 block, a tiled 1024 -> 4096 -> 1024 pair, and the
norm's per-conversation statistic."""

import numpy as np
import pytest

from perseus.examples.fhe_ssm import filling
from perseus.examples.fhe_ssm.fakes import make_runtime_fake
from perseus.impl import fake, prefill
from perseus.impl.layout import Dims
from perseus.impl.ops import FheOps
from perseus.impl.poly import LanePackedOps
from perseus.impl.rt import Rt

N, D, E = 32768, 1024, 4096


def session(N_=N, d=D, e=E, complex_payload=False):
    steps = filling.rotation_steps(N_, d)
    fhe, inf = make_runtime_fake(
        N=N_,
        hid=d,
        dim=d,
        H=1,
        H_real=1,
        E=e,
        E_real=e,
        rot_steps=steps | {N_ - s for s in steps},
        complex_payload=complex_payload,
    )
    ops = FheOps(inf, fake.core, fhe.unit)
    return fhe, inf, ops, Rt(ops, Dims.from_inf(inf))


def rows(n, seed, d=D):
    return 0.3 * np.random.default_rng(seed).standard_normal((n, d))


def weight(seed, shape=(D, D)):
    return np.random.default_rng(seed).standard_normal(shape) / np.sqrt(shape[0])


# ── the synthetic shape: the lazy table and the apply against upstream ───────────────────────────


@pytest.mark.parametrize("shape", [(32, 32), (32, 128), (128, 32)])
def test_lazy_diagonals_equal_diag_linear_slot_vectors(shape):
    W = weight(1, shape)
    lazy = filling.LazyDiagonals(W, 1024, 32)
    want = prefill.DiagLinear.slot_vectors(W, 1024, 32)
    assert lazy.shape == want.shape
    np.testing.assert_array_equal(lazy.dense(), want)
    for m in range(lazy.n_out):
        for gg in range(lazy.g):
            np.testing.assert_array_equal(lazy[m, gg], want[m, gg])


@pytest.mark.parametrize(("n_in", "n_out"), [(1, 1), (1, 4), (4, 1)])
def test_apply_diag_at_hint_2_is_upstreams_diag_apply(n_in, n_out):
    """Same op stream (the fake's call list) and the same numbers as ``prefill.diag_apply`` on the
    same inputs, with the inputs deep enough that the hint fires."""
    d = 32
    W = weight(2, (n_in * d, n_out * d))
    X = rows(64, 3, n_in * d)
    outs = []
    for which in ("upstream", "port"):
        fhe, _inf, _ops, rt = session(1024, d, 4 * d, complex_payload=True)
        xs = [fake.FakeCt(filling.pack_pairs(X[:, n * d : (n + 1) * d], 1024), 42, 2) for n in range(n_in)]
        if which == "upstream":
            lin = prefill.DiagLinear.encode(W, 1024, 32)
            (ys,) = prefill.diag_apply(rt, xs, [lin])
        else:
            (ys,) = filling.apply_diag(rt, xs, [filling.encode_diag(W, 1024, 32)], hint=2)
        outs.append((list(fhe.calls), [(y.vec.copy(), y.level, y.noise_deg) for y in ys], fhe.n_bootstraps))
    assert outs[0][0] == outs[1][0]
    assert outs[0][2] == outs[1][2] == n_in  # every input hinted at headroom(2) from effective 44
    for (ya, la, da), (yb, lb, db) in zip(outs[0][1], outs[1][1]):
        np.testing.assert_array_equal(ya, yb)
        assert (la, da) == (lb, db)
    got = np.concatenate([filling.unpack_pairs(y, d, 64) for y, _, _ in outs[1][1]], axis=1)
    np.testing.assert_allclose(got, X @ W, atol=1e-10)


def test_hint_1_lets_an_operand_at_the_threshold_through():
    """The fork's rule: an input at effective 44 passes and its one product sits at the ceiling."""
    d = 32
    for hint, fired in ((2, 1), (1, 0)):
        fhe, _inf, _ops, rt = session(1024, d, 4 * d, complex_payload=True)
        x = fake.FakeCt(filling.pack_pairs(rows(64, 4, d), 1024), 42, 2)
        (ys,) = filling.apply_diag(rt, [x], [filling.encode_diag(weight(5, (d, d)), 1024, 32)], hint=hint)
        assert fhe.n_bootstraps == fired
        assert fhe.n_reactive == 0
        assert rt.ops.lvl(ys[0]) == (36 + 2 if fired else 46)


# ── the model's shape ──────────────────────────────────────────────────────────────────────────


def test_tile_params_mirror_diag_linear_split():
    assert filling.tile_params(N, D) == (32, 32, 32)  # t, b, g
    assert filling.tile_params(N, E) == (8, 64, 64)  # one 4096-wide piece: 8 lanes only
    assert filling.rotation_steps(N, D) == set(prefill.diag_rot_steps(Dims(N, D, D, 1, 1, E, E))) | {
        32 << k for k in range(10)
    }


@pytest.mark.slow_shape
def test_one_block_32_conversations_exact_and_counted():
    fhe, _inf, _ops, rt = session()
    X, W = rows(32, 1), weight(2)
    (ys,) = filling.apply_diag(rt, [fake.FakeCt(filling.pack(X, N), 34)], [filling.encode_diag(W, N, 32)])
    err = np.abs(filling.unpack(ys[0].vec, D, 32) - X @ W).max()
    assert err < 1e-10
    assert sum(c.startswith("rotate") for c in fhe.calls) == 31 + 31  # babies + giants
    assert fhe.calls.count("mult_pt") == D  # one product per diagonal


@pytest.mark.slow_shape
def test_one_block_64_conversations_as_pairs_exact():
    fhe, _inf, _ops, rt = session(complex_payload=True)
    X, W = rows(64, 3), weight(4)
    (ys,) = filling.apply_diag(
        rt, [fake.FakeCt(filling.pack_pairs(X, N), 34)], [filling.encode_diag(W, N, 32)]
    )
    err = np.abs(filling.unpack_pairs(ys[0].vec, D, 64) - X @ W).max()
    assert err < 1e-10
    assert fhe.calls.count("mult_pt") == D  # 64 lanes, same 1,024 products


@pytest.mark.slow_shape
@pytest.mark.parametrize("pairs", [False, True])
def test_tiled_channel_mix_shapes_exact(pairs):
    _fhe, _inf, _ops, rt = session(complex_payload=pairs)
    n = 64 if pairs else 32
    X, Wk, Wv = rows(n, 5), weight(6, (D, E)), weight(7, (E, D))
    enc = filling.pack_pairs if pairs else filling.pack
    dec = filling.unpack_pairs if pairs else filling.unpack
    (hidden,) = filling.apply_diag(rt, [fake.FakeCt(enc(X, N), 34)], [filling.encode_diag(Wk, N, 32)])
    assert len(hidden) == 4  # 4,096 channels = 4 ciphertexts
    H = np.concatenate([dec(h.vec, D, n) for h in hidden], axis=1)
    assert np.abs(H - X @ Wk).max() < 1e-10
    chunks = [fake.FakeCt(enc(H[:, c * D : (c + 1) * D], N), 34) for c in range(4)]
    (out,) = filling.apply_diag(rt, chunks, [filling.encode_diag(Wv, N, 32)])
    assert np.abs(dec(out[0].vec, D, n) - H @ Wv).max() < 1e-10


@pytest.mark.slow_shape
def test_norm_statistic_32_conversations():
    fhe, _inf, ops, _rt = session()
    X = rows(32, 8)
    v = filling.per_conversation_sum(ops, ops.square(fake.FakeCt(filling.pack(X, N), 34)), N, D)
    want = (X**2).sum(axis=1)
    got = v.vec.reshape(D, 32)  # slot i*t + k -> [i, k]
    assert np.abs(got - want[None, :]).max() < 1e-10  # every channel slot of lane k
    assert sum(c.startswith("rotate") for c in fhe.calls) == 10  # log2(1024)


@pytest.mark.slow_shape
def test_norm_statistic_64_pairs_through_lane_packed_ops():
    _fhe, _inf, ops, _rt = session(complex_payload=True)
    X = rows(64, 9)
    lops = LanePackedOps(ops)
    sq = lops.square(fake.FakeCt(filling.pack_pairs(X, N), 34))  # a^2 + i b^2
    v = filling.per_conversation_sum(ops, sq, N, D)
    want = (X**2).sum(axis=1)
    got = v.vec.reshape(D, 32)
    assert np.abs(got.real - want[None, :32]).max() < 1e-9
    assert np.abs(got.imag - want[None, 32:]).max() < 1e-9
