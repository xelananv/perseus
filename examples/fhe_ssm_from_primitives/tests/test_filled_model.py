# SPDX-License-Identifier: MIT
"""The filled layout on the numpy fake: 32 conversations per ciphertext (real) and 64 as
real/imaginary pairs (split and lane routes).

Every piece is checked two independent ways, on edge cases: against the same kernels on
NumpyOps (``PlainGeometry`` / ``ref.MirrorModel``: a difference is a layout, packing or split
error) and against a closed form written directly in numpy (the math: ``x @ W``, the textbook
diagonal method, the polynomials, the trainer's forward). The edge cases are one conversation,
the capacity, one past the real axis, conversations at both ends of the norm's window, values
with a zero real or imaginary part, and an operand with its rescale pending."""

import contextlib

import numpy as np
import pytest
from examples.fhe_ssm_from_primitives import block, filling, geometry, pairs, ref, weights
from examples.fhe_ssm_from_primitives.model import FheSsmModel
from examples.fhe_ssm_from_primitives.runtime_fake import make_runtime_fake
from examples.fhe_ssm_from_primitives.synthetic import synthetic

from perseus.impl import fake
from perseus.impl.ops import FheOps, NumpyOps
from perseus.impl.poly import LanePackedOps

D, DFF, N = 32, 128, 1024  # t = 32 conversations per axis; tiles: s = 4, G = 8; 4 d_ffn tiles
T = N // D
S, G = 4, 8
ARMS = [("pairs", "split"), ("pairs", "lane"), ("real", "split")]
PRODUCTS = ("mult_cc", "mult_sc", "mult_pt", "square", "mult_const")


def session(complex_payload=True, **kw):
    fhe, inf = make_runtime_fake(
        N=N,
        hid=D,
        dim=D,
        H=1,
        H_real=1,
        E=DFF,
        E_real=DFF,
        rot_steps=filling.rotation_steps(N, D),
        complex_payload=complex_payload,
        **kw,
    )
    return fhe, inf, FheOps(inf, fake.core, fhe.unit)


def make(params, payload="pairs", route="split", **kw):
    fhe, inf = make_runtime_fake(
        N=N,
        hid=params.d,
        dim=params.d,
        H=1,
        H_real=1,
        E=params.d_ffn,
        E_real=params.d_ffn,
        rot_steps=(),
        complex_payload=payload == "pairs",
        **kw,
    )
    model = FheSsmModel(
        inf, params, core=fake.core, unit=fhe.unit, payload=payload, pair_route=route
    )
    fhe.loaded_rot_steps = sorted(model.geo.rotation_steps())
    fhe._rot = set(fhe.loaded_rot_steps)
    return fhe, inf, model


def window_rows(n, seed, d=D):
    """Rows whose mean squares sit inside the seeds' window [2e-4, 2e-3], the first and the
    last near its two ends."""
    rows = 0.03 * np.random.default_rng(seed).standard_normal((n, d))
    rows /= np.sqrt(np.mean(rows**2, axis=1, keepdims=True))
    ms = np.geomspace(2.4e-4, 1.8e-3, n) if n > 1 else np.array([9e-4])
    return rows * np.sqrt(ms)[:, None]


def naive_diagonal(xv, W, n_slots):
    """``x @ W`` in the filled layout by the textbook diagonal method (no baby/giant split):
    the input rotated by ``k t`` times diagonal ``k`` (``W[(j + k) mod d, j]`` on channel j's
    lanes), summed over ``k = 0 .. d-1``."""
    d = W.shape[0]
    t = n_slots // d
    j = np.arange(d)
    y = np.zeros_like(xv)
    for k in range(d):
        y = y + np.roll(xv, -k * t) * np.repeat(W[(j + k) % d, j], t)
    return y


def layout_vector(rows, pair):
    """The slot vector the layout prescribes, written as a reshape: slot ``i t + k`` is element
    ``(i, k)`` of a ``(d, t)`` array, i.e. the transpose of the rows."""
    if not pair:
        return rows.T.reshape(-1)
    return rows[:T].T.reshape(-1) + 1j * rows[T:].T.reshape(-1)


# ── the pair algebra ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("kind", ["complex", "real", "imaginary", "zero"])
def test_split_gives_twice_each_axis_without_a_level(kind):
    fhe, _inf, ops = session()
    rng = np.random.default_rng(1)
    re, im = rng.standard_normal(N), rng.standard_normal(N)
    v = {"complex": re + 1j * im, "real": re + 0j, "imaginary": 1j * im, "zero": 0j * re}[kind]
    x = fake.FakeCt(v, 37, 2)  # mid-chain, rescale pending
    a, b = pairs.split(ops, x)
    np.testing.assert_array_equal(a.vec, 2 * v.real)  # oracle 1: numpy, exactly
    np.testing.assert_array_equal(b.vec, 2 * v.imag)
    np.testing.assert_array_equal(pairs.repack(ops, a, b).vec, 2 * v)  # oracle 2: round trip
    assert (a.level, a.noise_deg, b.level, b.noise_deg) == (37, 2, 37, 2)
    assert fhe.calls.count("conjugate") == 1
    assert not [c for c in fhe.calls if c in PRODUCTS]


def test_lane_route_adds_a_plaintext_to_both_axes():
    _fhe, _inf, ops = session()
    rng = np.random.default_rng(2)
    re, im, v = rng.standard_normal(N), rng.standard_normal(N), rng.standard_normal(N)
    x = fake.FakeCt(re + 1j * im, 34)
    y = pairs.PairLaneOps(ops).add_pt(x, v, key="t.v")
    np.testing.assert_allclose(y.vec, x.vec + v * (1 + 1j), rtol=0, atol=1e-15)  # oracle 1
    np.testing.assert_allclose(y.vec.real, re + v, rtol=0, atol=1e-15)  # oracle 2: per axis
    np.testing.assert_allclose(y.vec.imag, im + v, rtol=0, atol=1e-15)
    # what the override exists for: Perseus's LanePackedOps forwards add_pt to the real axis only
    stock = LanePackedOps(ops).add_pt(x, v)
    np.testing.assert_array_equal(stock.vec.imag, im)


def test_pairs_refuse_a_session_without_complex_slots():
    params, *_ = synthetic(L=1)
    _fhe, inf = make_runtime_fake(
        N=N, hid=D, dim=D, H=1, H_real=1, E=DFF, E_real=DFF, rot_steps=(), complex_payload=False
    )
    with pytest.raises(ValueError, match="complex"):
        FheSsmModel(inf, params, core=fake.core, payload="pairs")
    real = FheSsmModel(inf, params, core=fake.core, payload="real")
    assert isinstance(real.geo, geometry.FilledGeometry)
    with pytest.raises(ValueError, match="route"):
        geometry.PairGeometry(real.geo, "x")


# ── packing ───────────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("n", [1, 2, T - 1, T])
def test_park_fills_with_conversation_zero(n):
    rows = np.random.default_rng(n).standard_normal((n, D))
    got = geometry.park(rows, T)
    assert got.shape == (T, D)
    np.testing.assert_array_equal(got[:n], rows)
    assert (got[n:] == rows[0]).all()


@pytest.mark.parametrize("shape", [(0, D), (T + 1, D), (D,), (2, D + 1)])
def test_rows_outside_the_layout_are_refused(shape):
    params, *_ = synthetic(L=1)
    _fhe, _inf, model = make(params, "real")
    with pytest.raises(ValueError, match="rows"):
        model.geo.encode_rows(np.zeros(shape))


@pytest.mark.parametrize(
    ("payload", "n"),
    [("real", 1), ("real", T), ("pairs", 1), ("pairs", T), ("pairs", T + 1), ("pairs", 2 * T)],
)
def test_rows_round_trip_and_sit_where_the_layout_says(payload, n):
    params, *_ = synthetic(L=1)
    _fhe, _inf, model = make(params, payload)
    rows = np.random.default_rng(n).standard_normal((n, D))
    ct = model.geo.encode_rows(rows)
    np.testing.assert_array_equal(model.geo.decode_rows(ct, n), rows)  # oracle 1: round trip
    full = geometry.park(rows, model.capacity)
    np.testing.assert_array_equal(ct.vec, layout_vector(full, payload == "pairs"))  # oracle 2
    assert (ct.level, ct.noise_deg) == (34, 1)  # a fresh encryption at the landing level


# ── linears ───────────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("pair", [False, True])
def test_up_and_down_match_numpy_and_the_naive_diagonal_method(pair):
    fhe, _inf, ops = session(complex_payload=pair)
    rng = np.random.default_rng(3)
    n = 2 * T if pair else T
    X = rng.standard_normal((n, D))
    Wk, Wr = rng.standard_normal((D, DFF)), rng.standard_normal((D, DFF))
    Wv = rng.standard_normal((DFF, D))
    xv = layout_vector(X, pair)
    ks, rs = filling.linear_multi(
        ops,
        [fake.FakeCt(xv, 34)],
        [filling.encode_tiled(Wk, N, D), filling.encode_tiled(Wr, N, D)],
    )
    for W, outs in ((Wk, ks), (Wr, rs)):
        assert len(outs) == DFF // D
        for j, y in enumerate(outs):
            Wj = W[:, j * D : (j + 1) * D]
            np.testing.assert_allclose(y.vec, layout_vector(X @ Wj, pair), atol=1e-12)
            np.testing.assert_allclose(y.vec, naive_diagonal(xv, Wj, N), atol=1e-12)
    # sharing: one set of baby rotations for both weights' 8 output tiles
    assert sum(c.startswith("rotate") for c in fhe.calls) == (S - 1) + 2 * 4 * (G - 1)
    assert fhe.calls.count("mult_pt") == 2 * 4 * D
    fhe.calls.clear()
    H = rng.standard_normal((n, DFF))
    hv = [layout_vector(H[:, j * D : (j + 1) * D], pair) for j in range(4)]
    y = filling.linear_tiled(ops, [fake.FakeCt(v, 34) for v in hv], filling.encode_tiled(Wv, N, D))
    assert len(y) == 1
    want = sum(naive_diagonal(hv[j], Wv[j * D : (j + 1) * D], N) for j in range(4))
    np.testing.assert_allclose(y[0].vec, layout_vector(H @ Wv, pair), atol=1e-12)
    np.testing.assert_allclose(y[0].vec, want, atol=1e-12)
    # sharing: one giant rotation per group, summed over the 4 input chunks
    assert sum(c.startswith("rotate") for c in fhe.calls) == 4 * (S - 1) + (G - 1)
    assert fhe.calls.count("mult_pt") == 4 * D


def test_mismatched_tiles_are_refused():
    _fhe, _inf, ops = session()
    x = fake.FakeCt(np.zeros(N), 34)
    with pytest.raises(ValueError, match="input chunks"):
        filling.linear_tiled(ops, [x, x], filling.encode_tiled(np.zeros((D, DFF)), N, D))
    with pytest.raises(ValueError, match="tile"):
        filling.encode_tiled(np.zeros((D, D + 1)), N, D)


# ── sections: norm and gates ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(("payload", "route"), ARMS)
def test_norm_section_matches_mirror_and_closed_form(payload, route):
    params, *_ = synthetic(L=1)
    _fhe, _inf, model = make(params, payload, route)
    n = model.capacity
    rows = window_rows(n, 4)
    site = params.block(0).norm_tm
    y = block.norm(model.geo, model.geo.encode_rows(rows), site, "t.norm", "perseus")
    got = model.geo.decode_rows(y, n)
    mirror = block.norm(geometry.PlainGeometry(NumpyOps()), rows, site, "t.norm", "perseus")
    np.testing.assert_allclose(got, mirror, rtol=1e-9, atol=1e-12)  # oracle 1
    np.testing.assert_allclose(got, ref.rmsnorm_exact(rows, 1.0), rtol=1e-6)  # oracle 2


def test_split_spends_the_real_depth_and_lane_spends_more():
    """Levels with no ceiling (so nothing refreshes): a split section costs what the real
    kernel costs; the lane route pays a level per ciphertext product for its 1/4."""
    params, *_ = synthetic(L=1)
    site, depth = params.block(0).norm_tm, {}
    for payload, route in ARMS:
        _fhe, _inf, model = make(params, payload, route, limit=10**6)
        rows = window_rows(model.capacity, 5)
        y = block.norm(model.geo, model.geo.encode_rows(rows), site, "t", "perseus")
        depth[payload, route] = model.rt.ops.lvl(y) - 34
    assert depth["pairs", "split"] == depth["real", "split"]
    assert depth["pairs", "lane"] > depth["real", "split"]


@pytest.mark.parametrize(("payload", "route"), ARMS)
def test_gate_sections_match_mirror_and_closed_form(payload, route):
    params, *_ = synthetic(L=1, seed=6)
    _fhe, _inf, model = make(params, payload, route)
    geo, p, n = model.geo, params.block(0), model.capacity
    w = weights.encode_block(p, geo, "b0.")
    wm = weights.encode_block(p, geometry.PlainGeometry(NumpyOps()), "b0.")
    rng = np.random.default_rng(7)
    Z = 0.3 * rng.standard_normal((n, D))
    z = geo.encode_rows(Z * geo.split_gain)  # what c, dd pre-halved make on a split pair
    g = geo.section(lambda o, _g, zs: [block.tm_gate(o, zs[0], w)], [z])[0]
    got = geo.decode_rows(g, n)
    np.testing.assert_allclose(got, block.tm_gate(NumpyOps(), Z, wm), rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(got, Z * (p.p0 + p.p1 * Z + p.p2 * Z**2), rtol=1e-9, atol=1e-12)
    K, R = 0.3 * rng.standard_normal((n, DFF)), 0.3 * rng.standard_normal((n, DFF))
    tiles = [
        geo.encode_rows(M[:, j * D : (j + 1) * D] * geo.split_gain)
        for M in (K, R)
        for j in range(4)
    ]
    out = geo.section(
        lambda o, _g, xs: [block.cm_gate(o, xs[j], xs[4 + j], w, j) for j in range(4)], tiles
    )
    got = np.concatenate([geo.decode_rows(o, n) for o in out], axis=1)
    mirror = block.cm_gate(NumpyOps(), K, R, wm)
    closed = (p.q0 + p.q1 * K + p.q2 * K**2 + p.q3 * K**3) * (p.r0 + p.r1 * R + p.r2 * R**2)
    np.testing.assert_allclose(got, mirror, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(got, closed, rtol=1e-9, atol=1e-12)


# ── the model ─────────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(("payload", "route"), ARMS)
def test_ticks_match_mirror_and_exact(payload, route):
    params, *_ = synthetic(L=2, seed=3)
    _fhe, _inf, model = make(params, payload, route)
    assert model.rotation_audit() == []
    n = model.capacity
    mirror, exact = ref.MirrorModel(params, n), ref.ExactModel(params, n)
    model.start()
    rng = np.random.default_rng(8)
    for pos in range(3):
        rows = params.emb[rng.integers(0, params.vocab, n)].astype(np.float64)
        got = model.step(rows, pos)["out"]
        assert got.shape == (n, D)
        assert ref.rel_err_rms(got, mirror.step(rows)) < 1e-9  # oracle 1
        assert ref.rel_err_rms(got, exact.step(rows)) < 1e-4  # oracle 2
        assert model.levels() == [(34, 34)] * 2


def test_routes_agree_on_the_same_conversations():
    """Three circuits, one answer: 64 pairs split, 64 pairs lane-wise, and the same 64
    conversations as two real ciphertexts of 32."""
    params, *_ = synthetic(L=2, seed=9)
    rows = params.emb[np.random.default_rng(10).integers(0, params.vocab, 2 * T)].astype(float)
    outs = {}
    for name, payload, route, part in (
        ("split", "pairs", "split", slice(None)),
        ("lane", "pairs", "lane", slice(None)),
        ("real_a", "real", "split", slice(0, T)),
        ("real_b", "real", "split", slice(T, 2 * T)),
    ):
        _fhe, _inf, model = make(params, payload, route)
        model.start()
        outs[name] = [model.step(rows[part], pos)["out"] for pos in range(2)]
    for pos in range(2):
        real = np.concatenate([outs["real_a"][pos], outs["real_b"][pos]])
        assert ref.rel_err_rms(outs["split"][pos], outs["lane"][pos]) < 1e-9
        assert ref.rel_err_rms(outs["split"][pos], real) < 1e-9


@pytest.mark.parametrize("n", [1, T + 1, 2 * T - 1])
def test_parked_lanes_leave_the_conversations_alone(n):
    params, *_ = synthetic(L=2, seed=11)
    _fhe, _inf, model = make(params, "pairs", "split")
    mirror, exact = ref.MirrorModel(params, n), ref.ExactModel(params, n)
    model.start()
    rng = np.random.default_rng(n)
    for pos in range(2):
        rows = params.emb[rng.integers(0, params.vocab, n)].astype(np.float64)
        got = model.step(rows, pos)["out"]
        assert got.shape == (n, D)
        assert ref.rel_err_rms(got, mirror.step(rows)) < 1e-9
        assert ref.rel_err_rms(got, exact.step(rows)) < 1e-4


@contextlib.contextmanager
def bootstrap_inputs(fhe):
    """The largest slot magnitude of every ciphertext refreshed inside the block; the fake's
    own bootstrap is restored on exit."""
    seen = []
    shadowed = "bootstrap" in vars(fhe)
    inner = fhe.bootstrap

    def recording(a):
        seen.append(float(np.max(np.abs(a.vec))))
        inner(a)

    fhe.bootstrap = recording
    try:
        yield seen
    finally:
        if shadowed:
            fhe.bootstrap = inner
        else:
            del fhe.bootstrap


@pytest.mark.parametrize("n", [1, T + 1])
def test_parking_keeps_every_bootstrap_input_in_range(n, monkeypatch):
    """A bootstrap needs every slot of its ciphertext in range, the unused lanes included. A
    lane left at zero enters each norm far below the Newton seed's window and its
    intermediates grow (measured here: 8x the full batch's peak); a parked lane copies
    conversation 0, so a partial batch refreshes what a full one does."""
    params, *_ = synthetic(L=2, seed=15)
    rows = params.emb[np.random.default_rng(16).integers(0, params.vocab, 2 * T)].astype(float)

    def peak(k):
        fhe, _inf, model = make(params, "pairs", "split")
        model.start()
        with bootstrap_inputs(fhe) as seen:
            for pos in range(2):
                model.step(rows[:k], pos)
        assert "bootstrap" not in vars(fhe)
        return max(seen)

    full, parked = peak(2 * T), peak(n)
    assert parked <= 1.5 * full
    monkeypatch.setattr(
        geometry,
        "park",
        lambda r, cap: np.concatenate([r, np.zeros((cap - r.shape[0], r.shape[1]))]),
    )
    assert peak(n) > 4 * full  # the hazard parking removes


def test_a_single_row_comes_back_as_a_row():
    params, *_ = synthetic(L=1, seed=12)
    _fhe, _inf, model = make(params, "pairs", "split")
    model.start()
    row = params.emb_row(5)
    got = model.step(row, 0)["out"]
    assert got.shape == (D,)
    assert ref.rel_err_rms(got, ref.MirrorModel(params, 1).step(row[None])[0]) < 1e-9


def test_full_shape_block_on_pairs():
    """One block at the real shape (ring 2^16: 32,768 slots, d = 1,024, d_ffn = 4,096) with 64
    conversations as split pairs: the tiling, the sharing and the sections at the sizes the GPU
    will run."""
    d, dff, n_slots = 1024, 4096, 32768
    params, *_ = synthetic(L=1, d=d, dff=dff, seed=13)
    fhe, inf = make_runtime_fake(
        N=n_slots,
        hid=d,
        dim=d,
        H=1,
        H_real=1,
        E=dff,
        E_real=dff,
        rot_steps=(),
        complex_payload=True,
    )
    model = FheSsmModel(inf, params, core=fake.core, unit=fhe.unit, resident_weights=False)
    fhe.loaded_rot_steps = sorted(model.geo.rotation_steps())
    fhe._rot = set(fhe.loaded_rot_steps)
    assert model.capacity == 64
    rows = params.emb[np.random.default_rng(14).integers(0, params.vocab, 64)].astype(float)
    model.start()
    c0 = len(fhe.calls)
    got = model.step(rows, 0)["out"]
    calls = fhe.calls[c0:]
    assert ref.rel_err_rms(got, ref.MirrorModel(params, 64).step(rows)) < 1e-9
    assert ref.rel_err_rms(got, ref.ExactModel(params, 64).step(rows)) < 1e-4
    _, s, g_ = filling.tile_params(n_slots, d)
    assert (
        calls.count("mult_pt") >= 14 * d
    )  # 14 tiles of 1,024 diagonals, plus the gates' constants
    linear_rotations = 2 * ((s - 1) + (g_ - 1)) + (s - 1) + 8 * (g_ - 1) + 4 * (s - 1) + (g_ - 1)
    norm_rotations = 3 * 2 * 10  # 3 norms (2 + the tail), 2 halves, log2(1024)
    assert sum(c.startswith("rotate") for c in calls) == linear_rotations + norm_rotations
