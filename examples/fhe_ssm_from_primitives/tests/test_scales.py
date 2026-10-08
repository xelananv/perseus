# SPDX-License-Identifier: MIT
"""The power-of-two scale folds (scales.py): exact by construction, so checked exactly.

The folded kernels against the unfolded ones times 2^e (bit for bit: a product by a power of
two is exact in floating point), the whole model with folds against the mirror with the same
folds and against the unfolded closed form, on all three arms and at the extreme exponents;
the exponent rule against a brute-force search; the calibration window against a hand-rolled
loop; files and validation at their edges."""

import dataclasses
import hashlib
import json

import numpy as np
import pytest
from examples.fhe_ssm_from_primitives import block, geometry, ref, scales, weights
from examples.fhe_ssm_from_primitives.model import FheSsmModel
from examples.fhe_ssm_from_primitives.runtime_fake import make_runtime_fake
from examples.fhe_ssm_from_primitives.synthetic import synthetic

from perseus.impl import fake
from perseus.impl.ops import NumpyOps

D, DFF, N = 32, 128, 1024
ARMS = [("pairs", "split"), ("pairs", "lane"), ("real", "split")]


def make(params, payload, route, sc):
    fhe, inf = make_runtime_fake(
        N=N, hid=D, dim=D, H=1, H_real=1, E=DFF, E_real=DFF, rot_steps=(),
        complex_payload=payload == "pairs",
    )  # fmt: skip
    model = FheSsmModel(
        inf, params, core=fake.core, unit=fhe.unit, payload=payload, pair_route=route, scales=sc
    )
    fhe.loaded_rot_steps = sorted(model.geo.rotation_steps())
    fhe._rot = set(fhe.loaded_rot_steps)
    return model


# ── the exponent rule ─────────────────────────────────────────────────────────────────────────


def brute(m, kappa=scales.KAPPA, top=scales.TOP):
    ok = [e for e in range(scales.MAX_EXP + 1) if kappa * 2**e * m <= top]
    return max(ok) if ok else 0


@pytest.mark.parametrize("m", [1.0, 0.5, 16.0, 16.0001, 1e-3, 0.27, 7.8, 1e-9, 1e3])
def test_exponent_matches_brute_force(m):
    assert scales.exponent(m) == brute(m)


@pytest.mark.parametrize("m", [0.0, -1.0, float("nan"), float("inf")])
def test_exponent_of_nothing_is_zero(m):
    assert scales.exponent(m) == 0


def test_exponent_is_capped_and_never_negative():
    assert scales.exponent(1e-30) == scales.MAX_EXP
    assert scales.exponent(1e30) == 0
    rng = np.random.default_rng(0)
    for m in 10 ** rng.uniform(-6, 3, 200):
        assert scales.exponent(m) == brute(m)


@pytest.mark.parametrize("bad", [-1, scales.MAX_EXP + 1, 1.5, "3"])
def test_block_scales_reject_bad_exponents(bad):
    with pytest.raises(ValueError, match="exponent"):
        scales.BlockScales(gate_tm=bad)


def test_choose_takes_each_familys_largest_product():
    maxima = [
        {"tm.p1z": 0.24, "tm.g": 0.57, "cm.q2k2": 0.5, "cm.r1r": 0.28, "cm.ag": 0.27, "cm.k": 99.0},
        {},
    ]
    got = scales.choose(maxima)
    e_ag = scales.exponent(0.27)
    a = min(scales.exponent(0.5), e_ag)
    assert got[0] == scales.BlockScales(
        scales.exponent(0.57), a, min(scales.exponent(0.28), e_ag - a)
    )
    assert got[1] == scales.BlockScales(0, 0, 0)  # nothing seen: no fold


def brute_pair(act, gate, ag, kappa=scales.KAPPA, top=scales.TOP):
    """The (a, b) with the largest a + b (then the largest a) such that every product a fold
    scales UP stays within top / kappa: no scaling is always allowed, even for a value that is
    already above it."""

    def fits(e, v):
        return e == 0 or kappa * 2**e * v <= top

    ok = [
        (a, b)
        for a in range(scales.MAX_EXP + 1)
        for b in range(scales.MAX_EXP + 1)
        if fits(a, act) and fits(b, gate) and fits(a + b, ag)
    ]
    return max(ok, key=lambda ab: (ab[0] + ab[1], ab[0]))


def test_choose_matches_brute_force_on_the_channel_mix():
    rng = np.random.default_rng(40)
    for _ in range(300):
        act, gate, ag = 10 ** rng.uniform(-4, 1.5, 3)
        got = scales.choose([{"cm.q1k": act, "cm.r1r": gate, "cm.ag": ag}])[0]
        assert (got.act_cm, got.gate_cm) == brute_pair(act, gate, ag)


def test_files_round_trip_and_refuse_short_lists(tmp_path):
    sc = [scales.BlockScales(3, 7, 2), scales.BlockScales(0, 14, 0)]
    scales.save(tmp_path / "s.json", sc, {"source": "test"})
    assert scales.load(tmp_path / "s.json") == sc
    assert json.loads((tmp_path / "s.json").read_text())["meta"] == {"source": "test"}
    with pytest.raises(ValueError, match="2 blocks of scales"):
        scales.load(tmp_path / "s.json", n_layers=3)


# ── exactness ─────────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(("g", "a", "b"), [(0, 0, 0), (5, 7, 3), (14, 14, 14), (0, 0, 9)])
def test_folded_kernels_are_the_unfolded_ones_times_two_to_the_e(g, a, b):
    params, *_ = synthetic(L=1, seed=31)
    p, geo, ops = params.block(0), geometry.PlainGeometry(NumpyOps()), NumpyOps()
    w0 = weights.encode_block(p, geo, "b0.")
    w1 = weights.encode_block(p, geo, "b0.", scales=scales.BlockScales(g, a, b))
    rng = np.random.default_rng(32)
    Z, K, R = (0.3 * rng.standard_normal((5, n)) for n in (D, DFF, DFF))
    np.testing.assert_array_equal(block.tm_gate(ops, Z, w1), 2.0**g * block.tm_gate(ops, Z, w0))
    np.testing.assert_array_equal(
        block.cm_gate(ops, K, R, w1), 2.0 ** (a + b) * block.cm_gate(ops, K, R, w0)
    )
    np.testing.assert_array_equal(w1.wout * 2.0**g, w0.wout)
    np.testing.assert_array_equal(w1.wv * 2.0 ** (a + b), w0.wv)


@pytest.mark.parametrize(("payload", "route"), ARMS)
def test_folded_model_matches_mirror_and_unfolded_exact(payload, route):
    params, *_ = synthetic(L=2, seed=33)
    sc = [scales.BlockScales(6, 9, 4), scales.BlockScales(14, 0, 14)]
    model = make(params, payload, route, sc)
    n = model.capacity
    mirror, exact = ref.MirrorModel(params, n, scales=sc), ref.ExactModel(params, n)
    model.start()
    rng = np.random.default_rng(34)
    for pos in range(2):
        rows = params.emb[rng.integers(0, params.vocab, n)].astype(np.float64)
        got = model.step(rows, pos)["out"]
        assert ref.rel_err_rms(got, mirror.step(rows)) < 1e-9  # the same folds, plain numpy
        assert ref.rel_err_rms(got, exact.step(rows)) < 1e-4  # the unfolded math


def test_too_few_scales_are_refused():
    params, *_ = synthetic(L=2, seed=35)
    with pytest.raises(ValueError, match="1 blocks of scales"):
        make(params, "pairs", "split", [scales.BlockScales(1, 1, 1)])


# ── calibration ───────────────────────────────────────────────────────────────────────────────


def test_measure_window_matches_a_hand_rolled_loop():
    params, *_ = synthetic(L=2, seed=36)
    rng = np.random.default_rng(37)
    streams = [list(rng.integers(0, params.vocab, 6)) for _ in range(3)]
    got = scales.measure(params, streams, 2, 5)
    want: list[dict[str, float]] = [{} for _ in range(2)]
    seen = {"on": False}

    def observe(b, name, value):
        if seen["on"]:
            want[b][name] = max(want[b].get(name, 0.0), float(np.max(np.abs(value))))

    exact = ref.ExactModel(params, 3, observe=observe)
    for pos in range(5):
        seen["on"] = pos >= 2
        exact.step(params.emb[[s[pos] for s in streams]].astype(float))
    assert got == want
    names = set(got[0])
    for family in (scales.GATE_TM_VALUES, scales.ACT_CM_VALUES, scales.GATE_CM_VALUES):
        assert set(family) <= names
    assert scales.PRODUCT_CM in names


@pytest.mark.parametrize(("start", "stop"), [(3, 3), (-1, 2), (0, 7)])
def test_measure_refuses_bad_windows(start, stop):
    params, *_ = synthetic(L=1, seed=38)
    with pytest.raises(ValueError, match="tokens"):
        scales.measure(params, [[1, 2, 3, 4, 5, 6]], start, stop)


def test_observing_changes_nothing():
    params, *_ = synthetic(L=2, seed=39)
    rows = params.emb[[1, 2, 3]].astype(float)
    quiet = ref.ExactModel(params, 3).step(rows)
    loud = ref.ExactModel(params, 3, observe=lambda *_a: None).step(rows)
    np.testing.assert_array_equal(quiet, loud)


# ── the shipped file and the drivers ──────────────────────────────────────────────────────────


def test_shipped_scales_are_sane_and_public():
    sc = scales.load(scales.SHIPPED)
    assert len(sc) == 24
    text = scales.SHIPPED.read_text()
    assert "/Users/" not in text
    assert "/home/" not in text
    meta = json.loads(text)["meta"]
    assert meta["kappa"] == scales.KAPPA
    assert meta["top"] == scales.TOP
    assert meta["validation_min_kept"] >= 1.0  # no held-out maximum went past the top
    assert scales.choose(meta["calibration_maxima"]) == sc  # the file is what its maxima give


def test_resolve_covers_every_spelling(tmp_path):
    assert scales.resolve("none") is None
    assert scales.resolve("NONE") is None
    assert scales.resolve(None) == scales.resolve("shipped") == scales.load(scales.SHIPPED)
    scales.save(tmp_path / "s.json", [scales.BlockScales(1, 2, 3)])
    assert scales.resolve(str(tmp_path / "s.json")) == [scales.BlockScales(1, 2, 3)]
    with pytest.raises(ValueError, match="24 blocks of scales"):
        scales.resolve("shipped", n_layers=25)
    with pytest.raises(FileNotFoundError):
        scales.resolve(str(tmp_path / "missing.json"))


@pytest.mark.parametrize("choice", ["none", "shipped"])
def test_driver_records_the_folds_it_ran(tmp_path, choice):
    from examples.fhe_ssm_from_primitives import run_fake

    _params, w, seeds, cfg = synthetic(L=2, seed=41)
    np.savez(tmp_path / "w.npz", **w)
    (tmp_path / "s.json").write_text(json.dumps(seeds))
    (tmp_path / "c.json").write_text(json.dumps(cfg))
    out = tmp_path / "r.json"
    argv = ["--weights", str(tmp_path / "w.npz"), "--seeds", str(tmp_path / "s.json"),
            "--config", str(tmp_path / "c.json"), "--tokens", "1", "--logn", "11",
            "--scales", choice, "--out", str(out)]  # fmt: skip
    assert run_fake.main(argv) == 0
    rep = json.loads(out.read_text())
    assert rep["ticks"][0]["relErrRms_vs_mirror"] < 1e-9
    assert rep["ticks"][0]["relErrRms_vs_exact"] < 1e-4
    if choice == "none":
        assert rep["scales"] is None
        assert "scales_file" not in rep
    else:
        shipped = scales.load(scales.SHIPPED)
        assert rep["scales"] == [list(dataclasses.astuple(s)) for s in shipped[:2]]
        assert (
            rep["scales_file"]["sha256"] == hashlib.sha256(scales.SHIPPED.read_bytes()).hexdigest()
        )
