# SPDX-License-Identifier: MIT
"""The speed axes, each two ways at the synthetic shape: off reproduces the reference circuit's op
stream and counts (the fork's formulas), on is checked against the mirror (the same kernels on rows)
and the closed form, on the edge cases each axis defines (the first tick's bootstrapped zero state
and a later tick, parked lanes, both halves of a pack, the output norm's own count, the hint rule)."""

import numpy as np
import pytest

from perseus.examples.fhe_ssm import block, fakes, filling, geometry, norm, pairs, ref, run_fake
from perseus.examples.fhe_ssm.lanemix import NB, NG, LaneMixKernel
from perseus.examples.fhe_ssm.model import FheSsmModel
from perseus.examples.fhe_ssm.synthetic import synthetic
from perseus.examples.fhe_ssm.weights import ModelParams
from perseus.impl import fake
from perseus.impl.ops import FheOps

D, DFF, N, T = 32, 128, 1024, 32
B, G = 8, 4  # DiagLinear.split(32)
#: the reference circuit's refreshes per tick at this shape (2 blocks, 64 split pairs, seeds window 2e-4..2e-3)
OFF_REFRESHES = 208
OFF_BY_STEP = {"tm.norm": 60, "tm.gate": 10, "cm.norm": 60, "cm.gate": 40, "carry": 4, "tail": 34}
#: the same circuit under upstream's hint rule (--diag-hint 2): the linears' inputs refreshed a level earlier
ON_REFRESHES = 173
ON_BY_STEP = {
    "tm.norm": 60,
    "tm.gate": 8,
    "cm.norm": 60,
    "cm.kr": 2,
    "cm.v": 8,
    "carry": 4,
    "tm.in": 1,
    "tail": 30,
}


def make(params, **kw):
    fhe, inf = fakes.make_runtime_fake(
        N=N, hid=D, dim=D, H=1, H_real=1, E=DFF, E_real=DFF, rot_steps=(), complex_payload=True
    )
    model = FheSsmModel(inf, params, core=fake.core, unit=fhe.unit, **kw)
    fhe.loaded_rot_steps = sorted(model.geo.rotation_steps())
    fhe._rot = set(fhe.loaded_rot_steps) | {s - N for s in fhe.loaded_rot_steps}
    return fhe, model


def tick(model, fhe, rows, pos):
    c0 = len(fhe.calls)
    out = model.step(rows, pos)["out"]
    return out, fhe.calls[c0:]


def census(calls):
    return {
        "mult_cc": calls.count("mult_cc") + calls.count("square"),
        "mult_pt": calls.count("mult_pt"),
        "rotations": sum(c.startswith("rotate") for c in calls),
        "conjugations": calls.count("conjugate"),
    }


def reference_counts(L):
    """The fork's formulas for the split-pair circuit at this shape: per block the linears'
    rotations ((B-1)+(G-1) for W_in and W_out, (B-1)+8(G-1) for W_k/W_r on one input, 4(B-1)+(G-1)
    for W_v), 2 halves x log2(t) per norm; 14 blocks of D diagonals and 6 + 2 x 22 constants; 11
    conjugations per block and 1 for the tail."""
    lin_rot = 2 * ((B - 1) + (G - 1)) + (B - 1) + 8 * (G - 1) + 4 * (B - 1) + (G - 1)
    norm_rot = 2 * 2 * 5
    return {
        "mult_pt": L * (14 * D + 6 + 2 * 22) + 1,
        "rotations": L * (lin_rot + norm_rot) + 2 * 5,
        "conjugations": L * 11 + 1,
    }


def run_arm(params, ticks=2, seed=8, **kw):
    fhe, model = make(params, **kw)
    n = model.capacity
    mirror = ref.MirrorModel(params, n, kw.get("newton_form", "perseus"))
    exact = ref.ExactModel(params, n)
    model.start()
    rng = np.random.default_rng(seed)
    out = []
    for pos in range(ticks):
        rows = params.emb[rng.integers(0, params.vocab, n)].astype(np.float64)
        got, calls = tick(model, fhe, rows, pos)
        out.append((got, calls, mirror.step(rows), exact.step(rows), fhe.n_bootstraps))
    return model, out


# ── off == the reference circuit ───────────────────────────────────────────────────────────────


def test_every_axis_off_is_the_reference_circuit():
    """The default settings are the fork's reference circuit: the same op stream (refreshes included) and the
    same numbers as every axis named at its off value, and that circuit's refresh count at this shape."""
    params, *_ = synthetic(L=2, seed=3)
    off = {
        "newton_form": "perseus",
        "pack_stats": "off",
        "linear": "diagonal",
        "diag_hint": 1,
        "pass_kernels": "off",
    }
    _m, plain = run_arm(params)
    _m, named = run_arm(params, **off)
    for (ga, ca, _, _, ba), (gb, cb, _, _, bb) in zip(plain, named):
        assert ca == cb  # the whole call list, "bootstrap" entries included
        np.testing.assert_array_equal(ga, gb)
        assert ba == bb
    c = census(plain[1][1])
    want = reference_counts(2)
    assert {k: c[k] for k in want} == want
    ids = np.random.default_rng(8).integers(0, params.vocab, size=(2, 64))
    rep = run_fake.run(params, ids, logn=11, log=lambda _s: None)
    assert rep["axes"] == {**off, "newton_sites": "served", "pcmm_wbits": None, "lane_mix": "off"}
    for t in rep["ticks"]:
        assert (t["bootstraps"], t["reactive"], t["hints"], t["carry"]) == (OFF_REFRESHES, 204, 0, 4)
        assert t["bootstraps_by_step"] == OFF_BY_STEP
        assert t["out_level"] == 38


def test_reference_counts_follow_the_formulas():
    params, *_ = synthetic(L=2, seed=3)
    _m, out = run_arm(params)
    c = census(out[1][1])
    want = reference_counts(2)
    assert {k: c[k] for k in want} == want
    # the Newton chains, per norm site and half: the statistic's square, 11 steps x 3 ct x ct products
    # (square, c*a, *ct; the affine first step and c = -0.5 v are scalar products) and the output product:
    # 1 + 33 + 1 = 35; the gates per half: tm 2 (z^2, z * poly), cm 4 per d_ffn block (k^2, q3k * k^2, r^2,
    # act * gate) x 4 blocks
    norm_cc = 2 * (1 + 3 * 11 + 1)
    gates_cc = 2 * (2 + 4 * 4)
    assert c["mult_cc"] == 2 * (2 * norm_cc + gates_cc) + norm_cc


# ── newton_form ────────────────────────────────────────────────────────────────────────────────


def test_depth2_is_the_same_polynomial_two_levels_per_step():
    params, *_ = synthetic(L=2, seed=3)
    _m, a = run_arm(params)
    _m, b = run_arm(params, newton_form="depth2")
    for (ga, ca, ma, ea, _), (gb, cb, mb, eb, _) in zip(a, b):
        assert ref.rel_err_rms(gb, mb) < 1e-9  # against the mirror in the same form
        assert ref.rel_err_rms(gb, eb) < 1e-4
        assert ref.rel_err_rms(ga, gb) < 1e-9  # the two forms agree (one polynomial)
        assert census(ca)["mult_pt"] == census(cb)["mult_pt"]
        assert census(ca)["mult_cc"] == census(cb)["mult_cc"]
    # the depth: a chain of n steps costs 3 levels per step in the perseus form, 2 in depth2
    site = params.block(0).norm_tm
    for form, per_step in (("perseus", 3), ("depth2", 2)):
        _fhe, inf = fakes.make_runtime_fake(N=N, hid=D, dim=D, H=1, H_real=1, E=DFF, E_real=DFF, limit=10**6)
        ops = FheOps(inf, fake.core, 2)
        v = fake.FakeCt(np.full(N, 1.0), 34, 1)
        y = norm.newton_rsqrt(ops, v, site.y0, site.iters, form)
        assert ops.lvl(y) - 34 == 2 * (1 + per_step * (site.iters - 1))  # 2 primes per level


# ── newton_sites ───────────────────────────────────────────────────────────────────────────────


def test_per_site_counts_in_block_and_keeps_the_output_norm():
    _p, w, seeds, cfg = synthetic(L=2, seed=3)
    served = ModelParams(w, seeds, cfg, newton_sites="served")
    per = ModelParams(w, seeds, cfg, newton_sites="per-site")
    for b in range(2):
        for site in ("norm_tm", "norm_cm"):
            s, q = getattr(served.block(b), site), getattr(per.block(b), site)
            assert (s.iters, q.iters) == (12, 9)
            assert q.y0 == pytest.approx(3.0 * s.y0)  # frac 0.3 against 0.1, same frame
            assert q.sigma == s.sigma
    assert per.norm_out == served.norm_out  # ln_out keeps 0.1 / 12
    with pytest.raises(ValueError, match="newton_sites"):
        ModelParams(w, seeds, cfg, newton_sites="x")
    _m, out = run_arm(per)
    _m, ref_out = run_arm(served)
    for (got, calls, mir, ex, _), (_, rcalls, _, _, _) in zip(out, ref_out):
        assert ref.rel_err_rms(got, mir) < 1e-9
        assert ref.rel_err_rms(got, ex) < 1e-4
        # 3 steps fewer at the four in-block sites, 2 halves each: 3 ct x ct per step saved; ln_out unchanged
        assert census(rcalls)["mult_cc"] - census(calls)["mult_cc"] == 2 * 2 * 2 * 3 * 3


# ── pack_stats ─────────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("form", norm.NEWTON_FORMS)
def test_packed_norm_equals_separate_to_rounding(form):
    """Both halves of a pair through one chain against the two separate chains, at the first
    norm of a tick (inputs at the landing level): equal to float64 rounding, the merged statistic
    stamped 2t-periodic, the levels as measured on the route-5 check (perseus: output 38 -> 40;
    depth2: 36 -> 40)."""
    params, *_ = synthetic(L=1, seed=5)
    site = params.block(0).norm_tm
    rows = (
        0.03
        * np.random.default_rng(6).standard_normal((2 * T, D))
        * np.random.default_rng(7).uniform(0.6, 1.4, (2 * T, 1))
    )
    got = {}
    for pack in ("off", "halves"):
        fhe, model = make(params, pack_stats=pack, newton_form=form)
        periods: list[int] = []
        inner = model.rt.ops.inf.tag_reduce

        def tagged(a, stride, _inner=inner, _periods=periods):
            _periods.append(int(stride))
            return _inner(a, stride)

        model.rt.ops.inf.tag_reduce = tagged
        x = model.geo.encode_rows(rows)
        c0 = len(fhe.calls)
        y = block.norm(model.geo, x, site, "t.norm", form, pack)
        got[pack] = (
            model.geo.decode_rows(y, 2 * T),
            census(fhe.calls[c0:]),
            model.rt.ops.lvl(y),
            periods,
            fhe.n_bootstraps,
        )
    np.testing.assert_allclose(got["halves"][0], got["off"][0], rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(got["halves"][0], ref.rmsnorm_exact(rows, 1.0), rtol=1e-6)
    assert got["halves"][3][-1] == 2 * T  # the merged statistic's period: K t = 64 slots
    assert got["off"][3] == [T, T]
    assert got["halves"][1]["mult_pt"] == got["off"][1]["mult_pt"] + 4  # 2 merge + 2 scatter masks
    assert got["halves"][1]["rotations"] == got["off"][1]["rotations"] + 2  # one scatter rotation per half
    assert got["halves"][1]["mult_cc"] == got["off"][1]["mult_cc"] - 33  # one chain of 11 x 3 saved
    assert (got["off"][2], got["halves"][2]) == ({"perseus": 38, "depth2": 36}[form], 40)
    assert (got["off"][4], got["halves"][4]) == (
        {"perseus": 30, "depth2": 28}[form],
        {"perseus": 17, "depth2": 16}[form],
    )


def test_packed_model_matches_mirror_and_exact_on_parked_lanes():
    params, *_ = synthetic(L=2, seed=3)
    for n in (2 * T, T + 1, 1):
        fhe, model = make(params, pack_stats="halves")
        mirror, exact = ref.MirrorModel(params, n), ref.ExactModel(params, n)
        model.start()
        rng = np.random.default_rng(n)
        for pos in range(2):
            rows = params.emb[rng.integers(0, params.vocab, n)].astype(np.float64)
            got, _ = tick(model, fhe, rows, pos)
            assert ref.rel_err_rms(got, mirror.step(rows)) < 1e-9
            assert ref.rel_err_rms(got, exact.step(rows)) < 1e-4
            assert model.levels() == [(34, 34)] * 2


def test_pack_stats_needs_the_split_pair_route():
    params, *_ = synthetic(L=1, seed=3)
    with pytest.raises(ValueError, match="split"):
        make(params, pack_stats="halves", pair_route="lane")
    with pytest.raises(ValueError, match="pack_stats"):
        make(params, pack_stats="x")


# ── linear: the hint rule and the refusals (PCMM itself needs d = 1024: test_pcmm.py) ─────────


def test_diag_hint_2_is_upstreams_rule_and_moves_only_refreshes():
    """``--diag-hint 2`` (upstream's ``diag_apply`` headroom) refreshes a linear's input one level earlier than the
    reference: the products and rotations are the same, the refresh sites move, the mirror still agrees."""
    params, *_ = synthetic(L=2, seed=3)
    _m, ref_arm = run_arm(params)
    _m, up = run_arm(params, diag_hint=2)
    for (g1, c1, m1, _, _), (g2, c2, m2, _, _) in zip(ref_arm, up):
        assert ref.rel_err_rms(g1, m1) < 1e-9
        assert ref.rel_err_rms(g2, m2) < 1e-9
        assert census(c1) == census(c2)
        assert c1 != c2
    ids = np.random.default_rng(8).integers(0, params.vocab, size=(2, 64))
    rep = run_fake.run(params, ids, logn=11, log=lambda _s: None, diag_hint=2)
    for t in rep["ticks"]:
        assert (t["bootstraps"], t["reactive"], t["hints"]) == (ON_REFRESHES, 158, 11)
        assert t["bootstraps_by_step"] == ON_BY_STEP
        assert t["out_level"] == 44
    with pytest.raises(ValueError, match="diag_hint"):
        make(params, diag_hint=3)


def test_pcmm_is_refused_on_a_core_with_in_process_keys():
    params, *_ = synthetic(L=1, seed=3)
    fhe, inf = fakes.make_runtime_fake(
        N=N, hid=D, dim=D, H=1, H_real=1, E=DFF, E_real=DFF, complex_payload=True
    )

    class Core:
        bts = object()

        def __getattr__(self, name):
            return getattr(fake.core, name)

    with pytest.raises(RuntimeError, match="secret key"):
        FheSsmModel(inf, params, core=Core(), unit=fhe.unit, linear="pcmm")
    with pytest.raises(ValueError, match="1024 channels"):
        FheSsmModel(inf, params, core=fake.core, unit=fhe.unit, linear="pcmm")
    with pytest.raises(ValueError, match="filled"):
        FheSsmModel(inf, params, core=fake.core, unit=fhe.unit, layout="decode", linear="pcmm")


# ── lane_mix ───────────────────────────────────────────────────────────────────────────────────


def test_lane_mix_kernel_slots_equal_rows_and_cost_what_the_docstring_says():
    fhe, inf = fakes.make_runtime_fake(
        N=N, hid=D, dim=D, H=1, H_real=1, E=DFF, E_real=DFF, complex_payload=True
    )
    steps = LaneMixKernel.rotation_steps(N)
    fhe._rot = steps | {s - N for s in steps}
    ops = FheOps(inf, fake.core, 2)
    k = LaneMixKernel.random(D, N, T, seed=5)
    rows = np.random.default_rng(1).standard_normal((2 * T, D))
    x = fake.FakeCt(filling.pack_pairs(rows, N), 36, 2)
    y = k.apply(ops, x)
    np.testing.assert_allclose(filling.unpack_pairs(y.vec, D, 2 * T), k.rows(rows), atol=1e-13)
    c = census(fhe.calls)
    assert c == {
        "mult_cc": 0,
        "mult_pt": 2 * NB * NG,
        "rotations": 2 * (NB - 1) + (NG - 1),
        "conjugations": 1,
    }
    assert ops.lvl(y) == 36 + 2 + 2  # one product above x (x carried a pending rescale)
    assert steps == {(-s) % N for s in (1, 2, 3, 4, 5, 6, 7, 8, 16, 24)}
    # lanes below the offset have no source: the term on lane j reads lanes <= j only
    one = np.zeros((2 * T, D))
    one[5] = 1.0  # conversation 5 on the real axis
    t = k.rows(one)
    assert np.abs(t[:5]).max() == 0.0
    assert np.abs(t[T : T + 5]).max() == 0.0
    assert np.abs(t[5:T]).max() > 0.0
    with pytest.raises(ValueError, match="lanes"):
        LaneMixKernel.random(D, 16 * D, 16, seed=1)


def test_lane_mix_model_matches_the_mirror_and_counts():
    _p, w, seeds, cfg = synthetic(L=2, seed=3)
    off = ModelParams(w, seeds, cfg)
    on = ModelParams(w, seeds, cfg, lane_mix="random")
    assert on.block(0).lanemix is not None
    assert off.block(0).lanemix is None
    _m, a = run_arm(off)
    model, b = run_arm(on)
    assert model.geo.base.lane_mix_on
    for (ga, ca, ma, ea, _), (gb, cb, mb, eb, _) in zip(a, b):
        assert ref.rel_err_rms(gb, mb) < 1e-9  # slots against rows with the same random kernel
        assert ref.rel_err_rms(ga, gb) > 1e-6  # the term is not a no-op
        d = {k: census(cb)[k] - census(ca)[k] for k in census(ca)}
        assert d == {"mult_cc": 0, "mult_pt": 2 * 64, "rotations": 2 * 17, "conjugations": 2}
    _fhe, inf = fakes.make_runtime_fake(
        N=N, hid=D, dim=D, H=1, H_real=1, E=DFF, E_real=DFF, complex_payload=True
    )
    with pytest.raises(ValueError, match="pairs"):
        FheSsmModel(inf, on, core=fake.core, unit=2, payload="real")
    with pytest.raises(AttributeError):
        on.lane_mix = "off"  # type: ignore[misc]  # the blocks are cached for one setting


@pytest.mark.parametrize("n", [1, 10, T + 1])
def test_lane_mix_parked_lanes_match_the_mirror(n):
    """Fewer conversations than the capacity: the fake parks the rest (copies of conversation 0) and the live
    lanes read the parked ones below them; the mirror parks its rows the same way."""
    _p, w, seeds, cfg = synthetic(L=2, seed=3)
    on = ModelParams(w, seeds, cfg, lane_mix="random")
    fhe, model = make(on)
    mirror = ref.MirrorModel(on, n)
    model.start()
    rng = np.random.default_rng(n)
    for pos in range(2):
        rows = on.emb[rng.integers(0, on.vocab, n)].astype(np.float64)
        got, _ = tick(model, fhe, rows, pos)
        assert got.shape == (n, D)
        assert ref.rel_err_rms(got, mirror.step(rows)) < 1e-9
    lm = on.block(0).lanemix
    assert lm is not None
    kern = geometry.PlainGeometry.lane_mix_kernel(*lm)
    with pytest.raises(ValueError, match="conversations"):
        kern.rows(np.zeros((2 * T + 1, D)))


# ── the levels setting ─────────────────────────────────────────────────────────────────────────


def test_preset_levels_come_from_upstreams_n32_preset():
    assert fakes.level_preset("fork") == (46, 34)
    assert fakes.level_preset("preset") == (44, 32)
    fhe, _inf = fakes.make_runtime_fake(
        N=N, hid=D, dim=D, H=1, H_real=1, E=DFF, E_real=DFF, limit=44, bts_out=32
    )
    b = fake.FakeCt(np.ones(N), 42, 2)
    fhe.bootstrap(b)
    assert (b.level, b.noise_deg) == (32, 2)
    with pytest.raises(ValueError, match="levels"):
        fakes.level_preset("x")


def test_split_route_halves_are_the_sections_inputs():
    """``PairGeometry.split`` (what pack_stats reads) is the section's own split."""
    params, *_ = synthetic(L=1, seed=3)
    _fhe, model = make(params)
    x = fake.FakeCt(
        np.random.default_rng(2).standard_normal(N) + 1j * np.random.default_rng(3).standard_normal(N), 34
    )
    ((a, b),) = model.geo.split([x])
    a2, b2 = pairs.split(model.rt.ops, x)
    np.testing.assert_array_equal(a.vec, a2.vec)
    np.testing.assert_array_equal(b.vec, b2.vec)
    assert isinstance(model.geo, geometry.PairGeometry)
