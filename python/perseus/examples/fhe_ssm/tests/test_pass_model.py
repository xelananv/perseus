# SPDX-License-Identifier: MIT
"""The pass kernels in the model (``pass_kernels="on"``) at the synthetic shape LOGN 11 (t = 32 lanes, d = 32,
N = 1,024 slots), two ways each: the scan, the token shift and the carry prep on the fake against numpy per lane-slot
and against ``ExactModel.pass_`` after the block; one pass of every scenario against ``ExactModel.pass_`` (per live
lane-slot) and against the mirror; two chunk passes of one sequence against 128 sequential steps of the trainer's
recurrence; a document's readers (odd lane-slots included) against the recurrence run to the document's end; the tick
configuration against ``--pass-kernels off``; the op stream (today's call list with the kernels in place, the census
per block); the levels (no reactive refresh from the kernels under both level settings); the lane-slot order; the
refusals."""

import json

import numpy as np
import pytest

from perseus.examples.fhe_ssm import block, fakes, filling, geometry, passkernel, passmasks, ref, run_fake
from perseus.examples.fhe_ssm.model import FheSsmModel
from perseus.examples.fhe_ssm.passmasks import PassConfig, Segment
from perseus.examples.fhe_ssm.synthetic import synthetic
from perseus.impl import fake, fakecap
from perseus.impl.ops import FheOps, NumpyOps

from .test_passmasks import CASES, carry_want, decay, scan_matrix, shift_want

D, DFF, N, T = 32, 128, 1024, 32
NQ = 2 * T
LABELS = ("tm.shift", "tm.scan", "carry.prep")
#: the reference circuit's refreshes per tick at this shape under fork levels (tests/test_axes.py)
OFF_BY_STEP = {"tm.norm": 60, "tm.gate": 10, "cm.norm": 60, "cm.gate": 40, "carry": 4, "tail": 34}


def params_of(L=2, seed=3):
    params, *_ = synthetic(L=L, d=D, seed=seed)
    return params


def make(params, levels="fork", **kw):
    lim, land = fakes.level_preset(levels)
    fhe, inf = fakes.make_runtime_fake(
        N=N, hid=D, dim=D, H=1, H_real=1, E=DFF, E_real=DFF, rot_steps=(), complex_payload=True,
        limit=lim, bts_out=land,
    )  # fmt: skip
    model = FheSsmModel(inf, params, core=fake.core, unit=fhe.unit, **kw)
    fhe.loaded_rot_steps = sorted(model.geo.rotation_steps())
    fhe._rot = set(fhe.loaded_rot_steps) | {s - N for s in fhe.loaded_rot_steps}
    return fhe, model


class Labeled(list):
    """``fhe.calls`` with the step label of every call beside it."""

    def __init__(self, ops):
        super().__init__()
        self.ops, self.labels = ops, []

    def append(self, x):
        super().append(x)
        self.labels.append(getattr(self.ops, "step_label", None))


def rows_of_vec(v):
    return passkernel.unpair_lanes(passkernel.lane_array(np.asarray(v), T))


def vec_of_rows(rows_q):
    return passkernel.slot_vector(passkernel.pair_lanes(rows_q))


def rel(got, want):
    return float(np.abs(got - want).max() / max(np.abs(want).max(), 1e-300))


def scenario(spec, params, seed=1, **kw):
    return passmasks.scenario(spec, vocab=params.vocab, seed=seed, **kw)


def run_on(params, cfgs, **kw):
    return run_fake.run(params, None, logn=11, log=lambda _s: None, pass_kernels="on", pass_list=cfgs, **kw)


# ── the three kernels on the fake, against numpy per lane-slot ─────────────────────────────────


def kernel_session():
    fhe, inf = fakes.make_runtime_fake(
        N=N, hid=D, dim=D, H=1, H_real=1, E=DFF, E_real=DFF, complex_payload=True
    )
    steps = passkernel.LaneKernel(T, -1).rotation_steps(N) | passkernel.LaneKernel(
        T, 1, False
    ).rotation_steps(N)
    fhe._rot = steps | {s - N for s in steps}
    passkernel.use_pass_hook(inf)
    return fhe, FheOps(inf, fake.core, fhe.unit)


#: the kernels read only the geometry's step scopes (a no-op on the plain geometry)
GEO = geometry.PlainGeometry(NumpyOps())


@pytest.mark.parametrize("case", sorted(CASES))
def test_kernels_on_the_fake_against_numpy(case):
    """``block.pass_scan``, ``block.pass_shift``, ``block.carry_prep`` on the fake's ciphertexts (the hook, the
    rotations, the conjugations of the runtime's ops) against the lane-slot maps written from the definitions: the
    scan as the lower-triangular Toeplitz of ``(1 - a) a^k`` per segment plus the carried weights, the shift as the
    previous lane-slot's ``n`` (``np.roll`` by one lane-slot inside a segment), the carry prep as the segment's last
    position copied over its home."""
    cfg = CASES[case]
    bp = decay(len(case), d=D)
    fhe, ops = kernel_session()
    pk = passmasks.SlotPass(passmasks.masks(cfg, bp, 0), 0, T)
    rng = np.random.default_rng(11)
    x_q, c_q, n_q, cn_q = (0.1 * rng.standard_normal((NQ, D)) for _ in range(4))
    ct = {
        k: fake.FakeCt(vec_of_rows(v), 36, 2) for k, v in (("x", x_q), ("c", c_q), ("n", n_q), ("cn", cn_q))
    }
    s = block.pass_scan(ops, GEO, ct["x"], ct["c"], pk)
    um = block.pass_shift(ops, GEO, ct["n"], ct["cn"], pk)
    M, k = scan_matrix(cfg, bp.a)
    want_s = np.einsum("iqr,ri->qi", M, x_q) + k.T * c_q
    # the scan by np.tril per segment, as a second construction of the same matrix
    for seg in cfg.segments:
        if seg.kind == "seq":
            p = np.arange(seg.h1 - seg.q0)
            for i in range(D):
                a = bp.a[i]
                toe = np.tril((1 - a) * a ** np.maximum(p[:, None] - p[None, :], 0))
                np.testing.assert_allclose(want_s[seg.q0 : seg.h1, i] - k[i, seg.q0 : seg.h1] * c_q[seg.q0 : seg.h1, i],
                                           toe @ x_q[seg.q0 : seg.h1, i], rtol=0, atol=1e-15)  # fmt: skip
    assert rel(rows_of_vec(s.vec), want_s) <= 1e-14
    assert rel(rows_of_vec(um.vec), shift_want(cfg, bp.mix, n_q, cn_q)) <= 1e-14
    st = block.BlockState(s, ct["n"])
    block.carry_prep(ops, GEO, st, pk)
    ws, wn = carry_want(cfg, rows_of_vec(s.vec), n_q)
    assert rel(rows_of_vec(st.s.vec), ws) <= 1e-14
    assert rel(rows_of_vec(st.n_prev.vec), wn) <= 1e-14
    # the levels: S and um one product above their inputs; both carries one product above max(S, n)
    assert ops.lvl(s) == ops.lvl(um) == 36 + 2 + 2
    lv = {name: (lev + (2 if deg == 2 else 0)) for name, (lev, deg) in pk.seen.items()}
    assert lv["Z"] == lv["C"] == lv["C_n"] == max(lv["S"], lv["n"]) + 2
    assert fhe.n_bootstraps == 0
    assert getattr(ops.inf, passkernel.COUNTER) == 66 + 6 + 34


# ── the model: the op stream ───────────────────────────────────────────────────────────────────


def labeled_pass(params, cfgs, **kw):
    """Per pass the labeled call list and the reply (lane-slot order) of the model with the pass kernels."""
    fhe, model = make(params, pass_kernels="on", **kw)
    fhe.calls = Labeled(model.rt.ops)
    model.start()
    out = []
    for pos, cfg in enumerate(cfgs):
        c0 = len(fhe.calls)
        rows = passmasks.pass_rows(cfg, params.emb_row)
        got = model.step(rows, pos, pass_cfg=cfg)["out"]
        out.append((list(zip(fhe.calls[c0:], fhe.calls.labels[c0:])), got, rows))
    return fhe, model, out


def labeled_ticks(params, rows_list, **kw):
    fhe, model = make(params, **kw)
    fhe.calls = Labeled(model.rt.ops)
    model.start()
    out = []
    for pos, rows in enumerate(rows_list):
        c0 = len(fhe.calls)
        got = model.step(rows, pos)["out"]
        out.append((list(zip(fhe.calls[c0:], fhe.calls.labels[c0:])), got))
    return fhe, model, out


def strip_today(calls):
    """Today's call list without the four products the kernels replace per block: the shift's two (the first two
    ``mult_pt`` under ``tm.in`` after the norm) and the scan's two (the first two under ``tm.gate``)."""
    out, seen, prev_label = [], {"tm.in": 0, "tm.gate": 0}, None
    for c, label in calls:
        if label in seen and label != prev_label:
            seen[label] = 0
        if label in seen and c == "mult_pt" and seen[label] < 2:
            seen[label] += 1
            prev_label = label
            continue
        out.append((c, label))
        prev_label = label
    return out


def test_the_kernels_slot_into_todays_op_stream():
    """In the tick configuration the call list with the kernels' own spans (``tm.shift``, ``tm.scan``,
    ``carry.prep``) removed is today's call list with the four products they replace removed: nothing else moves,
    refreshes included; the replies agree."""
    params = params_of()
    cfgs = scenario("tick:2", params)
    _f, _m, on = labeled_pass(params, cfgs)
    rows_r = [passmasks.to_layout(r) for _c, _g, r in on]
    _f, _m, off = labeled_ticks(params, rows_r)
    for (con, got_on, _r), (coff, got_off) in zip(on, off, strict=True):
        assert [x for x in con if x[1] not in LABELS] == strip_today(coff)
        spans = {lab: [c for c, l in con if l == lab] for lab in LABELS}
        assert spans["tm.shift"].count("mult_pt") == 2 * 6  # two blocks
        assert spans["tm.scan"].count("mult_pt") == 2 * 66
        assert spans["carry.prep"].count("mult_pt") == 2 * 34
        assert not any("bootstrap" in calls for calls in spans.values())
        np.testing.assert_array_equal(passmasks.to_layout(got_on), got_off)


def per_block_census(params, cfgs=None, rows_list=None, **kw):
    """The fake's census per block (block 0 and 1) of the last pass, through ``run_fake.stage_marks``."""
    if cfgs is not None:
        fhe, model = make(params, pass_kernels="on", **kw)
        items = [(passmasks.pass_rows(c, params.emb_row), c) for c in cfgs]
    else:
        fhe, model = make(params, **kw)
        items = [(r, None) for r in rows_list]
    with run_fake.stage_marks(model, fhe) as marks:
        model.start()
        for pos, (rows, cfg) in enumerate(items):
            marks.clear()
            if cfg is None:
                model.step(rows, pos)
            else:
                model.step(rows, pos, pass_cfg=cfg)
    return {
        b: {**run_fake.block_census(fhe.calls[m[0] : m[2]]), "pass_products": m[3] - m[1]}
        for b, m in marks.items()
    }


def test_the_census_per_block():
    """Per block, net of today's: +102 ct x pt (K1 64, shift 4, K2 34), +28 rotations (17 + 1 + 10), +8 conjugations
    (2 + 3 + 3), +4 mult_i, +3 ct - ct, +0 refreshes; 106 products through the pass hook (66 + 6 + 34)."""
    params = params_of()
    cfgs = scenario("tick:2", params)
    on = per_block_census(params, cfgs)
    off = per_block_census(
        params, rows_list=[passmasks.to_layout(passmasks.pass_rows(c, params.emb_row)) for c in cfgs]
    )
    for b in (0, 1):
        d = {k: on[b][k] - off[b][k] for k in off[b]}
        assert d == {"mult_cc": 0, "mult_pt": 102, "mult_scalar": 0, "rotations": 28, "conjugations": 8,
                     "mult_i": 4, "sub_cc": 3, "bootstraps": 0, "pass_products": 106}  # fmt: skip
    assert on[2] == off[2]  # the tail
    # the same count in every configuration (one op stream): the mixed scenario's last pass
    mixed = per_block_census(params, scenario("mixed", params))
    assert all(mixed[b] == on[b] for b in (0, 1))


@pytest.mark.parametrize("levels", ["fork", "preset"])
def test_no_reactive_refresh_from_the_kernels(levels):
    """The level probe: under both level settings the refreshes per pass equal today's tick's, step by step, in every
    scenario, and none fires inside the kernels' spans; the carry inputs sit one product above max(S, n)."""
    params = params_of()
    ids = np.random.default_rng(8).integers(0, params.vocab, size=(2, NQ))
    off = run_fake.run(params, ids, logn=11, log=lambda _s: None, levels=levels)
    want = off["ticks"][1]
    for spec in ("tick:2", "mixed", "prefill:65", "doc:130:64", "churn"):
        rep = run_on(params, scenario(spec, params), levels=levels)
        for t in rep["ticks"]:
            assert (t["bootstraps"], t["reactive"], t["hints"]) == (
                want["bootstraps"],
                want["reactive"],
                want["hints"],
            )
            assert t["bootstraps_by_step"] == want["bootstraps_by_step"]
            assert not set(t["bootstraps_by_step"]) & set(LABELS)
            for seen in t["pass_levels"].values():
                eff = {k: lev + (2 if deg == 2 else 0) for k, (lev, deg) in seen.items()}
                assert eff["Z"] == eff["C"] == eff["C_n"] == max(eff["S"], eff["n"]) + 2
    if levels == "fork":
        assert want["bootstraps_by_step"] == OFF_BY_STEP


# ── fidelity: every scenario, two references ───────────────────────────────────────────────────


SCENARIOS = ["tick", "prefill:320", "doc:258:64", "mixed", "churn", "prefill:1", "prefill:2", "prefill:3",
             "prefill:63", "prefill:64", "prefill:65", "doc:65:7"]  # fmt: skip


@pytest.mark.parametrize("spec", SCENARIOS)
def test_every_scenario_against_exact_and_mirror(spec):
    params = params_of()
    rep = run_on(params, scenario(spec, params, seed=len(spec)))
    assert rep["axes"]["pass_kernels"] == "on"
    for t in rep["ticks"]:
        assert t["finite"]
        assert t["relErrRms_vs_exact"] <= 1e-12
        assert t["relErrRms_vs_exact_worst_lane_slot"]["relErrRms"] <= 1e-12
        assert t["relErrRms_vs_mirror"] <= 1e-12
        assert t["relErrRms_vs_mirror_all_lane_slots"] <= 1e-12
        assert t["top1_agree"]["agree"] == t["top1_agree"]["of"]
        assert t["census_blocks_equal"]
        assert t["pass_products"] == 2 * 106


def test_two_chunks_equal_128_sequential_steps():
    """prefill:128 (two chunks of 64 on [0, 64), the second carried) and three generated tokens on lane-slot 63,
    against the trainer's recurrence stepped one token at a time (``ExactModel.step``, one lane, no pass logic)."""
    params = params_of()
    cfgs = scenario("prefill:128", params)
    rep = run_on(params, cfgs)
    _fhe, _model, on = labeled_pass(params, [passmasks.PassConfig.from_json(c) for c in rep["pass_list"]])
    seq = ref.ExactModel(params, 1)
    worst = 0.0
    for p, (_c, got, rows) in enumerate(on):
        qs = range(64) if p < 2 else [63]
        for q in qs:
            want = seq.step(rows[q][None])[0]
            worst = max(worst, ref.rel_err_rms(got[q], want))
    assert worst <= 1e-12


def test_readers_read_the_document():
    """doc:130:64: every reader, on an even or an odd lane-slot, starts from the document's final state: against the
    recurrence run through the document's 130 tokens on one lane, its state copied to 64 lanes, then the readers'
    questions."""
    params = params_of()
    cfgs = [
        passmasks.PassConfig.from_json(c) for c in run_on(params, scenario("doc:130:64", params))["pass_list"]
    ]
    _fhe, _model, on = labeled_pass(params, cfgs)
    doc = ref.ExactModel(params, 1)
    for p in range(3):
        cfg = cfgs[p]
        for q in cfg.live_slots():
            doc.step(on[p][2][q][None])
    readers = ref.ExactModel(params, NQ)
    readers.s = [np.repeat(s, NQ, axis=0) for s in doc.s]
    readers.u_prev = [np.repeat(u, NQ, axis=0) for u in doc.u_prev]
    for p in range(3, len(cfgs)):
        want = readers.step(on[p][2])
        got = on[p][1]
        per = [ref.rel_err_rms(got[q], want[q]) for q in range(NQ)]
        assert max(per) <= 1e-12
        assert max(per[1::2]) <= 1e-12  # the odd lane-slots (imaginary axes) too


def test_k1_and_the_shift_after_block_0_against_pass_(monkeypatch):
    """The scan's ``S`` and the norm's ``n`` per lane-slot after block 0, read off the fake's ciphertexts in the
    mixed scenario's three distinct configurations, against ``ExactModel.pass_``'s own lane-slot states."""
    params = params_of()
    cfgs = scenario("mixed", params)
    _fhe, model = make(params, pass_kernels="on")
    exact = ref.ExactModel(params, NQ)
    kept = {}
    orig = passmasks.SlotPass.note

    def note(self, ops, name, ct):
        orig(self, ops, name, ct)
        if self.b == 0:
            kept[name] = np.array(ct.vec)

    monkeypatch.setattr(passmasks.SlotPass, "note", note)
    model.start()
    for pos, cfg in enumerate(cfgs[:3]):
        rows = passmasks.pass_rows(cfg, params.emb_row)
        model.step(rows, pos, pass_cfg=cfg)
        exact.pass_(cfg, rows)
        s_ref, u_ref = exact.lane_states[0][0], exact.lane_states[1][0]
        assert rel(rows_of_vec(kept["S"]), s_ref) <= 1e-12
        assert rel(rows_of_vec(kept["n"]), u_ref) <= 1e-12


# ── the tick configuration against today's, the lane-slot order, parked sources ────────────────


@pytest.mark.parametrize("conversations", [NQ, 10, 1])
def test_tick_configuration_equals_pass_kernels_off(conversations):
    """``--pass-kernels on`` in the tick configuration against ``off`` on the same conversations (conversation k on
    lane-slot k, today on row k): every extra term multiplies an exact zero mask, and a lane with one live and one
    parked axis (one conversation) sums its shift pair before the total, so the replies agree to the last bit;
    fewer conversations park the rest (copies of lane-slot 0, run fresh)."""
    params = params_of()
    cfgs = scenario("tick:3", params, conversations=conversations)
    _f, _m, on = labeled_pass(params, cfgs)
    rows_off = [r[:conversations] for _c, _g, r in on]
    _f, _m, off = labeled_ticks(params, rows_off)
    for (_c, got_on, _r), (_c2, got_off) in zip(on, off, strict=True):
        assert np.isfinite(got_on).all()
        assert ref.rel_err_rms(got_on[:conversations], got_off) <= 1e-13
        np.testing.assert_array_equal(got_on[:conversations], got_off)


def test_the_lane_slot_order_round_trips():
    params = params_of()
    _fhe, model = make(params, pass_kernels="on")
    rows_q = np.random.default_rng(4).standard_normal((NQ, D)) * 0.03
    ct = model.encode_input(rows_q)
    np.testing.assert_array_equal(ct.vec, vec_of_rows(rows_q))  # lane-slot q on lane q div 2, axis q mod 2
    np.testing.assert_array_equal(model._decode(ct, NQ), rows_q)
    np.testing.assert_array_equal(filling.unpack_pairs(ct.vec, D, NQ), passmasks.to_layout(rows_q))
    with pytest.raises(ValueError, match="all 64 lane-slots"):
        model.encode_input(rows_q[:10])


def test_the_refusals():
    params = params_of()
    _p, w, seeds, cfg = synthetic(L=2, d=D, seed=3)
    from perseus.examples.fhe_ssm.weights import ModelParams

    with pytest.raises(ValueError, match="lane-mix"):
        make(ModelParams(w, seeds, cfg, lane_mix="random"), pass_kernels="on")
    with pytest.raises(ValueError, match="pairs"):
        make(params, pass_kernels="on", payload="real")
    with pytest.raises(ValueError, match="pass_kernels"):
        make(params, pass_kernels="maybe")
    _fhe, model = make(params, pass_kernels="on")
    model.start()
    rows = np.zeros((NQ, D))
    with pytest.raises(ValueError, match="configuration"):
        model.step(rows, 0)
    with pytest.raises(ValueError, match="lane-slots"):
        model.set_pass(
            PassConfig(tuple(Segment.conv(q, "fresh", f"c{q}") for q in range(32)), tuple([0] * 32))
        )
    _fhe, off = make(params)
    with pytest.raises(ValueError, match="pass_kernels='on'"):
        off.set_pass(scenario("tick:1", params)[0])
    # the capture recorder needs its own branch of the pass hook
    cfhe, cinf = fakes.make_capture_session(N, D, DFF)
    with pytest.raises(NotImplementedError, match="recorder"):
        FheSsmModel(cinf, params, core=fakecap.core, unit=cfhe.unit, pass_kernels="on")


# ── the driver ─────────────────────────────────────────────────────────────────────────────────


def checkpoint(tmp_path):
    _params, w, seeds, cfg = synthetic(L=2, d=D, seed=17)
    paths = {k: tmp_path / f"pbd_{k}" for k in ("weights.npz", "seeds.json", "config.json")}
    np.savez(paths["weights.npz"], **w)
    paths["seeds.json"].write_text(json.dumps(seeds))
    paths["config.json"].write_text(json.dumps(cfg))
    return paths


def test_the_driver_flags(tmp_path):
    paths = checkpoint(tmp_path)
    base = ["--weights", str(paths["weights.npz"]), "--seeds", str(paths["seeds.json"]),
            "--config", str(paths["config.json"]), "--logn", "11"]  # fmt: skip
    out = tmp_path / "rep.json"
    assert run_fake.main([*base, "--pass-kernels", "on", "--scenario", "mixed", "--out", str(out)]) == 0
    rep = json.loads(out.read_text())
    assert rep["scenario"] == "mixed"
    assert rep["axes"]["pass_kernels"] == "on"
    assert len(rep["ticks"]) == len(rep["pass_list"]) == 5
    t = rep["ticks"][2]
    assert t["token_values"] == 64
    assert t["sequences"] == {"B": 8}
    assert t["conversations"] == 56
    assert t["census_block_0"]["pass_products"] == 106
    assert t["relErrRms_vs_exact"] <= 1e-12
    # the same passes from a file
    cfg_file = tmp_path / "passes.json"
    cfg_file.write_text(json.dumps(rep["pass_list"]))
    out2 = tmp_path / "rep2.json"
    assert (
        run_fake.main([*base, "--pass-kernels", "on", "--lane-config", str(cfg_file), "--out", str(out2)])
        == 0
    )
    rep2 = json.loads(out2.read_text())
    assert rep2["scenario"] is None
    assert rep2["pass_list"] == rep["pass_list"]
    assert [x["relErrRms_vs_exact"] for x in rep2["ticks"]] == [x["relErrRms_vs_exact"] for x in rep["ticks"]]
    with pytest.raises(SystemExit, match="pass-kernels on"):
        run_fake.main([*base, "--scenario", "tick"])
    with pytest.raises(SystemExit, match="twice"):
        run_fake.main([*base, "--pass-kernels", "on", "--scenario", "tick", "--lane-config", str(cfg_file)])
    # --conversations with the pass kernels: the tick scenario parks the rest
    assert run_fake.main([*base, "--pass-kernels", "on", "--conversations", "10", "--out", str(out)]) == 0
    rep3 = json.loads(out.read_text())
    assert [t["parked"] for t in rep3["ticks"]] == [54] * 6
