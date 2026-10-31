# SPDX-License-Identifier: MIT
"""The planned driver under the pass kernels, everything a CPU can see, at the synthetic shape (LOGN 11, t = 32,
``synthetic(L=2, d=32)``, the preset's levels), against the stock plan of a merged capture (the configurations of
``tests/test_pass_capture.py`` without the histories, merged under ``max_each``): the twin's ``PassPt`` records equal
``passmasks.masks`` bit for bit per configuration and block (lean and not); the per-pass staging on the CPU stand-in
(``CheckedProgram.pt_keys`` / ``alias_pts``, ``FakeStore``) for every pass of every scenario: every program key reads
the twin's recorded value of that pass's configuration, no pass key aliases a static one, every (alias, level) a
program reads was staged, the staged count per configuration change equals the pass keys whose recorded values differ
(0 on a repeated configuration), the store sized by the distinct (alias, level) pairs holds exactly what was staged;
``run_planned --prepare-only`` per scenario (the stream static, 0 check errors, the budget both ways, the staging's dry
run); the lane-slot order on both ends of the device path; the refusals."""

import json
import types

import numpy as np
import pytest

from perseus.examples.fhe_ssm import fakes, filling, passmasks, passmerge, planned, run_fake, run_planned
from perseus.examples.fhe_ssm.model import FheSsmModel
from perseus.examples.fhe_ssm.synthetic import synthetic
from perseus.impl import fake
from perseus.impl.ops import FheOps

from .test_pass_capture import configurations
from .test_planned_contract import plan_of

D, L, LOGN, NQ = 32, 2, 11, 64
T = NQ // 2
LABELS4 = ("tick", "seqfresh", "seqcarry", "mixed")
SCENARIOS = ("tick", "prefill:320", "doc:258:64", "mixed", "churn")


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    """Four configuration captures (tick; one sequence on [0, 64) fresh; the same carried; mixed), merged under
    ``max_each``, and the stock plan of the merge."""
    d = tmp_path_factory.mktemp("passdrv")
    params, w, seeds, cfg = synthetic(L=L, d=D, seed=21)
    np.savez(d / "w.npz", **w)
    (d / "s.json").write_text(json.dumps(seeds))
    (d / "c.json").write_text(json.dumps(cfg))
    base = ["--weights", str(d / "w.npz"), "--seeds", str(d / "s.json"), "--config", str(d / "c.json"),
            "--logn", str(LOGN), "--levels", "preset"]  # fmt: skip
    confs = configurations(params.vocab)
    caps = {}
    for lab in LABELS4:
        cfgs, hist = confs[lab]
        assert hist is None
        f = d / f"{lab}.passes.json"
        f.write_text(json.dumps(passmasks.dump_list(cfgs)))
        argv = [*base, "--pass-kernels", "on", "--lane-config", str(f), "--capture", str(d / f"cap_{lab}"),
                "--capture-tick", "1", "--out", str(d / f"{lab}.json")]  # fmt: skip
        assert run_fake.main(argv) == 0, lab
        caps[lab] = d / f"cap_{lab}"
    merged = d / "cap_merged"
    passmerge.merge(caps["tick"], [caps[x] for x in LABELS4[1:]], merged, labels=list(LABELS4), log=lambda _s: None,
                    rule="max_each")  # fmt: skip
    plan = plan_of(merged, d, "syn_pass_stock")
    assert "check errors 0;" in (plan / "summary.txt").read_text()
    return {"dir": d, "base": base, "params": params, "plan": plan, "merged": merged, "caps": caps}


def twin(base, lean=False):
    """The driver's CPU twin under the pass kernels and its recorder."""
    a = run_planned.parser().parse_args([*base, "--pass-kernels", "on", "--plan", "x"])
    params, model, _twin_cfg, _levels = run_planned.make_twin(a)
    return params, model, planned.SsmRecorder(model, model.n_layers, lean=lean)


def recorded(base, cfg, seed=5, lean=False):
    """The twin recorded in ``cfg`` (two tokens, the second reading a carried state, as ``prepare``'s reference):
    ``{block: {pt name: recorded value}}`` of the second token."""
    params, model, rec = twin(base, lean)
    model.set_pass(cfg)
    rng = np.random.default_rng(seed)
    rows = [
        passmasks.pass_rows(run_planned.redrawn(cfg, rng, params.vocab), params.emb_row) for _ in range(2)
    ]
    rec.token(0, rows[0])
    blocks = rec.token(1, rows[1])
    return model, {b: dict(r.pts) for b, r in blocks.items()}


def make_run(synth, passes, tmp_path, lean=False):
    """A prepared ``PlannedSsmRun`` on the CPU (the compile check) with ``passes``."""
    a = run_planned.parser().parse_args(
        [*synth["base"], "--pass-kernels", "on", "--plan", str(synth["plan"]), "--prepare-only"]
    )
    params, model, _twin_cfg, _levels = run_planned.make_twin(a)
    cfg = run_planned.session_config(str(synth["plan"]))
    labels = [f"block_{b}" for b in range(model.n_layers + 1)]
    run = planned.PlannedSsmRun(
        planned.SsmRecorder(model, model.n_layers, lean=lean), str(synth["plan"]), cfg, labels, None, model=model,
        work_dir=str(tmp_path / "work"), passes=passes, log=lambda _m: None,
    )  # fmt: skip
    rng = np.random.default_rng(9)
    rows = [
        passmasks.pass_rows(run_planned.redrawn(passes[0], rng, params.vocab), params.emb_row)
        for _ in range(2)
    ]
    return run.prepare(rows), params


def store_key_refs(prog):
    """Every store reference of a stand-in program before aliasing: ``(store name, pt name, level)``."""
    return [
        (f"{prog.label}/{nm}", nm, int(lv))
        for nm, lv, _d, f in prog.pt_refs
        if f not in planned.GENERATED_FORMS
    ]


# ── the twin's records ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("lean", [False, True])
def test_the_twins_pass_records_are_the_masks_bit_for_bit(synth, lean):
    confs = configurations(synth["params"].vocab)
    for lab in LABELS4:
        cfg = confs[lab][0][1]
        model, pts = recorded(synth["base"], cfg, lean=lean)
        for b in range(L):
            want = passmasks.masks(cfg, model.block_decay(b), b)
            got: dict[str, list] = {}
            for v in pts[b].values():
                if isinstance(v, planned.PassPt):
                    got.setdefault(v.mask_id, []).append(v)
            assert sorted(got) == sorted(want), (lab, b)
            assert {k: len(v) for k, v in got.items() if len(v) != 1} == {"pass/k2/Po": 2}
            for mid, recs in got.items():
                for r in recs:
                    if lean:
                        assert isinstance(r.vec, planned.LeanValues)
                        assert r.digest == planned.digest(want[mid])
                    else:
                        assert r.vec.dtype == want[mid].dtype
                        assert r.vec.tobytes() == np.ascontiguousarray(want[mid]).tobytes(), (lab, b, mid)
        # the tail has no pass kernel; the static plaintexts are never PassPt
        assert not any(isinstance(v, planned.PassPt) for v in pts[L].values())


def test_values_equal_and_the_store_alias_of_a_record():
    v = np.arange(4.0)
    a, b = planned.PassPt("pass/b0/sh/V", v), planned.PassPt("pass/b0/sh/V", v.copy())
    assert planned.values_equal(a, b)
    assert not planned.values_equal(a, planned.PassPt("pass/b1/sh/V", v))  # another id
    assert not planned.values_equal(a, planned.PassPt("pass/b0/sh/V", v + 1))  # other values
    assert not planned.values_equal(a, v)  # a record against bare values
    lean = planned.PassPt("pass/b0/sh/V", planned.LeanValues(v))
    assert planned.values_equal(lean, planned.PassPt("pass/b0/sh/V", planned.LeanValues(v.copy())))
    assert lean.digest == a.digest == planned.digest(v)
    assert planned.store_alias(a) == "pass/b0/sh/V"
    assert planned.store_alias(v) == "v/" + planned.digest(v) == planned.store_alias(planned.LeanValues(v))
    with pytest.raises(ValueError, match="pass mask id"):
        planned.PassPt("v/abc", v)
    with pytest.raises(ValueError, match="pass mask id"):
        planned.PassPt("pass/b0#1", v)


# ── the stand-in staging ────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("spec", SCENARIOS)
def test_the_per_pass_staging_on_the_stand_in(synth, spec, tmp_path):
    params = synth["params"]
    passes = passmasks.scenario(spec, vocab=params.vocab, seed=3)
    run, _ = make_run(synth, passes, tmp_path)
    assert run.static
    # the twin recorded in every configuration of the scenario, independently of the driver
    keys = []
    for c in passes:
        if c.mask_key() not in keys:
            keys.append(c.mask_key())
    twins = {}
    for c in passes:
        if c.mask_key() not in twins:
            twins[c.mask_key()] = recorded(synth["base"], c)[1]
    run.store = planned.FakeStore()
    n0 = run.stage_static()
    progs = {run.labels[b]: (b, run.progs[run.labels[b]]) for b in sorted(run.ref)}
    # the alias partition: pass keys by mask id, static keys by digest, never one alias for both
    pass_al, static_al = set(), set()
    for b, prog in progs.values():
        for store, nm, _lv in store_key_refs(prog):
            a = prog.aliases[store]
            v = run.ref[b].pts[nm]
            if isinstance(v, planned.PassPt):
                assert a == v.mask_id
                pass_al.add(a)
            else:
                assert a == "v/" + planned.digest(v)
                static_al.add(a)
    assert pass_al
    assert static_al
    assert not pass_al & static_al
    # the store holds one slot per distinct (alias, level) pair, which is what session_kwargs reserves
    kw = run.session_kwargs()
    assert n0 == len(run.store) == len(kw["pt_levels"]) == kw["pt_slots"] - 8
    assert sorted(lv for _a, lv in run.store.digests) == sorted(kw["pt_levels"])

    def check(c):
        tw = twins[c.mask_key()]
        for lab, (b, prog) in progs.items():
            assert set(prog.pt_keys()) <= set(run.store.digests), lab  # every key a program reads is resident
            for store, nm, lv in store_key_refs(prog):
                got = run.store.resident[(prog.aliases[store], lv)]
                want = tw[b][nm]
                want = want.vec if isinstance(want, planned.PassPt) else want
                assert got.dtype == np.asarray(want).dtype
                assert got.tobytes() == np.ascontiguousarray(want).tobytes(), (spec, c.name, lab, nm)

    def digests(c):
        tw = twins[c.mask_key()]
        out = {}
        for b, prog in progs.values():
            for store, nm, lv in store_key_refs(prog):
                v = tw[b][nm]
                if isinstance(v, planned.PassPt):
                    out[(prog.aliases[store], lv)] = v.digest
        return out

    check(passes[0])  # setup staged the first pass's configuration (the warm-up's)
    assert run.stage_pass(passes[0])["staged"] == 0  # the warm-up restages nothing
    counts, prev = [], passes[0]
    for p, c in enumerate(passes):
        st = run.stage_pass(c)
        before, after = digests(prev), digests(c)
        assert set(before) == set(after) == set(run._pass_pairs)
        want = sum(1 for k in after if after[k] != before[k])
        assert st["staged"] == want, (spec, p)
        if c.mask_key() == prev.mask_key():
            assert st["staged"] == 0
        check(c)
        counts.append(st["staged"])
        prev = c
    assert len(run.store) == n0  # restaging is in place: no new key, no evict
    assert any(counts) == (len(keys) > 1)
    # the alias map is permanent: refused after a (stand-in) graph capture
    _b, prog0 = next(iter(progs.values()))
    prog0.n_graphs = 1
    with pytest.raises(RuntimeError, match="after a graph capture"):
        prog0.alias_pts({})


def test_the_lean_dry_run_stages_digests(synth, tmp_path):
    passes = passmasks.scenario("mixed", vocab=synth["params"].vocab, seed=3)
    full, _ = make_run(synth, passes, tmp_path / "full")
    lean, _ = make_run(synth, passes, tmp_path / "lean", lean=True)
    a, b = full.staging_dry_run(keep_values=True), lean.staging_dry_run(keep_values=False)
    assert a["every_program_key_resident"]
    assert b["every_program_key_resident"]
    assert [r["staged"] for r in a["passes"]] == [r["staged"] for r in b["passes"]]
    assert (a["setup_staged"], a["store_keys"], a["pass_keys"]) == (
        b["setup_staged"],
        b["store_keys"],
        b["pass_keys"],
    )
    assert full.store.digests == lean.store.digests  # the same keys and digests, values or not
    # a store that keeps values (the device's) refuses a lean record's digests
    again, _ = make_run(synth, passes, tmp_path / "lean2", lean=True)
    again.store = planned.FakeStore(keep_values=True)
    with pytest.raises(TypeError, match="lean recorder"):
        again.stage_static()


# ── run_planned --prepare-only under the pass kernels ───────────────────────────────────────────


@pytest.mark.parametrize("spec", SCENARIOS)
def test_prepare_only_per_scenario(synth, spec, tmp_path):
    out = tmp_path / "prep.json"
    argv = [*synth["base"], "--pass-kernels", "on", "--scenario", spec, "--plan", str(synth["plan"]),
            "--capture", str(synth["merged"]), "--prepare-only", "--out", str(out)]  # fmt: skip
    assert run_planned.main(argv) == 0
    d = json.loads(out.read_text())
    p = d["prepare"]
    assert (p["static_stream"], p["check_errors"], p["dynamic_steps"]) == (True, {}, {})
    assert d["capture"]["checked"] is True
    assert d["axes"]["pass_kernels"] == "on"
    assert p["pass_kernels"] is True
    pl = d["pass_list"]
    want = passmasks.scenario("tick:2" if spec == "tick" else spec, vocab=synth["params"].vocab, seed=7)
    assert pl["list"] == passmasks.dump_list(want)  # tick alone runs --ticks passes (default 2)
    assert pl["passes"] == len(want)
    assert pl["scenario"] == ("tick:2" if spec == "tick" else spec)
    # the budget both ways: by references (every pass product its own) and by distinct (alias, level) pairs
    bud = p["plaintexts"]
    ba = bud["by_alias"]
    assert bud["pass_refs"] == L * 106
    assert ba["pass_ids"] == L * (6 + 2 * T + 2) + T + 1
    assert ba["slots"] == ba["pass"] + ba["static"] < bud["plaintexts"]
    assert bud["bytes"] == bud["plaintexts"] * bud["rows"] * planned.SLOT_ROW_BYTES
    assert ba["bytes"] == ba["slots"] * bud["rows"] * planned.SLOT_ROW_BYTES
    dry = p["staging_dry_run"]
    assert dry["every_program_key_resident"] is True
    assert dry["setup_staged"] == dry["store_keys"] == ba["slots"]
    assert dry["pass_keys"] == ba["pass"]
    staged = [r["staged"] for r in dry["passes"]]
    assert staged[:2] == [0, 0]  # the warm-up and pass 0 run the configuration staged at setup
    ids = pl["config_ids"]
    for p_, (c_prev, c) in enumerate(zip([ids[0], *ids], ids, strict=False)):
        if c == c_prev:
            assert staged[p_ + 1] == 0


def test_tick_takes_ticks_passes_and_conversations(synth, tmp_path):
    out = tmp_path / "prep.json"
    argv = [*synth["base"], "--pass-kernels", "on", "--scenario", "tick", "--ticks", "4", "--conversations", "10",
            "--plan", str(synth["plan"]), "--prepare-only", "--out", str(out)]  # fmt: skip
    assert run_planned.main(argv) == 0
    d = json.loads(out.read_text())
    assert (d["pass_list"]["scenario"], d["pass_list"]["passes"]) == ("tick:4", 4)
    assert d["conversations"] == NQ  # a pass decodes every lane-slot; the scenario parks 54 of them
    segs = d["pass_list"]["list"][0]["segments"]
    assert sum(1 for s in segs if s["entry"] != "parked") == 10
    assert [r["staged"] for r in d["prepare"]["staging_dry_run"]["passes"]] == [0] * 5


def test_a_lane_config_file_runs_as_given(synth, tmp_path):
    confs = configurations(synth["params"].vocab)
    f = tmp_path / "mixed.json"
    f.write_text(json.dumps(passmasks.dump_list(confs["mixed"][0])))
    out = tmp_path / "prep.json"
    argv = [*synth["base"], "--pass-kernels", "on", "--lane-config", str(f), "--plan", str(synth["plan"]),
            "--prepare-only", "--out", str(out)]  # fmt: skip
    assert run_planned.main(argv) == 0
    pl = json.loads(out.read_text())["pass_list"]
    assert (pl["scenario"], pl["lane_config"], pl["passes"]) == (None, str(f), 2)
    assert pl["lane_config_sha256"] == run_planned.sha256_file(f)


# ── the lane-slot order on both ends ────────────────────────────────────────────────────────────


def fake_model(params):
    lim, land = fakes.level_preset("preset")
    fhe, inf = fakes.make_runtime_fake(
        N=1 << (LOGN - 1), hid=D, dim=D, H=1, H_real=1, E=params.d_ffn, E_real=params.d_ffn, rot_steps=(),
        complex_payload=True, limit=lim, bts_out=land,
    )  # fmt: skip
    model = FheSsmModel(inf, params, core=fake.core, unit=fhe.unit, pass_kernels="on")
    run_fake._rot_band(fhe, model)
    return fhe, inf, model


def pass_run_stub(model):
    """The pieces of ``PlannedSsmRun`` its packing reads (``input_ct``'s rows, ``reply``'s unpacking)."""
    run = planned.PlannedSsmRun.__new__(planned.PlannedSsmRun)
    run.pass_mode, run.d, run.t, run.n_conv = True, model.params.d, model.geo.base.t, model.capacity
    run.client, run.bundle = None, None
    return run


@pytest.mark.parametrize("conversations", [NQ, 10, 1])
def test_block_0s_input_is_the_twins_packing_through_pi(synth, conversations):
    params = synth["params"]
    fhe, inf, model = fake_model(params)
    cfg = passmasks.scenario("tick:1", vocab=params.vocab, seed=2, conversations=conversations)[0]
    rows_q = passmasks.pass_rows(cfg, params.emb_row)  # q order; the parked lane-slots copy lane-slot 0
    assert all(np.array_equal(rows_q[q], rows_q[0]) for q in range(conversations, NQ))
    x = model.encode_input(rows_q)  # the twin: pi applied in FheSsmModel._encode
    run = pass_run_stub(model)
    ops = FheOps(inf, fake.core, fhe.unit)
    y = planned.encode_pairs(ops, run.layout_rows(rows_q), params.d, run.t, 30)
    np.testing.assert_array_equal(y.vec, x.vec)
    np.testing.assert_array_equal(x.vec, filling.pack_pairs(passmasks.to_layout(rows_q), inf.slots))
    with pytest.raises(ValueError, match="all 64 lane-slots"):
        run.layout_rows(rows_q[:-1])


def test_the_reply_round_trips_through_pi_inverse(synth):
    params = synth["params"]
    _fhe, _inf, model = fake_model(params)
    rows_q = np.random.default_rng(4).standard_normal((NQ, params.d))
    vec = model.encode_input(
        rows_q
    ).vec  # what the session would decrypt: the layout's packing of q-order rows
    run = pass_run_stub(model)
    run.ops = types.SimpleNamespace(decrypt_slots_complex=lambda _ct: vec)
    run.ct = lambda _reg: types.SimpleNamespace(level=30)
    got, frame, level = run.reply("out")
    assert (frame, level, got.shape) == (None, 30, (NQ, params.d))
    np.testing.assert_allclose(got, rows_q, rtol=0, atol=1e-12)
    run.pass_mode = False  # the off path keeps the layout's order: the same values, permuted
    np.testing.assert_allclose(run.reply("out")[0], passmasks.to_layout(rows_q), rtol=0, atol=1e-12)


# ── refusals ────────────────────────────────────────────────────────────────────────────────────


def test_the_pass_kernels_refusals(synth, tmp_path, monkeypatch):
    base = [*synth["base"], "--plan", str(synth["plan"]), "--prepare-only", "--out", str(tmp_path / "x.json")]
    with pytest.raises(SystemExit, match="free run per lane-slot"):
        run_planned.main([*base, "--pass-kernels", "on", "--tokens", "t.json", "--free-run"])
    with pytest.raises(SystemExit, match="free run per lane-slot"):
        run_planned.main([*base, "--pass-kernels", "on", "--tokens", "t.json"])
    with pytest.raises(SystemExit, match="need --pass-kernels on"):
        run_planned.main([*base, "--scenario", "mixed"])
    with pytest.raises(SystemExit, match="twice"):
        run_planned.main([*base, "--pass-kernels", "on", "--scenario", "mixed", "--lane-config", "f.json"])
    with pytest.raises(SystemExit, match="tick scenario only"):
        run_planned.main([*base, "--pass-kernels", "on", "--scenario", "mixed", "--conversations", "3"])
    passes = passmasks.scenario("mixed", vocab=synth["params"].vocab, seed=3)
    a = run_planned.parser().parse_args([*synth["base"], "--plan", "x"])
    _p, off_model, *_ = run_planned.make_twin(a)
    _p, on_model, *_ = run_planned.make_twin(run_planned.parser().parse_args([*synth["base"], "--plan", "x",
                                                                              "--pass-kernels", "on"]))  # fmt: skip
    mk = lambda model, **kw: planned.PlannedSsmRun(
        None, "x", None, [], None, model=model, work_dir=str(tmp_path), **kw
    )
    with pytest.raises(ValueError, match="pass list"):
        mk(off_model, passes=passes)
    with pytest.raises(ValueError, match="pass list"):
        mk(on_model)
    with pytest.raises(ValueError, match="every lane-slot"):
        mk(on_model, passes=passes, conversations=10)
    gen = passmasks.scenario("prefill:70", vocab=synth["params"].vocab, seed=3)
    with pytest.raises(ValueError, match="generated token"):
        mk(on_model, passes=gen[-1:])
    with pytest.raises(RuntimeError, match="pass kernels are off"):
        planned.PlannedSsmRun(None, "x", None, [], None, model=off_model, work_dir=str(tmp_path)).stage_pass(
            passes[0]
        )
    # a stream that changes between the probes (a mask depending on something but the configuration) is refused
    real = passmasks.masks
    calls = {"n": 0}

    def drifting(cfg, bp, b):
        calls["n"] += 1
        out = real(cfg, bp, b)
        out[passmasks.mask_id("sh/V", b)] = out[passmasks.mask_id("sh/V", b)] + 1e-9 * calls["n"]
        return out

    monkeypatch.setattr(passmasks, "masks", drifting)
    with pytest.raises(planned.NotStatic, match="static"):
        make_run(synth, passes, tmp_path / "drift")
    with pytest.raises(SystemExit, match="static"):
        run_planned.main([*base, "--pass-kernels", "on", "--scenario", "mixed"])
    assert json.loads((tmp_path / "x.json").read_text())["prepare"]["static_stream"] is False


def test_the_stand_in_store_refuses_what_the_device_cannot_encode():
    st = planned.FakeStore()
    v = np.arange(8.0)
    assert st.stage([("a", 30, v), ("b", 28, v.astype(np.complex128))]) == 2
    assert st.stage([("a", 30, v + 1)]) == 1
    assert (len(st), st.slots[("a", 30)], st.calls) == (2, 0, [2, 1])  # restaged in place: the same slot
    assert st.has("b", 28)
    assert not st.has("b", 30)
    for bad, why in (
        (planned.PassPt("pass/k2/Po", v), "record"),
        (planned.LeanValues(v), "digest"),
        (np.zeros((2, 2)), "shape"),
    ):
        with pytest.raises(TypeError, match=why):
            st.stage([("c", 30, bad)])
    lean = planned.FakeStore(keep_values=False)
    assert lean.stage([("a", 30, planned.LeanValues(v)), ("b", 30, v)]) == 2
    assert lean.digests[("a", 30)] == lean.digests[("b", 30)] == planned.digest(v)
    assert lean.resident == {}
