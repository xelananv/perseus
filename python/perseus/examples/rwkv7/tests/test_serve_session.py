# SPDX-License-Identifier: MIT
"""``run_planned.Rwkv7ServeRun`` (continuous batching, SERVE_DESIGN.md §3): the CPU half of ONE session holding the
decode, prefill, batched and join programs.

On stand-in programs (no ``_pc.Program`` on a CPU-only build):
* the single-conversation programs' S registers ARE lane c's batched registers (``bind_lane``: the hand-over), their
  carries their own; the decode plan's own S registers are dropped and the registers numbered densely;
* the join program reads the lane-form carry registers and the single carry registers, writes the lane-form ones;
* the session arguments cover all four program sets (registers, their levels, plaintext levels);
* a (level, deg) mismatch in any hand-over direction stops ``prepare`` before a session exists, naming it;
* a join program for another block count is refused by name.
With ``--rwkv7-plan`` and the CUDA build: the same on real plans (1 block, T = 4, K = 4) compiled by Program.compile.
"""
import pytest

from perseus.examples.rwkv7 import run_planned as rp
from perseus.examples.rwkv7.tests.test_prefill_session import fake_run

K = 2
S_KEYS = ["s0", "s1"]
C_KEYS = ["xa", "xf"]


class JProg:
    def __init__(self, n_layers, lvl=32, deg=2, out_lvl=32, new_lvl=32):
        keys = [f"b{b}{k}" for b in range(n_layers) for k in C_KEYS]
        new = [f"b{b}n{k}" for b in range(n_layers) for k in C_KEYS]
        self.inputs = [(f"impl.cache.{k}-lvl=32", lvl, deg, 0.0) for k in keys] + \
            [(f"impl.cache.{k}-lvl=32", new_lvl, deg, 0.0) for k in new]
        self.outputs = [(f"out.{k}", out_lvl, deg, 0.0) for k in keys]
        self.pt_refs = [("pt_m", 31, 1, "weight"), ("pt_k", 31, 1, "weight")]
        self.relabelled = None

    def relabel(self, lab):
        self.relabelled = lab

    def output_index(self, nm):
        return [o[0] for o in self.outputs].index(nm)


class JRef:
    def __init__(self, p):
        self.p = p
        self.out_names = [o[0] for o in p.outputs]

    def inputs(self):
        return [i[0] for i in self.p.inputs]


def join_run(n_layers=1, **kw):
    from types import SimpleNamespace
    run = rp.Rwkv7JoinRun.__new__(rp.Rwkv7JoinRun)
    p = JProg(n_layers, **kw)
    run.labels, run.progs, run.ref = ["block_0"], {"block_0": p}, {0: JRef(p)}
    run.reg = {0: {"kv": [], "out": [(j, j) for j in range(len(p.outputs))]}}
    run.n_regs = len(p.outputs)
    run.model = SimpleNamespace(keys=[f"b{b}{k}" for b in range(n_layers) for k in C_KEYS],
                                new_keys=[f"b{b}n{k}" for b in range(n_layers) for k in C_KEYS],
                                L=SimpleNamespace(M=2), linears=lambda: {})
    run.al = {("block_0", "pt_m"): "join/m", ("block_0", "pt_k"): "join/keep"}
    run.variants, run.pool, run.pt_rows, run.dyn_rot, run.stc_cache = {}, 16, 0, [], 8
    run.pcmm_cf, run.pcmm_cf_map, run.log = 6, None, lambda *a: None
    run.rec = type("R", (), {"n_layers": 0})()
    run.prepare = lambda rows, argmax=False: run.map_io()
    return run


def bat_state(k=K):
    return [f"{s}c{c}" for c in range(k) for s in S_KEYS] + C_KEYS


def make(bat_lvl=32, join_kw=None, n_layers=1, join_layers=None):
    dec = fake_run(S_KEYS + C_KEYS, n_layers=n_layers)
    pf = fake_run(["xf", "s1", "xa", "s0"], n_layers=n_layers)
    bat = fake_run(bat_state(), n_layers=n_layers, lvl=bat_lvl)
    join = join_run(join_layers or n_layers, **(join_kw or {}))
    return rp.Rwkv7ServeRun(dec, pf, bat, join, T=4, K=K, log=lambda *a: None)


def test_register_map():
    sr = make().prepare(None, None, None, None)
    dec, pf, bat = sr.dec, sr.pf, sr.bat
    for c in (1, 0):
        sr.bind_lane(c)
        for s in S_KEYS:
            assert sr.dkv[0][s][2] == sr.pkv[0][s][2] == sr.bkv[0][f"{s}c{c}"][2]
    for k in C_KEYS:                     # the single carries: their own registers, shared by decode and prefill
        assert sr.dkv[0][k][2] == sr.pkv[0][k][2] != sr.bkv[0][k][2]
    used = {r for run in (dec, pf, bat) for b in run.reg for r in
            [x[2] for x in run.reg[b]["kv"]] + [x[1] for x in run.reg[b]["out"]]}
    # dense: K lanes x 2 S + 2 lane carries + 2 batched outs, 2 single carries, 2 decode outs + 2 prefill outs
    assert used == set(range(sr.n_regs)) and sr.n_regs == K * 2 + 2 + 2 + 2 + 2 + 2
    jm = sr.join_map()
    assert [(b, k) for b, k, *_ in jm] == [(0, "xa"), (0, "xf")]
    assert all(sr.join.ins[f"b0{k}"] == li and sr.join.ins[f"b0n{k}"] == ni for _b, k, li, ni, _jo in jm)
    assert bat.progs["block_0"].relabelled == "b_block_0" and sr.join.progs["block_0"].relabelled == "join_block_0"
    assert pf.progs["block_0"].relabelled == "pf_block_0"


def test_session_kwargs():
    sr = make().prepare(None, None, None, None)
    kw = sr.session_kwargs()
    assert kw["regs"] == sr.n_regs and len(kw["reg_levels"]) == sr.n_regs
    assert len(kw["programs"]) == 2 + 2 + 2 + 1          # decode, prefill, batched (block + tail), join
    assert sorted(set(kw["pt_levels"])) == [28, 30, 31]
    assert kw["pcmm"] and kw["pcmm"][0][0] == "b0.l1r"


@pytest.mark.parametrize("bad,match", [
    (dict(bat_lvl=30), "decode -> batched lane 0"),
    (dict(join_kw=dict(out_lvl=30)), "join -> batched"),
    (dict(join_kw=dict(new_lvl=30)), "decode -> join \\(natural form\\)"),
    (dict(join_kw=dict(lvl=30)), "batched -> join \\(lane form\\)"),
])
def test_landing_mismatch_stops_prepare(bad, match):
    with pytest.raises(RuntimeError, match="land the carried state differently") as e:
        make(**bad).prepare(None, None, None, None)
    assert __import__("re").search(match, str(e.value)), str(e.value)


def test_join_for_other_blocks_refused():
    with pytest.raises(RuntimeError, match="do not match"):
        make(n_layers=1, join_layers=2).prepare(None, None, None, None)


def test_serve_session_prepare(tmp_path, monkeypatch):
    """Real plans: 1-block decode, T = 4 prefill, K = 4 batched (lane carries) and K = 4 join, one shifts file,
    compiled by ``Program.compile`` (no session, no GPU); every hand-over lands where its reader reads. Needs
    --rwkv7-plan and the CUDA build's ``perseus._pc``."""
    import json
    import os
    import pathlib
    if os.environ.get("RWKV7_PLAN") != "1":
        pytest.skip("plans four times: run with --rwkv7-plan")
    pytest.importorskip("networkx")
    try:
        from perseus import _pc
        _pc.Program.compile                       # noqa: B018
    except (ImportError, AttributeError) as e:
        pytest.skip(f"perseus._pc has no Program.compile here: {e}")
    from perseus.examples.fhe_ssm.run_planned import session_config
    from perseus.examples.rwkv7 import common, ref
    from perseus.examples.rwkv7.model import load_coeffs
    from perseus.examples.rwkv7.run_capture import batch_probe, join_probe
    from perseus.examples.rwkv7.serve import make_join
    from perseus.examples.rwkv7.session import make_model
    from perseus.examples.rwkv7.tests.conftest import CKPT
    from perseus.examples.rwkv7.tests.test_capture import HIGH
    from perseus.examples.rwkv7.tests.test_prefill import _cap_and_plan
    from perseus.examples.rwkv7.tests.test_ref import COEFFS, tokenizer
    for k, v in HIGH.items():
        monkeypatch.setenv(k, v)
    T, Kb = 4, 4
    root = pathlib.Path(os.environ.get("RWKV7_B_DIR", str(tmp_path))) / "serve_prepare"
    cap_d, plan_d = _cap_and_plan(root, "dec", [], HIGH)
    sh = ["--shifts", str(cap_d / "pcmm_shifts.json")]
    _c, plan_p = _cap_and_plan(root, f"pf{T}", ["--prefill-chunk", str(T), *sh], HIGH)
    _c, plan_b = _cap_and_plan(root, f"b{Kb}", ["--batch", str(Kb), *sh], HIGH)
    _c, plan_j = _cap_and_plan(root, f"join{Kb}", ["--join", "--batch", str(Kb)], HIGH)
    shifts = json.load(open(cap_d / "pcmm_shifts.json"))
    P = ref.load_params(CKPT, 1)
    co = load_coeffs(COEFFS)
    tok = tokenizer()
    ids = tok.encode(common.CALIB_TEXT)
    _rf, st = common.prefix_state(P, COEFFS, ids, 1)
    cfg = session_config(str(plan_d))
    labels = ["block_0", "block_1"]
    _f, _i, dm = make_model(P, co, capture=True, shifts=shifts)
    _f, _i, pm = make_model(P, co, capture=True, shifts=shifts, chunk=T)
    _f, _i, bm = make_model(P, co, capture=True, shifts=shifts, batch=Kb, batch_carry="lane", prefill_keys=T)
    _f, _i, jm = make_join(P, Kb, capture=True)
    for m in (dm, pm, bm):
        m.init_state = st
    mk = dict(work_dir=None)

    def run(cls, model, plan, name, n, labs=labels):
        mk["work_dir"] = str(root / f"work_{name}")
        return cls(rp.Recorder(model, n), str(plan), cfg, labs, None, model=model, **mk)
    dec = run(rp.Rwkv7PlannedRun, dm, plan_d, "dec", 1)
    pf = run(rp.Rwkv7PlannedRun, pm, plan_p, "pf", 1)
    bat = run(rp.Rwkv7BatchedRun, bm, plan_b, "b", 1)
    join = run(rp.Rwkv7JoinRun, jm, plan_j, "j", 0, ["block_0"])
    long_ids = tok.encode(common.CALIB_TEXT_LONG)[1:]

    import types
    a = types.SimpleNamespace(prefix=1, ids=",".join(map(str, ids)), harness=None, vocab=None)
    jr = join_probe(P, COEFFS, a, jm.L, jm.n_layers)
    sr = rp.Rwkv7ServeRun(dec, pf, bat, join, T, Kb).prepare(
        rp.probe_rows(P, ids, ids[:1]), [[P.emb[t] for t in long_ids[i * T:(i + 1) * T]] for i in range(2)],
        batch_probe(P, long_ids, Kb, 2)[0], [jr, jr])
    kw = sr.session_kwargs()
    print("serve registers", sr.n_regs, "programs", len(kw["programs"]), "join ins", sorted(join.ins),
          "mask slots", join.mask_slots())
    assert len(kw["programs"]) == 2 + 2 + 2 + 1 and len(kw["reg_levels"]) == sr.n_regs
    for c in range(Kb):
        sr.bind_lane(c)
        assert all(sr.dkv[0][s][2] == sr.lane_reg(0, s, c) for s in sr.s_keys)


class _StubProgram:
    """``_pc.Program`` for the CPU: ``compile`` reads the work dir's graph and builds a program whose inputs carry the
    (level, deg) the driver passed in ``input_metas`` (with ``strict``, every one must be there; else the stage input
    enters at a stand-in meta) and whose outputs, ``live_out``, land where the carried inputs do."""

    strict = True

    def __init__(self, label, inputs, outputs, pts):
        self.label_, self.inputs, self.outputs, self.relabelled = label, inputs, outputs, None
        self.pt_refs = [(nm, 30, 1, "weight") for nm in pts]

    def relabel(self, lab):
        self.relabelled = lab

    def output_index(self, nm):
        return [o[0] for o in self.outputs].index(nm)

    def rot_steps(self):
        return []

    @staticmethod
    def compile(work_dir, plan_dir, config=None, blocks=None, live_out=None, input_metas=None, **_k):
        import json
        import os
        out, prev = {}, (36, 1)
        for lab in blocks:
            g = json.load(open(os.path.join(work_dir, f"block_{lab.split('_')[1]}", "graph.json")))
            made, ins, pts = set(), [], []
            for n in g["nodes"]:
                for x in n["inputs"]:
                    if x.startswith("pt_"):
                        pts.append(x) if x not in pts else None
                    elif not x.startswith(("const(", "rot(", "hint_lev(", "lvl_cap(")) and x not in made and x not in ins:
                        ins.append(x)
                made.add(n["output"])
            metas = dict((input_metas or {}).get(lab, {}))
            land = [m for x, m in metas.items() if "cache." in x]
            land = land[0] if land else (28, 2)            # every carried value lands at one (level, deg)
            if _StubProgram.strict:
                missing = [x for x in ins if x not in metas]
                assert not missing, f"{lab}: inputs without a meta {missing}"
            out[lab] = _StubProgram(lab, [(x, *metas.get(x, prev), 0.0) for x in ins],     # compile_dir's chain
                                    [(x, *land, 0.0) for x in (live_out or {}).get(lab, [])], pts)
            prev = land
        return out


def test_join_run_prepare_on_the_recorder(tmp_path, monkeypatch):
    """``Rwkv7JoinRun.prepare`` on the CPU twin (the join model on the recorder, ``Program.compile`` stubbed): every
    input is a named carried value with its recorded (level, deg), the outputs are the lane carries in key order, and
    the two masks are found by value (``join/m``, ``join/keep``) whatever lane is staged later."""
    import os
    import types

    from perseus import _pc
    from perseus.examples.rwkv7 import common, ref
    from perseus.examples.rwkv7.run_capture import join_probe
    from perseus.examples.rwkv7.serve import make_join
    from perseus.examples.rwkv7.tests.conftest import CKPT
    from perseus.examples.rwkv7.tests.test_ref import COEFFS, tokenizer
    if not (os.path.exists(CKPT) and os.path.exists(COEFFS)):
        pytest.skip("no checkpoint")
    monkeypatch.setattr(_pc, "Program", _StubProgram, raising=False)
    P = ref.load_params(CKPT, 2)
    _f, _i, jm = make_join(P, 4, capture=True)
    ids = tokenizer().encode(common.CALIB_TEXT)
    a = types.SimpleNamespace(prefix=1, ids=",".join(map(str, ids)), harness=None, vocab=None, text=None)
    rows = join_probe(P, COEFFS, a, jm.L, jm.n_layers)
    run = rp.Rwkv7JoinRun(rp.Recorder(jm, 0), str(tmp_path / "plan"), None, ["block_0"], None, model=jm,
                          work_dir=str(tmp_path / "work"))
    run.prepare([rows, rows])
    p = run.progs["block_0"]
    print("join ins", run.ins, "outs", run.outs, "metas", sorted({x[1:3] for x in p.inputs}), "masks", run.mask_slots())
    assert set(run.ins) == set(jm.keys) | set(jm.new_keys) and list(run.outs) == jm.keys
    assert sorted(run.outs.values()) == list(range(len(jm.keys)))
    assert len({x[1:3] for x in p.inputs}) == 1               # every carried input at the one landing
    assert {a for a, _lv in run.mask_slots()} == {"join/m", "join/keep"}


def test_device_server_register_flow(monkeypatch):
    """``DeviceServer`` on stand-in programs that record which register each input came from (provenance): a
    conversation prefilled for lane 1 writes S into lane 1's registers, its decode step reads them there, the join reads
    the lane carries (the prefix encryptions before any pass) and the single carries the decode step wrote, the batched
    pass then reads lane 1's S as the decode step left it, lane 0 from the prefix encryption and the carries the join
    wrote; after a leave, lane 1's S comes from the prefix encryption again (one encryption per state key serves
    every program and lane) and the carries from a join whose natural
    form inputs are the prefix carry's encryptions; the masks are staged for the joining / leaving lanes."""
    from types import SimpleNamespace

    import numpy as np

    from perseus.examples.rwkv7 import serve
    sr = make().prepare(None, None, None, None)
    log, regs, staged = [], {}, []

    def attach(run, tag):
        for lab, p in run.progs.items():
            def prun(ins, outs, p=p, lab=lab):
                log.append((tag, lab, [regs.get(x, x) if isinstance(x, int) else x for x in ins],
                            {p.outputs[j][0]: r for j, r in outs}))
                for j, r in outs:
                    regs[r] = (tag, lab, len(log) - 1, p.outputs[j][0])
            p.run = prun
    for run, tag in ((sr.dec, "dec"), (sr.pf, "pf"), (sr.bat, "bat"), (sr.join, "join")):
        attach(run, tag)
    L = SimpleNamespace(encode_rows_sh=lambda rows: [rows], decode_lanes_sh=lambda v, n: np.zeros((n, 4)),
                        decode_sh=lambda v: np.zeros(4))
    for run, tag in ((sr.dec, "dec"), (sr.pf, "pf"), (sr.bat, "bat")):
        names = [_state_key(run, 0, i) for i, _j, _r in run.reg[0]["kv"]]
        run.model.state_names = lambda names=names: names
        run.model.state_rows = lambda st, names=names: [[None] * len(names)] * sr.n_layers
        run.encrypt_input = lambda rows, p, i, what: (f"enc:{what}", None)
        run.input_ct = lambda t, rows, p: ["input"]
        run.zin = {b: [0] for b in run.ref}
        run.decrypt_regs = lambda regs_: ([0], 0)
        run.model.L = L
    sr.join.encrypt_input = lambda rows, p, i, what: (f"enc:{what}", None)
    sr.join.stage_masks = lambda lanes: staged.append(list(lanes))
    sr.join.model.L = L
    monkeypatch.setattr(serve, "new_carry_rows", lambda L_, n, st: {(b, k): None for b in range(n) for k in C_KEYS})
    dev = rp.DeviceServer(sr, None, log=lambda *a: None)
    assert len(dev.enc) == 4                       # s0, s1, xa, xf: one encryption each serves every reader

    def ins(entry, prog, key):
        p = prog
        names = [x[0] for x in p.inputs]
        return entry[2][names.index(f"impl.cache.{key}-lvl=32")]
    pf0, dec0, bat0, pj = (sr.pf.progs["block_0"], sr.dec.progs["block_0"], sr.bat.progs["block_0"],
                           sr.join.progs["block_0"])
    dev.prefill(1, [np.zeros(4)] * 4, True)
    e_pf = log[0]
    assert ins(e_pf, pf0, "s0") == "enc:prefix b0 s0" and e_pf[3]["fin.s0"] == sr.lane_reg(0, "s0", 1)
    dev.decode(1, np.zeros(4), False)
    e_dec = next(e for e in log if e[0] == "dec" and e[1] == "block_0")
    assert ins(e_dec, dec0, "s1")[0] == "pf" and ins(e_dec, dec0, "xa")[0] == "pf"
    assert e_dec[3]["fin.s1"] == sr.lane_reg(0, "s1", 1)
    dev.join(1)
    e_j = log[-1]
    assert e_j[0] == "join" and staged == [[1]]
    assert ins(e_j, pj, "b0xa") == "enc:prefix b0 xa" and ins(e_j, pj, "b0nxa")[0] == "dec"
    dev.batch(np.zeros((K, 4)))
    e_b = next(e for e in log if e[0] == "bat" and e[1] == "block_0")
    assert ins(e_b, bat0, "s0c1")[0] == "dec" and ins(e_b, bat0, "s0c0") == "enc:prefix b0 s0"
    assert ins(e_b, bat0, "xf")[0] == "join"
    dev.leave([1])
    e_l = log[-1]
    assert e_l[0] == "join" and staged[-1] == [1] and ins(e_l, pj, "b0nxa") == "enc:prefix b0 xa"
    assert ins(e_l, pj, "b0xa")[0] == "bat"
    n = len(log)
    dev.batch(np.zeros((K, 4)))
    e_b2 = next(e for e in log[n:] if e[0] == "bat" and e[1] == "block_0")
    assert ins(e_b2, bat0, "s0c1") == "enc:prefix b0 s0" and ins(e_b2, bat0, "s0c0")[0] == "bat"
    assert ins(e_b2, bat0, "xa")[2] == len(log[:n]) - 1          # the leave's join wrote the carries


def _state_key(run, b, i):
    return rp._state_key(run, b, i)


def test_main_serve_keyspec_on_stubs(tmp_path, monkeypatch):
    """``run_planned --serve ... --keyspec`` end to end on the CPU with ``Program.compile`` and the key source
    stubbed: arrivals read and tokenized, the four twins (decode, T = 2 prefill, K = 2 batched, join; 1 block) record
    their probes, ``Rwkv7ServeRun.prepare`` merges the registers and checks every hand-over on the recorded metas, and
    the session arguments cover the four program sets."""
    import os
    from types import SimpleNamespace

    from perseus import _pc
    from perseus.examples.fhe_ssm import run_planned as ssm_rp
    from perseus.examples.rwkv7.tests.conftest import CKPT
    from perseus.examples.rwkv7.tests.test_ref import COEFFS, HARNESS
    if not (os.path.exists(CKPT) and os.path.exists(COEFFS)):
        pytest.skip("no checkpoint")
    monkeypatch.setattr(_pc, "Program", _StubProgram, raising=False)
    monkeypatch.setattr(_pc, "KeySource", SimpleNamespace(generate=lambda *a, **k: None), raising=False)
    monkeypatch.setattr(ssm_rp, "session_config", lambda plan: None)
    monkeypatch.setattr(_StubProgram, "strict", False)
    got = {}
    monkeypatch.setattr(rp, "write_keyspec", lambda path, cfg, keys, kw, model: got.update(kw=kw, model=model) or 0)
    f = tmp_path / "arrivals.txt"
    f.write_text("0\tThe capital of Japan is\n1\tOnce upon a time, in a small village by the sea,\n")
    rc = rp.main(["--pth", CKPT, "--coeffs", COEFFS, "--harness", HARNESS, "--blocks", "1", "--serve", str(f),
                  "--batch", "2", "--prefill-chunk", "2", "--plan", "p", "--pf-plan", "p", "--b-plan", "p",
                  "--join-plan", "p", "--keyspec", str(tmp_path / "ks.json"), "--out", str(tmp_path / "o.jsonl"),
                  "--work-dir", str(tmp_path / "work")])
    kw = got["kw"]
    print("serve keyspec: programs", len(kw["programs"]), "registers", kw["regs"], "pt levels", sorted(set(kw["pt_levels"])))
    assert rc == 0 and len(kw["programs"]) == 2 + 2 + 2 + 1 and len(kw["reg_levels"]) == kw["regs"]
    assert got["model"].batch == 2 and got["model"].prefill_keys == 2
