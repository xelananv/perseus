# SPDX-License-Identifier: MIT
"""``run_planned.Rwkv7PrefillRun`` on stand-in programs (no ``_pc.Program`` on a CPU-only build): the register merge
and the session arguments of one session serving the decode and the prefill programs.

* the prefill state registers are the decode ones, by state name, whatever order each program lists its inputs in;
* the prefill stage outputs are numbered after the decode registers, and ``register_levels`` covers all of them;
* the prefill programs are relabelled ``pf_block_<b>``; plaintext levels and PCMM specs are merged;
* a (level, deg) mismatch between what a pass writes and what a decode step reads stops ``prepare`` by name.
"""
import pytest

from perseus.examples.rwkv7 import run_planned as rp

STATE = ["s0", "s1", "xa", "xf"]


class Prog:
    def __init__(self, label, order, lvl=32, deg=2):
        self.label_, self.relabelled = label, None
        # input 0 the stage input, then the carried state in `order`; outputs: exit, then the four final states
        self.inputs = [("x", 32, 1, 0.0)] + [(f"impl.cache.{s}-lvl=32", lvl, deg, 0.0) for s in order]
        self.outputs = [("out", 26, 1, 0.0)] + [(f"fin.{s}", lvl, deg, 0.0) for s in order]
        self.pt_refs = [("pt_a", 30, 1, "weight"), ("pt_b", 28, 1, "weight")]
        self.order = order

    def relabel(self, lab):
        self.relabelled = lab

    def output_index(self, nm):
        return [o[0] for o in self.outputs].index(nm)


class Ref:
    def __init__(self, prog):
        self.prog = prog

    def inputs(self):
        return [i[0] for i in self.prog.inputs]


class Model:
    def __init__(self):
        from types import SimpleNamespace
        self.lin = {"b0.l1r": SimpleNamespace(t=32, n_in=2, n_out=1, Wd=__import__("numpy").zeros((4, 4)),
                                              w_scale=1.0, shift=3)}
        self.L = SimpleNamespace(M=2)

    def linears(self):
        return self.lin


def fake_run(order, n_layers=1, lvl=32, deg=2, al_tag="d"):
    run = rp.Rwkv7PlannedRun.__new__(rp.Rwkv7PlannedRun)
    run.labels = [f"block_{b}" for b in range(n_layers + 1)]
    run.progs, run.ref, run.reg, run.kv = {}, {}, {}, {}
    n = 0
    for b in range(n_layers + 1):
        p = Prog(run.labels[b], order if b < n_layers else [], lvl, deg)
        run.progs[run.labels[b]] = p
        run.ref[b] = Ref(p)
        run.reg[b] = {"kv": [], "out": []}
        for k, s in enumerate(p.order):
            run.reg[b]["kv"].append((1 + k, 1 + k, n)); n += 1
        run.reg[b]["out"].append((0, n)); n += 1
    run.n_regs = n
    run.al = {(lab, nm): f"v/{nm}" for lab, p in run.progs.items() for nm, *_ in p.pt_refs}
    run.variants, run.pool, run.pt_rows, run.dyn_rot, run.stc_cache = {}, 96, 28, [], 160
    run.pcmm_cf, run.pcmm_cf_map, run.model, run.log = 6, None, Model(), lambda *a: None
    run.rec = type("R", (), {"n_layers": n_layers})()
    run.prepare = lambda rows, argmax=False: run
    return run


def test_register_merge_and_session_args():
    dec = fake_run(STATE)
    pf = fake_run(["xf", "s1", "xa", "s0"])          # another input order: the merge is by name
    both = rp.Rwkv7PrefillRun(dec, pf, 32, log=lambda *a: None).prepare(None, None)
    d = {pf.ref[0].inputs()[i]: r for i, _j, r in pf.reg[0]["kv"]}
    e = {dec.ref[0].inputs()[i]: r for i, _j, r in dec.reg[0]["kv"]}
    assert d == e
    outs = [r for b in pf.reg for _j, r in pf.reg[b]["out"]]
    assert outs == [dec.n_regs, dec.n_regs + 1] and both.n_regs == dec.n_regs + 2
    assert pf.progs["block_0"].relabelled == "pf_block_0"
    kw = both.session_kwargs()
    assert kw["regs"] == both.n_regs and len(kw["reg_levels"]) == both.n_regs
    assert len(kw["programs"]) == 4 and kw["pt_levels"] == [30, 28] or sorted(kw["pt_levels"]) == [28, 30]
    assert kw["pcmm"] and kw["pcmm"][0][0] == "b0.l1r"


def test_landing_mismatch_stops_prepare():
    dec = fake_run(STATE)
    pf = fake_run(STATE, lvl=30)
    with pytest.raises(RuntimeError, match="land the carried state differently"):
        rp.Rwkv7PrefillRun(dec, pf, 32, log=lambda *a: None).prepare(None, None)
