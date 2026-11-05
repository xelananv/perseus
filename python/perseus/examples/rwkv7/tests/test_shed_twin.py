# SPDX-License-Identifier: MIT
"""``RWKV7_SHED_TWIN`` (``run_planned.Recorder(shed=True)``, default off): the twin's recorded probe tokens drop the
value vectors only their record still references and share equal plaintext values; the recorded streams, plaintext
values, carried-state identities and the model's own values are unchanged (results/rwkv7-fhe-20261104/new_pod/
HOST_RAM_20261105.md)."""
import numpy as np
import pytest

from perseus.examples.rwkv7 import common, ref
from perseus.examples.rwkv7 import run_planned as rp
from perseus.examples.rwkv7.model import load_coeffs
from perseus.examples.rwkv7.run_planned import Recorder, probe_rows
from perseus.examples.rwkv7.session import make_model
from perseus.examples.rwkv7.tests.conftest import CKPT
from perseus.examples.rwkv7.tests.test_ref import COEFFS, need, tokenizer


def test_sole_ref_counts_the_dicts_own_names():
    class C:
        pass
    a, b = C(), C()
    d = {"x": C(), "y": a, "y2": a, "z": b}
    hold = b
    assert rp._sole_ref(d, "x")
    assert not rp._sole_ref(d, "y", 2)              # the local ``a`` holds it too
    del a
    assert not rp._sole_ref(d, "y")                 # twice in d: not sole at times = 1
    assert rp._sole_ref(d, "y", 2)
    assert not rp._sole_ref(d, "z")
    del hold


def test_default_is_off():
    """Off unless the environment asks (read once, at import)."""
    import os
    assert rp.SHED_TWIN is (os.environ.get("RWKV7_SHED_TWIN", "0") == "1")


def _record(P, st, rows, shed):
    _f, _i, model = make_model(P, load_coeffs(COEFFS), capture=True)
    model.init_state = st
    rec = Recorder(model, model.n_layers, shed=shed)
    recs = {t: rec.token(t, rows[t]) for t in (0, 1)}
    return rec, recs


def _held(recs):
    seen, n = set(), 0
    for rr in recs.values():
        for r in rr.values():
            for o in r.objs.values():
                v = getattr(o, "vec", None)
                if isinstance(v, np.ndarray) and id(v) not in seen:
                    seen.add(id(v))
                    n += v.nbytes
            for v in r.pts.values():
                if isinstance(v, np.ndarray) and id(v) not in seen:
                    seen.add(id(v))
                    n += v.nbytes
    return n


def _kv(recs):
    """``PlannedRun.prepare``'s carried-state match: (input index, the previous token's name) per block."""
    out = {}
    for b, r in recs[1].items():
        prev = recs[0][b]
        out[b] = [(i, prev.final[id(r.objs[n])]) for i, n in enumerate(r.inputs())
                  if n in r.objs and id(r.objs[n]) in prev.final]
    return out


@need
def test_shed_twin_records_the_same_streams_and_frees_the_vectors():
    P = ref.load_params(CKPT, 1)
    tok = tokenizer()
    ids = tok.encode("\n\nHello")
    _rf, st = common.prefix_state(P, COEFFS, ids, 1)
    rows = probe_rows(P, tok.encode(common.CALIB_TEXT), ids[:1])
    _r0, full = _record(P, st, rows, shed=False)
    held_full = _held(full)
    rec, shed = _record(P, st, rows, shed=True)
    held_shed = _held(shed)
    print("held bytes: unshed", held_full, "shed", held_shed, "stats", rec.shed_stats)
    for t in (0, 1):
        for b in full[t]:
            a, s = full[t][b], shed[t][b]
            assert a.nodes == s.nodes, (t, b)
            assert a.metas == s.metas
            assert a.out_names == s.out_names
            assert set(a.pts) == set(s.pts)
            assert all(np.array_equal(a.pts[k], s.pts[k]) for k in a.pts)
            assert sorted(a.objs) == sorted(s.objs)
            # the model's own values: the stage outputs (read by the next stage and the next token) are untouched
            ao = a.out if isinstance(a.out, (list, tuple)) else [a.out]
            so = s.out if isinstance(s.out, (list, tuple)) else [s.out]
            for x, y in zip(ao, so):
                assert x.vec.shape == y.vec.shape and np.array_equal(x.vec, y.vec)
    assert _kv(full) == _kv(shed)
    n_state = len(rec.model.state_names()) if hasattr(rec.model, "state_names") else None
    assert all(len(v) == (n_state or len(v)) for b, v in _kv(shed).items() if b < rec.n_layers)
    # what is kept: the outputs, inputs and carried state (a few vectors per block); the plaintexts once per value
    assert held_shed < 0.08 * held_full, (held_shed, held_full)
    assert rec.shed_stats["vectors"] > 1000
    assert all(not v.flags.writeable for v in rp._PT_INTERN.values())


def test_header_names_the_shed_only_when_on():
    from types import SimpleNamespace
    sess = SimpleNamespace(has_secret=False)
    base = {"key_mode": lambda: "bundle", "sess": sess, "client": object(), "bundle": "b"}
    off = rp.keys_header(SimpleNamespace(**base, rec=SimpleNamespace(shed=False)))
    on = rp.keys_header(SimpleNamespace(**base, rec=SimpleNamespace(shed=True, shed_stats={"vectors": 3})))
    assert "shed_twin" not in off
    assert on["shed_twin"]["vectors"] == 3 and "pt_distinct" in on["shed_twin"]


# ── the products of ``prepare`` (what reaches the device), shed against unshed, on stand-in programs ──

class _Prog:
    """``test_serve_session._StubProgram`` (inputs, outputs, ``pt_refs`` from the work dir's graph) plus the two calls
    ``pt_items`` makes (``pt_keys``, ``alias_pts``)."""

    @staticmethod
    def compile(*a, **k):
        from perseus.examples.rwkv7.tests.test_serve_session import _StubProgram
        out = _StubProgram.compile(*a, **k)
        for p in out.values():
            p.alias = {}
            p.pt_keys = _pt_keys(p)
            p.alias_pts = p.alias.update
        return out


def _pt_keys(q):
    def keys():
        return [(f"{q.label_}/{nm}", lv) for nm, lv, _d, _f in q.pt_refs]
    return keys


def _products(run, out_vecs=True):
    """Everything ``prepare`` hands the device side, as plain data; plus (``out_vecs``) the reference token's stage
    output vectors and carried-state vectors, the values a wrongly shed live vector would change."""
    def prog(p):
        return {"label": p.label_, "inputs": [tuple(x) for x in p.inputs], "outputs": [tuple(x) for x in p.outputs],
                "pt_refs": [tuple(x) for x in p.pt_refs]}
    items = run.pt_items()
    kw = run.session_kwargs()
    d = {"al": dict(run.al), "kv": {b: list(v) for b, v in run.kv.items()}, "reg": run.reg,
         "progs": {lab: prog(p) for lab, p in run.progs.items()},
         "pt_items": {k: np.array(v) for k, v in items.items()},
         "alias": {lab: dict(p.alias) for lab, p in run.progs.items()},
         "kwargs": {k: ([prog(p) for p in v] if k == "programs" else v) for k, v in kw.items()}}
    if out_vecs:
        vecs = {}
        for b, r in run.ref.items():
            outs = r.out if isinstance(r.out, (list, tuple)) else [r.out]
            vecs[(b, "out")] = [np.array(o.vec) for o in outs]
            vecs[(b, "state")] = [np.array(r.objs[r.inputs()[i]].vec) for i, _nm in run.kv[b]]
        d["vecs"] = vecs
    return d


def _diff(a, b, path="") -> list:
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        return [] if a.shape == b.shape and np.array_equal(a, b) else [path]
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return [f"{path} keys"]
        return [x for k in a for x in _diff(a[k], b[k], f"{path}/{k}")]
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return [f"{path} len"]
        return [x for i, (u, v) in enumerate(zip(a, b)) for x in _diff(u, v, f"{path}[{i}]")]
    return [] if a == b else [path]


def _prepared(kind, shed, tmp, sabotage=False, monkeypatch=None):
    import gc
    from types import SimpleNamespace

    from perseus.examples.rwkv7.run_capture import batch_probe, join_probe
    from perseus.examples.rwkv7.serve import make_join
    tok = tokenizer()
    work = str(tmp / f"{kind}_{int(shed)}_{int(sabotage)}")
    if kind == "join":
        P = ref.load_params(CKPT, 2)
        _f, _i, m = make_join(P, 4, capture=True)
        ids = tok.encode(common.CALIB_TEXT)
        a = SimpleNamespace(prefix=1, ids=",".join(map(str, ids)), harness=None, vocab=None, text=None)
        rows = join_probe(P, COEFFS, a, m.L, m.n_layers)
        run = rp.Rwkv7JoinRun(Recorder(m, 0, shed=shed), str(tmp / "plan"), None, ["block_0"], None, model=m,
                              work_dir=work)
        run.prepare([rows, rows])
    else:
        P = ref.load_params(CKPT, 1)
        ids = tok.encode("\n\nHello")
        _rf, st = common.prefix_state(P, COEFFS, ids, 1)
        kw = {"batch": 2, "batch_carry": "lane"} if kind == "batch" else {}
        _f, _i, m = make_model(P, load_coeffs(COEFFS), capture=True, **kw)
        m.init_state = st
        cls = rp.Rwkv7BatchedRun if kind == "batch" else rp.Rwkv7PlannedRun
        run = cls(Recorder(m, m.n_layers, shed=shed), str(tmp / "plan"), None, ["block_0", "block_1"], None,
                  model=m, work_dir=work)
        if kind == "batch":
            rows, _t = batch_probe(P, tok.encode(common.CALIB_TEXT_LONG)[1:], 2, 2)
        else:
            rows = probe_rows(P, tok.encode(common.CALIB_TEXT), ids[:1])
        run.prepare(rows)
    out = _products(run)
    stats = dict(run.rec.shed_stats)
    del run, m
    gc.collect()
    return out, stats



@need
@pytest.mark.parametrize("kind", ["decode", "batch", "join"])
def test_prepare_products_equal_shed_and_unshed(kind, tmp_path, monkeypatch):
    """``prepare``'s products (aliases, staged plaintext values, carried-state map, registers, the programs' inputs,
    outputs and ``pt_refs``, the session arguments) and the reference token's output and carried-state vectors are
    identical with and without the shed, for decode, batched K = 2 and the join (serve) path; and a sabotaged shed
    that also drops LIVE vectors is caught by the same comparison (non-vacuous)."""
    from perseus import _pc
    from perseus.examples.rwkv7.tests.test_serve_session import _StubProgram
    monkeypatch.setattr(_pc, "Program", _Prog, raising=False)
    monkeypatch.setattr(_StubProgram, "strict", kind == "join")
    off, _s0 = _prepared(kind, False, tmp_path)
    on, st = _prepared(kind, True, tmp_path)
    assert st["vectors"] > (8 if kind == "join" else 1000), st          # the join stage is small (16 shed)
    d = _diff(off, on)
    assert not d, d[:8]
    assert off["pt_items"] and off["kv"] is not None
    # sabotage 1, the comparison alone: at the reference token's end every vector counts as dead, the live outputs
    # and carried state included (the last probe token, so nothing reads them afterwards and nothing crashes)
    real_token, real_sole = Recorder.token, rp._sole_ref

    def token(self, t, row, argmax=False):
        blocks = real_token(self, t, row, argmax)
        if t == 1:
            rp._sole_ref = lambda *_a, **_k: True
            try:
                for r in blocks.values():
                    self._shed(r, final=True)
            finally:
                rp._sole_ref = real_sole
        return blocks
    monkeypatch.setattr(Recorder, "token", token)
    bad, _s = _prepared(kind, True, tmp_path, sabotage=True)
    caught = _diff(off, bad)
    print(kind, "sabotage (live vectors at token 1's end) caught by", caught[:3])
    assert caught
    monkeypatch.setattr(Recorder, "token", real_token)
    # sabotage 2: every vector counts as dead from the first block on; the twin then computes on shed live vectors
    monkeypatch.setattr(rp, "_sole_ref", lambda *_a, **_k: True)
    try:
        bad, _s = _prepared(kind, True, tmp_path, sabotage=True)
        caught = _diff(off, bad)
    except Exception as e:                      # noqa: BLE001 (a crash on a shed live vector also counts)
        caught = [repr(e)[:120]]
    print(kind, "sabotage (all from block 0) caught by", caught[:3])
    assert caught
