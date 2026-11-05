# SPDX-License-Identifier: MIT
"""``RWKV7_SHED_TWIN`` (``run_planned.Recorder(shed=True)``, default off): the twin's recorded probe tokens drop the
value vectors only their record still references and share equal plaintext values; the recorded streams, plaintext
values, carried-state identities and the model's own values are unchanged (results/rwkv7-fhe-20261104/new_pod/
HOST_RAM_20261105.md)."""
import numpy as np

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
