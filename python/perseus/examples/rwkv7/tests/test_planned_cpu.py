# SPDX-License-Identifier: MIT
"""The planned driver's CPU half on a 1-block twin: the recorder records two probe tokens from the public prefix state,
every block's carried state (S_0, S_1, xa, xf) is found among the reference token's inputs exactly as
``PlannedRun.prepare`` matches it (``prev.final[id(obj)]``), the stream is the same op for op across the probes
(``diff_block``: no dynamic step), and the plaintexts are equal (the static path)."""
import numpy as np

from perseus.examples.rwkv7 import common, ref
from perseus.examples.rwkv7.model import load_coeffs
from perseus.examples.rwkv7.run_planned import Recorder, probe_rows
from perseus.examples.rwkv7.session import make_model
from perseus.examples.rwkv7.tests.conftest import CKPT
from perseus.examples.rwkv7.tests.test_ref import COEFFS, need, tokenizer
from perseus.impl.driver import diff_block


@need
def test_twin_records_static_stream_with_carried_state():
    P = ref.load_params(CKPT, 1)
    ids = tokenizer().encode("\n\nHello")
    _rf, st = common.prefix_state(P, COEFFS, ids, 1)
    _fhe, _inf, model = make_model(P, load_coeffs(COEFFS), capture=True)
    model.init_state = st
    rec = Recorder(model, model.n_layers)
    rows = probe_rows(P, tokenizer().encode(common.CALIB_TEXT), ids[:1])
    recs = {t: rec.token(t, rows[t]) for t in (0, 1)}
    ref_t, prev = recs[1], recs[0]
    for b, r in ref_t.items():
        ins = r.inputs()
        kv = [(i, prev[b].final[id(r.objs[n])]) for i, n in enumerate(ins)
              if n in r.objs and id(r.objs[n]) in prev[b].final]
        names = [ins[i] for i, _ in kv]
        print(b, len(r.nodes), "nodes; carried inputs", names)
        if b < model.n_layers:
            assert len(kv) == model.L.M + 2
            assert sorted(n.split("cache.")[1].split("-")[0] for n in names) == ["s0", "s1", "xa", "xf"]
        s, k, _z = diff_block(r, prev[b])
        assert not s and not k
        assert set(r.pts) == set(prev[b].pts)
        assert all(np.array_equal(r.pts[x], prev[b].pts[x]) for x in r.pts)


@need
def test_prefix_state_encrypts_at_the_compiled_meta():
    """``Rwkv7PlannedRun.encrypt_input`` (the prefix state and block 0's input on the device): the slot vector is
    ``pack_tokens``'s layout (checked against the twin's own pack), and the Ct is encrypted at the compiled input's
    level AND degree (``Program.run`` refuses any other; ``pack_tokens`` is degree 1 only). The recorded state inputs'
    (level, deg) are printed: those are what the device programs are compiled at (``input_metas``)."""
    from types import SimpleNamespace
    from perseus.examples.rwkv7.run_planned import Rwkv7PlannedRun
    P = ref.load_params(CKPT, 1)
    tok = tokenizer()
    ids = tok.encode(common.DEFAULT_TEXT)
    _rf, st = common.prefix_state(P, COEFFS, ids, 1)
    fhe, _inf, model = make_model(P, load_coeffs(COEFFS), capture=True)
    model.init_state = st
    rec = Recorder(model, model.n_layers)
    rows = probe_rows(P, tok.encode(common.CALIB_TEXT), ids[:1])
    recs = {t: rec.token(t, rows[t]) for t in (0, 1)}
    r = recs[1][0]
    ins = r.inputs()
    kv = [i for i, n in enumerate(ins) if n in r.objs and id(r.objs[n]) in recs[0][0].final]
    print("block 0 input", ins[0], r.metas.get(ins[0]), "state inputs", [(ins[i], r.metas[ins[i]]) for i in kv])
    assert len(kv) == model.L.M + 2

    calls = []

    class Fhe:
        def encrypt(self, v, lvl, deg):
            calls.append((np.asarray(v), lvl, deg))
            return SimpleNamespace(level=lvl, deg=deg, nf=2.0 ** (40 * deg))

    srows = model.state_rows(st)[0]
    for k, rr in enumerate(srows):
        lvl, deg = 3 + k, 1 + k % 2
        p = SimpleNamespace(inputs=[("x", lvl, deg, 2.0 ** (40 * deg))])
        me = SimpleNamespace(fhe=Fhe())
        _ct, m = Rwkv7PlannedRun.encrypt_input(me, rr, p, 0, "test")
        v, l2, d2 = calls[-1]
        assert (l2, d2) == (lvl, deg) and m[:2] == (lvl, deg)
        want = np.real(np.asarray(model.rt.ops.pack_tokens(np.asarray(rr), model.rt.ops.bts_out()).vec))
        assert v.shape == want.shape and np.array_equal(v, want)
    p = SimpleNamespace(inputs=[("x", 3, 2, 2.0 ** 41)])                 # a compiled scale the fresh Ct lacks
    import pytest
    with pytest.raises(RuntimeError):
        Rwkv7PlannedRun.encrypt_input(SimpleNamespace(fhe=Fhe()), srows[0], p, 0, "test")


def _probe(P, st, row):
    _fhe, _inf, model = make_model(P, load_coeffs(COEFFS), capture=True)
    model.init_state = st
    rec = Recorder(model, model.n_layers)
    with np.errstate(over="raise", invalid="raise"):
        for t in (0, 1):
            rec.token(t, row[t])


@need
def test_probe_rows_stay_finite():
    """The setup's probe rows (``run_planned.probe_rows``: the public calibration text) keep the twin's values finite
    from the prefix state; the old row ``emb[0]`` (an untrained token) overflowed to NaN (box run 1's warnings)."""
    import pytest
    P = ref.load_params(CKPT, 1)
    tok = tokenizer()
    _rf, st = common.prefix_state(P, COEFFS, tok.encode(common.DEFAULT_TEXT), 1)
    rows = probe_rows(P, tok.encode(common.CALIB_TEXT), [0])
    assert not any(np.array_equal(r, P.emb[0]) for r in rows)
    _probe(P, st, rows)                                   # raises FloatingPointError on an overflow
    with pytest.raises(FloatingPointError):
        _probe(P, st, [P.emb[0]] * 2)
