# SPDX-License-Identifier: MIT
"""The planned driver's CPU half on a 1-block twin: the recorder records two probe tokens from the public prefix state,
every block's carried state (S_0, S_1, xa, xf) is found among the reference token's inputs exactly as
``PlannedRun.prepare`` matches it (``prev.final[id(obj)]``), the stream is the same op for op across the probes
(``diff_block``: no dynamic step), and the plaintexts are equal (the static path)."""
import numpy as np

from perseus.examples.rwkv7 import common, ref
from perseus.examples.rwkv7.model import load_coeffs
from perseus.examples.rwkv7.run_planned import Recorder
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
    recs = {t: rec.token(t, P.emb[0]) for t in (0, 1)}
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
