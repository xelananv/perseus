# SPDX-License-Identifier: MIT
"""``RWKV7_LEAN_PREFIX`` / ``RWKV7_POOL_TRACE`` (run_planned.py; results/rwkv7-fhe-20261104/new_pod/
DEVICE_MEM_20261105.md): the public prefix state's encryptions are shared by every carried input that reads the same
value at the same compiled meta, and the pool instrument records the lowest free count.

The device is not needed: ``state_cts`` runs on a stand-in run whose ``encrypt`` makes a tagged object (the value and
the meta it was asked for), so the test checks what each program input would receive, input by input, with and
without sharing; the batched rows are the real twin's (``Model.state_rows`` of a K-conversation model on the 0.1B's
first block)."""
import re
from types import SimpleNamespace

import numpy as np
import pytest

from perseus.examples.rwkv7 import run_planned as rp
from perseus.examples.rwkv7.run_planned import Rwkv7PlannedRun
from perseus.examples.rwkv7.tests.test_ref import need


class FakeCt:
    def __init__(self, v, lvl, deg):
        self.v, self.level, self.deg, self.nf = np.asarray(v), lvl, deg, 2.0 ** (40 * deg)


def stand_in(names, rows, metas, n_layers):
    """A run with ``n_layers`` blocks, each carrying ``names`` (inputs 0..), compiled at ``metas[b][i]``."""
    me = SimpleNamespace()
    me.calls = 0

    def enc(v, lvl, deg):
        me.calls += 1
        return FakeCt(v, lvl, deg)

    me.encrypt = enc
    me.fhe = None
    me.client = None
    me.log = lambda *_a, **_k: None
    me.model = SimpleNamespace(state_names=lambda: names, state_rows=lambda _st: rows)
    me.rec = SimpleNamespace(n_layers=n_layers)
    me.labels = [f"block_{b}" for b in range(n_layers)]
    me.progs = {f"block_{b}": SimpleNamespace(inputs=[(f"cache.{n}-lvl", *metas[b][i]) for i, n in enumerate(names)])
                for b in range(n_layers)}
    me.ref = {b: SimpleNamespace(inputs=lambda b=b: [f"cache.{n}-lvl" for n in names]) for b in range(n_layers)}
    me.reg = {b: {"kv": [(i, i, 100 + i) for i in range(len(names))]} for b in range(n_layers)}
    # the methods under test, bound to the stand-in
    me.encrypt_input = lambda r, p, i, what: Rwkv7PlannedRun.encrypt_input(_Enc(me), r, p, i, what)
    me.state_cts = lambda st, share=None: Rwkv7PlannedRun.state_cts(me, st, share)
    return me


class _Enc:
    """``encrypt_input`` calls ``Rwkv7PlannedRun.encrypt(self, ...)``: route it to the stand-in's counter."""

    def __init__(self, me):
        self.me = me
        self.client = None
        self.fhe = SimpleNamespace(encrypt=me.encrypt)


def _packed(r):
    r = np.asarray(r)
    return np.ascontiguousarray(r.astype(np.complex128 if np.iscomplexobj(r) else np.float64).T.ravel())


def _check_inputs(me, out, rows, metas):
    """Every input received a Ct holding exactly its own packed row at its compiled (level, deg)."""
    for b, cts in out.items():
        assert sorted(cts) == list(range(len(me.model.state_names())))
        for i, ct in cts.items():
            np.testing.assert_array_equal(ct.v, _packed(rows[b][i]))
            assert (ct.level, ct.deg) == tuple(metas[b][i][:2])


def test_share_one_encryption_per_value_and_meta():
    rng = np.random.default_rng(7)
    a, b2, c = (rng.standard_normal((4, 8)) for _ in range(3))
    names = ["s0c0", "s0c1", "s0c2", "xa", "xf", "s1c0"]
    # block 0: s0 of three conversations equal (one value), xa / xf distinct, s1c0 equal to s0's VALUE but at another
    # meta (must not share); block 1: the same values at the same metas (shared across blocks: Program.run copies)
    rows = [[a, a, a, b2, c, a], [a, a, a, b2, c, a]]
    m0 = (20, 1, 2.0 ** 40)
    m1 = (21, 1, 2.0 ** 40)
    metas = [[m0, m0, m0, m0, m0, m1], [m0, m0, m0, m0, m0, m1]]
    plain = stand_in(names, rows, metas, 2)
    out = plain.state_cts(None, share=False)
    assert plain.calls == 12 and plain.prefix_encryptions == 12
    _check_inputs(plain, out, rows, metas)
    lean = stand_in(names, rows, metas, 2)
    out = lean.state_cts(None, share=True)
    # distinct (meta, value): (m0, a), (m0, b2), (m0, c), (m1, a)
    assert lean.calls == 4 and lean.prefix_encryptions == 4
    _check_inputs(lean, out, rows, metas)
    assert out[0][0] is out[0][1] is out[0][2] is out[1][0]
    assert out[0][5] is not out[0][0]                       # same value, other level: its own encryption
    assert out[0][3] is not out[0][4]


def test_share_confirms_a_digest_hit_by_value(monkeypatch):
    """Two different vectors under a colliding digest: the second gets its own encryption (array_equal decides)."""
    monkeypatch.setattr(rp, "digest", lambda v: "same")
    rng = np.random.default_rng(3)
    a, b2 = rng.standard_normal((4, 8)), rng.standard_normal((4, 8))
    m = (20, 1, 2.0 ** 40)
    me = stand_in(["s0", "s1"], [[a, b2]], [[m, m]], 1)
    out = me.state_cts(None, share=True)
    assert me.calls == 2
    _check_inputs(me, out, [[a, b2]], [[m, m]])


def test_default_is_off():
    """The default path (flag unset) encrypts every input, as before the change."""
    if rp.LEAN_PREFIX:
        pytest.skip("RWKV7_LEAN_PREFIX=1 in this environment")
    a = np.ones((4, 8))
    m = (20, 1, 2.0 ** 40)
    me = stand_in(["s0c0", "s0c1"], [[a, a]], [[m, m]], 1)
    me.state_cts(None)
    assert me.calls == 2


def test_pool_trace(monkeypatch):
    frees = iter([90, 70, 80])
    run = SimpleNamespace(sess=SimpleNamespace(pool_free=lambda: next(frees)))
    monkeypatch.setattr(rp, "POOL_TRACE", False)
    rp.pool_mark(run, "x")
    assert getattr(run, "pool_low", None) is None              # off: no call, no field
    monkeypatch.setattr(rp, "POOL_TRACE", True)
    for w in ("a", "b", "c"):
        rp.pool_mark(run, w)
    assert run.pool_low == (70, "b")
    other = SimpleNamespace(pool_low=(75, "z"))
    assert rp.pool_low_of(run, None, other) == [70, "b"]
    assert rp.pool_low_of(SimpleNamespace()) is None


@need
@pytest.mark.parametrize("carry,layout", [("lane", "current"), ("decode", "current"), ("lane", "filled")])
def test_batched_twin_rows_collapse(carry, layout):
    """The real batched twin's prefix rows (K = 4, the 0.1B's first block): every conversation starts from the same
    public state, so the K x M state rows and the per-conversation carries collapse to one value per state name."""
    from perseus.examples.rwkv7 import common, ref
    from perseus.examples.rwkv7.model import load_coeffs
    from perseus.examples.rwkv7.session import make_model
    from perseus.examples.rwkv7.tests.conftest import CKPT
    from perseus.examples.rwkv7.tests.test_ref import COEFFS, tokenizer
    K = 4
    P = ref.load_params(CKPT, 1)
    ids = tokenizer().encode("\n\nHello")
    _rf, st = common.prefix_state(P, COEFFS, ids, 1)
    _f, _i, m = make_model(P, load_coeffs(COEFFS), capture=False, batch=K, batch_carry=carry, layout=layout)
    names, rows = m.state_names(), m.state_rows(st)
    assert len(rows[0]) == len(names)
    distinct = {}
    for n, r in zip(names, rows[0]):
        base = re.sub(r"c\d+$", "", n)
        distinct.setdefault(base, set()).add(_packed(r).tobytes())
    print(carry, layout, len(names), "inputs per block;", {k: len(v) for k, v in distinct.items()})
    assert all(len(v) == 1 for v in distinct.values())        # one value per state name
    n_inputs = len(names)
    n_shared = len(distinct)
    assert n_shared < n_inputs
    M = m.L.M
    assert n_inputs == (K * M + (2 if carry == "lane" else 2 * K)) * m.L.S
    assert n_shared == (M + 2) * m.L.S


@need
@pytest.mark.parametrize("layout", ["current", "filled"])
def test_batched_twin_metas_collapse(layout):
    """Review L2: sharing needs equal compiled metas as well as equal values. The carried inputs' compiled (level,
    deg) are the twin record's (``fhe_ssm/planned.py``: ``metas = {label: {name: r.metas[name]}}`` -> ``Program.compile(
    input_metas=...)``), so they are checked here on the real K = 4 batched twin (the 0.1B's first block, two public
    passes, as ``main_batch`` records them): every conversation's input of one state name is recorded at one (level,
    deg). The compiled nf is the plan's and is not visible on the CPU: ``state_cts`` keys on it anyway (a differing nf
    only costs sharing, never correctness), and the batched header's ``prefix_encryptions`` (written under
    ``RWKV7_LEAN_PREFIX`` or ``RWKV7_POOL_TRACE``) gives the device count on the first run (expected (M + 2) x blocks:
    0.4B filled 72, current 96)."""
    from perseus.examples.rwkv7 import common, ref
    from perseus.examples.rwkv7.model import load_coeffs
    from perseus.examples.rwkv7.run_capture import batch_probe
    from perseus.examples.rwkv7.run_planned import Recorder
    from perseus.examples.rwkv7.session import make_model
    from perseus.examples.rwkv7.tests.conftest import CKPT
    from perseus.examples.rwkv7.tests.test_ref import COEFFS, tokenizer
    K = 4
    P = ref.load_params(CKPT, 1)
    tok = tokenizer()
    ids = tok.encode("\n\nHello")
    _rf, st = common.prefix_state(P, COEFFS, ids, 1)
    _f, _i, model = make_model(P, load_coeffs(COEFFS), capture=True, batch=K, batch_carry="lane", layout=layout)
    model.init_state = st
    rec = Recorder(model, model.n_layers)
    rows, _t = batch_probe(P, tok.encode(common.CALIB_TEXT_LONG)[1:], K, 2)
    recs = {t: rec.token(t, rows[t]) for t in (0, 1)}
    r, prev = recs[1][0], recs[0][0]
    ins = r.inputs()
    kv = [ins[i] for i in range(len(ins)) if ins[i] in r.objs and id(r.objs[ins[i]]) in prev.final]
    names = [re.search(r"cache\.([a-z0-9]+)-lvl", n).group(1) for n in kv]
    assert len(kv) == K * model.L.M + 2                       # the carried inputs: K x M states + 2 lane carries
    groups = {}
    for n, nm in zip(kv, names):
        groups.setdefault(re.sub(r"c\d+$", "", nm), set()).add(tuple(r.metas[n]))
    print(layout, {k: sorted(v) for k, v in groups.items()})
    assert len(groups) == model.L.M + 2
    assert all(len(v) == 1 for v in groups.values())          # one (level, deg) per state name
