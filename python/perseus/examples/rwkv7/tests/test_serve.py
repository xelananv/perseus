# SPDX-License-Identifier: MIT
"""Continuous batching (serve.py): conversations prefilled in single-conversation mode, joined to a free lane of the
running batch (lane carries), freed when they end, against each conversation decoded ALONE from its prompt.

The schedule (K = 3 lanes unless stated): five conversations arrive at passes 0, 0, 0, 1, 9 with prompts of 5, 3, 8,
4, 6 tokens; T = 4, so a prompt is read as 1-2 prefill passes plus 0-2 decode steps (prompt 1: decode only; prompts
2 and 3: whole passes only); each then generates 2 tokens in the batch. Conversation 3 waits for a lane (it arrives
while all three are busy and joins lane 0 a pass later), lanes are reused, freed lanes run idle from the public prefix
state, and the clock jumps over the empty passes before conversation 4.

* every conversation, at every position (prefill lanes, decode steps, batched passes): hidden state and greedy picks
  equal the same conversation decoded alone from the public prefix state (mirror: bit for bit; fake / recorder:
  float64 rounding, and top-1 against its own poly model at every position);
* a leave followed by idle passes equals a fresh conversation from the public prefix state (every idle segment of
  every lane against one plaintext-free trajectory);
* non-vacuity: the conversations differ; a join onto the wrong lane breaks the equality; a leave that does not reset
  breaks the idle check.
2 blocks of the real checkpoint, 2^15 slots (t = 32 lanes).
"""
import numpy as np
import pytest

from perseus.examples.rwkv7 import common, ref
from perseus.examples.rwkv7.model import Mirror, load_coeffs
from perseus.examples.rwkv7.serve import FakeBackend, MirrorBackend, Server, read_arrivals, serve_loop
from perseus.examples.rwkv7.session import make_model
from perseus.examples.rwkv7.tests.test_batched import TEXT
from perseus.examples.rwkv7.tests.test_mirror import N_LAYERS, params2, rel_rms  # noqa: F401
from perseus.examples.rwkv7.tests.test_ref import COEFFS, need, tokenizer

LENS = [5, 3, 8, 4, 6]
ARRIVE = [0, 0, 0, 1, 9]
GEN = 2


def _setup(P):
    ids = tokenizer().encode(TEXT)
    _rf, st = common.prefix_state(P, COEFFS, ids, 1)
    body = ids[1:]
    arrivals = [(a, body[7 * k:7 * k + n]) for k, (a, n) in enumerate(zip(ARRIVE, LENS))]
    head = ref.RefModel(P, ref.Sites("poly", COEFFS, clamp=True)).head
    return st, arrivals, (lambda h: int(np.argmax(head(np.asarray(h, dtype=np.float64))))), body[0]


def _alone(step, P, prompt, n, pick):
    """A conversation decoded alone: the prompt token by token, then greedy picks fed back."""
    toks, hs = list(prompt), []
    for i in range(n):
        if i >= len(toks):
            toks.append(pick(hs[-1]))
        hs.append(np.asarray(step(P.emb[toks[i]])))
    return hs, toks


def _check(convs, idle, alone, pick, P, tol):
    """Max hidden error per conversation against its alone run (tokens must match), and per idle segment against
    the idle trajectory (``alone(idle prompt)``)."""
    errs = []
    for cv in convs:
        hs, toks = alone(cv.prompt, len(cv.prompt) + GEN)
        assert toks == cv.toks, (cv.k, toks, cv.toks)
        assert len(cv.hidden) == len(hs)
        errs.append(max(rel_rms(a, b) for a, b in zip(cv.hidden, hs)))
    n_idle = max((len(s) for segs in idle.values() for s in segs), default=0)
    ierrs = []
    if n_idle:
        first = next(s for segs in idle.values() for s in segs)[0][1]
        hs, toks = alone([first], n_idle)
        for c, segs in idle.items():
            for s in segs:
                assert [t for _q, t, _h in s] == toks[: len(s)], (c, s)
                ierrs.append(max(rel_rms(h, hs[i]) for i, (_q, _t, h) in enumerate(s)))
    return errs, ierrs


def _mirror_alone(P, co, st, pick):
    def alone(prompt, n):
        m = Mirror(P, co, 1 << 15, wbits=None)
        m.reset(st)
        return _alone(lambda row: m.step(row)[0], P, prompt, n, pick)
    return alone


@need
@pytest.mark.parametrize("K,T", [(3, 4), (2, 1), (3, 0)])
def test_mirror_serve_equals_alone(params2, K, T):  # noqa: F811
    co = load_coeffs(COEFFS)
    st, arrivals, pick, idle_tok = _setup(params2)
    srv = Server(MirrorBackend(params2, co, st, 1 << 15, K), K)
    ev = []
    convs, idle = serve_loop(srv, arrivals, K=K, T=T, generate=GEN, emb=params2.emb, pick=pick, idle_token=idle_tok,
                             emit=ev.append)
    alone = _mirror_alone(params2, co, st, pick)
    errs, ierrs = _check(convs, idle, alone, pick, params2, 0.0)
    kinds = [e["kind"] for e in ev]
    leaves = [(c, e["at_pass"]) for e in ev if e["kind"] == "leave" for c in e["lanes"]]
    post_leave = sum(1 for c in idle for s in idle[c] for q, _t, _h in s if any(c == lc and lq <= q for lc, lq in leaves))
    print(f"K={K} T={T}: passes {kinds.count('pass')}, prefill {kinds.count('prefill')}, decode {kinds.count('decode')}, "
          f"joins {kinds.count('join')}, leave runs {kinds.count('leave')}; hidden max {max(errs):.1e}, idle "
          f"segments {[len(s) for c in idle for s in idle[c]]} max {max(ierrs, default=0):.1e}; lanes "
          f"{[(cv.k, cv.lane, cv.joined, cv.left) for cv in convs]}")
    assert max(errs) < 1e-10 and max(ierrs, default=0.0) < 1e-10
    assert kinds.count("join") == len(arrivals) and post_leave > 0
    if T:
        assert kinds.count("prefill") > 0
    # the schedule exercised a queue (a conversation joined after its arrival pass) and lane reuse
    assert any(cv.joined > cv.arrival for cv in convs) and len({cv.lane for cv in convs}) < len(convs)
    # non-vacuity: the conversations differ ...
    assert rel_rms(convs[0].hidden[1], convs[1].hidden[1]) > 1e-2
    # ... a join onto the wrong lane breaks the equality ...
    bad = Server(MirrorBackend(params2, co, st, 1 << 15, K), K)
    bad.mask_lane = lambda c: (c + 1) % K
    bconvs, _ = serve_loop(bad, arrivals, K=K, T=T, generate=GEN, emb=params2.emb, pick=pick, idle_token=idle_tok)
    assert max(rel_rms(a, b) for cv, bc in zip(convs, bconvs) for a, b in zip(cv.hidden, bc.hidden)) > 1e-3
    # ... and a leave that does not reset the lane breaks the idle check
    nor = Server(MirrorBackend(params2, co, st, 1 << 15, K), K)
    nor.leave_resets = False
    _c, nidle = serve_loop(nor, arrivals, K=K, T=T, generate=GEN, emb=params2.emb, pick=pick, idle_token=idle_tok)
    worst = 0.0
    hs, _t = alone([idle_tok], max(len(s) for c in nidle for s in nidle[c]))
    for c in nidle:
        for s in nidle[c]:
            worst = max(worst, max(rel_rms(h, hs[i]) for i, (_q, _t2, h) in enumerate(s)))
    assert worst > 1e-3


@need
@pytest.mark.parametrize("capture", [False, True])
def test_fake_serve_equals_alone(params2, capture):  # noqa: F811
    """The same schedule on the fake (runtime level rules, PCMM fake, every carry and join landed by a refresh) and on
    the capture recorder; each conversation against itself decoded alone on a fresh fake, and top-1 against its own
    poly model at every position."""
    K, T = 3, 4
    co = load_coeffs(COEFFS)
    st, arrivals, pick, idle_tok = _setup(params2)
    fhe, _inf, m = make_model(params2, co, capture=capture, n_layers=N_LAYERS, batch=K, batch_carry="lane",
                              prefill_keys=T)
    if capture:
        fhe.magnitude_suppressed = True
    srv = Server(FakeBackend(m, st, T), K)
    convs, idle = serve_loop(srv, arrivals, K=K, T=T, generate=GEN, emb=params2.emb, pick=pick, idle_token=idle_tok)

    def alone(prompt, n):
        f, _i, dm = make_model(params2, co, capture=capture, n_layers=N_LAYERS)
        if capture:
            f.magnitude_suppressed = True
        dm.start(st)
        pos = [1]

        def step(row):
            h, _o = dm.step(row, pos[0])
            pos[0] += 1
            return h
        return _alone(step, params2, prompt, n, pick)
    errs, ierrs = _check(convs, idle, alone, pick, params2, 1e-9)
    agree, e_poly = [], []
    for cv in convs:
        rf = ref.RefModel(params2, ref.Sites("poly", COEFFS, clamp=True))
        rf.step(tokenizer().encode(TEXT)[0])
        for i, t in enumerate(cv.toks):
            lr, hr = rf.step(t)
            agree.append(int(np.argmax(rf.head(cv.hidden[i])) == np.argmax(lr)))
            e_poly.append(rel_rms(cv.hidden[i], hr))
    print(("capture" if capture else "fake"), f"serve vs alone max {max(errs):.1e}, idle max {max(ierrs, default=0):.1e},"
          f" top-1 {sum(agree)}/{len(agree)}, vs poly max {max(e_poly):.1e}, ops {dict(srv.counts)}")
    assert max(errs) < 1e-9 and max(ierrs, default=0.0) < 1e-9
    assert all(agree)


def test_read_arrivals(tmp_path):
    p = tmp_path / "serve.txt"
    p.write_text("# pass<TAB>prompt\n0\tHello\\nworld\n\n3 \\n\\nAlready prefixed\n12\tback\\\\slash\n")
    assert read_arrivals(str(p)) == [(0, "\n\nHello\nworld"), (3, "\n\nAlready prefixed"), (12, "\n\nback\\slash")]
    p.write_text("x\tno pass\n")
    with pytest.raises(ValueError, match="no arrival pass"):
        read_arrivals(str(p))


def test_serve_loop_needs_generate():
    with pytest.raises(ValueError, match="generate"):
        serve_loop(None, [(0, [1])], K=2, T=4, generate=0, emb=None, pick=None, idle_token=0)


@need
def test_join_capture_and_plan(tmp_path, monkeypatch):
    """run_capture --join (1 block; RWKV7_J_BLOCKS more): one stage whose inputs are every block's lane-form and
    natural-form carries, named as carried state, and whose outputs are the joined carries, one deliberate refresh
    each; with RWKV7_PLAN=1 the planner (LANDING=high) cuts its plan (RWKV7_B_DIR keeps it)."""
    import json
    import os
    import pathlib
    import subprocess
    import sys

    from perseus.examples.rwkv7 import run_capture
    from perseus.examples.rwkv7.tests.conftest import CKPT, ROOT
    from perseus.examples.rwkv7.tests.test_capture import HIGH
    from perseus.examples.rwkv7.tests.test_ref import HARNESS
    for k, v in HIGH.items():
        monkeypatch.setenv(k, v)
    K = int(os.environ.get("RWKV7_B_K", "4"))
    nb = int(os.environ.get("RWKV7_J_BLOCKS", "1"))
    root = pathlib.Path(os.environ.get("RWKV7_B_DIR", str(tmp_path)))
    cap = root / f"cap_join{K}_b{nb}"
    if not (cap / "capture_config.json").exists():
        assert run_capture.main(["--pth", CKPT, "--coeffs", COEFFS, "--harness", HARNESS, "--blocks", str(nb),
                                 "--join", "--batch", str(K), "--out", str(cap)]) == 0
    g = json.load(open(cap / "block_0" / "graph.json"))
    ops = {}
    for n in g["nodes"]:
        ops[n["op_type"]] = ops.get(n["op_type"], 0) + 1
    print(f"join K={K} blocks={nb}: {len(g['nodes'])} nodes", dict(sorted(ops.items())))
    names = {x for n in g["nodes"] for x in n["inputs"]}
    for b in range(nb):
        for k in ("xa", "xf", "nxa", "nxf"):
            assert any(f"cache.b{b}{k}-lvl" in x for x in names), (b, k)
    # per carry: two plaintext products, one add, one deliberate refresh (the carried state's landing)
    nc = len(json.load(open(cap / "rwkv7_capture.json"))["keys"])            # 2 carries per block and shard
    assert ops == {"mult": 2 * nc, "add": nc, "deliberate_bootstrap": nc}, ops
    meta = json.load(open(cap / "rwkv7_capture.json"))
    assert meta["kind"] == "join" and meta["batch"] == K
    if os.environ.get("RWKV7_PLAN") != "1":
        return
    name = f"plan_join{K}_b{nb}"
    env = {**os.environ, "LANDING": "high", "PLAN_ROOT": str(root), "MEMGUARD_GB": "",
           "PY": sys.executable, "PYTHONPATH": str(ROOT / "python")}
    plan = root / "bootstrap_placements" / name
    if not (plan / "PLAN_CMD.txt").exists():
        p = subprocess.run(["bash", str(ROOT / "tools/fhe_ssm/make_plan.sh"), str(cap), name], env=env,
                           capture_output=True, text=True, timeout=1800)
        print(p.stdout[-3000:], p.stderr[-1500:])
    assert (plan / "block_0_placement.json").exists()
    pl = json.load(open(plan / "block_0_placement.json"))
    print(f"join plan {name}: {len(pl['placements'])} planned refreshes;", (plan / "summary.txt").read_text()[-800:]
          if (plan / "summary.txt").exists() else "")
