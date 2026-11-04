# SPDX-License-Identifier: MIT
"""Batched decode (batched.py): K conversations per pass, conversation c on lane c, against K separate decode runs.

* mirror (numpy slots, no levels): every conversation's hidden state at every step, and its carried state at the end,
  equal K separate decode runs (float64 rounding); prompts of different lengths, then free-running greedy steps (each
  conversation's pick fed back to its own lane); K = 2, 5, 32; both carry layouts; a swapped lane assignment breaks it;
* fake (the runtime's level rules, PCMM's fake, the canonical carry) and the capture recorder: the batched pass
  matches K separate decode runs on fakes and gives the poly model's top-1 at every position of every conversation;
* K = 1 is the decode step: the same op stream, the same values.
2 blocks of the real checkpoint, 2^15 slots (t = 32 lanes).
"""
import numpy as np
import pytest

from perseus.examples.rwkv7 import common, ref
from perseus.examples.rwkv7.batched import BatchedMirror
from perseus.examples.rwkv7.model import Mirror, load_coeffs
from perseus.examples.rwkv7.session import make_model
from perseus.examples.rwkv7.tests.test_mirror import N_LAYERS, params2, rel_rms  # noqa: F401
from perseus.examples.rwkv7.tests.test_ref import COEFFS, need, tokenizer

TEXT = ("\n\nThe Eiffel tower is in the city of Paris, which is the capital of France. It was built for the "
        "world fair of 1889 and is now one of the most visited monuments in the world. In the beginning God "
        "created the heaven and the earth. And the earth was without form, and void; and darkness was upon the "
        "face of the deep. User: A shop sells 3 pencils for 2 dollars.")


def _prompts(K, lens):
    """K prompts (token ids after the public prefix) of the given lengths, each a different window of TEXT."""
    ids = tokenizer().encode(TEXT)[1:]
    return [ids[(5 * c) % (len(ids) - 8):][: lens[c]] for c in range(K)]


def _schedule(prompts, gen):
    return max(len(p) for p in prompts) + gen


def _batched_run(stepper, head, P, prompts, gen):
    """Runs the per-lane schedule: conversation c reads its prompt, then feeds its greedy pick back. ``stepper(rows)``
    returns (K, C) hidden states. Returns hs[c] (its first len(p) + gen positions) and the token streams."""
    K = len(prompts)
    toks = [list(p) for p in prompts]
    hs = [[] for _ in range(K)]
    for i in range(_schedule(prompts, gen)):
        rows = []
        for c in range(K):
            if i >= len(toks[c]):
                toks[c].append(int(np.argmax(head(hs[c][-1]))))
            rows.append(P.emb[toks[c][i]])
        h = stepper(np.stack(rows))
        for c in range(K):
            hs[c].append(h[c])
    return [np.stack(hs[c][: len(prompts[c]) + gen]) for c in range(K)], [t[: len(prompts[c]) + gen]
                                                                            for c, t in enumerate(toks)]


def _separate_run(make_step, head, P, prompt, n):
    step = make_step()
    toks, hs = list(prompt), []
    for i in range(n):
        if i >= len(toks):
            toks.append(int(np.argmax(head(hs[-1]))))
        hs.append(step(P.emb[toks[i]]))
    return np.stack(hs), toks


def _max_state_err(a, b):
    errs = []
    for sa, sb in zip(a, b):
        for x, y in zip([*sa.S, sa.xa, sa.xf], [*sb.S, sb.xa, sb.xf]):
            errs.append(float(np.max(np.abs(np.asarray(x) - np.asarray(y)))))
    return max(errs)


@need
@pytest.mark.parametrize("K,carry,hoist", [(2, "decode", 0), (5, "decode", 0), (5, "lane", 0), (5, "decode", 3),
                                           (32, "lane", 0), (32, "decode", 0)])
def test_mirror_batched_equals_separate_decodes(params2, K, carry, hoist):  # noqa: F811
    gen = 2
    lens = [2 + (c * 3) % 5 for c in range(K)]                 # 2..6 prompt tokens, different per conversation
    prompts = _prompts(K, lens)
    co = load_coeffs(COEFFS)
    ids = tokenizer().encode(TEXT)
    _rf, st = common.prefix_state(params2, COEFFS, ids, 1)
    head = ref.RefModel(params2, ref.Sites("poly", COEFFS, clamp=True)).head
    bm = BatchedMirror(params2, co, 1 << 15, K, carry=carry, wbits=None, hoist=hoist)
    bm.reset(st)
    hb, tb = _batched_run(lambda rows: bm.step(rows)[0], head, params2, prompts, gen)
    errs, serr = [], []
    n_sched = _schedule(prompts, gen)
    for c in range(K):
        sep = Mirror(params2, co, 1 << 15, wbits=None)
        sep.reset(st)

        def mk(sep=sep):
            return lambda row: sep.step(row)[0]
        # the separate run goes as far as the batch did (the batch keeps every lane running to the last pass)
        hs, ts = _separate_run(mk, head, params2, prompts[c], n_sched)
        assert ts[: len(tb[c])] == tb[c], (c, ts, tb[c])
        errs.append(max(rel_rms(hb[c][i], hs[i]) for i in range(len(hb[c]))))
        serr.append(_max_state_err(bm.conv_state(c), sep.state))
    print(f"K={K} carry={carry} hoist={hoist}: hidden rel max {max(errs):.1e}, end state abs max {max(serr):.1e}, "
          f"generated {[t[len(p):] for t, p in zip(tb, prompts)][:3]}")
    assert max(errs) < 1e-10 and max(serr) < 1e-9
    # non-vacuity: the conversations differ, and a wrong lane assignment breaks the equality
    assert rel_rms(hb[0][1], hb[1][1]) > 1e-2
    bad = BatchedMirror(params2, co, 1 << 15, K, carry=carry, wbits=None)
    bad.reset(st)
    rows0 = np.stack([params2.emb[p[0]] for p in prompts])
    bad.step(rows0)
    h_swapped = bad.step(np.stack([params2.emb[p[1]] for p in prompts])[::-1])[0]
    assert rel_rms(h_swapped[0], hb[K - 1][1]) > 1e-3


def _fake_batched(params2, co, st, prompts, gen, K, carry, capture):  # noqa: F811
    fhe, _inf, model = make_model(params2, co, capture=capture, n_layers=N_LAYERS, batch=K, batch_carry=carry)
    if capture:
        fhe.magnitude_suppressed = True
    model.start(st)
    head = ref.RefModel(params2, ref.Sites("poly", COEFFS, clamp=True)).head
    pos = [1]

    def stepper(rows):
        h, _o = model.step(rows, pos[0])
        pos[0] += 1
        return h
    return _batched_run(stepper, head, params2, prompts, gen)


@need
@pytest.mark.parametrize("capture,carry", [(False, "decode"), (False, "lane"), (True, "lane")])
def test_fake_batched_top1(params2, capture, carry):  # noqa: F811
    K, gen = 3, 2
    prompts = _prompts(K, [2, 4, 3])
    co = load_coeffs(COEFFS)
    ids = tokenizer().encode(TEXT)
    _rf, st = common.prefix_state(params2, COEFFS, ids, 1)
    hb, tb = _fake_batched(params2, co, st, prompts, gen, K, carry, capture)
    agree, e_poly, e_sep = [], [], []
    for c in range(K):
        rf = ref.RefModel(params2, ref.Sites("poly", COEFFS, clamp=True))
        rf.step(ids[0])
        fhe, _inf, dm = make_model(params2, co, capture=False, n_layers=N_LAYERS)
        dm.start(st)
        for i, t in enumerate(tb[c]):
            lr, hr = rf.step(t)
            agree.append(int(np.argmax(rf.head(hb[c][i])) == np.argmax(lr)))
            e_poly.append(rel_rms(hb[c][i], hr))
            if not capture:
                hd, _o = dm.step(params2.emb[t], 1 + i)
                e_sep.append(rel_rms(hb[c][i], hd))
    print(("capture" if capture else "fake"), carry, f"top-1 {sum(agree)}/{len(agree)}",
          f"vs poly max {max(e_poly):.1e}", f"vs separate fake decode max {max(e_sep):.1e}" if e_sep else "")
    assert all(agree)
    if e_sep:
        assert max(e_sep) < 1e-9


@need
def test_batched_capture_and_plan(tmp_path, monkeypatch):
    """run_capture --batch K (1 block): the pass's graph carries every conversation's state as inputs; with
    RWKV7_PLAN=1 the planner (LANDING=high) cuts a plan for it. RWKV7_B_K (default 4), RWKV7_B_CARRY (lane),
    RWKV7_B_HOIST (0), RWKV7_B_DIR (keep the capture and plan there), RWKV7_B_PLANENV (extra plan env, e.g.
    "PAIR=1")."""
    import json
    import os
    import pathlib
    import subprocess
    import sys

    from perseus.examples.rwkv7 import run_capture
    from perseus.examples.rwkv7.batched import state_names
    from perseus.examples.rwkv7.tests.conftest import CKPT, ROOT
    from perseus.examples.rwkv7.tests.test_capture import HIGH
    from perseus.examples.rwkv7.tests.test_ref import HARNESS
    for k, v in HIGH.items():
        monkeypatch.setenv(k, v)
    K = int(os.environ.get("RWKV7_B_K", "4"))
    carry = os.environ.get("RWKV7_B_CARRY", "lane")
    hoist = int(os.environ.get("RWKV7_B_HOIST", "0"))
    nb = int(os.environ.get("RWKV7_B_BLOCKS", "1"))
    tag = f"K{K}_{carry}" + (f"_h{hoist}" if hoist else "") + (f"_b{nb}" if nb > 1 else "")
    root = pathlib.Path(os.environ.get("RWKV7_B_DIR", str(tmp_path)))
    cap = root / f"cap_{tag}"
    if not (cap / "capture_config.json").exists():
        assert run_capture.main(["--pth", CKPT, "--coeffs", COEFFS, "--harness", HARNESS, "--blocks", str(nb),
                                 "--batch", str(K), "--batch-carry", carry, "--batch-hoist", str(hoist),
                                 "--out", str(cap)]) == 0
    g = json.load(open(cap / "block_0" / "graph.json"))
    ops = {}
    for n in g["nodes"]:
        ops[n["op_type"]] = ops.get(n["op_type"], 0) + 1
    print(f"K={K} {carry} block_0: {len(g['nodes'])} nodes", dict(sorted(ops.items())))
    names = {x for n in g["nodes"] for x in n["inputs"]}
    for s in state_names(K, 2, carry == "lane"):
        assert any(f"cache.{s}-lvl" in x for x in names), s
    if os.environ.get("RWKV7_PLAN") != "1":
        return
    extra = dict(kv.split("=", 1) for kv in os.environ.get("RWKV7_B_PLANENV", "").split())
    name = f"plan_{tag}" + "".join(f"_{k}{v}" for k, v in sorted(extra.items()))
    env = {**os.environ, **extra, "LANDING": "high", "PLAN_ROOT": str(root), "MEMGUARD_GB": "",
           "PY": sys.executable, "PYTHONPATH": str(ROOT / "python")}
    plan = root / "bootstrap_placements" / name
    if not (plan / "PLAN_CMD.txt").exists():
        p = subprocess.run(["bash", str(ROOT / "tools/fhe_ssm/make_plan.sh"), str(cap), name], env=env,
                           capture_output=True, text=True, timeout=5400)
        print(p.stdout[-3000:], p.stderr[-1500:])
    assert all((plan / f"block_{b}_placement.json").exists() for b in range(nb + 1))
    for b in range(nb):
        print(f"census {name} block {b}:", json.dumps(plan_census(cap, plan, b)))


def plan_census(cap, plan, b=0) -> dict:
    """Where block b's planned refreshes sit (by the capture's step scope), and the graph's op counts."""
    import collections
    import json
    g = json.load(open(f"{cap}/block_{b}/graph.json"))
    pl = json.load(open(f"{plan}/block_{b}_placement.json"))
    step = {n.get("output"): n.get("step") for n in g["nodes"]}
    by = collections.Counter(str(step.get(p.get("target_var"))) for p in pl["placements"])
    rot = collections.Counter(n.get("step") for n in g["nodes"] if n["op_type"] == "rotate")
    ops = collections.Counter(n["op_type"] for n in g["nodes"])
    node = {n.get("output"): n for n in g["nodes"]}
    users = collections.defaultdict(list)
    for n in g["nodes"]:
        for x in n["inputs"]:
            users[x].append(n)
    sample = []
    for p in pl["placements"]:
        n = node.get(p.get("target_var"))
        if n is not None and str(n.get("step")).endswith("update") and len(sample) < 6:
            sample.append({"var": n["output"], "op": n["op_type"], "lvl": n.get("output_level"),
                           "in_lvls": n.get("input_levels"), "inputs": n["inputs"],
                           "users": [(u["op_type"], u.get("step"), u["output"]) for u in users[n["output"]]]})
    return {"refreshes": len(pl["placements"]), "by_step": dict(by.most_common()), "ops": dict(ops),
            "rotate_by_step": dict(rot.most_common()), "update_sample": sample}


@need
def test_batched_session_prepare(tmp_path, monkeypatch):
    """The device runner's CPU half for --batch (``run_planned.Rwkv7BatchedRun.prepare``): a K = 4, 1-block batched
    plan compiles with ``Program.compile`` (no session, no GPU) and every conversation's carried state is recognised
    as block inputs (4 x 2 state + 2 lane carries); ``state_cts`` names map onto them; ``read_prompts`` unescapes.
    Needs --rwkv7-plan and the CUDA build's ``perseus._pc``."""
    import os
    import pathlib
    if os.environ.get("RWKV7_PLAN") != "1":
        pytest.skip("plans: run with --rwkv7-plan")
    pytest.importorskip("networkx")
    try:
        from perseus import _pc
        _pc.Program.compile                       # noqa: B018
    except (ImportError, AttributeError) as e:
        pytest.skip(f"perseus._pc has no Program.compile here: {e}")
    import json

    from perseus.examples.fhe_ssm.run_planned import session_config
    from perseus.examples.rwkv7 import run_planned as rp
    from perseus.examples.rwkv7.run_capture import batch_probe
    from perseus.examples.rwkv7.tests.conftest import CKPT
    from perseus.examples.rwkv7.tests.test_capture import HIGH
    from perseus.examples.rwkv7.tests.test_prefill import _cap_and_plan
    for k, v in HIGH.items():
        monkeypatch.setenv(k, v)
    K = 4
    root = pathlib.Path(os.environ.get("RWKV7_B_DIR", str(tmp_path))) / "session_prepare"
    cap, plan = _cap_and_plan(root, f"b{K}", ["--batch", str(K)], HIGH)
    shifts = json.load(open(cap / "pcmm_shifts.json"))
    P = ref.load_params(CKPT, 1)
    co = load_coeffs(COEFFS)
    tok = tokenizer()
    _rf, st = common.prefix_state(P, COEFFS, tok.encode(common.CALIB_TEXT), 1)
    _f, _i, m = make_model(P, co, capture=True, shifts=shifts, batch=K, batch_carry="lane")
    m.init_state = st
    run = rp.Rwkv7BatchedRun(rp.Recorder(m, 1), str(plan), session_config(str(plan)), ["block_0", "block_1"], None,
                             model=m, work_dir=str(root / "work_b"))
    public, _t = batch_probe(P, tok.encode(common.CALIB_TEXT_LONG)[1:], K, 2)
    run.prepare(public)
    keys = sorted(rp._state_key(run, 0, i) for i, _j, _r in run.reg[0]["kv"])
    print("batched kv", keys, "inputs", [x[:3] for x in run.progs["block_0"].inputs][:4])
    assert keys == sorted(m.state_names()) and len(keys) == 2 * K + 2


def test_read_prompts(tmp_path):
    from perseus.examples.rwkv7 import run_planned as rp
    p = tmp_path / "prompts.txt"
    p.write_text("Hello\\nworld\n\n\\n\\nAlready prefixed\nback\\\\slash\n")
    assert rp.read_prompts(str(p)) == ["\n\nHello\nworld", "\n\nAlready prefixed", "\n\nback\\slash"]


@need
def test_k1_is_decode(params2):  # noqa: F811
    """batch = 1: decode's circuit, state and op stream (the fake's call log), and its values."""
    co = load_coeffs(COEFFS)
    ids = tokenizer().encode(TEXT)[:4]
    _rf, st = common.prefix_state(params2, COEFFS, ids, 1)
    runs = []
    for batch in (0, 1):
        fhe, _inf, m = make_model(params2, co, n_layers=N_LAYERS, batch=batch)
        m.start(st)
        n0 = len(fhe.calls)
        hs = []
        for i, t in enumerate(ids[1:]):
            row = params2.emb[t] if batch == 0 else [params2.emb[t]]
            h, _o = m.step(row, 1 + i)
            hs.append(np.asarray(h).reshape(-1))
        runs.append((list(fhe.calls[n0:]), np.stack(hs), sorted(m.rotation_steps())))
    (c0, h0, r0), (c1, h1, r1) = runs
    print(f"decode calls {len(c0)}, batch=1 calls {len(c1)}")
    assert c0 == c1 and r0 == r1 and len(c0) > 1000
    assert np.array_equal(h0, h1)
