# SPDX-License-Identifier: MIT
"""Prefill (prefill.py): a T-token pass on token lanes against the decode step run token by token.

* mirror (numpy slots, no levels): the pass gives every lane the hidden state the decode step gives that token, and
  leaves the same carried state, so decoding after it continues identically (float64 rounding);
* fake (the runtime's level rules, PCMM's fake, the canonical carry) and the capture recorder: the pass matches the
  mirror's pass, and a prefilled chunk followed by decode steps gives the token-by-token run's top-1 and hidden
  states against the plaintext poly model.
2 blocks of the real checkpoint (the fake's slot vectors are numpy arrays at 32,768 slots, not CKKS ciphertexts).
"""
import pathlib

import numpy as np
import pytest

from perseus.examples.rwkv7 import common, ref
from perseus.examples.rwkv7.model import Mirror, load_coeffs
from perseus.examples.rwkv7.session import make_model
from perseus.examples.rwkv7.tests.test_mirror import N_LAYERS, params2, rel_rms  # noqa: F401
from perseus.examples.rwkv7.tests.test_ref import COEFFS, need, tokenizer

TEXT = ("\n\nThe Eiffel tower is in the city of Paris, which is the capital of France. It was built for the "
        "world fair of 1889 and is now one of the most visited monuments in the world.")


def _ids(n):
    ids = tokenizer().encode(TEXT)
    assert len(ids) >= n + 1, len(ids)
    return ids


def _states_close(a, b):
    errs = []
    for sa, sb in zip(a, b):
        for x, y in zip([*sa.S, sa.xa, sa.xf], [*sb.S, sb.xa, sb.xf]):
            errs.append(float(np.max(np.abs(np.asarray(x) - np.asarray(y)))))
    return max(errs)


@need
@pytest.mark.parametrize("n_slots,T", [(1 << 15, 32), (1 << 15, 7), (1 << 15, 1), (1 << 14, 16)])
def test_mirror_prefill_equals_token_by_token(params2, n_slots, T):  # noqa: F811
    ids = _ids(T + 3)
    co = load_coeffs(COEFFS)
    _rf, st = common.prefix_state(params2, COEFFS, ids, 1)
    seq, pf = (Mirror(params2, co, n_slots, wbits=None) for _ in range(2))
    seq.reset(st)
    pf.reset(st)
    chunk = ids[1:1 + T]
    hs = np.stack([seq.step(params2.emb[t])[0] for t in chunk])
    hp, z = pf.prefill([params2.emb[t] for t in chunk])
    e_h = max(rel_rms(hp[i], hs[i]) for i in range(T))
    e_s = _states_close(pf.state, seq.state)
    tail = ids[1 + T:1 + T + 2]                       # decode after the hand-over
    e_d = max(rel_rms(pf.step(params2.emb[t])[0], seq.step(params2.emb[t])[0]) for t in tail)
    print(f"slots {n_slots} T {T}: hidden {e_h:.1e} state {e_s:.1e} decode after {e_d:.1e}")
    # every extra op of the pass is exact in float64 (masks of 1 and 1/2, adding zeros, rotations), so equality can
    # be bit-exact; it is not vacuous: the states moved and the tokens differ, and a wrong lane breaks it (below)
    assert e_h < 1e-10 and e_s < 1e-9 and e_d < 1e-10
    if T > 1:
        assert rel_rms(hs[0], hs[1]) > 1e-2
        bad = Mirror(params2, co, n_slots, wbits=None)
        bad.reset(st)
        hb, _ = bad.prefill([params2.emb[t] for t in chunk[::-1]])      # the chunk reversed: lanes matter
        assert rel_rms(hb[-1], hs[-1]) > 1e-3
    assert np.abs(z.imag.reshape(1024, -1)[768:]).max() < 1e-9


def _fake_run(params2, co, ids, T, capture):  # noqa: F811
    """prefix state -> [prefill pass of T] -> decode the rest; and the token-by-token run, on fresh fakes."""
    _rf, st = common.prefix_state(params2, COEFFS, ids, 1)
    out = {}
    for mode in ("seq", "pf"):
        fhe, _inf, model = make_model(params2, co, capture=capture, n_layers=N_LAYERS, prefill_keys=T)
        if capture:
            fhe.magnitude_suppressed = True
        model.start(st)
        hs, counts = [], []
        body = ids[1:]
        pos = 1
        if mode == "pf":
            model.set_chunk(T)
            b0 = getattr(fhe, "n_bootstraps", 0)
            h, _o = model.step([params2.emb[t] for t in body[:T]], pos)
            counts.append(getattr(fhe, "n_bootstraps", 0) - b0)
            hs += list(h)
            model.set_chunk(0)
            body, pos = body[T:], pos + T
        for t in body:
            b0 = getattr(fhe, "n_bootstraps", 0)
            h, _o = model.step(params2.emb[t], pos)
            counts.append(getattr(fhe, "n_bootstraps", 0) - b0)
            hs.append(h)
            pos += 1
        out[mode] = (np.stack(hs), counts)
    return st, out


@need
@pytest.mark.parametrize("capture", [False, True])
def test_fake_prefill_then_decode(params2, capture):  # noqa: F811
    T = 6
    ids = _ids(T + 3)[: T + 4]                        # prefix + T prefilled + 3 decoded
    co = load_coeffs(COEFFS)
    st, out = _fake_run(params2, co, ids, T, capture)
    hs, hp = out["seq"][0], out["pf"][0]
    rf = ref.RefModel(params2, ref.Sites("poly", COEFFS, clamp=True))
    for t in ids[:1]:
        rf.step(t)
    agree, e_poly, e_seq = [], [], []
    for i, t in enumerate(ids[1:]):
        lr, hr = rf.step(t)
        agree.append(int(np.argmax(rf.head(hp[i])) == np.argmax(lr)) + int(np.argmax(rf.head(hs[i])) == np.argmax(lr)))
        e_poly.append(rel_rms(hp[i], hr))
        e_seq.append(rel_rms(hp[i], hs[i]))
    print("capture" if capture else "fake", "prefill-vs-seq", [f"{e:.1e}" for e in e_seq], "prefill-vs-poly",
          [f"{e:.1e}" for e in e_poly], "bootstraps seq", out["seq"][1], "pf", out["pf"][1])
    assert all(a == 2 for a in agree)                 # both runs agree with the poly model's top-1 at every position
    assert max(e_seq) < 1e-9


@need
def test_prefill_capture_and_plan(tmp_path, monkeypatch):
    """run_capture --prefill-chunk T (1 block): the pass's graph carries the state as inputs; with RWKV7_PLAN=1 the
    planner (LANDING=high) cuts a plan for it (RWKV7_PF_T, default 8, sets T)."""
    import json
    import os
    import subprocess
    import sys

    from perseus.examples.rwkv7 import run_capture
    from perseus.examples.rwkv7.tests.conftest import CKPT, ROOT
    from perseus.examples.rwkv7.tests.test_capture import HIGH
    from perseus.examples.rwkv7.tests.test_ref import HARNESS
    for k, v in HIGH.items():
        monkeypatch.setenv(k, v)
    T = int(os.environ.get("RWKV7_PF_T", "8"))
    nb = int(os.environ.get("RWKV7_PF_BLOCKS", "1"))
    cap = pathlib.Path(os.environ.get("RWKV7_PF_DIR", str(tmp_path))) / f"cap_T{T}_b{nb}"
    assert run_capture.main(["--pth", CKPT, "--coeffs", COEFFS, "--harness", HARNESS, "--blocks", str(nb),
                             "--prefill-chunk", str(T), "--out", str(cap)]) == 0
    g = json.load(open(cap / "block_0" / "graph.json"))
    ops = {}
    for n in g["nodes"]:
        ops[n["op_type"]] = ops.get(n["op_type"], 0) + 1
    print(f"T={T} block_0: {len(g['nodes'])} nodes", dict(sorted(ops.items())))
    names = [x for n in g["nodes"] for x in n["inputs"]]
    assert all(any(f"cache.{s}" in x for x in names) for s in ("s0", "s1", "xa", "xf"))
    if os.environ.get("RWKV7_PLAN") != "1":
        return
    root = cap.parent
    env = {**os.environ, "LANDING": "high", "PLAN_ROOT": str(root), "MEMGUARD_GB": "",
           "PY": sys.executable, "PYTHONPATH": str(ROOT / "python")}
    p = subprocess.run(["bash", str(ROOT / "tools/fhe_ssm/make_plan.sh"), str(cap), f"plan_T{T}_b{nb}"], env=env,
                       capture_output=True, text=True, timeout=5400)
    print(p.stdout[-4000:], p.stderr[-2000:])
    plan = root / "bootstrap_placements" / f"plan_T{T}_b{nb}"
    assert all((plan / f"block_{b}_placement.json").exists() for b in range(nb + 1))


def _cap_and_plan(root, name, extra, env_high):
    import os
    import subprocess
    import sys

    from perseus.examples.rwkv7 import run_capture
    from perseus.examples.rwkv7.tests.conftest import CKPT, ROOT
    from perseus.examples.rwkv7.tests.test_ref import HARNESS
    cap = root / f"cap_{name}"
    if not (cap / "capture_config.json").exists():
        assert run_capture.main(["--pth", CKPT, "--coeffs", COEFFS, "--harness", HARNESS, "--blocks", "1",
                                 "--out", str(cap), *extra]) == 0
    plan = root / "bootstrap_placements" / f"plan_{name}"
    if not (plan / "PLAN_CMD.txt").exists():
        env = {**os.environ, **env_high, "LANDING": "high", "PLAN_ROOT": str(root), "MEMGUARD_GB": "",
               "PY": sys.executable, "PYTHONPATH": str(ROOT / "python")}
        p = subprocess.run(["bash", str(ROOT / "tools/fhe_ssm/make_plan.sh"), str(cap), f"plan_{name}"], env=env,
                           capture_output=True, text=True, timeout=5400)
        assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    return cap, plan


@need
def test_prefill_session_prepare(tmp_path, monkeypatch):
    """The device runner's CPU half (``run_planned.Rwkv7PrefillRun.prepare``): a 1-block decode plan and a T = 4
    prefill plan (one shifts file) compile with the runtime's ``Program.compile`` (no session, no GPU); the prefill
    state registers map onto the decode ones, and the carried state's (level, deg) agree both ways (a pass writes
    what a decode step reads, and back), which ``Program.run`` requires. Needs --rwkv7-plan (it plans twice)."""
    import os
    if os.environ.get("RWKV7_PLAN") != "1":
        pytest.skip("plans twice: run with --rwkv7-plan")
    pytest.importorskip("networkx")
    try:
        from perseus import _pc
        _pc.Program.compile                       # noqa: B018
    except (ImportError, AttributeError) as e:
        pytest.skip(f"perseus._pc has no Program.compile here: {e}")
    from perseus.examples.fhe_ssm.run_planned import session_config
    from perseus.examples.rwkv7 import run_planned as rp
    from perseus.examples.rwkv7.tests.conftest import CKPT
    from perseus.examples.rwkv7.tests.test_capture import HIGH
    for k, v in HIGH.items():
        monkeypatch.setenv(k, v)
    T = 4
    root = pathlib.Path(os.environ.get("RWKV7_PF_DIR", str(tmp_path))) / "session_prepare"
    cap_d, plan_d = _cap_and_plan(root, "dec", [], HIGH)
    cap_p, plan_p = _cap_and_plan(root, f"pf{T}", ["--prefill-chunk", str(T), "--shifts",
                                                   str(cap_d / "pcmm_shifts.json")], HIGH)
    import json
    shifts = json.load(open(cap_d / "pcmm_shifts.json"))
    P = ref.load_params(CKPT, 1)
    co = load_coeffs(COEFFS)
    tok = tokenizer()
    ids = tok.encode(common.CALIB_TEXT)
    _rf, st = common.prefix_state(P, COEFFS, ids, 1)
    cfg = session_config(str(plan_d))
    labels = ["block_0", "block_1"]
    _f, _i, dm = make_model(P, co, capture=True, shifts=shifts)
    dm.init_state = st
    _f, _i, pm = make_model(P, co, capture=True, shifts=shifts, chunk=T)
    pm.init_state = st
    dec = rp.Rwkv7PlannedRun(rp.Recorder(dm, 1), str(plan_d), cfg, labels, None, model=dm,
                             work_dir=str(root / "work_dec"))
    pf = rp.Rwkv7PlannedRun(rp.Recorder(pm, 1), str(plan_p), cfg, labels, None, model=pm,
                            work_dir=str(root / "work_pf"))
    long_ids = tok.encode(common.CALIB_TEXT_LONG)[1:]
    both = rp.Rwkv7PrefillRun(dec, pf, T).prepare(rp.probe_rows(P, ids, ids[:1]),
                                                  [[P.emb[t] for t in long_ids[i * T:(i + 1) * T]] for i in range(2)])
    dregs = sorted(r for _i, _j, r in dec.reg[0]["kv"])
    pregs = sorted(r for _i, _j, r in pf.reg[0]["kv"])
    outs = [r for b in pf.reg for _j, r in pf.reg[b]["out"]]
    print("decode kv regs", dregs, "prefill kv regs", pregs, "prefill out regs", outs, "total", both.n_regs,
          "prefill inputs", [x[:3] for x in pf.progs["block_0"].inputs])
    assert dregs == pregs and len(dregs) == 4
    assert min(outs) >= dec.n_regs and both.n_regs == dec.n_regs + len(outs)
    assert pf.progs["block_0"].label() == "pf_block_0" if callable(getattr(pf.progs["block_0"], "label", None)) else True
