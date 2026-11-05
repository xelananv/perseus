# SPDX-License-Identifier: MIT
"""The carried state's refresh levers (results/rwkv7-fhe-20261104/SPEEDUP_MAP_20261105.md S1 / S2), capture side:

* ``RWKV7_CARRY_DROP=d``: the S carries refresh under raise drop d and the recorder lands them 2 d units deeper;
* ``RWKV7_STATE_ENTRY=L``: the recorded graph shows S entering every block at level L (the second pass, B, of a
  two-pass carry cycle);
* ``RWKV7_CARRY_SKIP_S=1``: the cycle's schedule, S unrefreshed at odd positions (the capture at tick 1 records pass A).

All default off. The checks are 1-block K = 4 filled batched captures (the configuration of the 0.1B K = 32 runs, at
K = 4) against a baseline capture of the same recipe: the carry nodes' landings and inputs, the state inputs' levels, and
with --rwkv7-plan the b2 + PAIR=1 plan's census (no extra placed refresh, the carries as expected) and the compile check
when its binary exists."""
import json
import os
import pathlib

import pytest

from perseus.examples.rwkv7.tests.conftest import CKPT, COEFFS, ROOT
from perseus.examples.rwkv7.tests.test_ref import HARNESS, need

LAND = 26          # the dense StC-first landing (make_plan.sh LANDING=high: BTS_LEVEL 28, PLAN_LEVEL_UNIT 2 -> 26)
PASS = 6           # units the state chain costs per pass (S in -> the carry refresh's input, graph.json)


def _capture(root, name, env, monkeypatch):
    from perseus.examples.rwkv7 import run_capture
    from perseus.examples.rwkv7.tests.test_capture import HIGH
    for k, v in {**HIGH, **env}.items():
        monkeypatch.setenv(k, v)
    cap = root / f"cap_{name}"
    if not (cap / "capture_config.json").exists():
        assert run_capture.main(["--pth", CKPT, "--coeffs", COEFFS, "--harness", HARNESS, "--blocks", "1",
                                 "--batch", "4", "--layout", "filled", "--out", str(cap)]) == 0
    for k in env:
        monkeypatch.delenv(k, raising=False)
    return cap


def _graph(cap):
    with open(cap / "block_0" / "graph.json") as f:
        return json.load(f)


def _carries(cap):
    """The carry refreshes of block 0 as {(input level, landing): count}, the S inputs' levels, the level_reduce
    count, and the level S leaves the block at (its stage output: the deepest S-named cache write's input)."""
    g = _graph(cap)
    nodes = g["nodes"]
    carry = [n for n in nodes if n["op_type"] == "deliberate_bootstrap" and str(n.get("step", "")).endswith("carry")]
    hist = {}
    for n in carry:
        k = (int(n["input_levels"][0]), int(n["output_level"]))
        hist[k] = hist.get(k, 0) + 1
    s_in = sorted({x for n in nodes for x in n["inputs"] if "cache.s0c" in x})
    levels = sorted({int(x.rsplit("lvl=", 1)[1]) for x in s_in})
    reduces = sum(n["op_type"] == "level_reduce" for n in nodes)
    return hist, levels, reduces


def _meta(cap):
    with open(cap / "rwkv7_capture.json") as f:
        return json.load(f)


def _plan(root, cap, name, monkeypatch):
    import subprocess
    import sys

    from perseus.examples.rwkv7.tests.test_batched import _planopt_env, plan_census
    (kv,) = _planopt_env("b2")
    monkeypatch.setenv(*kv.split("=", 1))
    plan = root / "bootstrap_placements" / f"plan_{name}"
    if not (plan / "PLAN_CMD.txt").exists():
        env = {**os.environ, "LANDING": "high", "PAIR": "1", "RUNTIME_EXTRA": "PC_EM_FUSE_RESCALE=1",
               "PLAN_ROOT": str(root), "MEMGUARD_GB": "", "PY": sys.executable, "PYTHONPATH": str(ROOT / "python")}
        p = subprocess.run(["bash", str(ROOT / "tools/fhe_ssm/make_plan.sh"), str(cap), f"plan_{name}"], env=env,
                           capture_output=True, text=True, timeout=5400, check=False)
        assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    c = plan_census(cap, plan, 0)
    cc = plan / "compile_check.json"
    check = None
    if cc.exists():
        with open(cc) as f:
            check = json.load(f)["blocks"][0]
    with open(plan / "summary.txt") as f:
        exit_at = [ln for ln in f.read().splitlines() if "block_0:" in ln]
    return c, check, exit_at


def test_default_off(monkeypatch):
    from perseus.examples.rwkv7.model import state_levers
    monkeypatch.delenv("RWKV7_CARRY_DROP", raising=False)
    monkeypatch.delenv("RWKV7_STATE_ENTRY", raising=False)
    assert state_levers() == (0, 0)
    monkeypatch.setenv("RWKV7_CARRY_DROP", "3")
    monkeypatch.setenv("RWKV7_STATE_ENTRY", "32")
    assert state_levers() == (3, 32) and state_levers(0, 0) == (0, 0)      # the arguments win
    monkeypatch.setenv("RWKV7_CARRY_DROP", "2")
    with pytest.raises(ValueError):
        state_levers()                                                      # only the built variants 1, 3, 5


@need
def test_carry_levers_in_capture(tmp_path, monkeypatch):
    root = pathlib.Path(os.environ.get("RWKV7_SL_DIR", str(tmp_path)))
    caps = {"base": _capture(root, "base", {}, monkeypatch),
            "drop3": _capture(root, "drop3", {"RWKV7_CARRY_DROP": "3"}, monkeypatch),
            "entry32": _capture(root, "entry32", {"RWKV7_STATE_ENTRY": "32"}, monkeypatch),
            "passA": _capture(root, "passA", {"RWKV7_CARRY_SKIP_S": "1"}, monkeypatch)}
    rec = {k: _carries(c) for k, c in caps.items()}
    for k, (hist, levels, reduces) in rec.items():
        print(f"{k:8s} carries {sorted(hist.items())} S in {levels} level_reduce {reduces}")
    # baseline: 4 S carries from the landing + the pass's 6 units, xa / xf from 42 / 44; S enters at the landing
    hist, levels, reduces = rec["base"]
    assert hist.get((LAND + PASS, LAND)) == 4 and sum(hist.values()) == 6 and levels == [LAND] and reduces == 0
    xaxf = {k: v for k, v in hist.items() if k != (LAND + PASS, LAND)}
    # drop 3: the S carries land 2 x 3 units deeper and start 6 units above that; xa / xf as before
    hist, levels, reduces = rec["drop3"]
    assert hist.get((LAND + 6 + PASS, LAND + 6)) == 4 and levels == [LAND + 6] and reduces == 0
    assert {k: v for k, v in hist.items() if k != (LAND + 6 + PASS, LAND + 6)} == xaxf
    assert _meta(caps["drop3"])["carry_drop"] == 3 and _meta(caps["drop3"])["state_entry"] == 0
    # entry 32 (pass B): S enters at 32, its carry refresh starts at 38 and lands at 26, one level_reduce per S
    hist, levels, reduces = rec["entry32"]
    assert hist.get((32 + PASS, LAND)) == 4 and levels == [32] and reduces == 4
    assert _meta(caps["entry32"])["state_entry"] == 32
    # pass A: no S carry at all (xa / xf only), S enters at the landing and leaves 6 units deeper, unrefreshed
    hist, levels, reduces = rec["passA"]
    assert sum(hist.values()) == 2 and hist == xaxf and levels == [LAND] and reduces == 0
    assert _meta(caps["passA"])["carry_skip_s"] is True and not _meta(caps["base"]).get("carry_skip_s", False)
    if os.environ.get("RWKV7_PLAN") != "1":
        return
    res = {}
    for name, cap in caps.items():
        c, check, exit_at = _plan(root, cap, name, monkeypatch)
        res[name] = {"placed": c["refreshes"], "by_step": c["by_step"], "exit": exit_at,
                     "compile": None if check is None else {"sites": check["sites"], "errors": check["errors"],
                                                             "richer": check["level_richer"], "routes": check["routes"],
                                                             "launches": check["bootstrap_launches"]}}
        print(name, json.dumps(res[name]))
    with open(root / "state_levers_plans.json", "w") as f:
        json.dump(res, f, indent=1)
    # the levers move no placed refresh: at most the baseline's count, nothing in the per-conversation scopes
    for name in ("drop3", "entry32", "passA"):
        assert res[name]["placed"] <= res["base"]["placed"], name
        assert not any(k.endswith(("update", "bcast", "readout")) for k in res[name]["by_step"]), name
    # pass B (entry 32) and pass A (no S carry) bind on this runtime: no error, no level richer than the capture's
    for name in ("entry32", "passA"):
        if res[name]["compile"] is not None:
            assert res[name]["compile"]["errors"] == [] and res[name]["compile"]["richer"] == 0, name
    # pass A plans the baseline's placed sites and compiles 4 refresh sites fewer (its S carries); S leaves it
    # unrefreshed (no carry node, no placed site in the state scopes: the graph and the census above)
    assert res["passA"]["placed"] == res["base"]["placed"]
    if res["passA"]["compile"] is not None and res["base"]["compile"] is not None:
        assert res["passA"]["compile"]["sites"] == res["base"]["compile"]["sites"] - 4
    # the dropped COMPLEX carries do not (2026-11-05, compile_check of this checkout): the dense StC-first route of a
    # complex payload has no drop variant in the compiler's landing table ("no route lands at (32, 2) from (38, 2)"),
    # while the real route has one (dense_stcf_real@32 in the same check). Until the runtime adds the variant the flag
    # is a capture-side experiment for the filled layout's S; every error must be that one or its consequences
    if res["drop3"]["compile"] is not None and res["drop3"]["compile"]["errors"]:
        errs = res["drop3"]["compile"]["errors"]
        root_cause = [e for e in errs if "no route lands at (32, 2)" in e]
        assert root_cause, errs[:3]
        assert all(("no route lands at (32, 2)" in e) or ("[plan_level_error]" in e) or ("op accounting" in e)
                   for e in errs), errs[:3]
        print(f"drop3: {len(errs)} compile errors: {len(root_cause)} 'no route lands at (32, 2)' (the complex route's "
              f"drop variant) and their consequences")
