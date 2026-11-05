# SPDX-License-Identifier: MIT
"""The carried state's refresh levers (results/rwkv7-fhe-20261104/SPEEDUP_MAP_20261105.md S1 / S2), capture side only:

* ``run_capture --carry-drop d``: the S carries refresh under raise drop d and the recorder lands them 2 d units deeper;
* ``run_capture --carry-skip-s``: the two-pass cycle's schedule, S unrefreshed at odd positions; ``--capture-tick 1``
  records pass A (S in at the landing, out 6 units deeper, no S carry), ``--capture-tick 2`` pass B (S in 6 units
  deeper, refreshed back to the landing).

Both are flags of run_capture, recorded in rwkv7_capture.json; the model's kwargs default off, the environment is not
read, and run_planned refuses a capture that carries one (the driver has no two-program cycle). The checks are 1-block
K = 4 filled batched captures (the configuration of the 0.1B K = 32 runs, at K = 4) against a baseline capture of the
same recipe: the carry nodes' landings and inputs, the state inputs' levels, the hand-over levels of the two passes in
both directions, and with --rwkv7-plan the b2 + PAIR=1 plan's census and the compile check when its binary exists."""
import inspect
import json
import os
import pathlib

import pytest

from perseus.examples.rwkv7.tests.conftest import CKPT, COEFFS, ROOT
from perseus.examples.rwkv7.tests.test_ref import HARNESS, need

LAND = 26          # the dense StC-first landing (make_plan.sh LANDING=high: BTS_LEVEL 28, PLAN_LEVEL_UNIT 2 -> 26)
PASS = 6           # units the state chain costs per pass (S in -> the carry refresh's input, graph.json)


def _capture(root, name, extra, monkeypatch):
    from perseus.examples.rwkv7 import run_capture
    from perseus.examples.rwkv7.tests.test_capture import HIGH
    for k, v in HIGH.items():
        monkeypatch.setenv(k, v)
    cap = root / f"cap_{name}"
    if not (cap / "capture_config.json").exists():
        assert run_capture.main(["--pth", CKPT, "--coeffs", COEFFS, "--harness", HARNESS, "--blocks", "1",
                                 "--batch", "4", "--layout", "filled", "--out", str(cap), *extra]) == 0
    return cap


def _graph(cap):
    with open(cap / "block_0" / "graph.json") as f:
        return json.load(f)


def _levels(cap):
    """Block 0's carry refreshes as {(input level, landing): count}, the S inputs' levels, and the levels of S' (the
    update's output read by the readout: the level S leaves the block at when nothing refreshes it)."""
    g = _graph(cap)
    nodes = g["nodes"]
    carry = [n for n in nodes if n["op_type"] == "deliberate_bootstrap" and str(n.get("step", "")).endswith("carry")]
    hist = {}
    for n in carry:
        k = (int(n["input_levels"][0]), int(n["output_level"]))
        hist[k] = hist.get(k, 0) + 1
    s_in = sorted({int(x.rsplit("lvl=", 1)[1]) for n in nodes for x in n["inputs"] if "cache.s0c" in x})
    # S' = the update's final add (S w + S a b + V k~), the deepest add of the update scope that feeds a product
    # there (its readout product S' r is issued inside state_step, i.e. in the update scope)
    producer = {n["output"]: n for n in nodes}
    cands = {int(producer[x]["output_level"]) for n in nodes
             if n["op_type"] == "mult" and str(n.get("step", "")).endswith("update")
             for x in n["inputs"] if x in producer and producer[x]["op_type"] == "add"
             and str(producer[x].get("step", "")).endswith("update")}
    s_prime = [max(cands)] if cands else []
    return hist, s_in, s_prime


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
    return c, check


def test_levers_are_capture_flags_only(tmp_path, monkeypatch):
    """The model's kwargs default off and the environment is not read; run_capture's flags default off; run_planned
    refuses a capture whose metadata carries a lever (and passes captures without one, or without the file)."""
    from perseus.examples.rwkv7 import run_capture, run_planned
    from perseus.examples.rwkv7.model import CARRY_DROPS, Rwkv7Fhe
    sig = inspect.signature(Rwkv7Fhe.__init__).parameters
    assert sig["carry_drop"].default == 0 and sig["carry_skip_s"].default is False
    assert CARRY_DROPS == (0, 1, 3, 5)
    for k in ("RWKV7_CARRY_DROP", "RWKV7_CARRY_SKIP_S", "RWKV7_STATE_ENTRY"):
        monkeypatch.setenv(k, "1")
    src = inspect.getsource(Rwkv7Fhe)
    assert "environ" not in src                                   # nothing in the model reads them
    ap = run_capture.argparse.ArgumentParser()
    run_capture.common.add_args(ap)
    # the capture's own flags, as main() declares them: default off, drops limited to the built variants
    main_src = inspect.getsource(run_capture.main)
    assert "--carry-drop" in main_src and "choices=(0, 1, 3, 5)" in main_src and "--carry-skip-s" in main_src
    # the driver's guard
    cap = tmp_path / "cap"
    cap.mkdir()
    shifts = cap / "pcmm_shifts.json"
    shifts.write_text("{}")
    run_planned.refuse_lever_capture(str(shifts))                 # no metadata: an older capture, allowed
    (cap / "rwkv7_capture.json").write_text(json.dumps({"carry_drop": 0, "carry_skip_s": False}))
    run_planned.refuse_lever_capture(str(shifts))                 # levers off: allowed
    for on in ({"carry_drop": 3}, {"carry_skip_s": True}):
        (cap / "rwkv7_capture.json").write_text(json.dumps({"carry_drop": 0, "carry_skip_s": False, **on}))
        with pytest.raises(SystemExit, match="refresh lever"):
            run_planned.refuse_lever_capture(str(shifts))
    run_planned.refuse_lever_capture(None)                        # no shifts file given: nothing to check


@need
def test_carry_levers_in_capture(tmp_path, monkeypatch):
    root = pathlib.Path(os.environ.get("RWKV7_SL_DIR", str(tmp_path)))
    caps = {"base": _capture(root, "base", [], monkeypatch),
            "drop3": _capture(root, "drop3", ["--carry-drop", "3"], monkeypatch),
            "passA": _capture(root, "passA", ["--carry-skip-s"], monkeypatch),
            "passB": _capture(root, "passB", ["--carry-skip-s", "--capture-tick", "2"], monkeypatch)}
    rec = {k: _levels(c) for k, c in caps.items()}
    for k, (hist, s_in, s_prime) in rec.items():
        print(f"{k:6s} carries {sorted(hist.items())} S in {s_in} S' {s_prime}")
    # baseline: 4 S carries from the landing + the pass's 6 units, xa / xf from 42 / 44; S enters at the landing
    hist, s_in, s_prime = rec["base"]
    assert hist.get((LAND + PASS, LAND)) == 4 and sum(hist.values()) == 6 and s_in == [LAND] and s_prime == [LAND + PASS]
    xaxf = {k: v for k, v in hist.items() if k != (LAND + PASS, LAND)}
    # drop 3: the S carries land 2 x 3 units deeper and start 6 units above that; xa / xf as before
    hist, s_in, s_prime = rec["drop3"]
    assert hist.get((LAND + 6 + PASS, LAND + 6)) == 4 and s_in == [LAND + 6] and s_prime == [LAND + 6 + PASS]
    assert {k: v for k, v in hist.items() if k != (LAND + 6 + PASS, LAND + 6)} == xaxf
    assert _meta(caps["drop3"])["carry_drop"] == 3 and not _meta(caps["drop3"]).get("carry_skip_s", False)
    # pass A (tick 1 of the schedule): no S carry, S enters at the landing and leaves 6 units deeper, unrefreshed
    hist, s_in, s_prime = rec["passA"]
    assert sum(hist.values()) == 2 and hist == xaxf and s_in == [LAND] and s_prime == [LAND + PASS]
    assert _meta(caps["passA"])["carry_skip_s"] is True and _meta(caps["passA"])["capture_tick"] == 1
    # pass B (tick 2): S enters 6 units deeper (pass A's output), its carry refresh runs 38 -> 26
    hist, s_in, s_prime = rec["passB"]
    assert hist.get((LAND + 2 * PASS, LAND)) == 4 and s_in == [LAND + PASS] and s_prime == [LAND + 2 * PASS]
    assert {k: v for k, v in hist.items() if k != (LAND + 2 * PASS, LAND)} == xaxf
    assert _meta(caps["passB"])["carry_skip_s"] is True and _meta(caps["passB"])["capture_tick"] == 2
    # the hand-over, both directions: A's output level is B's input level, B's output (its landing) is A's input
    a_out, b_in = rec["passA"][2], rec["passB"][1]
    b_out = sorted({land for (_lv, land), n in rec["passB"][0].items() if n == 4})
    a_in = rec["passA"][1]
    assert a_out == b_in == [LAND + PASS] and b_out == a_in == [LAND], (a_out, b_in, b_out, a_in)
    assert not _meta(caps["base"]).get("carry_skip_s", False)
    if os.environ.get("RWKV7_PLAN") != "1":
        return
    res = {}
    for name, cap in caps.items():
        c, check = _plan(root, cap, name, monkeypatch)
        res[name] = {"placed": c["refreshes"], "by_step": c["by_step"],
                     "compile": None if check is None else {"sites": check["sites"], "errors": check["errors"],
                                                             "richer": check["level_richer"], "routes": check["routes"],
                                                             "launches": check["bootstrap_launches"]}}
        print(name, json.dumps(res[name]))
    with open(root / "state_levers_plans.json", "w") as f:
        json.dump(res, f, indent=1)
    # the levers move no placed refresh into the per-conversation scopes
    for name in ("drop3", "passA", "passB"):
        assert not any(k.endswith(("update", "bcast", "readout")) for k in res[name]["by_step"]), name
    # both passes of the cycle bind on this runtime: no error, no level richer than the capture's; pass A compiles 4
    # refresh sites fewer than the baseline (its S carries), pass B the baseline's count
    for name in ("passA", "passB"):
        if res[name]["compile"] is not None:
            assert res[name]["compile"]["errors"] == [] and res[name]["compile"]["richer"] == 0, name
    if res["passA"]["compile"] is not None and res["base"]["compile"] is not None:
        assert res["passA"]["compile"]["sites"] == res["base"]["compile"]["sites"] - 4
        assert res["passB"]["compile"]["sites"] >= res["base"]["compile"]["sites"] - 1
    # the dropped carries do not bind (2026-11-05/06, compile_check of this checkout): the compiler resolves a
    # deliberate refresh by its captured landing through route_for_landing (crates/pc-program/src/compile/mod.rs:
    # 176-190), which tries each slot count's base route at the default raise drop and never the built drop
    # variants (real or complex), so "no route lands at (32, 2) from (38, 2)". Until the compiler searches
    # bts.raise_drops (a rebuild) the flag is a capture-side experiment; every error must be that one or its consequences
    if res["drop3"]["compile"] is not None and res["drop3"]["compile"]["errors"]:
        errs = res["drop3"]["compile"]["errors"]
        root_cause = [e for e in errs if "no route lands at (32, 2)" in e]
        assert root_cause, errs[:3]
        assert all(("no route lands at (32, 2)" in e) or ("[plan_level_error]" in e) or ("op accounting" in e)
                   for e in errs), errs[:3]
        print(f"drop3: {len(errs)} compile errors: {len(root_cause)} 'no route lands at (32, 2)' and their consequences")
