# SPDX-License-Identifier: MIT
"""The carried state's refresh levers (results/rwkv7-fhe-20261104/SPEEDUP_MAP_20261105.md S1 / S2), capture side:

* ``RWKV7_CARRY_DROP=d``: the S carries refresh under raise drop d and the recorder lands them 2 d units deeper;
* ``RWKV7_STATE_ENTRY=L``: the recorded graph shows S entering every block at level L (the second pass of a two-pass
  carry cycle), so the planner's acceptance of that pass is read on the CPU.

Both default off. The checks are 1-block K = 4 filled batched captures (the configuration of the 0.1B K = 32 runs, at
K = 4) against a baseline capture of the same recipe: the carry nodes' landings and inputs, the state inputs' levels, and
with --rwkv7-plan the b2 plan's census (no extra placed refresh, the carries as before) and the compile check when its
binary exists."""
import json
import os
import pathlib

import pytest

from perseus.examples.rwkv7.tests.conftest import CKPT, COEFFS, ROOT
from perseus.examples.rwkv7.tests.test_ref import HARNESS, need

LAND = 26          # the dense StC-first landing (make_plan.sh LANDING=high: BTS_LEVEL 28, PLAN_LEVEL_UNIT 2 -> 26)
UNIT = 2


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


def _carries(cap):
    """(input level, landing) of every carry refresh of a state ciphertext in block 0, and the S inputs' levels."""
    g = json.load(open(cap / "block_0" / "graph.json"))
    carry = [n for n in g["nodes"] if n["op_type"] == "deliberate_bootstrap" and str(n.get("step", "")).endswith("carry")]
    s_in = sorted({x for n in g["nodes"] for x in n["inputs"] if "cache.s0c" in x})
    levels = sorted({int(x.rsplit("lvl=", 1)[1]) for x in s_in})
    reduces = [n for n in g["nodes"] if n["op_type"] == "level_reduce"]
    return carry, levels, reduces


def _plan(root, cap, name, monkeypatch):
    import subprocess
    import sys

    from perseus.examples.rwkv7.tests.test_batched import _planopt_env, plan_census
    (kv,) = _planopt_env("b2")
    monkeypatch.setenv(*kv.split("=", 1))
    plan = root / "bootstrap_placements" / f"plan_{name}"
    if not (plan / "PLAN_CMD.txt").exists():
        env = {**os.environ, "LANDING": "high", "PAIR": "1", "PLAN_ROOT": str(root), "MEMGUARD_GB": "",
               "PY": sys.executable, "PYTHONPATH": str(ROOT / "python")}
        p = subprocess.run(["bash", str(ROOT / "tools/fhe_ssm/make_plan.sh"), str(cap), f"plan_{name}"], env=env,
                           capture_output=True, text=True, timeout=5400)
        assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    c = plan_census(cap, plan, 0)
    cc = plan / "compile_check.json"
    check = json.load(open(cc))["blocks"][0] if cc.exists() else None
    return plan, c, check


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
def test_carry_drop_and_entry_in_capture(tmp_path, monkeypatch):
    root = pathlib.Path(os.environ.get("RWKV7_SL_DIR", str(tmp_path)))
    base = _capture(root, "base", {}, monkeypatch)
    drop = _capture(root, "drop3", {"RWKV7_CARRY_DROP": "3"}, monkeypatch)
    entry = _capture(root, "entry32", {"RWKV7_STATE_ENTRY": "32"}, monkeypatch)
    cb, lb, rb = _carries(base)
    cd, ld, rd = _carries(drop)
    ce, le, re_ = _carries(entry)
    print("base  carries", sorted({(tuple(n["input_levels"]), n["output_level"]) for n in cb}), "S in", lb, "reduces", len(rb))
    print("drop3 carries", sorted({(tuple(n["input_levels"]), n["output_level"]) for n in cd}), "S in", ld, "reduces", len(rd))
    print("entry carries", sorted({(tuple(n["input_levels"]), n["output_level"]) for n in ce}), "S in", le, "reduces", len(re_))
    # baseline: 4 S carries + xa + xf, S in at the landing, the carry refresh from the landing + the pass's 6 units
    assert len(cb) == 6 and lb == [LAND] and not rb
    s_base = [n for n in cb if n["input_levels"][0] == LAND + 6]
    assert len(s_base) == 4 and all(n["output_level"] == LAND for n in s_base)
    # drop 3: the S carries land 2 x 3 units deeper and start 6 units above that; xa / xf as before
    s_drop = [n for n in cd if n["output_level"] == LAND + 2 * 3]
    assert len(s_drop) == 4 and ld == [LAND + 6] and all(n["input_levels"][0] == LAND + 6 + 6 for n in s_drop)
    assert sorted(n["output_level"] for n in cd if n not in s_drop) == [LAND, LAND]
    meta = json.load(open(drop / "rwkv7_capture.json"))
    assert meta["carry_drop"] == 3 and meta["state_entry"] == 0
    # entry 32: S enters at 32, its carry refresh starts at 38 and lands at 26, then one level_reduce per S per token
    assert le == [32] and len(re_) == 4
    s_entry = [n for n in ce if n["input_levels"][0] == 32 + 6]
    assert len(s_entry) == 4 and all(n["output_level"] == LAND for n in s_entry)
    assert json.load(open(entry / "rwkv7_capture.json"))["state_entry"] == 32
    if os.environ.get("RWKV7_PLAN") != "1":
        return
    res = {}
    for name, cap in (("base", base), ("drop3", drop), ("entry32", entry)):
        plan, c, check = _plan(root, cap, name, monkeypatch)
        res[name] = {"placed": c["refreshes"], "by_step": c["by_step"],
                     "compile": None if check is None else {"sites": check["sites"], "errors": check["errors"],
                                                             "richer": check["level_richer"], "routes": check["routes"],
                                                             "launches": check["bootstrap_launches"]}}
        print(name, json.dumps(res[name]))
    json.dump(res, open(root / "state_levers_plans.json", "w"), indent=1)
    # the levers move no placed refresh: the same placed count as the baseline, nothing in the per-conversation scopes
    for name in ("drop3", "entry32"):
        assert res[name]["placed"] <= res["base"]["placed"], name
        assert not any(k.endswith(("update", "bcast", "readout")) for k in res[name]["by_step"]), name
    # the second pass of a two-pass cycle binds on this runtime: no error, no level richer than the capture's
    if res["entry32"]["compile"] is not None:
        assert res["entry32"]["compile"]["errors"] == [] and res["entry32"]["compile"]["richer"] == 0
    # the dropped COMPLEX carries do not (2026-11-05, compile_check of this checkout): the dense StC-first route of a
    # complex payload has no drop variant in the compiler's landing table ("no route lands at (32, 2) from (38, 2)"),
    # while the real route has one (dense_stcf_real@32 in the same check). Until the runtime adds the variant the flag
    # is a capture-side experiment for the filled layout's S; every error must be of that one kind
    if res["drop3"]["compile"] is not None and res["drop3"]["compile"]["errors"]:
        errs = res["drop3"]["compile"]["errors"]
        root = [e for e in errs if "no route lands at (32, 2)" in e]
        assert root, errs[:3]
        # its consequences: the carries' outputs stay at 38 ([plan_level_error] expected 32 actual 38) and the op
        # accounting no longer balances; nothing else
        assert all(("no route lands at (32, 2)" in e) or ("[plan_level_error]" in e) or ("op accounting" in e)
                   for e in errs), errs[:3]
        print(f"drop3: {len(errs)} compile errors: {len(root)} 'no route lands at (32, 2)' (the complex route's drop "
              f"variant) and their consequences")
