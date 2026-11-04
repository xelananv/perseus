# SPDX-License-Identifier: MIT
"""The box's CPU steps on a 1-block model: shifts + capture (run_capture), then, when RWKV7_PLAN=1, the planner
(tools/fhe_ssm/make_plan.sh, LANDING=high) on that capture; the plan must hold every stage."""
import json
import os
import subprocess
import sys

import pytest

from perseus.examples.rwkv7 import run_capture
from perseus.examples.rwkv7.tests.conftest import CKPT, COEFFS, ROOT
from perseus.examples.rwkv7.tests.test_ref import HARNESS, need

HIGH = {"FIDESLIB_BTS_STC_FIRST": "2", "FIDESLIB_BTS_STC_FIRST_ROUTES": "dense", "FIDESLIB_BTS_STC_FIRST_CF": "13",
        "FIDESLIB_BTS_RAISE_DROP": "512:1", "AUTO_BTS_LEVEL": "44"}


@need
def test_capture_and_plan_one_block(tmp_path, monkeypatch):
    pytest.importorskip("networkx", reason="the planner (perseus.plan) needs networkx")
    monkeypatch.setenv("RWKV7_PLAN", "1")
    test_capture_one_block(tmp_path, monkeypatch)


@need
def test_capture_one_block(tmp_path, monkeypatch):
    for k, v in HIGH.items():
        monkeypatch.setenv(k, v)
    cap = tmp_path / "cap"
    assert run_capture.main(["--pth", CKPT, "--coeffs", COEFFS, "--harness", HARNESS, "--blocks", "1",
                             "--out", str(cap)]) == 0
    for b in (0, 1):
        g = json.load(open(cap / f"block_{b}" / "graph.json"))
        ops = {}
        for n in g["nodes"]:
            ops[n["op_type"]] = ops.get(n["op_type"], 0) + 1
        print(f"block_{b}: {len(g['nodes'])} nodes", dict(sorted(ops.items())))
    names = [n["inputs"] for n in json.load(open(cap / "block_0" / "graph.json"))["nodes"]]
    assert any("cache.s0" in x for ins in names for x in ins)
    shifts = json.load(open(cap / "pcmm_shifts.json"))
    print("shifts", shifts)
    assert set(shifts) == {"b0.l1r", "b0.l1k", "b0.l1v", "b0.l1a", "b0.l1w1", "b0.l1g1", "b0.w2", "b0.g2",
                           "b0.out", "b0.fk", "b0.fv"}
    if os.environ.get("RWKV7_PLAN") != "1":
        return
    env = {**os.environ, "LANDING": "high", "PLAN_ROOT": str(tmp_path), "MEMGUARD_GB": "",
           "PY": sys.executable, "PYTHONPATH": str(ROOT / "python")}
    p = subprocess.run(["bash", str(ROOT / "tools/fhe_ssm/make_plan.sh"), str(cap), "plan1"], env=env,
                       capture_output=True, text=True, timeout=3000)
    print(p.stdout[-3000:], p.stderr[-3000:])
    plan = tmp_path / "bootstrap_placements" / "plan1"
    assert (plan / "block_0_placement.json").exists() and (plan / "block_1_placement.json").exists()
