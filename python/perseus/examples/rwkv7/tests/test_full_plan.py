# SPDX-License-Identifier: MIT
"""The box's CPU steps on all 12 blocks: shifts + capture, then the plan (LANDING=high). slow_shape (about 4 GB,
a few minutes). Writes into RWKV7_FULL_OUT (default: a pytest tmp dir) so the products can be kept."""
import json
import os
import subprocess
import sys

import pytest

from perseus.examples.rwkv7 import run_capture
from perseus.examples.rwkv7.tests.conftest import CKPT, COEFFS, ROOT
from perseus.examples.rwkv7.tests.test_capture import HIGH
from perseus.examples.rwkv7.tests.test_ref import HARNESS, need


@need
@pytest.mark.slow_shape
def test_full_capture_and_plan(tmp_path, monkeypatch):
    pytest.importorskip("networkx")
    out = os.environ.get("RWKV7_FULL_OUT", str(tmp_path))
    for k, v in HIGH.items():
        monkeypatch.setenv(k, v)
    cap = os.path.join(out, "cap_rwkv7")
    if not os.path.exists(os.path.join(cap, "capture_config.json")):
        assert run_capture.main(["--pth", CKPT, "--coeffs", COEFFS, "--harness", HARNESS, "--out", cap]) == 0
    n = [len(json.load(open(os.path.join(cap, f"block_{b}", "graph.json")))["nodes"]) for b in range(13)]
    print("nodes per stage", n)
    env = {**os.environ, "LANDING": "high", "PLAN_ROOT": out, "MEMGUARD_GB": "", "PY": sys.executable,
           "PYTHONPATH": str(ROOT / "python")}
    p = subprocess.run(["bash", str(ROOT / "tools/fhe_ssm/make_plan.sh"), cap, "plan_rwkv7"], env=env,
                       capture_output=True, text=True, timeout=7200)
    print(p.stdout[-2500:])
    assert p.returncode == 0, p.stderr[-2000:]
