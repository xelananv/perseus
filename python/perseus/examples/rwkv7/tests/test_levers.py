# SPDX-License-Identifier: MIT
"""The fhe_ssm stack's levers that transfer to this port (PREFILL_DESIGN.md §6), on the CPU:

* the per-linear PCMM CF table (``fhe_ssm/pcmm_cf_table.py``) builds from this model's capture and covers every
  linear the session declares;
* ``--schedule siblings`` reaches ``Program.compile`` (``Rwkv7PlannedRun._compile_with_schedule``) and the module is
  restored after it;
* with --rwkv7-plan: a PAIR=1 plan (LANDING=high, PC_BTS_PAIR=1 and PC_EM_FUSE_RESCALE=1 on the runtime= line) cuts
  from the same capture.
"""
import json
import os
import subprocess
import sys

import pytest

from perseus.examples.rwkv7 import run_capture
from perseus.examples.rwkv7.tests.conftest import CKPT, COEFFS, ROOT
from perseus.examples.rwkv7.tests.test_capture import HIGH
from perseus.examples.rwkv7.tests.test_ref import HARNESS, need


@pytest.fixture(scope="module")
def capture1(tmp_path_factory):
    cap = tmp_path_factory.mktemp("levers") / "cap"
    env0 = {k: os.environ.get(k) for k in HIGH}
    os.environ.update(HIGH)
    try:
        assert run_capture.main(["--pth", CKPT, "--coeffs", COEFFS, "--harness", HARNESS, "--blocks", "1",
                                 "--out", str(cap)]) == 0
    finally:
        for k, v in env0.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return cap


@need
def test_pcmm_cf_table_from_capture(capture1):
    from perseus.examples.fhe_ssm import pcmm_cf_table
    doc = pcmm_cf_table.build(capture1)
    out = capture1.parent / "cf.json"
    out.write_text(json.dumps(doc))
    cfs = pcmm_cf_table.load(out)
    # block 0's linears (impl/pcmm.py's calibration dict is process-global: in a full pytest run the shifts file
    # also carries linears an earlier test calibrated)
    want = {k for k in json.load(open(capture1 / "pcmm_shifts.json")) if k.startswith("b0.")}
    print("CF per linear", cfs)
    assert len(want) == 11 and want <= set(cfs), sorted(want - set(cfs))
    assert all(1 <= c <= 20 for c in cfs.values())


def test_schedule_reaches_compile(monkeypatch):
    from perseus import _pc
    from perseus.examples.rwkv7.run_planned import Rwkv7PlannedRun
    seen = []

    class Prog:
        @staticmethod
        def compile(*a, **k):
            seen.append(k.get("schedule"))
            return {}
    monkeypatch.setattr(_pc, "Program", Prog, raising=False)
    run = Rwkv7PlannedRun.__new__(Rwkv7PlannedRun)
    run.schedule = "siblings"
    run._compile_with_schedule(lambda: _pc.Program.compile("g", "p", config=None))
    assert seen == ["siblings"] and _pc.Program is Prog
    run.schedule = "program"
    run._compile_with_schedule(lambda: _pc.Program.compile("g", "p"))
    assert seen == ["siblings", None]


@need
def test_pair_em_plan(capture1):
    if os.environ.get("RWKV7_PLAN") != "1":
        pytest.skip("run with --rwkv7-plan")
    pytest.importorskip("networkx")
    root = capture1.parent
    env = {**os.environ, "LANDING": "high", "PAIR": "1", "RUNTIME_EXTRA": "PC_EM_FUSE_RESCALE=1",
           "PLAN_ROOT": str(root), "MEMGUARD_GB": "", "PY": sys.executable, "PYTHONPATH": str(ROOT / "python")}
    p = subprocess.run(["bash", str(ROOT / "tools/fhe_ssm/make_plan.sh"), str(capture1), "plan_pair"], env=env,
                       capture_output=True, text=True, timeout=3000)
    print(p.stdout[-1500:], p.stderr[-1500:])
    assert p.returncode == 0
    cmd = (root / "bootstrap_placements" / "plan_pair" / "PLAN_CMD.txt").read_text()
    assert "PC_BTS_PAIR=1" in cmd and "PC_EM_FUSE_RESCALE=1" in cmd
