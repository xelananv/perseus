# SPDX-License-Identifier: MIT
"""P6 (ROBUST_NORMS.md): the 1/sqrt sites as Newton from a constant seed (`method: newton_const`).

The converter turns coeffs_mixed.json into coeffs_mixed_nc.json (only the 1/sqrt sites change) for 0.1B and 0.4B and
writes them next to the harness (`nc_coeffs/<tag>/`) and, for 0.1B, into .cache/rwkv7/ for the other tests
(RWKV7_COEFFS=.cache/rwkv7/coeffs_mixed_nc.json runs test_fake / test_mirror on them). The new sites agree with exact
1/sqrt across the widened range and stay finite and positive up to the divergence bound; P5's retired capture text
(CALIB_TEXT_LONG_V1) keeps every 1/sqrt site inside its range on the unclamped poly model."""
import json
import os
import pathlib
import re
import sys

import numpy as np
import pytest

from perseus.examples.rwkv7 import common, ref
from perseus.examples.rwkv7.tests.conftest import CACHE, CKPT, ROOT, TOOLS_COEFFS
from perseus.examples.rwkv7.tests.test_ref import HARNESS, tokenizer

RES = pathlib.Path(HARNESS).parents[1]                     # results/
MODELS = {
    "0.1b": pathlib.Path(CACHE) if (pathlib.Path(CACHE) / "coeffs_mixed.json").exists() else TOOLS_COEFFS / "0.1b",
    "0.4b": (RES / "rwkv7-fhe-20261104/new_pod/vd_runs_0.4b"
             if (RES / "rwkv7-fhe-20261104/new_pod/vd_runs_0.4b/coeffs_mixed.json").exists()
             else TOOLS_COEFFS / "0.4b"),
}
NC_01B = (pathlib.Path(CACHE) / "coeffs_mixed_nc.json"
          if (pathlib.Path(CACHE) / "coeffs_mixed_nc.json").exists()
          else TOOLS_COEFFS / "0.1b" / "coeffs_mixed_nc.json")
R, TOL = 10.0, 1e-3


def _convert():
    sys.path.insert(0, HARNESS)
    import convert_newton_const as cv
    return cv


@pytest.mark.parametrize("tag", sorted(MODELS))
def test_convert(tag, tmp_path):
    d = MODELS[tag]
    if not (d / "coeffs_mixed.json").exists():
        pytest.skip(f"no {d}/coeffs_mixed.json")
    cv = _convert()
    old = json.load(open(d / "coeffs_mixed.json"))
    new, rows = cv.convert(old, json.load(open(d / "ranges_ctx4096.json")), R, (), TOL)
    out_dir = tmp_path / "nc_coeffs" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(new, open(out_dir / "coeffs_mixed_nc.json", "w"), indent=1)
    json.dump({"rows": rows, "summary": cv.summary(rows)}, open(out_dir / "convert_report.json", "w"), indent=1)
    print(json.dumps(cv.summary(rows), indent=1))
    n_old = sum(s.get("method") == "newton" for s in old["sites"].values())
    assert len(rows) == n_old and n_old > 0
    for name, s in old["sites"].items():
        t = new["sites"][name]
        if s.get("method") != "newton":
            assert t == s, name                                # every other site unchanged
            continue
        assert t["method"] == "newton_const" and re.search(r"rsqrt$", name)
        assert t["lo"] <= t["calib_min"] / R * (1 + 1e-12) and t["hi"] >= t["calib_max"] * R * (1 - 1e-12)
        assert t["conv_hi"] >= 2.9 * t["hi"]                   # divergence bound u = 3 at 3x the widened top
        assert t["max_rel_err"] <= TOL


def test_newton_const_exact_across_range():
    if not NC_01B.exists():
        pytest.skip("run test_convert first")
    cfg = json.load(open(NC_01B))["sites"]
    worst = 0.0
    for name, s in cfg.items():
        if s.get("method") != "newton_const":
            continue
        x = np.geomspace(max(s["lo"], 1e-30), s["hi"], 3001)
        exact = ref.SITE_FUNCS[ref.site_kind(name)](x)
        rel = np.abs(ref.newton_const_rsqrt(x, s) / exact - 1.0)
        worst = max(worst, float(rel.max()))
        assert rel.max() <= TOL, (name, float(rel.max()))
        # between the widened top and the divergence bound: positive, finite, never above 1/sqrt (from below)
        xb = np.linspace(s["hi"], 0.98 * s["conv_hi"], 200)
        yb = ref.newton_const_rsqrt(xb, s)
        assert np.all(np.isfinite(yb)) and np.all(yb > 0) and np.all(yb <= ref.SITE_FUNCS[ref.site_kind(name)](xb)
                                                                        * (1 + 1e-12)), name
        # far below the range (var = 0 for the norms): finite, positive, under-normalised, no blow-up
        y0 = ref.newton_const_rsqrt(np.array([0.0]), s)
        assert np.isfinite(y0).all() and (y0 > 0).all()
    print("worst rel err over every widened range:", worst)


@pytest.mark.skipif(not (os.path.exists(CKPT) and NC_01B.exists()), reason="no checkpoint / nc coeffs")
def test_plan_cost_old_vs_nc(tmp_path, monkeypatch):
    """The refresh cost of the nc sites: a 1-block decode capture + plan (LANDING=high) with the old and the nc
    coefficients; prints the planner's refreshes per block and where they sit. Needs --rwkv7-plan; keeps the
    products under RWKV7_NC_DIR if set."""
    if os.environ.get("RWKV7_PLAN") != "1":
        pytest.skip("plans: run with --rwkv7-plan")
    pytest.importorskip("networkx")
    from perseus.examples.rwkv7 import run_capture
    from perseus.examples.rwkv7.tests.conftest import ROOT
    from perseus.examples.rwkv7.tests.test_batched import plan_census
    from perseus.examples.rwkv7.tests.test_capture import HIGH
    import subprocess
    for k, v in HIGH.items():
        monkeypatch.setenv(k, v)
    root = pathlib.Path(os.environ.get("RWKV7_NC_DIR", str(tmp_path)))
    res = {}
    for tag, co in (("old", MODELS["0.1b"] / "coeffs_mixed.json"), ("nc", NC_01B)):
        cap = root / f"cap_{tag}"
        if not (cap / "capture_config.json").exists():
            assert run_capture.main(["--pth", CKPT, "--coeffs", str(co), "--harness", HARNESS, "--blocks", "1",
                                     "--out", str(cap)]) == 0
        plan = root / "bootstrap_placements" / f"plan_{tag}"
        if not (plan / "block_0_placement.json").exists():
            env = {**os.environ, "LANDING": "high", "PLAN_ROOT": str(root), "MEMGUARD_GB": "",
                   "PY": sys.executable, "PYTHONPATH": str(ROOT / "python")}
            p = subprocess.run(["bash", str(ROOT / "tools/fhe_ssm/make_plan.sh"), str(cap), f"plan_{tag}"], env=env,
                               capture_output=True, text=True, timeout=5400)
            assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
        c = plan_census(cap, plan, 0)
        res[tag] = {"refreshes": c["refreshes"], "by_step": c["by_step"], "nodes": sum(c["ops"].values())}
        print(tag, json.dumps(res[tag]))
    json.dump(res, open(root / "plan_cost_old_vs_nc.json", "w"), indent=1)


@pytest.mark.skipif(not (os.path.exists(CKPT) and NC_01B.exists()), reason="no checkpoint / nc coeffs")
def test_scan_ranges_script(tmp_path):
    """The harness's scan_ranges.py (torch, unclamped) on P5's retired capture text: runs, and with the nc file no
    1/sqrt kind leaves its range or its convergence region."""
    pytest.importorskip("torch")
    sys.path.insert(0, HARNESS)
    import scan_ranges
    txt = tmp_path / "t.txt"
    txt.write_text(common.CALIB_TEXT_LONG_V1)
    out = tmp_path / "scan.json"
    assert scan_ranges.main(["--model", CKPT, "--coeffs", str(NC_01B), "--text", str(txt), "--tokens", "96",
                             "--ctx", "96", "--device", "cpu", "--out", str(out)]) == 0
    r = json.load(open(out))
    assert r["tokens"] > 32
    for k, d in r["kinds"].items():
        if k.endswith("rsqrt"):
            assert d["tokens_out"] == 0 and d["tokens_conv_out"] == 0 and d["nonfinite"] == 0, (k, d)


@pytest.mark.skipif(not (os.path.exists(CKPT) and NC_01B.exists()), reason="no checkpoint / nc coeffs")
def test_genesis_v1_capture_guard_passes_nc(tmp_path, monkeypatch):
    """P5's failing capture (retired text, 8 blocks, T = 32; the guard stopped at block 7 gn_rsqrt, pass 0 lane 17)
    with the nc coefficients: the SiteGuard (accuracy ranges for the nc sites, intervals elsewhere) passes."""
    from perseus.examples.rwkv7 import run_capture
    from perseus.examples.rwkv7.tests.test_capture import HIGH
    for k, v in HIGH.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(common, "CALIB_TEXT_LONG", common.CALIB_TEXT_LONG_V1)
    assert run_capture.main(["--pth", CKPT, "--coeffs", str(NC_01B), "--harness", HARNESS, "--blocks", "8",
                             "--prefill-chunk", "32", "--pcmm-shift", "zero", "--out", str(tmp_path / "cap")]) == 0


def test_harness_files_compile():
    """Syntax check of the P6 harness files where torch is absent (this Mac's venv)."""
    for f in ("scan_ranges.py", "rsqrt_const.py", "convert_newton_const.py", "fit.py", "rwkv7_model.py"):
        compile(open(os.path.join(HARNESS, f)).read(), f, "exec")


def test_coeffs_option_reaches_tests():
    """--rwkv7-coeffs X must reach the module-level COEFFS the other tests import (so test_fake / test_mirror run
    on the nc file)."""
    from perseus.examples.rwkv7.tests import test_ref
    want = os.environ.get("RWKV7_COEFFS")
    print("COEFFS =", test_ref.COEFFS)
    if want:
        assert test_ref.COEFFS == want


@pytest.mark.skipif(not (os.path.exists(CKPT) and NC_01B.exists()), reason="no checkpoint / nc coeffs")
def test_genesis_v1_passes_rsqrt_sites():
    """P5's retired capture text (block 7 gn_rsqrt at 15.82 vs [0.000984, 10.46]) on the unclamped poly model with
    the nc coefficients: no 1/sqrt site leaves its range; the non-norm sites it still drives out are printed."""
    from perseus.examples.rwkv7.run_scan import report, scan
    ids = tokenizer().encode(common.CALIB_TEXT_LONG_V1)
    first = scan(ref.load_params(CKPT), str(NC_01B), ids, 1, 80)
    print("\n".join(report(first)) or "all inside")
    bad = [k for k in first if k.endswith("rsqrt")]
    assert not bad, report({k: first[k] for k in bad})
