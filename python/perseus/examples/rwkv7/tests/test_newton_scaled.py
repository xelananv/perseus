# SPDX-License-Identifier: MIT
"""The 1/sqrt sites as scaled Newton (``method: newton_scaled``, tools/rwkv7/convert_newton_scaled.py; the speed-up
map's T1, results/rwkv7-fhe-20261104/SPEEDUP_MAP_20261105.md).

The converter turns a coefficients file's ``newton_const`` sites into ``newton_scaled`` ones (per-step constants from
``perseus.impl.poly.scaled_newton_steps``; every other site byte-equal). The tests convert the 0.1B's final
coefficients into a temporary file (a module fixture; nothing is written into .cache) and check: the error over the
accuracy range and the calibrated range (against the newton_const sites' own), the behaviour above the range up to the
divergence bound and the overflow beyond it, fewer steps, the port's ciphertext-side arithmetic against the numpy mirror
in and out of range, and (with --rwkv7-plan) the planner's count on a 1-block decode capture. Default off: no shipped
coefficients file carries the method."""
import json
import os
import pathlib
import re
import sys

import numpy as np
import pytest

from perseus.examples.rwkv7 import ref
from perseus.examples.rwkv7.tests.conftest import CACHE, CKPT, COEFFS, ROOT, TOOLS_COEFFS
from perseus.examples.rwkv7.tests.test_ref import HARNESS

FINAL = (pathlib.Path(CACHE) / "coeffs_final.json"
         if (pathlib.Path(CACHE) / "coeffs_final.json").exists()
         else TOOLS_COEFFS / "0.1b" / "coeffs_final.json")  # the 0.1B gate-3 coefficients (newton_const sites)
TOL = 1e-3


def _conv():
    sys.path.insert(0, str(ROOT / "tools" / "rwkv7"))
    import convert_newton_scaled as cv
    return cv


@pytest.fixture(scope="module")
def converted(tmp_path_factory):
    """(the newton_const file, the converted file's path, its dict, the converter's rows)."""
    if not FINAL.exists():
        pytest.skip("no .cache/rwkv7/coeffs_final.json")
    cv = _conv()
    with open(FINAL) as f:
        old = json.load(f)
    new, rows = cv.convert(old, TOL, 3.0)
    path = tmp_path_factory.mktemp("sc") / "coeffs_final_sc.json"
    with open(path, "w") as f:
        json.dump(new, f, indent=1)
    return old, path, new, rows


def test_convert_final(converted):
    cv = _conv()
    old, _path, new, rows = converted
    n_const = sum(s.get("method") == "newton_const" for s in old["sites"].values())
    assert n_const == 48 and len(rows) == n_const
    for name, s in old["sites"].items():
        t = new["sites"][name]
        if s.get("method") != "newton_const":
            assert t == s, name                                   # every other site unchanged
            continue
        assert t["method"] == "newton_scaled" and re.search(r"rsqrt$", name)
        for k in ("eps", "scale", "lo", "hi"):                    # the same range, scale and eps
            assert t[k] == float(s[k]), (name, k)
        assert t["conv_hi"] == 3.0 * float(s["scale"]) - float(s["eps"])
        assert t["n_steps"] < int(s["steps"]) and t["depth"] < 1 + 2 * (int(s["steps"]) - 1), name
        assert t["max_rel_err"] <= TOL * (1 + 1e-9), (name, t["max_rel_err"])
        assert abs(t["steps"][0][0] - 1.5) < 1e-9 and abs(t["steps"][0][1] - 0.5) < 1e-9   # step 1 = newton_const's
    s = cv.summary(rows)
    print(json.dumps(s, indent=1))
    tot_c = sum(v["const_depth"][2] for v in s.values())
    tot_s = sum(v["depth"][2] for v in s.values())
    print(f"mean depth of the four chains per block: {tot_c:.1f} -> {tot_s:.1f}")
    assert tot_s < 0.7 * tot_c


def test_error_profile_against_newton_const(converted):
    """The measured difference (ROBUST_NORMS.md "newton_scaled"): both methods are within the tolerance over the
    accuracy range; on the calibrated range newton_const's extra steps converge completely while the scaled steps
    spread the tolerance evenly; below the range the scaled sites are about twice as inaccurate; above conv_hi the
    scaled iteration overflows where newton_const stays bounded. The numbers are asserted as bounds."""
    old, _path, new, _rows = converted
    in_c, in_s, cal_c, cal_s, lo2_c, lo2_s = [], [], [], [], [], []
    for name, s in old["sites"].items():
        if s.get("method") != "newton_const":
            continue
        t = new["sites"][name]
        f = ref.SITE_FUNCS[ref.site_kind(name)]
        x = np.geomspace(max(s["lo"], 1e-30), s["hi"], 3001)
        in_c.append(np.abs(ref.newton_const_rsqrt(x, s) / f(x) - 1.0).max())
        in_s.append(np.abs(ref.newton_scaled_rsqrt(x, t) / f(x) - 1.0).max())
        xc = np.geomspace(max(s["calib_min"], 1e-30), s["calib_max"], 2001)
        cal_c.append(np.abs(ref.newton_const_rsqrt(xc, s) / f(xc) - 1.0).max())
        cal_s.append(np.abs(ref.newton_scaled_rsqrt(xc, t) / f(xc) - 1.0).max())
        xl = np.array([s["lo"] / 2])
        lo2_c.append(float(np.abs(ref.newton_const_rsqrt(xl, s) / f(xl) - 1.0)[0]))
        lo2_s.append(float(np.abs(ref.newton_scaled_rsqrt(xl, t) / f(xl) - 1.0)[0]))
        # above the range to conv_hi inclusive: finite, >= 0, from below (y = 0 exactly at conv_hi: under-normalised)
        xb = np.linspace(s["hi"], t["conv_hi"], 300)
        yb = ref.newton_scaled_rsqrt(xb, t)
        assert np.all(np.isfinite(yb)) and np.all(yb >= -1e-12) and np.all(yb <= f(xb) * (1 + 1e-12)), name
        assert np.isfinite(ref.newton_scaled_rsqrt(np.array([0.0]), t)).all(), name
        # one scaled step fewer breaks the tolerance somewhere on the range (not vacuous)
        short = dict(t, steps=t["steps"][:-1])
        assert np.abs(ref.newton_scaled_rsqrt(x, short) / f(x) - 1.0).max() > TOL, name
        # beyond the divergence bound: the scaled output overflows from blow_at x conv_hi (recorded by the converter)
        assert t["blow_at"] is not None and 1.0 < t["blow_at"] <= 1.5, (name, t["blow_at"])
    in_c, in_s, cal_c, cal_s = map(np.array, (in_c, in_s, cal_c, cal_s))
    worse = int((cal_s > cal_c).sum())
    print(f"accuracy range: max rel err const {in_c.max():.3e} scaled {in_s.max():.3e}; calibrated range: median "
          f"const {np.median(cal_c):.2e} scaled {np.median(cal_s):.2e}, max {cal_c.max():.2e} / {cal_s.max():.2e}, "
          f"scaled worse on {worse}/48; lo/2: median const {np.median(lo2_c):.2e} scaled {np.median(lo2_s):.2e}; "
          f"blow_at min {min(t['blow_at'] for t in new['sites'].values() if t.get('blow_at'))}")
    assert in_c.max() <= TOL and in_s.max() <= TOL
    assert cal_s.max() <= TOL and np.median(cal_s) < 1e-4 and np.median(cal_c) < 1e-12
    assert worse >= 40 and np.median(lo2_s) < 3 * np.median(lo2_c)


def test_port_newton_equals_mirror_in_and_out_of_range(converted):
    """Circuit.newton on NumpyOps (the ciphertext-side arithmetic: the affine map folded onto x, the d2 product order)
    against ref.newton_scaled_rsqrt, on the raw statistic with its xscale: inside the accuracy range, above it up to
    conv_hi, and at zero."""
    from perseus.examples.rwkv7.layout import Layout
    from perseus.examples.rwkv7.model import Circuit, load_coeffs
    from perseus.impl.ops import NumpyOps
    _old, path, _new, _rows = converted
    co = load_coeffs(str(path))
    L = Layout(1 << 15)
    circ = Circuit(NumpyOps(), L, co, None)
    rng = np.random.default_rng(0)
    for name, xscale in (("blocks.0.ln1.rsqrt", 1.0 / (4 * 768)), ("blocks.7.att.gn_rsqrt", 1.0),
                         ("blocks.3.att.kk_rsqrt", 1.0 / co["blocks.3.att.kk_rsqrt"]["scale"])):
        s = co[name]
        assert s["method"] == "newton_scaled"
        n = L.n
        inside = np.exp(rng.uniform(np.log(s["lo"] + 1e-30), np.log(s["hi"]), n - 1001))
        above = np.linspace(s["hi"], s["conv_hi"], 1000)                        # out of range, to the bound
        stat = np.concatenate([inside, above, [0.0]])                            # harness units
        x = stat / xscale                                                        # the port's raw statistic
        y = np.real(np.asarray(circ.newton(name, x.astype(np.complex128), xscale))) * s["scale"] ** -0.5
        want = ref.newton_scaled_rsqrt(stat, s)
        assert np.isfinite(y).all() and np.isfinite(want).all(), name
        # relative where the value is not tiny (it falls to 0 at conv_hi), absolute to the site's largest value there
        err = float(np.max(np.abs(y - want) / (np.abs(want) + 1e-6 * np.abs(want).max())))
        print(name, "port vs mirror max rel", err)
        assert err < 1e-9, (name, err)
        assert float(np.max(np.abs(y[: n - 1001] * np.sqrt(stat[: n - 1001] + s["eps"]) - 1.0))) <= TOL
        assert np.all(y[n - 1001:] >= -1e-12)                                  # y = 0 at conv_hi (rounding)


def test_default_coefficients_carry_no_scaled_site():
    if not os.path.exists(COEFFS):
        pytest.skip("no coefficients")
    with open(COEFFS) as f:
        cfg = json.load(f)["sites"]
    assert not any(s.get("method") == "newton_scaled" for s in cfg.values())


@pytest.mark.skipif(not (os.path.exists(CKPT) and FINAL.exists()), reason="no checkpoint / coefficients")
def test_plan_cost_const_vs_scaled(converted, tmp_path, monkeypatch):
    """The refresh cost: a 1-block decode capture + plan (LANDING=high) with the final (newton_const) and the scaled
    coefficients; prints the planner's refreshes per block by step. Needs --rwkv7-plan; keeps the products under
    RWKV7_SC_DIR if set."""
    if os.environ.get("RWKV7_PLAN") != "1":
        pytest.skip("plans: run with --rwkv7-plan")
    pytest.importorskip("networkx")
    import subprocess

    from perseus.examples.rwkv7 import run_capture
    from perseus.examples.rwkv7.tests.test_batched import plan_census
    from perseus.examples.rwkv7.tests.test_capture import HIGH
    for k, v in HIGH.items():
        monkeypatch.setenv(k, v)
    _old, sc_path, _new, _rows = converted
    root = pathlib.Path(os.environ.get("RWKV7_SC_DIR", str(tmp_path)))
    res = {}
    for tag, co in (("const", FINAL), ("scaled", sc_path)):
        cap = root / f"cap_{tag}"
        if not (cap / "capture_config.json").exists():
            assert run_capture.main(["--pth", CKPT, "--coeffs", str(co), "--harness", HARNESS, "--blocks", "1",
                                     "--out", str(cap)]) == 0
        plan = root / "bootstrap_placements" / f"plan_{tag}"
        if not (plan / "block_0_placement.json").exists():
            env = {**os.environ, "LANDING": "high", "PLAN_ROOT": str(root), "MEMGUARD_GB": "",
                   "PY": sys.executable, "PYTHONPATH": str(ROOT / "python")}
            p = subprocess.run(["bash", str(ROOT / "tools/fhe_ssm/make_plan.sh"), str(cap), f"plan_{tag}"], env=env,
                               capture_output=True, text=True, timeout=5400, check=False)
            assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
        c = plan_census(cap, plan, 0)
        res[tag] = {"refreshes": c["refreshes"], "by_step": c["by_step"], "nodes": sum(c["ops"].values())}
        print(tag, json.dumps(res[tag]))
    with open(root / "plan_cost_const_vs_scaled.json", "w") as f:
        json.dump(res, f, indent=1)
    assert res["scaled"]["refreshes"] < res["const"]["refreshes"]
