# SPDX-License-Identifier: MIT
"""The 1/sqrt sites as scaled Newton (``method: newton_scaled``, tools/rwkv7/convert_newton_scaled.py; the speed-up
map's T1, results/rwkv7-fhe-20261104/SPEEDUP_MAP_20261105.md).

The converter turns a coefficients file's ``newton_const`` sites into ``newton_scaled`` ones (per-step constants from
``perseus.impl.poly.scaled_newton_steps``; every other site byte-equal) and writes ``coeffs_final_sc.json`` next to the
0.1B's final coefficients in .cache/rwkv7/ (RWKV7_COEFFS=.cache/rwkv7/coeffs_final_sc.json runs test_fake / test_mirror
on it). The new sites agree with exact 1/sqrt over the accuracy range, stay finite, positive and from below up to the
divergence bound, take fewer steps than the constant-seed sites, and the port's ciphertext-side evaluation equals the
numpy mirror's. Default off: no shipped coefficients file carries the method."""
import json
import os
import pathlib
import re
import sys

import numpy as np
import pytest

from perseus.examples.rwkv7 import ref
from perseus.examples.rwkv7.tests.conftest import CACHE, CKPT, COEFFS, ROOT
from perseus.examples.rwkv7.tests.test_ref import HARNESS

FINAL = pathlib.Path(CACHE) / "coeffs_final.json"             # the 0.1B gate-3 coefficients (newton_const sites)
SC = pathlib.Path(CACHE) / "coeffs_final_sc.json"
TOL = 1e-3


def _conv():
    sys.path.insert(0, str(ROOT / "tools" / "rwkv7"))
    import convert_newton_scaled as cv
    return cv


@pytest.mark.skipif(not FINAL.exists(), reason="no .cache/rwkv7/coeffs_final.json")
def test_convert_final():
    cv = _conv()
    old = json.load(open(FINAL))
    new, rows = cv.convert(old, TOL, 3.0)
    json.dump(new, open(SC, "w"), indent=1)
    n_const = sum(s.get("method") == "newton_const" for s in old["sites"].values())
    assert n_const > 0 and len(rows) == n_const
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


@pytest.mark.skipif(not SC.exists(), reason="run test_convert_final first")
def test_scaled_exact_and_robust_across_range():
    """ROBUST_NORMS.md's checks on the converted file, through the port's numpy mirror (ref.py)."""
    cfg = json.load(open(SC))["sites"]
    worst = 0.0
    n = 0
    for name, s in cfg.items():
        if s.get("method") != "newton_scaled":
            continue
        n += 1
        f = ref.SITE_FUNCS[ref.site_kind(name)]
        x = np.geomspace(max(s["lo"], 1e-30), s["hi"], 3001)
        rel = np.abs(ref.newton_scaled_rsqrt(x, s) / f(x) - 1.0)
        worst = max(worst, float(rel.max()))
        assert rel.max() <= TOL, (name, float(rel.max()))
        xb = np.linspace(s["hi"], 0.98 * s["conv_hi"], 300)       # above the range: finite, positive, from below
        yb = ref.newton_scaled_rsqrt(xb, s)
        assert np.all(np.isfinite(yb)) and np.all(yb > 0) and np.all(yb <= f(xb) * (1 + 1e-12)), name
        y0 = ref.newton_scaled_rsqrt(np.array([0.0]), s)          # var = 0: finite, positive
        assert np.isfinite(y0).all() and (y0 > 0).all(), name
        # not vacuous: one scaled step fewer breaks the tolerance somewhere on the range
        short = dict(s, steps=s["steps"][:-1])
        assert np.abs(ref.newton_scaled_rsqrt(x, short) / f(x) - 1.0).max() > TOL, name
    assert n == 48
    print("worst rel err over every accuracy range:", worst)


@pytest.mark.skipif(not SC.exists(), reason="run test_convert_final first")
def test_port_newton_equals_mirror():
    """Circuit.newton on NumpyOps (the ciphertext-side arithmetic: the affine map folded onto x, the d2 product order)
    against ref.newton_scaled_rsqrt, on the raw statistic with its xscale, for a LayerNorm and a GroupNorm site."""
    from perseus.examples.rwkv7.layout import Layout
    from perseus.examples.rwkv7.model import Circuit, load_coeffs
    from perseus.impl.ops import NumpyOps
    co = load_coeffs(str(SC))
    L = Layout(1 << 15)
    circ = Circuit(NumpyOps(), L, co, None)
    rng = np.random.default_rng(0)
    for name, xscale in (("blocks.0.ln1.rsqrt", 1.0 / (4 * 768)), ("blocks.7.att.gn_rsqrt", 1.0),
                         ("blocks.3.att.kk_rsqrt", 1.0 / co["blocks.3.att.kk_rsqrt"]["scale"])):
        s = co[name]
        assert s["method"] == "newton_scaled"
        stat = np.exp(rng.uniform(np.log(s["lo"] + 1e-30), np.log(s["hi"]), L.n))   # harness units, in range
        x = stat / xscale                                                               # the port's raw statistic
        y = np.real(np.asarray(circ.newton(name, x.astype(np.complex128), xscale))) * s["scale"] ** -0.5
        want = ref.newton_scaled_rsqrt(stat, s)
        err = float(np.max(np.abs(y / want - 1.0)))
        print(name, "port vs mirror max rel", err)
        assert err < 1e-10, (name, err)
        assert float(np.max(np.abs(y * np.sqrt(stat + s["eps"]) - 1.0))) <= TOL


def test_default_coefficients_carry_no_scaled_site():
    if not os.path.exists(COEFFS):
        pytest.skip("no coefficients")
    cfg = json.load(open(COEFFS))["sites"]
    assert not any(s.get("method") == "newton_scaled" for s in cfg.values())


@pytest.mark.skipif(not (os.path.exists(CKPT) and FINAL.exists() and SC.exists()), reason="no checkpoint / files")
def test_plan_cost_const_vs_scaled(tmp_path, monkeypatch):
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
    root = pathlib.Path(os.environ.get("RWKV7_SC_DIR", str(tmp_path)))
    res = {}
    for tag, co in (("const", FINAL), ("scaled", SC)):
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
    json.dump(res, open(root / "plan_cost_const_vs_scaled.json", "w"), indent=1)
    assert res["scaled"]["refreshes"] < res["const"]["refreshes"]
