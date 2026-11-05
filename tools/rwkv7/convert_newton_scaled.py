#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""coeffs JSON with ``newton_const`` 1/sqrt sites -> the same sites as ``newton_scaled`` (every other site byte-equal).

    PYTHONPATH=python python tools/rwkv7/convert_newton_scaled.py --coeffs coeffs_final.json --out coeffs_final_sc.json
        [--tol 1e-3] [--zero 3.0]

Per-step (a, b) constants from ``perseus.impl.poly.scaled_newton_steps`` on newton_const's seed, pre-scale, eps,
accuracy range and divergence bound (step 1 is (1.5, 0.5), its zero at u = ``zero``; every later step keeps t = u y^2
<= 1): ~6.7x of range ratio per step instead of 2.25x, about half the depth. The error profile differs from
newton_const's (ROBUST_NORMS.md "newton_scaled"): the per-site checks here, and the gate, decide whether a converted file
may be used. Site fields: steps, n_steps, depth, const_steps, const_depth, max_rel_err, blow_at (x / conv_hi).
"""
from __future__ import annotations

import argparse
import json
import re
import sys

import numpy as np


def scaled_rsqrt_np(x, p: dict) -> np.ndarray:
    """1/sqrt(x + eps) by the scaled steps (the converter's check, the port's mirror and the harness use this)."""
    u = (np.asarray(x, dtype=np.float64) + p["eps"]) / p["scale"]
    a, b = p["steps"][0]
    y = a - b * u
    for a, b in p["steps"][1:]:
        y = a * y - b * u * y * y * y
    return y * p["scale"] ** -0.5


def exact(x, p: dict) -> np.ndarray:
    return 1.0 / np.sqrt(np.asarray(x, dtype=np.float64) + p["eps"])


def max_rel_err(p: dict, lo: float, hi: float, n: int = 4001) -> float:
    """max |sqrt(x + eps) y - 1| over [lo, hi] (log grid in the argument)."""
    arg = np.geomspace(lo + p["eps"], hi + p["eps"], n)
    return float(np.max(np.abs(scaled_rsqrt_np(arg - p["eps"], p) * np.sqrt(arg) - 1.0)))


def scaled_site(s: dict, tol: float, zero: float) -> dict:
    """The newton_scaled site for a newton_const site ``s`` (same range, scale and eps)."""
    from perseus.impl.poly import scaled_newton_steps
    u_lo = (float(s["lo"]) + float(s["eps"])) / float(s["scale"])
    if u_lo <= 0.0:
        raise ValueError("newton_scaled: the widened bottom plus eps must be > 0")
    ratio = 1.0 / u_lo
    steps = [[float(a), float(b)] for a, b in scaled_newton_steps(ratio, tol, zero)]
    out = {k: s[k] for k in ("kind", "short") if k in s}
    out.update({"method": "newton_scaled", "eps": float(s["eps"]), "scale": float(s["scale"]), "steps": steps,
                "n_steps": len(steps), "depth": 1 + 2 * (len(steps) - 1), "lo": float(s["lo"]), "hi": float(s["hi"]),
                "conv_lo": -float(s["eps"]), "conv_hi": zero * float(s["scale"]) - float(s["eps"]),
                "R": s.get("R"), "tol": float(tol), "zero": float(zero), "ratio": ratio,
                "calib_min": s.get("calib_min"), "calib_max": s.get("calib_max"),
                "const_steps": int(s["steps"]), "const_depth": 1 + 2 * (int(s["steps"]) - 1)})
    out["max_rel_err"] = max_rel_err(out, out["lo"], out["hi"])
    if out["calib_min"] is not None and out["calib_max"] is not None:
        out["max_rel_err_calib"] = max_rel_err(out, max(float(out["calib_min"]), 0.0), float(out["calib_max"]))
    return out


def robust(p: dict) -> dict:
    """The robustness checks on the converted site: finite, positive and from below between the widened top and the
    divergence bound conv_hi (inclusive); finite and positive at zero; and ``blow_at``, the smallest x / conv_hi in
    [1, 1.5] where the output is non-finite or 1,000x the exact value (None if it stays bounded there)."""
    xb = np.linspace(p["hi"], p["conv_hi"], 400)
    yb = scaled_rsqrt_np(xb, p)
    y0 = scaled_rsqrt_np(np.array([0.0]), p)
    xo = p["conv_hi"] * np.linspace(1.0, 1.5, 2001)
    with np.errstate(all="ignore"):
        yo = scaled_rsqrt_np(xo, p)
        bad = np.flatnonzero(~np.isfinite(yo) | (np.abs(yo) > 1e3 * exact(xo, p)))
    # at conv_hi itself u = 3 and y = 0 (rounding may give -1e-16): under-normalised, never sign-flipped
    return {"finite_to_conv_hi": bool(np.isfinite(yb).all()), "positive_to_conv_hi": bool((yb >= -1e-12).all()),
            "from_below_to_conv_hi": bool((yb <= exact(xb, p) * (1 + 1e-12)).all()),
            "finite_at_zero": bool(np.isfinite(y0).all() and (y0 > 0).all()),
            "blow_at": float(xo[bad[0]] / p["conv_hi"]) if bad.size else None}


def convert(coeffs: dict, tol: float = 1e-3, zero: float = 3.0) -> tuple[dict, list]:
    out = json.loads(json.dumps(coeffs))
    rows = []
    for name, s in coeffs["sites"].items():
        if s.get("method") != "newton_const":
            continue
        sc = scaled_site(s, tol, zero)
        r = robust(sc)
        ok = all(v for k, v in r.items() if k != "blow_at")
        if sc["max_rel_err"] > tol * (1 + 1e-9) or not ok:
            raise ValueError(f"{name}: rel err {sc['max_rel_err']:.3e}, robust {r}")
        sc["blow_at"] = r["blow_at"]
        out["sites"][name] = sc
        rows.append({"name": name, "short": s.get("short", name), "const_steps": int(s["steps"]),
                     "const_depth": sc["const_depth"], "steps": sc["n_steps"], "depth": sc["depth"],
                     "ratio": sc["ratio"], "rel": sc["max_rel_err"], "rel_calib": sc.get("max_rel_err_calib"), **r})
    out.setdefault("meta", {})["newton_scaled"] = {"tol": tol, "zero": zero, "converted": len(rows)}
    return out, rows


def summary(rows) -> dict:
    by = {}
    for r in rows:
        k = re.sub(r"^blocks\.\d+\.", "", r["name"])
        d = by.setdefault(k, {"n": 0, "const_steps": [], "steps": [], "const_depth": [], "depth": []})
        d["n"] += 1
        for f in ("const_steps", "steps", "const_depth", "depth"):
            d[f].append(r[f])
    return {k: {"n": d["n"], **{f: [min(d[f]), max(d[f]), round(sum(d[f]) / d["n"], 2)]
                                for f in ("const_steps", "steps", "const_depth", "depth")}} for k, d in by.items()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coeffs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--zero", type=float, default=3.0)
    a = ap.parse_args(argv)
    with open(a.coeffs) as f:
        coeffs = json.load(f)
    out, rows = convert(coeffs, a.tol, a.zero)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    for r in rows:
        calib = "n/a" if r["rel_calib"] is None else f"{r['rel_calib']:.2e}"
        print(f"{r['name']:28s} const {r['const_steps']:2d} st d{r['const_depth']:2d}  ->  scaled {r['steps']:2d} st "
              f"d{r['depth']:2d}  ratio {r['ratio']:.3g}  rel {r['rel']:.2e} (calib {calib})  blow_at {r['blow_at']}")
    s = summary(rows)
    print(json.dumps(s, indent=1))
    tot_c = sum(v["const_depth"][2] for v in s.values())
    tot_s = sum(v["depth"][2] for v in s.values())
    print(f"mean depth of the four chains per block: {tot_c:.1f} -> {tot_s:.1f} levels ({tot_s - tot_c:+.1f})")
    calibs = [r["rel_calib"] for r in rows if r["rel_calib"] is not None]
    blows = [r["blow_at"] for r in rows if r["blow_at"] is not None]
    print(f"max rel err: accuracy range {max(r['rel'] for r in rows):.3e}, calibrated range "
          f"{max(calibs):.3e}" if calibs else "n/a (no calib_min / calib_max)", end="; ")
    print(f"blow_at min {min(blows)}" if blows else "no overflow below 1.5 x conv_hi")
    print(f"wrote {a.out}: {len(rows)} sites converted")
    return 0


if __name__ == "__main__":
    sys.exit(main())
