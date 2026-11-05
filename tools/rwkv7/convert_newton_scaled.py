#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""coeffs JSON with ``newton_const`` 1/sqrt sites -> the same sites as ``newton_scaled`` (every other site byte-equal).

    PYTHONPATH=python python tools/rwkv7/convert_newton_scaled.py --coeffs coeffs_final.json --out coeffs_final_sc.json
        [--tol 1e-3] [--zero 3.0]

The mechanism (perseus/impl/poly.py ``scaled_newton_steps``, P6's "next lever" in ROBUST_NORMS.md): plain Newton for
1/sqrt(u) from the constant seed, y <- y (1.5 - 0.5 u y^2), raises y by at most 1.5x per step, so a site whose accuracy
range spans a ratio of 1e4 in u climbs for ~11 steps before it converges. With t = u y^2, a step maps t -> t (a - b t)^2;
the pair (a, b) = (1.5, 0.5) is one choice. Per-step constants that map the band [t_lo, 1] onto the widest image that stays
<= 1 (q peaks at exactly 1 inside the band, q(t_lo) = q(1)) cover ~6.7x of band ratio per step instead of 2.25x. The
first step keeps its zero at ``zero`` times the band top (3.0: the same divergence bound as newton_const, u = 3), and every
later step keeps t <= 1, so an input anywhere in (0, 3 scale) stays finite and positive and is under-normalised at worst,
never sign-flipped: the robustness argument of ROBUST_NORMS.md holds unchanged. The accuracy range [lo, hi], the pre-scale,
eps and the output convention (y scale^-1/2 = 1/sqrt(x + eps)) are the newton_const site's own.

Site JSON: method newton_scaled, steps [[a, b], ...] (step 1 affine: y = a - b u), n_steps, depth 1 + 2 (n_steps - 1),
and newton_const's fields (eps, scale, lo, hi, conv_lo, conv_hi, R, tol, calib_min, calib_max) plus const_steps.
"""
from __future__ import annotations

import argparse
import json
import math
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


def max_rel_err(p: dict, n: int = 4001) -> float:
    """max |sqrt(x + eps) y - 1| over the accuracy range (log grid in the argument)."""
    arg = np.geomspace(p["lo"] + p["eps"], p["hi"] + p["eps"], n)
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
    out["max_rel_err"] = max_rel_err(out)
    return out


def robust(p: dict) -> dict:
    """The robustness checks of ROBUST_NORMS.md on the converted site: finite, positive and from below between the
    widened top and the divergence bound; finite and positive at zero."""
    f = lambda x: 1.0 / np.sqrt(np.asarray(x, dtype=np.float64) + p["eps"])  # noqa: E731
    xb = np.linspace(p["hi"], 0.98 * p["conv_hi"], 400)
    yb = scaled_rsqrt_np(xb, p)
    y0 = scaled_rsqrt_np(np.array([0.0]), p)
    return {"finite_above": bool(np.isfinite(yb).all()), "positive_above": bool((yb > 0).all()),
            "from_below_above": bool((yb <= f(xb) * (1 + 1e-12)).all()),
            "finite_at_zero": bool(np.isfinite(y0).all() and (y0 > 0).all())}


def convert(coeffs: dict, tol: float = 1e-3, zero: float = 3.0) -> tuple[dict, list]:
    out = json.loads(json.dumps(coeffs))
    rows = []
    for name, s in coeffs["sites"].items():
        if s.get("method") != "newton_const":
            continue
        sc = scaled_site(s, tol, zero)
        r = robust(sc)
        if sc["max_rel_err"] > tol * (1 + 1e-9) or not all(r.values()):
            raise ValueError(f"{name}: rel err {sc['max_rel_err']:.3e}, robust {r}")
        out["sites"][name] = sc
        rows.append({"name": name, "short": s.get("short", name), "const_steps": int(s["steps"]),
                     "const_depth": sc["const_depth"], "steps": sc["n_steps"], "depth": sc["depth"],
                     "ratio": sc["ratio"], "rel": sc["max_rel_err"], **r})
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
    out, rows = convert(json.load(open(a.coeffs)), a.tol, a.zero)
    json.dump(out, open(a.out, "w"), indent=1)
    for r in rows:
        print(f"{r['name']:28s} const {r['const_steps']:2d} st d{r['const_depth']:2d}  ->  scaled {r['steps']:2d} st "
              f"d{r['depth']:2d}  ratio {r['ratio']:.3g}  rel {r['rel']:.2e}")
    s = summary(rows)
    print(json.dumps(s, indent=1))
    tot_c = sum(v["const_depth"][2] for v in s.values())
    tot_s = sum(v["depth"][2] for v in s.values())
    print(f"mean depth of the four chains per block: {tot_c:.1f} -> {tot_s:.1f} levels ({tot_s - tot_c:+.1f})")
    print(f"wrote {a.out}: {len(rows)} sites converted")
    return 0


if __name__ == "__main__":
    sys.exit(main())
