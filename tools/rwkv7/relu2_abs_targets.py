#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""ReLU^2 targets in absolute terms (REVIEW_20261105.md §5.A1): per block, E = REL x max|relu^2| over the site's
margin-M0 interval [m - M0 w, M + M0 w] (w = M - m, from the calibration ranges), printed as fit.py flags
``--site-target 'blocks\\.B\\.ffn\\.relu2$=E:abs'``, one per line. Fitting the wide interval (margin 0.5) with these
targets keeps ReLU^2's absolute error where gate 3's margin-0.1 fit had it, whatever the margin.

Why: fit.py's ``rel`` metric divides by max|f| over the fitted interval, and for relu^2 that maximum grows with the
square of the interval's top, so widening the interval from margin 0.1 to 0.5 loosened the absolute error ~2.8x (top-1
93.7 %, gate 2). An absolute target decouples accuracy from robustness; the price is degree (best degree-n error of
x|x|-type functions ~ (half-width)^2 / n^2, so about 2x the degree for 2x the width: one more level per site).

    python tools/rwkv7/relu2_abs_targets.py RANGES.json [--rel 1e-4] [--margin0 0.1]
"""
import argparse
import json
import re


def targets(ranges: dict, rel: float = 1e-4, margin0: float = 0.1) -> dict:
    out = {}
    for name, r in ranges["sites"].items():
        if not name.endswith("ffn.relu2"):
            continue
        m, M = float(r["min"]), float(r["max"])
        w = max(M - m, 1e-6)
        hi = M + margin0 * w
        out[name] = rel * max(hi, 0.0) ** 2
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ranges")
    ap.add_argument("--rel", type=float, default=1e-4)
    ap.add_argument("--margin0", type=float, default=0.1)
    ap.add_argument("--json", action="store_true", help="print {site: E} instead of the flags")
    a = ap.parse_args(argv)
    t = targets(json.load(open(a.ranges)), a.rel, a.margin0)
    if a.json:
        print(json.dumps(t, indent=1))
        return 0
    for name, e in sorted(t.items(), key=lambda kv: int(kv[0].split(".")[1])):
        print(f"--site-target={re.escape(name)}$={e:.6g}:abs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
