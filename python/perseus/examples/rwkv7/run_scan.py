# SPDX-License-Identifier: MIT
"""Plaintext pre-flight: the poly model (fit.py's coefficients, NO clamp, as a ciphertext evaluates them) over a
text's tokens after the public prefix; prints every site that leaves its calibrated interval (or goes non-finite),
with the first position, the count and the range. Exit 1 if any site leaves.

    python -m perseus.examples.rwkv7.run_scan --pth P --coeffs C --harness H [--text T | --long] [--n 64]
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from . import common, ref


def scan(P, coeffs, ids, n_prefix=1, n=64):
    """``{site: [first position, out count, min, max, lo, hi]}`` over ``ids[n_prefix:n_prefix + n]`` (positions
    counted after the prefix, which runs clamped, as ``common.prefix_state`` does)."""
    s = ref.Sites("poly", coeffs, clamp=True)
    m = ref.RefModel(P, s)
    for t in ids[:n_prefix]:
        m.step(t)
    s.clamp = False
    first, pos, cfg = {}, [0], s.cfg

    class Rec(dict):
        def __setitem__(self, k, v):
            if k in cfg:
                lo, hi = float(cfg[k]["lo"]), float(cfg[k]["hi"])
                bad = (v < lo) | (v > hi) | ~np.isfinite(v)
                if bad.any():
                    r = first.setdefault(k, [pos[0], 0, np.inf, -np.inf, lo, hi])
                    r[1] += int(bad.sum())
                    with np.errstate(all="ignore"):
                        r[2], r[3] = min(r[2], float(np.nanmin(v))), max(r[3], float(np.nanmax(v)))

    s.record = Rec()
    with np.errstate(all="ignore"):
        for i, t in enumerate(ids[n_prefix:n_prefix + n]):
            pos[0] = i
            _lg, h = m.step(t)
            if not np.all(np.isfinite(h)):
                first.setdefault("hidden(non-finite)", [i, 1, np.nan, np.nan, np.nan, np.nan])
                break
    return first


def report(first) -> list[str]:
    return [f"OUT {k}: first position {p}, {c} values, range [{a:.4g}, {b:.4g}] vs interval [{lo:.4g}, {hi:.4g}]"
            for k, (p, c, a, b, lo, hi) in sorted(first.items(), key=lambda kv: kv[1][0])]


def main(argv=None):
    ap = argparse.ArgumentParser()
    common.add_args(ap)
    ap.add_argument("--long", action="store_true", help="the prefill capture's text (common.CALIB_TEXT_LONG)")
    ap.add_argument("--n", type=int, default=64)
    a = ap.parse_args(argv)
    P = ref.load_params(a.pth, a.blocks)
    ids = common.token_ids(a, common.CALIB_TEXT_LONG if a.long else a.text)
    first = scan(P, a.coeffs, ids, a.prefix, a.n)
    print(f"{len(ids)} ids, scanned {min(a.n, len(ids) - a.prefix)} after a prefix of {a.prefix}")
    print("\n".join(report(first)))
    print("all sites inside their intervals" if not first else f"{len(first)} site(s) out of interval")
    return 1 if first else 0


if __name__ == "__main__":
    sys.exit(main())
