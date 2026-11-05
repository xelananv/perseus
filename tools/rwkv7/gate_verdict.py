#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""The quality gate of a coefficients file (results/rwkv7-fhe-20261104/new_pod/FINAL_PLAN.md, REVIEW_20261105.md
§5.A7): from the harness evals (exact, poly clamped, poly UNCLAMPED) and the unclamped range scans, PASS iff
  * the unclamped eval equals the clamped one (wt ppl within 1e-4 relative, LAMBADA within 0.1 point): what the
    ciphertext computes (no clamp) is what was evaluated;
  * every number is finite and every scan has zero non-finite site inputs and zero inputs past a newton_const site's
    convergence bound;
  * the coordinator's bars (FINAL_PLAN.md; the owner decides whether they are "reasonable quality"): ppl within 0.5 %
    of exact, LAMBADA within 0.5 points, top-1 vs exact >= 98 %.
Every excursion is listed per site kind (tokens out of the accuracy range, worst excursion as a fraction of the
interval width, the site). Exit 0 on PASS, 3 on FAIL.

    python tools/rwkv7/gate_verdict.py --eval EVAL.jsonl --exact-tag exact --clamp-tag C --unclamped-tag U \\
        --scan scan_wiki.json --scan scan_lambada.json --scan scan_gen.json [--out verdict.json]
"""
import argparse
import json
import math
import sys


def last(path, tag):
    hit = None
    for line in open(path):
        r = json.loads(line)
        if r.get("tag") == tag:
            hit = r
    if hit is None:
        raise SystemExit(f"{path}: no line tagged {tag!r}")
    return hit


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--exact-tag", default="exact")
    ap.add_argument("--exact-eval", default=None, help="the exact line's file, if not --eval")
    ap.add_argument("--clamp-tag", required=True)
    ap.add_argument("--unclamped-tag", required=True)
    ap.add_argument("--scan", action="append", default=[])
    ap.add_argument("--ppl-rel", type=float, default=0.005)
    ap.add_argument("--lambada-pts", type=float, default=0.5)
    ap.add_argument("--top1", type=float, default=0.98)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    ex = last(a.exact_eval or a.eval, a.exact_tag)
    cl, un = last(a.eval, a.clamp_tag), last(a.eval, a.unclamped_tag)
    checks = {}

    def fin(*xs):
        return all(isinstance(x, (int, float)) and math.isfinite(x) for x in xs)

    checks["finite"] = fin(cl["wt_ppl"], un["wt_ppl"], cl["lambada_acc"], un["lambada_acc"], un.get("top1_agree", 0.0))
    checks["unclamped_equals_clamped"] = checks["finite"] and abs(un["wt_ppl"] - cl["wt_ppl"]) <= 1e-4 * cl["wt_ppl"] \
        and abs(un["lambada_acc"] - cl["lambada_acc"]) <= 0.001
    checks["ppl_within"] = checks["finite"] and un["wt_ppl"] <= ex["wt_ppl"] * (1 + a.ppl_rel)
    checks["lambada_within"] = checks["finite"] and un["lambada_acc"] >= ex["lambada_acc"] - a.lambada_pts / 100
    checks["top1"] = checks["finite"] and un.get("top1_agree", 0.0) >= a.top1
    scans = {}
    for p in a.scan:
        d = json.load(open(p))
        kinds = d.get("kinds", {})
        nonfin = sum(int(k.get("nonfinite", 0)) for k in kinds.values())
        conv = sum(int(k.get("tokens_conv_out", 0)) for k in kinds.values())
        exc = {n: {"tokens_out": k.get("tokens_out"), "worst_excursion": k.get("worst_excursion"),
                   "worst_site": k.get("worst_site")} for n, k in kinds.items() if k.get("tokens_out")}
        scans[p] = {"tokens": d.get("tokens"), "nonfinite": nonfin, "tokens_conv_out": conv, "excursions": exc}
    checks["scans_clean"] = all(s["nonfinite"] == 0 and s["tokens_conv_out"] == 0 for s in scans.values())
    verdict = "PASS" if all(checks.values()) else "FAIL"
    rep = {"verdict": verdict, "checks": checks,
           "exact": {k: ex.get(k) for k in ("wt_ppl", "lambada_acc")},
           "clamped": {k: cl.get(k) for k in ("wt_ppl", "lambada_acc", "top1_agree", "kl_mean")},
           "unclamped": {k: un.get(k) for k in ("wt_ppl", "lambada_acc", "top1_agree", "kl_mean", "outside_frac_max")},
           "ppl_rel_vs_exact": (un["wt_ppl"] / ex["wt_ppl"] - 1) if checks["finite"] else None,
           "lambada_pts_vs_exact": 100 * (un["lambada_acc"] - ex["lambada_acc"]) if checks["finite"] else None,
           "bars": {"ppl_rel": a.ppl_rel, "lambada_pts": a.lambada_pts, "top1": a.top1}, "scans": scans}
    text = json.dumps(rep, indent=1)
    print(text)
    if a.out:
        open(a.out, "w").write(text)
    print(f"GATE {verdict}: " + ", ".join(f"{k}={'ok' if v else 'FAIL'}" for k, v in checks.items()), file=sys.stderr)
    return 0 if verdict == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
