#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""FILLED_DESIGN.md §1's cost table: per plan, the compile check's counts per block (``compile_check.json``:
``Program.compile``'s own refresh sites, pairs, and op census) and P4's price (BATCHED_DESIGN.md §4), extrapolated to
12 blocks as block 0 + 11 x block 1 + the tail.

    python tools/rwkv7/filled_table.py LABEL=PLAN_DIR [LABEL=PLAN_DIR ...] [--json]

Price (an ESTIMATE, R6: Perseus's published per-op times used as ratios, another card and build): refresh 28.77 ms,
rotation or conjugation (a key switch) 0.214 ms, ct x ct 0.203 ms, ct x pt 0.072 ms, add 0.033 ms; a refresh pair
(PAIR=1) is priced as one refresh. PCMM's GEMM is not in the graph (the same in both layouts) and mult_const,
mult_i and rescale are not priced. The "est. s per pass" column scales by 0.574-0.621 seconds per published second
(the old decode anchor); it is kept for reference only.

The "anchored s per pass" column (review lens 3, finding 8) scales each mode by its own measured run on pod 3's final
coefficients with LEVERS=best, so the published-unit model is only used as a ratio inside one mode:
  decode   the measured 9.703-9.7405 s per token (r01b_final/run_best_P1-P3.jsonl) over the "decode, current +best"
           row's model (scale ~0.896-0.899);
  prefill  the measured 24.12-24.53 s per T = 32 pass (r01b_final/run_pf32_best_L*.jsonl, "prefill" lines) over the
           "prefill T = 32, current +best" row's model;
  batched  the measured 29.8721 s per K = 32 pass (r01b_final/batch32_best.jsonl) over 50.40 published s, the per-op
           model of that run's own device block mix (its compiled sites, base + best; review lens 3's b3_model.py,
           transcribed): scale 0.5927 with no remainder; the alternative that keeps a fixed remainder fitted to the
           same anchor (3.3 s + 0.527 x the model) gives the other end of the range.
The batched pass needs no decode remainder: the old remainder (3.0-3.5 s) was the decode scale's own unit error (it
was calibrated on decode's planner count and applied to compiled sites). Reads JSON only: light.
"""
import json
import os
import sys

PRICE = {"refresh": 28.77, "ks": 0.214, "ctct": 0.203, "ctpt": 0.072, "add": 0.033}
SCALE = (0.574, 0.621)
DEC_MEAS = (9.703, 9.7405)          # r01b_final/run_best_P1-P3.jsonl seconds_per_token_median (min, max)
PF_MEAS = (24.12, 24.53)            # r01b_final/run_pf32_best_L*.jsonl "prefill" pass seconds (min, max)
B_MEAS, B_MODEL_PUB_S = 29.8721, 50.40   # batch32_best.jsonl seconds_per_pass_median; its per-op model (published s)
B_REMAINDER = (3.3, 0.527)          # fixed seconds + scale, fitted to the same anchor


def block_row(b: dict) -> dict:
    lim = b["limbs"]

    def n(k):
        return int(lim.get(k, {}).get("ops", 0))
    r = {"sites": int(b["sites"]), "pairs": int(b.get("pairs", 0)), "launches": int(b["launches"]),
         "rot": n("rotate"), "conj": n("conjugate"), "ctct": n("mult") + n("square"), "ctpt": n("mult_pt"),
         "add": n("add") + n("sub") + n("add_pt") + n("add_const")}
    r["refresh_cost"] = r["sites"] - r["pairs"]
    r["ms"] = (PRICE["refresh"] * r["refresh_cost"] + PRICE["ks"] * (r["rot"] + r["conj"]) +
               PRICE["ctct"] * r["ctct"] + PRICE["ctpt"] * r["ctpt"] + PRICE["add"] * r["add"])
    return r


def plan_row(plan: str, n_layers: int = 12) -> dict:
    d = json.load(open(os.path.join(plan, "compile_check.json")))
    blocks = {b["block"]: block_row(b) for b in d["blocks"]}
    b0, b1 = blocks["block_0"], blocks["block_1"]
    tail = blocks.get("block_2", {"sites": 1, "pairs": 0, "ms": PRICE["refresh"], "refresh_cost": 1})
    keys = ("sites", "pairs", "launches", "rot", "conj", "ctct", "ctpt", "add", "refresh_cost", "ms")
    full = {k: b0.get(k, 0) + (n_layers - 1) * b1.get(k, 0) + tail.get(k, 0) for k in keys}
    full["s_est"] = [round(full["ms"] / 1000 * s, 1) for s in SCALE]
    return {"b0": b0, "b1": b1, "pass12": full}


if __name__ == "__main__":
    rows = {}
    for arg in sys.argv[1:]:
        if arg.startswith("--"):
            continue
        label, plan = arg.rsplit("=", 1)
        rows[label] = plan_row(plan)
    def ref(mode):
        return next((r["pass12"]["ms"] / 1000 for lab, r in rows.items()
                     if lab.startswith(mode) and lab.rstrip().endswith("current +best")), None)
    for label, r in rows.items():
        pub = r["pass12"]["ms"] / 1000
        if label.startswith("decode") and ref("decode"):
            lo, hi = (pub * m / ref("decode") for m in DEC_MEAS)
        elif label.startswith("prefill") and ref("prefill"):
            lo, hi = (pub * m / ref("prefill") for m in PF_MEAS)
        elif label.startswith("batched"):
            a_, b_ = pub * B_MEAS / B_MODEL_PUB_S, B_REMAINDER[0] + B_REMAINDER[1] * pub
            lo, hi = min(a_, b_), max(a_, b_)
        else:
            lo = hi = None
        r["pass12"]["s_anchored"] = None if lo is None else [round(lo, 2), round(hi, 2)]
    if "--json" in sys.argv:
        print(json.dumps(rows, indent=1))
        sys.exit(0)
    print("| plan | sites b0 / b1 | pairs b0 / b1 | rotations + conj. b0 / b1 | ct x ct b0 / b1 | ct x pt b0 / b1 | "
          "12-block refreshes (sites - pairs) | 12-block rotations + conj. | model, published s | old est. s per pass | "
          "anchored s per pass |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for label, r in rows.items():
        b0, b1, f = r["b0"], r["b1"], r["pass12"]
        sa = f["s_anchored"]
        anch = "-" if sa is None else f"{sa[0]}-{sa[1]}"
        print(f"| {label} | {b0['sites']} / {b1['sites']} | {b0['pairs']} / {b1['pairs']} | "
              f"{b0['rot'] + b0['conj']:,} / {b1['rot'] + b1['conj']:,} | {b0['ctct']:,} / {b1['ctct']:,} | "
              f"{b0['ctpt']:,} / {b1['ctpt']:,} | {f['refresh_cost']:,} | {f['rot'] + f['conj']:,} | "
              f"{f['ms'] / 1000:.2f} | {f['s_est'][0]}-{f['s_est'][1]} | {anch} |")
