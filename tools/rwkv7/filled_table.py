#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""FILLED_DESIGN.md §1's cost table: per plan, the compile check's counts per block (``compile_check.json``:
``Program.compile``'s own refresh sites, pairs, and op census) and P4's price (BATCHED_DESIGN.md §4), extrapolated to
12 blocks as block 0 + 11 x block 1 + the tail.

    python tools/rwkv7/filled_table.py LABEL=PLAN_DIR [LABEL=PLAN_DIR ...] [--json]

Price (an ESTIMATE, R6: Perseus's published per-op times used as ratios, another card and build): refresh 28.77 ms,
rotation or conjugation (a key switch) 0.214 ms, ct x ct 0.203 ms, ct x pt 0.072 ms, add 0.033 ms; a refresh pair
(PAIR=1) is priced as one refresh. PCMM's GEMM is not in the graph (the same in both layouts) and mult_const,
mult_i and rescale are not priced. Scaled to this port's measured decode (9.54 s per token, 600 W box) by 0.574-0.621
seconds per published second. Reads JSON only: light.
"""
import json
import os
import sys

PRICE = {"refresh": 28.77, "ks": 0.214, "ctct": 0.203, "ctpt": 0.072, "add": 0.033}
SCALE = (0.574, 0.621)


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
    if "--json" in sys.argv:
        print(json.dumps(rows, indent=1))
        sys.exit(0)
    print("| plan | sites b0 / b1 | pairs b0 / b1 | rotations + conj. b0 / b1 | ct x ct b0 / b1 | ct x pt b0 / b1 | "
          "12-block refreshes (sites - pairs) | 12-block rotations + conj. | est. s per pass |")
    print("|---|---|---|---|---|---|---|---|---|")
    for label, r in rows.items():
        b0, b1, f = r["b0"], r["b1"], r["pass12"]
        print(f"| {label} | {b0['sites']} / {b1['sites']} | {b0['pairs']} / {b1['pairs']} | "
              f"{b0['rot'] + b0['conj']:,} / {b1['rot'] + b1['conj']:,} | {b0['ctct']:,} / {b1['ctct']:,} | "
              f"{b0['ctpt']:,} / {b1['ctpt']:,} | {f['refresh_cost']:,} | {f['rot'] + f['conj']:,} | "
              f"{f['s_est'][0]}-{f['s_est'][1]} |")
