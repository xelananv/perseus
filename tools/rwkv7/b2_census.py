#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""PLAN_B2.md's census: a batched plan's refreshes per block, split into the per-conversation scopes and the rest.

    python tools/rwkv7/b2_census.py CAPTURE_DIR PLAN_DIR [--json]

Per block: placements (planned refreshes) by step scope, deliberate refreshes (the capture's own: the state and lane
carries) by step scope, and the total the planner's summary reports (placements + deliberate + fired hints). The
per-conversation scopes are ``tm.state.bcast`` (lane c brought to lane 0 and broadcast), ``tm.state.update`` (decode's
``state_step`` on conversation c's own S) and ``tm.state.readout``; ``tm.state.src`` builds the three shared sources
once per pass. Reads JSON only (no planner, no capture): light enough to run beside nothing else.
"""
import collections
import glob
import json
import os
import re
import sys

PER_CONV = ("tm.state.bcast", "tm.state.update", "tm.state.readout")


def scope(step: str) -> str:
    s = re.sub(r"^blk\d+\.", "", str(step))
    return "tm.state.bcast" if s.startswith("tm.state.bcast") else s


def _load(path: str):
    with open(path) as f:
        return json.load(f)


def block_census(cap: str, plan: str, b: int) -> dict:
    g = _load(f"{cap}/block_{b}/graph.json")
    pl = _load(f"{plan}/block_{b}_placement.json")
    step = {n.get("output"): n.get("step") for n in g["nodes"]}
    placed = collections.Counter(scope(step.get(p.get("target_var"), "?")) for p in pl["placements"])
    delib = collections.Counter(scope(n.get("step")) for n in g["nodes"] if n["op_type"] == "deliberate_bootstrap")
    s = pl["summary"]
    return {"total": s.get("total_bootstraps"), "placements": len(pl["placements"]),
            "deliberate": sum(delib.values()), "hints_fired": s.get("num_hint_bootstraps"),
            "placed_per_conv_scopes": sum(v for k, v in placed.items() if k in PER_CONV),
            "placed_by_scope": dict(placed.most_common()), "deliberate_by_scope": dict(delib.most_common()),
            "exit_level": s.get("exit_level")}


def census(cap: str, plan: str) -> dict:
    n = len(glob.glob(os.path.join(plan, "block_*_placement.json")))
    cfg = _load(os.path.join(cap, "rwkv7_capture.json"))
    return {"K": cfg.get("batch"), "blocks": {b: block_census(cap, plan, b) for b in range(n)
                                              if os.path.exists(f"{cap}/block_{b}/graph.json")}}


if __name__ == "__main__":
    c = census(sys.argv[1], sys.argv[2])
    if "--json" in sys.argv:
        print(json.dumps(c, indent=1))
        sys.exit(0)
    print(f"K = {c['K']}  {sys.argv[2]}")
    for b, r in c["blocks"].items():
        print(f"  block {b}: total {r['total']} = placed {r['placements']} (per-conv scopes {r['placed_per_conv_scopes']})"
              f" + deliberate {r['deliberate']} + hints {r['hints_fired']}; exit {r['exit_level']}")
        print(f"    placed: {r['placed_by_scope']}")
        print(f"    deliberate: {r['deliberate_by_scope']}")
