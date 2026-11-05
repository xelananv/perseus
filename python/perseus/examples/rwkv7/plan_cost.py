# SPDX-License-Identifier: MIT
"""A plan's refreshes per stage (``summary.total_bootstraps`` of each ``block_<b>_placement.json``, what
``make_plan.sh`` prints) with the split into placed / deliberate, the capture graph's op counts, and the census of
where block b's placed refreshes sit by step scope. The per-size table of ANYSIZE_DESIGN.md is built from these.

    PYTHONPATH=python python -m perseus.examples.rwkv7.plan_cost CAPTURE_DIR PLAN_DIR [--census B]
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re


def plan_refreshes(cap: str, plan: str) -> dict:
    """{stage: {refreshes, placed, deliberate, nodes, rotate, conjugate, mult}} plus the total."""
    out, tot = {}, 0
    for f in sorted(glob.glob(os.path.join(plan, "block_*_placement.json")),
                    key=lambda p: int(re.search(r"block_(\d+)_placement", p).group(1))):
        b = int(re.search(r"block_(\d+)_placement", f).group(1))
        s = json.load(open(f))["summary"]
        row = {"refreshes": int(s.get("total_bootstraps") or 0), "placed": int(s.get("num_placements") or 0),
               "deliberate": int(s.get("num_deliberate_bootstrap_nodes") or 0), "exit_level": s.get("exit_level")}
        g = os.path.join(cap, f"block_{b}", "graph.json")
        if os.path.exists(g):
            ops = collections.Counter(n["op_type"] for n in json.load(open(g))["nodes"])
            row.update(nodes=sum(ops.values()), rotate=ops.get("rotate", 0), conjugate=ops.get("conjugate", 0),
                       mult=sum(v for k, v in ops.items() if k.startswith("mult") or k == "square"))
        out[f"block_{b}"] = row
        tot += row["refreshes"]
    return {"stages": out, "total": tot}


def census(cap: str, plan: str, b: int = 0) -> dict:
    """Block b's placed refreshes by the capture's step scope (``tm.kk``, ``cm.ffn``, ...)."""
    g = json.load(open(os.path.join(cap, f"block_{b}", "graph.json")))
    pl = json.load(open(os.path.join(plan, f"block_{b}_placement.json")))
    step = {n.get("output"): n.get("step") for n in g["nodes"]}
    by = collections.Counter(str(step.get(p.get("target_var"))) for p in pl["placements"])
    return dict(by.most_common())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("plan")
    ap.add_argument("--census", type=int, default=None)
    a = ap.parse_args(argv)
    rep = plan_refreshes(a.capture, a.plan)
    if a.census is not None:
        rep["census"] = census(a.capture, a.plan, a.census)
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
