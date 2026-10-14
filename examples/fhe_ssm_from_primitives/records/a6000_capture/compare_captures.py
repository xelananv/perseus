#!/usr/bin/env python3
"""Compare two graph captures of the same run node by node, and the plans made from them.

    compare_captures.py GRAPHS_A GRAPHS_B [PLANS_A PLANS_B] [--out summary.json]

GRAPHS_*: directories holding block_<b>/graph.json (gpu_run --mode capture). The node lists must align
(same op stream): every node's op_type and output name are checked first. For each of the five captured
magnitudes the relative difference per node is collected (the mean and the deviation from it over
max|z|, the AC coefficient maximum over the coefficient maximum: remainders are measured against the
value they are a remainder of), and the summary gives median, 99.9th percentile, max and the counts
above 1e-6 / 1e-3 / 1e-1 per field,
plus nodes carrying a field on one side only. Two captures are two runs with their own keys, so their
decrypted magnitudes differ by the FHE error of the two runs (1e-4 right after a refresh, up to ~1e-2
at a block's end); the magnitude distributions are therefore reported, not gated, except that any
relative difference above 0.1 is counted as suspicious. PLANS_*: bootstrap_placements directories; each
block's placement JSON is compared for deep equality (reported), for its decisions key by key
(placements, correction factors, sparse routes, hint decisions, rescale lists, rules, and the
summary's integer fields; reported), and for its cost: bootstraps per block and per tick, exit levels,
and the planner's own warnings. Exit 0 iff the node lists align, no field is one-sided, no magnitude
differs by more than 0.1, and (when given) the plan sets have the same files, per-tick bootstrap
totals within 0.5 %, and are both free of planner warnings or both not.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

FIELDS = (
    "output_max_abs",
    "output_mean",
    "output_max_dev",
    "output_max_coeff",
    "output_max_coeff_ac",
)


def rel(a: float, b: float, scale: float | None = None) -> float:
    if scale is not None:
        return abs(a - b) / scale if scale > 0 else 0.0
    s = max(abs(a), abs(b))
    return abs(a - b) / s if s > 0 else 0.0


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    k = min(len(ys) - 1, round(p * (len(ys) - 1)))
    return ys[k]


def compare_graphs(da: Path, db: Path) -> dict:
    blocks = sorted(
        int(p.name.split("_")[1]) for p in da.glob("block_*") if (p / "graph.json").exists()
    )
    blocks_b = {
        int(p.name.split("_")[1]) for p in db.glob("block_*") if (p / "graph.json").exists()
    }
    common = [b for b in blocks if b in blocks_b]
    out: dict = {
        "blocks_a": blocks,
        "blocks_b": sorted(blocks_b),
        "blocks_compared": common,
        "misaligned_blocks": [],
        "fields": {},
        "one_sided": dict.fromkeys(FIELDS, 0),
        "nodes": 0,
    }
    diffs: dict[str, list[float]] = {f: [] for f in FIELDS}
    worst_nodes: dict[str, tuple] = {}
    for b in common:
        na = json.loads((da / f"block_{b}" / "graph.json").read_text())["nodes"]
        nb = json.loads((db / f"block_{b}" / "graph.json").read_text())["nodes"]
        if len(na) != len(nb) or any(
            x.get("op_type") != y.get("op_type") or x.get("output") != y.get("output")
            for x, y in zip(na, nb)
        ):
            out["misaligned_blocks"].append({"block": b, "len_a": len(na), "len_b": len(nb)})
            continue
        for i, (x, y) in enumerate(zip(na, nb)):
            out["nodes"] += 1
            # the mean and the deviation from it are compared relative to the value's magnitude, the
            # AC coefficient maximum relative to the full coefficient maximum: those three are
            # remainders, and a remainder of a nearly constant value is itself noise-sized
            scale_abs = scale_coeff = None
            if x.get("output_max_abs") is not None and y.get("output_max_abs") is not None:
                scale_abs = max(float(x["output_max_abs"]), float(y["output_max_abs"]))
            if x.get("output_max_coeff") is not None and y.get("output_max_coeff") is not None:
                scale_coeff = max(float(x["output_max_coeff"]), float(y["output_max_coeff"]))
            scale_of = {
                "output_mean": scale_abs,
                "output_max_dev": scale_abs,
                "output_max_coeff_ac": scale_coeff,
            }
            for f in FIELDS:
                xa, xb = x.get(f), y.get(f)
                if (xa is None) != (xb is None):
                    out["one_sided"][f] += 1
                    continue
                if xa is None:
                    continue
                d = rel(float(xa), float(xb), scale_of.get(f))
                diffs[f].append(d)
                if f not in worst_nodes or d > worst_nodes[f][0]:
                    worst_nodes[f] = (d, b, i, x.get("op_type"), x.get("step"), xa, xb)
    for f in FIELDS:
        xs = diffs[f]
        out["fields"][f] = {
            "n": len(xs),
            "median": percentile(xs, 0.5),
            "p999": percentile(xs, 0.999),
            "max": max(xs) if xs else 0.0,
            "above_1e-6": sum(1 for d in xs if d > 1e-6),
            "above_1e-3": sum(1 for d in xs if d > 1e-3),
            "above_1e-1": sum(1 for d in xs if d > 1e-1),
            "worst": worst_nodes.get(f),
        }
    return out


#: the plan's decisions (what the runtime executes); everything else in the file is derived floats
DECISION_KEYS = (
    "placements",
    "correction_factor",
    "sparse_slots",
    "hint_fire",
    "hint_decisions_bound",
    "rescale_after",
    "rules",
    "weight_levels",
    "mask_levels",
)
SUMMARY_INTS = (
    "num_placements",
    "total_bootstraps",
    "exit_level",
    "exit_deg",
    "num_hint_bootstraps",
)


def _block_index(name: str) -> int:
    return int(name.split("_")[1])


def _warnings(plan: dict) -> int:
    bq = plan.get("summary", {}).get("bts_quality", {})
    ec = bq.get("env_cap", {})
    return (
        len(ec.get("reactive_past_cap", []))
        + len(ec.get("placed_past_cap", []))
        + len(ec.get("hints_predicted_past_cap", []))
        + (1 if ec.get("relaxed") else 0)
    )


def compare_plans(pa: Path, pb: Path) -> dict:
    """Deep equality is reported but not gated: a plan file embeds floats computed from the captured
    magnitudes (predicted errors, levels), which differ between two runs. The decisions are compared
    key by key, and the cost (bootstraps per block and per tick) and the planner's own warnings are
    what the gate reads: the min-cut picks among near-equal-cost sites, so two captures that differ by
    FHE noise can place differently at the same cost."""
    files_a = sorted((p.name for p in pa.glob("block_*_placement.json")), key=_block_index)
    files_b = sorted((p.name for p in pb.glob("block_*_placement.json")), key=_block_index)
    res: dict = {
        "files_a": len(files_a),
        "files_b": len(files_b),
        "identical": [],
        "different": [],
        "missing": [],
        "decision_identical": [],
        "decision_different": {},
        "per_block_total_a": [],
        "per_block_total_b": [],
        "total_bootstraps_a": 0,
        "total_bootstraps_b": 0,
        "exit_levels_a": [],
        "exit_levels_b": [],
        "warnings_a": 0,
        "warnings_b": 0,
    }
    for name in files_a:
        if name not in files_b:
            res["missing"].append(name)
            continue
        ja = json.loads((pa / name).read_text())
        jb = json.loads((pb / name).read_text())
        (res["identical"] if ja == jb else res["different"]).append(name)
        diff = [k for k in DECISION_KEYS if ja.get(k) != jb.get(k)]
        sa, sb = ja.get("summary", {}), jb.get("summary", {})
        diff += [f"summary.{k}" for k in SUMMARY_INTS if sa.get(k) != sb.get(k)]
        if diff:
            res["decision_different"][name] = diff
        else:
            res["decision_identical"].append(name)
        ta, tb = int(sa.get("total_bootstraps", 0)), int(sb.get("total_bootstraps", 0))
        res["per_block_total_a"].append(ta)
        res["per_block_total_b"].append(tb)
        res["total_bootstraps_a"] += ta
        res["total_bootstraps_b"] += tb
        res["exit_levels_a"].append(sa.get("exit_level"))
        res["exit_levels_b"].append(sb.get("exit_level"))
        res["warnings_a"] += _warnings(ja)
        res["warnings_b"] += _warnings(jb)
    ta, tb = res["total_bootstraps_a"], res["total_bootstraps_b"]
    res["total_rel_diff"] = abs(ta - tb) / max(ta, tb) if max(ta, tb) else 0.0
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("graphs_a")
    ap.add_argument("graphs_b")
    ap.add_argument("plans_a", nargs="?")
    ap.add_argument("plans_b", nargs="?")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    summary = {"graphs": compare_graphs(Path(a.graphs_a), Path(a.graphs_b))}
    ok = not summary["graphs"]["misaligned_blocks"] and summary["graphs"]["nodes"] > 0
    ok = ok and all(v == 0 for v in summary["graphs"]["one_sided"].values())
    ok = ok and all(f["above_1e-1"] == 0 for f in summary["graphs"]["fields"].values())
    if a.plans_a and a.plans_b:
        p = summary["plans"] = compare_plans(Path(a.plans_a), Path(a.plans_b))
        ok = (
            ok
            and not p["missing"]
            and p["files_a"] > 0
            and p["files_a"] == p["files_b"]
            and p["total_rel_diff"] <= 0.005
            and (p["warnings_a"] == 0) == (p["warnings_b"] == 0)
        )
    summary["pass"] = ok
    text = json.dumps(summary, indent=1, default=str)
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
