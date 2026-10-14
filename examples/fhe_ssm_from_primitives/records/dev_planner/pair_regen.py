#!/usr/bin/env python3
"""Regenerate every regenerable shipped plan in a worktree with the recipe in its PLAN_CMD.txt, exactly as
tests/test_paper_plans.py does, but keep the outputs (bootstrap_placements/_pair_<tag>_<name>).

  pair_regen.py run <worktree> <tag>                 # one side
  pair_regen.py compare <wtA> <tagA> <wtB> <tagB>    # A's regen vs B's regen (plan_equiv), and each vs the shipped plan
"""

import importlib.util
import os
import pathlib
import shlex
import subprocess
import sys

UPSTREAM_TOOLS = {"scripts/utils/orion_upstream/plan.sh", "scripts/utils/dacapo_upstream/plan.sh"}
MAIN = "gpt2_decode_python_n32"


def plan_dirs(wt):
    bp = wt / "bootstrap_placements"
    names = [MAIN]
    for p in sorted(bp.rglob("PLAN_CMD.txt")):
        rel = str(p.parent.relative_to(bp))
        if rel != MAIN and not rel.startswith("_"):
            names.append(rel)
    return names


def recipe(plan_dir):
    graph = rec = tool = None
    route = "sparse"
    for line in (plan_dir / "PLAN_CMD.txt").read_text().splitlines():
        if line.startswith("graph="):
            graph = line[len("graph=") :].split(" ")[0]
        elif line.startswith("recipe="):
            rec = shlex.split(line[len("recipe=") :])
        elif line.startswith("tool="):
            tool = line[len("tool=") :].split(" ")[0]
        elif line.startswith("route="):
            route = line[len("route=") :].strip()
    if rec is None:
        return None
    return graph, rec, tool, route


def out_name(tag, name):
    return f"_pair_{tag}_{name.replace('/', '_')}"


def run_side(wt, tag, logdir):
    logdir.mkdir(parents=True, exist_ok=True)
    for name in plan_dirs(wt):
        plan_dir = wt / "bootstrap_placements" / name
        parsed = recipe(plan_dir)
        if parsed is None:
            print(f"{name}: measured artifact, skipped", flush=True)
            continue
        graph, rec, tool, _route = parsed
        if not (wt / graph).is_dir():
            print(f"{name}: graph {graph} not present, skipped", flush=True)
            continue
        if tool in UPSTREAM_TOOLS:
            print(f"{name}: upstream tool {tool}, skipped", flush=True)
            continue
        out = out_name(tag, name)
        env = dict(
            os.environ,
            GRAPH_DIR=graph,
            OUT_NAME=out,
            PYTHON=sys.executable,
            CUDA_VISIBLE_DEVICES="",
            OMP_NUM_THREADS="4",
        )
        env.update(kv.split("=", 1) for kv in rec)
        if tool is None:
            cmds = [["bash", "scripts/utils/run_bootstrap_all_blocks.sh"]]
        else:
            cmds = [["bash", tool, graph, out], ["bash", tool, graph, out, "argmax"]]
            if (plan_dir / "block_0_feedback_placement.json").exists():
                cmds.append(["bash", tool, graph, out, "feedback"])
        rc = 0
        with (logdir / (name.replace("/", "_") + ".log")).open("w") as f:
            for cmd in cmds:
                rc = (
                    rc
                    or subprocess.run(
                        cmd, check=False, cwd=wt, env=env, stdout=f, stderr=subprocess.STDOUT
                    ).returncode
                )
        n = len(list((wt / "bootstrap_placements" / out).glob("block_*_placement.json")))
        print(f"{name}: rc={rc} files={n}", flush=True)


def compare(wta, taga, wtb, tagb):
    spec = importlib.util.spec_from_file_location(
        "plan_equiv", wta / "scripts" / "utils" / "plan_equiv.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    eq = mod.equivalent
    print(f"{'plan':48s} {'A=B':>5s} {'A=ship':>7s} {'B=ship':>7s}  first difference A vs B")
    for name in plan_dirs(wta):
        ship = wta / "bootstrap_placements" / name
        ra = wta / "bootstrap_placements" / out_name(taga, name)
        rb = wtb / "bootstrap_placements" / out_name(tagb, name)
        if not ra.is_dir() or not rb.is_dir():
            continue
        blocks = sorted(ship.glob("block_*_placement.json"))
        ab = asp = bsp = True
        why_ab = ""
        for ref in blocks:
            fa, fb = ra / ref.name, rb / ref.name
            if not fa.exists() or not fb.exists():
                ab = False
                why_ab = why_ab or f"{ref.name}: missing on one side"
                continue
            ok, why = eq(str(fa), str(fb))
            if not ok and ab:
                ab = False
                why_ab = f"{ref.name}: {why}"
            if not eq(str(fa), str(ref))[0]:
                asp = False
            if not eq(str(fb), str(ref))[0]:
                bsp = False
        print(f"{name:48s} {ab!s:>5s} {asp!s:>7s} {bsp!s:>7s}  {why_ab[:110]}", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "run":
        wt, tag = pathlib.Path(sys.argv[2]), sys.argv[3]
        run_side(wt, tag, pathlib.Path(os.environ["PAIR_LOGS"]) / tag)
    else:
        compare(pathlib.Path(sys.argv[2]), sys.argv[3], pathlib.Path(sys.argv[4]), sys.argv[5])
