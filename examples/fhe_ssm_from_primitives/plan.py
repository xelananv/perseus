# SPDX-License-Identifier: MIT
"""Plan the bootstraps of graphs captured by ``gpu_run.py --mode capture`` (CPU only).

Perseus's placer with the n32 recipe of its own GPT-2 example (examples/gpt2_from_primitives/
make_plan.sh): a planted refresh lands at 36 on the composite chain, a block's input and a
carried value arrive at 34 plus a pending rescale = 36, the plan may use the chain up to 50,
kappa = 2 on every captured magnitude, correction factors up to 20 priced with the measured
n32 accuracy table, sparse routes {512, 1} with their landings, redundant refreshes pruned, no
prescale (the scale folds, scales.py, already lift the small values). Then the capture's
contract (capture_env.json) is stamped into every plan, so a run under a different environment
refuses it.

    python -m examples.fhe_ssm_from_primitives.plan --graph-dir graphs/fhe_ssm \\
        --name fhe_ssm [--set PLAN_NO_PRESCALE=] [--dry-run]

Plans land in ``bootstrap_placements/<name>/block_<b>_placement.json`` (b = 0..24), where
``gpu_run.py --mode planned --plan-dir bootstrap_placements/<name>`` reads them, beside a
``PLAN_CMD.txt`` in the repository's form: the graph, the recipe the runner was given, and the
switches the plan needs at run time (``runtime_switches``), which the planned run exports.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from collections.abc import MutableMapping
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts" / "utils" / "run_bootstrap_all_blocks.sh"

#: make_plan.sh's plan_blocks() defaults, variable for variable (an empty value means "unset")
RECIPE = {
    "MAX_LEVEL": "50",
    "BTS_LEVEL": "36",
    "SRC_LEVEL": "36",
    "CACHE_READ_LEVEL": "36",
    "PLAN_LEVEL_UNIT": "2",
    "PLAN_CF_MAX": "20",
    "PLAN_NO_PRESCALE": "1",
    "PLAN_ACC_CHAIN": "n32",
    "PLAN_MAG_SAFETY": "2",
    "PLAN_HINT_ENV_VETO": "1",
    "PLAN_SPARSE_SLOTS": "512,1",
    "PLAN_SPARSE_BTS_OUT": "1:26,512:36",
    "PLAN_PRUNE": "1",
    "PLAN_REAL_ROUTE": "1",
    "PLAN_RAISE_DROP_SET": "1,3,5",
    "PLAN_SHARED_REALIZE": "1",
}


#: variables whose empty value means something (make_plan.sh): no measured table, a dense plan
SET_EMPTY = ("PLAN_ACC_CHAIN", "PLAN_SPARSE_SLOTS")


def dense(env: dict[str, str]) -> bool:
    """A dense plan: every refresh on the dense route (PLAN_SPARSE_SLOTS set and empty)."""
    return env.get("PLAN_SPARSE_SLOTS") == ""


def plan_env(graph_dir: str, name: str, overrides: dict[str, str] | None = None) -> dict[str, str]:
    """The environment the runner reads: the recipe, the caller's overrides, the graph and the
    output name. An empty override removes a variable (the runner tests ``${VAR:-}``), except
    where empty means something; a dense plan also clamps the deliberate refreshes' landing,
    as make_plan.sh does."""
    env = {**RECIPE, **(overrides or {}), "GRAPH_DIR": graph_dir, "OUT_NAME": name}
    if dense(env):
        env.setdefault("PLAN_DELIBERATE_CLAMP0", "1")
    return {k: v for k, v in env.items() if v != "" or k in SET_EMPTY}


def stamp(graph_dir: Path, plans: list[Path], is_dense: bool = False) -> int:
    """The capture's contract into every plan (make_plan.sh's stamp): a dense plan runs
    without the sparse keys, so its contract says so."""
    from perseus.plan import contract

    env_file = graph_dir / "capture_env.json"
    if not env_file.exists():
        raise SystemExit(f"{env_file} is missing: plans would bind to no environment")
    with open(env_file, encoding="utf-8") as f:
        captured = json.load(f)
    if is_dense:
        captured.setdefault("env", {}).update(SPARSE_AUTO="0", SPARSE_BTS_SLOTS="0")
    for p in plans:
        contract.stamp_file(str(p), captured)
    return len(plans)


def runtime_switches(plans: list[Path]) -> dict[str, str]:
    """The FIDESlib switches a planned run needs beyond the capture's contract, read from the plans
    themselves: the real-payload StC variant where a refresh takes the real route
    (``FIDESLIB_BTS_REAL=1``) and the raise variants the refreshes use (``FIDESLIB_BTS_RAISE_DROPS``),
    both built when the context loads and refused by the runtime when missing."""
    real = False
    drops: set[int] = set()
    for p in plans:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        real = real or bool(d.get("real_route"))
        drops.update(int(v) for v in (d.get("raise_drop") or {}).values())
    out: dict[str, str] = {}
    if real:
        out["FIDESLIB_BTS_REAL"] = "1"
    if drops:
        out["FIDESLIB_BTS_RAISE_DROPS"] = ",".join(str(k) for k in sorted(drops))
    return out


def write_plan_cmd(out_dir: Path, graph_dir: str, env: dict[str, str], plans: list[Path]) -> Path:
    """``PLAN_CMD.txt`` in the repository's form: ``graph=``, ``recipe=`` (the runner's variables,
    so ``tests/test_paper_plans.py`` regenerates the plan with the plain runner) and ``runtime=``
    (``runtime_switches``, exported by ``gpu_run.py --mode planned`` before the session opens)."""
    recipe = " ".join(
        f"{k}={shlex.quote(v)}" for k, v in env.items() if k not in ("GRAPH_DIR", "OUT_NAME")
    )
    runtime = " ".join(f"{k}={v}" for k, v in runtime_switches(plans).items())
    p = out_dir / "PLAN_CMD.txt"
    p.write_text(f"graph={graph_dir}\nrecipe={recipe}\nruntime={runtime}\n", encoding="utf-8")
    return p


def plan_runtime(plan_dir: str | Path) -> dict[str, str]:
    """The ``runtime=`` switches of a plan directory's ``PLAN_CMD.txt``; empty without the file."""
    cmd = Path(plan_dir) / "PLAN_CMD.txt"
    if not cmd.exists():
        return {}
    for line in cmd.read_text(encoding="utf-8").splitlines():
        if line.startswith("runtime="):
            return dict(kv.split("=", 1) for kv in shlex.split(line[len("runtime=") :]))
    return {}


def export_plan_runtime(
    plan_dir: str | Path, environ: MutableMapping[str, str] | None = None
) -> dict[str, str]:
    """Export the plan's runtime switches where they are not already set (an exported value wins,
    as everywhere in the repository); returns what the plan asks for."""
    env = os.environ if environ is None else environ
    runtime = plan_runtime(plan_dir)
    for k, v in runtime.items():
        env.setdefault(k, v)
    return runtime


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--graph-dir", required=True, help="block_<b>/graph.json captures")
    ap.add_argument("--name", required=True, help="bootstrap_placements/<name>/")
    ap.add_argument("--set", action="append", default=[], metavar="VAR=VALUE",
                    help="override a recipe variable (VAR= unsets it)")  # fmt: skip
    ap.add_argument("--dry-run", action="store_true", help="print the environment, plan nothing")
    a = ap.parse_args(argv)
    overrides = {}
    for kv in a.set:
        k, sep, v = kv.partition("=")
        if not sep or not k:
            ap.error(f"--set {kv!r}: VAR=VALUE")
        overrides[k] = v
    env = plan_env(a.graph_dir, a.name, overrides)
    if a.dry_run:
        print(json.dumps(env, indent=1, sort_keys=True))
        return 0
    graph_dir = Path(a.graph_dir)
    blocks = sorted(graph_dir.glob("block_*/graph.json"))
    if not blocks:
        raise SystemExit(f"no block_<b>/graph.json under {graph_dir}: capture first")
    full = {**{k: v for k, v in os.environ.items() if k not in RECIPE}, **env,
            "CUDA_VISIBLE_DEVICES": "", "PYTHON": sys.executable}  # fmt: skip
    subprocess.run(["bash", str(RUNNER)], cwd=ROOT, env=full, check=True)
    plans = sorted((ROOT / "bootstrap_placements" / a.name).glob("block_*_placement.json"))
    n = stamp(graph_dir, plans, dense(env))
    # the recipe line carries the planner switches the shell exported beyond the recipe (PLAN_HARD_ENV_CAP,
    # PLAN_PLACER, ...), so the plan regenerates from the file alone
    extra = {k: v for k, v in sorted(os.environ.items()) if k.startswith("PLAN_") and k not in env}
    cmd = write_plan_cmd(
        ROOT / "bootstrap_placements" / a.name, a.graph_dir, {**env, **extra}, plans
    )
    print(
        f"planned {len(blocks)} captured stages -> {len(plans)} plans; stamped {n}; {cmd.name}: {plan_runtime(cmd.parent) or 'no runtime switches'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
