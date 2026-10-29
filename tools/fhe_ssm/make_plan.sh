#!/bin/bash
# make_plan.sh — plan the bootstraps of an fhe-ssm capture on this branch (CPU only), with the plan-time speed axes.
#
#   bash tools/fhe_ssm/make_plan.sh GRAPH_DIR NAME            -> $PLAN_ROOT/bootstrap_placements/NAME/
#   DRY_RUN=1 bash tools/fhe_ssm/make_plan.sh GRAPH_DIR NAME  print the recipe, the runtime= line and the command only
#
# GRAPH_DIR: a capture (block_<b>/graph.json for every stage, capture_config.json or capture_env.json), made by the
# port's run_fake.py --capture UNDER THE SAME LANDING's runtime switches (the capture's own refreshes land where the
# fake's landing table says; a plan binds only when capture, recipe and runtime agree — the compile check below catches
# a mismatch). The port's capture takes the landing's runtime switches from the environment (fakes.runtime_config,
# fakes.LANDING_SWITCHES in python/perseus/examples/fhe_ssm/fakes.py) and stamps them in capture_config.json, so a
# LANDING=high plan needs a capture made under the high switches (run_gpu.sh capture exports them); on a stock capture
# its block 1 is infeasible. Every block is planned in ONE planner pass, block b entered at block b-1's planned exit
# (python/perseus/plan/run_bootstrap_all_blocks.sh -> python -m perseus.plan.placer).
#
# Axes (environment; default first):
#   LANDING=stock|high|fork   where a refresh lands, i.e. how many levels it hands back (levels count primes used, 0 =
#                             fresh, 50 = the top of the n32 chain; a refresh landing at L leaves 50 - L primes of work):
#       stock  this runtime without StC-first: dense refreshes in the shipped order land at 34, SparseB 512-slot at 24,
#              SPRU 1-slot at 16 (the landings of bootstrap_placements/gpt2_decode_python_n32_shipped, whose recipe
#              this reproduces; crates/pc-shadow/src/route.rs:92-110 and crates/pc-program/src/compile/mod.rs:246-293
#              are the runtime's own table). runtime: FIDESLIB_BTS_STC_FIRST=0, BTS_MAX_INPUT_LEVEL=50.
#       high   the GPT-2 speedrun recipe (speedrun/make_prefill_plan.sh RECIPE, :56-58, at its PRESET=n32): dense
#              refreshes StC-first at 28 (an input deeper than 46 falls back to the shipped order at 34), SparseB
#              512-slot at 26 (raise drop 1), SPRU 1-slot at 16. runtime: FIDESLIB_BTS_STC_FIRST=2 (dense),
#              STC_FIRST_CF=13, FIDESLIB_BTS_RAISE_DROP=512:1, AUTO_BTS_LEVEL=44, BTS_MAX_INPUT_LEVEL=48 (the n32
#              Decode-8 plan's line, bootstrap_placements/gpt2_decode_python_n32_kc16_cm; Decode-8 record 4's plan,
#              gpt2_decode_python_n32a4_kc16r_cm, adds CHAIN_PRESET=n32a4, the lean chain, which this port refuses).
#       fork   the fork's examples/fhe_ssm_from_primitives/plan.py RECIPE literally (dense and 512-slot land at 36,
#              1-slot at 26, MAX_LEVEL 50, PLAN_HARD_ENV_CAP=1): FIDESlib-C++ landings, which this runtime does not
#              realize. On a capture of this branch's fake such a plan still binds (GPT-2 decode blocks 0-12: 0 check
#              errors, 12,991 pins richer than planned) but over-refreshes: 194 refreshes against 135 for LANDING=stock
#              on the same capture, the planner placing refreshes for levels the runtime hands back anyway. Kept to
#              compare refresh COUNTS with the fork's records (953 per tick), not as a speed arm.
#   PAIR=0|1                  1: PC_BTS_PAIR=1 on the runtime= line: two sibling real-payload dense StC-first refreshes
#                             of one level and degree run as one complex refresh (crates/pc-program/src/program.rs:269-275).
#                             Needs LANDING=high (no StC-first, no pairs) and a compile schedule that brings the mates
#                             together: SCHEDULE=siblings for the check (the speedrun's PlannedPrefill default).
#   PLACER=mincut|ilp         ilp: PLAN_PLACER=ilp with the speedrun's PLAN_ILP_TIME_LIMIT=3600 PLAN_ILP_TIEBREAK=1
#                             (one optimum on every machine; needs scipy). mincut: the default placer.
#   ILP_OBJECTIVE=count|ms    count = --ilp-objective total (placed + fired hints); ms = each refresh priced at its
#                             route's latency PLAN_BTS_MS (default: perseus-core's own PRO 6000 table, one ciphertext,
#                             origin/main README.md:109: dense 20.3, StC-first 17.1, sparse-512 17.4, one slot 10.0 ms;
#                             a 32-slot route is priced at the 256-slot figure 16.4, which is unmeasured for 32).
#   SPARSE_SLOTS=512,1|512,32,1  the sparse routes the planner may use. 32 needs a landing in the runtime's table, which
#                             d97a6f7 does not have (compile/mod.rs:250 lists 1024/512/256 at 24): expect the compile
#                             check to refuse "no landing for route slots=32" until upstream adds one.
#   HARD_ENV_CAP=1|0|''       PLAN_HARD_ENV_CAP: 1 = the refresh envelope is absolute (the fork's plans needed it on a
#                             residual stream that crosses blocks unrefreshed: without it the envelope relaxes and the
#                             blocks' exits drift to the top of the window); '' = planner default.
#   SPARSE_B_K=''|16|24       FIDESLIB_BTS_SPARSE_B_K on the runtime= line (SparseB's EvalMod series; the Prefill-128
#                             record runs 16). Replay only: it moves no refresh.
#   RUNTIME_EXTRA='K=V ...'   more runtime= tokens (they win: later tokens win, crates/pc-program/src/plan.rs:152).
# Any recipe variable the caller exports wins over these defaults (MAX_LEVEL, BTS_LEVEL, PLAN_*; set-but-empty is
# kept, e.g. PLAN_SPARSE_SLOTS= for a dense plan).
# Other knobs: PLAN_ROOT (default <worktree>/.cache/fhe_ssm, gitignored; the plan lands in PLAN_ROOT/bootstrap_placements/
# NAME), PY (default <worktree>/.venv/bin/python), MEMGUARD_GB (the planner under <worktree>/../../tools/memguard.sh at
# that ceiling when the script exists; default 6, '' = no guard), CHECK=1|0 (run tools/fhe_ssm/compile_check on the
# result when its binary exists: bash tools/fhe_ssm/compile_check/build.sh), SCHEDULE=program|siblings|lockstep (the
# check's compile schedule; default siblings with PAIR=1, else program).
#
# Writes, beside block_<b>_placement.json (each stamped with the capture's Config, so a session under another Config
# refuses it): PLAN_CMD.txt (graph=, tool=, planner=, axes=, recipe=, runtime=: the lines pc-program reads,
# crates/pc-program/src/plan.rs:152-259), plan_rot_steps.json (every rotation step the stages' graphs use: the session's
# rotation-key scope, impl/driver.py:570-585), and summary.txt (per-block refresh count and exit level, the total, and
# the compile check's output).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
W=$(git -C "$HERE" rev-parse --show-toplevel)
GRAPH_DIR="${1:?usage: make_plan.sh GRAPH_DIR NAME}"; NAME="${2:?usage: make_plan.sh GRAPH_DIR NAME}"
# NAME becomes a directory that is deleted and rewritten: one plain path component, nothing else
[[ "$NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] && [[ "$NAME" != *..* ]] \
    || { echo "make_plan: NAME '$NAME' must be one path component of [A-Za-z0-9._-], starting alphanumeric, without '..'" >&2; exit 2; }
[ -d "$GRAPH_DIR" ] || { echo "make_plan: no graph dir $GRAPH_DIR" >&2; exit 2; }
GRAPH_DIR=$(cd "$GRAPH_DIR" && pwd)
ls "$GRAPH_DIR"/block_*/graph.json >/dev/null 2>&1 || { echo "make_plan: no block_<b>/graph.json under $GRAPH_DIR" >&2; exit 2; }
PY=${PY:-$W/.venv/bin/python}
PLAN_ROOT=${PLAN_ROOT:-$W/.cache/fhe_ssm}
LANDING=${LANDING:-stock}; PAIR=${PAIR:-0}; PLACER=${PLACER:-mincut}; ILP_OBJECTIVE=${ILP_OBJECTIVE:-count}
SPARSE_SLOTS=${SPARSE_SLOTS:-512,1}; HARD_ENV_CAP=${HARD_ENV_CAP-1}; SPARSE_B_K=${SPARSE_B_K:-}
MEMGUARD_GB=${MEMGUARD_GB-6}; CHECK=${CHECK:-1}
die() { echo "make_plan: $*" >&2; exit 2; }
case "$PAIR" in 0|1) ;; *) die "PAIR=$PAIR: 0|1" ;; esac
case "$PLACER" in mincut|ilp) ;; *) die "PLACER=$PLACER: mincut|ilp" ;; esac
case "$ILP_OBJECTIVE" in count|ms) ;; *) die "ILP_OBJECTIVE=$ILP_OBJECTIVE: count|ms" ;; esac
case "$SPARSE_SLOTS" in 512,1|512,32,1) ;; *) die "SPARSE_SLOTS=$SPARSE_SLOTS: 512,1|512,32,1" ;; esac
[ "$ILP_OBJECTIVE" = ms ] && [ "$PLACER" != ilp ] && die "ILP_OBJECTIVE=ms needs PLACER=ilp"
has32=0; [ "$SPARSE_SLOTS" = 512,32,1 ] && has32=1

# ---- the recipe: base (every LANDING), then the landing's own, then the axes; an exported value always wins
setd() { local k=$1 v=$2; if [ -z "${!k+x}" ]; then export "$k=$v"; fi; RECIPE_KEYS+=("$k"); }
RECIPE_KEYS=()
setd MAX_LEVEL 50; setd SRC_LEVEL 36; setd CACHE_READ_LEVEL 36; setd PLAN_LEVEL_UNIT 2; setd PLAN_CF_MAX 20
setd PLAN_NO_PRESCALE 1; setd PLAN_ACC_CHAIN n32; setd PLAN_MAG_SAFETY 2; setd PLAN_HINT_ENV_VETO 1
setd PLAN_SPARSE_SLOTS "$SPARSE_SLOTS"; setd PLAN_PRUNE 1; setd PLAN_REAL_ROUTE 1; setd PLAN_RAISE_DROP_SET 1,3,5
setd PLAN_SHARED_REALIZE 1
[ -n "$HARD_ENV_CAP" ] && setd PLAN_HARD_ENV_CAP "$HARD_ENV_CAP"
RT_COMMON="FIDESLIB_BTS_SHIFT=1 FIDESLIB_SPRU=64 FIDESLIB_BTS_RAISE_DROPS=1,3,5 FIDESLIB_BTS_REAL=1 FIDESLIB_BTS_SPARSE_B=1 FIDESLIB_ROT_KEY_BAND=32"
case "$LANDING" in
    stock)
        setd BTS_LEVEL 34; setd PLAN_SPARSE_BTS_OUT "1:16,512:24$([ $has32 = 1 ] && echo ,32:24)"
        setd PLAN_REFRESH_ENV_CAP 50
        RUNTIME="$RT_COMMON FIDESLIB_BTS_STC_FIRST=0 BTS_MAX_INPUT_LEVEL=50" ;;
    high)
        setd BTS_LEVEL 28; setd PLAN_SPARSE_BTS_OUT "1:16,512:26$([ $has32 = 1 ] && echo ,32:26)"
        setd PLAN_REFRESH_ENV_CAP 50
        setd PLAN_RAISE_DROP_MAX 0; setd PLAN_RAISE_LANDING_MAX 42; setd PLAN_RAISE_ROUTES 0
        setd PLAN_REFRESH_ENV_CAP_ROUTES 0:46; setd PLAN_SLIM_CUBIC 1; setd PLAN_STC_FIRST_ENTRY 46
        setd PLAN_STC_FIRST_FALLBACK 34:1
        RUNTIME="$RT_COMMON FIDESLIB_BTS_STC_FIRST=2 FIDESLIB_BTS_STC_FIRST_ROUTES=dense FIDESLIB_BTS_STC_FIRST_CF=13 FIDESLIB_BTS_RAISE_DROP=512:1 AUTO_BTS_LEVEL=44 BTS_MAX_INPUT_LEVEL=48" ;;
    fork)
        setd BTS_LEVEL 36; setd PLAN_SPARSE_BTS_OUT "1:26,512:36$([ $has32 = 1 ] && echo ,32:36)"
        RUNTIME="$RT_COMMON FIDESLIB_BTS_STC_FIRST=0 BTS_MAX_INPUT_LEVEL=50"
        echo "make_plan: LANDING=fork = FIDESlib-C++ landings this runtime does not realize: a count comparison, not a speed arm" >&2 ;;
    *) die "LANDING=$LANDING: stock|high|fork" ;;
esac
if [ "$PLACER" = ilp ]; then
    setd PLAN_PLACER ilp; setd PLAN_ILP_TIME_LIMIT 3600; setd PLAN_ILP_TIEBREAK 1
    if [ "$ILP_OBJECTIVE" = ms ]; then
        setd PLAN_ILP_OBJECTIVE ms
        dense_ms=$([ "$LANDING" = high ] && echo 17.1 || echo 20.3)
        setd PLAN_BTS_MS "0:$dense_ms,512:17.4,1:10.0$([ $has32 = 1 ] && echo ,32:16.4)"
    else
        setd PLAN_ILP_OBJECTIVE total
    fi
fi
[ "$PAIR" = 1 ] && RUNTIME="$RUNTIME PC_BTS_PAIR=1"
[ "$PAIR" = 1 ] && [ "$LANDING" != high ] && echo "make_plan: PAIR=1 without LANDING=high forms no pair (pairs need StC-first: crates/pc-program/src/compile/mod.rs:509-514)" >&2
[ -n "$SPARSE_B_K" ] && RUNTIME="$RUNTIME FIDESLIB_BTS_SPARSE_B_K=$SPARSE_B_K"
[ -n "${RUNTIME_EXTRA:-}" ] && RUNTIME="$RUNTIME $RUNTIME_EXTRA"
# the chain is a runtime Config field (CHAIN_PRESET): this model's plaintext rows and levels are n32's, so a runtime
# line naming any preset is refused here, before the planner and the compile check could run on another chain; a
# quote is refused too (the Rust parser splits the line on whitespace, the Python one with shlex: a quoted token
# would read differently on the two sides: crates/pc-program/src/plan.rs vs python/perseus/config/__init__.py)
case " $RUNTIME " in *" CHAIN_PRESET="*) echo "make_plan: the runtime line names a chain preset (CHAIN_PRESET=...); this model plans on n32 only: drop it" >&2; exit 2 ;; esac
case "$RUNTIME" in *\"*|*\'*) echo "make_plan: a quote in the runtime line; the runtime and the driver would parse it differently: drop it" >&2; exit 2 ;; esac
SCHEDULE=${SCHEDULE:-$([ "$PAIR" = 1 ] && echo siblings || echo program)}
# every PLAN_* the caller exported is part of the recipe too (the planner reads several straight from the env);
# PLAN_ROOT is where the plan lands, not a planner input: it stays out of the recipe line (a machine path otherwise)
for k in $(env | sed -n 's/^\(PLAN_[A-Z0-9_]*\)=.*/\1/p' | grep -vx PLAN_ROOT | sort); do
    case " ${RECIPE_KEYS[*]} " in *" $k "*) ;; *) RECIPE_KEYS+=("$k") ;; esac
done
for k in FIRST_ENTRY_LEVEL FIRST_ENTRY_DEG KV_ENTRY_LEVEL KV_ENTRY_DEG FORBID_STEPS BTS_OUT_DEG; do
    [ -n "${!k+x}" ] && RECIPE_KEYS+=("$k")
done
RECIPE=""
for k in "${RECIPE_KEYS[@]}"; do RECIPE="$RECIPE $k=${!k}"; done
RECIPE=${RECIPE# }
AXES="LANDING=$LANDING PAIR=$PAIR PLACER=$PLACER ILP_OBJECTIVE=$ILP_OBJECTIVE SPARSE_SLOTS=$SPARSE_SLOTS HARD_ENV_CAP=$HARD_ENV_CAP SPARSE_B_K=$SPARSE_B_K"
OUT="$PLAN_ROOT/bootstrap_placements/$NAME"
echo "make_plan: $GRAPH_DIR -> $OUT"
echo "  axes:    $AXES"
echo "  recipe:  $RECIPE"
echo "  runtime: $RUNTIME"

# ---- the capture's own switches vs the landing's (capture_config.json; a legacy capture_env.json is not checked)
"$PY" - "$GRAPH_DIR" "$LANDING" <<'PY' || true
import json, os, sys
g, landing = sys.argv[1:]
p = os.path.join(g, "capture_config.json")
if not os.path.exists(p):
    print("  capture: legacy capture_env.json only, its route switches are not recorded: not checked")
    sys.exit(0)
b = json.load(open(p)).get("bts", {})
# the switches that move where the capture's OWN refreshes (the deliberate ones) land: the StC-first order and the
# 512-slot raise drop; the sparse routes' kind (SparseB or classic) comes from the plan's runtime= line at compile time
stc, rd = int(b.get("stc_first") or 0), b.get("raise_drop")
rd = rd if rd not in (None, "", "0") else None
want = {"high": (2, "512:1"), "stock": (0, None), "fork": (0, None)}[landing]
print(f"  capture: stc_first={stc} raise_drop={rd} sparse_b={b.get('sparse_b')} auto_bts_level={b.get('auto_bts_level')}"
      + ("" if (stc, rd) == want else f"  ** MISMATCH: LANDING={landing} runs stc_first={want[0]} raise_drop={want[1]};"
      " the capture's own refreshes may not land where this runtime lands them: the compile check decides **"))
PY
if [ "${DRY_RUN:-0}" = 1 ]; then
    echo "  DRY_RUN: would run (cwd $PLAN_ROOT) GRAPH_DIR=$GRAPH_DIR OUT_NAME=$NAME PYTHON=$PY bash $W/python/perseus/plan/run_bootstrap_all_blocks.sh"
    exit 0
fi

# ---- plan (from PLAN_ROOT: the runner writes ./bootstrap_placements/<OUT_NAME>)
mkdir -p "$PLAN_ROOT"
rm -rf "$OUT"
GUARD=()
MG="$W/../../tools/memguard.sh"
[ -n "$MEMGUARD_GB" ] && [ -x "$MG" ] && GUARD=("$MG" "$MEMGUARD_GB")
t0=$(date +%s)
( cd "$PLAN_ROOT" && CUDA_VISIBLE_DEVICES="" PYTHONPATH="$W/python${PYTHONPATH:+:$PYTHONPATH}" GRAPH_DIR="$GRAPH_DIR" \
    OUT_NAME="$NAME" PYTHON="$PY" ${GUARD[@]+"${GUARD[@]}"} bash "$W/python/perseus/plan/run_bootstrap_all_blocks.sh" ) \
    > "$PLAN_ROOT/$NAME.plan.log" 2>&1 || { tail -25 "$PLAN_ROOT/$NAME.plan.log"; echo "make_plan: the planner failed (log $PLAN_ROOT/$NAME.plan.log)" >&2; exit 1; }
plan_s=$(( $(date +%s) - t0 ))
cp "$PLAN_ROOT/$NAME.plan.log" "$OUT/plan.log"
rm -f "$PLAN_ROOT/$NAME.plan.log"

# ---- stamp, rotation scope, PLAN_CMD.txt, summary
PLANNER="$(git -C "$W" log -1 --abbrev=7 --format=%h -- python/perseus/plan) (python/perseus/plan's last commit)"
GRAPH_MD5=$(cat "$GRAPH_DIR"/block_*/graph.json | (md5sum 2>/dev/null || md5 -q) | cut -c1-32)
CUDA_VISIBLE_DEVICES="" PYTHONPATH="$W/python" "$PY" - "$GRAPH_DIR" "$OUT" "$PLAN_SPARSE_SLOTS" <<'PY'
import glob, json, os, sys
from perseus import config
from perseus.plan import contract
g, out, slots = sys.argv[1:]
cfg = config.capture_config(g)
if slots == "":          # a dense plan from any capture runs without sparse keys (make_plan.sh's stamp())
    cfg = cfg.with_env({"SPARSE_AUTO": "0", "SPARSE_BTS_SLOTS": "0"})
plans = sorted(glob.glob(os.path.join(out, "block_*_placement.json")))
for p in plans:
    contract.stamp_file(p, cfg)
steps = set()
for f in glob.glob(os.path.join(g, "block_*", "graph.json")):
    for n in json.load(open(f))["nodes"]:
        steps |= {int(v) for x in n["inputs"] if x.startswith("rot(") for v in x[4:-1].split(",")}
json.dump(sorted(steps), open(os.path.join(out, "plan_rot_steps.json"), "w"))
print(f"  stamped {len(plans)} block plans with the capture's Config; plan_rot_steps.json: {len(steps)} steps")
PY
{
    echo "graph=$GRAPH_DIR (md5 of the stages' graph.json $GRAPH_MD5)"
    echo "tool=tools/fhe_ssm/make_plan.sh (one planner pass over every stage: python/perseus/plan/run_bootstrap_all_blocks.sh)"
    echo "planner=$PLANNER"
    echo "axes=$AXES"
    echo "recipe=$RECIPE"
    echo "runtime=$RUNTIME"
    echo "solver=scipy $(CUDA_VISIBLE_DEVICES="" "$PY" -c 'import scipy; print(scipy.__version__)' 2>/dev/null || echo '?') ($([ "$PLACER" = ilp ] && echo 'milp (HiGHS)' || echo 'not used: min-cut'))"
} > "$OUT/PLAN_CMD.txt"
"$PY" - "$OUT" "$plan_s" <<'PY' | tee "$OUT/summary.txt"
import glob, json, os, re, sys
out, secs = sys.argv[1], sys.argv[2]
key = lambda p: (int(re.search(r"block_(\d+)", p).group(1)), p)
tot, rows = 0, []
for p in sorted(glob.glob(os.path.join(out, "block_*_placement.json")), key=key):
    s = json.load(open(p))["summary"]
    n = s.get("total_bootstraps") or 0
    tot += n
    rows.append(f"{os.path.basename(p).replace('_placement.json', '')}:{n}@{s.get('exit_level')}")
print(f"  planned in {secs} s: {len(rows)} stages, {tot} refreshes (summary.total_bootstraps), per stage refreshes@exit level:")
print("    " + " ".join(rows))
PY
CC="$W/.cache/fhe_ssm/compile_check/target/release/compile_check"
if [ "$CHECK" = 1 ] && [ -x "$CC" ]; then
    echo "  compile check (schedule $SCHEDULE):" | tee -a "$OUT/summary.txt"
    set +e
    "$CC" "$OUT" "$GRAPH_DIR" --schedule "$SCHEDULE" --json "$OUT/compile_check.json" > "$OUT/compile_check.txt" 2>&1
    rc=$?
    set -e
    tail -2 "$OUT/compile_check.txt" | sed 's/^/    /' | tee -a "$OUT/summary.txt"
    [ $rc = 0 ] || echo "  ** the compile check FAILED (exit $rc): this plan does not bind on this runtime; $OUT/compile_check.txt **" | tee -a "$OUT/summary.txt"
elif [ "$CHECK" = 1 ]; then
    echo "  compile check: skipped (build it: bash tools/fhe_ssm/compile_check/build.sh)" | tee -a "$OUT/summary.txt"
fi
echo "make_plan: $OUT"
