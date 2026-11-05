#!/bin/bash
# b2_plan.sh — re-plan one batched capture under a PLANOPT variant (PLAN_B2.md), CPU only, and print its census.
#   bash tools/rwkv7/b2_plan.sh CAPTURE_DIR PLAN_ROOT NAME [PLANOPT]      (PLANOPT: base | b2 | any name in run_box.sh)
# The plan lands in PLAN_ROOT/bootstrap_placements/NAME; the census is tools/rwkv7/b2_census.py's.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
W=$(git -C "$HERE" rev-parse --show-toplevel)
CAP=${1:?capture}; ROOT=${2:?plan root}; NAME=${3:?name}; PLANOPT=${4:-base}
PY=${PY:-$W/.venv/bin/python}
# shellcheck source=tools/rwkv7/planopt.sh
. "$HERE/planopt.sh"
planopt_env "$PLANOPT"
echo "b2_plan: PLANOPT=$PLANOPT -> ${PLANOPT_ENV[*]:-(none)}"
env LANDING="${LANDING:-high}" ${PLANOPT_ENV[@]+"${PLANOPT_ENV[@]}"} PLAN_ROOT="$ROOT" PY="$PY" PYTHONPATH="$W/python" \
    MEMGUARD_GB="${MEMGUARD_GB-6}" bash "$W/tools/fhe_ssm/make_plan.sh" "$CAP" "$NAME" | tail -6
grep -E "placed_input_level_hist|placements=" "$ROOT/bootstrap_placements/$NAME/plan.log" | sed 's/^/  /'
grep -iE "warn|error|infeasible|forced" "$ROOT/bootstrap_placements/$NAME/plan.log" | sed 's/^/  ! /' || true
"$PY" "$HERE/b2_census.py" "$CAP" "$ROOT/bootstrap_placements/$NAME"
