#!/bin/bash
# One instrumented free run on an existing checkout (built core, the checkpoint and tokens.json in place as RUNS.md's
# setup leaves them): the run under the per-tick meter (gpu_run --metrics) with the box sampler beside it, its meta,
# the analysis, and one archive to send back. Nothing else should run on the box meanwhile.
#
#   bash examples/fhe_ssm_from_primitives/blips_run.sh                # planned, 80 ticks, GPU 0, out: blips_out/
#   MODE=eager TICKS=40 GPU=1 bash examples/fhe_ssm_from_primitives/blips_run.sh
#
# Knobs: MODE=planned|eager  TICKS=80  GPU=0  PLAN=bootstrap_placements/fhe_ssm  OUT=blips_out
#        BLIPS_STAGE_ARENA_GB=2 BLIPS_KV_ARENA_GB=1  (the port's pinned host arenas, as the 10-13 runs)
#        WEIGHTS=pbd430a_weights.npz SEEDS=pbd430a_seeds.json CONFIG=pbd430a_config.json TOKENS=tokens.json
#        FAKE=1 BLOCKS=n  (the numpy fake instead of a session, eager only: a smoke test of the plumbing)
# Writes OUT/<tag>.{json,jsonl,log,meta}, OUT/<tag>_samples/{gpu.csv,host.csv,procs.txt}, OUT/analysis/, and
# OUT/../<tag>_<host>_<date>.tar.gz with its sha256 beside it. A cut-off run keeps every finished tick.
set -o pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "$HERE/../.." && pwd)"; cd "$REPO" || exit 1
MODE="${MODE:-planned}"; TICKS="${TICKS:-80}"; GPU="${GPU:-0}"; PLAN="${PLAN:-bootstrap_placements/fhe_ssm}"
OUT="${OUT:-blips_out}"; TAG="blips_${MODE}${TICKS}"
# shellcheck disable=SC1091
source scripts/local_env.sh; export CHAIN="${CHAIN:-n32}"
# the port's host arenas (it streams no blocks and keeps no K/V cache), not local_env.sh's 24 GB GPT-2 defaults: the
# 10-13 runs used these; BLIPS_STAGE_ARENA_GB / BLIPS_KV_ARENA_GB override them
export CUDA_VISIBLE_DEVICES="$GPU" FHE_STAGE_ARENA_GB="${BLIPS_STAGE_ARENA_GB:-2}" KV_ARENA_GB="${BLIPS_KV_ARENA_GB:-1}"
unset FIDESLIB_ERRCHECK FHE_PROFILE          # stock: no per-launch polls, no per-step device syncs
PY="${PYTHON:-python}"
M=(--weights "${WEIGHTS:-pbd430a_weights.npz}" --seeds "${SEEDS:-pbd430a_seeds.json}" --config "${CONFIG:-pbd430a_config.json}")
T=(--tokens "${TOKENS:-tokens.json}")
for f in "${M[@]}" "${T[@]}"; do case "$f" in --*) ;; *) [ -f "$f" ] || { echo "missing $f (RUNS.md: Setup)"; exit 1; } ;; esac; done
extra=(); samp=()
if [ -n "${FAKE:-}" ]; then
    MODE=eager; TAG="blips_fake${TICKS}"; extra=(--fake --blocks "${BLOCKS:-1}"); samp=(--no-gpu --ps-every 0)
    export CUDA_VISIBLE_DEVICES=""
elif [ "$MODE" = planned ]; then
    [ -f "$PLAN/block_0_placement.json" ] || { echo "no plan at $PLAN (RUNS.md: the plan, once)"; exit 1; }
    extra=(--plan-dir "$PLAN")
fi
if pgrep -f "fhe_ssm_from_primitives\.(gpu_run|plan|sample_box)|cmake --build|nvcc|pytest" > /dev/null; then
    echo "BOX_NOT_IDLE:"; pgrep -fl "fhe_ssm_from_primitives\.(gpu_run|plan|sample_box)|cmake --build|nvcc|pytest"; exit 1
fi
mkdir -p "$OUT"; rm -f "$OUT/$TAG.jsonl"
{ date -u +"START %FT%TZ"; echo "HOST $(hostname)"; echo "HEAD $(git log -1 --format=%h 2>/dev/null) FIDESlib $(git -C third_party/FIDESlib log -1 --format=%h 2>/dev/null)"
  echo "core $(sha256sum perseus/_core*.so 2>/dev/null | cut -c1-16)"; echo "MODE=$MODE TICKS=$TICKS GPU=$GPU PLAN=$PLAN FAKE=${FAKE:-}"
  env | grep -E "^(CHAIN|LOGN|CUDA_VISIBLE_DEVICES|FIDESLIB_|FHE_|SPARSE_|LADDER|OMP_|MALLOC_|KV_)" | sort
  command -v nvidia-smi > /dev/null && nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader
  uptime; command -v free > /dev/null && free -g | head -2; command -v nproc > /dev/null && nproc; } > "$OUT/$TAG.meta"
echo "[$(date -u +%T)] $TAG starting (meta: $OUT/$TAG.meta)"
$PY -m examples.fhe_ssm_from_primitives.gpu_run "${M[@]}" --mode "$MODE" "${extra[@]}" "${T[@]}" --free-run --ticks "$TICKS" \
    --out "$OUT/$TAG.json" --metrics "$OUT/$TAG.jsonl" > "$OUT/$TAG.log" 2>&1 &
RUN=$!
$PY -m examples.fhe_ssm_from_primitives.sample_box --out "$OUT/${TAG}_samples" --gpu "$GPU" --pid "$RUN" --interval 1 --ps-every 5 "${samp[@]}" > "$OUT/${TAG}_sampler.log" 2>&1 &
SB=$!
wait "$RUN"; RC=$?
sleep 2; kill -TERM "$SB" 2>/dev/null; wait "$SB" 2>/dev/null
{ echo "${TAG}_RC=$RC"; date -u +"END %FT%TZ"; uptime; } >> "$OUT/$TAG.meta"
grep -E '^\{"pos"' "$OUT/$TAG.log" | $PY -c 'import sys, json; r = [json.loads(l) for l in sys.stdin]; print("ticks", len(r), "seconds", [x["seconds"] for x in r])'
if [ "$RC" != 0 ]; then grep -E "Traceback|Error|Killed|out of memory" "$OUT/$TAG.log" | head -5; echo "$TAG FAILED rc=$RC (log: $OUT/$TAG.log)"; fi
nv=(); [ -f "$OUT/${TAG}_samples/gpu.csv" ] && nv=(--nvsmi "$OUT/${TAG}_samples/gpu.csv")
$PY -m examples.fhe_ssm_from_primitives.blips --report "$OUT/$TAG.json" --metrics "$OUT/$TAG.jsonl" "${nv[@]}" \
    --host "$OUT/${TAG}_samples/host.csv" --label "$TAG" --out "$OUT/analysis" > "$OUT/analysis_$TAG.txt" 2>&1 \
    && sed -n 1,30p "$OUT/analysis_$TAG.txt" || echo "analysis failed (see $OUT/analysis_$TAG.txt); the records are still complete"
ARCH="$(dirname "$OUT")/${TAG}_$(hostname)_$(date -u +%Y%m%dT%H%MZ).tar.gz"
tar czf "$ARCH" -C "$(dirname "$OUT")" "$(basename "$OUT")" && sha256sum "$ARCH" | tee "$ARCH.sha256"
echo "[$(date -u +%T)] $TAG done rc=$RC; send $ARCH (and its .sha256)"
exit "$RC"
