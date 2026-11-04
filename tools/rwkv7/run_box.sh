#!/bin/bash
# run_box.sh — RWKV-7 0.1B encrypted decode on the GPU box (checkout /root/perseus-rwkv7, branch fhe-ssm-rwkv7).
#
#   bash tools/rwkv7/run_box.sh all          build, check, crosscheck, capture, plan, run (BLOCKS unset = 12 blocks)
#   BLOCKS=1 bash tools/rwkv7/run_box.sh all one block + the tail (the smoke: same steps, own directories)
#   bash tools/rwkv7/run_box.sh STEP         one step: build | check | crosscheck | capture | plan | run
#
# Steps (every product under $OUT; a step reuses what an earlier one left):
#   build       tools/fhe_ssm/run_gpu.sh build (uv venv, pip install -e .[hf,examples,dev], maturin develop --release)
#   check       the device classes of perseus._pc, the planner's imports (networkx, scipy), the CPU tests of the port
#   crosscheck  (needs TORCH_PY, a python with torch: the harness's own env) the harness's poly model (torch) vs the
#               numpy reference the run is scored against, on the run's tokens -> crosscheck.json. Skipped without it.
#   capture     CPU only: PCMM band shifts from the fake on a public calibration text, then the capture on the
#               recorder (token 1 after the public prefix "\n\n"), under the LANDING=high switches -> $CAP
#   plan        tools/fhe_ssm/make_plan.sh $CAP (LANDING=high: dense refreshes StC-first, as PCMM requires) -> $PLAN
#   run         perseus.examples.rwkv7.run_planned on the GPU: setup, warm-up, TOKENS decode steps after the prefix;
#               one JSON line per token (top-1 vs the poly model, logits relErrRms, hidden relErrRms vs the poly
#               model and the circuit's numpy mirror, seconds) and a summary line -> $OUT/run$S.jsonl, run$S.log
#
# Environment: OUT (default /root/out_rwkv7), PTH, COEFFS, HARNESS (defaults: the brief's paths), TOKENS (12),
# BLOCKS (unset = all 12), GPU (0), TEXT (the prompt; must start with "\n\n", the public prefix), PCMM_SHIFT
# (calib | zero), PCMM_CF (6), TORCH_PY (crosscheck), RUN_EXTRA (more run_planned flags).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
W=$(git -C "$HERE" rev-parse --show-toplevel)
PY=${PY:-$W/.venv/bin/python}
OUT=${OUT:-/root/out_rwkv7}
PTH=${PTH:-/root/d_models/RWKV-x070-World-0.1B-v2.8-20241210-ctx4096.pth}
COEFFS=${COEFFS:-/root/vd/runs/0.1b/coeffs_mixed.json}
HARNESS=${HARNESS:-/root/vd/harness}
TOKENS=${TOKENS:-12}; BLOCKS=${BLOCKS:-}; GPU=${GPU:-0}; PCMM_SHIFT=${PCMM_SHIFT:-calib}; PCMM_CF=${PCMM_CF:-6}
TEXT=${TEXT:-$'\n\nThe Eiffel tower is in the city of Paris, which is the capital of France and'}
S=${BLOCKS:+_b$BLOCKS}${PCMM_SHIFT/calib/}; S=${S/zero/_zero}
CAP=$OUT/cap_rwkv7$S
PLAN=$OUT/bootstrap_placements/plan_rwkv7$S
# the LANDING=high runtime switches (tools/fhe_ssm/run_gpu.sh CAPENV): PCMM needs the StC-first dense route
CAPENV="FIDESLIB_BTS_STC_FIRST=2 FIDESLIB_BTS_STC_FIRST_ROUTES=dense FIDESLIB_BTS_STC_FIRST_CF=13 FIDESLIB_BTS_RAISE_DROP=512:1 AUTO_BTS_LEVEL=44"
COMMON="--pth $PTH --coeffs $COEFFS --harness $HARNESS ${BLOCKS:+--blocks $BLOCKS}"
mkdir -p "$OUT"
log() { echo "[rwkv7 $(date -u +%H:%M:%S)] $*" | tee -a "$OUT/session.log"; }
die() { log "STOP: $*"; exit 1; }
pyrun() { ( cd "$W" && env PYTHONPATH="$W/python" "$@" ); }

step_build() { OUT=$OUT bash "$W/tools/fhe_ssm/run_gpu.sh" build || die "build failed"; }

step_check() {
    for f in "$PTH" "$COEFFS" "$HARNESS/rwkv_tokenizer.py"; do [ -f "$f" ] || die "missing $f"; done
    pyrun "$PY" -c "from perseus import _pc; miss=[n for n in ('Session','Program','KeySource','PtStore') if not hasattr(_pc,n)]; print('device classes missing:', miss or 'none'); raise SystemExit(1 if miss else 0)" \
        || die "perseus._pc is not the CUDA build"
    pyrun "$PY" -c "import networkx, scipy; print('planner deps ok')" || die "the planner needs networkx and scipy in $PY"
    pyrun env RWKV7_PTH="$PTH" RWKV7_COEFFS="$COEFFS" RWKV7_HARNESS="$HARNESS" \
        "$PY" -m pytest -q -p no:warnings "$W/python/perseus/examples/rwkv7/tests" \
        --deselect python/perseus/examples/rwkv7/tests/test_capture.py::test_capture_and_plan_one_block \
        > "$OUT/tests.log" 2>&1 || { tail -20 "$OUT/tests.log"; die "CPU tests failed ($OUT/tests.log)"; }
    tail -1 "$OUT/tests.log" | tee -a "$OUT/session.log"
}

ids() { pyrun "$PY" -m perseus.examples.rwkv7.run_ids --harness "$HARNESS" --text "$TEXT" --n $((TOKENS + 1)); }

step_crosscheck() {
    [ -n "${TORCH_PY:-}" ] || { log "crosscheck: skipped (set TORCH_PY to a python with torch)"; return 0; }
    local I; I=$(ids) || die "tokenizer failed"
    "$TORCH_PY" "$HERE/harness_logits.py" --harness "$HARNESS" --pth "$PTH" --coeffs "$COEFFS" --ids "$I" \
        --out "$OUT/harness_logits.npz" || die "harness logits failed"
    pyrun "$PY" -m perseus.examples.rwkv7.run_crosscheck --pth "$PTH" --coeffs "$COEFFS" \
        --harness-logits "$OUT/harness_logits.npz" --out "$OUT/crosscheck.json" | tee -a "$OUT/session.log"
}

step_capture() {
    [ -f "$CAP/capture_config.json" ] && { log "capture: reusing $CAP"; return 0; }
    log "capture -> $CAP ($CAPENV)"
    pyrun env CUDA_VISIBLE_DEVICES="" $CAPENV "$PY" -m perseus.examples.rwkv7.run_capture $COMMON \
        --pcmm-shift "$PCMM_SHIFT" --out "$CAP" > "$OUT/capture$S.log" 2>&1 || { tail -20 "$OUT/capture$S.log"; die "capture failed"; }
    tail -3 "$OUT/capture$S.log" | tee -a "$OUT/session.log"
}

step_plan() {
    [ -f "$PLAN/PLAN_CMD.txt" ] && { log "plan: reusing $PLAN"; return 0; }
    log "plan $CAP -> $PLAN (LANDING=high)"
    LANDING=high PLAN_ROOT="$OUT" PY="$PY" MEMGUARD_GB= bash "$W/tools/fhe_ssm/make_plan.sh" "$CAP" "plan_rwkv7$S" \
        > "$OUT/plan$S.log" 2>&1 || { tail -30 "$OUT/plan$S.log"; die "plan failed ($OUT/plan$S.log)"; }
    tail -8 "$OUT/plan$S.log" | tee -a "$OUT/session.log"
}

step_run() {
    [ -f "$PLAN/PLAN_CMD.txt" ] || die "no plan $PLAN (run the plan step)"
    local busy; busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$GPU" 2>/dev/null)
    [ -z "$busy" ] || log "WARNING: GPU $GPU has other processes ($busy): timings are not clean"
    local I; I=$(ids) || die "tokenizer failed"
    log "run: $TOKENS tokens after the prefix, ids $I"
    pyrun env CUDA_VISIBLE_DEVICES=$GPU $CAPENV "$PY" -m perseus.examples.rwkv7.run_planned $COMMON --ids "$I" \
        --tokens "$TOKENS" --plan "$PLAN" --shifts "$CAP/pcmm_shifts.json" --pcmm-cf "$PCMM_CF" \
        --out "$OUT/run$S.jsonl" ${RUN_EXTRA:-} > "$OUT/run$S.log" 2>&1
    local rc=$?
    grep '"kind": "token"' "$OUT/run$S.jsonl" 2>/dev/null | cut -c1-220 | tee -a "$OUT/session.log"
    grep '"kind": "summary"' "$OUT/run$S.jsonl" 2>/dev/null | tee -a "$OUT/session.log"
    [ $rc = 0 ] || { tail -30 "$OUT/run$S.log"; die "run failed ($OUT/run$S.log)"; }
}

cmd=${1:-all}
case "$cmd" in
    build|check|crosscheck|capture|plan|run) "step_$cmd" ;;
    all) step_build; step_check; step_crosscheck; step_capture; step_plan; step_run ;;
    *) die "unknown step $cmd" ;;
esac
log "$cmd: done"
