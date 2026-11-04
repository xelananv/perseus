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
#
# Prefill (python/perseus/examples/rwkv7/PREFILL_DESIGN.md): T prompt tokens per pass on token lanes, then decode
# from the state the pass left in the registers, in ONE session (the decode capture and plan above are reused):
#   capture_pf  CPU only: run_capture --prefill-chunk $PF_T with the decode capture's pcmm_shifts.json -> $PFCAP
#   plan_pf     make_plan.sh $PFCAP -> $PFPLAN (same LANDING / PAIR / runtime switches as the decode plan)
#   pfcheck     the combined session's CPU half on the device build (Program.compile of a 1-block decode plan and a
#               T = 4 prefill plan; the carried state must land at the same (level, deg) in both): run it once
#   run_pf      run_planned --prefill-chunk $PF_T --pf-plan $PFPLAN on PF_TEXT: floor(n / T) passes, the remainder
#               and PF_GENERATE free-running steps by decode; every prefilled position scored like a decoded one
#               -> $OUT/run_pf$PF_T$S.jsonl ("kind": "prefill" lines: seconds per pass and per token)
#   pf          capture, plan, capture_pf, plan_pf, run_pf
#
# Levers from the fhe_ssm stack (PREFILL_DESIGN.md §6), opt-in, on their own capture-free plan dirs:
#   LEVERS=best   PAIR=1 (pairs of sibling real refreshes in one complex refresh) + PC_EM_FUSE_RESCALE=1 (the fused
#                 EvalMod) on the plans' runtime= line, and --schedule siblings on the run
#   CFTABLE=1     --pcmm-cf-table from the capture (fhe_ssm/pcmm_cf_table.py); A/B it against PCMM_CF first
#
# Environment: OUT (default /root/out_rwkv7), PTH, COEFFS, HARNESS (defaults: the brief's paths), TOKENS (12),
# BLOCKS (unset = all 12), GPU (0), TEXT (the prompt; must start with "\n\n", the public prefix), PCMM_SHIFT
# (calib | zero), PCMM_CF (6), TORCH_PY (crosscheck), RUN_EXTRA (more run_planned flags), STC_CACHE (160), POOL (96;
# run_pf 192), PF_T (32), PF_TEXT, PF_TOKENS (70), PF_GENERATE (4), LEVERS, CFTABLE.
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
LEVERS=${LEVERS:-}; CFTABLE=${CFTABLE:-}
case "$LEVERS" in
    "") PLANENV="LANDING=high"; SCHED=program ;;
    best) PLANENV="LANDING=high PAIR=1 RUNTIME_EXTRA=PC_EM_FUSE_RESCALE=1"; SCHED=siblings ;;
    *) echo "LEVERS=$LEVERS: unset | best" >&2; exit 2 ;;
esac
LV=${LEVERS:+_$LEVERS}
PLAN=$OUT/bootstrap_placements/plan_rwkv7$S$LV
PF_T=${PF_T:-32}; PF_TOKENS=${PF_TOKENS:-70}; PF_GENERATE=${PF_GENERATE:-4}
STC_CACHE=${STC_CACHE:-160}; POOL=${POOL:-96}
PF_TEXT=${PF_TEXT:-$'\n\nThe Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower from 1887 to 1889. Locally nicknamed the Iron Lady, it was constructed as the centrepiece of the 1889 World\'s Fair, and to crown the centennial anniversary of the French Revolution. Although initially criticised by some of France\'s leading artists and intellectuals for its design, it has since become a global cultural icon of France and one of the most recognisable structures in the world.'}
PFCAP=$OUT/cap_rwkv7_pf$PF_T$S
PFPLAN=$OUT/bootstrap_placements/plan_rwkv7_pf$PF_T$S$LV
CFT=$OUT/pcmm_cf_table$S.json
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

plan_one() {   # plan_one CAPTURE PLAN_DIR LOG
    local cap=$1 plan=$2 lg=$3
    [ -f "$plan/PLAN_CMD.txt" ] && { log "plan: reusing $plan"; return 0; }
    log "plan $cap -> $plan ($PLANENV)"
    env $PLANENV PLAN_ROOT="$OUT" PY="$PY" MEMGUARD_GB= bash "$W/tools/fhe_ssm/make_plan.sh" "$cap" "$(basename "$plan")" \
        > "$lg" 2>&1 || { tail -30 "$lg"; die "plan failed ($lg)"; }
    tail -8 "$lg" | tee -a "$OUT/session.log"
}

step_plan() { plan_one "$CAP" "$PLAN" "$OUT/plan$S$LV.log"; }

step_capture_pf() {
    [ -f "$PFCAP/capture_config.json" ] && { log "capture_pf: reusing $PFCAP"; return 0; }
    [ -f "$CAP/pcmm_shifts.json" ] || die "no $CAP/pcmm_shifts.json (run the capture step first: one shifts file serves both)"
    log "capture_pf T=$PF_T -> $PFCAP"
    pyrun env CUDA_VISIBLE_DEVICES="" $CAPENV "$PY" -m perseus.examples.rwkv7.run_capture $COMMON \
        --prefill-chunk "$PF_T" --shifts "$CAP/pcmm_shifts.json" --out "$PFCAP" > "$OUT/capture_pf$PF_T$S.log" 2>&1 \
        || { tail -20 "$OUT/capture_pf$PF_T$S.log"; die "capture_pf failed"; }
    tail -3 "$OUT/capture_pf$PF_T$S.log" | tee -a "$OUT/session.log"
}

step_plan_pf() { plan_one "$PFCAP" "$PFPLAN" "$OUT/plan_pf$PF_T$S$LV.log"; }

step_pfcheck() {
    pyrun env RWKV7_PTH="$PTH" RWKV7_COEFFS="$COEFFS" RWKV7_HARNESS="$HARNESS" $CAPENV \
        "$PY" -m pytest -q -s -p no:warnings "$W/python/perseus/examples/rwkv7/tests/test_prefill.py" \
        -k session_prepare --rwkv7-plan --rwkv7-pf-dir "$OUT/pfcheck" > "$OUT/pfcheck.log" 2>&1 \
        || { tail -30 "$OUT/pfcheck.log"; die "pfcheck failed ($OUT/pfcheck.log)"; }
    grep -E "decode kv regs|passed|skipped" "$OUT/pfcheck.log" | tee -a "$OUT/session.log"
    grep -q "1 passed" "$OUT/pfcheck.log" || die "pfcheck did not run (skipped?) — see $OUT/pfcheck.log"
}

cf_flags() {
    [ -n "$CFTABLE" ] || return 0
    [ -f "$CFT" ] || pyrun "$PY" -m perseus.examples.fhe_ssm.pcmm_cf_table "$CAP" "$CFT" >> "$OUT/session.log" 2>&1 \
        || die "pcmm_cf_table failed"
    echo "--pcmm-cf-table $CFT"
}

step_run_pf() {
    [ -f "$PLAN/PLAN_CMD.txt" ] || die "no plan $PLAN"
    [ -f "$PFPLAN/PLAN_CMD.txt" ] || die "no prefill plan $PFPLAN (run capture_pf and plan_pf)"
    local busy; busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$GPU" 2>/dev/null)
    [ -z "$busy" ] || log "WARNING: GPU $GPU has other processes ($busy): timings are not clean"
    local I; I=$(pyrun "$PY" -m perseus.examples.rwkv7.run_ids --harness "$HARNESS" --text "$PF_TEXT" --n $((PF_TOKENS + 1))) \
        || die "tokenizer failed"
    local R=$OUT/run_pf$PF_T$S$LV${CFTABLE:+_cft}
    log "run_pf: T=$PF_T, $PF_TOKENS prompt tokens after the prefix + $PF_GENERATE generated, ids $I"
    pyrun env CUDA_VISIBLE_DEVICES=$GPU $CAPENV "$PY" -m perseus.examples.rwkv7.run_planned $COMMON --ids "$I" \
        --tokens "$PF_TOKENS" --generate "$PF_GENERATE" --plan "$PLAN" --shifts "$CAP/pcmm_shifts.json" \
        --prefill-chunk "$PF_T" --pf-plan "$PFPLAN" --schedule "$SCHED" $(cf_flags) --pcmm-cf "$PCMM_CF" \
        --stc-cache "$STC_CACHE" --pool "${POOL_PF:-192}" --out "$R.jsonl" ${RUN_EXTRA:-} > "$R.log" 2>&1
    local rc=$?
    grep -E '"kind": "(prefill|warmup)"' "$R.jsonl" 2>/dev/null | tee -a "$OUT/session.log"
    grep '"kind": "summary"' "$R.jsonl" 2>/dev/null | cut -c1-900 | tee -a "$OUT/session.log"
    [ $rc = 0 ] || { tail -30 "$R.log"; die "run_pf failed ($R.log)"; }
}

step_run() {
    [ -f "$PLAN/PLAN_CMD.txt" ] || die "no plan $PLAN (run the plan step)"
    local busy; busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$GPU" 2>/dev/null)
    [ -z "$busy" ] || log "WARNING: GPU $GPU has other processes ($busy): timings are not clean"
    local I; I=$(ids) || die "tokenizer failed"
    log "run: $TOKENS tokens after the prefix, ids $I"
    local R=$OUT/run$S$LV${CFTABLE:+_cft}
    pyrun env CUDA_VISIBLE_DEVICES=$GPU $CAPENV "$PY" -m perseus.examples.rwkv7.run_planned $COMMON --ids "$I" \
        --tokens "$TOKENS" --plan "$PLAN" --shifts "$CAP/pcmm_shifts.json" --pcmm-cf "$PCMM_CF" \
        --schedule "$SCHED" $(cf_flags) --stc-cache "$STC_CACHE" --pool "$POOL" \
        --out "$R.jsonl" ${RUN_EXTRA:-} > "$R.log" 2>&1
    local rc=$?
    grep '"kind": "token"' "$R.jsonl" 2>/dev/null | cut -c1-220 | tee -a "$OUT/session.log"
    grep '"kind": "summary"' "$R.jsonl" 2>/dev/null | tee -a "$OUT/session.log"
    [ $rc = 0 ] || { tail -30 "$R.log"; die "run failed ($R.log)"; }
}

cmd=${1:-all}
case "$cmd" in
    build|check|crosscheck|capture|plan|run|capture_pf|plan_pf|pfcheck|run_pf) "step_$cmd" ;;
    all) step_build; step_check; step_crosscheck; step_capture; step_plan; step_run ;;
    pf) step_capture; step_plan; step_capture_pf; step_plan_pf; step_run_pf ;;
    *) die "unknown step $cmd" ;;
esac
log "$cmd: done"
