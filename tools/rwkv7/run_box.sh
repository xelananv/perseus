#!/bin/bash
# run_box.sh — RWKV-7 (any RWKV-x070-World size) encrypted decode on the GPU box (checkout /root/perseus-rwkv7).
# tools/rwkv7/size_pipeline.sh drives it for a new size (calibrate, fit, scan, gate, then these steps).
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
# Batched decode (python/perseus/examples/rwkv7/BATCHED_DESIGN.md): B_K conversations per pass, conversation c on
# lane c, each with its own carried state; the decode capture's shifts file is reused:
#   capture_b   CPU only: run_capture --batch $B_K --batch-carry $B_CARRY -> $BCAP
#   plan_b      make_plan.sh $BCAP -> $BPLAN (same LANDING / PAIR / runtime switches as the decode plan)
#   bcheck      the batched runner's CPU half on the device build (Program.compile of a K = 4, 1-block batched plan;
#               every conversation's state must be recognised as block inputs): run it once
#   batch       run_planned --batch $B_K --prompts $B_PROMPTS --generate $B_GENERATE: every pass decodes one token
#               of every conversation; one "pass" line per pass (seconds) and one "token" line per conversation and
#               position (top-1 vs that conversation's poly model) -> $OUT/batch$B_K$S.jsonl, summary line last
#   b           capture, plan, capture_b, plan_b, batch
#
# Continuous batching (python/perseus/examples/rwkv7/SERVE_DESIGN.md): conversations arrive at batched passes
# (S_FILE rows "<pass><TAB><prompt>"), are read by prefill (PF_T) and decode steps while the batch waits, join a free
# lane of the B_K pass, generate S_GENERATE tokens there and leave (the lane back to the public prefix state). ONE
# session holds the decode, prefill, batched and join programs; the decode, prefill and batched captures above are reused:
#   capture_j   CPU only: run_capture --join --batch $B_K (every block's carries; no weights) -> $JCAP
#   plan_j      make_plan.sh $JCAP -> $JPLAN (one deliberate refresh per carry, nothing placed)
#   scheck      the serve session's CPU half on the device build (Program.compile of 1-block decode, T = 4 prefill,
#               K = 4 batched and join plans; every hand-over must land where its reader reads): run it once
#   serve       run_planned --serve $S_FILE: "prefill" / "decode" / "join" / "leave" / "pass" lines with seconds, one
#               "token" line per conversation and position (top-1 vs its own poly model, hidden vs the mirror run on
#               the same schedule) -> $OUT/serve$B_K$S.jsonl, summary line last
#   s           capture, plan, capture_pf, plan_pf, capture_b, plan_b, capture_j, plan_j, serve
#   Start with B_K=4 (and BLOCKS=1 for the smoke): a K = 32 batched pass alone peaked at 94.9 GB of the 97.9 GB card,
#   and this session also holds the decode and prefill programs.
#
# Levers from the fhe_ssm stack (PREFILL_DESIGN.md §6), opt-in, on their own capture-free plan dirs:
#   LEVERS=best   PAIR=1 (pairs of sibling real refreshes in one complex refresh) + PC_EM_FUSE_RESCALE=1 (the fused
#                 EvalMod) on the plans' runtime= line, and --schedule siblings on the run
#   CFTABLE=1     --pcmm-cf-table from the capture (fhe_ssm/pcmm_cf_table.py); A/B it against PCMM_CF first
#   PLANOPT=b2    the batched plan (plan_b, and the batch / serve steps that read it) without planned refreshes inside
#                 any one conversation's own state work (tools/rwkv7/planopt.sh; PLAN_B2.md): the cut covers it from
#                 shared values instead. Own plan dir and run records (suffix _b2); decode, prefill and join plans
#                 are not affected (they have none of those steps). Combines with LEVERS=best
#
# Keys (run_planned --keys; the 11-02 fhe_ssm publication path): KEYS=inproc (default: the session generates every key
# and holds the secret; recorded insecure-inproc) | bundle (the client's bundle $OUT/keys_<mode>$S$LV, made below) | DIR.
#   keyspec     run_planned --keyspec for KEYMODE (decode | pf | b): the session's key request for exactly the plans
#               that run -> $OUT/keys_<mode>$S$LV/keyspec.json (CPU half + Session.key_manifest; device build)
#   keygen      the client role: run_planned --keygen DIR --keyspec-file DIR/keyspec.json -> DIR/keys.pckb and
#               DIR/client_secret.bin (a fresh secret; the session never sees it); then KEYS=bundle for run / run_pf /
#               batch: the session opens with Session.from_bundle (has_secret false in the header), the client object
#               encrypts every input and decrypts every reply as wire frames
#
# Any size (ANYSIZE_DESIGN.md): every shape is read from PTH. LANES (0 = one ciphertext per vector: t = 32 up to width
# 1,024, 16 up to 2,048, 8 up to 4,096 | 32 = channel shards of 1,024 | 16 | 8), LORA (auto | fold | split), SHARD_LAND
# (1 | 0: --no-shard-land) pass through to every step; B_K and PF_T must be <= the lanes.
#
# Environment: OUT (default /root/out_rwkv7), PTH, COEFFS, HARNESS (defaults: the brief's paths), TOKENS (12),
# BLOCKS (unset = all 12), GPU (0), TEXT (the prompt; must start with "\n\n", the public prefix), PCMM_SHIFT
# (calib | zero), PCMM_CF (6), TORCH_PY (crosscheck), RUN_EXTRA (more run_planned flags), RUN_TAG (suffix of the run's
# record names: run_<tag>.jsonl), STC_CACHE (160), POOL (96;
# run_pf 192), PF_T (32), PF_TEXT, PF_TOKENS (70), PF_GENERATE (4), LEVERS, CFTABLE, PLANOPT (base | b2), B_K (32), B_CARRY (lane |
# decode), B_PROMPTS (tools/rwkv7/batch_prompts.txt), B_GENERATE (16), POOL_B (192), STC_CACHE_B ($STC_CACHE),
# S_FILE (tools/rwkv7/serve_arrivals.txt), S_GENERATE (8), POOL_S (256), STC_CACHE_S ($STC_CACHE).
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
PLANOPT=${PLANOPT:-}
# shellcheck source=tools/rwkv7/planopt.sh
. "$HERE/planopt.sh"
planopt_env "$PLANOPT" || exit 2
BPLANENV="${PLANOPT_ENV[*]:-}"
PO=${PLANOPT:+_$PLANOPT}; PO=${PO/_base/}
PLAN=$OUT/bootstrap_placements/plan_rwkv7$S$LV
PF_T=${PF_T:-32}; PF_TOKENS=${PF_TOKENS:-70}; PF_GENERATE=${PF_GENERATE:-4}
STC_CACHE=${STC_CACHE:-160}; POOL=${POOL:-96}
PF_TEXT=${PF_TEXT:-$'\n\nThe Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower from 1887 to 1889. Locally nicknamed the Iron Lady, it was constructed as the centrepiece of the 1889 World\'s Fair, and to crown the centennial anniversary of the French Revolution. Although initially criticised by some of France\'s leading artists and intellectuals for its design, it has since become a global cultural icon of France and one of the most recognisable structures in the world.'}
PFCAP=$OUT/cap_rwkv7_pf$PF_T$S
PFPLAN=$OUT/bootstrap_placements/plan_rwkv7_pf$PF_T$S$LV
CFT=$OUT/pcmm_cf_table$S.json
B_K=${B_K:-32}; B_CARRY=${B_CARRY:-lane}; B_PROMPTS=${B_PROMPTS:-$HERE/batch_prompts.txt}; B_GENERATE=${B_GENERATE:-16}
BCAP=$OUT/cap_rwkv7_b$B_K${B_CARRY/lane/}$S
BPLAN=$OUT/bootstrap_placements/plan_rwkv7_b$B_K${B_CARRY/lane/}$S$LV$PO
S_FILE=${S_FILE:-$HERE/serve_arrivals.txt}; S_GENERATE=${S_GENERATE:-8}
JCAP=$OUT/cap_rwkv7_join$B_K$S
JPLAN=$OUT/bootstrap_placements/plan_rwkv7_join$B_K$S$LV
# the LANDING=high runtime switches (tools/fhe_ssm/run_gpu.sh CAPENV): PCMM needs the StC-first dense route
CAPENV="FIDESLIB_BTS_STC_FIRST=2 FIDESLIB_BTS_STC_FIRST_ROUTES=dense FIDESLIB_BTS_STC_FIRST_CF=13 FIDESLIB_BTS_RAISE_DROP=512:1 AUTO_BTS_LEVEL=44"
LORA=${LORA:-auto}; SHARD_LAND=${SHARD_LAND:-1}; KEYS=${KEYS:-inproc}; KEYMODE=${KEYMODE:-decode}; LANES=${LANES:-0}
COMMON="--pth $PTH --coeffs $COEFFS --harness $HARNESS ${BLOCKS:+--blocks $BLOCKS} --lora $LORA --lanes $LANES"
[ "$SHARD_LAND" = 1 ] || COMMON="$COMMON --no-shard-land"
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

plan_one() {   # plan_one CAPTURE PLAN_DIR LOG [EXTRA_ENV]
    local cap=$1 plan=$2 lg=$3 extra=${4:-}
    [ -f "$plan/PLAN_CMD.txt" ] && { log "plan: reusing $plan"; return 0; }
    log "plan $cap -> $plan ($PLANENV${extra:+ $extra})"
    env $PLANENV $extra PLAN_ROOT="$OUT" PY="$PY" MEMGUARD_GB= bash "$W/tools/fhe_ssm/make_plan.sh" "$cap" "$(basename "$plan")" \
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

keydir() {   # keydir decode|pf|b|s (the b and s bundles are made from the PLANOPT batched plan: their own dir)
    case "$1" in b|s) echo "$OUT/keys_$1$S$LV$PO" ;; *) echo "$OUT/keys_$1$S$LV" ;; esac
}

keys_flag() {   # keys_flag decode|pf|b (need_keys has checked it in the calling shell)
    case "$KEYS" in
        inproc) echo "--keys inproc" ;;
        bundle) echo "--keys $(keydir "$1")" ;;
        *) echo "--keys $KEYS" ;;
    esac
}

need_keys() {   # stops here (not inside a $(...)) when the bundle is missing
    case "$KEYS" in
        inproc) log "keys: inproc (INSECURE: the session holds the secret; recorded insecure-inproc)" ;;
        bundle) [ -f "$(keydir "$1")/keys.pckb" ] || die "KEYS=bundle: no $(keydir "$1")/keys.pckb (KEYMODE=$1 keyspec, keygen)" ;;
        *) [ -f "$KEYS/keys.pckb" ] || die "KEYS=$KEYS: no keys.pckb there" ;;
    esac
}

mode_flags() {   # the run_planned flags of a key mode (the same plans the run opens)
    case "$1" in
        decode) echo "--plan $PLAN --shifts $CAP/pcmm_shifts.json --ids $(ids)" ;;
        pf) echo "--plan $PLAN --pf-plan $PFPLAN --prefill-chunk $PF_T --shifts $CAP/pcmm_shifts.json --ids $(ids)" ;;
        b) echo "--plan $BPLAN --batch $B_K --batch-carry $B_CARRY --prompts $B_PROMPTS --shifts $CAP/pcmm_shifts.json" ;;
        s) echo "--serve $S_FILE --plan $PLAN --pf-plan $PFPLAN --prefill-chunk $PF_T --b-plan $BPLAN --join-plan $JPLAN --batch $B_K --shifts $CAP/pcmm_shifts.json" ;;
        *) die "KEYMODE=$1: decode | pf | b | s" ;;
    esac
}

step_keyspec() {
    local d; d=$(keydir "$KEYMODE"); mkdir -p "$d"
    log "keyspec $KEYMODE -> $d/keyspec.json"
    pyrun env CUDA_VISIBLE_DEVICES=$GPU $CAPENV "$PY" -m perseus.examples.rwkv7.run_planned $COMMON $(mode_flags "$KEYMODE") \
        --schedule "$SCHED" --pcmm-cf "$PCMM_CF" --stc-cache "$STC_CACHE" --pool "$POOL" --keyspec "$d/keyspec.json" \
        --out "$d/keyspec_run.jsonl" > "$d/keyspec.log" 2>&1 || { tail -20 "$d/keyspec.log"; die "keyspec failed ($d/keyspec.log)"; }
    tail -2 "$d/keyspec.log" | tee -a "$OUT/session.log"
}

step_keygen() {
    local d; d=$(keydir "$KEYMODE")
    [ -f "$d/keyspec.json" ] || die "no $d/keyspec.json (run keyspec first, KEYMODE=$KEYMODE)"
    log "keygen $KEYMODE (the client role) -> $d/keys.pckb"
    pyrun env CUDA_VISIBLE_DEVICES=$GPU "$PY" -m perseus.examples.rwkv7.run_planned --keygen "$d" --keyspec-file "$d/keyspec.json" \
        --threads "${KEYGEN_THREADS:-8}" > "$d/keygen.log" 2>&1 || { tail -20 "$d/keygen.log"; die "keygen failed ($d/keygen.log)"; }
    tail -1 "$d/keygen.log" | cut -c1-400 | tee -a "$OUT/session.log"
}

step_run_pf() {
    need_keys pf
    [ -f "$PLAN/PLAN_CMD.txt" ] || die "no plan $PLAN"
    [ -f "$PFPLAN/PLAN_CMD.txt" ] || die "no prefill plan $PFPLAN (run capture_pf and plan_pf)"
    local busy; busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$GPU" 2>/dev/null)
    [ -z "$busy" ] || log "WARNING: GPU $GPU has other processes ($busy): timings are not clean"
    local I; I=$(pyrun "$PY" -m perseus.examples.rwkv7.run_ids --harness "$HARNESS" --text "$PF_TEXT" --n $((PF_TOKENS + 1))) \
        || die "tokenizer failed"
    local R=$OUT/run_pf$PF_T$S$LV${CFTABLE:+_cft}${RUN_TAG:+_$RUN_TAG}
    log "run_pf: T=$PF_T, $PF_TOKENS prompt tokens after the prefix + $PF_GENERATE generated, ids $I"
    pyrun env CUDA_VISIBLE_DEVICES=$GPU $CAPENV "$PY" -m perseus.examples.rwkv7.run_planned $COMMON --ids "$I" \
        --tokens "$PF_TOKENS" --generate "$PF_GENERATE" --plan "$PLAN" --shifts "$CAP/pcmm_shifts.json" \
        --prefill-chunk "$PF_T" --pf-plan "$PFPLAN" --schedule "$SCHED" $(cf_flags) --pcmm-cf "$PCMM_CF" \
        --stc-cache "$STC_CACHE" --pool "${POOL_PF:-192}" $(keys_flag pf) --out "$R.jsonl" ${RUN_EXTRA:-} > "$R.log" 2>&1
    local rc=$?
    grep -E '"kind": "(prefill|warmup)"' "$R.jsonl" 2>/dev/null | tee -a "$OUT/session.log"
    grep '"kind": "summary"' "$R.jsonl" 2>/dev/null | cut -c1-900 | tee -a "$OUT/session.log"
    [ $rc = 0 ] || { tail -30 "$R.log"; die "run_pf failed ($R.log)"; }
}

step_capture_b() {
    [ -f "$BCAP/capture_config.json" ] && { log "capture_b: reusing $BCAP"; return 0; }
    [ -f "$CAP/pcmm_shifts.json" ] || die "no $CAP/pcmm_shifts.json (run the capture step first: one shifts file serves both)"
    log "capture_b K=$B_K carry=$B_CARRY -> $BCAP"
    pyrun env CUDA_VISIBLE_DEVICES="" $CAPENV "$PY" -m perseus.examples.rwkv7.run_capture $COMMON \
        --batch "$B_K" --batch-carry "$B_CARRY" --shifts "$CAP/pcmm_shifts.json" --out "$BCAP" \
        > "$OUT/capture_b$B_K$S.log" 2>&1 || { tail -20 "$OUT/capture_b$B_K$S.log"; die "capture_b failed"; }
    tail -3 "$OUT/capture_b$B_K$S.log" | tee -a "$OUT/session.log"
}

step_plan_b() { plan_one "$BCAP" "$BPLAN" "$OUT/plan_b$B_K$S$LV$PO.log" "$BPLANENV"; }

step_bcheck() {
    pyrun env RWKV7_PTH="$PTH" RWKV7_COEFFS="$COEFFS" RWKV7_HARNESS="$HARNESS" $CAPENV \
        "$PY" -m pytest -q -s -p no:warnings "$W/python/perseus/examples/rwkv7/tests/test_batched.py" \
        -k session_prepare --rwkv7-plan --rwkv7-b-dir "$OUT/bcheck" > "$OUT/bcheck.log" 2>&1 \
        || { tail -30 "$OUT/bcheck.log"; die "bcheck failed ($OUT/bcheck.log)"; }
    grep -E "batched kv|passed|skipped" "$OUT/bcheck.log" | tee -a "$OUT/session.log"
    grep -q "1 passed" "$OUT/bcheck.log" || die "bcheck did not run (skipped?) — see $OUT/bcheck.log"
}

step_batch() {
    need_keys b
    [ -f "$BPLAN/PLAN_CMD.txt" ] || die "no batched plan $BPLAN (run capture_b and plan_b)"
    [ -f "$B_PROMPTS" ] || die "no prompts file $B_PROMPTS"
    local busy; busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$GPU" 2>/dev/null)
    [ -z "$busy" ] || log "WARNING: GPU $GPU has other processes ($busy): timings are not clean"
    local R=$OUT/batch$B_K${B_CARRY/lane/}$S$LV$PO${CFTABLE:+_cft}${RUN_TAG:+_$RUN_TAG}
    log "batch: K=$B_K carry=$B_CARRY, prompts $B_PROMPTS, $B_GENERATE generated each, pool ${POOL_B:-192}"
    pyrun env CUDA_VISIBLE_DEVICES=$GPU $CAPENV "$PY" -m perseus.examples.rwkv7.run_planned $COMMON \
        --batch "$B_K" --batch-carry "$B_CARRY" --prompts "$B_PROMPTS" --generate "$B_GENERATE" \
        --plan "$BPLAN" --shifts "$CAP/pcmm_shifts.json" --pcmm-cf "$PCMM_CF" --schedule "$SCHED" $(cf_flags) \
        --stc-cache "${STC_CACHE_B:-$STC_CACHE}" --pool "${POOL_B:-192}" $(keys_flag b) --out "$R.jsonl" ${RUN_EXTRA:-} > "$R.log" 2>&1
    local rc=$?
    grep -E '"kind": "(pass|warmup)"' "$R.jsonl" 2>/dev/null | tee -a "$OUT/session.log"
    grep '"kind": "summary"' "$R.jsonl" 2>/dev/null | cut -c1-1200 | tee -a "$OUT/session.log"
    [ $rc = 0 ] || { tail -30 "$R.log"; die "batch failed ($R.log)"; }
}

step_capture_j() {
    [ -f "$JCAP/capture_config.json" ] && { log "capture_j: reusing $JCAP"; return 0; }
    log "capture_j K=$B_K -> $JCAP"
    pyrun env CUDA_VISIBLE_DEVICES="" $CAPENV "$PY" -m perseus.examples.rwkv7.run_capture $COMMON \
        --join --batch "$B_K" --out "$JCAP" > "$OUT/capture_j$B_K$S.log" 2>&1 \
        || { tail -20 "$OUT/capture_j$B_K$S.log"; die "capture_j failed"; }
    tail -2 "$OUT/capture_j$B_K$S.log" | tee -a "$OUT/session.log"
}

step_plan_j() { plan_one "$JCAP" "$JPLAN" "$OUT/plan_j$B_K$S$LV.log"; }

step_scheck() {
    pyrun env RWKV7_PTH="$PTH" RWKV7_COEFFS="$COEFFS" RWKV7_HARNESS="$HARNESS" $CAPENV \
        "$PY" -m pytest -q -s -p no:warnings "$W/python/perseus/examples/rwkv7/tests/test_serve_session.py" \
        -k serve_session_prepare --rwkv7-plan --rwkv7-b-dir "$OUT/scheck" > "$OUT/scheck.log" 2>&1 \
        || { tail -30 "$OUT/scheck.log"; die "scheck failed ($OUT/scheck.log)"; }
    grep -E "serve registers|passed|skipped" "$OUT/scheck.log" | tee -a "$OUT/session.log"
    grep -q "1 passed" "$OUT/scheck.log" || die "scheck did not run (skipped?) — see $OUT/scheck.log"
}

step_serve() {
    need_keys s
    [ "$B_CARRY" = lane ] || die "serve joins lane-form carries: B_CARRY=lane"
    for f in "$PLAN" "$PFPLAN" "$BPLAN" "$JPLAN"; do
        [ -f "$f/PLAN_CMD.txt" ] || die "no plan $f (serve needs plan, plan_pf, plan_b, plan_j)"
    done
    [ -f "$S_FILE" ] || die "no arrivals file $S_FILE"
    local busy; busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$GPU" 2>/dev/null)
    [ -z "$busy" ] || log "WARNING: GPU $GPU has other processes ($busy): timings are not clean"
    local R=$OUT/serve$B_K$S$LV$PO${CFTABLE:+_cft}${RUN_TAG:+_$RUN_TAG}
    log "serve: K=$B_K T=$PF_T, arrivals $S_FILE, $S_GENERATE generated each, pool ${POOL_S:-256}"
    pyrun env CUDA_VISIBLE_DEVICES=$GPU $CAPENV "$PY" -m perseus.examples.rwkv7.run_planned $COMMON \
        $(mode_flags s) --generate "$S_GENERATE" --pcmm-cf "$PCMM_CF" --schedule "$SCHED" $(cf_flags) \
        --stc-cache "${STC_CACHE_S:-$STC_CACHE}" --pool "${POOL_S:-256}" $(keys_flag s) --out "$R.jsonl" ${RUN_EXTRA:-} \
        > "$R.log" 2>&1
    local rc=$?
    grep -E '"kind": "(prefill|join|leave|pass|warmup)"' "$R.jsonl" 2>/dev/null | cut -c1-200 | tee -a "$OUT/session.log"
    grep '"kind": "summary"' "$R.jsonl" 2>/dev/null | cut -c1-1500 | tee -a "$OUT/session.log"
    [ $rc = 0 ] || { tail -30 "$R.log"; die "serve failed ($R.log)"; }
}

step_run() {
    need_keys decode
    [ -f "$PLAN/PLAN_CMD.txt" ] || die "no plan $PLAN (run the plan step)"
    local busy; busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$GPU" 2>/dev/null)
    [ -z "$busy" ] || log "WARNING: GPU $GPU has other processes ($busy): timings are not clean"
    local I; I=$(ids) || die "tokenizer failed"
    log "run: $TOKENS tokens after the prefix, ids $I"
    local R=$OUT/run$S$LV${CFTABLE:+_cft}${RUN_TAG:+_$RUN_TAG}
    pyrun env CUDA_VISIBLE_DEVICES=$GPU $CAPENV "$PY" -m perseus.examples.rwkv7.run_planned $COMMON --ids "$I" \
        --tokens "$TOKENS" --plan "$PLAN" --shifts "$CAP/pcmm_shifts.json" --pcmm-cf "$PCMM_CF" \
        --schedule "$SCHED" $(cf_flags) --stc-cache "$STC_CACHE" --pool "$POOL" $(keys_flag decode) \
        --out "$R.jsonl" ${RUN_EXTRA:-} > "$R.log" 2>&1
    local rc=$?
    grep '"kind": "token"' "$R.jsonl" 2>/dev/null | cut -c1-220 | tee -a "$OUT/session.log"
    grep '"kind": "summary"' "$R.jsonl" 2>/dev/null | tee -a "$OUT/session.log"
    [ $rc = 0 ] || { tail -30 "$R.log"; die "run failed ($R.log)"; }
}

cmd=${1:-all}
case "$cmd" in
    build|check|crosscheck|capture|plan|run|capture_pf|plan_pf|pfcheck|run_pf|capture_b|plan_b|bcheck|batch|keyspec|keygen) "step_$cmd" ;;
    capture_j|plan_j|scheck|serve) "step_$cmd" ;;
    all) step_build; step_check; step_crosscheck; step_capture; step_plan; step_run ;;
    pf) step_capture; step_plan; step_capture_pf; step_plan_pf; step_run_pf ;;
    b) step_capture; step_plan; step_capture_b; step_plan_b; step_batch ;;
    s) step_capture; step_plan; step_capture_pf; step_plan_pf; step_capture_b; step_plan_b; step_capture_j; step_plan_j
       step_serve ;;
    *) die "unknown step $cmd" ;;
esac
log "$cmd: done"
