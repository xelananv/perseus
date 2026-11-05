#!/bin/bash
# run_box.sh — RWKV-7 (any RWKV-x070-World size) encrypted decode on the GPU box (checkout /root/perseus-rwkv7).
# tools/rwkv7/size_pipeline.sh drives it for a new size (calibrate, fit, scan, gate, then these steps).
#
#   KEYS=inproc bash tools/rwkv7/run_box.sh all            build, check, crosscheck, capture, plan, run (BLOCKS unset = all)
#   KEYS=inproc BLOCKS=1 bash tools/rwkv7/run_box.sh all   one block + the tail (the smoke: same steps, own directories)
#   bash tools/rwkv7/run_box.sh STEP [STEP ...]           steps in order: build | check | crosscheck | capture | plan | run ...
#   bash tools/rwkv7/run_box.sh names                     every product name of these settings, KEY=VALUE (no side effect)
#
# Steps (every product under $OUT; a step reuses what an earlier one left):
#   build       tools/fhe_ssm/run_gpu.sh build (uv venv, pip install -e .[hf,examples,dev], maturin develop --release)
#   check       the device classes of perseus._pc, the planner's imports (networkx, scipy), the CPU tests of the port
#               (on PTH when it is the 768-wide 0.1B, else on CHECK_PTH / CHECK_COEFFS, the 0.1B: the suite is written
#               against it), and PTH's width-agnostic checks (shapes, lanes, preset, a coefficient site per block)
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
#                 shared values instead. Own plan dir, key bundles and run records (suffix _pob2); decode, prefill and join plans
#                 are not affected (they have none of those steps). Combines with LEVERS=best
#
# Keys (run_planned --keys; the 11-02 fhe_ssm publication path), REQUIRED for run / run_pf / batch / serve (no default,
# as fhe_ssm's run_gpu.sh): KEYS=inproc (the session generates every key and holds the secret; recorded insecure-inproc)
# | bundle (the client's bundle, `run_box.sh names` prints its KEYDIR_<mode>, made below) | DIR.
#   keyspec     run_planned --keyspec for KEYMODE (decode | pf | b): the session's key request for exactly the plans
#               that run -> KEYDIR_<mode>/keyspec.json (CPU half + Session.key_manifest; device build)
#   keygen      the client role: run_planned --keygen DIR --keyspec-file DIR/keyspec.json -> DIR/keys.pckb and
#               DIR/client_secret.bin (a fresh secret, mode 600; the session object never sees it; client and session are
#               objects in ONE process, not separate processes); then KEYS=bundle for run / run_pf /
#               batch: the session opens with Session.from_bundle (has_secret false in the header), the client object
#               encrypts every input and decrypts every reply as wire frames
#
# State layout (FILLED_DESIGN.md): LAYOUT=current (default) | filled (two value groups of the WKV state per ciphertext,
# one on each axis: half the state ciphertexts, carry refreshes, state products and rotations). Every capture, plan,
# key bundle and record of a filled run carries the suffix _filled (pfcheck, bcheck and scheck pass RWKV7_LAYOUT); combines with LEVERS, PLANOPT, B_K, PF_T.
#
# Any size (ANYSIZE_DESIGN.md): every shape is read from PTH. LANES (0 = one ciphertext per vector: t = 32 up to width
# 1,024, 16 up to 2,048, 8 up to 4,096 | 32 = channel shards of 1,024 | 16 | 8), LORA (auto | fold | split), SHARD_LAND
# (1 | 0: --no-shard-land) pass through to every step and name; B_K and PF_T must be <= the lanes; B_K >= 2 for the
# batched steps. Before a capture or plan is reused, and before every run step, tools/rwkv7/capture_check.py compares
# its rwkv7_capture.json (pth / coeffs sha256, lanes, layout, blocks, batch, T, lora, shard_land) and the plan's graph
# md5 with these settings and stops on a mismatch (CAPTURE_MISMATCH_OK=1 overrides, printed).
#
# Environment: OUT (default /root/out_rwkv7), PTH, COEFFS, HARNESS (defaults: the brief's paths), TOKENS (12),
# BLOCKS (unset = all 12), GPU (0), TEXT (the prompt; must start with "\n\n", the public prefix), PCMM_SHIFT
# (calib | zero), PCMM_CF (6), TORCH_PY (crosscheck), RUN_EXTRA (more run_planned flags), RUN_TAG (suffix of the run's
# record names: run..._<tag>.jsonl), CHECK_PTH / CHECK_COEFFS (the 0.1B for the check step's suite), STC_CACHE (160), POOL (96;
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
LAYOUT=${LAYOUT:-current}; LORA=${LORA:-auto}; SHARD_LAND=${SHARD_LAND:-1}; LANES=${LANES:-0}
case "$LAYOUT" in current|filled) ;; *) echo "LAYOUT=$LAYOUT: current | filled" >&2; exit 2 ;; esac
case "$PCMM_SHIFT" in calib|zero) ;; *) echo "PCMM_SHIFT=$PCMM_SHIFT: calib | zero" >&2; exit 2 ;; esac
case "$LORA" in auto|fold|split) ;; *) echo "LORA=$LORA: auto | fold | split" >&2; exit 2 ;; esac
case "$SHARD_LAND" in 0|1) ;; *) echo "SHARD_LAND=$SHARD_LAND: 1 | 0" >&2; exit 2 ;; esac
[[ "$LANES" =~ ^(0|8|16|32)$ ]] || { echo "LANES=$LANES: 0 | 32 | 16 | 8" >&2; exit 2; }
[[ -z "$BLOCKS" || "$BLOCKS" =~ ^[0-9]+$ ]] || { echo "BLOCKS=$BLOCKS: a number or unset" >&2; exit 2; }
# Names (review lens 2, finding 1): every setting that changes a capture, plan, key bundle or record is a token with its
# own prefix, in a fixed order, so two configurations never share a directory: _blk<BLOCKS> _zero _l<LANES, 0 = none>
# _lora<fold|split> _nsl (SHARD_LAND=0) _filled  [S]; _k<B_K>[_c<B_CARRY>] (batched), _pf<PF_T> (prefill), _join_k<B_K>;
# _best (LEVERS) _po<PLANOPT> _cft (CFTABLE). PTH and COEFFS are not in the names: capture_check.py compares their sha256
# (and lanes, layout, blocks, batch, T, lora, shard_land) with every capture and plan before it is reused or run.
S=${BLOCKS:+_blk$BLOCKS}
[ "$PCMM_SHIFT" = zero ] && S="${S}_zero"
[ "$LANES" = 0 ] || S="${S}_l$LANES"
[ "$LORA" = auto ] || S="${S}_lora$LORA"
[ "$SHARD_LAND" = 1 ] || S="${S}_nsl"
[ "$LAYOUT" = filled ] && S="${S}_filled"
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
PO=; case "$PLANOPT" in ""|base) ;; *) PO=_po$PLANOPT ;; esac
PLAN=$OUT/bootstrap_placements/plan_rwkv7$S$LV
PF_T=${PF_T:-32}; PF=_pf$PF_T; PF_TOKENS=${PF_TOKENS:-70}; PF_GENERATE=${PF_GENERATE:-4}
STC_CACHE=${STC_CACHE:-160}; POOL=${POOL:-96}
PF_TEXT=${PF_TEXT:-$'\n\nThe Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower from 1887 to 1889. Locally nicknamed the Iron Lady, it was constructed as the centrepiece of the 1889 World\'s Fair, and to crown the centennial anniversary of the French Revolution. Although initially criticised by some of France\'s leading artists and intellectuals for its design, it has since become a global cultural icon of France and one of the most recognisable structures in the world.'}
PFCAP=$OUT/cap_rwkv7$PF$S
PFPLAN=$OUT/bootstrap_placements/plan_rwkv7$PF$S$LV
CFT=$OUT/pcmm_cf_table$S.json
B_K=${B_K:-32}; B_CARRY=${B_CARRY:-lane}; B_PROMPTS=${B_PROMPTS:-$HERE/batch_prompts.txt}; B_GENERATE=${B_GENERATE:-16}
[[ "$B_K" =~ ^[0-9]+$ && "$PF_T" =~ ^[0-9]+$ ]] || { echo "B_K=$B_K PF_T=$PF_T: numbers" >&2; exit 2; }
case "$B_CARRY" in lane) BC=_k$B_K ;; decode) BC=_k${B_K}_c$B_CARRY ;; *) echo "B_CARRY=$B_CARRY: lane | decode" >&2; exit 2 ;; esac
BCAP=$OUT/cap_rwkv7$BC$S
BPLAN=$OUT/bootstrap_placements/plan_rwkv7$BC$S$LV$PO
S_FILE=${S_FILE:-$HERE/serve_arrivals.txt}; S_GENERATE=${S_GENERATE:-8}
JCAP=$OUT/cap_rwkv7_join_k$B_K$S
JPLAN=$OUT/bootstrap_placements/plan_rwkv7_join_k$B_K$S$LV
CFS=${CFTABLE:+_cft}
PCF=; [ "$PCMM_CF" = 6 ] || PCF=_cf$PCMM_CF                  # a non-default PCMM CF changes the keys and the records
# the run records and the key bundles (a key bundle is made for exactly the plans its mode opens: every name above)
R_RUN=$OUT/run$S$LV$PCF$CFS; R_PF=$OUT/run$PF$S$LV$PCF$CFS; R_BATCH=$OUT/batch$BC$S$LV$PO$PCF$CFS
R_SERVE=$OUT/serve$BC$PF$S$LV$PO$PCF$CFS
keydir() {   # keydir decode|pf|b|s
    case "$1" in
        decode) echo "$OUT/keys_decode$S$LV$PCF$CFS" ;;
        pf) echo "$OUT/keys$PF$S$LV$PCF$CFS" ;;
        b) echo "$OUT/keys_b$BC$S$LV$PO$PCF$CFS" ;;
        s) echo "$OUT/keys_s$BC$PF$S$LV$PO$PCF$CFS" ;;
        *) echo "$OUT/keys_$1_unknown" ;;
    esac
}
# the LANDING=high runtime switches (tools/fhe_ssm/run_gpu.sh CAPENV): PCMM needs the StC-first dense route
CAPENV="FIDESLIB_BTS_STC_FIRST=2 FIDESLIB_BTS_STC_FIRST_ROUTES=dense FIDESLIB_BTS_STC_FIRST_CF=13 FIDESLIB_BTS_RAISE_DROP=512:1 AUTO_BTS_LEVEL=44"
KEYS=${KEYS:-}; KEYMODE=${KEYMODE:-decode}
COMMON="--pth $PTH --coeffs $COEFFS --harness $HARNESS ${BLOCKS:+--blocks $BLOCKS} --lora $LORA --lanes $LANES --layout $LAYOUT"
[ "$SHARD_LAND" = 1 ] || COMMON="$COMMON --no-shard-land"

if [ "${1:-}" = names ]; then   # every product name of these settings, KEY=VALUE (size_pipeline.sh evals it); no side effect
    for v in CAP PLAN PFCAP PFPLAN BCAP BPLAN JCAP JPLAN CFT R_RUN R_PF R_BATCH R_SERVE; do printf '%s=%q\n' "$v" "${!v}"; done
    for m in decode pf b s; do printf 'KEYDIR_%s=%q\n' "$m" "$(keydir $m)"; done
    exit 0
fi

mkdir -p "$OUT"
log() { echo "[rwkv7 $(date -u +%H:%M:%S)] $*" | tee -a "$OUT/session.log"; }
die() { log "STOP: $*"; exit 1; }
pyrun() { ( cd "$W" && env PYTHONPATH="$W/python" "$@" ); }

# capcheck KIND CAPTURE [PLAN]: the capture (and the plan cut from it) belong to these settings, else stop
# (tools/rwkv7/capture_check.py; CAPTURE_MISMATCH_OK=1 overrides, printed)
capcheck() {
    local kind=$1 cap=$2 plan=${3:-} K=0 T=0
    case "$kind" in b|join) K=$B_K ;; pf) T=$PF_T ;; esac
    pyrun "$PY" "$HERE/capture_check.py" "$cap" --kind "$kind" --pth "$PTH" --coeffs "$COEFFS" --lanes "$LANES" \
        --layout "$LAYOUT" --blocks "$BLOCKS" --batch "$K" --batch-carry "$B_CARRY" --prefill-chunk "$T" --lora "$LORA" \
        --shard-land "$SHARD_LAND" ${plan:+--plan "$plan"} --sha-cache "$OUT/sha256_cache.json" 2>&1 | tee -a "$OUT/session.log"
    [ "${PIPESTATUS[0]}" = 0 ] || die "capture_check refused $cap${plan:+ / $plan}"
}

step_build() { OUT=$OUT bash "$W/tools/fhe_ssm/run_gpu.sh" build || die "build failed"; }

# The CPU suite is written against the 0.1B (test_mirror's padding check at C = 768, K = 32 and T = 32 at t = 32;
# review lens 2, finding 3): it runs on $PTH only when $PTH is 768 wide, else on CHECK_PTH / CHECK_COEFFS (the 0.1B
# and its final coefficients on the box); $PTH itself gets the width-agnostic checks (shapes, the layout and preset,
# a coefficient site for every block).
CHECK_PTH=${CHECK_PTH:-/root/d_models/RWKV-x070-World-0.1B-v2.8-20241210-ctx4096.pth}
CHECK_COEFFS=${CHECK_COEFFS:-/root/vd/runs/0.1b/coeffs_final.json}
step_check() {
    for f in "$PTH" "$COEFFS" "$HARNESS/rwkv_tokenizer.py"; do [ -f "$f" ] || die "missing $f"; done
    pyrun "$PY" -c "from perseus import _pc; miss=[n for n in ('Session','Program','KeySource','PtStore') if not hasattr(_pc,n)]; print('device classes missing:', miss or 'none'); raise SystemExit(1 if miss else 0)" \
        || die "perseus._pc is not the CUDA build"
    pyrun "$PY" -c "import networkx, scipy; print('planner deps ok')" || die "the planner needs networkx and scipy in $PY"
    local w; w=$(pyrun "$PY" - "$PTH" "$COEFFS" <<'PY'
import json, sys
from perseus.examples.rwkv7.ckpt_info import model_shapes, preset, shapes
m = model_shapes(shapes(sys.argv[1]))
sites = json.load(open(sys.argv[2]))["sites"]
miss = [b for b in range(m["n_layer"]) if not any(n.startswith(f"blocks.{b}.") for n in sites)]
p = preset(m)
print(f"check {sys.argv[1]}: {m['n_layer']} x {m['width']}, heads {m['heads']}, d_ffn {m['d_ffn']}, lanes {p['LANES']}, "
      f"fits one card {p['fits_one_card']}; blocks without a coefficient site: {miss or 'none'}", file=sys.stderr)
print(m["width"] if not miss else "MISSING")
PY
) || die "the width-agnostic checks of $PTH failed"
    [ "$w" != MISSING ] || die "$COEFFS has no site for some block of $PTH"
    local cp=$PTH cc=$COEFFS
    if [ "$w" != 768 ]; then
        cp=$CHECK_PTH; cc=$CHECK_COEFFS
        for f in "$cp" "$cc"; do [ -f "$f" ] || die "the CPU suite needs the 0.1B: missing $f (CHECK_PTH / CHECK_COEFFS)"; done
        log "check: width $w; the CPU suite runs on the 0.1B ($cp)"
    fi
    pyrun env RWKV7_PTH="$cp" RWKV7_COEFFS="$cc" RWKV7_HARNESS="$HARNESS" \
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
    [ -f "$CAP/capture_config.json" ] && { capcheck decode "$CAP"; log "capture: reusing $CAP"; return 0; }
    log "capture -> $CAP ($CAPENV)"
    pyrun env CUDA_VISIBLE_DEVICES="" $CAPENV "$PY" -m perseus.examples.rwkv7.run_capture $COMMON \
        --pcmm-shift "$PCMM_SHIFT" --out "$CAP" > "$OUT/capture$S.log" 2>&1 || { tail -20 "$OUT/capture$S.log"; die "capture failed"; }
    tail -3 "$OUT/capture$S.log" | tee -a "$OUT/session.log"
}

plan_one() {   # plan_one KIND CAPTURE PLAN_DIR LOG [EXTRA_ENV]
    local kind=$1 cap=$2 plan=$3 lg=$4 extra=${5:-}
    if [ -f "$plan/PLAN_CMD.txt" ]; then capcheck "$kind" "$cap" "$plan"; log "plan: reusing $plan"; return 0; fi
    capcheck "$kind" "$cap"
    log "plan $cap -> $plan ($PLANENV${extra:+ $extra})"
    env $PLANENV $extra PLAN_ROOT="$OUT" PY="$PY" MEMGUARD_GB= bash "$W/tools/fhe_ssm/make_plan.sh" "$cap" "$(basename "$plan")" \
        > "$lg" 2>&1 || { tail -30 "$lg"; die "plan failed ($lg)"; }
    tail -8 "$lg" | tee -a "$OUT/session.log"
}

step_plan() { plan_one decode "$CAP" "$PLAN" "$OUT/plan$S$LV.log"; }

step_capture_pf() {
    [ -f "$PFCAP/capture_config.json" ] && { capcheck pf "$PFCAP"; log "capture_pf: reusing $PFCAP"; return 0; }
    [ -f "$CAP/pcmm_shifts.json" ] || die "no $CAP/pcmm_shifts.json (run the capture step first: one shifts file serves both)"
    capcheck decode "$CAP"
    log "capture_pf T=$PF_T -> $PFCAP"
    pyrun env CUDA_VISIBLE_DEVICES="" $CAPENV "$PY" -m perseus.examples.rwkv7.run_capture $COMMON \
        --prefill-chunk "$PF_T" --shifts "$CAP/pcmm_shifts.json" --out "$PFCAP" > "$OUT/capture$PF$S.log" 2>&1 \
        || { tail -20 "$OUT/capture$PF$S.log"; die "capture_pf failed"; }
    tail -3 "$OUT/capture$PF$S.log" | tee -a "$OUT/session.log"
}

step_plan_pf() { plan_one pf "$PFCAP" "$PFPLAN" "$OUT/plan$PF$S$LV.log"; }

step_pfcheck() {
    pyrun env RWKV7_PTH="$PTH" RWKV7_COEFFS="$COEFFS" RWKV7_HARNESS="$HARNESS" RWKV7_LAYOUT="$LAYOUT" $CAPENV \
        "$PY" -m pytest -q -s -p no:warnings "$W/python/perseus/examples/rwkv7/tests/test_prefill.py" \
        -k session_prepare --rwkv7-plan --rwkv7-pf-dir "$OUT/pfcheck$S" > "$OUT/pfcheck$S.log" 2>&1 \
        || { tail -30 "$OUT/pfcheck$S.log"; die "pfcheck failed ($OUT/pfcheck$S.log)"; }
    grep -E "decode kv regs|passed|skipped" "$OUT/pfcheck$S.log" | tee -a "$OUT/session.log"
    grep -q "1 passed" "$OUT/pfcheck$S.log" || die "pfcheck did not run (skipped?) — see $OUT/pfcheck$S.log"
}

cf_flags() {
    [ -n "$CFTABLE" ] || return 0
    [ -f "$CFT" ] || pyrun "$PY" -m perseus.examples.fhe_ssm.pcmm_cf_table "$CAP" "$CFT" >> "$OUT/session.log" 2>&1 \
        || die "pcmm_cf_table failed"
    echo "--pcmm-cf-table $CFT"
}

keys_flag() {   # keys_flag decode|pf|b|s (need_keys has checked it in the calling shell)
    case "$KEYS" in
        inproc) echo "--keys inproc" ;;
        bundle) echo "--keys $(keydir "$1")" ;;
        *) echo "--keys $KEYS" ;;
    esac
}

need_keys() {   # stops here (not inside a $(...)) when the bundle is missing; KEYS has no default (fhe_ssm's rule)
    case "$KEYS" in
        "") die "KEYS is required for the device steps: inproc (INSECURE: the session holds the secret) | bundle | DIR" ;;
        inproc) log "keys: inproc (INSECURE: the session holds the secret; recorded insecure-inproc)" ;;
        bundle) [ -f "$(keydir "$1")/keys.pckb" ] || die "KEYS=bundle: no $(keydir "$1")/keys.pckb (KEYMODE=$1 keyspec, keygen)" ;;
        *) [ -f "$KEYS/keys.pckb" ] || die "KEYS=$KEYS: no keys.pckb there" ;;
    esac
}

guard_mode() {   # guard_mode decode|pf|b|s: every capture and plan the mode opens belongs to these settings
    capcheck decode "$CAP" "$PLAN"
    case "$1" in
        pf) capcheck pf "$PFCAP" "$PFPLAN" ;;
        b) capcheck b "$BCAP" "$BPLAN" ;;
        s) capcheck pf "$PFCAP" "$PFPLAN"; capcheck b "$BCAP" "$BPLAN"; capcheck join "$JCAP" "$JPLAN" ;;
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
    guard_mode "$KEYMODE"
    log "keyspec $KEYMODE -> $d/keyspec.json"
    pyrun env CUDA_VISIBLE_DEVICES=$GPU $CAPENV "$PY" -m perseus.examples.rwkv7.run_planned $COMMON $(mode_flags "$KEYMODE") \
        --schedule "$SCHED" $(cf_flags) --pcmm-cf "$PCMM_CF" --stc-cache "$STC_CACHE" --pool "$POOL" --keyspec "$d/keyspec.json" \
        --out "$d/keyspec_run.jsonl" > "$d/keyspec.log" 2>&1 || { tail -20 "$d/keyspec.log"; die "keyspec failed ($d/keyspec.log)"; }
    tail -2 "$d/keyspec.log" | tee -a "$OUT/session.log"
}

step_keygen() {
    local d; d=$(keydir "$KEYMODE")
    [ -f "$d/keyspec.json" ] || die "no $d/keyspec.json (run keyspec first, KEYMODE=$KEYMODE)"
    log "keygen $KEYMODE (the client role) -> $d/keys.pckb"
    pyrun env CUDA_VISIBLE_DEVICES=$GPU "$PY" -m perseus.examples.rwkv7.run_planned --keygen "$d" --keyspec-file "$d/keyspec.json" \
        --threads "${KEYGEN_THREADS:-8}" > "$d/keygen.log" 2>&1 || { tail -20 "$d/keygen.log"; die "keygen failed ($d/keygen.log)"; }
    # the secret stays with the client role: owner-only permissions (the split is between objects in one process)
    [ -f "$d/client_secret.bin" ] && chmod 600 "$d/client_secret.bin"
    tail -1 "$d/keygen.log" | cut -c1-400 | tee -a "$OUT/session.log"
}

gpu_busy() {
    local busy; busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$GPU" 2>/dev/null)
    [ -z "$busy" ] || log "WARNING: GPU $GPU has other processes ($busy): timings are not clean"
}

step_run_pf() {
    need_keys pf
    [ -f "$PLAN/PLAN_CMD.txt" ] || die "no plan $PLAN"
    [ -f "$PFPLAN/PLAN_CMD.txt" ] || die "no prefill plan $PFPLAN (run capture_pf and plan_pf)"
    guard_mode pf
    gpu_busy
    local I; I=$(pyrun "$PY" -m perseus.examples.rwkv7.run_ids --harness "$HARNESS" --text "$PF_TEXT" --n $((PF_TOKENS + 1))) \
        || die "tokenizer failed"
    local R=$R_PF${RUN_TAG:+_$RUN_TAG}
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
    [ "$B_K" -ge 2 ] || die "B_K=$B_K: a batched capture needs K >= 2 (run_capture takes the batched path only for K > 1)"
    [ -f "$BCAP/capture_config.json" ] && { capcheck b "$BCAP"; log "capture_b: reusing $BCAP"; return 0; }
    [ -f "$CAP/pcmm_shifts.json" ] || die "no $CAP/pcmm_shifts.json (run the capture step first: one shifts file serves both)"
    capcheck decode "$CAP"
    log "capture_b K=$B_K carry=$B_CARRY -> $BCAP"
    pyrun env CUDA_VISIBLE_DEVICES="" $CAPENV "$PY" -m perseus.examples.rwkv7.run_capture $COMMON \
        --batch "$B_K" --batch-carry "$B_CARRY" --shifts "$CAP/pcmm_shifts.json" --out "$BCAP" \
        > "$OUT/capture$BC$S.log" 2>&1 || { tail -20 "$OUT/capture$BC$S.log"; die "capture_b failed"; }
    tail -3 "$OUT/capture$BC$S.log" | tee -a "$OUT/session.log"
}

step_plan_b() { plan_one b "$BCAP" "$BPLAN" "$OUT/plan$BC$S$LV$PO.log" "$BPLANENV"; }

step_bcheck() {
    pyrun env RWKV7_PTH="$PTH" RWKV7_COEFFS="$COEFFS" RWKV7_HARNESS="$HARNESS" RWKV7_LAYOUT="$LAYOUT" $CAPENV \
        "$PY" -m pytest -q -s -p no:warnings "$W/python/perseus/examples/rwkv7/tests/test_batched.py" \
        -k session_prepare --rwkv7-plan --rwkv7-b-dir "$OUT/bcheck$S" > "$OUT/bcheck$S.log" 2>&1 \
        || { tail -30 "$OUT/bcheck$S.log"; die "bcheck failed ($OUT/bcheck$S.log)"; }
    grep -E "batched kv|passed|skipped" "$OUT/bcheck$S.log" | tee -a "$OUT/session.log"
    grep -q "1 passed" "$OUT/bcheck$S.log" || die "bcheck did not run (skipped?) — see $OUT/bcheck$S.log"
}

step_batch() {
    [ "$B_K" -ge 2 ] || die "B_K=$B_K: a batched run needs K >= 2"
    need_keys b
    [ -f "$BPLAN/PLAN_CMD.txt" ] || die "no batched plan $BPLAN (run capture_b and plan_b)"
    [ -f "$B_PROMPTS" ] || die "no prompts file $B_PROMPTS"
    guard_mode b
    gpu_busy
    local R=$R_BATCH${RUN_TAG:+_$RUN_TAG}
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
    [ -f "$JCAP/capture_config.json" ] && { capcheck join "$JCAP"; log "capture_j: reusing $JCAP"; return 0; }
    log "capture_j K=$B_K -> $JCAP"
    pyrun env CUDA_VISIBLE_DEVICES="" $CAPENV "$PY" -m perseus.examples.rwkv7.run_capture $COMMON \
        --join --batch "$B_K" --out "$JCAP" > "$OUT/capture_join_k$B_K$S.log" 2>&1 \
        || { tail -20 "$OUT/capture_join_k$B_K$S.log"; die "capture_j failed"; }
    tail -2 "$OUT/capture_join_k$B_K$S.log" | tee -a "$OUT/session.log"
}

step_plan_j() { plan_one join "$JCAP" "$JPLAN" "$OUT/plan_join_k$B_K$S$LV.log"; }

step_scheck() {
    pyrun env RWKV7_PTH="$PTH" RWKV7_COEFFS="$COEFFS" RWKV7_HARNESS="$HARNESS" RWKV7_LAYOUT="$LAYOUT" $CAPENV \
        "$PY" -m pytest -q -s -p no:warnings "$W/python/perseus/examples/rwkv7/tests/test_serve_session.py" \
        -k serve_session_prepare --rwkv7-plan --rwkv7-b-dir "$OUT/scheck$S" > "$OUT/scheck$S.log" 2>&1 \
        || { tail -30 "$OUT/scheck$S.log"; die "scheck failed ($OUT/scheck$S.log)"; }
    grep -E "serve registers|passed|skipped" "$OUT/scheck$S.log" | tee -a "$OUT/session.log"
    grep -q "1 passed" "$OUT/scheck$S.log" || die "scheck did not run (skipped?) — see $OUT/scheck$S.log"
}

step_serve() {
    need_keys s
    [ "$B_CARRY" = lane ] || die "serve joins lane-form carries: B_CARRY=lane"
    for f in "$PLAN" "$PFPLAN" "$BPLAN" "$JPLAN"; do
        [ -f "$f/PLAN_CMD.txt" ] || die "no plan $f (serve needs plan, plan_pf, plan_b, plan_j)"
    done
    [ -f "$S_FILE" ] || die "no arrivals file $S_FILE"
    guard_mode s
    gpu_busy
    local R=$R_SERVE${RUN_TAG:+_$RUN_TAG}
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
    guard_mode decode
    gpu_busy
    local I; I=$(ids) || die "tokenizer failed"
    log "run: $TOKENS tokens after the prefix, ids $I"
    local R=$R_RUN${RUN_TAG:+_$RUN_TAG}
    pyrun env CUDA_VISIBLE_DEVICES=$GPU $CAPENV "$PY" -m perseus.examples.rwkv7.run_planned $COMMON --ids "$I" \
        --tokens "$TOKENS" --plan "$PLAN" --shifts "$CAP/pcmm_shifts.json" --pcmm-cf "$PCMM_CF" \
        --schedule "$SCHED" $(cf_flags) --stc-cache "$STC_CACHE" --pool "$POOL" $(keys_flag decode) \
        --out "$R.jsonl" ${RUN_EXTRA:-} > "$R.log" 2>&1
    local rc=$?
    grep '"kind": "token"' "$R.jsonl" 2>/dev/null | cut -c1-220 | tee -a "$OUT/session.log"
    grep '"kind": "summary"' "$R.jsonl" 2>/dev/null | tee -a "$OUT/session.log"
    [ $rc = 0 ] || { tail -30 "$R.log"; die "run failed ($R.log)"; }
}

run_cmd() {
    case "$1" in
        build|check|crosscheck|capture|plan|run|capture_pf|plan_pf|pfcheck|run_pf|capture_b|plan_b|bcheck|batch|keyspec|keygen) "step_$1" ;;
        capture_j|plan_j|scheck|serve) "step_$1" ;;
        all) step_build; step_check; step_crosscheck; step_capture; step_plan; step_run ;;
        pf) step_capture; step_plan; step_capture_pf; step_plan_pf; step_run_pf ;;
        b) step_capture; step_plan; step_capture_b; step_plan_b; step_batch ;;
        s) step_capture; step_plan; step_capture_pf; step_plan_pf; step_capture_b; step_plan_b; step_capture_j; step_plan_j
           step_serve ;;
        *) die "unknown step $1" ;;
    esac
    log "$1: done"
}

[ $# -gt 0 ] || set -- all
for cmd in "$@"; do run_cmd "$cmd"; done
