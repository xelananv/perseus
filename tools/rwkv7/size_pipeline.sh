#!/bin/bash
# size_pipeline.sh <pth> <tag> [steps] — one RWKV-x070-World checkpoint of ANY size, from the .pth to the encrypted runs.
# The deliverable is this pipeline (python/perseus/examples/rwkv7/ANYSIZE_DESIGN.md §5); every step is logged to
# $OUT/pipeline.log with its rc and seconds, and the first failing step stops it.
#
#   bash tools/rwkv7/size_pipeline.sh /root/d_models/RWKV-x070-World-1.5B-v3-20250127-ctx4096.pth 1.5b
#   DRY=1 bash tools/rwkv7/size_pipeline.sh M.pth 0.1b            # everything up to the GPU (stops before keyspec)
#   bash tools/rwkv7/size_pipeline.sh M.pth 1.5b gate              # one step (or a comma list: fit,scan,gate)
#
# Steps (default: all, in this order):
#   info      the checkpoint's shapes without loading it (layers, width, heads, d_ffn, LoRA ranks), the shard count
#             and the memory preset for that width (ANYSIZE_DESIGN.md §4) -> $OUT/info.json
#   calib     harness calibrate.py: ctx 4096, 200,000 tokens of wiki.valid, batch 2 -> $RUNS/ranges_ctx4096.json
#   fit       the final recipe (gate 3, new_pod/RESULTS.md): fit.py --range minmax --margin 0.5 --target-err 1e-3
#             --metric rel --max-degree 255 --newton-target 1e-3 --site-target 'ffn\.relu2=1e-4' --rsqrt newton-const
#             --rsqrt-range-factor 10 (-> coeffs_wide.json), then ReLU^2 by RELU2:
#               m02 (default, gate 3): a second fit at margin 0.2, its ffn.relu2 entries merged (merge_relu2.py)
#               abs (REVIEW §5.A1): ffn.relu2 at the wide margin with an ABSOLUTE target per block, RELU2_REL x max|f|
#                   of the margin-0.1 interval (relu2_abs_targets.py): one fit, accuracy independent of the margin
#             -> $RUNS/coeffs_final.json
#   gen       the model's own free-running text: 32 prompts (tools/rwkv7/batch_prompts.txt) x 256 greedy tokens on the
#             poly model UNCLAMPED (gen_greedy.py) -> $RUNS/gen_final.{jsonl,txt}
#   scan      harness scan_ranges.py, unclamped (poly feed), every token of wiki.test, LAMBADA's texts and the
#             generations; each site kind's excursions listed -> $RUNS/scanfull_final_{wiki,lambada,gen}.json
#   gate      harness eval.py exact, poly clamped, poly UNCLAMPED (WikiText-103 test + LAMBADA), then gate_verdict.py:
#             PASS iff unclamped == clamped, all finite, every free-running generation finite (gen_final.jsonl), scans clean, ppl within 0.5 % of exact, LAMBADA within 0.5
#             points, top-1 >= 98 % (FINAL_PLAN.md's bars) -> $RUNS/verdict_final.json; a FAIL stops the pipeline
#   capture   PCMM band shifts + the decode capture (run_box.sh capture; CPU)
#   plan      the decode plan (run_box.sh plan), and plan_cost.py's refreshes per stage -> $OUT/plan_cost_decode.json
#   pf        the prefill capture + plan (run_box.sh capture_pf, plan_pf; T = PF_T from the preset: 16 at 0.4B and above)
#   b         the batched capture + plan (run_box.sh capture_b, plan_b; K = B_K from the preset, <= 16 at 0.4B and above;
#             skipped when B_K < 2). Why 16 (pod 3, 2026-11-05): the 0.4B T = 32 prefill and K = 32 batched sessions
#             were killed by the HOST out-of-memory killer at setup (~185 GB of 181, the GPU idle); the preset also
#             holds the host-RAM model (ckpt_info.host_gb; logged at the start) under 92 % of 181 GB
#   ---- DRY=1 stops here (no GPU, no device build needed) ----
#   keys      KEYS=bundle: run_box.sh keyspec + keygen for decode, pf and b (the client role; has_secret false; b only
#             when B_K >= 2)
#   run       decode P1-P3 + 32 free-running steps each (run_box.sh run, LEVERS from env)
#   run_pf    prefill on P3 + the long prompts (PF_TEXTS, a file of one prompt per line), 32 generated each
#   batch     batched K = B_K, the 32 prompts, 32 generated each. Only when the session refuses memory (the runtime's
#             "RuntimeError: memory:", a CUDA out-of-memory, or a SIGKILL = the host's OOM killer, which run_box.sh
#             notes as "Killed (rc 137 ...)" in the run's log) it falls back to K / 2 with that K's own
#             capture, plan and, under KEYS=bundle, keyspec + keygen (the bundle is made for exactly the plan that runs);
#             any other failure stops the pipeline. Skipped when B_K < 2 (2.9B and above: the preset fits no batch)
#   records   the FINAL_PLAN.md record per run (tools/rwkv7/run_records.py): s/token, top-1, relErrRms median/max,
#             poly_sites_out_of_range, setup, the sampler window (each GPU step is named after its record) -> records.json
#
# Overwrites: calib, fit, gen, scan and gate write into RUNS. Each refuses to overwrite its product (ranges_ctx4096.json,
# coeffs_final.json, gen_final.jsonl, scanfull_final_*.json, eval_final.jsonl) unless FORCE=1, which first moves the old
# file to <name>.prev-<UTC time>: RUNS defaults to /root/vd/runs/<tag>_pipe, never a gate's published directory.
#
# Environment: OUT (/root/out_any/<tag>), RUNS (/root/vd/runs/<tag>_pipe), FORCE (1: overwrite RUNS products), HARNESS (/root/vd/harness), TP (the harness's
# python with torch: /root/venv_d/bin/python), DATA (/root/d_data: wiki.valid.raw, wiki.test.raw, data/lambada_test.jsonl),
# RELU2 (m02 | abs), RELU2_REL (1e-4), DRY (1: stop before the GPU), SMOKE (1: tiny token counts for a CPU check of the
# harness steps; SMOKE_TEXT = a text file standing in for every corpus), KEYS (inproc | bundle; default bundle on a full
# run), LEVERS (unset | best), BLOCKS (unset = all), and the memory preset overrides POOL, STC_CACHE, POOL_PF, POOL_B,
# STC_CACHE_B, B_K, PF_T (B_K / PF_T set in the environment are used as given, not capped at 16).
set -uo pipefail
PTH=${1:?usage: size_pipeline.sh <pth> <tag> [steps]}; TAG=${2:?tag}; STEPS=${3:-all}
[ -f "$PTH" ] || { echo "no checkpoint $PTH" >&2; exit 2; }
PTH=$(cd "$(dirname "$PTH")" && pwd)/$(basename "$PTH")           # every step runs from its own directory
HERE=$(cd "$(dirname "$0")" && pwd)
W=$(git -C "$HERE" rev-parse --show-toplevel)
PY=${PY:-$W/.venv/bin/python}
if [ -z "${OUT:-}" ]; then
    if [ -d /root/out_any ] || { [ -w /root ] && [ -d /root/d_models ]; }; then OUT=/root/out_any/$TAG
    else OUT=$W/.cache/rwkv7_any/$TAG; fi
fi
if [ -z "${RUNS:-}" ]; then
    if [ -d /root/vd/runs ] || { [ -w /root ] && [ -d /root/d_models ]; }; then RUNS=/root/vd/runs/${TAG}_pipe
    else RUNS=$W/.cache/rwkv7_runs/${TAG}_pipe; fi
fi
FORCE=${FORCE:-}
if [ -z "${HARNESS:-}" ]; then
    [ -f "$HERE/harness/rwkv_tokenizer.py" ] && HARNESS=$HERE/harness || HARNESS=/root/vd/harness
fi
if [ -z "${TP:-}" ]; then
    [ -x /root/venv_d/bin/python ] && TP=/root/venv_d/bin/python || TP=$PY
fi
if [ -z "${DATA:-}" ]; then
    [ -f /root/d_data/wiki.test.raw ] && DATA=/root/d_data || DATA=$W/.cache/rwkv7/data
fi
RELU2=${RELU2:-m02}; RELU2_REL=${RELU2_REL:-1e-4}; DRY=${DRY:-}; SMOKE=${SMOKE:-}
mkdir -p "$OUT" "$RUNS"
LOG=$OUT/pipeline.log
log() { echo "[size_pipeline $TAG $(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG"; }
# st NAME CMD...: run a step, log rc and seconds; a failure stops the pipeline
st() { local n=$1; shift; local t0; t0=$(date +%s); log "step $n: $*"; "$@"; local rc=$?
       log "step $n rc=$rc $(( $(date +%s) - t0 )) s"; [ $rc = 0 ] || { log "STOP at $n"; exit $rc; }; }
want() { [ "$STEPS" = all ] || [[ ",$STEPS," == *",$1,"* ]]; }
# st_try NAME CMD...: as st, but returns the rc (the caller decides)
st_try() { local n=$1; shift; local t0; t0=$(date +%s); log "step $n: $*"; "$@"; local rc=$?
           log "step $n rc=$rc $(( $(date +%s) - t0 )) s"; return $rc; }
# fresh STEP FILE...: a RUNS product is never overwritten silently (review lens 2, finding 4): stop unless FORCE=1,
# which moves the old file aside first
fresh() { local n=$1 f stamp; shift; stamp=$(date -u +%Y%m%dT%H%M%SZ)
          for f in "$@"; do
              [ -e "$f" ] || continue
              if [ "$FORCE" = 1 ]; then mv "$f" "$f.prev-$stamp"; log "$n: FORCE=1, moved $f to $f.prev-$stamp"
              else log "STOP at $n: $f exists (FORCE=1 overwrites it, keeping the old one as .prev-<time>)"; exit 4; fi
          done; return 0; }
pyw() { ( cd "$W" && env PYTHONPATH="$W/python" "$@" ); }

# corpora (SMOKE: one small public text for all of them, and a 20-line LAMBADA stand-in made from it)
if [ -n "$SMOKE" ]; then
    T=${SMOKE_TEXT:?SMOKE=1 needs SMOKE_TEXT (a text file)}
    VALID=$T; TEST=$T; LAMB_TXT=$T; LAMB=$OUT/smoke_lambada.jsonl
    python3 - "$T" "$LAMB" <<'PY'
import json, sys
lines = [l.strip() for l in open(sys.argv[1]) if len(l.split()) > 8][:20]
with open(sys.argv[2], "w") as f:
    for l in lines:
        f.write(json.dumps({"text": l}) + "\n")
PY
    CAL_TOK=2000; EVAL_X="--max-tokens 2048 --max-lambada 20 --batch 1"; SCAN_TOK=2048; GEN_N=${GEN_N:-8}; GEN_P=${GEN_P:-2}
    DEV="--device cpu"
else
    VALID=$DATA/wiki.valid.raw; TEST=$DATA/wiki.test.raw; LAMB=$DATA/data/lambada_test.jsonl; LAMB_TXT=$DATA/lambada_as_text.raw
    CAL_TOK=200000; EVAL_X=""; SCAN_TOK=2000000; GEN_N=${GEN_N:-256}; GEN_P=${GEN_P:-32}; DEV=""
fi

# ── info: the shapes, the shards and the memory preset ──
if want info || [ ! -f "$OUT/info.json" ]; then
    st info pyw "$PY" -m perseus.examples.rwkv7.ckpt_info "$PTH" --out "$OUT/info.json"
fi
eval "$(python3 - "$OUT/info.json" <<'PY'
import json, shlex, sys
d = json.load(open(sys.argv[1]))
for k, v in d["preset"].items():
    print(f'P_{k}={shlex.quote(" ".join(map(str, v)) if isinstance(v, list) else str(v))}')
print(f'C={d["width"]}; NL={d["n_layer"]}; SH={d["shards"]}; AUTO_LANES={d.get("auto_lanes", 0)}')
PY
)"
# LANES=0 (the default layout) when the preset keeps it, so these runs share run_box.sh's default names
[ -z "${LANES:-}" ] && [ "$P_LANES" = "$AUTO_LANES" ] && P_LANES=0
export LANES=${LANES:-$P_LANES} POOL=${POOL:-$P_POOL} STC_CACHE=${STC_CACHE:-$P_STC_CACHE} POOL_PF=${POOL_PF:-$P_POOL_PF}
export POOL_B=${POOL_B:-$P_POOL_B} STC_CACHE_B=${STC_CACHE_B:-$P_STC_CACHE_B}
# 0.4B and above start at T = 16 and K <= 16 (pod 3, 2026-11-05: the 0.4B T = 32 prefill and K = 32 batched sessions
# were killed by the HOST out-of-memory killer at setup, ~185 GB of 181 with the GPU idle; ckpt_info.big / host_gb);
# PF_T / B_K in the environment override it
BIG=$( [ "$NL" -ge 24 ] || [ "$C" -gt 1024 ] && echo 1 )
if [ -n "${B_K:-}" ]; then log "B_K=$B_K from the environment (preset $P_B_K)"
else B_K=$P_B_K; [ -n "$BIG" ] && [ "$B_K" -gt 16 ] && B_K=16; fi
if [ -n "${PF_T:-}" ]; then log "PF_T=$PF_T from the environment (preset $P_PF_T)"
else PF_T=$P_PF_T; [ -n "$BIG" ] && [ "$PF_T" -gt 16 ] && PF_T=16; fi
export B_K PF_T
[ "$P_fits_one_card" = True ] || log "WARNING: the memory model puts this size over one card (${P_memory_decode_mb} MB decode; ANYSIZE_DESIGN.md §4)"
log "width $C, $NL blocks, lanes $LANES, $SH shard(s); preset POOL=$POOL STC_CACHE=$STC_CACHE POOL_PF=$POOL_PF POOL_B=$POOL_B STC_CACHE_B=$STC_CACHE_B B_K=$B_K PF_T=$PF_T"
log "host RAM model (GB, setup; cap ${P_host_cap_gb}): decode ${P_host_decode_gb}, batched K=$P_B_K ${P_host_batched_gb}, prefill T=$P_PF_T ${P_host_prefill_gb// / to } (ANYSIZE_DESIGN.md §4)"

H() { ( cd "$HARNESS" && "$TP" "$@" ); }
FINAL=$RUNS/coeffs_final.json

want calib && fresh calib "$RUNS/ranges_ctx4096.json" && st calib H calibrate.py --model "$PTH" --text "$VALID" --tokens $CAL_TOK --ctx 4096 --batch 2 $DEV \
    --out "$RUNS/ranges_ctx4096.json"

if want fit; then
    fresh fit "$FINAL" "$RUNS/coeffs_wide.json" "$RUNS/coeffs_m02.json"
    RECIPE=(--ranges "$RUNS/ranges_ctx4096.json" --range minmax --target-err 1e-3 --metric rel --max-degree 255
            --newton-target 1e-3 --rsqrt newton-const --rsqrt-range-factor 10)
    case "$RELU2" in
        m02)
            st fit_wide H fit.py "${RECIPE[@]}" --margin 0.5 --site-target 'ffn\.relu2=1e-4' --out "$RUNS/coeffs_wide.json"
            st fit_m02 H fit.py "${RECIPE[@]}" --margin 0.2 --site-target 'ffn\.relu2=1e-4' --out "$RUNS/coeffs_m02.json"
            st merge python3 "$HERE/merge_relu2.py" "$RUNS/coeffs_wide.json" "$RUNS/coeffs_m02.json" "$FINAL" ;;
        abs)
            ABS=(); while IFS= read -r f; do ABS+=("$f"); done < <(python3 "$HERE/relu2_abs_targets.py" \
                "$RUNS/ranges_ctx4096.json" --rel "$RELU2_REL")
            [ ${#ABS[@]} -gt 0 ] || { log "no ffn.relu2 sites in the ranges"; exit 2; }
            log "relu2 absolute targets: ${ABS[*]:0:3} ..."
            st fit_abs H fit.py "${RECIPE[@]}" --margin 0.5 "${ABS[@]}" --out "$FINAL" ;;
        *) log "RELU2=$RELU2: m02 | abs"; exit 2 ;;
    esac
    python3 -c "import json,sys; d=json.load(open(sys.argv[1])); s=[n for n,v in d['sites'].items() if v.get('target_met') is False]; print('sites missing their target:', s or 'none')" "$FINAL" | tee -a "$LOG"
fi

if want gen; then
    fresh gen "$RUNS/gen_final.jsonl" "$RUNS/gen_final.txt"
    st gen H "$HERE/gen_greedy.py" --harness "$HARNESS" --model "$PTH" --coeffs "$FINAL" --prompts "$HERE/batch_prompts.txt" \
        --max-prompts $GEN_P --n $GEN_N --out "$RUNS/gen_final.jsonl" --text-out "$RUNS/gen_final.txt"
fi

if want scan; then
    if [ -z "$SMOKE" ] && [ ! -f "$LAMB_TXT" ]; then
        python3 -c "import json,sys; open(sys.argv[2],'w').write(''.join(json.loads(l)['text']+'\n\n' for l in open(sys.argv[1])))" "$LAMB" "$LAMB_TXT"
    fi
    fresh scan "$RUNS"/scanfull_final_{wiki,lambada,gen}.json
    for n in wiki lambada gen; do
        case $n in wiki) f=$TEST ;; lambada) f=$LAMB_TXT ;; gen) f=$RUNS/gen_final.txt ;; esac
        st scan_$n H scan_ranges.py --model "$PTH" --coeffs "$FINAL" --text "$f" --tokens $SCAN_TOK --ctx 1024 --feed poly \
            $DEV --out "$RUNS/scanfull_final_$n.json"
    done
fi

if want gate; then
    fresh gate "$RUNS/eval_final.jsonl" "$RUNS/verdict_final.json"
    st eval_exact H eval.py --model "$PTH" --wikitext "$TEST" --lambada "$LAMB" --mode exact $EVAL_X $DEV --tag exact \
        --out "$RUNS/eval_final.jsonl"
    st eval_clamp H eval.py --model "$PTH" --wikitext "$TEST" --lambada "$LAMB" --mode poly --coeffs "$FINAL" --clamp $EVAL_X \
        $DEV --tag final_clamp --out "$RUNS/eval_final.jsonl"
    st eval_unclamped H eval.py --model "$PTH" --wikitext "$TEST" --lambada "$LAMB" --mode poly --coeffs "$FINAL" $EVAL_X \
        $DEV --tag final_unclamped --out "$RUNS/eval_final.jsonl"
    python3 "$HERE/gate_verdict.py" --eval "$RUNS/eval_final.jsonl" --clamp-tag final_clamp --unclamped-tag final_unclamped \
        --scan "$RUNS/scanfull_final_wiki.json" --scan "$RUNS/scanfull_final_lambada.json" \
        --scan "$RUNS/scanfull_final_gen.json" --gen "$RUNS/gen_final.jsonl" --out "$RUNS/verdict_final.json" \
        > /dev/null 2> "$OUT/gate.txt"
    rc=$?; cat "$OUT/gate.txt" | tee -a "$LOG"
    if [ $rc != 0 ]; then
        if [ -n "$SMOKE" ]; then log "gate FAIL under SMOKE (tiny corpora: the verdict is not meaningful); continuing"
        else log "STOP: the gate failed ($RUNS/verdict_final.json); review the verdict before going on"; exit 3; fi
    fi
fi

# ── the encrypted model: run_box.sh with this checkpoint and these coefficients ──
export OUT PTH COEFFS=$FINAL HARNESS
RB="bash $HERE/run_box.sh"
# run_box.sh's own names for these settings (CAP, PLAN, BCAP, ..., R_RUN, KEYDIR_<mode>): never re-derived here
names() { env "$@" bash "$HERE/run_box.sh" names; }
NAMES=$(names) || { log "run_box.sh names failed"; exit 2; }
eval "$NAMES"
want capture && st capture $RB capture
if want plan; then
    st plan $RB plan
    pyw "$PY" -m perseus.examples.rwkv7.plan_cost "$CAP" "$PLAN" --census 0 > "$OUT/plan_cost_decode.json" && \
        log "decode plan: $(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['total'], 'refreshes per token')" "$OUT/plan_cost_decode.json")"
fi
want pf && { st capture_pf $RB capture_pf; st plan_pf $RB plan_pf; }
if [ "$B_K" -lt 2 ]; then
    BATCHED=; log "batched steps (b, keys b, batch) skipped: the preset's B_K=$B_K (< 2: no batch fits the memory model)"
else BATCHED=1; fi
want b && [ -n "$BATCHED" ] && { st capture_b $RB capture_b; st plan_b $RB plan_b; }
if [ -n "$DRY" ]; then log "DRY: stopping before the GPU steps (keys, run, run_pf, batch)"; exit 0; fi

# ── the GPU: the sampler runs throughout ──
bash "$HERE/sampler.sh" "$OUT/sampler.csv" 5 & SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT
export KEYS=${KEYS:-bundle}
if want keys && [ "$KEYS" = bundle ]; then
    for m in decode pf ${BATCHED:+b}; do st keyspec_$m env KEYMODE=$m $RB keyspec; st keygen_$m env KEYMODE=$m $RB keygen; done
fi
P1=$'\n\nThe scientists discovered a new species of'
P2=$'\n\nIn the context of language learning, American pronunciation'
P3=$'\n\nUser: A shop sells 3 pencils for 2 dollars. How much do 12 pencils cost? Answer with the number.\n\nAssistant:'
if want run; then
    for p in P1 P2 P3; do
        st "$(basename "$R_RUN")_$p" env TEXT="${!p}" TOKENS=64 RUN_EXTRA="--generate 32" RUN_TAG=$p $RB run
    done
fi
if want run_pf; then
    # P3 and the long prompts (PF_TEXTS: one per line, \n escapes; FINAL_PLAN.md: longrun_prompts64.txt's 116-128-token
    # lines), each read by prefill passes of PF_T and then 32 free-running steps
    i=0
    while IFS= read -r line; do
        [ -n "$line" ] || continue
        txt=$(printf '%b' "$line"); [[ "$txt" == $'\n\n'* ]] || txt=$'\n\n'"$txt"
        n=$(pyw "$PY" -m perseus.examples.rwkv7.run_ids --harness "$HARNESS" --text "$txt" | tr ',' '\n' | wc -l)
        st "$(basename "$R_PF")_p$i" env PF_TEXT="$txt" PF_TOKENS=$((n - 1)) PF_GENERATE=32 RUN_TAG=p$i $RB run_pf
        i=$((i + 1))
    done < <(printf '%s\n' "${P3//$'\n'/\\n}"; [ -n "${PF_TEXTS:-}" ] && cat "$PF_TEXTS")
fi
if want batch && [ -n "$BATCHED" ]; then
    st_try "$(basename "$R_BATCH")" env B_GENERATE=32 $RB batch; rc=$?
    if [ $rc != 0 ]; then
        # fall back only on a memory refusal (review lens 2, finding 2); the K/2 run gets its own capture, plan and
        # key bundle (run_box.sh names carry _k<K>), so the bundle is made for exactly the plan that runs
        if [ "$KEYS" != bundle ] && [ "$KEYS" != inproc ]; then
            log "STOP at batch: rc=$rc; KEYS=$KEYS is a fixed bundle made for K=$B_K's plan: no K/2 fallback"; exit $rc
        elif grep -qE "RuntimeError: memory:|out of memory|CUDA_ERROR_OUT_OF_MEMORY|^Killed" "$R_BATCH.log" 2>/dev/null \
               && [ "$B_K" -ge 4 ]; then
            # a memory refusal: the runtime's "RuntimeError: memory: ..." (the device), a CUDA out-of-memory, or the
            # host's OOM killer (pod 3: no traceback, the GPU idle at setup; run_box.sh appends "Killed (rc 137 ...)"
            # to the run's log when the process died of SIGKILL)
            K2=$((B_K / 2)); HN=$(names B_K=$K2) || { log "run_box.sh names failed"; exit 2; }
            eval "$(printf '%s\n' "$HN" | grep -E '^(R_BATCH|KEYDIR_b)=' | sed 's/^/H_/')"
            why=$(grep -m1 -oE 'RuntimeError: memory:.{0,160}|out of memory.{0,80}|^Killed.*' "$R_BATCH.log")
            log "batch K=$B_K refused memory (${why:-killed at setup with no traceback: the host OOM killer}); K=$K2"
            H=$K2
            st capture_b_k$H env B_K=$H $RB capture_b
            st plan_b_k$H env B_K=$H $RB plan_b
            if [ "$KEYS" = bundle ]; then
                st keyspec_b_k$H env B_K=$H KEYMODE=b $RB keyspec
                st keygen_b_k$H env B_K=$H KEYMODE=b $RB keygen
            fi
            st "$(basename "$H_R_BATCH")" env B_K=$H B_GENERATE=32 $RB batch
        else
            log "STOP at batch: rc=$rc and not a memory refusal ($R_BATCH.log)"; exit $rc
        fi
    fi
fi
want records && st records pyw "$PY" "$HERE/run_records.py" "$OUT" --sampler "$OUT/sampler.csv" --out "$OUT/records.json"
log "done: $STEPS"
