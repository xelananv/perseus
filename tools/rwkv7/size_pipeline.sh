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
#             PASS iff unclamped == clamped, all finite, scans clean, ppl within 0.5 % of exact, LAMBADA within 0.5
#             points, top-1 >= 98 % (FINAL_PLAN.md's bars) -> $RUNS/verdict_final.json; a FAIL stops the pipeline
#   capture   PCMM band shifts + the decode capture (run_box.sh capture; CPU)
#   plan      the decode plan (run_box.sh plan), and plan_cost.py's refreshes per stage -> $OUT/plan_cost_decode.json
#   pf        the prefill capture + plan (run_box.sh capture_pf, plan_pf; T = PF_T)
#   b         the batched capture + plan (run_box.sh capture_b, plan_b; K = B_K from the preset)
#   ---- DRY=1 stops here (no GPU, no device build needed) ----
#   keys      KEYS=bundle: run_box.sh keyspec + keygen for decode, pf and b (the client role; has_secret false)
#   run       decode P1-P3 + 32 free-running steps each (run_box.sh run, LEVERS from env)
#   run_pf    prefill on P3 + the long prompts (PF_TEXTS, a file of one prompt per line), 32 generated each
#   batch     batched K = B_K, the 32 prompts, 32 generated each (falls back to B_K / 2 if the session refuses memory)
#   records   the FINAL_PLAN.md record per run (tools/rwkv7/run_records.py): s/token, top-1, relErrRms median/max,
#             poly_sites_out_of_range, setup, the sampler window -> $OUT/records.json
#
# Environment: OUT (/root/out_any/<tag>), RUNS (/root/vd/runs/<tag>), HARNESS (/root/vd/harness), TP (the harness's
# python with torch: /root/venv_d/bin/python), DATA (/root/d_data: wiki.valid.raw, wiki.test.raw, data/lambada_test.jsonl),
# RELU2 (m02 | abs), RELU2_REL (1e-4), DRY (1: stop before the GPU), SMOKE (1: tiny token counts for a CPU check of the
# harness steps; SMOKE_TEXT = a text file standing in for every corpus), KEYS (inproc | bundle; default bundle on a full
# run), LEVERS (unset | best), BLOCKS (unset = all), and the memory preset overrides POOL, STC_CACHE, POOL_PF, POOL_B,
# STC_CACHE_B, B_K.
set -uo pipefail
PTH=${1:?usage: size_pipeline.sh <pth> <tag> [steps]}; TAG=${2:?tag}; STEPS=${3:-all}
[ -f "$PTH" ] || { echo "no checkpoint $PTH" >&2; exit 2; }
PTH=$(cd "$(dirname "$PTH")" && pwd)/$(basename "$PTH")           # every step runs from its own directory
HERE=$(cd "$(dirname "$0")" && pwd)
W=$(git -C "$HERE" rev-parse --show-toplevel)
PY=${PY:-$W/.venv/bin/python}
OUT=${OUT:-/root/out_any/$TAG}; RUNS=${RUNS:-/root/vd/runs/$TAG}
HARNESS=${HARNESS:-/root/vd/harness}; TP=${TP:-/root/venv_d/bin/python}; DATA=${DATA:-/root/d_data}
RELU2=${RELU2:-m02}; RELU2_REL=${RELU2_REL:-1e-4}; DRY=${DRY:-}; SMOKE=${SMOKE:-}
mkdir -p "$OUT" "$RUNS"
LOG=$OUT/pipeline.log
log() { echo "[size_pipeline $TAG $(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG"; }
# st NAME CMD...: run a step, log rc and seconds; a failure stops the pipeline
st() { local n=$1; shift; local t0; t0=$(date +%s); log "step $n: $*"; "$@"; local rc=$?
       log "step $n rc=$rc $(( $(date +%s) - t0 )) s"; [ $rc = 0 ] || { log "STOP at $n"; exit $rc; }; }
want() { [ "$STEPS" = all ] || [[ ",$STEPS," == *",$1,"* ]]; }
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
import json, sys
d = json.load(open(sys.argv[1]))
for k, v in d["preset"].items():
    print(f'P_{k}={v}')
print(f'C={d["width"]}; NL={d["n_layer"]}; SH={d["shards"]}')
PY
)"
export LANES=${LANES:-$P_LANES} POOL=${POOL:-$P_POOL} STC_CACHE=${STC_CACHE:-$P_STC_CACHE} POOL_PF=${POOL_PF:-$P_POOL_PF}
export POOL_B=${POOL_B:-$P_POOL_B} STC_CACHE_B=${STC_CACHE_B:-$P_STC_CACHE_B} B_K=${B_K:-$P_B_K}
export PF_T=${PF_T:-$LANES}
[ "$P_fits_one_card" = True ] || log "WARNING: the memory model puts this size over one card (${P_memory_decode_mb} MB decode; ANYSIZE_DESIGN.md §4)"
log "width $C, $NL blocks, lanes $LANES, $SH shard(s); preset POOL=$POOL STC_CACHE=$STC_CACHE POOL_PF=$POOL_PF POOL_B=$POOL_B STC_CACHE_B=$STC_CACHE_B B_K=$B_K"

H() { ( cd "$HARNESS" && "$TP" "$@" ); }
FINAL=$RUNS/coeffs_final.json

want calib && st calib H calibrate.py --model "$PTH" --text "$VALID" --tokens $CAL_TOK --ctx 4096 --batch 2 $DEV \
    --out "$RUNS/ranges_ctx4096.json"

if want fit; then
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
    st gen H "$HERE/gen_greedy.py" --harness "$HARNESS" --model "$PTH" --coeffs "$FINAL" --prompts "$HERE/batch_prompts.txt" \
        --max-prompts $GEN_P --n $GEN_N --out "$RUNS/gen_final.jsonl" --text-out "$RUNS/gen_final.txt"
fi

if want scan; then
    if [ -z "$SMOKE" ] && [ ! -f "$LAMB_TXT" ]; then
        python3 -c "import json,sys; open(sys.argv[2],'w').write(''.join(json.loads(l)['text']+'\n\n' for l in open(sys.argv[1])))" "$LAMB" "$LAMB_TXT"
    fi
    for n in wiki lambada gen; do
        case $n in wiki) f=$TEST ;; lambada) f=$LAMB_TXT ;; gen) f=$RUNS/gen_final.txt ;; esac
        st scan_$n H scan_ranges.py --model "$PTH" --coeffs "$FINAL" --text "$f" --tokens $SCAN_TOK --ctx 1024 --feed poly \
            $DEV --out "$RUNS/scanfull_final_$n.json"
    done
fi

if want gate; then
    st eval_exact H eval.py --model "$PTH" --wikitext "$TEST" --lambada "$LAMB" --mode exact $EVAL_X $DEV --tag exact \
        --out "$RUNS/eval_final.jsonl"
    st eval_clamp H eval.py --model "$PTH" --wikitext "$TEST" --lambada "$LAMB" --mode poly --coeffs "$FINAL" --clamp $EVAL_X \
        $DEV --tag final_clamp --out "$RUNS/eval_final.jsonl"
    st eval_unclamped H eval.py --model "$PTH" --wikitext "$TEST" --lambada "$LAMB" --mode poly --coeffs "$FINAL" $EVAL_X \
        $DEV --tag final_unclamped --out "$RUNS/eval_final.jsonl"
    python3 "$HERE/gate_verdict.py" --eval "$RUNS/eval_final.jsonl" --clamp-tag final_clamp --unclamped-tag final_unclamped \
        --scan "$RUNS/scanfull_final_wiki.json" --scan "$RUNS/scanfull_final_lambada.json" \
        --scan "$RUNS/scanfull_final_gen.json" --out "$RUNS/verdict_final.json" > /dev/null 2> "$OUT/gate.txt"
    rc=$?; cat "$OUT/gate.txt" | tee -a "$LOG"
    if [ $rc != 0 ]; then
        if [ -n "$SMOKE" ]; then log "gate FAIL under SMOKE (tiny corpora: the verdict is not meaningful); continuing"
        else log "STOP: the gate failed ($RUNS/verdict_final.json); the owner decides"; exit 3; fi
    fi
fi

# ── the encrypted model: run_box.sh with this checkpoint and these coefficients ──
export OUT PTH COEFFS=$FINAL HARNESS
RB="bash $HERE/run_box.sh"
want capture && st capture $RB capture
if want plan; then
    st plan $RB plan
    PL=$(ls -d "$OUT"/bootstrap_placements/plan_rwkv7* | head -1); CP=$(ls -d "$OUT"/cap_rwkv7 "$OUT"/cap_rwkv7_b* 2>/dev/null | head -1)
    pyw "$PY" -m perseus.examples.rwkv7.plan_cost "$CP" "$PL" --census 0 > "$OUT/plan_cost_decode.json" && \
        log "decode plan: $(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['total'], 'refreshes per token')" "$OUT/plan_cost_decode.json")"
fi
want pf && { st capture_pf $RB capture_pf; st plan_pf $RB plan_pf; }
want b && { st capture_b $RB capture_b; st plan_b $RB plan_b; }
if [ -n "$DRY" ]; then log "DRY: stopping before the GPU steps (keys, run, run_pf, batch)"; exit 0; fi

# ── the GPU: the sampler runs throughout ──
bash "$HERE/sampler.sh" "$OUT/sampler.csv" 5 & SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT
export KEYS=${KEYS:-bundle}
if want keys && [ "$KEYS" = bundle ]; then
    for m in decode pf b; do st keyspec_$m env KEYMODE=$m $RB keyspec; st keygen_$m env KEYMODE=$m $RB keygen; done
fi
P1=$'\n\nThe scientists discovered a new species of'
P2=$'\n\nIn the context of language learning, American pronunciation'
P3=$'\n\nUser: A shop sells 3 pencils for 2 dollars. How much do 12 pencils cost? Answer with the number.\n\nAssistant:'
if want run; then
    for p in P1 P2 P3; do
        st run_$p env TEXT="${!p}" TOKENS=64 RUN_EXTRA="--generate 32" RUN_TAG=$p $RB run
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
        st run_pf_$i env PF_TEXT="$txt" PF_TOKENS=$((n - 1)) PF_GENERATE=32 RUN_TAG=p$i $RB run_pf
        i=$((i + 1))
    done < <(printf '%s\n' "${P3//$'\n'/\\n}"; [ -n "${PF_TEXTS:-}" ] && cat "$PF_TEXTS")
fi
if want batch; then
    env B_GENERATE=32 $RB batch || { log "batch K=$B_K failed; retrying at K=$((B_K / 2))"; st batch_half env B_K=$((B_K / 2)) B_GENERATE=32 $RB b; }
fi
want records && st records pyw "$PY" "$HERE/run_records.py" "$OUT" --sampler "$OUT/sampler.csv" --out "$OUT/records.json"
log "done: $STEPS"
