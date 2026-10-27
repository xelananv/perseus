#!/bin/bash
# run_gpu.sh — the fhe-ssm port's GPU session on a fresh box, one subcommand per step of
# results/perseus-20261006/fork_port/gpu_20261011/speedrun_port_20261027/RUNSHEET_GPU.md (research repository).
#
#   bash tools/fhe_ssm/run_gpu.sh box              record the box: GPU, driver, nvcc, RAM, disk, load, commit
#   bash tools/fhe_ssm/run_gpu.sh build            uv venv + editable install + maturin develop --release
#   bash tools/fhe_ssm/run_gpu.sh verify           _pc on the device, upstream's CPU tests, the port's tests, additive,
#                                                  the CPU compile check built
#   bash tools/fhe_ssm/run_gpu.sh bench            this build's refresh prices (pc-bench bts --all --k 1,8) -> PLAN_BTS_MS
#   bash tools/fhe_ssm/run_gpu.sh arms             the axis arms and what each sets
#   bash tools/fhe_ssm/run_gpu.sh capture ARM      the CPU-twin capture of ARM (no GPU; reused when present)
#   bash tools/fhe_ssm/run_gpu.sh plan ARM         tools/fhe_ssm/make_plan.sh on that capture (+ the compile check)
#   bash tools/fhe_ssm/run_gpu.sh run ARM          the planned run: $DRIVER --ticks $TICKS --profile, nvidia-smi sampled
#   bash tools/fhe_ssm/run_gpu.sh smoke            capture + plan + run of `ref` at BLOCKS=1 (the first device contact)
#   bash tools/fhe_ssm/run_gpu.sh matrix [ARM...]  capture + plan + run per arm, strictly one after another
#
# Environment: OUT (default <worktree>/.cache/fhe_ssm/gpu_<UTC date>; every product lands there, one dir per arm),
# DATA (a directory holding the model's weights .npz, seeds .json and config .json; WEIGHTS / SEEDS / CONFIG override
# each), DRIVER (the planned driver's python module, default perseus.examples.fhe_ssm.run_planned: its contract is in
# RUNSHEET_GPU.md §4), TICKS (2), BLOCKS (all), GPU (0), KEYS (inproc | bundle), PY (<worktree>/.venv/bin/python),
# PERSEUS_CORE_ARCH (build: default from nvidia-smi's compute capability), PC_NVCC_JOBS (build: default 8).
# Nothing here edits a tracked file; captures, plans and records go under OUT.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
W=$(git -C "$HERE" rev-parse --show-toplevel)
PY=${PY:-$W/.venv/bin/python}
OUT=${OUT:-$W/.cache/fhe_ssm/gpu_$(date -u +%Y%m%d)}
DRIVER=${DRIVER:-perseus.examples.fhe_ssm.run_planned}
TICKS=${TICKS:-2}; BLOCKS=${BLOCKS:-}; GPU=${GPU:-0}; KEYS=${KEYS:-inproc}
mkdir -p "$OUT"
log() { echo "[run_gpu $(date -u +%H:%M:%S)] $*" | tee -a "$OUT/session.log"; }
die() { log "STOP: $*"; exit 1; }
resolve_data() {   # WEIGHTS / SEEDS / CONFIG, else the first match of each in DATA; sets MODEL_ARGS
    local v g f
    for v in WEIGHTS:'*weights*.npz' SEEDS:'*seed*.json' CONFIG:'*config*.json'; do
        g=${v#*:}; v=${v%%:*}
        if [ -z "${!v:-}" ]; then
            [ -n "${DATA:-}" ] || die "set DATA (or WEIGHTS, SEEDS, CONFIG)"
            f=$(ls "$DATA"/$g 2>/dev/null | head -1); [ -n "$f" ] || die "no $g in $DATA"
            printf -v "$v" '%s' "$f"
        fi
        [ -f "${!v}" ] || die "$v=${!v} is not a file"
    done
    MODEL_ARGS="--weights $WEIGHTS --seeds $SEEDS --config $CONFIG --levels preset${BLOCKS:+ --blocks $BLOCKS}"
}

# ---- the arms: capture flags | make_plan.sh environment | run flags
# The capture flags are run_fake.py's (python/perseus/examples/fhe_ssm/run_fake.py); the run repeats them so the
# driver rebuilds the same model on its CPU twin. LANDING=high also exports the landing's runtime switches to the
# capture (the fake reads its routes and landings from the Config in the environment, impl/fakecap.py:841-860).
ARMS="ref depth2 persite packstats high pair sbk16 emfuse pcmmtrunk24 pcmm24 best lockstep2 lockstep4"
arm() {   # sets CAP, PLANENV, RUNF, NOTE for arm $1
    CAP=""; PLANENV="LANDING=stock"; RUNF=""; NOTE=""
    case "$1" in
        ref)        NOTE="the reference: the shipped circuit, this runtime's landings without StC-first (diagonal linears)" ;;
        depth2)     CAP="--newton-form depth2" ;;
        persite)    CAP="--newton-sites per-site" ;;
        packstats)  CAP="--pack-stats halves" ;;
        high)       PLANENV="LANDING=high"; NOTE="dense refreshes StC-first at 28, SparseB-512 at 26" ;;
        pair)       PLANENV="LANDING=high PAIR=1"; RUNF="--schedule siblings"; NOTE="refresh pairs; compare with high" ;;
        sbk16)      PLANENV="LANDING=high SPARSE_B_K=16"; NOTE="SparseB EvalMod K=16, replay only; compare with high" ;;
        emfuse)     PLANENV="LANDING=high RUNTIME_EXTRA=PC_EM_FUSE_RESCALE=1"; NOTE="kernel switch, binds no plan; compare with high" ;;
        pcmmtrunk24) CAP="--linear pcmm-trunk --pcmm-wbits 24"; PLANENV="LANDING=high"; NOTE="PCMM needs StC-first: compare with high" ;;
        pcmm24)     CAP="--linear pcmm --pcmm-wbits 24"; PLANENV="LANDING=high"; NOTE="compare with high" ;;
        best)       CAP="--newton-form depth2 --newton-sites per-site --pack-stats halves --linear pcmm --pcmm-wbits 24"
                    PLANENV="LANDING=high PAIR=1"; RUNF="--schedule siblings"; NOTE="the stack; compare with ref and high" ;;
        lockstep2|lockstep4)
                    CAP="--streams ${1#lockstep}"; PLANENV="LANDING=high"; RUNF="--schedule lockstep"
                    NOTE="BLOCKED until the port captures k ciphertext sets per stage (UPSTREAM_API §2.4)" ;;
        *) die "unknown arm $1 (arms: $ARMS)" ;;
    esac
    case "$PLANENV" in *LANDING=high*) CAPENV="FIDESLIB_BTS_STC_FIRST=2 FIDESLIB_BTS_STC_FIRST_ROUTES=dense FIDESLIB_BTS_STC_FIRST_CF=13 FIDESLIB_BTS_RAISE_DROP=512:1 AUTO_BTS_LEVEL=44" ;;
                       *) CAPENV="FIDESLIB_BTS_STC_FIRST=0" ;; esac
}
capname() {   # the capture an arm uses: arms that differ only at plan or run time share one
    arm "$1"
    local land; land=$(echo "$PLANENV" | sed -n 's/.*LANDING=\([a-z]*\).*/\1/p')
    local key="${land}_$(echo "$CAP" | tr -c 'a-z0-9\n' '_' | tr -s '_')"
    echo "cap_${key%_}${BLOCKS:+_b$BLOCKS}"
}

gpu_free() {
    local apps; apps=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i "$GPU" 2>/dev/null)
    [ -z "$apps" ] || die "GPU $GPU runs other processes ($apps): a timing needs an idle GPU (CLAUDE.md R6)"
}
build_id() {
    echo "commit=$(git -C "$W" rev-parse --short HEAD) dirty=$(git -C "$W" status --porcelain --untracked-files=no | wc -l | tr -d ' ')" \
         "pc_so=$(sha256sum "$W"/python/perseus/_pc*.so 2>/dev/null | cut -c1-16)" \
         "nvcc=$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9.]*\).*/\1/p')" \
         "arch=${PERSEUS_CORE_ARCH:-?} load=$(cut -d' ' -f1-3 /proc/loadavg 2>/dev/null)"
}

cmd=${1:-}; shift || true
case "$cmd" in
box)
    {   echo "date_utc=$(date -u +%FT%TZ)"; echo "host=$(hostname)"; uname -a
        nvidia-smi --query-gpu=index,name,memory.total,driver_version,compute_cap --format=csv
        nvidia-smi topo -m 2>/dev/null | head -12
        nvcc --version 2>/dev/null | tail -2; echo "CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.2 (default)}"
        free -g 2>/dev/null; df -h "$W" | tail -1; nproc; cat /proc/loadavg 2>/dev/null
        command -v uv cargo rustup; git -C "$W" log -1 --format='%H %s'
    } > "$OUT/box.txt" 2>&1
    cat "$OUT/box.txt"; log "box recorded in $OUT/box.txt" ;;
build)
    cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i "$GPU" 2>/dev/null | head -1 | tr -d '.')
    export PERSEUS_CORE_ARCH=${PERSEUS_CORE_ARCH:-${cap:-120}} PC_NVCC_JOBS=${PC_NVCC_JOBS:-8}
    export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.2}; export PATH="$CUDA_HOME/bin:$PATH"
    log "build: PERSEUS_CORE_ARCH=$PERSEUS_CORE_ARCH PC_NVCC_JOBS=$PC_NVCC_JOBS CUDA_HOME=$CUDA_HOME"
    t0=$(date +%s)
    ( cd "$W" && uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e ".[hf,examples,dev]" \
        && .venv/bin/maturin develop --release ) > "$OUT/build.log" 2>&1 || { tail -30 "$OUT/build.log"; die "build failed ($OUT/build.log)"; }
    log "build: done in $(( $(date +%s) - t0 )) s; $(build_id)" ;;
verify)
    ( cd "$W" && "$PY" -c "from perseus import _pc; print('perseus._pc', [n for n in ('Session','Program','KeySource','Client') if hasattr(_pc, n)], 'device free GB', _pc.device_free_gb())" ) 2>&1 | tee -a "$OUT/session.log" || die "_pc does not load with a device"
    ( cd "$W" && CUDA_VISIBLE_DEVICES="" "$PY" -m pytest -q -p no:cacheprovider -m 'not gpu' python/tests/test_fake_shadow.py \
        python/tests/test_pairing_cpu.py python/tests/test_rmsnorm_example.py python/tests/test_prefill_cpu.py 2>&1 | tail -3 ) | tee -a "$OUT/session.log"
    if [ -d "$W/python/perseus/examples/fhe_ssm/tests" ]; then
        ( cd "$W" && CUDA_VISIBLE_DEVICES="" "$PY" -m pytest -q -p no:cacheprovider python/perseus/examples/fhe_ssm/tests 2>&1 | tail -3 ) | tee -a "$OUT/session.log"
    fi
    bash "$HERE/check_additive.sh" | tee -a "$OUT/session.log"
    bash "$HERE/compile_check/build.sh" | tee -a "$OUT/session.log"
    log "verify: $(build_id)" ;;
bench)
    gpu_free
    ( cd "$W" && CUDA_VISIBLE_DEVICES=$GPU cargo run --release -p pc-tools --bin pc-bench -- bts --all --k 1,8 --reps 20 --keys packed ) \
        > "$OUT/bench_bts.txt" 2>&1 || { tail -20 "$OUT/bench_bts.txt"; die "pc-bench failed"; }
    tail -25 "$OUT/bench_bts.txt"; log "bench: $OUT/bench_bts.txt (set PLAN_BTS_MS from it for ILP_OBJECTIVE=ms); $(build_id)" ;;
arms)
    for a in $ARMS; do arm "$a"; printf '%-12s capture[%s] plan[%s] run[%s] -> %s  %s\n' "$a" "$CAP" "$PLANENV" "$RUNF" "$(capname "$a")" "$NOTE"; done ;;
capture)
    a=${1:?arm}; arm "$a"; c="$OUT/$(capname "$a")"
    case "$NOTE" in BLOCKED*) die "$a: $NOTE" ;; esac
    if [ -f "$c/capture_config.json" ]; then log "capture $a: reusing $c"; exit 0; fi
    resolve_data
    log "capture $a -> $c (CPU only): $CAPENV run_fake $CAP"
    t0=$(date +%s)
    ( cd "$W" && env CUDA_VISIBLE_DEVICES="" PYTHONPATH="$W/python" $CAPENV "$PY" -m perseus.examples.fhe_ssm.run_fake \
        $MODEL_ARGS --tokens 2 $CAP --capture "$c.part" --capture-tick 1 --out "$OUT/$(capname "$a").fake.json" ) \
        > "$OUT/$(capname "$a").capture.log" 2>&1 || { tail -20 "$OUT/$(capname "$a").capture.log"; die "capture $a failed"; }
    rm -rf "$c"; mv "$c.part" "$c"
    log "capture $a: done in $(( $(date +%s) - t0 )) s" ;;
plan)
    a=${1:?arm}; arm "$a"; c="$OUT/$(capname "$a")"
    [ -d "$c" ] || die "no capture $c: run_gpu.sh capture $a"
    ( env $PLANENV PLAN_ROOT="$OUT" MEMGUARD_GB= bash "$HERE/make_plan.sh" "$c" "plan_$a" ) 2>&1 | tee "$OUT/plan_$a.log" | tail -8
    grep -q "compile check FAILED" "$OUT/bootstrap_placements/plan_$a/summary.txt" && die "plan $a does not bind (compile check)"
    log "plan $a: $OUT/bootstrap_placements/plan_$a" ;;
run)
    a=${1:?arm}; arm "$a"; p="$OUT/bootstrap_placements/plan_$a"
    [ -f "$p/PLAN_CMD.txt" ] || die "no plan $p: run_gpu.sh plan $a"
    "$PY" -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('$DRIVER') else 1)" 2>/dev/null \
        || die "no driver module $DRIVER yet (RUNSHEET_GPU.md §4 is its contract)"
    case "$CAP" in *"--linear pcmm"*) ;; *) log "note: $a uses diagonal linears: the driver must stream them (UPSTREAM_API §0.2)";; esac
    gpu_free
    r="$OUT/run_$a"; mkdir -p "$r"
    echo "$(build_id)" > "$r/build_id.txt"
    nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,power.draw --format=csv -lms 500 -i "$GPU" > "$r/nvsmi.csv" 2>/dev/null &
    smi=$!
    log "run $a: $DRIVER --plan $p --ticks $TICKS --profile $CAP $RUNF --keys $KEYS"
    t0=$(date +%s)
    resolve_data
    ( cd "$W" && env CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH="$W/python" $CAPENV "$PY" -m "$DRIVER" $MODEL_ARGS --plan "$p" --ticks "$TICKS" \
        --profile --keys "$KEYS" $CAP $RUNF --out "$r/record.json" ) > "$r/run.log" 2>&1
    rc=$?
    kill "$smi" 2>/dev/null; wait "$smi" 2>/dev/null
    peak=$(awk -F', ' 'NR>1 {gsub(/ MiB/,"",$2); if ($2+0>m) m=$2+0} END {print m+0}' "$r/nvsmi.csv")
    log "run $a: exit $rc in $(( $(date +%s) - t0 )) s; nvidia-smi peak ${peak} MiB; $r/record.json; $(build_id)"
    tail -15 "$r/run.log"
    [ $rc = 0 ] || die "run $a failed ($r/run.log)" ;;
smoke)
    export BLOCKS=1
    bash "$0" capture ref && bash "$0" plan ref && bash "$0" run ref ;;
matrix)
    [ $# -gt 0 ] || set -- ref depth2 persite packstats high pair sbk16 emfuse pcmmtrunk24 pcmm24 best
    for a in "$@"; do
        arm "$a"; case "$NOTE" in BLOCKED*) log "skip $a: $NOTE"; continue ;; esac
        bash "$0" capture "$a" && bash "$0" plan "$a" && bash "$0" run "$a" || log "arm $a stopped (above); continuing"
    done ;;
*)
    sed -n '2,24p' "$0"; exit 2 ;;
esac
