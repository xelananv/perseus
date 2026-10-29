#!/bin/bash
# run_gpu.sh — the fhe-ssm port's GPU session on a fresh box, one subcommand per step of tools/fhe_ssm/RUNSHEET_GPU.md.
#
#   bash tools/fhe_ssm/run_gpu.sh box              record the box: GPU, driver, nvcc, RAM, disk, load, commit
#   bash tools/fhe_ssm/run_gpu.sh build            uv venv + editable install + maturin develop --release
#   bash tools/fhe_ssm/run_gpu.sh verify           _pc loads (its device classes present), the GPU's memory, upstream's
#                                                  CPU tests, the port's tests, the change check, the CPU compile check
#   bash tools/fhe_ssm/run_gpu.sh bench            this build's refresh prices (pc-bench bts --all --k 1,8) -> PLAN_BTS_MS
#   bash tools/fhe_ssm/run_gpu.sh arms             the axis arms and what each sets
#   bash tools/fhe_ssm/run_gpu.sh capture ARM      the CPU-twin capture of ARM (no GPU; reused when present)
#   bash tools/fhe_ssm/run_gpu.sh plan ARM         tools/fhe_ssm/make_plan.sh on that capture (+ the compile check)
#   bash tools/fhe_ssm/run_gpu.sh prepare ARM      the driver's CPU part (--prepare-only: record, compile check, report)
#   bash tools/fhe_ssm/run_gpu.sh keyspec ARM      the key request of ARM's plan ($DRIVER --keyspec: the model's needs and,
#                                                  on the device build, the session's) -> $OUT/keys_ARM/keyspec.json
#   bash tools/fhe_ssm/run_gpu.sh keygen ARM       the client role for that request ($DRIVER --keygen: a fresh secret,
#                                                  keys.pckb); then KEYS=bundle FHE_SSM_KEYS=$OUT/keys_ARM for run
#   bash tools/fhe_ssm/run_gpu.sh run ARM          the planned run: $DRIVER --ticks $TICKS --profile, nvidia-smi sampled
#                                                  (diagonal linears only under DIAG_SOURCE=generated: the device
#                                                  diagonal encoder; else the PCMM arms only)
#   bash tools/fhe_ssm/run_gpu.sh smoke            capture + plan + run of `pcmm24` at BLOCKS=1 (the first device contact)
#   bash tools/fhe_ssm/run_gpu.sh smoke-diag       capture + plan + run of `ref` at BLOCKS=1 under DIAG_SOURCE=generated
#                                                  (the device diagonal encoder's first device contact)
#   bash tools/fhe_ssm/run_gpu.sh matrix [ARM...]  capture + plan + run (prepare for a CPU-only arm), one arm at a time
#
# Environment: OUT (default <worktree>/.cache/fhe_ssm/gpu_<UTC date>; every product lands there, one dir per arm),
# DATA (a directory holding the model's weights .npz, seeds .json and config .json; WEIGHTS / SEEDS / CONFIG override
# each), DRIVER (the planned driver's python module, default perseus.examples.fhe_ssm.run_planned: its contract is in
# RUNSHEET_GPU.md §4), TICKS (2), BLOCKS (all), GPU (0), KEYS (NO DEFAULT; run, smoke and matrix refuse without it:
# inproc = in-process keys, INSECURE, the server holds the secret, a timing run only | bundle = the client's key directory
# in FHE_SSM_KEYS, made by keyspec + keygen | that directory), PROFILE_EVERY (2: a stage fence on every 2nd tick, so one run
# holds fenced and unfenced ticks; 1 = every tick), PREPARE_EXTRA (more `prepare` flags, e.g. --lean on a small host),
# PY (<worktree>/.venv/bin/python), PERSEUS_CORE_ARCH (build: default from nvidia-smi's compute capability),
# PC_NVCC_JOBS (build: default 8), DIAG_SOURCE (values | generated, default values: where the diagonal linears'
# plaintexts come from on the device, the driver's --diag-source; generated = the device diagonal encoder, which makes
# every non-blocked arm a device arm; a driver flag only, so the captures and plans are the same under both).
# Nothing here edits a tracked file; captures, plans and records go under OUT.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
W=$(git -C "$HERE" rev-parse --show-toplevel)
PY=${PY:-$W/.venv/bin/python}
OUT=${OUT:-$W/.cache/fhe_ssm/gpu_$(date -u +%Y%m%d)}
DRIVER=${DRIVER:-perseus.examples.fhe_ssm.run_planned}
TICKS=${TICKS:-2}; BLOCKS=${BLOCKS:-}; GPU=${GPU:-0}; KEYS=${KEYS:-}; PROFILE_EVERY=${PROFILE_EVERY:-2}
DIAG_SOURCE=${DIAG_SOURCE:-values}
case "$DIAG_SOURCE" in values|generated) ;; *) echo "run_gpu: DIAG_SOURCE=$DIAG_SOURCE (values | generated)" >&2; exit 2 ;; esac
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
# driver rebuilds the same model on its CPU twin. LANDING=high's runtime switches (CAPENV) are exported to the capture
# and the run; the port's session Config takes them from the environment (fakes.runtime_config, fakes.LANDING_SWITCHES
# in python/perseus/examples/fhe_ssm/fakes.py), so a high capture's capture_config.json carries bts.stc_first=2 and
# bts.raise_drop=512:1 and its own dense refreshes land StC-first. `capture` still refuses a LANDING=high capture whose
# stamp lacks them (a capture made without the switches exported, or by a port without that Config).
ARMS="ref depth2 persite packstats high pair sbk16 emfuse pcmmtrunk24 pcmm24 best lockstep2 lockstep4"
arm() {   # sets CAP, PLANENV, RUNF, NOTE, DEVICE for arm $1
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
                    NOTE="BLOCKED until the port captures k ciphertext sets per stage (RUNSHEET_GPU.md §3)" ;;
        *) die "unknown arm $1 (arms: $ARMS)" ;;
    esac
    case "$PLANENV" in *LANDING=high*) CAPENV="FIDESLIB_BTS_STC_FIRST=2 FIDESLIB_BTS_STC_FIRST_ROUTES=dense FIDESLIB_BTS_STC_FIRST_CF=13 FIDESLIB_BTS_RAISE_DROP=512:1 AUTO_BTS_LEVEL=44" ;;
                       *) CAPENV="FIDESLIB_BTS_STC_FIRST=0" ;; esac
    # HIGH_CAPTURE_OK=0 blocks every LANDING=high arm before its 11-minute capture (for a checkout whose capture Config
    # does not take the switches); the default 1 runs them, and the post-capture check in `capture` guards the stamp.
    case "$PLANENV" in *LANDING=high*)
        [ "${HIGH_CAPTURE_OK:-1}" = 1 ] || NOTE="BLOCKED (HIGH_CAPTURE_OK=0): LANDING=high needs a capture Config with stc_first=2 / raise_drop=512:1; $NOTE" ;;
    esac
    # DEVICE=1: the arm runs on a device. Under DIAG_SOURCE=values only the all-PCMM arms do: a values session keeps
    # every plaintext it reads resident and the diagonals are 344,064 per tick (about 2.3 TiB at 28 rows; the driver's
    # --pt-budget-gb refuses them), so a diagonal arm runs its CPU part only (capture, plan, prepare). Under
    # DIAG_SOURCE=generated the device builds each diagonal inside its group's fused sum from the declared tiles
    # (2.625 GiB of float64 matrices, no store slot per diagonal): every non-blocked arm is a device arm, and the
    # driver flag rides in RUNF (run, prepare, keyspec), never in CAP, so the capture name and the plan stay shared.
    case "$CAP" in *"--linear pcmm --"*|*"--linear pcmm") DEVICE=1 ;; *) DEVICE=0 ;; esac
    if [ "$DIAG_SOURCE" = generated ]; then
        RUNF="${RUNF:+$RUNF }--diag-source generated"
        case "$NOTE" in BLOCKED*) ;; *) DEVICE=1 ;; esac
    fi
}
capname() {   # the capture an arm uses: arms that differ only at plan or run time share one
    arm "$1"
    local land; land=$(echo "$PLANENV" | sed -n 's/.*LANDING=\([a-z]*\).*/\1/p')
    local key="${land}_$(echo "$CAP" | tr -c 'a-z0-9\n' '_' | tr -s '_')"
    echo "cap_${key%_}${BLOCKS:+_b$BLOCKS}"
}

gpu_free() {
    local apps; apps=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i "$GPU" 2>/dev/null)
    [ -z "$apps" ] || die "GPU $GPU runs other processes ($apps): a timing needs an idle GPU (RUNSHEET_GPU.md §0)"
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
    # the device classes of the full extension (crates/pc-py/src/lib.rs:31-48); a session's own memory is its
    # `mem_report` and `fhe.mem_info()` = (free, total) bytes (crates/pc-py/src/fhe.rs:1307-1310), which the driver records
    ( cd "$W" && "$PY" -c "from perseus import _pc; import sys; want = ('Session', 'Program', 'KeySource', 'Client', 'PtStore'); miss = [n for n in want if not hasattr(_pc, n)]; print('perseus._pc loaded; device classes missing:', miss or 'none'); sys.exit(1 if miss else 0)" ) 2>&1 | tee -a "$OUT/session.log"
    [ "${PIPESTATUS[0]}" = 0 ] || die "_pc is not the full (CUDA) extension"
    nvidia-smi --query-gpu=index,name,memory.used,memory.free,memory.total --format=csv -i "$GPU" 2>&1 | tee -a "$OUT/session.log"
    ( cd "$W" && CUDA_VISIBLE_DEVICES="" "$PY" -m pytest -q -p no:cacheprovider -m 'not gpu' python/tests/test_fake_shadow.py \
        python/tests/test_pairing_cpu.py python/tests/test_rmsnorm_example.py python/tests/test_prefill_cpu.py > "$OUT/verify_upstream_tests.log" 2>&1 )
    rc=$?; tail -3 "$OUT/verify_upstream_tests.log" | tee -a "$OUT/session.log"
    [ "$rc" = 0 ] || die "upstream's CPU tests failed ($OUT/verify_upstream_tests.log)"
    if [ -d "$W/python/perseus/examples/fhe_ssm/tests" ]; then
        ( cd "$W" && CUDA_VISIBLE_DEVICES="" "$PY" -m pytest -q -p no:cacheprovider python/perseus/examples/fhe_ssm/tests > "$OUT/verify_port_tests.log" 2>&1 )
        rc=$?; tail -3 "$OUT/verify_port_tests.log" | tee -a "$OUT/session.log"
        [ "$rc" = 0 ] || die "the port's tests failed ($OUT/verify_port_tests.log)"
    fi
    bash "$HERE/check_additive.sh" | tee -a "$OUT/session.log"
    [ "${PIPESTATUS[0]}" = 0 ] || die "the branch changes an unlisted upstream file (tools/fhe_ssm/UPSTREAM_CHANGES.md)"
    bash "$HERE/compile_check/build.sh" | tee -a "$OUT/session.log"
    [ "${PIPESTATUS[0]}" = 0 ] || die "the compile check did not build (tools/fhe_ssm/compile_check/build.sh): plans could not be gated"
    log "verify: $(build_id)" ;;
bench)
    gpu_free
    ( cd "$W" && CUDA_VISIBLE_DEVICES=$GPU cargo run --release -p pc-tools --bin pc-bench -- bts --all --k 1,8 --reps 20 --keys packed ) \
        > "$OUT/bench_bts.txt" 2>&1 || { tail -20 "$OUT/bench_bts.txt"; die "pc-bench failed"; }
    tail -25 "$OUT/bench_bts.txt"; log "bench: $OUT/bench_bts.txt (set PLAN_BTS_MS from it for ILP_OBJECTIVE=ms); $(build_id)" ;;
arms)
    for a in $ARMS; do arm "$a"; printf '%-12s device[%s] capture[%s] plan[%s] run[%s] -> %s  %s\n' "$a" "$([ $DEVICE = 1 ] && echo yes || echo cpu-only)" "$CAP" "$PLANENV" "$RUNF" "$(capname "$a")" "$NOTE"; done ;;
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
    case "$PLANENV" in *LANDING=high*)
        "$PY" -c "import json, sys; b = json.load(open(sys.argv[1]))['bts']; sys.exit(0 if int(b.get('stc_first') or 0) == 2 and b.get('raise_drop') == '512:1' else 1)" "$c.part/capture_config.json" \
            || die "capture $a: capture_config.json lacks stc_first=2 / raise_drop=512:1, the capture did not take the LANDING=high switches (fakes.LANDING_SWITCHES; header comment); kept at $c.part" ;;
    esac
    rm -rf "$c"; mv "$c.part" "$c"
    log "capture $a: done in $(( $(date +%s) - t0 )) s" ;;
plan)
    a=${1:?arm}; arm "$a"; c="$OUT/$(capname "$a")"
    [ -d "$c" ] || die "no capture $c: run_gpu.sh capture $a"
    ( env $PLANENV PLAN_ROOT="$OUT" MEMGUARD_GB= bash "$HERE/make_plan.sh" "$c" "plan_$a" ) 2>&1 | tee "$OUT/plan_$a.log" | tail -8
    rc=${PIPESTATUS[0]}
    [ "$rc" = 0 ] || die "plan $a: make_plan.sh exit $rc (a PlanInfeasible block, a bad NAME, ...): $OUT/plan_$a.log"
    grep -q "compile check FAILED" "$OUT/bootstrap_placements/plan_$a/summary.txt" && die "plan $a does not bind (compile check)"
    grep -q "compile check: skipped" "$OUT/bootstrap_placements/plan_$a/summary.txt" && die "plan $a was not gated: the compile check is missing (run_gpu.sh verify builds it)"
    log "plan $a: $OUT/bootstrap_placements/plan_$a" ;;
prepare)
    a=${1:?arm}; arm "$a"; p="$OUT/bootstrap_placements/plan_$a"
    [ -f "$p/PLAN_CMD.txt" ] || die "no plan $p: run_gpu.sh plan $a"
    PYTHONPATH="$W/python" "$PY" -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('$DRIVER') else 1)" 2>/dev/null \
        || die "no driver module $DRIVER (RUNSHEET_GPU.md §4 is its contract)"
    resolve_data
    r="$OUT/prepare_$a"; mkdir -p "$r"
    log "prepare $a (CPU only): $CAPENV $DRIVER --plan $p --prepare-only $CAP $RUNF ${PREPARE_EXTRA:-}"
    t0=$(date +%s)
    ( cd "$W" && env CUDA_VISIBLE_DEVICES="" PYTHONPATH="$W/python" $CAPENV "$PY" -m "$DRIVER" $MODEL_ARGS --plan "$p" \
        --prepare-only ${PREPARE_EXTRA:-} $CAP $RUNF --out "$r/record.json" ) > "$r/prepare.log" 2>&1
    rc=$?
    log "prepare $a: exit $rc in $(( $(date +%s) - t0 )) s; $r/record.json"
    tail -8 "$r/prepare.log"
    [ $rc = 0 ] || die "prepare $a failed ($r/prepare.log)" ;;
run)
    a=${1:?arm}; arm "$a"; p="$OUT/bootstrap_placements/plan_$a"
    [ "$DEVICE" = 1 ] || die "run $a: under DIAG_SOURCE=values a diagonal linear does not run on the device path (344,064 resident plaintexts per tick, about 2.3 TiB; the driver refuses it): DIAG_SOURCE=generated runs it on the device diagonal encoder, or run_gpu.sh prepare $a runs its CPU part (RUNSHEET_GPU.md §3)"
    [ -f "$p/PLAN_CMD.txt" ] || die "no plan $p: run_gpu.sh plan $a"
    PYTHONPATH="$W/python" "$PY" -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('$DRIVER') else 1)" 2>/dev/null \
        || die "no driver module $DRIVER (RUNSHEET_GPU.md §4 is its contract)"
    [ -n "$KEYS" ] || die "run $a: set KEYS. inproc = in-process keys (INSECURE: the server generates every key and holds the secret; a timing run only, recorded as insecure-inproc) | bundle = the client's key directory in FHE_SSM_KEYS (run_gpu.sh keyspec $a, then keygen $a) | <that directory>"
    [ "$KEYS" != bundle ] || [ -d "${FHE_SSM_KEYS:-}" ] || die "KEYS=bundle reads the client's key directory from FHE_SSM_KEYS: export it (or KEYS=<dir>)"
    gpu_free
    resolve_data
    r="$OUT/run_$a"; mkdir -p "$r"
    echo "$(build_id)" > "$r/build_id.txt"
    nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,power.draw --format=csv -lms 500 -i "$GPU" > "$r/nvsmi.csv" 2>/dev/null &
    smi=$!
    trap 'kill "$smi" 2>/dev/null; wait "$smi" 2>/dev/null' EXIT       # the sampler dies with the step, whatever ends it
    log "run $a: $DRIVER --plan $p --ticks $TICKS --profile --profile-every $PROFILE_EVERY $CAP $RUNF --keys $KEYS"
    t0=$(date +%s)
    ( cd "$W" && env CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH="$W/python" $CAPENV "$PY" -m "$DRIVER" $MODEL_ARGS --plan "$p" --ticks "$TICKS" \
        --profile --profile-every "$PROFILE_EVERY" --keys "$KEYS" $CAP $RUNF --out "$r/record.json" ) > "$r/run.log" 2>&1
    rc=$?
    kill "$smi" 2>/dev/null; wait "$smi" 2>/dev/null; trap - EXIT
    peak=$(awk -F', ' 'NR>1 {gsub(/ MiB/,"",$2); if ($2+0>m) m=$2+0} END {print m+0}' "$r/nvsmi.csv")
    log "run $a: exit $rc in $(( $(date +%s) - t0 )) s; nvidia-smi peak ${peak} MiB; $r/record.json; $(build_id)"
    tail -15 "$r/run.log"
    [ $rc = 0 ] || die "run $a failed ($r/run.log)" ;;
keyspec)
    a=${1:?arm}; arm "$a"; p="$OUT/bootstrap_placements/plan_$a"
    [ -f "$p/PLAN_CMD.txt" ] || die "no plan $p: run_gpu.sh plan $a"
    resolve_data
    r="$OUT/keys_$a"; mkdir -p "$r"
    log "keyspec $a: $DRIVER --plan $p --keyspec $r/keyspec.json $CAP $RUNF"
    ( cd "$W" && env CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH="$W/python" $CAPENV "$PY" -m "$DRIVER" $MODEL_ARGS --plan "$p" \
        --keyspec "$r/keyspec.json" $CAP $RUNF ) > "$r/keyspec.log" 2>&1
    rc=$?
    log "keyspec $a: exit $rc; $r/keyspec.json"
    tail -5 "$r/keyspec.log"
    [ $rc = 0 ] || die "keyspec $a failed ($r/keyspec.log)" ;;
keygen)
    a=${1:?arm}; r="$OUT/keys_$a"
    [ -f "$r/keyspec.json" ] || die "no key request $r/keyspec.json: run_gpu.sh keyspec $a"
    log "keygen $a (the client role, on this box for the test): $DRIVER --keygen $r --keyspec-file $r/keyspec.json"
    ( cd "$W" && env CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH="$W/python" "$PY" -m "$DRIVER" --keygen "$r" \
        --keyspec-file "$r/keyspec.json" --threads "${KEYGEN_THREADS:-8}" ) > "$r/keygen.log" 2>&1
    rc=$?
    log "keygen $a: exit $rc; bundle $r/keys.pckb ($(du -h "$r/keys.pckb" 2>/dev/null | cut -f1)); for run: KEYS=bundle FHE_SSM_KEYS=$r"
    tail -5 "$r/keygen.log"
    [ $rc = 0 ] || die "keygen $a failed ($r/keygen.log)" ;;
smoke)
    export BLOCKS=1
    bash "$0" capture pcmm24 && bash "$0" plan pcmm24 && bash "$0" run pcmm24 ;;
smoke-diag)
    export BLOCKS=1 DIAG_SOURCE=generated
    bash "$0" capture ref && bash "$0" plan ref && bash "$0" run ref ;;
matrix)
    [ $# -gt 0 ] || set -- ref depth2 persite packstats high pair sbk16 emfuse pcmmtrunk24 pcmm24 best
    for a in "$@"; do
        arm "$a"; case "$NOTE" in BLOCKED*) log "skip $a: $NOTE"; continue ;; esac
        last=run; [ "$DEVICE" = 1 ] || last=prepare
        bash "$0" capture "$a" && bash "$0" plan "$a" && bash "$0" "$last" "$a" || log "arm $a stopped (above); continuing"
    done ;;
*)
    sed -n '2,36p' "$0"; exit 2 ;;
esac
