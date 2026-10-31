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
#   bash tools/fhe_ssm/run_gpu.sh plan ARM         tools/fhe_ssm/make_plan.sh on that capture (+ the compile check); a plan
#                                                  already in $OUT (one shipped from elsewhere) is reused after its graph=
#                                                  md5 is checked against the capture (REPLAN=1 re-cuts it; an arm whose
#                                                  plan is shipped only, cfg-pcmm24-tick, is never cut here)
#   bash tools/fhe_ssm/run_gpu.sh prepare ARM      the driver's CPU part (--prepare-only: record, compile check, report)
#   bash tools/fhe_ssm/run_gpu.sh keyspec ARM      the key request of ARM's plan ($DRIVER --keyspec: the model's needs and,
#                                                  on the device build, the session's) -> $OUT/keys_ARM[_bN]/keyspec.json
#   bash tools/fhe_ssm/run_gpu.sh keygen ARM       the client role for that request ($DRIVER --keygen: a fresh secret,
#                                                  keys.pckb); then KEYS=bundle FHE_SSM_KEYS=$OUT/keys_ARM[_bN] for run
#   bash tools/fhe_ssm/run_gpu.sh run ARM          the planned run: $DRIVER --ticks $TICKS --profile, nvidia-smi sampled
#                                                  (diagonal linears only under DIAG_SOURCE=generated: the device
#                                                  diagonal encoder; else the PCMM arms only); the cfg-* arms run the pass
#                                                  kernels under --scenario $SCENARIO (tick: $TICKS passes)
#   bash tools/fhe_ssm/run_gpu.sh smoke            capture + plan + run of `pcmm24` at BLOCKS=1 (the first device contact)
#   bash tools/fhe_ssm/run_gpu.sh diag-tests       the device diagonal encoder's device checks in acceptance order: 1 the
#                                                  kernel selftest (encode / encode_many / diag_gather lines), 2 the crates'
#                                                  GPU tests (a)-(c), 3 the port's gpu tests (a, b, d); e (DIAG_E=1: DATA and
#                                                  the whole-model plan_ref) and f (DIAG_F=1); DIAG_ONLY=1,2,3,e,f picks steps;
#                                                  a passed step is skipped on the next call unless DIAG_RERUN=1
#   bash tools/fhe_ssm/run_gpu.sh pass-tests       the pass kernels' device checks (tests/test_pass_gpu.py: per-pass staging
#                                                  keeps the graphs, the replies per live lane-slot; a skip fails)
#   bash tools/fhe_ssm/run_gpu.sh smoke-diag       capture + plan + run of `ref` at BLOCKS=1 under DIAG_SOURCE=generated
#                                                  (the device diagonal encoder's first planned run; after diag-tests; KEYS
#                                                  defaults to inproc here: a contact test, recorded insecure-inproc)
#   bash tools/fhe_ssm/run_gpu.sh matrix [ARM...]  capture + plan + run (prepare for a CPU-only arm), one arm at a time
#
# Environment: OUT (default <worktree>/.cache/fhe_ssm/gpu_<UTC date>; every product lands there, one dir per arm; with
# BLOCKS=N every capture, plan, prepare, run and key directory of an arm carries the suffix _bN, so a smoke never
# replaces the whole model's plan), RUN_TAG (default empty: a further suffix _<tag> of every run, prepare and key
# directory, run_<arm>[_bN][_<tag>], so the runs of one bracket never overwrite each other; [A-Za-z0-9._-] only),
# SCENARIO (the cfg-* arms' --scenario: tick (default; $TICKS passes) | tick:K | prefill:T | doc:T:READERS | mixed |
# churn; a run flag, never a capture's),
# DATA (a directory holding the model's weights .npz, seeds .json and config .json; WEIGHTS / SEEDS / CONFIG override
# each), DRIVER (the planned driver's python module, default perseus.examples.fhe_ssm.run_planned: its contract is in
# RUNSHEET_GPU.md §4), TICKS (2), BLOCKS (all), GPU (0), KEYS (NO DEFAULT; run, smoke and matrix refuse without it:
# inproc = in-process keys, INSECURE, the server holds the secret, a timing run only | bundle = the client's key directory
# in FHE_SSM_KEYS, made by keyspec + keygen | that directory), PROFILE_EVERY (2: a stage fence on every 2nd tick, so one run
# holds fenced and unfenced ticks; 1 = every tick), PREPARE_EXTRA (more `prepare` flags, e.g. --lean on a small host),
# PY (<worktree>/.venv/bin/python), PERSEUS_CORE_ARCH (build: default from nvidia-smi's compute capability),
# PC_NVCC_JOBS (build: default 8), DIAG_SOURCE (values | generated, default values: where the diagonal linears'
# plaintexts come from on the device, the driver's --diag-source; generated = the device diagonal encoder, which makes
# every non-blocked arm a device arm; a driver flag only, so the captures and plans are the same under both), DIAG_E /
# DIAG_F (diag-tests: 1 = also run (e) the model shape and (f) the timings), REPLAN (1: plan re-cuts an existing plan).
# `run`, `prepare` and `keyspec` pass the driver --capture <the capture the arm's plan was cut from> when that directory
# is in $OUT (the plan's graph= line names the machine it was cut on, so without it the md5 check does not run).
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
SCENARIO=${SCENARIO:-tick}; RUN_TAG=${RUN_TAG:-}
[[ "$SCENARIO" =~ ^[a-z]+(:[0-9]+)*$ ]] || { echo "run_gpu: SCENARIO=$SCENARIO (tick[:K] | prefill:T | doc:T:READERS | mixed | churn)" >&2; exit 2; }
[[ -z "$RUN_TAG" || "$RUN_TAG" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "run_gpu: RUN_TAG=$RUN_TAG ([A-Za-z0-9._-] only)" >&2; exit 2; }
S=${BLOCKS:+_b$BLOCKS}   # the BLOCKS suffix of every product name (capname gives captures the same one)
T=${RUN_TAG:+_$RUN_TAG}  # the RUN_TAG suffix of every run, prepare and key directory
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
# The cfg-* arms run the pass kernels (python/perseus/examples/fhe_ssm/passmasks.py: a pass in any lane configuration).
# --pass-kernels on is a capture flag (the capture name carries it); the shipped captures of these arms are the merged
# ones (every configuration's magnitudes), the per-configuration ones beside them as <capname>__<label>. cfg-pcmm24-tick
# shares cfg-pcmm24's capture flags but its plan was cut on <capname>__tick alone (a pure-kernel control): shipped only.
ARMS="ref depth2 persite packstats high pair sbk16 emfuse pcmmtrunk24 pcmm24 best lockstep2 lockstep4 cfg-pcmm24 cfg-pcmm24-tick cfg-ref"
arm() {   # sets CAP, PLANENV, RUNF, NOTE, DEVICE, PLANCAP (the suffix of the capture its plan was cut from), SHIPPED
    CAP=""; PLANENV="LANDING=stock"; RUNF=""; NOTE=""; PLANCAP=""; SHIPPED=0
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
        cfg-pcmm24) CAP="--linear pcmm --pcmm-wbits 24 --pass-kernels on"; PLANENV="LANDING=high"; RUNF="--scenario $SCENARIO"
                    NOTE="the pass kernels on pcmm24 (the whole model's plan: the merged capture's; at BLOCKS=1 the tick configuration's capture); compare with pcmm24" ;;
        cfg-pcmm24-tick)
                    CAP="--linear pcmm --pcmm-wbits 24 --pass-kernels on"; PLANENV="LANDING=high"; RUNF="--scenario $SCENARIO"
                    PLANCAP="__tick"; SHIPPED=1
                    NOTE="cfg-pcmm24's kernels, plan cut on the tick configuration's capture alone (shipped, never cut here); compare with cfg-pcmm24" ;;
        cfg-ref)    CAP="--pass-kernels on"; RUNF="--scenario $SCENARIO"
                    NOTE="the pass kernels on the reference (diagonal linears: a device arm under DIAG_SOURCE=generated); compare with ref" ;;
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
plancap() {   # the capture an arm's plan was cut from: capname, or a per-configuration capture beside it (PLANCAP)
    local c; c=$(capname "$1"); arm "$1"
    echo "${c}${PLANCAP}"
}
capture_flag() {   # the driver's --capture for arm $1 when that capture is in $OUT (sets CAPF; logs either way)
    local c; c="$OUT/$(plancap "$1")"
    if [ -d "$c" ]; then CAPF="--capture $c"; else CAPF=""; log "$1: no capture $c here: the plan's capture md5 is not checked"; fi
}
md5_of() {   # cat block_*/graph.json | md5, as make_plan.sh writes it into PLAN_CMD.txt
    cat "$1"/block_*/graph.json | (md5sum 2>/dev/null || md5 -q) | cut -c1-32
}

cuda_env() {   # the build's CUDA environment for every step that compiles the kernels: build.rs re-runs nvcc whenever
               # PERSEUS_CORE_ARCH or PC_NVCC_JOBS differ from the last build's, so build, bench and diag-tests share one
    local cc; cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i "$GPU" 2>/dev/null | head -1 | tr -d '.')
    export PERSEUS_CORE_ARCH=${PERSEUS_CORE_ARCH:-${cc:-120}} PC_NVCC_JOBS=${PC_NVCC_JOBS:-8}
    export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.2}; export PATH="$CUDA_HOME/bin:$PATH"
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
    cuda_env
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
    cuda_env
    ( cd "$W" && CUDA_VISIBLE_DEVICES=$GPU cargo run --release -p pc-tools --bin pc-bench -- bts --all --k 1,8 --reps 20 --keys packed ) \
        > "$OUT/bench_bts.txt" 2>&1 || { tail -20 "$OUT/bench_bts.txt"; die "pc-bench failed"; }
    tail -25 "$OUT/bench_bts.txt"; log "bench: $OUT/bench_bts.txt (set PLAN_BTS_MS from it for ILP_OBJECTIVE=ms); $(build_id)" ;;
arms)
    for a in $ARMS; do
        arm "$a"
        pc=$(plancap "$a"); c=$(capname "$a")
        printf '%-15s device[%s] capture[%s] plan[%s] run[%s] -> %s%s  %s\n' "$a" "$([ $DEVICE = 1 ] && echo yes || echo cpu-only)" "$CAP" "$PLANENV" "$RUNF" "$c" "$([ "$pc" = "$c" ] || echo " (plan cut on $pc)")" "$NOTE"
    done ;;
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
    a=${1:?arm}; arm "$a"; c="$OUT/$(plancap "$a")"; p="$OUT/bootstrap_placements/plan_$a$S"
    if [ -f "$p/PLAN_CMD.txt" ] && [ "${REPLAN:-0}" != 1 ]; then
        # a plan in place (shipped): reused, never re-cut, once its graph= md5 is the capture's at hand
        want=$(sed -n 's/^graph=.*(md5 of the stages.* graph.json \([0-9a-f]*\)).*/\1/p' "$p/PLAN_CMD.txt")
        [ -d "$c" ] || die "plan $a: reusing $p needs its capture $c here (the md5 check)"
        got=$(md5_of "$c")
        [ -n "$want" ] && [ "$got" = "$want" ] || die "plan $a: $p was cut from a capture of md5 ${want:-?}, $c has $got"
        grep -q "check errors 0;" "$p/summary.txt" || die "plan $a: $p's summary holds no 'check errors 0;' line"
        log "plan $a: reusing $p (capture md5 $got; REPLAN=1 re-cuts it)"; exit 0
    fi
    [ "$SHIPPED" = 1 ] && die "plan $a: shipped only (cut elsewhere on $(plancap "$a")): never cut here; ship $p"
    [ -d "$c" ] || die "no capture $c: run_gpu.sh capture $a"
    ( env $PLANENV PLAN_ROOT="$OUT" MEMGUARD_GB= bash "$HERE/make_plan.sh" "$c" "plan_$a$S" ) 2>&1 | tee "$OUT/plan_$a$S.log" | tail -8
    rc=${PIPESTATUS[0]}
    [ "$rc" = 0 ] || die "plan $a: make_plan.sh exit $rc (a PlanInfeasible block, a bad NAME, ...): $OUT/plan_$a$S.log"
    grep -q "compile check FAILED" "$OUT/bootstrap_placements/plan_$a$S/summary.txt" && die "plan $a does not bind (compile check)"
    grep -q "compile check: skipped" "$OUT/bootstrap_placements/plan_$a$S/summary.txt" && die "plan $a was not gated: the compile check is missing (run_gpu.sh verify builds it)"
    log "plan $a: $OUT/bootstrap_placements/plan_$a$S" ;;
prepare)
    a=${1:?arm}; arm "$a"; p="$OUT/bootstrap_placements/plan_$a$S"
    [ -f "$p/PLAN_CMD.txt" ] || die "no plan $p: run_gpu.sh plan $a"
    PYTHONPATH="$W/python" "$PY" -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('$DRIVER') else 1)" 2>/dev/null \
        || die "no driver module $DRIVER (RUNSHEET_GPU.md §4 is its contract)"
    resolve_data
    capture_flag "$a"
    r="$OUT/prepare_$a$S$T"; mkdir -p "$r"
    log "prepare $a (CPU only): $CAPENV $DRIVER --plan $p --prepare-only $CAP $RUNF $CAPF ${PREPARE_EXTRA:-}"
    t0=$(date +%s)
    ( cd "$W" && env CUDA_VISIBLE_DEVICES="" PYTHONPATH="$W/python" $CAPENV "$PY" -m "$DRIVER" $MODEL_ARGS --plan "$p" \
        --prepare-only ${PREPARE_EXTRA:-} $CAP $RUNF $CAPF --out "$r/record.json" ) > "$r/prepare.log" 2>&1
    rc=$?
    log "prepare $a: exit $rc in $(( $(date +%s) - t0 )) s; $r/record.json"
    tail -8 "$r/prepare.log"
    [ $rc = 0 ] || die "prepare $a failed ($r/prepare.log)" ;;
run)
    a=${1:?arm}; arm "$a"; p="$OUT/bootstrap_placements/plan_$a$S"
    [ "$DEVICE" = 1 ] || die "run $a: under DIAG_SOURCE=values a diagonal linear does not run on the device path (344,064 resident plaintexts per tick, about 2.3 TiB; the driver refuses it): DIAG_SOURCE=generated runs it on the device diagonal encoder, or run_gpu.sh prepare $a runs its CPU part (RUNSHEET_GPU.md §3)"
    [ -f "$p/PLAN_CMD.txt" ] || die "no plan $p: run_gpu.sh plan $a"
    PYTHONPATH="$W/python" "$PY" -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('$DRIVER') else 1)" 2>/dev/null \
        || die "no driver module $DRIVER (RUNSHEET_GPU.md §4 is its contract)"
    [ -n "$KEYS" ] || die "run $a: set KEYS. inproc = in-process keys (INSECURE: the server generates every key and holds the secret; a timing run only, recorded as insecure-inproc) | bundle = the client's key directory in FHE_SSM_KEYS (run_gpu.sh keyspec $a, then keygen $a) | <that directory>"
    [ "$KEYS" != bundle ] || [ -d "${FHE_SSM_KEYS:-}" ] || die "KEYS=bundle reads the client's key directory from FHE_SSM_KEYS: export it (or KEYS=<dir>)"
    gpu_free
    resolve_data
    capture_flag "$a"
    r="$OUT/run_$a$S$T"; mkdir -p "$r"
    echo "$(build_id)" > "$r/build_id.txt"
    nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,power.draw --format=csv -lms 500 -i "$GPU" > "$r/nvsmi.csv" 2>/dev/null &
    smi=$!
    trap 'kill "$smi" 2>/dev/null; wait "$smi" 2>/dev/null' EXIT       # the sampler dies with the step, whatever ends it
    log "run $a${RUN_TAG:+ (tag $RUN_TAG)}: $DRIVER --plan $p --ticks $TICKS --profile --profile-every $PROFILE_EVERY $CAP $RUNF $CAPF --keys $KEYS"
    t0=$(date +%s)
    ( cd "$W" && env CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH="$W/python" $CAPENV "$PY" -m "$DRIVER" $MODEL_ARGS --plan "$p" --ticks "$TICKS" \
        --profile --profile-every "$PROFILE_EVERY" --keys "$KEYS" $CAP $RUNF $CAPF --out "$r/record.json" ) > "$r/run.log" 2>&1
    rc=$?
    kill "$smi" 2>/dev/null; wait "$smi" 2>/dev/null; trap - EXIT
    peak=$(awk -F', ' 'NR>1 {gsub(/ MiB/,"",$2); if ($2+0>m) m=$2+0} END {print m+0}' "$r/nvsmi.csv")
    log "run $a: exit $rc in $(( $(date +%s) - t0 )) s; nvidia-smi peak ${peak} MiB; $r/record.json; $(build_id)"
    tail -15 "$r/run.log"
    [ $rc = 0 ] || die "run $a failed ($r/run.log)" ;;
keyspec)
    a=${1:?arm}; arm "$a"; p="$OUT/bootstrap_placements/plan_$a$S"
    [ -f "$p/PLAN_CMD.txt" ] || die "no plan $p: run_gpu.sh plan $a"
    resolve_data
    capture_flag "$a"
    r="$OUT/keys_$a$S$T"; mkdir -p "$r"
    log "keyspec $a: $DRIVER --plan $p --keyspec $r/keyspec.json $CAP $RUNF $CAPF"
    ( cd "$W" && env CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH="$W/python" $CAPENV "$PY" -m "$DRIVER" $MODEL_ARGS --plan "$p" \
        --keyspec "$r/keyspec.json" $CAP $RUNF $CAPF --out "$r/keyspec_record.json" ) > "$r/keyspec.log" 2>&1
    rc=$?
    log "keyspec $a: exit $rc; $r/keyspec.json"
    tail -5 "$r/keyspec.log"
    [ $rc = 0 ] || die "keyspec $a failed ($r/keyspec.log)" ;;
keygen)
    a=${1:?arm}; r="$OUT/keys_$a$S$T"
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
diag-tests)
    # RUNSHEET_GPU.md §2a, in order, stopping at the first failure; logs under $OUT/diag_tests. Steps: 1 2 3 (default),
    # e (DIAG_E=1), f (DIAG_F=1), or the list in DIAG_ONLY (e.g. DIAG_ONLY=e,f). A passed step leaves <name>.passed and is
    # skipped on the next call unless DIAG_RERUN=1.
    gpu_free
    cuda_env
    # every module loaded at context creation (no lazy-load allocation after the first tick: the free-memory checks
    # compare ticks exactly); a test that cannot find the encoder's session fails instead of skipping
    export CUDA_MODULE_LOADING=EAGER FHE_SSM_REQUIRE_DEVICE=1
    r="$OUT/diag_tests"; mkdir -p "$r"
    only=${DIAG_ONLY:-}
    if [ -z "$only" ]; then
        only=1,2,3; [ "${DIAG_E:-0}" = 1 ] && only="$only,e"; [ "${DIAG_F:-0}" = 1 ] && only="$only,f"
    fi
    want() { case ",$only," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }
    skip_done() {   # skip_done <name>: true (and logged) when the step passed before and DIAG_RERUN is not 1
        [ -f "$r/$1.passed" ] && [ "${DIAG_RERUN:-0}" != 1 ] && { log "diag-tests $1: passed before ($r/$1.passed; DIAG_RERUN=1 reruns)"; return 0; }
        return 1
    }
    dfail() {   # dfail <name> <message>: the failing lines of the step's log, then stop
        grep -E 'FAILED|panicked at|Error|error\[' "$r/$1.log" 2>/dev/null | head -20 | tee -a "$OUT/session.log"
        die "diag-tests $1: $2 ($r/$1.log)"
    }
    dstep() {   # dstep <name> <ERE the log must hold> <cmd...>: run in the checkout on $GPU, log to $r/<name>.log
        local name=$1 expect=$2; shift 2
        log "diag-tests $name: $*"
        ( cd "$W" && env CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$W/python" "$@" ) > "$r/$name.log" 2>&1
        local rc=$?
        tail -4 "$r/$name.log" | tee -a "$OUT/session.log"
        [ $rc = 0 ] || dfail "$name" "exit $rc"
        grep -qE "$expect" "$r/$name.log" || dfail "$name" "exit 0 but no line matching '$expect' (a skipped or deselected test passes nothing)"
        touch "$r/$name.passed"
    }
    PYT=("$PY" -m pytest -q -rs -x -p no:cacheprovider -m gpu python/perseus/examples/fhe_ssm/tests/test_diag_gpu.py)
    # 1. the kernels against host references, every built instance (the default w32 build: u32 x Barrett64 and
    #    Barrett32, both with NTT): exit 0, the final "total mismatches 0" line, and per instance 3 encode, 4 encode_many
    #    and 2 diag_gather lines, every one at 0 mismatches (encode's lines are the refactored kernels' own host check)
    if want 1 && ! skip_done 1_selftest; then
        log "diag-tests 1_selftest: cargo run --release -p pc-kernels --bin pc-kernels-selftest -- --no-bench"
        ( cd "$W" && CUDA_VISIBLE_DEVICES="$GPU" cargo run --release -p pc-kernels --bin pc-kernels-selftest -- --no-bench ) > "$r/1_selftest.log" 2>&1
        rc=$?
        L="$r/1_selftest.log"
        grep -E '\| (encode |encode_many |diag_gather )' "$L" | tee -a "$OUT/session.log"
        inst=$(grep -cE '^\[selftest\] [^|]*: [0-9]+ checks, [0-9]+ mismatches$' "$L")
        nenc=$(grep -cE '\| encode [^|]*\| +[0-9]+ mismatches' "$L"); nem=$(grep -cE '\| encode_many [^|]*\| +[0-9]+ mismatches' "$L")
        ndg=$(grep -cE '\| diag_gather [^|]*\| +[0-9]+ mismatches' "$L")
        bad=$(grep -E '\| (encode |encode_many |diag_gather )' "$L" | grep -cvE '\| +0 mismatches of')
        [ $rc = 0 ] || dfail 1_selftest "exit $rc"
        grep -qE '^\[selftest\] instances .*: total mismatches 0$' "$L" || dfail 1_selftest "no final 'total mismatches 0' line"
        [ "$inst" -ge 1 ] && [ "$nenc" = $((3 * inst)) ] && [ "$nem" = $((4 * inst)) ] && [ "$ndg" = $((2 * inst)) ] && [ "$bad" = 0 ] \
            || dfail 1_selftest "$inst instance(s): $nenc encode, $nem encode_many, $ndg diag_gather lines (want 3, 4, 2 each), $bad not at 0"
        touch "$r/1_selftest.passed"
    fi
    # 2. (a) generated rows == eager DeviceFft of the host vector, (b) encode_many == single encodes, (c) fused sum == store loop
    if want 2 && ! skip_done 2_crates_abc; then
        dstep 2_crates_abc 'test result: ok\. 3 passed' cargo test --release -p pc-program --features gpu --test diag_gen_gpu -- \
            --ignored --test-threads=1 --nocapture --skip f_device_ms_per_plaintext
    fi
    # 3. the session's probe against the port's oracle (a, b), one block against the exact forward (d), the negative
    #    control (a perturbed tile must leave the tolerance), chunks of 1 and 128
    if want 3 && ! skip_done 3_port_abd; then
        dstep 3_port_abd '(^| )3 passed' "${PYT[@]}" -k "not model_shape"
    fi
    # e. the whole model, 2 ticks on the ref arm's whole-model stock plan (never the BLOCKS=1 smoke's plan_ref_b1)
    if want e && ! skip_done 5_port_e; then
        resolve_data
        p="$OUT/bootstrap_placements/plan_ref"
        [ -f "$p/PLAN_CMD.txt" ] || die "diag-tests (e): no whole-model plan $p (run_gpu.sh capture ref; run_gpu.sh plan ref, without BLOCKS)"
        wcap=$(BLOCKS=; capname ref)
        g=$(sed -n 's/^graph=\([^ ]*\).*/\1/p' "$p/PLAN_CMD.txt")
        [ "$(basename "$g")" = "$wcap" ] || die "diag-tests (e): $p was cut on $(basename "$g"), not the whole-model capture $wcap"
        nb=$(ls "$p"/block_*_placement.json 2>/dev/null | wc -l | tr -d ' ')
        [ "$nb" -ge 25 ] || die "diag-tests (e): $p holds $nb block plans; the model has 25 stages"
        dstep 5_port_e '(^| )1 passed' env FHE_SSM_WEIGHTS="$WEIGHTS" FHE_SSM_SEEDS="$SEEDS" FHE_SSM_CONFIG="$CONFIG" \
            FHE_SSM_PLAN="$p" FHE_SSM_E_OUT="$r/5_model_shape.json" "${PYT[@]}" -k model_shape
    fi
    # f. device ms per plaintext at B = 1, 32, 128 and 18 / 10 limbs
    if want f && ! skip_done 6_crates_f; then
        dstep 6_crates_f 'test result: ok\. 1 passed' cargo test --release -p pc-program --features gpu --test diag_gen_gpu -- \
            --ignored --exact f_device_ms_per_plaintext --nocapture
        grep "\[diag_gen\] (f)" "$r/6_crates_f.log" | tee -a "$OUT/session.log"
    fi
    log "diag-tests: steps $only passed ($r); $(build_id)" ;;
pass-tests)
    # the pass kernels on a device (RUNSHEET_GPU.md §3 step 2a): one block at d = 1,024, PCMM-24, LANDING=high, the tick,
    # mixed and doc scenarios; FHE_SSM_REQUIRE_DEVICE=1 (a test that cannot find the device fails, never skips) and
    # CUDA_MODULE_LOADING=EAGER (no lazy module load after the warm-up: the free-memory check is exact)
    gpu_free
    cuda_env
    r="$OUT/pass_tests"; mkdir -p "$r"
    log "pass-tests: pytest -m gpu python/perseus/examples/fhe_ssm/tests/test_pass_gpu.py"
    ( cd "$W" && env CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$W/python" CUDA_MODULE_LOADING=EAGER FHE_SSM_REQUIRE_DEVICE=1 \
        "$PY" -m pytest -q -rs -x -p no:cacheprovider -m gpu python/perseus/examples/fhe_ssm/tests/test_pass_gpu.py ) \
        > "$r/pass_tests.log" 2>&1
    rc=$?
    tail -4 "$r/pass_tests.log" | tee -a "$OUT/session.log"
    [ $rc = 0 ] || { grep -E 'FAILED|Error' "$r/pass_tests.log" | head -20 | tee -a "$OUT/session.log"; die "pass-tests failed ($r/pass_tests.log)"; }
    grep -qE '(^| )3 passed' "$r/pass_tests.log" || die "pass-tests: exit 0 but no '3 passed' line (a skipped or deselected test passes nothing)"
    log "pass-tests: passed ($r); $(build_id)" ;;
smoke-diag)
    export BLOCKS=1 DIAG_SOURCE=generated KEYS=${KEYS:-inproc}
    log "smoke-diag: KEYS=$KEYS (a contact test; inproc = in-process keys, recorded insecure-inproc)"
    bash "$0" capture ref && bash "$0" plan ref && bash "$0" run ref ;;
matrix)
    [ $# -gt 0 ] || set -- ref depth2 persite packstats high pair sbk16 emfuse pcmmtrunk24 pcmm24 best
    for a in "$@"; do
        arm "$a"; case "$NOTE" in BLOCKED*) log "skip $a: $NOTE"; continue ;; esac
        last=run; [ "$DEVICE" = 1 ] || last=prepare
        bash "$0" capture "$a" && bash "$0" plan "$a" && bash "$0" "$last" "$a" || log "arm $a stopped (above); continuing"
    done ;;
*)
    sed -n '2,/^set -uo pipefail/p' "$0" | sed '$d'; exit 2 ;;
esac
