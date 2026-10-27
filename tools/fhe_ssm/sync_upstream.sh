#!/bin/bash
# sync_upstream.sh — bring the fhe-ssm port up to upstream's latest `speedrun` and check it still holds.
#
#   bash tools/fhe_ssm/sync_upstream.sh            fetch, rebase, additive check, CPU _pc rebuild, the port's tests
#   DRY_RUN=1 bash tools/fhe_ssm/sync_upstream.sh  fetch only: report what the rebase would do, run the additive check
#
# Steps (each prints one line; the first failure stops the script with its own exit code):
#   1. refuse a dirty worktree (tracked edits): a rebase needs a clean tree; untracked files are left alone
#   2. git fetch origin speedrun                                     (the only network access)
#   3. git rebase origin/speedrun on BRANCH (default fhe-ssm-speedrun); on a conflict the rebase is ABORTED, the
#      conflicting paths are printed and the script exits 3. The branch is then exactly where it was.
#   4. tools/fhe_ssm/check_additive.sh origin/speedrun (exit 4 on a violation)
#   5. rebuild the CPU-only perseus._pc (tag, meta, shadow, config over pc-core / pc-shadow / pc-config) with the
#      research repository's tools/perseus_cpu_pc/build.sh, when the crates it compiles changed or the .so is missing
#      (FORCE_BUILD=1 always; SKIP_BUILD=1 never, e.g. on a GPU box where `maturin develop --release` builds the real
#      one) (exit 5)
#   6. python -m pytest -q python/perseus/examples/fhe_ssm/tests, skipped with a note while the package does not exist
#      (exit 6 on a failure)
#
# Environment: BRANCH (default fhe-ssm-speedrun), PY (default <worktree>/.venv/bin/python), CPU_PC_BUILD (default
# <worktree>/../../tools/perseus_cpu_pc/build.sh, i.e. the research repository's), MEMGUARD (default
# <worktree>/../../tools/memguard.sh; the tests run under it at MEMGUARD_GB, default 5, when it exists).
# Nothing is pushed. A rebase rewrites the branch's local commits; the public fork is never force-pushed by this script.
set -uo pipefail
W=$(git -C "$(dirname "$0")" rev-parse --show-toplevel) || exit 2
HERE="$W/tools/fhe_ssm"
BRANCH=${BRANCH:-fhe-ssm-speedrun}
PY=${PY:-$W/.venv/bin/python}
CPU_PC_BUILD=${CPU_PC_BUILD:-$W/../../tools/perseus_cpu_pc/build.sh}
MEMGUARD=${MEMGUARD:-$W/../../tools/memguard.sh}
MEMGUARD_GB=${MEMGUARD_GB:-5}
say() { echo "[sync] $*"; }
g() { git -C "$W" "$@"; }

# 1. clean tree, right branch
cur=$(g rev-parse --abbrev-ref HEAD)
[ "$cur" = "$BRANCH" ] || { say "on branch $cur, not $BRANCH: refusing (BRANCH=$cur to sync this one)"; exit 2; }
if ! g diff --quiet HEAD --; then
    say "tracked files have uncommitted edits: commit or set them aside first"; g diff --stat HEAD | tail -5; exit 2
fi
old_head=$(g rev-parse HEAD); old_base=$(g rev-parse origin/speedrun 2>/dev/null || echo none)

# 2. fetch
g fetch -q origin speedrun || { say "git fetch origin speedrun failed"; exit 2; }
new_base=$(g rev-parse origin/speedrun)
n_up=$( [ "$old_base" = none ] && echo "?" || g rev-list --count "$old_base..$new_base")
n_behind=$(g rev-list --count "HEAD..$new_base")
say "origin/speedrun ${old_base:0:7} -> ${new_base:0:7} ($n_up fetched; $n_behind upstream commit(s) not in $BRANCH yet)"

# 3. rebase
mb=$(g merge-base HEAD origin/speedrun)
if [ "${DRY_RUN:-0}" = 1 ]; then
    say "DRY_RUN: would $([ "$mb" = "$new_base" ] && echo "not rebase (already on origin/speedrun)" || echo "replay $(g rev-list --count "$mb..HEAD") local commit(s) from ${mb:0:7} onto ${new_base:0:7}"); upstream files changed: $(g diff --name-only "$mb" "$new_base" | wc -l | tr -d ' ')"
    bash "$HERE/check_additive.sh" origin/speedrun; exit $?
fi
if [ "$mb" = "$new_base" ]; then
    say "rebase: $BRANCH ${old_head:0:7} is already on origin/speedrun (no-op)"
else
    n_ours=$(g rev-list --count "$mb..HEAD")
    log=$(mktemp -t sync_rebase.XXXXXX)
    if g rebase -q origin/speedrun >"$log" 2>&1; then
        say "rebase: $n_ours local commit(s) replayed onto ${new_base:0:7}: ${old_head:0:7} -> $(g rev-parse --short HEAD)"
    else
        conflicts=$(g diff --name-only --diff-filter=U | tr '\n' ' ')
        g rebase --abort >/dev/null 2>&1
        say "rebase CONFLICT on: ${conflicts:-<see log>}; aborted, $BRANCH is back at $(g rev-parse --short HEAD)"
        tail -5 "$log"; rm -f "$log"
        exit 3
    fi
    rm -f "$log"
fi

# 4. additive
if bash "$HERE/check_additive.sh" origin/speedrun; then :; else say "the branch is not additive (above)"; exit 4; fi

# 5. CPU _pc
so="$W/python/perseus/_pc.abi3.so"
pc_src="crates/pc-py/src/tag.rs crates/pc-py/src/meta.rs crates/pc-py/src/shadow.rs crates/pc-py/src/config.rs
        crates/pc-core crates/pc-shadow crates/pc-config Cargo.toml Cargo.lock"
need_build=0
[ -f "$so" ] || need_build=1
[ "${FORCE_BUILD:-0}" = 1 ] && need_build=1
if [ "$need_build" = 0 ] && [ "$old_base" != none ] && ! g diff --quiet "$old_base" "$new_base" -- $pc_src; then need_build=1; fi
if [ "${SKIP_BUILD:-0}" = 1 ]; then
    say "CPU _pc: skipped (SKIP_BUILD=1)"
elif [ "$need_build" = 1 ]; then
    [ -f "$CPU_PC_BUILD" ] || { say "CPU _pc: no build script at $CPU_PC_BUILD (CPU_PC_BUILD=...)"; exit 5; }
    t0=$(date +%s)
    if bash "$CPU_PC_BUILD" "$W" "$PY"; then say "CPU _pc: rebuilt in $(( $(date +%s) - t0 )) s"
    else say "CPU _pc: build FAILED"; exit 5; fi
else
    say "CPU _pc: up to date (no change under the crates it compiles; FORCE_BUILD=1 to rebuild)"
fi

# 6. the port's tests
T="$W/python/perseus/examples/fhe_ssm/tests"
if [ ! -d "$T" ]; then
    say "tests: skipped (python/perseus/examples/fhe_ssm/tests does not exist yet)"
else
    guard=()
    [ -x "$MEMGUARD" ] && guard=("$MEMGUARD" "$MEMGUARD_GB")
    t0=$(date +%s)
    if (cd "$W" && CUDA_VISIBLE_DEVICES="" PYTHONPATH="$W/python" ${guard[@]+"${guard[@]}"} "$PY" -m pytest -q -p no:cacheprovider \
            python/perseus/examples/fhe_ssm/tests 2>&1 | tail -15; exit "${PIPESTATUS[0]}"); then
        say "tests: passed in $(( $(date +%s) - t0 )) s"
    else
        say "tests: FAILED (above)"; exit 6
    fi
fi
say "done: $BRANCH at $(g rev-parse --short HEAD) on origin/speedrun ${new_base:0:7}"
