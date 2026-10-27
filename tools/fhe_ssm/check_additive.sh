#!/bin/bash
# check_additive.sh — the fhe-ssm port is purely additive to upstream's `speedrun` branch.
#
#   bash tools/fhe_ssm/check_additive.sh [BASE]          (BASE default: origin/speedrun)
#
# The rule: every path this branch adds since BASE (committed, staged or modified in the worktree)
#   1. lies under python/perseus/examples/fhe_ssm/ or tools/fhe_ssm/, and
#   2. does not exist on BASE (no upstream file is edited, renamed, deleted or shadowed).
# Untracked files outside those directories are listed as warnings: they break the rule the moment they are
# committed (a plan written to bootstrap_placements/<name>, a capture under graphs/).
#
# Exit 0 = additive; 1 = a violation (each one printed); 2 = usage / not a git checkout / BASE unknown.
set -uo pipefail
W=$(git -C "$(dirname "$0")" rev-parse --show-toplevel 2>/dev/null) || { echo "check_additive: not in a git checkout" >&2; exit 2; }
BASE=${1:-origin/speedrun}
git -C "$W" rev-parse --verify -q "$BASE^{commit}" >/dev/null || { echo "check_additive: unknown base $BASE" >&2; exit 2; }
ALLOWED='^(python/perseus/examples/fhe_ssm/|tools/fhe_ssm/)'
bad=0

check_path() {   # check_path <origin> <status> <path>
    local origin=$1 st=$2 p=$3
    if ! [[ "$p" =~ $ALLOWED ]]; then
        echo "VIOLATION ($origin, $st): $p is outside python/perseus/examples/fhe_ssm/ and tools/fhe_ssm/"
        bad=1
    elif git -C "$W" cat-file -e "$BASE:$p" 2>/dev/null; then
        echo "VIOLATION ($origin, $st): $p exists on $BASE (an upstream file)"
        bad=1
    fi
}

# 1. committed: what HEAD adds since its merge base with BASE (three dots)
n_commit=0
while IFS=$'\t' read -r st a b; do
    [ -n "$st" ] || continue
    n_commit=$((n_commit + 1))
    check_path committed "$st" "$a"
    [ -n "${b:-}" ] && check_path committed "$st" "$b"      # a rename / copy names both sides
done < <(git -C "$W" diff --name-status -M "$BASE...HEAD")

# 2. staged and unstaged edits of tracked files
n_work=0
while IFS=$'\t' read -r st a b; do
    [ -n "$st" ] || continue
    n_work=$((n_work + 1))
    check_path worktree "$st" "$a"
    [ -n "${b:-}" ] && check_path worktree "$st" "$b"
done < <(git -C "$W" diff --name-status -M HEAD)

# 3. untracked, not ignored: warnings only
n_warn=0
while IFS= read -r p; do
    [ -n "$p" ] || continue
    if ! [[ "$p" =~ $ALLOWED ]]; then
        echo "warning (untracked): $p is outside the port's directories; do not commit it"
        n_warn=$((n_warn + 1))
    fi
done < <(git -C "$W" ls-files --others --exclude-standard)

head_s=$(git -C "$W" rev-parse --short HEAD); base_s=$(git -C "$W" rev-parse --short "$BASE")
mb=$(git -C "$W" merge-base HEAD "$BASE" | cut -c1-7)
echo "check_additive: HEAD $head_s, $BASE $base_s, merge base $mb; $n_commit committed path(s), $n_work worktree" \
     "edit(s), $n_warn untracked warning(s): $([ $bad = 0 ] && echo ADDITIVE || echo VIOLATED)"
[ "$mb" = "$base_s" ] || echo "note: HEAD is not on top of $BASE (merge base $mb): run tools/fhe_ssm/sync_upstream.sh"
exit $bad
