#!/bin/bash
# check_additive.sh — the fhe-ssm port changes upstream's `speedrun` branch only where tools/fhe_ssm/UPSTREAM_CHANGES.md
# says so.
#
#   bash tools/fhe_ssm/check_additive.sh [BASE]          (BASE default: origin/speedrun)
#
# The rule, for every path this branch changes since BASE (committed, staged or edited in the worktree):
#   1. a path under python/perseus/examples/fhe_ssm/ or tools/fhe_ssm/ is the port's own and must not exist on BASE;
#   2. any other path (an upstream file edited, renamed or deleted, or a new file elsewhere) is accepted only when it is
#      listed in tools/fhe_ssm/UPSTREAM_CHANGES.md (a table row whose first cell is the path in backticks).
# Listed paths the branch does not change are reported as stale rows. Untracked files outside the port's directories are
# warnings: they break the rule the moment they are committed (a plan under bootstrap_placements/, a capture in graphs/).
#
# Exit 0 = every change is the port's or listed; 1 = a violation (each one printed); 2 = usage / not a git checkout.
set -uo pipefail
W=$(git -C "$(dirname "$0")" rev-parse --show-toplevel 2>/dev/null) || { echo "check_additive: not in a git checkout" >&2; exit 2; }
BASE=${1:-origin/speedrun}
git -C "$W" rev-parse --verify -q "$BASE^{commit}" >/dev/null || { echo "check_additive: unknown base $BASE" >&2; exit 2; }
ALLOWED='^(python/perseus/examples/fhe_ssm/|tools/fhe_ssm/)'
LIST="$W/tools/fhe_ssm/UPSTREAM_CHANGES.md"
LISTED=$( [ -f "$LIST" ] && sed -n 's/^| *`\([^`]*\)` *|.*/\1/p' "$LIST" | sort -u )
bad=0; n_listed_used=0; USED=""

is_listed() { [ -n "$LISTED" ] && printf '%s\n' "$LISTED" | grep -Fxq -- "$1"; }

check_path() {   # check_path <origin> <status> <path>
    local origin=$1 st=$2 p=$3
    if [[ "$p" =~ $ALLOWED ]]; then
        if git -C "$W" cat-file -e "$BASE:$p" 2>/dev/null; then
            echo "VIOLATION ($origin, $st): $p is in the port's directories but exists on $BASE (an upstream file)"
            bad=1
        fi
    elif is_listed "$p"; then
        echo "listed ($origin, $st): $p (tools/fhe_ssm/UPSTREAM_CHANGES.md)"
        case " $USED " in *" $p "*) ;; *) USED="$USED $p"; n_listed_used=$((n_listed_used + 1)) ;; esac
    else
        echo "VIOLATION ($origin, $st): $p is outside the port's directories and not listed in tools/fhe_ssm/UPSTREAM_CHANGES.md"
        bad=1
    fi
}

# 1. committed: what HEAD changes since its merge base with BASE (three dots)
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
    if ! [[ "$p" =~ $ALLOWED ]] && ! is_listed "$p"; then
        echo "warning (untracked): $p is outside the port's directories and not listed; do not commit it"
        n_warn=$((n_warn + 1))
    fi
done < <(git -C "$W" ls-files --others --exclude-standard)

# 4. stale rows: listed, but nothing changes them
n_stale=0
for p in $LISTED; do
    case " $USED " in *" $p "*) ;; *)
        if ! git -C "$W" ls-files --others --exclude-standard -- "$p" | grep -q .; then
            echo "note: $p is listed in UPSTREAM_CHANGES.md but this branch does not change it (stale row?)"
            n_stale=$((n_stale + 1))
        fi ;;
    esac
done

head_s=$(git -C "$W" rev-parse --short HEAD); base_s=$(git -C "$W" rev-parse --short "$BASE")
mb=$(git -C "$W" merge-base HEAD "$BASE" | cut -c1-7)
echo "check_additive: HEAD $head_s, $BASE $base_s, merge base $mb; $n_commit committed path(s), $n_work worktree" \
     "edit(s), $n_listed_used listed upstream change(s), $n_stale stale row(s), $n_warn untracked warning(s):" \
     "$([ $bad = 0 ] && echo OK || echo VIOLATED)"
[ "$mb" = "$base_s" ] || echo "note: HEAD is not on top of $BASE (merge base $mb): run tools/fhe_ssm/sync_upstream.sh"
exit $bad
