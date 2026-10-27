#!/bin/bash
# The model-shape arms (24 blocks, ring 2^16, 64 split pairs, the published weights), one at a time, each in its own
# process under a memory guard, 2 ticks each; the reports and logs land in OUT. Runs on bash 3.2 (macOS) upwards.
#   bash python/perseus/examples/fhe_ssm/tests/model_shape_runs.sh OUT [ARM ...]
# Environment: FHE_SSM_WEIGHTS, FHE_SSM_SEEDS, FHE_SSM_CONFIG (the npz / seeds / config paths), MEMGUARD (a wrapper
# such as "tools/memguard.sh 3"; empty = none), MEM_PROBE (a probe script run before each arm; empty = none),
# PYTHON (default .venv/bin/python of this checkout), TICKS (default 2).
set -euo pipefail
OUT="${1:?output dir}"; shift || true
PKG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$PKG/../../../.." && pwd)"
PYTHON="${PYTHON:-$REPO/.venv/bin/python}"
W="${FHE_SSM_WEIGHTS:?}"; S="${FHE_SSM_SEEDS:?}"; C="${FHE_SSM_CONFIG:?}"
TICKS="${TICKS:-2}"
mkdir -p "$OUT"
flags_of() {  # arm name -> its flags (every other flag at its default: 24 blocks, filled split pairs, shipped scales)
  case "$1" in
    ref_fork_hint1) echo "--diag-hint 1" ;;
    default) echo "" ;;
    depth2) echo "--newton-form depth2" ;;
    per_site) echo "--newton-sites per-site" ;;
    pack_stats) echo "--pack-stats halves" ;;
    pcmm_trunk24) echo "--linear pcmm-trunk --pcmm-wbits 24" ;;
    pcmm16) echo "--linear pcmm" ;;
    lane_mix) echo "--lane-mix random" ;;
    levels_preset) echo "--levels preset" ;;
    rules_stock) echo "--rules stock" ;;
    depth2_hint1) echo "--newton-form depth2 --diag-hint 1" ;;
    per_site_hint1) echo "--newton-sites per-site --diag-hint 1" ;;
    pack_stats_hint1) echo "--pack-stats halves --diag-hint 1" ;;
    pcmm_trunk24_hint1) echo "--linear pcmm-trunk --pcmm-wbits 24 --diag-hint 1" ;;
    pcmm16_hint1) echo "--linear pcmm --diag-hint 1" ;;
    lane_mix_hint1) echo "--lane-mix random --diag-hint 1" ;;
    *) echo "unknown arm: $1" >&2; exit 2 ;;
  esac
}
ORDER="ref_fork_hint1 default depth2 per_site pack_stats pcmm_trunk24 pcmm16 lane_mix levels_preset rules_stock"
[ $# -gt 0 ] && ORDER="$*"
cd "$REPO"
for arm in $ORDER; do
  flags="$(flags_of "$arm")"
  echo "== $arm: $flags  ($(date '+%F %T'))" | tee -a "$OUT/RUNS.log"
  if [ -n "${MEM_PROBE:-}" ]; then bash "$MEM_PROBE" | tee -a "$OUT/RUNS.log" || true; fi
  # shellcheck disable=SC2086
  if PYTHONPATH="$REPO/python" ${MEMGUARD:-} "$PYTHON" -m perseus.examples.fhe_ssm.run_fake \
      --weights "$W" --seeds "$S" --config "$C" --tokens "$TICKS" --blocks 24 $flags --out "$OUT/$arm.json" \
      > "$OUT/$arm.log" 2>&1; then
    tail -n 1 "$OUT/$arm.log" | tee -a "$OUT/RUNS.log"
  else
    echo "FAILED $arm (exit $?)" | tee -a "$OUT/RUNS.log"
    tail -n 5 "$OUT/$arm.log" | tee -a "$OUT/RUNS.log"
  fi
done
