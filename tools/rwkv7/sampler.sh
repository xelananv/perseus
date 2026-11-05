#!/bin/bash
# sampler.sh OUT.csv [PERIOD_S]: one row per period (GPU util, power, VRAM; CPU busy %, RAM used MiB) until killed.
# The pipeline starts it before the first GPU step and kills it after the last; every run's window is read from it.
O=$1; P=${2:-5}
echo "ts,gpu_util,power_w,vram_mib,cpu_busy_pct,ram_used_mib" > "$O"
prev=($(head -1 /proc/stat 2>/dev/null | awk '{print $2+$3+$4+$7+$8, $5+$6}'))
while true; do
    g=$(nvidia-smi --query-gpu=utilization.gpu,power.draw,memory.used --format=csv,noheader,nounits -i "${GPU:-0}" 2>/dev/null | head -1 | tr -d ' ')
    cur=($(head -1 /proc/stat 2>/dev/null | awk '{print $2+$3+$4+$7+$8, $5+$6}'))
    busy=""; if [ -n "${cur[0]:-}" ] && [ -n "${prev[0]:-}" ]; then
        db=$(( cur[0]-prev[0] )); di=$(( cur[1]-prev[1] )); [ $((db+di)) -gt 0 ] && busy=$(( 100*db/(db+di) )); fi
    prev=("${cur[@]}")
    ram=$(free -m 2>/dev/null | awk '/Mem:/{print $3}')
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ),${g:-,,},${busy},${ram}" >> "$O"
    sleep "$P"
done
