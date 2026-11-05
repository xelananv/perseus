#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""The FINAL_PLAN.md record of every run in a size_pipeline OUT dir (results/rwkv7-fhe-20261104/new_pod/FINAL_PLAN.md,
"Records for each run"): per-token (or per-pass) seconds, top-1 against the poly model, logits relErrRms median and
max (R3), the tokens whose plaintext poly reference left a site's interval, setup seconds, the keys (insecure-inproc
or bundle, has_secret), the generated ids, and the sampler's window (GPU util, power, VRAM, CPU busy, RAM) between the
step's start and end lines in pipeline.log.

    python tools/rwkv7/run_records.py OUT [--sampler OUT/sampler.csv] [--out OUT/records.json]
"""
import argparse
import csv
import glob
import json
import os
import re
import statistics as st


def windows(log_path):
    """{step name: (start ts, end ts)} from pipeline.log's 'step NAME: ...' and 'step NAME rc=' lines."""
    out, start = {}, {}
    if not os.path.exists(log_path):
        return out
    for line in open(log_path):
        m = re.match(r"\[size_pipeline \S+ (\S+)\] step (\S+?)(:| rc=)", line)
        if not m:
            continue
        ts, name, kind = m.groups()
        if kind == ":":
            start[name] = ts
        elif name in start:
            out[name] = (start[name], ts)
    return out


def sampler_window(path, t0, t1):
    if not path or not os.path.exists(path):
        return None
    rows = [r for r in csv.DictReader(open(path)) if t0 <= r["ts"] <= t1]

    def col(k):
        v = [float(r[k]) for r in rows if r.get(k) not in (None, "")]
        return v
    if not rows:
        return {"rows": 0}
    g, p, vr, c, ram = (col(k) for k in ("gpu_util", "power_w", "vram_mib", "cpu_busy_pct", "ram_used_mib"))
    return {"rows": len(rows), "gpu_util_mean": round(st.mean(g), 1) if g else None,
            "power_mean_w": round(st.mean(p), 0) if p else None, "power_max_w": max(p) if p else None,
            "vram_peak_mib": max(vr) if vr else None, "cpu_busy_mean": round(st.mean(c), 1) if c else None,
            "ram_peak_mib": max(ram) if ram else None}


def record(path):
    head, summ = None, None
    for line in open(path):
        r = json.loads(line)
        if r.get("kind") == "header":
            head = r
        elif r.get("kind") == "summary":
            summ = r
    if summ is None:
        return {"file": path, "complete": False}
    keep = ("tokens", "top1_agree", "top1_agree_prefill", "logits_relErrRms_median", "logits_relErrRms_max",
            "seconds_per_token_median", "seconds_per_token_after_first", "prefill_seconds_per_token",
            "seconds_per_pass_median", "aggregate_tokens_per_s_full", "aggregate_tokens_per_s",
            "poly_sites_out_of_range_tokens", "setup_s", "wall_s", "generated_ids", "batch", "passes")
    out = {"file": path, "complete": True, **{k: summ[k] for k in keep if k in summ}}
    if head:
        out.update({k: head.get(k) for k in ("keys", "has_secret", "input_encryption", "width", "shards", "blocks",
                                             "schedule", "pcmm_cf", "pth_sha256", "coeffs_sha256")})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--sampler", default=None)
    ap.add_argument("--json-out", "--out", dest="json_out", default=None)
    a = ap.parse_args(argv)
    win = windows(os.path.join(a.out, "pipeline.log"))
    recs = []
    for f in sorted(glob.glob(os.path.join(a.out, "run*.jsonl")) + glob.glob(os.path.join(a.out, "batch*.jsonl"))):
        r = record(f)
        tag = re.sub(r"\.jsonl$", "", os.path.basename(f))
        step = next((s for s in win if s.lower().endswith(tag.split("_")[-1].lower())), None)
        r["step"] = step
        if step:
            r["window"] = win[step]
            r["sampler"] = sampler_window(a.sampler, *win[step])
        recs.append(r)
    text = json.dumps(recs, indent=1)
    print(text)
    if a.json_out:
        open(a.json_out, "w").write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
