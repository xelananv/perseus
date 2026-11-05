#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Planner refresh counts for one model size: capture (run_capture, the LANDING=high switches) + plan
(tools/fhe_ssm/make_plan.sh) of a 1-2 block model in decode, batched (K) and prefill (T) form, each in its own
process (one heavy process at a time), and the refreshes per stage of each plan (plan_cost). The per-size table of
python/perseus/examples/rwkv7/ANYSIZE_DESIGN.md comes from these JSON lines.

    # a real checkpoint (the box; the tokenizer from the harness)
    PYTHONPATH=python python tools/rwkv7/plan_sizes.py --pth M.pth --coeffs C.json --harness H --blocks 2 \\
        --modes decode,b4,b8,pf8 --out DIR
    # a synthetic one (synth.py; this Mac): --width W makes DIR/ckpt.npz and DIR/coeffs.json first
    PYTHONPATH=python python tools/rwkv7/plan_sizes.py --width 2048 --blocks 2 --modes decode,b4 --out DIR

Writes DIR/sizes.jsonl: one line per mode {width, shards, blocks, mode, stages: {block_b: {refreshes, ...}}}.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
HIGH = {"FIDESLIB_BTS_STC_FIRST": "2", "FIDESLIB_BTS_STC_FIRST_ROUTES": "dense", "FIDESLIB_BTS_STC_FIRST_CF": "13",
        "FIDESLIB_BTS_RAISE_DROP": "512:1", "AUTO_BTS_LEVEL": "44", "CUDA_VISIBLE_DEVICES": ""}


def mode_flags(mode: str) -> list[str]:
    if mode == "decode":
        return []
    if mode.startswith("b"):
        return ["--batch", mode[1:], "--batch-carry", "lane"]
    if mode.startswith("pf"):
        return ["--prefill-chunk", mode[2:]]
    raise SystemExit(f"mode {mode}: decode | b<K> | pf<T>")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pth", default=None)
    ap.add_argument("--coeffs", default=None)
    ap.add_argument("--harness", default=None)
    ap.add_argument("--width", type=int, default=None, help="a synthetic checkpoint of this width (synth.py)")
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--modes", default="decode")
    ap.add_argument("--lora", default="auto")
    ap.add_argument("--extra", default="", help="more run_capture flags (e.g. --no-shard-land)")
    ap.add_argument("--planenv", default="", help="extra plan env, e.g. 'PAIR=1'")
    ap.add_argument("--memguard-gb", default=os.environ.get("MEMGUARD_GB", "10"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    env = {**os.environ, **HIGH, "PYTHONPATH": str(ROOT / "python")}
    guard = [str(ROOT.parent.parent / "tools" / "memguard.sh"), str(a.memguard_gb)]
    if not pathlib.Path(guard[0]).exists():
        guard = []
    ids_flag: list[str] = []
    if a.width:
        ck, cj = out / "ckpt.npz", out / "coeffs.json"
        if not (ck.exists() and cj.exists()):
            subprocess.run([*guard, py, "-m", "perseus.examples.rwkv7.synth", "--width", str(a.width), "--layers",
                            str(a.blocks), "--calib-tokens", "96", "--out", str(ck), "--coeffs", str(cj)],
                           env=env, check=True)
        doc = json.load(open(cj))
        ids_flag = ["--ids", ",".join(str(t) for t in doc["meta"]["calib_tokens"])]
        a.pth, a.coeffs = str(ck), str(cj)
    if not (a.pth and a.coeffs):
        raise SystemExit("--pth and --coeffs, or --width")
    common = ["--pth", a.pth, "--coeffs", a.coeffs, "--blocks", str(a.blocks), "--lora", a.lora, *a.extra.split()]
    common += ["--harness", a.harness] if a.harness else ids_flag
    shifts = None
    rows = []
    for mode in [m for m in a.modes.split(",") if m]:
        cap = out / f"cap_{mode}_b{a.blocks}"
        t0 = time.time()
        if not (cap / "capture_config.json").exists():
            cmd = [*guard, py, "-m", "perseus.examples.rwkv7.run_capture", *common, *mode_flags(mode),
                   *(["--shifts", str(shifts)] if shifts else []), "--out", str(cap)]
            with open(out / f"capture_{mode}.log", "w") as lg:
                rc = subprocess.run(cmd, env=env, stdout=lg, stderr=subprocess.STDOUT).returncode
            if rc:
                print(json.dumps({"mode": mode, "step": "capture", "rc": rc, "log": str(out / f"capture_{mode}.log")}))
                return rc
        if mode == "decode" or shifts is None:
            shifts = cap / "pcmm_shifts.json"          # every mode's capture reads the decode capture's shifts
        t1 = time.time()
        name = f"plan_{mode}_b{a.blocks}" + "".join(f"_{kv.replace('=', '')}" for kv in a.planenv.split())
        plan = out / "bootstrap_placements" / name
        if not (plan / "PLAN_CMD.txt").exists():
            penv = {**env, "LANDING": "high", "PLAN_ROOT": str(out), "MEMGUARD_GB": "", "PY": py,
                    **dict(kv.split("=", 1) for kv in a.planenv.split())}
            with open(out / f"plan_{mode}.log", "w") as lg:
                rc = subprocess.run(["bash", str(ROOT / "tools/fhe_ssm/make_plan.sh"), str(cap), name], env=penv,
                                    stdout=lg, stderr=subprocess.STDOUT).returncode
            if rc:
                print(json.dumps({"mode": mode, "step": "plan", "rc": rc, "log": str(out / f"plan_{mode}.log")}))
                return rc
        sys.path.insert(0, str(ROOT / "python"))
        from perseus.examples.rwkv7.plan_cost import census, plan_refreshes
        meta = json.load(open(cap / "rwkv7_capture.json"))
        rep = plan_refreshes(str(cap), str(plan))
        row = {"width": meta.get("width"), "shards": meta.get("shards"), "lanes": meta.get("lanes"), "blocks": a.blocks,
               "mode": mode,
               "lora": meta.get("lora"), "shard_land": meta.get("shard_land"), "planenv": a.planenv,
               "pth": a.pth, "coeffs": a.coeffs, "total": rep["total"], "stages": rep["stages"],
               "census_block0": census(str(cap), str(plan), 0),
               "census_block1": census(str(cap), str(plan), 1) if a.blocks > 1 else None,
               "capture_s": round(t1 - t0, 1), "plan_s": round(time.time() - t1, 1)}
        rows.append(row)
        with open(out / "sizes.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps({k: row[k] for k in ("width", "shards", "mode", "total", "capture_s", "plan_s")} |
                         {"per_stage": {k: v["refreshes"] for k, v in row["stages"].items()}}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
