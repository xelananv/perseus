#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Does a capture (and the plan cut from it) belong to the current run settings? run_box.sh calls this before it
reuses a capture or a plan, and before every run step, so a directory left by another configuration is refused
instead of silently reused (the rwkv7 counterpart of fhe_ssm run_gpu.sh's --capture md5 check).

    python tools/rwkv7/capture_check.py CAPTURE --kind decode|pf|b|join --pth P --coeffs C --lanes 0 \\
        --layout current --blocks '' --batch 0 --batch-carry lane --prefill-chunk 0 --lora auto --shard-land 1 \\
        [--plan PLAN_DIR] [--sha-cache FILE]

Compared with the capture's rwkv7_capture.json: pth_sha256 and coeffs_sha256 (sha256 of the files now), lanes (LANES,
0 = layout.auto_lanes(width)), layout, blocks (BLOCKS, unset = the checkpoint's depth), batch (and the carry form),
prefill_chunk, lora (auto = fold at one shard, split above), shard_land (a 1-shard model never lands). With --plan,
the md5 of the capture's block_*/graph.json against the md5 the plan's PLAN_CMD.txt recorded (make_plan.sh's graph=
line): the plan was cut from this capture. A key the capture does not record (older captures, the join capture's
layout) is printed as unrecorded and not compared. rc 0 = consistent, 3 = mismatch (one line per field), 2 = no
capture. CAPTURE_MISMATCH_OK=1 prints the mismatches and returns 0.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python"))


def _sha256(path: str, cache: str | None) -> str:
    st = os.stat(path)
    key = f"{os.path.abspath(path)}|{st.st_size}|{int(st.st_mtime)}"
    memo = {}
    if cache and os.path.exists(cache):
        try:
            with open(cache) as f:
                memo = json.load(f)
        except (OSError, ValueError):
            memo = {}
    if key in memo:
        return memo[key]
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    memo[key] = h.hexdigest()
    if cache:
        try:
            with open(cache, "w") as f:
                json.dump(memo, f, indent=1)
        except OSError:
            pass
    return memo[key]


def graph_md5(cap: str) -> str:
    """md5 of the stages' graph.json concatenated in the shell's glob order (make_plan.sh: cat block_*/graph.json)."""
    names = sorted(d for d in os.listdir(cap) if d.startswith("block_") and os.path.isfile(os.path.join(cap, d, "graph.json")))
    h = hashlib.md5()
    for d in names:
        with open(os.path.join(cap, d, "graph.json"), "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def expected(a, meta: dict) -> dict:
    from perseus.examples.rwkv7.layout import auto_lanes
    exp = {}
    if a.pth:
        exp["pth_sha256"] = _sha256(a.pth, a.sha_cache)
    if a.coeffs:
        exp["coeffs_sha256"] = _sha256(a.coeffs, a.sha_cache)
    width = meta.get("width")
    if width is not None:
        exp["lanes"] = int(a.lanes) or auto_lanes(int(width))
    if a.kind != "join":
        exp["layout"] = a.layout
    if a.blocks not in (None, ""):
        exp["blocks"] = int(a.blocks)
    elif a.pth:
        try:
            from perseus.examples.rwkv7.ckpt_info import model_shapes, shapes
            exp["blocks"] = int(model_shapes(shapes(a.pth))["n_layer"])
        except Exception as e:                          # noqa: BLE001 (a checkpoint the shape reader does not know)
            print(f"  blocks: the checkpoint's depth is unreadable ({type(e).__name__}): not compared")
    K = int(a.batch or 0)
    if a.kind in ("b", "join"):
        exp["batch"] = K
    else:
        exp["batch"] = 0
    if a.kind == "b":
        exp["batch_carry"] = a.batch_carry
    exp["prefill_chunk"] = int(a.prefill_chunk or 0) if a.kind == "pf" else 0
    shards = meta.get("shards")
    if a.kind != "join" and shards is not None:
        exp["lora"] = ("fold" if int(shards) == 1 else "split") if a.lora == "auto" else a.lora
        exp["shard_land"] = (a.shard_land == "1") and int(shards) > 1
    return exp


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("--kind", required=True, choices=("decode", "pf", "b", "join"))
    ap.add_argument("--pth", default=None)
    ap.add_argument("--coeffs", default=None)
    ap.add_argument("--lanes", default="0")
    ap.add_argument("--layout", default="current")
    ap.add_argument("--blocks", default="")
    ap.add_argument("--batch", default="0")
    ap.add_argument("--batch-carry", default="lane")
    ap.add_argument("--prefill-chunk", default="0")
    ap.add_argument("--lora", default="auto")
    ap.add_argument("--shard-land", default="1")
    ap.add_argument("--plan", default=None, help="a plan dir cut from this capture: its PLAN_CMD.txt graph md5 is checked")
    ap.add_argument("--sha-cache", default=None, help="a JSON file memoising sha256 by path, size and mtime")
    a = ap.parse_args(argv)
    mp = os.path.join(a.capture, "rwkv7_capture.json")
    if not os.path.exists(mp):
        print(f"capture_check: no {mp}")
        return 2
    with open(mp) as f:
        meta = json.load(f)
    if a.kind == "join" and meta.get("kind") != "join":
        meta_kind_bad = f"kind: the capture is not a join capture ({meta.get('kind')!r})"
    elif a.kind != "join" and meta.get("kind") == "join":
        meta_kind_bad = "kind: the capture is a join capture"
    else:
        meta_kind_bad = None
    exp = expected(a, meta)
    bad, unrec = [], []
    if meta_kind_bad:
        bad.append(meta_kind_bad)
    for k, v in exp.items():
        if k not in meta:
            unrec.append(k)
            continue
        got = meta[k]
        if k == "batch":
            got = int(got or 0)
            if a.kind not in ("b", "join") and got <= 1:
                got = 0
        elif k == "prefill_chunk":
            got = int(got or 0)
        if got != v:
            bad.append(f"{k}: capture {got!r}, settings {v!r}")
    if a.plan:
        cmd = os.path.join(a.plan, "PLAN_CMD.txt")
        if os.path.exists(cmd):
            with open(cmd) as f:
                m = re.search(r"md5 of the stages' graph.json ([0-9a-f]{32})", f.read())
            if m is None:
                unrec.append("plan graph md5")
            else:
                g = graph_md5(a.capture)
                if g != m.group(1):
                    bad.append(f"plan: {a.plan} was cut from graphs with md5 {m.group(1)}, this capture's are {g}")
    if unrec:
        print(f"capture_check {a.capture}: unrecorded, not compared: {', '.join(unrec)}")
    if bad:
        for b in bad:
            print(f"capture_check {a.capture}: MISMATCH {b}")
        if os.environ.get("CAPTURE_MISMATCH_OK") == "1":
            print("capture_check: CAPTURE_MISMATCH_OK=1, continuing")
            return 0
        print("capture_check: refusing (a capture or plan of another configuration; remove it, change OUT, or set "
              "CAPTURE_MISMATCH_OK=1)")
        return 3
    print(f"capture_check {a.capture}: consistent ({len(exp) - len(unrec)} fields{' + plan md5' if a.plan else ''})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
