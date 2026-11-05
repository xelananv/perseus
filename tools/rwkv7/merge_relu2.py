#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Gate 3's merge (results/rwkv7-fhe-20261104/new_pod/pod3_gate3.sh): BASE (the wide fit: every Chebyshev site at
margin 0.5, degree <= 255, 1/sqrt newton_const) with every ffn.relu2 entry taken from RELU2 (the margin-0.2 fit).

    python tools/rwkv7/merge_relu2.py BASE.json RELU2.json OUT.json
"""
import json
import sys


def merge(base: dict, relu: dict) -> int:
    n = 0
    for k in list(base["sites"]):
        if k.endswith("ffn.relu2"):
            base["sites"][k] = relu["sites"][k]
            n += 1
    base.setdefault("meta", {})["relu2_from"] = relu.get("meta", {})
    return n


if __name__ == "__main__":
    b, r = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
    n = merge(b, r)
    json.dump(b, open(sys.argv[3], "w"))
    print(f"merged {n} ffn.relu2 entries -> {sys.argv[3]}")
