# SPDX-License-Identifier: MIT
"""Print the checkpoint's shapes and the coefficient JSON's sites (degree / Newton steps, interval, error).

    python -m perseus.examples.rwkv7.inspect_coeffs --coeffs .cache/rwkv7/coeffs_mixed.json [--pth CKPT]
"""
from __future__ import annotations

import argparse
import json


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coeffs", required=True)
    ap.add_argument("--pth", default=None)
    ap.add_argument("--blocks", default="0,1,11")
    a = ap.parse_args(argv)
    c = json.load(open(a.coeffs))
    print("meta", {k: v for k, v in c["meta"].items() if k != "site_degree"})
    want = tuple(f"blocks.{b}." for b in a.blocks.split(","))
    for n, s in c["sites"].items():
        if not n.startswith(want) and n.startswith("blocks."):
            continue
        row = {k: (len(v) if isinstance(v, list) else v) for k, v in s.items() if k not in ("kind", "short")}
        print(n, row)
    if a.pth:
        from .ckpt import load_state_dict
        sd = load_state_dict(a.pth)
        for k, v in sd.items():
            if k.startswith(("blocks.0.", "emb", "head", "ln_out")) or k.startswith("blocks.1.att.v"):
                print(k, tuple(v.shape), v.dtype)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
