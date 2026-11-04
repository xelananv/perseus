# SPDX-License-Identifier: MIT
"""Run by a Python WITH torch (the research harness's environment), not the perseus venv: the harness's own
RWKV-7 (results/vector-d-20261104/harness/rwkv7_model.py) in poly mode, clamp on, float32 on CPU, GPT mode over the
token list; writes the logits of every position to an .npz. run_box.sh's `crosscheck` step compares them with the
numpy reference the encrypted run is scored against (perseus.examples.rwkv7.run_crosscheck).

    TORCH_PY harness_logits.py --harness H --pth P --coeffs C --ids 261,... --out logits.npz
"""
import argparse
import json
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness", required=True)
    ap.add_argument("--pth", required=True)
    ap.add_argument("--coeffs", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.harness)
    import torch
    from rwkv7_model import Sites, load_rwkv7
    sites = Sites("poly")
    sites.load_poly(json.load(open(a.coeffs)), clamp=True)
    model = load_rwkv7(a.pth, device="cpu", dtype=torch.float32, sites=sites)
    ids = torch.tensor([[int(x) for x in a.ids.split(",")]])
    with torch.no_grad():
        logits = model(ids)[0].double().numpy()
    np.savez(a.out, logits=logits, ids=ids.numpy()[0])
    print(f"harness poly logits {logits.shape} -> {a.out}")


if __name__ == "__main__":
    main()
