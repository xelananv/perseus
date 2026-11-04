# SPDX-License-Identifier: MIT
"""The numpy reference (``ref.py``, float64) against the research harness's own poly model (torch, float32; logits
from ``tools/rwkv7/harness_logits.py``): top-1 agreement and the logits' relErrRms per position. Run on the box,
where the harness has torch; the two should agree to float32 rounding.

    python -m perseus.examples.rwkv7.run_crosscheck --pth P --coeffs C --ids ... --harness-logits L.npz --out X.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from . import common, ref


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    common.add_args(ap)
    ap.add_argument("--harness-logits", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    z = np.load(a.harness_logits)
    ids = [int(x) for x in z["ids"]]
    P = ref.load_params(a.pth, a.blocks)
    m = ref.RefModel(P, ref.Sites("poly", a.coeffs, clamp=True))
    rows = []
    for i, t in enumerate(ids):
        lr, _ = m.step(t)
        lh = z["logits"][i]
        rows.append({"pos": i, "token": t, "top1_ref": int(np.argmax(lr)), "top1_harness": int(np.argmax(lh)),
                     "logits_relErrRms": common.rel_rms(lr, lh)})
    doc = {"positions": len(rows), "top1_agree": sum(r["top1_ref"] == r["top1_harness"] for r in rows),
           "logits_relErrRms_max": max(r["logits_relErrRms"] for r in rows), "rows": rows}
    json.dump(doc, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in doc.items() if k != "rows"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
