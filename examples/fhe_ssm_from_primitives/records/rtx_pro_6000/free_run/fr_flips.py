"""Every pick flip of a free-running run, with its closed-form margin. Replays the float64 closed form
on the run's own fed tokens (the report's per-tick `tokens`), applies the run's decode policy with each
conversation's fed history, and lists each (pos, conversation) where the run's pick differs from the
closed form's, with the policy-modified top-two margin there, whether the run took the closed form's
second choice and whether the conversation was generating; the same for raw top-1 flips. The replay's
per-tick agreement must equal the run's recorded pick_agree_vs_exact and top1_agree_vs_exact.
One JSON object on stdout.

    python fr_flips.py fr_planned148.json --weights W --seeds S --config C
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from examples.fhe_ssm_from_primitives.decode import Policy
from examples.fhe_ssm_from_primitives.ref import ExactModel
from examples.fhe_ssm_from_primitives.weights import load_model

ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
ap.add_argument("report")
ap.add_argument("--weights", required=True)
ap.add_argument("--seeds", required=True)
ap.add_argument("--config", required=True)
a = ap.parse_args()

doc = json.loads(Path(a.report).read_text(encoding="utf-8"))
params = load_model(a.weights, a.seeds, a.config)
policy = Policy(**{k: (tuple(v) if k == "ban" else v) for k, v in doc["decode_policy"].items()})
prompts, ticks = doc["prompt_lengths"], doc["ticks"]
n = len(ticks[0]["tokens"])
emb = np.asarray(params.emb, dtype=np.float64)
exact = ExactModel(params, n)
hist: list[list[int]] = [[] for _ in range(n)]
pick_flips, top1_flips, mismatch = [], [], []
t0 = time.time()
for t in ticks:
    pos = t["pos"]
    for k in range(n):
        hist[k].append(int(t["tokens"][k]))
    logits = exact.step(np.asarray(emb[t["tokens"]], dtype=np.float64)) @ emb.T
    agree_p = agree_t = 0
    for k in range(n):
        m = policy.modified(logits[k], hist[k])
        o = np.argsort(-m)
        r = np.argsort(-logits[k])
        gen = prompts[k] <= pos + 1
        if t["pick"][k] == int(o[0]):
            agree_p += 1
        else:
            pick_flips.append(
                {
                    "pos": pos,
                    "conv": k,
                    "run": t["pick"][k],
                    "exact": int(o[0]),
                    "margin": round(float(m[o[0]] - m[o[1]]), 6),
                    "run_is_second": int(o[1]) == t["pick"][k],
                    "generating": gen,
                }
            )
        if t["top1"][k] == int(r[0]):
            agree_t += 1
        else:
            top1_flips.append(
                {
                    "pos": pos,
                    "conv": k,
                    "margin": round(float(logits[k][r[0]] - logits[k][r[1]]), 6),
                    "run_is_second": int(r[1]) == t["top1"][k],
                    "generating": gen,
                }
            )
    off_p = abs(agree_p / n - t["pick_agree_vs_exact"]) > 1e-12
    off_t = abs(agree_t / n - t["top1_agree_vs_exact"]) > 1e-12
    if off_p or off_t:
        mismatch.append({"pos": pos, "replay_pick": agree_p / n, "replay_top1": agree_t / n})
print(
    json.dumps(
        {
            "report": a.report,
            "mode": doc["mode"],
            "ticks": len(ticks),
            "conversations": n,
            "pick_flips": pick_flips,
            "top1_flips": top1_flips,
            "pick_flips_generating": sum(f["generating"] for f in pick_flips),
            "max_pick_flip_margin": max((f["margin"] for f in pick_flips), default=None),
            "max_top1_flip_margin": max((f["margin"] for f in top1_flips), default=None),
            "replay_mismatches": mismatch,
            "seconds": round(time.time() - t0, 1),
        }
    )
)
