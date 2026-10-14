"""Where must a faithful free run leave the published record? Teacher-forced through the float64 closed
form on the record's own tokens, per conversation: every generated position where the record's decode
policy, applied to the closed form's logits with the conversation's fed history, picks a different
token than the record has next, with the policy-modified top-two margin there. A run whose picks equal
the closed form's leaves the record at the first such position + 1 (gpu_run.py FreeRun.diverged_at).
One JSON object on stdout.

    python record_departures.py --weights W --seeds S --config C --record tokens.json
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
ap.add_argument("--weights", required=True)
ap.add_argument("--seeds", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--record", required=True, help="the published run's tokens.json")
a = ap.parse_args()

rec = json.loads(Path(a.record).read_text(encoding="utf-8"))
params = load_model(a.weights, a.seeds, a.config)
policy = Policy.from_record(rec)
streams = [p + g for p, g in zip(rec["ids_lanes"], rec["generated_lanes"], strict=True)]
prompts = [len(p) for p in rec["ids_lanes"]]
n, ticks = len(streams), min(len(s) for s in streams) - 1
emb = np.asarray(params.emb, dtype=np.float64)
exact = ExactModel(params, n)
diffs: dict[int, list[dict]] = {k: [] for k in range(n)}
t0 = time.time()
for pos in range(ticks):
    logits = exact.step(np.asarray(emb[[s[pos] for s in streams]], dtype=np.float64)) @ emb.T
    for k in range(n):
        if prompts[k] <= pos + 1 < len(streams[k]):
            nxt, hist = streams[k][pos + 1], streams[k][: pos + 1]
            p = policy.pick(logits[k], hist)
            if p != nxt:
                m = policy.modified(logits[k], hist)
                order = np.argsort(-m)
                diffs[k].append(
                    {
                        "pos": pos,
                        "diverged_at": pos + 1,
                        "pick": int(p),
                        "record": int(nxt),
                        "margin": round(float(m[order[0]] - m[order[1]]), 6),
                        "record_is_second": int(order[1]) == nxt,
                        "record_minus_pick": round(float(m[nxt] - m[p]), 6),
                    }
                )
first = [d[0]["diverged_at"] if d else None for d in diffs.values()]
print(
    json.dumps(
        {
            "policy": policy.as_dict(),
            "ticks": ticks,
            "lanes_with_a_departure": sum(f is not None for f in first),
            "departures": sum(len(d) for d in diffs.values()),
            "first_diverged_at": first,
            "per_lane": {k: d for k, d in diffs.items() if d},
            "seconds": round(time.time() - t0, 1),
        }
    )
)
