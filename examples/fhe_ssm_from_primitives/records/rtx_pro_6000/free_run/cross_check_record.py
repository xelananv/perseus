"""Are the record's departures from the closed form (record_departures.out) the published run's own
encrypted pick flips? Reads that run's per-conversation, per-tick record (fidelity.json in the fhe-ssm
repository: encPick, refPick, pickAgree, generating per lane and tick) and compares, (tick, lane) by
(tick, lane), with the departures: same positions, the record's token = its encPick, the closed form's
pick = its refPick.

    python cross_check_record.py record_departures.out fidelity.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

dep = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
rows = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
rows = rows.get("rows", rows) if isinstance(rows, dict) else rows
flips = {(r["tick"], r["lane"]): r for r in rows if r.get("pickAgree") is False and r["generating"]}
ours = {(x["pos"], int(k)): x for k, v in dep["per_lane"].items() for x in v}
both = sorted(set(flips) & set(ours))
print(
    f"published run's generating pick flips: {len(flips)}; departures from the closed form: {len(ours)}; same (tick, lane): {len(both)}"
)
print(
    f"record token = its encPick: {sum(ours[p]['record'] == flips[p]['encPick'] for p in both)} of {len(both)}"
)
print(
    f"closed-form pick = its refPick: {sum(ours[p]['pick'] == flips[p]['refPick'] for p in both)} of {len(both)}"
)
other = sum(1 for r in rows if r.get("pickAgree") is False and not r["generating"])
print(f"published pick flips on prompt rows (nothing fed back): {other}")
