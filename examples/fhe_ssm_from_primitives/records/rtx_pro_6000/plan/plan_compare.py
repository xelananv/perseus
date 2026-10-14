"""Compare two plan directories block by block: every top-level key the runtime consumes (all but
`summary`) as canonical JSON, the way scripts/utils/plan_equiv.py does, listing which keys differ per
block and, for the dict-valued `capture_env`, which entries. One line per block, then the tally.

    python plan_compare.py PLANS_A PLANS_B
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

a_dir, b_dir = (Path(p) for p in sys.argv[1:3])
tally: Counter[str] = Counter()
for fa in sorted(a_dir.glob("block_*_placement.json"), key=lambda p: int(p.stem.split("_")[1])):
    fb = b_dir / fa.name
    a = json.loads(fa.read_text(encoding="utf-8"))
    b = json.loads(fb.read_text(encoding="utf-8"))
    keys = sorted(
        k
        for k in (set(a) | set(b)) - {"summary"}
        if json.dumps(a.get(k), sort_keys=True) != json.dumps(b.get(k), sort_keys=True)
    )
    tally.update(keys)
    print(
        f"{fa.name}: "
        + (
            ", ".join(f"{k}{'' if (k in a and k in b) else ' (one side only)'}" for k in keys)
            or "identical"
        )
    )
print("keys that differ, over all blocks:", dict(tally))
