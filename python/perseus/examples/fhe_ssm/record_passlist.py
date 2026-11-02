# SPDX-License-Identifier: MIT
"""A serve run's token record as a pass list for the pass kernels: the tick-mode free run (``decode.FreeRun``) in
pass form. Every pass is the tick configuration (one live conversation per lane-slot, stream ``r<q>``); lane-slot q
feeds conversation q's prompt token at position p while its prompt lasts and ``GENERATED`` after it, which
``run_planned --pass-policy`` resolves with the record's decode policy, as FreeRun feeds the policy's pick back.

    python -m perseus.examples.fhe_ssm.record_passlist RECORD.json OUT.passes.json [--passes 148] [--lanes 64]
"""

from __future__ import annotations

import argparse
import json

from . import passmasks
from .decode import prompt_lengths, streams_from_record


def build(doc: dict, passes: int, lanes: int = 64) -> list[passmasks.PassConfig]:
    """``passes`` passes of ``lanes`` conversations from the record ``doc`` (``ids_lanes`` + ``generated_lanes``)."""
    streams = streams_from_record(doc)
    if len(streams) < lanes:
        raise ValueError(f"the record has {len(streams)} conversations, fewer than {lanes} lane-slots")
    prompts = prompt_lengths(doc, lanes)
    out = []
    for p in range(passes):
        entry = "fresh" if p == 0 else "carry"
        segs = tuple(passmasks.Segment("conv", (q, q + 1), q, entry, f"r{q}", None) for q in range(lanes))
        toks = tuple(int(streams[q][p]) if p < prompts[q] else passmasks.GENERATED for q in range(lanes))
        cfg = passmasks.PassConfig(segs, toks, f"rec.{p}")
        passmasks.validate(cfg)
        out.append(cfg)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("record")
    ap.add_argument("out")
    ap.add_argument("--passes", type=int, default=148)
    ap.add_argument("--lanes", type=int, default=64)
    a = ap.parse_args(argv)
    with open(a.record, encoding="utf-8") as f:
        doc = json.load(f)
    cfgs = build(doc, a.passes, a.lanes)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"passes": passmasks.dump_list(cfgs)}, f)
    gen = sum(t == passmasks.GENERATED for c in cfgs for t in c.tokens)
    print(
        f"{len(cfgs)} passes x {a.lanes} lane-slots; {gen} generated, {len(cfgs) * a.lanes - gen} prompt tokens -> {a.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
