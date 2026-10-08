# SPDX-License-Identifier: MIT
"""Calibrate the power-of-two scale folds (scales.py) on token streams; write them as JSON.

    python -m examples.fhe_ssm_from_primitives.calibrate --weights W.npz --seeds S.json \\
        --config C.json --tokens tokens.json [--calibrate 0:74] [--validate 74:148] --out scales.json
    python -m examples.fhe_ssm_from_primitives.calibrate ... --random 64x32 --out scales.json

``--tokens``: a serve run's token record, each conversation's prompt then its generation
(``{"ids_lanes": [...], "generated_lanes": [...]}``), or ``{"streams": [[...], ...]}``.
``--random CxT``: C random streams of T tokens instead. The closed-form forward runs every
stream teacher-forced; the exponents come from the maxima over the ``--calibrate`` ticks, and
the ``--validate`` ticks (held out) report how much of the margin kappa each block keeps:
``kept = top / (2^e * max)``, which the exponent rule makes at least kappa on calibration.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from . import scales
from .weights import load_model


def streams_from_record(doc: dict) -> list[list[int]]:
    """One token id list per conversation from a serve run's token record."""
    if "streams" in doc:
        return [[int(t) for t in s] for s in doc["streams"]]
    ids, gen = doc.get("ids_lanes"), doc.get("generated_lanes")
    if not ids or gen is None or len(ids) != len(gen):
        raise ValueError("a token record needs 'streams', or 'ids_lanes' and 'generated_lanes'")
    return [[int(t) for t in a + b] for a, b in zip(ids, gen)]


def window(text: str) -> tuple[int, int]:
    a, sep, b = text.partition(":")
    if not sep:
        raise argparse.ArgumentTypeError(f"{text!r}: a window is START:STOP")
    return int(a), int(b)


def kept_margin(
    maxima: list[dict[str, float]], chosen: list[scales.BlockScales], top: float
) -> list[dict[str, float]]:
    """Per block and family, top / (2^e * max): the margin left on these maxima."""
    out = []
    for m, s in zip(maxima, chosen):
        row = {}
        for fam, names, e in (
            ("gate_tm", scales.GATE_TM_VALUES, s.gate_tm),
            ("act_cm", scales.ACT_CM_VALUES, s.act_cm),
            ("gate_cm", scales.GATE_CM_VALUES, s.gate_cm),
            ("ag", (scales.PRODUCT_CM,), s.act_cm + s.gate_cm),
        ):
            v = max((m.get(k, 0.0) for k in names), default=0.0)
            row[fam] = float("inf") if v <= 0 else top / (2**e * v)
        out.append(row)
    return out


def calibrate(params, streams, cal, val=None, kappa=scales.KAPPA, top=scales.TOP):
    """(exponents per block, report) from the maxima over the calibration window."""
    cal_max = scales.measure(params, streams, *cal)
    chosen = scales.choose(cal_max, kappa, top)
    report: dict = {
        "kappa": kappa,
        "top": top,
        "calibrate_ticks": list(cal),
        "conversations": len(streams),
        "calibration_maxima": cal_max,
        "calibration_kept": kept_margin(cal_max, chosen, top),
    }
    if val is not None:
        val_max = scales.measure(params, streams, *val)
        kept = kept_margin(val_max, chosen, top)
        report.update(validate_ticks=list(val), validation_maxima=val_max, validation_kept=kept)
        report["validation_min_kept"] = min(min(r.values()) for r in kept)
    return chosen, report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--weights", required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--config", required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--tokens", help="a serve run's token record (JSON)")
    src.add_argument("--random", help="CxT: C random streams of T tokens")
    ap.add_argument(
        "--calibrate", type=window, default=None, help="START:STOP (default: first half)"
    )
    ap.add_argument(
        "--validate", type=window, default=None, help="START:STOP (default: second half)"
    )
    ap.add_argument("--kappa", type=float, default=scales.KAPPA)
    ap.add_argument("--top", type=float, default=scales.TOP)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    params = load_model(a.weights, a.seeds, a.config)
    if a.tokens:
        with open(a.tokens, encoding="utf-8") as f:
            streams = streams_from_record(json.load(f))
        source = {"tokens": a.tokens}
    else:
        c, _, t = a.random.partition("x")
        rng = np.random.default_rng(a.seed)
        streams = [list(rng.integers(0, params.vocab, int(t))) for _ in range(int(c))]
        source = {"random": a.random, "seed": a.seed}
    length = min(len(s) for s in streams)
    cal = a.calibrate or (0, length // 2)
    val = a.validate or ((length // 2, length) if a.calibrate is None else None)
    chosen, report = calibrate(params, streams, cal, val, a.kappa, a.top)
    scales.save(a.out, chosen, {**source, **report})
    print(
        json.dumps(
            {
                "blocks": [[s.gate_tm, s.act_cm, s.gate_cm] for s in chosen],
                "validation_min_kept": report.get("validation_min_kept"),
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
