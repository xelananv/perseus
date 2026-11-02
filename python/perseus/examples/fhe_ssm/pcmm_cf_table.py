# SPDX-License-Identifier: MIT
"""A correction factor per PCMM linear, from a capture: the table ``run_planned.py --pcmm-cf-table`` reads
(``Session(pcmm_cf_map=...)``; a linear not in it refreshes at ``--pcmm-cf``).

    python -m perseus.examples.fhe_ssm.pcmm_cf_table CAPTURE OUT.json [--safety S] [--offset O]
        [--cf-min a] [--cf-max b]

The magnitude. A PCMM finish refreshes each output ``y_m / 2^shift`` (``pc-gpu`` ``matmul/pcmm.rs`` ``linear``:
the stamp ``w_scale 2^shift``, then ``x 2^shift`` after the refresh). The capture records each finish as a
``deliberate_bootstrap`` node with ``args.external`` (the descriptor: ``linear``, ``out``, ``shift``) whose
``output_max_abs`` is ``max |y_m|`` (a refresh keeps the value). The StC-first route's EvalMod sees the slot values,
so the slot maximum (not the coefficient one) is the magnitude; one CF per linear is sized by its largest output.

The rule. The planner's band-centring CF (``perseus/plan/btserr.py`` ``cf_for_magnitude``:
``CF* = log2(|m| / 0.01)``, the band ``[0.003, 0.03] 2^CF``), on the magnitude times the planner's drift margin κ
(``PLAN_MAG_SAFETY``, 2), plus a global offset, rounded up and clamped to ``[cf_min, cf_max]`` (``DEFAULT_CF_MIN``,
``DEFAULT_CF_MAX``)::

    cf = clamp(ceil(log2(κ max|y| / 0.01) + offset), cf_min, cf_max)

The offset is a calibration, not a derivation: a single CF over one block (``pcmm24``, LANDING=high) scored best at
8 and overflowed at 6 (``GPU_DIAGNOSIS_20261102.md`` F0), and the largest finish of that block (``b0.win``,
max|y| 6.25 on the staged one-block capture) is the one a single CF must hold. The default -2.5 puts that site at 8
(``log2(2 x 6.25 / 0.01) = 10.29``; ``ceil(7.79) = 8``) and every smaller one lower by its log2 ratio. Why the form
is log2(m) + const: the StC-first EvalMod's linearization error grows as ``(m 2^-CF)^2`` and the refresh's noise,
multiplied back by ``2^CF``, as ``2^CF`` (×4 / ×2 per step, the sweep's two branches), so each site's best CF sits at
one ratio ``m / 2^CF`` [a model, not measured per site].
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from collections import Counter

from perseus.plan.btserr import DEFAULT_CF_MAX, DEFAULT_CF_MIN, cf_for_magnitude

#: the planner's drift margin κ (PLAN_MAG_SAFETY in the plan recipe)
SAFETY = 2.0
#: the calibration above: the one-block pcmm24 capture's largest finish at CF 8
OFFSET = -2.5


def finish_magnitudes(capture) -> dict[str, float]:
    """``{linear: max over its outputs of output_max_abs / 2^shift}`` over every ``block_<b>/graph.json`` of
    ``capture`` (the PCMM finishes: ``deliberate_bootstrap`` nodes whose ``args.external`` is a ``pcmm``
    descriptor)."""
    root = pathlib.Path(capture)
    files = sorted(root.glob("block_*/graph.json"), key=lambda p: int(p.parent.name.split("_")[1]))
    if not files:
        raise ValueError(f"no block_<b>/graph.json under {root}")
    out: dict[str, float] = {}
    for p in files:
        with open(p, encoding="utf-8") as f:
            nodes = json.load(f)["nodes"]
        for n in nodes:
            d = (n.get("args") or {}).get("external")
            if isinstance(d, str):
                d = json.loads(d)
            if n.get("op_type") != "deliberate_bootstrap" or not isinstance(d, dict) or d.get("op") != "pcmm":
                continue
            m = n.get("output_max_abs")
            if m is None or not math.isfinite(float(m)):
                raise ValueError(
                    f"{p}: the finish of {d.get('linear')} output {d.get('out')} has no magnitude"
                )
            name = str(d["linear"])
            out[name] = max(out.get(name, 0.0), float(m) / 2.0 ** int(d.get("shift", 0)))
    if not out:
        raise ValueError(f"no PCMM finish in {root} (a capture of --linear pcmm?)")
    return out


def cf_of(
    mag: float, safety: float = SAFETY, offset: float = OFFSET, cf_min=DEFAULT_CF_MIN, cf_max=DEFAULT_CF_MAX
):
    """The module rule for one magnitude."""
    return int(min(cf_max, max(cf_min, math.ceil(cf_for_magnitude(safety * mag) + offset))))


def build(capture, safety=SAFETY, offset=OFFSET, cf_min=DEFAULT_CF_MIN, cf_max=DEFAULT_CF_MAX) -> dict:
    """The table file's content: ``cf`` (the map the driver reads), the magnitudes and the rule's parameters."""
    if cf_min > cf_max:
        raise ValueError(f"--cf-min {cf_min} > --cf-max {cf_max}")
    mags = finish_magnitudes(capture)
    cfs = {k: cf_of(m, safety, offset, cf_min, cf_max) for k, m in sorted(mags.items())}
    return {
        "cf": cfs,
        "max_abs": dict(sorted(mags.items())),
        "rule": "clamp(ceil(log2(safety max|y| / 0.01) + offset), cf_min, cf_max) (perseus/plan/btserr.py "
        "cf_for_magnitude)",
        "safety": safety,
        "offset": offset,
        "cf_min": cf_min,
        "cf_max": cf_max,
        "capture": str(capture),
    }


def load(path) -> dict[str, int]:
    """The ``{linear: CF}`` map of a table file: the ``cf`` object of what ``build`` writes, or a flat object. Every
    value an integer >= 1."""
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    if not isinstance(doc, dict):
        raise TypeError(f"{path}: a CF table is a JSON object, got {type(doc).__name__}")
    table = doc["cf"] if isinstance(doc.get("cf"), dict) else doc
    out: dict[str, int] = {}
    for k, v in table.items():
        if isinstance(v, bool) or not isinstance(v, int) or v < 1:
            raise ValueError(f"{path}: CF of {k!r} must be an integer >= 1, got {v!r}")
        out[str(k)] = v
    if not out:
        raise ValueError(f"{path}: an empty CF table")
    return out


def histogram(cfs: dict[str, int]) -> str:
    c = Counter(cfs.values())
    return ", ".join(f"CF {k}: {c[k]}" for k in sorted(c))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("capture", help="a capture directory (block_<b>/graph.json)")
    ap.add_argument("out", help="the table to write (JSON)")
    ap.add_argument("--safety", type=float, default=SAFETY, help="κ, the drift margin on each magnitude")
    ap.add_argument(
        "--offset", type=float, default=OFFSET, help="added to the band-centring CF before rounding"
    )
    ap.add_argument("--cf-min", type=int, default=DEFAULT_CF_MIN)
    ap.add_argument("--cf-max", type=int, default=DEFAULT_CF_MAX)
    a = ap.parse_args(argv)
    doc = build(a.capture, a.safety, a.offset, a.cf_min, a.cf_max)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")
    cfs = doc["cf"]
    worst = max(doc["max_abs"], key=doc["max_abs"].get)
    print(f"[pcmm_cf_table] {len(cfs)} linears -> {a.out}: {histogram(cfs)}")
    print(f"[pcmm_cf_table] largest finish {worst} max|y| {doc['max_abs'][worst]:.4g} -> CF {cfs[worst]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
