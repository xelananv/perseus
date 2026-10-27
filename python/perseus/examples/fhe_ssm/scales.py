# SPDX-License-Identifier: MIT
"""Power-of-two scale folds: the small values a bootstrap refreshes, lifted into its band.

Three exponents per block (``gate_tm``, ``act_cm``, ``gate_cm``) scale the time-mix gate, the channel-mix
activation and its gate, each folded into constants the block multiplies anyway and undone in the linear
that follows (exact: powers of two). ``calibrate`` takes the largest exponents with ``kappa * 2^e * max <=
top`` over token streams; ``scales_pbd430a.json`` is the published checkpoint's. README.md: the mechanism.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

#: the product outputs each exponent scales: names of ref.ExactModel's observations
GATE_TM_VALUES = ("tm.p1z", "tm.p2z2", "tm.g")
ACT_CM_VALUES = ("cm.q1k", "cm.q2k2", "cm.q3k", "cm.q3k3")
GATE_CM_VALUES = ("cm.r1r", "cm.r2r2")
PRODUCT_CM = "cm.ag"  # scales with act_cm + gate_cm
KAPPA, TOP, MAX_EXP = 2.0, 32.0, 14
#: the exponents calibrated for the published checkpoint (pbd430a), shipped beside this file
SHIPPED = Path(__file__).with_name("scales_pbd430a.json")


@dataclasses.dataclass(frozen=True)
class BlockScales:
    gate_tm: int = 0
    act_cm: int = 0
    gate_cm: int = 0

    def __post_init__(self):
        for name in ("gate_tm", "act_cm", "gate_cm"):
            e = getattr(self, name)
            if not isinstance(e, int) or not 0 <= e <= MAX_EXP:
                raise ValueError(f"{name}={e!r}: an integer exponent in [0, {MAX_EXP}]")


def exponent(value_max: float, kappa: float = KAPPA, top: float = TOP, cap: int = MAX_EXP) -> int:
    """The largest e in [0, cap] with kappa * 2^e * value_max <= top (0 when the value is
    already that large, or was never seen)."""
    if not math.isfinite(value_max) or value_max <= 0.0:
        return 0
    e = math.floor(math.log2(top / (kappa * value_max)))
    return int(min(max(e, 0), cap))


def choose(maxima: list[dict[str, float]], kappa: float = KAPPA, top: float = TOP) -> list[BlockScales]:
    """Exponents per block from the per-block maxima of the observed values: each as large as
    its own products allow, the activation's and the gate's together no larger than their
    product before W_v allows (the activation's first)."""

    def e_of(m, names):
        return exponent(max((m.get(k, 0.0) for k in names), default=0.0), kappa, top)

    out = []
    for m in maxima:
        # the product's own allowance bounds a + b, which may exceed one exponent's cap
        ag = m.get(PRODUCT_CM, 0.0)
        e_ag = exponent(ag, kappa, top, cap=2 * MAX_EXP) if ag > 0 else 2 * MAX_EXP
        a = min(e_of(m, ACT_CM_VALUES), e_ag)
        b = min(e_of(m, GATE_CM_VALUES), e_ag - a)
        out.append(BlockScales(e_of(m, GATE_TM_VALUES), a, b))
    return out


def save(path, scales: list[BlockScales], meta: dict | None = None) -> None:
    doc = {"blocks": [dataclasses.asdict(s) for s in scales], "meta": meta or {}}
    Path(path).write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")


def load(path, n_layers: int | None = None) -> list[BlockScales]:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    scales = [BlockScales(**b) for b in doc["blocks"]]
    if n_layers is not None and len(scales) < n_layers:
        raise ValueError(f"{path}: {len(scales)} blocks of scales for a {n_layers}-block model")
    return scales


def measure(params, streams, start: int, stop: int) -> list[dict[str, float]]:
    """Per block, the largest |value| of every observed quantity of the closed-form forward on
    ticks ``start .. stop-1`` of ``streams`` (one token id list per conversation, teacher-forced).
    The recurrence always runs from tick 0, so a later window sees the state it would see."""
    from .ref import ExactModel

    if not 0 <= start < stop or any(len(s) < stop for s in streams):
        raise ValueError(f"ticks {start}..{stop - 1} need every stream to hold {stop} tokens")
    maxima: list[dict[str, float]] = [{} for _ in range(params.n_layers)]
    recording = [False]

    def observe(b, name, value):
        if recording[0]:
            m = maxima[b]
            m[name] = max(m.get(name, 0.0), float(abs(value).max()))

    exact = ExactModel(params, len(streams), observe=observe)
    for pos in range(stop):
        recording[0] = pos >= start
        exact.step(params.emb[[s[pos] for s in streams]].astype(float))
    return maxima


def resolve(choice: str | None, n_layers: int | None = None) -> list[BlockScales] | None:
    """A driver's ``--scales``: None or "shipped" for SHIPPED, "none" for no folds, else a path."""
    if choice is not None and choice.lower() == "none":
        return None
    path = SHIPPED if choice in (None, "shipped") else Path(choice)
    return load(path, n_layers)
