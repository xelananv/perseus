# SPDX-License-Identifier: MIT
"""A pass's lane configuration: its rules across passes, the scenarios, the lane-slot order, and the pass kernels'
masks as a pure function of the configuration and the block's decay and shift mix.

**Lane-slots.** A pass is one ciphertext stream through the blocks and the tail. In the filled pair layout channel
``i`` of lane ``j`` sits at slot ``i t + j`` (``t = N / d`` lanes: 32 at the model's ring 2^16 and at the synthetic
shape's LOGN 11); lane-slot ``q = 2j + e`` is lane ``j``'s real axis (``e = 0``) or imaginary axis (``e = 1``): ``2t``
lane-slots per channel group, the same partition in every channel. A configuration partitions them into

- ``seq`` segments: a home ``[h0, h1)`` with ``h0``, ``h1`` even (both lane-slots of every home lane belong to it), the
  sequence's consecutive positions at ``[q0, h1)`` in ``q`` order (position ``p`` of the chunk at ``q0 + p``: even
  positions on the real axis, odd on the imaginary), ``[h0, q0)`` front padding of any length (copies of row ``q0``, run
  as conversation-fresh lane-slots, never scored, overwritten by the segment's broadcast at the end of the pass).
  The last position is always ``h1 - 1``, an imaginary axis. Entry ``fresh`` (zero state) or ``carry`` (the state the
  previous pass's broadcast left on the home: a continuing segment's home lies inside its previous home);
- ``conv`` lane-slots: one conversation, one token, entry ``fresh`` / ``carry``, or ``parked``: a copy of a named live
  lane-slot's row (``parked_from``), run fresh, never scored.

``Tracker`` follows which stream last wrote each lane-slot's register and refuses what the masks cannot honour
(``validate`` refuses a malformed pass on its own): an overlapping or incomplete partition, an odd home bound, a
continuing ``seq`` whose home is not inside its previous home, ``carry`` on a register another stream wrote, a parked
lane-slot wrote, nothing wrote yet, or a ``seq`` that still continues wrote; a finished segment's home lanes may turn
``conv carry`` (the hand-over: every home lane-slot, padding included, holds the segment's final state after its last
pass). Not representable (stated gaps): a segment that widens, a conversation's state handed to a multi-lane segment,
a segment ending on a real axis.

**The lane-slot order.** The client packs the ``2t`` rows of a pass in ``q`` order; today's packing puts row ``r`` on
lane ``r mod t``, axis ``r div t`` (``geometry.PairGeometry.encode_rows``), so one permutation ``pi(q) = t (q mod 2) +
q div 2`` (``to_layout`` / ``from_layout``) is applied once before the encode and once after the decode
(``model.FheSsmModel`` under ``pass_kernels="on"``).

**The masks** (block ``b``, channel ``i``, decay ``a = a_{b,i}``, mix ``m = mix_{b,i}``; every mask target-indexed,
``(t, d)`` per offset ``s``). Two complex products realize any real-linear map of a source pair ``(u, v)`` into a target
pair: from the four real coefficients ``(c_ee, c_eo, c_oe, c_oo)`` (target axis, source axis),
``p = (c_ee + c_oo)/2``, ``p' = (c_ee - c_oo)/2``, ``q' = (c_oe + c_eo)/2``, ``q = (c_oe - c_eo)/2``, ``P = p + iq``,
``P' = p' + iq'`` (passkernel.py). Every 1/2 rides a mask (a scalar product costs a level).

- K1, the scan ``S[q] = sum_{q'} c(q, q') x[q'] + kappa(q) C[q]``: ``c(q, q') = (1 - a) a^{q - q'}`` for ``q'`` in
  ``[q0, q]`` of ``q``'s segment (a position), ``c(q, q) = 1 - a`` on a conv, parked or padding lane-slot, else 0 (the
  offset-0 cross term ``c(2j, 2j + 1)`` is a later position: 0); ``P_s``, ``P'_s`` from the four coefficients of target
  lane ``j``, source lane ``j - s``. The state pair, axis-local because a home lane's register holds the state on both
  axes: ``kappa(q) = a^{q - q0 + 1}`` on a carried position, ``a`` on a carried conv, 0 on a fresh, parked or padding
  lane-slot; ``P^c = (kappa(2j) + kappa(2j + 1))/2``, ``P'^c = (kappa(2j) - kappa(2j + 1))/2`` (real).
- The shift ``um[q] = m n[q] + (1 - m) n_prev[q]``, ``n_prev[q] = n[q - 1]`` when ``q > q0`` is a position, else the
  carried ``C_n[q]`` (``kappa_n(q) = 1`` on a carried first position or a carried conv), else 0. ``n[q - 1]`` is
  ``Re n[j]`` for ``q = 2j + 1`` (offset 0, cross-axis) and ``Im n[j - 1]`` for ``q = 2j`` (offset 1, cross-axis); with
  ``sigma(q) = [q > q0 a position]``: ``mixU0 = m + i (1 - m) sigma(2j + 1)/2``, ``U0c = i (1 - m) sigma(2j + 1)/2``,
  ``U1 = -i (1 - m) sigma(2j)/2``, ``U1c = i (1 - m) sigma(2j)/2``, ``V = (1 - m)(kappa_n(2j) + kappa_n(2j + 1))/2``,
  ``Vc = (1 - m)(kappa_n(2j) - kappa_n(2j + 1))/2``.
- K2, the carry prep: ``R_s[j] = 1/4`` where lane ``j + s`` is the end lane ``h1/2 - 1`` of ``j``'s home, else 0 (the
  map is a copy, ``c_ee = c_oo``, ``c_eo = c_oe = 0``, so ``P'`` is 0 in every configuration and not issued);
  ``P_o[j] = 1`` on a lane outside every home (whole lanes, real), else 0. Block- and channel-independent.

``coefficients`` returns them target-indexed (``(t, t, d)`` for K1's ``P``, ``Pc``; ``(t, d)`` for the state pair and
the shift; ``(t, t)`` / ``(t,)`` for ``R`` / ``P_o``): the mirror's form (``RowsPass``). ``masks`` returns the slot
vectors the kernels multiply, keyed by id (``SlotPass``): K1's ``P_s`` / ``P'_s`` pre-rotated for the causal giants
(``passkernel.prerotate``, ``sign -1``), K2's ``R_s`` for the anti-causal ones (``sign +1``), the rest as they are. Ids
(fixed: never a function of the configuration, never a ``#``): per block ``pass/b<b>/k1/P/<s>``,
``pass/b<b>/k1/Pc/<s>``, ``pass/b<b>/k1/S/P``, ``pass/b<b>/k1/S/Pc``, ``pass/b<b>/sh/{mixU0,U0c,U1,U1c,V,Vc}``; shared by
every block ``pass/k2/R/<s>``, ``pass/k2/Po``. Today's tick (every lane-slot ``conv carry``) is ``P_0 = 1 - a``,
``P_{s >= 1} = P' = 0``, ``P^c = a``, ``P'^c = 0``, ``U = 0``, ``V = 1 - m``, ``Vc = 0``, ``R = 0``, ``P_o = 1``.

**Scenarios** (``scenario``): ``tick[:K]``, ``prefill:T``, ``doc:T:readers``, ``mixed``, ``churn`` (the module's
functions say what each holds). A token of ``GENERATED`` (-1) is the argmax of the tied head on the previous pass's
reply at the same lane-slot (``resolve_generated``); padding and parked lane-slots copy their source's token.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np

from . import passkernel
from .passkernel import ANTICAUSAL, CAUSAL, LaneKernel, bsgs_split, prerotate, slot_vector

KINDS = ("seq", "conv")
SEQ_ENTRIES = ("fresh", "carry")
CONV_ENTRIES = ("fresh", "carry", "parked")
#: a token the driver generates from the previous pass's reply at the same lane-slot
GENERATED = -1
SCENARIOS = ("tick", "prefill", "doc", "mixed", "churn")
#: the per-block masks after K1's 2t offsets, in the order the block issues them
K1_STATE = ("k1/S/P", "k1/S/Pc")
SHIFT = ("sh/mixU0", "sh/U0c", "sh/U1", "sh/U1c", "sh/V", "sh/Vc")


def mask_id(name: str, b: int) -> str:
    """A mask's id: ``pass/<name>`` for the carry prep's block-independent masks (``k2/...``), else ``pass/b<b>/<name>``."""
    if "#" in name:
        raise ValueError(f"a pass mask name never holds '#': {name!r}")
    return f"pass/{name}" if name.startswith("k2/") else f"pass/b{int(b)}/{name}"


# ── the configuration ──────────────────────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class Segment:
    """One segment of a pass (the module docstring). A ``conv`` lane-slot ``q`` has ``home = (q, q + 1)``,
    ``q0 = q``; a parked one has no stream and names ``parked_from``."""

    kind: str
    home: tuple[int, int]
    q0: int
    entry: str
    stream: str | None = None
    parked_from: int | None = None

    @classmethod
    def seq(cls, h0: int, h1: int, q0: int, entry: str, stream: str) -> Segment:
        return cls("seq", (int(h0), int(h1)), int(q0), str(entry), str(stream))

    @classmethod
    def conv(cls, q: int, entry: str, stream: str | None = None, parked_from: int | None = None) -> Segment:
        pf = None if parked_from is None else int(parked_from)
        return cls(
            "conv", (int(q), int(q) + 1), int(q), str(entry), None if stream is None else str(stream), pf
        )

    @property
    def h0(self) -> int:
        return self.home[0]

    @property
    def h1(self) -> int:
        return self.home[1]

    @property
    def live(self) -> bool:
        return self.entry != "parked"

    @property
    def positions(self) -> range:
        """The scored lane-slots: a sequence's positions, a live conversation's lane-slot."""
        return range(self.q0, self.h1) if self.live else range(0)

    @property
    def padding(self) -> range:
        return range(self.h0, self.q0) if self.kind == "seq" else range(0)

    @property
    def last(self) -> int:
        """The lane-slot whose reply is the next-token prediction."""
        return self.h1 - 1

    def to_json(self) -> dict:
        return {
            "kind": self.kind,
            "home": [self.h0, self.h1],
            "q0": self.q0,
            "entry": self.entry,
            "stream": self.stream,
            "parked_from": self.parked_from,
        }

    @classmethod
    def from_json(cls, d: dict) -> Segment:
        unknown = set(d) - {"kind", "home", "q0", "entry", "stream", "parked_from"}
        if unknown:
            raise ValueError(
                f"a segment has kind, home, q0, entry, stream, parked_from; got also {sorted(unknown)}"
            )
        home = d["home"]
        if not (isinstance(home, (list, tuple)) and len(home) == 2):
            raise ValueError(f"a segment's home is [h0, h1], got {home!r}")
        pf = d.get("parked_from")
        stream = d.get("stream")
        return cls(
            str(d["kind"]),
            (int(home[0]), int(home[1])),
            int(d["q0"]),
            str(d["entry"]),
            None if stream is None else str(stream),
            None if pf is None else int(pf),
        )


@dataclasses.dataclass(frozen=True)
class PassConfig:
    """One pass: its segments and a token per lane-slot (``q`` order), optionally named for the record."""

    segments: tuple[Segment, ...]
    tokens: tuple[int, ...]
    name: str = ""

    @property
    def lane_slots(self) -> int:
        return len(self.tokens)

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "segments": [s.to_json() for s in self.segments],
            "tokens": list(self.tokens),
        }

    @classmethod
    def from_json(cls, d: dict) -> PassConfig:
        unknown = set(d) - {"name", "segments", "tokens"}
        if unknown:
            raise ValueError(f"a pass has name, segments, tokens; got also {sorted(unknown)}")
        return cls(
            tuple(Segment.from_json(s) for s in d["segments"]),
            tuple(int(k) for k in d["tokens"]),
            str(d.get("name", "")),
        )

    def roles(self) -> list[tuple[str, int]]:
        """Per lane-slot ``(role, segment index)``: role ``pos`` (a sequence position), ``pad`` (front padding),
        ``conv`` (a live conversation) or ``parked``. ``validate`` first: this assumes a partition."""
        out: list[tuple[str, int]] = [("", -1)] * self.lane_slots
        for k, seg in enumerate(self.segments):
            if seg.kind == "seq":
                for q in seg.padding:
                    out[q] = ("pad", k)
                for q in seg.positions:
                    out[q] = ("pos", k)
            else:
                out[seg.q0] = ("conv" if seg.live else "parked", k)
        return out

    def live_slots(self) -> list[int]:
        """The lane-slots a pass is scored on: every sequence position and every live conversation."""
        return sorted(q for seg in self.segments for q in seg.positions)

    def next_token_slots(self) -> list[int]:
        """The lane-slots whose replies are next-token predictions: each sequence's last position, each live conv."""
        return sorted(seg.last for seg in self.segments if seg.live)

    def mask_key(self) -> tuple:
        """What the masks depend on (two passes with equal keys have equal masks for every block): the partition and
        the entries, a parked lane-slot counted as fresh; not the streams or the tokens."""
        return tuple(
            (s.kind, s.home, s.q0, "fresh" if s.entry == "parked" else s.entry)
            for s in sorted(self.segments, key=lambda s: s.home)
        )

    def describe(self) -> str:
        """One line for a record."""
        parts = [
            f"seq {s.stream} [{s.h0},{s.h1}) q0 {s.q0} {s.entry}" for s in self.segments if s.kind == "seq"
        ]
        convs = [s for s in self.segments if s.kind == "conv"]
        for entry in CONV_ENTRIES:
            qs = [s.q0 for s in convs if s.entry == entry]
            if qs:
                parts.append(f"conv {entry} x{len(qs)}")
        return "; ".join(parts)


def validate(cfg: PassConfig) -> None:
    """The rules of one pass on its own (the module docstring); raises ``ValueError``."""
    n = cfg.lane_slots
    if n < 2 or n % 2:
        raise ValueError(f"a pass holds 2t lane-slots, got {n} tokens")
    cover = [-1] * n
    streams: dict[str, int] = {}
    for k, seg in enumerate(cfg.segments):
        if seg.kind not in KINDS:
            raise ValueError(f"segment {k}: kind must be one of {KINDS}, got {seg.kind!r}")
        h0, h1 = seg.home
        if seg.kind == "seq":
            if h0 % 2 or h1 % 2:
                raise ValueError(f"segment {k}: a seq home has even bounds (whole lanes), got [{h0}, {h1})")
            if not 0 <= h0 <= seg.q0 < h1 <= n:
                raise ValueError(
                    f"segment {k}: a seq needs 0 <= h0 <= q0 < h1 <= {n}, got [{h0}, {h1}) q0 {seg.q0}"
                )
            if seg.entry not in SEQ_ENTRIES:
                raise ValueError(f"segment {k}: a seq's entry is one of {SEQ_ENTRIES}, got {seg.entry!r}")
            if not seg.stream or seg.parked_from is not None:
                raise ValueError(f"segment {k}: a seq names its stream and parks nothing")
        else:
            if h1 != h0 + 1 or seg.q0 != h0 or not 0 <= h0 < n:
                raise ValueError(
                    f"segment {k}: a conv lane-slot q has home [q, q + 1) and q0 = q, got [{h0}, {h1})"
                )
            if seg.entry not in CONV_ENTRIES:
                raise ValueError(f"segment {k}: a conv's entry is one of {CONV_ENTRIES}, got {seg.entry!r}")
            if seg.entry == "parked":
                if seg.parked_from is None or seg.stream is not None:
                    raise ValueError(f"segment {k}: a parked lane-slot names its source and no stream")
            elif not seg.stream or seg.parked_from is not None:
                raise ValueError(f"segment {k}: a live conv names its stream and parks nothing")
        if seg.live:
            if seg.stream in streams:
                raise ValueError(f"stream {seg.stream!r} occupies segments {streams[seg.stream]} and {k}")
            streams[str(seg.stream)] = k
        for q in range(h0, h1):
            if cover[q] >= 0:
                raise ValueError(f"lane-slot {q} lies in segments {cover[q]} and {k}: the segments overlap")
            cover[q] = k
    missing = [q for q in range(n) if cover[q] < 0]
    if missing:
        raise ValueError(f"lane-slots {missing} lie in no segment: a pass partitions all {n}")
    roles = cfg.roles()
    tokens = cfg.tokens
    for k, seg in enumerate(cfg.segments):
        if seg.entry == "parked":
            src = int(seg.parked_from)  # type: ignore[arg-type]
            if not (0 <= src < n and roles[src][0] in ("pos", "conv")):
                raise ValueError(
                    f"segment {k}: parked lane-slot {seg.q0} copies {src}, which is not live in this pass"
                )
            if tokens[seg.q0] != tokens[src]:
                raise ValueError(
                    f"parked lane-slot {seg.q0} holds token {tokens[seg.q0]}, its source {tokens[src]}"
                )
        for q in seg.padding:
            if tokens[q] != tokens[seg.q0]:
                raise ValueError(
                    f"padding lane-slot {q} holds token {tokens[q]}, its segment's first {tokens[seg.q0]}"
                )
    for q in range(n):
        if tokens[q] < 0 and (tokens[q] != GENERATED or roles[q][0] == "pos"):
            raise ValueError(
                f"lane-slot {q}: token {tokens[q]} (only a conversation's token may be GENERATED)"
            )


@dataclasses.dataclass(frozen=True)
class Writer:
    """Who wrote a lane-slot's register in the last pass: a sequence's broadcast, a conversation, or a parked copy."""

    kind: str
    stream: str | None


class Tracker:
    """The registers across passes (the module docstring's rules): ``resolve`` checks a pass and says, per segment,
    the stream whose state it enters from (None: zeros); ``commit`` records what the pass leaves in the registers."""

    def __init__(self, lane_slots: int):
        self.n = int(lane_slots)
        self.writer: list[Writer | None] = [None] * self.n
        self.homes: dict[str, tuple[int, int]] = {}
        self.passes = 0

    def resolve(self, cfg: PassConfig) -> list[str | None]:
        validate(cfg)
        if cfg.lane_slots != self.n:
            raise ValueError(f"a pass of {cfg.lane_slots} lane-slots on registers of {self.n}")
        continuing = {s.stream for s in cfg.segments if s.kind == "seq"}
        src: list[str | None] = []
        for k, seg in enumerate(cfg.segments):
            if seg.entry != "carry":
                src.append(None)
                continue
            if seg.kind == "seq":
                mine = Writer("seq", seg.stream)
                if any(self.writer[q] != mine for q in range(seg.h0, seg.h1)):
                    prev = self.homes.get(str(seg.stream))
                    was_seq = prev is not None and any(w == mine for w in self.writer)
                    if was_seq:
                        raise ValueError(
                            f"segment {k}: seq {seg.stream!r} continues on home [{seg.h0}, {seg.h1}), not inside its "
                            f"previous home [{prev[0]}, {prev[1]})"  # type: ignore[index]
                        )
                    raise ValueError(
                        f"segment {k}: seq {seg.stream!r} carries, but no previous pass left its state on its home "
                        "(a conversation cannot hand its state to a seq segment; a seq that skipped a pass was "
                        "overwritten)"
                    )
                src.append(seg.stream)
                continue
            w = self.writer[seg.q0]
            where = f"segment {k}: conv {seg.stream!r} carries on lane-slot {seg.q0}"
            if w is None:
                raise ValueError(f"{where}, whose register no pass wrote (use fresh)")
            if w.kind == "parked":
                raise ValueError(f"{where}, whose register a parked copy wrote")
            if w.kind == "conv":
                if w.stream != seg.stream:
                    raise ValueError(f"{where}, whose register stream {w.stream!r} wrote")
                src.append(seg.stream)
                continue
            if w.stream in continuing:
                raise ValueError(f"{where}, whose register seq {w.stream!r} wrote and still continues")
            src.append(w.stream)  # the hand-over: the finished segment's final state
        return src

    def commit(self, cfg: PassConfig) -> None:
        for seg in cfg.segments:
            if seg.kind == "seq":
                for q in range(seg.h0, seg.h1):
                    self.writer[q] = Writer("seq", seg.stream)
                self.homes[str(seg.stream)] = seg.home
            elif seg.entry == "parked":
                self.writer[seg.q0] = Writer("parked", None)
            else:
                self.writer[seg.q0] = Writer("conv", seg.stream)
        self.passes += 1


def validate_list(cfgs: Iterable[PassConfig]) -> list[list[str | None]]:
    """Every pass of a list through one ``Tracker``: the entry streams per pass (raises on the first refusal)."""
    cfgs = list(cfgs)
    if not cfgs:
        raise ValueError("an empty pass list")
    tr = Tracker(cfgs[0].lane_slots)
    out = []
    for p, cfg in enumerate(cfgs):
        try:
            out.append(tr.resolve(cfg))
        except ValueError as e:
            raise ValueError(f"pass {p}: {e}") from e
        tr.commit(cfg)
    return out


def dump_list(cfgs: Iterable[PassConfig]) -> list[dict]:
    return [c.to_json() for c in cfgs]


def load_list(obj: Any) -> list[PassConfig]:
    """A pass list from JSON (a path, or the parsed value): a list of passes, or ``{"passes": [...]}``."""
    if isinstance(obj, (str, bytes)) or hasattr(obj, "__fspath__"):
        with open(obj, encoding="utf-8") as f:
            obj = json.load(f)
    if isinstance(obj, dict):
        obj = obj["passes"]
    return [PassConfig.from_json(d) for d in obj]


# ── the lane-slot order and the rows ───────────────────────────────────────────────────────────


def layout_index(lane_slots: int) -> np.ndarray:
    """``pi(q) = t (q mod 2) + q div 2``: the row of today's packing (``r`` on lane ``r mod t``, axis ``r div t``) that
    holds lane-slot ``q``."""
    n = int(lane_slots)
    if n < 2 or n % 2:
        raise ValueError(f"2t lane-slots, got {n}")
    q = np.arange(n)
    return (n // 2) * (q % 2) + q // 2


def to_layout(rows_q) -> np.ndarray:
    """Rows in ``q`` order -> today's row order (``rows_r[pi(q)] = rows_q[q]``)."""
    rows_q = np.asarray(rows_q)
    out = np.empty_like(rows_q)
    out[layout_index(rows_q.shape[0])] = rows_q
    return out


def from_layout(rows_r) -> np.ndarray:
    """Today's row order -> ``q`` order (``to_layout``'s inverse)."""
    rows_r = np.asarray(rows_r)
    return rows_r[layout_index(rows_r.shape[0])]


def pass_rows(cfg: PassConfig, emb_row: Callable[[int], np.ndarray]) -> np.ndarray:
    """The ``2t`` input rows of a pass in ``q`` order: the embedding row of each lane-slot's token (padding and parked
    lane-slots hold their source's token, so their rows are copies)."""
    if any(k == GENERATED for k in cfg.tokens):
        raise ValueError(f"pass {cfg.name!r}: generated tokens are resolved first (resolve_generated)")
    return np.stack([np.asarray(emb_row(int(k)), dtype=np.float64) for k in cfg.tokens])


def resolve_generated(cfg: PassConfig, prev_reply_q, head) -> PassConfig:
    """``cfg`` with every ``GENERATED`` token replaced by ``argmax(prev_reply_q[q] @ head.T)`` at its own lane-slot
    (copies follow their source): what a client does with the previous pass's decrypted reply."""
    toks = list(cfg.tokens)
    if GENERATED not in toks:
        return cfg
    if prev_reply_q is None:
        raise ValueError(f"pass {cfg.name!r}: a generated token needs the previous pass's reply")
    head = np.asarray(head, dtype=np.float64)
    roles = cfg.roles()
    for q, k in enumerate(toks):
        if k == GENERATED and roles[q][0] == "conv":
            toks[q] = int(np.argmax(np.asarray(prev_reply_q[q], dtype=np.float64) @ head.T))
    for seg in cfg.segments:
        if seg.entry == "parked" and toks[seg.q0] == GENERATED:
            toks[seg.q0] = toks[int(seg.parked_from)]  # type: ignore[arg-type]
    if GENERATED in toks:
        raise ValueError(
            f"pass {cfg.name!r}: a generated token on a lane-slot that is not a live conversation"
        )
    return dataclasses.replace(cfg, tokens=tuple(toks))


# ── the masks ──────────────────────────────────────────────────────────────────────────────────


def coefficients(cfg: PassConfig, a, mix) -> dict[str, np.ndarray]:
    """The target-indexed coefficients of one block (the module docstring): ``k1/P``, ``k1/Pc`` ``(t, t, d)`` complex
    (offset, lane, channel); ``k1/S/P``, ``k1/S/Pc`` ``(t, d)`` real; ``sh/mixU0``, ``sh/U0c``, ``sh/U1``, ``sh/U1c``
    ``(t, d)`` complex; ``sh/V``, ``sh/Vc`` ``(t, d)`` real; ``k2/R`` ``(t, t)`` real (offset, lane); ``k2/Po`` ``(t,)``
    real."""
    a = np.asarray(a, dtype=np.float64)
    m = np.asarray(mix, dtype=np.float64)
    if a.ndim != 1 or m.shape != a.shape:
        raise ValueError(f"a and mix are (d,), got {a.shape} and {m.shape}")
    n, d = cfg.lane_slots, a.shape[0]
    t = n // 2
    roles = cfg.roles()
    segs = cfg.segments
    # K1: c(q, q') = (1 - a) a^E[q, q'] where E >= 0; index n of the power table is the zero row
    E = np.full((n, n), n, dtype=np.int64)
    kap = np.zeros((n, d))
    sig = np.zeros(n)
    kap_n = np.zeros(n)
    for q, (role, k) in enumerate(roles):
        seg = segs[k]
        if role == "pos":
            qq = np.arange(seg.q0, q + 1)
            E[q, qq] = q - qq
            sig[q] = float(q > seg.q0)
            if seg.entry == "carry":
                kap[q] = a ** (q - seg.q0 + 1)
                kap_n[q] = float(q == seg.q0)
        else:
            E[q, q] = 0
            if role == "conv" and seg.entry == "carry":
                kap[q] = a
                kap_n[q] = 1.0
    pw = np.concatenate([(1.0 - a)[None, :] * a[None, :] ** np.arange(n)[:, None], np.zeros((1, d))])
    P = np.zeros((t, t, d), dtype=np.complex128)
    Pc = np.zeros_like(P)
    for s in range(t):
        j = np.arange(s, t)
        jj = j - s
        cee, ceo = pw[E[2 * j, 2 * jj]], pw[E[2 * j, 2 * jj + 1]]
        coe, coo = pw[E[2 * j + 1, 2 * jj]], pw[E[2 * j + 1, 2 * jj + 1]]
        P[s, j] = 0.5 * (cee + coo) + 0.5j * (coe - ceo)
        Pc[s, j] = 0.5 * (cee - coo) + 0.5j * (coe + ceo)
    im = (1.0 - m)[None, :]
    s_odd, s_even = sig[1::2][:, None], sig[0::2][:, None]
    R = np.zeros((t, t))
    Po = np.ones(t)
    for seg in segs:
        if seg.kind == "seq":
            e = seg.h1 // 2 - 1
            for lane in range(seg.h0 // 2, seg.h1 // 2):
                R[e - lane, lane] = 0.25
                Po[lane] = 0.0
    return {
        "k1/P": P,
        "k1/Pc": Pc,
        "k1/S/P": 0.5 * (kap[0::2] + kap[1::2]),
        "k1/S/Pc": 0.5 * (kap[0::2] - kap[1::2]),
        "sh/mixU0": m[None, :] + 0.5j * im * s_odd,
        "sh/U0c": 0.5j * im * s_odd,
        "sh/U1": -0.5j * im * s_even,
        "sh/U1c": 0.5j * im * s_even,
        "sh/V": 0.5 * im * (kap_n[0::2] + kap_n[1::2])[:, None],
        "sh/Vc": 0.5 * im * (kap_n[0::2] - kap_n[1::2])[:, None],
        "k2/R": R,
        "k2/Po": Po,
    }


def lane_coefs(co: dict[str, np.ndarray], name: str, d: int) -> np.ndarray:
    """A coefficient of ``coefficients`` by its mask name (``k1/P/<s>``, ``k2/R/<s>``, ...) as a ``(t, d)`` lane
    array."""
    base, _, s = name.rpartition("/")
    v = co[base][int(s)] if base in ("k1/P", "k1/Pc", "k2/R") and s.isdigit() else co[name]
    return np.repeat(v[:, None], d, axis=1) if v.ndim == 1 else v


def mask_names(t: int) -> list[str]:
    """Every mask a block issues, in the block's order: the shift's six (``tm.in``), K1's ``2t`` (``P_s``, ``P'_s`` per
    offset) and its state pair (``tm.gate``), K2's ``t`` and ``P_o`` (``carry.prep``; ``P_o`` multiplies ``S`` and ``n``:
    one id, two products)."""
    k1 = [f"k1/{h}/{s}" for s in range(t) for h in ("P", "Pc")]
    return [*SHIFT, *k1, *K1_STATE, *(f"k2/R/{s}" for s in range(t)), "k2/Po"]


def masks(cfg: PassConfig, bp, b: int) -> dict[str, np.ndarray]:
    """``{mask id: the slot vector the block multiplies}`` for block ``b`` (``bp`` has the block's ``a`` and ``mix``,
    ``(d,)``): a pure function of the configuration and the block's two vectors."""
    co = coefficients(cfg, bp.a, bp.mix)
    t = cfg.lane_slots // 2
    d = int(np.asarray(bp.a).shape[0])
    nb, _ng = bsgs_split(t)
    out: dict[str, np.ndarray] = {}
    for name in mask_names(t):
        v = slot_vector(lane_coefs(co, name, d))
        if name.startswith("k1/P"):
            v = prerotate(v, int(name.rpartition("/")[2]), nb, CAUSAL)
        elif name.startswith("k2/R/"):
            v = prerotate(v, int(name.rpartition("/")[2]), nb, ANTICAUSAL)
        out[mask_id(name, b)] = v
    return out


# ── the per-block kernel objects block.py runs ─────────────────────────────────────────────────


class SlotPass:
    """One block's pass kernels on the slots: the masks of ``masks`` (by id) through ``passkernel.passmult``. The
    protocol block.py's pass functions use: ``enter`` / ``leave`` (identity here), ``lanes(ops, x, "k1" | "k2")``,
    ``mult(ops, ct, name)``, ``rot1(ops, x)`` (``rot(x, -1)``), ``note(ops, name, ct)`` (records level and degree)."""

    def __init__(self, mask_vectors: dict[str, np.ndarray], b: int, t: int):
        self.masks, self.b, self.t = mask_vectors, int(b), int(t)
        self.k1 = LaneKernel(self.t, CAUSAL, conj_half=True)
        self.k2 = LaneKernel(self.t, ANTICAUSAL, conj_half=False)
        self.seen: dict[str, tuple[int, int]] = {}

    @staticmethod
    def enter(x):
        return x

    @staticmethod
    def leave(x):
        return x

    def note(self, ops, name: str, ct) -> None:
        self.seen[name] = (int(ops.level(ct)), int(ops.noise_deg(ct)))

    def mult(self, ops, ct, name: str):
        mid = mask_id(name, self.b)
        return passkernel.passmult(ops, ct, mid, self.masks[mid])

    def lanes(self, ops, x, which: str):
        if which == "k1":
            return self.k1.apply(
                ops, x, lambda o, ct, s, half: self.mult(o, ct, f"k1/{'Pc' if half else 'P'}/{s}")
            )
        if which == "k2":
            return self.k2.apply(ops, x, lambda o, ct, s, _half: self.mult(o, ct, f"k2/R/{s}"))
        raise ValueError(f"the lane terms are k1 and k2, got {which!r}")

    @staticmethod
    def rot1(ops, x):
        return ops.rotate(x, -1)


class RowsPass:
    """One block's pass kernels on ``(2t, d)`` rows in ``q`` order (the mirror's form): the same functions of block.py
    on the ``(t, d)`` complex lanes of the rows, the lane terms by ``passkernel.direct`` with the target-indexed
    coefficients (no pre-rotation, no slots), ``rot1`` the lane shift with zero fill."""

    def __init__(self, coefs: dict[str, np.ndarray]):
        self.co = coefs
        self.d = int(coefs["k1/P"].shape[2])

    @staticmethod
    def enter(rows_q):
        return passkernel.pair_lanes(rows_q)

    @staticmethod
    def leave(X):
        return passkernel.unpair_lanes(X)

    def note(self, ops, name: str, ct) -> None:
        pass

    def mult(self, ops, X, name: str):  # noqa: ARG002 (the protocol)
        return X * lane_coefs(self.co, name, self.d)

    def lanes(self, ops, X, which: str):  # noqa: ARG002 (the protocol)
        if which == "k1":
            return passkernel.direct(X, self.co["k1/P"], self.co["k1/Pc"], CAUSAL)
        if which == "k2":
            return passkernel.direct(X, self.co["k2/R"][:, :, None], None, ANTICAUSAL)
        raise ValueError(f"the lane terms are k1 and k2, got {which!r}")

    @staticmethod
    def rot1(ops, X):  # noqa: ARG004 (the protocol)
        return passkernel.lane_shift(X, 1, CAUSAL)


# ── scenarios ──────────────────────────────────────────────────────────────────────────────────


class _Tokens:
    def __init__(self, vocab: int, seed: int):
        self.vocab = int(vocab)
        self.rng = np.random.default_rng(seed)

    def draw(self, n: int) -> list[int]:
        return [int(k) for k in self.rng.integers(0, self.vocab, n)]


def _chunks(T: int, n: int) -> list[int]:
    """``q0`` of each chunk of a ``T``-position sequence on a home ``[0, n)``: the full chunks first, the remainder
    last at ``q0 = n - (T mod n)`` (front padding below it)."""
    if T < 1:
        raise ValueError(f"a sequence of {T} positions")
    full, r = divmod(int(T), n)
    return [0] * full + ([n - r] if r else [])


def _seq_pass(n: int, q0: int, entry: str, stream: str, toks: _Tokens, name: str, rest=()) -> PassConfig:
    t = toks.draw(n - q0)
    tokens = [t[0]] * q0 + t
    return PassConfig((Segment.seq(0, n, q0, entry, stream), *rest), tuple(tokens), name)


def _conv_pass(
    n: int, live: dict[int, tuple[str, str]], source: int, tokens: dict[int, int], name: str
) -> PassConfig:
    """Live conversations ``{q: (entry, stream)}``, the other lane-slots parked from ``source``."""
    segs, toks = [], []
    for q in range(n):
        if q in live:
            entry, stream = live[q]
            segs.append(Segment.conv(q, entry, stream))
            toks.append(tokens[q])
        else:
            segs.append(Segment.conv(q, "parked", parked_from=source))
            toks.append(tokens[source])
    return PassConfig(tuple(segs), tuple(toks), name)


def tick(n: int, toks: _Tokens, passes: int = 6, conversations: int | None = None) -> list[PassConfig]:
    """``passes`` ticks of ``conversations`` (default all ``n``) conversations on lane-slots ``0 .. c - 1`` (fresh in
    pass 0, carried after), the rest parked copies of lane-slot 0: today's tick in ``q`` order."""
    c = n if conversations is None else int(conversations)
    if not 1 <= c <= n:
        raise ValueError(f"{c} conversations on {n} lane-slots")
    out = []
    for p in range(int(passes)):
        drawn = toks.draw(c)
        live = {q: ("fresh" if p == 0 else "carry", f"c{q}") for q in range(c)}
        out.append(_conv_pass(n, live, 0, dict(enumerate(drawn)), f"tick.{p}"))
    return out


def prefill(n: int, toks: _Tokens, T: int, generate: int = 3) -> list[PassConfig]:
    """One conversation ``L`` reading a ``T``-token prompt on the home ``[0, n)`` (``_chunks``; pass 0 fresh, the rest
    carried), then ``generate`` passes of ``L`` as a conversation on lane-slot ``n - 1`` (the token generated from the
    previous pass's reply there), the other lane-slots parked copies of it."""
    out = []
    for c, q0 in enumerate(_chunks(T, n)):
        out.append(_seq_pass(n, q0, "fresh" if c == 0 else "carry", "L", toks, f"prefill.{c}"))
    out.extend(
        _conv_pass(n, {n - 1: ("carry", "L")}, n - 1, {n - 1: GENERATED}, f"generate.{g}")
        for g in range(int(generate))
    )
    return out


def doc(n: int, toks: _Tokens, T: int, readers: int, passes: int = 3) -> list[PassConfig]:
    """A document ``D`` of ``T`` tokens read on the home ``[0, n)`` (``_chunks``), then ``passes`` passes of
    ``readers`` conversations ``R<k>`` on lane-slots ``0 .. readers - 1`` (each starting from the document's final
    state: the hand-over, then its own), a question token each per pass, the rest parked copies of lane-slot 0."""
    w = int(readers)
    if not 1 <= w <= n:
        raise ValueError(f"{w} readers on {n} lane-slots")
    out = []
    for c, q0 in enumerate(_chunks(T, n)):
        out.append(_seq_pass(n, q0, "fresh" if c == 0 else "carry", "D", toks, f"doc.{c}"))
    for r in range(int(passes)):
        drawn = toks.draw(w)
        live = {q: ("carry", f"R{q}") for q in range(w)}
        out.append(_conv_pass(n, live, 0, dict(enumerate(drawn)), f"readers.{r}"))
    return out


def mixed(n: int, toks: _Tokens) -> list[PassConfig]:
    """The owner's mix on 64 lane-slots. Pass 0: seq A home ``[0, 32)`` ``q0`` 0 fresh; seq B home ``[32, 42)`` ``q0`` 33
    (9 positions, 1 padding) fresh; conversations on ``[42, 64)`` fresh. Pass 1: A, B carried; the conversations
    carried but lane-slots 45, 50, 61 (new conversations, fresh). Pass 2: A finished, 32 readers ``RA<k>`` on
    ``[0, 32)`` (the hand-over); lane-slots 32, 33 two new conversations (fresh); B on home ``[34, 42)`` ``q0`` 34
    carried; ``[42, 64)`` carried. Passes 3 and 4: pass 2's configuration again (new conversations on 32 and 33 each
    pass): equal masks, nothing to restage. Every live lane-slot draws a token per pass."""
    if n != 64:
        raise ValueError(f"the mixed scenario is written for 64 lane-slots, got {n}")
    conv_streams = {q: f"c{q}" for q in range(42, 64)}
    out = []
    for p in range(5):
        tokens = toks.draw(n)
        segs: list[Segment] = []
        if p < 2:
            entry = "fresh" if p == 0 else "carry"
            segs += [Segment.seq(0, 32, 0, entry, "A"), Segment.seq(32, 42, 33, entry, "B")]
            tokens[32] = tokens[33]
            fresh = set(range(42, 64)) if p == 0 else {45, 50, 61}
            if p == 1:
                conv_streams.update({q: f"c{q}.1" for q in (45, 50, 61)})
        else:
            segs += [Segment.conv(q, "carry", f"RA{q}") for q in range(32)]
            segs += [Segment.conv(q, "fresh", f"n{q}.{p}") for q in (32, 33)]
            segs.append(Segment.seq(34, 42, 34, "carry", "B"))
            fresh = set()
        segs += [Segment.conv(q, "fresh" if q in fresh else "carry", conv_streams[q]) for q in range(42, 64)]
        out.append(PassConfig(tuple(segs), tuple(tokens), f"mixed.{p}"))
    return out


def churn(n: int, toks: _Tokens, seed: int) -> list[PassConfig]:
    """Conversations leaving (their lane-slot parked) and joining (fresh, a new stream) across passes 0-3, then two
    passes unchanged (every live one carried): pass 0 all fresh; pass 1: four leave, two are replaced; pass 2: two of
    the four rejoin, two more replaced; pass 3: the other two rejoin, one leaves, one replaced; passes 4, 5: no change.
    Lane-slot 0 never churns (the parked copies' source)."""
    rng = np.random.default_rng(seed + 1)
    order = [int(q) for q in rng.permutation(n) if q != 0]
    if len(order) < 10:
        raise ValueError(f"churn needs at least 11 lane-slots, got {n}")
    leave1, rep1 = order[0:4], order[4:6]
    rep2, leave3, rep3 = order[6:8], order[8:9], order[9:10]
    plan = [
        ({}, set(), True),
        (dict.fromkeys(rep1, "fresh"), set(leave1), False),
        (dict.fromkeys((*leave1[:2], *rep2), "fresh"), set(leave1[2:]), False),
        (dict.fromkeys((*leave1[2:], *rep3), "fresh"), set(leave3), False),
        ({}, set(leave3), False),
        ({}, set(leave3), False),
    ]
    gen: dict[int, int] = dict.fromkeys(range(n), 0)
    out = []
    for p, (fresh, parked, first) in enumerate(plan):
        live = {}
        for q in range(n):
            if q in parked:
                continue
            if first or q in fresh:
                gen[q] += 0 if first else 1
                live[q] = ("fresh", f"c{q}.{gen[q]}")
            else:
                live[q] = ("carry", f"c{q}.{gen[q]}")
        drawn = toks.draw(n)
        out.append(_conv_pass(n, live, 0, dict(enumerate(drawn)), f"churn.{p}"))
    return out


def scenario(spec: str, *, vocab: int, seed: int = 0, lane_slots: int = 64, conversations: int | None = None):
    """The pass list a ``--scenario`` names: ``tick`` / ``tick:K`` (6 passes by default; ``conversations`` live, the
    rest parked), ``prefill:T``, ``doc:T:readers``, ``mixed``, ``churn``; validated through one ``Tracker``."""
    name, *args = str(spec).split(":")
    n = int(lane_slots)
    toks = _Tokens(vocab, seed)
    if name == "tick" and len(args) <= 1:
        cfgs = tick(n, toks, int(args[0]) if args else 6, conversations)
    elif name == "prefill" and len(args) == 1:
        cfgs = prefill(n, toks, int(args[0]))
    elif name == "doc" and len(args) == 2:
        cfgs = doc(n, toks, int(args[0]), int(args[1]))
    elif name == "mixed" and not args:
        cfgs = mixed(n, toks)
    elif name == "churn" and not args:
        cfgs = churn(n, toks, seed)
    else:
        raise ValueError(f"scenario {spec!r}: tick[:K] | prefill:T | doc:T:readers | mixed | churn")
    validate_list(cfgs)
    return cfgs
