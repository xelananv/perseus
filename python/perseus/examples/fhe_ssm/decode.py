# SPDX-License-Identifier: MIT
"""The client's side of a run over a token record: the token streams of a serve record, the decode policy a
free-running session picks its next token with, and the free-running loop over the decrypted logits.

    greedy     the argmax
    norepeat   the argmax after three changes to the logits: the banned tokens removed; a repetition penalty over
               the last ``window`` fed tokens, the prompt included (a positive logit divided by ``penalty``, a
               negative one multiplied by it); every token that would complete an ``ngram`` already in the
               history banned

The policies and their parameters are the fhe-ssm repository's ``spec_decode/decode_policy.py``; a serve record
names the one it ran under (``decodePolicy``). Deterministic, so the decrypted logits and the closed form's
logits, picked on the same history, compare pick for pick.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

POLICIES = ("greedy", "norepeat")
FIELDS = ("policy", "penalty", "window", "ngram", "ban")


def streams_from_record(doc: dict) -> list[list[int]]:
    """One token id list per conversation from a serve run's token record (``streams``, or ``ids_lanes`` followed
    by ``generated_lanes``)."""
    if "streams" in doc:
        return [[int(t) for t in s] for s in doc["streams"]]
    ids, gen = doc.get("ids_lanes"), doc.get("generated_lanes")
    if not ids or gen is None or len(ids) != len(gen):
        raise ValueError("a token record needs 'streams', or 'ids_lanes' and 'generated_lanes'")
    return [[int(t) for t in a + b] for a, b in zip(ids, gen, strict=True)]


def prompt_lengths(doc: dict, n: int) -> list[int]:
    """Each conversation's prompt length in a serve record (``ids_lanes``): where its generation starts."""
    ids = doc.get("ids_lanes")
    if not ids or len(ids) < n or any(len(x) < 1 for x in ids[:n]):
        raise ValueError(
            "a free run needs a serve record whose 'ids_lanes' give every conversation its prompt"
        )
    return [len(x) for x in ids[:n]]


@dataclass(frozen=True)
class Policy:
    policy: str = "greedy"
    penalty: float = 1.3
    window: int = 64
    ngram: int = 4
    ban: tuple[int, ...] = ()

    @classmethod
    def from_record(cls, doc: dict) -> Policy:
        """The ``decodePolicy`` of a serve record; refused when missing, of an unknown kind, or with a field this
        module does not apply."""
        p = doc.get("decodePolicy")
        if not isinstance(p, dict) or p.get("policy") not in POLICIES:
            raise ValueError(f"the record names no decode policy of {POLICIES}: {p!r}")
        extra = sorted(set(p) - set(FIELDS))
        if extra:
            raise ValueError(f"decode policy fields not applied here: {extra}")
        d = cls()
        return cls(
            policy=str(p["policy"]),
            penalty=float(p.get("penalty", d.penalty)),
            window=int(p.get("window", d.window)),
            ngram=int(p.get("ngram", d.ngram)),
            ban=tuple(int(b) for b in p.get("ban", d.ban)),
        )

    def as_dict(self) -> dict:
        return {
            "policy": self.policy,
            "penalty": self.penalty,
            "window": self.window,
            "ngram": self.ngram,
            "ban": list(self.ban),
        }

    def modified(self, logits: np.ndarray, history: Sequence[int]) -> np.ndarray:
        """The logits the policy takes its argmax over (a float64 copy; greedy: unchanged). ``history``: the tokens
        fed so far to this conversation, the prompt and the token that produced these logits included."""
        z = np.array(logits, dtype=np.float64, copy=True)
        if self.policy == "greedy":
            return z
        for b in self.ban:
            z[b] = -np.inf
        h = [int(t) for t in history]
        if self.penalty and self.penalty != 1.0 and h:
            idx = np.unique(np.asarray(h[-self.window :], dtype=np.int64))
            v = z[idx]
            z[idx] = np.where(v > 0, v / self.penalty, v * self.penalty)
        n = self.ngram
        if n and n > 1 and len(h) >= n - 1:
            tail = tuple(h[-(n - 1) :])
            for i in range(len(h) - n + 1):
                if tuple(h[i : i + n - 1]) == tail:
                    z[h[i + n - 1]] = -np.inf
        return z

    def pick(self, logits: np.ndarray, history: Sequence[int]) -> int:
        """The token the client feeds next."""
        if self.policy == "greedy":
            return int(np.argmax(logits))
        return int(np.argmax(self.modified(logits, history)))


class FreeRun:
    """Free running under a record's decode policy. Each conversation's prompt stays teacher-forced; from its first
    generated position on, the policy's pick on the decrypted logits, with the conversation's fed tokens as the
    policy's history, is what it is fed next. Fidelity is the same policy on the closed form's logits on the same
    history; agreement with the record is scored only over conversations whose fed tokens still match the record."""

    def __init__(self, streams: list[list[int]], prompts: list[int], policy: Policy):
        self.streams = [list(s) for s in streams]  # the fed tokens, extended by the feedback
        self.record = [list(s) for s in streams]  # the record's own tokens
        self.prompts = list(prompts)
        self.policy = policy
        self.diverged_at: list[int | None] = [None] * len(streams)

    def tick(self, pos: int, got_logits: np.ndarray, want_logits: np.ndarray) -> dict:
        n = len(self.streams)
        fed = [int(s[pos]) for s in self.streams]
        for k in range(n):
            if self.diverged_at[k] is None and fed[k] != self.record[k][pos]:
                self.diverged_at[k] = pos
        hist = [s[: pos + 1] for s in self.streams]
        pick = [self.policy.pick(got_logits[k], hist[k]) for k in range(n)]
        pick_exact = [self.policy.pick(want_logits[k], hist[k]) for k in range(n)]
        gen = [k for k in range(n) if self.prompts[k] <= pos + 1 < len(self.record[k])]
        on_record = [k for k in gen if self.diverged_at[k] is None]
        agree = [pick[k] == self.record[k][pos + 1] for k in on_record]
        for k in gen:
            self.streams[k][pos + 1] = pick[k]
        return {
            "tokens": fed,
            "top1": [int(x) for x in got_logits.argmax(axis=1)],
            "pick": pick,
            "pick_agree_vs_exact": float(np.mean([a == b for a, b in zip(pick, pick_exact, strict=True)])),
            "on_record": len(on_record),
            "pick_agree_vs_record": float(np.mean(agree)) if on_record else None,
            "fed_back": len(gen),
        }
