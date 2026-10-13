# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Alexander Ananyev
"""The client's decode policy of a free-running session: how a client turns the decrypted logits
of its conversation into the next token. The policies and their parameters are those of the
fhe-ssm repository's ``spec_decode/decode_policy.py``, and a serve run's token record names the
one it ran under (``decodePolicy``).

    greedy     the argmax
    norepeat   the argmax after three changes to the logits: the banned tokens removed; a
               repetition penalty over the last ``window`` fed tokens, the prompt included (a
               positive logit divided by ``penalty``, a negative one multiplied by it); and every
               token that would complete an ``ngram`` already in the history banned

Deterministic, so the decrypted logits and the closed form's logits, picked on the same history,
can be compared pick for pick.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

POLICIES = ("greedy", "norepeat")
FIELDS = ("policy", "penalty", "window", "ngram", "ban")


@dataclass(frozen=True)
class Policy:
    policy: str = "greedy"
    penalty: float = 1.3
    window: int = 64
    ngram: int = 4
    ban: tuple[int, ...] = ()

    @classmethod
    def from_record(cls, doc: dict) -> Policy:
        """The ``decodePolicy`` of a serve record; refused when missing, of an unknown kind, or
        with a field this module does not apply."""
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
        return {"policy": self.policy, "penalty": self.penalty, "window": self.window,
                "ngram": self.ngram, "ban": list(self.ban)}  # fmt: skip

    def modified(self, logits: np.ndarray, history: Sequence[int]) -> np.ndarray:
        """The logits the policy takes its argmax over (a float64 copy; greedy: unchanged).
        ``history``: the tokens fed so far to this conversation, the prompt and the token that
        produced these logits included."""
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
