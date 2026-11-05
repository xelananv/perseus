# SPDX-License-Identifier: MIT
"""Batched decode: K <= t independent conversations per encrypted pass, conversation ``c`` on lane ``c``
(BATCHED_DESIGN.md).

The decode step copies ONE token over the t = 32 lanes of the filled layout. Here lane ``c`` holds conversation ``c``'s
current token instead. Every token-side op of the block (PCMM linears, polynomial sites, Newton chains, LayerNorm,
GroupNorm, the ``kk`` and bonus reductions, ``z = h + i v_first``) is per lane already (PREFILL_DESIGN.md §1), so it
serves K conversations for the price of one. Two things are per conversation:

* **the WKV state**: conversation ``c`` keeps its own M state ciphertexts per block, in decode's layout (value index on
  the lanes). Per conversation its six vectors are moved from lane ``c`` to natural form (the three pairs
  ``w + i kk``, ``(kk a) + i k~``, ``r + i v`` share one broadcast each; one conjugation splits a pair: prefill's
  machinery), decode's own ``state_step`` updates its state, and the readout, left on lane 0 by
  ``untranspose_lane0``, goes back to lane ``c`` (Horner: ``Q <- rot(Q, +1) + acc_c``, one rotation by ``-(K - 1)`` at
  the end). Unlike prefill the K updates are independent: conversation ``c``'s update reads only its own state.
* **the token shift**: two layouts of the carried LayerNorm outputs ``xa`` / ``xf``:
  - ``carry="decode"`` (the brief's layout): conversation ``c`` keeps its own carry in natural form, as a decode step
    does. ``x_prev = sum_c carry_c * mask(lane c)`` (K plaintext products, one level), and the new carry of ``c`` is
    lane ``c`` of ``x`` copied to every lane (a rotation by ``c``, a lane-0 mask, the 5-rotation lane copy).
  - ``carry="lane"`` (a Goal-2 lever): ONE ciphertext per carry, conversation ``c``'s on lane ``c``. That is ``x``
    itself, so the token shift is decode's code with no op at all: ``x_prev = carry``, ``carry <- x``. It drops
    12 K rotations, K plaintext products and 2 (K - 1) carry refreshes per block.

Lanes past K repeat conversation K - 1's input row, its token-shift partner and its readout, so they hold in-range
values that are never read back (with ``carry="lane"`` they equal lane K - 1 exactly, by induction from the public
prefix state, which is the same on every lane).

K = 1 is decode itself (``Rwkv7Fhe.set_batch(1)`` keeps the decode circuit and its op stream).
"""
from __future__ import annotations

import numpy as np

from .model import STATE, Circuit, Mirror, State


class BatchState:
    """One block's carried state for K conversations: ``S[c]`` (its S M ciphertexts, decode's sharded layout) and the
    carries: per conversation a list of shard ciphertexts in natural form (``lane=False``: ``xa[c][j]``), or one list
    of lane-form shard ciphertexts (``lane=True``: ``xa[j]``)."""

    def __init__(self, S, xa, xf, lane: bool):
        self.S = [list(s) for s in S]
        self.xa, self.xf, self.lane = xa, xf, bool(lane)

    @property
    def K(self):
        return len(self.S)

    @property
    def Sh(self):
        return len(self.xa) if self.lane else len(self.xa[0])

    def cts(self):
        out = [c for Sc in self.S for c in Sc]
        if self.lane:
            return out + [*self.xa, *self.xf]
        return out + [c for xs in self.xa for c in xs] + [c for xs in self.xf for c in xs]

    def names(self):
        return state_names(self.K, len(self.S[0]) // self.Sh, self.lane, self.Sh)

    def conv(self, c) -> State:
        """Conversation c's state as a decode ``State`` (its natural-form carries; lane form: None)."""
        return State(self.S[c], None if self.lane else self.xa[c], None if self.lane else self.xf[c])


def state_names(K: int, M: int, lane: bool, Sh: int = 1) -> list[str]:
    """``s<m>[h<j>]c<c>`` per conversation (shard-major within it), then the carries: ``xa[h<j>]`` (lane form) or
    ``xa[h<j>]c<c>`` (conversation-major). Shard 0 keeps the single-shard names."""
    sfx = [""] + [f"h{j}" for j in range(1, Sh)]
    out = [f"{STATE[m]}{x}c{c}" for c in range(K) for x in sfx for m in range(M)]
    if lane:
        return out + [f"xa{x}" for x in sfx] + [f"xf{x}" for x in sfx]
    return out + [f"xa{x}c{c}" for c in range(K) for x in sfx] + [f"xf{x}c{c}" for c in range(K) for x in sfx]


class BatchedCircuit(Circuit):
    """``Circuit`` with conversation ``c`` on lane ``c`` (K <= t conversations)."""

    def __init__(self, ops, L, coeffs, linear, K: int, carry: str = "decode", hoist: int = 0, step=None):
        super().__init__(ops, L, coeffs, linear, step=step)
        if not 2 <= int(K) <= L.t:
            raise ValueError(f"batch K = {K} must be 2..{L.t} (K = 1 is the decode step)")
        if carry not in ("decode", "lane"):
            raise ValueError(f"carry {carry!r}: decode | lane")
        self.K, self.carry, self.hoist = int(K), carry, int(hoist)
        self.m_0 = L.lane_mask([0])
        self.m_0h = L.lane_mask([0], 0.5)
        # conversation c's lanes for the token-shift partner: c, and K - 1 also every lane past K
        self.m_conv = [L.lane_mask([c]) for c in range(self.K - 1)] + [L.lane_mask(range(self.K - 1, L.t))]
        self.m_past = L.lane_mask(range(self.K, L.t)) if self.K < L.t else None

    # ── lane c -> every lane ──
    def bcast_lane0(self, x, value_mask):
        """Lane 0 of ``x`` (times the mask's value) on every lane of its channel: a mask and the 5-rotation copy."""
        return self.ops.rotate_and_sum(self.ops.mult_pt(x, value_mask), -1, self.L.t)

    def lanes_to_0(self, x):
        """Yields ``rot(x, c)`` for c = 0 .. K-1 (lane c of ``x`` on lane 0), lazily, so a pass holds one rotated
        copy per source at a time (B with hoisting). ``hoist = 0``: a running rotation by +1 (one key, chained);
        ``hoist = B``: hoisted rotations from ``x`` itself, B offsets per call (keys 1..K-1)."""
        ops, K = self.ops, self.K
        yield x
        if not self.hoist:
            cur = x
            for _ in range(1, K):
                cur = ops.rotate(cur, 1)
                yield cur
            return
        offs = list(range(1, K))
        for i in range(0, len(offs), self.hoist):
            yield from ops.rotate_many(x, offs[i:i + self.hoist])

    # ── the two per-conversation steps ──
    def token_shift(self, x, carry):
        """``x``: the LN output shards. Lane carries: decode's own token shift (the carry is ``x``). Decode carries:
        ``carry[c]`` conversation c's natural-form shards; per shard the partner is ``sum_c carry_c * mask(lane c)``
        and the new carry of c is lane c of ``x`` on every lane."""
        if self.carry == "lane":
            return carry, x                                       # decode's own token shift: the carry is x
        ops = self.ops
        xp, new = [], [[] for _ in range(self.K)]
        for j, xj in enumerate(x):
            xp.append(ops.mult_pt_sum([cc[j] for cc in carry], self.m_conv))
            for c, xc in enumerate(self.lanes_to_0(xj)):
                new[c].append(self.bcast_lane0(xc, self.m_0))
        return xp, new

    def wkv_shard(self, st: BatchState, j, w, kkn, bb, kt, r, v):
        """Shard j's heads: every conversation's update of its own M state ciphertexts of that shard."""
        ops, L, K, M = self.ops, self.L, self.K, self.L.M
        with self._scope("src"):
            srcs = [ops.add(w, ops.mult_i(kkn)), ops.add(bb, ops.mult_i(kt)), ops.add(r, ops.mult_i(v))]
            at0 = [self.lanes_to_0(s) for s in srcs]
        Q = acc = None
        for c in range(K):
            with self._scope("bcast"):
                vec = []
                for i in range(3):
                    with self._scope("src"):
                        xc = next(at0[i])
                    B = self.bcast_lane0(xc, self.m_0h)                  # (x + i y) / 2 on every lane
                    Bc = ops.conjugate(B)
                    vec += [ops.add(B, Bc), ops.mult_i(ops.sub(Bc, B))]
                wc, kkc, bbc, ktc, rc, vc = vec
                V = L.transpose(ops, vc)
            with self._scope("update"):
                Sj, Ps = self.state_step(st.S[c][j * M:(j + 1) * M], wc, kkc, bbc, ktc, rc, V)
                st.S[c][j * M:(j + 1) * M] = Sj
            with self._scope("readout"):
                acc = L.untranspose_lane0(ops, Ps)
                Q = acc if Q is None else ops.add(ops.rotate(Q, 1), acc)
        with self._scope("readout"):
            y = ops.rotate(Q, -(K - 1))
            if self.m_past is not None:
                y = ops.add(y, ops.mult_pt(ops.rotate_and_sum(acc, -1, L.t), self.m_past))
        return y


def rotation_steps(L, K: int, hoist: int = 0) -> set[int]:
    """The steps a K-conversation pass adds to the decode step's."""
    s = {1, -(K - 1)}
    if hoist:
        s |= set(range(1, K))
    return {x % L.n for x in s} - {0}


def batch_state(lane: bool, K: int, M: int, make, Sh: int = 1) -> BatchState:
    """A BatchState of fresh ciphertexts / vectors: ``make(kind, c, i)`` with kind "S" (i: the state index, shard-major
    over the S M), "xa", "xf" (i: the shard); with ``lane`` the carries are made once (c = None)."""
    S = [[make("S", c, i) for i in range(Sh * M)] for c in range(K)]
    if lane:
        return BatchState(S, [make("xa", None, j) for j in range(Sh)], [make("xf", None, j) for j in range(Sh)], True)
    return BatchState(S, [[make("xa", c, j) for j in range(Sh)] for c in range(K)],
                      [[make("xf", c, j) for j in range(Sh)] for c in range(K)], False)


class BatchedMirror(Mirror):
    """The batched circuit on numpy slot vectors: ``step(rows)`` with row c conversation c's token."""

    def __init__(self, P, coeffs, n_slots, K, carry="decode", n_layers=None, wbits=24, fuse_l1=False, hoist=0,
                 lora="auto", lanes=0, layout="current"):
        self.K, self.carry_kind, self.hoist = int(K), carry, int(hoist)
        super().__init__(P, coeffs, n_slots, n_layers=n_layers, wbits=wbits, fuse_l1=fuse_l1, lora=lora, lanes=lanes,
                         layout=layout)
        self.bcirc = BatchedCircuit(self.circ.ops, self.L, coeffs, self.circ.linear, self.K, carry, hoist)

    def reset(self, state=None):
        Mirror.reset(self, state)
        single = self.state
        lane = self.carry_kind == "lane"
        self.state = [batch_state(lane, self.K, self.L.M,
                                  lambda kind, c, i, s=s: (s.S[i] if kind == "S" else getattr(s, kind)[i]).copy(),
                                  self.L.S)
                      for s in single]

    def step(self, rows):
        rows = np.asarray(rows, dtype=np.float64)
        assert rows.shape[0] == self.K, rows.shape
        z = [v.astype(np.complex128) for v in self.L.chunk_sh(rows)]
        for b in range(self.n_layers):
            z = self.bcirc.block(b, z, self.state[b], self.W[b])
        z = [zi * lv for zi, lv in zip(z, self.L.lives)]
        return self.L.decode_lanes_sh([np.real(zi) for zi in z], self.K), z

    def conv_state(self, c) -> list[State]:
        """Conversation c's carried state in decode's form (natural-form carries; from lane c with ``carry=lane``)."""
        out = []
        for st in self.state:
            if st.lane:
                bc = self.bcirc
                xa = [np.real(bc.bcast_lane0(np.roll(x, -c), bc.m_0)) for x in st.xa]
                xf = [np.real(bc.bcast_lane0(np.roll(x, -c), bc.m_0)) for x in st.xf]
                out.append(State([s.copy() for s in st.S[c]], xa, xf))
            else:
                out.append(State([s.copy() for s in st.S[c]], [x.copy() for x in st.xa[c]],
                                 [x.copy() for x in st.xf[c]]))
        return out
