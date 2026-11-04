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
    """One block's carried state for K conversations: ``S[c]`` (M ciphertexts each, decode's layout) and the carries,
    a list of K natural-form ciphertexts each (``lane=False``) or one lane-form ciphertext each (``lane=True``)."""

    def __init__(self, S, xa, xf, lane: bool):
        self.S = [list(s) for s in S]
        self.xa, self.xf, self.lane = xa, xf, bool(lane)

    @property
    def K(self):
        return len(self.S)

    def cts(self):
        out = [c for Sc in self.S for c in Sc]
        return out + ([self.xa, self.xf] if self.lane else [*self.xa, *self.xf])

    def names(self):
        K, M = self.K, len(self.S[0])
        out = [f"{STATE[m]}c{c}" for c in range(K) for m in range(M)]
        return out + (["xa", "xf"] if self.lane else [f"xac{c}" for c in range(K)] + [f"xfc{c}" for c in range(K)])

    def conv(self, c) -> State:
        """Conversation c's state as a decode ``State`` (the carry: its natural-form carry; lane form: None)."""
        return State(self.S[c], None if self.lane else self.xa[c], None if self.lane else self.xf[c])


def state_names(K: int, M: int, lane: bool) -> list[str]:
    return BatchState([[None] * M for _ in range(K)], None if lane else [None] * K,
                      None if lane else [None] * K, lane).names()


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
        if self.carry == "lane":
            return carry, x                                       # decode's own token shift: the carry is x
        ops = self.ops
        xp = ops.mult_pt_sum(list(carry), self.m_conv)
        return xp, [self.bcast_lane0(xc, self.m_0) for xc in self.lanes_to_0(x)]   # noqa: C416

    def wkv(self, st: BatchState, w, kkn, bb, kt, r, v):
        ops, L, K = self.ops, self.L, self.K
        with self._scope("src"):
            srcs = [ops.add(w, ops.mult_i(kkn)), ops.add(bb, ops.mult_i(kt)), ops.add(r, ops.mult_i(v))]
            at0 = [self.lanes_to_0(s) for s in srcs]
        Q = acc = None
        for c in range(K):
            with self._scope("bcast"):
                vec = []
                for j in range(3):
                    with self._scope("src"):
                        xc = next(at0[j])
                    B = self.bcast_lane0(xc, self.m_0h)                  # (x + i y) / 2 on every lane
                    Bc = ops.conjugate(B)
                    vec += [ops.add(B, Bc), ops.mult_i(ops.sub(Bc, B))]
                wc, kkc, bbc, ktc, rc, vc = vec
                V = L.transpose(ops, vc)
            with self._scope("update"):
                st.S[c], Ps = self.state_step(st.S[c], wc, kkc, bbc, ktc, rc, V)
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


def batch_state(lane: bool, K: int, M: int, make) -> BatchState:
    """A BatchState of fresh ciphertexts / vectors: ``make(kind, c)`` with kind in ("S", "xa", "xf"); with
    ``lane`` the carries are made once (c = None)."""
    S = [[make("S", c, m) for m in range(M)] for c in range(K)]
    if lane:
        return BatchState(S, make("xa", None, None), make("xf", None, None), True)
    return BatchState(S, [make("xa", c, None) for c in range(K)], [make("xf", c, None) for c in range(K)], False)


class BatchedMirror(Mirror):
    """The batched circuit on numpy slot vectors: ``step(rows)`` with row c conversation c's token."""

    def __init__(self, P, coeffs, n_slots, K, carry="decode", n_layers=None, wbits=24, fuse_l1=False, hoist=0):
        self.K, self.carry_kind, self.hoist = int(K), carry, int(hoist)
        super().__init__(P, coeffs, n_slots, n_layers=n_layers, wbits=wbits, fuse_l1=fuse_l1)
        self.bcirc = BatchedCircuit(self.circ.ops, self.L, coeffs, self.circ.linear, self.K, carry, hoist)

    def reset(self, state=None):
        Mirror.reset(self, state)
        single = self.state
        lane = self.carry_kind == "lane"
        self.state = [batch_state(lane, self.K, self.L.M,
                                  lambda kind, c, m, s=s: (s.S[m] if kind == "S" else getattr(s, kind)).copy())
                      for s in single]

    def step(self, rows):
        rows = np.asarray(rows, dtype=np.float64)
        assert rows.shape[0] == self.K, rows.shape
        z = self.L.chunk(rows).astype(np.complex128)
        for b in range(self.n_layers):
            z = self.bcirc.block(b, z, self.state[b], self.W[b])
        z = z * self.L.live
        return self.L.decode_lanes(np.real(z), self.K), z

    def conv_state(self, c) -> list[State]:
        """Conversation c's carried state in decode's form (natural-form carries; from lane c with ``carry=lane``)."""
        out = []
        for st in self.state:
            if st.lane:
                bc = self.bcirc
                xa, xf = (np.real(bc.bcast_lane0(np.roll(x, -c), bc.m_0)) for x in (st.xa, st.xf))
                out.append(State([s.copy() for s in st.S[c]], xa, xf))
            else:
                out.append(State([s.copy() for s in st.S[c]], st.xa[c].copy(), st.xf[c].copy()))
        return out
