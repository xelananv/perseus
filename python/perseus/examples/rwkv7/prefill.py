# SPDX-License-Identifier: MIT
"""Prefill: a chunk of T <= t prompt tokens in one encrypted pass (PREFILL_DESIGN.md).

The decode step copies ONE token over the t = 32 lanes of the filled layout. A prefill pass puts token ``l`` of the
chunk on lane ``l`` instead ("token lanes"). Every token-parallel step of the block is then the decode step's own
code at the decode step's cost: PCMM applies the same matrix on every lane, every polynomial site and Newton chain is
slot-wise, and the LayerNorm, GroupNorm, ``kk`` and bonus reductions (``channel_sum``, ``block_allreduce``) sum
over channels within a lane. Two steps change:

* **token shift**: ``x_prev`` of lane ``l`` is lane ``l - 1`` of the same ciphertext (one rotation by -1, a mask), and
  lane 0 takes the carried LN output of the previous step (natural form, every lane equal). The carry left for the
  next pass or decode step is lane ``T - 1`` copied over every lane (a rotation, a lane-0 mask, the lane copy).
* **WKV**: the recurrence ``S_t = S_{t-1} (diag(w_t) - kk_t^T (a_t kk_t)) + v_t^T k~_t`` stays sequential, on the
  decode step's state layout (the value index on the lanes), so a pass ends with exactly the state a decode step
  carries. Per token ``t`` its six vectors are moved from lane ``t`` to natural form: the three pairs
  ``w + i kk``, ``(kk a) + i k~``, ``r + i v`` share one broadcast each (a running rotation by +1, a lane-0 mask with
  the factor 1/2, the lane copy), and one conjugation splits a pair (``x = B + conj B``, ``y = i (conj B - B)``).
  The decode step's ``state_step`` then runs unchanged, and its readout, left on lane 0 by ``untranspose_lane0``, is
  accumulated Horner-style (``Q <- rot(Q, +1) + acc_t``; one rotation by ``-(T - 1)`` at the end puts token ``t``'s
  readout on lane ``t``). Lanes past a short chunk (T < t) get the last token's readout.

The chunk's lanes past T repeat its last row (``Layout.encode_rows``) and are never read back.
"""
from __future__ import annotations

from .model import Circuit, State


class PrefillCircuit(Circuit):
    """``Circuit`` on token lanes: lane ``l`` holds token ``l`` of a T-token chunk."""

    def __init__(self, ops, L, coeffs, linear, T: int, step=None):
        super().__init__(ops, L, coeffs, linear, step=step)
        if not 1 <= int(T) <= L.t:
            raise ValueError(f"prefill chunk T = {T} must be 1..{L.t} (the lanes)")
        self.T = int(T)
        self.m_not0 = L.lane_mask(range(1, L.t))
        self.m_0 = L.lane_mask([0])
        self.m_0h = L.lane_mask([0], 0.5)
        self.m_past = L.lane_mask(range(self.T, L.t)) if self.T < L.t else None

    def bcast_lane0(self, x, value_mask):
        """Lane 0 of ``x`` (times the mask's value) on every lane of its channel."""
        return self.ops.rotate_and_sum(self.ops.mult_pt(x, value_mask), -1, self.L.t)

    def token_shift(self, x, carry):
        ops, T = self.ops, self.T
        xp = ops.add(ops.mult_pt(ops.rotate(x, -1), self.m_not0), ops.mult_pt(carry, self.m_0))
        last = x if T == 1 else ops.rotate(x, T - 1)
        return xp, self.bcast_lane0(last, self.m_0)

    def wkv(self, st: State, w, kkn, bb, kt, r, v):
        ops, L = self.ops, self.L
        srcs = [ops.add(w, ops.mult_i(kkn)), ops.add(bb, ops.mult_i(kt)), ops.add(r, ops.mult_i(v))]
        Q = acc = None
        for t in range(self.T):
            if t:
                srcs = [ops.rotate(s, 1) for s in srcs]
            vec = []
            for s in srcs:
                B = self.bcast_lane0(s, self.m_0h)                 # (x + i y) / 2 on every lane
                Bc = ops.conjugate(B)
                vec += [ops.add(B, Bc), ops.mult_i(ops.sub(Bc, B))]
            wt, kkt, bbt, ktt, rt, vt = vec
            st.S, Ps = self.state_step(st.S, wt, kkt, bbt, ktt, rt, L.transpose(ops, vt))
            acc = L.untranspose_lane0(ops, Ps)
            Q = acc if Q is None else ops.add(ops.rotate(Q, 1), acc)
        y = Q if self.T == 1 else ops.rotate(Q, -(self.T - 1))
        if self.m_past is not None:
            y = ops.add(y, ops.mult_pt(ops.rotate_and_sum(acc, -1, L.t), self.m_past))
        return y


def rotation_steps(L, T: int) -> set[int]:
    """The steps a T-token pass adds to the decode step's."""
    s = {1, -1}
    if T > 1:
        s |= {T - 1, -(T - 1)}
    return {x % L.n for x in s} - {0}
