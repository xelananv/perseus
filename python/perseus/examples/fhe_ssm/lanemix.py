# SPDX-License-Identifier: MIT
"""A cost-only lane-mixing term: the Toeplitz over a filled ciphertext's lanes that a chunked prefill would
apply (the A2i form), with a random kernel; its count is the point, not its value.

``T(x) = sum_s P_s (.) rot(x, -s) + P'_s (.) conj(rot(x, -s))`` over the ``t = 32`` lane offsets, as a
baby-step giant-step (``s = 8 g + b``): the 7 baby rotations of ``x`` and of ``conj x`` from one hoisted
decomposition each, pre-rotated plaintexts per group (zero where a group's sum would cross into the next
channel's lanes, and on the lanes below the offset), one giant rotation per group. Per block: 64 plaintext
products, 14 hoisted + 3 giant rotations, one conjugation, depth 1. ``LaneMixKernel.rows`` is the same term
on rows (the mirror). README.md derives the plaintexts and the keys.
"""

from __future__ import annotations

import numpy as np

#: baby and giant steps over the t = 32 lanes
NB, NG = 8, 4


class LaneMixKernel:
    """``P`` and ``Pc``: ``(t, d)`` complex coefficients, offset ``s`` per channel, for ``y_s`` and ``conj(y_s)``."""

    def __init__(self, P, Pc, N: int, t: int):
        self.P, self.Pc = np.asarray(P, dtype=np.complex128), np.asarray(Pc, dtype=np.complex128)
        self.N, self.t = int(N), int(t)
        if self.t != NB * NG:
            raise ValueError(f"the lane-mix BSGS is cut for t = {NB * NG} lanes, got t = {self.t}")
        if self.P.shape != (self.t, self.N // self.t) or self.Pc.shape != self.P.shape:
            raise ValueError(f"kernels must be (t, d) = ({self.t}, {self.N // self.t}), got {self.P.shape}")

    @staticmethod
    def random(d: int, N: int, t: int, seed: int, scale: float = 1e-2) -> LaneMixKernel:
        rng = np.random.default_rng(seed)
        P, Pc = (scale * (rng.standard_normal((t, d)) + 1j * rng.standard_normal((t, d))) for _ in range(2))
        return LaneMixKernel(P, Pc, N, t)

    def pt(self, g: int, b: int, conj: bool) -> np.ndarray:
        """The pre-rotated plaintext of baby ``b`` in giant group ``g``."""
        s = NB * g + b
        lanes = np.arange(self.t)
        src = lanes + NB * g
        valid = ((src < self.t) & (src >= s)).astype(np.float64)
        coef = (self.Pc if conj else self.P)[s]
        return (coef[:, None] * valid[None, :]).reshape(-1)

    def apply(self, ops, x):
        """``T(x)`` on the slots: the op stream of the module docstring."""
        xc = ops.conjugate(x)
        babies = []
        for v in (x, xc):
            rot = [v, *ops.rotate_many(v, [-b for b in range(1, NB)])]
            for r in rot:
                ops.realize(r)
            babies.append(rot)
        total = None
        for g in range(NG):
            acc = None
            for b in range(NB):
                p = ops.add(
                    ops.mult_pt(babies[0][b], self.pt(g, b, False)),
                    ops.mult_pt(babies[1][b], self.pt(g, b, True)),
                )
                acc = p if acc is None else ops.add(acc, p)
            if g:
                acc = ops.rotate(acc, -NB * g)
            total = acc if total is None else ops.add(total, acc)
        return total

    def rows(self, rows) -> np.ndarray:
        """``T`` on ``(2t, d)`` rows (the mirror): conversation ``j`` real, ``j + t`` imaginary."""
        rows = np.asarray(rows, dtype=np.float64)
        t = self.t
        if rows.shape[0] != 2 * t:
            raise ValueError(f"the lane mix needs {2 * t} conversations (pairs), got {rows.shape[0]}")
        X = rows[:t] + 1j * rows[t:]
        T = np.zeros_like(X)
        for s in range(t):
            y = np.zeros_like(X)
            y[s:] = X[: t - s]
            T += self.P[s][None, :] * y + self.Pc[s][None, :] * np.conj(y)
        return np.concatenate([T.real, T.imag])

    @staticmethod
    def rotation_steps(N: int) -> set[int]:
        return {(-b) % N for b in range(1, NB)} | {(-NB * g) % N for g in range(1, NG)}
