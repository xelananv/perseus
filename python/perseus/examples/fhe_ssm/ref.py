# SPDX-License-Identifier: MIT
"""Float64 references for one or more sequences (rows of ``[B, d]``).

``exact_step`` is spec_decode/plain_recurrent.py statement by statement with the closed-form
inverse square root; ``MirrorModel`` runs the port's own kernels on ``NumpyOps`` through
``PlainGeometry`` (the same Newton, no slots), so ``fake - mirror`` is a layout or protocol
error, ``mirror - exact`` the Newton approximation. The lane-mix axis has no closed form (its
kernel is random): the mirror applies it on rows, the exact model never does.

A pass in any lane configuration (passmasks.py): ``ExactModel.pass_(cfg, rows)`` keeps a state per **stream** (not per
lane-slot) and steps each conversation once and each sequence segment once per position from its stream's state
(fresh: zeros; a reader's first token from the document's final state, the hand-over), padding and parked lane-slots
once from zeros (run, never scored, never stored); ``MirrorModel.pass_`` runs the port's pass kernels on the rows
(``passmasks.RowsPass``: the direct lane form, no slots), so ``fake - mirror`` stays the layout error.
"""

from __future__ import annotations

import numpy as np

from perseus.impl.ops import NumpyOps

from . import passmasks
from .block import BlockState, carry, ssm_block
from .geometry import PlainGeometry, park
from .norm import rmsnorm
from .weights import LANE_MIX_T, ModelParams, encode_block


def rmsnorm_exact(x, g, eps=1e-5):
    x = np.asarray(x, dtype=np.float64)
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * g


class ExactModel:
    """The trainer's forward with the carried state, float64, closed-form norm. The gains
    stay folded into the linears' rows, where they act exactly as on the normalized value."""

    def __init__(self, params: ModelParams, lanes: int, observe=None):
        """``observe(block, name, value)``, when given, sees every product the port's kernels
        refresh (scales.py names them) and changes nothing."""
        self.p = params
        self.lanes = lanes
        self.observe = observe
        self.start()

    def start(self):
        d = self.p.d
        self.u_prev = [np.zeros((self.lanes, d)) for _ in range(self.p.n_layers)]
        self.s = [np.zeros((self.lanes, d)) for _ in range(self.p.n_layers)]
        #: pass_: per stream, its (s, u_prev) per block (``(d,)`` each); the registers' writers
        self.streams: dict[str, tuple[list[np.ndarray], list[np.ndarray]]] = {}
        self.tracker: passmasks.Tracker | None = None
        #: pass_: what every lane-slot computed in the last pass, per block (s, u), ``(2t, d)`` each
        self.lane_states: tuple[list[np.ndarray], list[np.ndarray]] = ([], [])

    def step(self, rows):
        def shift(i, u):
            prev = self.u_prev[i]
            self.u_prev[i] = u
            return prev

        def scan(i, b, x):
            s = b.a * self.s[i] + (1.0 - b.a) * x
            self.s[i] = s
            return s

        return self._forward(np.asarray(rows, dtype=np.float64), shift, scan)

    def sequences(self, rows, chunk: int = 0):
        """The trainer's recurrence over whole sequences from the zero state: ``rows`` ``(B, T, d)``, the embedding rows of
        ``B`` sequences of ``T`` positions; returns ``(s, u, out)``, per block the scan state and the normalized input
        after the last position (``(B, d)`` each: what ``step`` holds after ``T`` calls on the ``B`` rows) and the replies
        ``(B, T, d)``. The positions run together through ``_forward`` (a sequence's shift and scan in place of ``step``'s
        registers), so each block's matrices are read once per call instead of once per position; per row the
        arithmetic is ``step``'s statement for statement (the scan's own expression, position by position), up to the
        summation order of a matrix product over many rows instead of one. ``chunk`` > 0: that many sequences per
        ``_forward`` (an intermediate holds ``chunk T d_ffn`` floats)."""
        R = np.asarray(rows, dtype=np.float64)
        if R.ndim != 3 or R.shape[2] != self.p.d:
            raise ValueError(f"sequences are (B, T, {self.p.d}) rows, got {R.shape}")
        B, T, d = R.shape
        L = self.p.n_layers
        per = B if chunk <= 0 else int(chunk)
        s_out = [np.empty((B, d)) for _ in range(L)]
        u_out = [np.empty((B, d)) for _ in range(L)]
        out = np.empty((B, T, d))
        for c0 in range(0, B, per):
            nb = min(B, c0 + per) - c0
            fin_s: list[np.ndarray] = [np.empty(0)] * L
            fin_u: list[np.ndarray] = [np.empty(0)] * L

            def shift(i, u, nb=nb, fin_u=fin_u):
                U = u.reshape(nb, T, d)
                prev = np.zeros_like(U)
                prev[:, 1:] = U[:, :-1]
                fin_u[i] = U[:, -1].copy()
                return prev.reshape(nb * T, d)

            def scan(i, b, x, nb=nb, fin_s=fin_s):
                X = x.reshape(nb, T, d)
                S = np.empty_like(X)
                sp = np.zeros((nb, d))
                for t in range(T):
                    sp = b.a * sp + (1.0 - b.a) * X[:, t]
                    S[:, t] = sp
                fin_s[i] = S[:, -1].copy()
                return S.reshape(nb * T, d)

            o = self._forward(R[c0 : c0 + nb].reshape(nb * T, d), shift, scan)
            out[c0 : c0 + nb] = o.reshape(nb, T, d)
            for i in range(L):
                s_out[i][c0 : c0 + nb] = fin_s[i]
                u_out[i][c0 : c0 + nb] = fin_u[i]
        return s_out, u_out, out

    def pass_(self, cfg: passmasks.PassConfig, rows):
        """One pass of a lane configuration (the module docstring): the ``2t`` reply rows in lane-slot order. Checks
        the pass against the registers (``passmasks.Tracker``: raises on a refusal) before stepping anything."""
        rows = np.asarray(rows, dtype=np.float64)
        n, d, L = cfg.lane_slots, self.p.d, self.p.n_layers
        if rows.shape != (n, d):
            raise ValueError(f"a pass of {n} lane-slots takes ({n}, {d}) rows, got {rows.shape}")
        if self.tracker is None:
            self.tracker = passmasks.Tracker(n)
        src = self.tracker.resolve(cfg)
        # per lane-slot: the previous position it reads (-1: none) or the stream whose state it enters from
        prev = [-1] * n
        entry: list[str | None] = [None] * n
        for q, (role, k) in enumerate(cfg.roles()):
            if role == "pos" and q > cfg.segments[k].q0:
                prev[q] = q - 1
            elif role in ("pos", "conv"):
                entry[q] = src[k]
        lane_s: list[np.ndarray] = [np.zeros((n, d))] * L
        lane_u: list[np.ndarray] = [np.zeros((n, d))] * L

        def shift(i, u):
            out = np.zeros_like(u)
            for q in range(n):
                e = entry[q]
                if prev[q] >= 0:
                    out[q] = u[prev[q]]
                elif e is not None:
                    out[q] = self.streams[e][1][i]
            lane_u[i] = u
            return out

        def scan(i, b, x):
            s = np.empty_like(x)
            for q in range(n):  # in q order: a position's predecessor is computed before it
                e = entry[q]
                if prev[q] >= 0:
                    sp = s[prev[q]]
                elif e is not None:
                    sp = self.streams[e][0][i]
                else:
                    sp = np.zeros(d)
                s[q] = b.a * sp + (1.0 - b.a) * x[q]
            lane_s[i] = s
            return s

        out = self._forward(rows, shift, scan)
        new = {
            str(seg.stream): (
                [lane_s[i][seg.last].copy() for i in range(L)],
                [lane_u[i][seg.last].copy() for i in range(L)],
            )
            for seg in cfg.segments
            if seg.live
        }
        self.streams.update(new)
        self.tracker.commit(cfg)
        self.lane_states = (lane_s, lane_u)
        return out

    def _forward(self, h, shift, scan):
        """The forward over the blocks for rows ``h``: ``shift(i, u)`` is block i's previous-token normalized input
        (given this token's ``u``), ``scan(i, b, x)`` block i's scan state; the rest statement by statement."""
        p = self.p
        see = self.observe or (lambda *_: None)
        for i, b in enumerate(p.blocks()):
            u = rmsnorm_exact(h, 1.0)
            um = u * b.mix + shift(i, u) * (1.0 - b.mix)
            x = um @ b.win
            s = scan(i, b, x)
            z = b.c * s + b.dd * x
            p1z, p2z2 = b.p1 * z, b.p2 * (z * z)
            poly = b.p0 + p1z + p2z2
            g = z * poly
            for name, v in (("tm.p1z", p1z), ("tm.p2z2", p2z2), ("tm.poly", poly), ("tm.g", g)):
                see(i, name, v)
            h = h + g @ b.wout
            u2 = rmsnorm_exact(h, 1.0)
            k, r = u2 @ b.wk, u2 @ b.wr
            q1k, q2k2, q3k = b.q1 * k, b.q2 * (k * k), b.q3 * k
            q3k3 = q3k * (k * k)
            act = b.q0 + q1k + q2k2 + q3k3
            r1r, r2r2 = b.r1 * r, b.r2 * (r * r)
            gate = b.r0 + r1r + r2r2
            ag = act * gate
            for name, v in (
                ("cm.q1k", q1k),
                ("cm.q2k2", q2k2),
                ("cm.q3k", q3k),
                ("cm.q3k3", q3k3),
                ("cm.act", act),
                ("cm.r1r", r1r),
                ("cm.r2r2", r2r2),
                ("cm.gate", gate),
                ("cm.ag", ag),
            ):
                see(i, name, v)
            h = h + ag @ b.wv
        return rmsnorm_exact(h, p.g_out)


class MirrorModel:
    """The port's kernels on NumpyOps: ``[B, d]`` arrays, every plaintext broadcast. With the lane mix on,
    fewer than ``2t`` conversations are parked as the fake parks its lanes (copies of conversation 0 that
    carry their own state across blocks and ticks, read by the live lanes above them) and dropped on output."""

    def __init__(
        self,
        params: ModelParams,
        lanes: int,
        newton_form="perseus",
        scales=None,
        linear="diagonal",
        pcmm_wbits=16,
    ):
        """``linear`` / ``pcmm_wbits`` as the model's: the mirror then multiplies by the matrices PCMM rounds to."""
        self.p = params
        self.lanes = lanes
        self.form = newton_form
        self.scales = scales
        self.ops = NumpyOps()
        self.geo = PlainGeometry(self.ops, linear=linear, pcmm_wbits=pcmm_wbits)
        # the rows the kernels run on: the live conversations, parked up to 2t with the lane mix on
        self.rows_held = 2 * LANE_MIX_T if params.lane_mix != "off" and lanes < 2 * LANE_MIX_T else lanes
        self.start()

    def start(self):
        z = np.zeros((self.rows_held, self.p.d))
        self.state = [BlockState(z.copy(), z.copy()) for _ in range(self.p.n_layers)]

    def pass_(self, cfg: passmasks.PassConfig, rows):
        """One pass of a lane configuration on the rows (lane-slot order, ``2t`` of them): the port's blocks with the
        pass kernels in their rows form (``passmasks.RowsPass``), the registers per lane-slot as on the slots."""
        rows = np.asarray(rows, dtype=np.float64)
        if rows.shape[0] != self.rows_held or cfg.lane_slots != rows.shape[0]:
            raise ValueError(f"a pass of {cfg.lane_slots} lane-slots on a mirror of {self.rows_held} rows")
        passmasks.validate(cfg)
        h = rows.copy()
        for i, st in enumerate(self.state):
            sc = None if self.scales is None else self.scales[i]
            bp = self.p.block(i)
            w = encode_block(bp, self.geo, f"b{i}.", scales=sc)
            pk = passmasks.RowsPass(passmasks.coefficients(cfg, bp.a, bp.mix))
            h = ssm_block(self.ops, self.geo, h, w, st, self.form, "off", pk)
            carry(self.ops, self.geo, st)
        n = rmsnorm(self.ops, self.geo, h, self.p.norm_out, "out.norm", self.form)
        return n * self.geo.lane(self.p.g_out)

    def step(self, rows):
        rows = np.asarray(rows, dtype=np.float64)
        if rows.shape[0] != self.lanes:
            raise ValueError(f"{rows.shape[0]} rows for a mirror of {self.lanes} conversations")
        h = park(rows, self.rows_held).copy()
        for i, st in enumerate(self.state):
            sc = None if self.scales is None else self.scales[i]
            w = encode_block(self.p.block(i), self.geo, f"b{i}.", scales=sc)
            h = ssm_block(self.ops, self.geo, h, w, st, self.form)
            carry(self.ops, self.geo, st)
        n = rmsnorm(self.ops, self.geo, h, self.p.norm_out, "out.norm", self.form)
        return (n * self.geo.lane(self.p.g_out))[: self.lanes]


def rel_err_rms(got, want):
    got, want = np.asarray(got, dtype=np.float64), np.asarray(want, dtype=np.float64)
    return float(np.sqrt(np.mean((got - want) ** 2)) / np.sqrt(np.mean(want**2)))
