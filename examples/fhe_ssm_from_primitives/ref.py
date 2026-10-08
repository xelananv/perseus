# SPDX-License-Identifier: MIT
"""Float64 references for one or more sequences (rows of ``[B, d]``).

``exact_step`` is spec_decode/plain_recurrent.py statement by statement with the closed-form
inverse square root; ``MirrorModel`` runs the port's own kernels on ``NumpyOps`` through
``PlainGeometry`` (the same Newton, no slots), so ``fake - mirror`` is a layout or protocol
error, ``mirror - exact`` the Newton approximation.
"""

from __future__ import annotations

import numpy as np

from perseus.impl.ops import NumpyOps

from .block import BlockState, carry, ssm_block
from .geometry import PlainGeometry
from .norm import rmsnorm
from .weights import ModelParams, encode_block


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

    def step(self, rows):
        p, h = self.p, np.asarray(rows, dtype=np.float64)
        see = self.observe or (lambda *_: None)
        for i, b in enumerate(p.blocks()):
            u = rmsnorm_exact(h, 1.0)
            um = u * b.mix + self.u_prev[i] * (1.0 - b.mix)
            self.u_prev[i] = u
            x = um @ b.win
            s = b.a * self.s[i] + (1.0 - b.a) * x
            self.s[i] = s
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
    """The port's kernels on NumpyOps: ``[B, d]`` arrays, every plaintext broadcast."""

    def __init__(self, params: ModelParams, lanes: int, newton_form="perseus", scales=None):
        self.p = params
        self.lanes = lanes
        self.form = newton_form
        self.scales = scales
        self.ops = NumpyOps()
        self.geo = PlainGeometry(self.ops)
        self.start()

    def start(self):
        z = np.zeros((self.lanes, self.p.d))
        self.state = [BlockState(z.copy(), z.copy()) for _ in range(self.p.n_layers)]

    def step(self, rows):
        h = np.array(rows, dtype=np.float64)
        for i, st in enumerate(self.state):
            sc = None if self.scales is None else self.scales[i]
            w = encode_block(self.p.block(i), self.geo, f"b{i}.", scales=sc)
            h = ssm_block(self.ops, self.geo, h, w, st, self.form)
            carry(self.ops, self.geo, st)
        n = rmsnorm(self.ops, self.geo, h, self.p.norm_out, "out.norm", self.form)
        return n * self.geo.lane(self.p.g_out)


def rel_err_rms(got, want):
    got, want = np.asarray(got, dtype=np.float64), np.asarray(want, dtype=np.float64)
    return float(np.sqrt(np.mean((got - want) ** 2)) / np.sqrt(np.mean(want**2)))
