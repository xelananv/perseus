"""How a per-channel vector meets a ciphertext in the two layouts the kernels run on.

Slot layout (the cachemir token packing, perseus.impl.layout): channel ``i`` of a ``d``-wide
value sits at slot ``i*t`` with ``t = N / hid`` and the gaps are the linear's workspace; a
``d_ffn``-wide value follows the up-linear's output layout, slot ``m*tp_E`` carrying channel
``interleave_idx(m, hid, d_ffn)`` (what ``encode_bias_vector`` with ``fill=False`` produces).
A fresh encode and every norm output are zero off the lanes; a linear output is not, so every
value that enters a linear or a reduction passes through a per-channel plaintext first, and the
norm's mask is that plaintext for the residual stream.

Plain layout: numpy vectors of ``d`` or ``d_ffn`` entries, the mirror. The same kernels run on
both through this object and ``NumpyOps``.
"""

from __future__ import annotations

import contextlib

import numpy as np

from perseus.impl.layout import encode_bias_vector, lane_vec, linear_rot_steps, stride_mask
from perseus.impl.linear import EncodedLinear, linear, linear_multi


class SlotGeometry:
    def __init__(self, rt):
        self.rt = rt
        self.dims = rt.dims

    def lane(self, v):
        return lane_vec(v, self.dims.N, self.dims.t)

    def expanded(self, v):
        d = self.dims
        return encode_bias_vector(v, d.N, d.hid, d.E, fill=False)

    def weight(self, w, name):
        d_in, d_out = w.shape
        return EncodedLinear.encode(w, self.dims.N, d_in, d_out, name=name)

    def masked_scale(self, x, scale, site):
        """``scale`` on the lane slots, zero elsewhere: the one plaintext product that both
        cleans a residual-stream value for the reduction and scales it."""
        d = self.dims
        return self.rt.mult_mask(x, (site, scale), lambda: stride_mask(d.N, d.dim, d.t, scale))

    def linear(self, x, w):
        return linear(self.rt, x, w)

    def linear_multi(self, x, ws):
        return linear_multi(self.rt, x, ws)

    def sum_all(self, x):
        """The sum over every slot, broadcast to every slot (rotate_and_sum_all), tagged as a
        1-periodic payload so a planned refresh of it can route sparse."""
        ops = self.rt.ops
        return ops.tag_reduce(ops.rotate_and_sum(x, 1, self.dims.N), 1)

    def hint(self, x, products=1):
        """Refresh ``x`` unless ``products`` more products fit under the ceiling. On the runtime
        a product lands at its operand's effective level with the rescale pending and refreshes
        when that level reaches the ceiling, so ``x`` must go once its effective level exceeds
        ``limit - products * unit``: Perseus's ``level_headroom(1)`` before every linear
        (src/model/mlp.cu). The stock numpy fake charges the rescale at once and refreshes one
        product earlier; runtime_fake.py has the runtime's rule."""
        ops = self.rt.ops
        return ops.bootstrap_hint(x, ops.headroom(products), True)

    def step(self, label):
        """A step scope: labels the ops inside in the captured graph and the timing layer."""
        return self.rt.step(label)

    def rotation_steps(self):
        """Every rotation the port issues: the three linear shapes and the all-slot ladder."""
        d = self.dims
        steps = set()
        for d_in, d_out in ((d.hid, d.hid), (d.hid, d.E), (d.E, d.hid)):
            steps |= linear_rot_steps(d.N, d_in, d_out)
        gap = 1
        while gap < d.N:
            steps.add(gap)
            gap *= 2
        return steps


class PlainGeometry:
    def __init__(self, ops):
        self.ops = ops

    @staticmethod
    def lane(v):
        return np.asarray(v, dtype=np.float64)

    expanded = lane

    @staticmethod
    def weight(w, name):  # noqa: ARG004 (the geometry protocol)
        return np.asarray(w, dtype=np.float64)

    def masked_scale(self, x, scale, site):  # noqa: ARG002 (the geometry protocol)
        return self.ops.mult(x, scale)

    @staticmethod
    def linear(x, w):
        return x @ w

    @staticmethod
    def linear_multi(x, ws):
        return [x @ w for w in ws]

    @staticmethod
    def sum_all(x):
        """Each sequence's sum broadcast over its own row: what the all-slot ladder gives a
        payload that is zero off the lanes, one row per sequence."""
        return np.broadcast_to(np.sum(x, axis=-1, keepdims=True), np.shape(x)).copy()

    def hint(self, x, products=1):  # noqa: ARG002 (the geometry protocol)
        return x

    @staticmethod
    def step(label):  # noqa: ARG004 (the geometry protocol)
        return contextlib.nullcontext()
