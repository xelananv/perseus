# SPDX-License-Identifier: MIT
"""How values and per-channel constants meet a ciphertext in the layouts the kernels run on.

Every geometry offers the same protocol, so ``block.py`` and ``norm.py`` are written once:

    lane(v) / ffn(v)        a d-wide / d_ffn-wide per-channel constant (ffn: one vector per tile)
    weight(W, name)         a linear in this layout's form
    linear(x, w)            d -> d
    up(x, [w, ...])         d -> d_ffn for several weights on one prepared input: [[tiles], ...]
    down(xs, w)             d_ffn (tiles) -> d
    masked_scale(x, c, s)   c x on the lanes, zero elsewhere (one product)
    sum_all(x)              each conversation's sum over its channels, at all its slots
    hint(x, products)       refresh x unless `products` more products fit
    section(fn, xs)         run a nonlinear kernel fn(ops, geo, xs) -> outputs
    split_gain              what a section's inputs must be multiplied by (0.5 on split pairs)
    capacity                conversations per ciphertext
    encode_rows / decode_rows   the client's side of the layout

``SlotGeometry``: the cachemir decode packing (perseus.impl.layout). Channel ``i`` of a d-wide
value sits at slot ``i*t`` with ``t = N / hid`` and the gaps are the linear's workspace; a
d_ffn-wide value follows the up-linear's output layout, slot ``m*tp_E`` carrying channel
``interleave_idx(m, hid, d_ffn)``. One sequence per ciphertext. A fresh encode and every norm
output are zero off the lanes; a linear output is not, so every value that enters a linear or a
reduction passes through a per-channel plaintext first, and the norm's mask is that plaintext
for the residual stream.

``FilledGeometry``: the dev branch's filling packing. Channel ``i`` of conversation ``k`` sits
at slot ``i*t + k``, ``t = N / d`` conversations per ciphertext, every slot live; a d_ffn-wide
value is ``d_ffn / d`` ciphertexts of ``d`` channels (filling.py).

``PairGeometry``: the filled layout with conversation ``k`` on the real axis and ``k + t`` on
the imaginary axis (pairs.py): ``2t`` conversations per ciphertext. Linears, plaintext products
and reductions are those of the filled layout; nonlinear sections split or run lane-wise.

``PlainGeometry``: numpy rows ``[B, d]``, the mirror. The same kernels run on it through
``NumpyOps``.
"""

from __future__ import annotations

import contextlib

import numpy as np

from perseus.impl.layout import encode_bias_vector, lane_vec, linear_rot_steps, stride_mask
from perseus.impl.linear import EncodedLinear, linear, linear_multi

from . import filling, pairs

ROUTES = ("split", "lane")


def _rows(rows, d: int, capacity: int) -> np.ndarray:
    """``(n, d)`` float64 rows, ``1 <= n <= capacity``."""
    rows = np.asarray(rows, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != d:
        raise ValueError(f"rows must be (n, {d}), got {rows.shape}")
    if not 1 <= rows.shape[0] <= capacity:
        raise ValueError(f"{rows.shape[0]} rows for a capacity of {capacity}")
    return rows


def park(rows: np.ndarray, capacity: int) -> np.ndarray:
    """``rows`` filled up to ``capacity`` with copies of the first row. A lane left at zero
    would enter every norm far below its Newton seed's window and diverge, and a bootstrap
    needs every slot of a ciphertext in range; a parked lane runs a copy of conversation 0
    instead, and its outputs are dropped."""
    n = rows.shape[0]
    if n >= capacity:
        return rows
    return np.concatenate([rows, np.repeat(rows[:1], capacity - n, axis=0)])


class SlotGeometry:
    split_gain = 1.0
    capacity = 1

    def __init__(self, rt):
        self.rt = rt
        self.dims = rt.dims

    def lane(self, v):
        return lane_vec(v, self.dims.N, self.dims.t)

    def expanded(self, v):
        d = self.dims
        return encode_bias_vector(v, d.N, d.hid, d.E, fill=False)

    def ffn(self, v):
        return [self.expanded(v)]

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

    def up(self, x, ws):
        return [[y] for y in linear_multi(self.rt, x, ws)]

    def down(self, xs, w):
        if len(xs) != 1:
            raise ValueError(f"the decode layout holds d_ffn in one ciphertext, got {len(xs)}")
        return linear(self.rt, xs[0], w)

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

    def section(self, fn, xs):
        return fn(self.rt.ops, self, xs)

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

    def encode_rows(self, rows):
        rows = _rows(rows, self.dims.dim, 1)
        return self.rt.ops.encode_token(rows[0])

    def decode_rows(self, ct, n: int):
        if n != 1:
            raise ValueError(f"the decode layout holds one sequence, asked for {n}")
        return self.rt.ops.decode_token(ct)[None, : self.dims.dim]


class FilledGeometry:
    split_gain = 1.0

    def __init__(self, rt, d: int, d_ffn: int):
        n = rt.dims.N
        if n % d or d_ffn % d:
            raise ValueError(f"N={n} and d_ffn={d_ffn} must be multiples of d={d}")
        self.rt = rt
        self.N, self.d, self.d_ffn = n, d, d_ffn
        self.t = n // d
        self.n_tiles = d_ffn // d

    @property
    def ops(self):
        return self.rt.ops

    @property
    def capacity(self) -> int:
        return self.t

    def lane(self, v):
        v = np.asarray(v, dtype=np.float64)
        if v.shape != (self.d,):
            raise ValueError(f"a per-channel constant must be ({self.d},), got {v.shape}")
        return filling.lane_const(v, self.N)

    def ffn(self, v):
        v = np.asarray(v, dtype=np.float64)
        if v.shape != (self.d_ffn,):
            raise ValueError(f"a d_ffn constant must be ({self.d_ffn},), got {v.shape}")
        return [self.lane(v[j * self.d : (j + 1) * self.d]) for j in range(self.n_tiles)]

    def weight(self, w, name):
        return filling.encode_tiled(w, self.N, self.d, name)

    def linear(self, x, w):
        return filling.linear_tiled(self.ops, [x], w)[0]

    def up(self, x, ws):
        return filling.linear_multi(self.ops, [x], ws)

    def down(self, xs, w):
        return filling.linear_tiled(self.ops, xs, w)[0]

    def masked_scale(self, x, scale, site):  # noqa: ARG002 (the geometry protocol)
        """Every slot is a lane here, so the scale is a scalar product (one level, no mask)."""
        return self.ops.mult(x, float(scale))

    def sum_all(self, x):
        return filling.per_conversation_sum(self.ops, x, self.N, self.d)

    def hint(self, x, products=1):
        ops = self.ops
        return ops.bootstrap_hint(x, ops.headroom(products), True)

    def step(self, label):
        return self.rt.step(label)

    def section(self, fn, xs):
        return fn(self.ops, self, xs)

    def rotation_steps(self):
        return filling.rotation_steps(self.N, self.d)

    def encode_rows(self, rows):
        """Up to ``t`` conversations (the rest parked), encrypted fresh at the bootstrap landing
        level."""
        rows = park(_rows(rows, self.d, self.t), self.t)
        return self.ops.pack_tokens(rows, self.ops.bts_out())

    def decode_rows(self, ct, n: int):
        if not 1 <= n <= self.t:
            raise ValueError(f"{n} conversations for a capacity of {self.t}")
        return filling.unpack(self.ops.decrypt_slots(ct), self.d, n)


class PairGeometry:
    def __init__(self, base: FilledGeometry, route: str = "split"):
        if route not in ROUTES:
            raise ValueError(f"route must be one of {ROUTES}, got {route!r}")
        if not getattr(base.ops, "complex_payload", False):
            raise ValueError("pairs need a session whose slots carry complex values")
        self.base, self.route = base, route
        self.split_gain = 0.5 if route == "split" else 1.0
        self._lane_ops = pairs.PairLaneOps(base.ops) if route == "lane" else None

    @property
    def rt(self):
        return self.base.rt

    @property
    def ops(self):
        return self.base.ops

    @property
    def capacity(self) -> int:
        return 2 * self.base.t

    def lane(self, v):
        return self.base.lane(v)

    def ffn(self, v):
        return self.base.ffn(v)

    def weight(self, w, name):
        return self.base.weight(w, name)

    def linear(self, x, w):
        return self.base.linear(x, w)

    def up(self, x, ws):
        return self.base.up(x, ws)

    def down(self, xs, w):
        return self.base.down(xs, w)

    def masked_scale(self, x, scale, site):
        return self.base.masked_scale(x, scale, site)

    def sum_all(self, x):
        return self.base.sum_all(x)

    def hint(self, x, products=1):
        return self.base.hint(x, products)

    def step(self, label):
        return self.base.step(label)

    def rotation_steps(self):
        return self.base.rotation_steps()

    def section(self, fn, xs):
        """The kernel on both axes of every pair. ``split``: on the real halves ``2a`` and
        ``2b`` (``split_gain`` tells the caller to have halved the inputs), repacked as
        ``a' + i b'``; ``lane``: once on the pairs, with lane-wise products."""
        if self._lane_ops is not None:
            return fn(self._lane_ops, self.base, xs)
        ops = self.base.ops
        halves = [pairs.split(ops, x) for x in xs]
        out_a = fn(ops, self.base, [a for a, _ in halves])
        out_b = fn(ops, self.base, [b for _, b in halves])
        if len(out_a) != len(out_b):
            raise RuntimeError("a section returned different output counts for its two halves")
        return [pairs.repack(ops, a, b) for a, b in zip(out_a, out_b)]

    def encode_rows(self, rows):
        """Up to ``2t`` conversations (the rest parked): the first ``t`` on the real axis, the
        rest on the imaginary axis (two fresh encryptions, joined by a level-free product by
        ``i``)."""
        t = self.base.t
        rows = park(_rows(rows, self.base.d, 2 * t), 2 * t)
        return pairs.repack(
            self.ops, self.base.encode_rows(rows[:t]), self.base.encode_rows(rows[t:])
        )

    def decode_rows(self, ct, n: int):
        if not 1 <= n <= self.capacity:
            raise ValueError(f"{n} conversations for a capacity of {self.capacity}")
        return filling.unpack_pairs(self.ops.decrypt_slots_complex(ct), self.base.d, n)


class PlainGeometry:
    split_gain = 1.0

    def __init__(self, ops):
        self.ops = ops

    @staticmethod
    def lane(v):
        return np.asarray(v, dtype=np.float64)

    expanded = lane

    def ffn(self, v):
        return [self.lane(v)]

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
    def up(x, ws):
        return [[x @ w] for w in ws]

    @staticmethod
    def down(xs, w):
        if len(xs) != 1:
            raise ValueError(f"the plain mirror holds d_ffn in one array, got {len(xs)}")
        return xs[0] @ w

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

    def section(self, fn, xs):
        return fn(self.ops, self, xs)
