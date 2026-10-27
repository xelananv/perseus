# SPDX-License-Identifier: MIT
"""How values and per-channel constants meet a ciphertext in each layout the kernels run on.

One protocol (``Geometry``, ``Ops``), four geometries: ``SlotGeometry`` (the cachemir decode packing, one
sequence per ciphertext), ``FilledGeometry`` (the dense token layout, ``t = N / d`` conversations; its linears
``diagonal`` / ``pcmm-trunk`` / ``pcmm``), ``PairGeometry`` (the filled layout with a conversation on each axis
of a slot, ``2t`` per ciphertext; split or lane-wise sections) and ``PlainGeometry`` (numpy rows, the mirror,
which rounds the matrices PCMM would). The protocol's methods: ``lane`` / ``ffn`` (constants), ``weight``,
``linear`` / ``up`` / ``down``, ``masked_scale``, ``sum_all``, ``hint``, ``section``, ``lane_mix_kernel`` /
``lane_mix``, ``split_gain``, ``capacity``, ``encode_rows`` / ``decode_rows``. README.md: the layouts.
"""

from __future__ import annotations

import contextlib
import math
from typing import Any, Protocol

import numpy as np

from perseus.impl import pcmm
from perseus.impl.layout import encode_bias_vector, lane_vec, linear_rot_steps, stride_mask
from perseus.impl.linear import EncodedLinear, linear, linear_multi
from perseus.impl.pcmm import PcmmLinear

from . import filling, pairs
from .lanemix import LaneMixKernel

ROUTES = ("split", "lane")
LINEARS = ("diagonal", "pcmm-trunk", "pcmm")
PCMM_WBITS = (16, 24)
#: the matrices pcmm-trunk sends through PCMM (the d -> d and d_ffn -> d ones)
TRUNK = ("win", "wout", "wv")
KINDS = ("win", "wout", "wk", "wr", "wv")


class Ops(Protocol):
    """The leaf operations the kernels (block.py, norm.py) take: ``perseus.impl.ops.FheOps`` on
    a session or the fake, ``NumpyOps`` on arrays, the lane-wise ops of a split pair."""

    def add(self, a: Any, b: Any) -> Any: ...
    def mult(self, a: Any, b: Any) -> Any: ...
    def square(self, a: Any) -> Any: ...
    def mult_pt(self, a: Any, v: Any, key: str | None = None) -> Any: ...
    def add_pt(self, a: Any, v: Any, key: str | None = None) -> Any: ...
    def bootstrap(self, a: Any) -> Any: ...


class Geometry(Protocol):
    """What the kernels ask of a layout (the list in the module docstring, as the type checker
    sees it); ``SlotGeometry``, ``FilledGeometry``, ``PairGeometry`` and ``PlainGeometry``
    implement it, the model's client side (``capacity``, ``encode_rows``, ``decode_rows``,
    ``rotation_steps``) sits beside it on the first three."""

    split_gain: float

    def lane(self, v: Any) -> Any: ...
    def ffn(self, v: Any) -> Any: ...
    def weight(self, w: Any, name: Any, kind: str) -> Any: ...
    def linear(self, x: Any, w: Any) -> Any: ...
    def up(self, x: Any, ws: Any) -> Any: ...
    def down(self, xs: Any, w: Any) -> Any: ...
    def masked_scale(self, x: Any, scale: Any, site: Any) -> Any: ...
    def sum_all(self, x: Any) -> Any: ...
    def hint(self, x: Any, products: int = 1) -> Any: ...
    def section(self, fn: Any, xs: Any) -> Any: ...
    def step(self, label: Any) -> Any: ...
    def lane_mix_kernel(self, P: Any, Pc: Any) -> Any: ...
    def lane_mix(self, x: Any, kernel: Any) -> Any: ...


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


def pcmm_quantize(W, wbits: int) -> np.ndarray:
    """What ``PcmmLinear.encode`` keeps of ``W``: one power-of-two scale per matrix from its largest entry
    (``2^floor(log2(top / max|W|))``, ``top`` the balanced digits' reach), every weight rounded to that
    grid (pcmm.py:238-247). The mirror multiplies by this matrix, so the fake-against-mirror error of a
    PCMM linear is the layout's alone and the rounding shows against the closed form."""
    W = np.asarray(W, dtype=np.float64)
    nb = (int(wbits) + 7) // 8
    top = min(2 ** (int(wbits) - 1) - 1, 127 * (256**nb - 1) // 255)
    ws = 2.0 ** math.floor(math.log2(top / np.abs(W).max()))
    return np.rint(W * ws) / ws


def pcmm_guard(core) -> None:
    """PCMM on a device session runs planned, with the ring-switch keys of the client's key bundle
    (crates/pc-client, crates/pc-py/src/session.rs ``Entry::Pcmm``); a core that carries the legacy
    in-process key generation (``core.bts``: perseus.impl.pcmm's ``make_keys`` from the secret key on
    the evaluating side) is refused."""
    if getattr(core, "bts", None) is not None:
        raise RuntimeError(
            "PCMM on this session would generate its ring-switch keys in process from the secret key "
            "(perseus.impl.pcmm.make_keys); run it planned with the client's key bundle instead"
        )


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

    def weight(self, w, name, kind):  # noqa: ARG002 (the geometry protocol)
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
        ``limit - products * unit``: Perseus's ``level_headroom(1)`` before every linear."""
        ops = self.rt.ops
        return ops.bootstrap_hint(x, ops.headroom(products), True)

    def step(self, label):
        """A step scope: labels the ops inside in the captured graph and the timing layer."""
        return self.rt.step(label)

    def section(self, fn, xs):
        return fn(self.rt.ops, self, xs)

    def lane_mix_kernel(self, P, Pc):  # noqa: ARG002 (the geometry protocol)
        raise ValueError("the lane mix needs the filled layout with pairs")

    def lane_mix(self, x, kernel):  # noqa: ARG002 (the geometry protocol)
        raise ValueError("the lane mix needs the filled layout with pairs")

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

    def __init__(
        self,
        rt,
        d: int,
        d_ffn: int,
        linear: str = "diagonal",
        diag_hint: int = 1,
        pcmm_wbits: int = 16,
        lane_mix: bool = False,
    ):
        n = rt.dims.N
        if n % d or d_ffn % d:
            raise ValueError(f"N={n} and d_ffn={d_ffn} must be multiples of d={d}")
        if linear not in LINEARS:
            raise ValueError(f"linear must be one of {LINEARS}, got {linear!r}")
        if diag_hint not in (1, 2):
            raise ValueError(
                f"diag_hint is 1 (the reference circuit) or 2 (upstream's diag_apply rule), got {diag_hint!r}"
            )
        if pcmm_wbits not in PCMM_WBITS:
            raise ValueError(f"pcmm_wbits must be one of {PCMM_WBITS}, got {pcmm_wbits!r}")
        self.rt = rt
        self.N, self.d, self.d_ffn = n, d, d_ffn
        self.t = n // d
        self.n_tiles = d_ffn // d
        self.linear_mode, self.diag_hint, self.pcmm_wbits = linear, int(diag_hint), int(pcmm_wbits)
        # the lane-mix term's rotation keys are part of the band from the start (lanemix.py)
        self.lane_mix_on = bool(lane_mix)
        # PCMM bookkeeping for the driver's report: calls, inputs hinted, outputs (external refreshes)
        self.pcmm_stats = {"calls": 0, "inputs": 0, "outputs": 0}
        if self.uses_pcmm:
            pcmm_guard(rt.ops.core)
            if d % pcmm.Shape(self.t).nch:
                raise ValueError(
                    f"PCMM needs d a multiple of its {pcmm.Shape(self.t).nch} channels at t = {self.t}, got d = {d}"
                )

    @property
    def ops(self):
        return self.rt.ops

    @property
    def capacity(self) -> int:
        return self.t

    @property
    def uses_pcmm(self) -> bool:
        return self.linear_mode != "diagonal"

    def is_pcmm(self, kind: str) -> bool:
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        return self.linear_mode == "pcmm" or (self.linear_mode == "pcmm-trunk" and kind in TRUNK)

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

    def weight(self, w, name, kind):
        if not name:
            raise ValueError(
                "a linear needs its name (b<b>.<kind>): a PCMM finish is looked up by it on the device"
            )
        if self.is_pcmm(kind):
            return PcmmLinear.encode(
                np.asarray(w, dtype=np.float64), self.t, 1.0, name=name, wbits=self.pcmm_wbits
            )
        return filling.encode_diag(w, self.N, self.t, name)

    def _pcmm(self, xs, w):
        self.pcmm_stats["calls"] += 1
        self.pcmm_stats["inputs"] += w.n_in
        self.pcmm_stats["outputs"] += w.n_out
        return pcmm.run(self.rt, list(xs), w)

    def linear(self, x, w):
        if isinstance(w, PcmmLinear):
            return self._pcmm([x], w)[0]
        return filling.apply_diag(self.rt, [x], [w], self.diag_hint)[0][0]

    def up(self, x, ws):
        if any(isinstance(w, PcmmLinear) for w in ws):
            if not all(isinstance(w, PcmmLinear) for w in ws):
                raise ValueError("the up-projections of one input share a linear form")
            return [self._pcmm([x], w) for w in ws]
        return filling.apply_diag(self.rt, [x], list(ws), self.diag_hint)

    def down(self, xs, w):
        if isinstance(w, PcmmLinear):
            return self._pcmm(xs, w)[0]
        return filling.apply_diag(self.rt, list(xs), [w], self.diag_hint)[0][0]

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

    def lane_mix_kernel(self, P, Pc):
        if not self.lane_mix_on:
            raise ValueError("a lane-mix kernel on a geometry built without the axis (its rotation keys)")
        return LaneMixKernel(P, Pc, self.N, self.t)

    def lane_mix(self, x, kernel):
        return kernel.apply(self.ops, x)

    def rotation_steps(self):
        steps = filling.rotation_steps(self.N, self.d)
        if self.uses_pcmm:
            steps |= set(pcmm.rot_steps())
        if self.lane_mix_on:
            steps |= LaneMixKernel.rotation_steps(self.N)
        return steps

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

    def weight(self, w, name, kind):
        return self.base.weight(w, name, kind)

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

    def lane_mix_kernel(self, P, Pc):
        return self.base.lane_mix_kernel(P, Pc)

    def lane_mix(self, x, kernel):
        return self.base.lane_mix(x, kernel)

    def rotation_steps(self):
        return self.base.rotation_steps()

    def split(self, xs):
        """The real halves ``(2a, 2b)`` of every pair in ``xs`` (the split route's conjugations)."""
        return [pairs.split(self.ops, x) for x in xs]

    def section(self, fn, xs):
        """The kernel on both axes of every pair. ``split``: on the real halves ``2a`` and
        ``2b`` (``split_gain`` tells the caller to have halved the inputs), repacked as
        ``a' + i b'``; ``lane``: once on the pairs, with lane-wise products."""
        if self._lane_ops is not None:
            return fn(self._lane_ops, self.base, xs)
        ops = self.base.ops
        halves = self.split(xs)
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
        return pairs.repack(self.ops, self.base.encode_rows(rows[:t]), self.base.encode_rows(rows[t:]))

    def decode_rows(self, ct, n: int):
        if not 1 <= n <= self.capacity:
            raise ValueError(f"{n} conversations for a capacity of {self.capacity}")
        return filling.unpack_pairs(self.ops.decrypt_slots_complex(ct), self.base.d, n)


class PlainGeometry:
    """``linear`` / ``pcmm_wbits``: the mirror rounds the matrices PCMM would (``pcmm_quantize``)."""

    split_gain = 1.0

    def __init__(self, ops, linear: str = "diagonal", pcmm_wbits: int = 16):
        if linear not in LINEARS:
            raise ValueError(f"linear must be one of {LINEARS}, got {linear!r}")
        self.ops = ops
        self.linear_mode, self.pcmm_wbits = linear, int(pcmm_wbits)

    @staticmethod
    def lane(v):
        return np.asarray(v, dtype=np.float64)

    expanded = lane

    def ffn(self, v):
        return [self.lane(v)]

    def weight(self, w, name, kind):  # noqa: ARG002 (the geometry protocol)
        if self.linear_mode == "pcmm" or (self.linear_mode == "pcmm-trunk" and kind in TRUNK):
            return pcmm_quantize(w, self.pcmm_wbits)
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

    @staticmethod
    def lane_mix_kernel(P, Pc):
        """The mirror keeps the kernel at the shape of the slots' (t = 32 lanes, d channels)."""
        P = np.asarray(P)
        return LaneMixKernel(P, Pc, P.shape[0] * P.shape[1], P.shape[0])

    @staticmethod
    def lane_mix(x, kernel):
        return kernel.rows(x)
