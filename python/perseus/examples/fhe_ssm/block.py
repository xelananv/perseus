# SPDX-License-Identifier: MIT
"""One fhe-ssm block on the ops protocol (the circuit of gpu_real_model_x.cu's layer loop).

Polynomials run at their minimal depth (the cubic gate as ``z (p0 + p1 z + p2 z^2)``, the cubic activation as
``q0 + q1 k + q2 k^2 + (q3 k) k^2``); every linear is preceded by a hint (refresh unless its one product
fits). The nonlinear parts run as the geometry's sections (on a real layout, on a split pair's two halves,
or lane-wise); on split pairs a section's inputs carry half their value (``weights.encode_block`` pre-halves
the operands that make them). ``pack_stats`` (``norm.rmsnorm_packed``) and the lane-mix term (a kernel on
the block's weights) are the two axes that live here.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

from . import pairs
from .geometry import PairGeometry
from .norm import rmsnorm, rmsnorm_packed

if TYPE_CHECKING:
    from .geometry import Geometry, Ops

PACK_STATS = ("off", "halves")


@dataclasses.dataclass
class BlockState:
    """What a block carries from one token to the next: the scan state and the previous
    token's normalized input (both at the canonical level after the carry)."""

    s: Any
    n_prev: Any


def norm(geo: Geometry, h, site, tag: str, form: str, pack_stats: str = "off"):
    """RMSNorm of ``h`` as a section, the split's factor 2 absorbed by the masks. With
    ``pack_stats="halves"`` on a split pair, both halves' statistics share one Newton chain."""
    if pack_stats not in PACK_STATS:
        raise ValueError(f"pack_stats must be one of {PACK_STATS}, got {pack_stats!r}")
    gain = geo.split_gain
    if pack_stats == "halves" and isinstance(geo, PairGeometry) and geo.route == "split":
        ops, base = geo.ops, geo.base
        ((a, b),) = geo.split([h])
        out = rmsnorm_packed(ops, base, [a, b], site, tag, form, gain=gain)
        return pairs.repack(ops, out[0], out[1])

    def kernel(ops: Ops, g: Geometry, xs):
        return [rmsnorm(ops, g, xs[0], site, tag, form, gain=gain)]

    return geo.section(kernel, [h])[0]


def tm_gate(ops: Ops, z, w):
    """``z (p0 + p1 z + p2 z^2)``."""
    poly = ops.add(
        ops.mult_pt(z, w.p1, key=w.key("p1")),
        ops.mult_pt(ops.square(z), w.p2, key=w.key("p2")),
    )
    return ops.mult(z, ops.add_pt(poly, w.p0, key=w.key("p0")))


def cm_gate(ops: Ops, k, r, w, j: int = 0):
    """``(q0 + q1 k + q2 k^2 + q3 k^3) (r0 + r1 r + r2 r^2)`` on d_ffn block ``j``."""

    def key(name):
        return w.key(name if len(w.q0) == 1 else f"{name}.{j}")

    k2 = ops.square(k)
    act = ops.add(ops.mult_pt(k, w.q1[j], key=key("q1")), ops.mult_pt(k2, w.q2[j], key=key("q2")))
    act = ops.add(act, ops.mult(ops.mult_pt(k, w.q3[j], key=key("q3")), k2))
    act = ops.add_pt(act, w.q0[j], key=key("q0"))
    gate = ops.add(
        ops.mult_pt(r, w.r1[j], key=key("r1")),
        ops.mult_pt(ops.square(r), w.r2[j], key=key("r2")),
    )
    gate = ops.add_pt(gate, w.r0[j], key=key("r0"))
    return ops.mult(act, gate)


def time_mix(ops: Ops, geo: Geometry, h, w, st: BlockState, form: str, pack_stats: str = "off"):
    """``h + W_out g``; updates ``st`` in place (``s``, ``n_prev`` not yet carried)."""
    with geo.step("tm.norm"):
        n = norm(geo, h, w.norm_tm, w.key("tm.norm"), form, pack_stats)
    with geo.step("tm.in"):
        um = ops.add(
            ops.mult_pt(n, w.mix, key=w.key("mix")),
            ops.mult_pt(st.n_prev, w.imix, key=w.key("imix")),
        )
        x = geo.linear(geo.hint(um), w.win)
    with geo.step("tm.gate"):
        s = ops.add(ops.mult_pt(st.s, w.a, key=w.key("a")), ops.mult_pt(x, w.b, key=w.key("b")))
        if w.lanemix is not None:
            with geo.step("lanemix"):
                s = ops.add(s, geo.lane_mix(x, w.lanemix))
        z = ops.add(ops.mult_pt(s, w.c, key=w.key("c")), ops.mult_pt(x, w.dd, key=w.key("dd")))
        g = geo.section(lambda o, _g, zs: [tm_gate(o, zs[0], w)], [z])[0]
    st.s, st.n_prev = s, n
    with geo.step("tm.out"):
        return ops.add(h, geo.linear(geo.hint(g), w.wout))


def channel_mix(ops: Ops, geo: Geometry, h, w, form: str, pack_stats: str = "off"):
    """``h + W_v (act (.) gate)``, block by block over d_ffn."""
    with geo.step("cm.norm"):
        n = norm(geo, h, w.norm_cm, w.key("cm.norm"), form, pack_stats)
    with geo.step("cm.kr"):
        ks, rs = geo.up(geo.hint(n), [w.wk, w.wr])
    with geo.step("cm.gate"):
        n_tiles = len(ks)

        def kernel(o, _g, xs):
            return [cm_gate(o, xs[j], xs[n_tiles + j], w, j) for j in range(n_tiles)]

        ags = geo.section(kernel, [*ks, *rs])
    with geo.step("cm.v"):
        return ops.add(h, geo.down([geo.hint(a) for a in ags], w.wv))


def ssm_block(ops: Ops, geo: Geometry, h, w, st: BlockState, form: str = "perseus", pack_stats: str = "off"):
    return channel_mix(ops, geo, time_mix(ops, geo, h, w, st, form, pack_stats), w, form, pack_stats)


def carry(ops: Ops, geo: Geometry, st: BlockState):
    """The canonical carry: both carried values back to the bootstrap landing level, so the
    next token enters the identical circuit."""
    with geo.step("carry"):
        ops.bootstrap(st.s)
        ops.bootstrap(st.n_prev)
    return st
