"""One fhe-ssm block on the ops protocol (the circuit of gpu_real_model_x.cu's layer loop).

Polynomials are evaluated at their minimal depth: the cubic gate as ``z (p0 + p1 z + p2 z^2)``
(three levels above ``z``), the cubic activation as ``q0 + q1 k + q2 k^2 + (q3 k) k^2`` (two
levels above ``k``). Every linear is preceded by the hint the GPT-2 port uses before its
projections (``geometry.SlotGeometry.hint``): refresh unless the linear's one product fits.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from .norm import rmsnorm


@dataclasses.dataclass
class BlockState:
    """What a block carries from one token to the next: the scan state and the previous
    token's normalized input (both at the canonical level after the carry)."""

    s: Any
    n_prev: Any


def time_mix(ops, geo, h, w, st: BlockState, form: str):
    """``h + W_out g``; updates ``st`` in place (``s``, ``n_prev`` not yet carried)."""
    with geo.step("tm.norm"):
        n = rmsnorm(ops, geo, h, w.norm_tm, w.key("tm.norm"), form)
    with geo.step("tm.in"):
        um = ops.add(
            ops.mult_pt(n, w.mix, key=w.key("mix")),
            ops.mult_pt(st.n_prev, w.imix, key=w.key("imix")),
        )
        x = geo.linear(geo.hint(um), w.win)
    with geo.step("tm.gate"):
        s = ops.add(ops.mult_pt(st.s, w.a, key=w.key("a")), ops.mult_pt(x, w.b, key=w.key("b")))
        z = ops.add(ops.mult_pt(s, w.c, key=w.key("c")), ops.mult_pt(x, w.dd, key=w.key("dd")))
        poly = ops.add(
            ops.mult_pt(z, w.p1, key=w.key("p1")),
            ops.mult_pt(ops.square(z), w.p2, key=w.key("p2")),
        )
        g = ops.mult(z, ops.add_pt(poly, w.p0, key=w.key("p0")))
    st.s, st.n_prev = s, n
    with geo.step("tm.out"):
        return ops.add(h, geo.linear(geo.hint(g), w.wout))


def channel_mix(ops, geo, h, w, form: str):
    """``h + W_v (act (.) gate)``."""
    with geo.step("cm.norm"):
        n = rmsnorm(ops, geo, h, w.norm_cm, w.key("cm.norm"), form)
    with geo.step("cm.kr"):
        k, r = geo.linear_multi(geo.hint(n), [w.wk, w.wr])
    with geo.step("cm.gate"):
        k2 = ops.square(k)
        act = ops.add(ops.mult_pt(k, w.q1, key=w.key("q1")), ops.mult_pt(k2, w.q2, key=w.key("q2")))
        act = ops.add(act, ops.mult(ops.mult_pt(k, w.q3, key=w.key("q3")), k2))
        act = ops.add_pt(act, w.q0, key=w.key("q0"))
        gate = ops.add(
            ops.mult_pt(r, w.r1, key=w.key("r1")),
            ops.mult_pt(ops.square(r), w.r2, key=w.key("r2")),
        )
        gate = ops.add_pt(gate, w.r0, key=w.key("r0"))
        ag = ops.mult(act, gate)
    with geo.step("cm.v"):
        return ops.add(h, geo.linear(geo.hint(ag), w.wv))


def ssm_block(ops, geo, h, w, st: BlockState, form: str = "perseus"):
    return channel_mix(ops, geo, time_mix(ops, geo, h, w, st, form), w, form)


def carry(ops, geo, st: BlockState):
    """The canonical carry: both carried values back to the bootstrap landing level, so the
    next token enters the identical circuit."""
    with geo.step("carry"):
        ops.bootstrap(st.s)
        ops.bootstrap(st.n_prev)
    return st
