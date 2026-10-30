# SPDX-License-Identifier: MIT
"""One fhe-ssm block on the ops protocol (the circuit of gpu_real_model_x.cu's layer loop).

Polynomials run at their minimal depth (the cubic gate as ``z (p0 + p1 z + p2 z^2)``, the cubic activation as
``q0 + q1 k + q2 k^2 + (q3 k) k^2``); every linear is preceded by a hint (refresh unless its one product
fits). The nonlinear parts run as the geometry's sections (on a real layout, on a split pair's two halves,
or lane-wise); on split pairs a section's inputs carry half their value (``weights.encode_block`` pre-halves
the operands that make them). ``pack_stats`` (``norm.rmsnorm_packed``) and the lane-mix term (a kernel on
the block's weights) are two axes that live here; the pass kernels (``pk``: one block's ``passmasks.SlotPass`` on the
slots or ``passmasks.RowsPass`` on the mirror's rows) are the third: the scan, the token shift and the carry prep of
a pass in any lane configuration (``pass_scan``, ``pass_shift``, ``carry_prep``).
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


def pass_shift(ops: Ops, geo: Geometry, n, cn, pk):
    """The token shift of a pass: ``um[q] = mix n[q] + (1 - mix) n_prev[q]`` with ``n_prev[q] = n[q - 1]`` inside a
    sequence and the carried ``C_n[q]`` (or 0, fresh) elsewhere, as six products (passmasks.py: ``mix + U_0`` one
    mask), one rotation by -1, three conjugations; depth 1 (``um`` at today's level). Each product pair (a mask and
    its conjugate half) is summed before it joins the total, as the lane kernel sums its pairs: on a lane whose two
    axes read the register differently the pair is ``(1 - mix)/2 (C + conj C)``, whose real part doubles one rounded
    product exactly, so the tick configuration rounds as today's ``mix (.) n + (1 - mix) (.) n_prev``."""
    with geo.step("tm.shift"):
        nv, cv = pk.enter(n), pk.enter(cn)
        r = pk.rot1(ops, nv)
        nc, rc, cc = ops.conjugate(nv), ops.conjugate(r), ops.conjugate(cv)
        um = ops.add(pk.mult(ops, nv, "sh/mixU0"), pk.mult(ops, nc, "sh/U0c"))
        um = ops.add(um, ops.add(pk.mult(ops, r, "sh/U1"), pk.mult(ops, rc, "sh/U1c")))
        um = ops.add(um, ops.add(pk.mult(ops, cv, "sh/V"), pk.mult(ops, cc, "sh/Vc")))
        return pk.leave(um)


def pass_scan(ops: Ops, geo: Geometry, x, c, pk):
    """K1, the scan of a pass: ``S[q] = sum_{q'} c(q, q') x[q'] + kappa(q) C[q]`` (passmasks.py), the lane term over the
    ``t`` offsets as a causal baby-step giant-step on ``x`` (``2t`` products, ``2(NB - 1) + NG - 1`` rotations, one
    conjugation) plus the state term ``P^c (.) C + P'^c (.) conj(C)`` (two products, one conjugation); depth 1: the
    products of ``x`` land where today's ``b (.) x`` lands, those of ``C`` where today's ``a (.) C`` lands."""
    with geo.step("tm.scan"):
        xv, cv = pk.enter(x), pk.enter(c)
        lanes = pk.lanes(ops, xv, "k1")
        state = ops.add(pk.mult(ops, cv, "k1/S/P"), pk.mult(ops, ops.conjugate(cv), "k1/S/Pc"))
        return pk.leave(ops.add(lanes, state))


def carry_prep(ops: Ops, geo: Geometry, st: BlockState, pk):
    """K2, before the canonical carry: every lane of a sequence's home gets the segment's final state (its last
    position's ``S`` and ``n``, on the imaginary axis of the home's last lane) on both axes, a conversation's
    lane-slots keep their own. Every 1/2 rides a mask (a scalar product costs a level)::

        Y2 = mult_i(conj S - S) + (n - conj n)        = 2 Im S + 2i Im n       (level-free)
        Z  = sum_s R_s (.) rot(Y2, +s)                R_s = 1/4 at the end lane (anti-causal, no conj half)
        W  = Z + conj Z = S_end;  V = mult_i(conj Z - Z) = n_end                (real-valued, level-free)
        C_next = P_o (.) S + W + mult_i(W);  C_n_next = P_o (.) n + V + mult_i(V)

    ``t + 2`` products, ``NB - 1 + NG - 1`` rotations, three conjugations; both carries one product above
    ``max(S, n)``. Updates ``st`` in place."""
    with geo.step("carry.prep"):
        s, n = pk.enter(st.s), pk.enter(st.n_prev)
        pk.note(ops, "S", s)
        pk.note(ops, "n", n)
        sc, nc = ops.conjugate(s), ops.conjugate(n)
        y2 = ops.add(ops.mult_i(ops.sub(sc, s)), ops.sub(n, nc))
        z = pk.lanes(ops, y2, "k2")
        pk.note(ops, "Z", z)
        zc = ops.conjugate(z)
        w = ops.add(z, zc)
        v = ops.mult_i(ops.sub(zc, z))
        c_next = ops.add(ops.add(pk.mult(ops, s, "k2/Po"), w), ops.mult_i(w))
        cn_next = ops.add(ops.add(pk.mult(ops, n, "k2/Po"), v), ops.mult_i(v))
        pk.note(ops, "C", c_next)
        pk.note(ops, "C_n", cn_next)
        st.s, st.n_prev = pk.leave(c_next), pk.leave(cn_next)
    return st


def time_mix(ops: Ops, geo: Geometry, h, w, st: BlockState, form: str, pack_stats: str = "off", pk=None):
    """``h + W_out g``; updates ``st`` in place (``s``, ``n_prev`` not yet carried). ``pk``: the pass kernels in place
    of the token shift and the scan (``pass_shift``, ``pass_scan``)."""
    with geo.step("tm.norm"):
        n = norm(geo, h, w.norm_tm, w.site("tm.norm"), form, pack_stats)
    with geo.step("tm.in"):
        if pk is None:
            um = ops.add(
                ops.mult_pt(n, w.mix, key=w.key("mix")),
                ops.mult_pt(st.n_prev, w.imix, key=w.key("imix")),
            )
        else:
            um = pass_shift(ops, geo, n, st.n_prev, pk)
        x = geo.linear(geo.hint(um), w.win)
    with geo.step("tm.gate"):
        if pk is None:
            s = ops.add(ops.mult_pt(st.s, w.a, key=w.key("a")), ops.mult_pt(x, w.b, key=w.key("b")))
        else:
            s = pass_scan(ops, geo, x, st.s, pk)
        if w.lanemix is not None:
            if pk is not None:
                raise ValueError(
                    "the lane-mix term adds a cost-only term to the scan state: refused with the pass kernels"
                )
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
        n = norm(geo, h, w.norm_cm, w.site("cm.norm"), form, pack_stats)
    with geo.step("cm.kr"):
        ks, rs = geo.up(geo.hint(n), [w.wk, w.wr])
    with geo.step("cm.gate"):
        n_tiles = len(ks)

        def kernel(o, _g, xs):
            return [cm_gate(o, xs[j], xs[n_tiles + j], w, j) for j in range(n_tiles)]

        ags = geo.section(kernel, [*ks, *rs])
    with geo.step("cm.v"):
        return ops.add(h, geo.down([geo.hint(a) for a in ags], w.wv))


def ssm_block(
    ops: Ops, geo: Geometry, h, w, st: BlockState, form: str = "perseus", pack_stats: str = "off", pk=None
):
    """One block; with the pass kernels (``pk``) the block ends with the carry prep (``carry_prep``)."""
    h = channel_mix(ops, geo, time_mix(ops, geo, h, w, st, form, pack_stats, pk), w, form, pack_stats)
    if pk is not None:
        carry_prep(ops, geo, st, pk)
    return h


def carry(ops: Ops, geo: Geometry, st: BlockState):
    """The canonical carry: both carried values back to the bootstrap landing level, so the
    next token enters the identical circuit."""
    with geo.step("carry"):
        ops.bootstrap(st.s)
        ops.bootstrap(st.n_prev)
    return st
