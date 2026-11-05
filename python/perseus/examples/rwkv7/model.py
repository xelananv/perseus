# SPDX-License-Identifier: MIT
"""The RWKV-7 block on ciphertexts (``Circuit``), its weights folded for PCMM (``LayerW``), and the model as an
``ImplModel`` (``Rwkv7Fhe``: one stage per block plus a tail, the carried state named for the planner).

The stage interface is ONE ciphertext ``z = h + i v_first``: the residual stream ``h`` on the real axis and layer 0's
value ``v_first`` (which every later block's value-residual reads) on the imaginary axis, both in natural form
(layout.py). A block reads ``2h = z + conj z`` (level-free; the factor 2 is folded into the LayerNorm constants) and
``2 v_first = i (conj z - z)`` (the factor 2 folded into the v-gate polynomial, halved), and adds its real outputs to
``z``, so ``v_first`` rides along without a register. Block 0 writes it: ``z += i v``.

Per block (the harness's names; ``x_prev`` are the carried LayerNorm outputs, the token shift):

    x   = LN1(h)                                             channel sums + Newton (site ln1.rsqrt)
    [r, k, v, a', w', g', v'] = PCMM([x, xa_prev])           the token-shift mixes folded into the matrices
    w   = decay(w0 + w2 tanh(w'))                            sites att.w_tanh, att.decay
    a   = sigmoid(a0 + a');  g = g2 sigmoid(g')              sites att.a_sigmoid, att.g_sigmoid
    v  += (v_first - v) sigmoid(v0 + v')        (b >= 1)     site att.v_sigmoid
    kk  = k k_k / ||k k_k||_head                             per-head sum + Newton (site att.kk_rsqrt)
    k~  = k (1 + (a - 1) k_a)
    S_m = S_m w + bcast_j(-(S_m kk)) (kk a) + V_m k~         V = transpose(v)   (m = 0, 1)
    y   = untranspose(S_0 r, S_1 r)
    y   = GroupNorm(y) + (sum_head r k~ r_k) v               per-head sums + Newton (site att.gn_rsqrt)
    h  += Wo (y g)
    x2  = LN2(h);  h += Wv relu2(PCMM([x2, xf_prev]))        sites ln2.rsqrt, ffn.relu2

Every site evaluates the coefficient JSON's polynomial with ``impl.poly.eval_chebyshev_bsgs`` (no coefficient trim)
on its interval, without a clamp (a ciphertext cannot be clamped); the inverse square roots run fit.py's Newton form
(a Chebyshev seed on ``u = (x + eps) / scale``, then ``y <- y (1.5 - 0.5 u y^2)`` in the two-level form
``1.5 y + (y (-u/2)) y^2``) with the affine map of ``u`` and ``scale^-1/2`` folded into constants.
"""
from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass

import numpy as np

from perseus.impl.driver import ImplModel
from perseus.impl.poly import eval_chebyshev_bsgs

from .layout import D, HEAD, Layout
from .ref import Params

STATE = tuple(f"s{m}" for m in range(16))   # names of the M = 64 / t state ciphertexts (2 at t = 32, 8 at t = 8)


def load_coeffs(path_or_doc) -> dict:
    doc = json.load(open(path_or_doc)) if isinstance(path_or_doc, str) else path_or_doc
    return {n: s for n, s in doc["sites"].items() if not s.get("exact")}


def _pad(W, rows, cols):
    out = np.zeros((rows, cols))
    out[: W.shape[0], : W.shape[1]] = W
    return out


@dataclass
class Lin:
    """A linear ``y = x W`` over whole 1,024-channel blocks: ``W`` (n_in 1024, n_out 1024) float64, its name, and the
    PCMM encoding (``pcmm``, built on demand: ``impl.pcmm.PcmmLinear``)."""
    name: str
    W: np.ndarray
    n_in: int
    n_out: int
    pcmm: object = None
    Wq: np.ndarray | None = None      # the PCMM-rounded matrix (the mirror multiplies by it)

    def __post_init__(self):
        # float32: the checkpoint is bf16, and a <= 24-bit rounded matrix is exact in float32
        self.W = np.asarray(self.W, dtype=np.float32)


def _blocks(n: int, D: int = D) -> int:
    return -(-int(n) // D)


#: how a no-nonlinearity LoRA pair (a1 a2, v1 v2) enters the circuit: "fold" = one C x C matrix (one PCMM, S outputs;
#: S^2 1,024-blocks of planes per input), "split" = two PCMMs through the rank (S + ceil(r/1024) outputs, 2 S
#: ceil(r/1024) blocks: fewer planes and GEMM, one more PCMM refresh and its levels); "auto" folds at one shard
LORA_MODES = ("auto", "fold", "split")


class LayerW:
    """Block ``b``'s operands: the PCMM linears and the natural-form constants (one per channel shard: lists)."""

    def __init__(self, P: Params, b: int, L: Layout, coeffs: dict, *, fuse_l1=False, lora="auto"):
        p, C = P.layers[b].p, P.C
        Sh, Cp, D = L.S, L.Cp, L.D
        Fp = L.n_ffn * D                                  # d_ffn zero-padded to whole ciphertexts
        self.b = b
        if lora not in LORA_MODES:
            raise ValueError(f"lora {lora!r}: {LORA_MODES}")
        self.lora = ("fold" if Sh == 1 else "split") if lora == "auto" else lora
        one = np.ones(C)
        mixes = {k: p[k] for k in ("x_r", "x_w", "x_k", "x_v", "x_a", "x_g")}
        has_v = b > 0
        if self.lora == "fold":
            ma, mv = [("a", "x_a", p["a1"] @ p["a2"])], [("vg", "x_v", p["v1"] @ p["v2"])] if has_v else []
        else:
            ma, mv = [("a1", "x_a", p["a1"])], [("v1", "x_v", p["v1"])] if has_v else []
        mats = [("r", "x_r", p["receptance_w"].T), ("k", "x_k", p["key_w"].T), ("v", "x_v", p["value_w"].T),
                *ma, ("w1", "x_w", p["w1"]), ("g1", "x_g", p["g1"])]
        mats += mv
        self.l1_kinds = [m[0] for m in mats]
        blocks, nouts = [], []
        for _kind, mix, Wt in mats:
            mu = mixes[mix]
            no = _blocks(Wt.shape[1], D)
            top = _pad((one - mu)[:, None] * Wt, Cp, no * D)        # x       (the current LN output)
            bot = _pad(mu[:, None] * Wt, Cp, no * D)                # x_prev  (the carried one)
            blocks.append(np.vstack([top, bot]))
            nouts.append(no)
        self.l1_nout = dict(zip(self.l1_kinds, nouts))
        if fuse_l1:
            self.l1 = [Lin(f"b{b}.l1", np.hstack(blocks), 2 * Sh, sum(nouts))]
        else:
            self.l1 = [Lin(f"b{b}.l1{kind}", blk, 2 * Sh, no) for kind, blk, no in zip(self.l1_kinds, blocks, nouts)]
        self.lora2 = []
        if self.lora == "split":
            ra = _blocks(p["a2"].shape[0], D)
            self.a2 = Lin(f"b{b}.a2", _pad(p["a2"], ra * D, Cp), ra, Sh)
            self.lora2.append(self.a2)
            if b > 0:
                rv = _blocks(p["v2"].shape[0], D)
                self.v2 = Lin(f"b{b}.v2", _pad(p["v2"], rv * D, Cp), rv, Sh)
                self.lora2.append(self.v2)
        rw, rg = _blocks(p["w2"].shape[0], D), _blocks(p["g2"].shape[0], D)
        self.w2 = Lin(f"b{b}.w2", _pad(p["w2"], rw * D, Cp), rw, Sh)
        self.g2 = Lin(f"b{b}.g2", _pad(p["g2"], rg * D, Cp), rg, Sh)
        self.wo = Lin(f"b{b}.out", _pad(p["output_w"].T, Cp, Cp), Sh, Sh)
        fx = p["ffn_x_k"]
        Wk = p["ffn_key_w"].T                                      # (C, d_ffn)
        self.fk = Lin(f"b{b}.fk", np.vstack([_pad((one - fx)[:, None] * Wk, Cp, Fp),
                                             _pad(fx[:, None] * Wk, Cp, Fp)]), 2 * Sh, L.n_ffn)
        self.fv = Lin(f"b{b}.fv", _pad(p["ffn_value_w"].T, Fp, Cp), L.n_ffn, Sh)
        lane = L.lanes
        self.w0, self.a0 = lane(p["w0"]), lane(p["a0"])
        self.v0 = lane(p["v0"]) if b > 0 else None
        self.ln = {}
        for which in ("ln1", "ln2"):
            s = coeffs[f"blocks.{b}.{which}.rsqrt"]
            # x = 2(h - mu) y ; LN = (h - mu) / sqrt(var + eps) gamma + beta = x y (gamma scale^-1/2 / 2) + beta
            self.ln[which] = (lane(p[f"{which}_w"] * 0.5 * s["scale"] ** -0.5), lane(p[f"{which}_b"]))
        skk = coeffs[f"blocks.{b}.att.kk_rsqrt"]
        self.kk_c = skk["scale"] ** -0.5                          # kk carries scale^-1/2 (folded into k_k)
        self.k_k = lane(p["k_k"] * self.kk_c)
        self.kk_pad = L.pad_heads_sh(0.5 * (skk["lo"] + skk["hi"]) * self.kk_c ** 2)
        self.k_a, self.one_m_ka = lane(p["k_a"]), lane(1.0 - p["k_a"])
        sgn = coeffs[f"blocks.{b}.att.gn_rsqrt"]
        self.gn_pad = L.pad_heads_sh(0.5 * (sgn["lo"] + sgn["hi"]))
        self.gn_w = lane(p["ln_x_w"] * sgn["scale"] ** -0.5)
        self.gn_b = lane(p["ln_x_b"])
        self.r_k = lane(p["r_k"].reshape(-1))

    def linears(self) -> list[Lin]:
        return [*self.l1, *self.lora2, self.w2, self.g2, self.wo, self.fk, self.fv]


#: the raise drops the runtime builds dense-route variants for (make_plan.sh FIDESLIB_BTS_RAISE_DROPS=1,3,5)
CARRY_DROPS = (0, 1, 3, 5)


def state_names(Sh: int, M: int) -> list[str]:
    """One block's carried ciphertexts, in ``State.cts()`` order: shard 0 keeps the single-shard names (``s0 s1 xa
    xf``), shard j > 0 adds ``h<j>`` (``s0h1``, ``xah1``)."""
    sfx = [""] + [f"h{j}" for j in range(1, Sh)]
    return ([f"{STATE[m]}{x}" for x in sfx for m in range(M)] + [f"xa{x}" for x in sfx] + [f"xf{x}" for x in sfx])


class State:
    """One block's carried state: ``S`` the S M state ciphertexts (shard-major), ``xa`` / ``xf`` the token-shift
    carries (one per shard)."""

    def __init__(self, S, xa, xf):
        self.S = list(S)
        self.xa = list(xa) if isinstance(xa, (list, tuple)) else xa
        self.xf = list(xf) if isinstance(xf, (list, tuple)) else xf

    def cts(self):
        return [*self.S, *self.xa, *self.xf]

    def s_cts(self):
        """The state ciphertexts alone (the carries xa / xf excluded), in ``cts()`` order."""
        return list(self.S)

    def names(self):
        return state_names(len(self.xa), len(self.S) // len(self.xa))


def _sum(ops, xs):
    acc = xs[0]
    for x in xs[1:]:
        acc = ops.add(acc, x)
    return acc


class Circuit:
    """The block on any ops (``linear(xs, lin)`` supplies the matrix product: PCMM on a session or the fake, numpy
    on the mirror). Every width-C vector is a list of ``L.S`` shard ciphertexts (one at C <= 1,024)."""

    def __init__(self, ops, L: Layout, coeffs: dict, linear, step=None):
        self.ops, self.L, self.coeffs, self.linear = ops, L, coeffs, linear
        self.step = step

    def _scope(self, label):
        import contextlib
        return self.step(label) if self.step is not None else contextlib.nullcontext()

    #: a ``SiteGuard`` (the capture sets one): every site's input checked against its interval before it is evaluated
    guard = None

    # ── sites ──
    def cheb(self, name, x, scale=1.0):
        s = self.coeffs[name]
        c = [float(v) * scale for v in s["coeffs"]]
        if self.guard is not None:
            self.guard(name, x, float(s["lo"]), float(s["hi"]))
        return eval_chebyshev_bsgs(self.ops, x, c, float(s["lo"]), float(s["hi"]), eps=0.0)

    def newton(self, name, x, xscale=1.0):
        """y with y scale^-1/2 = 1/sqrt(xscale x + eps) (fit.py's Newton form, on the raw statistic x)."""
        ops, s = self.ops, self.coeffs[name]
        a, b0 = float(xscale) / s["scale"], s["eps"] / s["scale"]
        if s.get("method") == "newton_scaled":
            # per-step constants (tools/rwkv7/convert_newton_scaled.py; poly.scaled_newton_steps): the same constant
            # seed, accuracy range and convergence region as newton_const, ~6.7x of range ratio per step instead of
            # 2.25x. Step 1 is affine in u = a x + b0 (one scalar level); every later step y <- A y - B u y^3 in the
            # d2 product order ((c y) y^2, c = -B u folded onto x: one scalar level, never deeper than y): two levels
            if self.guard is not None:
                self.guard(name, x, float(s["lo"]) / xscale, float(s["hi"]) / xscale)
            (A, B), rest = s["steps"][0], s["steps"][1:]
            y = ops.add(ops.mult(x, -B * a), A - B * b0)
            for A, B in rest:
                c = ops.add(ops.mult(x, -B * a), -B * b0)
                y = ops.add(ops.mult(y, A), ops.mult(ops.mult(c, y), ops.square(y)))
            return y
        if s.get("method") == "newton_const":
            # P6: constant seed 1 (ROBUST_NORMS.md). The guard judges the accuracy range [lo, hi] (x units of the
            # harness statistic, i.e. xscale x); the iteration itself converges on the wider (conv_lo, conv_hi).
            if self.guard is not None:
                self.guard(name, x, float(s["lo"]) / xscale, float(s["hi"]) / xscale)
            # step 1 closed form: y1 = 1.5 - 0.5 u (one scalar product); then inv_sqrt_newton_d2 on u = a x + b0
            y = ops.add(ops.mult(x, -0.5 * a), 1.5 - 0.5 * b0)
            n = int(s["steps"]) - 1
            if n <= 0:
                return y
            from perseus.impl.poly import inv_sqrt_newton_d2
            return inv_sqrt_newton_d2(ops, ops.add(x, b0 / a), y, n, x_scale=a)
        lo, hi = (s["seed_lo"] - b0) / a, (1.0 - b0) / a
        if self.guard is not None:
            self.guard(name, x, lo, hi)
        y = eval_chebyshev_bsgs(ops, x, [float(c) for c in s["seed_coeffs"]], lo, hi, eps=0.0)
        un = ops.add(ops.mult(x, -0.5 * a), -0.5 * b0)          # -u/2
        for _ in range(int(s["steps"])):
            y = ops.add(ops.mult(y, 1.5), ops.mult(ops.mult(y, un), ops.square(y)))
        return y

    def layer_norm(self, h2, name, consts):
        """LayerNorm over the C channels of the shards ``h2`` (each ``2h``): ``channel_sum`` is linear, so the
        shards are added first and summed once (one rotate-and-sum whatever S), and one Newton chain serves all."""
        ops, L = self.ops, self.L
        s1 = L.channel_sum(ops, _sum(ops, h2))
        xc = [ops.sub(h, ops.mult_pt(s1, lv / L.C)) for h, lv in zip(h2, L.lives)]   # 2 (h - mu), zero on padding
        q = L.channel_sum(ops, _sum(ops, [ops.square(x) for x in xc]))              # 4 C var
        y = self.newton(name, q, 1.0 / (4 * L.C))
        g, b = consts
        return [ops.add_pt(ops.mult_pt(ops.mult(x, y), gi), bi) for x, gi, bi in zip(xc, g, b)]

    # ── the two steps a prefill pass replaces (prefill.py) ──
    def token_shift(self, x, carry):
        """(the token-shift partner of ``x``, the carry the next step reads): one token, the carried LN output
        (lists of shards)."""
        return carry, x

    def state_step(self, S, w, kkn, bb, kt, r, V):
        """One token's update of one shard's M state ciphertexts and its readout terms ``S_m r`` (natural-form
        inputs; ``V = transpose(v)``)."""
        ops, L = self.ops, self.L
        newS, Ps = [], []
        for m, Sm in enumerate(S):
            Sa = L.block_allreduce(ops, ops.mult(Sm, kkn), -1.0)     # S a with a = -kk, broadcast over j
            Sn = ops.add(ops.add(ops.mult(Sm, w), ops.mult(Sa, bb)), ops.mult(V[m], kt))
            newS.append(Sn)
            Ps.append(ops.mult(Sn, r))
        return newS, Ps

    def wkv_shard(self, st: State, j, w, kkn, bb, kt, r, v):
        """Shard j's heads: its M state ciphertexts updated in ``st.S``, its readout returned."""
        M = self.L.M
        Sj, Ps = self.state_step(st.S[j * M:(j + 1) * M], w, kkn, bb, kt, r, self.L.transpose(self.ops, v))
        st.S[j * M:(j + 1) * M] = Sj
        return self.L.untranspose(self.ops, Ps)

    def wkv(self, st: State, w, kkn, bb, kt, r, v):
        return [self.wkv_shard(st, j, *xs) for j, xs in enumerate(zip(w, kkn, bb, kt, r, v))]

    # ── the block ──
    def block(self, b, z, st: State, W: LayerW):
        """``z``: the S stage ciphertexts ``h + i v_first`` (a list); returns the next block's."""
        ops, L = self.ops, self.L
        pre = f"blocks.{b}."
        z = list(z)
        with self._scope("tm.norm"):
            x = self.layer_norm([ops.add(zi, ops.conjugate(zi)) for zi in z], pre + "ln1.rsqrt", W.ln["ln1"])
        with self._scope("tm.in"):
            xp, st.xa = self.token_shift(x, st.xa)
            outs = []
            for lin in W.l1:
                outs += self.linear([*x, *xp], lin)
            o, i0 = {}, 0
            for kind in W.l1_kinds:
                o[kind] = outs[i0:i0 + W.l1_nout[kind]]
                i0 += W.l1_nout[kind]
        with self._scope("tm.lora"):
            zw = self.linear([self.cheb(pre + "att.w_tanh", c) for c in o["w1"]], W.w2)
            w = [self.cheb(pre + "att.decay", ops.add_pt(c, w0)) for c, w0 in zip(zw, W.w0)]
            ain = o["a"] if W.lora == "fold" else self.linear(o["a1"], W.a2)
            a = [self.cheb(pre + "att.a_sigmoid", ops.add_pt(c, a0)) for c, a0 in zip(ain, W.a0)]
            g = self.linear([self.cheb(pre + "att.g_sigmoid", c) for c in o["g1"]], W.g2)
            r, k, v = o["r"], o["k"], list(o["v"])
            if b > 0:
                vin = o["vg"] if W.lora == "fold" else self.linear(o["v1"], W.v2)
                for i, zi in enumerate(z):
                    q = ops.mult_i(ops.sub(ops.conjugate(zi), zi))       # 2 v_first
                    sv = self.cheb(pre + "att.v_sigmoid", ops.add_pt(vin[i], W.v0[i]), scale=0.5)
                    v[i] = ops.add(v[i], ops.mult(ops.sub(q, ops.add(v[i], v[i])), sv))
        with self._scope("tm.kk"):
            kkn, kt, bb = [], [], []
            for i in range(L.S):
                kk = ops.mult_pt(k[i], W.k_k[i])                         # k k_k scale^-1/2
                nk = ops.add_pt(L.block_allreduce(ops, ops.square(kk)), W.kk_pad[i])
                kkn.append(ops.mult(kk, self.newton(pre + "att.kk_rsqrt", nk, 1.0 / W.kk_c ** 2)))
                kt.append(ops.mult(k[i], ops.add_pt(ops.mult_pt(a[i], W.k_a[i]), W.one_m_ka[i])))
                bb.append(ops.mult(kkn[i], a[i]))
        with self._scope("tm.state"):
            y = self.wkv(st, w, kkn, bb, kt, r, v)
        with self._scope("tm.gn"):
            yo = []
            for i in range(L.S):
                yc = ops.sub(y[i], L.block_allreduce(ops, y[i], 1.0 / HEAD))
                var = ops.add_pt(L.block_allreduce(ops, ops.square(yc), 1.0 / HEAD), W.gn_pad[i])
                yn = ops.add_pt(ops.mult_pt(ops.mult(yc, self.newton(pre + "att.gn_rsqrt", var)), W.gn_w[i]),
                                W.gn_b[i])
                bonus = ops.mult(L.block_allreduce(ops, ops.mult_pt(ops.mult(r[i], kt[i]), W.r_k[i])), v[i])
                yo.append(ops.mult(ops.add(yn, bonus), g[i]))
        with self._scope("tm.out"):
            dx = self.linear(yo, W.wo)
            for i in range(L.S):
                if b == 0:
                    z[i] = ops.add(z[i], ops.mult_i(v[i]))               # v_first onto the imaginary axis
                z[i] = ops.add(z[i], dx[i])
        with self._scope("cm.norm"):
            x2 = self.layer_norm([ops.add(zi, ops.conjugate(zi)) for zi in z], pre + "ln2.rsqrt", W.ln["ln2"])
        with self._scope("cm.ffn"):
            xp2, st.xf = self.token_shift(x2, st.xf)
            kf = self.linear([*x2, *xp2], W.fk)
            rf = [self.cheb(pre + "ffn.relu2", kj) for kj in kf]
            f = self.linear(rf, W.fv)
            z = [ops.add(zi, fi) for zi, fi in zip(z, f)]
        return z


# ── linears ──

class SiteOutOfInterval(RuntimeError):
    """A site's input left its calibrated interval (or went non-finite) on the fake / capture run."""


class SiteGuard:
    """The capture's fail-fast check (``Circuit.guard``): before each polynomial / Newton site, the input's slots
    (decrypted on the fake / recorder ops; never on a session) must be finite and inside the site's interval
    ``[lo, hi]`` (for a Newton site the seed's interval in raw-statistic units), else ``SiteOutOfInterval`` names the
    site, block, pass, lane (the token of a prefill pass), channel and the value. A ciphertext cannot clamp: a value
    outside makes the Chebyshev / Newton evaluation diverge, the capture carries NaN and the planner refuses the block
    much later. ``tol`` is a slack relative to the interval width (default 0: the interval itself)."""

    def __init__(self, ops, L, chunk=0, tol=0.0):
        self.ops, self.L, self.chunk, self.tol = ops, L, int(chunk), float(tol)
        self.pos = 0
        self.checked = 0

    def __call__(self, name, x, lo, hi):
        v = np.asarray(self.ops.decrypt_slots(x))
        re = v.real.astype(np.float64)
        slack = self.tol * (hi - lo)
        with np.errstate(invalid="ignore"):
            bad = ~np.isfinite(v) | (re < lo - slack) | (re > hi + slack)
        self.checked += 1
        if not bad.any():
            return
        idx = np.flatnonzero(bad)
        i = int(idx[0])
        t = self.L.t
        lanes = sorted({int(j) % t for j in idx})
        where = (f"pass {self.pos}, lane {i % t} (token {self.pos * self.chunk + i % t} of the capture text)"
                 if self.chunk else f"token {self.pos}, lane {i % t}")
        with np.errstate(all="ignore"):
            fin = re[np.isfinite(re)]
            rng = f"[{fin.min():.4g}, {fin.max():.4g}]" if fin.size else "[none finite]"
        raise SiteOutOfInterval(
            f"site {name} (block {name.split('.')[1] if name.startswith('blocks.') else '?'}): {idx.size} slot(s) "
            f"out of [{lo:.4g}, {hi:.4g}] at {where}, channel {i // t}: value {v[i]!r}; lanes hit {lanes[:8]}"
            f"{'...' if len(lanes) > 8 else ''}; input range {rng}, {np.count_nonzero(~np.isfinite(v))} non-finite")


def pcmm_encode(lin: Lin, t: int, wbits: int, shifts: dict | None = None):
    from perseus.impl.pcmm import PcmmLinear
    if lin.pcmm is None:
        sh = int((shifts or {}).get(lin.name, 0))
        lin.pcmm = PcmmLinear.encode(np.asarray(lin.W, dtype=np.float64), t, 1.0, name=lin.name, wbits=wbits,
                                     shift=sh)
    return lin.pcmm


def quantized(lin: Lin, wbits: int) -> np.ndarray:
    if lin.Wq is None:
        from perseus.examples.fhe_ssm.geometry import pcmm_quantize
        lin.Wq = np.asarray(pcmm_quantize(np.asarray(lin.W, dtype=np.float64), wbits), dtype=np.float32)
    return lin.Wq


def numpy_linear(L: Layout, wbits: int | None):
    """The mirror's linear: ``Y_m = sum_c W[c, m]^T X_c`` per lane on the (1,024, t) view, with the PCMM-rounded
    matrix (``wbits``) or the exact one (None)."""
    def run(xs, lin: Lin):
        W = lin.W if wbits is None else quantized(lin, wbits)
        D = L.D
        X = np.concatenate([np.asarray(x).reshape(D, L.t) for x in xs], axis=0)
        Y = W.T.astype(np.float64) @ X
        return [Y[m * D:(m + 1) * D].reshape(-1) for m in range(lin.n_out)]
    return run


def pcmm_shifts(calib: dict, band=0.25, margin=1.5) -> dict:
    """``impl.pcmm.band_shift``'s rule on a calibration ``{linear: max |y|}`` (``PF_PCMM_CALIB``'s file)."""
    import math
    return {k: max(0, math.ceil(math.log2(v * margin / band))) if v > 0 else 0 for k, v in calib.items()}


# ── the model ──

class Rwkv7Fhe(ImplModel):
    """The model on a session, the fake or the capture recorder. Stages ``blk0..blk{n-1}`` and ``tail`` (the reply:
    ``z`` masked to the live channels; the client decrypts it and applies ``ln_out`` and the head). A stage's input
    and output are the ``L.S`` shard ciphertexts of ``z`` (a list; one at width <= 1,024). With ``S > 1`` and
    ``shard_land`` every block also refreshes its output shards (a deliberate refresh, as the carried state), so each
    shard enters the next block at the canonical landing: the stage chain of the planned runtime binds only the first
    output to the next block's entry (``compile_dir``), the others enter as registers at a fixed (level, deg)."""

    def __init__(self, inf, P: Params, coeffs: dict, *, core=None, n_layers=None, unit=None, wbits=24,
                 fuse_l1=False, shifts=None, carry=True, chunk=0, prefill_keys=0, batch=0, batch_carry="decode",
                 batch_hoist=0, lora="auto", shard_land=True, lanes=0, layout="current", carry_drop=0,
                 carry_skip_s=False):
        super().__init__(inf, core=core, unit=unit)
        self.P = P
        self.n_layers = int(P.n_layer if n_layers is None else n_layers)
        self.coeffs = coeffs
        from .layout import auto_lanes
        lanes = int(lanes) or auto_lanes(P.C, self.rt.dims.N)
        self.L = Layout(self.rt.dims.N, P.C, P.d_ffn, D=self.rt.dims.N // lanes, layout=layout)
        self.wbits, self.fuse_l1, self.shifts, self.carry_on = int(wbits), bool(fuse_l1), shifts, bool(carry)
        self.shard_land = bool(shard_land) and self.L.S > 1
        self.W = [LayerW(P, b, self.L, coeffs, fuse_l1=fuse_l1, lora=lora) for b in range(self.n_layers)]
        self.decode_circ = Circuit(self.rt.ops, self.L, coeffs, self._pcmm, step=self.rt.step)
        self.state: list[State] = []
        self.init_state = None          # a public ref.State that start() lands instead of zeros (the prefix's)
        self.pcmm_outputs = 0
        # prefill (prefill.py): chunk T > 0 runs every step as a T-token pass on token lanes; prefill_keys T also
        # loads a T-token pass's rotation keys while decoding (a session that serves both)
        self.prefill_keys = int(prefill_keys)
        self.batch, self.batch_carry, self.batch_hoist = 0, str(batch_carry), int(batch_hoist)
        # the carried state's refresh levers (SPEEDUP_MAP_20261105.md S1 / S2): run_capture's --carry-drop and
        # --carry-skip-s set them on the capture model only (recorded in rwkv7_capture.json; run_planned refuses a
        # capture that carries one: the driver has no two-program cycle), default off:
        #   carry_drop=d     the S carries (and S's landing in start()) refresh under raise drop d (1, 3 or 5: the
        #                    runtime's built variants): the capture recorder lands them 2 d units deeper
        #                    (impl/fakecap.py inner_bootstrap, CapFhe.raise_drop); xa / xf keep drop 0
        #   carry_skip_s     the two-pass cycle's schedule: at odd positions the carry refreshes xa / xf only and
        #                    leaves S unrefreshed (pass A: S in at the landing, out 6 units deeper), at even
        #                    positions S is refreshed (pass B: S in 6 units deeper, out at the landing); start()
        #                    still lands S. A capture at tick 1 records pass A, at tick 2 pass B
        self.carry_drop = int(carry_drop)
        if self.carry_drop not in CARRY_DROPS:
            raise ValueError(f"carry_drop={self.carry_drop}: the runtime builds the drop variants {CARRY_DROPS[1:]}")
        self.carry_skip_s = bool(carry_skip_s)
        self.set_chunk(chunk)
        if batch:
            self.set_batch(batch)

    def set_batch(self, K: int):
        """K conversations per pass, conversation c on lane c (batched.py). K = 1 is the decode step itself (its
        circuit, its state, its op stream); ``step`` then takes and returns one row in a list."""
        from .batched import BatchedCircuit
        if self.chunk:
            raise ValueError("a batched pass and a prefill chunk do not combine")
        self.batch = int(K)
        self.circ = (self.decode_circ if self.batch <= 1 else
                     BatchedCircuit(self.rt.ops, self.L, self.coeffs, self._pcmm, self.batch, self.batch_carry,
                                    self.batch_hoist, step=self.rt.step))

    def set_mode(self, chunk: int = 0, batch: int = 0):
        """One twin, three programs (serve.py): the decode step (both 0), a T-token prefill pass (``chunk`` T) or a
        K-conversation batched pass (``batch`` K >= 2). The carried state stays the caller's: decode and prefill read
        a ``State`` list, a batched pass a ``BatchState`` list (``self.state``). Build the model with ``batch=K`` and
        ``prefill_keys=T`` so its rotation keys cover all three."""
        if chunk and batch:
            raise ValueError("a batched pass and a prefill chunk do not combine")
        self.batch = 0
        self.set_chunk(int(chunk))
        if batch:
            self.set_batch(int(batch))

    def set_chunk(self, T: int):
        """0: decode steps (one token on every lane); T: T-token prefill passes. The carried state is the same."""
        from .prefill import PrefillCircuit
        self.chunk = int(T)
        self.circ = (self.decode_circ if not self.chunk else
                     PrefillCircuit(self.rt.ops, self.L, self.coeffs, self._pcmm, self.chunk, step=self.rt.step))
        self.prefill_keys = max(self.prefill_keys, self.chunk)

    def _pcmm(self, xs, lin: Lin):
        from perseus.impl import pcmm
        self.pcmm_outputs += lin.n_out
        return pcmm.run(self.rt, list(xs), pcmm_encode(lin, self.L.t, self.wbits, self.shifts))

    def linears(self) -> dict:
        return {lin.name: pcmm_encode(lin, self.L.t, self.wbits, self.shifts)
                for W in self.W for lin in W.linears()}

    def rotation_steps(self) -> set[int]:
        from perseus.impl import pcmm
        from .prefill import rotation_steps as prefill_steps
        s = self.L.rotation_steps() | {s % self.L.n for s in pcmm.rot_steps()}
        if self.batch > 1:
            from .batched import rotation_steps as batch_steps
            s |= batch_steps(self.L, self.batch, self.batch_hoist)
        return s | prefill_steps(self.L, self.prefill_keys) if self.prefill_keys else s

    def zero_ct(self):
        ops = self.rt.ops
        return ops.pack_tokens(np.zeros((self.L.t, self.L.D)), ops.bts_out())

    def state_rows(self, state) -> list[list[np.ndarray]]:
        """A plaintext state (``ref.State``: the public prefix's) as ``pack_tokens`` rows per block, in
        ``State.cts()`` order (the S M state ciphertexts, the xa shards, the xf shards); batched: every conversation
        starts from it, in ``BatchState.cts()`` order."""
        L = self.L
        out = []
        for b in range(self.n_layers):
            # the filled state's rows are complex (group q real, group q + M imaginary); the current layout's real
            rows = [(v if L.filled else np.real(v)).reshape(L.D, L.t).T.copy() for v in L.state_slots_sh(state.S[b])]
            xa, xf = L.encode_row_sh(state.xa[b]), L.encode_row_sh(state.xf[b])
            if self.batch > 1:
                K = self.batch
                carries = xa + xf if self.batch_carry == "lane" else xa * K + xf * K
                out.append(rows * K + carries)
            else:
                out.append(rows + xa + xf)
        return out

    def state_names(self) -> list[str]:
        """One block's carried ciphertexts' names, in ``cts()`` order."""
        if self.batch > 1:
            from .batched import state_names as bnames
            return bnames(self.batch, self.L.M, self.batch_carry == "lane", self.L.S)
        return state_names(self.L.S, self.L.M)

    def make_state(self, cts):
        """One block's State / BatchState from its carried ciphertexts in ``state_names()`` order."""
        Sh, M = self.L.S, self.L.M
        if self.batch > 1:
            from .batched import BatchState
            K, SM = self.batch, Sh * M
            S = [cts[c * SM:(c + 1) * SM] for c in range(K)]
            rest = cts[K * SM:]
            if self.batch_carry == "lane":
                return BatchState(S, rest[:Sh], rest[Sh:2 * Sh], True)
            return BatchState(S, [rest[c * Sh:(c + 1) * Sh] for c in range(K)],
                              [rest[(K + c) * Sh:(K + c + 1) * Sh] for c in range(K)], False)
        return State(cts[: Sh * M], cts[Sh * M:Sh * M + Sh], cts[Sh * M + Sh:Sh * M + 2 * Sh])

    @contextlib.contextmanager
    def _carry_drop(self):
        """The S carries' refreshes under raise drop ``carry_drop`` (the capture recorder's ``raise_drop``; a session
        or the plain fake ignores the attribute: only the recorded graph carries the deeper landing)."""
        fhe = self.rt.ops.fhe
        if not self.carry_drop or not hasattr(fhe, "inner_bootstrap"):
            yield
            return
        old = getattr(fhe, "raise_drop", 0)
        fhe.raise_drop = self.carry_drop
        try:
            yield
        finally:
            fhe.raise_drop = old

    def land_state(self, st, skip_s=False):
        """The carry refresh of one block's state: S under the drop, xa / xf plain; ``skip_s`` leaves S unrefreshed
        (the two-pass cycle's pass A)."""
        ops = self.rt.ops
        s_cts = st.s_cts()
        if not skip_s:
            with self._carry_drop():
                for c in s_cts:
                    ops.bootstrap(c)
        for c in st.cts()[len(s_cts):]:
            ops.bootstrap(c)

    def start(self, state=None):
        """Zero state (or ``state``, a public ``ref.State``), each ciphertext landed by a bootstrap (as every carried
        value is), so the first token enters the circuit every later token does."""
        state = self.init_state if state is None else state
        rows = None if state is None else self.state_rows(state)
        self.state = []
        n = len(self.state_names())
        for b in range(self.n_layers):
            if rows is None:
                cts = [self.zero_ct() for _ in range(n)]
            else:
                cts = [self.pack_rows(r) for r in rows[b]]
            st = self.make_state(cts)
            self.land_state(st)
            self.state.append(st)

    def pack_rows(self, r):
        """``pack_tokens`` of real rows; complex rows (the filled state) as ``pack(Re) + i pack(Im)`` (level-free)."""
        ops = self.rt.ops
        r = np.asarray(r)
        if not np.iscomplexobj(r):
            return ops.pack_tokens(r, ops.bts_out())
        return ops.add(ops.pack_tokens(r.real, ops.bts_out()), ops.mult_i(ops.pack_tokens(r.imag, ops.bts_out())))

    def encode_input(self, row):
        """Block 0's input shards: the embedding row (ln0 applied, client side) on every lane, at the landing level;
        in a prefill pass the chunk's rows (T, C), row l on lane l; batched: row c on lane c."""
        ops, L = self.rt.ops, self.L
        if self.batch:
            # batched: row c (conversation c's token) on lane c; K = 1: decode's input, row 0 on every lane
            rows = L.encode_rows_sh(row) if self.batch > 1 else L.encode_row_sh(np.asarray(row)[0])
        else:
            rows = L.encode_rows_sh(row) if self.chunk else L.encode_row_sh(row)
        return [ops.pack_tokens(r, ops.bts_out()) for r in rows]

    def name_state(self, st: State):
        for name, ct in zip(st.names(), st.cts()):
            lvl = self.rt.ops.level(ct)
            self.inf.name_ct(ct, f"{self.inf.block_prefix}cache.{name}-lvl={lvl}")

    def stage_caches(self, i):
        if i >= self.n_layers or not self.state:
            return None
        st = self.state[i]
        return st.cts(), [f"impl.b{i}.{n}" for n in st.names()]

    def token_stages(self, x, pos):  # noqa: ARG002
        def block_fn(b):
            def compute(z):
                cap = self.enter_stage(b, pos)
                st = self.state[b]
                self.name_state(st)
                with self.rt.step(f"blk{b}"):
                    z = self.circ.block(b, z, st, self.W[b])
                    if self.carry_on:
                        with self.rt.step("carry"):
                            self.land_state(st, skip_s=self.carry_skip_s and pos % 2 == 1)
                            if self.shard_land:
                                for c in z:
                                    self.rt.ops.bootstrap(c)
                self.exit_stage(b, cap)
                return z
            return compute

        def tail_fn(z):
            b = self.n_layers
            cap = self.enter_stage(b, pos)
            out = []
            with self.rt.step("tail"):
                # one deliberate refresh per shard: a stage the planner sees with no refresh at all trips its summary
                # (a None in a format), and a refreshed reply leaves the client the most noise headroom
                for zi, lv in zip(z, self.L.lives):
                    o = self.rt.ops.copy(zi)
                    self.rt.ops.bootstrap(o)
                    out.append(self.rt.ops.mult_pt(o, lv))
            self.exit_stage(b, cap)
            return out

        stages = [(f"blk{b}", None, block_fn(b)) for b in range(self.n_layers)]
        stages.append(("tail", None, tail_fn))
        return stages

    def step(self, row, pos):
        """One token on the fake / recorder: returns (hidden (C,), the reply shards); in a prefill pass ``row`` is the
        chunk (T, C) and the hidden states are (T, C)."""
        self.token_begin(pos)
        x = self.encode_input(row)
        out = self.run_stages(x, pos, self.token_stages(x, pos))
        self.token_tail(pos)
        vals = [self.rt.ops.decrypt_slots(o) for o in out]
        L = self.L
        if self.batch:
            return (L.decode_lanes_sh(vals, len(row)) if self.batch > 1 else L.decode_sh(vals)[None]), out
        return (L.decode_lanes_sh(vals, len(row)) if self.chunk else L.decode_sh(vals)), out


class Mirror:
    """The same circuit on numpy slot vectors (``NumpyOps``): no levels, no noise, no clamp; with ``wbits`` the
    PCMM-rounded matrices. ``fake - mirror`` is the runtime's arithmetic, ``mirror - poly reference`` the layout and
    the folds (and any unclamped excursion). ``step`` returns (hidden, the S output slot vectors)."""

    def __init__(self, P: Params, coeffs: dict, n_slots: int, n_layers=None, wbits: int | None = 24, fuse_l1=False,
                 lora="auto", lanes=0, layout="current"):
        """``lanes``: t (channels per ciphertext n_slots / t); 0 = ``layout.auto_lanes`` (at 2^15 slots the device's
        t; at width <= 1,024 it is n_slots / 1,024, the tests' t = 16 at 2^14). ``layout``: the state layout."""
        from perseus.impl.ops import NumpyOps
        from .layout import auto_lanes
        self.P = P
        if lanes == 0:
            lanes = auto_lanes(P.C, n_slots)
        self.L = Layout(n_slots, P.C, P.d_ffn, **({"D": n_slots // int(lanes)} if lanes else {}), layout=layout)
        self.n_layers = int(P.n_layer if n_layers is None else n_layers)
        self.W = [LayerW(P, b, self.L, coeffs, fuse_l1=fuse_l1, lora=lora) for b in range(self.n_layers)]
        self.circ = Circuit(NumpyOps(), self.L, coeffs, numpy_linear(self.L, wbits))
        self.reset()

    def reset(self, state=None):
        """Zero state, or ``state`` (a ``ref.State``)."""
        L = self.L
        z = np.zeros(L.n, dtype=np.complex128)
        if state is None:
            self.state = [State([z.copy() for _ in range(L.S * L.M)], [z.copy() for _ in range(L.S)],
                                [z.copy() for _ in range(L.S)]) for _ in range(self.n_layers)]
            return
        c = np.complex128
        self.state = [State([s.astype(c) for s in L.state_slots_sh(state.S[b])],
                            [x.astype(c) for x in L.natural_sh(state.xa[b])],
                            [x.astype(c) for x in L.natural_sh(state.xf[b])]) for b in range(self.n_layers)]

    def step(self, row):
        z = [v.astype(np.complex128) for v in self.L.natural_sh(row)]
        for b in range(self.n_layers):
            z = self.circ.block(b, z, self.state[b], self.W[b])
        z = [zi * lv for zi, lv in zip(z, self.L.lives)]
        return self.L.decode_sh([np.real(zi) for zi in z]), z

    def prefill(self, rows):
        """A chunk of T rows in one pass on token lanes (prefill.py): (hidden (T, C), the S slot vectors)."""
        from .prefill import PrefillCircuit
        T = len(rows)
        circ = PrefillCircuit(self.circ.ops, self.L, self.circ.coeffs, self.circ.linear, T)
        z = [v.astype(np.complex128) for v in self.L.chunk_sh(rows)]
        for b in range(self.n_layers):
            z = circ.block(b, z, self.state[b], self.W[b])
        z = [zi * lv for zi, lv in zip(z, self.L.lives)]
        return self.L.decode_lanes_sh([np.real(zi) for zi in z], T), z
