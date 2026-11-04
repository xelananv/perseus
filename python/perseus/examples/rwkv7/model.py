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

import json
from dataclasses import dataclass

import numpy as np

from perseus.impl.driver import ImplModel
from perseus.impl.poly import eval_chebyshev_bsgs

from .layout import D, HEAD, Layout
from .ref import Params

STATE = ("s0", "s1", "s2", "s3")      # names of the M state ciphertexts (M = 2 at ring 2^16)


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


class LayerW:
    """Block ``b``'s operands: the PCMM linears and the natural-form constants."""

    def __init__(self, P: Params, b: int, L: Layout, coeffs: dict, *, fuse_l1=False):
        p, C = P.layers[b].p, P.C
        self.b = b
        one = np.ones(C)
        mixes = {k: p[k] for k in ("x_r", "x_w", "x_k", "x_v", "x_a", "x_g")}
        mats = [("r", "x_r", p["receptance_w"].T), ("k", "x_k", p["key_w"].T), ("v", "x_v", p["value_w"].T),
                ("a", "x_a", p["a1"] @ p["a2"]), ("w1", "x_w", p["w1"]), ("g1", "x_g", p["g1"])]
        if b > 0:
            mats.append(("vg", "x_v", p["v1"] @ p["v2"]))
        self.l1_kinds = [m[0] for m in mats]
        blocks = []
        for _kind, mix, Wt in mats:
            mu = mixes[mix]
            top = _pad((one - mu)[:, None] * Wt, D, D)        # x       (the current LN output)
            bot = _pad(mu[:, None] * Wt, D, D)                # x_prev  (the carried one)
            blocks.append(np.vstack([top, bot]))
        if fuse_l1:
            self.l1 = [Lin(f"b{b}.l1", np.hstack(blocks), 2, len(blocks))]
        else:
            self.l1 = [Lin(f"b{b}.l1{kind}", blk, 2, 1) for kind, blk in zip(self.l1_kinds, blocks)]
        self.w2 = Lin(f"b{b}.w2", _pad(p["w2"], D, D), 1, 1)
        self.g2 = Lin(f"b{b}.g2", _pad(p["g2"], D, D), 1, 1)
        self.wo = Lin(f"b{b}.out", _pad(p["output_w"].T, D, D), 1, 1)
        fx = p["ffn_x_k"]
        Wk = p["ffn_key_w"].T                                      # (C, d_ffn)
        self.fk = Lin(f"b{b}.fk", np.vstack([_pad((one - fx)[:, None] * Wk, D, P.d_ffn),
                                             _pad(fx[:, None] * Wk, D, P.d_ffn)]), 2, P.d_ffn // D)
        self.fv = Lin(f"b{b}.fv", _pad(p["ffn_value_w"].T, P.d_ffn, D), P.d_ffn // D, 1)
        lane = L.lane
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
        self.kk_pad = L.pad_heads(0.5 * (skk["lo"] + skk["hi"]) * self.kk_c ** 2)
        self.k_a, self.one_m_ka = lane(p["k_a"]), lane(1.0 - p["k_a"])
        sgn = coeffs[f"blocks.{b}.att.gn_rsqrt"]
        self.gn_pad = L.pad_heads(0.5 * (sgn["lo"] + sgn["hi"]))
        self.gn_w = lane(p["ln_x_w"] * sgn["scale"] ** -0.5)
        self.gn_b = lane(p["ln_x_b"])
        self.r_k = lane(p["r_k"].reshape(-1))

    def linears(self) -> list[Lin]:
        return [*self.l1, self.w2, self.g2, self.wo, self.fk, self.fv]


class State:
    def __init__(self, S, xa, xf):
        self.S, self.xa, self.xf = list(S), xa, xf

    def cts(self):
        return [*self.S, self.xa, self.xf]

    def names(self):
        return [*STATE[: len(self.S)], "xa", "xf"]


class Circuit:
    """The block on any ops (``linear(xs, lin)`` supplies the matrix product: PCMM on a session or the fake, numpy
    on the mirror)."""

    def __init__(self, ops, L: Layout, coeffs: dict, linear, step=None):
        self.ops, self.L, self.coeffs, self.linear = ops, L, coeffs, linear
        self.step = step

    def _scope(self, label):
        import contextlib
        return self.step(label) if self.step is not None else contextlib.nullcontext()

    # ── sites ──
    def cheb(self, name, x, scale=1.0):
        s = self.coeffs[name]
        c = [float(v) * scale for v in s["coeffs"]]
        return eval_chebyshev_bsgs(self.ops, x, c, float(s["lo"]), float(s["hi"]), eps=0.0)

    def newton(self, name, x, xscale=1.0):
        """y with y scale^-1/2 = 1/sqrt(xscale x + eps) (fit.py's Newton form, on the raw statistic x)."""
        ops, s = self.ops, self.coeffs[name]
        a, b0 = float(xscale) / s["scale"], s["eps"] / s["scale"]
        lo, hi = (s["seed_lo"] - b0) / a, (1.0 - b0) / a
        y = eval_chebyshev_bsgs(ops, x, [float(c) for c in s["seed_coeffs"]], lo, hi, eps=0.0)
        un = ops.add(ops.mult(x, -0.5 * a), -0.5 * b0)          # -u/2
        for _ in range(int(s["steps"])):
            y = ops.add(ops.mult(y, 1.5), ops.mult(ops.mult(y, un), ops.square(y)))
        return y

    def layer_norm(self, h2, name, consts):
        ops, L = self.ops, self.L
        s1 = L.channel_sum(ops, h2)
        xc = ops.sub(h2, ops.mult_pt(s1, L.live / L.C))           # 2 (h - mu), zero on the padding
        q = L.channel_sum(ops, ops.square(xc))                    # 4 C var
        y = self.newton(name, q, 1.0 / (4 * L.C))
        g, b = consts
        return ops.add_pt(ops.mult_pt(ops.mult(xc, y), g), b)

    # ── the two steps a prefill pass replaces (prefill.py) ──
    def token_shift(self, x, carry):
        """(the token-shift partner of ``x``, the carry the next step reads): one token, the carried LN output."""
        return carry, x

    def state_step(self, S, w, kkn, bb, kt, r, V):
        """One token's update of the M state ciphertexts and its readout terms ``S_m r`` (natural-form inputs;
        ``V = transpose(v)``)."""
        ops, L = self.ops, self.L
        newS, Ps = [], []
        for m, Sm in enumerate(S):
            Sa = L.block_allreduce(ops, ops.mult(Sm, kkn), -1.0)     # S a with a = -kk, broadcast over j
            Sn = ops.add(ops.add(ops.mult(Sm, w), ops.mult(Sa, bb)), ops.mult(V[m], kt))
            newS.append(Sn)
            Ps.append(ops.mult(Sn, r))
        return newS, Ps

    def wkv(self, st: State, w, kkn, bb, kt, r, v):
        st.S, Ps = self.state_step(st.S, w, kkn, bb, kt, r, self.L.transpose(self.ops, v))
        return self.L.untranspose(self.ops, Ps)

    # ── the block ──
    def block(self, b, z, st: State, W: LayerW):
        ops, L = self.ops, self.L
        pre = f"blocks.{b}."
        with self._scope("tm.norm"):
            x = self.layer_norm(ops.add(z, ops.conjugate(z)), pre + "ln1.rsqrt", W.ln["ln1"])
        with self._scope("tm.in"):
            xp, st.xa = self.token_shift(x, st.xa)
            outs = []
            for lin in W.l1:
                outs += self.linear([x, xp], lin)
            o = dict(zip(W.l1_kinds, outs))
        with self._scope("tm.lora"):
            zdec = ops.add_pt(self.linear([self.cheb(pre + "att.w_tanh", o["w1"])], W.w2)[0], W.w0)
            w = self.cheb(pre + "att.decay", zdec)
            a = self.cheb(pre + "att.a_sigmoid", ops.add_pt(o["a"], W.a0))
            g = self.linear([self.cheb(pre + "att.g_sigmoid", o["g1"])], W.g2)[0]
            r, k, v = o["r"], o["k"], o["v"]
            if b > 0:
                q = ops.mult_i(ops.sub(ops.conjugate(z), z))       # 2 v_first
                sv = self.cheb(pre + "att.v_sigmoid", ops.add_pt(o["vg"], W.v0), scale=0.5)
                v = ops.add(v, ops.mult(ops.sub(q, ops.add(v, v)), sv))
        with self._scope("tm.kk"):
            kk = ops.mult_pt(k, W.k_k)                            # k k_k scale^-1/2
            nk = ops.add_pt(L.block_allreduce(ops, ops.square(kk)), W.kk_pad)
            kkn = ops.mult(kk, self.newton(pre + "att.kk_rsqrt", nk, 1.0 / W.kk_c ** 2))
            kt = ops.mult(k, ops.add_pt(ops.mult_pt(a, W.k_a), W.one_m_ka))
            bb = ops.mult(kkn, a)
        with self._scope("tm.state"):
            y = self.wkv(st, w, kkn, bb, kt, r, v)
        with self._scope("tm.gn"):
            yc = ops.sub(y, L.block_allreduce(ops, y, 1.0 / HEAD))
            var = ops.add_pt(L.block_allreduce(ops, ops.square(yc), 1.0 / HEAD), W.gn_pad)
            yn = ops.add_pt(ops.mult_pt(ops.mult(yc, self.newton(pre + "att.gn_rsqrt", var)), W.gn_w), W.gn_b)
            bonus = ops.mult(L.block_allreduce(ops, ops.mult_pt(ops.mult(r, kt), W.r_k)), v)
            yo = ops.mult(ops.add(yn, bonus), g)
        with self._scope("tm.out"):
            dx = self.linear([yo], W.wo)[0]
            if b == 0:
                z = ops.add(z, ops.mult_i(v))                    # v_first onto the imaginary axis
            z = ops.add(z, dx)
        with self._scope("cm.norm"):
            x2 = self.layer_norm(ops.add(z, ops.conjugate(z)), pre + "ln2.rsqrt", W.ln["ln2"])
        with self._scope("cm.ffn"):
            xp2, st.xf = self.token_shift(x2, st.xf)
            kf = self.linear([x2, xp2], W.fk)
            rf = [self.cheb(pre + "ffn.relu2", kj) for kj in kf]
            z = ops.add(z, self.linear(rf, W.fv)[0])
        return z


# ── linears ──

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
    ``z`` masked to the live channels; the client decrypts it and applies ``ln_out`` and the head)."""

    def __init__(self, inf, P: Params, coeffs: dict, *, core=None, n_layers=None, unit=None, wbits=24,
                 fuse_l1=False, shifts=None, carry=True, chunk=0, prefill_keys=0):
        super().__init__(inf, core=core, unit=unit)
        self.P = P
        self.n_layers = int(P.n_layer if n_layers is None else n_layers)
        self.coeffs = coeffs
        self.L = Layout(self.rt.dims.N, P.C, P.d_ffn)
        self.wbits, self.fuse_l1, self.shifts, self.carry_on = int(wbits), bool(fuse_l1), shifts, bool(carry)
        self.W = [LayerW(P, b, self.L, coeffs, fuse_l1=fuse_l1) for b in range(self.n_layers)]
        self.decode_circ = Circuit(self.rt.ops, self.L, coeffs, self._pcmm, step=self.rt.step)
        self.state: list[State] = []
        self.init_state = None          # a public ref.State that start() lands instead of zeros (the prefix's)
        self.pcmm_outputs = 0
        # prefill (prefill.py): chunk T > 0 runs every step as a T-token pass on token lanes; prefill_keys T also
        # loads a T-token pass's rotation keys while decoding (a session that serves both)
        self.prefill_keys = int(prefill_keys)
        self.set_chunk(chunk)

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
        return s | prefill_steps(self.L, self.prefill_keys) if self.prefill_keys else s

    def zero_ct(self):
        ops = self.rt.ops
        return ops.pack_tokens(np.zeros((self.L.t, D)), ops.bts_out())

    def state_rows(self, state) -> list[list[np.ndarray]]:
        """A plaintext state (``ref.State``: the public prefix's) as ``pack_tokens`` rows per block, in
        ``State.cts()`` order (S_0 .. S_{M-1}, xa, xf)."""
        L = self.L
        out = []
        for b in range(self.n_layers):
            rows = [np.real(v).reshape(D, L.t).T.copy() for v in L.state_slots(state.S[b])]
            out.append(rows + [L.encode_row(state.xa[b]), L.encode_row(state.xf[b])])
        return out

    def start(self, state=None):
        """Zero state (or ``state``, a public ``ref.State``), each ciphertext landed by a bootstrap (as every carried
        value is), so the first token enters the circuit every later token does."""
        ops = self.rt.ops
        state = self.init_state if state is None else state
        rows = None if state is None else self.state_rows(state)
        self.state = []
        for b in range(self.n_layers):
            if rows is None:
                cts = [self.zero_ct() for _ in range(self.L.M + 2)]
            else:
                cts = [ops.pack_tokens(r, ops.bts_out()) for r in rows[b]]
            for c in cts:
                ops.bootstrap(c)
            self.state.append(State(cts[: self.L.M], cts[self.L.M], cts[self.L.M + 1]))

    def encode_input(self, row):
        """Block 0's input: the embedding row (ln0 applied, client side) on every lane, at the landing level; in a
        prefill pass the chunk's rows (T, C), row l on lane l."""
        ops = self.rt.ops
        rows = self.L.encode_rows(row) if self.chunk else self.L.encode_row(row)
        return ops.pack_tokens(rows, ops.bts_out())

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
                            for c in st.cts():
                                self.rt.ops.bootstrap(c)
                self.exit_stage(b, cap)
                return z
            return compute

        def tail_fn(z):
            b = self.n_layers
            cap = self.enter_stage(b, pos)
            with self.rt.step("tail"):
                # one deliberate refresh: a stage the planner sees with no refresh at all trips its summary
                # (a None in a format), and a refreshed reply leaves the client the most noise headroom
                out = self.rt.ops.copy(z)
                self.rt.ops.bootstrap(out)
                out = self.rt.ops.mult_pt(out, self.L.live)
            self.exit_stage(b, cap)
            return out

        stages = [(f"blk{b}", None, block_fn(b)) for b in range(self.n_layers)]
        stages.append(("tail", None, tail_fn))
        return stages

    def step(self, row, pos):
        """One token on the fake / recorder: returns (hidden (C,), the reply ciphertext); in a prefill pass ``row``
        is the chunk (T, C) and the hidden states are (T, C)."""
        self.token_begin(pos)
        x = self.encode_input(row)
        out = self.run_stages(x, pos, self.token_stages(x, pos))
        self.token_tail(pos)
        vals = self.rt.ops.decrypt_slots(out)
        return (self.L.decode_lanes(vals, len(row)) if self.chunk else self.L.decode(vals)), out


class Mirror:
    """The same circuit on numpy slot vectors (``NumpyOps``): no levels, no noise, no clamp; with ``wbits`` the
    PCMM-rounded matrices. ``fake - mirror`` is the runtime's arithmetic, ``mirror - poly reference`` the layout and
    the folds (and any unclamped excursion)."""

    def __init__(self, P: Params, coeffs: dict, n_slots: int, n_layers=None, wbits: int | None = 24, fuse_l1=False):
        from perseus.impl.ops import NumpyOps
        self.P, self.L = P, Layout(n_slots, P.C, P.d_ffn)
        self.n_layers = int(P.n_layer if n_layers is None else n_layers)
        self.W = [LayerW(P, b, self.L, coeffs, fuse_l1=fuse_l1) for b in range(self.n_layers)]
        self.circ = Circuit(NumpyOps(), self.L, coeffs, numpy_linear(self.L, wbits))
        self.reset()

    def reset(self, state=None):
        """Zero state, or ``state`` (a ``ref.State``)."""
        z = np.zeros(self.L.n, dtype=np.complex128)
        if state is None:
            self.state = [State([z.copy() for _ in range(self.L.M)], z.copy(), z.copy())
                          for _ in range(self.n_layers)]
            return
        L = self.L
        self.state = [State([s.astype(np.complex128) for s in L.state_slots(state.S[b])],
                            L.natural(state.xa[b]).astype(np.complex128),
                            L.natural(state.xf[b]).astype(np.complex128)) for b in range(self.n_layers)]

    def step(self, row):
        z = self.L.natural(row).astype(np.complex128)
        for b in range(self.n_layers):
            z = self.circ.block(b, z, self.state[b], self.W[b])
        z = z * self.L.live
        return self.L.decode(np.real(z)), z

    def prefill(self, rows):
        """A chunk of T rows in one pass on token lanes (prefill.py): (hidden (T, C), the slot vector)."""
        from .prefill import PrefillCircuit
        T = len(rows)
        circ = PrefillCircuit(self.circ.ops, self.L, self.circ.coeffs, self.circ.linear, T)
        z = self.L.chunk(rows).astype(np.complex128)
        for b in range(self.n_layers):
            z = circ.block(b, z, self.state[b], self.W[b])
        z = z * self.L.live
        return self.L.decode_lanes(np.real(z), T), z
