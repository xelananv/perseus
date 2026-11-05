# SPDX-License-Identifier: MIT
"""The plaintext reference: RWKV-7 (x070) in RNN mode, numpy float64, one token at a time, every non-polynomial op
a named site evaluated ``exact`` or ``poly`` exactly as the research harness does
(``results/vector-d-20261104/harness/rwkv7_model.py``: ``Sites._poly``, ``cheb_eval``, ``newton_rsqrt_eval``; the
harness computes them in float32 under torch, this module in float64). The math is the harness's GPT-mode forward
written as a recurrence (``wkv7_recurrence``: ``S <- S diag(w) + (S a) b^T + v k^T``, ``y = S r`` with
``a = -kk``, ``b = kk * a``); the token shift's previous input is zero at the first token, as ``F.pad`` makes it.

Sites (the harness's names): ``blocks.L.ln1.rsqrt``, ``blocks.L.ln2.rsqrt`` (1/sqrt(var + 1e-5)), ``att.w_tanh``,
``att.decay`` (exp(-e^-0.5 sigmoid z)), ``att.a_sigmoid``, ``att.v_sigmoid`` (L >= 1), ``att.g_sigmoid``,
``att.kk_rsqrt`` (1/max(sqrt x, 1e-12)), ``att.gn_rsqrt`` (1/sqrt(var + 64e-5)), ``ffn.relu2``; ``ln0`` and
``ln_out`` exact (client side). The head is exact (client side).
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field

import numpy as np

EXP_M05 = math.exp(-0.5)


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


SITE_FUNCS = {
    "ln.rsqrt": lambda x: 1.0 / np.sqrt(x + 1e-5),
    "att.gn_rsqrt": lambda x: 1.0 / np.sqrt(x + 64e-5),
    "att.kk_rsqrt": lambda x: 1.0 / np.maximum(np.sqrt(np.maximum(x, 0.0)), 1e-12),
    "att.decay": lambda x: np.exp(-EXP_M05 * _sig(x)),
    "att.w_tanh": np.tanh,
    "att.a_sigmoid": _sig,
    "att.v_sigmoid": _sig,
    "att.g_sigmoid": _sig,
    "ffn.relu2": lambda x: np.maximum(x, 0.0) ** 2,
}


def site_kind(name: str) -> str:
    base = re.sub(r"^blocks\.\d+\.", "", name)
    if re.fullmatch(r"(ln\d*|ln_out)\.rsqrt", base):
        return "ln.rsqrt"
    return base


def cheb_eval(x, coeffs, lo, hi):
    """sum_k c_k T_k(t), t = (2x - lo - hi)/(hi - lo), by Clenshaw (the harness's cheb_eval, float64)."""
    t = (2.0 * x - (lo + hi)) / (hi - lo)
    c = np.asarray(coeffs, dtype=np.float64)
    if c.size == 1:
        return np.zeros_like(t) + c[0]
    b1 = np.zeros_like(t)
    b2 = np.zeros_like(t)
    for k in range(c.size - 1, 0, -1):
        b1, b2 = 2.0 * t * b1 - b2 + c[k], b1
    return t * b1 - b2 + c[0]


def newton_rsqrt(x, p, steps=None):
    """The harness's newton_rsqrt_eval: a Chebyshev seed on u = (x + eps)/scale in [seed_lo, 1], Newton on u."""
    u = (x + p["eps"]) / p["scale"]
    y = cheb_eval(u, p["seed_coeffs"], p["seed_lo"], 1.0)
    for _ in range(p["steps"] if steps is None else steps):
        y = y * (1.5 - 0.5 * u * y * y)
    return y * p["scale"] ** -0.5


def newton_scaled_rsqrt(x, p):
    """The converter's newton_scaled (tools/rwkv7/convert_newton_scaled.py): the constant seed, per-step (a, b):
    y_1 = a_0 - b_0 u, then y <- a y - b u y^3 on u = (x + eps)/scale; the accuracy range and the convergence region
    are newton_const's."""
    u = (x + p["eps"]) / p["scale"]
    a, b = p["steps"][0]
    y = a - b * u
    for a, b in p["steps"][1:]:
        y = a * y - b * u * y * y * y
    return y * p["scale"] ** -0.5


def newton_const_rsqrt(x, p, steps=None):
    """The harness's newton_const (rsqrt_const.py, P6): Newton from the constant seed 1 on u = (x + eps)/scale;
    converges for every u in (0, 3), accurate to the site's tol on [lo, hi]."""
    u = (x + p["eps"]) / p["scale"]
    y = np.ones_like(u)
    for _ in range(p["steps"] if steps is None else steps):
        y = y * (1.5 - 0.5 * u * y * y)
    return y * p["scale"] ** -0.5


class Sites:
    """``mode``: 'exact' or 'poly' (with ``coeffs``, the fit.py JSON); ``clamp``: clamp to [lo, hi] before a
    polynomial (poly mode's ``--clamp``); ``out_counts`` counts values outside each site's interval."""

    def __init__(self, mode="exact", coeffs=None, clamp=True):
        if mode not in ("exact", "poly"):
            raise ValueError(mode)
        self.mode, self.clamp = mode, bool(clamp)
        self.cfg = {}
        if coeffs is not None:
            cfg = json.load(open(coeffs)) if isinstance(coeffs, str) else coeffs
            self.cfg = {n: s for n, s in cfg["sites"].items() if not s.get("exact")}
        if mode == "poly" and not self.cfg:
            raise ValueError("poly mode needs a coefficients JSON")
        self.out_counts: dict[str, list[int]] = {}
        self.record: dict[str, np.ndarray] | None = None

    def __call__(self, name, x):
        x = np.asarray(x, dtype=np.float64)
        if self.record is not None:
            self.record[name] = x.copy()
        if self.mode == "poly" and name in self.cfg:
            s = self.cfg[name]
            lo, hi = float(s["lo"]), float(s["hi"])
            c = self.out_counts.setdefault(name, [0, 0])
            c[0] += x.size
            c[1] += int(((x < lo) | (x > hi)).sum())
            xx = np.clip(x, lo, hi) if self.clamp else x
            if s.get("method") == "newton":
                return newton_rsqrt(xx, s)
            if s.get("method") == "newton_const":
                return newton_const_rsqrt(xx, s)
            if s.get("method") == "newton_scaled":
                return newton_scaled_rsqrt(xx, s)
            return cheb_eval(xx, s["coeffs"], lo, hi)
        return SITE_FUNCS[site_kind(name)](x)


@dataclass
class Layer:
    p: dict          # the block's parameters, float64, by short name (att.receptance.weight -> receptance)


@dataclass
class Params:
    n_layer: int
    C: int
    H: int
    N: int
    d_ffn: int
    vocab: int
    emb: np.ndarray                  # ln0 folded (client side), (vocab, C) float32
    layers: list
    ln_out_w: np.ndarray
    ln_out_b: np.ndarray
    head: np.ndarray                 # (vocab, C) float32
    meta: dict = field(default_factory=dict)

    @staticmethod
    def from_state_dict(sd: dict, n_layer: int | None = None) -> "Params":
        L = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("blocks."))
        n = L if n_layer is None else int(n_layer)
        vocab, C = sd["emb.weight"].shape
        H, N = sd["blocks.0.att.r_k"].shape
        d_ffn = sd["blocks.0.ffn.key.weight"].shape[0]
        e = sd["emb.weight"].astype(np.float64)
        mu = e.mean(-1, keepdims=True)
        var = ((e - mu) ** 2).mean(-1, keepdims=True)
        emb = ((e - mu) / np.sqrt(var + 1e-5) * sd["blocks.0.ln0.weight"] + sd["blocks.0.ln0.bias"]).astype(np.float32)
        layers = []
        for b in range(n):
            pre = f"blocks.{b}."
            p = {}
            for k, v in sd.items():
                if not k.startswith(pre) or ".ln0." in k:
                    continue
                short = k[len(pre):].replace("att.", "").replace(".weight", "_w").replace(".bias", "_b")
                if k.startswith(pre + "ffn."):
                    short = "ffn_" + k[len(pre) + 4:].replace(".weight", "_w")
                p[short] = np.asarray(v, dtype=np.float64).reshape(v.shape)
            for k in ("x_r", "x_w", "x_k", "x_v", "x_a", "x_g", "w0", "a0", "v0", "k_k", "k_a"):
                if k in p:
                    p[k] = p[k].reshape(-1)
            p["ffn_x_k"] = p["ffn_x_k"].reshape(-1)
            layers.append(Layer(p))
        return Params(n, C, H, N, d_ffn, vocab, emb, layers, sd["ln_out.weight"].astype(np.float64),
                      sd["ln_out.bias"].astype(np.float64), sd["head.weight"])


def load_params(path: str, n_layer: int | None = None) -> Params:
    from .ckpt import load_state_dict
    return Params.from_state_dict(load_state_dict(path), n_layer)


class State:
    def __init__(self, P: Params):
        self.xa = [np.zeros(P.C) for _ in range(P.n_layer)]
        self.xf = [np.zeros(P.C) for _ in range(P.n_layer)]
        self.S = [np.zeros((P.H, P.N, P.N)) for _ in range(P.n_layer)]


def layer_norm(sites, name, x, w, b):
    mu = x.mean()
    xc = x - mu
    var = (xc * xc).mean()
    return xc * sites(name, np.array([var]))[0] * w + b


class RefModel:
    """``step(token)`` -> (logits, hidden) per token; ``hidden`` is the last block's output (before ``ln_out``),
    the value the encrypted circuit hands to the client."""

    def __init__(self, P: Params, sites: Sites):
        self.P, self.sites = P, sites
        self.state = State(P)

    def reset(self):
        self.state = State(self.P)

    def head(self, h):
        P = self.P
        mu = h.mean()
        xc = h - mu
        x = xc / np.sqrt((xc * xc).mean() + 1e-5) * P.ln_out_w + P.ln_out_b
        return P.head.astype(np.float64) @ x

    def block(self, b, x, v_first):
        P, s, st, p = self.P, self.sites, self.state, self.P.layers[b].p
        H, N = P.H, P.N
        pre = f"blocks.{b}."
        xa = layer_norm(s, pre + "ln1.rsqrt", x, p["ln1_w"], p["ln1_b"])
        xx = st.xa[b] - xa
        st.xa[b] = xa
        xr, xw, xk, xv, xa_, xg = (xa + xx * p[m] for m in ("x_r", "x_w", "x_k", "x_v", "x_a", "x_g"))
        r = p["receptance_w"] @ xr
        z = p["w0"] + s(pre + "att.w_tanh", xw @ p["w1"]) @ p["w2"]
        w = s(pre + "att.decay", z)
        k = p["key_w"] @ xk
        v = p["value_w"] @ xv
        if b == 0:
            v_first = v
        else:
            v = v + (v_first - v) * s(pre + "att.v_sigmoid", p["v0"] + (xv @ p["v1"]) @ p["v2"])
        a = s(pre + "att.a_sigmoid", p["a0"] + (xa_ @ p["a1"]) @ p["a2"])
        g = s(pre + "att.g_sigmoid", xg @ p["g1"]) @ p["g2"]
        kk = (k * p["k_k"]).reshape(H, N)
        kk = (kk * s(pre + "att.kk_rsqrt", (kk * kk).sum(-1, keepdims=True))).reshape(-1)
        k = k * (1 + (a - 1) * p["k_a"])
        S = st.S[b]
        rh, wh, kh, vh = (t.reshape(H, N) for t in (r, w, k, v))
        ah, bh = (-kk).reshape(H, N), (kk * a).reshape(H, N)
        Sa = np.einsum("hij,hj->hi", S, ah)
        S = S * wh[:, None, :] + Sa[:, :, None] * bh[:, None, :] + vh[:, :, None] * kh[:, None, :]
        st.S[b] = S
        y = np.einsum("hij,hj->hi", S, rh)
        yc = y - y.mean(-1, keepdims=True)
        var = (yc * yc).mean(-1, keepdims=True)
        y = (yc * s(pre + "att.gn_rsqrt", var)).reshape(-1) * p["ln_x_w"] + p["ln_x_b"]
        y = y + ((rh * kh * p["r_k"]).sum(-1, keepdims=True) * vh).reshape(-1)
        x = x + p["output_w"] @ (y * g)
        x2 = layer_norm(s, pre + "ln2.rsqrt", x, p["ln2_w"], p["ln2_b"])
        xx2 = st.xf[b] - x2
        st.xf[b] = x2
        kf = p["ffn_key_w"] @ (x2 + xx2 * p["ffn_x_k"])
        x = x + p["ffn_value_w"] @ s(pre + "ffn.relu2", kf)
        return x, v_first

    def step(self, token: int):
        x = self.P.emb[int(token)].astype(np.float64)
        v_first = None
        for b in range(self.P.n_layer):
            x, v_first = self.block(b, x, v_first)
        return self.head(x), x
