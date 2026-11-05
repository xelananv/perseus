# SPDX-License-Identifier: MIT
"""Synthetic RWKV-7 (x070) checkpoints of any width, for the CPU checks of the any-size port (ANYSIZE_DESIGN.md).

The 1.5B / 2.9B / 7.2B / 13.3B ``.pth`` files are not on the development machine. A synthetic checkpoint has the real
structure (every tensor name and shape of ``torch.save(state_dict)`` of an ``RWKV-x070-World-*`` model: the heads of
64, d_ffn = 4 C, the LoRA ranks of w, a, v, g) with random values drawn at magnitudes like the 0.1B model's (matrices
scaled by 1/sqrt(fan-in)), written as ``.npz`` (``ckpt.load_state_dict`` reads it). Its coefficients come from
``fit_coeffs``: every site's input range recorded on the exact model over a few tokens, a Chebyshev interpolant on
that range widened by ``margin`` (degrees by site kind from the final 0.1B fit, so the depth per site is a real one),
and the 1/sqrt sites as ``newton_const`` (P6, ``ROBUST_NORMS.md``: accuracy range [m/R, M R], constant seed, the
fewest steps reaching ``tol``). Values are not a model: these checkpoints test the circuit's structure (layout,
shards, folds, the planner's refresh placement), never quality.

    python -m perseus.examples.rwkv7.synth --width 2048 --layers 2 --out ckpt.npz --coeffs coeffs.json
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np

from . import ref

#: the LoRA ranks of RWKV-7's training code (RWKV-LM v7 ``train_temp``), as functions of the width C: decay and a
#: max(32, round(2.5 sqrt C / 32) 32), v max(32, round(1.7 sqrt C / 32) 32), gate max(32, round(0.6 C^0.8 / 32) 32);
#: they give the 0.1B checkpoint's 64 / 64 / 32 / 128 at C = 768. A real checkpoint's own shapes always win (ref.py
#: reads every shape from the state dict).
def lora_ranks(C: int) -> dict:
    def r(x):
        return max(32, int(round(x / 32)) * 32)
    return {"w": r(2.5 * C ** 0.5), "a": r(2.5 * C ** 0.5), "v": r(1.7 * C ** 0.5), "g": r(0.6 * C ** 0.8)}


#: the published RWKV-7 World sizes (layers, width); the pipeline reads the real ones from the checkpoint
SIZES = {"0.1B": (12, 768), "0.4B": (24, 1024), "1.5B": (24, 2048), "2.9B": (32, 2560), "7.2B": (32, 4096),
         "13.3B": (61, 4096)}

#: per-kind Chebyshev degree and the 1/sqrt tolerance: the medians of the final 0.1B fit (gate 3,
#: results/rwkv7-fhe-20261104/new_pod/gate/0.1b/coeffs_final.json; degrees 194 / 113 / 154 / 65 / 141 / 56)
DEGREES = {"att.w_tanh": 194, "att.decay": 113, "att.a_sigmoid": 154, "att.g_sigmoid": 65, "att.v_sigmoid": 141,
           "ffn.relu2": 56}
EPS = {"ln.rsqrt": 1e-5, "att.gn_rsqrt": 64e-5, "att.kk_rsqrt": 0.0}


def make_state_dict(C: int, n_layer: int, *, vocab: int = 4096, seed: int = 0, ranks: dict | None = None) -> dict:
    """A random state dict with the x070 structure at width C (float32)."""
    if C % 64:
        raise ValueError(f"C = {C}: whole heads of 64")
    g = np.random.default_rng(seed)
    H, N, F = C // 64, 64, 4 * C
    rk = ranks or lora_ranks(C)
    f = math.sqrt(768.0 / C)

    def nrm(shape, std, mean=0.0):
        return (mean + std * g.standard_normal(shape)).astype(np.float32)

    def uni(shape, lo, hi):
        return g.uniform(lo, hi, shape).astype(np.float32)

    sd = {"emb.weight": nrm((vocab, C), 0.0155)}
    for b in range(n_layer):
        p = f"blocks.{b}."
        sd[p + "ln1.weight"], sd[p + "ln1.bias"] = nrm((C,), 0.05, 0.45), nrm((C,), 0.06)
        sd[p + "ln2.weight"], sd[p + "ln2.bias"] = nrm((C,), 0.1, 1.6), nrm((C,), 0.3)
        if b == 0:
            sd[p + "ln0.weight"], sd[p + "ln0.bias"] = nrm((C,), 0.01, 0.08), nrm((C,), 0.06)
        a = p + "att."
        for k, (lo, hi) in {"x_r": (0, 0.3), "x_w": (0.3, 0.9), "x_k": (0.5, 1.4), "x_v": (0, 0.6), "x_a": (0.3, 0.9),
                            "x_g": (0, 0.2)}.items():
            sd[a + k] = uni((1, 1, C), lo, hi)
        sd[a + "w0"] = np.clip(nrm((1, 1, C), 1.5, -3.0), -7, 1).astype(np.float32)
        sd[a + "r_k"] = nrm((H, N), 0.5, 0.5)
        sd[a + "w1"], sd[a + "w2"] = nrm((C, rk["w"]), 0.5 * f), nrm((rk["w"], C), 0.5)
        sd[a + "a1"], sd[a + "a2"] = nrm((C, rk["a"]), 0.3 * f), nrm((rk["a"], C), 0.3)
        sd[a + "a0"] = nrm((1, 1, C), 0.4, -0.3)
        sd[a + "g1"], sd[a + "g2"] = nrm((C, rk["g"]), 0.3 * f), nrm((rk["g"], C), 0.3)
        if b > 0:
            sd[a + "v2"], sd[a + "v1"] = nrm((rk["v"], C), 0.15), nrm((C, rk["v"]), 0.13 * f)
            sd[a + "v0"] = nrm((1, 1, C), 0.3, 0.8)
        sd[a + "k_k"], sd[a + "k_a"] = uni((1, 1, C), 0.5, 1.0), uni((1, 1, C), 0.9, 1.0)
        for k in ("receptance", "key", "value", "output"):
            sd[a + k + ".weight"] = nrm((C, C), 0.018 * f)
        sd[a + "ln_x.weight"], sd[a + "ln_x.bias"] = nrm((C,), 0.02, 0.15), nrm((C,), 0.05)
        sd[p + "ffn.x_k"] = uni((1, 1, C), 0.5, 1.0)
        sd[p + "ffn.key.weight"] = nrm((F, C), 0.018 * f)
        sd[p + "ffn.value.weight"] = nrm((C, F), 0.018 * f / 2)
    sd["ln_out.weight"], sd["ln_out.bias"] = nrm((C,), 0.5, 6.0), nrm((C,), 0.3)
    sd["head.weight"] = nrm((vocab, C), 0.019 * f)
    return sd


class _RangeSites(ref.Sites):
    """Exact sites that record every site's input min / max."""

    def __init__(self):
        super().__init__("exact")
        self.rng: dict[str, list[float]] = {}

    def __call__(self, name, x):
        x = np.asarray(x, dtype=np.float64)
        r = self.rng.setdefault(name, [np.inf, -np.inf])
        r[0], r[1] = min(r[0], float(x.min())), max(r[1], float(x.max()))
        return super().__call__(name, x)


def site_ranges(P: ref.Params, ids) -> dict:
    s = _RangeSites()
    m = ref.RefModel(P, s)
    for t in ids:
        m.step(int(t))
    return s.rng


def cheb_coeffs(fn, lo, hi, deg) -> list[float]:
    """The degree-``deg`` Chebyshev interpolant of ``fn`` on [lo, hi] (coefficients on t = (2x - lo - hi)/(hi - lo))."""
    from numpy.polynomial import chebyshev as Ch
    return [float(c) for c in Ch.chebinterpolate(lambda t: fn(0.5 * (hi - lo) * t + 0.5 * (hi + lo)), deg)]


def newton_const_site(m: float, M: float, eps: float, R: float = 10.0, tol: float = 1e-3, max_steps: int = 40) -> dict:
    """``rsqrt_const.py``'s design: accuracy range [m/R, M R], scale = M R + eps, seed 1, the fewest steps with max
    relative error <= tol over the accuracy range (a 4,001-point log grid)."""
    m = max(float(m), 1e-12)
    lo, hi = m / R, max(float(M), m) * R
    scale = hi + eps
    x = np.geomspace(lo, hi, 4001)
    u = (x + eps) / scale
    want = 1.0 / np.sqrt(x + eps)
    y = np.ones_like(u)
    for steps in range(1, max_steps + 1):
        y = y * (1.5 - 0.5 * u * y * y)
        err = float(np.max(np.abs(y * scale ** -0.5 - want) / want))
        if err <= tol:
            break
    return {"method": "newton_const", "eps": eps, "scale": scale, "steps": steps, "lo": lo, "hi": hi,
            "conv_lo": -eps, "conv_hi": 3 * scale - eps, "R": R, "tol": tol, "max_rel_err": err,
            "calib_min": m, "calib_max": float(M)}


def fit_coeffs(P: ref.Params, ids, *, margin: float = 0.5, degrees: dict | None = None, R: float = 10.0) -> dict:
    """A coefficients document (fit.py's format: ``{"meta": ..., "sites": {name: ...}}``) for ``P`` from the site
    ranges over ``ids`` on the exact model."""
    degrees = {**DEGREES, **(degrees or {})}
    rng = site_ranges(P, ids)
    sites = {"ln0.rsqrt": {"exact": True, "reason": "client-side"},
             "ln_out.rsqrt": {"exact": True, "reason": "client-side"}}
    for name, (m, M) in sorted(rng.items()):
        kind = ref.site_kind(name)
        if kind.endswith("rsqrt"):
            sites[name] = {"kind": kind, **newton_const_site(m, M, EPS[kind], R)}
            continue
        w = max(M - m, 1e-3)
        lo, hi = m - margin * w, M + margin * w
        deg = int(degrees[kind])
        sites[name] = {"kind": kind, "lo": lo, "hi": hi, "degree": deg,
                       "coeffs": cheb_coeffs(ref.SITE_FUNCS[kind], lo, hi, deg), "calib_min": m, "calib_max": M}
    return {"meta": {"synthetic": True, "margin": margin, "R": R, "calib_tokens": [int(t) for t in ids],
                     "degrees": degrees}, "sites": sites}


def write(C: int, n_layer: int, out: str, coeffs_out: str | None = None, *, vocab=4096, seed=0, n_calib=24,
          margin=0.5) -> tuple[str, str | None]:
    sd = make_state_dict(C, n_layer, vocab=vocab, seed=seed)
    np.savez(out, **sd)
    if coeffs_out:
        P = ref.Params.from_state_dict(sd)
        ids = np.random.default_rng(seed + 1).integers(1, vocab, n_calib)
        with open(coeffs_out, "w") as f:
            json.dump(fit_coeffs(P, ids, margin=margin), f)
    return out, coeffs_out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--vocab", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--calib-tokens", type=int, default=24)
    ap.add_argument("--margin", type=float, default=0.5)
    ap.add_argument("--out", required=True, help="the checkpoint (.npz)")
    ap.add_argument("--coeffs", default=None, help="also write a fitted coefficients JSON here")
    a = ap.parse_args(argv)
    write(a.width, a.layers, a.out, a.coeffs, vocab=a.vocab, seed=a.seed, n_calib=a.calib_tokens, margin=a.margin)
    print(json.dumps({"checkpoint": a.out, "coeffs": a.coeffs, "width": a.width, "layers": a.layers,
                      "lora_ranks": lora_ranks(a.width)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
