"""A random checkpoint in the trainer's state-dict names, with Newton seeds fitted to a
mean-square window: what the tests (and a GPU-free smoke run) use instead of the real weights.
The shapes follow the real model's ratios (d_ffn = 4 d); the sizes are the caller's."""

from __future__ import annotations

import numpy as np

from . import weights


def linear_seed(m, M):
    """``y0 = a + b v`` fitted to ``1/sqrt(v)`` on ``[m, M]`` (the fhe-ssm repository's
    ml-eval/fhe_circuit_sim.linear_rsqrt_seed)."""
    grid = np.geomspace(m, M, 64)
    tgt = 1.0 / np.sqrt(grid)
    A = np.stack([np.ones_like(grid), grid], axis=1) / tgt[:, None]
    sol, *_ = np.linalg.lstsq(A, np.ones_like(grid), rcond=None)
    return float(sol[0]), float(sol[1])


def synthetic(L=2, d=32, dff=128, seed=0, ms=(2e-4, 2e-3)):
    """A random checkpoint in the trainer's state-dict names with seeds for the given
    mean-square window, plus the config the loader checks."""
    rng = np.random.default_rng(seed)
    w = {
        "g_out": 1.0 + 0.1 * rng.standard_normal(d),
        "emb.weight": 0.03 * rng.standard_normal((64, d)),
    }
    seeds = {}
    a, b = linear_seed(*ms)
    for i in range(L):
        p = f"blocks.{i}."
        w[p + "tm.g1"] = 1.0 + 0.1 * rng.standard_normal(d)
        w[p + "tm.mix"] = rng.uniform(0.3, 0.9, d)
        w[p + "tm.a_raw"] = rng.standard_normal(d)
        for k in ("c", "dd"):
            w[p + f"tm.{k}"] = 0.5 * rng.standard_normal(d)
        w[p + "tm.p0"], w[p + "tm.p1"], w[p + "tm.p2"] = (
            0.5 + 0.1 * rng.standard_normal(d),
            0.25 * rng.standard_normal(d),
            0.05 * rng.standard_normal(d),
        )
        w[p + "tm.win.weight"] = rng.standard_normal((d, d)) / np.sqrt(d) * 0.03
        w[p + "tm.wout.weight"] = rng.standard_normal((d, d)) / np.sqrt(d) * 0.03
        w[p + "cm.g2"] = 1.0 + 0.1 * rng.standard_normal(d)
        w[p + "cm.wk.weight"] = rng.standard_normal((dff, d)) / np.sqrt(d) * 0.03
        w[p + "cm.wr.weight"] = rng.standard_normal((dff, d)) / np.sqrt(d) * 0.03
        w[p + "cm.wv.weight"] = rng.standard_normal((d, dff)) / np.sqrt(dff) * 0.03
        for k, s in (
            ("q0", 0.1),
            ("q1", 1.0),
            ("q2", 0.1),
            ("q3", 0.05),
            ("r0", 0.5),
            ("r1", 0.25),
            ("r2", 0.05),
        ):
            w[p + f"cm.{k}"] = s * (1.0 + 0.1 * rng.standard_normal(dff))
        seeds[f"L{i}.tm"] = seeds[f"L{i}.cm"] = {"a": a, "b": b, "iters": 3}
    seeds["ln_out"] = {"a": a, "b": b, "iters": 3}
    cfg = {
        "d_model": d,
        "n_layer": L,
        "d_ffn": dff,
        "a_min": 0.9,
        "a_max": 0.9995,
        "alpha_res": 1.0,
        "nonlin": "poly",
        "shift_mix": True,
        "gates": "expanded",
        "vocab": 64,
    }
    return weights.ModelParams(w, seeds, cfg), w, seeds, cfg
