"""The checkpoint (``<tag>_weights.npz``, the trainer's state dict) and the exported Newton
seeds (``<tag>_seeds.json``) -> the operands one block needs.

Folds, all exact: the norm gains ``g1`` / ``g2`` ride into the rows of the linears that consume
the normalized value (``W_in``; ``W_k`` and ``W_r``), the
residual scale ``alpha_res`` into ``W_out``, and the decay enters as the two plaintexts
``a = a_min + (a_max - a_min) sigmoid(a_raw)`` and ``1 - a`` (the trainer's ``decay()``). Linear
weights are stored ``(out, in)`` and used as ``x @ W``, so they are transposed here.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any

import numpy as np

from .norm import NormSite

VECTORS_TM = ("mix", "c", "dd", "p0", "p1", "p2")
VECTORS_CM = ("q0", "q1", "q2", "q3", "r0", "r1", "r2")


@dataclasses.dataclass(frozen=True)
class BlockParams:
    """One block in float64, before any layout: matrices as ``(d_in, d_out)``."""

    win: np.ndarray
    wout: np.ndarray
    wk: np.ndarray
    wr: np.ndarray
    wv: np.ndarray
    a: np.ndarray
    mix: np.ndarray
    c: np.ndarray
    dd: np.ndarray
    p0: np.ndarray
    p1: np.ndarray
    p2: np.ndarray
    q0: np.ndarray
    q1: np.ndarray
    q2: np.ndarray
    q3: np.ndarray
    r0: np.ndarray
    r1: np.ndarray
    r2: np.ndarray
    norm_tm: NormSite
    norm_cm: NormSite


def decay(a_raw, a_min: float, a_max: float) -> np.ndarray:
    return a_min + (a_max - a_min) / (1.0 + np.exp(-np.asarray(a_raw, dtype=np.float64)))


def _f64(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def block_params(w, seeds, cfg, b: int, *, norm_kw=None) -> BlockParams:
    """Block ``b`` from the state dict ``w`` (any mapping of name -> array), the seeds mapping
    and the training config."""
    p = f"blocks.{b}."
    d, norm_kw = int(cfg["d_model"]), dict(norm_kw or {})
    site = {
        k: NormSite.from_seed(seeds[k]["a"], seeds[k]["b"], seeds[k]["iters"], d, **norm_kw)
        for k in (f"L{b}.tm", f"L{b}.cm")
    }
    tm, cm = site[f"L{b}.tm"], site[f"L{b}.cm"]
    g1 = _f64(w[p + "tm.g1"])
    g2 = _f64(w[p + "cm.g2"])
    vec = {k: _f64(w[p + f"tm.{k}"]) for k in VECTORS_TM}
    vec.update({k: _f64(w[p + f"cm.{k}"]) for k in VECTORS_CM})
    return BlockParams(
        win=_f64(w[p + "tm.win.weight"]).T * g1[:, None],
        wout=_f64(w[p + "tm.wout.weight"]).T * float(cfg.get("alpha_res", 1.0)),
        wk=_f64(w[p + "cm.wk.weight"]).T * g2[:, None],
        wr=_f64(w[p + "cm.wr.weight"]).T * g2[:, None],
        wv=_f64(w[p + "cm.wv.weight"]).T,
        a=decay(w[p + "tm.a_raw"], float(cfg["a_min"]), float(cfg["a_max"])),
        norm_tm=tm,
        norm_cm=cm,
        **vec,
    )


class ModelParams:
    """The model over a lazily read state dict (``numpy.load`` of the npz, or any mapping):
    ``block(b)`` builds one block in float64 on demand and keeps the last two, so a 24-block pass
    never holds the whole checkpoint in float64."""

    def __init__(self, store, seeds, cfg, *, n_layers=None, norm_kw=None, keep=2):
        self.store, self.seeds, self.cfg = store, seeds, cfg
        self.norm_kw = dict(norm_kw or {})
        self.d, self.d_ffn = int(cfg["d_model"]), int(cfg["d_ffn"])
        self.n_layers = int(cfg["n_layer"] if n_layers is None else n_layers)
        s = seeds["ln_out"]
        self.norm_out = NormSite.from_seed(s["a"], s["b"], s["iters"], self.d, **self.norm_kw)
        self.g_out = _f64(store["g_out"])
        self._emb = None
        self._blocks, self._keep = {}, int(keep)

    @property
    def emb(self) -> np.ndarray:
        """The tied embedding as stored (float32): the client's input rows and output head."""
        if self._emb is None:
            self._emb = np.asarray(self.store["emb.weight"])
        return self._emb

    @property
    def vocab(self) -> int:
        return int(self.emb.shape[0])

    def emb_row(self, token: int) -> np.ndarray:
        return _f64(self.emb[int(token)])

    def block(self, b: int) -> BlockParams:
        p = self._blocks.get(b)
        if p is None:
            p = block_params(self.store, self.seeds, self.cfg, b, norm_kw=self.norm_kw)
            if len(self._blocks) >= self._keep:
                del self._blocks[next(iter(self._blocks))]
            self._blocks[b] = p
        return p

    def blocks(self):
        return (self.block(b) for b in range(self.n_layers))


def load_model(weights_npz: str, seeds_json: str, config_json: str, **kw) -> ModelParams:
    with open(seeds_json, encoding="utf-8") as f:
        seeds = json.load(f)
    with open(config_json, encoding="utf-8") as f:
        cfg = json.load(f)
    for k, want in (("nonlin", "poly"), ("shift_mix", True), ("gates", "expanded")):
        if cfg.get(k) != want:
            raise ValueError(f"config {k}={cfg.get(k)!r}: this port is written for {want!r}")
    if cfg.get("block", "sequential") != "sequential":
        raise ValueError("the parallel block is not ported")
    return ModelParams(np.load(weights_npz), seeds, cfg, **kw)


@dataclasses.dataclass(frozen=True)
class BlockWeights:
    """The block's operands in one geometry: the linears as the geometry's weight objects, every
    d-wide per-channel vector as a plaintext slot vector, every d_ffn-wide one as a list of them
    (one per tile)."""

    tag: str
    win: Any
    wout: Any
    wk: Any
    wr: Any
    wv: Any
    a: np.ndarray
    b: np.ndarray
    mix: np.ndarray
    imix: np.ndarray
    c: np.ndarray
    dd: np.ndarray
    p0: np.ndarray
    p1: np.ndarray
    p2: np.ndarray
    q0: list[np.ndarray]
    q1: list[np.ndarray]
    q2: list[np.ndarray]
    q3: list[np.ndarray]
    r0: list[np.ndarray]
    r1: list[np.ndarray]
    r2: list[np.ndarray]
    norm_tm: NormSite
    norm_cm: NormSite

    def key(self, name: str) -> str:
        return self.tag + name


def encode_block(p: BlockParams, geo, tag: str, named: bool = True) -> BlockWeights:
    """``named=False`` leaves the linears' plaintexts unnamed (encoded at every use, nothing
    cached by the runtime): the low-memory choice for a CPU pass over many blocks.

    On split pairs (``geo.split_gain = 0.5``) the operands that make the inputs of the two gate
    sections are halved, exactly (a power of two): ``c`` and ``dd`` (they make ``z``) and
    ``W_k``, ``W_r`` (they make ``k`` and ``r``), so the split's halves ``2 Re`` and ``2 Im``
    carry the true values."""
    lane, ffn, half = geo.lane, geo.ffn, float(geo.split_gain)

    def lin(w, name):
        return geo.weight(w, tag + name if named else None)

    return BlockWeights(
        tag=tag,
        win=lin(p.win, "win"),
        wout=lin(p.wout, "wout"),
        wk=lin(p.wk * half, "wk"),
        wr=lin(p.wr * half, "wr"),
        wv=lin(p.wv, "wv"),
        a=lane(p.a),
        b=lane(1.0 - p.a),
        mix=lane(p.mix),
        imix=lane(1.0 - p.mix),
        c=lane(p.c * half),
        dd=lane(p.dd * half),
        p0=lane(p.p0),
        p1=lane(p.p1),
        p2=lane(p.p2),
        q0=ffn(p.q0),
        q1=ffn(p.q1),
        q2=ffn(p.q2),
        q3=ffn(p.q3),
        r0=ffn(p.r0),
        r1=ffn(p.r1),
        r2=ffn(p.r2),
        norm_tm=p.norm_tm,
        norm_cm=p.norm_cm,
    )
