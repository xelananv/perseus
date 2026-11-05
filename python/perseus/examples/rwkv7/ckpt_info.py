# SPDX-License-Identifier: MIT
"""A checkpoint's shapes without loading its tensors (the pickle's storage records only), the channel-shard layout
they imply, and the device-memory model of ANYSIZE_DESIGN.md §4 with the run preset for that size.

    python -m perseus.examples.rwkv7.ckpt_info M.pth [--out info.json]

The memory model is transcribed from the pod-3 session reports of the FINAL-coefficient runs (``mem_report_text`` of
results/rwkv7-fhe-20261104/new_pod/r01b_final/run_best_P1.jsonl (total 70,980.0 MB), r04b_final/run_best_P1.jsonl
(77,844.9), r01b_final/batch32_best.jsonl (91,019.6); recalibrated after review lens 3, finding 5: the older
r01b / r04b / r01b_batch reports had a smaller plaintext store and replay slab, and the model calibrated on them
under-predicted every final session by about 2 %): a carried ciphertext (register) 14.0 MB, a pool slot 31.5 MB, an
StC-first cache slot 205.2 MB, keys 8,437.2 MB (43 rotation keys + the refresh keys); the PCMM planes are exact
(24-bit weights = 3 int8 planes per 1,024 x 1,024 block: 3 MiB); the plaintext store 205.3 MB per block and shard
(0.1B final decode 2,463.5 / 12; the 0.4B's 3,570.0 / 24 = 148.75 and the batched 2,245.0 / 12 = 187.1 are smaller, so
the store is an upper value); "other" (workspaces, the dense bootstrap tables, replay slabs and side buffers, PCMM
scratch) = the report's total minus the items above: decode 21,428.6 (0.1B) / 20,813.9 (0.4B, at 205.3 MB per block),
batched 22,531.0, taken as 21,430 / 22,530 (the larger, so the model reproduces 0.1B decode and batched to +0.0 % and
0.4B decode to +0.8 %). The card: 97,887 MiB (RTX PRO 6000 Workstation, new_pod/RESULTS.md).
"""
from __future__ import annotations

import argparse
import json
import pickle
import zipfile

D = 1024
REG_MB, POOL_MB, STC_MB, KEYS_MB = 14.0, 31.5, 205.2, 8437.2
OTHER_MB = {"decode": 21430.0, "batched": 22530.0}   # total - (the items modelled here), final runs: see the docstring
PT_MB_PER_BLOCK_SHARD = 205.3
CARD_MB = 97887.0


class _Shape:
    def __init__(self, size):
        self.shape = tuple(int(s) for s in size)


def _rebuild(_storage, _offset, size, *_a, **_k):
    return _Shape(size)


class _ShapeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return _rebuild
        if module == "torch._utils" and name == "_rebuild_parameter":
            return lambda data, *_a, **_k: data
        if module == "collections" and name == "OrderedDict":
            import collections
            return collections.OrderedDict
        if module == "torch":
            return name
        raise pickle.UnpicklingError(f"unsupported global {module}.{name}")

    def persistent_load(self, pid):
        return None


def shapes(path: str) -> dict[str, tuple]:
    """{tensor name: shape} of a torch.save'd state dict (or an .npz), reading no tensor data."""
    if str(path).endswith(".npz"):
        import numpy as np
        with np.load(path, mmap_mode="r") as z:
            return {k: tuple(z[k].shape) for k in z.files}
    with zipfile.ZipFile(path) as zf:
        pkl = next(n for n in zf.namelist() if n.endswith("data.pkl"))
        with zf.open(pkl) as f:
            sd = _ShapeUnpickler(f).load()
    return {k: v.shape for k, v in sd.items()}


def model_shapes(sh: dict) -> dict:
    n_layer = 1 + max(int(k.split(".")[1]) for k in sh if k.startswith("blocks."))
    vocab, C = sh["emb.weight"]
    H, N = sh["blocks.0.att.r_k"]
    return {"n_layer": n_layer, "width": C, "heads": H, "head_size": N, "vocab": vocab,
            "d_ffn": sh["blocks.0.ffn.key.weight"][0],
            "lora": {"w": sh["blocks.0.att.w1"][1], "a": sh["blocks.0.att.a1"][1], "g": sh["blocks.0.att.g1"][1],
                     "v": sh["blocks.1.att.v1"][1] if "blocks.1.att.v1" in sh else None}}


def _lanes(m: dict, lanes: int = 0) -> int:
    from .layout import auto_lanes
    return int(lanes) or auto_lanes(int(m["width"]))


def plane_blocks(m: dict, b: int, lora: str = "auto", D: int = D) -> int:
    """PCMM blocks of D x D (D channels per ciphertext) of block b (model.LayerW's linears)."""
    S, F = -(-int(m["width"]) // D), -(-int(m["d_ffn"]) // D)
    fold = (S == 1) if lora == "auto" else lora == "fold"
    r = {k: -(-int(v) // D) for k, v in m["lora"].items() if v}
    lr = 2 * S * S if fold else 2 * S * r["a"] + r["a"] * S          # a (and v)
    n = 3 * (2 * S * S) + lr + (2 * S * r["w"] + r["w"] * S) + (2 * S * r["g"] + r["g"] * S) + S * S
    n += 2 * S * F + F * S
    if b > 0:
        n += 2 * S * S if fold else 2 * S * r["v"] + r["v"] * S
    return n


def memory_mb(m: dict, *, K: int = 1, pool: int = 128, stc: int = 160, lora: str = "auto", blocks=None,
              lanes: int = 0) -> dict:
    """The session's device memory (MB) by item, decode (K = 1) or batched K (lane carries), at ``lanes`` (0: the
    default layout, ``layout.auto_lanes``)."""
    t = _lanes(m, lanes)
    Dc = 32768 // t
    S = -(-int(m["width"]) // Dc)
    nl = int(blocks or m["n_layer"])
    M = 64 // t
    per_block = (S * M * K + 2 * S + S) if K > 1 else (S * M + 2 * S + S)
    regs = (nl * per_block + S) * REG_MB
    # 3 int8 planes per D x D block; the reports' "MB" are MiB (0.1B: 310 blocks of 1,024^2 = 930.0)
    planes = sum(plane_blocks(m, b, lora, Dc) for b in range(nl)) * 3.0 * (Dc // D) ** 2
    pt = nl * S * PT_MB_PER_BLOCK_SHARD
    items = {"registers": regs, "pcmm_planes": planes, "pt_store": pt, "pool": pool * POOL_MB, "stc_cache": stc * STC_MB,
             "keys": KEYS_MB, "other": OTHER_MB["batched" if K > 1 else "decode"] + 300.0 * (S - 1)}
    items["total"] = sum(items.values())
    items["card"] = CARD_MB
    return {k: round(v, 1) for k, v in items.items()}


def pool_b(K: int, S: int) -> int:
    """Pool slots for a batched pass: 0.1B at K = 32 exhausted 192 and ran at 1,024 (new_pod/RESULTS.md); taken
    proportional to the conversations and shards a pass holds temporaries for (an assumption, not a measurement)."""
    return min(2048, 32 * K * S)


def preset(m: dict, lanes: int = 0) -> dict:
    """The run preset; ``lanes`` 0 = the default layout, unless it does not fit one card and another layout does
    (wider ciphertexts pad the LoRA ranks and d_ffn to whole D-channel blocks: at 2,560 / 4,096 the t = 8 planes are
    1.2-2.2x the sharded ones): the first that fits of (auto, 16, 32), else the smallest."""
    if lanes:
        return _preset(m, lanes)
    t0 = _lanes(m, 0)
    cands = [_preset(m, t) for t in (t0, 16, 32) if t >= t0]
    fit = [c for c in cands if c["fits_one_card"]]
    return fit[0] if fit else min(cands, key=lambda c: c["memory_decode_mb"])


def _preset(m: dict, lanes: int = 0) -> dict:
    """LANES / POOL / STC_CACHE / POOL_PF / POOL_B / STC_CACHE_B / B_K for this size: the measured 0.1B / 0.4B settings
    at one ciphertext per vector of 1,024 channels; above, the largest StC cache (<= 160) and K <= t (a power of two)
    that the memory model fits under 92 % of the card. Batched: the largest K that fits with an StC cache of 64
    slots (both measured batched runs used 64), else of 32; at each K the 64-slot cache is tried first, so halving K
    restores it (review lens 3, finding 6)."""
    t = _lanes(m, lanes)
    S = -(-int(m["width"]) // (32768 // t))
    cap = 0.92 * CARD_MB
    K, stc_b = t, 64
    while K > 1:
        fit = [c for c in (64, 32) if memory_mb(m, K=K, pool=pool_b(K, S), stc=c, lanes=t)["total"] <= cap]
        if fit:
            stc_b = fit[0]
            break
        K //= 2
    if K <= 1:
        stc_b = 64
    stc = 160
    while stc > 32 and memory_mb(m, pool=128 * S, stc=stc, lanes=t)["total"] > cap:
        stc -= 16
    big = m["n_layer"] > 12
    pool = (224 if big else 128) * S
    tot = memory_mb(m, pool=pool, stc=stc, lanes=t)["total"]
    return {"LANES": t, "POOL": pool, "STC_CACHE": stc, "POOL_PF": 256 * S, "POOL_B": pool_b(K, S), "STC_CACHE_B": stc_b,
            "B_K": K, "fits_one_card": tot <= cap, "memory_decode_mb": round(tot, 1)}


def info(path: str, lanes: int = 0) -> dict:
    m = model_shapes(shapes(path))
    p = preset(m, lanes)
    t = p["LANES"]
    Dc = 32768 // t
    S = -(-int(m["width"]) // Dc)
    return {**m, "lanes": t, "channels_per_ct": Dc, "shards": S, "heads_per_shard": Dc // m["head_size"],
            "state_cts_per_block": S * (64 // t), "preset": p,
            "memory_decode_mb": memory_mb(m, pool=p["POOL"], stc=p["STC_CACHE"], lanes=t),
            "memory_batched_mb": memory_mb(m, K=p["B_K"], pool=p["POOL_B"], stc=p["STC_CACHE_B"], lanes=t)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pth")
    ap.add_argument("--lanes", type=int, default=0, help="0: the default layout (layout.auto_lanes)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    d = info(a.pth, a.lanes)
    text = json.dumps(d, indent=1)
    print(text)
    if a.out:
        open(a.out, "w").write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
