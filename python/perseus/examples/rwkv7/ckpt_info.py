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

Two pod-3 device facts of 2026-11-05 (new_pod/RESULTS.md, "0.4B batched" and "0.4B prefill") enter the model:

* Device, more blocks in a batched session. The 0.4B batched K = 16 session (24 blocks, POOL_B 1,024, STC_CACHE_B 64)
  was refused at open: "memory: ct_pool = 33822867456: plan needs 103450379264 B, device has 101389893632 B free"
  (98,658.0 MiB needed, 96,692.9 MiB free; the pool alone 1,024 x 31.5 MiB). The model calibrated on the 12-block
  0.1B said 95,139 MiB: 3,519 MiB (3.6 %) short. The batched "other" therefore grows by 293.2 MiB per block beyond
  the 0.1B's 12 (3,519 / 12; a one-point fit: the replay slab and side buffers of a batched program scale with its
  blocks; the refusal names no item). The model reproduces the refusal's 98,658 MiB by construction.
* The presets' ceiling is the largest PLANNED total measured to run, 91,020 MiB (r01b_final/batch32_best: 91,019.6
  MiB planned, VRAM peak 94,858 MiB of 97,887, the sampler), not a percentage of the card: the runtime compares its
  plan with the free memory at session open (96,692.9 MiB on pod 3), and what the plan does not count (context,
  fragmentation) took the other ~3.8 GB.
* Host RAM (``host_gb``), a second budget: session setup (the twins' probe recording and ``Program.compile``) runs
  on the host before the device is touched, and on pod 3 (181 GB for the container; the sampler's total 186,252 MiB)
  it killed two 0.4B sessions with the GPU idle. Modelled on the sampler's peaks (monitor/usage_20261104.csv and the
  windows quoted in new_pod/RESULTS.md, MiB / 1,000 = GB):
    decode    0.1B 34.7-35.1 GB (12 blocks), 0.4B 64.1-64.2 GB (24 blocks): 6.07 GB + 2.42 GB per block;
    batched   0.1B K = 32 150.5 GB (12 blocks), 0.4B K = 16 160.9 GB (24 blocks): 6.07 GB + (0.868 + 0.349 K) GB
              per block, floored at decode's 2.42 (the fit's intercept is below it: an artefact of two points); the
              0.4B K = 32 session was OOM-killed at ~185.6 GB (the model: 295 GB);
    prefill   the session holds the decode programs and the T-token pass's: 6.07 + (2.42 + p(T)) GB per block, with
              p(T) = p32 T / 32 and p32 only bracketed: the 0.1B T = 32 session fitted (< 181 GB: p32 < 12.15) and
              the 0.4B T = 32 one was OOM-killed at ~184.9 GB (> 181: p32 > 4.87). So 0.4B T = 16 is 122-210 GB in
              the model (unsettled: the first run records it) and T = 8 93-137 GB.
  Per block the cost follows the graph's node count, which at one ciphertext per vector is width-independent (3,334
  nodes at 2,048 / t = 16 against 3,458 at the 0.1B, ANYSIZE_DESIGN.md section 3); it is scaled by the ciphertexts
  per vector S (shards double the nodes) and the batched per-conversation term by the state ciphertexts per
  conversation and block over the 0.1B's 2. Width 768 and 1,024 are confounded in the two batched anchors.
"""
from __future__ import annotations

import argparse
import json
import pickle
import zipfile

D = 1024
REG_MB, POOL_MB, STC_MB, KEYS_MB = 14.0, 31.5, 205.2, 8437.2
OTHER_MB = {"decode": 21430.0, "batched": 22530.0}   # total - (the items modelled here), final runs: see the docstring
OTHER_B_MB_PER_BLOCK = 293.2                          # batched, per block beyond 12: the 0.4B K = 16 refusal (docstring)
PT_MB_PER_BLOCK_SHARD = 205.3
CARD_MB = 97887.0
CAP_MB = 91020.0          # the largest planned total measured to run (r01b_final/batch32_best 91,019.6 MiB)
DEVICE_FREE_MB = 96692.9  # free at session open on pod 3 (the 0.4B K = 16 refusal's "device has ... B free")
# host RAM (GB), fitted on pod 3's sampler peaks (docstring)
HOST_BASE_GB, HOST_DEC_GB_PER_BLOCK = 6.07, 2.4225
HOST_B_GB_PER_BLOCK, HOST_B_GB_PER_CONV_BLOCK = 0.868, 0.3489
HOST_PF32_GB_PER_BLOCK = (4.87, 12.15)                 # bracket: 0.4B T = 32 killed, 0.1B T = 32 fitted
HOST_GB = 181.0           # pod 3's container
HOST_CAP_GB = 0.92 * HOST_GB                           # 166.5: the 0.4B K = 16 session ran at 160.9


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
             "keys": KEYS_MB, "other": OTHER_MB["batched" if K > 1 else "decode"] + 300.0 * (S - 1)
             + (OTHER_B_MB_PER_BLOCK * max(0, nl - 12) if K > 1 else 0.0)}
    items["total"] = sum(items.values())
    items["card"] = CARD_MB
    return {k: round(v, 1) for k, v in items.items()}


def host_gb(m: dict, *, K: int = 1, T: int = 0, blocks=None, lanes: int = 0) -> tuple[float, float]:
    """Host RAM (GB) of a session's setup (the docstring's model): decode (K = 1, T = 0), batched K, or prefill T
    (decode + prefill programs). Returns (low, high); only prefill has a range (p32 is bracketed)."""
    t = _lanes(m, lanes)
    S = -(-int(m["width"]) // (32768 // t))
    nl = int(blocks or m["n_layer"])
    conv = S * (64 // t) / 2.0                     # state cts per conversation and block over the 0.1B's 2
    if K > 1:
        # floored at decode's per-block cost: the two-point fit's intercept (0.868) is below it, an artefact (a
        # batched block's graph holds at least a decode block's)
        h = HOST_BASE_GB + nl * S * max(HOST_DEC_GB_PER_BLOCK, HOST_B_GB_PER_BLOCK + HOST_B_GB_PER_CONV_BLOCK * K * conv / S)
        return round(h, 1), round(h, 1)
    dec = HOST_BASE_GB + nl * S * HOST_DEC_GB_PER_BLOCK
    if not T:
        return round(dec, 1), round(dec, 1)
    return tuple(round(dec + nl * S * p * T / 32, 1) for p in HOST_PF32_GB_PER_BLOCK)


def pool_b(K: int, S: int) -> int:
    """The batched pool's floor: 0.1B at K = 32 exhausted 192 slots and ran at 1,024 (new_pod/RESULTS.md); taken
    proportional to the conversations and shards a pass holds temporaries for (an assumption, not a measurement).
    The preset then grows the pool up to 1,024 S slots while the device model fits (pool exhaustion is a hard stop)."""
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


def big(m: dict) -> bool:
    """0.4B and above (24+ blocks or wider than 1,024): pod 3 killed the 0.4B T = 32 prefill and K = 32 batched
    sessions on host RAM, so these sizes start at T <= 16 and K <= 16 (size_pipeline.sh; overridable)."""
    return int(m["n_layer"]) >= 24 or int(m["width"]) > 1024


def _preset(m: dict, lanes: int = 0) -> dict:
    """LANES / POOL / STC_CACHE / POOL_PF / PF_T / POOL_B / STC_CACHE_B / B_K for this size, under two budgets: the
    device model at most CAP_MB (the largest planned total measured to run) and the host model at most HOST_CAP_GB.
    Decode: the largest StC cache (<= 160) that fits. Batched: the largest K <= t (a power of two; <= 16 for ``big``
    sizes) whose host model fits and whose device model fits at the pool floor ``pool_b`` with an StC cache of 64
    slots (both measured batched runs used 64), else 32; at each K the 64-slot cache is tried first, so halving K
    restores it (review lens 3, finding 6); then the pool grows in steps of 32 up to 1,024 S slots while the device
    model fits (0.4B K = 16: 768, where 1,024 was refused on pod 3). Prefill: T = min(t, 32), 16 for ``big`` sizes."""
    t = _lanes(m, lanes)
    S = -(-int(m["width"]) // (32768 // t))
    K, stc_b = (min(t, 16) if big(m) else t), 64
    while K > 1:
        if host_gb(m, K=K, lanes=t)[1] <= HOST_CAP_GB:
            fit = [c for c in (64, 32) if memory_mb(m, K=K, pool=pool_b(K, S), stc=c, lanes=t)["total"] <= CAP_MB]
            if fit:
                stc_b = fit[0]
                break
        K //= 2
    if K <= 1:
        stc_b, pb = 64, 32
    else:
        pb = pool_b(K, S)
        while pb + 32 <= 1024 * S and memory_mb(m, K=K, pool=pb + 32, stc=stc_b, lanes=t)["total"] <= CAP_MB:
            pb += 32
    stc = 160
    while stc > 32 and memory_mb(m, pool=128 * S, stc=stc, lanes=t)["total"] > CAP_MB:
        stc -= 16
    pool = (224 if m["n_layer"] > 12 else 128) * S
    tot = memory_mb(m, pool=pool, stc=stc, lanes=t)["total"]
    pf_t = 16 if big(m) else min(t, 32)
    return {"LANES": t, "POOL": pool, "STC_CACHE": stc, "POOL_PF": 256 * S, "PF_T": min(pf_t, t), "POOL_B": pb,
            "STC_CACHE_B": stc_b, "B_K": K, "fits_one_card": tot <= CAP_MB, "memory_decode_mb": round(tot, 1),
            "host_decode_gb": host_gb(m, lanes=t)[1], "host_batched_gb": host_gb(m, K=K, lanes=t)[1] if K > 1 else None,
            "host_prefill_gb": list(host_gb(m, T=min(pf_t, t), lanes=t)), "host_cap_gb": round(HOST_CAP_GB, 1)}


def info(path: str, lanes: int = 0) -> dict:
    m = model_shapes(shapes(path))
    p = preset(m, lanes)
    t = p["LANES"]
    Dc = 32768 // t
    S = -(-int(m["width"]) // Dc)
    return {**m, "lanes": t, "auto_lanes": _lanes(m, 0), "channels_per_ct": Dc, "shards": S, "heads_per_shard": Dc // m["head_size"],
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
