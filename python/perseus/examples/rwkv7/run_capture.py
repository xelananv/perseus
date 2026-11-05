# SPDX-License-Identifier: MIT
"""CPU only: (1) the PCMM band shifts, from the fake run on a public calibration text (``max |y|`` per linear,
``impl.pcmm.band_shift``'s rule), and (2) the capture the planner cuts the refresh plan from: the model on the
capture recorder (values on, so the graph carries magnitudes) for tokens 0..CAPTURE_TICK of the same text, the
graphs of token CAPTURE_TICK (it reads a carried state) written as ``block_<b>/graph.json`` with
``capture_config.json`` / ``capture_env.json``. Run it under the plan's runtime switches (run_box.sh exports them).

    python -m perseus.examples.rwkv7.run_capture --pth P --coeffs C --harness H --out DIR [--blocks N]
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from . import common
from .session import make_model


def model_kw(a) -> dict:
    """The any-size switches every model of a run shares (capture, twin, mirror)."""
    return {"lora": getattr(a, "lora", "auto"), "shard_land": not getattr(a, "no_shard_land", False),
            "lanes": getattr(a, "lanes", 0), "layout": getattr(a, "layout", "current")}


def _ckpt_depth(path):
    """The checkpoint's own depth (its blocks.N keys, read without loading tensors), whatever --blocks loaded."""
    try:
        from .ckpt_info import model_shapes, shapes
        return int(model_shapes(shapes(path))["n_layer"])
    except Exception:                                   # noqa: BLE001 (a format the shape reader does not know)
        return None


def batch_probe(P, ids, K: int, n_passes: int):
    """Public batched passes: pass i, lane c reads ``ids[s c + i]`` (each conversation its own window of the public
    text, stride s <= 7). Returns (the passes' row matrices (K, C), the token ids [[pass 0's K ids], ...])."""
    s = min(7, (len(ids) - n_passes) // max(1, K - 1))
    if s < 1:
        raise SystemExit(f"the batch probe text has {len(ids)} tokens, {K - 1 + n_passes} needed")
    toks = [[int(ids[s * c + i]) for c in range(K)] for i in range(n_passes)]
    return [[P.emb[t] for t in row] for row in toks], toks


def join_probe(P, coeffs, a, L, n_layers):
    """The join twin's rows (public): the lane-form carries of the public prefix state (every lane equal, as a fresh
    batch holds them) and the natural-form carries after the public calibration text (a conversation's)."""
    from .serve import join_rows
    ids = common.token_ids(a, common.CALIB_TEXT)
    _rf, lane_st = common.prefix_state(P, coeffs, ids, a.prefix)
    _rf, new_st = common.prefix_state(P, coeffs, ids, len(ids))
    return join_rows(L, n_layers, lane_st, new_st)


def capture_join(a, P) -> int:
    """``--join``: the join program's capture (``block_0/graph.json``: one stage, no weights, no rotations)."""
    from .serve import make_join
    K = int(a.batch)
    if K < 2:
        raise SystemExit("--join needs --batch K >= 2 (the batch the conversations join)")
    t0 = time.time()
    os.environ["FHE_GRAPH_CAPTURE_TOKEN"] = str(a.capture_tick)
    _fhe, _inf, jm = make_join(P, K, capture=True, lanes=a.lanes)
    jm.set_capture(a.out)
    rows = join_probe(P, a.coeffs, a, jm.L, jm.n_layers)
    for pos in range(a.capture_tick + 1):
        out = jm.step(rows, pos)
        print(f"[capture] join tick {pos}: {len(out)} carries out at level {jm.rt.ops.level(out[0])}", flush=True)
    meta = {"kind": "join", "blocks": jm.n_layers, "batch": K, "lane": jm.lane, "capture_tick": a.capture_tick,
            "keys": jm.keys, "new_keys": jm.new_keys, "prefix": a.prefix, "pth_sha256": common.sha256(a.pth),
            "coeffs_sha256": common.sha256(a.coeffs), "width": P.C, "shards": jm.L.S, "lanes": jm.L.t,
            "environment": common.environment()}
    with open(os.path.join(a.out, "rwkv7_capture.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"[capture] wrote the join capture {a.out} ({jm.n_layers} blocks, K = {K}) in {time.time() - t0:.1f} s",
          flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    common.add_args(ap)
    ap.add_argument("--out", required=True)
    ap.add_argument("--capture-tick", type=int, default=1)
    ap.add_argument("--calib-tokens", type=int, default=6)
    ap.add_argument("--pcmm-shift", choices=("calib", "zero"), default="calib")
    ap.add_argument("--no-site-guard", action="store_true",
                    help="do not stop on a site input out of its interval / non-finite (model.SiteGuard)")
    ap.add_argument("--site-guard-tol", type=float, default=0.0,
                    help="the guard's slack, relative to each interval's width (default 0)")
    ap.add_argument("--join", action="store_true",
                    help="capture the JOIN program instead (serve.py, SERVE_DESIGN.md): every block's lane-form carries "
                    "joined with one conversation's natural-form carries on lane 0 of a --batch K pass, one refresh each")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    P, co, _ = common.load(a)
    if a.join:
        return capture_join(a, P)
    ids = common.token_ids(a, common.CALIB_TEXT)
    _rf, st = common.prefix_state(P, a.coeffs, ids, a.prefix)
    ids = ids[a.prefix:]
    t0 = time.time()
    shifts = {}
    T = int(a.prefill_chunk)
    if a.shifts:
        # given (a session serving decode AND prefill uploads each linear once, with one shift: both captures read
        # the same file)
        shifts = json.load(open(a.shifts))
    elif a.pcmm_shift == "calib":
        calib = os.path.join(a.out, "pcmm_calib.json")
        os.environ["PF_PCMM_CALIB"] = calib
        _fhe, _inf, model = make_model(P, co, capture=False, fuse_l1=a.fuse_l1, wbits=a.wbits, prefill_keys=T,
                                       **model_kw(a))
        model.start(st if a.prefix else None)
        for pos, t in enumerate(ids[: a.calib_tokens]):
            model.step(P.emb[t], pos)
        if T:
            # prefill: the band must also hold every token of the long text, read as two T-token passes
            long_ids = common.token_ids(a, common.CALIB_TEXT_LONG)[a.prefix:]
            model.start(st if a.prefix else None)
            model.set_chunk(T)
            for i in range(min(2, len(long_ids) // T)):
                model.step([P.emb[t] for t in long_ids[i * T:(i + 1) * T]], i)
        del os.environ["PF_PCMM_CALIB"]
        from .model import pcmm_shifts
        shifts = pcmm_shifts(json.load(open(calib)))
    else:
        _fhe, _inf, model = make_model(P, co, capture=False, fuse_l1=a.fuse_l1, wbits=a.wbits, **model_kw(a))
        shifts = {name: 0 for name in model.linears()}
    with open(os.path.join(a.out, "pcmm_shifts.json"), "w") as f:
        json.dump(shifts, f, indent=1, sort_keys=True)
    print(f"[capture] shifts ({a.pcmm_shift}) for {len(shifts)} linears in {time.time() - t0:.1f} s", flush=True)
    os.environ["FHE_GRAPH_CAPTURE_TOKEN"] = str(a.capture_tick)
    T = int(a.prefill_chunk)
    K = int(a.batch)
    fhe, _inf, model = make_model(P, co, capture=True, fuse_l1=a.fuse_l1, wbits=a.wbits, shifts=shifts, chunk=T,
                                  batch=K, batch_carry=a.batch_carry, batch_hoist=a.batch_hoist, **model_kw(a))
    model.set_capture(a.out)
    guard = None
    if not a.no_site_guard:
        # fail fast: a site input out of its interval (or NaN) stops here, not in the planner 25 minutes later
        from .model import SiteGuard
        guard = SiteGuard(model.rt.ops, model.L, chunk=T, tol=a.site_guard_tol)
        model.circ.guard = model.decode_circ.guard = guard
    model.start(st if a.prefix else None)
    if K > 1:
        # a K-conversation pass per tick: conversation c reads its own window of the long public text
        steps, ids = batch_probe(P, common.token_ids(a, common.CALIB_TEXT_LONG)[a.prefix:], K, a.capture_tick + 1)
    elif T:
        # a T-token pass per tick, on the long public text (tick 1 reads the state tick 0 carried)
        long_ids = common.token_ids(a, common.CALIB_TEXT_LONG)[a.prefix:]
        if len(long_ids) < T * (a.capture_tick + 1):
            raise SystemExit(f"the prefill capture text has {len(long_ids)} tokens, {T * (a.capture_tick + 1)} needed")
        steps = [[P.emb[t] for t in long_ids[i * T:(i + 1) * T]] for i in range(a.capture_tick + 1)]
        ids = long_ids[: T * (a.capture_tick + 1)]
    else:
        steps = [P.emb[t] for t in ids[: a.capture_tick + 1]]
        ids = ids[: a.capture_tick + 1]
    for pos, row in enumerate(steps):
        t1 = time.time()
        if guard is not None:
            guard.pos = pos
        h, out = model.step(row, pos)
        if not np.all(np.isfinite(h)):
            raise SystemExit(f"[capture] {'pass' if T else 'token'} {pos}: non-finite hidden state")
        print(f"[capture] {'pass' if T else 'token'} {pos} out level {model.rt.ops.level(out[0])} in "
              f"{time.time() - t1:.1f} s", flush=True)
    meta = {"blocks": model.n_layers, "capture_tick": a.capture_tick, "tokens": ids, "prefill_chunk": T,
            "batch": K, "batch_carry": a.batch_carry if K > 1 else None, "batch_hoist": a.batch_hoist if K > 1 else None,
            "prefix": a.prefix,
            "pth_sha256": common.sha256(a.pth), "coeffs_sha256": common.sha256(a.coeffs), "wbits": a.wbits,
            "fuse_l1": a.fuse_l1, "pcmm_shift": a.pcmm_shift, "width": P.C, "blocks_loaded": P.n_layer, "n_layer_ckpt": _ckpt_depth(a.pth),
            "shards": model.L.S, "lanes": model.L.t, "channels_per_ct": model.L.D, "layout": model.L.layout, "lora": model.W[0].lora if model.W else None, "shard_land": model.shard_land,
            "state_names": model.state_names(), "carry_drop": model.carry_drop, "state_entry": model.state_entry,
            "environment": common.environment()}
    with open(os.path.join(a.out, "rwkv7_capture.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"[capture] wrote {a.out} ({model.n_layers} blocks + tail) in {time.time() - t0:.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
