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


def batch_probe(P, ids, K: int, n_passes: int):
    """Public batched passes: pass i, lane c reads ``ids[s c + i]`` (each conversation its own window of the public
    text, stride s <= 7). Returns (the passes' row matrices (K, C), the token ids [[pass 0's K ids], ...])."""
    s = min(7, (len(ids) - n_passes) // max(1, K - 1))
    if s < 1:
        raise SystemExit(f"the batch probe text has {len(ids)} tokens, {K - 1 + n_passes} needed")
    toks = [[int(ids[s * c + i]) for c in range(K)] for i in range(n_passes)]
    return [[P.emb[t] for t in row] for row in toks], toks


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
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    P, co, _ = common.load(a)
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
        _fhe, _inf, model = make_model(P, co, capture=False, fuse_l1=a.fuse_l1, wbits=a.wbits, prefill_keys=T)
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
        _fhe, _inf, model = make_model(P, co, capture=False, fuse_l1=a.fuse_l1, wbits=a.wbits)
        shifts = {name: 0 for name in model.linears()}
    with open(os.path.join(a.out, "pcmm_shifts.json"), "w") as f:
        json.dump(shifts, f, indent=1, sort_keys=True)
    print(f"[capture] shifts ({a.pcmm_shift}) for {len(shifts)} linears in {time.time() - t0:.1f} s", flush=True)
    os.environ["FHE_GRAPH_CAPTURE_TOKEN"] = str(a.capture_tick)
    T = int(a.prefill_chunk)
    K = int(a.batch)
    fhe, _inf, model = make_model(P, co, capture=True, fuse_l1=a.fuse_l1, wbits=a.wbits, shifts=shifts, chunk=T,
                                  batch=K, batch_carry=a.batch_carry, batch_hoist=a.batch_hoist)
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
        print(f"[capture] {'pass' if T else 'token'} {pos} out level {model.rt.ops.level(out)} in "
              f"{time.time() - t1:.1f} s", flush=True)
    meta = {"blocks": model.n_layers, "capture_tick": a.capture_tick, "tokens": ids, "prefill_chunk": T,
            "batch": K, "batch_carry": a.batch_carry if K > 1 else None, "batch_hoist": a.batch_hoist if K > 1 else None,
            "prefix": a.prefix,
            "pth_sha256": common.sha256(a.pth), "coeffs_sha256": common.sha256(a.coeffs), "wbits": a.wbits,
            "fuse_l1": a.fuse_l1, "pcmm_shift": a.pcmm_shift, "environment": common.environment()}
    with open(os.path.join(a.out, "rwkv7_capture.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"[capture] wrote {a.out} ({model.n_layers} blocks + tail) in {time.time() - t0:.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
