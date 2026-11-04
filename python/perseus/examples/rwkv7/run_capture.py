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

from . import common
from .session import make_model


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    common.add_args(ap)
    ap.add_argument("--out", required=True)
    ap.add_argument("--capture-tick", type=int, default=1)
    ap.add_argument("--calib-tokens", type=int, default=6)
    ap.add_argument("--pcmm-shift", choices=("calib", "zero"), default="calib")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    P, co, _ = common.load(a)
    ids = common.token_ids(a, common.CALIB_TEXT)
    _rf, st = common.prefix_state(P, a.coeffs, ids, a.prefix)
    ids = ids[a.prefix:]
    t0 = time.time()
    shifts = {}
    if a.pcmm_shift == "calib":
        calib = os.path.join(a.out, "pcmm_calib.json")
        os.environ["PF_PCMM_CALIB"] = calib
        _fhe, _inf, model = make_model(P, co, capture=False, fuse_l1=a.fuse_l1, wbits=a.wbits)
        model.start(st if a.prefix else None)
        for pos, t in enumerate(ids[: a.calib_tokens]):
            model.step(P.emb[t], pos)
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
    fhe, _inf, model = make_model(P, co, capture=True, fuse_l1=a.fuse_l1, wbits=a.wbits, shifts=shifts)
    model.set_capture(a.out)
    model.start(st if a.prefix else None)
    for pos, t in enumerate(ids[: a.capture_tick + 1]):
        t1 = time.time()
        h, out = model.step(P.emb[t], pos)
        print(f"[capture] token {pos} ({t}) out level {model.rt.ops.level(out)} in {time.time() - t1:.1f} s",
              flush=True)
    meta = {"blocks": model.n_layers, "capture_tick": a.capture_tick, "tokens": ids[: a.capture_tick + 1],
            "prefix": a.prefix,
            "pth_sha256": common.sha256(a.pth), "coeffs_sha256": common.sha256(a.coeffs), "wbits": a.wbits,
            "fuse_l1": a.fuse_l1, "pcmm_shift": a.pcmm_shift, "environment": common.environment()}
    with open(os.path.join(a.out, "rwkv7_capture.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"[capture] wrote {a.out} ({model.n_layers} blocks + tail) in {time.time() - t0:.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
