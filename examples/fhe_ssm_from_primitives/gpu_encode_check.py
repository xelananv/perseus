# SPDX-License-Identifier: MIT
"""Key-free check of the device encoder route from Python.

A session with bootstrap off and no rotation keys (about 2 GB, so it runs beside another job),
one or more random d x d tiles, the diagonals of a giant index made on the device by
``encode_group_gpu`` and multiplied into one encrypted vector with ``mult_ptx``, against the CPU
route's ``mult_pt`` of the same diagonals, both decrypted and compared with numpy. Exercises the
batch shapes the model uses: B = 32 (one tile), 128 (four tiles sharing the babies, W_v's down
chunks) and 256 (eight tiles, W_k + W_r's up tiles).

    python -m examples.fhe_ssm_from_primitives.gpu_encode_check [--d 1024] [--out FILE]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

from . import filling
from .gpu_smoke import SMOKE_ENV
from .ref import rel_err_rms


def open_plain_session(d: int, *, logn: int = 16, depth: int = 26):
    """The n32 session of gpu_smoke.open_session without bootstrap keys or rotation keys."""
    for k, v in SMOKE_ENV.items():
        os.environ.setdefault(k, v)
    from perseus import _backend, session
    from perseus.profile import SessionProfile

    prof = SessionProfile.custom_n32()
    prof.logN = logn
    prof.ckks_complex = True
    prof.apply()
    opts = prof.options()
    opts.dim = opts.hidDim = d
    opts.expanded = opts.expDim = 4 * d
    opts.numHeads = opts.numHeadsReal = 16
    opts.packing_kind = _backend.default().PackingKind.Diagonal
    ck = opts.ckks
    ck.logN = logn
    ck.ckks_complex_payload = True
    ck.enable_bootstrap = False
    ck.depth = depth
    ck.h_weight = 0
    ck.extra_rot_steps = []
    opts.ckks = ck
    return session(options=opts, family="generic", mode="sync")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--d", type=int, default=1024)
    ap.add_argument("--level", type=int, default=34, help="the input ciphertext's level")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    from perseus import _core
    from perseus.impl.ops import FheOps

    t0 = time.time()
    rng = np.random.default_rng(a.seed)
    report: dict = {"d": a.d, "level": a.level, "cells": []}
    with open_plain_session(a.d) as sess:
        inf = sess.inf
        ops = FheOps(inf, _core, unit=2)
        N, d = inf.slots, a.d
        t, s, G = filling.tile_params(N, d)
        report.update({"N": N, "t": t, "s": s, "G": G, "gpu_encode": bool(ops.gpu_encode),
                       "open_seconds": round(time.time() - t0, 1)})
        print(json.dumps({k: v for k, v in report.items() if k != "cells"}), flush=True)
        if not ops.gpu_encode:
            raise SystemExit("the session has no device encoder")

        rows = rng.uniform(-0.5, 0.5, (t, d))
        x = ops.pack_tokens(rows, a.level)
        lx = ops.level(x)
        if not np.allclose(filling.unpack(ops.decrypt_slots(x), d, t), rows, atol=1e-6):
            raise SystemExit("round trip of the input failed")

        def one_group(tiles_W, g, babies_checked):
            """tiles_W: the tiles sharing this group's babies (B = len * s)."""
            devs = [filling.EncodedTile.encode(W, N, f"t{i}", dev_ops=ops) for i, W in enumerate(tiles_W)]
            cpus = [filling.EncodedTile.encode(W, N, f"c{i}") for i, W in enumerate(tiles_W)]
            ids = [w.dev for w in devs for _ in range(s)]
            baby = [b for _ in devs for b in range(s)]
            names = [w.key(g, b) for w in devs for b in range(s)]
            t1 = time.time()
            pts = ops.encode_group(ids, baby, g, s, x, names)
            enc_s = time.time() - t1
            cell = {"B": len(ids), "g": g, "encode_wall_s": round(enc_s, 4), "pt_level": int(pts[0].level),
                    "ct_level": lx, "checks": []}
            worst = {"rel_dev": 0.0, "rel_cpu": 0.0, "max_dev_minus_cpu": 0.0, "level_mismatch": 0}
            try:
                for i, (W, wc) in enumerate(zip(tiles_W, cpus)):
                    for b in babies_checked:
                        y_dev = ops.mult_ptx(x, pts[i * s + b])
                        y_cpu = ops.mult_pt(x, wc.pt(g, b), key=wc.key(g, b))
                        r_dev = filling.unpack(ops.decrypt_slots(y_dev), d, t)
                        r_cpu = filling.unpack(ops.decrypt_slots(y_cpu), d, t)
                        # diagonal (g, b) is pre-rotated by g*s*t slots = g*s channels (filling._Diagonal):
                        # the slots of channel i' carry channel i = (i' - g*s) mod d's weight W[(i+g*s+b) mod d][i]
                        j = np.arange(d)
                        diag = np.roll(W[(j + g * s + b) % d, j], g * s)
                        want = rows * diag[None, :]
                        c = {"tile": i, "b": b,
                             "rel_dev_vs_exact": rel_err_rms(r_dev, want),
                             "rel_cpu_vs_exact": rel_err_rms(r_cpu, want),
                             "max_abs_dev_minus_cpu": float(np.max(np.abs(r_dev - r_cpu))),
                             "max_abs_exact": float(np.max(np.abs(want))),
                             "level_dev": ops.level(y_dev), "level_cpu": ops.level(y_cpu)}
                        cell["checks"].append(c)
                        worst["rel_dev"] = max(worst["rel_dev"], c["rel_dev_vs_exact"])
                        worst["rel_cpu"] = max(worst["rel_cpu"], c["rel_cpu_vs_exact"])
                        worst["max_dev_minus_cpu"] = max(worst["max_dev_minus_cpu"], c["max_abs_dev_minus_cpu"])
                        worst["level_mismatch"] += int(c["level_dev"] != c["level_cpu"])
            finally:
                for pt in pts:
                    ops.evict_ptx(pt)
            cell["worst"] = worst
            cell["pass"] = (worst["max_dev_minus_cpu"] <= 1e-9 and worst["rel_dev"] <= 1e-8
                            and worst["level_mismatch"] == 0)
            print(json.dumps({k: v for k, v in cell.items() if k != "checks"}), flush=True)
            report["cells"].append(cell)
            return cell["pass"]

        ok = True
        W1 = [rng.uniform(-0.3, 0.3, (d, d))]
        for g in (0, 3, 31):
            ok &= one_group(W1, g, range(s) if g == 3 else (0, 7, 31))
        W4 = [rng.uniform(-0.3, 0.3, (d, d)) for _ in range(4)]
        ok &= one_group(W4, 5, (0, 31))
        W8 = [rng.uniform(-0.3, 0.3, (d, d)) for _ in range(8)]
        ok &= one_group(W8, 17, (0, 15))
        report["stats"] = dict(_core.gpu_encode_stats(inf))
        _core.release_tiles(inf)
    report["pass"] = bool(ok)
    report["total_seconds"] = round(time.time() - t0, 1)
    print(json.dumps({"stats": report["stats"], "pass": report["pass"], "total_seconds": report["total_seconds"]}), flush=True)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1, default=str)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
