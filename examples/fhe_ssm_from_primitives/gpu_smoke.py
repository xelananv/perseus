# SPDX-License-Identifier: MIT
"""A smoke test of the port on a live session: open a Perseus session sized for this model
(d = 1,024, d_ffn = 4,096, ring 2^16, complex slots, the filled layout's rotation keys) and run
three checks on it, each recorded even when an earlier one failed:

    roundtrip   every conversation encrypted and decrypted: the encryption noise
    linear      one 1024 x 1024 tile (block 0's W_in) on them, against numpy
    ticks       the model (``--blocks`` blocks) for two ticks, against the mirror (the same
                kernels on numpy) and the trainer's closed-form forward: the reply's error,
                the logits' error and top-1 agreement on the tied head (the published run's
                metrics), and the tick time scaled to 24 blocks (eager, unplanned)

    source scripts/local_env.sh      # the n32 chain's defaults
    python -m examples.fhe_ssm_from_primitives.gpu_smoke --weights W --seeds S --config C \\
        [--blocks 1] [--payload pairs|real] [--pair-route split|lane] [--out gpu_smoke.json]

On a session the errors are CKKS noise; on the fake (``--fake``: the same checks, no GPU) they
are rounding, 1e-12 and below. ``run_checks`` takes any session, so the tests run it on the fake;
``open_session`` needs the CUDA extension and runs only on a GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

import numpy as np

from . import filling
from .model import PAYLOADS, FheSsmModel
from .ref import ExactModel, MirrorModel, rel_err_rms

#: exported before the extension loads: small host arenas (this port streams no blocks and keeps
#: no K/V cache), complex slots; every other value comes from scripts/local_env.sh
SMOKE_ENV = {"CKKS_COMPLEX": "1", "FHE_STAGE_ARENA_GB": "2", "KV_ARENA_GB": "1"}


def _free_gb(core):
    f = getattr(core, "device_free_gb", None)
    try:
        return None if f is None else round(float(f()), 2)
    except (RuntimeError, TypeError):
        return None


def run_checks(inf, core, params, *, ticks=2, payload="pairs", route="split", seed=3, log=print):
    """The three checks on session ``inf`` (``core``: its data-plane module). Returns a dict
    with one entry per check; a failed check records its error and the rest still run."""
    model = FheSsmModel(
        inf, params, core=core, payload=payload, pair_route=route, resident_weights=False
    )
    missing = model.rotation_audit()
    results: dict = {"rotation_audit": {"n_missing": len(missing), "missing": missing[:16]}}
    log(json.dumps(results))
    if missing:
        return results
    geo, n = model.geo, model.capacity
    rng = np.random.default_rng(seed)
    rows = [
        np.asarray(params.emb[rng.integers(0, params.vocab, n)], dtype=np.float64)
        for _ in range(ticks)
    ]

    def roundtrip():
        ct = geo.encode_rows(rows[0])
        return {"relErrRms": rel_err_rms(geo.decode_rows(ct, n), rows[0]), "level": ct.level}

    def linear():
        w = params.block(0).win
        y = geo.linear(geo.encode_rows(rows[0]), geo.weight(w, None))
        return {"relErrRms": rel_err_rms(geo.decode_rows(y, n), rows[0] @ w)}

    def tick_loop():
        mirror, exact = MirrorModel(params, n, model.form), ExactModel(params, n)
        emb = np.asarray(params.emb, dtype=np.float64)
        model.start()
        per = []
        for pos in range(ticks):
            t0 = time.time()
            rec = model.step(rows[pos], pos)
            seconds = time.time() - t0
            want = exact.step(rows[pos])
            got_logits, want_logits = rec["out"] @ emb.T, want @ emb.T
            per.append(
                {
                    "pos": pos,
                    "seconds": round(seconds, 2),
                    "projected_full_tick_s": round(seconds * 24 / params.n_layers, 1),
                    "relErrRms_vs_mirror": rel_err_rms(rec["out"], mirror.step(rows[pos])),
                    "relErrRms_vs_exact": rel_err_rms(rec["out"], want),
                    "logits_relErrRms_vs_exact": rel_err_rms(got_logits, want_logits),
                    "top1_agree_vs_exact": float(
                        np.mean(got_logits.argmax(axis=1) == want_logits.argmax(axis=1))
                    ),
                    "out_level": rec["out_level"],
                    "bootstraps_total": model.rt.ops.n_bootstraps,
                }
            )
        return {"ticks": per}

    for name, fn in (("roundtrip", roundtrip), ("linear", linear), ("ticks", tick_loop)):
        free0, t0 = _free_gb(core), time.time()
        try:
            res = {"ok": True, **fn()}
        except Exception as e:  # noqa: BLE001 (a smoke test records every failure and goes on)
            res = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            res["trace"] = traceback.format_exc()[-3000:]
        res.update(
            seconds=round(time.time() - t0, 2), free_gb_before=free0, free_gb_after=_free_gb(core)
        )
        results[name] = res
        log(json.dumps({name: {k: v for k, v in res.items() if k != "trace"}}))
    return results


def open_session(params, *, logn=16, auto_bts_level=46):
    """A Perseus session for this model on the n32 chain: the model's own sizes (no padding),
    complex slots, and the filled layout's rotation steps as extra keys. Needs the CUDA
    extension (perseus._core); a context manager (``with open_session(...) as s:``)."""
    for k, v in SMOKE_ENV.items():
        os.environ.setdefault(k, v)
    from perseus import session
    from perseus.profile import SessionProfile

    prof = SessionProfile.custom_n32()
    prof.logN = logn
    prof.ckks_complex = True
    prof.auto_bts_level = auto_bts_level
    prof.apply()
    opts = prof.options()
    opts.dim = opts.hidDim = params.d
    opts.expanded = opts.expDim = params.d_ffn
    opts.numHeads = opts.numHeadsReal = 16
    ck = opts.ckks
    ck.logN = logn
    ck.ckks_complex_payload = True
    ck.extra_rot_steps = sorted(filling.rotation_steps(1 << (logn - 1), params.d))
    opts.ckks = ck
    return session(options=opts, family="generic", mode="sync")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--weights", required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--blocks", type=int, default=1)
    ap.add_argument("--ticks", type=int, default=2)
    ap.add_argument("--payload", default="pairs", choices=PAYLOADS)
    ap.add_argument("--pair-route", default="split", choices=("split", "lane"))
    ap.add_argument("--fake", action="store_true", help="run the checks on the numpy fake")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    from .weights import load_model

    params = load_model(a.weights, a.seeds, a.config, n_layers=a.blocks)
    kw = {"ticks": a.ticks, "payload": a.payload, "route": a.pair_route}
    t0 = time.time()
    if a.fake:
        from perseus.impl import fake

        from .runtime_fake import make_runtime_fake

        _fhe, inf = make_runtime_fake(
            N=32768,
            hid=params.d,
            dim=params.d,
            H=1,
            H_real=1,
            E=params.d_ffn,
            E_real=params.d_ffn,
            rot_steps=filling.rotation_steps(32768, params.d),
            complex_payload=a.payload == "pairs",
        )
        report = {"session": "fake", **run_checks(inf, fake.core, params, **kw)}
    else:
        from perseus import _core

        with open_session(params) as sess:
            opened = round(time.time() - t0, 1)
            report = {
                "session": "gpu",
                "open_seconds": opened,
                "build": str(getattr(_core, "build_info", lambda: None)()),
                **run_checks(sess.inf, _core, params, **kw),
            }
    report["total_seconds"] = round(time.time() - t0, 1)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1, default=str)
    ok = all(v.get("ok", True) for v in report.values() if isinstance(v, dict))
    return 0 if ok and not report["rotation_audit"]["n_missing"] else 1


if __name__ == "__main__":
    sys.exit(main())
