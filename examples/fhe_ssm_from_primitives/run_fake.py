"""Ticks of the fhe-ssm model on a numpy fake session against the float64 references.

    python -m fhe_ssm_from_primitives.run_fake --weights W.npz --seeds S.json --config C.json \
        --tokens 4 --blocks 24 [--rules runtime|stock] [--form perseus|depth2] [--out report.json]

Prints per tick the relative RMS error of the reply against the mirror (layout / protocol
check) and against the exact forward (Newton approximation), the bootstrap count split into
reactive refreshes, hints and the canonical carry, and the carried levels. ``--rules runtime``
(the default) applies the runtime's level rules (runtime_fake.py), ``stock`` Perseus's fake as
shipped; both default to the n32 decode preset (ceiling 46, landing 34, two primes per level).
The times are the fake's own and say nothing about a GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import resource
import subprocess
import sys
import time

import numpy as np

import perseus
from perseus.impl import fake

from .model import FheSsmModel
from .ref import ExactModel, MirrorModel, rel_err_rms
from .runtime_fake import make_runtime_fake
from .weights import load_model

RULES = ("runtime", "stock")
RSS_UNIT = 1 if sys.platform == "darwin" else 1024


def make_session(params, logn: int, *, rules="runtime", limit=46, bts_out=34, unit=2, strict=False):
    if rules not in RULES:
        raise ValueError(f"rules must be one of {RULES}, got {rules!r}")
    make = make_runtime_fake if rules == "runtime" else fake.make_fake
    return make(
        N=1 << (logn - 1),
        hid=params.d,
        dim=params.d,
        H=1,
        H_real=1,
        E=params.d_ffn,
        E_real=params.d_ffn,
        limit=limit,
        bts_out=bts_out,
        unit=unit,
        strict=strict,
        rot_steps=(),
    )


def peak_rss_gb():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * RSS_UNIT / 2**30, 2)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def perseus_head():
    root = pathlib.Path(perseus.__file__).resolve().parents[1]
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


def run(
    params,
    token_ids,
    *,
    logn=16,
    rules="runtime",
    form="perseus",
    limit=46,
    bts_out=34,
    unit=2,
    strict=False,
    canonical_carry=True,
    resident_weights=False,
    log=print,
):
    fhe, inf = make_session(
        params, logn, rules=rules, limit=limit, bts_out=bts_out, unit=unit, strict=strict
    )
    model = FheSsmModel(
        inf,
        params,
        core=fake.core,
        unit=unit,
        newton_form=form,
        canonical_carry=canonical_carry,
        resident_weights=resident_weights,
    )
    fhe.loaded_rot_steps = sorted(model.geo.rotation_steps())
    fhe._rot = set(fhe.loaded_rot_steps)
    exact, mirror = ExactModel(params, 1), MirrorModel(params, 1, form)
    by_step: dict[str, int] = {}
    inner = fhe.bootstrap

    def counted(ct):
        label = getattr(model.rt.ops, "step_label", None) or "?"
        by_step[label] = by_step.get(label, 0) + 1
        inner(ct)

    fhe.bootstrap = counted
    model.start()
    start_bootstraps = fhe.n_bootstraps
    by_step.clear()
    ticks = []
    for pos, tok in enumerate(token_ids):
        row = params.emb_row(tok)
        b0, r0, c0 = fhe.n_bootstraps, fhe.n_reactive, len(fhe.calls)
        rec = model.step(row, pos)
        calls = fhe.calls[c0:]
        t0 = time.time()
        want_m, want_e = mirror.step(row[None])[0], exact.step(row[None])[0]
        levels = model.levels()
        tick = {
            "pos": pos,
            "token": int(tok),
            "model_s": round(rec["decode_s"], 2),
            "refs_s": round(time.time() - t0, 2),
            "bootstraps": fhe.n_bootstraps - b0,
            "reactive": fhe.n_reactive - r0,
            "carry": 2 * model.n_layers if model.canonical_carry else 0,
            "mult_cc": calls.count("mult_cc") + calls.count("square"),
            "mult_pt": calls.count("mult_pt"),
            "rotations": sum(1 for c in calls if c.startswith("rotate")),
            "relErrRms_vs_mirror": rel_err_rms(rec["out"], want_m),
            "relErrRms_vs_exact": rel_err_rms(rec["out"], want_e),
            "out_level": rec["out_level"],
            "state_level_max": max(max(lv) for lv in levels),
            "levels": levels[:2] + ([levels[-1]] if model.n_layers > 2 else []),
            "peak_rss_gb": peak_rss_gb(),
        }
        tick["hints"] = tick["bootstraps"] - tick["reactive"] - tick["carry"]
        tick["bootstraps_by_step"], by_step = dict(by_step), {}
        ticks.append(tick)
        log(json.dumps({k: v for k, v in tick.items() if k != "levels"}))
    return {
        "instrument": f"perseus.impl.fake, {rules} level rules",
        "rules": rules,
        "logn": logn,
        "form": form,
        "limit": limit,
        "bts_out": bts_out,
        "unit": unit,
        "hint": "bootstrap_hint(x, headroom(products), acct=True) before each linear",
        "n_layers": model.n_layers,
        "resident_weights": resident_weights,
        "named_slot_encodes": model.rt.ops.n_encodes,
        "start_bootstraps": start_bootstraps,
        "ticks": ticks,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--weights", required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--tokens", type=int, default=4)
    ap.add_argument("--blocks", type=int, default=None)
    ap.add_argument("--logn", type=int, default=16)
    ap.add_argument("--rules", default="runtime", choices=RULES)
    ap.add_argument("--form", default="perseus", choices=("perseus", "depth2"))
    ap.add_argument("--limit", type=int, default=46)
    ap.add_argument("--bts-out", type=int, default=34)
    ap.add_argument("--unit", type=int, default=2)
    ap.add_argument("--ms-norm-k", type=float, default=5.5, help="sigma = -K b / a (--ms-norm)")
    ap.add_argument("--newton-frac", type=float, default=0.1, help="constant seed fraction")
    ap.add_argument("--newton-min-iters", type=int, default=12)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument(
        "--no-carry",
        action="store_true",
        help="skip the canonical carry: with a huge --limit the levels then measure the depth",
    )
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    norm_kw = {"k": a.ms_norm_k, "frac": a.newton_frac, "min_iters": a.newton_min_iters}
    params = load_model(a.weights, a.seeds, a.config, n_layers=a.blocks, norm_kw=norm_kw)
    ids = np.random.default_rng(a.seed).integers(0, params.vocab, size=a.tokens)
    rep = run(
        params,
        ids,
        logn=a.logn,
        rules=a.rules,
        form=a.form,
        limit=a.limit,
        bts_out=a.bts_out,
        unit=a.unit,
        canonical_carry=not a.no_carry,
    )
    rep["norm"] = {**norm_kw, "eps": params.norm_out.eps}
    rep["perseus_head"] = perseus_head()
    for name in ("weights", "seeds", "config"):
        path = getattr(a, name)
        rep[name] = {"path": path, "sha256": sha256(path)}
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=1, default=float)
    return 0


if __name__ == "__main__":
    sys.exit(main())
