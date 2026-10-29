# SPDX-License-Identifier: MIT
"""Ticks of the fhe-ssm model on the numpy fake against the float64 references, every axis a flag.

    python -m perseus.examples.fhe_ssm.run_fake --weights W.npz --seeds S.json --config C.json \\
        --tokens 2 --blocks 24 [--newton-form ...] [--newton-sites ...] [--pack-stats ...] [--linear ...]
        [--pcmm-wbits ...] [--lane-mix ...] [--diag-hint ...] [--levels ...] [--rules ...] [--layout ...]
        [--payload ...] [--pair-route ...] [--conversations N] [--capture DIR [--capture-tick 1]] [--out F]

Per tick: the reply's relative RMS error against the mirror and the exact forward, the refreshes (reactive /
hints / carry / PCMM, and by step), the op counts, the levels. ``--capture DIR`` runs on the capture recorder
and writes ``block_<b>/graph.json`` per stage at the capture tick. ``--help`` names every default; the
README the meaning of each flag.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
import os
import pathlib
import resource
import subprocess
import sys
import time

import numpy as np

import perseus
from perseus.impl import fake, fakecap

from . import fakes
from . import scales as scales_mod
from .block import PACK_STATS
from .geometry import LINEARS, PCMM_WBITS, ROUTES
from .model import LAYOUTS, PAYLOADS, FheSsmModel
from .norm import NEWTON_FORMS, NEWTON_SITES
from .ref import ExactModel, MirrorModel, rel_err_rms
from .weights import LANE_MIX, load_model

RSS_UNIT = 1 if sys.platform == "darwin" else 1024
CAPTURE_TOKEN = "FHE_GRAPH_CAPTURE_TOKEN"


def make_session(params, logn: int, *, rules="runtime", limit=46, bts_out=34, unit=2, strict=False,
                 complex_payload=False, capture=False):  # fmt: skip
    """(fhe, inf, core) at the model's dims: the fake under ``rules``, or the capture recorder."""
    N = 1 << (logn - 1)
    if capture:
        if rules != "runtime":
            raise ValueError("the capture recorder runs the runtime's rules only")
        fhe, inf = fakes.make_capture_session(
            N,
            params.d,
            params.d_ffn,
            limit=limit,
            bts_out=bts_out,
            unit=unit,
            complex_payload=complex_payload,
        )
        return fhe, inf, fakecap.core
    fhe, inf = fakes.make_fake(
        rules,
        N=N,
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
        complex_payload=complex_payload,
    )
    return fhe, inf, fake.core


def _rot_band(fhe, model):
    """Give the session exactly the rotation keys the layout (and the axes) ask for."""
    n = model.rt.dims.N
    steps = {int(s) % n for s in model.geo.rotation_steps()}
    fhe.loaded_rot_steps = sorted(steps)
    fhe._rot = steps | {s - n for s in steps}


@contextlib.contextmanager
def bootstraps_by_step(fhe, ops):
    """Count the session's bootstraps by the step scope they fire in; the session's own refresh
    entry point (``inner_bootstrap`` on the capture recorder, ``bootstrap`` on the fake) is
    restored on exit."""
    counts: dict[str, int] = {}
    name = "inner_bootstrap" if hasattr(fhe, "inner_bootstrap") else "bootstrap"
    shadowed = name in vars(fhe)
    inner = getattr(fhe, name)

    def counted(ct, *args, **kw):
        label = getattr(ops, "step_label", None) or "?"
        counts[label] = counts.get(label, 0) + 1
        return inner(ct, *args, **kw)

    setattr(fhe, name, counted)
    try:
        yield counts
    finally:
        if shadowed:
            setattr(fhe, name, inner)
        else:
            delattr(fhe, name)


def peak_rss_gb():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * RSS_UNIT / 2**30, 2)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def perseus_head():
    root = pathlib.Path(perseus.__file__).resolve().parents[2]
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


def op_counts(calls) -> dict:
    """The fake's call list -> the report's op counts."""
    return {
        "mult_cc": calls.count("mult_cc") + calls.count("square"),
        "mult_pt": calls.count("mult_pt"),
        "mult_scalar": calls.count("mult_sc") + calls.count("mult_const"),
        "rotations": sum(1 for c in calls if c.startswith("rotate")),
        "conjugations": calls.count("conjugate"),
    }


def graph_census(graph_dir) -> dict:
    """The op counts of a capture (every ``block_<b>/graph.json`` under ``graph_dir``), the same
    fields as ``op_counts`` plus the refreshes the graphs hold, by block."""
    total = {"mult_cc": 0, "mult_pt": 0, "mult_scalar": 0, "rotations": 0, "conjugations": 0,
             "bootstraps": 0, "nodes": 0}  # fmt: skip
    blocks = {}
    root = pathlib.Path(graph_dir)
    for p in sorted(root.glob("block_*/graph.json"), key=lambda q: int(q.parent.name.split("_")[1])):
        with open(p, encoding="utf-8") as f:
            nodes = json.load(f)["nodes"]
        c = dict.fromkeys(total, 0)
        c["nodes"] = len(nodes)
        for n in nodes:
            op, args = n["op_type"], n.get("args") or {}
            if op in ("mult", "square"):
                if "pt" in args:
                    c["mult_pt"] += 1
                elif "scalar" in args:
                    c["mult_scalar"] += 1
                else:
                    c["mult_cc"] += 1
            elif op in ("rotate", "rotate_sum"):
                c["rotations"] += 1
            elif op == "conjugate":
                c["conjugations"] += 1
            elif op in ("deliberate_bootstrap", "auto_bootstrap") or (op == "hint" and args.get("fired")):
                c["bootstraps"] += 1
        blocks[p.parent.name] = c
        for k in total:
            total[k] += c[k]
    return {"blocks": blocks, "total": total}


def run(
    params,
    token_ids,
    *,
    logn=16,
    rules="runtime",
    levels="fork",
    form="perseus",
    limit=None,
    bts_out=None,
    unit=2,
    strict=False,
    canonical_carry=True,
    resident_weights=False,
    layout="filled",
    payload="pairs",
    pair_route="split",
    scales=None,
    pack_stats="off",
    linear="diagonal",
    diag_hint=1,
    pcmm_wbits=16,
    capture=None,
    capture_tick=1,
    log=print,
):
    """``token_ids``: ``(ticks, conversations)`` (or ``(ticks,)`` for one conversation);
    ``scales``: the power-of-two folds per block (scales.py), or None; ``capture``: a graph dir."""
    ids = np.asarray(token_ids)
    ids = ids[:, None] if ids.ndim == 1 else ids
    lim, land = fakes.level_preset(levels)
    limit = lim if limit is None else int(limit)
    bts_out = land if bts_out is None else int(bts_out)
    complex_payload = layout == "filled" and payload == "pairs"
    fhe, inf, core = make_session(
        params, logn, rules=rules, limit=limit, bts_out=bts_out, unit=unit, strict=strict,
        complex_payload=complex_payload, capture=capture is not None,
    )  # fmt: skip
    model = FheSsmModel(
        inf,
        params,
        core=core,
        unit=unit,
        newton_form=form,
        canonical_carry=canonical_carry,
        resident_weights=resident_weights,
        layout=layout,
        payload=payload,
        pair_route=pair_route,
        scales=scales,
        pack_stats=pack_stats,
        linear=linear,
        diag_hint=diag_hint,
        pcmm_wbits=pcmm_wbits,
    )
    n = ids.shape[1]
    if not 1 <= n <= model.capacity:
        raise ValueError(f"{n} conversations for a capacity of {model.capacity}")
    _rot_band(fhe, model)
    if capture is not None:
        os.environ[CAPTURE_TOKEN] = str(int(capture_tick))
        model.set_capture(str(capture))
    exact = ExactModel(params, n)
    mirror = MirrorModel(params, n, form, scales=scales, linear=linear, pcmm_wbits=pcmm_wbits)
    filled = model.filled
    pcmm_stats = None if filled is None else filled.pcmm_stats
    ticks = []
    with bootstraps_by_step(fhe, model.rt.ops) as by_step:
        model.start()
        start_bootstraps = fhe.n_bootstraps
        by_step.clear()
        for pos, toks in enumerate(ids):
            rows = np.stack([params.emb_row(int(k)) for k in toks])
            b0, r0, c0 = fhe.n_bootstraps, fhe.n_reactive, len(fhe.calls)
            p0 = 0 if pcmm_stats is None else pcmm_stats["outputs"]
            rec = model.step(rows, pos)
            calls = fhe.calls[c0:]
            t0 = time.time()
            want_m, want_e = mirror.step(rows), exact.step(rows)
            lv = model.levels()
            per_conv = [rel_err_rms(rec["out"][k], want_e[k]) for k in range(n)]
            tick = {
                "pos": pos,
                "tokens": [int(k) for k in toks],
                "conversations": n,
                "model_s": round(rec["decode_s"], 2),
                "refs_s": round(time.time() - t0, 2),
                "bootstraps": fhe.n_bootstraps - b0,
                "reactive": fhe.n_reactive - r0,
                "carry": 2 * model.n_layers if model.canonical_carry else 0,
                "pcmm_external": 0 if pcmm_stats is None else pcmm_stats["outputs"] - p0,
                **op_counts(calls),
                "relErrRms_vs_mirror": rel_err_rms(rec["out"], want_m),
                "relErrRms_vs_exact": rel_err_rms(rec["out"], want_e),
                "relErrRms_vs_exact_worst_conversation": max(per_conv),
                "out_level": rec["out_level"],
                "state_level_max": max(max(x) for x in lv),
                "levels": lv[:2] + ([lv[-1]] if model.n_layers > 2 else []),
                "peak_rss_gb": peak_rss_gb(),
            }
            tick["hints"] = tick["bootstraps"] - tick["reactive"] - tick["carry"] - tick["pcmm_external"]
            tick["bootstraps_by_step"] = dict(by_step)
            by_step.clear()
            ticks.append(tick)
            log(json.dumps({k: v for k, v in tick.items() if k not in ("levels", "tokens")}))
    instrument = (
        "perseus.impl.fakecap (capture recorder)"
        if capture is not None
        else f"perseus.impl.fake, {rules} level rules"
    )
    rep = {
        "instrument": instrument,
        "rules": rules,
        "levels": levels,
        "layout": layout,
        "payload": payload if layout == "filled" else "real",
        "pair_route": pair_route if complex_payload else None,
        "axes": model.axes(),
        "scales": None if scales is None else [dataclasses.astuple(x) for x in scales[: model.n_layers]],
        "capacity": model.capacity,
        "conversations": n,
        "logn": logn,
        "form": form,
        "limit": limit,
        "bts_out": bts_out,
        "unit": unit,
        "hint": "bootstrap_hint(x, headroom(products), acct=True) before each linear",
        "n_layers": model.n_layers,
        "resident_weights": resident_weights,
        "runtime_config": json.loads(fhe.config.to_json()),
        "landing_switches": fakes.landing_switches_set(),
        "named_slot_encodes": model.rt.ops.n_encodes,
        "start_bootstraps": start_bootstraps,
        "rotation_steps": len(fhe.loaded_rot_steps),
        "pcmm": pcmm_stats,
        "ticks": ticks,
    }
    if capture is not None:
        rep["capture"] = {"dir": str(capture), "tick": int(capture_tick), **graph_census(capture)}
        if "lane_mix" in rep["axes"] and rep["axes"]["lane_mix"] != "off":
            rep["lane_mix_note"] = "random kernel: relErrRms_vs_exact is meaningless with this axis on"
    elif params.lane_mix != "off":
        rep["lane_mix_note"] = "random kernel: relErrRms_vs_exact is meaningless with this axis on"
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("--weights", required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--tokens", type=int, default=4, help="ticks")
    ap.add_argument("--blocks", type=int, default=None, help="blocks to run (default: the config's)")
    ap.add_argument("--logn", type=int, default=16, help="ring dimension log2 (N/2 slots)")
    ap.add_argument("--rules", default="runtime", choices=fakes.RULES, help="the fake's level rules")
    ap.add_argument("--levels", default="fork", choices=fakes.LEVELS,
                    help="ceiling / landing: fork = 46 / 34 (the validated session), preset = the n32 preset's 44 / 32")  # fmt: skip
    ap.add_argument("--limit", type=int, default=None, help="override the ceiling")
    ap.add_argument("--bts-out", type=int, default=None, help="override the landing")
    ap.add_argument("--unit", type=int, default=2, help="primes per level")
    ap.add_argument(
        "--layout",
        default="filled",
        choices=LAYOUTS,
        help="filled: t conversations per ciphertext; decode: one",
    )
    ap.add_argument(
        "--payload", default="pairs", choices=PAYLOADS, help="pairs: 2t conversations (both axes); real: t"
    )
    ap.add_argument(
        "--pair-route",
        default="split",
        choices=ROUTES,
        help="the nonlinear sections on a pair: split or lane-wise",
    )
    ap.add_argument("--scales", default="shipped",
                    help="power-of-two folds: 'shipped' (scales_pbd430a.json), 'none', or a file")  # fmt: skip
    ap.add_argument("--conversations", type=int, default=None,
                    help="conversations per tick (default: the layout's capacity: 64 pairs, 32 real, 1)")  # fmt: skip
    ap.add_argument("--newton-form", "--form", dest="newton_form", default="perseus", choices=NEWTON_FORMS,
                    help="the Newton step: perseus (3 levels per step) or depth2 (2 levels, same polynomial)")  # fmt: skip
    ap.add_argument("--newton-sites", default="served", choices=NEWTON_SITES,
                    help="served = frac 0.1 / 12 steps everywhere; per-site = 0.3 / 9 in-block, ln_out at 0.1 / 12")  # fmt: skip
    ap.add_argument("--pack-stats", default="off", choices=PACK_STATS,
                    help="halves = one Newton chain for a split pair's two halves per norm site")  # fmt: skip
    ap.add_argument("--linear", default="diagonal", choices=LINEARS,
                    help="diagonal = prefill.DiagLinear everywhere; pcmm-trunk = W_in, W_out, W_v through PCMM; pcmm = all")  # fmt: skip
    ap.add_argument(
        "--pcmm-wbits", type=int, default=16, choices=PCMM_WBITS, help="PCMM integer weight width"
    )
    ap.add_argument(
        "--diag-hint",
        type=int,
        default=1,
        choices=(1, 2),
        help="headroom (product units) of the diagonal linear's input hint: 1 = the reference circuit (the fork's), "
        "2 = upstream's diag_apply rule",
    )
    ap.add_argument(
        "--lane-mix",
        default="off",
        choices=LANE_MIX,
        help="random = the cost-only lane-mixing term per block (no trained kernel: counts only)",
    )
    ap.add_argument("--ms-norm-k", type=float, default=5.5, help="sigma = -K b / a (--ms-norm)")
    ap.add_argument(
        "--newton-frac",
        type=float,
        default=0.1,
        help="constant seed fraction (served; ln_out under per-site)",
    )
    ap.add_argument(
        "--newton-min-iters", type=int, default=12, help="minimum steps (served; ln_out under per-site)"
    )
    ap.add_argument("--seed", type=int, default=7, help="the random tokens' seed")
    ap.add_argument("--no-carry", action="store_true",
                    help="skip the canonical carry: with a huge --limit the levels then measure the depth")  # fmt: skip
    ap.add_argument("--capture", default=None, metavar="DIR",
                    help="run on the capture recorder and write block_<b>/graph.json under DIR")  # fmt: skip
    ap.add_argument("--capture-tick", type=int, default=1, help="the tick whose stages are captured")
    ap.add_argument("--out", default=None, help="the report (JSON)")
    a = ap.parse_args(argv)
    switches = fakes.landing_switches_set()
    if switches:
        print(
            "run_fake: landing switches set in the environment ("
            + ", ".join(f"{k}={v}" for k, v in switches.items())
            + "): this session's refreshes land by them, not by the reference circuit's table; the report stamps them",
            file=sys.stderr,
        )
    if a.capture is not None and a.tokens <= a.capture_tick:
        raise SystemExit(f"--capture-tick {a.capture_tick} needs more than {a.capture_tick} ticks (--tokens)")
    norm_kw = {"k": a.ms_norm_k, "frac": a.newton_frac, "min_iters": a.newton_min_iters}
    params = load_model(
        a.weights, a.seeds, a.config, n_layers=a.blocks, norm_kw=norm_kw,
        newton_sites=a.newton_sites, lane_mix=a.lane_mix,
    )  # fmt: skip
    t = (1 << (a.logn - 1)) // params.d
    capacity = 1 if a.layout == "decode" else (t if a.payload == "real" else 2 * t)
    n = capacity if a.conversations is None else a.conversations
    ids = np.random.default_rng(a.seed).integers(0, params.vocab, size=(a.tokens, n))
    rep = run(
        params,
        ids,
        scales=scales_mod.resolve(a.scales, params.n_layers),
        layout=a.layout,
        payload=a.payload,
        pair_route=a.pair_route,
        logn=a.logn,
        rules=a.rules,
        levels=a.levels,
        form=a.newton_form,
        limit=a.limit,
        bts_out=a.bts_out,
        unit=a.unit,
        canonical_carry=not a.no_carry,
        pack_stats=a.pack_stats,
        linear=a.linear,
        diag_hint=a.diag_hint,
        pcmm_wbits=a.pcmm_wbits,
        capture=a.capture,
        capture_tick=a.capture_tick,
    )
    rep["norm"] = {
        **norm_kw,
        "eps": params.norm_out.eps,
        "in_block": params.norm_kw,
        "ln_out": params.norm_kw_out,
    }
    rep["perseus_head"] = perseus_head()
    rep["argv"] = list(sys.argv[1:] if argv is None else argv)
    for name in ("weights", "seeds", "config"):
        path = getattr(a, name)
        rep[name] = {"path": path, "sha256": sha256(path)}
    if a.scales.lower() != "none":
        path = scales_mod.SHIPPED if a.scales == "shipped" else a.scales
        rep["scales_file"] = {"path": str(path), "sha256": sha256(path)}
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=1, default=float)
    return 0


if __name__ == "__main__":
    sys.exit(main())
