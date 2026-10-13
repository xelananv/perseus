# SPDX-License-Identifier: MIT
"""Ticks of the model on a live session (or on the fake) in one of Perseus's three modes:

    eager      refreshes fire where the levels run out (reactive), plus the port's hints
    capture    as eager, and every stage's graph is recorded on tick ``--capture-tick`` (default
               1: tick 0's carried state is zero) for the planner (plan.py); the magnitudes are
               probed on the device (FHE_MAG_GPU=0 selects the host decrypt instead)
    planned    every stage runs under its plan (``block_<b>_placement.json``, b = 0..24)

    source scripts/local_env.sh      # the n32 chain's defaults
    python -m examples.fhe_ssm_from_primitives.gpu_run --weights W --seeds S --config C \\
        --mode eager|capture|planned [--graph-dir DIR] [--plan-dir DIR] [--ticks 4] [--blocks 24] \\
        [--tokens tokens.json] [--scales shipped] [--out run.json] [--fake]

Every tick is checked against the mirror (the same kernels and folds on numpy) and the
trainer's closed form: the reply's error, the logits' error and top-1 agreement on the tied
head. The report is rewritten after every tick, so an interrupted run keeps what it measured.
``--tokens`` teacher-forces a serve run's token record (prompts then generations, as
calibrate.py reads it); without it every conversation gets random tokens. ``--free-run`` keeps
each conversation's prompt teacher-forced and then feeds back the model's own top-1 on the tied
head (the client's decode), as the published run generated its tokens; each tick then records the
tokens it fed, the top-1s, and the share that picked the token the record has next.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from . import filling
from . import plan as plan_mod
from . import scales as scales_mod
from .calibrate import streams_from_record
from .gpu_smoke import open_session
from .model import PAYLOADS, FheSsmModel
from .ref import ExactModel, MirrorModel, rel_err_rms
from .weights import load_model

MODES = ("eager", "capture", "planned")


def write_json(path: Path, doc: dict) -> None:
    """Atomic: the report on disk is always a whole one."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1, default=str)
    tmp.replace(path)


def token_rows(params, streams, pos):
    return np.asarray(params.emb[[s[pos] for s in streams]], dtype=np.float64)


def prompt_lengths(doc: dict, n: int) -> list[int]:
    """Each conversation's prompt length in a serve record (``ids_lanes``): where its
    free-running generation starts."""
    ids = doc.get("ids_lanes")
    if not ids or len(ids) < n or any(len(x) < 1 for x in ids[:n]):
        raise SystemExit(
            "--free-run needs a serve record whose 'ids_lanes' give every conversation its prompt"
        )
    return [len(x) for x in ids[:n]]


def feed_back(streams: list[list[int]], top1: list[int], prompts: list[int], pos: int) -> int:
    """Free-running: conversation k's input at pos + 1 becomes its own top-1 at pos once its
    prompt is done (pos + 1 >= prompts[k]); the prompt stays teacher-forced. Returns how many
    conversations were fed back."""
    fed = 0
    for k, s in enumerate(streams):
        if prompts[k] <= pos + 1 < len(s):
            s[pos + 1] = int(top1[k])
            fed += 1
    return fed


def token_streams(a, vocab: int, n: int, ticks: int) -> tuple[list[list[int]], list[int] | None]:
    """The conversations' token streams, from ``--tokens`` or random, and with ``--free-run``
    each conversation's prompt length."""
    prompts: list[int] | None = None
    if a.tokens:
        with open(a.tokens, encoding="utf-8") as f:
            doc = json.load(f)
        streams = streams_from_record(doc)[:n]
        if len(streams) < n:
            raise SystemExit(f"{a.tokens} holds {len(streams)} conversations, the layout {n}")
        if a.free_run:
            prompts = prompt_lengths(doc, n)
    else:
        rng = np.random.default_rng(a.seed)
        streams = [[int(t) for t in rng.integers(0, vocab, ticks)] for _ in range(n)]
    if any(len(st) < ticks for st in streams):
        raise SystemExit(f"the token streams hold fewer than {ticks} tokens")
    return streams, prompts


def free_run_step(
    rec: dict, streams: list[list[int]], record, prompts: list[int], pos: int
) -> None:
    """One tick of free running: the tokens this tick was fed, the share of the conversations now
    generating whose top-1 is the record's next token, then the feedback for the next tick."""
    rec["tokens"] = [int(s[pos]) for s in streams]
    gen = [k for k in range(len(streams)) if prompts[k] <= pos + 1 < len(record[k])]
    agree = [rec["top1"][k] == record[k][pos + 1] for k in gen]
    rec["top1_agree_vs_record"] = float(np.mean(agree)) if gen else None
    rec["fed_back"] = feed_back(streams, rec["top1"], prompts, pos)


def tick_record(model, rows, pos, mirror, exact, emb, before, top1=False):
    """One tick: the model against both references, its time and its refreshes; with ``top1``,
    each conversation's top-1 on the tied head too (what the client would feed back)."""
    # the runtime's own count (deliberate + reactive + iterations), where it exposes one: the
    # port's n_bootstraps counts only its explicit hints and carries
    fhe = getattr(model.inf, "fhe", None)
    rt0 = getattr(fhe, "total_bootstraps", None)
    t0 = time.time()
    rec = model.step(rows, pos)
    seconds = time.time() - t0
    rt1 = getattr(fhe, "total_bootstraps", None)
    want = exact.step(rows)
    got_logits, want_logits = rec["out"] @ emb.T, want @ emb.T
    per_conv = [rel_err_rms(got_logits[k], want_logits[k]) for k in range(len(rows))]
    extra = {"top1": [int(t) for t in got_logits.argmax(axis=1)]} if top1 else {}
    return {
        "pos": pos,
        "seconds": round(seconds, 2),
        "relErrRms_vs_mirror": rel_err_rms(rec["out"], mirror.step(rows)),
        "relErrRms_vs_exact": rel_err_rms(rec["out"], want),
        "logits_relErrRms_vs_exact": rel_err_rms(got_logits, want_logits),
        "logits_relErrRms_worst_conversation": max(per_conv),
        "top1_agree_vs_exact": float(
            np.mean(got_logits.argmax(axis=1) == want_logits.argmax(axis=1))
        ),
        "bootstraps": model.rt.ops.n_bootstraps - before,
        "runtime_bootstraps": None if rt0 is None or rt1 is None else int(rt1) - int(rt0),
        "out_level": rec["out_level"],
        "free_gb": rec.get("free_gb"),
        **extra,
    }


def fake_session(params, payload):
    """(inf, core) of the numpy fake with the runtime's level rules and the layout's band."""
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
        complex_payload=payload == "pairs",
    )
    return contextlib.nullcontext(inf), fake.core


def gpu_session(params):
    """(the session, a context manager whose value has ``.inf``; perseus._core)."""
    from perseus import _core

    return open_session(params), _core


def run(a) -> dict:
    params = load_model(a.weights, a.seeds, a.config, n_layers=a.blocks)
    folds = scales_mod.resolve(a.scales, params.n_layers)
    if a.mode == "capture":
        # before the session opens, as Perseus's own runner (gpt2_from_primitives/run_decode.py)
        os.environ["FHE_GRAPH_DIR"] = str(a.graph_dir)
        os.environ["FHE_GRAPH_CAPTURE_TOKEN"] = str(a.capture_tick)
        # the capture's magnitudes come from the device probe unless FHE_MAG_GPU=0 asks for the
        # host decrypt (the same five statistics; the probe is ~40x faster on the capture tick)
        os.environ.setdefault("FHE_MAG_GPU", "1")
    runtime: dict[str, str] = {}
    if a.mode == "planned":
        # before the session opens: the switches the plan declares in its PLAN_CMD.txt (plan.py), such
        # as the real-payload StC variant for real-route refreshes; an exported value wins
        runtime = plan_mod.export_plan_runtime(a.plan_dir)
    session, core = fake_session(params, a.payload) if a.fake else gpu_session(params)
    out = Path(a.out)
    report: dict = {"mode": a.mode, "fake": a.fake, "blocks": params.n_layers, "ticks": []}
    if a.mode == "planned":
        effective = plan_mod.effective_runtime(runtime)
        report["plan_runtime"] = {"plan": runtime, "effective": effective}
        if effective != runtime:
            print(
                f"[gpu_run] the environment overrides the plan's runtime switches: the plan asks for "
                f"{runtime}, in effect {effective}",
                file=sys.stderr,
                flush=True,
            )
    with session as s:
        inf = s if a.fake else s.inf
        model = FheSsmModel(
            inf,
            params,
            core=core,
            payload=a.payload,
            pair_route=a.pair_route,
            resident_weights=False,
            scales=folds,
        )
        missing = model.rotation_audit()
        report["rotation_audit_missing"] = missing[:16]
        if missing:
            write_json(out, report)
            raise SystemExit(f"rotation keys missing from the session: {missing[:8]}")
        n = model.capacity
        ticks = max(a.ticks, a.capture_tick + 1) if a.mode == "capture" else a.ticks
        streams, prompts = token_streams(a, params.vocab, n, ticks)
        if prompts is not None:
            report.update(free_run=True, prompt_lengths=prompts)
        if a.mode == "capture":
            model.set_capture(a.graph_dir)
        elif a.mode == "planned":
            model.load_plans(a.plan_dir)
            report["plans"] = sorted(model.plans or {})
        report.update(
            conversations=n, scales=a.scales, tokens=a.tokens or f"random (seed {a.seed})"
        )
        mirror = MirrorModel(params, n, model.form, scales=folds)
        exact = ExactModel(params, n)
        emb = np.asarray(params.emb, dtype=np.float64)
        record = [list(s) for s in streams]  # the record's own tokens, before any feedback
        model.start()
        for pos in range(ticks):
            before = model.rt.ops.n_bootstraps
            rows = token_rows(params, streams, pos)
            rec = tick_record(
                model, rows, pos, mirror, exact, emb, before, top1=prompts is not None
            )
            if prompts is not None:
                free_run_step(rec, streams, record, prompts, pos)
            report["ticks"].append(rec)
            write_json(out, report)
            print(json.dumps(rec), flush=True)
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--weights", required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", default="eager", choices=MODES)
    ap.add_argument("--graph-dir", default="graphs/fhe_ssm")
    ap.add_argument("--plan-dir", default="bootstrap_placements/fhe_ssm")
    ap.add_argument("--capture-tick", type=int, default=1)
    ap.add_argument("--ticks", type=int, default=4)
    ap.add_argument("--blocks", type=int, default=None)
    ap.add_argument("--payload", default="pairs", choices=PAYLOADS)
    ap.add_argument("--pair-route", default="split", choices=("split", "lane"))
    ap.add_argument("--scales", default="shipped", help="'shipped' (default), 'none', or a file")
    ap.add_argument("--tokens", default=None, help="a serve run's token record (teacher-forced)")
    ap.add_argument("--free-run", action="store_true",
                    help="after each prompt of --tokens, feed back the model's own top-1")  # fmt: skip
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--fake", action="store_true", help="the numpy fake instead of a GPU session")
    ap.add_argument("--out", default="gpu_run.json")
    a = ap.parse_args(argv)
    if a.capture_tick < 0:
        ap.error("--capture-tick must be >= 0")
    if a.fake and a.mode == "planned":
        ap.error("planned mode needs a GPU session: the fake cannot execute a plan")
    if a.free_run and not a.tokens:
        ap.error("--free-run needs --tokens: a serve record's prompts")
    run(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
