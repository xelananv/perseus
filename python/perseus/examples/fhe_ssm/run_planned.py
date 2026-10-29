# SPDX-License-Identifier: MIT
"""The model under a plan on a GPU (the planned device path), with a CPU-checkable setup.

    python -m perseus.examples.fhe_ssm.run_planned --weights W --seeds S --config C [--levels preset] [axis flags]
        --plan DIR --ticks T [--tokens tokens.json [--free-run]] [--schedule program|siblings] [--profile]
        [--diag-source values|generated] --keys inproc|bundle|DIR | --insecure-test-keys --out record.json
    ... --prepare-only [--lean]          the CPU part: record the twin, compile (check) against the plan, no session
    ... --keyspec [FILE]                 the key request: the model's needs, and the session's on the device build
    ... --keygen DIR --keyspec-file F    the client role: a secret and the bundle DIR/keys.pckb for a key request

The record (rewritten after every tick) holds per tick the wall, the stage walls under --profile, the runtime's
counters, the reply's relErrRms against the exact forward and the tied head's top-1 agreement, the free run's
picks. README.md "Running under a plan on a GPU" names what each field measures and what is GPU-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import platform
import shutil
import subprocess
import sys
import time
from typing import Any

import numpy as np

from perseus import config
from perseus._paths import resolve
from perseus.plan import contract

from . import diaggen, fakes, run_fake
from . import scales as scales_mod
from .block import PACK_STATS
from .decode import FreeRun, Policy, prompt_lengths, streams_from_record
from .geometry import LINEARS, PCMM_WBITS, ROUTES
from .model import FheSsmModel
from .norm import NEWTON_FORMS, NEWTON_SITES
from .planned import LOCKSTEP_REFUSED, N32_Q_PRIMES, SCHEDULES, PlannedSsmRun, SsmRecorder, key_needs
from .ref import ExactModel, MirrorModel, rel_err_rms
from .weights import LANE_MIX, load_model

PRESET = "n32_gpt2_decode"
#: the chain the port is laid out for (``planned.N32_Q_PRIMES``, the ``--levels`` ladder, make_plan.sh's recipe)
CHAIN = "n32"
KEYS_INPROC = "inproc"
INSECURE_BANNER = (
    "INSECURE: keys generated in process, the session holds the secret key: a timing run, never a deployment"
)
#: the environment the record keeps beside the flags (the runtime's switches, the device, the planner's)
ENV_PREFIXES = ("FIDESLIB_", "PLAN_", "PF_", "PC_", "BTS_", "AUTO_BTS", "CUDA", "SPARSE_", "LADDER_", "IMPL_",
                "CKKS_", "CORRECTION_", "PERSEUS_", "FHE_")  # fmt: skip
GiB = float(1 << 30)


# ── small helpers ───────────────────────────────────────────────────────────────────────────────────────────────


def write_json(path, doc: dict) -> None:
    """Atomic: the record on disk is always a whole one."""
    path = pathlib.Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1, default=_jsonable)
    tmp.replace(path)


def _jsonable(x):
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (set, frozenset)):
        return sorted(x)
    return str(x)


def plan_cmd_lines(plan: str) -> dict[str, str]:
    """The ``key=value`` lines of the plan's PLAN_CMD.txt (graph=, axes=, recipe=, runtime=, ...)."""
    p = pathlib.Path(plan) / "PLAN_CMD.txt"
    out = {}
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            k, sep, v = line.partition("=")
            if sep and k.isidentifier():
                out[k] = v
    return out


def graphs_md5(graph_dir) -> str:
    """``cat block_*/graph.json | md5`` as make_plan.sh writes it: the files in the shell's glob order (lexicographic)."""
    h = hashlib.md5()
    files = sorted(pathlib.Path(graph_dir).glob("block_*/graph.json"), key=str)
    if not files:
        raise ValueError(f"no block_*/graph.json under {graph_dir}")
    for f in files:
        with open(f, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 24), b""):
                h.update(chunk)
    return h.hexdigest()


def plan_capture(lines: dict[str, str]) -> tuple[str | None, str | None]:
    """(the capture dir, its md5) the plan's ``graph=`` line names."""
    g = lines.get("graph")
    if not g:
        return None, None
    path, sep, rest = g.partition(" (md5 of the stages' graph.json ")
    md5 = rest.rstrip(")").strip() if sep else None
    return path.strip(), md5


def check_capture(plan: str, lines: dict[str, str], capture: str | None) -> dict:
    """The capture the plan was cut from against the one at hand (``--capture``, else the plan's own path when it
    exists here): a differing md5 refuses the run (the stream the plan binds is that capture's)."""
    path, want = plan_capture(lines)
    use = capture or (path if path and os.path.isdir(path) else None)
    out = {"plan_graph": path, "plan_md5": want, "capture": use, "md5": None, "checked": False}
    if use is None:
        return out
    if want is None:
        raise SystemExit(f"{plan}: PLAN_CMD.txt names no capture md5 to check {use} against")
    got = graphs_md5(use)
    out.update(md5=got, checked=True)
    if got != want:
        raise SystemExit(
            f"the plan {plan} was cut from a capture of md5 {want}; {use} has {got}: not the same stream"
        )
    return out


def sha256_file(path) -> str:
    return run_fake.sha256(path)


def build_id() -> dict:
    """What the timings are a property of (a wall-clock compares only inside one build): the commit, the extension's
    digest, the compiler, the arch."""
    from perseus import _pc  # type: ignore[attr-defined]

    root = pathlib.Path(__file__).resolve().parents[4]
    out: dict = {"python": sys.version.split()[0], "platform": platform.platform()}
    try:
        out["commit"] = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        out["dirty"] = len(
            subprocess.run(
                ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.splitlines()
        )
    except (OSError, subprocess.CalledProcessError):
        out["commit"] = None
    so = getattr(_pc, "__file__", None)
    out["pc_so"] = sha256_file(so)[:16] if so and os.path.isfile(so) else None
    out["pc_device_classes"] = [
        n for n in ("Session", "Program", "KeySource", "Client", "PtStore") if hasattr(_pc, n)
    ]
    nvcc = shutil.which("nvcc")
    if nvcc:
        v = subprocess.run([nvcc, "--version"], capture_output=True, text=True, check=False).stdout
        out["nvcc"] = next((ln.strip() for ln in v.splitlines() if "release" in ln), None)
    out["arch"] = os.environ.get("PERSEUS_CORE_ARCH")
    return out


def environment() -> dict[str, str]:
    return {k: v for k, v in sorted(os.environ.items()) if k.startswith(ENV_PREFIXES)}


#: planned_stats fields that are states, not counters (no delta per tick)
STATS_STATES = ("stc_keys", "bound", "slab", "group_bytes", "stc_slots")


def stats_delta(before: dict, after: dict) -> dict:
    """The per-tick increments of the runtime's counters (``Session.planned_stats``)."""
    out: dict[str, Any] = {}
    for k, v in after.items():
        if k in STATS_STATES:
            continue
        u = before.get(k)
        if isinstance(v, list) and isinstance(u, list):
            out[k] = [x - y for x, y in zip(v, u, strict=True)]
        elif isinstance(v, (int, float)) and isinstance(u, (int, float)):
            out[k] = v - u
    return out


class TokenRows:
    """``rows[t]`` = the conversations' embedding rows at tick ``t`` from their token streams (read at use, so a
    free run's fed-back tokens are seen)."""

    def __init__(self, params, streams: list[list[int]], ticks: int):
        self.params, self.streams, self.ticks = params, streams, int(ticks)

    def __len__(self) -> int:
        return self.ticks

    def __getitem__(self, t: int) -> np.ndarray:
        return np.asarray(self.params.emb[[s[t] for s in self.streams]], dtype=np.float64)


def token_streams(a, vocab: int, n: int, ticks: int) -> tuple[list[list[int]], FreeRun | None, dict]:
    """The conversations' token streams from ``--tokens`` (teacher-forced) or random, with the free run over them."""
    free = None
    meta: dict = {}
    if a.tokens:
        with open(a.tokens, encoding="utf-8") as f:
            doc = json.load(f)
        streams = streams_from_record(doc)[:n]
        if len(streams) < n:
            raise SystemExit(f"{a.tokens} holds {len(streams)} conversations, the layout {n}")
        meta = {"tokens": a.tokens, "tokens_sha256": sha256_file(a.tokens)}
        if a.free_run:
            try:
                policy = Policy.from_record(doc)
                free = FreeRun(streams, prompt_lengths(doc, n), policy)
            except ValueError as e:
                raise SystemExit(f"--free-run: {e}") from None
            meta.update(free_run=True, prompt_lengths=free.prompts, decode_policy=policy.as_dict())
    else:
        rng = np.random.default_rng(a.seed)
        streams = [[int(t) for t in rng.integers(0, vocab, ticks)] for _ in range(n)]
        meta = {"tokens": f"random (seed {a.seed})"}
    if any(len(st) < ticks for st in streams):
        raise SystemExit(f"the token streams hold fewer than {ticks} tokens")
    return streams, free, meta


# ── the keys ────────────────────────────────────────────────────────────────────────────────────────────────────


def resolve_keys(a) -> tuple[str, pathlib.Path | None]:
    """``("inproc", None)`` for the loud in-process mode, ``("bundle", dir)`` for a client's bundle; a device run
    without either is refused (there is no default)."""
    if a.insecure_test_keys and a.keys not in (None, KEYS_INPROC):
        raise SystemExit("--insecure-test-keys and --keys name two key sources")
    k = KEYS_INPROC if a.insecure_test_keys else a.keys
    if k is None:
        raise SystemExit(
            "a device run needs its keys: --keys DIR (a client's bundle: keys.pckb, optionally client_secret.bin) or, "
            f"for a timing run only, --insecure-test-keys (= --keys {KEYS_INPROC})"
        )
    if k == KEYS_INPROC:
        return KEYS_INPROC, None
    d = os.environ.get("FHE_SSM_KEYS") if k == "bundle" else k
    if not d:
        raise SystemExit("--keys bundle reads the bundle directory from FHE_SSM_KEYS")
    bundle = pathlib.Path(d)
    if not (bundle / "keys.pckb").is_file():
        raise SystemExit(
            f"{bundle}: no keys.pckb (the client's bundle for this run's key request: --keyspec, then --keygen)"
        )
    return "bundle", bundle


def check_bundle(bundle: pathlib.Path, cfg, needs: dict) -> dict:
    """The bundle against this session before it opens: the plan contract, the chain, the key kinds the layout
    needs (the exact key list is checked by the session's own import)."""
    from perseus import _pc  # type: ignore[attr-defined]

    info = dict(_pc.bundle_info(str(bundle / "keys.pckb")))
    want = _pc.contract_hash(cfg)
    if info.get("contract") != want:
        raise SystemExit(
            f"{bundle}: made for plan contract {info.get('contract')}, this session's Config is {want}"
        )
    ent = dict(info.get("entries") or {})
    if needs["pcmm_t"] and not ent.get("pcmm"):
        raise SystemExit(
            f"{bundle}: no PCMM ring-switch keys, the linears need them (pcmm_t {needs['pcmm_t']})"
        )
    least = needs["n_rotation_steps"] + 2
    if int(ent.get("ksk", 0)) < least:
        raise SystemExit(
            f"{bundle}: {ent.get('ksk', 0)} switching keys, the layout needs at least {least} (rotations + conjugation + relinearization)"
        )
    if not ent.get("public_key"):
        raise SystemExit(f"{bundle}: no public key (the server encrypts block 0's input with it)")
    return {
        "bytes": info.get("bytes"),
        "entries": ent,
        "contract": info.get("contract"),
        "params_hash": info.get("params_hash"),
    }


def keygen(a) -> int:
    """The client role: a fresh secret, the bundle for a key request, the secret kept beside it for the run's
    decryptions (``_pc.Client`` only; the session never sees it)."""
    from perseus import _pc  # type: ignore[attr-defined]

    if not hasattr(_pc, "Client"):
        raise SystemExit(
            "--keygen needs the device build of perseus._pc (Client); the CPU-only build has none"
        )
    if not a.keyspec_file:
        raise SystemExit("--keygen DIR needs --keyspec-file F (run_planned --keyspec F on the server side)")
    with open(a.keyspec_file, encoding="utf-8") as f:
        doc = json.load(f)
    spec = doc.get("session") if isinstance(doc, dict) and "session" in doc else doc
    if not isinstance(spec, dict) or "keys" not in spec:
        raise SystemExit(f"{a.keyspec_file}: no session key request (made on the device build: --keyspec)")
    out = pathlib.Path(a.keygen)
    out.mkdir(parents=True, exist_ok=True)
    secret = _pc.ClientSecret.generate()
    cl = _pc.Client(secret, complex=True)
    t0 = time.time()
    st = dict(cl.keygen(json.dumps(spec), str(out / "keys.pckb"), threads=a.threads))
    (out / "client_secret.bin").write_bytes(bytes(secret.to_bytes()))
    st["wall_s"] = round(time.time() - t0, 1)
    st["info"] = dict(_pc.bundle_info(str(out / "keys.pckb")))
    print(
        json.dumps(
            {"keygen": st, "bundle": str(out / "keys.pckb"), "client_secret": str(out / "client_secret.bin")},
            default=_jsonable,
        )
    )
    return 0


# ── the flags ───────────────────────────────────────────────────────────────────────────────────────────────────


def add_model_args(ap: argparse.ArgumentParser) -> None:
    """run_fake.py's model and axis flags (the same spellings, so run_gpu.sh repeats a capture's flags verbatim)."""
    ap.add_argument("--weights", default=None, help="required in every mode but --keygen")
    ap.add_argument("--seeds", default=None, help="required in every mode but --keygen")
    ap.add_argument("--config", default=None, help="required in every mode but --keygen")
    ap.add_argument("--blocks", type=int, default=None, help="blocks to run (default: the config's)")
    ap.add_argument("--logn", type=int, default=16, help="ring dimension log2 (N/2 slots)")
    ap.add_argument("--levels", default="preset", choices=fakes.LEVELS,
                    help="ceiling / landing of the twin: preset = this runtime's n32 preset (44 / 32), fork = 46 / 34")  # fmt: skip
    ap.add_argument("--limit", type=int, default=None, help="override the ceiling")
    ap.add_argument("--bts-out", type=int, default=None, help="override the landing")
    ap.add_argument("--unit", type=int, default=2, help="primes per level")
    ap.add_argument(
        "--layout", default="filled", choices=("filled",), help="the planned driver packs the filled layout"
    )
    ap.add_argument("--payload", default="pairs", choices=("pairs",), help="64 conversations as split pairs")
    ap.add_argument("--pair-route", default="split", choices=ROUTES, help="the nonlinear sections on a pair")
    ap.add_argument("--scales", default="shipped", help="power-of-two folds: 'shipped', 'none', or a file")
    ap.add_argument(
        "--conversations", type=int, default=None, help="conversations per tick (default: the capacity, 64)"
    )
    ap.add_argument("--newton-form", "--form", dest="newton_form", default="perseus", choices=NEWTON_FORMS)
    ap.add_argument("--newton-sites", default="served", choices=NEWTON_SITES)
    ap.add_argument("--pack-stats", default="off", choices=PACK_STATS)
    ap.add_argument("--linear", default="diagonal", choices=LINEARS)
    ap.add_argument("--pcmm-wbits", type=int, default=16, choices=PCMM_WBITS)
    ap.add_argument("--diag-hint", type=int, default=1, choices=(1, 2))
    ap.add_argument("--lane-mix", default="off", choices=LANE_MIX)
    ap.add_argument("--ms-norm-k", type=float, default=5.5)
    ap.add_argument("--newton-frac", type=float, default=0.1)
    ap.add_argument("--newton-min-iters", type=int, default=12)
    ap.add_argument("--no-carry", action="store_true", help="refused: a plan binds the carried circuit")
    ap.add_argument("--streams", type=int, default=1, help="ciphertext sets per stage (only 1 is captured)")


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_model_args(ap)
    g = ap.add_argument_group("the plan and the run")
    g.add_argument(
        "--plan", default=None, metavar="DIR", help="the plan (block_<b>_placement.json, PLAN_CMD.txt)"
    )
    g.add_argument(
        "--capture",
        default=None,
        metavar="DIR",
        help="the capture the plan was cut from (md5-checked; default: the plan's own path when present)",
    )
    g.add_argument("--ticks", type=int, default=2, help="timed ticks after the warm-up")
    g.add_argument("--tokens", default=None, help="a serve run's token record (teacher-forced)")
    g.add_argument(
        "--free-run",
        action="store_true",
        help="after each prompt of --tokens, feed back the record's decode policy's pick",
    )
    g.add_argument("--seed", type=int, default=7, help="the random tokens' seed")
    g.add_argument("--schedule", default="program", choices=SCHEDULES, help="Program.compile's schedule")
    g.add_argument(
        "--diag-source",
        default="values",
        choices=diaggen.SOURCES,
        help="where the diagonal linears' plaintexts come from: values = resident store plaintexts (the PCMM arms "
        "only fit); generated = built on the device inside each group's fused sum from the declared tiles (the device "
        "diagonal encoder). A driver flag: the capture and the plan are shared by both",
    )
    g.add_argument(
        "--profile",
        action="store_true",
        help="a device fence after every stage: stage walls (an attribution)",
    )
    g.add_argument(
        "--profile-every", type=int, default=1, metavar="K", help="with --profile, fence every K-th tick only"
    )
    g.add_argument(
        "--mirror", action="store_true", help="also score the reply against the mirror (the kernels on numpy)"
    )
    g.add_argument("--no-warmup", action="store_true", help="skip the untimed warm-up tick")
    g.add_argument(
        "--no-graphs", action="store_true", help="replay every block eagerly (exec.use_graphs = 0)"
    )
    g.add_argument("--out", default="planned_record.json", help="the record (rewritten after every tick)")
    g.add_argument(
        "--work-dir", default=None, help="the reference stream's graph dir (default: beside --out)"
    )
    k = ap.add_argument_group("keys")
    k.add_argument("--keys", default=None, metavar="inproc|bundle|DIR",
                   help="DIR: a client's bundle (keys.pckb [+ client_secret.bin]); bundle: the DIR in $FHE_SSM_KEYS; "
                        "inproc: in-process keys, INSECURE (= --insecure-test-keys)")  # fmt: skip
    k.add_argument(
        "--insecure-test-keys",
        action="store_true",
        help="keys and the secret in this process: a timing run only, loudly marked",
    )
    k.add_argument(
        "--keyspec",
        nargs="?",
        const="-",
        default=None,
        metavar="FILE",
        help="write the key request (the model's needs; the session's on the device build) and stop",
    )
    k.add_argument(
        "--keygen",
        default=None,
        metavar="DIR",
        help="the client role: a secret and the bundle for --keyspec-file",
    )
    k.add_argument("--keyspec-file", default=None, help="the key request --keygen answers")
    k.add_argument("--threads", type=int, default=8, help="--keygen threads")
    s = ap.add_argument_group("setup knobs")
    s.add_argument(
        "--prepare-only",
        action="store_true",
        help="the CPU part only: record, compile (check), report; no session",
    )
    s.add_argument(
        "--lean",
        action="store_true",
        help="with --prepare-only: keep plaintext digests, not values (the model shape on a small machine)",
    )
    s.add_argument(
        "--ref-token",
        type=int,
        default=1,
        help="the tick whose stream is compiled (>= 1: it reads a carried state)",
    )
    s.add_argument(
        "--probe", type=int, default=2, help="ticks recorded to find the dynamic steps (> --ref-token)"
    )
    s.add_argument(
        "--pt-rows", type=int, default=28, help="rows per resident plaintext slot (0 = every prime)"
    )
    s.add_argument("--stc-cache", type=int, default=8, help="StC-first last-stage cache entries")
    s.add_argument("--pool", type=int, default=16, help="eager ciphertext slots (block 0's input, the reply)")
    s.add_argument("--pcmm-cf", type=int, default=6, help="the PCMM service's correction factor")
    s.add_argument(
        "--pt-budget-gb",
        type=float,
        default=40.0,
        help="refuse when the resident plaintexts would exceed this",
    )
    return ap


# ── the run ─────────────────────────────────────────────────────────────────────────────────────────────────────


def make_twin(a):
    """The model on the capture recorder (the CPU twin), as run_fake builds it for a capture, with the plan's levels."""
    norm_kw = {"k": a.ms_norm_k, "frac": a.newton_frac, "min_iters": a.newton_min_iters}
    params = load_model(
        a.weights,
        a.seeds,
        a.config,
        n_layers=a.blocks,
        norm_kw=norm_kw,
        newton_sites=a.newton_sites,
        lane_mix=a.lane_mix,
    )
    lim, land = fakes.level_preset(a.levels)
    limit = lim if a.limit is None else int(a.limit)
    bts_out = land if a.bts_out is None else int(a.bts_out)
    fhe, inf, core = run_fake.make_session(
        params,
        a.logn,
        rules="runtime",
        limit=limit,
        bts_out=bts_out,
        unit=a.unit,
        complex_payload=True,
        capture=True,
    )
    scales = scales_mod.resolve(a.scales, params.n_layers)
    model = FheSsmModel(
        inf, params, core=core, unit=a.unit, newton_form=a.newton_form, canonical_carry=True, resident_weights=False,
        layout=a.layout, payload=a.payload, pair_route=a.pair_route, scales=scales, pack_stats=a.pack_stats,
        linear=a.linear, diag_hint=a.diag_hint, pcmm_wbits=a.pcmm_wbits,
    )  # fmt: skip
    run_fake._rot_band(fhe, model)
    diaggen.use_source(inf, a.diag_source)
    twin_cfg = fakes.runtime_config(limit, bts_out, a.unit)
    return params, model, twin_cfg, {"limit": limit, "bts_out": bts_out, "unit": a.unit, "scales": a.scales}


def check_plan_contract(plan: str, twin_cfg) -> dict | None:
    """The plan's stamp (the capture's load-bearing Config) against the twin's: a plan cut at other levels or routes
    does not bind the stream recorded here."""
    files = sorted(pathlib.Path(plan).glob("block_*_placement.json"))
    if not files:
        raise SystemExit(f"{plan}: no block_<b>_placement.json")
    stamp = contract.read_stamp(files[0])
    if not stamp:
        return None
    bad = contract.contract_mismatches(stamp, session=twin_cfg)
    if bad:
        raise SystemExit(f"{plan}: cut under another Config than the twin's:\n  " + "\n  ".join(bad))
    return stamp


def session_config(plan: str):
    """The session's Config: ``PRESET`` with the plan's stamp and ``runtime=`` line over it (pc-program's
    ``PlanDir::config`` reads a plan the same way). The chain is a runtime Config field (``chain.preset``,
    ``CHAIN_PRESET``): a runtime line that names another chain builds the session, its shadow and its key manifest
    on that chain, while this model's plaintext rows and levels are the n32 chain's; such a plan is refused."""
    raw = plan_cmd_lines(plan).get("runtime", "")
    if any(q in raw for q in "\"'"):
        raise SystemExit(
            f"{plan}: a quote in its runtime= line; the runtime splits that line on whitespace and this driver "
            "with shlex, so a quoted token would read differently on the two sides"
        )
    cfg = config.plan_runtime(plan, CHAIN, base=config.preset(PRESET))
    chain = cfg["chain.preset"]
    if chain not in (None, CHAIN):
        raise SystemExit(
            f"{plan}: its Config names the chain preset {chain!r}; this model runs on {CHAIN} only "
            f"({N32_Q_PRIMES} Q primes, the --levels ladder)"
        )
    return cfg


def sum_census(run: PlannedSsmRun) -> dict:
    """The fused sums the programs hold: per source the count and the term-count histogram (the compile check's
    ``sums`` row on the CPU build, ``Program.coeff_sums`` / ``diag_sums`` on the device build)."""
    out: dict[str, Any] = {"diag": 0, "stream": 0, "diag_terms": {}, "stream_terms": {}, "per_program": {}}
    for lab, p in run.progs.items():
        row = getattr(p, "row", None)
        if isinstance(row, dict) and "sums" in row:
            sm = row["sums"]
            hist = {
                src: {int(k): int(v) for k, v in sm[f"{src}_terms"].items()} for src in ("diag", "stream")
            }
        else:
            hist = {}
            for src, terms in (
                ("diag", getattr(p, "diag_sums", [])),
                ("stream", getattr(p, "coeff_sums", [])),
            ):
                h: dict[int, int] = {}
                for k in terms:
                    h[int(k)] = h.get(int(k), 0) + 1
                hist[src] = h
        for src in ("diag", "stream"):
            out[src] += sum(hist[src].values())
            for k, v in hist[src].items():
                out[f"{src}_terms"][k] = out[f"{src}_terms"].get(k, 0) + v
        out["per_program"][lab] = sum(hist["diag"].values()) + sum(hist["stream"].values())
    return out


def prepare_report(run: PlannedSsmRun) -> dict:
    """What the CPU part established: the compile's verdict per program, the registers, the carried names, the
    static-stream check, the plaintext budget, the fused sums, the diagonal tiles, the PCMM declarations."""
    progs = run.program_report()
    errors = {lab: p["check"][1] for lab, p in progs.items() if p["check"] and p["check"][0]}
    rep = {
        "compiled_by": run.compiled_by,
        "schedule": run.schedule,
        "conversations": run.n_conv,
        "programs": progs,
        "refreshes_planned": sum(int(p["n_bootstraps"] or 0) for p in progs.values()),
        "check_errors": errors,
        "registers": run.n_regs,
        "dynamic_steps": {str(b): sorted(v) for b, v in run.dyn.items() if v},
        "static_stream": run.static,
        "carried": {str(b): names for b, names in run.kv_names().items()},
        "plaintexts": run.plaintext_budget(),
        "diag_source": run.diag_source,
        "sums": sum_census(run),
        "prepare_s": round(run.prepare_s, 1),
        "rotation_steps": sorted({int(s) for p in run.progs.values() for s in p.rot_steps()}),
    }
    if run.pcmm_specs():
        rep["pcmm"] = {"declared": [s[0] for s in run.pcmm_specs()], **run.check_pcmm_names()}
    tiles = run.diag_tiles()
    if tiles:
        rep["diag_tiles"] = {
            "n": len(tiles),
            "bytes": sum(8 * d_in * d_out for d_in, d_out, _t in tiles.values()),
        }
    return rep


def run_mode(a) -> str:
    if a.keygen:
        return "keygen"
    if a.keyspec is not None:
        return "keyspec"
    if a.prepare_only:
        return "prepare"
    return "run"


def main(argv=None) -> int:
    ap = parser()
    a = ap.parse_args(argv)
    if a.schedule == "lockstep":
        raise SystemExit(LOCKSTEP_REFUSED)
    if a.streams != 1:
        raise SystemExit(f"--streams {a.streams}: {LOCKSTEP_REFUSED}")
    if a.no_carry:
        raise SystemExit("--no-carry: a plan binds the carried circuit (every tick one stream)")
    if a.ref_token < 1 or a.probe <= a.ref_token:
        raise SystemExit(f"--ref-token {a.ref_token} must be >= 1 and below --probe {a.probe}")
    if a.free_run and not a.tokens:
        raise SystemExit("--free-run needs --tokens: a serve record's prompts and decode policy")
    mode = run_mode(a)
    if mode == "keygen":
        return keygen(a)
    for k in ("weights", "seeds", "config"):
        if not getattr(a, k):
            raise SystemExit(f"--{k} is required (every mode but the client role, --keygen, takes the model)")
    if not a.plan:
        raise SystemExit("--plan DIR is required")
    keys_mode, bundle = resolve_keys(a) if mode == "run" else (None, None)
    from perseus import _pc  # type: ignore[attr-defined]

    device = hasattr(_pc, "Session")
    if mode == "run" and not device:
        raise SystemExit(
            "a device run needs the GPU build of perseus._pc (Session, Program); the CPU build runs --prepare-only / --keyspec"
        )
    if (
        device
        and mode in ("run", "keyspec")
        and a.diag_source == "generated"
        and not hasattr(_pc.Session, "register_tile")
    ):
        raise SystemExit(
            "--diag-source generated: this build of perseus._pc has no device diagonal encoder (Session.register_tile, "
            "Session(diag_tiles=...)); --diag-source values runs the PCMM arms"
        )
    log = lambda m: print(m, flush=True)
    plan = str(resolve(a.plan))
    lines = plan_cmd_lines(plan)
    cap = check_capture(plan, lines, a.capture)
    params, model, twin_cfg, levels = make_twin(a)
    n = model.capacity if a.conversations is None else int(a.conversations)
    if not 1 <= n <= model.capacity:
        raise SystemExit(f"{n} conversations for a capacity of {model.capacity}")
    stamp = check_plan_contract(plan, twin_cfg)
    cfg = session_config(plan)
    if a.no_graphs:
        cfg["exec.use_graphs"] = False
    ticks = max(int(a.ticks), 0)
    streams, free, tok_meta = token_streams(a, params.vocab, n, max(ticks, a.probe))
    # a free run feeds its picks back into its own copy of the streams: the rows read that copy
    rows = TokenRows(params, streams if free is None else free.streams, max(ticks, a.probe))
    labels = [f"block_{b}" for b in range(model.n_layers + 1)]
    keys = _pc.KeySource.generate([], conj=True, layout="packed") if device else None
    out = pathlib.Path(a.out)
    work = a.work_dir or str(out.with_suffix("")) + ".work"
    client = None
    if bundle is not None and (bundle / "client_secret.bin").is_file():
        client = _pc.Client(
            _pc.ClientSecret.from_bytes(list((bundle / "client_secret.bin").read_bytes())), complex=True
        )
    run = PlannedSsmRun(
        SsmRecorder(model, model.n_layers, lean=a.lean and mode != "run"), plan, cfg, labels, keys, model=model,
        conversations=n, work_dir=work, schedule=a.schedule, bundle=None if bundle is None else bundle / "keys.pckb", client=client,
        pcmm_cf=a.pcmm_cf, diag_source=a.diag_source, profile=a.profile, profile_every=a.profile_every, ref_token=a.ref_token, probe=a.probe,
        pt_rows=a.pt_rows, stc_cache=a.stc_cache, pool=a.pool, log=log,
    )  # fmt: skip
    needs = key_needs(model)
    header: dict = {
        "driver": "perseus.examples.fhe_ssm.run_planned",
        "mode": mode,
        "build_id": build_id(),
        "plan": {
            "dir": plan,
            **{k: lines.get(k) for k in ("axes", "recipe", "runtime", "planner", "tool")},
            "stamp": stamp,
        },
        "capture": cap,
        "argv": list(sys.argv[1:] if argv is None else argv),
        "flags": dict(vars(a)),
        "environment": environment(),
        "model": {
            k: {"path": getattr(a, k), "sha256": sha256_file(getattr(a, k))}
            for k in ("weights", "seeds", "config")
        },
        "axes": model.axes(),
        "diag_source": a.diag_source,
        "levels": levels,
        "conversations": n,
        "capacity": model.capacity,
        "key_needs": needs,
        "keys": {
            "mode": "insecure-inproc" if keys_mode == KEYS_INPROC else keys_mode,
            "bundle": None if bundle is None else str(bundle),
            "client_secret": client is not None,
        },
        "perseus_head": run_fake.perseus_head(),
        **tok_meta,
    }
    if a.scales.lower() != "none":
        path = scales_mod.SHIPPED if a.scales == "shipped" else a.scales
        header["scales_file"] = {"path": str(path), "sha256": sha256_file(path)}
    if keys_mode == KEYS_INPROC:
        log(f"[planned] {INSECURE_BANNER}")
    run.prepare(rows)
    header["prepare"] = prepare_report(run)
    if mode == "keyspec":
        doc: dict[str, Any] = {"model": needs, "session": None, "note": None}
        if device:
            doc["session"] = json.loads(_pc.Session.key_manifest(run.cfg, run.keys, **run.session_kwargs()))
        else:
            doc["note"] = (
                "the session's key request (Session.key_manifest) needs the device build of perseus._pc; the model's needs are above"
            )
        text = json.dumps(doc, indent=1, default=_jsonable)
        if a.keyspec == "-":
            print(text)
        else:
            pathlib.Path(a.keyspec).write_text(text, encoding="utf-8")
        return 0
    if mode == "prepare":
        header["ticks"] = []
        write_json(out, header)
        errs = header["prepare"]["check_errors"]
        log(
            f"[planned] prepare-only: {run.compiled_by}, {len(errs)} program(s) with check errors, record {out}"
        )
        return 1 if errs else 0
    # the device path
    budget = header["prepare"]["plaintexts"]
    if budget["bytes"] > a.pt_budget_gb * GiB:
        raise SystemExit(
            f"--linear {a.linear}: the programs read {budget['plaintexts']} distinct plaintexts, resident at "
            f"{budget['rows']} rows each = {budget['bytes'] / GiB:.1f} GiB (PlannedRun keeps one slot per plaintext); "
            f"over --pt-budget-gb {a.pt_budget_gb}. The diagonal linears run on the device diagonal encoder "
            "(--diag-source generated: no store slot per diagonal; README.md); the PCMM arms run resident"
        )
    if bundle is not None:
        header["keys"]["bundle_info"] = check_bundle(bundle, cfg, needs)
    t0 = time.time()
    run.setup(rows)
    header["setup"] = {
        "total_s": round(time.time() - t0, 1),
        "prepare_s": round(run.prepare_s, 1),
        "session_s": run.sess.setup_seconds()[0],
        "keygen_s": run.sess.setup_seconds()[1],
        "mem_report": run.sess.mem_report,
        "mem_free_bytes": run.fhe.mem_info()[0],
        "mem_total_bytes": run.fhe.mem_info()[1],
        "rotation_keys": list(run.rot_keys),
        "registers": run.n_regs,
        "pcmm_uploaded": list(run.uploaded),
        "diag_tiles_uploaded": len(run.tiles_uploaded),
        "has_secret": bool(run.sess.has_secret),
        "profile": "a device fence after every stage on the fenced ticks: stage walls are attributions, not the production wall"
        if a.profile
        else None,
    }
    header["ticks"] = []
    write_json(out, header)
    exact = ExactModel(params, n)
    mirror = (
        MirrorModel(params, n, model.form, scales=model.scales, linear=a.linear, pcmm_wbits=a.pcmm_wbits)
        if a.mirror
        else None
    )
    emb = np.asarray(params.emb, dtype=np.float64)
    replies = out.with_name(out.stem + ".replies")

    def tick(t, src: TokenRows, timed: bool) -> dict:
        """Tick ``t`` of ``src`` issued, its reply read and scored against the exact forward."""
        rows_t = src[t]
        c0 = run.counters()
        t1 = time.perf_counter()
        reg, host = run.token(t, src)
        got, frame, level = run.reply(reg)
        seconds = time.perf_counter() - t1
        c1 = run.counters()
        row = {
            "pos": t, "timed": timed, "seconds": round(seconds, 3), "fenced": host["fenced"], "issue_s": host["issue_s"],
            "served": host["served"], "out_level": level, "refreshes_planned": header["prepare"]["refreshes_planned"],
            "counters": stats_delta(c0["planned_stats"], c1["planned_stats"]), "mem_free_bytes": c1["mem_free_bytes"],
            "stages": host["stages"] if a.profile else None,
        }  # fmt: skip
        want = exact.step(rows_t)
        if got is None:
            replies.mkdir(parents=True, exist_ok=True)
            p = replies / f"tick_{t}{'' if timed else '_warmup'}.ct"
            p.write_bytes(frame)
            row["reply_frame"] = str(p)
            return row
        got_logits, want_logits = got @ emb.T, want @ emb.T
        per_conv = [rel_err_rms(got[k], want[k]) for k in range(n)]
        row.update(
            relErrRms_vs_exact=rel_err_rms(got, want),
            relErrRms_vs_exact_worst_conversation=max(per_conv),
            logits_relErrRms_vs_exact=rel_err_rms(got_logits, want_logits),
            top1_agree_vs_exact=float(np.mean(got_logits.argmax(axis=1) == want_logits.argmax(axis=1))),
        )
        if mirror is not None:
            row["relErrRms_vs_mirror"] = rel_err_rms(got, mirror.step(rows_t))
        if free is not None and timed:
            row.update(free.tick(t, got_logits, want_logits))
        return row

    if not a.no_warmup:
        warm = TokenRows(
            params, [[int(x)] for x in np.random.default_rng(a.seed + 1).integers(0, params.vocab, n)], 1
        )
        exact.start()
        header["warmup"] = tick(0, warm, False)
        header["warmup"]["note"] = "untimed, random public tokens, from zero state; captures the CUDA graphs"
        write_json(out, header)
        log(json.dumps({k: v for k, v in header["warmup"].items() if k != "stages"}, default=_jsonable))
    exact.start()
    if mirror is not None:
        mirror.start()
    for t in range(ticks):
        row = tick(t, rows, True)
        header["ticks"].append(row)
        if free is not None:
            header["diverged_at"] = free.diverged_at
        write_json(out, header)
        log(
            json.dumps(
                {k: v for k, v in row.items() if k not in ("stages", "tokens", "top1", "pick")},
                default=_jsonable,
            )
        )
    run.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
