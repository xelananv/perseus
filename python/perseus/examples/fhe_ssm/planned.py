# SPDX-License-Identifier: MIT
"""The model under a plan on this runtime's device path (``impl.driver.PlannedRun``), and what the CPU can check.

``SsmRecorder`` is upstream's ``Recorder`` on the CPU twin plus the hook this model needs: the carried state it names
at stage entry (``cache.<name>-lvl=<L>``) is registered as a block input, so ``prepare`` carries it in registers as
GPT-2's K/V. ``PlannedSsmRun`` is ``PlannedRun`` with the compile schedule passed through, block 0's input packed as
split pairs, the PCMM linears declared and uploaded by name, the session on in-process keys (a timing run, marked
insecure) or on a client's key bundle (no secret on the server), every plaintext staged once and the ticks issued
without per-tick recording (the stream is the same every tick, verified on the probes), the reply decrypted in
process or exported as a frame for the client, and a device fence per stage under ``--profile``. On the CPU-only
build ``prepare`` compiles through ``tools/fhe_ssm/compile_check`` and ``CheckedProgram`` stands in for the
programs. Under ``diag_source="generated"`` (``diaggen``) the diagonals are made on the device from declared tiles.
Under the pass kernels (``FheSsmModel(pass_kernels="on")`` and ``passes=`` a list of ``passmasks.PassConfig``) one op
stream serves every lane configuration: the twin records the probe configuration (the first pass's), keeping each
pass-mask product as a ``PassPt`` (its mask id and values); the store keys of the masks are aliased by mask id and the
static ones by value digest, the store is sized by the distinct (alias, level) pairs, and before every pass the keys
whose values change are re-encoded in place (``stage_pass``); block 0's input and the reply go through the lane-slot
permutation (``passmasks.to_layout`` / ``from_layout``). ``CheckedProgram.pt_keys`` / ``alias_pts`` and ``FakeStore``
stand in for the device's on the CPU. README.md "Running under a plan on a GPU", "The device diagonal encoder", "A pass
in any lane configuration".
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import subprocess
import time
from typing import Any, cast

import numpy as np

import perseus
from perseus.impl import pcmm as pcmm_mod
from perseus.impl.driver import PlannedRun, Recorder, diff_block
from perseus.impl.ops import FheOps

from . import diaggen, filling, pairs, passmasks
from .geometry import KINDS, PairGeometry, park
from .model import STATE_NAMES, FheSsmModel

SCHEDULES = ("program", "siblings", "lockstep")
LOCKSTEP_REFUSED = (
    "schedule lockstep needs k > 1 independent ciphertext sets per stage in the captured stream; this port captures "
    "one stream (k = 1), so a lockstep compile would reorder nothing (tools/fhe_ssm/RUNSHEET_GPU.md section 3)"
)
#: the n32 chain's Q primes: a plaintext at level L needs 54 - L rows (impl.prefill.PlannedPrefill.session_kwargs)
N32_Q_PRIMES = 54
#: one row of a device plaintext slot: N = 2^16 coefficients of 32-bit words
SLOT_ROW_BYTES = (1 << 16) * 4
#: plaintext forms generated inside a fused sum (``Program.pt_refs``' form; crates/pc-program/src/compile/diag.rs):
#: never a store slot, never staged
GENERATED_FORMS = ("stream", "diag")
#: the forms of a v2 graph's ``args.pt.kind`` as the compiler reads them (compile/resolve.rs ``pt_form``)
PT_KIND_FORMS = {"stream": "stream", "diag": "diag", "cached": "cached"}
#: a device program's static counters the record keeps
PROGRAM_FIELDS = (
    "n_ops",
    "n_launches",
    "n_bootstraps",
    "bts_batches",
    "n_pairs",
    "n_finishes",
    "check",
    "mem",
)
#: a pass-mask plaintext's store alias is its mask id (``passmasks.mask_id``: ``pass/b<b>/...`` or ``pass/k2/...``), a
#: static plaintext's ``v/<the digest of its values>``: the two prefixes keep the partitions apart
PASS_PREFIX = "pass/"
STATIC_PREFIX = "v/"


# ── what the layout asks of the keys ────────────────────────────────────────────────────────────────────────────


def key_needs(model: FheSsmModel) -> dict:
    """What the model's layout asks of a key bundle before any program exists: the rotation steps (modulo the slot
    count), the conjugation key (the split route), one PCMM key set per token-lane count. The session's own request
    (``Session.key_manifest``) adds the refresh routes' keys and the relinearization key on top of these."""
    n = model.rt.dims.N
    steps = sorted({int(s) % n for s in model.geo.rotation_steps()} - {0})
    f = model.filled
    return {
        "slots": n,
        "d": model.params.d,
        "capacity": model.capacity,
        "rotation_steps": steps,
        "n_rotation_steps": len(steps),
        "conjugation": True,
        "pcmm_t": [f.t] if f is not None and f.uses_pcmm else [],
        "linear": None if f is None else f.linear_mode,
    }


# ── plaintext values the recorder keeps ─────────────────────────────────────────────────────────────────────────


def digest(v) -> str:
    a = np.ascontiguousarray(v)
    return hashlib.blake2b(f"{a.dtype.str}{a.shape}".encode() + a.tobytes(), digest_size=12).hexdigest()


class LeanValues:
    """A plaintext's digest in place of its slot vector: the CPU bind check at the model shape, where the diagonal
    linears read 344,064 vectors of 512 KB per tick, which the recorder would otherwise hold for every block."""

    __slots__ = ("digest", "shape")

    def __init__(self, v):
        a = np.asarray(v)
        self.digest, self.shape = digest(a), tuple(a.shape)


class PassPt:
    """What the planned twin's recorder keeps for a pass-mask plaintext (``passkernel.pass_mult_recorder``: a ``Pt`` with
    ``src = ("pass", mask_id)``): the mask id and its full slot vector in every mode (``LeanValues`` of it under
    ``--lean``). Never a marker without values: a marker staged on the device would reach ``PtStore.stage``'s float64
    conversion (``crates/pc-py/src/store.rs`` ``host_values``). The driver stages a pass key under its id and restages it
    per pass from ``passmasks.masks`` (``PlannedSsmRun.stage_static`` / ``stage_pass``); the recorded values are the
    probe configuration's."""

    __slots__ = ("_digest", "mask_id", "vec")

    def __init__(self, mask_id: str, vec):
        mask_id = str(mask_id)
        if not mask_id.startswith(PASS_PREFIX) or "#" in mask_id:
            raise ValueError(f"a pass mask id starts with {PASS_PREFIX!r} and holds no '#': {mask_id!r}")
        self.mask_id, self.vec = mask_id, vec
        self._digest: str | None = None

    @property
    def digest(self) -> str:
        """The digest of the values (``digest``; a lean record's own)."""
        if self._digest is None:
            self._digest = self.vec.digest if isinstance(self.vec, LeanValues) else digest(self.vec)
        return self._digest

    def __repr__(self) -> str:
        return f"PassPt({self.mask_id!r}, {self.digest})"


def store_alias(v) -> str:
    """The store alias of a recorded plaintext (``PlannedSsmRun.pt_aliases``): a ``PassPt`` by identity, its mask id; any
    other plaintext by value, ``v/<digest>`` (a ``LeanValues`` carries the same digest as the values it stands for)."""
    if isinstance(v, PassPt):
        return v.mask_id
    if isinstance(v, diaggen.GeneratedPt):
        raise TypeError(f"{v!r} is generated inside a fused sum: it has no store slot")
    return STATIC_PREFIX + (v.digest if isinstance(v, LeanValues) else digest(v))


def values_equal(a, b) -> bool:
    if isinstance(a, diaggen.GeneratedPt) or isinstance(b, diaggen.GeneratedPt):
        return a == b
    if isinstance(a, PassPt) or isinstance(b, PassPt):
        return (
            isinstance(a, PassPt)
            and isinstance(b, PassPt)
            and a.mask_id == b.mask_id
            and values_equal(a.vec, b.vec)
        )
    if isinstance(a, LeanValues) or isinstance(b, LeanValues):
        return isinstance(a, LeanValues) and isinstance(b, LeanValues) and a.digest == b.digest
    return a is b or bool(np.array_equal(a, b))


def stream_info(nodes) -> dict:
    """What ``prepare`` and the CPU check read of a block's stream without its nodes: the (level, deg) of every
    output, the plaintext references (name, level, deg, form: the form from ``args.pt.kind`` as the compiler reads it,
    so a generated diagonal is ``diag``), the rotation steps, the PCMM finishes by linear, the op count, and a digest
    of the stream's identity (op, inputs, output, args per node: what ``diff_block`` compares)."""
    by_out: dict[str, tuple[int, int]] = {}
    refs: list[tuple[str, int, int, str]] = []
    rot: set[int] = set()
    fin: dict[str, list[dict]] = {}
    h = hashlib.blake2b(digest_size=16)
    # a block input's name (anon_N) is numbered across blocks and tokens: as diff_block, it is its first-use position
    made: set[str] = set()
    pos: dict[str, int] = {}
    for n in nodes:
        by_out[n["output"]] = (int(n["output_level"]), int(n.get("output_noise_level", 1)))
        lv = n.get("input_levels") or []
        args = n.get("args") or {}
        kind = PT_KIND_FORMS.get(str((args.get("pt") or {}).get("kind", "")), "at_use")
        canon = []
        for j, x in enumerate(n["inputs"]):
            if x.startswith("pt_"):
                name = str(args["pt"]["name"]) if kind in GENERATED_FORMS else x
                refs.append((name, int(lv[j]) if j < len(lv) else 0, 1, kind))
            elif x.startswith("rot("):
                rot.update(int(v) for v in x[4:-1].split(","))
            if x.startswith(("pt_", "const(", "rot(", "hint_lev(", "lvl_cap(")) or x in made:
                canon.append(x)
            else:
                canon.append(f"in{pos.setdefault(x, len(pos))}")
        made.add(n["output"])
        ext = args.get("external")
        if isinstance(ext, dict) and ext.get("op") == "pcmm":
            fin.setdefault(str(ext["linear"]), []).append(ext)
        h.update(json.dumps([n["op_type"], canon, n["output"], args], sort_keys=True, default=str).encode())
    return {
        "by_out": by_out,
        "pt_refs": refs,
        "rot_steps": sorted(rot),
        "finishes": fin,
        "n_ops": len(nodes),
        "digest": h.hexdigest(),
    }


class SsmRecorder(Recorder):
    """``Recorder`` whose block records know the carried state. ``name_ct`` names a ciphertext in the current block
    generation, which ``Recorder.rec_var`` then takes for an already registered one (driver.py:485): a state named by
    ``FheSsmModel.name_state`` at stage entry would never enter ``_rec_objs`` and ``prepare`` would not carry it.
    ``lean``: a bind check without a session on a small machine: plaintext values kept as digests (``LeanValues``),
    and each block's nodes reduced to ``stream_info`` at stage end (``infos[(t, b)]``), the reference token's graph
    written to ``sink = (work_dir, ref_token)`` as it goes (the model's 25 stages hold about 280 MB of nodes each). A
    pass-mask product (``src = ("pass", mask_id)``, the pass kernels) is kept as a ``PassPt``: the id and the full vector
    in every mode (its ``LeanValues`` when lean)."""

    def __init__(self, model, n_layers, lean=False, sink=None):
        super().__init__(model, n_layers)
        fhe = model.inf.fhe
        self.lean = bool(lean)
        self.sink = sink
        self.t: int | None = None
        self.infos: dict[tuple[int, int], dict] = {}
        if not hasattr(fhe, "_rec_orig_name_ct"):
            fhe._rec_orig_name_ct = fhe.name_ct
        name_ct = fhe._rec_orig_name_ct

        def rec_name_ct(ct, name, overwrite=True):
            name_ct(ct, name, overwrite)
            if fhe.nodes is not None and name and ct.name == name and name not in fhe._rec_objs:
                fhe._rec_objs[name] = ct
                fhe._rec_metas[name] = (int(ct.level), int(ct.noise_deg))

        fhe.name_ct = rec_name_ct
        # a generated diagonal (diaggen.diag_sum_recorder) is kept by name, never by its vector, in every mode; a pass
        # mask (passkernel.pass_mult_recorder) as a PassPt: its id and its full vector (a digest of it when lean)
        kept_pt_var = fhe.pt_var  # Recorder's rec_pt_var (this recorder's), wrapped below once more when lean
        base_pt_var = fhe._rec_orig[0]

        def pass_pt_var(pt):
            new = not (pt.gen == fhe.gen and pt.name)
            name = base_pt_var(pt)
            if new and fhe.nodes is not None:
                fhe._rec_pts[name] = PassPt(pt.src[1], LeanValues(pt.vec) if self.lean else pt.vec)
            return name

        def diag_pt_var(pt):
            if pt.src[0] == "pass":
                return pass_pt_var(pt)
            if pt.src[0] != "diag":
                return kept_pt_var(pt)
            new = not (pt.gen == fhe.gen and pt.name)
            name = base_pt_var(pt)
            if new and fhe.nodes is not None:
                fhe._rec_pts[name] = diaggen.GeneratedPt(pt.src[1])
            return name

        fhe.pt_var = diag_pt_var
        if self.lean:
            pt_var = fhe._rec_orig[0]

            def lean_pt_var(pt):
                if pt.src[0] in ("diag", "pass"):
                    return diag_pt_var(pt)
                new = not (pt.gen == fhe.gen and pt.name)
                name = pt_var(pt)
                if new and fhe.nodes is not None:
                    fhe._rec_pts[name] = LeanValues(pt.vec)
                return name

            fhe.pt_var = lean_pt_var
            core = model.core
            end = core.end_subgraph_capture

            def lean_end(inf, b):
                end(inf, b)
                self._slim(b)

            core.end_subgraph_capture = lean_end

    def token(self, t, row, argmax=False):
        self.t = int(t)
        try:
            return super().token(t, row, argmax)
        finally:
            # Recorder.token sets the class-wide CapCt.novec for its values mode and leaves it: restore it, so a
            # capture or a fake made later in the same process keeps its values
            self._fakecap.CapCt.novec = False

    def _slim(self, b: int) -> None:
        """Block ``b`` just recorded: its info kept, its graph written when it is the reference token's, its nodes dropped."""
        r = self.blocks[b]
        if r.nodes is None:
            return
        r.inputs()  # cached while the nodes exist
        self.infos[(int(self.t or 0), b)] = stream_info(r.nodes)
        if self.sink is not None and self.t == self.sink[1]:
            d = os.path.join(str(self.sink[0]), f"block_{b}")
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "graph.json"), "w", encoding="utf-8") as f:
                json.dump({"version": 2, "nodes": r.nodes}, f)
        r.nodes = None


# ── the compile on either build ─────────────────────────────────────────────────────────────────────────────────


def worktree_root() -> pathlib.Path:
    return pathlib.Path(perseus.__file__).resolve().parents[2]


def default_check_bin() -> str:
    """tools/fhe_ssm/compile_check's binary (``$FHE_SSM_COMPILE_CHECK``, else where its build.sh puts it)."""
    env = os.environ.get("FHE_SSM_COMPILE_CHECK")
    if env:
        return env
    return str(
        worktree_root() / ".cache" / "fhe_ssm" / "compile_check" / "target" / "release" / "compile_check"
    )


class CheckedProgram:
    """What ``PlannedRun.prepare`` reads of a compiled ``_pc.Program``, on the CPU-only build: the inputs, outputs
    and plaintext references from the recorded stream, the counters from the compile check's row
    (``compile_check --json``). The staging's stand-in (with ``FakeStore``): ``pt_keys`` derived from the recorded
    references as the bind names them (a graph's ``pt_N`` under its block, ``<block>/pt_N``, at its level; a plaintext
    generated inside a fused sum holds no key: ``crates/pc-program/src/bind.rs``), ``alias_pts`` renaming them and
    recording its map (refused once ``n_graphs`` > 0, as the device refuses it after a graph capture). The levels are the
    twin's recorded ones; a device program's are the compiled ones. The device's ``bind`` / ``run`` / ``mark_dynamic``
    raise."""

    def __init__(self, label: str, rec, live: list[str], row: dict, info: dict | None = None):
        self.label, self.row = label, dict(row)
        if info is None:
            if rec.nodes is None:
                raise ValueError(f"{label}: a record without nodes needs its stream info")
            info = stream_info(rec.nodes)
        by_out = info["by_out"]
        self.inputs = [(n, *rec.metas.get(n, (0, 1)), 0.0) for n in rec.inputs()]
        self.outputs = [(n, *by_out.get(n, (0, 1)), 0.0) for n in live]
        self._out_pos = {n: i for i, n in enumerate(live)}
        self.pt_refs = list(info["pt_refs"])
        # one key per store reference, in reference order, renamed in place by alias_pts (Bound.pts on the device)
        self._keys: list[list] = [
            [f"{label}/{nm}", int(lv)] for nm, lv, _d, f in self.pt_refs if f not in GENERATED_FORMS
        ]
        #: alias_pts' maps, merged ({store name: alias}); a stand-in graph capture counter (alias_pts refused after one)
        self.aliases: dict[str, str] = {}
        self.n_graphs = 0
        self._rot = list(info["rot_steps"])
        self.n_ops = int(info["n_ops"])
        self.n_launches = int(row.get("launches", 0))
        self.n_bootstraps = int(row.get("sites", 0))
        self.bts_batches = (
            row.get("k_max"),
            row.get("bootstrap_launches"),
            row.get("bootstrap_hist"),
            row.get("routes"),
        )
        self.n_pairs = int(row.get("pairs", 0))
        self.n_finishes = (int(row.get("finishes", 0)), 0)
        errs = list(row.get("errors") or [])
        self.check = (len(errs), errs)
        self.mem = (int(row.get("arena_bytes", 0)), int(row.get("live_peak_bytes", 0)))

    def output_index(self, name: str) -> int:
        if name not in self._out_pos:
            raise ValueError(f"{self.label}: {name} is not an output (compile live_out)")
        return self._out_pos[name]

    def rot_steps(self) -> list[int]:
        return list(self._rot)

    def _device(self, what: str):
        raise RuntimeError(
            f"{self.label}.{what}: a device program needs perseus._pc built with Program / Session (the GPU build); "
            "this is the CPU compile check's stand-in"
        )

    def bind(self, *_a, **_k):
        self._device("bind")

    def run(self, *_a, **_k):
        self._device("run")

    def pt_keys(self) -> list[tuple[str, int]]:
        """``Program.pt_keys`` on the CPU: the distinct ``(store name, level)`` keys in first-use order, under their
        aliases once ``alias_pts`` renamed them (``Bound::pt_keys``)."""
        out: list[tuple[str, int]] = []
        seen: set[tuple[str, int]] = set()
        for nm, lv in self._keys:
            k = (str(nm), int(lv))
            if k not in seen:
                seen.add(k)
                out.append(k)
        return out

    def alias_pts(self, alias: dict[str, str]) -> int:
        """``Program.alias_pts`` on the CPU: every reference whose store name ``alias`` maps is renamed (the count is
        returned, as the device's); refused after a graph capture (``crates/pc-py/src/program.rs``: the alias map is
        permanent once a graph baked the rows)."""
        if self.n_graphs > 0:
            raise RuntimeError(f"{self.label}: alias_pts after a graph capture")
        n = 0
        for k in self._keys:
            to = alias.get(k[0])
            if to is not None:
                k[0] = str(to)
                n += 1
        self.aliases.update(alias)
        return n

    def mark_dynamic(self, *_a):
        self._device("mark_dynamic")


class FakeStore:
    """The CPU stand-in of the session's ``PtStore`` (``crates/pc-py/src/store.rs``) for the staging the driver does:
    ``stage`` takes ``[(name, level, values[, deg])]`` as the device's does and returns the count; each key stays
    resident under its ``(name, level)`` and keeps its slot when staged again (the device re-encodes a resident key in
    place); ``log`` holds ``(name, level, digest)`` per staged item, ``calls`` the items per call. ``keep_values=False``
    keeps digests only (a lean recorder's static plaintexts are digests: the model-shape dry run). Values must be what
    the device converts (a real or complex numpy vector): a recorder's marker (``PassPt``, ``GeneratedPt``) is refused,
    as is a digest where values are kept."""

    def __init__(self, keep_values: bool = True):
        self.keep_values = bool(keep_values)
        self.resident: dict[tuple[str, int], np.ndarray] = {}
        self.digests: dict[tuple[str, int], str] = {}
        self.slots: dict[tuple[str, int], int] = {}
        self.log: list[tuple[str, int, str]] = []
        self.calls: list[int] = []

    def stage(self, items) -> int:
        n = 0
        for it in items:
            if len(it) < 3:
                raise ValueError("FakeStore.stage item = (name, level, values[, deg])")
            name, level, v = str(it[0]), int(it[1]), it[2]
            if isinstance(v, (PassPt, diaggen.GeneratedPt)):
                raise TypeError(f"FakeStore.stage {name}@{level}: {v!r} is a recorder's record, not values")
            key = (name, level)
            if isinstance(v, LeanValues):
                if self.keep_values:
                    raise TypeError(f"FakeStore.stage {name}@{level}: a digest where the store keeps values")
                dg = v.digest
            else:
                a = np.ascontiguousarray(v)
                if a.ndim != 1 or a.dtype.kind not in "fc":
                    raise TypeError(f"FakeStore.stage {name}@{level}: a {a.dtype} array of shape {a.shape}")
                dg = digest(a)
                if self.keep_values:
                    self.resident[key] = a
            self.slots.setdefault(key, len(self.slots))
            self.digests[key] = dg
            self.log.append((name, level, dg))
            n += 1
        self.calls.append(n)
        return n

    def has(self, name: str, level: int) -> bool:
        return (str(name), int(level)) in self.digests

    def __len__(self) -> int:
        return len(self.digests)


class NotStatic(RuntimeError):
    """``prepare`` under the pass kernels found the stream not static across the probes: refused (a configuration may
    enter the stream only through pass-mask plaintexts, never through a scalar, a rotation step or a skipped op)."""


def compile_programs(
    work_dir,
    plan_dir,
    cfg,
    labels,
    live,
    metas,
    schedule,
    *,
    recs,
    infos=None,
    check_bin=None,
    cpu_check=False,
    log=print,
):
    """``{label: program}`` of the recorded stream under ``plan_dir``: ``_pc.Program.compile`` with the schedule on
    the device build; on the CPU-only build, or on any build under ``cpu_check=True`` (the CPU part of a run,
    ``run_planned --prepare-only``: its staging dry run reads ``CheckedProgram.pt_keys``, where a device program that
    no session has bound raises), tools/fhe_ssm/compile_check on the same directories (its compile is the same call
    without ``live_out`` / ``input_metas``: the carried inputs take the compiler's seeds) and ``CheckedProgram``s (from
    ``infos`` {label: stream_info} for records whose nodes were streamed out). Returns ``(programs, "device" |
    "cpu-check")``."""
    from perseus import _pc  # type: ignore[attr-defined]

    if schedule not in SCHEDULES:
        raise ValueError(f"schedule must be one of {SCHEDULES}, got {schedule!r}")
    if hasattr(_pc, "Program") and not cpu_check:
        progs = _pc.Program.compile(
            str(work_dir),
            str(plan_dir),
            config=cfg,
            blocks=list(labels),
            live_out=live,
            input_metas=metas,
            schedule=schedule,
        )
        return progs, "device"
    binary = check_bin or default_check_bin()
    if not os.access(binary, os.X_OK):
        raise RuntimeError(
            f"no compile check binary at {binary}: build it (bash tools/fhe_ssm/compile_check/build.sh) or set "
            "FHE_SSM_COMPILE_CHECK; the device build compiles through _pc.Program.compile"
        )
    out = os.path.join(str(work_dir), "compile_check.json")
    cmd = [binary, str(plan_dir), str(work_dir), "--schedule", schedule, "--json", out]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, check=False)
    with open(os.path.join(str(work_dir), "compile_check.txt"), "w", encoding="utf-8") as f:
        f.write(p.stdout + p.stderr)
    if p.returncode == 2 or not os.path.exists(out):
        raise RuntimeError(
            f"compile check failed to run ({' '.join(cmd)}): {p.stderr.strip() or p.stdout.strip()}"
        )
    with open(out, encoding="utf-8") as f:
        rows = {r["block"]: r for r in json.load(f)["blocks"]}
    log(
        f"[planned] compile check ({schedule}): {p.stdout.strip().splitlines()[-2] if p.stdout else ''} ({time.time() - t0:.1f} s)"
    )
    progs = {}
    for lab in labels:
        if lab not in rows:
            raise RuntimeError(
                f"compile check: no row for {lab} (a block the plan does not hold, or a compile error)"
            )
        progs[lab] = CheckedProgram(lab, recs[lab], live.get(lab, []), rows[lab], (infos or {}).get(lab))
    return progs, "cpu-check"


# ── block 0's input ─────────────────────────────────────────────────────────────────────────────────────────────


def encode_pairs(ops, rows, d: int, t: int, level: int):
    """Block 0's input on any ops: up to ``2t`` conversations (the rest parked as copies of the first) as split pairs
    at ``level``, the first ``t`` on the real axis and the rest on the imaginary one: two ``pack_tokens`` joined by
    the level-free product by ``i`` (``pairs.repack``), what ``PairGeometry.encode_rows`` records on the twin, at
    the compiled input level instead of the session's landing."""
    rows = np.asarray(rows, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != d or not 1 <= rows.shape[0] <= 2 * t:
        raise ValueError(f"rows must be (1..{2 * t}, {d}), got {rows.shape}")
    rows = park(rows, 2 * t)
    return pairs.repack(ops, ops.pack_tokens(rows[:t], int(level)), ops.pack_tokens(rows[t:], int(level)))


# ── the run ─────────────────────────────────────────────────────────────────────────────────────────────────────


class PlannedSsmRun(PlannedRun):
    """``PlannedRun`` for this model (the module docstring). ``model``: the twin ``FheSsmModel`` the recorder runs
    (its geometry and weights); ``bundle``: a client's key bundle file (``from_bundle``; None: in-process keys);
    ``client``: a ``_pc.Client`` holding the client's secret, which decrypts the exported reply (None on a bundle
    session: the reply is returned as a frame); ``check_bin``: the CPU compile check's binary, ``cpu_check``: take it
    on any build (the CPU part, ``run_planned --prepare-only``, and the tests' stand-ins); ``passes``: under the
    pass kernels (``model.pass_kernels == "on"``, and only then) the pass list, ``passmasks.PassConfig`` per pass, its
    first configuration the probes' and the warm-up's (README.md "A pass in any lane configuration")."""

    #: the pass kernels' state (class defaults, so a run built without ``__init__`` reads the off path)
    pass_mode = False
    passes: list[passmasks.PassConfig] | None = None

    def __init__(
        self,
        recorder,
        plan_dir,
        config,
        labels,
        keys,
        *,
        model: FheSsmModel,
        work_dir,
        conversations: int | None = None,
        schedule="program",
        bundle=None,
        client=None,
        pcmm_cf=6,
        diag_source="values",
        profile=False,
        profile_every=1,
        check_bin=None,
        cpu_check=False,
        ref_token=1,
        probe=2,
        pt_rows=28,
        stc_cache=8,
        pool=16,
        passes=None,
        log=print,
    ):
        if schedule not in SCHEDULES:
            raise ValueError(f"schedule must be one of {SCHEDULES}, got {schedule!r}")
        if schedule == "lockstep":
            raise ValueError(LOCKSTEP_REFUSED)
        if not isinstance(model.geo, PairGeometry):
            raise TypeError(
                "the planned driver packs split pairs: the model needs the filled layout with pairs"
            )
        if ref_token < 1 or probe <= ref_token:
            raise ValueError(
                f"the reference token must read a carried state (ref_token >= 1) and be probed: got {ref_token} / {probe}"
            )
        pass_mode = getattr(model, "pass_kernels", "off") == "on"
        if pass_mode != (passes is not None):
            raise ValueError(
                "a pass list runs on a model with pass_kernels='on', and such a model runs a pass list: model "
                f"pass_kernels={getattr(model, 'pass_kernels', 'off')!r}, passes {'given' if passes is not None else 'None'}"
            )
        if pass_mode and conversations not in (None, model.capacity):
            raise ValueError(
                f"under the pass kernels a pass decodes every lane-slot ({model.capacity}); the conversations of a "
                f"tick scenario are its configuration's, got conversations={conversations}"
            )
        n_conv = model.capacity if conversations is None else int(conversations)
        if not 1 <= n_conv <= model.capacity:
            raise ValueError(f"{n_conv} conversations for a capacity of {model.capacity}")
        if diag_source not in diaggen.SOURCES:
            raise ValueError(f"diag_source must be one of {diaggen.SOURCES}, got {diag_source!r}")
        chain = config["chain.preset"] if config is not None else None
        if chain not in (None, "n32"):  # the plaintext rows below are sized with N32_Q_PRIMES
            raise ValueError(f"the session Config names chain preset {chain!r}; this model runs on n32 only")
        super().__init__(
            recorder,
            plan_dir,
            config,
            labels,
            keys,
            ref_token=ref_token,
            probe=probe,
            work_dir=work_dir,
            pt_rows=pt_rows,
            dyn_rot=[],
            stc_cache=stc_cache,
            pool=pool,
            log=log,
        )
        self.model = model
        base = model.geo.base
        self.d, self.t = base.d, base.t
        self.n_conv = n_conv  # reply() unpacks n_conv rows; the parked lanes are dropped
        self.schedule, self.bundle, self.client = schedule, bundle, client
        self.pcmm_cf, self.check_bin = int(pcmm_cf), check_bin
        self.cpu_check = bool(cpu_check)
        self.diag_source = diag_source
        self.tiles_uploaded: list[str] = []
        #: TEST ONLY (a negative control): {tile: factor} uploaded scaled, the twin and the exact forward unchanged
        self.tile_scale: dict[str, float] = {}
        self.profile, self.profile_every = bool(profile), max(1, int(profile_every))
        self.static = False
        self.staged_at_setup: int | None = None
        self.pass_setup: dict | None = None
        self.compiled_by: str | None = None
        self.stage_walls: list[dict] = []
        self._fence = False
        self.ops: FheOps | None = None
        self.uploaded: list[str] = []
        self.pass_mode = pass_mode
        self.probe_cfg: passmasks.PassConfig | None = None
        if pass_mode:
            self.passes = list(passes)
            passmasks.validate_list(self.passes)
            if passmasks.GENERATED in self.passes[0].tokens:
                raise ValueError("pass 0 holds a generated token: no previous reply to generate it from")
            # the probes record one fixed configuration, the first pass's (the warm-up runs it too, so its masks are the
            # resident ones when the graphs are captured); the static check across the probes then proves the masks a
            # pure function of the configuration
            self.probe_cfg = self.passes[0]
        #: {(program label, pt name): store alias} (pt_aliases); the pass keys {(mask id, level): a block reading it}
        self._aliases: dict[tuple[str, str], str] | None = None
        self._pass_pairs: dict[tuple[str, int], int] = {}
        #: the digest each pass key holds in the store now (stage_static, stage_pass)
        self._pass_staged: dict[tuple[str, int], str] = {}
        #: per configuration (its mask_key): {(mask id, level): (values, digest)}, and the seconds it took
        self._mask_cache: dict[tuple, dict[tuple[str, int], tuple[Any, str]]] = {}
        self.mask_seconds: dict[tuple, float] = {}

    # ── setup (CPU) ──
    def prepare(self, rows, argmax=False):
        """``PlannedRun.prepare`` (driver.py:622-690) with the schedule passed to the compile, the CPU check in place
        of the device compile when the build has no ``Program``, the carried state checked per block and the stream
        checked static across the probes."""
        if argmax:
            raise ValueError("this model has no CutMax stage")
        if self.pass_mode:
            # the probes' configuration, fixed across the probes
            self.model.set_pass(cast("passmasks.PassConfig", self.probe_cfg))
        t0 = time.time()
        # structure only unless debugging; PCMM's fake computes its products on the slot vectors (impl.pcmm
        # Pcmm._fake reshapes the input's values), so a PCMM arm records with values
        self.rec.values = self.debug or bool(self.model.filled is not None and self.model.filled.uses_pcmm)
        os.makedirs(self.work_dir, exist_ok=True)
        lean = isinstance(self.rec, SsmRecorder) and self.rec.lean
        if lean:
            self.rec.sink = (
                self.work_dir,
                self.ref_token,
            )  # the reference graphs written as they are recorded
        for t in range(min(self.probe, len(rows))):
            self._record(t, rows, argmax)
        if self.ref_token not in self.records or self.ref_token - 1 not in self.records:
            raise RuntimeError(f"the reference token {self.ref_token} needs {self.ref_token + 1} input rows")
        ref = self.records[self.ref_token]
        self.ref = ref
        live, self.kv = {}, {}
        for b, r in ref.items():
            lab = self.labels[b]
            d = os.path.join(self.work_dir, f"block_{lab.split('_')[1]}")
            os.makedirs(d, exist_ok=True)
            if r.nodes is not None:
                with open(os.path.join(d, "graph.json"), "w", encoding="utf-8") as f:
                    json.dump({"version": 2, "nodes": r.nodes}, f)
            elif not os.path.exists(os.path.join(d, "graph.json")):
                raise RuntimeError(f"block {b}: the lean record streamed no graph for the reference token")
            prev = self.records[self.ref_token - 1][b]
            ins = r.inputs()
            kv = [
                (i, prev.final[id(r.objs[n])])
                for i, n in enumerate(ins)
                if n in r.objs and id(r.objs[n]) in prev.final
            ]
            live[lab] = list(r.out_names) + [nm for _, nm in kv]
            self.kv[b] = kv
        bad = [b for b in range(self.rec.n_layers) if len(self.kv.get(b, ())) != len(STATE_NAMES)]
        if bad:
            raise RuntimeError(
                f"blocks {bad}: the carried state was not recognised as a block input (expected {len(STATE_NAMES)} "
                "per block: the recorder must register the named state)"
            )
        metas = {
            self.labels[b]: {r.inputs()[i]: r.metas[r.inputs()[i]] for i, _ in self.kv[b]}
            for b, r in ref.items()
        }
        labels = [self.labels[b] for b in ref]
        infos = (
            {
                self.labels[b]: self.rec.infos[(self.ref_token, b)]
                for b in ref
                if (self.ref_token, b) in self.rec.infos
            }
            if lean
            else None
        )
        self.progs, self.compiled_by = compile_programs(
            self.work_dir,
            str(self.plan_dir),
            self.cfg,
            labels,
            live,
            metas,
            self.schedule,
            recs={self.labels[b]: r for b, r in ref.items()},
            infos=infos,
            check_bin=self.check_bin,
            cpu_check=self.cpu_check,
            log=self.log,
        )
        dyn: dict[int, set] = {b: set() for b in ref}
        for t, rec in self.records.items():
            for b in ref:
                if lean:
                    # the nodes are gone: equal stream digests mean no dynamic step; different ones need the nodes
                    if self.rec.infos[(t, b)]["digest"] != self.rec.infos[(self.ref_token, b)]["digest"]:
                        raise RuntimeError(
                            f"token {t} block {b}: the stream differs from the reference token's; a lean record "
                            "cannot name the dynamic steps, rerun without --lean"
                        )
                    continue
                s, k, _ = diff_block(ref[b], rec[b])
                dyn[b] |= set(s) | set(k)
        self.reg = {}
        n = 0
        for b in sorted(ref):
            p = self.progs[self.labels[b]]
            kv_regs: list[tuple[int, int, int]] = []
            out_regs: list[tuple[int, int]] = []
            for i, nm in self.kv[b]:
                kv_regs.append((i, p.output_index(nm), n))
                n += 1
            for o in ref[b].out_names:
                out_regs.append((p.output_index(o), n))
                n += 1
            self.reg[b] = {"kv": kv_regs, "out": out_regs}
        self.ref_dyn = {}
        for b, r in ref.items():
            vals: dict[str, tuple] = {}
            byname = {nd["output"]: nd for nd in r.nodes} if dyn[b] else {}  # a lean record has no nodes
            for nm in dyn[b]:
                nd = byname[nm]
                vals[nm] = (
                    ("r", int(nd["inputs"][1][4:-1]))
                    if nd["op_type"] == "rotate"
                    else ("s", float(nd["args"]["scalar"]))
                )
            self.ref_dyn[b] = vals
        self.dyn, self.n_regs = dyn, n
        self.static = self._static_stream()
        self.prepare_s = time.time() - t0
        self.log(
            f"[planned] prepared ({self.compiled_by}): {len(self.progs)} programs from token {self.ref_token}'s stream, "
            f"{sum(len(v) for v in dyn.values())} dynamic steps, {n} registers, stream "
            f"{'static' if self.static else 'NOT static'}, {self.prepare_s:.1f} s"
        )
        if self.pass_mode and not self.static:
            raise NotStatic(
                "under the pass kernels the stream must be static across the probes (one op stream for every "
                f"configuration, which enters through pass-mask plaintexts only): {sum(len(v) for v in dyn.values())} "
                "dynamic steps, or a plaintext that changed between the probes; refused"
            )
        return self

    def _static_stream(self) -> bool:
        """The same op stream and the same plaintext values at every probe token (no patch per tick)."""
        if any(self.dyn.values()):
            return False
        for t, rec in self.records.items():
            if t == self.ref_token:
                continue
            for b, r in self.ref.items():
                o = rec[b]
                if set(o.pts) != set(r.pts) or any(not values_equal(o.pts[k], r.pts[k]) for k in r.pts):
                    return False
        return True

    def out_reg(self, b: int) -> int:
        """The register that holds stage ``b``'s output (the next stage's input; the tail's is the reply)."""
        outs = cast("list[tuple[int, int]]", self.reg[b]["out"])
        return int(outs[0][1])

    def kv_names(self) -> dict[int, list[str]]:
        """Per block, the names of its carried inputs in the reference stream (``cache.<name>-lvl=<L>``)."""
        return {b: [self.ref[b].inputs()[i] for i, _nm in kv] for b, kv in self.kv.items()}

    def plaintext_refs(self) -> set[tuple[str, str, int]]:
        """The distinct (program, name, level) plaintext references the store holds: a store key is the block's
        (``<block>/pt_N``), so the same ``pt_N`` in two blocks is two slots; a plaintext generated inside a fused sum
        (a streamed row, a generated diagonal) holds none (``StreamedPlannedRun`` filters the same way,
        ``driver.py:1265, :1279``)."""
        return {
            (lab, nm, int(lv))
            for lab, p in self.progs.items()
            for nm, lv, _d, f in p.pt_refs
            if f not in GENERATED_FORMS
        }

    def pt_aliases(self) -> dict[tuple[str, str], str]:
        """``{(program label, pt name): store alias}`` for every plaintext the programs read from the store (their
        ``pt_refs``, the forms generated inside a fused sum left out): a pass mask (a ``PassPt`` of the reference
        stream) by identity, its mask id (``pass/b<b>/...`` per block, ``pass/k2/...`` shared by every block, at the
        levels each block reads it); every other plaintext by the digest of its values (``v/<digest>``, the name
        ``PlannedRun.precompute`` gives a slot). Built from the compiled programs and the twin's reference record, so it
        exists before the session (``session_kwargs`` sizes the store by it) and ``stage_static`` applies the same map.
        No pass alias equals a static one (the prefixes differ: checked)."""
        if self._aliases is not None:
            return self._aliases
        out: dict[tuple[str, str], str] = {}
        for b in sorted(self.ref):
            lab, r = self.labels[b], self.ref[b]
            for nm, _lv, _d, f in self.progs[lab].pt_refs:
                if f in GENERATED_FORMS:
                    continue
                if nm not in r.pts:
                    raise RuntimeError(
                        f"{lab}: the program reads {nm}, which the twin's reference record does not hold"
                    )
                out[(lab, nm)] = store_alias(r.pts[nm])
        passes = {a for a in out.values() if a.startswith(PASS_PREFIX)}
        odd = sorted(a for a in out.values() if not a.startswith((PASS_PREFIX, STATIC_PREFIX)))
        if odd:
            raise RuntimeError(f"pt_aliases: aliases in neither partition (pass/, v/): {odd[:4]}")
        if passes and not self.pass_mode:
            raise RuntimeError(
                "pt_aliases: pass-mask plaintexts in a stream recorded without the pass kernels"
            )
        self._aliases = out
        return out

    def plaintext_pairs(self) -> set[tuple[str, int]]:
        """The distinct (store alias, level) pairs the programs read: one store slot each (``session_kwargs``)."""
        al = self.pt_aliases()
        return {
            (al[(lab, nm)], int(lv))
            for lab, p in self.progs.items()
            for nm, lv, _d, f in p.pt_refs
            if f not in GENERATED_FORMS
        }

    def plaintext_budget(self) -> dict:
        """What the resident plaintexts would take on the device, both ways: by references, one slot per distinct
        (program, name, level) (``plaintexts``, ``bytes``: what ``PlannedRun.session_kwargs`` reserves upstream), and by
        aliases, one slot per distinct (store alias, level) (``by_alias``: what this driver reserves, ``reserved``), each
        slot of ``pt_rows`` rows (at least the rows of the deepest plaintext, n32: 54 - level). The pass masks' share is
        named in both (``pass_refs``; ``by_alias.pass``, ``by_alias.pass_ids``)."""
        refs = self.plaintext_refs()
        pairs = self.plaintext_pairs()
        al = self.pt_aliases()
        levels = [lv for _lab, _nm, lv in refs]
        rows = self.pt_rows and max(self.pt_rows, N32_Q_PRIMES - min(levels, default=N32_Q_PRIMES))
        rows = int(rows or N32_Q_PRIMES)
        pass_refs = sum(1 for lab, nm, _lv in refs if al[(lab, nm)].startswith(PASS_PREFIX))
        pass_pairs = [a for a, _lv in pairs if a.startswith(PASS_PREFIX)]
        return {
            "plaintexts": len(refs),
            "rows": rows,
            "bytes": len(refs) * rows * SLOT_ROW_BYTES,
            "pass_refs": pass_refs,
            "static_refs": len(refs) - pass_refs,
            "by_alias": {
                "slots": len(pairs),
                "pass": len(pass_pairs),
                "static": len(pairs) - len(pass_pairs),
                "pass_ids": len(set(pass_pairs)),
                "bytes": len(pairs) * rows * SLOT_ROW_BYTES,
            },
            "reserved": "by_alias",
        }

    def linears(self) -> dict[str, pcmm_mod.PcmmLinear]:
        """The PCMM linears by the name their finishes carry (``b<b>.<kind>``), from the twin's weights."""
        out: dict[str, pcmm_mod.PcmmLinear] = {}
        for b in range(self.rec.n_layers):
            bw = self.model.block_weights(b)
            for kind in KINDS:
                w = getattr(bw, kind)
                if isinstance(w, pcmm_mod.PcmmLinear):
                    if w.name is None or w.name in out:
                        raise RuntimeError(
                            f"block {b} {kind}: a PCMM linear without a unique name ({w.name!r})"
                        )
                    out[w.name] = w
        return out

    def pcmm_specs(self) -> list[tuple]:
        """``Session(pcmm=...)``'s ``(name, t, n_in, n_out, planes, w_scale, shift)`` per linear."""
        return [
            (
                lin.name,
                int(lin.t or self.t),
                int(lin.n_in),
                int(lin.n_out),
                int(lin.Wd.shape[0]),
                float(lin.w_scale),
                int(lin.shift),
            )
            for lin in self.linears().values()
        ]

    def finishes(self) -> dict[str, list[dict]]:
        """The PCMM finish descriptors of the reference stream by linear name (the capture's ``args.external``)."""
        out: dict[str, list[dict]] = {}
        for b, r in self.ref.items():
            info = self.rec.infos.get((self.ref_token, b)) if isinstance(self.rec, SsmRecorder) else None
            fin = info["finishes"] if info is not None else stream_info(r.nodes)["finishes"]
            for name, descs in fin.items():
                out.setdefault(name, []).extend(descs)
        return out

    def check_pcmm_names(self) -> dict:
        """Every finish names a declared linear with the shape it was compiled for, and every declared linear is
        read: the device service's own checks (crates/pc-py/src/pcmm.rs), run here before any session."""
        fin, specs = self.finishes(), {s[0]: s for s in self.pcmm_specs()}
        missing = sorted(set(fin) - set(specs))
        unused = sorted(set(specs) - set(fin))
        if missing or unused:
            raise RuntimeError(
                f"PCMM linears: finishes name undeclared {missing}, declared but never read {unused}"
            )
        for name, descs in fin.items():
            _n, t, n_in, n_out, _planes, _ws, shift = specs[name]
            for d in descs:
                got = (int(d["n_in"]), int(d["n_out"]), int(d["t"]), int(d["shift"]))
                if got != (n_in, n_out, t, shift):
                    raise RuntimeError(
                        f"PCMM {name}: finish (n_in, n_out, t, shift) {got} vs the declared {(n_in, n_out, t, shift)}"
                    )
            if len(descs) != n_out:
                raise RuntimeError(f"PCMM {name}: {len(descs)} finishes for {n_out} outputs")
        return {"linears": len(specs), "finishes": sum(len(v) for v in fin.values())}

    def size_dict(self) -> dict[str, int]:
        """``Session(size=)``: the geometry ``pack_tokens`` lays rows out by (``t = slots / hidDim``, ``dim`` features)."""
        d, dff = self.model.params.d, self.model.params.d_ffn
        return {
            "hidDim": d,
            "dim": d,
            "numHeads": 1,
            "numHeadsReal": 1,
            "expDim": dff,
            "expanded": dff,
            "seqLen": 1,
        }

    def diag_tiles(self) -> dict[str, tuple[int, int, int]]:
        """``Session(diag_tiles=...)``: ``{linear: (d_in, d_out, t)}`` of the diagonal linears under
        ``diag_source="generated"``, else empty (``diaggen.diag_tiles``)."""
        return diaggen.diag_tiles(self.model) if self.diag_source == "generated" else {}

    def session_kwargs(self):
        kw = super().session_kwargs()
        kw["size"] = self.size_dict()
        specs = self.pcmm_specs()
        kw["pcmm"] = specs or None
        kw["pcmm_cf"] = self.pcmm_cf
        # the store holds the plaintexts read from it only, one slot per distinct (store alias, level) pair, which is
        # what stage_static stages: PlannedRun.session_kwargs (driver.py:725-731) counts every reference, generated
        # ones included (344,064 diagonals per tick at the model shape), and every pass-mask product is a reference of
        # its own (+2,448 per pass at 28 rows: +16.8 GiB [derived])
        kw["pt_levels"] = [lv for _a, lv in sorted(self.plaintext_pairs())]
        kw["pt_slots"] = len(kw["pt_levels"]) + 8
        if kw["pt_levels"] and self.pt_rows:
            kw["pt_rows"] = max(self.pt_rows, N32_Q_PRIMES - min(kw["pt_levels"]))
        tiles = self.diag_tiles()
        if tiles:
            kw["diag_tiles"] = tiles
        return kw

    def device_has_encoder(self) -> bool:
        """Whether this build's session takes generated diagonals (``Session.register_tile``, with the
        ``diag_tiles`` declaration)."""
        from perseus import _pc  # type: ignore[attr-defined]

        return hasattr(getattr(_pc, "Session", None), "register_tile")

    # ── the session (device) ──
    def open_session(self, kw):
        from perseus import _pc  # type: ignore[attr-defined]

        if self.bundle is None:
            return _pc.Session(self.cfg, self.keys, **kw)
        sess = _pc.Session.from_bundle(self.cfg, self.keys, str(self.bundle), **kw)
        if sess.has_secret:
            raise RuntimeError("the bundle session holds a secret")
        return sess

    def setup(self, rows, argmax=False):
        """``PlannedRun.setup`` (the session, every program bound, the arena frozen), then the PCMM planes uploaded
        by name and every plaintext staged once; under the pass kernels the first pass's masks staged with the static
        plaintexts and every configuration's masks computed (``precompute_passes``)."""
        from perseus import _pc  # type: ignore[attr-defined]

        if self.diag_tiles() and not self.device_has_encoder():
            raise RuntimeError(
                "diag_source generated: this build of perseus._pc has no device diagonal encoder "
                "(Session.register_tile / Session(diag_tiles=...)); run --diag-source values or a build with it"
            )
        super().setup(rows, argmax)
        self.ops = FheOps(self.sess.fhe, _pc, self.model.rt.ops.unit)
        if self.bundle is not None and self.fhe.has_secret_key:
            raise RuntimeError("the bundle session's op surface holds a secret key")
        for name, lin in self.linears().items():
            self.sess.pcmm_upload(name, np.ascontiguousarray(lin.Wd, dtype=np.int8).tobytes())
            self.uploaded.append(name)
        self.upload_tiles()
        n = self.stage_static()
        self.staged_at_setup = n
        self.pass_setup = self.precompute_passes() if self.pass_mode else None
        self.log(
            f"[planned] {len(self.uploaded)} PCMM linears uploaded, {len(self.tiles_uploaded)} diagonal tiles "
            f"uploaded, {n} plaintext slots staged"
            + (f"; pass masks: {json.dumps(self.pass_setup)}" if self.pass_setup else "")
        )
        return self

    def upload_tiles(self) -> int:
        """Every declared tile's matrix to the device (after ``complete_setup``, like ``pcmm_upload``), one block's
        matrices built at a time; returns the tiles uploaded (0 under ``diag_source="values"``)."""
        tiles = self.diag_tiles()
        if not tiles:
            return 0
        unknown = sorted(set(self.tile_scale) - set(tiles))
        if unknown:
            raise RuntimeError(f"tile_scale names undeclared tiles {unknown}")
        for name, W in diaggen.tile_matrices(self.model):
            if W.shape != tiles[name][:2]:
                raise RuntimeError(f"tile {name}: matrix {W.shape}, declared {tiles[name][:2]}")
            up = np.ascontiguousarray(W * float(self.tile_scale[name])) if name in self.tile_scale else W
            self.sess.register_tile(name, up)
            self.tiles_uploaded.append(name)
        missing = sorted(set(tiles) - set(self.tiles_uploaded))
        if missing:
            raise RuntimeError(f"declared tiles never uploaded: {missing[:8]}")
        return len(self.tiles_uploaded)

    def _keeps_values(self) -> bool:
        return bool(getattr(self.store, "keep_values", True))

    def stage_static(self) -> int:
        """Every plaintext the programs read staged once, before the warm-up captures the graphs: the alias map is
        permanent (``alias_pts`` is refused after a graph capture, ``crates/pc-py/src/program.rs``), so it is built here
        once (``pt_aliases``). A static plaintext shares a slot with every equal one, by the digest of its values (as
        ``PlannedRun.precompute`` names a slot per position); a pass mask (the pass kernels) is aliased by its mask id,
        never by value (one configuration's all-zero masks would otherwise share one slot that the next configuration
        needs nonzero), and staged at each level a program reads it (``(alias, compiled level)``) with the first
        pass's values; ``stage_pass`` restages it per pass. One item per distinct (alias, level) pair, the slots the
        store holds (``session_kwargs``). Returns the items staged."""
        if self._alias:
            raise RuntimeError("stage_static: the plaintexts are already staged")
        if not self.static:
            raise RuntimeError(
                "stage_static: the stream is not static (prepare found a dynamic step or a changed plaintext)"
            )
        al = self.pt_aliases()
        items: dict[tuple[str, int], Any] = {}
        alias: dict[str, str] = {}
        pairs: dict[tuple[str, int], int] = {}
        for b in sorted(self.ref):
            lab, r = self.labels[b], self.ref[b]
            keys = self.progs[lab].pt_keys()
            amap: dict[str, str] = {}
            for nm, lv in keys:
                if "#" in nm:  # Bound::pt_keys skips every plaintext generated inside a sum (bind.rs in_sum)
                    raise RuntimeError(f"stage_static: {nm} is a generated plaintext, not a store key")
                pt = nm.split("/", 1)[1]
                a = al.get((lab, pt))
                if a is None:
                    raise RuntimeError(f"stage_static: {lab} reads {nm}, which no plaintext reference names")
                amap[nm] = a
                if a.startswith(PASS_PREFIX):
                    pairs.setdefault((a, int(lv)), b)
                    continue
                v = r.pts[pt]
                if isinstance(v, LeanValues) and self._keeps_values():
                    raise TypeError("stage_static: a lean recorder holds digests, not values")
                items.setdefault((a, int(lv)), v)
            self.progs[lab].alias_pts(amap)
            alias.update(amap)
        self._alias = alias
        self._pass_pairs = pairs
        if pairs and not self.pass_mode:
            raise RuntimeError("stage_static: pass-mask keys in a stream recorded without the pass kernels")
        staged = [(nm, lv, v) for (nm, lv), v in items.items()]
        if self.pass_mode:
            staged += self._pass_items(cast("list[passmasks.PassConfig]", self.passes)[0], every=True)
        return int(self.store.stage(staged))

    # ── the pass kernels' per-pass staging ──
    def pass_masks(self, cfg: passmasks.PassConfig) -> dict[tuple[str, int], tuple[Any, str]]:
        """``{(mask id, level): (values, digest)}`` of configuration ``cfg`` for every pass key the programs read
        (``stage_static`` found them): ``passmasks.masks`` per block from the cached decay and mix
        (``FheSsmModel.block_decay``), one vector per id (a block-independent id, ``pass/k2/...``, is checked equal in
        every block), cached per ``cfg.mask_key()`` with its seconds (``mask_seconds``). Values are kept only when the
        store keeps them (``FakeStore(keep_values=False)``: digests)."""
        key = cfg.mask_key()
        got = self._mask_cache.get(key)
        if got is not None:
            return got
        if not self._pass_pairs:
            raise RuntimeError("pass_masks before stage_static (no pass key is known yet)")
        t0 = time.perf_counter()
        keep = self._keeps_values()
        levels: dict[str, list[int]] = {}
        for mid, lv in self._pass_pairs:
            levels.setdefault(mid, []).append(lv)
        per_id: dict[str, tuple[Any, str]] = {}
        first: dict[str, int] = {}
        for b in range(self.rec.n_layers):
            for mid, vec in passmasks.masks(cfg, self.model.block_decay(b), b).items():
                if mid not in levels:
                    continue
                dg = digest(vec)
                if mid in per_id:
                    if per_id[mid][1] != dg:
                        raise RuntimeError(
                            f"pass mask {mid}: block {b}'s values differ from block {first[mid]}'s (one alias for "
                            "every block needs one value)"
                        )
                    continue
                per_id[mid], first[mid] = (vec if keep else LeanValues(vec), dg), b
        missing = sorted(set(levels) - set(per_id))
        if missing:
            raise RuntimeError(
                f"pass_masks: the programs read {missing[:4]}, which passmasks.masks does not make"
            )
        out = {(mid, lv): per_id[mid] for mid, lvs in levels.items() for lv in lvs}
        self._mask_cache[key] = out
        self.mask_seconds[key] = time.perf_counter() - t0
        return out

    def _pass_items(self, cfg: passmasks.PassConfig, every: bool = False) -> list[tuple[str, int, Any]]:
        """The pass keys to stage before a pass of ``cfg``: those whose digest differs from the resident one (every key
        with ``every``), the bookkeeping updated."""
        masks = self.pass_masks(cfg)
        out = []
        for k in sorted(self._pass_pairs):
            vec, dg = masks[k]
            if every or self._pass_staged.get(k) != dg:
                out.append((k[0], k[1], vec))
                self._pass_staged[k] = dg
        return out

    def precompute_passes(self) -> dict:
        """Setup: the masks and digests of every configuration of the pass list (``pass_masks``), so a pass's in-clock
        work is staging what changed. Returns what was computed."""
        if not self.pass_mode:
            raise RuntimeError("precompute_passes: the pass kernels are off")
        t0 = time.perf_counter()
        keys = []
        for cfg in cast("list[passmasks.PassConfig]", self.passes):
            if cfg.mask_key() not in keys:
                keys.append(cfg.mask_key())
                self.pass_masks(cfg)
        return {
            "configurations": len(keys),
            "pass_keys": len(self._pass_pairs),
            "seconds": round(time.perf_counter() - t0, 3),
            "per_configuration_s": [round(self.mask_seconds[k], 3) for k in keys],
        }

    def stage_pass(self, cfg: passmasks.PassConfig) -> dict:
        """Before a pass of ``cfg``: every pass key whose values differ from the resident ones (by digest) restaged in
        place (``PtStore.stage``: the key's resident slot re-encoded, so the captured graphs keep replaying; never an
        evict, never a new key). ``{"staged", "stage_s", "masks_s"}``: ``masks_s`` > 0 only for a configuration first
        seen here, its masks computed in the clock."""
        if not self.pass_mode:
            raise RuntimeError("stage_pass: the pass kernels are off")
        if not self._alias:
            raise RuntimeError("stage_pass before stage_static")
        t0 = time.perf_counter()
        new = cfg.mask_key() not in self._mask_cache
        items = self._pass_items(cfg)
        t1 = time.perf_counter()
        n = int(self.store.stage(items)) if items else 0
        return {
            "staged": n,
            "stage_s": round(time.perf_counter() - t0, 4),
            "masks_s": round(t1 - t0, 4) if new else 0.0,
        }

    def staging_dry_run(self, keep_values: bool = True) -> dict:
        """The setup's and every pass's staging on the CPU stand-ins (``FakeStore``, ``CheckedProgram.pt_keys`` /
        ``alias_pts``), before any session: what ``setup`` and each pass would stage, the warm-up first (the first
        pass's configuration). ``keep_values=False`` stages digests (a lean record). The programs stay aliased."""
        if not self.pass_mode:
            raise RuntimeError("staging_dry_run: the pass kernels are off")
        self.store = FakeStore(keep_values=keep_values)
        t0 = time.perf_counter()
        n0 = self.stage_static()
        setup_s = time.perf_counter() - t0
        pre = self.precompute_passes()
        passes = cast("list[passmasks.PassConfig]", self.passes)
        rows = [{"pass": "warmup", "config": passes[0].name, **self.stage_pass(passes[0])}]
        rows += [{"pass": p, "config": c.name, **self.stage_pass(c)} for p, c in enumerate(passes)]
        held = {k for p in self.progs.values() for k in p.pt_keys()}
        return {
            "store": "FakeStore (the CPU stand-in)",
            "keep_values": keep_values,
            "setup_staged": n0,
            "setup_s": round(setup_s, 3),
            "store_keys": len(self.store),
            "pass_keys": len(self._pass_pairs),
            "every_program_key_resident": held <= set(cast("FakeStore", self.store).digests),
            "masks": pre,
            "passes": rows,
        }

    # ── per tick ──
    def input_ct(self, t, rows, p):
        """Block 0's input of tick ``t``: ``rows[t]`` (the conversations' rows) as split pairs at the compiled input
        level (``p.inputs[0]``); under the pass kernels the ``2t`` lane-slot rows of pass ``t`` in ``q`` order, permuted
        to the layout's row order here (``passmasks.to_layout``, the one place on this end)."""
        if self.ops is None:
            raise RuntimeError("input_ct before setup")
        level, deg = int(p.inputs[0][1]), int(p.inputs[0][2])
        if deg != 1:
            raise RuntimeError(
                f"block 0's input is compiled at degree {deg}; a fresh encryption has degree 1"
            )
        return encode_pairs(self.ops, self.layout_rows(rows[t]), self.d, self.t, level)

    def layout_rows(self, rows) -> np.ndarray:
        """Block 0's rows in the layout's order: under the pass kernels all ``2t`` lane-slot rows in ``q`` order through
        ``passmasks.to_layout``; else as given."""
        if not self.pass_mode:
            return rows
        rows = np.asarray(rows, dtype=np.float64)
        if rows.shape != (2 * self.t, self.d):
            raise ValueError(f"a pass packs all {2 * self.t} lane-slots in q order, got {rows.shape}")
        return passmasks.to_layout(rows)

    def _run_block(self, b, t, zero, first_in, variant=None):
        t0 = time.perf_counter()
        info = super()._run_block(b, t, zero, first_in, variant)
        rec = {"block": int(b), **{k: info[k] for k in ("served", "host_us", "launch_us") if k in info}}
        if self._fence:
            self.fhe.device_sync()
            rec["wall_s"] = round(time.perf_counter() - t0, 4)
        else:
            rec["issue_s"] = round(time.perf_counter() - t0, 4)
        self.stage_walls.append(rec)
        return info

    def token(self, t, rows, argmax=False, entry=None, variant=None):
        """Tick ``t`` issued on the device: the static stream needs no recording and no patch, so the blocks run
        straight from the registers (``PlannedRun.token`` otherwise). Asynchronous until the reply is read;
        returns (the reply's register, the host record with a row per stage)."""
        if argmax or variant is not None:
            raise ValueError("this model has no CutMax stage and no program variants")
        if not self.static:
            if self.pass_mode:
                raise NotStatic(
                    "token: under the pass kernels the stream must be static (prepare refuses otherwise)"
                )
            return super().token(t, rows, argmax, entry, variant)
        tt = time.time()
        self.stage_walls = []
        self._fence = self.profile and t % self.profile_every == 0
        n_blocks = self.rec.n_layers + 1
        info = []
        for b in range(n_blocks):
            if b == 0:
                x = self.input_ct(t, rows, self.progs[self.labels[0]]) if entry is None else entry
            else:
                x = self.out_reg(b - 1)
            info.append(self._run_block(b, t, {}, x))
        rec = {
            "issue_s": round(time.time() - tt, 3),
            "served": "".join(i["served"][0] for i in info),
            "fenced": self._fence,
            "stages": list(self.stage_walls),
        }
        return self.out_reg(n_blocks - 1), rec

    def reply(self, reg):
        """The reply register read: ``(rows, frame, level)`` with ``rows`` the ``(conversations, d)`` values when a
        secret is at hand (the session's own, or the client's through ``_pc.Client``), else None and the exported
        frame for the client; under the pass kernels the ``2t`` lane-slot rows in ``q`` order (``passmasks.from_layout``,
        the one place on this end). Synchronous (waits for the tick)."""
        from perseus import _pc  # type: ignore[attr-defined]

        ct = self.ct(reg)
        level = int(ct.level)
        if self.client is not None:
            frame = bytes(_pc.export_ct(self.sess, ct))
            vals = np.asarray(self.client.decrypt(frame)[0], dtype=np.complex128)
        elif self.bundle is None:
            if self.ops is None:
                raise RuntimeError("reply before setup")
            vals = self.ops.decrypt_slots_complex(ct)
        else:
            return None, bytes(_pc.export_ct(self.sess, ct)), level
        out = filling.unpack_pairs(vals, self.d, self.n_conv)
        return (passmasks.from_layout(out) if self.pass_mode else out), None, level

    # ── the record ──
    def counters(self) -> dict:
        """``planned_stats()`` and the device memory, JSON-ready; under ``generated`` the device diagonal encoder's
        counters too (``Session.diag_stats``)."""
        st = {k: (list(v) if isinstance(v, tuple) else v) for k, v in dict(self.sess.planned_stats()).items()}
        free, total = self.fhe.mem_info()
        out = {"planned_stats": st, "mem_free_bytes": int(free), "mem_total_bytes": int(total)}
        if self.diag_source == "generated":
            out["diag_stats"] = {
                k: (list(v) if isinstance(v, tuple) else v) for k, v in dict(self.sess.diag_stats()).items()
            }
        return out

    def probe_diagonals(self, k: int, seed: int = 0) -> dict:
        """The device diagonal encoder against the port's oracle (``Session.diag_probe``): ``k`` random diagonals of
        the declared tiles (at most the encoder's batch), their slot vectors from ``filling.diag_gather_reference``
        over the registered folded matrices, at the deepest diagonal sum's limbs and at half of them. Mismatch counts,
        all 0 when the encoder is right: the gathered slots equal the oracle vectors bit for bit, ``encode_many``
        equals one ``encode`` per job, and the eager ``DeviceFft`` encode of the oracle vector runs the same kernels
        on the same values."""
        tiles = self.diag_tiles()
        if not tiles:
            raise RuntimeError("probe_diagonals: diag_source generated only")
        st = dict(self.sess.diag_stats())
        picks = diaggen.probe_picks(self.model, tiles, min(int(k), int(st["batch"])), seed)
        names = [nm for nm, _p, _j in picks]
        vals = np.ascontiguousarray(
            np.stack([filling.diag_gather_reference(p.W, p.N, p.t, [j])[0] for _nm, p, j in picks]),
            dtype=np.float64,
        )
        out: dict[str, Any] = {"jobs": len(picks), "names": names[:4], "levels": {}}
        full = int(st["max_rows"])
        for limbs in sorted({full, max(1, full // 2)}, reverse=True):
            level = N32_Q_PRIMES - limbs
            r = self.sess.diag_probe(names, level, vals)
            out["levels"][str(level)] = {
                "limbs": limbs,
                "slots_vs_oracle": int(
                    np.count_nonzero(r["slots"].real != vals) + np.count_nonzero(r["slots"].imag)
                ),
                "generated_vs_single": int(np.count_nonzero(r["generated"] != r["single"])),
                "generated_vs_oracle": int(np.count_nonzero(r["generated"] != r["oracle"])),
                "words": int(r["generated"].size),
            }
        return out

    def program_report(self) -> dict:
        out = {}
        for lab, p in self.progs.items():
            row = {}
            for k in PROGRAM_FIELDS:
                v = getattr(p, k, None)
                row[k] = list(v) if isinstance(v, tuple) else v
            out[lab] = row
        return out
