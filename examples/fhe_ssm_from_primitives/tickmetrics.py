# SPDX-License-Identifier: MIT
"""Per-tick metrics of a ``gpu_run``, one JSON line per tick, written as the run goes
(``gpu_run --metrics run.jsonl``).

A tick's ``seconds`` is the wall time of ``model.step`` alone; when a tick is slow, this file
says where the time went and what the process and the box looked like at that moment. The
first line is a header (``kind: header``: the build identity, the environment and the load
state at the start); every other line is one tick (``kind: tick``), appended and flushed before
the next tick starts, so a cut-off run keeps every tick it finished. Per tick:

    t_start, t_end            wall clock (epoch) of the timed step, to align the box samplers
    seconds, wall_s           the driver's timer and the meter's own (they agree to the ms)
    cpu_s, thread_s           CPU time of the process and of the calling thread over the step: a
                              stall that burns CPU (a copy, a rehash, an encode) shows here, a wait
                              (a lock, the driver, a page-in) does not
    minflt, majflt            page faults: a reallocated or first-touched host buffer faults once per
                              page, a swapped-out one majors
    ctx_vol, ctx_invol        context switches (waits; preemption by another process)
    sched                     the scheduler's account of the process's threads: run time and the
                              time spent runnable but waiting for a CPU (a neighbour stealing cores)
    io                        bytes read and written by the process (a log or a report being written)
    gc                        Python's collector: collections per generation, their time, objects freed
    malloc                    glibc's arena (mallinfo2): heap and mmap'd bytes in use and free
    host                      the process's VmRSS / VmHWM / VmSwap / VmData / VmLck / VmPin / Threads,
                              the box's MemAvailable / SwapFree / Dirty, the load averages, the process
                              counts: the load state beside the number
    runtime                   the runtime's own counters over the step: bootstraps (its total and the
                              port's), encodes, the encode cache (hits / misses / size), device free GB
    driver                    what ImplModel.token_end measured (ring gaps, mask staging, decode_s)
    stages                    every stage (24 blocks and the tail): wall, CPU and faults inside it, so a
                              slow tick is charged to a block or to the gaps between blocks
    the report's fields       the fidelity numbers of the tick, as gpu_run's report has them

Linux ``/proc`` fields are None where the file does not exist (macOS); nothing here needs
the GPU, so the meter runs unchanged on the fake.
"""

from __future__ import annotations

import contextlib
import ctypes
import gc
import hashlib
import json
import os
import platform
import resource
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ENV_PREFIXES = (
    "CHAIN",
    "LOGN",
    "CUDA_VISIBLE_DEVICES",
    "FIDESLIB_",
    "FHE_",
    "SPARSE_",
    "LADDER",
    "OMP_",
    "MALLOC_",
    "KV_",
    "AUTO_BTS",
    "BTS_",
    "CKKS_",
    "CORRECTION",
    "COMPOSITE",
    "LEVEL_BUDGET",
    "FIRST_MOD",
    "SCALE_BITS",
    "BTP_",
    "NUM_LARGE",
    "H_WEIGHT",
    "IMPL_",
    "PLAN_",
    "FUSED_",
    "SM_",
)
STATUS_FIELDS = ("VmRSS", "VmHWM", "VmSwap", "VmData", "VmLck", "VmPin", "Threads")
MEMINFO_FIELDS = ("MemTotal", "MemFree", "MemAvailable", "Cached", "SwapTotal", "SwapFree", "Dirty")
PROC = Path("/proc")


# ── /proc readers (None off Linux) ──
def _kv_file(path: Path, fields: tuple[str, ...]) -> dict[str, int | None] | None:
    try:
        text = path.read_text()
    except OSError:
        return None
    out: dict[str, int | None] = dict.fromkeys(fields)
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        if key in out:
            tok = rest.split()
            out[key] = int(tok[0]) if tok else None
    return out


def proc_status(proc: Path = PROC, pid: str = "self") -> dict[str, int | None] | None:
    """VmRSS, VmHWM, VmSwap, VmData, VmLck, VmPin (kB) and Threads of a process."""
    return _kv_file(proc / pid / "status", STATUS_FIELDS)


def meminfo(proc: Path = PROC) -> dict[str, int | None] | None:
    return _kv_file(proc / "meminfo", MEMINFO_FIELDS)


def loadavg(proc: Path = PROC) -> dict[str, float] | None:
    try:
        tok = (proc / "loadavg").read_text().split()
        running, total = tok[3].split("/")
        return {
            "load1": float(tok[0]),
            "load5": float(tok[1]),
            "load15": float(tok[2]),
            "procs_running": int(running),
            "procs_total": int(total),
        }
    except (OSError, IndexError, ValueError):
        return None


def nprocs(proc: Path = PROC) -> int | None:
    try:
        return sum(1 for p in proc.iterdir() if p.name.isdigit())
    except OSError:
        return None


def proc_io(proc: Path = PROC) -> dict[str, int] | None:
    try:
        text = (proc / "self" / "io").read_text()
    except OSError:
        return None
    out = {}
    for line in text.splitlines():
        key, _, val = line.partition(":")
        if key in ("rchar", "wchar", "read_bytes", "write_bytes"):
            out[key] = int(val)
    return out or None


def schedstat(proc: Path = PROC) -> dict[str, int] | None:
    """Run time and run-queue wait time (ns) and time slices, summed over the process's threads."""
    task = proc / "self" / "task"
    try:
        tids = [p for p in task.iterdir() if p.name.isdigit()]
    except OSError:
        return None
    run = wait = slices = 0
    for t in tids:
        try:
            a, b, c = (t / "schedstat").read_text().split()[:3]
        except (OSError, ValueError):
            continue
        run, wait, slices = run + int(a), wait + int(b), slices + int(c)
    return {"run_ns": run, "wait_ns": wait, "timeslices": slices, "threads": len(tids)}


class _Mallinfo2(ctypes.Structure):
    _fields_ = [(n, ctypes.c_size_t) for n in ("arena", "ordblks", "smblks", "hblks", "hblkhd", "usmblks", "fsmblks", "uordblks", "fordblks", "keepcost")]  # fmt: skip


def mallinfo() -> dict[str, int] | None:
    """glibc's ``mallinfo2`` (bytes): arena (brk heap), hblkhd (mmap'd), uordblks (in use),
    fordblks (free in the heap), keepcost (trimmable top)."""
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.mallinfo2.restype = _Mallinfo2
    except (OSError, AttributeError):
        return None
    m = libc.mallinfo2()
    return {n: int(getattr(m, n)) for n in ("arena", "hblkhd", "uordblks", "fordblks", "keepcost")}


# ── the Python collector ──
class GcWatch:
    """Counts and times the collector's runs through ``gc.callbacks``."""

    def __init__(self):
        self.reset()
        self._t0: dict[int, float] = {}
        gc.callbacks.append(self._cb)

    def reset(self):
        self.n = [0, 0, 0]
        self.s = [0.0, 0.0, 0.0]
        self.collected = 0

    def _cb(self, phase: str, info: dict):
        g = int(info.get("generation", 0))
        if phase == "start":
            self._t0[g] = time.perf_counter()
        elif phase == "stop":
            self.n[g] += 1
            self.s[g] += time.perf_counter() - self._t0.pop(g, time.perf_counter())
            self.collected += int(info.get("collected", 0))

    def snapshot(self) -> dict[str, Any]:
        return {
            "gen0": self.n[0],
            "gen1": self.n[1],
            "gen2": self.n[2],
            "gen0_s": round(self.s[0], 4),
            "gen1_s": round(self.s[1], 4),
            "gen2_s": round(self.s[2], 4),
            "collected": self.collected,
            "counts": list(gc.get_count()),
            "tracked": len(gc.get_objects()) if gc.get_count()[2] == 0 else None,
        }

    def close(self):
        if self._cb in gc.callbacks:
            gc.callbacks.remove(self._cb)


# ── identity ──
def git_head(root: Path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=False, timeout=10).stdout.strip() or None  # fmt: skip
    except (OSError, subprocess.SubprocessError):
        return None


def core_identity(core) -> dict[str, Any]:
    out: dict[str, Any] = {"module": getattr(core, "__name__", str(core))}
    f = getattr(core, "__file__", None)
    if f and Path(f).is_file():
        out["file"] = f
        out["sha256"] = hashlib.sha256(Path(f).read_bytes()).hexdigest()
    info = getattr(core, "build_info", None)
    if callable(info):
        with contextlib.suppress(RuntimeError, TypeError):
            out["build_info"] = dict(info())
    return out


def nvidia_smi(args: list[str]) -> list[str] | None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        r = subprocess.run([exe, *args], capture_output=True, text=True, check=False, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip().splitlines() if r.returncode == 0 else None


def _counter(obj, name):
    v = getattr(obj, name, None)
    try:
        return None if v is None else int(v)
    except (TypeError, ValueError):
        return None


# ── the meter ──
class TickMeter:
    """``attach(model)`` wraps the model's stages with timers; ``begin(pos)`` and ``end(pos, ...)``
    bracket the timed step; ``close()`` detaches. One JSON line per tick goes to ``path``."""

    def __init__(self, path, *, core=None, root: Path | None = None, run: dict | None = None):
        self.path = Path(path)
        self.f = self.path.open("a", encoding="utf-8")
        self.core = core
        self.gc = GcWatch()
        self.model: Any = None
        self._stages: list[dict] = []
        self._b: dict[str, Any] = {}
        self.write({"kind": "header", **self.header(root, run or {})})

    def header(self, root: Path | None, run: dict) -> dict[str, Any]:
        root = root or Path(__file__).resolve().parents[2]
        env = {k: v for k, v in sorted(os.environ.items()) if k.startswith(ENV_PREFIXES)}
        return {
            "t": time.time(),
            "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "pid": os.getpid(),
            "cpu_count": os.cpu_count(),
            "argv": sys.argv,
            "cwd": os.getcwd(),
            "git_head": git_head(root),
            "core": core_identity(self.core) if self.core is not None else None,
            "env": env,
            "run": run,
            "nvidia_smi": nvidia_smi(
                ["--query-gpu=index,name,driver_version,memory.total,uuid", "--format=csv,noheader"]
            ),
            "gpu_processes": nvidia_smi(
                ["--query-compute-apps=pid,used_memory", "--format=csv,noheader"]
            ),
            "meminfo": meminfo(),
            "loadavg": loadavg(),
            "nprocs": nprocs(),
            "status": proc_status(),
        }

    # the stage timers
    def attach(self, model):
        self.model = model
        orig = model.token_stages

        def token_stages(x, pos):
            return [(lab, pre, self._timed(lab, fn)) for lab, pre, fn in orig(x, pos)]

        model.token_stages = token_stages
        return self

    def _timed(self, label, fn):
        def compute(h):
            t0, c0, th0 = time.time(), time.process_time(), time.thread_time()
            r0 = resource.getrusage(resource.RUSAGE_SELF)
            h = fn(h)
            r1 = resource.getrusage(resource.RUSAGE_SELF)
            self._stages.append({
                "label": label,
                "wall_s": round(time.time() - t0, 4),
                "cpu_s": round(time.process_time() - c0, 4),
                "thread_s": round(time.thread_time() - th0, 4),
                "minflt": r1.ru_minflt - r0.ru_minflt,
                "majflt": r1.ru_majflt - r0.ru_majflt,
                "ctx_vol": r1.ru_nvcsw - r0.ru_nvcsw,
                "ctx_invol": r1.ru_nivcsw - r0.ru_nivcsw,
            })  # fmt: skip
            return h

        return compute

    def _runtime(self) -> dict[str, Any]:
        m = self.model
        if m is None:
            return {}
        ops = getattr(getattr(m, "rt", None), "ops", None)
        fhe = getattr(getattr(m, "inf", None), "fhe", None)
        out: dict[str, Any] = {
            "total_bootstraps": _counter(fhe, "total_bootstraps"),
            "n_bootstraps": _counter(ops, "n_bootstraps"),
            "n_encodes": _counter(ops, "n_encodes"),
        }
        stats = getattr(ops, "enc_cache_stats", None)
        if callable(stats):
            with contextlib.suppress(RuntimeError, TypeError):
                out["enc_cache"] = {
                    k: v for k, v in dict(stats()).items() if isinstance(v, (int, float))
                }
        free = getattr(self.core, "device_free_gb", None)
        if callable(free):
            with contextlib.suppress(RuntimeError, TypeError):
                out["device_free_gb"] = round(float(free()), 2)
        return out

    def begin(self, pos: int):
        self._stages = []
        self.gc.reset()
        self._b = {
            "pos": pos,
            "t": time.time(),
            "cpu": time.process_time(),
            "thr": time.thread_time(),
            "ru": resource.getrusage(resource.RUSAGE_SELF),
            "io": proc_io(),
            "sched": schedstat(),
            "runtime": self._runtime(),
        }

    def end(
        self, pos: int, seconds: float, driver: dict | None = None, row: dict | None = None
    ) -> dict[str, Any]:
        b, now = self._b, time.time()
        ru0, ru1 = b["ru"], resource.getrusage(resource.RUSAGE_SELF)
        rt0, rt1 = b["runtime"], self._runtime()
        rec: dict[str, Any] = {
            "kind": "tick",
            "pos": pos,
            "t_start": b["t"],
            "t_end": now,
            "seconds": seconds,
            "wall_s": round(now - b["t"], 3),
            "cpu_s": round(time.process_time() - b["cpu"], 3),
            "thread_s": round(time.thread_time() - b["thr"], 3),
            "minflt": ru1.ru_minflt - ru0.ru_minflt,
            "majflt": ru1.ru_majflt - ru0.ru_majflt,
            "ctx_vol": ru1.ru_nvcsw - ru0.ru_nvcsw,
            "ctx_invol": ru1.ru_nivcsw - ru0.ru_nivcsw,
            "maxrss_kb": ru1.ru_maxrss if sys.platform != "darwin" else ru1.ru_maxrss // 1024,
            "io": _delta(b["io"], proc_io()),
            "sched": _delta(b["sched"], schedstat(), keep=("threads",)),
            "gc": self.gc.snapshot(),
            "malloc": mallinfo(),
            "host": {
                **(proc_status() or {}),
                **(meminfo() or {}),
                **(loadavg() or {}),
                "nprocs": nprocs(),
            },
            "runtime": {
                **{
                    k: _sub(rt1.get(k), rt0.get(k))
                    for k in ("total_bootstraps", "n_bootstraps", "n_encodes")
                },
                "enc_cache": rt1.get("enc_cache"),
                "enc_cache_delta": _delta(rt0.get("enc_cache"), rt1.get("enc_cache")),
                "device_free_gb_start": rt0.get("device_free_gb"),
                "device_free_gb_end": rt1.get("device_free_gb"),
            },
            "driver": {k: v for k, v in (driver or {}).items() if k not in ("out", "pos")},
            "stages": self._stages,
        }
        if row:
            rec.update(
                {
                    k: v
                    for k, v in row.items()
                    if k not in ("tokens", "top1", "pick", "pos", "seconds")
                }
            )
        self.write(rec)
        return rec

    def write(self, rec: dict):
        self.f.write(json.dumps(rec, default=str) + "\n")
        self.f.flush()
        os.fsync(self.f.fileno())

    def close(self):
        self.gc.close()
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def _sub(a, b):
    return None if a is None or b is None else a - b


def _delta(before: dict | None, after: dict | None, keep: tuple[str, ...] = ()) -> dict | None:
    if before is None or after is None:
        return None
    out = {
        k: (after[k] - before[k])
        for k in after
        if k in before and k not in keep and isinstance(after[k], (int, float))
    }
    out.update({k: after[k] for k in keep if k in after})
    return out


def read_metrics(path) -> tuple[dict, list[dict]]:
    """(header, ticks) of a metrics file."""
    header, ticks = {}, []
    with Path(path).open(encoding="utf-8") as f:
        for raw in f:
            if raw.strip():
                rec = json.loads(raw)
                if rec.get("kind") == "header":
                    header = rec
                elif rec.get("kind") == "tick":
                    ticks.append(rec)
    return header, ticks
