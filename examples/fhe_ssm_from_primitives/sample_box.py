# SPDX-License-Identifier: MIT
"""Sample the box beside a run: the GPU every second, the host every second, the other
processes every few seconds, each to its own file under ``--out`` as it goes.

    python -m examples.fhe_ssm_from_primitives.sample_box --out DIR --gpu 0 [--pid PID] \\
        [--interval 1] [--ps-every 5] [--duration S]

    gpu.csv     nvidia-smi's own stream (``-lms``), one row per second, an ``epoch`` column
                prepended as each row arrives: memory used and reserved, utilisation, SM and
                memory clocks, temperature, power, the active clock-event (throttle) reasons; the
                field list is probed first and falls back to what this nvidia-smi accepts
    host.csv    ``/proc``: MemAvailable / MemFree / SwapFree / Cached / Dirty, the load averages,
                the running and total process counts, and with ``--pid`` that process's VmRSS /
                VmHWM / VmSwap / VmData / VmLck / VmPin / Threads
    procs.txt   every ``--ps-every`` seconds: the top CPU consumers (``ps``) and the processes
                holding the GPU (``nvidia-smi --query-compute-apps``), each block stamped with
                the epoch: the neighbours a timing must be read beside

Stops on SIGTERM / SIGINT or after ``--duration`` seconds. ``--no-gpu`` and ``--ps-every 0``
leave those files out (a box without nvidia-smi, or a test); ``--proc-root`` points the host
sampler at a copy of ``/proc`` for tests.
"""

from __future__ import annotations

import argparse
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from .tickmetrics import MEMINFO_FIELDS, STATUS_FIELDS, loadavg, meminfo, nprocs, proc_status

GPU_FIELD_SETS = (
    "timestamp,index,memory.used,memory.reserved,utilization.gpu,utilization.memory,clocks.sm,clocks.mem,temperature.gpu,power.draw,clocks_event_reasons.active",
    "timestamp,index,memory.used,memory.reserved,utilization.gpu,utilization.memory,clocks.sm,clocks.mem,temperature.gpu,power.draw,clocks_throttle_reasons.active",
    "timestamp,index,memory.used,utilization.gpu,utilization.memory,clocks.sm,temperature.gpu,power.draw",
    "timestamp,index,memory.used,utilization.gpu",
)
HOST_COLUMNS = ("epoch", *(f"{k}_kB" for k in MEMINFO_FIELDS), "load1", "load5", "load15", "procs_running", "procs_total", "nprocs", *(f"pid_{k}_kB" if k != "Threads" else "pid_Threads" for k in STATUS_FIELDS))  # fmt: skip


def gpu_fields(exe: str, gpu: str) -> str | None:
    """The first field set this nvidia-smi accepts for this GPU."""
    for fields in GPU_FIELD_SETS:
        r = subprocess.run(
            [exe, "-i", gpu, f"--query-gpu={fields}", "--format=csv"],
            capture_output=True,
            text=True,
            check=False,
        )
        if r.returncode == 0 and r.stdout.count("\n") >= 2:
            return fields
    return None


def stream_gpu(
    exe: str, gpu: str, fields: str, interval_ms: int, out: Path, stop: threading.Event
) -> None:
    """nvidia-smi's own loop, each row stamped with the epoch at which it arrived."""
    cmd = [exe, "-i", gpu, f"--query-gpu={fields}", "--format=csv", f"-lms={interval_ms}"]
    with (
        out.open("w", encoding="utf-8") as f,
        subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True) as p,
    ):
        assert p.stdout is not None
        first = True
        for raw in p.stdout:
            line = raw.rstrip("\n")
            if not line:
                continue
            f.write(("epoch, " if first else f"{time.time():.3f}, ") + line + "\n")
            f.flush()
            first = False
            if stop.is_set():
                p.terminate()
                break


def host_row(proc: Path, pid: str | None) -> str:
    mi, la, st = (
        meminfo(proc) or {},
        loadavg(proc) or {},
        (proc_status(proc, pid) if pid else None) or {},
    )
    vals: list[object] = [f"{time.time():.3f}", *(mi.get(k) for k in MEMINFO_FIELDS)]
    vals += [la.get(k) for k in ("load1", "load5", "load15", "procs_running", "procs_total")]
    vals += [nprocs(proc), *(st.get(k) for k in STATUS_FIELDS)]
    return ",".join("" if v is None else str(v) for v in vals)


def ps_block(gpu_exe: str | None) -> str:
    lines = [f"=== {time.time():.3f} {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}"]
    try:
        r = subprocess.run(
            ["ps", "-eo", "pid,pcpu,pmem,rss,etimes,comm", "--sort=-pcpu"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        lines += r.stdout.splitlines()[:14] if r.returncode == 0 else [f"ps rc={r.returncode}"]
    except (OSError, subprocess.SubprocessError) as e:
        lines.append(f"ps failed: {e}")
    if gpu_exe:
        try:
            r = subprocess.run(
                [gpu_exe, "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
            lines += ["gpu processes:", *(r.stdout.splitlines() or ["(none)"])]
        except (OSError, subprocess.SubprocessError) as e:
            lines.append(f"nvidia-smi apps failed: {e}")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", default="0", help="nvidia-smi index of the card to sample")
    ap.add_argument("--pid", default=None, help="the run's process, for its /proc/<pid>/status")
    ap.add_argument(
        "--interval", type=float, default=1.0, help="seconds between GPU and host samples"
    )
    ap.add_argument(
        "--ps-every", type=float, default=5.0, help="seconds between process listings (0: none)"
    )
    ap.add_argument("--duration", type=float, default=None, help="stop after this many seconds")
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--proc-root", default="/proc")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    proc = Path(a.proc_root)
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    exe = None if a.no_gpu else shutil.which("nvidia-smi")
    threads = []
    if exe:
        fields = gpu_fields(exe, a.gpu)
        if fields is None:
            print(
                "sample_box: nvidia-smi accepts none of the field sets; no gpu.csv", file=sys.stderr
            )
        else:
            print(f"sample_box: gpu fields {fields}", file=sys.stderr)
            t = threading.Thread(
                target=stream_gpu,
                args=(exe, a.gpu, fields, int(a.interval * 1000), out / "gpu.csv", stop),
                daemon=True,
            )
            t.start()
            threads.append(t)
    elif not a.no_gpu:
        print("sample_box: no nvidia-smi on PATH; no gpu.csv", file=sys.stderr)
    t_end = None if a.duration is None else time.time() + a.duration
    next_ps = time.time()
    with (
        (out / "host.csv").open("w", encoding="utf-8") as host,
        (out / "procs.txt").open("w", encoding="utf-8") as procs,
    ):
        host.write(",".join(HOST_COLUMNS) + "\n")
        while not stop.is_set() and (t_end is None or time.time() < t_end):
            host.write(host_row(proc, a.pid) + "\n")
            host.flush()
            if a.ps_every > 0 and time.time() >= next_ps:
                procs.write(ps_block(exe))
                procs.flush()
                next_ps = time.time() + a.ps_every
            stop.wait(a.interval)
    stop.set()
    for t in threads:
        t.join(timeout=5)
    return 0


if __name__ == "__main__":
    sys.exit(main())
