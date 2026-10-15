# SPDX-License-Identifier: MIT
"""The per-tick meter (tickmetrics.py) on the fake through ``gpu_run --metrics``, the blip
analysis (blips.py) on a synthetic run with planted slow ticks and a synthetic nvidia-smi
stream, and the box sampler's host readers on a copy of ``/proc``."""

import contextlib
import csv
import json
import math
from typing import Any

import numpy as np
import pytest
from examples.fhe_ssm_from_primitives import blips, gpu_run, sample_box, tickmetrics
from examples.fhe_ssm_from_primitives.synthetic import synthetic


def checkpoint(tmp_path):
    _params, w, seeds, cfg = synthetic(L=2, d=32, dff=128, seed=51)
    np.savez(tmp_path / "w.npz", **w)
    (tmp_path / "s.json").write_text(json.dumps(seeds))
    (tmp_path / "c.json").write_text(json.dumps(cfg))
    return ["--weights", str(tmp_path / "w.npz"), "--seeds", str(tmp_path / "s.json"),
            "--config", str(tmp_path / "c.json")]  # fmt: skip


@pytest.fixture
def small_fake(monkeypatch):
    """The fake at the synthetic checkpoint's width (as test_run_and_plan.py)."""
    from examples.fhe_ssm_from_primitives import filling
    from examples.fhe_ssm_from_primitives.runtime_fake import make_runtime_fake

    from perseus.impl import fake

    def session(params, payload):
        _fhe, inf = make_runtime_fake(
            N=1024, hid=params.d, dim=params.d, H=1, H_real=1, E=params.d_ffn,
            E_real=params.d_ffn, rot_steps=filling.rotation_steps(1024, params.d),
            complex_payload=payload == "pairs",
        )  # fmt: skip
        return contextlib.nullcontext(inf), fake.core

    monkeypatch.setattr(gpu_run, "fake_session", session)


# ── the meter ──
@pytest.mark.usefixtures("small_fake")
def test_metrics_file_has_a_header_and_one_line_per_tick_with_the_stages(tmp_path):
    out, met = tmp_path / "run.json", tmp_path / "run.jsonl"
    argv = [
        *checkpoint(tmp_path),
        "--fake",
        "--ticks",
        "3",
        "--out",
        str(out),
        "--metrics",
        str(met),
    ]
    assert gpu_run.main(argv) == 0
    header, ticks = tickmetrics.read_metrics(met)
    assert header["kind"] == "header"
    assert header["run"]["ticks"] == 3
    assert header["run"]["fake"] is True
    assert "LADDER_RADIX_BITS" in header["env"]  # the port's own switch, exported before the fake
    rep = json.loads(out.read_text())
    assert rep["metrics"] == str(met)
    assert [t["pos"] for t in ticks] == [0, 1, 2]
    for t, r in zip(ticks, rep["ticks"], strict=True):
        assert t["seconds"] == pytest.approx(r["seconds"], abs=0.01)
        assert t["t_start"] < t["t_end"]
        assert t["wall_s"] == pytest.approx(t["t_end"] - t["t_start"], abs=0.002)
        assert t["cpu_s"] >= 0
        assert math.isfinite(t["cpu_s"])
        assert t["minflt"] >= 0
        assert t["majflt"] >= 0
        assert [s["label"] for s in t["stages"]] == ["blk0", "blk1", "tail"]
        assert sum(s["wall_s"] for s in t["stages"]) <= t["wall_s"] + 0.01
        assert t["runtime"]["n_bootstraps"] == r["bootstraps"]
        assert t["gc"]["gen0"] >= 0
        assert t["logits_relErrRms_vs_exact"] == r["logits_relErrRms_vs_exact"]
        assert "tokens" not in t  # the token lists stay in the report
        assert set(t["driver"]) >= {"decode_s", "cpu_s", "minflt"}  # ImplModel.token_end's numbers


@pytest.mark.usefixtures("small_fake")
def test_metrics_are_appended_as_the_run_goes(tmp_path):
    """A run cut after any tick leaves that tick on disk: every line is complete JSON and the
    file grows by one line per tick (no buffering to the end)."""
    out, met = tmp_path / "run.json", tmp_path / "run.jsonl"
    seen = []
    orig = gpu_run.write_json

    def spy(path, doc):
        orig(path, doc)
        seen.append(sum(1 for line in met.read_text().splitlines() if line.strip()))

    gpu_run.write_json = spy
    try:
        assert (
            gpu_run.main(
                [
                    *checkpoint(tmp_path),
                    "--fake",
                    "--ticks",
                    "2",
                    "--out",
                    str(out),
                    "--metrics",
                    str(met),
                ]
            )
            == 0
        )
    finally:
        gpu_run.write_json = orig
    assert seen == [2, 3]  # header + 1 tick after tick 0, header + 2 after tick 1


def test_proc_readers_return_none_off_linux_and_parse_a_copy(tmp_path):
    proc = tmp_path / "proc"
    (proc / "self" / "task" / "7").mkdir(parents=True)
    (proc / "self" / "status").write_text(
        "Name:\tpython\nVmRSS:\t  1234 kB\nVmSwap:\t     0 kB\nThreads:\t5\n"
    )
    (proc / "meminfo").write_text(
        "MemTotal:  263000000 kB\nMemAvailable:  200000000 kB\nSwapFree:  0 kB\n"
    )
    (proc / "loadavg").write_text("1.26 1.35 1.51 2/1234 99999\n")
    (proc / "self" / "io").write_text("rchar: 10\nwchar: 20\nread_bytes: 0\nwrite_bytes: 4096\n")
    (proc / "self" / "task" / "7" / "schedstat").write_text("1000 200 3\n")
    for d in ("12", "34"):
        (proc / d).mkdir()
    st = tickmetrics.proc_status(proc)
    mi = tickmetrics.meminfo(proc)
    assert mi is not None
    assert st == {
        "VmRSS": 1234,
        "VmHWM": None,
        "VmSwap": 0,
        "VmData": None,
        "VmLck": None,
        "VmPin": None,
        "Threads": 5,
    }
    assert mi["MemAvailable"] == 200000000
    assert tickmetrics.loadavg(proc) == {
        "load1": 1.26,
        "load5": 1.35,
        "load15": 1.51,
        "procs_running": 2,
        "procs_total": 1234,
    }
    assert tickmetrics.nprocs(proc) == 2
    assert tickmetrics.proc_io(proc) == {
        "rchar": 10,
        "wchar": 20,
        "read_bytes": 0,
        "write_bytes": 4096,
    }
    assert tickmetrics.schedstat(proc) == {
        "run_ns": 1000,
        "wait_ns": 200,
        "timeslices": 3,
        "threads": 1,
    }
    assert tickmetrics.proc_status(tmp_path / "nowhere") is None


def test_gc_watch_counts_a_forced_collection():
    w = tickmetrics.GcWatch()
    try:
        import gc

        gc.collect()
        s = w.snapshot()
        assert s["gen2"] >= 1
        assert s["gen2_s"] >= 0
    finally:
        w.close()


# ── the analysis ──
def synthetic_run(
    tmp_path, ticks=80, blip_at=((16, 1.2), (34, 1.9), (70, 4.1)), seed=5, cadence=2.0
):
    """A report of ``ticks`` ticks at 46.3 s with 0.1 s noise, the warm-up tick slow, the listed
    ticks slower by the listed excess, and a planted metric (``minflt``) that spikes on them;
    beside it an nvidia-smi stream at a fixed cadence over the whole timeline, idle in the host
    phase between ticks (1.8 s) and, on a blip, for the excess in the middle of the tick.
    Returns the report, the stream, the start time and the ticks whose stall the stream saw."""
    rng = np.random.default_rng(seed)
    base = 46.3 + rng.normal(0, 0.1, ticks)
    base[0] = 58.7
    excess = dict(blip_at)
    start, session_open, host_phase = 1_700_000_000.0, 40.0, 1.8
    rows: list[dict[str, Any]] = []
    stalls: list[tuple[int, float, float]] = []
    t = session_open
    for p in range(ticks):
        s = round(float(base[p] + excess.get(p, 0.0)), 2)
        rows.append({"pos": p, "seconds": s, "minflt": int(500 + rng.integers(0, 50) + (40000 if p in excess else 0)),
                     "free_gb": 59.9, "runtime_bootstraps": 953, "tokens": [1, 2], "top1": [1, 2], "pick": [1, 2]})  # fmt: skip
        if p in excess:
            stalls.append((p, t + 0.4 * s, t + 0.4 * s + excess[p]))
        t += s + host_phase
    end = t
    busy: list[tuple[float, float]] = []  # [t0, t1) the GPU is busy: each tick minus its stall
    t = session_open
    for r in rows:
        s = float(r["seconds"])
        st = next(((a, b) for p, a, b in stalls if p == r["pos"]), None)
        busy += [(t, t + s)] if st is None else [(t, st[0]), (st[1], t + s)]
        t += s + host_phase
    samples: list[tuple[float, int, int]] = []
    seen: set[int] = set()
    n_stall = 0
    for k in range(int(end / cadence) + 2):
        ts = k * cadence
        util = 98 if any(a <= ts < b for a, b in busy) else 0
        samples.append((start + ts, 35961, util))
        hit = [p for p, a, b in stalls if a <= ts < b]
        seen.update(hit)
        n_stall += len(hit)
    report = tmp_path / "run.json"
    report.write_text(json.dumps({"mode": "planned", "ticks": rows}))
    nv = tmp_path / "nvsmi.csv"
    with nv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", " memory.used [MiB]", " utilization.gpu [%]"])
        for ts, mem, util in samples:
            w.writerow([f"{ts:.3f}", f" {mem} MiB", f" {util} %"])
    return report, nv, start, seen, n_stall


def test_blips_are_found_with_their_pattern_and_the_planted_metric(tmp_path):
    report, nv, start, seen, n_stall = synthetic_run(tmp_path)
    assert seen >= {70}  # a 4.1 s stall cannot hide between 2-second samples
    out = tmp_path / "out"
    iso = blips.dt.datetime.fromtimestamp(start, tz=blips.dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    assert (
        blips.main(
            [
                "--report",
                str(report),
                "--nvsmi",
                str(nv),
                "--start",
                iso,
                "--label",
                "syn",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    res = json.loads((out / "syn_blips.json").read_text())
    a = res["analysis"]
    assert [b["pos"] for b in a["blips"]] == [16, 34, 70]
    assert a["warmup_tick0_s"] == 58.7
    assert a["pattern"]["gaps"] == [18, 36]
    assert a["pattern"]["doubling"]["c"] == 2
    assert a["pattern"]["excess_vs_tick"]["slope"] == pytest.approx(0.054, abs=0.01)
    by = {m["metric"]: m for m in a["metrics"]}
    assert all(v["z"] > 50 and v["pct"] == 100.0 for v in by["minflt"]["at_blips"].values())
    assert by["runtime_bootstraps"]["constant"]
    assert all(v["z"] == 0.0 for v in by["runtime_bootstraps"]["at_blips"].values())
    # the between-tick idles are placed between ticks by the window fit, and the stalls the
    # stream saw are attributed to their ticks, nothing else
    wf = res["window_fit"]
    assert wf["idle_samples_between_ticks"] == wf["idle_samples"] - n_stall
    assert wf["overhead_per_tick_s"] == pytest.approx(1.8, abs=0.3)
    mids = {m["pos"] for m in res["gpu_idles_mid_tick"] if m["pos"] > 0}
    assert mids == seen
    assert by["gpu.utilization.gpu.min"]["at_blips"]["70"]["value"] == 0
    md = (out / "syn_blips.md").read_text()
    assert "doubling offset c = 2" in md
    assert "- tick 70:" in md


def test_blips_from_a_metrics_file_use_its_windows(tmp_path):
    """With per-tick timestamps no fit is needed and the stage columns join the table."""
    met = tmp_path / "run.jsonl"
    start = 1_700_000_000.0
    with met.open("w") as f:
        f.write(json.dumps({"kind": "header", "git_head": "abc"}) + "\n")
        t = start
        for p in range(40):
            s = 46.3 + (2.0 if p == 16 else 0.0) + 0.01 * (p % 3)
            stages = [
                {
                    "label": f"blk{b}",
                    "wall_s": 1.9 + (2.0 if (p == 16 and b == 5) else 0.0),
                    "cpu_s": 0.1,
                }
                for b in range(24)
            ]
            f.write(
                json.dumps(
                    {
                        "kind": "tick",
                        "pos": p,
                        "t_start": t,
                        "t_end": t + s,
                        "seconds": s,
                        "cpu_s": 3.0,
                        "stages": stages,
                        "host": {"VmRSS_kB": 1000 + p},
                    }
                )
                + "\n"
            )
            t += s + 1.8
    out = tmp_path / "out"
    assert blips.main(["--metrics", str(met), "--label", "m", "--out", str(out)]) == 0
    res = json.loads((out / "m_blips.json").read_text())
    assert res["header"]["git_head"] == "abc"
    assert "window_fit" not in res
    a = res["analysis"]
    assert [b["pos"] for b in a["blips"]] == [16]
    by = {m["metric"]: m for m in a["metrics"]}
    assert by["stage.blk5.wall_s"]["at_blips"]["16"]["z"] > 50
    assert by["stage.argmax"]["at_blips"]["16"]["value"] == 5
    assert by["host.VmRSS_kB"]["spearman_vs_seconds"] == pytest.approx(0.0, abs=0.4)


def test_spearman_and_ranks():
    x = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
    assert blips.spearman(x, x**2) == pytest.approx(1.0)
    assert blips.spearman(x, -x) == pytest.approx(-1.0)
    assert blips.rankdata(np.asarray([3.0, 1.0, 3.0])).tolist() == [2.5, 1.0, 2.5]
    assert math.isnan(blips.spearman(x, np.ones(5)))


# ── the sampler ──
def test_sample_box_host_file_on_a_proc_copy(tmp_path):
    proc = tmp_path / "proc"
    (proc / "4242").mkdir(parents=True)
    (proc / "4242" / "status").write_text("VmRSS:\t 2048 kB\nVmHWM:\t 4096 kB\nThreads:\t3\n")
    (proc / "meminfo").write_text("MemTotal: 10 kB\nMemAvailable: 8 kB\nSwapFree: 1 kB\n")
    (proc / "loadavg").write_text("0.5 0.4 0.3 1/50 777\n")
    out = tmp_path / "samples"
    rc = sample_box.main(
        [
            "--out",
            str(out),
            "--no-gpu",
            "--ps-every",
            "0",
            "--proc-root",
            str(proc),
            "--pid",
            "4242",
            "--interval",
            "0.05",
            "--duration",
            "0.3",
        ]
    )
    assert rc == 0
    with (out / "host.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) >= 3
    r = rows[0]
    assert r["MemAvailable_kB"] == "8"
    assert r["SwapFree_kB"] == "1"
    assert r["load1"] == "0.5"
    assert r["procs_total"] == "50"
    assert r["nprocs"] == "1"
    assert r["pid_VmRSS_kB"] == "2048"
    assert r["pid_Threads"] == "3"
    assert r["pid_VmSwap_kB"] == ""  # absent in the copy: an empty cell, not a crash
    assert float(r["epoch"]) > 1_600_000_000
    assert not (out / "gpu.csv").exists()
    # the analysis reads the host file by its epoch column
    times, cols = blips.load_samples(out / "host.csv", "epoch")
    assert len(times) == len(rows)
    assert cols["pid_VmRSS_kB"][0] == 2048
