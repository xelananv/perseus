# SPDX-License-Identifier: MIT
"""The slow ticks of a run ("blips") against every metric sampled beside them.

A blip is a tick whose ``seconds`` sits more than ``--threshold`` robust standard deviations
(1.4826 x the median absolute deviation) above the run's median; ticks between ``--candidate``
and the threshold are listed as candidates. Tick 0 (the warm-up) is reported apart. For every
other per-tick metric the script gives its value at each blip tick as a robust z-score against
the non-blip ticks, its Spearman correlation with ``seconds`` over the whole run, and, for the
stage timings, which stage carries the excess; the ``nvidia-smi`` and host samples taken beside
the run are cut into the ticks' windows and summarised per tick, so they join the same table.

    python -m examples.fhe_ssm_from_primitives.blips --report run.json [--metrics run.jsonl] \\
        [--nvsmi nvsmi.csv] [--host host.csv] [--start 2026-10-13T18:02:35Z] [--label NAME] --out DIR

``--report`` is a ``gpu_run`` report; ``--metrics`` the per-tick JSONL ``gpu_run --metrics``
writes (tickmetrics.py), which carries each tick's wall-clock window; without it the windows
are fitted to the report's seconds and ``--start`` (the run's start time, from its ``.meta``),
with a constant per-tick overhead outside the timer chosen so that the GPU-idle samples fall
between ticks. Writes ``<label>_blips.md``, ``<label>_blips.json`` and, with matplotlib,
``<label>_blips.png``. Numpy only; nothing of the runtime is imported.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np

SAMPLER_LAG_S = 2.5  # a sampler reports the previous interval: ignore samples this close to an edge
IDLE_UTIL = 30  # a GPU sample below this (%) counts as idle


# ── inputs ──
def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_time(s: str) -> float:
    """An ISO time (``2026-10-13T18:02:35Z``) or nvidia-smi's ``2026/10/13 18:02:36.033`` as
    a UTC epoch."""
    s = s.strip()
    for fmt in (
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S",
        "%Y/%m/%d %H:%M:%S.%f",
        "%Y/%m/%d %H:%M:%S",
    ):
        try:
            return dt.datetime.strptime(s, fmt).replace(tzinfo=dt.UTC).timestamp()
        except ValueError:
            continue
    raise ValueError(f"unrecognised time {s!r}")


def flatten(d: dict, prefix: str = "") -> dict[str, float]:
    """Nested dicts to dotted keys; numeric leaves only (lists of numbers become ``key.i``)."""
    out: dict[str, float] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        elif isinstance(v, list) and v and all(isinstance(x, (int, float)) for x in v):
            if len(v) <= 64:
                out.update({f"{key}.{i}": float(x) for i, x in enumerate(v)})
        elif isinstance(v, bool) or (isinstance(v, (int, float)) and math.isfinite(v)):
            out[key] = float(v)
    return out


def load_report(path: Path) -> dict[int, dict[str, float]]:
    doc = json.loads(path.read_text())
    rows = {}
    for t in doc.get("ticks", []):
        row = {k: v for k, v in t.items() if k not in ("tokens", "top1", "pick")}
        rows[int(t["pos"])] = flatten(row)
    return rows


def load_metrics(path: Path) -> tuple[dict, dict[int, dict[str, float]]]:
    """(header, per-tick rows) of a tickmetrics JSONL; a stage list becomes one column per
    stage label and per field (``stage.blk3.wall_s``)."""
    header: dict = {}
    rows: dict[int, dict[str, float]] = {}
    with path.open(encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("kind") == "header":
                header = rec
                continue
            if rec.get("kind") != "tick":
                continue
            stages = rec.pop("stages", None) or []
            row = flatten(rec)
            for st in stages:
                lab = st.get("label", "?")
                row.update(
                    {
                        f"stage.{lab}.{k}": float(v)
                        for k, v in st.items()
                        if isinstance(v, (int, float))
                    }
                )
            walls = [float(st.get("wall_s", 0.0)) for st in stages]
            if walls:
                row["stage.max_wall_s"] = max(walls)
                row["stage.argmax"] = float(int(np.argmax(walls)))
                row["stage.sum_wall_s"] = float(sum(walls))
            rows[int(rec["pos"])] = row
    return header, rows


def read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open(encoding="utf-8", newline="") as f:
        rows = [r for r in csv.reader(f) if r]
    names = [c.strip().split(" [")[0] for c in rows[0]]
    return names, rows[1:]


def number(cell: str) -> float:
    """``35961 MiB`` -> 35961; ``0x0000000000000000`` -> 0; ``[N/A]`` -> nan."""
    tok = cell.strip().split(" ")[0]
    if tok.startswith("0x"):
        return float(int(tok, 16))
    try:
        return float(tok)
    except ValueError:
        return math.nan


def load_samples(
    path: Path, time_col: str = "timestamp"
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """A sampler CSV as (epoch times, {column: values})."""
    names, rows = read_csv(path)
    time_col = "epoch" if "epoch" in names else time_col
    ti = names.index(time_col) if time_col in names else 0
    times: list[float] = []
    cols: dict[str, list[float]] = {n: [] for i, n in enumerate(names) if i != ti}
    for r in rows:
        if len(r) != len(names):
            continue
        cell = r[ti].strip()
        times.append(float(cell) if cell.replace(".", "", 1).isdigit() else parse_time(cell))
        for i, n in enumerate(names):
            if i != ti:
                cols[n].append(number(r[i]))
    return np.asarray(times), {n: np.asarray(v) for n, v in cols.items()}


# ── statistics ──
def robust(x: np.ndarray) -> tuple[float, float]:
    """(median, 1.4826 x MAD); the scale floors at 1e-9 so a constant series gives z = 0 or inf."""
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    return med, max(1.4826 * mad, 1e-9)


def base_scale(base: np.ndarray) -> tuple[float, float]:
    """(median, scale) of a reference distribution: 1.4826 x MAD, or the standard deviation when
    the MAD is zero (most values equal), or 0 when every value is the same (a constant)."""
    med = float(np.median(base))
    scale = 1.4826 * float(np.median(np.abs(base - med)))
    return med, (scale or float(np.std(base)))


def zscores(x: np.ndarray, base: np.ndarray) -> np.ndarray:
    med, scale = robust(base)
    return (x - med) / scale


def rankdata(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties share the mean rank)."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 4 or np.ptp(a[ok]) == 0 or np.ptp(b[ok]) == 0:
        return math.nan
    ra, rb = rankdata(a[ok]), rankdata(b[ok])
    return float(np.corrcoef(ra, rb)[0, 1])


def linear_fit(x: np.ndarray, y: np.ndarray) -> dict:
    if len(x) < 2:
        return {"slope": math.nan, "intercept": math.nan, "r2": math.nan}
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res, ss_tot = float(np.sum((y - pred) ** 2)), float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else math.nan
    return {"slope": float(slope), "intercept": float(intercept), "r2": r2}


def doubling_offset(ticks: list[int]) -> dict:
    """The constant c for which t + c doubles best from blip to blip (the signature of a
    container that doubles its capacity): the c in 0..8 with the smallest spread of
    log2((t2 + c) / (t1 + c)) over consecutive blips, with those ratios."""
    if len(ticks) < 2:
        return {"c": None, "ratios": [], "spread": math.nan}
    cands = []
    for c in range(9):
        lr = [math.log2((b + c) / (a + c)) for a, b in itertools.pairwise(ticks)]
        cands.append((max(lr) - min(lr), c, [round(2**r, 3) for r in lr]))
    spread, c, ratios = min(cands, key=lambda t: t[0])
    return {"c": c, "ratios": ratios, "spread": round(spread, 4)}


# ── tick windows ──
def windows_from_metrics(rows: dict[int, dict[str, float]]) -> dict[int, tuple[float, float]]:
    return {
        p: (r["t_start"], r["t_end"]) for p, r in rows.items() if "t_start" in r and "t_end" in r
    }


def fit_windows(
    seconds: dict[int, float], start: float, idle_t: np.ndarray
) -> tuple[dict[int, tuple[float, float]], dict]:
    """Windows for a run without per-tick timestamps: tick k starts at start + S + sum over
    earlier ticks of (seconds + g); S (the session open) and g (the host work outside the timer)
    are chosen on a grid so that as many GPU-idle samples as possible fall between ticks."""
    pos = sorted(seconds)
    secs = np.asarray([seconds[p] for p in pos])
    rel = idle_t - start
    rel = rel[rel > 300]  # after the warm-up
    cands = []
    for s0 in np.arange(20.0, 90.0, 0.5):
        for g in np.arange(0.5, 6.0, 0.05):
            starts = s0 + np.concatenate([[0.0], np.cumsum(secs[:-1] + g)])
            k = np.searchsorted(starts, rel, side="right") - 1
            k = np.clip(k, 0, len(pos) - 1)
            phase = rel - starts[k]
            inside = int(np.sum(phase >= secs[k] - SAMPLER_LAG_S))
            cands.append((inside, float(s0), float(g)))
    inside, s0, g = max(cands, key=lambda t: t[0])
    starts = s0 + np.concatenate([[0.0], np.cumsum(secs[:-1] + g)])
    win = {p: (start + st, start + st + seconds[p]) for p, st in zip(pos, starts)}
    return win, {
        "session_open_s": s0,
        "overhead_per_tick_s": g,
        "idle_samples_between_ticks": inside,
        "idle_samples": len(rel),
    }


def per_tick_samples(
    times: np.ndarray,
    cols: dict[str, np.ndarray],
    windows: dict[int, tuple[float, float]],
    prefix: str,
) -> tuple[dict[int, dict[str, float]], list[dict]]:
    """Each sampled column inside each tick's window as min / max / mean (and first-to-last
    delta); for the GPU utilisation also the idle samples inside the window, which are the
    mid-tick idles when the window excludes the between-tick host phase."""
    out: dict[int, dict[str, float]] = {}
    idles: list[dict] = []
    util = next((c for c in cols if c.startswith("utilization.gpu")), None)
    for p, (t0, t1) in windows.items():
        sel = (times >= t0 + SAMPLER_LAG_S) & (times <= t1)
        if not sel.any():
            continue
        row: dict[str, float] = {}
        for name, v in cols.items():
            x = v[sel]
            x = x[np.isfinite(x)]
            if len(x) == 0:
                continue
            key = f"{prefix}.{name}"
            row[f"{key}.min"], row[f"{key}.max"], row[f"{key}.mean"] = (
                float(x.min()),
                float(x.max()),
                float(x.mean()),
            )
            row[f"{key}.delta"] = float(x[-1] - x[0])
        if util is not None:
            u = cols[util][sel]
            idle = u < IDLE_UTIL
            row[f"{prefix}.idle_samples"] = float(idle.sum())
            for ts, uu in zip(times[sel][idle], u[idle]):
                idles.append(
                    {
                        "pos": p,
                        "t": float(ts),
                        "phase_s": round(float(ts - t0), 1),
                        "util": float(uu),
                    }
                )
        out[p] = row
    return out, idles


# ── the analysis ──
def analyse(table: dict[int, dict[str, float]], threshold: float, candidate: float) -> dict:
    pos = sorted(p for p in table if p >= 1 and "seconds" in table[p])
    secs = np.asarray([table[p]["seconds"] for p in pos])
    z = zscores(secs, secs)
    blips = [p for p, zz in zip(pos, z) if zz > threshold]
    cands = [p for p, zz in zip(pos, z) if candidate < zz <= threshold]
    med, scale = robust(secs)
    base_mask = np.asarray([p not in blips for p in pos])
    out: dict = {
        "ticks": len(pos),
        "warmup_tick0_s": table.get(0, {}).get("seconds"),
        "median_s": med,
        "robust_sd_s": scale,
        "normal_mean_s": float(secs[base_mask].mean()),
        "normal_sd_s": float(secs[base_mask].std()),
        "threshold": threshold,
        "blips": [
            {
                "pos": p,
                "seconds": table[p]["seconds"],
                "excess_s": round(table[p]["seconds"] - med, 2),
                "z": round(float(z[pos.index(p)]), 1),
            }
            for p in blips
        ],
        "candidates": [
            {
                "pos": p,
                "seconds": table[p]["seconds"],
                "excess_s": round(table[p]["seconds"] - med, 2),
                "z": round(float(z[pos.index(p)]), 1),
            }
            for p in cands
        ],
    }
    bt = np.asarray(blips, dtype=float)
    ex = np.asarray([table[p]["seconds"] - med for p in blips])
    out["pattern"] = {
        "gaps": [int(b - a) for a, b in itertools.pairwise(blips)],
        "doubling": doubling_offset(blips),
        "excess_vs_tick": linear_fit(bt, ex),
        "excess_ratios": [round(float(b / a), 2) for a, b in itertools.pairwise(ex)],
    }
    out["metrics"] = metric_table(table, pos, secs, blips, base_mask)
    return out


def metric_table(table, pos, secs, blips, base_mask) -> list[dict]:
    names = sorted({k for p in pos for k in table[p]} - {"seconds", "pos"})
    rows = []
    for name in names:
        v = np.asarray([table[p].get(name, math.nan) for p in pos], dtype=float)
        fin = np.isfinite(v)
        if fin.sum() < 10:
            continue
        base = v[fin & base_mask]
        if len(base) < 5:
            continue
        med, scale = base_scale(base)
        constant = scale == 0.0
        at = {}
        for b in blips:
            x = table[b].get(name)
            if x is None or not math.isfinite(x):
                continue
            # a constant base has no scale: a blip value off it is "differs" (z None), on it z = 0
            z = (0.0 if x == med else None) if constant else round((x - med) / scale, 1)
            at[b] = {"value": x, "z": z, "pct": round(float(np.mean(base <= x)) * 100, 1)}
        if not at:
            continue
        rows.append(
            {
                "metric": name,
                "base_median": med,
                "base_robust_sd": None if constant else scale,
                "constant": constant,
                "spearman_vs_seconds": round(spearman(secs[fin], v[fin]), 3),
                "at_blips": at,
                "max_abs_z_at_blips": max(
                    (math.inf if a["z"] is None else abs(a["z"])) for a in at.values()
                ),
            }
        )
    rows.sort(key=lambda r: (-r["max_abs_z_at_blips"], r["metric"]))
    return rows


# ── output ──
def fmt(x) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        if math.isnan(x):
            return "nan"
        return f"{x:.4g}" if abs(x) < 1e-2 or abs(x) >= 1e5 else f"{x:.3f}".rstrip("0").rstrip(".")
    return str(x)


def markdown(res: dict, inputs: dict, label: str) -> str:
    a = res["analysis"]
    lines = [f"## {label}: slow ticks against every sampled metric", ""]
    lines += [
        "Inputs (sha256):",
        *[f"- `{k}`: `{v['path']}` {v['sha256'][:16]}…" for k, v in inputs.items()],
        "",
    ]
    lines += [
        (
            f"Ticks 1..{a['ticks']}: median {fmt(a['median_s'])} s, robust sd {fmt(a['robust_sd_s'])} s, normal ticks "
            f"mean {fmt(a['normal_mean_s'])} s sd {fmt(a['normal_sd_s'])} s; tick 0 (warm-up) {fmt(a['warmup_tick0_s'])} s. "
            f"Blip = z > {a['threshold']}."
        ),
        "",
        "| tick | seconds | excess s | z |",
        "|---|---|---|---|",
        *[
            f"| {b['pos']} | {fmt(b['seconds'])} | {b['excess_s']:+.2f} | {b['z']} |"
            for b in a["blips"]
        ],
        "",
    ]
    if a["candidates"]:
        lines += [
            "Candidates ("
            + ", ".join(
                f"tick {c['pos']} {c['excess_s']:+.2f} s z {c['z']}" for c in a["candidates"]
            )
            + ").",
            "",
        ]
    pat = a["pattern"]
    lines += [
        (
            f"Pattern: gaps {pat['gaps']}; best doubling offset c = {pat['doubling']['c']} (ratios of (t + c): "
            f"{pat['doubling']['ratios']}, log2 spread {pat['doubling']['spread']}); excess vs tick: slope "
            f"{fmt(pat['excess_vs_tick']['slope'])} s per tick, intercept {fmt(pat['excess_vs_tick']['intercept'])} s, "
            f"R² {fmt(pat['excess_vs_tick']['r2'])}; excess ratios blip to blip {pat['excess_ratios']}."
        ),
        "",
    ]
    if res.get("window_fit"):
        wf = res["window_fit"]
        lines += [
            (
                f"Tick windows fitted (no per-tick timestamps): session open {wf['session_open_s']} s, overhead outside the timer "
                f"{wf['overhead_per_tick_s']:.2f} s per tick; {wf['idle_samples_between_ticks']} of {wf['idle_samples']} idle GPU samples "
                "fall between ticks."
            ),
            "",
        ]
    if res.get("gpu_idles_mid_tick") is not None:
        mids = res["gpu_idles_mid_tick"]
        lines += [
            "GPU samples idle inside a tick's window (host-side stalls when the window excludes the between-tick phase):",
            "",
        ]
        lines += [
            f"- tick {m['pos']}: {m['phase_s']} s into the tick, utilisation {m['util']:.0f} %"
            for m in mids
        ] or ["- none"]
        lines += [""]
    lines += [
        "| metric | base median | robust sd | Spearman vs seconds | "
        + " | ".join(f"tick {b['pos']} (z, pct)" for b in a["blips"])
        + " |",
        "|---|---|---|---|" + "---|" * len(a["blips"]),
    ]
    for r in a["metrics"][:40]:
        cells = []
        for b in a["blips"]:
            at = r["at_blips"].get(b["pos"])
            cells.append(
                "—"
                if at is None
                else f"{fmt(at['value'])} ({'differs' if at['z'] is None else at['z']}, {at['pct']})"
            )
        const = " (constant)" if r["constant"] else ""
        lines.append(
            f"| `{r['metric']}`{const} | {fmt(r['base_median'])} | {fmt(r['base_robust_sd'])} | {fmt(r['spearman_vs_seconds'])} | "
            + " | ".join(cells)
            + " |"
        )
    lines += [
        "",
        (
            f"{len(a['metrics'])} metrics compared; the table shows the 40 with the largest |z| at a blip. "
            "A constant metric scores z = 0 at a blip whose value equals the base, "
            "and 'differs' when it does not."
        ),
        "",
    ]
    return "\n".join(lines)


def figure(res: dict, table: dict[int, dict[str, float]], path: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    a = res["analysis"]
    pos = sorted(p for p in table if "seconds" in table[p])
    secs = [table[p]["seconds"] for p in pos]
    blips = [b["pos"] for b in a["blips"]]
    cands = [c["pos"] for c in a["candidates"]]
    wanted = [
        "rss_gb",
        "host.VmRSS_kB",
        "minflt",
        "cpu_s",
        "ring_gap_s",
        "stage.max_wall_s",
        "gpu.utilization.gpu.min",
        "gpu.memory.used.max",
        "host.MemAvailable_kB.min",
        "host.load1.max",
        "gc.gen2_s",
        "io.wchar",
    ]
    present = [m for m in wanted if any(m in table[p] for p in pos)]
    n = 2 + len(present)
    fig, axes = plt.subplots(n, 1, figsize=(11, 2.2 * n), sharex=True)
    ax = axes[0]
    ax.plot(pos, secs, "-", color="0.4", lw=0.8)
    ax.plot(blips, [table[p]["seconds"] for p in blips], "o", color="crimson", label="blip")
    ax.plot(cands, [table[p]["seconds"] for p in cands], "o", color="orange", label="candidate")
    ax.set_ylabel("seconds / tick")
    ax.legend(loc="upper left", fontsize=8)
    ax = axes[1]
    ex = [b["excess_s"] for b in a["blips"]]
    ax.plot(blips, ex, "o", color="crimson")
    fit = a["pattern"]["excess_vs_tick"]
    if blips and math.isfinite(fit["slope"]):
        xs = np.asarray([min(blips), max(blips)], dtype=float)
        ax.plot(
            xs,
            fit["slope"] * xs + fit["intercept"],
            "--",
            color="0.5",
            label=f"{fit['slope']:.3f} s/tick, R² {fit['r2']:.2f}",
        )
        ax.legend(loc="upper left", fontsize=8)
    ax.set_ylabel("excess s at blips")
    for ax, m in zip(axes[2:], present):
        ys = [table[p].get(m, math.nan) for p in pos]
        ax.plot(pos, ys, "-", color="0.3", lw=0.8)
        ax.plot(blips, [table[p].get(m, math.nan) for p in blips], "o", color="crimson")
        ax.set_ylabel(m, fontsize=7)
    axes[-1].set_xlabel("tick")
    fig.suptitle(f"{res['label']}: {len(blips)} blips (z > {a['threshold']})")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return True


def load_inputs(a) -> tuple[dict[str, dict], dict[int, dict[str, float]], dict]:
    """(inputs with their sha256, the per-tick table, the metrics header) of --report / --metrics."""
    inputs: dict[str, dict] = {}
    table: dict[int, dict[str, float]] = {}
    header: dict = {}
    if a.report:
        p = Path(a.report)
        inputs["report"] = {"path": str(p), "sha256": sha256_of(p)}
        table = load_report(p)
    if a.metrics:
        p = Path(a.metrics)
        inputs["metrics"] = {"path": str(p), "sha256": sha256_of(p)}
        header, rows = load_metrics(p)
        for pos, row in rows.items():
            table.setdefault(pos, {}).update(row)
    return inputs, table, header


def idle_times(samples: list) -> np.ndarray:
    """The times of the GPU samples below IDLE_UTIL (empty without a GPU utilisation column)."""
    for kind, times, cols in samples:
        util = next((c for c in cols if c.startswith("utilization.gpu")), None)
        if kind == "gpu" and util is not None:
            return times[cols[util] < IDLE_UTIL]
    return np.asarray([])


def place_samples(samples: list, windows: dict, seconds: dict, table: dict, res: dict) -> None:
    """Cut every sampler stream into the ticks' windows (into the table) and keep the GPU
    samples idle inside a tick (the mid-tick stalls)."""
    for kind, times, cols in samples:
        rows, idles = per_tick_samples(times, cols, windows, kind)
        for pos, row in rows.items():
            table.setdefault(pos, {}).update(row)
        if kind == "gpu":
            res["gpu_idles_mid_tick"] = [
                i for i in idles if i["phase_s"] < seconds.get(i["pos"], 0) - SAMPLER_LAG_S
            ]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--report", help="a gpu_run report (JSON)")
    ap.add_argument("--metrics", help="the per-tick metrics JSONL of gpu_run --metrics")
    ap.add_argument("--nvsmi", help="nvidia-smi samples (CSV with a timestamp column)")
    ap.add_argument("--host", help="host samples (CSV of sample_box.py)")
    ap.add_argument(
        "--start", help="the run's start time (ISO, UTC) when --metrics gives no windows"
    )
    ap.add_argument(
        "--threshold", type=float, default=6.0, help="robust z above which a tick is a blip"
    )
    ap.add_argument(
        "--candidate", type=float, default=3.0, help="robust z above which a tick is listed"
    )
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", required=True, help="output directory")
    a = ap.parse_args(argv)
    if not a.report and not a.metrics:
        ap.error("--report or --metrics is needed")
    inputs, table, header = load_inputs(a)
    label = a.label or Path(a.report or a.metrics).stem
    res: dict = {
        "label": label,
        "inputs": inputs,
        "header": {k: header[k] for k in header if k != "kind"},
    }
    windows = windows_from_metrics(table)
    seconds = {p: r["seconds"] for p, r in table.items() if "seconds" in r}
    samples = []
    for kind, path in (("gpu", a.nvsmi), ("host", a.host)):
        if path:
            pth = Path(path)
            inputs[kind] = {"path": str(pth), "sha256": sha256_of(pth)}
            samples.append((kind, *load_samples(pth, "timestamp" if kind == "gpu" else "epoch")))
    if samples and not windows:
        if not a.start:
            ap.error(
                "--start is needed to place the samples: the inputs carry no per-tick timestamps"
            )
        windows, res["window_fit"] = fit_windows(seconds, parse_time(a.start), idle_times(samples))
    place_samples(samples, windows, seconds, table, res)
    res["analysis"] = analyse(table, a.threshold, a.candidate)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{label}_blips.json").write_text(json.dumps(res, indent=1, default=str))
    md = markdown(res, inputs, label)
    (out / f"{label}_blips.md").write_text(md)
    if figure(res, table, out / f"{label}_blips.png"):
        md += f"\nFigure: `{label}_blips.png`\n"
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
