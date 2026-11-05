#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""The per-size table of ANYSIZE_DESIGN.md §4 from the planner census (plan_sizes.py's sizes.jsonl files) and the
memory model (ckpt_info.py). Every derived number says how it was derived (models, not measurements).

Units. Every count here is a PLANNER count (``summary.total_bootstraps``, what plan_cost.py reads), and every price
is a price per PLANNER refresh, anchored on a run whose planner count is on record. The device's compiled
``n_bootstraps`` is a different unit (the compile prunes and merges sites: the final batched plan is 1,424 planner
refreshes and 1,257 compiled sites), so a price per compiled site must never multiply a planner count (review lens 3,
finding 1).

  refreshes per token (decode) = f x (block_0 + (layers - 1) x block_1 + tail), from a 2-block plan (a 1-block plan:
      block_1 = block_0 x the 2-block plans' mean block_1 / block_0), f = the real 0.1B (final coefficients) over the
      synthetic 768-wide model, the same 2-block plans;
  refreshes per K-pass = refreshes per token + K x (slope_0 + (layers - 1) x slope_1), slope_b = (block_b(K = 8) -
      block_b(K = 4)) / 4 refreshes per conversation and block, from the synthetic plans, UNSCALED (review lens 3,
      finding 2): the real 0.1B's base plan shows 4 per conversation in block 1 against the synthetic 2, but that is
      the min-cut prune artefact PLANOPT=b2 removes (PLAN_B2.md: under b2 the real 0.1B is 36 + 2.0 K / 40 + 1.96 K),
      not a synthetic under-count; the synthetic slopes are b2-like (2 at t = 32, 4 at t = 16: the state ciphertexts
      per conversation and block). Linear in K: BATCHED_DESIGN.md §2. So the pass column prices a b2 plan;
  seconds = refreshes x a price per planned refresh from pod 3's FINAL-coefficient runs, LEVERS=best (R6: one card,
      one build family, not this code's build):
      decode  9.703 / 9.7283 / 9.7405 s per token (r01b_final/run_best_P1-P3.jsonl, seconds_per_token_median) over 497
              planned refreshes (r01b_final/plan_best.log), and 19.7367 / 19.7413 / 19.7286 s (r04b_final/run_best_P1-P3)
              over 1,011 (r04b_final/plan_best.log): 19.51 - 19.60 ms per planned refresh;
      batched 29.8721 s per pass (r01b_final/batch32_best.jsonl, seconds_per_pass_median) over 1,424 planned
              refreshes (r01b_final/plan_b32_best.log): 20.98 ms per planned refresh. That plan is a BASE plan (33
              paired refreshes per block); a b2 plan removes refreshes about half of which were paired, so its price
              per planned refresh is higher and this column is LOW for b2 plans: at the 0.1B, 26.6 s here against the
              per-op model's 28.3 s (review lens 3, finding 10). So every pass time is a range: the low end at 20.98 ms,
              the high end at the price that makes the 0.1B row equal the per-op model's b2 estimate (28.45 s, review
              lens 3's remainder model; B2_HI_S below) over this table's own 0.1B count.
  The price includes everything else a token does (rotations, products, PCMM GEMMs) at the 0.1B / 0.4B's mix; the GEMM
  grows with the parameters, refreshes with the ciphertext count, so at 1.5B and above it is an underestimate.
  Memory: ckpt_info.memory_mb (device, MiB / 1,000) and ckpt_info.host_gb (host RAM of session setup, GB), both
  modelled on pod 3's records (ckpt_info's docstring: the final-coefficient mem reports, the 0.4B K = 16 refusal,
  the sampler's host peaks and the two host OOM kills); K is the preset's (both budgets).

    python tools/rwkv7/size_table.py --census DIR [DIR ...] --real DIR [--estimate W:t:b0:b1:slope ...] \\
        [--anchors results/rwkv7-fhe-20261104/new_pod] [--json OUT] [--md OUT]

--anchors re-derives the anchor prices from the named records and stops if they differ from the constants below.
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))
from perseus.examples.rwkv7.ckpt_info import CAP_MB, HOST_CAP_GB, host_gb, memory_mb, preset
from perseus.examples.rwkv7.synth import SIZES, lora_ranks

# the anchors (transcribed; --anchors re-derives them): (record, planned refreshes)
DEC_RUNS = {"r01b_final/run_best_P1.jsonl": (9.703, 497), "r01b_final/run_best_P2.jsonl": (9.7283, 497),
            "r01b_final/run_best_P3.jsonl": (9.7405, 497), "r04b_final/run_best_P1.jsonl": (19.7367, 1011),
            "r04b_final/run_best_P2.jsonl": (19.7413, 1011), "r04b_final/run_best_P3.jsonl": (19.7286, 1011)}
DEC_PLANS = {"r01b_final/plan_best.log": 497, "r04b_final/plan_best.log": 1011}
B_RUN = ("r01b_final/batch32_best.jsonl", 29.8721, "r01b_final/plan_b32_best.log", 1424)
MS_DEC = (min(s / n for s, n in DEC_RUNS.values()), max(s / n for s, n in DEC_RUNS.values()))
MS_B = B_RUN[1] / B_RUN[3]
B2_HI_S = 28.45          # the per-op model's b2 pass at the 0.1B, K = 32 (review lens 3, finding 10: 28.28-28.45 s)
#: the measured final runs the table is checked against (same records), and the tolerance stated for decode
MEASURED = {"0.1B": {"refreshes": 497, "s": (9.703, 9.7405)}, "0.4B": {"refreshes": 1011, "s": (19.7286, 19.7413)}}
TOL = 0.03
#: the memory anchors the device / host models are fitted on (ckpt_info's docstring), printed beside the table
MEASURED_MEM = {
    "0.1B": (("decode 70,980.0 MiB planned (r01b_final/run_best_P1), batched K = 32 91,019.6 MiB (batch32_best, VRAM "
              "peak 94,858 MiB): the model 71.0 / 91.0"),
             ("decode 34.7-35.1 GB, batched K = 32 150.5 GB (sampler peaks); prefill T = 32 fitted (no peak on "
              "record): the model 35.1 / 150.5 / 93.6-180.9")),
    "0.4B": (("decode 77,844.9 MiB planned (r04b_final/run_best_P1); batched K = 16 at POOL_B 1,024 / STC_CACHE_B 64 "
              "REFUSED needing 98,658.0 MiB of 96,692.9 free: the model 98.7 there (fitted), 90.6 at the preset's "
              "POOL_B 768"),
             ("decode 64.1-64.2 GB, batched K = 16 160.9 GB; K = 32 and prefill T = 32 OOM-killed at ~185 GB of 181: "
              "the model 64.2 / 160.9 / 294.9 (K = 32) / 181.1-355.8 (T = 32)")),
}


def hd_fmt(x):
    return f"{x:.0f}"


def check_anchors(d: str) -> None:
    """Re-derive every anchor from the records in ``d`` (new_pod/); stop on a difference."""
    bad = []
    for rel, (s, _n) in DEC_RUNS.items():
        summ = [json.loads(x) for x in open(os.path.join(d, rel)) if '"summary"' in x][-1]
        if round(summ["seconds_per_token_median"], 4) != round(s, 4):
            bad.append(f"{rel}: seconds_per_token_median {summ['seconds_per_token_median']} != {s}")
    plans = dict(DEC_PLANS)
    plans[B_RUN[2]] = B_RUN[3]
    for rel, n in plans.items():
        m = re.search(r"(\d+) refreshes \(summary\.total_bootstraps\)", open(os.path.join(d, rel)).read())
        if not m or int(m.group(1)) != n:
            bad.append(f"{rel}: planned refreshes {m.group(1) if m else None} != {n}")
    summ = [json.loads(x) for x in open(os.path.join(d, B_RUN[0])) if '"summary"' in x][-1]
    if round(summ["seconds_per_pass_median"], 4) != round(B_RUN[1], 4):
        bad.append(f"{B_RUN[0]}: seconds_per_pass_median {summ['seconds_per_pass_median']} != {B_RUN[1]}")
    if bad:
        raise SystemExit("anchors differ from the records:\n  " + "\n  ".join(bad))
    print(f"anchors re-derived from {d}: decode {MS_DEC[0] * 1e3:.2f}-{MS_DEC[1] * 1e3:.2f} ms, batched "
          f"{MS_B * 1e3:.2f} ms per planned refresh")


def load(d):
    rows = {}
    for line in open(os.path.join(d, "sizes.jsonl")):
        r = json.loads(line)
        if r.get("lanes") is None:                 # older census lines: the capture's own record
            meta = os.path.join(d, f"cap_{r['mode']}_b{r['blocks']}", "rwkv7_capture.json")
            r["lanes"] = json.load(open(meta)).get("lanes", 32) if os.path.exists(meta) else 32
        rows[r["mode"]] = r
    return rows


def blocks(r):
    st = r["stages"]
    return st["block_0"]["refreshes"], (st["block_1"]["refreshes"] if r["blocks"] > 1 else None), \
        st[f"block_{r['blocks']}"]["refreshes"]


def secs(R, ms):
    return (R * ms[0], R * ms[1]) if isinstance(ms, tuple) else R * ms


def fmt_range(x, nd=1):
    return f"{x[0]:.{nd}f}-{x[1]:.{nd}f}" if isinstance(x, tuple) else f"{x:.{nd}f}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", nargs="+", required=True)
    ap.add_argument("--real", required=True, help="the real 0.1B census dir (final coefficients)")
    ap.add_argument("--estimate", nargs="*", default=[],
                    help="W:t:b0:b1:slope for a width with no plan (marked 'estimate' in the table)")
    ap.add_argument("--anchors", default=None, help="new_pod/: re-derive the anchor prices from the records")
    ap.add_argument("--json", default=None)
    ap.add_argument("--md", default=None, help="write the header lines and the table here as well")
    a = ap.parse_args(argv)
    if a.anchors:
        check_anchors(a.anchors)
    lines = []

    def say(s):
        print(s)
        lines.append(s)
    by_w = {}
    for d in a.census:
        rows = load(d)
        by_w[rows["decode"]["width"]] = rows
    real = load(a.real)
    rb0, rb1, rt = blocks(real["decode"])
    sb0, sb1, st_ = blocks(by_w[768]["decode"])
    f = (rb0 + 11 * rb1 + rt) / (sb0 + 11 * sb1 + st_)
    two = [blocks(r["decode"]) for r in by_w.values() if r["decode"]["blocks"] > 1]
    ratio = sum(b1 / b0 for b0, b1, _ in two) / len(two)          # block_1 / block_0 over the 2-block plans
    est = {}
    for e in a.estimate:
        W, t, e0, e1, sl = e.split(":")
        est[int(W)] = (int(t), int(e0), int(e1), float(sl))
    out = []
    for name, (nl, C) in SIZES.items():
        rows = by_w.get(C)
        m = {"n_layer": nl, "width": C, "d_ffn": 4 * C, "lora": lora_ranks(C)}
        estimate = rows is None
        if rows is None and C not in est:
            continue
        if estimate:
            t, b0, b1, sl = est[C]
            s0 = s1 = sl
            tl = 1
            note = " (estimate, not planned)"
        else:
            b0, b1, tl = blocks(rows["decode"])
            note = ""
            if b1 is None:
                b1 = round(b0 * ratio)
                note = " (b1 = b0 x ratio)"
            t = rows["decode"].get("lanes") or 32
            if "b4" in rows and "b8" in rows and rows["b8"]["blocks"] > 1:
                k4, k8 = blocks(rows["b4"]), blocks(rows["b8"])
                s0, s1 = (k8[0] - k4[0]) / 4, (k8[1] - k4[1]) / 4
            elif "b4" in rows:
                k4 = blocks(rows["b4"])
                s0 = s1 = (k4[0] - b0) / 4                  # one block's per-conversation refreshes, from K = 4
            else:
                s0 = s1 = None
        R = f * (b0 + (nl - 1) * b1 + tl)
        p = preset(m, t)
        K = p["B_K"]
        RK = R + K * (s0 + (nl - 1) * s1) if (K > 1 and s0 is not None) else None
        md = memory_mb(m, pool=p["POOL"], stc=p["STC_CACHE"], lanes=t)["total"] / 1000
        mb = memory_mb(m, K=K, pool=p["POOL_B"], stc=p["STC_CACHE_B"], lanes=t)["total"] / 1000 if K > 1 else None
        hd = host_gb(m, lanes=t)[1]
        hb = host_gb(m, K=K, lanes=t)[1] if K > 1 else None
        hp = host_gb(m, T=p["PF_T"], lanes=t)
        S = -(-C // (32768 // t))
        out.append({"size": name, "layers": nl, "width": C, "lanes": t, "shards": S, "state_cts_per_block": S * 64 // t,
                    "estimate": estimate, "note": note, "decode_b0": b0, "decode_b1": b1, "slope": [s0, s1],
                    "refreshes_per_token_f": R, "refreshes_per_pass_f": RK, "K": K, "PF_T": p["PF_T"],
                    "mem_decode_gb": round(md, 1), "mem_batched_gb": round(mb, 1) if mb else None,
                    "host_decode_gb": hd, "host_batched_gb": hb, "host_prefill_gb": list(hp),
                    "fits_one_card": p["fits_one_card"], "preset": p})
    say(f"- synthetic-to-real factor {f:.4f} (real 0.1B 2-block decode {rb0}/{rb1}/{rt}, synthetic 768 {sb0}/{sb1}/{st_}); "
        f"1-block widths: block_1 = block_0 x {ratio:.3f}")
    say("- per-conversation slopes: the synthetic plans' (K = 4 -> 8), unscaled and added to the decode count (not "
        "multiplied by the factor); the pass column is a b2 plan's")
    say(f"- memory: device MiB / 1,000 under the presets' ceiling {CAP_MB / 1000:.2f} (the largest planned total measured "
        f"to run); host RAM of session setup in GB under {HOST_CAP_GB:.1f} (92 % of pod 3's 181 GB); prefill: the "
        f"decode + T-token programs, a range (only bracketed by pod 3's 0.1B fit and 0.4B OOM kill at T = 32)")
    r01 = next((r for r in out if r["size"] == "0.1B" and r["refreshes_per_pass_f"]), None)
    ms_b = (MS_B, max(MS_B, B2_HI_S / r01["refreshes_per_pass_f"])) if r01 else MS_B
    say(f"- prices per PLANNED refresh (pod 3, final coefficients, LEVERS=best): decode {MS_DEC[0] * 1e3:.2f}-"
        f"{MS_DEC[1] * 1e3:.2f} ms; batched {fmt_range(tuple(x * 1e3 for x in ms_b), 2) if r01 else f'{MS_B * 1e3:.2f}'} "
        f"ms (low end: the measured base pass; high end: the 0.1B b2 pass at the per-op model's {B2_HI_S} s)")
    say("")
    say("| size | layers x width | lanes t | cts / vector | state cts / block / conv. | planned refreshes / block (b0, b1) "
        "| slope / conv. / block | refreshes / token | s / token, K = 1 | K (memory) | refreshes / pass | s / pass "
        "| tok/s aggregate | device memory decode / batched GB | host RAM decode / batched / prefill (T) GB |")
    say("|" + "---|" * 15)
    for row in out:
        R, RK, K, S, t = row["refreshes_per_token_f"], row["refreshes_per_pass_f"], row["K"], row["shards"], row["lanes"]
        s0, s1 = row["slope"]
        sd = secs(R, MS_DEC)
        sp = secs(RK, ms_b) if RK else None
        agg = ((round(K / sp[1], 3), round(K / sp[0], 3)) if sp else (round(1 / sd[1], 4), round(1 / sd[0], 4)))
        row.update({"refreshes_per_token": round(R), "s_per_token": [round(x, 2) for x in sd],
                    "refreshes_per_pass": round(RK) if RK else None,
                    "s_per_pass": [round(x, 1) for x in sp] if sp else None, "agg_tok_s": list(agg)})
        sl_txt = f"{s1:g}" if s0 is not None and s0 == s1 else (f"{s0:g}, {s1:g}" if s0 is not None else "-")
        over = "" if row["fits_one_card"] else " (over the ceiling)"
        mbt = f"{row['mem_batched_gb']:.0f}" if row["mem_batched_gb"] else "-"
        hbt = f"{row['host_batched_gb']:.0f}" if row["host_batched_gb"] else "-"
        hp = row["host_prefill_gb"]
        say(f"| {row['size']} | {row['layers']} x {row['width']} | {t} | {S} | {row['state_cts_per_block']} | "
            f"{row['decode_b0']}, {row['decode_b1']}{row['note']} | {sl_txt} | {round(R)} | {fmt_range(sd, 2)} | {K} | "
            f"{round(RK) if RK else '-'} | {fmt_range(sp, 1) if sp else '-'} | {agg[0]}-{agg[1]} | "
            f"{row['mem_decode_gb']:.0f} / {mbt}{over} | {hd_fmt(row['host_decode_gb'])} / {hbt} / "
            f"{hp[0]:.0f}-{hp[1]:.0f} (T = {row['PF_T']}) |")
    # the checks against the measured final runs (same records as the anchors): stated tolerance TOL on decode
    say("")
    for r in out:
        ms = MEASURED.get(r["size"])
        if not ms or r["estimate"]:
            continue
        dn = r["refreshes_per_token"] / ms["refreshes"] - 1
        lo, hi = r["s_per_token"][0] / ms["s"][1] - 1, r["s_per_token"][1] / ms["s"][0] - 1
        ok = abs(dn) <= TOL and max(abs(lo), abs(hi)) <= TOL
        say(f"- check {r['size']} decode: {r['refreshes_per_token']} planned refreshes vs the final plan's "
            f"{ms['refreshes']} ({100 * dn:+.1f} %); {r['s_per_token'][0]:.2f}-{r['s_per_token'][1]:.2f} s vs measured "
            f"{ms['s'][0]:.2f}-{ms['s'][1]:.2f} s ({100 * min(lo, hi):+.1f} to {100 * max(lo, hi):+.1f} %): "
            f"{'within' if ok else 'OUTSIDE'} the stated +/-{100 * TOL:.0f} %")
    if r01:
        say(f"- check 0.1B batched (K = {r01['K']}): {r01['refreshes_per_pass']} planned refreshes per pass for a b2 plan; "
            f"no measured b2 pass exists. The real 0.1B's K = 32 b2 2-block plan (planner 100 / 103 for blocks 0 / 1, "
            f"PLAN_B2.md) extrapolates to 100 + 11 x 103 + 1 = 1234 ({100 * (r01['refreshes_per_pass'] / 1234 - 1):+.1f} %); "
            f"the measured BASE pass is {B_RUN[3]} planned / {B_RUN[1]:.2f} s (the low price itself), and the per-op "
            f"model puts b2 at 28.28-28.45 s (the high price); this row: {fmt_range(tuple(r01['s_per_pass']), 1)} s")
    for name, (dev_mb, host) in MEASURED_MEM.items():
        r = next((x for x in out if x["size"] == name), None)
        if r:
            say(f"- check {name} memory: {dev_mb}; host {host}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)
    if a.md:
        with open(a.md, "w") as f:
            f.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
