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
              per-op model's 28.3 s (review lens 3, finding 10). A range 26.6 - 28.3 s belongs with every pass time.
  The price includes everything else a token does (rotations, products, PCMM GEMMs) at the 0.1B / 0.4B's mix; the GEMM
  grows with the parameters, refreshes with the ciphertext count, so at 1.5B and above it is an underestimate.

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
from perseus.examples.rwkv7.ckpt_info import memory_mb, preset  # noqa: E402
from perseus.examples.rwkv7.synth import SIZES, lora_ranks       # noqa: E402

# the anchors (transcribed; --anchors re-derives them): (record, planned refreshes)
DEC_RUNS = {"r01b_final/run_best_P1.jsonl": (9.703, 497), "r01b_final/run_best_P2.jsonl": (9.7283, 497),
            "r01b_final/run_best_P3.jsonl": (9.7405, 497), "r04b_final/run_best_P1.jsonl": (19.7367, 1011),
            "r04b_final/run_best_P2.jsonl": (19.7413, 1011), "r04b_final/run_best_P3.jsonl": (19.7286, 1011)}
DEC_PLANS = {"r01b_final/plan_best.log": 497, "r04b_final/plan_best.log": 1011}
B_RUN = ("r01b_final/batch32_best.jsonl", 29.8721, "r01b_final/plan_b32_best.log", 1424)
MS_DEC = (min(s / n for s, n in DEC_RUNS.values()), max(s / n for s, n in DEC_RUNS.values()))
MS_B = B_RUN[1] / B_RUN[3]
#: the measured final runs the table is checked against (same records), and the tolerance stated for decode
MEASURED = {"0.1B": {"refreshes": 497, "s": (9.703, 9.7405)}, "0.4B": {"refreshes": 1011, "s": (19.7286, 19.7413)}}
TOL = 0.03


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
    out = []
    say(f"- synthetic-to-real factor {f:.4f} (real 0.1B 2-block decode {rb0}/{rb1}/{rt}, synthetic 768 {sb0}/{sb1}/{st_}); "
        f"1-block widths: block_1 = block_0 x {ratio:.3f}")
    say("- per-conversation slopes: the synthetic plans' (K = 4 -> 8), unscaled and added to the decode count (not "
        "multiplied by the factor); the pass column is a b2 plan's")
    say(f"- prices per PLANNED refresh (pod 3, final coefficients, LEVERS=best): decode {MS_DEC[0] * 1e3:.2f}-"
        f"{MS_DEC[1] * 1e3:.2f} ms; batched {MS_B * 1e3:.2f} ms (a base plan; low for b2 plans, see below)")
    say("")
    say("| size | layers x width | lanes t | cts / vector | state cts / block / conv. | planned refreshes / block (b0, b1) "
        "| slope / conv. / block | refreshes / token | s / token, K = 1 | K (memory) | refreshes / pass | s / pass "
        "| tok/s aggregate | memory decode / batched GB |")
    say("|" + "---|" * 14)
    est = {}
    for e in a.estimate:
        W, t, e0, e1, sl = e.split(":")
        est[int(W)] = (int(t), int(e0), int(e1), float(sl))
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
        mb = memory_mb(m, K=K, pool=p["POOL_B"], stc=p["STC_CACHE_B"], lanes=t)["total"] / 1000
        S = -(-C // (32768 // t))
        sd = secs(R, MS_DEC)
        sp = secs(RK, MS_B) if RK else None
        agg = (round(K / sp, 3) if sp else (round(1 / sd[1], 4), round(1 / sd[0], 4)))
        row = {"size": name, "layers": nl, "width": C, "lanes": t, "shards": S, "state_cts_per_block": S * 64 // t,
               "estimate": estimate, "decode_b0": b0, "decode_b1": b1, "slope": [s0, s1],
               "refreshes_per_token": round(R), "s_per_token": [round(x, 2) for x in sd], "K": K,
               "refreshes_per_pass": round(RK) if RK else None, "s_per_pass": round(sp, 1) if sp else None,
               "agg_tok_s": agg, "mem_decode_gb": round(md, 1), "mem_batched_gb": round(mb, 1),
               "fits_one_card": p["fits_one_card"], "preset": p}
        out.append(row)
        sl_txt = f"{s1:g}" if s0 is not None and s0 == s1 else (f"{s0:g}, {s1:g}" if s0 is not None else "-")
        agg_txt = f"{agg}" if sp else f"{agg[0]}-{agg[1]}"
        say(f"| {name} | {nl} x {C} | {t} | {S} | {S * 64 // t} | {b0}, {b1}{note} | {sl_txt} | {round(R)} | "
            f"{fmt_range(sd, 2)} | {K} | {round(RK) if RK else '-'} | {f'{sp:.1f}' if sp else '-'} | {agg_txt} | "
            f"{md:.0f} / {mb:.0f}{'' if p['fits_one_card'] else ' (over the 92 % margin)'} |")
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
    r01 = next((r for r in out if r["size"] == "0.1B"), None)
    if r01 and r01["refreshes_per_pass"]:
        say(f"- check 0.1B batched (K = {r01['K']}): {r01['refreshes_per_pass']} planned refreshes per pass for a b2 plan; "
            f"no measured b2 pass exists. The real 0.1B's K = 32 b2 2-block plan (planner 100 / 103 for blocks 0 / 1, "
            f"PLAN_B2.md) extrapolates to 100 + 11 x 103 + 1 = 1234; the measured BASE pass is "
            f"{B_RUN[3]} planned / 29.87 s, i.e. the price anchor itself. {r01['s_per_pass']} s here is low for b2: the "
            f"per-op model gives 28.3 s (review lens 3)")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)
    if a.md:
        with open(a.md, "w") as f:
            f.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
