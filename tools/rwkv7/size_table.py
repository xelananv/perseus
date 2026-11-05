#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""The per-size table of ANYSIZE_DESIGN.md §4 from the planner census (plan_sizes.py's sizes.jsonl files) and the
memory model (ckpt_info.py). Every derived number says how it was derived:

  refreshes per token (decode) = block_0 + (layers - 1) x block_1 + tail, from a 2-block plan (a 1-block plan:
      block_1 = block_0 x the 2,048-wide plan's block_1 / block_0), times the synthetic-to-real factor (the real 0.1B
      with the final coefficients over the synthetic 768-wide model, the same 2-block plans);
  refreshes per K-pass = the same with block_b(K) = block_b(decode) + slope_b K, slope_b = (block_b(8) - block_b(4)) / 4
      (linear in K: the 0.1B planner's 55 / 79 / 127 / 222 at K = 4 / 8 / 16 / 32, BATCHED_DESIGN.md §2), times the
      real / synthetic slope ratio at 768 (the synthetic weights under-count the per-conversation refreshes);
  seconds = refreshes x the measured price per planned refresh on pod 3 (R6: one card, one build family, not this
      code's build): decode 9.54 s / 478 = 19.96 ms (new_pod/RESULTS.md P1-P3, the decode plan of README.md);
      batched 46.68 s / 2,063 = 22.63 ms (r01b_batch/batch32.jsonl: summary seconds_per_pass_median, the header's
      compiled n_bootstraps summed over the 13 programs).

    python tools/rwkv7/size_table.py --census DIR [DIR ...] --real DIR
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))
from perseus.examples.rwkv7.ckpt_info import memory_mb, preset  # noqa: E402
from perseus.examples.rwkv7.synth import SIZES, lora_ranks       # noqa: E402

MS_DEC, MS_B = 9.54 / 478, 46.68 / 2063


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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", nargs="+", required=True)
    ap.add_argument("--real", required=True, help="the real 0.1B census dir (final coefficients)")
    ap.add_argument("--estimate", nargs="*", default=[],
                    help="W:t:b0:b1:slope for a width with no plan (marked 'estimate' in the table)")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    by_w = {}
    for d in a.census:
        rows = load(d)
        by_w[rows["decode"]["width"]] = rows
    real = load(a.real)
    rb0, rb1, rt = blocks(real["decode"])
    sb0, sb1, st_ = blocks(by_w[768]["decode"])
    f = (rb0 + 11 * rb1 + rt) / (sb0 + 11 * sb1 + st_)
    # per-conversation slopes: the synthetic 768 model under-counts the real 0.1B's (block 1: 2 vs 4 per conversation);
    # every synthetic slope is scaled by real / synthetic at 768, block by block
    def slopes(rows):
        k4, k8 = blocks(rows["b4"]), blocks(rows["b8"])
        return (k8[0] - k4[0]) / 4, (k8[1] - k4[1]) / 4
    (rs0, rs1), (ss0, ss1) = slopes(real), slopes(by_w[768])
    fk0, fk1 = rs0 / ss0, rs1 / ss1
    print(f"per-conversation slope factors (real / synthetic at 768): block 0 {fk0:.2f} ({rs0}/{ss0}), block 1 "
          f"{fk1:.2f} ({rs1}/{ss1})")
    two = [blocks(r["decode"]) for r in by_w.values() if r["decode"]["blocks"] > 1]
    ratio = sum(b1 / b0 for b0, b1, _ in two) / len(two)          # block_1 / block_0 over the 2-block plans
    out = []
    print(f"synthetic-to-real factor {f:.3f} (real 0.1B 2-block decode {rb0}/{rb1}/{rt}, synthetic 768 {sb0}/{sb1}/{st_}); "
          f"1-block widths: block_1 = block_0 x {ratio:.3f}")
    hdr = ("| size | layers x width | lanes t | cts / vector | state cts / block / conv. | planned refreshes / block "
           "(b0, b1) | refreshes / token | s / token, K = 1 | K (memory) | refreshes / pass | s / pass | tok/s "
           "aggregate | memory decode / batched GB |")
    print(hdr)
    print("|" + "---|" * 13)
    est = {}
    for e in a.estimate:
        W, t, e0, e1, sl = e.split(":")
        est[int(W)] = (int(t), int(e0), int(e1), float(sl))
    for name, (nl, C) in SIZES.items():
        rows = by_w.get(C)
        if rows is None and C in est:
            t, b0, b1, sl = est[C]
            m = {"n_layer": nl, "width": C, "d_ffn": 4 * C, "lora": lora_ranks(C)}
            p = preset(m, t)
            K = p["B_K"]
            R = f * (b0 + (nl - 1) * b1 + 1)
            RK = f * (b0 + (nl - 1) * b1 + nl * fk1 * sl * K + 1) if K > 1 else None
            md = memory_mb(m, pool=p["POOL"], stc=p["STC_CACHE"], lanes=t)["total"] / 1000
            mb = memory_mb(m, K=K, pool=p["POOL_B"], stc=p["STC_CACHE_B"], lanes=t)["total"] / 1000
            S = -(-C // (32768 // t))
            sp = RK * MS_B if RK else None
            out.append({"size": name, "layers": nl, "width": C, "lanes": t, "estimate": True, "decode_b0": b0,
                        "decode_b1": b1, "refreshes_per_token": round(R), "s_per_token": round(R * MS_DEC, 1), "K": K,
                        "mem_decode_gb": round(md, 1), "mem_batched_gb": round(mb, 1),
                        "fits_one_card": p["fits_one_card"], "preset": p})
            print(f"| {name} | {nl} x {C} | {t} | {S} | {S * 64 // t} | {b0}, {b1} (estimate, not planned) | {round(R)} | "
                  f"{R * MS_DEC:.1f} | {K} | {round(RK) if RK else '-'} | {f'{sp:.1f}' if sp else '-'} | "
                  f"{round(K / sp, 3) if sp else round(1 / (R * MS_DEC), 4)} | {md:.0f} / {mb:.0f}"
                  f"{'' if p['fits_one_card'] else ' (over the card)'} |")
            continue
        if rows is None:
            continue
        b0, b1, tl = blocks(rows["decode"])
        if b1 is None:
            b1 = round(b0 * ratio)
        R = f * (b0 + (nl - 1) * b1 + tl)
        m = {"n_layer": nl, "width": C, "d_ffn": 4 * C, "lora": lora_ranks(C)}
        t = rows["decode"].get("lanes") or 32
        p = preset(m, t)
        K = p["B_K"]
        if "b4" in rows and "b8" in rows and rows["b8"]["blocks"] > 1:
            k4, k8 = blocks(rows["b4"]), blocks(rows["b8"])
            s0, s1 = fk0 * (k8[0] - k4[0]) / 4, fk1 * (k8[1] - k4[1]) / 4
            pb0, pb1 = b0 + s0 * K, b1 + s1 * K
            RK = f * (pb0 + (nl - 1) * pb1 + tl)
        elif "b4" in rows:
            k4 = blocks(rows["b4"])
            per = fk0 * (k4[0] - b0) / 4                   # one block's per-conversation refreshes, from K = 4
            RK = f * (b0 + per * K + (nl - 1) * (b1 + per * ratio * K) + tl) if K > 1 else R
        else:
            RK = None
        md = memory_mb(m, pool=p["POOL"], stc=p["STC_CACHE"], lanes=t)["total"] / 1000
        mb = memory_mb(m, K=K, pool=p["POOL_B"], stc=p["STC_CACHE_B"], lanes=t)["total"] / 1000
        S = -(-C // (32768 // t))
        sp = RK * MS_B if (RK and K > 1) else None
        row = {"size": name, "layers": nl, "width": C, "lanes": t, "shards": S, "state_cts_per_block": S * 64 // t,
               "decode_b0": b0, "decode_b1": b1, "refreshes_per_token": round(R), "s_per_token": round(R * MS_DEC, 1),
               "K": K, "refreshes_per_pass": round(RK) if RK else None, "s_per_pass": round(sp, 1) if sp else None,
               "agg_tok_s": round(K / sp, 3) if sp else round(1 / (R * MS_DEC), 4),
               "mem_decode_gb": round(md, 1), "mem_batched_gb": round(mb, 1), "fits_one_card": p["fits_one_card"],
               "preset": p}
        out.append(row)
        print(f"| {name} | {nl} x {C} | {t} | {S} | {S * 64 // t} | {b0}, {b1}{'' if rows['decode']['blocks'] > 1 else ' (b1 = b0 x ratio)'} | {round(R)} | {R * MS_DEC:.1f} | {K} | "
              f"{round(RK) if RK else '-'} | {f'{sp:.1f}' if sp else '-'} | {row['agg_tok_s']} | {md:.0f} / {mb:.0f}"
              f"{'' if p['fits_one_card'] else ' (over the card)'} |")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
