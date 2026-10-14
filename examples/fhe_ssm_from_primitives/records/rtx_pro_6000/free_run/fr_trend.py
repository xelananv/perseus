#!/usr/bin/env python3
"""The free-running runs as tables, from gpu_run.py --free-run reports
(fr_<mode><ticks>.json: one record per tick; per-tick fields are aggregates over the 64 conversations, except
tokens/top1/pick, which are per conversation). Prints, per block of ticks: the mean logit relErrRms against the
closed form (the AGGREGATE rms over all conversations, rms(diff)/rms(ref) of the whole logits matrix, which
is NOT the published 09-19 run's per-lane median, so levels and slopes compare only loosely; one
conversation's spike dominates a tick's aggregate, hence also the median over each block's ticks), the mean of the
per-tick worst conversation, raw top-1 and decode-policy pick disagreements with the closed form (counted per
conversation-tick), generating conversation-ticks, on-record agreement; then a log-log least-squares slope of the
block level (a description of these records, not a law), the divergence list, and, given two reports, whether the
two runs fed the same tokens. Timings are printed as recorded; the two free runs shared one host.
With --predicted record_departures.out, each departure is marked "predicted" (where the published run's own pick
flipped: a run that picks as the closed form does leaves the record there) or "own flip" (this run's pick left the
closed form's); a predicted departure not made means this run flipped the same near-tie the published run did.
usage: fr_trend.py fr_eager148.json [fr_planned148.json] [--block 10] [--predicted record_departures.out]"""

import argparse
import json
import math
import statistics as st

ap = argparse.ArgumentParser()
ap.add_argument("reports", nargs="+")
ap.add_argument("--block", type=int, default=10)
ap.add_argument(
    "--predicted", help="record_departures.py output: the first departures a faithful run must make"
)
a = ap.parse_args()


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


PRED = None
if a.predicted:
    with open(a.predicted, encoding="utf-8") as f:
        PRED = json.load(f)["first_diverged_at"]


def trend(path, doc):
    T = doc["ticks"]
    n = doc.get("conversations") or len(T[0]["tokens"])
    print(
        f"\n## {path}: mode {doc['mode']}, {len(T)} ticks, {n} conversations, policy {doc.get('decode_policy')}"
    )
    if doc.get("plan_runtime"):
        print(f"- plan runtime switches: {doc['plan_runtime']}")
    print(
        "\n| ticks | mean logits relErrRms (aggregate) | median over the block's ticks | mean worst conversation | top-1 flips | pick flips | generating conv-ticks | on-record agree | median s/tick | runtime boots/tick | free GB |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    centres, levels = [], []
    for b0 in range(0, len(T), a.block):
        tt = T[b0 : b0 + a.block]
        lv = [t["logits_relErrRms_vs_exact"] for t in tt]
        wc = [t["logits_relErrRms_worst_conversation"] for t in tt]
        top1 = sum(round((1 - t["top1_agree_vs_exact"]) * n) for t in tt)
        pick = sum(round((1 - t["pick_agree_vs_exact"]) * n) for t in tt)
        gen = sum(t["fed_back"] for t in tt)
        onr = sum(t["on_record"] for t in tt)
        agr = sum(round(t["pick_agree_vs_record"] * t["on_record"]) for t in tt if t["on_record"])
        rb = sorted({t["runtime_bootstraps"] for t in tt})
        fg = sorted({t["free_gb"] for t in tt})
        print(
            f"| {tt[0]['pos']}-{tt[-1]['pos']} | {st.mean(lv):.6f} | {st.median(lv):.6f} | {st.mean(wc):.6f} | {top1} | {pick} | {gen}/{n * len(tt)} | {agr}/{onr} | {st.median(t['seconds'] for t in tt):.2f} | {rb} | {fg} |"
        )
        if len(tt) == a.block:
            centres.append(tt[0]["pos"] + a.block / 2)
            levels.append(st.mean(lv))
    top1 = sum(round((1 - t["top1_agree_vs_exact"]) * n) for t in T)
    pick = sum(round((1 - t["pick_agree_vs_exact"]) * n) for t in T)
    print(
        f"\n- raw top-1 agreement {n * len(T) - top1}/{n * len(T)} = {100 * (1 - top1 / (n * len(T))):.3f} %; pick agreement {n * len(T) - pick}/{n * len(T)} = {100 * (1 - pick / (n * len(T))):.3f} %"
    )
    print(
        f"- seconds per tick: first {T[0]['seconds']}, median {st.median(t['seconds'] for t in T):.2f}, min {min(t['seconds'] for t in T)}, max {max(t['seconds'] for t in T)}"
    )
    if len(centres) >= 3:
        X = [math.log(c) for c in centres]
        Y = [math.log(v) for v in levels]
        mx, my = st.mean(X), st.mean(Y)
        sl = sum((x - mx) * (y - my) for x, y in zip(X, Y)) / sum((x - mx) ** 2 for x in X)
        print(
            f"- least-squares power law through {len(centres)} full blocks: level = {math.exp(my - sl * mx):.6f} * tick^{sl:.3f} (description of these records only)"
        )
        print(
            f"- block level first/last: {levels[0]:.6f} / {levels[-1]:.6f} (ratio {levels[-1] / levels[0]:.2f})"
        )
    dv = doc.get("diverged_at") or []
    off = [(k, p) for k, p in enumerate(dv) if p is not None]
    print(
        f"- conversations that left the record: {len(off)} of {len(dv)}"
        + (": " + ", ".join(f"conv {k} at pos {p}" for k, p in off) if off else "")
    )
    for k, p in off:
        t = T[p - 1]  # the tick whose pick was fed at pos p
        tag = (
            ""
            if PRED is None
            else (" predicted" if PRED[k] == p else f" own flip (closed form predicts {PRED[k]})")
        )
        print(
            f"  - conv {k} at pos {p}:{tag}; pick at pos {p - 1} was {t['pick'][k]} (top-1 {t['top1'][k]}); pick agreement with the closed form at that tick {t['pick_agree_vs_exact']}"
        )
    if PRED is not None:
        last = T[-1]["pos"]
        due = [(p, k) for k, p in enumerate(PRED) if p is not None and p <= last]
        kept = [(p, k) for p, k in due if dv[k] is None]
        later = [(p, k) for p, k in due if dv[k] is not None and dv[k] > p]
        print(
            f"- predicted departures due by pos {last}: {len(due)}; of them made: {sum(1 for p, k in due if dv[k] == p)}, "
            f"pre-empted by an earlier own flip: {sum(1 for p, k in due if dv[k] is not None and dv[k] < p)}, "
            f"not made because this run flipped the same near-tie as the published run (it picked the record's token, the closed form's second choice, and stays on the record): {len(kept)}"
            + (f" {kept}" if kept else "")
            + f"; the same, but the conversation left later by an own flip: {len(later)}"
            + (f" {[(p, k, dv[k]) for p, k in later]}" if later else "")
        )
    return T


runs = [(p, load(p)) for p in a.reports]
ticks = [trend(p, d) for p, d in runs]
if len(ticks) == 2:
    A, B = ticks
    m = min(len(A), len(B))
    same = [A[i]["tokens"] == B[i]["tokens"] for i in range(m)]
    first = next((i for i, s in enumerate(same) if not s), None)
    print(
        f"\n## {runs[0][0]} vs {runs[1][0]}: fed tokens identical at {sum(same)} of {m} common ticks"
        + (f"; first difference at pos {first}" if first is not None else "")
    )
    if first is not None:
        diff = [
            k
            for k in range(len(A[first]["tokens"]))
            if A[first]["tokens"][k] != B[first]["tokens"][k]
        ]
        print(f"- conversations differing at pos {first}: {diff}")
