# SPDX-License-Identifier: MIT
"""Several captures of one op stream (the pass kernels' configurations) checked node for node and merged into the one
capture a plan is cut on; the per-site margins of a plan against them.

    python -m perseus.examples.fhe_ssm.passmerge diff A B
    python -m perseus.examples.fhe_ssm.passmerge merge OUT BASE EXTRA [EXTRA ...] [--labels L0 L1 ...] [--json F]
    python -m perseus.examples.fhe_ssm.passmerge margins PLAN MERGED LABEL=CAPTURE [...] [--kappa 2] [--json F]

**One op stream.** Under ``--pass-kernels on`` every configuration issues the same products, rotations and refreshes
with plaintexts whose graph record holds no value (``passkernel.pass_mult_recorder``), so two captures of the model in
two configurations hold the same ``block_<b>/graph.json`` in every field (``op_type``, ``inputs``, ``input_levels``,
``output``, ``step``, the levels and degrees, ``tag``, ``pack_*``, ``pack_field``, ``shadow``, ``args`` with
``args.pt`` = kind, level, degree, name) but the five magnitude statistics the recorder measures on the values
(``STATS``: ``impl/magtape.node_stats``). ``structure_diff`` compares two captures that way, names verbatim (no
renumbering: the captures run the same stream before the capture tick, so even the ``anon_<n>`` names agree).

**The merge** (``merge``): the base capture copied whole (``capture_config.json``, ``capture_env.json`` and every block's
stamps), then per node every statistic taken from the capture with the largest ``output_max_abs`` (the base on a tie,
then the earliest extra): one capture's record of one value, so ``output_mean`` / ``output_max_dev`` / the coefficient
maxima stay consistent with the slot maximum the planner sizes the refresh by (``placer/refresh.py``: ``max_abs``,
``max_coeff``, ``max_coeff_ac``, ``mean``, ``max_dev`` of the site's producer). Upstream's ``scripts/utils/merge_mags.py``
maxes each ``output_max_*`` field separately and keeps the base's ``output_mean``; this rule keeps a node's record
whole. What it gives up is printed beside the merge: per block the nodes whose ``output_max_coeff`` in another capture
exceeds the chosen capture's (the coefficient maximum is the planner's CF key when captured, ``btserr.choose_site``) and
the largest such ratio. A capture whose structure differs from the base's is refused (the merged plan would bind one of
them only). The per-block lines follow ``merge_mags.py``'s ("grew": nodes whose record came from another capture than
the base, with the ratio of the chosen ``output_max_abs`` to the base's).

**The margins** (``margins``): for every refresh site of a plan (its placements, its fired hints, and the graph's
deliberate refreshes: the carry), each capture's ``output_max_abs`` and ``output_max_coeff`` of the site's producer
against the merged capture's value times κ (the planner's ``PLAN_MAG_SAFETY``, 2): a ratio above 1 is a value the plan's
correction factor was not sized to hold.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys
from collections.abc import Sequence
from typing import Any

#: the per-node statistics a capture measures on the values (``impl/magtape.node_stats``): everything else is structure
STATS = ("output_max_abs", "output_mean", "output_max_dev", "output_max_coeff", "output_max_coeff_ac")


def block_dirs(cap) -> list[pathlib.Path]:
    """The ``block_<b>`` directories of a capture holding a ``graph.json``, in block order."""
    root = pathlib.Path(cap)
    out = sorted((p.parent for p in root.glob("block_*/graph.json")), key=lambda d: int(d.name.split("_")[1]))
    if not out:
        raise ValueError(f"no block_<b>/graph.json under {root}")
    return out


def graphs(cap) -> dict[str, list[dict]]:
    """``{block_<b>: nodes}`` of a capture."""
    out = {}
    for d in block_dirs(cap):
        with open(d / "graph.json", encoding="utf-8") as f:
            out[d.name] = json.load(f)["nodes"]
    return out


def structure(node: dict) -> dict:
    """A node without its magnitude statistics."""
    return {k: v for k, v in node.items() if k not in STATS}


def structure_diff(a, b, show: int = 3) -> dict[str, dict]:
    """Per block of ``a`` (and of ``b``): ``{"nodes": (len a, len b), "differences": n, "first": [...]}``; a difference
    is a node index and a field whose values differ outside ``STATS`` (a missing block or node counts once)."""
    ga, gb = graphs(a), graphs(b)
    out: dict[str, dict] = {}
    for blk in sorted(set(ga) | set(gb), key=lambda s: int(s.split("_")[1])):
        na, nb = ga.get(blk), gb.get(blk)
        if na is None or nb is None:
            out[blk] = {"nodes": (len(na or []), len(nb or [])), "differences": 1, "first": ["block missing"]}
            continue
        n, first = abs(len(na) - len(nb)), []
        if len(na) != len(nb):
            first.append(f"{len(na)} vs {len(nb)} nodes")
        for i, (x, y) in enumerate(zip(na, nb)):
            sx, sy = structure(x), structure(y)
            if sx == sy:
                continue
            for k in sorted(set(sx) | set(sy)):
                if sx.get(k) != sy.get(k):
                    n += 1
                    if len(first) < show:
                        first.append(f"node {i} ({x.get('output')}) {k}: {sx.get(k)!r} vs {sy.get(k)!r}")
        out[blk] = {"nodes": (len(na), len(nb)), "differences": n, "first": first}
    return out


def _chosen(nodes: Sequence[dict]) -> int:
    """The index of the record with the largest ``output_max_abs`` (the first one on a tie)."""
    best, k = None, 0
    for i, n in enumerate(nodes):
        v = n.get("output_max_abs")
        if v is None:
            continue
        if best is None or float(v) > best:
            best, k = float(v), i
    return k


def merge(base, extras: Sequence, out, labels: Sequence[str] | None = None, log=print) -> dict:
    """The merged capture at ``out`` (a new directory) from ``base`` and ``extras`` (the module docstring); returns the
    record that is also written as ``out/passmerge.json``."""
    base, out = pathlib.Path(base), pathlib.Path(out)
    caps = [base, *(pathlib.Path(e) for e in extras)]
    labels = list(labels) if labels else [p.name for p in caps]
    if len(labels) != len(caps):
        raise ValueError(f"{len(labels)} labels for {len(caps)} captures")
    if out.exists():
        raise ValueError(f"{out} exists")
    for e in caps[1:]:
        bad = {b: r for b, r in structure_diff(base, e).items() if r["differences"]}
        if bad:
            b, r = next(iter(bad.items()))
            raise ValueError(
                f"{e} differs from {base} outside the magnitudes ({len(bad)} blocks; {b}: {r['first']})"
            )
    gs = [graphs(c) for c in caps]
    shutil.copytree(base, out)
    rec: dict = {"base": str(base), "captures": [str(c) for c in caps], "labels": labels, "rule": "max_abs record",
                 "stats": list(STATS), "blocks": {}}  # fmt: skip
    for blk in gs[0]:
        nodes = gs[0][blk]
        took = dict.fromkeys(labels, 0)
        grown, coeff_lost = [], []
        for i, n in enumerate(nodes):
            cand = [g[blk][i] for g in gs]  # cand[0] is n itself: read everything before n is overwritten
            k = _chosen(cand)
            took[labels[k]] += 1
            src = cand[k]
            cm = float(src.get("output_max_coeff") or 0.0)
            top = max(range(len(cand)), key=lambda j: float(cand[j].get("output_max_coeff") or 0.0))
            ct = float(cand[top].get("output_max_coeff") or 0.0)
            if ct > cm:
                coeff_lost.append((ct / max(cm, 1e-300), n["output"], labels[top]))
            if k:
                b0 = float(n.get("output_max_abs") or 0.0)
                grown.append((float(src["output_max_abs"]) / max(b0, 1e-300), n["output"], labels[k]))
                for s in STATS:
                    if s in src:
                        n[s] = src[s]
                    else:
                        n.pop(s, None)
        with open(out / blk / "graph.json", "w", encoding="utf-8") as f:
            json.dump({"version": 2, "nodes": nodes}, f)
        grown.sort(reverse=True)
        coeff_lost.sort(reverse=True)
        rec["blocks"][blk] = {
            "nodes": len(nodes),
            "grew": len(grown),
            "largest": [{"output": v, "ratio": r, "from": lab} for r, v, lab in grown[:3]],
            "records_from": took,
            "coeff_above_chosen": len(coeff_lost),
            "coeff_largest": [{"output": v, "ratio": r, "from": lab} for r, v, lab in coeff_lost[:3]],
        }
        log(
            f"{blk}: {len(nodes)} nodes, {len(grown)} grew"
            + ("; largest " + ", ".join(f"{v} x{r:.3g} ({lab})" for r, v, lab in grown[:3]) if grown else "")
            + f"; coefficient max above the chosen record's at {len(coeff_lost)}"
            + (
                " (largest " + ", ".join(f"{v} x{r:.3g} ({lab})" for r, v, lab in coeff_lost[:3]) + ")"
                if coeff_lost
                else ""
            )
        )
    with open(out / "passmerge.json", "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1)
    return rec


def plan_sites(plan_dir, graph) -> dict[str, list[tuple[str, str]]]:
    """Per block, the refresh sites of a plan as ``(kind, var)``: ``placed`` (``placements``), ``hint`` (``hint_fire``)
    and ``deliberate`` (the input of every ``deliberate_bootstrap`` node of ``graph``'s block, the carry and the PCMM
    finishes' inputs excluded: an external finish lists its inputs, not one var)."""
    pd = pathlib.Path(plan_dir)
    g = graphs(graph)
    out: dict[str, list[tuple[str, str]]] = {}
    for blk, nodes in g.items():
        p = pd / f"{blk}_placement.json"
        if not p.exists():
            continue
        with open(p, encoding="utf-8") as f:
            doc = json.load(f)
        sites = [("placed", x["target_var"]) for x in doc.get("placements", [])]
        sites += [("hint", v) for v in doc.get("hint_fire", [])]
        sites += [
            ("deliberate", n["inputs"][0])
            for n in nodes
            if n["op_type"] == "deliberate_bootstrap" and "external" not in (n.get("args") or {})
        ]
        out[blk] = sites
    return out


def margins(plan_dir, merged, captures: dict[str, str], kappa: float = 2.0) -> dict:
    """The per-site margin table (the module docstring): per site, per capture label, ``abs`` = the capture's
    ``output_max_abs`` / (κ × the merged one), ``coeff`` the same of ``output_max_coeff``, and the worst of all."""
    gm = {b: {n["output"]: n for n in ns} for b, ns in graphs(merged).items()}
    gc = {
        lab: {b: {n["output"]: n for n in ns} for b, ns in graphs(c).items()} for lab, c in captures.items()
    }
    rows = []
    for blk, sites in plan_sites(plan_dir, merged).items():
        for kind, v in sites:
            m = gm[blk].get(v)
            if m is None:
                rows.append({"block": blk, "kind": kind, "var": v, "missing": True})
                continue
            ma, mc = float(m.get("output_max_abs") or 0.0), float(m.get("output_max_coeff") or 0.0)
            ratios: dict[str, dict[str, float]] = {"abs": {}, "coeff": {}}
            for lab in captures:
                n = gc[lab][blk].get(v) or {}
                a, c = float(n.get("output_max_abs") or 0.0), float(n.get("output_max_coeff") or 0.0)
                ratios["abs"][lab] = a / max(kappa * ma, 1e-300)
                ratios["coeff"][lab] = c / max(kappa * mc, 1e-300)
            worst = max(
                ((x, lab, st) for st, xs in ratios.items() for lab, x in xs.items()),
                default=(0.0, None, None),
            )
            row: dict[str, Any] = {"block": blk, "kind": kind, "var": v, "step": m.get("step"), "merged_abs": ma,
                                   "merged_coeff": mc, **ratios}  # fmt: skip
            row["worst"] = {"ratio": worst[0], "capture": worst[1], "stat": worst[2]}
            rows.append(row)
    return {
        "plan": str(plan_dir),
        "merged": str(merged),
        "captures": dict(captures),
        "kappa": kappa,
        "rows": rows,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("diff", help="structure_diff of two captures")
    d.add_argument("a")
    d.add_argument("b")
    m = sub.add_parser("merge", help="the merged capture")
    m.add_argument("out")
    m.add_argument("base")
    m.add_argument("extra", nargs="+")
    m.add_argument("--labels", nargs="+", default=None)
    m.add_argument("--json", default=None, help="also write the record here")
    g = sub.add_parser("margins", help="the per-site margin table of a plan")
    g.add_argument("plan")
    g.add_argument("merged")
    g.add_argument("captures", nargs="+", metavar="LABEL=CAPTURE")
    g.add_argument("--kappa", type=float, default=2.0)
    g.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "diff":
        rep = structure_diff(a.a, a.b)
        n = 0
        for blk, r in rep.items():
            n += r["differences"]
            print(f"{blk}: {r['nodes'][0]} vs {r['nodes'][1]} nodes, {r['differences']} differences"
                  + ("; " + "; ".join(r["first"]) if r["first"] else ""))  # fmt: skip
        print(f"total differences outside {', '.join(STATS)}: {n}")
        return 0 if n == 0 else 1
    if a.cmd == "merge":
        rec = merge(a.base, a.extra, a.out, a.labels)
        if a.json:
            pathlib.Path(a.json).write_text(json.dumps(rec, indent=1))
        return 0
    caps = dict(x.split("=", 1) for x in a.captures)
    rep = margins(a.plan, a.merged, caps, a.kappa)
    rows = [r for r in rep["rows"] if not r.get("missing")]
    for r in sorted(rows, key=lambda r: -r["worst"]["ratio"])[:12]:
        print(f"{r['block']} {r['kind']} {r['var']} ({r['step']}): worst {r['worst']['ratio']:.4g} "
              f"({r['worst']['capture']}, {r['worst']['stat']})")  # fmt: skip
    print(f"sites {len(rep['rows'])}, missing {sum(1 for r in rep['rows'] if r.get('missing'))}, "
          f"worst {max((r['worst']['ratio'] for r in rows), default=0.0):.4g}")  # fmt: skip
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
