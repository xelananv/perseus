# SPDX-License-Identifier: MIT
"""Several captures of one op stream (the pass kernels' configurations) checked node for node and merged into the one
capture a plan is cut on; the per-site margins of a plan against them.

    python -m perseus.examples.fhe_ssm.passmerge diff A B
    python -m perseus.examples.fhe_ssm.passmerge merge OUT BASE EXTRA [EXTRA ...] [--labels L0 L1 ...] [--rule record|max_each] [--json F]
    python -m perseus.examples.fhe_ssm.passmerge margins PLAN MERGED LABEL=CAPTURE [...] [--kappa 2] [--json F]

**One op stream.** Under ``--pass-kernels on`` every configuration issues the same products, rotations and refreshes
with plaintexts whose graph record holds no value (``passkernel.pass_mult_recorder``), so two captures of the model in
two configurations hold the same ``block_<b>/graph.json`` in every field (``op_type``, ``inputs``, ``input_levels``,
``output``, ``step``, the levels and degrees, ``tag``, ``pack_*``, ``pack_field``, ``shadow``, ``args`` with
``args.pt`` = kind, level, degree, name) but the five magnitude statistics the recorder measures on the values
(``STATS``: ``impl/magtape.node_stats``). ``structure_diff`` compares two captures that way, names verbatim (no
renumbering: the captures run the same stream before the capture tick, so even the ``anon_<n>`` names agree).

**The merge** (``merge``): the base capture's stamps copied (``capture_config.json``, ``capture_env.json`` and every
block's; not its ``capture_pass.json``, which names one configuration: the sources' stamps go into ``passmerge.json``),
then per node every statistic taken from the capture with the largest ``output_max_abs`` (the base on a tie, then the
earliest extra): one capture's record of one value, so ``output_mean`` / ``output_max_dev`` / the coefficient maxima stay
consistent with the slot maximum the planner sizes the refresh by (``placer/refresh.py``: ``max_abs``, ``max_coeff``,
``max_coeff_ac``, ``mean``, ``max_dev`` of the site's producer). Upstream's ``scripts/utils/merge_mags.py`` maxes each
``output_max_*`` field separately and keeps the base's ``output_mean``; the ``record`` rule keeps a node's record whole.
What it gives up is printed beside the merge: per block the nodes whose ``output_max_coeff`` in another capture exceeds
the chosen capture's (the coefficient maximum is the planner's CF key when captured, ``btserr.choose_site``) and the
largest such ratio. Measured at the model shape on the diagonal arm's stock plan, what it gives up is not small: a
fired hint's coefficient maximum under the continuing-chunk configuration reached 5.0 times the record's times κ
(``blk22.tm.out``), so the ``max_each`` rule (``RULES``) is the one the plans are cut on: every magnitude statistic
its maximum over the captures, the signed mean the largest-slot record's. A capture whose structure differs from the
base's is refused (the merged plan would bind one of them only;
the partial output is removed). The per-block lines follow ``merge_mags.py``'s ("grew": nodes whose record came from
another capture than the base, with the ratio of the chosen ``output_max_abs`` to the base's; a node that is an exact
zero in the base, a product by a mask that is zero in the base's configuration, is counted apart and has no ratio).

**The margins** (``margins``): for every refresh site of a plan (its placements, its fired hints, and the graph's
deliberate refreshes: the carry), each capture's ``output_max_abs`` and ``output_max_coeff`` of the site's producer
against the merged capture's value times κ (the planner's ``PLAN_MAG_SAFETY``, 2): a ratio above 1 is a value the plan's
correction factor was not sized to hold. Every function reads one block at a time (a model-shape capture of the
diagonal arm is about 20 MB of JSON per block).
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


def block_names(cap) -> list[str]:
    """The ``block_<b>`` directories of a capture holding a ``graph.json``, in block order."""
    root = pathlib.Path(cap)
    out = sorted((p.parent.name for p in root.glob("block_*/graph.json")), key=lambda s: int(s.split("_")[1]))
    if not out:
        raise ValueError(f"no block_<b>/graph.json under {root}")
    return out


def load_block(cap, blk: str) -> list[dict] | None:
    """The nodes of one block of a capture (None when the block is missing)."""
    p = pathlib.Path(cap) / blk / "graph.json"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)["nodes"]


def graphs(cap) -> dict[str, list[dict]]:
    """``{block_<b>: nodes}`` of a capture (every block in memory: for small captures)."""
    return {b: load_block(cap, b) or [] for b in block_names(cap)}


def structure(node: dict) -> dict:
    """A node without its magnitude statistics."""
    return {k: v for k, v in node.items() if k not in STATS}


def _block_diff(na: list[dict], nb: list[dict], show: int) -> tuple[int, list[str]]:
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
    return n, first


def structure_diff(a, b, show: int = 3) -> dict[str, dict]:
    """Per block of ``a`` (and of ``b``): ``{"nodes": (len a, len b), "differences": n, "first": [...]}``; a difference
    is a node index and a field whose values differ outside ``STATS`` (a missing block or node counts once)."""
    blocks = sorted(set(block_names(a)) | set(block_names(b)), key=lambda s: int(s.split("_")[1]))
    out: dict[str, dict] = {}
    for blk in blocks:
        na, nb = load_block(a, blk), load_block(b, blk)
        if na is None or nb is None:
            out[blk] = {"nodes": (len(na or []), len(nb or [])), "differences": 1, "first": ["block missing"]}
            continue
        n, first = _block_diff(na, nb, show)
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


#: the merge rules: ``record`` keeps every statistic of the capture with the largest ``output_max_abs`` (one capture's
#: record of one value); ``max_each`` takes the maximum of every magnitude statistic over the captures separately
#: (``output_max_abs``, ``output_max_dev``, ``output_max_coeff``, ``output_max_coeff_ac``: upstream's ``merge_mags.py``
#: rule) and the signed ``output_mean`` of that largest-slot record, so a site the planner sizes by the coefficient
#: maximum (a dense refresh in the shipped order, where the bootstrap's sine sees the coefficients) is sized for the
#: configuration whose coefficients are largest even when another configuration's slots are: nothing is given up
RULES = ("record", "max_each")
#: the statistics ``max_each`` takes the maximum of (every one but the signed mean)
MAXED = ("output_max_abs", "output_max_dev", "output_max_coeff", "output_max_coeff_ac")


def _merge_block(
    blocks: list[list[dict]], labels: list[str], rule: str = "record"
) -> tuple[list[dict], dict]:
    """One block's merged nodes (``blocks[0]``'s list, overwritten in place) and its record, under ``rule``."""
    if rule not in RULES:
        raise ValueError(f"rule must be one of {RULES}, got {rule!r}")
    nodes = blocks[0]
    took = dict.fromkeys(labels, 0)
    grown, coeff_lost, from_zero = [], [], 0
    for i, n in enumerate(nodes):
        cand = [g[i] for g in blocks]  # cand[0] is n itself: read everything before n is overwritten
        k = _chosen(cand)
        took[labels[k]] += 1
        src = cand[k]
        cm = float(src.get("output_max_coeff") or 0.0)
        top = max(range(len(cand)), key=lambda j: float(cand[j].get("output_max_coeff") or 0.0))
        ct = float(cand[top].get("output_max_coeff") or 0.0)
        if ct > cm and rule == "record":
            coeff_lost.append((ct / max(cm, 1e-300), n["output"], labels[top]))
        if k:
            b0 = float(n.get("output_max_abs") or 0.0)
            if b0 > 0.0:
                grown.append((float(src["output_max_abs"]) / b0, n["output"], labels[k]))
            else:
                from_zero += 1  # an exact zero in the base (a mask zero in its configuration): no ratio
        if rule == "record":
            if k:
                for s in STATS:
                    if s in src:
                        n[s] = src[s]
                    else:
                        n.pop(s, None)
            continue
        # max_each: the signed mean from the largest-slot record, every other statistic its maximum over the captures
        if "output_mean" in src:
            n["output_mean"] = src["output_mean"]
        else:
            n.pop("output_mean", None)
        for s in MAXED:
            vals = [float(c[s]) for c in cand if c.get(s) is not None]
            if vals:
                n[s] = max(vals)
            else:
                n.pop(s, None)
    grown.sort(reverse=True)
    coeff_lost.sort(reverse=True)
    rec = {
        "nodes": len(nodes),
        "grew": len(grown) + from_zero,
        "grew_from_zero": from_zero,
        "largest": [{"output": v, "ratio": r, "from": lab} for r, v, lab in grown[:3]],
        "records_from": took,
        "coeff_above_chosen": len(coeff_lost),
        "coeff_largest": [{"output": v, "ratio": r, "from": lab} for r, v, lab in coeff_lost[:3]],
    }
    return nodes, rec


def _merge_line(blk: str, rec: dict) -> str:
    """``merge_mags.py``'s per-block line, with the coefficient maxima the record rule gives up."""

    def few(xs):
        return ", ".join(f"{x['output']} x{x['ratio']:.3g} ({x['from']})" for x in xs)

    line = f"{blk}: {rec['nodes']} nodes, {rec['grew']} grew ({rec['grew_from_zero']} from an exact zero in the base)"
    if rec["largest"]:
        line += "; largest " + few(rec["largest"])
    line += f"; coefficient max above the chosen record's at {rec['coeff_above_chosen']}"
    if rec["coeff_largest"]:
        line += " (largest " + few(rec["coeff_largest"]) + ")"
    return line


def _merge_one(
    caps: list[pathlib.Path], labels: list[str], blk: str, out: pathlib.Path, rule: str = "record"
) -> dict:
    """Block ``blk`` of every capture: checked against the base's structure, merged, written under ``out``."""
    got = [load_block(c, blk) or [] for c in caps]
    for c, g in zip(caps[1:], got[1:], strict=True):
        n, first = _block_diff(got[0], g, 3)
        if n:
            raise ValueError(f"{c} differs from {caps[0]} outside the magnitudes ({blk}: {first})")
    nodes, brec = _merge_block(got, labels, rule)
    with open(out / blk / "graph.json", "w", encoding="utf-8") as f:
        json.dump({"version": 2, "nodes": nodes}, f)
    return brec


def merge(
    base, extras: Sequence, out, labels: Sequence[str] | None = None, log=print, rule: str = "record"
) -> dict:
    """The merged capture at ``out`` (a new directory) from ``base`` and ``extras`` (the module docstring) under
    ``rule`` (``RULES``); returns the record that is also written as ``out/passmerge.json``."""
    if rule not in RULES:
        raise ValueError(f"rule must be one of {RULES}, got {rule!r}")
    base, out = pathlib.Path(base), pathlib.Path(out)
    caps = [base, *(pathlib.Path(e) for e in extras)]
    labels = list(labels) if labels else [p.name for p in caps]
    if len(labels) != len(caps):
        raise ValueError(f"{len(labels)} labels for {len(caps)} captures")
    if out.exists():
        raise ValueError(f"{out} exists")
    blocks = block_names(base)
    for c in caps[1:]:
        if block_names(c) != blocks:
            raise ValueError(f"{c} holds blocks {block_names(c)}, {base} holds {blocks}")
    stamps = {}
    for lab, c in zip(labels, caps, strict=True):
        p = c / "capture_pass.json"
        stamps[lab] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    shutil.copytree(
        base, out, ignore=shutil.ignore_patterns("graph.json", "capture_pass.json", "passmerge.json")
    )
    rec: dict = {"base": str(base), "captures": [str(c) for c in caps], "labels": labels,
                 "rule": "max_abs record" if rule == "record" else "max of each statistic, the mean of the max_abs record",
                 "stats": list(STATS), "sources": stamps, "blocks": {}}  # fmt: skip
    try:
        for blk in blocks:
            rec["blocks"][blk] = _merge_one(caps, labels, blk, out, rule)
            log(_merge_line(blk, rec["blocks"][blk]))
    except ValueError:
        shutil.rmtree(out, ignore_errors=True)
        raise
    with open(out / "passmerge.json", "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1)
    return rec


def block_sites(placement: dict, nodes: list[dict]) -> list[tuple[str, str]]:
    """One block's refresh sites as ``(kind, var)``: ``placed`` (``placements``), ``hint`` (``hint_fire``) and
    ``deliberate`` (the input of every ``deliberate_bootstrap`` node of ``nodes``, the external PCMM finishes left out:
    a finish lists its inputs, not one value)."""
    sites = [("placed", x["target_var"]) for x in placement.get("placements", [])]
    sites += [("hint", v) for v in placement.get("hint_fire", [])]
    sites += [
        ("deliberate", n["inputs"][0])
        for n in nodes
        if n["op_type"] == "deliberate_bootstrap" and "external" not in (n.get("args") or {})
    ]
    return sites


def margins(plan_dir, merged, captures: dict[str, str], kappa: float = 2.0) -> dict:
    """The per-site margin table (the module docstring): per site, per capture label, ``abs`` = the capture's
    ``output_max_abs`` / (κ × the merged one), ``coeff`` the same of ``output_max_coeff``, and the worst of all."""
    pd = pathlib.Path(plan_dir)
    rows = []
    for blk in block_names(merged):
        p = pd / f"{blk}_placement.json"
        if not p.exists():
            continue
        with open(p, encoding="utf-8") as f:
            placement = json.load(f)
        mnodes = load_block(merged, blk) or []
        gm = {n["output"]: n for n in mnodes}
        gc = {lab: {n["output"]: n for n in (load_block(c, blk) or [])} for lab, c in captures.items()}
        for kind, v in block_sites(placement, mnodes):
            m = gm.get(v)
            if m is None:
                rows.append({"block": blk, "kind": kind, "var": v, "missing": True})
                continue
            ma, mc = float(m.get("output_max_abs") or 0.0), float(m.get("output_max_coeff") or 0.0)
            ratios: dict[str, dict[str, float]] = {"abs": {}, "coeff": {}}
            for lab in captures:
                n = gc[lab].get(v) or {}
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
    m.add_argument(
        "--rule",
        default="record",
        choices=RULES,
        help="record: one capture's whole record per node; max_each: every statistic its maximum",
    )
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
        rec = merge(a.base, a.extra, a.out, a.labels, rule=a.rule)
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
