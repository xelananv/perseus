# SPDX-License-Identifier: MIT
"""The pass kernels on the capture recorder at the synthetic shape (LOGN 11, ``synthetic(L=2, d=32)``, t = 32, the
preset's levels): one op stream in every configuration, the recorder's branch of the pass hook, the history injection
and the merge. Six captures at capture tick 1 (tick; one sequence on ``[0, 64)`` entering fresh; the same entering
carried, ``prefill:320``'s second chunk; ``mixed``'s second pass; a chunk of a 606-position document and a tick at
position 149 of 64 conversations, both entered from a history on the float64 reference) hold the same
``graph.json`` in every field but the five magnitude statistics, tags included; 106 pass products per block under the
105 mask ids; a pass product's tag is value-free; the REAL halves are today's; the merged capture takes each node's
record from the capture with the largest slot maximum; the stock plan of the merged capture compiles against every
configuration's capture with 0 check errors."""

import json
import os
import pathlib
import subprocess

import numpy as np
import pytest

from perseus.examples.fhe_ssm import fakes, passkernel, passmasks, passmerge, planned, ref, run_fake
from perseus.examples.fhe_ssm.model import FheSsmModel
from perseus.examples.fhe_ssm.passmasks import PassConfig, Segment
from perseus.examples.fhe_ssm.synthetic import synthetic
from perseus.impl import fake

from .test_planned_contract import plan_of

D, DFF, L, LOGN, NQ = 32, 128, 2, 11, 64
SLOTS = 1 << (LOGN - 1)
LABELS = ("tick", "seqfresh", "seqcarry", "mixed", "longdoc", "longtick")
DOC_T0, TICK_T0 = 606, 148


def params_of(seed=21):
    return synthetic(L=L, d=D, seed=seed)


def tokens(rng, n, vocab):
    return [int(x) for x in rng.integers(0, vocab, n)]


def configurations(vocab, seed=1):
    """``{label: (two passes, history or None)}``: the captured pass is the second (capture tick 1)."""
    rng = np.random.default_rng(seed)
    tick = passmasks.scenario("tick:2", vocab=vocab, seed=seed)
    fresh = PassConfig((Segment.seq(0, NQ, 0, "fresh", "L"),), tuple(tokens(rng, NQ, vocab)), "seqfresh.1")
    doc = [
        PassConfig((Segment.seq(0, NQ, 0, "carry", "D"),), tuple(tokens(rng, NQ, vocab)), f"longdoc.{p}")
        for p in range(2)
    ]
    long_tick = passmasks.scenario("tick:2", vocab=vocab, seed=seed + 1)
    long_tick[1] = PassConfig(
        long_tick[1].segments, tuple(passmasks.GENERATED for _ in range(NQ)), long_tick[1].name
    )
    return {
        "tick": (tick, None),
        "seqfresh": ([tick[0], fresh], None),
        "seqcarry": (passmasks.scenario("prefill:320", vocab=vocab, seed=seed)[:2], None),
        "mixed": (passmasks.scenario("mixed", vocab=vocab, seed=seed)[:2], None),
        "longdoc": (
            doc,
            {"positions": DOC_T0, "streams": {"D": tokens(rng, DOC_T0, vocab)}, "source": "drawn"},
        ),
        "longtick": (
            long_tick,
            {
                "positions": TICK_T0,
                "streams": {f"c{q}": tokens(rng, TICK_T0, vocab) for q in range(NQ)},
                "source": "drawn",
            },
        ),
    }


@pytest.fixture(scope="module")
def caps(tmp_path_factory):
    d = tmp_path_factory.mktemp("passcap")
    params, w, seeds, cfg = params_of()
    np.savez(d / "w.npz", **w)
    (d / "s.json").write_text(json.dumps(seeds))
    (d / "c.json").write_text(json.dumps(cfg))
    base = ["--weights", str(d / "w.npz"), "--seeds", str(d / "s.json"), "--config", str(d / "c.json"),
            "--logn", str(LOGN), "--levels", "preset"]  # fmt: skip
    out: dict = {"dir": d, "base": base, "params": params, "configs": configurations(params.vocab)}
    for lab, (cfgs, hist) in out["configs"].items():
        f = d / f"{lab}.passes.json"
        f.write_text(json.dumps(passmasks.dump_list(cfgs)))
        extra = []
        if hist is not None:
            h = d / f"{lab}.history.json"
            h.write_text(json.dumps(hist))
            extra = ["--inject-from-exact", str(h)]
        argv = [*base, "--pass-kernels", "on", "--lane-config", str(f), *extra,
                "--capture", str(d / f"cap_{lab}"), "--capture-tick", "1", "--out", str(d / f"{lab}.json")]  # fmt: skip
        assert run_fake.main(argv) == 0, lab
        out[lab] = d / f"cap_{lab}"
    argv = [*base, "--tokens", "2", "--capture", str(d / "cap_off"), "--capture-tick", "1",
            "--out", str(d / "off.json")]  # fmt: skip
    assert run_fake.main(argv) == 0
    out["off"] = d / "cap_off"
    return out


def nodes_of(cap, blk="block_0"):
    return passmerge.graphs(cap)[blk]


def pass_nodes(nodes):
    return [n for n in nodes if (n.get("args") or {}).get("pt", {}).get("kind") == "pass"]


# ── one op stream ───────────────────────────────────────────────────────────────────────────────


def test_every_configuration_records_the_same_graph_but_the_magnitudes(caps):
    blocks = sorted(passmerge.graphs(caps["tick"]))
    assert blocks == ["block_0", "block_1", "block_2"]
    for lab in LABELS[1:]:
        rep = passmerge.structure_diff(caps["tick"], caps[lab])
        assert sorted(rep) == blocks
        assert {b: r["differences"] for b, r in rep.items()} == dict.fromkeys(blocks, 0), (lab, rep)
    # and the comparison sees something: the magnitudes differ (a seq chunk against conversations)
    a, b = nodes_of(caps["tick"]), nodes_of(caps["seqcarry"])
    moved = sum(1 for x, y in zip(a, b, strict=True) if x.get("output_max_abs") != y.get("output_max_abs"))
    assert moved > len(a) // 2
    # the off capture is another op stream (no pass products), so it differs
    off = passmerge.structure_diff(caps["off"], caps["tick"])
    assert all(r["differences"] > 0 for b, r in off.items() if b != "block_2")


def test_106_pass_products_per_block_under_the_105_mask_ids(caps):
    names = passmasks.mask_names(NQ // 2)
    for lab in LABELS:
        g = passmerge.graphs(caps[lab])
        for b in range(L):
            ps = pass_nodes(g[f"block_{b}"])
            assert len(ps) == 106, (lab, b)
            got = [n["args"]["pt"]["name"] for n in ps]
            assert set(got) == {passmasks.mask_id(x, b) for x in names}
            assert len(set(got)) == 105
            assert [x for x in set(got) if got.count(x) == 2] == ["pass/k2/Po"]
            by_step = {
                s: sum(1 for n in ps if n["step"].endswith(s)) for s in ("tm.shift", "tm.scan", "carry.prep")
            }
            assert by_step == {"tm.shift": 6, "tm.scan": 66, "carry.prep": 34}
            assert all(n["op_type"] == "mult" and n["args"]["pt"]["deg"] == 1 for n in ps)
        assert pass_nodes(g["block_2"]) == []  # the tail has no pass kernel


def test_a_pass_products_tag_is_value_free(caps):
    """``CapFhe.mult`` stamps ``t_mult(ct tag, Tag())``: known with period = slots and field UNKNOWN where the ciphertext's
    tag is known, unknown where it is not (the carried state the carry prep's own products feed); never a pack_field."""
    counts = set()
    for lab in LABELS:
        for b in range(L):
            ps = pass_nodes(nodes_of(caps[lab], f"block_{b}"))
            known = [n for n in ps if n["tag"]["period"]]
            for n in ps:
                assert "pack_field" not in n
                assert n["tag"]["field"] == 0
            for n in known:
                assert n["tag"]["period"] == SLOTS == n["pack_period"]
            for n in ps:
                if not n["tag"]["period"]:
                    assert "pack_period" not in n
            counts.add((len(known), len(ps) - len(known)))
    assert counts == {(69, 37)}  # the same split in every configuration and block


def test_the_real_halves_are_todays(caps):
    """The norm and gate halves keep their REAL field (the split's conjugate link) as in today's capture; the carry prep
    adds three REAL values (its folded halves ``2 Im S``, ``Z + conj Z``, ``mult_i(conj Z - Z)``)."""

    def real_by_step(cap):
        out: dict[str, int] = {}
        for n in nodes_of(cap):
            if n.get("pack_field") == 1:
                k = "carry.prep" if n["step"].endswith("carry.prep") else ".".join(n["step"].split(".")[1:3])
                out[k] = out.get(k, 0) + 1
        return out

    off, on = real_by_step(caps["off"]), real_by_step(caps["tick"])
    assert on.pop("carry.prep") == 3
    assert on == off
    assert set(on) == {"tm.norm", "tm.gate", "cm.norm", "cm.gate"}


def test_each_capture_records_its_configuration(caps):
    for lab, (cfgs, hist) in caps["configs"].items():
        st = json.loads((caps[lab] / "capture_pass.json").read_text())
        assert st["capture_tick"] == 1
        assert st["pass"]["name"] == cfgs[1].name
        assert st["pass"]["segments"] == [s.to_json() for s in cfgs[1].segments]
        assert st["mask_key"] == json.loads(json.dumps(cfgs[1].mask_key()))
        if hist is None:
            assert st["inject"] is None
        else:
            assert st["inject"]["positions"] == hist["positions"]
            assert st["inject"]["streams"] == sorted(hist["streams"])
    rep = json.loads((caps["dir"] / "longdoc.json").read_text())
    assert rep["inject"]["path"].endswith("longdoc.history.json")
    assert rep["ticks"][0]["relErrRms_vs_exact"] <= 1e-12  # scored from the injected state


# ── the history injection ───────────────────────────────────────────────────────────────────────


def fake_model(params):
    lim, land = fakes.level_preset("preset")
    fhe, inf = fakes.make_runtime_fake(
        N=SLOTS, hid=D, dim=D, H=1, H_real=1, E=DFF, E_real=DFF, rot_steps=(), complex_payload=True,
        limit=lim, bts_out=land,
    )  # fmt: skip
    model = FheSsmModel(inf, params, core=fake.core, unit=fhe.unit, pass_kernels="on")
    run_fake._rot_band(fhe, model)
    return model


@pytest.mark.parametrize("lab", ["longdoc", "longtick"])
def test_an_injected_history_continues_on_the_fake(caps, lab):
    """The registers landed from the float64 reference's history: the fake's replies of both passes against the
    trainer's recurrence stepped one position at a time through the history and the passes (``ExactModel.step``,
    no pass logic): a document chunk at positions 606..733 on one lane, a tick at positions 148 and 149 of 64
    conversations (the second pass's tokens generated from the first's reply)."""
    params = caps["params"]
    cfgs, hist = caps["configs"][lab]
    per, state = run_fake.history_states(params, cfgs[0], hist)
    model = fake_model(params)
    model.start(state=state)
    emb = np.asarray(params.emb, dtype=np.float64)
    if lab == "longdoc":
        seq = ref.ExactModel(params, 1)
        for k in hist["streams"]["D"]:
            seq.step(params.emb_row(k)[None])
        for i in range(L):  # the history in one call is the one stepped position by position
            assert np.abs(per["D"][0][i] - seq.s[i][0]).max() <= 1e-15
            assert np.abs(per["D"][1][i] - seq.u_prev[i][0]).max() <= 1e-15
    else:
        seq = ref.ExactModel(params, NQ)
        for t in range(TICK_T0):
            seq.step(np.stack([params.emb_row(hist["streams"][f"c{q}"][t]) for q in range(NQ)]))
    worst, prev = 0.0, None
    for p, cfg0 in enumerate(cfgs):
        cfg = passmasks.resolve_generated(cfg0, prev, emb)
        rows = passmasks.pass_rows(cfg, params.emb_row)
        got = model.step(rows, p, pass_cfg=cfg)["out"]
        if lab == "longdoc":
            want = np.stack([seq.step(rows[q][None])[0] for q in range(NQ)])
        else:
            want = seq.step(rows)
        worst = max(worst, *(ref.rel_err_rms(got[q], want[q]) for q in range(NQ)))
        prev = got
    assert worst <= 1e-12


def test_a_history_names_every_carried_stream():
    params, *_ = params_of()
    cfg = PassConfig((Segment.seq(0, NQ, 0, "carry", "D"),), tuple([1] * NQ), "x")
    with pytest.raises(ValueError, match="missing"):
        run_fake.history_states(params, cfg, {"positions": 3, "streams": {"E": [1, 2, 3]}})
    with pytest.raises(ValueError, match="holds 3 tokens"):
        run_fake.history_states(params, cfg, {"positions": 3, "streams": {"D": [1, 2]}})
    with pytest.raises(ValueError, match="pass_kernels"):
        run_fake.run(params, np.zeros((2, 1), dtype=int), logn=LOGN, inject={"positions": 1, "streams": {}})
    _per, (s_rows, _u) = run_fake.history_states(params, cfg, {"positions": 2, "streams": {"D": [1, 2]}})
    assert all(np.array_equal(r, np.repeat(r[:1], NQ, axis=0)) for r in s_rows)  # the home's broadcast


# ── the merge ───────────────────────────────────────────────────────────────────────────────────


def fake_capture(root: pathlib.Path, maxes: dict[str, list[float]]):
    """A two-block capture of hand-made nodes: ``maxes[block]`` the ``output_max_abs`` of its nodes, the other four
    statistics derived from it (so a record copied whole is visible)."""
    root.mkdir(parents=True)
    (root / "capture_config.json").write_text('{"bts": {}}')
    (root / "capture_env.json").write_text("{}")
    for blk, ms in maxes.items():
        (root / blk).mkdir()
        stats = [{"output_max_abs": m, "output_mean": m / 10, "output_max_dev": m * 1.1,
                  "output_max_coeff": m / 100, "output_max_coeff_ac": m / 200} for m in ms]  # fmt: skip
        nodes = [
            {
                "op_type": "mult",
                "inputs": [f"v_{i}", "pt_1"],
                "output": f"v_{i + 1}",
                "step": "blk0.x",
                "output_level": 30,
                "tag": {"period": 0},
                **st,
            }
            for i, st in enumerate(stats)
        ]
        (root / blk / "graph.json").write_text(json.dumps({"version": 2, "nodes": nodes}))
    return root


def test_merge_takes_the_largest_record_whole(tmp_path):
    a = fake_capture(tmp_path / "a", {"block_0": [1.0, 2.0, 3.0], "block_1": [5.0]})
    b = fake_capture(tmp_path / "b", {"block_0": [4.0, 2.0, 1.0], "block_1": [5.0]})
    c = fake_capture(tmp_path / "c", {"block_0": [0.5, 2.5, 1.0], "block_1": [7.0]})
    lines: list[str] = []
    rec = passmerge.merge(a, [b, c], tmp_path / "m", labels=["A", "B", "C"], log=lines.append)
    g = passmerge.graphs(tmp_path / "m")
    want = {"block_0": [(4.0, "B"), (2.5, "C"), (3.0, "A")], "block_1": [(7.0, "C")]}
    for blk, rows in want.items():
        for n, (m, _lab) in zip(g[blk], rows, strict=True):
            assert (n["output_max_abs"], n["output_mean"], n["output_max_dev"]) == (m, m / 10, m * 1.1)
            assert (n["output_max_coeff"], n["output_max_coeff_ac"]) == (m / 100, m / 200)
    assert rec["blocks"]["block_0"]["records_from"] == {"A": 1, "B": 1, "C": 1}
    assert rec["blocks"]["block_0"]["grew"] == 2
    assert rec["blocks"]["block_0"]["largest"][0] == {"output": "v_1", "ratio": 4.0, "from": "B"}
    assert lines[0].startswith(
        "block_0: 3 nodes, 2 grew (0 from an exact zero in the base); largest v_1 x4 (B), v_2 x1.25 (C)"
    )
    # an exact zero in the base grows without a ratio
    z = fake_capture(tmp_path / "z", {"block_0": [0.0, 2.0, 3.0], "block_1": [5.0]})
    rz = passmerge.merge(z, [b], tmp_path / "mz", log=lambda _s: None)["blocks"]["block_0"]
    assert (rz["grew"], rz["grew_from_zero"], [x["output"] for x in rz["largest"]]) == (1, 1, [])
    # a tie keeps the base's record
    t = fake_capture(tmp_path / "t", {"block_0": [1.0, 2.0, 3.0], "block_1": [5.0]})
    tt = json.loads((t / "block_0" / "graph.json").read_text())
    tt["nodes"][0]["output_mean"] = 99.0
    (t / "block_0" / "graph.json").write_text(json.dumps(tt))
    passmerge.merge(a, [t], tmp_path / "m2", log=lambda _s: None)
    assert passmerge.graphs(tmp_path / "m2")["block_0"][0]["output_mean"] == 0.1
    # the stamps travel; the record is written; a structure difference and an existing directory are refused
    assert (tmp_path / "m" / "capture_config.json").exists()
    assert json.loads((tmp_path / "m" / "passmerge.json").read_text())["labels"] == ["A", "B", "C"]
    x = fake_capture(tmp_path / "x", {"block_0": [1.0, 2.0, 3.0], "block_1": [5.0]})
    xx = json.loads((x / "block_0" / "graph.json").read_text())
    xx["nodes"][1]["output_level"] = 28
    (x / "block_0" / "graph.json").write_text(json.dumps(xx))
    with pytest.raises(ValueError, match="outside the magnitudes"):
        passmerge.merge(a, [x], tmp_path / "m3", log=lambda _s: None)
    assert not (tmp_path / "m3").exists()  # the partial output is removed
    with pytest.raises(ValueError, match="exists"):
        passmerge.merge(a, [b], tmp_path / "m", log=lambda _s: None)
    assert passmerge.main(["diff", str(a), str(b)]) == 0
    assert passmerge.main(["diff", str(a), str(x)]) == 1


def test_the_merged_synthetic_capture_is_the_slot_max_of_the_six(caps, tmp_path):
    out = tmp_path / "merged"
    rec = passmerge.merge(
        caps["tick"], [caps[x] for x in LABELS[1:]], out, labels=list(LABELS), log=lambda _s: None
    )
    gm = passmerge.graphs(out)
    gs = [passmerge.graphs(caps[x]) for x in LABELS]
    for blk, nodes in gm.items():
        for i, n in enumerate(nodes):
            got = [g[blk][i].get("output_max_abs") for g in gs]
            assert n.get("output_max_abs") == max((x for x in got if x is not None), default=None)
        assert sum(rec["blocks"][blk]["records_from"].values()) == len(nodes)
    assert passmerge.structure_diff(caps["tick"], out)["block_0"]["differences"] == 0
    # the merged capture holds no single configuration: the sources' stamps are in its record instead
    assert not (out / "capture_pass.json").exists()
    assert (out / "capture_config.json").exists()
    assert (out / "block_0" / "capture_config.json").exists()
    src = json.loads((out / "passmerge.json").read_text())["sources"]
    assert src["longdoc"]["inject"]["positions"] == DOC_T0
    assert src["tick"]["pass"]["name"] == "tick.1"


# ── the plan of the merged capture binds every configuration ───────────────────────────────────


def test_the_merged_stock_plan_compiles_against_every_configuration(caps, tmp_path):
    merged = tmp_path / "merged"
    passmerge.merge(
        caps["tick"], [caps[x] for x in LABELS[1:]], merged, labels=list(LABELS), log=lambda _s: None
    )
    plan = plan_of(merged, tmp_path, "syn_pass_stock")
    assert "check errors 0;" in (plan / "summary.txt").read_text()
    # the planner chains block b + 1 from block b's last op that is not a deliberate refresh (emit.py exit_var): the
    # residual's add, as today (with the carry prep last in the block, it was the carry prep's final add)
    gm = passmerge.graphs(merged)
    for b in range(L):
        s = json.loads((plan / f"block_{b}_placement.json").read_text())["summary"]
        exit_node = next(n for n in gm[f"block_{b}"] if n["output"] == s["exit_var"])
        assert exit_node["step"] == f"blk{b}.cm.v", exit_node["step"]
        assert exit_node["op_type"] == "add"
    for lab in LABELS:
        js = tmp_path / f"cc_{lab}.json"
        p = subprocess.run(
            [str(planned.default_check_bin()), str(plan), str(caps[lab]), "--json", str(js)],
            capture_output=True, text=True, check=False, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        )  # fmt: skip
        assert p.returncode == 0, p.stdout[-1500:]
        assert "check errors 0;" in p.stdout, lab
        tot = json.loads(js.read_text())["totals"]
        assert tot["errors"] == 0
        assert tot["delib"] == 2 * L  # the canonical carry, two per block
    # the margins of the merged plan: no captured configuration exceeds its merged value
    rep = passmerge.margins(plan, merged, {lab: str(caps[lab]) for lab in LABELS})
    rows = [r for r in rep["rows"] if not r.get("missing")]
    assert rows
    assert len(rows) == len(rep["rows"])
    # kappa = 2 over the slot max they were merged from
    assert max(max(r["abs"].values()) for r in rows) <= 1 / 2
    assert {r["kind"] for r in rows} >= {"placed", "deliberate"}


def test_the_pass_hook_branch_on_the_recorder_counts_its_products(caps):
    rep = json.loads((caps["dir"] / "tick.json").read_text())
    assert [t["pass_products"] for t in rep["ticks"]] == [L * 106, L * 106]
    assert passkernel.COUNTER == "pass_products"


def test_merge_max_each_takes_every_statistics_maximum(tmp_path):
    """The ``max_each`` rule: each magnitude statistic its maximum over the captures, the signed mean the largest-slot
    record's; nothing given up (no coefficient above the chosen record)."""
    a = fake_capture(tmp_path / "a", {"block_0": [1.0, 2.0, 3.0], "block_1": [5.0]})
    b = fake_capture(tmp_path / "b", {"block_0": [4.0, 2.0, 1.0], "block_1": [5.0]})
    bb = json.loads((b / "block_0" / "graph.json").read_text())
    bb["nodes"][2]["output_max_coeff"] = 9.0  # the smaller-slot capture holds the larger coefficient maximum
    bb["nodes"][2]["output_mean"] = -7.0
    (b / "block_0" / "graph.json").write_text(json.dumps(bb))
    lines: list[str] = []
    rec = passmerge.merge(a, [b], tmp_path / "m", labels=["A", "B"], log=lines.append, rule="max_each")
    g = passmerge.graphs(tmp_path / "m")["block_0"]
    assert g[0]["output_max_abs"] == 4.0  # B's record (larger slots)
    assert g[0]["output_max_coeff"] == 4.0 / 100
    assert g[2]["output_max_abs"] == 3.0  # A's slots
    assert g[2]["output_max_dev"] == 3.0 * 1.1
    assert g[2]["output_max_coeff"] == 9.0  # B's coefficient maximum taken although A's slots are larger
    assert g[2]["output_mean"] == 3.0 / 10  # the mean of the largest-slot record (A), not B's -7
    assert rec["rule"].startswith("max of each statistic")
    assert rec["blocks"]["block_0"]["coeff_above_chosen"] == 0
    assert rec["blocks"]["block_0"]["coeff_largest"] == []
    assert rec["blocks"]["block_0"]["records_from"] == {"A": 2, "B": 1}
    with pytest.raises(ValueError, match="rule"):
        passmerge.merge(a, [b], tmp_path / "m2", log=lambda _s: None, rule="other")
