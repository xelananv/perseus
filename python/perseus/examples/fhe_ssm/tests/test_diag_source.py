# SPDX-License-Identifier: MIT
"""``--diag-source generated`` (the device diagonal encoder's CPU-checkable half): the gather's index arithmetic
against the lazy diagonal table at the synthetic and the model shape; the fake's reply bit-identical under both
sources (the vectors built by the gather, so the test covers the arithmetic: a wrong roll changes the reply); the
generated capture equal to the values capture but for each diagonal plaintext's ``args.pt``; the planned twin keeping
names, not vectors; the stock plan cut on the values capture binding the generated recording with the same refreshes,
one fused sum per ``(m, gg)`` group and no store plaintext named ``<linear>#...``; the session's arguments and the
tile upload."""

import json
import pathlib

import numpy as np
import pytest

from perseus.examples.fhe_ssm import diaggen, fakes, filling, planned, run_fake, run_planned
from perseus.examples.fhe_ssm.model import FheSsmModel
from perseus.examples.fhe_ssm.synthetic import synthetic
from perseus.examples.fhe_ssm.tests.test_planned_contract import LOGN, checkpoint, plan_of
from perseus.impl import fake, prefill

# the synthetic shape of the planned-contract tests: 2^9 slots, d = 32, t = 16, B = 32, b = 8, g = 4
N_SYN, D_SYN, T_SYN = 1 << (LOGN - 1), 32, 16
SHAPES = ((32, 32), (32, 128), (128, 32))  # win / wout, wk / wr, wv at d = 32, d_ffn = 128
# per block: 14 tiles of 32 diagonals = 448 generated plaintexts in 44 groups (40 of b = 8 terms, wv's 4 of 4 x 8)
SYN_DIAGS, SYN_SUMS, SYN_TERMS = 448, 44, {8: 40, 32: 4}


def weight(seed, shape):
    return np.random.default_rng(seed).standard_normal(shape) / np.sqrt(shape[0])


# ── 1. the index arithmetic ────────────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("shape", SHAPES)
def test_the_gather_is_the_lazy_table_for_every_plaintext_of_the_synthetic_shape(shape):
    W = weight(3, shape)
    lazy = filling.LazyDiagonals(W, N_SYN, T_SYN)
    jobs = [
        (n, m, gg, jj)
        for m in range(lazy.n_out)
        for gg in range(lazy.g)
        for n in range(lazy.n_in)
        for jj in range(lazy.b)
    ]
    got = filling.diag_gather_reference(W, N_SYN, T_SYN, jobs)
    want = np.stack([lazy[m, gg][n, jj] for n, m, gg, jj in jobs])
    np.testing.assert_array_equal(got, want)
    # and upstream's own expansion (DiagLinear.slot_vectors: (n_out, g, n_in, b, N))
    up = prefill.DiagLinear.slot_vectors(W, N_SYN, T_SYN)
    np.testing.assert_array_equal(got, np.stack([up[m, gg, n, jj] for n, m, gg, jj in jobs]))
    assert len(jobs) == (shape[0] // 32) * (shape[1] // 32) * 32


@pytest.mark.parametrize("shape", [(1024, 4096), (4096, 1024)])
def test_the_gather_is_the_lazy_table_for_64_random_jobs_at_the_model_shape(shape):
    n_slots, t = 1 << 15, 32
    W = weight(5, shape)
    lazy = filling.LazyDiagonals(W, n_slots, t)
    assert (lazy.B, lazy.b, lazy.g) == (1024, 32, 32)
    rng = np.random.default_rng(11)
    jobs = [
        (
            int(rng.integers(lazy.n_in)),
            int(rng.integers(lazy.n_out)),
            int(rng.integers(lazy.g)),
            int(rng.integers(lazy.b)),
        )
        for _ in range(64)
    ]
    got = filling.diag_gather_reference(W, n_slots, t, jobs)
    for q, (n, m, gg, jj) in enumerate(jobs):
        np.testing.assert_array_equal(got[q], lazy[m, gg][n, jj])
    # one element by hand: slot v of (n, m, gg, jj) reads W[n B + ((i + gg b + jj) mod B), m B + i], i = ((v - gg b t)
    # mod n_slots) // t (np.roll moves elements to higher indices)
    n, m, gg, jj = jobs[0]
    for v in (0, 1, 31, 32, gg * 32 * 32, n_slots - 1):
        i = ((v - gg * 32 * t) % n_slots) // t
        assert got[0][v] == W[n * 1024 + (i + gg * 32 + jj) % 1024, m * 1024 + i]
        assert filling.diag_slot_index(v, n, m, gg, jj, 1024, t) == (
            n * 1024 + (i + gg * 32 + jj) % 1024,
            m * 1024 + i,
        )


# ── 2. the fake: one reply under both sources ──────────────────────────────────────────────────────────────────


def fake_model(source, linear="diagonal"):
    params, *_ = synthetic(L=2, seed=5)
    fhe, inf = fakes.make_runtime_fake(N=N_SYN, hid=params.d, dim=params.d, H=1, H_real=1, E=params.d_ffn,
                                       E_real=params.d_ffn, limit=44, bts_out=32, complex_payload=True)  # fmt: skip
    model = FheSsmModel(inf, params, core=fake.core, unit=fhe.unit, linear=linear, resident_weights=False)
    run_fake._rot_band(fhe, model)
    diaggen.use_source(inf, source)
    return params, fhe, model


def replies(source, ticks=3):
    params, fhe, model = fake_model(source)
    rng = np.random.default_rng(9)
    model.start()
    out = []
    for pos in range(ticks):
        rows = 0.3 * rng.standard_normal((2 * model.geo.base.t, params.d))
        rec = model.step(rows, pos)
        out.append((np.asarray(rec["out"]).copy(), rec["out_level"], fhe.n_bootstraps, len(fhe.calls)))
    return out


def test_the_fakes_reply_is_bit_identical_under_both_sources(monkeypatch):
    calls = []
    gather = filling.diag_gather_reference

    def counted(*a, **k):
        calls.append(len(a[3]))
        return gather(*a, **k)

    monkeypatch.setattr(filling, "diag_gather_reference", counted)
    v = replies("values")
    assert calls == []  # the values source never gathers
    g = replies("generated")
    for (ov, lv, bv, cv), (og, lg, bg, cg) in zip(v, g, strict=True):
        np.testing.assert_array_equal(og, ov)
        assert (lg, bg, cg) == (lv, bv, cv)
    # the generated run gathered every group of every tick (2 blocks x 44 groups x 3 ticks), the values run none
    assert len(calls) == 2 * SYN_SUMS * 3
    assert sorted(set(calls)) == [8, 32]


def test_a_wrong_roll_in_the_gather_changes_the_reply(monkeypatch):
    """The bit-identity above holds only if the index arithmetic is right: the gather with the roll's sign flipped
    (np.roll moving elements to lower indices) gives another reply."""
    good = filling.diag_slot_index

    def flipped(v, n, m, gg, jj, B, t):
        b, _g = prefill.DiagLinear.split(B)
        return good(np.asarray(v) + 2 * gg * b * t, n, m, gg, jj, B, t)

    v = replies("values", ticks=1)
    monkeypatch.setattr(filling, "diag_slot_index", flipped)
    g = replies("generated", ticks=1)
    assert not np.array_equal(g[0][0], v[0][0])


def test_the_source_switch():
    _p, _fhe, model = fake_model("generated")
    inf = model.inf
    assert diaggen.source_of(inf) == "generated"
    diaggen.use_source(inf, "values")
    assert diaggen.source_of(inf) == "values"
    with pytest.raises(ValueError, match="diag source"):
        diaggen.use_source(inf, "rows")
    with pytest.raises(TypeError, match="fake or the capture recorder"):
        diaggen.use_source(object(), "generated")


# ── 3. the capture: the same graph but args.pt ─────────────────────────────────────────────────────────────────


def graphs(d: pathlib.Path) -> dict[str, list[dict]]:
    return {p.parent.name: json.loads(p.read_text())["nodes"] for p in sorted(d.glob("block_*/graph.json"))}


@pytest.fixture(scope="module")
def captures(tmp_path_factory):
    """The synthetic checkpoint captured under both sources (2 blocks + the tail, ring 2^10, the preset's levels),
    and the stock plan of the values capture."""
    d = tmp_path_factory.mktemp("diagsrc")
    base = checkpoint(d)
    caps = {}
    for src in diaggen.SOURCES:
        caps[src] = d / f"cap_{src}"
        argv = [
            *base,
            "--tokens",
            "3",
            "--capture",
            str(caps[src]),
            "--capture-tick",
            "1",
            "--out",
            str(d / f"{src}.json"),
        ]
        assert run_fake.main([*argv, "--diag-source", src]) == 0
    return {"base": base, "dir": d, **caps, "plan": plan_of(caps["values"], d, "syn_stock_v")}


def test_the_generated_capture_is_the_values_capture_but_each_diagonals_args_pt(captures):
    gv, gg_ = graphs(captures["values"]), graphs(captures["generated"])
    assert sorted(gv) == sorted(gg_) == ["block_0", "block_1", "block_2"]
    for blk in gv:
        a, b = gv[blk], gg_[blk]
        assert len(a) == len(b)
        n_diag = 0
        for nx, ny in zip(a, b, strict=True):
            px, py = (nx.get("args") or {}).get("pt"), (ny.get("args") or {}).get("pt")
            if py is not None and py["kind"] == "diag":
                n_diag += 1
                assert px == {"kind": "values", "level": py["level"], "deg": py["deg"]}
                lin, idx = py["name"].rsplit("#", 1)
                assert lin.startswith(f"b{blk.split('_')[1]}.")
                assert len(idx.split(".")) == 4
                nx = {**nx, "args": {**nx["args"], "pt": None}}  # noqa: PLW2901 (the node without its args.pt)
                ny = {**ny, "args": {**ny["args"], "pt": None}}  # noqa: PLW2901
            assert nx == ny  # every op, var name, level, tag, pack field and magnitude
        assert n_diag == (SYN_DIAGS if blk != "block_2" else 0), blk
    # the plan's graph= line names the values capture's md5; the generated capture has its own
    md5 = run_planned.graphs_md5(captures["values"])
    assert run_planned.plan_capture(run_planned.plan_cmd_lines(str(captures["plan"])))[1] == md5
    assert run_planned.graphs_md5(captures["generated"]) != md5


def test_a_diag_plaintext_names_its_group_term_in_the_order_the_values_loop_reads(captures):
    names = [
        n["args"]["pt"]["name"]
        for n in graphs(captures["generated"])["block_0"]
        if (n.get("args") or {}).get("pt", {}).get("kind") == "diag"
    ]
    assert len(names) == len(set(names)) == SYN_DIAGS
    by_lin: dict[str, list[tuple[int, ...]]] = {}
    for s in names:
        lin, idx = s.rsplit("#", 1)
        by_lin.setdefault(lin, []).append(tuple(int(x) for x in idx.split(".")))
    assert sorted(by_lin) == ["b0.win", "b0.wk", "b0.wout", "b0.wr", "b0.wv"]
    # within a linear: m, then gg, then n, then jj (apply_diag's loops, DiagLinear._sum's n-outer / jj-inner terms)
    for lin, idx in by_lin.items():
        assert idx == sorted(idx, key=lambda q: (q[1], q[2], q[0], q[3])), lin


# ── 4. the planned twin and the compile ────────────────────────────────────────────────────────────────────────


def prepare(captures, tmp_path, source, *extra):
    out = tmp_path / f"prep_{source}.json"
    rc = run_planned.main(
        [
            *captures["base"],
            "--plan",
            str(captures["plan"]),
            "--prepare-only",
            "--diag-source",
            source,
            *extra,
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    d = json.loads(out.read_text())
    cc = json.loads((tmp_path / f"prep_{source}.work" / "compile_check.json").read_text())
    return d, cc


def test_the_stock_plan_binds_the_generated_recording_with_one_sum_per_group(captures, tmp_path):
    dv, cv = prepare(captures, tmp_path, "values")
    dg, cg = prepare(captures, tmp_path, "generated")
    pv, pg = dv["prepare"], dg["prepare"]
    assert pv["check_errors"] == pg["check_errors"] == {}
    assert pg["refreshes_planned"] == pv["refreshes_planned"] > 0
    assert pg["static_stream"] is True
    assert pg["dynamic_steps"] == {}
    assert pg["registers"] == pv["registers"]
    assert dg["capture"]["checked"]
    assert dg["capture"]["md5"] == dg["capture"]["plan_md5"]
    # the census: 44 sums per block, 40 of 8 terms and wv's 4 of 32; no store plaintext carries a generated name
    tg, tv = cg["totals"], cv["totals"]
    assert (tg["diag_sums"], tg["stream_sums"], tg["form_errors"], tg["store_named_generated"]) == (
        2 * SYN_SUMS,
        0,
        0,
        0,
    )
    assert (tv["diag_sums"], tv["store_named_generated"]) == (0, 0)
    assert {int(k): v for k, v in cg["sum_terms"]["diag"].items()} == {k: 2 * v for k, v in SYN_TERMS.items()}
    assert cg["pt_forms"]["diag"] == 2 * SYN_DIAGS
    assert tg["store_refs"] == tv["store_refs"] - 2 * SYN_DIAGS
    assert tg["sites"] == tv["sites"]
    rows_v = {r["block"]: r for r in cv["blocks"]}
    for r in cg["blocks"]:
        s = r["sums"]
        if r["block"] == "block_2":
            assert s["diag"] == 0
            assert r["launches"] == rows_v[r["block"]]["launches"]
            continue
        assert s["diag"] == SYN_SUMS
        assert s["fused_vars"] == 2 * SYN_DIAGS - 2 * SYN_SUMS
        # products + intra-group adds - sums: 448 + (448 - 44) - 44 = 808 launches fewer per block
        assert rows_v[r["block"]]["launches"] - r["launches"] == SYN_DIAGS + (SYN_DIAGS - SYN_SUMS) - SYN_SUMS
        assert (r["sites"], r["placed"], r["hint"], r["deliberate"]) == tuple(
            rows_v[r["block"]][k] for k in ("sites", "placed", "hint", "deliberate")
        )
    # the driver's own report: the plaintext budget counts store plaintexts only, the sums by source
    assert pg["plaintexts"]["plaintexts"] == pv["plaintexts"]["plaintexts"] - 2 * SYN_DIAGS
    assert pg["sums"]["diag"] == 2 * SYN_SUMS
    assert pv["sums"]["diag"] == 0
    assert pg["diag_tiles"] == {"n": 10, "bytes": 2 * 8 * (32 * 32 * 2 + 32 * 128 * 3)}
    assert "diag_tiles" not in pv
    # the key request reads no plaintext: the same under both sources
    assert dg["key_needs"] == dv["key_needs"]


def test_lean_prepare_under_generated_still_binds(captures, tmp_path):
    dg, cg = prepare(captures, tmp_path, "generated", "--lean")
    assert dg["prepare"]["check_errors"] == {}
    assert dg["prepare"]["static_stream"] is True
    assert cg["totals"]["diag_sums"] == 2 * SYN_SUMS


def planned_run(captures, tmp_path, source):
    a = run_planned.parser().parse_args(
        [*captures["base"], "--plan", str(captures["plan"]), "--prepare-only", "--diag-source", source]
    )
    params, model, _twin_cfg, _levels = run_planned.make_twin(a)
    cfg = run_planned.session_config(str(captures["plan"]))
    rng = np.random.default_rng(4)
    streams = [[int(x) for x in rng.integers(0, params.vocab, 3)] for _ in range(model.capacity)]
    rows = run_planned.TokenRows(params, streams, 3)
    labels = [f"block_{b}" for b in range(model.n_layers + 1)]
    run = planned.PlannedSsmRun(planned.SsmRecorder(model, model.n_layers), str(captures["plan"]), cfg, labels, None,
                                model=model, work_dir=str(tmp_path / f"w_{source}"), diag_source=source, log=lambda _m: None)  # fmt: skip
    return run.prepare(rows), model


def test_the_twin_keeps_names_not_vectors_and_the_session_arguments_filter_them(captures, tmp_path):
    run, model = planned_run(captures, tmp_path, "generated")
    for b in range(model.n_layers):
        kept = [v for v in run.ref[b].pts.values() if isinstance(v, diaggen.GeneratedPt)]
        assert len(kept) == SYN_DIAGS
        assert all("#" in v.name for v in kept)
    kv = run.session_kwargs()
    tiles = kv["diag_tiles"]
    assert sorted(tiles) == sorted(f"b{b}.{k}" for b in range(2) for k in ("win", "wout", "wk", "wr", "wv"))
    assert tiles["b1.wk"] == (32, 128, T_SYN)
    assert tiles["b0.wv"] == (128, 32, T_SYN)
    run_v, _ = planned_run(captures, tmp_path, "values")
    kvv = run_v.session_kwargs()
    assert "diag_tiles" not in kvv
    assert kvv["pt_slots"] - kv["pt_slots"] == 2 * SYN_DIAGS
    assert kv["pt_slots"] == len(kv["pt_levels"]) + 8
    assert run.plaintext_budget()["plaintexts"] == run_v.plaintext_budget()["plaintexts"] - 2 * SYN_DIAGS

    class Sess:  # what upload_tiles needs of a device session
        def __init__(self):
            self.got = {}

        def register_tile(self, name, W):
            assert W.flags["C_CONTIGUOUS"]
            assert W.dtype == np.float64
            self.got[name] = W

    run.sess = Sess()
    assert run.upload_tiles() == 10
    for name, W in run.sess.got.items():
        b, kind = name[1:].split(".")
        want = getattr(model.block_weights(int(b)), kind).pts.W
        np.testing.assert_array_equal(W, want)  # the folded matrix LazyDiagonals reads
        assert W.shape == tiles[name][:2]
    run_v.sess = Sess()
    assert run_v.upload_tiles() == 0
    assert run_v.sess.got == {}


def test_a_device_build_without_the_encoder_is_refused(captures, tmp_path, monkeypatch):
    from perseus import _pc  # type: ignore[attr-defined]

    class Session:  # a device build of today: no register_tile
        pass

    monkeypatch.setattr(_pc, "Session", Session, raising=False)
    with pytest.raises(SystemExit, match="no device diagonal encoder"):
        run_planned.main(
            [
                *captures["base"],
                "--plan",
                str(captures["plan"]),
                "--diag-source",
                "generated",
                "--keyspec",
                str(tmp_path / "k.json"),
            ]
        )
    run, _ = planned_run(captures, tmp_path, "generated")
    assert run.device_has_encoder() is False
    with pytest.raises(RuntimeError, match="no device diagonal encoder"):
        run.setup([])
