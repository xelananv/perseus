# SPDX-License-Identifier: MIT
"""The per-linear PCMM CF table (``pcmm_cf_table.py``) and its fallback in the driver: the table file's parsing, the
magnitudes read from a capture's finishes, the rule's calibration point, and which linears take the table's CF and
which ``--pcmm-cf`` (``PlannedSsmRun.applied_cf_map``, what ``Session(pcmm_cf_map=...)`` receives)."""

from __future__ import annotations

import json
import types

import pytest

from .. import pcmm_cf_table
from ..planned import PlannedSsmRun


def finish(linear, out, mag, shift=0):
    d = {"op": "pcmm", "linear": linear, "out": out, "n_in": 1, "n_out": 2, "t": 32, "shift": shift}
    return {"op_type": "deliberate_bootstrap", "inputs": ["v_0"], "output": f"v_{linear}_{out}",
            "output_max_abs": mag, "args": {"external": d, "n_inputs": 1}}  # fmt: skip


def capture(tmp_path, blocks):
    for b, nodes in enumerate(blocks):
        (tmp_path / f"block_{b}").mkdir()
        (tmp_path / f"block_{b}" / "graph.json").write_text(json.dumps({"version": 2, "nodes": nodes}))
    return tmp_path


def test_load_reads_the_cf_object_or_a_flat_map(tmp_path):
    p = tmp_path / "t.json"
    p.write_text(json.dumps({"cf": {"b0.win": 8, "b0.wv": 3}, "safety": 2.0}))
    assert pcmm_cf_table.load(p) == {"b0.win": 8, "b0.wv": 3}
    p.write_text(json.dumps({"b0.win": 9}))
    assert pcmm_cf_table.load(p) == {"b0.win": 9}


@pytest.mark.parametrize(
    ("bad", "err", "msg"),
    [
        ({"b0.win": 0}, ValueError, "integer >= 1"),
        ({"b0.win": 7.5}, ValueError, "integer >= 1"),
        ({"b0.win": True}, ValueError, "integer >= 1"),
        ({"b0.win": "8"}, ValueError, "integer >= 1"),
        ({}, ValueError, "empty"),
        ([8], TypeError, "JSON object"),
    ],
)
def test_load_refuses_what_is_not_a_cf(tmp_path, bad, err, msg):
    p = tmp_path / "t.json"
    p.write_text(json.dumps(bad))
    with pytest.raises(err, match=msg):
        pcmm_cf_table.load(p)


def test_magnitudes_are_the_largest_output_over_its_shift(tmp_path):
    nodes = [
        finish("b0.wk", 0, 0.95),
        finish("b0.wk", 1, 1.40),
        finish("b0.win", 0, 6.25),
        finish("b1.wv", 0, 0.8, shift=2),
        {
            "op_type": "deliberate_bootstrap",
            "inputs": ["v_1"],
            "output": "v_2",
            "output_max_abs": 99.0,
            "args": {},
        },
        {"op_type": "rotate", "inputs": ["v_1"], "output": "v_3", "output_max_abs": 50.0},
    ]
    mags = pcmm_cf_table.finish_magnitudes(capture(tmp_path, [nodes[:3], nodes[3:]]))
    assert mags == {"b0.wk": 1.40, "b0.win": 6.25, "b1.wv": 0.2}


def test_the_calibration_point_and_the_clamp():
    """The default rule puts the one-block capture's largest finish (b0.win, 6.25) at the best single CF, 8, and
    moves one CF per factor 2 of magnitude."""
    assert pcmm_cf_table.cf_of(6.25) == 8
    assert pcmm_cf_table.cf_of(6.25 / 2) == 7
    assert pcmm_cf_table.cf_of(1e-9) == pcmm_cf_table.DEFAULT_CF_MIN
    assert pcmm_cf_table.cf_of(1e9) == pcmm_cf_table.DEFAULT_CF_MAX
    assert pcmm_cf_table.cf_of(6.25, cf_max=7) == 7


def test_build_writes_the_map_and_its_rule(tmp_path):
    doc = pcmm_cf_table.build(capture(tmp_path, [[finish("b0.win", 0, 6.25), finish("b0.wv", 0, 0.2)]]))
    assert doc["cf"] == {"b0.win": 8, "b0.wv": 3}
    assert doc["max_abs"] == {"b0.win": 6.25, "b0.wv": 0.2}
    with pytest.raises(ValueError, match="--cf-min 9 > --cf-max 8"):
        pcmm_cf_table.build(tmp_path, cf_min=9, cf_max=8)


def test_the_session_map_keeps_the_declared_linears_and_the_rest_fall_back():
    logs: list[str] = []
    run = types.SimpleNamespace(pcmm_cf_map={"b0.win": 8, "b5.wk": 7}, pcmm_cf=6, log=logs.append)
    got = PlannedSsmRun.applied_cf_map(run, ["b0.win", "b0.wk"])  # type: ignore[arg-type]
    assert got == {"b0.win": 8}
    assert run.cf_map_report == {
        "applied": {"b0.win": 8},
        "fallback": ["b0.wk"],
        "ignored": ["b5.wk"],
        "pcmm_cf": 6,
    }
    assert "1 linears from the table, 1 at --pcmm-cf 6, 1 table entries not declared here" in logs[0]
