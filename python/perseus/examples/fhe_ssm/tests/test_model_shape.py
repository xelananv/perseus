# SPDX-License-Identifier: MIT
"""Checks at the model's own shape that need the published weights (``FHE_SSM_WEIGHTS``, ``FHE_SSM_SEEDS``,
``FHE_SSM_CONFIG``; skipped without them) or a 1,024-wide block: the reference circuit's counts, pinned, and a
PCMM capture whose finishes name their linears."""

import json
import os

import numpy as np
import pytest

from perseus.examples.fhe_ssm import run_fake
from perseus.examples.fhe_ssm.geometry import KINDS, TRUNK
from perseus.examples.fhe_ssm.synthetic import synthetic
from perseus.examples.fhe_ssm.weights import load_model

pytestmark = pytest.mark.slow_shape

#: the fork's reference circuit on the published weights: one tick, 24 blocks, 64 split pairs, shipped scales
REFERENCE = {
    "bootstraps": 1968,
    "by_step": {
        "tm.norm": 808,
        "cm.norm": 808,
        "tm.gate": 142,
        "cm.gate": 40,
        "cm.v": 88,
        "carry": 48,
        "tail": 34,
    },
    "reactive": 1832,
    "hints": 88,
    "mult_pt": 345265,
    "rotations": 14372,
    "conjugations": 265,
    "mult_cc": 4294,
}


def published():
    paths = [os.environ.get(k) for k in ("FHE_SSM_WEIGHTS", "FHE_SSM_SEEDS", "FHE_SSM_CONFIG")]
    if not all(paths):
        pytest.skip("the published checkpoint: set FHE_SSM_WEIGHTS, FHE_SSM_SEEDS, FHE_SSM_CONFIG")
    return paths


def test_default_settings_pin_the_reference_counts_on_the_published_weights():
    w, s, c = published()
    params = load_model(w, s, c, n_layers=24)
    ids = np.random.default_rng(7).integers(0, params.vocab, size=(1, 64))
    rep = run_fake.run(params, ids, logn=16, log=lambda _s: None)
    t = rep["ticks"][0]
    assert t["bootstraps_by_step"] == REFERENCE["by_step"]
    for k in ("bootstraps", "reactive", "hints", "mult_pt", "rotations", "conjugations", "mult_cc"):
        assert t[k] == REFERENCE[k], k
    assert t["relErrRms_vs_mirror"] < 1e-13
    assert t["relErrRms_vs_exact"] < 1e-13
    assert (t["out_level"], t["state_level_max"]) == (38, 34)


@pytest.mark.parametrize(("linear", "names"), [("pcmm-trunk", TRUNK), ("pcmm", KINDS)])
def test_pcmm_capture_names_every_finish_after_its_linear(tmp_path, linear, names):
    """The capture's PCMM finishes carry the linear's name (``b<b>.<kind>``), which the device service looks the
    weights up by; one block at the model's width with synthetic weights."""
    params, *_ = synthetic(L=1, d=1024, dff=4096, seed=13)
    ids = np.random.default_rng(1).integers(0, params.vocab, size=(2, 64))
    graphs = tmp_path / "graphs"
    rep = run_fake.run(params, ids, logn=16, log=lambda _s: None, linear=linear, capture=graphs)
    assert rep["capture"]["total"]["bootstraps"] == rep["ticks"][1]["bootstraps"]
    nodes = json.loads((graphs / "block_0" / "graph.json").read_text())["nodes"]
    finishes = [
        n["args"]["external"] for n in nodes if isinstance((n.get("args") or {}).get("external"), dict)
    ]
    declared = {f"b0.{k}" for k in names}
    assert {f["linear"] for f in finishes} == declared
    assert all(f["op"] == "pcmm" and f["t"] == 32 for f in finishes)
    outs = {k: 4 if k in ("wk", "wr") else 1 for k in names}
    assert {k: sum(1 for f in finishes if f["linear"] == f"b0.{k}") for k in names} == outs
    assert len(finishes) == rep["ticks"][1]["pcmm_external"]
