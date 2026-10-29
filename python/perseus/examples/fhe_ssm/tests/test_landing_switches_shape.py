# SPDX-License-Identifier: MIT
"""The landing's runtime switches at the model's own shape: the fake under LANDING=high's switches pins
the reference circuit's eager counts that the 24-block capture under the same switches recorded (two
instruments, one count; skipped without ``FHE_SSM_WEIGHTS`` / ``FHE_SSM_SEEDS`` / ``FHE_SSM_CONFIG``), and
a one-block PCMM capture at the model's width under them stamps the switches and refreshes StC-first."""

import json
import os

import numpy as np
import pytest

from perseus.examples.fhe_ssm import run_fake
from perseus.examples.fhe_ssm.synthetic import synthetic
from perseus.examples.fhe_ssm.weights import load_model

pytestmark = pytest.mark.slow_shape

HIGH = {
    "FIDESLIB_BTS_STC_FIRST": "2",
    "FIDESLIB_BTS_STC_FIRST_ROUTES": "dense",
    "FIDESLIB_BTS_STC_FIRST_CF": "13",
    "FIDESLIB_BTS_RAISE_DROP": "512:1",
    "AUTO_BTS_LEVEL": "44",
}
#: the reference circuit under the switches at --levels preset, one tick: the 24-block capture's counts
REFERENCE_HIGH = {
    "bootstraps": 1344,
    "reactive": 1272,
    "hints": 24,
    "by_step": {"tm.norm": 620, "tm.out": 24, "cm.norm": 626, "carry": 48, "tail": 26},
    "out_level": 30,
    "state_level_max": 26,
}


def published():
    paths = [os.environ.get(k) for k in ("FHE_SSM_WEIGHTS", "FHE_SSM_SEEDS", "FHE_SSM_CONFIG")]
    if not all(paths):
        pytest.skip("the published checkpoint: set FHE_SSM_WEIGHTS, FHE_SSM_SEEDS, FHE_SSM_CONFIG")
    return paths


def test_the_fake_under_the_high_switches_counts_what_the_capture_recorded(monkeypatch):
    w, s, c = published()
    for k, v in HIGH.items():
        monkeypatch.setenv(k, v)
    params = load_model(w, s, c, n_layers=24)
    ids = np.random.default_rng(7).integers(0, params.vocab, size=(1, 64))
    rep = run_fake.run(params, ids, logn=16, levels="preset", log=lambda _s: None)
    t = rep["ticks"][0]
    assert t["bootstraps_by_step"] == REFERENCE_HIGH["by_step"]
    for k in ("bootstraps", "reactive", "hints", "out_level", "state_level_max"):
        assert t[k] == REFERENCE_HIGH[k], k
    assert t["relErrRms_vs_mirror"] < 1e-13


def test_a_pcmm_capture_under_the_switches_refreshes_stc_first(tmp_path, monkeypatch):
    for k, v in HIGH.items():
        monkeypatch.setenv(k, v)
    params, *_ = synthetic(L=1, d=1024, dff=4096, seed=13)
    ids = np.random.default_rng(1).integers(0, params.vocab, size=(2, 64))
    graphs = tmp_path / "graphs"
    rep = run_fake.run(
        params,
        ids,
        logn=16,
        levels="preset",
        log=lambda _s: None,
        linear="pcmm",
        pcmm_wbits=24,
        capture=graphs,
    )
    bts = json.loads((graphs / "capture_config.json").read_text())["bts"]
    assert (bts["stc_first"], bts["raise_drop"]) == (2, "512:1")
    assert rep["capture"]["total"]["bootstraps"] == rep["ticks"][1]["bootstraps"]
    nodes = json.loads((graphs / "block_0" / "graph.json").read_text())["nodes"]
    routes = [
        (n["args"] or {}).get("route")
        for n in nodes
        if n["op_type"] in ("auto_bootstrap", "deliberate_bootstrap")
        or (n["op_type"] == "hint" and (n.get("args") or {}).get("fired"))
    ]
    internal = [r for r in routes if r is not None]
    assert internal
    assert all(r["slots"] == 0 and r["stc_first"] for r in internal)
    assert len(routes) - len(internal) == rep["ticks"][1]["pcmm_external"] == 11
