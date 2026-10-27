# SPDX-License-Identifier: MIT
"""``--capture`` at the synthetic shape: the capture recorder writes one graph per stage at the capture
tick only, the graphs name the carried state at its landing, their op census equals the fake's call
counts for the same circuit (two instruments, one op stream), and the recorder's levels agree with the
fake's (the same Shadow mode and landing)."""

import json

import numpy as np
import pytest

from perseus.examples.fhe_ssm import run_fake
from perseus.examples.fhe_ssm.synthetic import synthetic


def checkpoint(tmp_path):
    _params, w, seeds, cfg = synthetic(L=2, seed=21)
    np.savez(tmp_path / "w.npz", **w)
    (tmp_path / "s.json").write_text(json.dumps(seeds))
    (tmp_path / "c.json").write_text(json.dumps(cfg))
    return ["--weights", str(tmp_path / "w.npz"), "--seeds", str(tmp_path / "s.json"),
            "--config", str(tmp_path / "c.json"), "--tokens", "3", "--logn", "11"]  # fmt: skip


def test_capture_writes_every_stage_at_the_capture_tick_and_counts_like_the_fake(tmp_path):
    base = checkpoint(tmp_path)
    graphs = tmp_path / "graphs"
    assert (
        run_fake.main(
            [*base, "--capture", str(graphs), "--capture-tick", "1", "--out", str(tmp_path / "cap.json")]
        )
        == 0
    )
    assert run_fake.main([*base, "--out", str(tmp_path / "fake.json")]) == 0
    cap = json.loads((tmp_path / "cap.json").read_text())
    fk = json.loads((tmp_path / "fake.json").read_text())
    assert sorted(cap["capture"]["blocks"]) == [f"block_{b}" for b in range(3)]
    assert (graphs / "capture_env.json").exists()
    assert (graphs / "capture_config.json").exists()
    total = cap["capture"]["total"]
    tick = fk["ticks"][1]
    for k in ("mult_cc", "mult_pt", "mult_scalar", "rotations", "conjugations"):
        assert total[k] == tick[k], k
    # refreshes: the graphs hold tick 1's (the state's own carry included); the fake counted the same tick
    assert total["bootstraps"] == tick["bootstraps"]
    assert [t["bootstraps"] for t in cap["ticks"]] == [t["bootstraps"] for t in fk["ticks"]]
    assert [t["out_level"] for t in cap["ticks"]] == [t["out_level"] for t in fk["ticks"]]
    for t in cap["ticks"]:
        assert t["relErrRms_vs_mirror"] < 1e-9
    nodes = json.loads((graphs / "block_0" / "graph.json").read_text())["nodes"]
    names = {n["output"] for n in nodes} | {i for n in nodes for i in n["inputs"]}
    assert {"block_0.cache.s-lvl=34", "block_0.cache.n_prev-lvl=34"} <= names
    assert {n["step"].split(".")[0] for n in nodes} >= {"blk0"}
    assert any(n["op_type"] == "deliberate_bootstrap" for n in nodes)  # the carry


def test_capture_needs_a_tick_after_the_capture_tick(tmp_path):
    base = checkpoint(tmp_path)
    with pytest.raises(SystemExit, match="capture-tick"):
        run_fake.main([*base[:-4], "--tokens", "1", "--logn", "11", "--capture", str(tmp_path / "g")])
