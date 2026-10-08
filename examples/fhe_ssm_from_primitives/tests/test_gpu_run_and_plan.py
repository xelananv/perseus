# SPDX-License-Identifier: MIT
"""The GPU runner on the fake (its eager and capture modes; planned needs a GPU and is refused
there) and the planner wrapper without a GPU: its recipe against Perseus's own make_plan.sh,
read from that file; its overrides; the contract stamp read back through Perseus's contract
module."""

import json
import re
from pathlib import Path

import numpy as np
import pytest
from examples.fhe_ssm_from_primitives import gpu_run, plan
from examples.fhe_ssm_from_primitives.synthetic import synthetic

from perseus.plan import contract

ROOT = Path(__file__).resolve().parents[3]


def checkpoint(tmp_path):
    _params, w, seeds, cfg = synthetic(L=2, d=32, dff=128, seed=51)
    np.savez(tmp_path / "w.npz", **w)
    (tmp_path / "s.json").write_text(json.dumps(seeds))
    (tmp_path / "c.json").write_text(json.dumps(cfg))
    return ["--weights", str(tmp_path / "w.npz"), "--seeds", str(tmp_path / "s.json"),
            "--config", str(tmp_path / "c.json")]  # fmt: skip


@pytest.fixture
def small_fake(monkeypatch):
    """The fake at the synthetic checkpoint's width: N = 1024 slots for d = 32 (t = 32)."""
    import contextlib

    from examples.fhe_ssm_from_primitives import filling
    from examples.fhe_ssm_from_primitives.runtime_fake import make_runtime_fake

    from perseus.impl import fake

    def session(params, payload):
        _fhe, inf = make_runtime_fake(
            N=1024, hid=params.d, dim=params.d, H=1, H_real=1, E=params.d_ffn,
            E_real=params.d_ffn, rot_steps=filling.rotation_steps(1024, params.d),
            complex_payload=payload == "pairs",
        )  # fmt: skip
        return contextlib.nullcontext(inf), fake.core

    monkeypatch.setattr(gpu_run, "fake_session", session)


@pytest.mark.usefixtures("small_fake")
@pytest.mark.parametrize("mode", ["eager", "capture"])
def test_runner_modes_on_the_fake(tmp_path, mode):
    out = tmp_path / "run.json"
    argv = [*checkpoint(tmp_path), "--fake", "--mode", mode, "--ticks", "2",
            "--graph-dir", str(tmp_path / "graphs"), "--out", str(out)]  # fmt: skip
    assert gpu_run.main(argv) == 0
    rep = json.loads(out.read_text())
    assert rep["mode"] == mode
    assert rep["conversations"] == 64
    assert len(rep["ticks"]) == 2
    for tick in rep["ticks"]:
        assert tick["relErrRms_vs_mirror"] < 1e-9  # oracle 1: the same kernels on numpy
        assert tick["relErrRms_vs_exact"] < 1e-4  # oracle 2: the trainer's closed form
        assert tick["top1_agree_vs_exact"] == 1.0
        assert tick["bootstraps"] > 0
    assert not list(tmp_path.glob("*.tmp"))  # every write was whole
    if mode == "capture":
        assert (tmp_path / "graphs" / "capture_env.json").exists()


@pytest.mark.usefixtures("small_fake")
def test_capture_runs_past_the_capture_tick(tmp_path):
    out = tmp_path / "run.json"
    argv = [*checkpoint(tmp_path), "--fake", "--mode", "capture", "--ticks", "1",
            "--capture-tick", "2", "--graph-dir", str(tmp_path / "g"), "--out", str(out)]  # fmt: skip
    gpu_run.main(argv)
    assert len(json.loads(out.read_text())["ticks"]) == 3


def test_planned_mode_is_refused_on_the_fake(tmp_path):
    with pytest.raises(SystemExit):
        gpu_run.main([*checkpoint(tmp_path), "--fake", "--mode", "planned"])


@pytest.mark.usefixtures("small_fake")
def test_short_token_records_are_refused(tmp_path):
    rec = tmp_path / "tokens.json"
    rec.write_text(json.dumps({"streams": [[1, 2, 3]] * 10}))
    with pytest.raises(SystemExit, match="10 conversations"):
        gpu_run.main(
            [
                *checkpoint(tmp_path),
                "--fake",
                "--tokens",
                str(rec),
                "--out",
                str(tmp_path / "r.json"),
            ]
        )
    rec.write_text(json.dumps({"streams": [[1, 2]] * 64}))
    with pytest.raises(SystemExit, match="fewer than 4 tokens"):
        gpu_run.main(
            [
                *checkpoint(tmp_path),
                "--fake",
                "--tokens",
                str(rec),
                "--out",
                str(tmp_path / "r.json"),
            ]
        )


def make_plan_defaults():
    """plan_blocks()'s ${VAR:-default} / ${VAR-default} pairs, read from Perseus's script."""
    text = (ROOT / "examples" / "gpt2_from_primitives" / "make_plan.sh").read_text()
    body = text[text.index("plan_blocks() {") :]
    body = body[: body.index("\n}\n")]
    found = dict(re.findall(r'(\w+)="\$\{\1:?-([^}]*)\}"', body))
    if 'PLAN_PRUNE="${PLAN_PRUNE-1}"' in text:
        found["PLAN_PRUNE"] = "1"
    return found


def test_recipe_is_perseus_own():
    theirs = make_plan_defaults()
    assert theirs, "make_plan.sh's plan_blocks() no longer parses"
    assert theirs == plan.RECIPE


def test_overrides_unset_and_dense():
    env = plan.plan_env("g", "n", {"PLAN_NO_PRESCALE": "", "PLAN_CF_MAX": "12"})
    assert "PLAN_NO_PRESCALE" not in env
    assert env["PLAN_CF_MAX"] == "12"
    assert (env["GRAPH_DIR"], env["OUT_NAME"]) == ("g", "n")
    dense = plan.plan_env("g", "n", {"PLAN_SPARSE_SLOTS": ""})
    assert dense["PLAN_SPARSE_SLOTS"] == ""  # set and empty: a dense plan
    assert dense["PLAN_DELIBERATE_CLAMP0"] == "1"
    assert plan.plan_env("g", "n", {"PLAN_ACC_CHAIN": ""})["PLAN_ACC_CHAIN"] == ""


@pytest.mark.parametrize("is_dense", [False, True])
def test_stamp_writes_the_capture_contract(tmp_path, is_dense):
    captured = {"env": {"CHAIN": "n32", "SPARSE_AUTO": "2", "SPARSE_BTS_SLOTS": "512,1"}}
    (tmp_path / "capture_env.json").write_text(json.dumps(captured))
    p = tmp_path / "block_0_placement.json"
    p.write_text(json.dumps({"placements": []}))
    assert plan.stamp(tmp_path, [p], is_dense) == 1
    got = contract.read_stamp(str(p))
    want = {
        **captured["env"],
        **({"SPARSE_AUTO": "0", "SPARSE_BTS_SLOTS": "0"} if is_dense else {}),
    }
    assert got["env"] == want


def test_planning_without_a_capture_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="capture first"):
        plan.main(["--graph-dir", str(tmp_path), "--name", "x"])
    with pytest.raises(SystemExit, match=r"capture_env\.json"):
        plan.stamp(tmp_path, [])
