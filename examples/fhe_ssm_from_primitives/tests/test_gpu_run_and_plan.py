# SPDX-License-Identifier: MIT
"""The GPU runner on the fake (its eager and capture modes; planned needs a GPU and is refused
there) and the planner wrapper without a GPU: its recipe against Perseus's own make_plan.sh,
read from that file; its overrides; the contract stamp read back through Perseus's contract
module."""

import json
import os
import re
import shlex
import subprocess
import sys
import types
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
def test_runner_modes_on_the_fake(tmp_path, monkeypatch, mode):
    # set through monkeypatch so the runner's own writes are undone after the test
    monkeypatch.setenv("FHE_GRAPH_DIR", "before")
    monkeypatch.setenv("FHE_GRAPH_CAPTURE_TOKEN", "before")
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
        assert os.environ["FHE_GRAPH_DIR"] == str(tmp_path / "graphs")
        assert os.environ["FHE_GRAPH_CAPTURE_TOKEN"] == "1"


@pytest.mark.usefixtures("small_fake")
def test_capture_runs_past_the_capture_tick(tmp_path, monkeypatch):
    monkeypatch.setenv("FHE_GRAPH_DIR", "before")
    monkeypatch.setenv("FHE_GRAPH_CAPTURE_TOKEN", "before")
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
    """Every ``VAR="${VAR:-default}"`` / ``${VAR-default}`` of Perseus's script, whether inside
    ``plan_blocks()`` or exported around it, so a default the script adds anywhere shows up here.
    Left out: the interpreter, a default that is itself a shell variable (a stage's entry
    plumbing), and the dense plan's landing clamp, which ``plan_env`` mirrors by its condition.
    A name with two different defaults in the script is an error."""
    text = (ROOT / "examples" / "gpt2_from_primitives" / "make_plan.sh").read_text()
    found: dict[str, str] = {}
    for name, default in re.findall(r'(\w+)="\$\{\1:?-([^}]*)\}"', text):
        if name in ("PYTHON", "PLAN_DELIBERATE_CLAMP0") or "$" in default:
            continue
        assert found.get(name, default) == default, f"{name}: {found[name]!r} and {default!r}"
        found[name] = default
    assert "PLAN_DELIBERATE_CLAMP0" in text, (
        "the dense clamp left make_plan.sh; drop it from plan_env"
    )
    return found


def test_recipe_is_perseus_own():
    theirs = make_plan_defaults()
    assert theirs, "make_plan.sh no longer parses"
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


def test_plan_cmd_records_the_recipe_and_the_plans_switches(tmp_path):
    env = plan.plan_env("graphs/x", "n", {"PLAN_SPARSE_SLOTS": ""})
    a = tmp_path / "block_0_placement.json"
    b = tmp_path / "block_1_placement.json"
    a.write_text(
        json.dumps({"placements": [], "real_route": ["v_1"], "raise_drop": {"v_2": 3, "v_3": 1}})
    )
    b.write_text(json.dumps({"placements": []}))
    plan.write_plan_cmd(tmp_path, "graphs/x", {**env, "PLAN_HARD_ENV_CAP": "1"}, [a, b])
    lines = (tmp_path / "PLAN_CMD.txt").read_text().splitlines()
    assert lines[0] == "graph=graphs/x"
    recipe = dict(kv.split("=", 1) for kv in shlex.split(lines[1][len("recipe=") :]))
    assert recipe["PLAN_SPARSE_SLOTS"] == ""
    assert recipe["PLAN_HARD_ENV_CAP"] == "1"
    assert "GRAPH_DIR" not in recipe
    assert "OUT_NAME" not in recipe
    assert {k: v for k, v in recipe.items() if k in plan.RECIPE} == {
        k: v for k, v in env.items() if k in plan.RECIPE
    }
    assert plan.plan_runtime(tmp_path) == {
        "FIDESLIB_BTS_REAL": "1",
        "FIDESLIB_BTS_RAISE_DROPS": "1,3",
    }
    # plans without those routes declare nothing, and a directory without the file declares nothing
    plan.write_plan_cmd(tmp_path, "graphs/x", env, [b])
    assert plan.plan_runtime(tmp_path) == {}
    assert plan.plan_runtime(tmp_path / "missing") == {}


def test_planned_runs_export_the_plans_switches_unless_already_set(tmp_path):
    (tmp_path / "PLAN_CMD.txt").write_text(
        "graph=g\nrecipe=\nruntime=FIDESLIB_BTS_REAL=1 FIDESLIB_BTS_RAISE_DROPS=1,3\n"
    )
    environ = {"FIDESLIB_BTS_RAISE_DROPS": "1,3,5"}
    got = plan.export_plan_runtime(tmp_path, environ)
    assert got == {"FIDESLIB_BTS_REAL": "1", "FIDESLIB_BTS_RAISE_DROPS": "1,3"}
    assert environ == {"FIDESLIB_BTS_RAISE_DROPS": "1,3,5", "FIDESLIB_BTS_REAL": "1"}
    # what the run reports beside the plan's request: the values in effect
    assert plan.effective_runtime(got, environ) == {
        "FIDESLIB_BTS_REAL": "1",
        "FIDESLIB_BTS_RAISE_DROPS": "1,3,5",
    }
    assert plan.effective_runtime({"FIDESLIB_BTS_REAL": "1"}, {}) == {"FIDESLIB_BTS_REAL": ""}


def test_the_runner_variables_come_from_the_script():
    names = plan.runner_vars()
    assert {"MAX_LEVEL", "BTS_LEVEL", "FORBID_STEPS", "FIRST_ENTRY_LEVEL", "BTS_OUT_DEG"} <= names
    assert {"PLAN_PRUNE", "PLAN_SHARED_REALIZE"} <= names
    assert not {"GRAPH_DIR", "OUT_NAME", "PYTHON"} & names


def test_an_override_wins_and_the_plan_file_records_what_the_runner_got():
    shell = {
        "PLAN_SHARED_REALIZE": "1",
        "PLAN_KS_REALIZE": "1",
        "PLAN_HARD_ENV_CAP": "1",
        "FORBID_STEPS": "tail",
        "HOME": "/h",
    }
    over = {"PLAN_SHARED_REALIZE": "", "PLAN_KS_REALIZE": "", "PLAN_CF_MAX": "12"}
    full = plan.runner_env("graphs/x", "n", over, shell)
    # VAR= unsets VAR for the runner, whether or not the recipe has it and whatever the shell exports
    assert "PLAN_SHARED_REALIZE" not in full
    assert "PLAN_KS_REALIZE" not in full
    assert full["PLAN_CF_MAX"] == "12"
    assert full["FORBID_STEPS"] == "tail"
    assert full["HOME"] == "/h"
    rec = plan.recipe_of(full)
    assert rec == {k: full[k] for k in rec}  # every value as the runner got it
    assert rec["FORBID_STEPS"] == "tail"
    assert rec["PLAN_HARD_ENV_CAP"] == "1"
    assert "PLAN_SHARED_REALIZE" not in rec
    assert not {"HOME", "PYTHON", "CUDA_VISIBLE_DEVICES", "GRAPH_DIR", "OUT_NAME"} & set(rec)
    assert set(plan.RECIPE) - {"PLAN_SHARED_REALIZE"} <= set(rec)


def test_main_writes_the_runners_environment_into_the_plan_file(tmp_path, monkeypatch):
    graphs = tmp_path / "graphs"
    (graphs / "block_0").mkdir(parents=True)
    (graphs / "block_0" / "graph.json").write_text("{}")
    (graphs / "capture_env.json").write_text(json.dumps({"env": {"CHAIN": "n32"}}))
    seen: dict[str, str] = {}

    def fake_run(cmd, **kw):
        assert cmd == ["bash", str(plan.RUNNER)]
        seen.update(kw["env"])
        out = tmp_path / "bootstrap_placements" / kw["env"]["OUT_NAME"]
        out.mkdir(parents=True)
        plan_file = out / "block_0_placement.json"
        plan_file.write_text(json.dumps({"placements": [], "real_route": ["v_1"]}))

    monkeypatch.setattr(plan, "ROOT", tmp_path)
    monkeypatch.setattr(plan, "subprocess", types.SimpleNamespace(run=fake_run))
    monkeypatch.setenv("PLAN_SHARED_REALIZE", "1")
    monkeypatch.setenv("FORBID_STEPS", "tail")
    plan.main(["--graph-dir", str(graphs), "--name", "p", "--set", "PLAN_SHARED_REALIZE="])
    assert "PLAN_SHARED_REALIZE" not in seen
    assert seen["FORBID_STEPS"] == "tail"
    lines = (tmp_path / "bootstrap_placements" / "p" / "PLAN_CMD.txt").read_text().splitlines()
    recipe = dict(kv.split("=", 1) for kv in shlex.split(lines[1][len("recipe=") :]))
    assert recipe == plan.recipe_of(seen)
    assert lines[2] == "runtime=FIDESLIB_BTS_REAL=1"


def test_collecting_this_example_sets_no_environment():
    """Importing the package and this directory's conftest, which is what collection does, leaves
    LADDER_RADIX_BITS alone, so the repository's own tests in the same run keep their default."""
    code = (
        "import os, runpy; import examples.fhe_ssm_from_primitives; "
        f"runpy.run_path({str(Path(__file__).with_name('conftest.py'))!r}); "
        "print(os.environ.get('LADDER_RADIX_BITS'))"
    )
    env = {k: v for k, v in os.environ.items() if k != "LADDER_RADIX_BITS"}
    r = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, check=True
    )
    assert r.stdout.strip() == "None"


def test_the_capture_environment_does_not_leak_between_tests():
    assert os.environ.get("FHE_GRAPH_DIR") in (None, "before")
