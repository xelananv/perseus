# SPDX-License-Identifier: MIT
"""The planned driver's pieces a CPU can see: the model's hooks (``stage_caches``, ``encode_input``), the recorder
registering the named carried state, ``prepare`` at the synthetic shape against a plan the stock recipe makes of a
synthetic capture (the compile check's verdict, the registers, the static stream, the schedule), the pair
packing of block 0's input on the fake against the twin's own, the key needs the model emits, and the refusal
paths (no keys, lockstep, streams, a plan whose capture md5 differs, a plan cut under other levels, a plan on
another chain)."""

import json
import os
import pathlib
import shutil
import subprocess
import types

import numpy as np
import pytest

from perseus.examples.fhe_ssm import fakes, filling, planned, run_fake, run_planned
from perseus.examples.fhe_ssm.model import STATE_NAMES, FheSsmModel
from perseus.examples.fhe_ssm.synthetic import synthetic
from perseus.impl import fake, fakecap
from perseus.impl.ops import FheOps

LOGN = 10
HERE = pathlib.Path(__file__).resolve()
W = HERE.parents[5]
MAKE_PLAN = W / "tools" / "fhe_ssm" / "make_plan.sh"


def checkpoint(tmp_path, L=2, seed=21):
    _params, w, seeds, cfg = synthetic(L=L, seed=seed)
    np.savez(tmp_path / "w.npz", **w)
    (tmp_path / "s.json").write_text(json.dumps(seeds))
    (tmp_path / "c.json").write_text(json.dumps(cfg))
    return ["--weights", str(tmp_path / "w.npz"), "--seeds", str(tmp_path / "s.json"), "--config", str(tmp_path / "c.json"),
            "--logn", str(LOGN), "--levels", "preset"]  # fmt: skip


def plan_of(capture: pathlib.Path, root: pathlib.Path, name: str, **env) -> pathlib.Path:
    """``make_plan.sh`` on a capture (the stock recipe; the compile check when its binary is built)."""
    if not os.access(planned.default_check_bin(), os.X_OK):
        pytest.skip("tools/fhe_ssm/compile_check is not built (bash tools/fhe_ssm/compile_check/build.sh)")
    e = {**os.environ, "PLAN_ROOT": str(root), "MEMGUARD_GB": "", "CUDA_VISIBLE_DEVICES": "", **env}
    p = subprocess.run(
        ["bash", str(MAKE_PLAN), str(capture), name], env=e, capture_output=True, text=True, check=False
    )
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    return root / "bootstrap_placements" / name


@pytest.fixture(scope="module")
def synthetic_plan(tmp_path_factory):
    """One synthetic capture (2 blocks, ring 2^10, the preset's levels) and its stock plan, shared by the module."""
    d = tmp_path_factory.mktemp("planned")
    base = checkpoint(d)
    cap = d / "cap"
    assert (
        run_fake.main(
            [
                *base,
                "--tokens",
                "3",
                "--capture",
                str(cap),
                "--capture-tick",
                "1",
                "--out",
                str(d / "cap.json"),
            ]
        )
        == 0
    )
    return {"base": base, "capture": cap, "plan": plan_of(cap, d, "syn_stock"), "dir": d}


# ── the model's hooks ──────────────────────────────────────────────────────────────────────────────────────────


def small_model(linear="diagonal", recorder=False):
    """A 2-block model on the fake (the runtime's rules), or on the capture recorder (``recorder``: what the planned
    driver's twin runs on), at the preset's levels."""
    params, *_ = synthetic(L=2, seed=5)
    n = 1 << (LOGN - 1)
    if recorder:
        fhe, inf = fakes.make_capture_session(n, params.d, params.d_ffn, limit=44, bts_out=32)
    else:
        fhe, inf = fakes.make_runtime_fake(N=n, hid=params.d, dim=params.d, H=1, H_real=1, E=params.d_ffn,
                                           E_real=params.d_ffn, limit=44, bts_out=32, complex_payload=True)  # fmt: skip
    core = fakecap.core if recorder else fake.core
    model = FheSsmModel(inf, params, core=core, unit=fhe.unit, linear=linear, resident_weights=False)
    run_fake._rot_band(fhe, model)
    return params, fhe, inf, model


def test_stage_caches_return_the_two_carried_ciphertexts_with_their_keys():
    _params, _fhe, _inf, model = small_model()
    assert model.stage_caches(0) is None  # before start(): no state yet
    model.start()
    for b in range(model.n_layers):
        cts, keys = model.stage_caches(b)
        assert cts == [model.state[b].s, model.state[b].n_prev]
        assert keys == [f"impl.b{b}.{n}" for n in STATE_NAMES] == model.cache_keys(b)
    assert model.stage_caches(model.n_layers) is None  # the tail carries nothing


def test_encode_input_is_the_geometry_packing_and_encode_pairs_matches_it():
    params, fhe, inf, model = small_model()
    t, d = model.geo.base.t, params.d
    rows = np.random.default_rng(3).standard_normal((2 * t, d))
    x = model.encode_input(rows)
    np.testing.assert_allclose(x.vec, filling.pack_pairs(rows, inf.slots), rtol=0, atol=0)
    # the planned driver's packing at the compiled level: the same slots, two fresh encryptions joined by i
    ops = FheOps(inf, fake.core, fhe.unit)
    y = planned.encode_pairs(ops, rows, d, t, 30)
    np.testing.assert_allclose(y.vec, x.vec, rtol=0, atol=0)
    assert ops.level(y) == 30
    # fewer conversations are parked as copies of the first; wrong shapes are refused
    z = planned.encode_pairs(ops, rows[:3], d, t, 30)
    np.testing.assert_allclose(
        filling.unpack_pairs(z.vec, d, 2 * t)[3:], np.repeat(rows[:1], 2 * t - 3, axis=0)
    )
    with pytest.raises(ValueError, match="rows must be"):
        planned.encode_pairs(ops, rows[:, :-1], d, t, 30)


def test_key_needs_are_the_layouts_rotations_the_conjugation_and_pcmm_per_lane_count():
    _params, _fhe, inf, model = small_model()
    needs = planned.key_needs(model)
    n, d = inf.slots, model.params.d
    assert needs["rotation_steps"] == sorted(filling.rotation_steps(n, d))
    assert needs["conjugation"] is True
    assert needs["pcmm_t"] == []
    assert needs["linear"] == "diagonal"
    assert needs["capacity"] == 2 * (n // d)


# ── prepare at the synthetic shape ─────────────────────────────────────────────────────────────────────────────


def test_prepare_only_binds_the_recorded_stream_to_the_plan(synthetic_plan, tmp_path):
    out = tmp_path / "prep.json"
    rc = run_planned.main(
        [*synthetic_plan["base"], "--plan", str(synthetic_plan["plan"]), "--prepare-only", "--out", str(out)]
    )
    assert rc == 0
    d = json.loads(out.read_text())
    p = d["prepare"]
    assert p["compiled_by"] == "cpu-check"
    assert p["check_errors"] == {}
    assert p["static_stream"] is True
    assert p["dynamic_steps"] == {}
    # two carried ciphertexts per block, named as the capture names them, and one output register per stage
    L = 2
    assert p["registers"] == 2 * L + (L + 1)
    for b in range(L):
        names = p["carried"][str(b)]
        assert sorted(names) == sorted(f"block_{b}.cache.{n}-lvl=32" for n in STATE_NAMES)
    assert str(L) not in p["carried"] or p["carried"][str(L)] == []
    # the plan's count is the compile check's count, program by program
    summary = (synthetic_plan["plan"] / "summary.txt").read_text()
    total = int(summary.split("stages, ")[1].split(" refreshes")[0])
    assert p["refreshes_planned"] == total
    assert sum(v["n_bootstraps"] for v in p["programs"].values()) == total
    # the capture the plan was cut from is the one checked
    assert d["capture"]["checked"]
    assert d["capture"]["md5"] == d["capture"]["plan_md5"]
    assert d["plan"]["axes"].startswith("LANDING=stock")
    # the plan's rotation scope covers the programs' steps (what plan_rot_scope checks on the device)
    scope = set(json.loads((synthetic_plan["plan"] / "plan_rot_steps.json").read_text()))
    assert set(p["rotation_steps"]) <= {s % (1 << (LOGN - 1)) for s in scope}
    assert d["keys"]["mode"] is None
    assert d["mode"] == "prepare"


def test_prepare_only_with_the_siblings_schedule(synthetic_plan, tmp_path):
    out = tmp_path / "prep_sib.json"
    rc = run_planned.main(
        [
            *synthetic_plan["base"],
            "--plan",
            str(synthetic_plan["plan"]),
            "--prepare-only",
            "--schedule",
            "siblings",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    d = json.loads(out.read_text())["prepare"]
    assert d["schedule"] == "siblings"
    assert d["check_errors"] == {}
    assert (tmp_path / "prep_sib.work" / "compile_check.txt").read_text().startswith("compile_check")


def test_prepare_only_lean_keeps_digests_and_still_binds(synthetic_plan, tmp_path):
    out = tmp_path / "prep_lean.json"
    rc = run_planned.main(
        [
            *synthetic_plan["base"],
            "--plan",
            str(synthetic_plan["plan"]),
            "--prepare-only",
            "--lean",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    assert json.loads(out.read_text())["prepare"]["static_stream"] is True


def test_recorder_registers_the_named_state_and_prepare_carries_it():
    """Upstream's Recorder alone would not: a ciphertext named by name_ct in the block's generation is 'not new'
    to rec_var (driver.py:485). The subclass registers it, so every block has its two carried inputs."""
    from perseus.impl.driver import Recorder

    params, _fhe, _inf, model = small_model(recorder=True)
    rec = Recorder(model, model.n_layers)
    rec.values = False  # structure only, as the planned driver records
    rows = np.random.default_rng(1).standard_normal((2, 2 * model.geo.base.t, params.d))
    r0 = rec.token(0, rows[0])
    r1 = rec.token(1, rows[1])
    cache = [n for n in r1[0].inputs() if ".cache." in n]
    assert sorted(cache) == sorted(f"block_0.cache.{n}-lvl=32" for n in STATE_NAMES)  # read as inputs ...
    assert not any(n in r1[0].objs for n in cache)  # ... but upstream never registered their objects
    params, _fhe, _inf, model = small_model(recorder=True)
    rec = planned.SsmRecorder(model, model.n_layers)
    rec.values = False
    r0 = rec.token(0, rows[0])
    r1 = rec.token(1, rows[1])
    cache = [n for n in r1[0].inputs() if ".cache." in n]
    assert sorted(cache) == sorted(f"block_0.cache.{n}-lvl=32" for n in STATE_NAMES)
    assert all(n in r1[0].objs and id(r1[0].objs[n]) in r0[0].final for n in cache)
    assert all(r1[0].metas[n] == (32, 2) for n in cache)
    # the end-of-block names of the state are outputs the next tick reads: the same names every tick
    assert {r0[0].final[id(r1[0].objs[n])] for n in cache} == {
        r1[0].final[id(c)] for c in model.stage_caches(0)[0]
    }


# ── refusals ───────────────────────────────────────────────────────────────────────────────────────────────────


def test_lockstep_and_streams_are_refused_before_anything_runs(tmp_path):
    base = checkpoint(tmp_path)
    with pytest.raises(SystemExit, match="lockstep"):
        run_planned.main([*base, "--plan", "x", "--schedule", "lockstep"])
    with pytest.raises(SystemExit, match="k = 1"):
        run_planned.main([*base, "--plan", "x", "--streams", "2"])
    with pytest.raises(ValueError, match="lockstep"):
        planned.PlannedSsmRun(
            None, "x", None, [], None, model=small_model()[3], work_dir=str(tmp_path), schedule="lockstep"
        )


def test_the_reply_is_the_conversations_asked_for(tmp_path):
    """``--conversations N`` runs N of the 64 pairs (the rest parked as copies): ``reply()`` decodes N rows (the
    real axis first, then the imaginary one), so the tick compares N rows with the exact forward's N; outside
    1..capacity the run is refused before anything is recorded."""
    model = small_model()[3]
    for bad in (0, model.capacity + 1):
        with pytest.raises(ValueError, match="capacity"):
            planned.PlannedSsmRun(
                None, "x", None, [], None, model=model, work_dir=str(tmp_path), conversations=bad
            )
    d, t = model.geo.base.d, model.geo.base.t
    rng = np.random.default_rng(3)
    for n in (1, 3, t, t + 1, 2 * t):
        rows = rng.standard_normal((n, d))
        # what the session decrypts: the packed pairs, the parked lanes zero; reply() is exercised on the
        # in-process decrypt path with the session's two calls stubbed
        vec = filling.pack_pairs(rows, d * t)
        run = planned.PlannedSsmRun.__new__(planned.PlannedSsmRun)
        run.d, run.t, run.n_conv, run.client, run.bundle = d, t, n, None, None
        run.ops = types.SimpleNamespace(decrypt_slots_complex=lambda _ct, v=vec: v)  # type: ignore[assignment]
        run.ct = lambda _reg: types.SimpleNamespace(level=30)  # type: ignore[method-assign]
        got, frame, level = run.reply("out")
        assert (frame, level, got.shape) == (None, 30, (n, d))
        np.testing.assert_allclose(got, rows, rtol=0, atol=1e-12)


def test_prepare_records_the_conversations_the_run_decodes(synthetic_plan, tmp_path):
    out = tmp_path / "prep3.json"
    rc = run_planned.main(
        [*synthetic_plan["base"], "--plan", str(synthetic_plan["plan"]), "--prepare-only", "--conversations", "3",
         "--out", str(out)]
    )  # fmt: skip
    assert rc == 0
    d = json.loads(out.read_text())
    assert (d["conversations"], d["prepare"]["conversations"]) == (3, 3)
    assert d["capacity"] > 3  # the parked lanes are copies of the first conversation; reply() drops them


def test_keygen_is_the_client_role_and_takes_no_model(tmp_path):
    from perseus import _pc  # type: ignore[attr-defined]

    # the device build answers a key request for real: the refusal is the CPU build's
    if not hasattr(_pc, "Client"):
        with pytest.raises(SystemExit, match="device build"):
            run_planned.main(["--keygen", str(tmp_path / "k"), "--keyspec-file", str(tmp_path / "ks.json")])
    with pytest.raises(SystemExit, match="--weights is required"):
        run_planned.main(["--plan", "x", "--ticks", "1"])


def test_a_device_run_needs_its_keys(tmp_path):
    base = checkpoint(tmp_path)
    with pytest.raises(SystemExit, match="needs its keys"):
        run_planned.main([*base, "--plan", "x", "--ticks", "1"])
    with pytest.raises(SystemExit, match=r"no keys\.pckb"):
        run_planned.main([*base, "--plan", "x", "--keys", str(tmp_path / "nokeys")])
    os.environ.pop("FHE_SSM_KEYS", None)
    with pytest.raises(SystemExit, match="FHE_SSM_KEYS"):
        run_planned.main([*base, "--plan", "x", "--keys", "bundle"])
    with pytest.raises(SystemExit, match="two key sources"):
        run_planned.main([*base, "--plan", "x", "--keys", str(tmp_path), "--insecure-test-keys"])


def test_a_plan_whose_capture_md5_differs_is_refused(synthetic_plan, tmp_path):
    other = tmp_path / "other_cap"
    shutil.copytree(synthetic_plan["capture"], other)
    g = other / "block_0" / "graph.json"
    doc = json.loads(g.read_text())
    doc["nodes"][0]["step"] = "changed"
    g.write_text(json.dumps(doc))
    with pytest.raises(SystemExit, match="not the same stream"):
        run_planned.main(
            [
                *synthetic_plan["base"],
                "--plan",
                str(synthetic_plan["plan"]),
                "--capture",
                str(other),
                "--prepare-only",
                "--out",
                str(tmp_path / "x.json"),
            ]
        )
    assert (
        run_planned.graphs_md5(synthetic_plan["capture"])
        == run_planned.plan_capture(run_planned.plan_cmd_lines(str(synthetic_plan["plan"])))[1]
    )


def test_a_plan_cut_under_other_levels_is_refused(synthetic_plan, tmp_path):
    base = list(synthetic_plan["base"])
    base[base.index("preset")] = "fork"  # the twin at 46 / 34, the plan at the preset's 44 / 32
    with pytest.raises(SystemExit, match="another Config"):
        run_planned.main(
            [
                *base,
                "--plan",
                str(synthetic_plan["plan"]),
                "--prepare-only",
                "--out",
                str(tmp_path / "x.json"),
            ]
        )


def test_a_plan_on_another_chain_is_refused(synthetic_plan, tmp_path):
    """The chain is a runtime Config field (``CHAIN_PRESET``): a plan whose runtime line names n32a4 would open the
    session on that chain; the driver refuses it before anything runs. The plan as made names none (n32), and an
    explicit n32 is the same chain."""
    cfg = run_planned.session_config(str(synthetic_plan["plan"]))
    assert cfg["chain.preset"] is None
    for chain, ok in (("n32", True), ("n32a4", False)):
        plan = tmp_path / f"plan_{chain}"
        shutil.copytree(synthetic_plan["plan"], plan)
        cmd = plan / "PLAN_CMD.txt"
        lines = [
            x + f" CHAIN_PRESET={chain}" if x.startswith("runtime=") else x
            for x in cmd.read_text().splitlines()
        ]
        cmd.write_text("\n".join(lines) + "\n")
        if ok:
            assert run_planned.session_config(str(plan))["chain.preset"] == "n32"
            continue
        with pytest.raises(SystemExit, match="chain preset 'n32a4'"):
            run_planned.session_config(str(plan))
        with pytest.raises(SystemExit, match="chain preset 'n32a4'"):
            run_planned.main(
                [
                    *synthetic_plan["base"],
                    "--plan",
                    str(plan),
                    "--prepare-only",
                    "--out",
                    str(tmp_path / "x.json"),
                ]
            )
    # a quoted runtime token reads differently to the runtime (whitespace split) and to this driver (shlex)
    plan = tmp_path / "plan_quoted"
    shutil.copytree(synthetic_plan["plan"], plan)
    cmd = plan / "PLAN_CMD.txt"
    cmd.write_text(
        "\n".join(
            x + ' KV_ARENA_GB="0 CHAIN_PRESET=n32a4 "' if x.startswith("runtime=") else x
            for x in cmd.read_text().splitlines()
        )
        + "\n"
    )
    with pytest.raises(SystemExit, match="quote"):
        run_planned.session_config(str(plan))
    # the run object itself refuses a Config on another chain, whoever built it
    from perseus import config as pconfig

    other = pconfig.preset(run_planned.PRESET).with_env({"CHAIN_PRESET": "n32a4"})
    with pytest.raises(ValueError, match="chain preset 'n32a4'"):
        planned.PlannedSsmRun(None, "x", other, [], None, model=small_model()[3], work_dir=str(tmp_path))


def test_keyspec_on_the_cpu_build_emits_the_models_needs(synthetic_plan, tmp_path):
    rc = run_planned.main(
        [
            *synthetic_plan["base"],
            "--plan",
            str(synthetic_plan["plan"]),
            "--keyspec",
            str(tmp_path / "ks.json"),
            "--out",
            str(tmp_path / "r.json"),
        ]
    )
    assert rc == 0
    doc = json.loads((tmp_path / "ks.json").read_text())
    n, d = 1 << (LOGN - 1), 32
    assert doc["model"]["rotation_steps"] == sorted(filling.rotation_steps(n, d))
    assert doc["model"]["conjugation"] is True
    assert doc["model"]["pcmm_t"] == []
    from perseus import _pc  # type: ignore[attr-defined]

    assert (doc["session"] is None) == (not hasattr(_pc, "Session"))


@pytest.mark.slow_shape
def test_pcmm_linears_are_declared_by_the_names_the_finishes_carry(tmp_path):
    """One block at the model's width under PCMM (synthetic weights): the plan of its capture, the recorded stream
    compiled against it, and the linears the driver would declare to the session are exactly the finishes' names
    with their shapes (the device service's checks, run here)."""
    _params, w, seeds, cfg = synthetic(L=1, d=1024, dff=4096, seed=13)
    np.savez(tmp_path / "w.npz", **w)
    (tmp_path / "s.json").write_text(json.dumps(seeds))
    (tmp_path / "c.json").write_text(json.dumps(cfg))
    base = ["--weights", str(tmp_path / "w.npz"), "--seeds", str(tmp_path / "s.json"), "--config", str(tmp_path / "c.json"),
            "--levels", "preset", "--linear", "pcmm", "--pcmm-wbits", "24"]  # fmt: skip
    cap = tmp_path / "cap"
    assert (
        run_fake.main(
            [
                *base,
                "--tokens",
                "2",
                "--capture",
                str(cap),
                "--capture-tick",
                "1",
                "--out",
                str(tmp_path / "cap.json"),
            ]
        )
        == 0
    )
    plan = plan_of(cap, tmp_path, "pcmm_stock")
    out = tmp_path / "prep.json"
    assert run_planned.main([*base, "--plan", str(plan), "--prepare-only", "--out", str(out)]) == 0
    p = json.loads(out.read_text())["prepare"]
    assert p["check_errors"] == {}
    assert p["pcmm"]["declared"] == [f"b0.{k}" for k in ("win", "wout", "wk", "wr", "wv")]
    assert p["pcmm"]["finishes"] == 1 + 1 + 4 + 4 + 1  # one per output block: wk, wr are 1 -> 4
    assert p["plaintexts"]["bytes"] < 40 * 2**30  # the PCMM arm is resident
    assert p["static_stream"] is True


def test_the_model_flags_are_run_fakes():
    """run_gpu.sh repeats a capture's run_fake flags to the driver: every axis flag must parse the same."""
    import argparse

    a1 = run_planned.parser()
    opts = {o for act in a1._actions for o in act.option_strings}
    for flag in ("--weights", "--seeds", "--config", "--blocks", "--logn", "--levels", "--newton-form", "--newton-sites",
                 "--pack-stats", "--linear", "--pcmm-wbits", "--diag-hint", "--lane-mix", "--scales", "--conversations",
                 "--pair-route", "--ms-norm-k", "--newton-frac", "--newton-min-iters", "--limit", "--bts-out", "--unit"):  # fmt: skip
        assert flag in opts, flag
    ns = a1.parse_args(
        [
            "--weights",
            "w",
            "--seeds",
            "s",
            "--config",
            "c",
            "--linear",
            "pcmm",
            "--pcmm-wbits",
            "24",
            "--newton-form",
            "depth2",
        ]
    )
    assert isinstance(ns, argparse.Namespace)
    assert (ns.linear, ns.pcmm_wbits, ns.levels, ns.newton_form) == ("pcmm", 24, "preset", "depth2")
