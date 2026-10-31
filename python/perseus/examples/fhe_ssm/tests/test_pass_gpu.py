# SPDX-License-Identifier: MIT
"""The planned driver under the pass kernels on a device (skipped without the GPU build of ``perseus._pc``; a failure
instead under ``FHE_SSM_REQUIRE_DEVICE=1``): one block at the model's width (d = 1,024, synthetic weights, every linear
PCMM at 24 bits), captured under the pass kernels in the four capture configurations (``passmasks.capture_configurations``:
tick, one sequence fresh, the same carried, mixed) with the LANDING=high runtime switches (``run_gpu.sh``'s ``CAPENV``),
merged under ``max_each`` and planned by the high recipe (a plan cut on the tick configuration alone sizes its refreshes
by the tick's magnitudes, and a failure on ``mixed`` or ``doc`` would then not separate a staging fault from an
under-sized refresh), then run with in-process keys under the ``tick``, ``mixed`` and ``doc`` scenarios. On every timed pass: each stage replayed from its captured graph (``served`` all
``r``); on a pass that restaged pass masks, no graph capture and no invalidation (the in-place re-encode keeps the
graphs: ``crates/pc-py/src/store.rs``, ``crates/pc-program/src/replay.rs``); relErrRms < 1e-3 on every live lane-slot
against ``ref.ExactModel.pass_`` (the port's device tolerance, ``tests/test_planned_gpu.py``); the reply at every
sequence's last position scored; the free device memory equal on every pass (no allocation after setup: run under
``CUDA_MODULE_LOADING=EAGER``, as ``run_gpu.sh pass-tests`` does). At one block top-1 is recorded, not gated (the input
rows are the tokens' own tied embeddings, ``tests/test_diag_gpu.py``)."""

import contextlib
import json
import os
import pathlib
import subprocess

import numpy as np
import pytest

from perseus.examples.fhe_ssm import passmasks, passmerge, run_fake, run_planned
from perseus.examples.fhe_ssm.synthetic import synthetic

pytestmark = pytest.mark.gpu

W = pathlib.Path(__file__).resolve().parents[5]
REL_ERR_RMS = 1e-3
#: run_gpu.sh's CAPENV for a LANDING=high arm: the capture, the plan's stamp and the run's twin share them
HIGH = {
    "FIDESLIB_BTS_STC_FIRST": "2",
    "FIDESLIB_BTS_STC_FIRST_ROUTES": "dense",
    "FIDESLIB_BTS_STC_FIRST_CF": "13",
    "FIDESLIB_BTS_RAISE_DROP": "512:1",
    "AUTO_BTS_LEVEL": "44",
}
SCENARIOS = ("tick:4", "mixed", "doc:130:64")


def device():
    from perseus import _pc  # type: ignore[attr-defined]

    if not hasattr(_pc, "Session"):
        why = "the GPU build of perseus._pc (Session, Program)"
        if os.environ.get("FHE_SSM_REQUIRE_DEVICE") == "1":
            pytest.fail(f"FHE_SSM_REQUIRE_DEVICE=1, and this perseus._pc is not {why}")
        pytest.skip(why)
    return _pc


@contextlib.contextmanager
def high_switches():
    saved = {k: os.environ.get(k) for k in HIGH}
    os.environ.update(HIGH)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@pytest.fixture(scope="module")
def one_block_pass(tmp_path_factory):
    """A 1-block checkpoint at d = 1,024, the four capture configurations' captures under the pass kernels (capture
    tick 1, the high switches) merged under ``max_each``, and the LANDING=high plan of the merge."""
    device()
    d = tmp_path_factory.mktemp("pass_gpu")
    params, w, seeds, cfg = synthetic(L=1, d=1024, dff=4096, seed=13)
    np.savez(d / "w.npz", **w)
    (d / "s.json").write_text(json.dumps(seeds))
    (d / "c.json").write_text(json.dumps(cfg))
    base = ["--weights", str(d / "w.npz"), "--seeds", str(d / "s.json"), "--config", str(d / "c.json"), "--levels", "preset",
            "--linear", "pcmm", "--pcmm-wbits", "24", "--pass-kernels", "on"]  # fmt: skip
    caps = {}
    with high_switches():
        for lab, cfgs in passmasks.capture_configurations(params.vocab).items():
            f = d / f"{lab}.passes.json"
            f.write_text(json.dumps(passmasks.dump_list(cfgs)))
            rc = run_fake.main([*base, "--lane-config", str(f), "--capture", str(d / f"cap_{lab}"), "--capture-tick",
                                "1", "--out", str(d / f"{lab}.json")])  # fmt: skip
            assert rc == 0, lab
            caps[lab] = d / f"cap_{lab}"
    cap = d / "cap"
    labels = list(passmasks.CAPTURE_LABELS)
    passmerge.merge(caps[labels[0]], [caps[x] for x in labels[1:]], cap, labels=labels, log=lambda _s: None,
                    rule="max_each")  # fmt: skip
    e = {**os.environ, "PLAN_ROOT": str(d), "MEMGUARD_GB": "", "CUDA_VISIBLE_DEVICES": "", "LANDING": "high"}
    p = subprocess.run(
        ["bash", str(W / "tools" / "fhe_ssm" / "make_plan.sh"), str(cap), "pass_high"],
        env=e,
        capture_output=True,
        text=True,
        check=False,
    )
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    plan = d / "bootstrap_placements" / "pass_high"
    assert "check errors 0;" in (plan / "summary.txt").read_text()
    return {"base": base, "plan": plan, "capture": cap}


@pytest.mark.parametrize("spec", SCENARIOS)
def test_per_pass_staging_keeps_the_graphs_and_the_replies(one_block_pass, spec, tmp_path, monkeypatch):
    for k, v in HIGH.items():
        monkeypatch.setenv(k, v)
    out = tmp_path / "rec.json"
    rc = run_planned.main([*one_block_pass["base"], "--scenario", spec, "--plan", str(one_block_pass["plan"]),
                           "--capture", str(one_block_pass["capture"]), "--profile", "--profile-every", "2",
                           "--insecure-test-keys", "--out", str(out)])  # fmt: skip
    assert rc == 0
    d = json.loads(out.read_text())
    assert d["prepare"]["compiled_by"] == "device"
    assert d["prepare"]["check_errors"] == {}
    assert d["prepare"]["static_stream"] is True
    assert d["capture"]["checked"] is True
    assert d["keys"]["mode"] == "insecure-inproc"
    # the store holds one slot per distinct (alias, level) pair, all staged at setup
    bud = d["prepare"]["plaintexts"]
    assert d["setup"]["plaintexts_staged"] == bud["by_alias"]["slots"]
    assert d["setup"]["pass_masks"]["configurations"] == d["pass_list"]["configurations"]
    passes = d["ticks"]
    assert len(passes) == d["pass_list"]["passes"]
    for t in passes:
        assert set(t["served"]) == {"r"}, t  # every stage replayed its captured graph after the warm-up
        assert t["counters"]["runs"] == 2  # the block and the tail
        if t["staged"]:
            assert t["counters"]["captures"] == 0, t
            assert t["counters"]["invalidations"] == 0, t
        assert t["relErrRms_vs_exact_worst_lane_slot"]["relErrRms"] < REL_ERR_RMS, t
        assert t["relErrRms_vs_exact"] < REL_ERR_RMS
        assert t["top1"]["of"] == len(config_of(t).next_token_slots())
    # each sequence's last position is a scored lane-slot of its pass
    for t in passes:
        for seg in config_of(t).segments:
            if seg.kind == "seq":
                assert seg.last in t["top1"]["slots"]
    if d["pass_list"]["configurations"] > 1:
        assert any(t["staged"] for t in passes)  # a configuration change restaged in place
    assert len({t["mem_free_bytes"] for t in passes}) == 1  # no device allocation after setup
    assert set(d["warmup"]["served"]) == {"c"}  # the warm-up captured every stage's graph


def config_of(t) -> passmasks.PassConfig:
    """A pass row's configuration (its segments; the tokens play no part here)."""
    n = max(seg["home"][1] for seg in t["segments"])
    return passmasks.PassConfig.from_json({"name": t["config"], "segments": t["segments"], "tokens": [0] * n})
