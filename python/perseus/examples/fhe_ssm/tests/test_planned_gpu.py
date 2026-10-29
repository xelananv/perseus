# SPDX-License-Identifier: MIT
"""The planned driver on a device (skipped without the GPU build of ``perseus._pc``): one block of the model at the
model's width with synthetic PCMM weights, planned by the stock recipe, run with in-process keys: the reply against
the exact forward, the carried state across ticks, the key request, and a bundle session round trip (the client's
secret only ever in ``_pc.Client``)."""

import json
import os
import pathlib
import subprocess

import numpy as np
import pytest

from perseus.examples.fhe_ssm import run_fake, run_planned
from perseus.examples.fhe_ssm.synthetic import synthetic

pytestmark = pytest.mark.gpu

W = pathlib.Path(__file__).resolve().parents[5]


def device():
    from perseus import _pc  # type: ignore[attr-defined]

    if not hasattr(_pc, "Session"):
        pytest.skip("the GPU build of perseus._pc (Session, Program) is needed")
    return _pc


@pytest.fixture(scope="module")
def one_block_pcmm(tmp_path_factory):
    """A 1-block checkpoint at d = 1,024 (PCMM's shape), its capture under PCMM and the stock plan."""
    device()
    d = tmp_path_factory.mktemp("gpu")
    _params, w, seeds, cfg = synthetic(L=1, d=1024, dff=4096, seed=13)
    np.savez(d / "w.npz", **w)
    (d / "s.json").write_text(json.dumps(seeds))
    (d / "c.json").write_text(json.dumps(cfg))
    base = ["--weights", str(d / "w.npz"), "--seeds", str(d / "s.json"), "--config", str(d / "c.json"), "--levels", "preset",
            "--linear", "pcmm", "--pcmm-wbits", "24"]  # fmt: skip
    cap = d / "cap"
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
                str(d / "cap.json"),
            ]
        )
        == 0
    )
    e = {**os.environ, "PLAN_ROOT": str(d), "MEMGUARD_GB": "", "CUDA_VISIBLE_DEVICES": ""}
    p = subprocess.run(
        ["bash", str(W / "tools" / "fhe_ssm" / "make_plan.sh"), str(cap), "pcmm_stock"],
        env=e,
        capture_output=True,
        text=True,
        check=False,
    )
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    return {"base": base, "plan": d / "bootstrap_placements" / "pcmm_stock", "dir": d}


def test_ticks_match_the_exact_forward_and_the_state_carries(one_block_pcmm, tmp_path):
    out = tmp_path / "rec.json"
    rc = run_planned.main([*one_block_pcmm["base"], "--plan", str(one_block_pcmm["plan"]), "--ticks", "3", "--profile",
                           "--insecure-test-keys", "--out", str(out)])  # fmt: skip
    assert rc == 0
    d = json.loads(out.read_text())
    assert d["keys"]["mode"] == "insecure-inproc"
    assert d["setup"]["has_secret"] is True
    assert d["prepare"]["compiled_by"] == "device"
    assert d["prepare"]["check_errors"] == {}
    assert d["setup"]["pcmm_uploaded"] == ["b0.win", "b0.wout", "b0.wk", "b0.wr", "b0.wv"]
    assert d["warmup"]["served"]
    assert len(d["ticks"]) == 3
    for t in d["ticks"]:
        # 24-bit PCMM rounding: 2.1e-6 on the fake's reply (README); the device adds the bootstrap's noise
        assert t["relErrRms_vs_exact"] < 1e-3, t
        assert t["top1_agree_vs_exact"] > 0.95
        assert t["fenced"]
        assert all("wall_s" in s for s in t["stages"])
        assert t["counters"]["runs"] == 2  # one program per stage: the block and the tail
    # the state carried: tick 2's reply differs from what a zero state would give (the model is recurrent)
    assert d["ticks"][2]["relErrRms_vs_exact"] < 1e-3


def test_keyspec_and_a_bundle_session_round_trip(one_block_pcmm, tmp_path):
    ks = tmp_path / "ks.json"
    assert (
        run_planned.main(
            [
                *one_block_pcmm["base"],
                "--plan",
                str(one_block_pcmm["plan"]),
                "--keyspec",
                str(ks),
                "--out",
                str(tmp_path / "r.json"),
            ]
        )
        == 0
    )
    doc = json.loads(ks.read_text())
    assert doc["session"] is not None
    assert doc["session"]["pcmm_t"] == [32]
    assert doc["session"]["public_key"] is True
    roles = [k["role"] for k in doc["session"]["keys"]]
    assert "conj" in roles
    assert "relin" in roles
    steps = {k["step"] for k in doc["session"]["keys"] if k["role"] == "rot"}
    assert set(doc["model"]["rotation_steps"]) <= {s % (1 << 15) for s in steps}
    keys = tmp_path / "keys"
    assert (
        run_planned.main(
            [
                "--weights",
                "w",
                "--seeds",
                "s",
                "--config",
                "c",
                "--keygen",
                str(keys),
                "--keyspec-file",
                str(ks),
            ]
        )
        == 0
    )
    out = tmp_path / "rec_bundle.json"
    rc = run_planned.main(
        [
            *one_block_pcmm["base"],
            "--plan",
            str(one_block_pcmm["plan"]),
            "--ticks",
            "2",
            "--keys",
            str(keys),
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    d = json.loads(out.read_text())
    assert d["keys"]["mode"] == "bundle"
    assert d["keys"]["client_secret"] is True
    assert d["setup"]["has_secret"] is False
    assert all(t["relErrRms_vs_exact"] < 1e-3 for t in d["ticks"])


def test_diagonal_linears_are_refused_on_the_device_path(one_block_pcmm, tmp_path):
    base = list(one_block_pcmm["base"])
    base[base.index("pcmm")] = "diagonal"
    with pytest.raises(SystemExit, match="distinct plaintexts"):
        run_planned.main(
            [
                *base,
                "--plan",
                str(one_block_pcmm["plan"]),
                "--ticks",
                "1",
                "--insecure-test-keys",
                "--out",
                str(tmp_path / "x.json"),
            ]
        )
