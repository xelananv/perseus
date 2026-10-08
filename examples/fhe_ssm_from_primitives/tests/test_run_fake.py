# SPDX-License-Identifier: MIT
"""The fake driver end to end on a synthetic checkpoint written to disk, for each arm. Its report
is checked two ways: the errors it computes against the references, and its counts against an
independent tally (the per-step bootstrap counts must add up to the totals, the totals to their
parts, the recorded hashes to the files)."""

import hashlib
import json

import numpy as np
import pytest
from examples.fhe_ssm_from_primitives import run_fake
from examples.fhe_ssm_from_primitives.runtime_fake import make_runtime_fake
from examples.fhe_ssm_from_primitives.synthetic import synthetic

from perseus.impl.layout import Dims
from perseus.impl.ops import FheOps
from perseus.impl.rt import Rt

ARMS = [
    ("filled", "pairs", "split", 64),
    ("filled", "pairs", "lane", 64),
    ("filled", "real", "split", 32),
    ("decode", "pairs", "split", 1),
]


def checkpoint(tmp_path):
    _params, w, seeds, cfg = synthetic(L=2, seed=17)
    paths = {k: tmp_path / f"pbd_{k}" for k in ("weights.npz", "seeds.json", "config.json")}
    np.savez(paths["weights.npz"], **w)
    paths["seeds.json"].write_text(json.dumps(seeds))
    paths["config.json"].write_text(json.dumps(cfg))
    return paths


@pytest.mark.parametrize(("layout", "payload", "route", "capacity"), ARMS)
def test_driver_report(tmp_path, layout, payload, route, capacity):
    paths = checkpoint(tmp_path)
    out = tmp_path / "report.json"
    argv = [
        "--weights", str(paths["weights.npz"]),
        "--seeds", str(paths["seeds.json"]),
        "--config", str(paths["config.json"]),
        "--tokens", "2",
        "--logn", "11",
        "--layout", layout,
        "--payload", payload,
        "--pair-route", route,
        "--out", str(out),
    ]  # fmt: skip
    assert run_fake.main(argv) == 0
    rep = json.loads(out.read_text())
    assert rep["conversations"] == rep["capacity"] == capacity
    assert len(rep["ticks"]) == 2
    for tick in rep["ticks"]:
        assert tick["relErrRms_vs_mirror"] < 1e-9
        assert tick["relErrRms_vs_exact"] < 1e-4
        assert tick["relErrRms_vs_exact_worst_conversation"] >= tick["relErrRms_vs_exact"] * 0.5
        assert sum(tick["bootstraps_by_step"].values()) == tick["bootstraps"]
        assert tick["bootstraps"] == tick["reactive"] + tick["hints"] + tick["carry"]
        assert tick["carry"] == 2 * rep["n_layers"]
        assert (tick["conjugations"] > 0) == (layout == "filled" and payload == "pairs")
    for name, path in (
        ("weights", "weights.npz"),
        ("seeds", "seeds.json"),
        ("config", "config.json"),
    ):
        assert rep[name]["sha256"] == hashlib.sha256(paths[path].read_bytes()).hexdigest()


def test_too_many_conversations_are_refused(tmp_path):
    paths = checkpoint(tmp_path)
    argv = [
        "--weights", str(paths["weights.npz"]),
        "--seeds", str(paths["seeds.json"]),
        "--config", str(paths["config.json"]),
        "--tokens", "1",
        "--logn", "11",
        "--payload", "real",
        "--conversations", "33",
    ]  # fmt: skip
    with pytest.raises(ValueError, match="capacity"):
        run_fake.main(argv)


def test_bootstrap_counting_restores_the_fake():
    fhe, inf = make_runtime_fake(N=1024, hid=32, dim=32, H=1, H_real=1, E=128, E_real=128)
    rt = Rt(FheOps(inf, run_fake.fake.core, fhe.unit), Dims.from_inf(inf))
    ct = run_fake.fake.FakeCt(np.zeros(1024), 40)
    with run_fake.bootstraps_by_step(fhe, rt.ops) as counts:
        with rt.step("here"):
            rt.ops.bootstrap(ct)
        assert "bootstrap" in vars(fhe)
    assert counts == {"here": 1}
    assert "bootstrap" not in vars(fhe)
    with pytest.raises(RuntimeError), run_fake.bootstraps_by_step(fhe, rt.ops):
        raise RuntimeError
    assert "bootstrap" not in vars(fhe)
