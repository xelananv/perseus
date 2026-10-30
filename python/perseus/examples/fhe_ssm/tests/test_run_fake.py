# SPDX-License-Identifier: MIT
"""The fake driver end to end on a synthetic checkpoint written to disk, for each arm. Its report
is checked two ways: the errors it computes against the references, and its counts against an
independent tally (the per-step bootstrap counts must add up to the totals, the totals to their
parts, the recorded hashes to the files)."""

import hashlib
import json

import numpy as np
import pytest

from perseus.examples.fhe_ssm import fakes, run_fake
from perseus.examples.fhe_ssm.fakes import make_runtime_fake
from perseus.examples.fhe_ssm.synthetic import synthetic
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


def base_argv(paths, tmp_path):
    return [
        "--weights", str(paths["weights.npz"]),
        "--seeds", str(paths["seeds.json"]),
        "--config", str(paths["config.json"]),
        "--tokens", "2",
        "--logn", "11",
        "--out", str(tmp_path / "report.json"),
    ]  # fmt: skip


@pytest.mark.parametrize(("layout", "payload", "route", "capacity"), ARMS)
def test_driver_report(tmp_path, layout, payload, route, capacity):
    paths = checkpoint(tmp_path)
    argv = [*base_argv(paths, tmp_path), "--layout", layout, "--payload", payload, "--pair-route", route]
    assert run_fake.main(argv) == 0
    rep = json.loads((tmp_path / "report.json").read_text())
    assert rep["conversations"] == rep["capacity"] == capacity
    assert len(rep["ticks"]) == 2
    assert rep["axes"] == {
        "newton_form": "perseus", "newton_sites": "served", "pack_stats": "off",
        "linear": None if layout == "decode" else "diagonal",
        "diag_hint": None if layout == "decode" else 1, "pcmm_wbits": None, "lane_mix": "off",
        "pass_kernels": "off",
    }  # fmt: skip
    assert (rep["limit"], rep["bts_out"]) == (46, 34)
    for tick in rep["ticks"]:
        assert tick["relErrRms_vs_mirror"] < 1e-9
        assert tick["relErrRms_vs_exact"] < 1e-4
        assert tick["relErrRms_vs_exact_worst_conversation"] >= tick["relErrRms_vs_exact"] * 0.5
        assert sum(tick["bootstraps_by_step"].values()) == tick["bootstraps"]
        assert tick["bootstraps"] == tick["reactive"] + tick["hints"] + tick["carry"] + tick["pcmm_external"]
        assert tick["carry"] == 2 * rep["n_layers"]
        assert tick["pcmm_external"] == 0
        assert (tick["conjugations"] > 0) == (layout == "filled" and payload == "pairs")
    for name, path in (
        ("weights", "weights.npz"),
        ("seeds", "seeds.json"),
        ("config", "config.json"),
    ):
        assert rep[name]["sha256"] == hashlib.sha256(paths[path].read_bytes()).hexdigest()


def test_the_axes_reach_the_report(tmp_path):
    paths = checkpoint(tmp_path)
    argv = [*base_argv(paths, tmp_path), "--newton-form", "depth2", "--newton-sites", "per-site",
            "--pack-stats", "halves", "--lane-mix", "random", "--diag-hint", "1", "--levels", "preset"]  # fmt: skip
    assert run_fake.main(argv) == 0
    rep = json.loads((tmp_path / "report.json").read_text())
    assert rep["axes"] == {
        "newton_form": "depth2", "newton_sites": "per-site", "pack_stats": "halves", "linear": "diagonal",
        "diag_hint": 1, "pcmm_wbits": None, "lane_mix": "random", "pass_kernels": "off",
    }  # fmt: skip
    assert (rep["limit"], rep["bts_out"]) == (44, 32)
    assert rep["norm"]["in_block"] == {"k": 5.5, "frac": 0.3, "min_iters": 9}
    assert rep["norm"]["ln_out"] == {"k": 5.5, "frac": 0.1, "min_iters": 12}
    assert "lane_mix_note" in rep
    assert rep["ticks"][1]["relErrRms_vs_mirror"] < 1e-9  # the mirror runs the same kernel


def test_linears_are_named_whatever_the_residency():
    """A PCMM finish is looked up by its linear's name on the device: every linear is ``b<b>.<kind>``, resident
    or streamed; ``named`` only keys the per-channel constants."""
    from perseus.examples.fhe_ssm import fakes, weights
    from perseus.examples.fhe_ssm.model import FheSsmModel
    from perseus.impl import fake

    params, *_ = synthetic(L=2, seed=17)
    fhe, inf = fakes.make_runtime_fake(
        N=1024, hid=32, dim=32, H=1, H_real=1, E=128, E_real=128, complex_payload=True
    )
    for resident in (False, True):
        model = FheSsmModel(inf, params, core=fake.core, unit=fhe.unit, resident_weights=resident)
        w = model.block_weights(1)
        assert [getattr(w, k).name for k in ("win", "wout", "wk", "wr", "wv")] == [
            f"b1.{k}" for k in ("win", "wout", "wk", "wr", "wv")
        ]
        assert w.key("mix") == ("b1.mix" if resident else None)
    with pytest.raises(ValueError, match="name"):
        model.geo.weight(np.zeros((32, 32)), None, "win")
    bw = weights.encode_block(params.block(0), model.geo, "b0.", named=False)
    assert bw.named is False
    assert bw.wv.name == "b0.wv"


def test_landing_switches_are_stamped_and_warned(tmp_path, monkeypatch, capsys):
    """The fake's Config takes the landing switches from the shell: a report must say which were set (the
    counts move with them) and the run must warn, so a switched run cannot pass for the reference circuit."""
    paths = checkpoint(tmp_path)
    argv = [*base_argv(paths, tmp_path), "--levels", "preset"]
    for k in (*fakes.LANDING_SWITCHES, "AUTO_BTS_LEVEL"):
        monkeypatch.delenv(k, raising=False)
    assert run_fake.main(argv) == 0
    ref = json.loads((tmp_path / "report.json").read_text())
    assert ref["landing_switches"] == {}
    assert isinstance(ref["runtime_config"], dict)
    assert "landing switches" not in capsys.readouterr().err
    high = {
        "FIDESLIB_BTS_STC_FIRST": "2", "FIDESLIB_BTS_STC_FIRST_ROUTES": "dense",
        "FIDESLIB_BTS_STC_FIRST_CF": "13", "FIDESLIB_BTS_RAISE_DROP": "512:1",
    }  # fmt: skip
    for k, v in high.items():
        monkeypatch.setenv(k, v)
    assert run_fake.main(argv) == 0
    rep = json.loads((tmp_path / "report.json").read_text())
    assert rep["landing_switches"] == high
    assert rep["runtime_config"] != ref["runtime_config"]
    assert rep["ticks"][1]["bootstraps"] < ref["ticks"][1]["bootstraps"]  # StC-first hands back more levels
    err = capsys.readouterr().err
    assert "landing switches set in the environment" in err
    assert "FIDESLIB_BTS_STC_FIRST=2" in err


def test_too_many_conversations_are_refused(tmp_path):
    paths = checkpoint(tmp_path)
    argv = [*base_argv(paths, tmp_path)[:-2], "--payload", "real", "--conversations", "33"]
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
