# SPDX-License-Identifier: MIT
"""The Colab helpers and the GPU smoke checks, without a network or a GPU: the download is a
local copy (checked against a good and a corrupted file), the smoke checks run on the fake (on
the real layout's band, and on a band that lacks the layout's rotations)."""

import json
import shutil

import numpy as np
import pytest
from examples.fhe_ssm_from_primitives import colab, filling, gpu_smoke
from examples.fhe_ssm_from_primitives.runtime_fake import make_runtime_fake
from examples.fhe_ssm_from_primitives.synthetic import synthetic

from perseus.impl import fake

D, DFF, N = 32, 128, 1024


def write_checkpoint(dirpath, seed=21):
    _params, w, seeds, cfg = synthetic(L=2, seed=seed)
    dirpath.mkdir(parents=True, exist_ok=True)
    np.savez(dirpath / "pbd430a_weights.npz", **w)
    (dirpath / "pbd430a_seeds.json").write_text(json.dumps(seeds))
    (dirpath / "pbd430a_config.json").write_text(json.dumps(cfg))
    return dirpath


def local_hub(src):
    """hf_hub_download's signature, served from a local directory."""

    def download(repo, filename, local_dir):
        assert repo == colab.HF_REPO
        out = shutil.copy(src / filename, local_dir)
        return str(out)

    return download


def test_fetch_checks_the_weights_hash(tmp_path, monkeypatch):
    src = write_checkpoint(tmp_path / "hub")
    good = {k: (n, colab.sha256(src / n)) for k, (n, _h) in colab.FILES.items()}
    monkeypatch.setattr(colab, "FILES", good)
    paths = colab.fetch_model(tmp_path / "dest", download=local_hub(src))
    for key, (name, want) in good.items():
        assert paths[key] == tmp_path / "dest" / name
        assert colab.sha256(paths[key]) == want  # oracle 2: the copy's own hash
    with open(src / "pbd430a_weights.npz", "r+b") as f:  # one flipped byte
        f.seek(100)
        b = f.read(1)
        f.seek(100)
        f.write(bytes([b[0] ^ 1]))
    with pytest.raises(RuntimeError, match="sha256"):
        colab.fetch_model(tmp_path / "dest2", download=local_hub(src))


def test_fetch_reports_but_accepts_other_seeds(tmp_path, monkeypatch, capsys):
    src = write_checkpoint(tmp_path / "hub")
    files = {k: (n, colab.sha256(src / n)) for k, (n, _h) in colab.FILES.items()}
    files["seeds"] = (files["seeds"][0], "0" * 64)
    monkeypatch.setattr(colab, "FILES", files)
    colab.fetch_model(tmp_path / "dest", download=local_hub(src))
    assert "differs from the recorded" in capsys.readouterr().out


def test_recorded_hashes_are_well_formed():
    for name, h in colab.FILES.values():
        assert name.startswith("pbd430a_")
        assert len(h) == 64
        int(h, 16)


def test_machine_report_has_the_build_relevant_fields():
    m = colab.machine()
    for key in ("python", "numpy", "cpus", "memory_gb", "disk_free_gb", "gpu", "nvcc", "gcc"):
        assert key in m
    assert m["cpus"] >= 1


@pytest.mark.parametrize(("payload", "route"), [("pairs", "split"), ("real", "split")])
def test_fake_check_passes_on_a_consistent_checkpoint(tmp_path, payload, route):
    src = write_checkpoint(tmp_path)
    paths = {k: src / n for k, (n, _h) in colab.FILES.items()}
    out = colab.fake_check(
        paths, ticks=2, payload=payload, route=route, logn=11, log=lambda _s: None
    )
    v = out["verdict"]
    assert v["pass"]
    assert v["conversations"] == (64 if payload == "pairs" else 32)
    assert v["worst_relErrRms_vs_mirror"] < colab.MIRROR_TOL
    assert len(out["report"]["ticks"]) == 2


def smoke_session(params, rot_steps, complex_payload=True):
    _fhe, inf = make_runtime_fake(
        N=N,
        hid=params.d,
        dim=params.d,
        H=1,
        H_real=1,
        E=params.d_ffn,
        E_real=params.d_ffn,
        rot_steps=rot_steps,
        complex_payload=complex_payload,
    )
    return inf


@pytest.mark.parametrize(
    ("payload", "route"), [("pairs", "split"), ("pairs", "lane"), ("real", "split")]
)
def test_smoke_checks_pass_on_the_fake(payload, route):
    params, *_ = synthetic(L=2, seed=22)
    inf = smoke_session(params, filling.rotation_steps(N, D), payload == "pairs")
    res = gpu_smoke.run_checks(
        inf, fake.core, params, payload=payload, route=route, log=lambda _s: None
    )
    assert res["rotation_audit"]["n_missing"] == 0
    for name in ("roundtrip", "linear", "ticks"):
        assert res[name]["ok"], res[name].get("error")
    assert res["roundtrip"]["relErrRms"] == 0.0  # the fake adds no noise
    assert res["linear"]["relErrRms"] < 1e-12
    for tick in res["ticks"]["ticks"]:
        assert tick["relErrRms_vs_mirror"] < 1e-9
        assert tick["relErrRms_vs_exact"] < 1e-4
        assert tick["logits_relErrRms_vs_exact"] < 1e-4
        assert tick["top1_agree_vs_exact"] == 1.0
        assert tick["projected_full_tick_s"] >= tick["seconds"]


def test_smoke_stops_on_a_band_without_the_layouts_rotations():
    params, *_ = synthetic(L=1, seed=23)
    inf = smoke_session(params, {1, 2, 4})
    res = gpu_smoke.run_checks(inf, fake.core, params, log=lambda _s: None)
    assert res["rotation_audit"]["n_missing"] > 0
    assert "roundtrip" not in res


def test_smoke_records_a_failing_check_and_goes_on(monkeypatch):
    params, *_ = synthetic(L=1, seed=24)
    inf = smoke_session(params, filling.rotation_steps(N, D))

    def broken(*_a, **_k):
        raise RuntimeError("no tile today")

    monkeypatch.setattr(filling, "linear_tiled", broken)
    res = gpu_smoke.run_checks(inf, fake.core, params, ticks=1, log=lambda _s: None)
    assert res["roundtrip"]["ok"]
    assert not res["linear"]["ok"]
    assert "no tile today" in res["linear"]["error"]
    assert not res["ticks"]["ok"]  # the model's linears go through the same function
