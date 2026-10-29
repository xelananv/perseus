# SPDX-License-Identifier: MIT
"""The landing's runtime switches (``fakes.LANDING_SWITCHES``) reach the session's Config from the
environment: with none set the Config, and so the capture's ``capture_config.json``, is the four-field
one; with LANDING=high's set, the stamp carries them, the recorder and the fake refresh dense in the
StC-first order at its landing (two instruments, one count), and a LANDING=high plan binds on such a
capture and on no other (the CPU compile check, when built)."""

import json
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

from perseus.examples.fhe_ssm import fakes, run_fake
from perseus.examples.fhe_ssm.synthetic import synthetic
from perseus.impl import fake, fakecap
from perseus.impl.layout import Dims

#: tools/fhe_ssm/make_plan.sh LANDING=high's capture switches (its runtime= line has these and more)
HIGH = {
    "FIDESLIB_BTS_STC_FIRST": "2",
    "FIDESLIB_BTS_STC_FIRST_ROUTES": "dense",
    "FIDESLIB_BTS_STC_FIRST_CF": "13",
    "FIDESLIB_BTS_RAISE_DROP": "512:1",
    "AUTO_BTS_LEVEL": "44",
}
#: the four-field Config at --levels preset, as the reference capture stamped it before the switches existed
FOUR_FIELDS = {
    "chain": {"chain": "n32", "composite_degree": 2},
    "bts": {"auto_bts_level": 44},
    "quirks": {"fake_bts_out": 32, "fakecap_landings": "0:32,1:14,256:22,512:24"},
}
W = pathlib.Path(__file__).resolve().parents[5]
COMPILE_CHECK = W / ".cache" / "fhe_ssm" / "compile_check" / "target" / "release" / "compile_check"


def test_no_switch_leaves_the_four_field_config():
    for env in ({}, {"FIDESLIB_BTS_STC_FIRST": "", "PATH": "/bin", "FIDESLIB_BTS_SPARSE_B": "1"}):
        cfg = fakes.runtime_config(44, 32, 2, env)
        assert json.loads(cfg.to_json()) == FOUR_FIELDS
        assert cfg.to_json() == fakes.runtime_config(44, 32, 2, {}).to_json()


def test_the_high_switches_are_typed_into_the_config():
    cfg = fakes.runtime_config(44, 32, 2, HIGH)
    bts = json.loads(cfg.to_json())["bts"]
    assert bts == {
        "auto_bts_level": 44,
        "raise_drop": "512:1",
        "stc_first": 2,
        "stc_first_cf": 13,
        "stc_first_routes": ["dense"],
    }
    assert cfg["bts.stc_first"] == 2
    assert cfg["bts.raise_drop"] == "512:1"
    # one switch alone sets one field
    one = json.loads(fakes.runtime_config(44, 32, 2, {"FIDESLIB_BTS_STC_FIRST": "2"}).to_json())["bts"]
    assert one == {"auto_bts_level": 44, "stc_first": 2}
    # an explicit off is recorded as such
    off = json.loads(fakes.runtime_config(44, 32, 2, {"FIDESLIB_BTS_STC_FIRST": "0"}).to_json())["bts"]
    assert off == {"auto_bts_level": 44, "stc_first": 0}


def test_the_ceiling_is_the_levels_settings():
    assert fakes.runtime_config(44, 32, 2, {"AUTO_BTS_LEVEL": "44"})["bts.auto_bts_level"] == 44
    with pytest.raises(ValueError, match="AUTO_BTS_LEVEL=44"):
        fakes.runtime_config(46, 34, 2, {"AUTO_BTS_LEVEL": "44"})


@pytest.mark.parametrize(
    "env",
    [
        {},
        {"FIDESLIB_BTS_STC_FIRST": "2"},
        {"FIDESLIB_BTS_STC_FIRST": "0", "FIDESLIB_BTS_STC_FIRST_ROUTES": "dense"},
        {"FIDESLIB_BTS_STC_FIRST": "2", "FIDESLIB_BTS_STC_FIRST_ROUTES": "sparse"},
        {"FIDESLIB_BTS_STC_FIRST": "2", "FIDESLIB_BTS_STC_FIRST_ROUTES": "512,dense"},
        {"FIDESLIB_BTS_STC_FIRST": "2", "FIDESLIB_BTS_STC_FIRST_STAGED_ONLY": "1"},
        {"FIDESLIB_BTS_STC_FIRST": "1", "FIDESLIB_BTS_STC_FIRST_ROUTES": "dense"},
    ],
)
def test_the_fake_reads_the_order_as_the_recorder_does(env):
    cfg = fakes.runtime_config(44, 32, 2, env)
    fhe, _inf = fakes.make_runtime_fake(limit=44, bts_out=32, env=env)
    rec, _rinf = fakecap.make_session(
        Dims(1024, 32, 32, 1, 1, 128, 128), set(), limit=44, bts_out=32, unit=2, config=cfg
    )
    assert fhe.stc_first() == rec.stc_first() == fakes.stc_first_configured(cfg)
    assert fhe.bootstrap_output_level() == rec.bootstrap_output_level()
    # a refresh from the ceiling lands where the recorder's does
    a, b = fake.FakeCt(np.zeros(8), 44, 2), fakecap.CapCt(np.zeros(8), 44, 2)
    fhe.bootstrap(a)
    rec.inner_bootstrap(b)
    assert (a.level, a.noise_deg) == (b.level, b.noise_deg)
    assert (a.level, a.noise_deg) == ((26, 2) if fhe.stc_first() else (32, 2))


def checkpoint(tmp_path):
    _params, w, seeds, cfg = synthetic(L=2, seed=21)
    np.savez(tmp_path / "w.npz", **w)
    (tmp_path / "s.json").write_text(json.dumps(seeds))
    (tmp_path / "c.json").write_text(json.dumps(cfg))
    return ["--weights", str(tmp_path / "w.npz"), "--seeds", str(tmp_path / "s.json"),
            "--config", str(tmp_path / "c.json"), "--tokens", "3", "--logn", "11", "--levels", "preset"]  # fmt: skip


def refreshes(graph_dir):
    """Every refresh node of a capture: (op, route slots, stc_first, input (level, deg), output (level, deg)),
    each input's degree read from its producer (a carried input: its named level, degree 2)."""
    out = []
    for p in sorted(pathlib.Path(graph_dir).glob("block_*/graph.json")):
        made: dict[str, tuple[int, int]] = {}
        for n in json.loads(p.read_text())["nodes"]:
            args = n.get("args") or {}
            fired = n["op_type"] == "hint" and args.get("fired")
            if n["op_type"] in ("auto_bootstrap", "deliberate_bootstrap") or fired:
                r = args["route"]
                src = made.get(n["inputs"][0], (n["input_levels"][0], 2))
                out.append(
                    (
                        n["op_type"],
                        r["slots"],
                        r["stc_first"],
                        src,
                        (n["output_level"], n["output_noise_level"]),
                    )
                )
            made[n["output"]] = (n["output_level"], n.get("output_noise_level"))
    return out


def capture(tmp_path, monkeypatch, name, env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    base = checkpoint(tmp_path)
    graphs = tmp_path / name
    assert run_fake.main([*base, "--capture", str(graphs), "--out", str(tmp_path / f"{name}.json")]) == 0
    assert run_fake.main([*base, "--out", str(tmp_path / f"{name}_fake.json")]) == 0
    for k in env:
        monkeypatch.delenv(k)
    rep = json.loads((tmp_path / f"{name}.json").read_text())
    fk = json.loads((tmp_path / f"{name}_fake.json").read_text())
    return graphs, rep, fk


def test_a_capture_carries_the_switches_and_refreshes_in_their_order(tmp_path, monkeypatch):
    stock, rs, fs = capture(tmp_path, monkeypatch, "stock", {})
    high, rh, fh = capture(tmp_path, monkeypatch, "high", HIGH)
    # the stamp: unchanged without the switches, the switches added with them
    assert (stock / "capture_config.json").read_text() == fakes.runtime_config(44, 32, 2, {}).to_json()
    assert json.loads((stock / "capture_config.json").read_text()) == FOUR_FIELDS
    stamp = json.loads((high / "capture_config.json").read_text())
    assert stamp["bts"]["stc_first"] == 2
    assert stamp["bts"]["raise_drop"] == "512:1"
    assert {k: v for k, v in stamp.items() if k != "bts"} == {
        k: v for k, v in FOUR_FIELDS.items() if k != "bts"
    }
    # the refreshes, from the graphs: the route's rule (crates/pc-shadow/src/route.rs BtsConfig::landing in the
    # fakecap mode: StC-first lands 6 below the dense raw landing 32 unless the effective input passes 46)
    for op, slots, stc, (lv, dg), land in refreshes(stock):
        assert (slots, stc, land) == (0, False, (32, 2) if op != "hint" else (34, 1)), op
    rows = refreshes(high)
    assert rows
    for op, slots, stc, (lv, dg), land in rows:
        eff = lv + 2 * (dg - 1)
        want = (26, 2) if eff <= 46 else (32, 2)
        if op == "hint":
            want = (want[0] + 2, 1)  # the hint's landing is realized
        assert (slots, stc, land) == (0, True, want), (op, lv, dg)
    # two instruments, one count: the fake and the recorder under the same switches
    for rep, fk in ((rs, fs), (rh, fh)):
        assert [t["bootstraps"] for t in rep["ticks"]] == [t["bootstraps"] for t in fk["ticks"]]
        assert [t["hints"] for t in rep["ticks"]] == [t["hints"] for t in fk["ticks"]]
        assert [t["out_level"] for t in rep["ticks"]] == [t["out_level"] for t in fk["ticks"]]
        assert rep["capture"]["total"]["bootstraps"] == fk["ticks"][1]["bootstraps"]
        for t in rep["ticks"]:
            assert t["relErrRms_vs_mirror"] < 1e-9
    # the synthetic shape's counts (2026-10-29): StC-first hands back 6 more levels per refresh
    assert [t["bootstraps"] for t in fs["ticks"]] == [208] * 3
    assert [t["bootstraps"] for t in fh["ticks"]] == [134] * 3
    assert [(t["reactive"], t["hints"], t["carry"]) for t in fh["ticks"]] == [(128, 2, 4)] * 3
    assert [t["out_level"] for t in fs["ticks"]] == [36] * 3
    assert [t["out_level"] for t in fh["ticks"]] == [30] * 3
    # the op stream itself does not depend on where the refreshes land
    for k in ("mult_cc", "mult_pt", "mult_scalar", "rotations", "conjugations"):
        assert rs["capture"]["total"][k] == rh["capture"]["total"][k], k


def make_plan(graphs, name, landing, root):
    env = {
        **os.environ,
        "LANDING": landing,
        "PLAN_ROOT": str(root),
        "MEMGUARD_GB": "",
        "PY": sys.executable,
    }
    for k in HIGH:
        env.pop(k, None)
    res = subprocess.run(
        ["bash", str(W / "tools" / "fhe_ssm" / "make_plan.sh"), str(graphs), name],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, res.stdout + res.stderr
    out = root / "bootstrap_placements" / name
    return json.loads((out / "compile_check.json").read_text()), (out / "summary.txt").read_text()


@pytest.mark.skipif(
    not COMPILE_CHECK.exists(), reason="the compile check is not built (tools/fhe_ssm/compile_check)"
)
def test_a_high_plan_binds_on_a_high_capture_only(tmp_path, monkeypatch):
    stock, _rs, _fs = capture(tmp_path, monkeypatch, "stock", {})
    high, _rh, _fh = capture(tmp_path, monkeypatch, "high", HIGH)
    root = tmp_path / "plans"
    ok, _summary = make_plan(high, "high_on_high", "high", root)
    assert sum(len(b["errors"]) for b in ok["blocks"]) == 0
    assert sum(b["sites"] for b in ok["blocks"]) > 0
    assert any(r.startswith("dense_stcf@26") for r in ok["routes"])
    bad, summary = make_plan(stock, "high_on_stock", "high", root)
    assert sum(len(b["errors"]) for b in bad["blocks"]) > 0
    assert "compile check FAILED" in summary
