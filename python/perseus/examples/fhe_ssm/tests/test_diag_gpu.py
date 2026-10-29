# SPDX-License-Identifier: MIT
"""The device diagonal encoder (``--diag-source generated``) on a GPU, after the crates' GPU tests (a)-(c)
(crates/pc-program/tests/diag_gen_gpu.rs; tools/fhe_ssm/RUNSHEET_GPU.md step 10): (a, b) the session's probe at
the model's width against the port's oracle, ``filling.diag_gather_reference`` over the registered folded matrices
(``LazyDiagonals.W``), and against one ``encode`` per job; (d) one block under ``generated`` against the exact
forward within the port's device tolerance; (e) the model shape, 2 ticks (needs ``FHE_SSM_DATA`` and the stock plan
``FHE_SSM_PLAN``). Skipped without the GPU build of ``perseus._pc`` that has ``Session.register_tile``."""

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
#: the port's device tolerance (tests/test_planned_gpu.py: one block at the model's width, the PCMM arm)
REL_ERR_RMS, TOP1 = 1e-3, 0.95
#: diagonal sums per block at d = 1,024, d_ffn = 4,096: 32 + 32 + 128 + 128 + 32 groups (the compile check's census)
SUMS_PER_BLOCK = 352


def device():
    from perseus import _pc  # type: ignore[attr-defined]

    if not hasattr(getattr(_pc, "Session", None), "register_tile"):
        pytest.skip("the GPU build of perseus._pc with the device diagonal encoder (Session.register_tile)")
    return _pc


@pytest.fixture(scope="module")
def one_block_diag(tmp_path_factory):
    """A 1-block checkpoint at d = 1,024 (diagonal linears: B = 1,024 at t = 32), its values capture and the stock
    plan; the generated runs share both (``--diag-source`` is a driver flag)."""
    device()
    d = tmp_path_factory.mktemp("diag_gpu")
    _params, w, seeds, cfg = synthetic(L=1, d=1024, dff=4096, seed=13)
    np.savez(d / "w.npz", **w)
    (d / "s.json").write_text(json.dumps(seeds))
    (d / "c.json").write_text(json.dumps(cfg))
    base = ["--weights", str(d / "w.npz"), "--seeds", str(d / "s.json"), "--config", str(d / "c.json"), "--levels", "preset",
            "--linear", "diagonal"]  # fmt: skip
    cap = d / "cap"
    rc = run_fake.main(
        [*base, "--tokens", "2", "--capture", str(cap), "--capture-tick", "1", "--out", str(d / "cap.json")]
    )
    assert rc == 0
    e = {**os.environ, "PLAN_ROOT": str(d), "MEMGUARD_GB": "", "CUDA_VISIBLE_DEVICES": ""}
    p = subprocess.run(
        ["bash", str(W / "tools" / "fhe_ssm" / "make_plan.sh"), str(cap), "diag_stock"],
        env=e,
        capture_output=True,
        text=True,
        check=False,
    )
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    return {"base": base, "plan": d / "bootstrap_placements" / "diag_stock"}


def run(base, plan, out, *extra) -> dict:
    rc = run_planned.main([*base, "--plan", str(plan), "--diag-source", "generated", "--insecure-test-keys",
                           "--out", str(out), *extra])  # fmt: skip
    assert rc == 0
    return json.loads(pathlib.Path(out).read_text())


def check_probe(probe: dict, jobs: int) -> None:
    """(a) the gathered slots == the port's oracle vectors and the rows == the eager DeviceFft encode of them; (b) the
    rows of encode_many == one encode per job: every word, at both probed limb counts."""
    assert probe["jobs"] == jobs
    assert len(probe["levels"]) == 2
    for lv in probe["levels"].values():
        assert (lv["slots_vs_oracle"], lv["generated_vs_single"], lv["generated_vs_oracle"]) == (0, 0, 0), lv
        assert lv["words"] == jobs * lv["limbs"] * (1 << 16)


def test_a_b_probe_and_d_one_block_against_the_exact_forward(one_block_diag, tmp_path):
    d = run(
        one_block_diag["base"],
        one_block_diag["plan"],
        tmp_path / "rec.json",
        "--ticks",
        "3",
        "--diag-probe",
        "32",
    )
    st = d["setup"]["diag"]
    assert (st["tiles"], st["uploaded"], st["missing"], st["batch"]) == (5, 5, [], 32)
    assert (
        st["jobs"][0] == st["jobs"][1] == 320 * 32 + 32 * 128
    )  # a job-table row per term of the block's sums
    assert "diag.tiles" in d["setup"]["mem_report"]
    assert "diag.scratch" in d["setup"]["mem_report"]
    check_probe(d["setup"]["diag_probe"], 32)
    assert d["prepare"]["sums"]["diag"] == SUMS_PER_BLOCK
    free = {t["mem_free_bytes"] for t in d["ticks"]}
    assert len(free) == 1, free  # no device allocation after setup: free memory flat across ticks
    for t in d["ticks"]:
        # the generated arm's plaintexts are DeviceFft encodes (llround of a double FFT), the values arm's HostFft
        # (exact i128): they differ by +-1 in rare coefficients, so a generated run is scored against the exact
        # forward within the port's tolerance, never word for word against a values run
        assert t["relErrRms_vs_exact"] < REL_ERR_RMS, t
        assert t["top1_agree_vs_exact"] > TOP1, t
        assert t["diag"]["splits"] == 0
    # sums / chunks count what the host issued (eager runs and graph captures; a replayed graph adds none): whole
    # blocks of 352 sums in 320 + 32 x 4 = 448 chunks (32-term sums one chunk each, 128-term sums four at batch 32)
    issued = [t["diag"] for t in [d["warmup"], *d["ticks"]]]
    n_sums, n_chunks = sum(x["sums"] for x in issued), sum(x["chunks"] for x in issued)
    assert n_sums > 0
    assert n_sums % SUMS_PER_BLOCK == 0
    assert n_chunks == n_sums // SUMS_PER_BLOCK * 448


def test_d_every_chunk_size_within_tolerance(one_block_diag, tmp_path):
    """Chunks of 1 and 128 jobs (``host.diag_batch``; the longest sum has 128 terms) run end to end within the
    tolerance. Word equality across chunk sizes is the crates' test (c) (one session: the keys and the encryption
    noise of two sessions differ)."""
    for b in (1, 128):
        d = run(one_block_diag["base"], one_block_diag["plan"], tmp_path / f"rec_{b}.json", "--ticks", "1", "--diag-batch", str(b))  # fmt: skip
        assert d["setup"]["diag"]["batch"] == b
        for t in d["ticks"]:
            assert t["relErrRms_vs_exact"] < REL_ERR_RMS, (b, t)
            assert t["diag"]["splits"] == 0


@pytest.mark.skipif(
    not (os.environ.get("FHE_SSM_DATA") and os.environ.get("FHE_SSM_PLAN")),
    reason="(e) the model shape: FHE_SSM_DATA (pbd430a weights, seeds, config) and FHE_SSM_PLAN (its stock plan)",
)
def test_e_model_shape_two_ticks(tmp_path):
    """The 24-block model under ``generated`` on the stock plan, 2 ticks, scored as the PCMM arm is (against the exact
    forward) and recorded; the generated arm's DeviceFft plaintexts differ from the values arm's HostFft ones by +-1
    in rare coefficients, so this is a tolerance score, never a word-for-word comparison with a values run."""
    device()
    data = pathlib.Path(os.environ["FHE_SSM_DATA"])
    pick = {k: next(iter(sorted(data.glob(g)))) for k, g in (("weights", "*weights*.npz"), ("seeds", "*seed*.json"),
                                                             ("config", "*config*.json"))}  # fmt: skip
    base = ["--weights", str(pick["weights"]), "--seeds", str(pick["seeds"]), "--config", str(pick["config"]),
            "--levels", "preset", "--linear", "diagonal"]  # fmt: skip
    out = pathlib.Path(os.environ.get("FHE_SSM_E_OUT", str(tmp_path / "model_shape.json")))
    d = run(base, os.environ["FHE_SSM_PLAN"], out, "--ticks", "2", "--profile", "--diag-probe", "32")
    st = d["setup"]["diag"]
    assert (st["tiles"], st["uploaded"], st["missing"], st["splits"]) == (120, 120, [], 0)
    assert d["prepare"]["sums"]["diag"] == 24 * SUMS_PER_BLOCK
    check_probe(d["setup"]["diag_probe"], 32)
    assert len({t["mem_free_bytes"] for t in d["ticks"]}) == 1
    for t in d["ticks"]:
        assert t["diag"]["splits"] == 0
        assert t["top1_agree_vs_exact"] > TOP1, t
