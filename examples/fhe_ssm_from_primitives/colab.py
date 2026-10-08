# SPDX-License-Identifier: MIT
"""What the Colab notebook (``colab_check.ipynb``) runs, as functions the tests can call too:
fetch the published checkpoint, describe the machine, and check the port on the numpy fake at the
full size of the model.

    from examples.fhe_ssm_from_primitives import colab
    colab.machine()                          # the runtime: CPU, memory, GPU, toolchain
    paths = colab.fetch_model("/content/pbd430a")
    result = colab.fake_check(paths, ticks=2)

The fake check passes when every tick's reply equals the mirror (the same kernels on numpy) to
1e-9 and the trainer's closed-form forward to 1e-4, relative RMS, for every conversation.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HF_REPO = "xelananv/fhe-ssm-pbd430a"
#: file name and sha256: the weights' hash is the one the published run recorded, the seeds' and
#: the config's those of the files the published run read (fhe-ssm repository, ml-eval/artifacts/)
FILES = {
    "weights": (
        "pbd430a_weights.npz",
        "3ec3f1d8b9b71e7eee38055a7621213991c6417a89576f1c02590963293e784b",
    ),
    "seeds": (
        "pbd430a_seeds.json",
        "d63288181655ebbb5370a16be4b59d7bd6745fcf98bf8ca14e1ca15ec51245b7",
    ),
    "config": (
        "pbd430a_config.json",
        "1829d34eeddbbfc4ce023a43d828085c6e71a2a941e7da8aa17790544b770ba7",
    ),
}
#: the pass thresholds of ``fake_check`` (relative RMS)
MIRROR_TOL, EXACT_TOL = 1e-9, 1e-4


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def _run(cmd: list[str]) -> str | None:
    """A tool's first output line, or None when it is missing or fails."""
    if shutil.which(cmd[0]) is None:
        return None
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    text = (out.stdout or out.stderr).strip()
    return text.splitlines()[0] if text else None


def _mem_gb() -> float | None:
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
    except (ValueError, OSError, AttributeError):
        return None


def machine() -> dict:
    """The runtime this notebook landed on: what decides whether the GPU build can work."""
    nvcc = _run(["nvcc", "--version"])
    if nvcc is not None:
        full = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, check=False)
        nvcc = next((ln for ln in full.stdout.splitlines() if "release" in ln), nvcc)
    gpu = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,compute_cap,memory.total,driver_version",
            "--format=csv,noheader",
        ]
    )
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "cpus": os.cpu_count(),
        "memory_gb": _mem_gb(),
        "disk_free_gb": round(shutil.disk_usage(Path.cwd()).free / 2**30, 1),
        "gpu": gpu,
        "nvcc": nvcc,
        "gcc": _run(["gcc", "--version"]),
        "cmake": _run(["cmake", "--version"]),
    }


def fetch_model(dest, *, download=None) -> dict[str, Path]:
    """The checkpoint, Newton seeds and config from the Hugging Face model repository into
    ``dest``. The weights must match their recorded hash (the published run's); a seeds or
    config file that differs from its recorded hash is reported, since the fake check stays
    valid for any consistent set. ``download(repo, filename, local_dir)`` defaults to
    ``huggingface_hub.hf_hub_download``."""
    if download is None:
        from huggingface_hub import hf_hub_download as download
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    paths = {}
    for key, (name, want) in FILES.items():
        path = Path(download(HF_REPO, name, local_dir=str(dest)))
        got = sha256(path)
        if got != want:
            if key == "weights":
                raise RuntimeError(f"{name}: sha256 {got}, expected {want}")
            print(f"note: {name} sha256 {got} differs from the recorded {want}")
        paths[key] = path
    return paths


def fake_check(
    paths, *, ticks=2, blocks=None, payload="pairs", route="split", logn=16, seed=7, log=print
) -> dict:
    """``ticks`` ticks of the model on the numpy fake with every lane a live conversation,
    against both references; returns the driver's report and a verdict."""
    from . import run_fake
    from .weights import load_model

    params = load_model(
        str(paths["weights"]), str(paths["seeds"]), str(paths["config"]), n_layers=blocks
    )
    t = (1 << (logn - 1)) // params.d
    n = 2 * t if payload == "pairs" else t
    ids = np.random.default_rng(seed).integers(0, params.vocab, size=(ticks, n))
    rep = run_fake.run(params, ids, logn=logn, payload=payload, pair_route=route, log=log)
    worst_mirror = max(tk["relErrRms_vs_mirror"] for tk in rep["ticks"])
    worst_exact = max(tk["relErrRms_vs_exact_worst_conversation"] for tk in rep["ticks"])
    verdict = {
        "conversations": n,
        "ticks": ticks,
        "blocks": params.n_layers,
        "worst_relErrRms_vs_mirror": worst_mirror,
        "worst_conversation_relErrRms_vs_exact": worst_exact,
        "pass": worst_mirror < MIRROR_TOL and worst_exact < EXACT_TOL,
        "hashes": {k: sha256(paths[k]) for k in FILES},
    }
    log(json.dumps(verdict))
    return {"report": rep, "verdict": verdict}
