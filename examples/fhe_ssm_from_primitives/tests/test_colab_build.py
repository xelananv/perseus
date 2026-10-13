# SPDX-License-Identifier: MIT
"""The Colab build's decisions without a GPU: version parsing on real tool output, the plan for
a fresh runtime and for a ready one (checked as explicit step lists and as invariants every
plan must keep), the refusals, the step runner on harmless commands, and the NCCL prefix."""

import sys

import pytest
from examples.fhe_ssm_from_primitives import colab_build as cb

FRESH = {
    "gpu": "NVIDIA L4, 8.9, 23034 MiB",
    "driver": (580, 82),
    "nvcc": (12, 5),
    "nvcc_home": "/usr/local/cuda-12.5",
    "cuda_new": False,
    "gcc": (11,),
    "gcc12": False,
    "cmake": (3, 31),
    "libarchive": False,
    "python_h": True,
    "python": "3.12",
    "pybind11": False,
    "nccl_wheel": None,
    "distro": "ubuntu2204",
    "root": True,
}
READY = {
    **FRESH,
    "gpu": "NVIDIA A100-SXM4-80GB, 8.0, 81920 MiB",
    "nvcc": (13, 0),
    "nvcc_home": "/opt/cuda-13.0",
    "gcc": (13,),
    "libarchive": True,
    "pybind11": True,
    "nccl_wheel": "/x/nvidia/nccl",
}


@pytest.mark.parametrize(
    ("text", "pattern", "want"),
    [
        ("Cuda compilation tools, release 12.5, V12.5.82\n", r"release (\d+)\.(\d+)", (12, 5)),
        ("11\n", r"^(\d+)", (11,)),
        ("cmake version 3.31.6\n\nCMake suite", r"version (\d+)\.(\d+)", (3, 31)),
        ('NAME="Ubuntu"\nVERSION_ID="22.04"\n', r'^VERSION_ID="(\d+)\.(\d+)"', (22, 4)),
        ("", r"release (\d+)\.(\d+)", None),
        (None, r"release (\d+)\.(\d+)", None),
    ],
)
def test_version_parsing(text, pattern, want):
    assert cb.version(text, pattern) == want


def names(steps):
    return [s.name for s in steps]


def by_name(steps, name):
    return next(s for s in steps if s.name == name)


def test_fresh_runtime_gets_every_install():
    steps = cb.plan(FRESH, 2, "/n")
    assert names(steps) == [
        "nvidia-repo",
        "apt",
        "pip",
        "nccl",
        "submodule",
        "deps",
        "core",
        "import",
    ]
    assert "ubuntu2204" in by_name(steps, "nvidia-repo").cmd[-1]
    apt = by_name(steps, "apt").cmd
    for pkg in (cb.CUDA_PKG, "libarchive-dev", "gcc-12", "g++-12"):
        assert pkg in apt
    assert {cb.PYBIND11_PIN, cb.NCCL_PIN} <= set(by_name(steps, "pip").cmd)
    assert cb.CMAKE_PIN not in by_name(steps, "pip").cmd  # the runtime's cmake is new enough
    assert "/tmp/" not in by_name(steps, "nvidia-repo").cmd[-1]  # the keyring lands in a mktemp dir
    env = by_name(steps, "deps").env
    assert env["CUDA_HOME"] == cb.CUDA_HOME_NEW
    assert (env["CC"], env["CXX"]) == ("gcc-12", "g++-12")


def test_ready_runtime_installs_nothing():
    steps = cb.plan(READY, 8, "/n")
    assert names(steps) == ["nccl", "submodule", "deps", "core", "import"]
    env = by_name(steps, "core").env
    assert env["CUDA_HOME"] == "/opt/cuda-13.0"
    assert env["CPATH"].split(":")[0] == "/opt/cuda-13.0/include/cccl"
    assert (env["CC"], env["CHAIN"], env["JOBS"]) == ("gcc", "n32", "8")


def test_a_cuda_12_runtime_gets_cuda_13():
    # nvcc 12.x crashes on the pinned FIDESlib, so a runtime that has one still gets the 13.0 toolkit
    for nvcc in ((12, 6), (12, 8), (12, 9)):
        steps = cb.plan(
            {**READY, "nvcc": nvcc, "nvcc_home": f"/opt/cuda-{nvcc[0]}.{nvcc[1]}"}, 8, "/n"
        )
        assert names(steps)[:2] == ["nvidia-repo", "apt"]
        assert cb.CUDA_PKG in by_name(steps, "apt").cmd
        assert by_name(steps, "core").env["CUDA_HOME"] == cb.CUDA_HOME_NEW


@pytest.mark.parametrize(
    "probe", [FRESH, READY, {**FRESH, "cuda_new": True}, {**READY, "python_h": False}]
)
def test_every_plan_keeps_the_invariants(probe):
    steps = cb.plan(probe, 4, "/n")
    assert names(steps)[-4:] == ["submodule", "deps", "core", "import"]
    deps, core = by_name(steps, "deps"), by_name(steps, "core")
    assert deps.env["NATIVE_SIZE"] == "32"
    assert core.env["PYTHON"] == sys.executable
    for s in (deps, core):
        assert s.env["NCCL_HOME"] == "/n"
        assert s.env["PATH"].startswith(s.env["CUDA_HOME"] + "/bin:")
    apt = by_name(steps, "apt").cmd if "apt" in names(steps) else []
    assert ("gcc-12" in apt or probe["gcc12"]) == (deps.env["CC"] == "gcc-12")
    assert (f"python{probe['python']}-dev" in apt) == (not probe["python_h"])


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"gpu": None}, "no GPU"),
        ({"gpu": "Tesla T4, 7.5, 15360 MiB"}, "needs 8.0 or newer"),
        ({"gpu": "garbled"}, "compute capability"),
        ({"root": False}, "root"),
        ({"driver": (575, 64)}, "driver 580.65 or newer"),
        ({"distro": None}, "distribution"),
    ],
)
def test_refusals(change, message):
    with pytest.raises(SystemExit, match=message):
        cb.plan({**FRESH, **change}, 2, "/n")


def test_runner_stops_at_the_first_failure_and_shows_its_log(tmp_path):
    out: list[str] = []
    steps = [
        cb.Step("one", [sys.executable, "-c", "print('first')"]),
        cb.Step("two", [sys.executable, "-c", "import sys; print('boom'); sys.exit(3)"]),
        cb.Step("three", [sys.executable, "-c", "print('never')"]),
    ]
    assert not cb.run(steps, tmp_path, echo=out.append)
    assert (tmp_path / "one.log").read_text().strip() == "first"
    assert "boom" in (tmp_path / "two.log").read_text()
    assert not (tmp_path / "three.log").exists()
    assert any("boom" in line for line in out)
    assert cb.run(steps[:1], tmp_path / "ok", echo=out.append)


def test_nccl_prefix_links_the_wheel(tmp_path, monkeypatch):
    wheel = tmp_path / "wheel"
    (wheel / "lib").mkdir(parents=True)
    (wheel / "include").mkdir()
    (wheel / "lib" / "libnccl.so.2").write_text("lib")
    (wheel / "include" / "nccl.h").write_text("hdr")
    monkeypatch.setattr(cb, "_module_dir", lambda _name: str(wheel))
    home = cb.link_nccl(str(tmp_path / "home"))
    assert (home / "lib" / "libnccl.so").read_text() == "lib"
    assert (home / "lib" / "libnccl.so.2").resolve() == (wheel / "lib" / "libnccl.so.2").resolve()
    assert (home / "include" / "nccl.h").read_text() == "hdr"
    cb.link_nccl(str(tmp_path / "home"))  # idempotent
    monkeypatch.setattr(cb, "_module_dir", lambda _name: None)
    with pytest.raises(SystemExit, match="nvidia-nccl-cu12"):
        cb.link_nccl(str(tmp_path / "other"))


def test_env_file_carries_the_build_toolchain(tmp_path):
    steps = cb.plan(FRESH, 2, "/n")
    text = cb.write_env(steps, tmp_path / "env.sh").read_text()
    assert f'export CUDA_HOME="{cb.CUDA_HOME_NEW}"' in text
    assert f'export CPATH="{cb.CUDA_HOME_NEW}/include/cccl' in text
    assert 'export NCCL_HOME="/n"' in text
    assert 'export CC="gcc-12"' in text
    assert "LD_LIBRARY_PATH" in text


@pytest.mark.parametrize(
    ("text", "want"),
    [
        ("Tesla T4, 7.5, 15360 MiB", ("Tesla T4", (7, 5), 15360)),
        ("NVIDIA L4, 8.9, 23034 MiB\n", ("NVIDIA L4", (8, 9), 23034)),
        (
            "NVIDIA H100 80GB HBM3, 9.0, 81559 MiB\nNVIDIA H100 80GB HBM3, 9.0, 81559 MiB",
            ("NVIDIA H100 80GB HBM3", (9, 0), 81559),
        ),
        (
            "NVIDIA RTX PRO 6000 Blackwell, 12.0, 97887 MiB",
            ("NVIDIA RTX PRO 6000 Blackwell", (12, 0), 97887),
        ),
        ("", None),
        (None, None),
        ("Tesla T4, [N/A], 15360 MiB", None),
    ],
)
def test_gpu_parsing(text, want):
    assert cb.parse_gpu(text) == want


def test_memory_note_below_the_readme_figure():
    assert len(cb.notes(FRESH)) == 1  # an L4, 22 GB
    assert "may not fit" in cb.notes(FRESH)[0]
    assert cb.notes(READY) == []  # 80 GB
    assert cb.notes({"gpu": None}) == []
