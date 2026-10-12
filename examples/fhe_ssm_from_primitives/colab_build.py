# SPDX-License-Identifier: MIT
"""Build Perseus's CUDA extension (the n32 chain) on a Google Colab GPU runtime (Ubuntu, root)
and check that it imports.

The machine is probed first; what the repository's build needs and the runtime lacks is
installed: CUDA >= 12.6 from NVIDIA's apt repository, gcc 12, LibArchive's and Python's
headers, CMake >= 3.24 and pybind11, and NCCL from NVIDIA's ``nvidia-nccl-cu12`` wheel linked
into a prefix the way the repository's README describes. Then the repository's own scripts run:
``scripts/install_deps.sh`` (the patched OpenFHE and FIDESlib32bits) and
``scripts/local_build_core.sh`` (``perseus._core``). Each step logs to its own file; a failing
step prints the tail of its log and stops.

    python examples/fhe_ssm_from_primitives/colab_build.py [--logs DIR] [--jobs N] [--dry-run]

``--dry-run`` prints the plan for this machine and runs nothing. ``plan(probe, ...)`` is a pure
function of the probe, so the tests check its decisions without a GPU.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import sysconfig
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CUDA_MIN, GCC_MIN, CMAKE_MIN = (12, 6), 12, (3, 24)
CUDA_PKG = "cuda-toolkit-12-6"
CUDA_HOME_NEW = "/usr/local/cuda-12.6"
# exact versions, the ones a GPU build of this example was made with, so a rebuild is reproducible
CMAKE_PIN, PYBIND11_PIN, NCCL_PIN = "cmake==4.4.4", "pybind11==3.1.0", "nvidia-nccl-cu12==2.32.3"
KEYRING = (
    "https://developer.download.nvidia.com/compute/cuda/repos/{distro}/x86_64/"
    "cuda-keyring_1.1-1_all.deb"
)


def version(text: str | None, pattern: str) -> tuple[int, ...] | None:
    """The first ``pattern`` match in ``text`` as a tuple of ints (None when absent)."""
    m = re.search(pattern, text or "", re.MULTILINE)
    return tuple(int(g) for g in m.groups()) if m else None


def _out(cmd: list[str]) -> str | None:
    """A command's stdout and stderr, or None when it is missing or cannot run."""
    if shutil.which(cmd[0]) is None and not Path(cmd[0]).exists():
        return None
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return (r.stdout or "") + (r.stderr or "")


def _module_dir(name: str) -> str | None:
    try:
        spec = importlib.util.find_spec(name)
    except ModuleNotFoundError:
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    return next(iter(spec.submodule_search_locations))


#: FIDESlib32bits's CKKS/Discard.cu emits the PTX ``discard.global.L2`` instruction, which ptxas
#: accepts only for sm_80 and newer (a Colab T4, 7.5, fails there after ~11 minutes of build)
MIN_CAP = (8, 0)
#: Perseus's README: "a GPU with 48 GB or more of memory" for its encrypted runs
README_GB = 48


def parse_gpu(text: str | None) -> tuple[str, tuple[int, int], int] | None:
    """``(name, compute capability, MiB)`` from nvidia-smi's first CSV line
    (``name, compute_cap, memory.total``), or None when it does not parse."""
    line = (text or "").strip().splitlines()[0] if (text or "").strip() else ""
    parts = [x.strip() for x in line.split(",")]
    cap = version(parts[1], r"^(\d+)\.(\d+)$") if len(parts) >= 3 else None
    mib = version(parts[2], r"^(\d+) MiB$") if cap else None
    if cap is None or mib is None:
        return None
    return parts[0], (cap[0], cap[1]), mib[0]


def notes(p: dict) -> list[str]:
    """What the build will run into on this machine that is not worth refusing for."""
    gpu = parse_gpu(p.get("gpu"))
    if gpu is None or gpu[2] >= README_GB * 1024:
        return []
    return [
        f"{gpu[0]} has {gpu[2] / 1024:.0f} GB; Perseus asks for {README_GB} GB, so the session's keys may not fit"
    ]


def probe() -> dict:
    """What decides the plan: the GPU, the toolchain's versions, the distribution."""
    os_release = Path("/etc/os-release")
    text = os_release.read_text(encoding="utf-8") if os_release.exists() else ""
    distro = version(text, r'^VERSION_ID="(\d+)\.(\d+)"')
    gpu = _out(["nvidia-smi", "--query-gpu=name,compute_cap,memory.total", "--format=csv,noheader"])
    pybind = subprocess.run(
        [sys.executable, "-c", "import pybind11"], capture_output=True, check=False
    )
    return {
        "gpu": gpu.strip() if gpu and gpu.strip() else None,
        "nvcc": version(_out(["nvcc", "--version"]), r"release (\d+)\.(\d+)"),
        "nvcc_home": str(Path(nvcc).resolve().parents[1])
        if (nvcc := shutil.which("nvcc"))
        else None,
        "cuda_new": Path(CUDA_HOME_NEW, "bin", "nvcc").exists(),
        "gcc": version(_out(["gcc", "-dumpversion"]), r"^(\d+)"),
        "gcc12": shutil.which("gcc-12") is not None,
        "cmake": version(_out(["cmake", "--version"]), r"version (\d+)\.(\d+)"),
        "libarchive": Path("/usr/include/archive.h").exists(),
        "python_h": Path(sysconfig.get_paths()["include"], "Python.h").exists(),
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "pybind11": pybind.returncode == 0,
        "nccl_wheel": _module_dir("nvidia.nccl"),
        "distro": f"ubuntu{distro[0]}{distro[1]:02d}" if distro else None,
        "root": hasattr(os, "geteuid") and os.geteuid() == 0,
    }


@dataclass
class Step:
    name: str
    cmd: list[str]
    env: dict[str, str] = field(default_factory=dict)


def require(p: dict) -> None:
    """Refuse a machine the build cannot succeed on: no GPU, a GPU older than FIDESlib32bits
    supports, no root (the toolchain is installed with apt)."""
    if not p.get("gpu"):
        raise SystemExit("no GPU visible: Runtime > Change runtime type > a GPU, then rerun")
    gpu = parse_gpu(p["gpu"])
    if gpu is None:
        raise SystemExit(f"cannot read the GPU's compute capability from {p['gpu']!r}")
    if gpu[1] < MIN_CAP:
        raise SystemExit(
            f"{gpu[0]} is compute capability {gpu[1][0]}.{gpu[1][1]}; FIDESlib32bits needs "
            f"{MIN_CAP[0]}.{MIN_CAP[1]} or newer (its CKKS/Discard.cu uses the PTX discard "
            "instruction): pick an L4, A100 or newer runtime"
        )
    if not p.get("root"):
        raise SystemExit("this script installs system packages: run it as root (Colab does)")


def plan(p: dict, jobs: int, nccl_home: str) -> list[Step]:
    """The build for probe ``p``: only the installs this machine needs, then the repository's
    scripts with the toolchain made explicit. ``nccl_home`` is the prefix ``link_nccl``
    fills."""
    require(p)
    steps: list[Step] = []
    nvcc_ok = p.get("nvcc") is not None and p["nvcc"] >= CUDA_MIN
    need_cuda = not nvcc_ok and not p.get("cuda_new")
    if need_cuda:
        if not p.get("distro"):
            raise SystemExit("unknown distribution: cannot add NVIDIA's apt repository")
        keyring = KEYRING.format(distro=p["distro"])
        steps.append(
            Step(
                "nvidia-repo",
                [
                    "bash",
                    "-c",
                    (
                        f'd=$(mktemp -d) && wget -q {keyring} -O "$d/cuda-keyring.deb" && '
                        'dpkg -i "$d/cuda-keyring.deb" && apt-get update -q && rm -rf "$d"'
                    ),
                ],
            )
        )
    pkgs = [CUDA_PKG] if need_cuda else []
    if not p.get("libarchive"):
        pkgs.append("libarchive-dev")
    if not p.get("python_h"):
        pkgs.append(f"python{p['python']}-dev")
    gcc_ok = p.get("gcc") is not None and p["gcc"][0] >= GCC_MIN
    if not gcc_ok and not p.get("gcc12"):
        pkgs += ["gcc-12", "g++-12"]
    if pkgs:
        steps.append(Step("apt", ["apt-get", "install", "-y", "-q", *pkgs]))
    pip = []
    if p.get("cmake") is None or p["cmake"] < CMAKE_MIN:
        pip.append(CMAKE_PIN)
    if not p.get("pybind11"):
        pip.append(PYBIND11_PIN)
    if not p.get("nccl_wheel"):
        pip.append(NCCL_PIN)
    if pip:
        steps.append(Step("pip", [sys.executable, "-m", "pip", "install", "-q", *pip]))
    steps.append(Step("nccl", [sys.executable, __file__, "--link-nccl", nccl_home]))
    cuda_home = (p.get("nvcc_home") or "/usr/local/cuda") if nvcc_ok else CUDA_HOME_NEW
    cc, cxx = ("gcc", "g++") if gcc_ok else ("gcc-12", "g++-12")
    tool_env = {
        "CUDA_HOME": cuda_home,
        "CC": cc,
        "CXX": cxx,
        "CUDAHOSTCXX": cxx,
        "JOBS": str(jobs),
        "NCCL_HOME": nccl_home,
        "PATH": f"{cuda_home}/bin:{os.environ.get('PATH', '')}",
        "LD_LIBRARY_PATH": f"{nccl_home}/lib:{os.environ.get('LD_LIBRARY_PATH', '')}",
    }
    steps += [
        Step("submodule", ["git", "-C", str(ROOT), "submodule", "update", "--init", "third_party/FIDESlib"]),
        Step("deps", ["bash", str(ROOT / "scripts" / "install_deps.sh")], {**tool_env, "NATIVE_SIZE": "32"}),
        Step(
            "core",
            ["bash", str(ROOT / "scripts" / "local_build_core.sh")],
            {**tool_env, "CHAIN": "n32", "PYTHON": sys.executable},
        ),
        Step(
            "import",
            [sys.executable, "-c", "import perseus._core as c; print(c.chain, c.build_info())"],
            {"PYTHONPATH": str(ROOT), "LD_LIBRARY_PATH": tool_env["LD_LIBRARY_PATH"]},
        ),
    ]  # fmt: skip
    return steps


def link_nccl(dest: str) -> Path:
    """A prefix holding ``lib/libnccl.so``, ``lib/libnccl.so.2`` and ``include/nccl.h``, linked
    from the ``nvidia-nccl-cu12`` wheel (which ships only ``libnccl.so.2``): the repository
    README's route for a machine without a system NCCL."""
    src = _module_dir("nvidia.nccl")
    if src is None:
        raise SystemExit("nvidia-nccl-cu12 is not installed (pip install nvidia-nccl-cu12)")
    lib, inc = Path(src, "lib", "libnccl.so.2"), Path(src, "include", "nccl.h")
    for f in (lib, inc):
        if not f.exists():
            raise SystemExit(f"the NCCL wheel lacks {f}")
    home = Path(dest)
    (home / "lib").mkdir(parents=True, exist_ok=True)
    (home / "include").mkdir(parents=True, exist_ok=True)
    for link, target in (
        (home / "lib" / "libnccl.so", lib),
        (home / "lib" / "libnccl.so.2", lib),
        (home / "include" / "nccl.h", inc),
    ):
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(target)
    return home


def write_env(steps: list[Step], path: Path) -> Path:
    """The toolchain the build used, as a shell file the session's shell sources before
    ``scripts/local_env.sh`` (which would otherwise pick its own CUDA and miss the NCCL prefix)."""
    env = next(s.env for s in steps if s.name == "deps")
    keys = ("CUDA_HOME", "NCCL_HOME", "CC", "CXX")
    lines = [f'export {k}="{env[k]}"' for k in keys]
    lines.append('export LD_LIBRARY_PATH="$NCCL_HOME/lib:${LD_LIBRARY_PATH:-}"')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run(steps: list[Step], logs: Path, *, tail: int = 40, echo=print) -> bool:
    """Each step to ``logs/<name>.log``; on a failure the log's last ``tail`` lines and stop."""
    logs.mkdir(parents=True, exist_ok=True)
    for step in steps:
        log = logs / f"{step.name}.log"
        echo(f"== {step.name}: {' '.join(step.cmd)[:160]}")
        t0 = time.time()
        with open(log, "w", encoding="utf-8") as f:
            rc = subprocess.run(
                step.cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                env={**os.environ, **step.env},
                cwd=ROOT,
                check=False,
            ).returncode
        if rc:
            echo(f"!! {step.name} failed (exit {rc}) after {time.time() - t0:.0f} s; end of {log}:")
            echo("\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-tail:]))
            return False
        echo(f"   ok ({time.time() - t0:.0f} s)")
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--logs", default="/content/perseus_build_logs")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--nccl-home", default=str(ROOT / ".cache" / "nccl"))
    ap.add_argument("--dry-run", action="store_true", help="print the plan, run nothing")
    ap.add_argument("--link-nccl", metavar="DEST", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.link_nccl:
        print(f"NCCL prefix: {link_nccl(a.link_nccl)}")
        return 0
    p = probe()
    print(f"probe: {p}")
    steps = plan(p, a.jobs, a.nccl_home)
    for note in notes(p):
        print(f"note: {note}")
    if a.dry_run:
        for s in steps:
            print(f"{s.name}: {' '.join(s.cmd)}" + (f"  env {s.env}" if s.env else ""))
        return 0
    env_file = write_env(steps, ROOT / ".cache" / "colab_env.sh")
    print(f"toolchain: {env_file}")
    return 0 if run(steps, Path(a.logs)) else 1


if __name__ == "__main__":
    sys.exit(main())
