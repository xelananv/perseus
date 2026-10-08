# SPDX-License-Identifier: MIT
"""Lint, format and type-check this example: fix what the tools can fix, then fail on anything
left (the repository's ruff rules plus ``ruff.toml``'s, and ``mypy.ini``). The tools come from
``$RUFF`` / ``$MYPY``, else from PATH; mypy checks against the interpreter running this script,
so it sees that environment's numpy and perseus.

    python examples/fhe_ssm_from_primitives/lint.py           fix, then check
    python examples/fhe_ssm_from_primitives/lint.py --check   check only (changes nothing)
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def tool(env: str, name: str) -> str:
    path = os.environ.get(env) or shutil.which(name)
    if not path:
        raise SystemExit(f"lint: {name} not found (pip install ruff mypy, or set {env})")
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="check only, change nothing")
    a = ap.parse_args(argv)
    ruff, mypy = tool("RUFF", "ruff"), tool("MYPY", "mypy")
    cfg = ["--config", str(HERE / "ruff.toml")]
    if not a.check:
        subprocess.run([ruff, "check", "--fix", "--quiet", *cfg, str(HERE)], check=False)
        subprocess.run([ruff, "format", "--quiet", *cfg, str(HERE)], check=True)
    checks = {
        "ruff check": [ruff, "check", *cfg, str(HERE)],
        "ruff format": [ruff, "format", "--check", *cfg, str(HERE)],
        "mypy": [
            mypy,
            "--config-file",
            str(HERE / "mypy.ini"),
            "--python-executable",
            sys.executable,
            "-p",
            "examples.fhe_ssm_from_primitives",
        ],
    }
    env = {**os.environ, "MYPYPATH": str(ROOT)}
    failed = [
        name
        for name, cmd in checks.items()
        if subprocess.run(cmd, cwd=ROOT, env=env, check=False).returncode
    ]
    if failed:
        print(f"lint: failed: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
