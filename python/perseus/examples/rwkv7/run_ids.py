# SPDX-License-Identifier: MIT
"""Print a text's RWKV World token ids, comma-separated (the harness's tokenizer, torch-free).

    python -m perseus.examples.rwkv7.run_ids --harness H [--text T] [--n N]
"""
from __future__ import annotations

import argparse

from . import common


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness", required=True)
    ap.add_argument("--vocab", default=None)
    ap.add_argument("--text", default=common.DEFAULT_TEXT)
    ap.add_argument("--n", type=int, default=0, help="first N ids only (0: all)")
    a = ap.parse_args(argv)
    ids = common.tokenizer(a.harness, a.vocab).encode(a.text)
    print(",".join(str(i) for i in (ids[: a.n] if a.n else ids)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
