# SPDX-License-Identifier: MIT
"""Thin wrapper: ``python -m perseus.examples.rwkv7.run_scan`` (the plaintext interval pre-flight)."""
import sys

from perseus.examples.rwkv7.run_scan import main

if __name__ == "__main__":
    sys.exit(main())
