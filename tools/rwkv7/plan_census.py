#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Where a plan's refreshes sit (by the capture's step scope), and the block graph's op counts.

    PYTHONPATH=python python tools/rwkv7/plan_census.py CAPTURE_DIR PLAN_DIR [BLOCK]
"""
import json
import sys

from perseus.examples.rwkv7.tests.test_batched import plan_census

if __name__ == "__main__":
    print(json.dumps(plan_census(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 0), indent=1))
