# SPDX-License-Identifier: MIT
import os
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[5]          # the checkout
CACHE = ROOT / ".cache" / "rwkv7"
CKPT = os.environ.get("RWKV7_PTH", str(CACHE / "RWKV-x070-World-0.1B-v2.8-20241210-ctx4096.pth"))
COEFFS = os.environ.get("RWKV7_COEFFS", str(CACHE / "coeffs_mixed.json"))
