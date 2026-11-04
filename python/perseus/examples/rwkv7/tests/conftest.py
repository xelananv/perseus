# SPDX-License-Identifier: MIT
import os
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[5]          # the checkout
CACHE = ROOT / ".cache" / "rwkv7"
CKPT = os.environ.get("RWKV7_PTH", str(CACHE / "RWKV-x070-World-0.1B-v2.8-20241210-ctx4096.pth"))
COEFFS = os.environ.get("RWKV7_COEFFS", str(CACHE / "coeffs_mixed.json"))


def pytest_addoption(parser):
    # the same switches as the RWKV7_PLAN / RWKV7_PF_T / RWKV7_PF_DIR environment variables
    parser.addoption("--rwkv7-plan", action="store_true", help="also run the planner in the capture tests")
    parser.addoption("--rwkv7-pf-t", default=None, help="the prefill capture test's chunk T")
    parser.addoption("--rwkv7-pf-dir", default=None, help="keep the prefill capture and plan under this dir")
    parser.addoption("--rwkv7-pf-blocks", default=None, help="blocks in the prefill capture test (default 1)")


def pytest_configure(config):
    for opt, env in (("rwkv7_plan", "RWKV7_PLAN"), ("rwkv7_pf_t", "RWKV7_PF_T"), ("rwkv7_pf_dir", "RWKV7_PF_DIR"),
                     ("rwkv7_pf_blocks", "RWKV7_PF_BLOCKS")):
        v = config.getoption(opt, default=None)
        if v:
            os.environ[env] = "1" if v is True else str(v)
