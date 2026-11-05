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
    # the batched capture test (test_batched.py::test_batched_capture_and_plan)
    parser.addoption("--rwkv7-b-k", default=None, help="the batched capture test's K (default 4)")
    parser.addoption("--rwkv7-b-carry", default=None, help="lane | decode (default lane)")
    parser.addoption("--rwkv7-b-hoist", default=None, help="hoisted lane rotations per call (default 0)")
    parser.addoption("--rwkv7-b-dir", default=None, help="keep the batched capture and plan under this dir")
    parser.addoption("--rwkv7-b-planenv", default=None, help="extra plan env, e.g. 'PAIR=1'")
    parser.addoption("--rwkv7-b-blocks", default=None, help="blocks in the batched capture test (default 1)")
    parser.addoption("--rwkv7-coeffs", default=None, help="the coefficients JSON (as RWKV7_COEFFS; P6: the _nc file)")
    parser.addoption("--rwkv7-heavy", action="store_true",
                     help="also the any-size checks that hold 5-9 GB (widths 2,560 / 4,096, the 2,048 fakes, K = 32): "
                     "run them one test per process (RWKV7_HEAVY=1)")


def pytest_configure(config):
    global COEFFS
    c = config.getoption("rwkv7_coeffs", default=None)
    if c:
        os.environ["RWKV7_COEFFS"] = COEFFS = str(c)
    for opt, env in (("rwkv7_plan", "RWKV7_PLAN"), ("rwkv7_pf_t", "RWKV7_PF_T"), ("rwkv7_pf_dir", "RWKV7_PF_DIR"),
                     ("rwkv7_pf_blocks", "RWKV7_PF_BLOCKS"), ("rwkv7_b_k", "RWKV7_B_K"),
                     ("rwkv7_b_carry", "RWKV7_B_CARRY"), ("rwkv7_b_hoist", "RWKV7_B_HOIST"),
                     ("rwkv7_b_dir", "RWKV7_B_DIR"), ("rwkv7_b_planenv", "RWKV7_B_PLANENV"),
                     ("rwkv7_b_blocks", "RWKV7_B_BLOCKS"), ("rwkv7_heavy", "RWKV7_HEAVY")):
        v = config.getoption(opt, default=None)
        if v:
            os.environ[env] = "1" if v is True else str(v)
