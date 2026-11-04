# SPDX-License-Identifier: MIT
"""The prefill capture's public text keeps every polynomial / Newton site inside its calibrated interval on the
unclamped poly model (a ciphertext cannot clamp): a site leaving it diverges, the capture goes NaN and the planner
refuses the block 25 minutes later (results/rwkv7-fhe-20261104/new_pod/r01b_pf_fail)."""
import os

import pytest

from perseus.examples.rwkv7 import common, ref
from perseus.examples.rwkv7.run_scan import report, scan
from perseus.examples.rwkv7.tests.conftest import CKPT, COEFFS
from perseus.examples.rwkv7.tests.test_ref import HARNESS, tokenizer

need = pytest.mark.skipif(not (os.path.exists(CKPT) and os.path.exists(COEFFS)
                               and os.path.exists(os.path.join(HARNESS, "rwkv_tokenizer.py"))), reason="no checkpoint")


@need
def test_prefill_capture_text_inside_intervals():
    P = ref.load_params(CKPT)
    ids = tokenizer().encode(common.CALIB_TEXT_LONG)
    first = scan(P, COEFFS, ids, 1, 64)
    print("\n".join(report(first)) or "all inside")
    assert not first, report(first)


@need
def test_capture_guard_stops_at_the_site(tmp_path, monkeypatch):
    """The retired capture text on 8 blocks at T = 32: the capture's SiteGuard stops in pass 0 at block 7's GroupNorm
    Newton, lane 17 (the plaintext scan's position 17), instead of writing a NaN capture."""
    from perseus.examples.rwkv7 import run_capture
    from perseus.examples.rwkv7.model import SiteOutOfInterval
    from perseus.examples.rwkv7.tests.test_capture import HIGH
    for k, v in HIGH.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(common, "CALIB_TEXT_LONG", common.CALIB_TEXT_LONG_V1)
    with pytest.raises(SiteOutOfInterval) as e:
        run_capture.main(["--pth", CKPT, "--coeffs", COEFFS, "--harness", HARNESS, "--blocks", "8",
                          "--prefill-chunk", "32", "--pcmm-shift", "zero", "--out", str(tmp_path / "cap")])
    print(e.value)
    assert "blocks.7.att.gn_rsqrt" in str(e.value) and "pass 0, lane 17" in str(e.value)


PF_TEXT = ("\n\nThe Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after "
           "the engineer Gustave Eiffel, whose company designed and built the tower from 1887 to 1889. Locally nicknamed "
           "the Iron Lady, it was constructed as the centrepiece of the 1889 World's Fair, and to crown the centennial "
           "anniversary of the French Revolution.")
CANDIDATES = {"pf_text": PF_TEXT, "decode_calib": common.CALIB_TEXT, "default": common.DEFAULT_TEXT,
              "genesis_v1": common.CALIB_TEXT_LONG_V1}


@need
@pytest.mark.parametrize("name", sorted(CANDIDATES))
def test_scan_text_info(name):
    """Informational (no assert): the sites a text drives out of interval, with the first offending tokens."""
    tok = tokenizer()
    ids = tok.encode(CANDIDATES[name])
    first = scan(ref.load_params(CKPT), COEFFS, ids, 1, 80)
    print(f"\n{len(ids)} ids; first out-of-interval positions:",
          [(p, repr(tok.decode([ids[1 + p]]))) for p in sorted({v[0] for v in first.values()})][:3])
    print("\n".join(report(first)[:4]) or "all inside")
