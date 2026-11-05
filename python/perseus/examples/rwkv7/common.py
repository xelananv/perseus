# SPDX-License-Identifier: MIT
"""What the runners share: arguments, the checkpoint, the coefficients, the tokens, the scores."""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np

from . import ref
from .model import load_coeffs

DEFAULT_TEXT = "\n\nThe Eiffel tower is in the city of Paris, which is the capital of"
#: the calibration / capture text (public; the PCMM band shifts and the capture's magnitudes come from it)
CALIB_TEXT = "\n\nIn the beginning God created the heaven and the earth. And the earth was without form"
#: the prefill capture's text (public, Genesis 1:3-8 KJV; 86 tokens after the "\n\n"): two chunks of up to 32 tokens.
#: Every polynomial / Newton site stays inside its calibrated interval on its first 64 tokens (tests/test_scan.py);
#: a ciphertext cannot clamp, and the capture's magnitudes must come from in-interval values.
CALIB_TEXT_LONG = (
    "\n\nAnd God said, Let there be light: and there was light. And God saw the light, that it was good: and God "
    "divided the light from the darkness. And God called the light Day, and the darkness he called Night. And the "
    "evening and the morning were the first day. And God said, Let there be a firmament in the midst of the waters, "
    "and let it divide the waters from the waters.")
#: the previous capture text (Genesis 1:1-8), retired 2026-11-05: its token 17 after the "\n\n" (the "," of "without
#: form, and void") drives block 7's GroupNorm variance (att.gn_rsqrt) of 2 heads to 15.82, above the interval's
#: 10.46; the Newton inverse square root diverges, block 7 goes NaN in pass 0 and the xf carry takes it into pass 1
#: (results/rwkv7-fhe-20261104/new_pod/r01b_pf_fail). Kept for the guard's regression test.
CALIB_TEXT_LONG_V1 = (
    "\n\nIn the beginning God created the heaven and the earth. And the earth was without form, and void; and "
    "darkness was upon the face of the deep. And the Spirit of God moved upon the face of the waters. And God said, "
    "Let there be light: and there was light. And God saw the light, that it was good: and God divided the light "
    "from the darkness. And God called the light Day, and the darkness he called Night. And the evening and the "
    "morning were the first day. And God said, Let there be a firmament in the midst of the waters, and let it "
    "divide the waters from the waters.")


def add_args(ap, n_tokens=12):
    ap.add_argument("--pth", required=True, help="RWKV-x070 .pth (read without torch)")
    ap.add_argument("--coeffs", required=True, help="coeffs_mixed.json (fit.py)")
    ap.add_argument("--harness", default=None, help="the research harness dir (its rwkv_tokenizer); else --ids")
    ap.add_argument("--vocab", default=None, help="rwkv_vocab_v20230424.txt (default: the tokenizer's)")
    ap.add_argument("--text", default=DEFAULT_TEXT)
    ap.add_argument("--ids", default=None, help="comma-separated token ids instead of --text")
    ap.add_argument("--tokens", type=int, default=n_tokens, help="decode steps (the prompt fed one per step)")
    ap.add_argument("--blocks", type=int, default=None, help="first N blocks only (default: all)")
    ap.add_argument("--wbits", type=int, default=24, choices=(16, 24))
    ap.add_argument("--fuse-l1", action="store_true", help="one 2->7 PCMM for the first linears (default: one each)")
    ap.add_argument("--shifts", default=None, help="PCMM band shifts JSON (run_capture writes it); default none")
    ap.add_argument("--prefill-chunk", type=int, default=0,
                    help="T > 0: the prompt is read in passes of T tokens on token lanes (prefill.py; T <= 32), the "
                    "remainder and any generation by decode steps; the capture then holds a T-token pass")
    ap.add_argument("--batch", type=int, default=0,
                    help="K > 1: K conversations per pass, conversation c on lane c (batched.py; K <= 32); the capture "
                    "then holds a K-conversation pass, the device run reads --prompts")
    ap.add_argument("--batch-carry", default="lane", choices=("lane", "decode"),
                    help="the token-shift carries of a batched pass: one lane-form ciphertext (lane: the lever, no op) "
                    "or one natural-form ciphertext per conversation (decode: decode's layout)")
    ap.add_argument("--batch-hoist", type=int, default=0,
                    help="B > 0: the lane-to-lane-0 rotations of a batched pass hoisted, B per call (keys 1..K-1); "
                    "0: a running rotation by +1")
    ap.add_argument("--prefix", type=int, default=1, help="leading tokens run in plaintext as a public prefix "
                    "(default 1: the text's leading \\n\\n); 0 = every token encrypted from zero state")
    ap.add_argument("--lora", default="auto", choices=("auto", "fold", "split"),
                    help="the a / v LoRA pairs: one folded C x C PCMM (fold), two PCMMs through the rank (split); auto "
                    "folds at width <= 1,024 (one channel shard) and splits above (ANYSIZE_DESIGN.md)")
    ap.add_argument("--lanes", type=int, default=0, choices=(0, 32, 16, 8),
                    help="lanes per ciphertext (t): 0 (default) = one ciphertext per vector (32 up to width 1,024, 16 "
                    "up to 2,048, 8 up to 4,096; layout.auto_lanes); 32 = 1,024 channels per ciphertext, wider models "
                    "as channel shards (K and T up to 32); 16 / 8 = 2,048 / 4,096 channels (batched K and prefill T "
                    "<= t). ANYSIZE_DESIGN.md §3 compares them by planned refreshes")
    ap.add_argument("--no-shard-land", action="store_true",
                    help="width > 1,024: do not refresh the stage's output shards at block end (then the shards enter "
                    "the next block at the compiled exit, checked before the session opens)")


def tokenizer(harness, vocab=None):
    if harness is None:
        return None
    sys.path.insert(0, harness)
    from rwkv_tokenizer import RWKVTokenizer
    return RWKVTokenizer(vocab) if vocab else RWKVTokenizer()


def token_ids(a, text=None):
    """``--ids``, else the tokenizer's ids of ``text`` (default ``--text``). Without a tokenizer (``--harness`` unset)
    ``--ids`` stands in for every text (the synthetic checkpoints: public calibration ids)."""
    if a.ids and (text is None or a.harness is None):
        ids = [int(x) for x in a.ids.split(",")]
    else:
        tok = tokenizer(a.harness, a.vocab)
        if tok is None:
            raise SystemExit("--harness (the tokenizer) or --ids is required")
        ids = tok.encode(text if text is not None else a.text)
    return ids


def load(a):
    P = ref.load_params(a.pth, a.blocks)
    co = load_coeffs(a.coeffs)
    shifts = json.load(open(a.shifts)) if a.shifts else None
    return P, co, shifts


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel_rms(a, b) -> float:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean((a - b) ** 2)) / max(np.sqrt(np.mean(b ** 2)), 1e-300))


class Scorer:
    """Per token: the poly reference (clamp on, as the brief's comparison) and the circuit's mirror, side by side with
    the decrypted hidden state; top-1 and logits' relErrRms against the poly model."""

    def __init__(self, P, a, mirror=None, poly=None):
        self.P = P
        self.poly = poly if poly is not None else ref.RefModel(P, ref.Sites("poly", a.coeffs, clamp=True))
        self.mirror = mirror
        self.sites = self.poly.sites

    def score(self, pos, tok, hidden):
        ex_before = {k: v[1] for k, v in self.sites.out_counts.items()}
        lr, hr = self.poly.step(tok)
        out_now = {k: v[1] - ex_before.get(k, 0) for k, v in self.sites.out_counts.items()
                   if v[1] - ex_before.get(k, 0)}
        lf = self.poly.head(np.asarray(hidden, dtype=np.float64))
        rec = {"pos": pos, "token": int(tok), "top1_fhe": int(np.argmax(lf)), "top1_poly": int(np.argmax(lr)),
               "top1_agree": bool(np.argmax(lf) == np.argmax(lr)), "logits_relErrRms": rel_rms(lf, lr),
               "hidden_relErrRms_vs_poly": rel_rms(hidden, hr), "poly_sites_out_of_range": out_now}
        if self.mirror is not None:
            hm, _ = self.mirror.step(self.P.emb[int(tok)])
            rec["hidden_relErrRms_vs_mirror"] = rel_rms(hidden, hm)
        return rec


def prefix_state(P, coeffs_path, ids, n_prefix):
    """The public prefix (``ids[:n_prefix]``, by default the document separator ``"\\n\\n"``) run in plaintext on the
    poly model (clamp on): (the RefModel, positioned after it; a copy of its state). The encrypted decode starts from
    this state, so position 0 (whose activations leave several sites' calibrated intervals, which a ciphertext
    cannot clamp: GroupNorm's Newton in block 8 diverges there) is never evaluated on ciphertexts."""
    import copy
    m = ref.RefModel(P, ref.Sites("poly", coeffs_path, clamp=True))
    for t in ids[:n_prefix]:
        m.step(t)
    return m, copy.deepcopy(m.state)


def environment() -> dict:
    keys = ("FIDESLIB_BTS_STC_FIRST", "FIDESLIB_BTS_STC_FIRST_ROUTES", "FIDESLIB_BTS_STC_FIRST_CF",
            "FIDESLIB_BTS_RAISE_DROP", "AUTO_BTS_LEVEL", "CUDA_VISIBLE_DEVICES")
    return {k: os.environ.get(k) for k in keys if os.environ.get(k) is not None}
