# SPDX-License-Identifier: MIT
"""The decode policy in pass mode (passmasks.resolve_with_policy, run_planned --pass-policy) and the serve record as
a pass list (record_passlist): CPU only, small shapes."""

import numpy as np

from perseus.examples.fhe_ssm import passmasks, record_passlist
from perseus.examples.fhe_ssm.decode import Policy

L, D, V = 64, 4, 12
GEN = passmasks.GENERATED


def tick_cfg(tokens, entry="carry"):
    segs = tuple(passmasks.Segment("conv", (q, q + 1), q, entry, f"r{q}", None) for q in range(L))
    cfg = passmasks.PassConfig(segs, tuple(tokens), "t")
    passmasks.validate(cfg)
    return cfg


def replies(seed):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((L, D)), rng.standard_normal((V, D))


def test_no_pick_is_the_argmax_bit_for_bit():
    prev, head = replies(0)
    cfg = tick_cfg([GEN] * L)
    got = passmasks.resolve_generated(cfg, prev, head)
    assert list(got.tokens) == [int(np.argmax(prev[q] @ head.T)) for q in range(L)]
    fixed = passmasks.resolve_generated(cfg, prev, head, pick=lambda _q, _logits: 3)
    assert set(fixed.tokens) == {3}


def test_greedy_policy_matches_the_argmax_and_counts_no_flips():
    prev, head = replies(1)
    cfg = tick_cfg([5] * (L // 2) + [GEN] * (L // 2))
    hist: dict[str, list[int]] = {}
    got, info = passmasks.resolve_with_policy(cfg, prev, prev, head, Policy(), hist)
    assert list(got.tokens)[L // 2 :] == [int(np.argmax(prev[q] @ head.T)) for q in range(L // 2, L)]
    assert info == {"picks": L // 2, "flips": []}


def test_a_lane_whose_exact_reply_disagrees_is_a_flip():
    prev, head = replies(2)
    want = prev.copy()
    want[7] = -want[7]  # another argmax at lane-slot 7 (the logits change sign)
    cfg = tick_cfg([GEN] * L)
    _, info = passmasks.resolve_with_policy(cfg, prev, want, head, Policy(), {})
    assert 7 in info["flips"]
    assert info["picks"] == L


def test_norepeat_bans_and_uses_the_streams_history():
    prev, head = replies(3)
    top = int(np.argmax(prev[0] @ head.T))
    pol = Policy(policy="norepeat", penalty=1.3, window=64, ngram=4, ban=(top,))
    cfg = tick_cfg([GEN] + [1] * (L - 1))
    got, _ = passmasks.resolve_with_policy(cfg, prev, prev, head, pol, {"r0": [1, 2, 3]})
    assert got.tokens[0] != top  # the banned token is never picked


def test_feed_history_reads_sequences_and_conversations():
    segs = (
        passmasks.Segment("seq", (0, 32), 0, "fresh", "A", None),
        *(passmasks.Segment("conv", (q, q + 1), q, "carry", f"c{q}", None) for q in range(32, L)),
    )
    cfg = passmasks.PassConfig(segs, tuple(range(L)), "m")
    passmasks.validate(cfg)
    hist: dict[str, list[int]] = {"A": [99]}
    passmasks.feed_history(cfg, hist)
    assert hist["A"] == [99, *range(32)]
    assert hist["c40"] == [40]


def test_record_passlist_feeds_prompts_then_generated():
    doc = {
        "ids_lanes": [[q, q + 1, q + 2][: 1 + q % 3] for q in range(L)],
        "generated_lanes": [[0] * 9 for _ in range(L)],
    }
    cfgs = record_passlist.build(doc, passes=5)
    assert len(cfgs) == 5
    assert cfgs[0].segments[0].entry == "fresh"
    assert cfgs[1].segments[0].entry == "carry"
    for q in (0, 1, 2):
        plen = 1 + q % 3
        assert [c.tokens[q] for c in cfgs] == [doc["ids_lanes"][q][p] if p < plen else GEN for p in range(5)]
