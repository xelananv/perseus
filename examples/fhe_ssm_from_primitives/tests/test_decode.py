# SPDX-License-Identifier: MIT
"""The decode policy (decode.py) against hand-built cases and an independent re-implementation."""

import numpy as np
import pytest
from examples.fhe_ssm_from_primitives.decode import Policy

NOREPEAT = Policy("norepeat", penalty=1.3, window=64, ngram=4, ban=(0,))


def test_greedy_is_the_argmax_whatever_the_history():
    z = np.array([0.1, 3.0, -2.0, 2.9])
    assert Policy("greedy").pick(z, [1, 1, 1, 1]) == 1


def test_a_banned_token_is_never_picked():
    z = np.array([5.0, 1.0, 0.5])
    assert NOREPEAT.pick(z, [2]) == 1


def test_the_penalty_divides_positive_and_multiplies_negative_logits_in_the_window():
    pol = Policy("norepeat", penalty=2.0, window=2, ngram=0)
    z = np.array([3.0, 2.0, -1.0, 1.0])
    m = pol.modified(z, [3, 0, 2])  # the window is the last two fed tokens: 0 and 2
    assert m.tolist() == [1.5, 2.0, -2.0, 1.0]  # token 3 fell out of the window
    assert pol.pick(z, [3, 0, 2]) == 1


def test_a_token_completing_a_repeated_ngram_is_banned():
    pol = Policy("norepeat", penalty=1.0, window=64, ngram=3, ban=())
    z = np.array([0.0, 0.0, 0.0, 0.0, 9.0, 1.0])
    # history ... 1 2 4 ... 1 2: the 3-gram (1, 2, 4) exists, so 4 after (1, 2) is banned
    assert pol.pick(z, [1, 2, 4, 3, 1, 2]) == 5
    # the same tail without an earlier occurrence bans nothing
    assert pol.pick(z, [3, 1, 2]) == 4


def _reference(logits, history, penalty, window, ngram, ban):
    """The policy written out loop by loop, for the comparison below."""
    z = [float(x) for x in logits]
    for b in ban:
        z[b] = float("-inf")
    if penalty != 1.0 and history:
        for t in set(history[-window:]):
            z[t] = z[t] / penalty if z[t] > 0 else z[t] * penalty
    if ngram > 1 and len(history) >= ngram - 1:
        tail = history[len(history) - (ngram - 1) :]
        for i in range(len(history) - ngram + 1):
            if history[i : i + ngram - 1] == tail:
                z[history[i + ngram - 1]] = float("-inf")
    return max(range(len(z)), key=lambda j: (z[j], -j))  # the first maximum, as numpy's argmax


@pytest.mark.parametrize("seed", range(40))
def test_norepeat_matches_a_loop_by_loop_reading(seed):
    rng = np.random.default_rng(seed)
    v = 12
    logits = rng.standard_normal(v) * 2
    history = [int(t) for t in rng.integers(0, v, int(rng.integers(0, 90)))]
    pol = Policy("norepeat", penalty=float(rng.choice([1.0, 1.3, 2.0])), window=int(rng.integers(1, 70)),
                 ngram=int(rng.choice([0, 2, 3, 4])), ban=tuple(int(b) for b in rng.choice(v, 2, replace=False)))  # fmt: skip
    want = _reference(logits, history, pol.penalty, pol.window, pol.ngram, pol.ban)
    assert pol.pick(logits, history) == want


def test_a_record_names_its_policy_or_is_refused():
    p = Policy.from_record({"decodePolicy": {"policy": "norepeat", "penalty": 1.3, "window": 64,
                                             "ngram": 4, "ban": [0]}})  # fmt: skip
    assert p == NOREPEAT
    assert p.as_dict() == {
        "policy": "norepeat",
        "penalty": 1.3,
        "window": 64,
        "ngram": 4,
        "ban": [0],
    }
    for doc in ({}, {"decodePolicy": {"policy": "beam"}}, {"decodePolicy": "norepeat"}):
        with pytest.raises(ValueError, match="no decode policy"):
            Policy.from_record(doc)
    with pytest.raises(ValueError, match="temperature"):
        Policy.from_record({"decodePolicy": {"policy": "greedy", "temperature": 0.7}})
