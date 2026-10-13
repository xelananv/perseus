# SPDX-License-Identifier: MIT
"""The fhe-ssm recurrent language model (tag ``pbd430a``) on ``perseus.impl``.

One block is a time-mix followed by a channel-mix, both residual, every nonlinearity a learned
polynomial; the only approximated operation is the RMSNorm's inverse square root (Newton from a
constant seed, 12 steps). Per block the model carries two values across tokens: the scan state
``S`` and the previous token's normalized input ``u_prev`` (the token shift).

    time-mix     n  = rmsnorm(h)                        norm.rmsnorm
                 um = mix (.) n + (1 - mix) (.) n_prev
                 x  = W_in um                           linear (g1 folded in, weights.py)
                 S  = a (.) S + (1 - a) (.) x           a trained, fixed at inference
                 z  = c (.) S + dd (.) x
                 g  = z (.) (p0 + p1 z + p2 z^2)
                 h  = h + W_out g
    channel-mix  n2 = rmsnorm(h)
                 k  = W_k n2 ; r = W_r n2               one prepared input, two linears
                 act  = q0 + q1 k + q2 k^2 + q3 k^3
                 gate = r0 + r1 r + r2 r^2
                 h  = h + W_v (act (.) gate)

The reference is spec_decode/plain_recurrent.py of the fhe-ssm repository; ``ref.py`` here is
the same forward in float64 with the closed-form rsqrt (``exact_step``), and the kernels run
unchanged on ``NumpyOps`` (``mirror``) so that FHE noise, layout errors and the Newton
approximation are separable, the convention of examples/gpt2_from_primitives.

    geometry    the two layouts the kernels run on: cachemir slots (one token per ciphertext,
                channel i at slot i*t) and plain numpy vectors
    weights     the npz checkpoint + the exported Newton seeds -> per-block operands
    norm        RMSNorm from primitives (mask, ladder, constant-seed Newton)
    block       time_mix / channel_mix / ssm_block on the ops protocol
    model       FheSsmModel(ImplModel): stages, the carried state, the canonical carry
    ref         float64 references
    run_fake    the CPU driver: ticks on the numpy fake against the references
"""

import os

#: The rotate-and-sum ladders run plain on both routes. The two routes read ``LADDER_RADIX_BITS`` with
#: different defaults (3 in ``perseus.impl.layout``, so the fake hoists radix groups; 0 in the C++
#: ``rotate_and_sum``), and either hoists a group wherever the session holds its rotation keys, which
#: this layout's own steps do for two of the norm ladder's three multi-rung groups at d = 1,024. One
#: value on both routes keeps the fake's counts a GPU session's; the plain ladder is what was measured.
LADDER_RADIX_BITS = "0"


def port_env() -> None:
    """Export the switches this example pins on both routes, before a session or a fake is made.

    The write is process-wide, and the C++ reads the variable once, at its first ladder: run this
    example in a process of its own (its runners are), or set the variable before any model runs."""
    os.environ["LADDER_RADIX_BITS"] = LADDER_RADIX_BITS
