# SPDX-License-Identifier: MIT
"""The fhe-ssm recurrent language model (tag ``pbd430a``) on ``perseus.impl``, with its speed axes as flags.

A block is a time-mix then a channel-mix, both residual, every nonlinearity a learned polynomial; the one
approximated operation is the RMSNorm's inverse square root (Newton from a constant seed); each block
carries the scan state and the previous token's normalized input. The equations, the folds, the layouts
and the axes are in README.md. Modules: ``geometry`` (the layouts the kernels run on), ``filling`` (the
dense token layout and its diagonal linear), ``weights``, ``norm``, ``lanemix``, ``block``, ``model``
(``FheSsmModel``), ``fakes`` (the fake with the runtime's rules; the capture session), ``ref`` (the
float64 references), ``run_fake`` (the CPU driver, every axis a flag).
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
