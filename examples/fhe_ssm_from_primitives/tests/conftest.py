"""Put the repository root on sys.path so the tests import ``examples.fhe_ssm_from_primitives``
the way the repository's own tests import ``examples.gpt2_from_primitives``."""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
