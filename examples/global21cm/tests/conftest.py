"""Make ``global21cm`` importable and turn on float64 before any array exists.

These tests are not collected by the repository's suite (its ``testpaths``
is ``tests``); run them explicitly, see the README.
"""

import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

EXAMPLES = Path(__file__).resolve().parents[2]
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))
