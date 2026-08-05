"""``tests/evidence/`` runs only under ``jax_enable_x64``. This is the gate.

The evidence layer's arithmetic does not survive float32 -- a stored factor's
offset scalar is the time-bandwidth product, ~7.2e11 for one RHINO night,
against a difference of ~1e5 -- so these tests need x64. The rest of the suite
must not get it: float32 is the package's production dtype, and eighteen tests
elsewhere assert refusals that only float32 forces. That was measured, not
assumed: running the whole suite with ``JAX_ENABLE_X64=1`` fails exactly those
eighteen, and each is correct to fail.

Which is why this file does NOT call
``jax.config.update("jax_enable_x64", True)``. A conftest in a subdirectory is
imported while the whole session is being collected, so the update would land
before those eighteen ever run and break every one of them. The flag arrives
from the environment or it does not arrive.

Two further traps, both checked rather than assumed:

``pytest_collection_modifyitems`` declared here is a *session* hook. It is
handed every item in the run, not the subtree this conftest sits in --
measured by probe: collecting ``tests/core/test_basis.py tests/evidence``
called it once with 35 items, 34 of them outside this directory. An unfiltered
loop would therefore skip the entire suite whenever x64 is off, which in the
default session is always. Hence the explicit path test below.

Skip, not deselect. ``tests/test_readme_counts.py`` derives the README's stated
test count from a plain ``--collect-only`` and asserts the prose matches it
exactly. A deselected test is gone from that count; a skipped test is still
collected. Deselecting here would quietly make the README's number describe
something other than the suite.
"""

from pathlib import Path

import jax
import pytest

_DIRECTORY = Path(__file__).resolve().parent

_REASON = (
    "the evidence layer needs float64, and jax_enable_x64 is process-global -- "
    "run `JAX_ENABLE_X64=1 .venv/bin/python -m pytest tests/evidence`. "
    "This skip is not a hole in the default suite: "
    "tests/test_evidence_session.py runs exactly that command as a subprocess "
    "and fails if it fails, or if it turns out to have run nothing."
)


def pytest_collection_modifyitems(config, items):
    """Defer this directory's tests when the process is not in x64."""
    if jax.config.read("jax_enable_x64"):
        return
    gate = pytest.mark.skip(reason=_REASON)
    for item in items:
        path = getattr(item, "path", None)
        # Filter by path rather than trusting pytest to scope a subdirectory
        # conftest's hook, because it does not -- see the module docstring.
        if path is not None and Path(path).is_relative_to(_DIRECTORY):
            item.add_marker(gate)
