"""Shared setup for the comparisons against bayesmith.

These files run the same inputs through this package and through bayesmith
and compare the outputs. They were bayesmith's ``tests/crosscheck/`` until its
``61d4644`` removed that directory so that its suite no longer imports its
downstream; the two files here are the ones that hold an implementation this
package still has a copy of. ``docs/stability.md`` lists them.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _heal_the_global_x64_flag():
    """Restore the process-global x64 flag a call inside a float64 block can leak.

    Measured under xdist in bayesmith's suite: this package's diagnostics
    force x64 with a save and restore built on ``jax.config.read`` and
    ``update``. Run inside ``with jax.enable_x64(True):``, the ``read``
    returns the effective value, True, and the ``finally`` writes True into
    the global config. When the context manager exits its override goes and
    the global True stays, so every later test in the same worker runs in
    float64. The suite's main session asserts refusals only float32 forces.

    Setup and teardown run outside any test-body context manager, so ``read``
    here is the global value and putting it back undoes the leak.
    """
    import jax

    was = jax.config.read("jax_enable_x64")
    yield
    if jax.config.read("jax_enable_x64") != was:
        jax.config.update("jax_enable_x64", was)
