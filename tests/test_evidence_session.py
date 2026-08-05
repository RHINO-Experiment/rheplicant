"""The evidence layer's float64 session, run from inside the float32 one.

The streaming-evidence subsystem cannot be tested at the suite's default
precision. Under ``RadiometerNoise`` a stored factor's offset scalar is the
time-bandwidth product -- about 7.2e11 for one RHINO night -- while the
difference the quadratic form has to resolve is about 1e5. float32 carries
roughly seven decimal digits, so that difference does not survive the
subtraction at all: the quantity is annihilated, not merely imprecise. Every
test under ``tests/evidence/`` therefore needs ``jax_enable_x64``.

The flag cannot simply be switched on for the whole suite. float32 is this
package's production dtype (s2fft and healpix bound it there), and eighteen
tests assert refusals that only float32 forces -- measured, by running the
whole suite with ``JAX_ENABLE_X64=1``: exactly eighteen fail, in
``tests/core/test_coordinates.py`` (10), ``tests/radio/test_cw_time_axis.py``
(3), ``tests/radio/test_rhino.py`` (1), ``tests/inference/test_gls.py`` (1),
``tests/inference/test_loss_sense.py`` (1) and
``tests/inference/test_stochastic_twin.py`` (2), and every one of them is right
to. Nor can the flag be scoped to a block: jax 0.11.0 has no
``jax.experimental.enable_x64`` context manager, and the setting is read once,
before the first array exists. A separate process carrying
``JAX_ENABLE_X64=1`` in its environment is the only mechanism available.

So ``tests/evidence/`` is a second pytest session, and this file is what keeps
plain ``pytest`` authoritative over it: the default suite runs that session and
goes red when it goes red. This matters because ``tests/evidence/conftest.py``
skips those tests in the default session, and a skip reads as a pass to anyone
scanning the summary line. That is why the exit code is not the only thing
asserted below -- a child that collected nothing, or skipped everything, exits
0 exactly like a healthy one. Those two outcomes are the silent false green
this file exists to close, so they are checked by name.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

_PASSED = re.compile(r"(\d+) passed")
_SKIPPED = re.compile(r"(\d+) skipped")


@pytest.fixture(scope="module")
def session() -> subprocess.CompletedProcess:
    """One child run, shared by the assertions that read it.

    Module-scoped because the run costs a full interpreter start plus a jax
    import; the alternative is paying that once per assertion, for output that
    cannot differ between them.
    """
    return subprocess.run(
        # No ``-q`` of our own. ``addopts`` in pyproject.toml already carries
        # one, and a second makes it ``-qq``, which drops the summary line
        # entirely -- verified by running both forms by hand: with the extra
        # flag the child printed only the progress dots, without it
        # "1 passed in 0.05s". No summary line means no passed-count to read,
        # and the returncode alone cannot tell a green run from an empty one.
        [sys.executable, "-m", "pytest", "tests/evidence", "--no-cov"],
        cwd=ROOT,
        env={**os.environ, "JAX_ENABLE_X64": "1"},
        capture_output=True,
        text=True,
    )


def test_the_gated_x64_session_passes(session):
    """The whole point: ``tests/evidence/`` is not optional, it is deferred."""
    assert session.returncode == 0, session.stdout + session.stderr


def test_the_gated_x64_session_actually_ran_tests(session):
    """A green child that ran nothing is the failure this guard is for.

    Rename the directory, break the collection, mis-scope the conftest's skip
    hook so it fires even under x64 -- each leaves the child exiting 0 with
    nothing executed, and the assertion above would applaud. So the count is
    read out of the summary and required to be positive, and skips are required
    to be absent: inside the child the flag IS set, so a skip there means the
    gate misfired rather than that the tests were correctly deferred.
    """
    output = session.stdout + session.stderr
    passed = _PASSED.search(output)
    skipped = _SKIPPED.search(output)
    assert passed and int(passed.group(1)) > 0, (
        f"the x64 session reported no passing tests, which a broken collection "
        f"and a healthy empty directory both do while exiting 0:\n{output}"
    )
    assert not skipped, (
        f"the x64 session skipped tests, but it runs with JAX_ENABLE_X64=1, so "
        f"tests/evidence/conftest.py's dtype gate should not have fired:\n{output}"
    )
