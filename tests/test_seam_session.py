"""The adapter's float64 session, run from inside the float32 one.

``tests/seam/`` holds P1's ten acceptance examples for
``rheplicant.inference.graph_bridge``. Its deterministic tier compares the
number that comes back through the seam against a dense solve of the same
normal equations at ``rtol <= 1e-12``, and float32 carries about seven decimal
digits -- so at the suite's default precision that tier would be measuring the
dtype and not the seam.

The flag cannot be switched on for the whole suite, and it cannot be scoped to
a block: ``jax_enable_x64`` is process-global and read before the first array
exists. The reasons, the population of tests that are RIGHT to fail under x64,
and the command that reproduces them are recorded once in
``tests/test_evidence_session.py`` and deliberately not repeated here.

So this is the same idiom as that file, for a second directory, and it exists
for the same reason: ``tests/seam/conftest.py`` skips those tests in the
default session, and a skip reads as a pass to anyone scanning the summary
line. Exit code alone cannot tell a healthy run from one that collected
nothing, so the passed-count and the gate's own skip reason are both checked.

**When does this file go?** Not when the evidence layer migrates. The adapter
is a PERMANENT surface -- it is what this package will be after the migration,
not something the migration removes -- so its acceptance tier keeps needing
float64 for as long as it makes a machine-precision claim. That is the
difference from ``tests/evidence/``, whose x64 need leaves with the layer.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

_PASSED = re.compile(r"(\d+) passed")

#: A distinctive fragment of tests/seam/conftest.py's skip reason. Matched as
#: TEXT rather than by importing the constant, because the point is to observe
#: what the child process emitted, not what this process can import.
_GATE_MARKER = "jax_enable_x64 is process-global"


def _parallel() -> list[str]:
    """``-n 2`` when xdist is installed, nothing when it is not.

    Two and not four: this directory is twenty tests, several of which draw 400
    GCR samples, and the parent is itself a parallel session. The evidence
    session next door takes four because it is an order of magnitude longer;
    copying that number here would oversubscribe the box for a saving the run
    is too short to contain.

    Conditional because ``pytest-xdist`` is dev-group only, and a hard ``-n``
    would turn a thin environment's run into a usage error -- which arrives as
    exit code 4 and reads, to anything checking only for non-zero, exactly like
    the seam session failing.
    """
    try:
        import xdist  # noqa: F401
    except ImportError:
        return []
    return ["-n", "2"]


@pytest.fixture(scope="module")
def session() -> subprocess.CompletedProcess:
    """One child run, shared by the assertions that read it."""
    return subprocess.run(
        # `-rs` so skip REASONS are printed: the assertion below has to tell a
        # skip the dtype gate caused from one an optional dependency caused,
        # and only the reason distinguishes them.
        [sys.executable, "-m", "pytest", "tests/seam", "--no-cov", "-rs", *_parallel()],
        cwd=ROOT,
        env={**os.environ, "JAX_ENABLE_X64": "1"},
        capture_output=True,
        text=True,
    )


def test_the_gated_x64_session_passes(session):
    """The whole point: ``tests/seam/`` is not optional, it is deferred."""
    assert session.returncode == 0, session.stdout + session.stderr


def test_the_gated_x64_session_actually_ran_tests(session):
    """A green child that ran nothing is the failure this guard is for.

    Rename the directory, break the collection, mis-scope the conftest's skip
    hook so it fires even under x64 -- each leaves the child exiting 0 with
    nothing executed, and the assertion above would applaud.
    """
    output = session.stdout + session.stderr
    passed = _PASSED.search(output)
    assert passed and int(passed.group(1)) > 0, (
        "the seam session reported no passing tests, which a broken collection "
        f"and a healthy empty directory both do while exiting 0:\n{output}"
    )
    assert _GATE_MARKER not in output, (
        "the seam session skipped tests through tests/seam/conftest.py's dtype "
        "gate, but it runs with JAX_ENABLE_X64=1 so the gate should not have "
        f"fired -- the flag is not reaching the child:\n{output}"
    )
