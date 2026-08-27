"""The evidence layer's float64 session, run from inside the float32 one.

The streaming-evidence subsystem cannot be tested at the suite's default
precision. Under ``RadiometerNoise`` a stored factor's offset scalar is the
time-bandwidth product -- about 7.2e11 for one RHINO night -- while the
difference the quadratic form has to resolve is about 1e5. float32 carries
roughly seven decimal digits, so that difference does not survive the
subtraction at all: the quantity is annihilated, not merely imprecise. Every
test under ``tests/evidence/`` therefore needs ``jax_enable_x64``.

The flag cannot simply be switched on for the whole suite. float32 is this
package's production dtype (s2fft and healpix bound it there), and a population
of tests elsewhere assert refusals that only float32 forces. **This module is
the one place that population is written down**; everywhere else that mentions
it -- ``README.md``, ``DESIGN.md``, ``docs/install.md``,
``tests/evidence/conftest.py``, ``tests/inference/test_gradient_transition.py``
-- points here rather than restating the number, because six copies of one
measurement is six things to update and one thing that actually gets updated.

**Measured 2026-08-25 on ``main`` 49eef1e: twenty-two**, in
``tests/core/test_coordinates.py`` (10), ``tests/radio/test_cw_time_axis.py``
(3), ``tests/radio/test_filters.py`` (2),
``tests/inference/test_stochastic_twin.py`` (2), ``tests/radio/test_rhino.py``
(1), ``tests/inference/test_gls.py`` (1), ``tests/inference/test_loss_sense.py``
(1), ``tests/inference/test_linear_blocks.py`` (1) and
``tests/inference/test_conjugate_transition.py`` (1) -- and every one of them is
right to fail. The command::

    JAX_ENABLE_X64=1 .venv/bin/python -m pytest -n 8 --ignore=tests/config

**``--ignore=tests/config`` is load-bearing, and it was not needed when this
paragraph was first written.** The claim used to read "running the whole suite
with ``JAX_ENABLE_X64=1`` fails exactly eighteen" and was true when measured
(``a26c64d``, 2026-08-05). Thirteen days later ``52f3ea3`` added the bootstrap
runtime audit, which refuses a document declaring ``runtime.jax_enable_x64:
false`` inside a process that has it on -- correctly, and 1388 times, every one
of them in ``tests/config``. So the unmodified command now reports 1137 failures
and 316 errors, and the eighteen it was pointing at are a 1.2 % minority of the
noise.

Worth naming, because the shape is nastier than a wrong number: **the
conclusion survived and only the reproduction died.** All eighteen originally
listed still fail, still for the reason given; four more have joined since. A
reader re-running the documented command does not find a claim that is subtly
off, they find a swamp -- and the natural inference from a swamp is that the
claim was wrong, which it was not. A stale count gets corrected; a stale recipe
gets the correct finding discarded. (One further caveat, so a re-run is not
misread: ``tests/gui/e2e/test_packaged_frontend.py`` drives 188 browser tests on
14 workers and has been seen to fail under this command while passing in the
default suite and in the unmodified x64 run. It is oversubscription, not dtype.)

Nor can the flag be scoped to a block: jax 0.11.0 has no
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

**When the evidence layer migrates, this file goes -- and the twenty-two do
not become anybody's problem.** The migration spec's §六 step 3 calls the merge
of the two sessions "一个可能的红利", a possible dividend, and says to re-assess
this module's reason to exist when it lands. Measured on 2026-08-25, ahead of
that: ``tests/evidence/`` was then the only directory in the suite carrying an
x64 collection gate, so once it leaves for bayesmith there is no x64 consumer
left and the second session has nothing to run. The merge is therefore
**"delete this file and that conftest"**, not "turn x64 on for the whole
suite".

**Corrected 2026-08-27: it is no longer the only one, and the conclusion above
survives only for THIS file.** ``tests/seam/`` now carries the same gate, for
the adapter's deterministic tier -- a machine-precision comparison against a
dense reference, which float32 cannot state. Its driver is
``tests/test_seam_session.py``. The difference that matters when the merge is
assessed: the evidence layer's x64 need LEAVES with the layer, and the
adapter's does not, because the adapter is what this package keeps rather than
what it hands over. So "the two sessions merge" is now, at best, "one of the
two goes"; a reader planning that step from the paragraph above alone would
plan to delete a directory that has to stay.

The distinction is the whole point of measuring early, because the phrase "the
two sessions merge" reads equally well as the second thing, and the second
thing is a multi-day trap: it costs the twenty-two refusals above, and only six
of them live under ``tests/inference/`` and leave with the migration. The other
sixteen are in ``tests/core/`` and ``tests/radio/``, which are not migrating,
and they are load-bearing -- float32 stays this instrument's production dtype
whatever the inference layer does.

What does NOT leave with the evidence layer is the x64 *child process* idiom:
``tests/radio/test_driftscan_projector.py``, ``tests/radio/test_sky_abstraction.py``,
``tests/config/test_config_delivery.py`` and ``tests/config/test_config_section_runtime.py``
each spawn their own, and none of them is migrating. So "rheplicant stops
needing float64" would be the wrong summary to carry forward; the accurate one
is that it stops needing a float64 *pytest session*.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

_PASSED = re.compile(r"(\d+) passed")

#: A distinctive fragment of tests/evidence/conftest.py's skip reason. Matched
#: as text rather than importing the constant, because the point is to observe
#: what the CHILD process emitted, not what this process can import.
_GATE_MARKER = "jax_enable_x64 is process-global"


def _parallel() -> list[str]:
    """``-n 4`` when xdist is installed, nothing when it is not.

    This child is the whole suite's critical path, and it ran serially while
    the parent's other seven workers sat idle. Measured on this machine:
    175 s serial, 80 s at ``-n 8``, 78 s at ``-n 4`` -- so the win is the
    first few workers and the rest is noise. **Four, not eight**: the parent
    is itself an ``-n 8`` session, so eight here would oversubscribe the box
    for a second that the measurement says is not there.

    Conditional because ``pytest-xdist`` is dev-group only, and this module's
    contract is that plain ``pytest`` stays authoritative over
    ``tests/evidence/``. A hard ``-n`` would turn a thin environment's run
    into a usage error, and a usage error here reads as the evidence session
    failing -- which is exactly the alarm this file exists to raise honestly.
    """
    try:
        import xdist  # noqa: F401
    except ImportError:
        return []
    return ["-n", "4"]


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
        # `-rs` so skip REASONS are printed: the assertion below has to tell a
        # skip the dtype gate caused from one an optional dependency caused,
        # and only the reason distinguishes them.
        [sys.executable, "-m", "pytest", "tests/evidence", "--no-cov", "-rs",
         *_parallel()],
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
    read out of the summary and required to be positive.

    What is NOT required is that nothing skipped at all. An earlier version
    asserted that, and it was wrong in a way that would have surfaced on
    somebody else's machine: ``numpyro`` is an optional extra
    (``pyproject.toml``), ``tests/evidence/test_memory_numpyro.py``
    ``importorskip``s it, and without the extra installed the child reports
    "66 passed, 1 skipped" -- whereupon the default suite went red pointing at
    ``tests/evidence/conftest.py``'s dtype gate, which had not fired and was
    not involved. The predicate wanted is "no skip the gate caused", so the
    gate's own reason is what gets searched for.
    """
    output = session.stdout + session.stderr
    passed = _PASSED.search(output)
    assert passed and int(passed.group(1)) > 0, (
        f"the x64 session reported no passing tests, which a broken collection "
        f"and a healthy empty directory both do while exiting 0:\n{output}"
    )
    assert _GATE_MARKER not in output, (
        f"the x64 session skipped tests through tests/evidence/conftest.py's "
        f"dtype gate, but it runs with JAX_ENABLE_X64=1 so the gate should not "
        f"have fired -- the flag is not reaching the child:\n{output}"
    )
