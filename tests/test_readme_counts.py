"""The README's test count is counted, not remembered.

``README.md``'s Status section said "1354 tests, ~97 % coverage" while the
suite held 2113 and covered 99.7 %. It had drifted by 759 tests, and nothing
would have said so: a number in prose is a claim no run checks, and the stale
version reads exactly as authoritative as the true one.

The operator census (``tests/radio/test_placeholder_census.py``) already solved
this shape for "17 of the 29 are placeholders" by deriving the count and
asserting the prose quotes it. This does the same for the test count, which is
the number a reader is most likely to take at face value.

No ``@pytest.mark.slow``: the project registers no markers, and an
unregistered one raises ``PytestUnknownMarkWarning`` -- a warning nobody reads
is precisely the failure mode this file exists to close. The collection
subprocess costs about three seconds.

**Coverage is deliberately not pinned here.** It moves with every line added to
``src/``, so an exact assertion would fail on unrelated work and be loosened
until it meant nothing; and the run that would measure it costs twelve minutes,
which is not a price to pay on every invocation of the suite. The README states
it to one decimal and ``pyproject.toml``'s ``--cov-fail-under`` is what actually
holds the floor. What is pinned is the count that a plain collection already
knows.
"""

import re
import subprocess
import sys
from functools import cache
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
TESTS = ROOT / "tests"

#: A module that defines at least one test, and so is expected to contribute to
#: the count. Modules gated by a module-level ``pytest.importorskip`` still
#: define their tests -- they simply do not collect when what they need is
#: absent, which is exactly the condition worth detecting.
_DEFINES_A_TEST = re.compile(r"^\s*(?:async\s+)?def\s+test_", re.MULTILINE)

#: Modules that can only collect where someone holds an unpublishable local
#: dataset, and whose tests are therefore **excluded** from the number the
#: README states.
#:
#: Not an exemption, a correction. Everything else on the optional list can be
#: installed -- ``h5py``, ``numpyro``, ``rhino-cal-jax`` all have a documented
#: command -- so an environment that collects them is reachable and the count is
#: checkable. ``RHEPLICANT_RHINO_CAL`` names a checkout that is on no index and
#: cannot be published, so counting its tests would make the README's number
#: verifiable on exactly one machine, and this check would stand down everywhere
#: else forever. That is this file's own failure mode, one level down: the guard
#: would never fail because it would never run. Excluding them makes the number
#: mean the same thing in every environment, so the assertion always happens.
_OPT_IN_LOCAL = frozenset({"tests/radio/test_ingestion_vs_reference.py"})


@cache
def _collected() -> dict[str, int]:
    """How many tests each module contributed, as pytest counts them.

    A subprocess rather than a plugin hook: the count has to be of the WHOLE
    suite, and a hook inside a partial run (``pytest tests/radio``) would see
    only that part and report a mismatch that is not one.

    The contributing modules come back too, because the total alone cannot be
    read. Several modules sit behind a module-level ``pytest.importorskip`` and
    contribute nothing when what they need is missing, so a venv without
    ``h5py`` or ``rhino-cal-jax`` reports a smaller number for the same suite --
    a partial view, not a shrunken suite. No count is quoted here on purpose:
    this file exists because numbers in prose go stale, and a docstring is
    prose.
    """
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "--collect-only", "-q", "--no-cov"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.skip(f"collection failed, nothing to compare against: {result.stderr[-200:]}")
    # `-q` prints "path/to/test_x.py: N" per file. Summing those is exact,
    # where parsing a summary line is not: `addopts` already carries `-q`, so
    # this run is effectively `-qq` and prints no summary line at all.
    #
    # It prints a WARNINGS SUMMARY, though, and the sentence above missed it.
    # Measured at Plan 3A's Task 12, the first task whose checks emit a
    # `ConfigWarning` from a document a test module loads at import time: the
    # summary's group header is bare `path/to/x.py:LINENO`, which the split
    # below read as "LINENO tests in path/to/x.py". Two headers reading
    # `tests/config/exit_helpers.py:226` credited 452 phantom tests and named a
    # helper module as a collector, and the guard reported 6137 against a true
    # 5685 -- a wrong number offered to a reader told to trust this message.
    # The separator is what tells them apart: a count is `": "` and a line
    # reference is `":"`.
    per_module: dict[str, int] = {}
    for line in result.stdout.splitlines():
        if not line.startswith("tests/"):
            continue
        path, sep, count = line.rpartition(": ")
        if not sep or not count.strip().isdigit():
            continue
        per_module[path] = per_module.get(path, 0) + int(count)
    return per_module


def _modules_that_define_tests() -> set[str]:
    """Every test module that ought to contribute, read off the source."""
    return {
        path.relative_to(ROOT).as_posix()
        for path in TESTS.rglob("test_*.py")
        if _DEFINES_A_TEST.search(path.read_text())
    }


def _quoted_counts() -> list[int]:
    """Every "N tests" the README states."""
    return [int(n.replace(",", "")) for n in re.findall(r"([\d,]+) tests\b", README.read_text())]


def test_the_readme_quotes_a_test_count_at_all():
    """Guard the guard: a README that stopped stating one would pass vacuously."""
    assert _quoted_counts(), "README states no test count for this to check"


def test_the_collection_probe_can_still_see_the_suite():
    """Guard the guard, in the direction that would silence it.

    The check below stands down when a module did not collect. If either half of
    that stopped matching -- the ``def test_`` scan, or the ``path: N`` parse --
    every module would look absent, the count check would stand down forever, and
    the number could drift exactly as it did before this file existed. A guard
    that cannot fail is the failure mode being closed, so both halves are pinned.
    """
    defined = _modules_that_define_tests()
    assert len(defined) > 50, (
        f"only {len(defined)} test modules found under {TESTS.name}/; the "
        "'def test_' scan has stopped matching rather than the suite having shrunk."
    )
    per_module = _collected()
    total = sum(per_module.values())
    assert total > 1000, f"collection reported only {total} tests; the parse has drifted"
    assert per_module.keys() <= defined, (
        "collection credited modules the source scan does not know about: "
        f"{sorted(per_module.keys() - defined)[:5]}"
    )
    # A typo in _OPT_IN_LOCAL would silently excuse a module that is not really
    # opt-in -- it would read as "expected absent" and never be missed again.
    assert _OPT_IN_LOCAL <= defined, (
        f"_OPT_IN_LOCAL names {sorted(_OPT_IN_LOCAL - defined)}, which define no "
        "tests. A stale path here excuses a module from the count for good."
    )


def test_every_test_count_in_the_readme_is_the_real_one():
    """Exact, not approximate.

    A tolerance here would be a licence to drift, which is the failure being
    fixed -- 1354 was once exact too. Adding tests is meant to be a two-line
    change: the tests, and the number.

    Exact against a *reachable* environment. The count is not a property of the
    source alone: modules behind a module-level ``pytest.importorskip`` drop out
    when what they need is absent, so a thinner venv collects fewer of the same
    tests. Told only "README says N; the suite collects M", a reader does the
    obvious thing and writes M into the README -- breaking it for everyone who
    has the extras, and doing it on the authority of a test that says "Update
    README.md". So an environment missing an *installable* dependency is not
    allowed to have an opinion about the number, and says why.

    Modules gated on an unpublishable local dataset are a different case and are
    subtracted instead of waited for -- see ``_OPT_IN_LOCAL``. Waiting for them
    would mean this assertion never ran anywhere.
    """
    per_module = _collected()
    actual = sum(n for path, n in per_module.items() if path not in _OPT_IN_LOCAL)

    absent = sorted(_modules_that_define_tests() - _OPT_IN_LOCAL - per_module.keys())
    if absent:
        shown = ", ".join(absent[:4]) + (" ..." if len(absent) > 4 else "")
        pytest.skip(
            f"this environment collects {actual} tests, which is not the size of "
            f"the suite: {len(absent)} module(s) did not collect at all, each "
            "standing down because an optional package or an opt-in local dataset "
            f"is absent. DO NOT put {actual} in the README -- it is a partial view "
            "and would be wrong for a complete environment. See docs/install.md for "
            "the extras and the RHEPLICANT_* variables, then re-run to get a number "
            f"worth comparing. Did not collect: {shown}"
        )

    for quoted in _quoted_counts():
        assert quoted == actual, (
            f"README says {quoted} tests; the suite collects {actual}. Every module "
            "collected, so this is real drift rather than a thin environment. "
            "Update README.md rather than loosening this assertion -- the count "
            "went stale by 759 once because nothing checked it."
        )
