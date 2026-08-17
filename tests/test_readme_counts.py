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

**Coverage's VALUE is deliberately not pinned here.** It moves with every line
added to ``src/``, so an exact assertion would fail on unrelated work and be
loosened until it meant nothing; and the run that would measure it costs twelve
minutes, which is not a price to pay on every invocation of the suite. The
README states it to one decimal and ``pyproject.toml``'s ``--cov-fail-under``
is what actually holds the floor. What is pinned exactly is the count that a
plain collection already knows.

**Its SHAPE is pinned, and that is new in Plan 3C.** Until now the count was
pinned by equality while the percentage standing beside it was matched by
nothing at all -- so the two numbers in one parenthesis had completely
different guarantees, and a reader had no way to know which. What
:func:`test_the_readme_states_one_live_coverage_figure_beside_its_test_count`
asserts is everything about that figure that is checkable without a
twelve-minute run: that there is exactly ONE live claim, that it stands in the
same parenthesis as the pinned count so the two are edited together, that it is
written to the one decimal the README promises, and that it is not below the
floor the suite actually enforces. The README carries a SECOND percentage --
99.7 %, a claim about a past commit -- and keeping the live one distinguishable
from it is most of the value.
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
    # helper module as a collector -- a wrong number offered to a reader whom
    # every task brief tells to take the count from this message. Re-measured
    # at Task 12's fix commit by reverting the token below: 6189 reported
    # against a true 5737, the same delta of 452 and the same sole culprit.
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


#: The LIVE coverage claim: a percentage standing in the same parenthesis as a
#: test count. Written as one pattern over both numbers on purpose -- it is
#: what makes the pinned count and the unpinned percentage a single edit, and
#: it is what tells the live claim apart from the README's other percentage
#: ("the 99.7 % it was before the evidence layer landed"), which is a statement
#: about a past commit and is deliberately not touched.
_LIVE_COVERAGE = re.compile(r"\(([\d,]+) tests, (\d+\.\d) % coverage")

#: Any sentence claiming a coverage percentage, live or not. The count of
#: these minus the live one is what must be explained as historical.
_ANY_COVERAGE = re.compile(r"(\d+(?:\.\d+)?) % coverage")


def _cov_fail_under() -> int:
    """The floor ``pyproject.toml`` makes the suite actually enforce."""
    found = re.search(r"--cov-fail-under=(\d+)",
                      (ROOT / "pyproject.toml").read_text())
    assert found, (
        "pyproject.toml no longer carries --cov-fail-under. It is the only "
        "thing that holds the coverage floor, and the README's figure is "
        "compared against it below -- with it gone, that comparison would "
        "either crash or, if softened, pass vacuously forever."
    )
    return int(found.group(1))


def test_the_readme_quotes_a_test_count_at_all():
    """Guard the guard: a README that stopped stating one would pass vacuously."""
    assert _quoted_counts(), "README states no test count for this to check"


def test_the_readme_states_one_live_coverage_figure_beside_its_test_count():
    """The percentage next to the pinned count was matched by nothing.

    Four properties, and each one can fail on its own:

    1. **The live claim exists at all**, in the "(N tests, X % coverage"
       shape. A README that moved the figure into a sentence of its own would
       separate it from the count this file pins by equality, and the next
       reader updating the count would have no reason to touch it.
    2. **The number beside it is the number the sibling test pins.** The same
       match yields both, so they cannot be read from two different sentences.
    3. **Exactly one percentage on the page is a live coverage claim.** The
       README carries a second -- 99.7 %, about a past commit -- and the whole
       reason this guard reads a *pairing* rather than a percentage is to tell
       them apart. A second live claim, or the historical one reworded into
       the live shape, is caught here.
    4. **The live figure is not below the floor the suite enforces.** A README
       claiming less coverage than ``--cov-fail-under`` states something every
       passing run disproves.

    **What this deliberately does NOT assert, said plainly so nobody assumes
    otherwise:** that the figure is the true coverage. That needs the
    twelve-minute run, and this file's own docstring explains why an exact
    pin would be loosened until it meant nothing. The floor comparison is a
    bound, not a measurement.
    """
    text = README.read_text()
    live = _LIVE_COVERAGE.findall(text)
    assert len(live) == 1, (
        f"the README states {len(live)} '(N tests, X % coverage' pairings and "
        "this guard needs exactly one. The pairing is what keeps the pinned "
        "count and the unpinned percentage a single edit; splitting them "
        "leaves the percentage with no reader again."
    )
    counted, figure = live[0]
    assert int(counted.replace(",", "")) in _quoted_counts(), (
        f"the count beside the coverage figure is {counted}, which is not one "
        "of the counts the equality pin above reads. The two guards are "
        "looking at different sentences."
    )
    everywhere = _ANY_COVERAGE.findall(text)
    assert everywhere.count(figure) == 1 and len(everywhere) == 1, (
        f"the README states {len(everywhere)} coverage percentages "
        f"({everywhere}) and exactly one may be a live claim. The 99.7 % "
        "figure is a statement about a past commit and is written as one "
        "('rather than the 99.7 % it was before ...'); if a second live claim "
        "is really wanted, it needs its own guard, not a widening of this."
    )
    floor = _cov_fail_under()
    assert float(figure) >= floor, (
        f"README says {figure} % coverage while pyproject.toml enforces a "
        f"--cov-fail-under={floor} floor that every passing run clears. One "
        "of the two is wrong, and it is not the run."
    )


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
