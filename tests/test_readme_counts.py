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
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"


def _collected() -> int:
    """How many tests this suite actually has, by asking pytest.

    A subprocess rather than a plugin hook: the count has to be of the WHOLE
    suite, and a hook inside a partial run (``pytest tests/radio``) would see
    only that part and report a mismatch that is not one.
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
    return sum(
        int(line.rsplit(":", 1)[1])
        for line in result.stdout.splitlines()
        if line.startswith("tests/") and line.rsplit(":", 1)[-1].strip().isdigit()
    )


def _quoted_counts() -> list[int]:
    """Every "N tests" the README states."""
    return [int(n.replace(",", "")) for n in re.findall(r"([\d,]+) tests\b", README.read_text())]


def test_the_readme_quotes_a_test_count_at_all():
    """Guard the guard: a README that stopped stating one would pass vacuously."""
    assert _quoted_counts(), "README states no test count for this to check"


def test_every_test_count_in_the_readme_is_the_real_one():
    """Exact, not approximate.

    A tolerance here would be a licence to drift, which is the failure being
    fixed -- 1354 was once exact too. Adding tests is meant to be a two-line
    change: the tests, and the number.
    """
    actual = _collected()
    for quoted in _quoted_counts():
        assert quoted == actual, (
            f"README says {quoted} tests; the suite collects {actual}. "
            f"Update README.md rather than loosening this assertion -- the "
            f"count went stale by 759 once because nothing checked it."
        )
