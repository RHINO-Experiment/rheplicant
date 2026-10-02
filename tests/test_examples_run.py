"""Every example script runs to its end.

``docs/examples.md`` lists each script under ``examples/`` with what it does,
how long it takes and what it needs. Nothing ran them. ``gibbs_plan.py``
stopped at its second estimate with ``LinearityRefused`` at ``v0.9.0`` while
three pages gave its running time, and the refusal it exists to show had
become a different one: the script catches that exception and prints it, so
an exit code alone would not have shown the change.

Each script is run the way a reader runs it: its own interpreter, no
arguments, a scratch working directory. The page's table is read for two
things, so neither is spelled a second time here:

* **Needs.** A script whose row names ``cal`` or ``numpyro`` is skipped where
  that module does not import, and run everywhere else. A script that needs a
  module its row does not name fails here, which is how the row is kept
  true.
* **Time.** A row of :data:`SLOW_SECONDS` or more runs only when
  ``RHEPLICANT_ALL_EXAMPLES`` is set. At 0.9.1 that is three scripts and
  about five minutes between them.

On a CI runner (``CI`` set, as GitHub Actions sets it) no script runs unless
that variable is set, and the workflow does not set it: the scripts cost
runner minutes, about a minute and a half of them on a laptop for the twelve
quick ones.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from tests.test_tour_runs import _importable_in_a_fresh_interpreter

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
PAGE = ROOT / "docs" / "examples.md"

SCRIPTS = sorted(path.name for path in EXAMPLES.glob("*.py"))

#: A documented time at or above this is opt-in.
SLOW_SECONDS = 45.0
#: Set, every script runs: the slow ones, and on a CI runner too.
ALL_VARIABLE = "RHEPLICANT_ALL_EXAMPLES"

#: The page's "Needs" vocabulary, as the module each word imports.
NEEDS = {"cal": "rhino_cal_jax", "numpyro": "numpyro"}

_ROW = re.compile(r"^\* - `(\w+\.py)`")
_SECONDS = re.compile(r"^\**([0-9.]+) s\**$")

#: What a script must print, for the scripts whose failure is not an exit
#: code. ``gibbs_plan.py`` catches two refusals and prints each: the twin with
#: its noise operator still in it, and the plan over four switch positions.
#: Each being the refusal the script is there to show is asserted on the
#: output, the second with its directions named; the two exits follow.
MUST_PRINT = {
    "gibbs_plan.py": (
        r"^ParameterSpaceError: This ParameterSpace was given a forward model containing Noise",
        r"its joint Jacobian has nullity 2 of 34",
        r"^  direction 0: ",
        r"^  direction 1: ",
        r"^estimate\s+sweeps \d+\s+converged True",
        r"^  rank 34 of 34, nullity 0$",
        r"^sample\s+sweeps 120\s+warmup 20\s+kept 100\s",
    ),
}


def _rows() -> dict[str, tuple[float, tuple[str, ...]]]:
    """``{script: (seconds, modules)}`` from the page's three tables."""
    rows: dict[str, list[str]] = {}
    name = None
    for line in PAGE.read_text().splitlines():
        opened = _ROW.match(line)
        if opened:
            name = opened.group(1)
            rows[name] = []
        elif line.startswith((":::", "* - ")):
            name = None
        elif name is not None and line.startswith("  - "):
            rows[name].append(line[4:].strip())
        elif name is not None and line.startswith("    ") and rows[name]:
            rows[name][-1] += " " + line.strip()

    read = {}
    for script, cells in rows.items():
        assert len(cells) == 3, f"{PAGE.name}: the row for {script} has {len(cells)} cells, not 3"
        _, time, needs = cells
        seconds = _SECONDS.match(time)
        assert seconds, f"{PAGE.name}: the time for {script} reads {time!r}, not '<number> s'"
        words = [word.strip(" `") for word in needs.split(",")]
        unknown = [word for word in words if word != "—" and word not in NEEDS]
        assert not unknown, (
            f"{PAGE.name}: {script} needs {unknown}; the page defines {sorted(NEEDS)}"
        )
        read[script] = (
            float(seconds.group(1)),
            tuple(NEEDS[word] for word in words if word != "—"),
        )
    return read


def test_every_script_has_a_row_this_test_can_read() -> None:
    """The table is this module's input, so its coverage is asserted first."""
    assert len(SCRIPTS) >= 15, f"only {len(SCRIPTS)} scripts under {EXAMPLES}"
    rows = _rows()
    assert sorted(rows) == SCRIPTS


@pytest.mark.parametrize("script", SCRIPTS)
def test_the_script_runs_to_its_end(script: str, tmp_path: Path) -> None:
    seconds, modules = _rows()[script]
    everything = bool(os.environ.get(ALL_VARIABLE))
    if os.environ.get("CI") and not everything:
        pytest.skip(f"example scripts do not run on a CI runner; set {ALL_VARIABLE}=1 to run them")
    if seconds >= SLOW_SECONDS and not everything:
        pytest.skip(f"{script} is documented at {seconds:g} s; set {ALL_VARIABLE}=1 to run it")
    for module in modules:
        if not _importable_in_a_fresh_interpreter(module, cwd=tmp_path):
            pytest.skip(f"{script} needs {module}, which is not installed")

    result = subprocess.run(
        [sys.executable, str(EXAMPLES / script)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        timeout=1800,
    )
    assert result.returncode == 0, (
        f"examples/{script} exited {result.returncode}.\n\n{result.stderr[-4000:]}"
    )
    for pattern in MUST_PRINT.get(script, ()):
        assert re.search(pattern, result.stdout, re.MULTILINE), (
            f"examples/{script} ran to its end without printing {pattern!r}.\n\n"
            f"{result.stdout[-4000:]}"
        )


def test_the_printed_claims_name_scripts_that_exist() -> None:
    assert set(MUST_PRINT) <= set(SCRIPTS)
