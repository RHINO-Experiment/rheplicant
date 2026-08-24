"""The three TypeScript gates are RUN, not merely described.

``AGENTS.md`` lists three commands under "TypeScript gates, all of which must
pass". Two of them already have a Python module here -- but each of those
checks the SHAPE of its project (which files are in the program, which flags
the project states) and says so in its own words: *"the return code is
deliberately not asserted: a guard with two jobs reports neither clearly."*
That is a good reason to keep those tests narrow and not a reason for the exit
code to be asserted nowhere, which is where it was.

Measured cost of that gap: ``check:tests`` went red in ``402f3d5``, which added
``units`` to ``ProjectedWidget`` and updated the Python drift test but not the
nine widget fixtures in ``ConfigForms.test.tsx`` and the three beside them. The
whole Python suite stayed green for a day, because nothing in it ran the gate.
``npm test`` would not have found it either -- it runs a different, much
smaller config.

So this module has exactly one job, stated three times. It runs each gate as a
human runs it and asserts that it passed.
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_REACT = _ROOT / "tools/config_gui_spike/react"

#: Each gate, exactly as ``AGENTS.md`` spells it. ``test:session`` takes the
#: separating ``--`` because it forwards to vitest, which otherwise watches.
_GATES = (
    pytest.param(["npm", "run", "check:tests"], id="check:tests"),
    pytest.param(["npm", "run", "check:e2e"], id="check:e2e"),
    pytest.param(["npm", "run", "test:session", "--", "--run"], id="test:session"),
)


def _toolchain_or_skip() -> None:
    """Skip only for an absent toolchain, and say that a skip is not a pass.

    The same shape ``tests/test_readme_counts.py`` uses, for the same reason it
    gives: a guard that stands down quietly is one that can be missing for
    weeks with real failures behind it -- which is precisely what happened to
    the gate this module exists to run.
    """
    if shutil.which("npm") is None:
        pytest.skip(
            "npm is not on PATH, so the TypeScript gates did not run. THIS IS "
            "NOT A PASS: check:tests, check:e2e and test:session are three of "
            "the gates AGENTS.md requires, and this suite is the only thing "
            "that runs them. Install Node and re-run before believing the "
            "frontend is green."
        )
    if not (_REACT / "node_modules" / ".bin").is_dir():
        pytest.skip(
            f"{_REACT}/node_modules is not installed, so the TypeScript gates "
            "did not run. THIS IS NOT A PASS -- see the message above; run "
            "`npm ci` in that directory."
        )


@pytest.mark.parametrize("command", _GATES)
def test_the_gate_passes(command: list[str]) -> None:
    """One gate, one job: it exited zero.

    The output is attached to the failure whole rather than summarised. A tsc
    error names a file, a line and the two types it could not reconcile, and
    every one of those is what the reader needs; a message that said only
    "check:tests failed" would send them to run it again by hand.
    """
    _toolchain_or_skip()

    completed = subprocess.run(
        command, cwd=_REACT, check=False, capture_output=True, text=True
    )

    assert completed.returncode == 0, (
        f"{' '.join(command)} failed with {completed.returncode}. This is one of "
        "the three gates AGENTS.md requires; the whole output follows.\n"
        f"--- stdout ---\n{completed.stdout}\n--- stderr ---\n{completed.stderr}"
    )


def test_this_module_runs_every_gate_agents_md_lists() -> None:
    """The census, so a fourth gate cannot be added and left unrun.

    ``AGENTS.md`` is the list; this reads it rather than restating it, because
    a second copy of the list is the copy that goes stale -- and a gate added
    there and forgotten here is exactly the hole this module was written to
    close.
    """
    fenced = (_ROOT / "AGENTS.md").read_text()
    _, marker, after = fenced.partition("TypeScript gates, all of which must pass")
    assert marker, "AGENTS.md no longer introduces the gates in the phrasing read here"
    block, _, _ = after.partition("```\n\n")
    listed = {
        line.split("#")[0].strip()
        for line in block.splitlines()
        if line.strip().startswith("npm ")
    }
    assert listed, f"no npm gate parsed out of AGENTS.md's block: {block!r}"

    # AGENTS.md pads `test:session` for column alignment, so both sides are
    # compared on their words rather than their whitespace.
    listed = {" ".join(entry.split()) for entry in listed}
    run = {" ".join(command.values[0]) for command in _GATES}

    assert listed == run, (
        f"AGENTS.md lists {sorted(listed)}; this module runs {sorted(run)}."
    )
