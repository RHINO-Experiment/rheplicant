"""The guided tour claims to be a script; this runs it and proves the claim.

``docs/tour.md`` says its snippets "pasted top to bottom form a working script".
Nothing was enforcing that. ``myst-parser`` renders code blocks, it does not run
them the way ``myst-nb`` would, so the promise was kept by hand -- and had already
broken: section 3 called ``DriftScanProjector.from_beam_maps(beam_maps, ...)``
with ``beam_maps`` never defined anywhere in the document. A reader pasting the
tour hit a ``NameError`` two thirds of the way down, and the failure was
invisible to the whole test suite.

**Why a subprocess rather than ``exec``.** The claim under test is precisely "a
reader who pastes this into a file and runs it gets a working script", so the
test does that: concatenate the fences, write a file, run it with
``sys.executable``. It also keeps the tour's global effects -- it enables x64,
which is process-wide and irreversible -- out of the process running the rest of
the suite.

Two categories of block are not part of the script, and each has its own guard.

**Optional extras.** The worked example needs ``rhino_cal_jax``, which is not a
dependency (``NoiseWaveOperator`` is an adapter over it -- D15) and is not even
an entry in ``[project.optional-dependencies]``. A block declares what it needs
with a first-line marker::

    # needs-extra: rhino_cal_jax

The name is the module that is IMPORTED, not the pip extra: a first draft wrote
``limtod`` where the code does ``import limtod_jax``, ``find_spec`` said absent,
the block was skipped, and the ``beam_maps`` bug it was meant to catch stayed
invisible behind a green test. Because the tour is one continuous script, a
missing extra skips the whole run rather than the block -- the blocks after it
use its names -- which follows this suite's ``pytest.importorskip`` convention.

**Sketches.** A few fences are one-line illustrations of a signature
(``Pipeline(sky, beam, gain)``) whose names are deliberately abstract. They are
marked ``# sketch``, excluded from the script, and still checked: each must parse
as Python, and there is a cap, so "mark it a sketch" cannot quietly become the
way an unrunnable snippet gets into the tour.
"""

import ast
import re
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path

import pytest

TOUR = Path(__file__).resolve().parents[1] / "docs" / "tour.md"

#: ```python fences, in document order.
_FENCE = re.compile(r"^```python\n(.*?)^```", re.MULTILINE | re.DOTALL)

#: ``# needs-extra: <module>`` -- the module the block imports.
_NEEDS_EXTRA = re.compile(r"^\s*#\s*needs-extra:\s*(\S+)", re.MULTILINE)

#: ``# sketch`` -- an illustrative fence, excluded from the script.
_SKETCH = re.compile(r"^\s*#\s*sketch\s*$", re.MULTILINE)

#: The modules a marker may name. A typo would otherwise skip the run forever,
#: and silently, since a misspelled module and an uninstalled one are the same
#: thing to ``find_spec``.
_KNOWN_EXTRAS = frozenset({"rhino_cal_jax", "limtod_jax", "numpyro"})

#: Below this, the fence pattern has stopped matching rather than the tour
#: having shrunk -- a vacuous pass is the failure mode worth naming.
_MIN_BLOCKS = 10

#: Sketches earn their exemption by being rare. Raising this is a decision, not
#: a formality: every sketch is a snippet nothing executes.
_MAX_SKETCHES = 6


def _blocks() -> list[str]:
    return _FENCE.findall(TOUR.read_text())


def test_the_tour_is_a_script_that_runs(tmp_path: Path) -> None:
    """Paste every runnable python block into one file, in order, and run it."""
    blocks = _blocks()
    assert len(blocks) >= _MIN_BLOCKS, (
        f"Only {len(blocks)} python blocks found in {TOUR.name}; the fence pattern "
        "has probably stopped matching rather than the tour having shrunk."
    )

    for number, source in enumerate(blocks, start=1):
        for name in _NEEDS_EXTRA.findall(source):
            if find_spec(name) is None:
                pytest.skip(
                    f"{TOUR.name} block {number} needs {name}, which is not installed; "
                    "the tour is one continuous script, so the run cannot be verified "
                    "without it."
                )

    parts = [
        f"# ---- {TOUR.name} block {number} ----\n{source}"
        for number, source in enumerate(blocks, start=1)
        if not _SKETCH.search(source)
    ]
    script = tmp_path / "tour_as_pasted.py"
    script.write_text("\n".join(parts))

    result = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, timeout=900
    )
    if result.returncode != 0:
        raise AssertionError(
            f"{TOUR.name} is not a runnable script -- pasting it top to bottom fails.\n"
            f"Script written to {script} (the '# ---- block N ----' comments locate "
            f"the failure in the document).\n\n{result.stderr}"
        )


def test_sketches_are_rare_and_still_parse() -> None:
    """A ``# sketch`` is exempt from running, not from being Python."""
    sketches = [(n, s) for n, s in enumerate(_blocks(), start=1) if _SKETCH.search(s)]
    assert len(sketches) <= _MAX_SKETCHES, (
        f"{len(sketches)} sketch blocks in {TOUR.name}, cap is {_MAX_SKETCHES}. "
        "A sketch is a snippet nothing executes; if the tour needs more of them, "
        "raise the cap deliberately and say why."
    )
    for number, source in sketches:
        try:
            ast.parse(source)
        except SyntaxError as exc:
            raise AssertionError(
                f"{TOUR.name} block {number} is marked '# sketch' but is not valid "
                f"Python: {exc}\n\n{source}"
            ) from exc


def test_every_needs_extra_marker_names_a_real_module() -> None:
    """A marker with a typo would skip the whole run forever, and silently."""
    for number, source in enumerate(_blocks(), start=1):
        for name in _NEEDS_EXTRA.findall(source):
            assert name in _KNOWN_EXTRAS, (
                f"{TOUR.name} block {number} declares 'needs-extra: {name}', which is "
                f"not one of {sorted(_KNOWN_EXTRAS)}. If it is a new one, add it to "
                "_KNOWN_EXTRAS; otherwise it is a typo that would skip the run."
            )


def test_the_graph_assembly_heading_survives_verbatim() -> None:
    """``docs/index.md``, ``docs/operators.md`` and ``conf.py`` deep-link this.

    They link ``tour.md#graph-assembly``, and that anchor is the *slug of the
    heading text* -- ``myst_heading_anchors = 3``. An explicit MyST target,
    ``(graph-assembly)=``, does NOT satisfy such a link: measured, a build with
    the target and a reworded heading still emits three
    ``local id not found in doc 'tour': 'graph-assembly'`` warnings. So the
    heading has to stay exactly these two words.

    myst-parser 5.1.0 does warn on a dead fragment, so ``sphinx -n`` shows the
    damage -- but warnings do not fail the build and nothing reads the log, so
    the broken links would ship regardless.
    """
    assert "\n## Graph assembly\n" in TOUR.read_text(), (
        "tour.md's '## Graph assembly' heading has been reworded or removed. Its "
        "slug is the anchor that docs/index.md, docs/operators.md and docs/conf.py "
        "link to; rewording it dead-links all three with only a build warning."
    )
