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


def _importable_in_a_fresh_interpreter(name: str, cwd: Path) -> bool:
    """Can the process that will run the script import this? Not: can we?

    ``find_spec`` in *this* process was the original check and it is the wrong
    process. ``tests/radio/test_ingestion_vs_reference.py`` used to put a
    rhino-cal checkout on ``sys.path`` at import time -- above its own
    ``importorskip``, so the insert happened even when that module then bowed out
    -- and such a checkout carries a top-level ``rhino_cal_jax``. Anyone with one
    therefore had a pytest process that could find a module the pasted script's
    interpreter could not: the skip below did not fire, the script ran, and it
    died on ``import rhino_cal_jax`` with a ``ModuleNotFoundError`` that said
    nothing about the tour and everything about another test file. It was
    order-dependent rather than random, which is why it survived a while:
    ``pytest tests/test_tour_runs.py`` skipped while
    ``pytest tests/radio/test_ingestion_vs_reference.py tests/test_tour_runs.py``
    failed, so every isolated re-run of the "failing" test passed.

    That import-time insert is gone -- it is scoped to a fixture now -- but the
    check stays asked of the right process, because the bug was never really the
    other file. Any future path manipulation, in any test or plugin or sitecustomize,
    reintroduces exactly this divergence, and only asking the interpreter that
    has to run the script is immune to all of them.

    So: a fresh interpreter, in the directory the script will run from, and by
    importing rather than locating -- a module that is findable but broken is not
    one the tour can use either.
    """
    try:
        probe = subprocess.run(
            [sys.executable, "-c", f"import importlib; importlib.import_module({name!r})"],
            capture_output=True,
            cwd=cwd,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        # An import that hangs -- a device scan, a network probe, a stale lock --
        # is not one the tour can rely on either, and the run below carries its
        # own timeout for the same reason. Answering "no" beats hanging the suite
        # on a question nobody asked.
        return False
    return probe.returncode == 0


def test_the_tour_is_a_script_that_runs(tmp_path: Path) -> None:
    """Paste every runnable python block into one file, in order, and run it."""
    blocks = _blocks()
    assert len(blocks) >= _MIN_BLOCKS, (
        f"Only {len(blocks)} python blocks found in {TOUR.name}; the fence pattern "
        "has probably stopped matching rather than the tour having shrunk."
    )

    for number, source in enumerate(blocks, start=1):
        for name in _NEEDS_EXTRA.findall(source):
            if not _importable_in_a_fresh_interpreter(name, cwd=tmp_path):
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


def test_a_module_only_this_process_can_see_does_not_count_as_installed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The regression for the order-dependent failure described in the helper.

    Another test file puts a checkout on ``sys.path``; this process can then
    import things the tour's subprocess cannot. Asking ``find_spec`` here said
    "installed", the skip did not fire, and the tour failed on an import. The
    fake package below reproduces that shape exactly -- visible here, invisible
    to a fresh interpreter -- and the check must call it absent.
    """
    polluted = tmp_path / "polluted"
    (polluted / "ghost_extra").mkdir(parents=True)
    (polluted / "ghost_extra" / "__init__.py").write_text("")
    monkeypatch.syspath_prepend(str(polluted))

    assert find_spec("ghost_extra") is not None, (
        "the fake package is not visible in this process, so this test is not "
        "reproducing the sys.path pollution it exists to pin"
    )
    assert not _importable_in_a_fresh_interpreter("ghost_extra", cwd=tmp_path), (
        "a module importable only because THIS process' sys.path was mutated is "
        "being counted as installed. The tour runs in a subprocess that cannot "
        "see it, so the run would fail with ModuleNotFoundError instead of "
        "skipping."
    )


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
