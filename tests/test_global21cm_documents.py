"""The global 21 cm example's six documents, from what git tracks.

``examples/global21cm/`` holds six YAML documents that ``rheplicant run``
executes, and rheplicant-agent checks this repository out by commit and runs
them as its acceptance. Until 0.9.1 the arrays they read were ignored by git,
and the step that writes them needs inputs that are not public, so a checkout
could validate none of the six.

Three things are held here:

* every ``file:`` a document names is tracked, which is asked of git and not
  of the disk, because on the machine that made the arrays the files exist
  either way;
* the arrays are the ones the kept scores were computed from:
  ``results/analysis/fom.json`` records a sha256 for each. That is asked of
  the working tree and of HEAD, because on the machine that regenerates the
  arrays the two can differ;
* ``rheplicant validate`` accepts each document. That needs the emulator the
  documents' hooks import, which ``examples/global21cm/requirements.txt``
  installs. Where it is absent the six cases skip, and it is absent on the CI
  runner, which does not install it.

The example has a suite of its own under ``examples/global21cm/tests``, which
this repository's ``testpaths`` does not collect. It tests the example. This
module tests that the engine still reads the example's documents.
"""

import functools
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from tests.test_tour_runs import _importable_in_a_fresh_interpreter

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "global21cm"
FULL = ("oracle", "beamconv", "physical")
DOCUMENTS = tuple(f"{name}{suffix}" for name in FULL for suffix in ("", "_quick"))

#: What validating a document imports beyond rheplicant, by document.
NEEDS = {name: ("global21cm_jax",) for name in DOCUMENTS}
NEEDS |= {name: ("global21cm_jax", "pygdsm") for name in ("physical", "physical_quick")}


def _files(node) -> set[str]:
    """Every ``file: {path: ...}`` under ``node``, as written."""
    found: set[str] = set()
    if isinstance(node, dict):
        entry = node.get("file")
        if isinstance(entry, dict) and "path" in entry:
            found.add(entry["path"])
        for value in node.values():
            found |= _files(value)
    elif isinstance(node, list):
        for value in node:
            found |= _files(value)
    return found


def _read(name: str) -> set[str]:
    return _files(yaml.safe_load((EXAMPLE / f"{name}.yaml").read_text()))


def _tracked(path: Path) -> bool:
    done = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(path.relative_to(ROOT))],
        cwd=ROOT,
        capture_output=True,
    )
    return done.returncode == 0


def _at_head(path: Path) -> bytes | None:
    """What HEAD holds for ``path``, or None where there is no HEAD to read."""
    if not (ROOT / ".git").exists() or shutil.which("git") is None:
        return None
    done = subprocess.run(
        ["git", "show", f"HEAD:{path.relative_to(ROOT).as_posix()}"],
        cwd=ROOT,
        capture_output=True,
    )
    return done.stdout if done.returncode == 0 else None


@functools.cache
def _installed(module: str) -> bool:
    return _importable_in_a_fresh_interpreter(module, cwd=ROOT)


def test_the_documents_read_sixteen_arrays_between_them() -> None:
    """The walk finds what the documents declare: a count, so it cannot go blind."""
    every = set().union(*(_read(name) for name in DOCUMENTS))
    assert len(every) == 16, sorted(every)
    assert {path.split("/")[1] for path in every} == {"sim", "sim_stress"}
    for name in FULL:
        assert _read(name) == _read(f"{name}_quick")


@pytest.mark.parametrize("name", FULL)
def test_every_file_a_document_names_is_tracked(name: str) -> None:
    if not (ROOT / ".git").exists() or shutil.which("git") is None:
        pytest.skip("not a git checkout, or no git to ask, so there is no index to read")
    missing = sorted(path for path in _read(name) if not _tracked(EXAMPLE / path))
    assert not missing, (
        f"{name}.yaml reads {missing}, which git does not track, so a checkout "
        "cannot validate the document. See examples/global21cm/.gitignore."
    )


@pytest.mark.parametrize("view", ["the working tree", "HEAD"])
@pytest.mark.parametrize("name", FULL)
def test_the_arrays_are_the_ones_the_kept_scores_were_made_from(name: str, view: str) -> None:
    read = Path.read_bytes if view == "the working tree" else _at_head
    record = read(EXAMPLE / "results" / "analysis" / "fom.json")
    if record is None:
        pytest.skip("no HEAD to read: not a git checkout, or fom.json is not in it")
    recorded = json.loads(record)["input_sha256"]
    for path in sorted(_read(name)):
        assert path in recorded, f"fom.json records no hash for {path}"
        held = read(EXAMPLE / path)
        assert held is not None, f"{path} is not in {view}"
        assert hashlib.sha256(held).hexdigest() == recorded[path], (
            f"{path} in {view} is not the file results/analysis/fom.json was computed from"
        )


@pytest.mark.parametrize("name", DOCUMENTS)
def test_the_document_validates(name: str) -> None:
    for module in NEEDS[name]:
        if not _installed(module):
            pytest.skip(
                f"{name}.yaml needs {module}; examples/global21cm/requirements.txt installs it"
            )
    inherited = os.environ.get("PYTHONPATH")
    path = str(ROOT / "examples") + (os.pathsep + inherited if inherited else "")
    done = subprocess.run(
        [sys.executable, "-m", "_rheplicant_bootstrap", "validate", str(EXAMPLE / f"{name}.yaml")],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": path},
        timeout=900,
    )
    assert done.returncode == 0, (
        f"rheplicant validate {name}.yaml exited {done.returncode}.\n\n"
        f"{done.stdout[-2000:]}\n{done.stderr[-4000:]}"
    )
    assert "configuration valid: base + 1 variants" in done.stdout
