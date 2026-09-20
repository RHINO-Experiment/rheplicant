"""Bootstrap imports must remain independent of the scientific stack."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).parents[2] / "src"
PACKAGE = "_rheplicant_bootstrap"
#: Top-level names no bootstrap module may pull in. ``rheplicant`` covers every
#: ``rheplicant.`` submodule, because the census compares the first dotted
#: component; ``_rheplicant_bootstrap`` itself does not match it.
SCIENTIFIC_STACK = frozenset({"jax", "jaxlib", "equinox", "numpyro", "bayesmith", "rheplicant"})
IMPORT_CENSUS_PROGRAM = (
    "import _rheplicant_bootstrap; "
    "import json, sys; "
    "print(json.dumps(sorted({name.split('.')[0] for name in sys.modules})))"
)


def _bootstrap_modules():
    """Every module of the bootstrap package, read from the source tree.

    ``__main__`` is left out because importing it runs the CLI.
    """
    names = []
    for path in sorted((SRC / PACKAGE).rglob("*.py")):
        parts = path.relative_to(SRC).with_suffix("").parts
        if parts[-1] == "__main__":
            continue
        if parts[-1] == "__init__":
            parts = parts[:-1]
        names.append(".".join(parts))
    return names


BOOTSTRAP_MODULES = _bootstrap_modules()


def _run(program):
    """Run ``program`` in a fresh isolated interpreter with ``src`` first on
    the path. ``-I`` ignores every ``PYTHON*`` variable, including
    ``PYTHONDONTWRITEBYTECODE``, so ``-B`` is passed explicitly: without it
    each subprocess writes ``__pycache__`` into ``src/_rheplicant_bootstrap``,
    the stale-bytecode hazard CLAUDE.md records for mutation runs."""
    return subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            f"import sys; sys.path.insert(0, {str(SRC)!r})\n" + program,
        ],
        check=True,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )


def test_importing_bootstrap_does_not_import_the_scientific_stack():
    imported = set(json.loads(_run(IMPORT_CENSUS_PROGRAM).stdout))
    assert imported.isdisjoint(SCIENTIFIC_STACK)


def test_the_module_census_found_the_package():
    """The census below is parametrized from a directory walk, and a walk of
    the wrong directory would collect no cases and pass."""
    assert PACKAGE in BOOTSTRAP_MODULES
    assert f"{PACKAGE}.cli" in BOOTSTRAP_MODULES
    assert f"{PACKAGE}.output.manager" in BOOTSTRAP_MODULES


@pytest.mark.parametrize("module", BOOTSTRAP_MODULES)
def test_each_bootstrap_module_imports_without_the_scientific_stack(module):
    """Importing the package alone loads few of its modules, so each module is
    imported by itself in a fresh interpreter."""
    program = (
        "import importlib, json\n"
        f"importlib.import_module({module!r})\n"
        "print(json.dumps(sorted(sys.modules)))"
    )
    loaded = json.loads(_run(program).stdout)
    reached = sorted({name.split(".")[0] for name in loaded} & SCIENTIFIC_STACK)
    assert not reached, f"importing {module} imports {reached}"


def test_bootstrap_main_imports_cli_only_when_called():
    """The CLI module is absent after ``import _rheplicant_bootstrap`` and
    present once ``main`` runs. The second half shows the module name is the
    right one: this case used to assert that ``rheplicant.cli``, a module that
    has never existed, was absent, and that could not fail."""
    program = (
        "import contextlib, io\n"
        "import _rheplicant_bootstrap\n"
        "before = '_rheplicant_bootstrap.cli' in sys.modules\n"
        "with contextlib.redirect_stdout(io.StringIO()):\n"
        "    try:\n"
        "        _rheplicant_bootstrap.main(['--help'])\n"
        "    except SystemExit:\n"
        "        pass\n"
        "print(before, '_rheplicant_bootstrap.cli' in sys.modules)"
    )
    before, after = _run(program).stdout.split()
    assert before == "False", "import _rheplicant_bootstrap imported the CLI eagerly"
    assert after == "True", "main() did not import _rheplicant_bootstrap.cli"
