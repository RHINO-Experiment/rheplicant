"""Bootstrap imports must remain independent of the scientific stack."""

import json
import os
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).parents[2] / "src"
IMPORT_CENSUS_PROGRAM = (
    "import _rheplicant_bootstrap; "
    "import json, sys; "
    "print(json.dumps(sorted({name.split('.')[0] for name in sys.modules})))"
)


def test_importing_bootstrap_does_not_import_the_scientific_stack():
    program = f"import sys; sys.path.insert(0, {str(SRC)!r}); " + IMPORT_CENSUS_PROGRAM
    completed = subprocess.run(
        [sys.executable, "-I", "-c", program],
        check=True,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    imported = set(json.loads(completed.stdout))
    assert imported.isdisjoint({"jax", "jaxlib", "equinox", "numpyro", "rheplicant"})


def test_bootstrap_main_imports_cli_only_when_called():
    program = f"import sys; sys.path.insert(0, {str(SRC)!r}); " + (
        "import _rheplicant_bootstrap; "
        "assert 'rheplicant.cli' not in sys.modules; "
        "print('ok')"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", program],
        check=True,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    assert completed.stdout.strip() == "ok"
