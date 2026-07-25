"""Guard the architectural hard rule: replicant.core is domain-agnostic.

DESIGN.md: "replicant.core never imports from radio or inference. If the
framework proves reusable beyond radio astronomy, core graduates to its own
package by moving one directory." This test makes that promise mechanical.
"""

import pathlib

import replicant.core

CORE_DIR = pathlib.Path(replicant.core.__file__).parent
# Match actual import statements, not prose mentions in docstrings.
FORBIDDEN = (
    "from replicant.radio",
    "import replicant.radio",
    "from replicant.inference",
    "import replicant.inference",
)


def test_core_never_imports_domain_layers():
    offenders = []
    for path in CORE_DIR.glob("*.py"):
        source = path.read_text()
        for banned in FORBIDDEN:
            if banned in source:
                offenders.append(f"{path.name}: contains {banned!r}")
    assert not offenders, (
        "replicant.core must stay domain-agnostic (extractable), but:\n"
        + "\n".join(offenders)
    )
