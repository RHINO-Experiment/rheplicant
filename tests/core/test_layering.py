"""Guard the architectural hard rule: rheplicant.core is domain-agnostic.

DESIGN.md: "rheplicant.core never imports from radio or inference. If the
framework proves reusable beyond radio astronomy, core graduates to its own
package by moving one directory." This test makes that promise mechanical.
"""

import pathlib

import rheplicant.core

CORE_DIR = pathlib.Path(rheplicant.core.__file__).parent
# Match actual import statements, not prose mentions in docstrings.
FORBIDDEN = (
    "from rheplicant.radio",
    "import rheplicant.radio",
    "from rheplicant.inference",
    "import rheplicant.inference",
    # config assembles core objects from a document; core knows nothing about
    # documents. Without this entry the direction is a convention, and the
    # glob below never leaves core/, so a core -> config import would be
    # silent -- and DESIGN.md's "core graduates by moving one directory"
    # would quietly stop being true.
    "from rheplicant.config",
    "import rheplicant.config",
)


def test_core_never_imports_domain_layers():
    offenders = []
    for path in CORE_DIR.glob("*.py"):
        source = path.read_text()
        for banned in FORBIDDEN:
            if banned in source:
                offenders.append(f"{path.name}: contains {banned!r}")
    assert not offenders, (
        "rheplicant.core must stay domain-agnostic (extractable), but:\n"
        + "\n".join(offenders)
    )
