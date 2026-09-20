"""The API reference covers exactly the frozen ``__all__`` lists.

Stage 4.2 asks for that in one sentence, and it was false in a way nobody
could see from either side. Measured 2026-09-20, before this file existed:
twenty modules defining ninety-five public names had no ``automodule``
directive -- the whole of ``rheplicant.gui``, the capability registry, the
document loader, the findings vocabulary. Every one of them is counted by
``docs/stability.md`` as part of the public surface, and none of them was in
the reference a reader is sent to.

Neither half could notice on its own. ``docs/stability.md`` counts names from
``__all__`` and never looks at the reference; the reference lists modules and
never looks at ``__all__``. This is the join.

Both directions are asserted. A module documented here that exports nothing
public is not an error -- a page may explain machinery -- so the reverse claim
is narrower and stated where it is made.
"""

from __future__ import annotations

import importlib
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "docs" / "api.md"

#: The six namespaces ``docs/stability.md`` counts. Named here rather than
#: discovered, because "what is public" is a decision and not a measurement.
NAMESPACES = (
    "rheplicant",
    "rheplicant.core",
    "rheplicant.radio",
    "rheplicant.inference",
    "rheplicant.config",
    "rheplicant.gui",
)


def documented_modules() -> set[str]:
    return set(re.findall(r"automodule:: ([\w.]+)", REFERENCE.read_text(encoding="utf-8")))


def home_of(namespace: str, name: str) -> str | None:
    """Which module defines a public name.

    ``__module__`` answers for a class or a function and is ``None`` for plain
    data -- ``ACCEPTED_UNITS`` is a frozenset, ``VALUE_FORMS`` a tuple -- so a
    census built on it alone silently skipped every module-level constant on
    the public surface. The fallback reads the namespace's OWN imports, which
    is where a re-exported constant's home is actually written down.
    """
    import ast

    module = importlib.import_module(namespace)
    declared = getattr(getattr(module, name), "__module__", None)
    if declared is not None:
        return declared
    source = pathlib.Path(module.__file__)
    tree = ast.parse(source.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if any(alias.asname == name or alias.name == name for alias in node.names):
                return node.module
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return namespace
    return None


def test_the_only_exempt_public_names_are_dunders():
    """``__version__`` is the whole exemption, and it is stated rather than
    silent.

    A version string has no signature to document and no docstring worth
    rendering; an API reference entry for it would be noise. Anything else
    beginning with a dunder appearing on the public surface is a question, so
    this fails rather than widening quietly.
    """
    exempt = {
        name
        for namespace in NAMESPACES
        for name in importlib.import_module(namespace).__all__
        if name.startswith("__")
    }
    assert exempt == {"__version__"}, sorted(exempt)


def test_the_reference_documents_something():
    """Guard the guard: an empty reference would satisfy a subset claim."""
    assert len(documented_modules()) > 50


@pytest.mark.parametrize("namespace", NAMESPACES)
def test_every_public_name_has_its_module_in_the_reference(namespace):
    documented = documented_modules()
    module = importlib.import_module(namespace)
    missing = sorted(
        {
            f"{home_of(namespace, name)}.{name}"
            for name in module.__all__
            if not name.startswith("__") and home_of(namespace, name) not in documented
        }
    )
    assert not missing, (
        f"{namespace} exports these from modules docs/api.md does not "
        f"document: {missing}. A name counted as public by "
        "docs/stability.md and absent from the reference is a promise with "
        "nowhere to read it"
    )


def test_the_bootstrap_appears_only_for_the_names_it_re_exports():
    """The one exception, and the shape that keeps it an exception.

    ``_rheplicant_bootstrap`` is private by name and by intent. Two names in
    ``rheplicant.config``'s public surface are defined there, so the reference
    has to reach them -- with an explicit ``:members:`` list, never a whole
    module, so documenting the private layer cannot happen by accident.
    """
    text = REFERENCE.read_text(encoding="utf-8")
    for match in re.finditer(r"automodule:: (_rheplicant_bootstrap[\w.]*)\n(.*?)\n\n", text, re.S):
        assert ":members:" in match.group(2), match.group(1)
        listed = re.search(r":members: (.+)", match.group(2))
        assert listed and listed.group(1).strip(), (
            f"{match.group(1)} is documented wholesale; name the re-exported members instead"
        )
