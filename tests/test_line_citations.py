"""Three citations pinned by CONTENT, now that they are pinned by name.

``tests/test_citation_census.py`` asks whether every ``file.py::qualname``
citation names something its file defines. That is the structural half, and it
is new: a line number was true of any file long enough to have that line, so
nothing could ask it.

This is the other half, for the handful of citations where the sentence makes
a claim about what is IN the thing it names. A name existing does not make the
sentence true; the definition still has to contain what the citation says it
does. Nothing decides that in general, which is why it is three by hand rather
than a rule.

**These were line citations until 2026-09-20 and the migration is why this
file changed shape.** The line citation for that claim was pinned against
``selected = tuple(values)``; the same claim is now made about
``score_directions``, the function that line sits in. The guarantee is the
same and it no longer moves when something above it does -- which was the
whole complaint: the citation "turned out to exist in five different
spellings" because five sentences had each recorded a different line for one
claim. There is one spelling now, and five sentences citing it agree by
construction.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]

#: ``(citation, what that definition must still contain)``. Keyed by the PAIR,
#: because three of these name one function and a dict keyed by the citation
#: would keep only the last -- which is exactly what happened when the
#: migration collapsed the three line ranges into one name, and is why the
#: shape of this table changed with them.
EXPECTED = (
    ("reduced_basis.py::score_directions", "def score_directions"),
    ("reduced_basis.py::score_directions", "selected = tuple(values)"),
    (
        "reduced_basis.py::score_directions",
        "Iterate `selected`, never `jacobian.items()`",
    ),
)

#: Where a citation counts as LIVE. Historical plans and specs are excluded on
#: purpose; see the module docstring of the census beside this one.
LIVE = ("src", "tests")


def _source_of(citation: str) -> str:
    """The source of the definition a citation names."""
    target, _, qualname = citation.partition("::")
    matches = [
        path
        for top in LIVE
        for path in (ROOT / top).rglob("*.py")
        if path.parts[-len(target.split("/")) :] == tuple(target.split("/"))
    ]
    assert len(matches) == 1, (citation, matches)
    text = matches[0].read_text(encoding="utf-8")
    tree = ast.parse(text)
    wanted = qualname.split(".")

    def find(node, names):
        for child in ast.iter_child_nodes(node):
            if (
                isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                and child.name == names[0]
            ):
                if len(names) == 1:
                    return child
                return find(child, names[1:])
        return None

    found = find(tree, wanted)
    assert found is not None, (citation, "names nothing in that file")
    lines = text.split("\n")
    start = min([found.lineno] + [d.lineno for d in found.decorator_list])
    return "\n".join(lines[start - 1 : found.end_lineno])


@pytest.mark.parametrize(("citation", "expected"), EXPECTED, ids=[row[1][:40] for row in EXPECTED])
def test_the_named_definition_still_contains_what_it_is_cited_for(citation, expected):
    assert expected in _source_of(citation), (
        f"{citation} no longer contains {expected!r}. Either the claim moved "
        "to another definition -- in which case the citations move with it -- "
        "or it stopped being true, in which case the sentence does."
    )


def test_every_pinned_citation_is_actually_written_somewhere():
    """A pin nobody cites is a pin guarding nothing.

    The three above are cited from five sentences in
    ``config/sections/diagnostics.py``. If they all stopped citing it, this
    file would go on passing about a claim no reader is being made.
    """
    cited = {
        citation
        for citation, _expected in EXPECTED
        for top in LIVE
        for path in (ROOT / top).rglob("*.py")
        if path.name != pathlib.Path(__file__).name and citation in path.read_text(encoding="utf-8")
    }
    assert cited == {citation for citation, _expected in EXPECTED}
