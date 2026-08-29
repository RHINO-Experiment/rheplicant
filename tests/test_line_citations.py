"""A citation that names a line number rots the moment the file moves.

``src/`` and ``tests/`` cite ``reduced_basis.py:<line>`` eight times, and every
one of them names the same thing: the permutation bug that comes back whenever
a consumer zips a jacobian's own key order against a declared-order list. The
citations are load-bearing -- they are how a reader of
``config/sections/diagnostics.py`` finds out why comparison order is checked
there at all.

**Nothing checked that they still pointed at that code**, and on 2026-08-29 the
Wave C `reduced_basis` half-switch moved it: delegating two array-level helpers
that sat above ``score_directions`` shifted the block from 171-180 to 159-168.
Every citation would have gone on naming a line range that had become somebody
else's code, silently, and a reader following one would have found nothing and
concluded the comment had been deleted.

Finding them all was itself three passes. A grep for ``171-180`` restricted to
``*.py`` found six; widening to ``*.md`` found three more; and the citations
turned out to exist in five different spellings (``:114``, ``:114-119``,
``:114-120``, ``:164``, ``:171-180``). This test is the answer to that: the
table below must cover **every** citation the scan finds, so a new one added in
a new spelling fails here rather than joining the ones nobody knew about.

Historical documents under ``docs/superpowers/`` are deliberately NOT scanned.
A dated plan said what it said; rewriting line numbers inside one would falsify
the record rather than maintain it.
"""

import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
CITED = ROOT / "src" / "rheplicant" / "inference" / "reduced_basis.py"

#: Each live citation, and a string the cited range must contain. The point is
#: not the number -- it is that the number still reaches the claim.
EXPECTED = {
    "reduced_basis.py:102-108": "def score_directions",
    "reduced_basis.py:152": "selected = tuple(values)",
    "reduced_basis.py:159-168": "Iterate `selected`, never `jacobian.items()`",
}

#: Where a citation counts as LIVE. Historical plans and specs are excluded on
#: purpose; see the module docstring.
LIVE = ("src", "tests")

_CITATION = re.compile(r"reduced_basis\.py:(\d+)(?:-(\d+))?")


def _live_citations() -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for top in LIVE:
        for path in (ROOT / top).rglob("*.py"):
            if path.name == "test_line_citations.py":
                continue
            for match in _CITATION.finditer(path.read_text(encoding="utf-8")):
                found.setdefault(match.group(0), []).append(
                    str(path.relative_to(ROOT))
                )
    return found


@pytest.mark.parametrize("citation, expected", sorted(EXPECTED.items()))
def test_the_cited_range_still_contains_what_it_names(citation, expected):
    match = _CITATION.match(citation)
    first = int(match.group(1))
    last = int(match.group(2)) if match.group(2) else first
    lines = CITED.read_text(encoding="utf-8").split("\n")
    assert last <= len(lines), (citation, "past the end of the file", len(lines))
    excerpt = "\n".join(lines[first - 1 : last])
    assert expected in excerpt, (
        f"{citation} no longer contains {expected!r}. The file moved and the "
        "citations did not; find the new line range and update every live "
        "citation, not only the ones your first grep matched."
    )


def test_every_live_citation_is_in_the_table():
    """The half with teeth: a citation nobody listed is a citation nobody checks.

    Without this, the table above guards only the three spellings that happened
    to exist when it was written -- and the reason this file exists is that a
    grep for one spelling missed two others.
    """
    unlisted = set(_live_citations()) - set(EXPECTED)
    assert not unlisted, (
        "these citations are not in EXPECTED, so nothing verifies they point "
        "anywhere: " + repr({c: _live_citations()[c] for c in sorted(unlisted)})
    )


def test_the_check_would_notice_a_shift():
    """Anti-vacuity: prove the assertion can fail, since it is a containment.

    A range wide enough to contain the string no matter where it moved would
    pass for ever. The cited comment block is ten lines in an 800-line file, so
    an off-by-twelve -- which is exactly what the switch caused -- must fail.
    """
    lines = CITED.read_text(encoding="utf-8").split("\n")
    shifted = "\n".join(lines[171 - 1 : 180])
    assert "Iterate `selected`" not in shifted, (
        "the old range 171-180 still contains the comment, so this file's "
        "premise is wrong and the citations never moved"
    )
