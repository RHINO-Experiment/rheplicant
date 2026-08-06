"""Cross-page links in the documentation resolve, and the D-range is not stale.

Both of these were drifting when the checks were written, and neither drift is
visible without looking.

**Anchors.** ``docs/conf.py`` emitted ``tour.md#4-graph-assembly`` for a heading
that had been renamed to ``Graph assembly``; the link had been dead for as long
as the rename. myst-parser 5.1.0 *does* warn (``myst.xref_missing``), but the
project builds with ``sphinx -n``, not ``-W``, and ``.readthedocs.yaml`` sets no
``fail_on_warning`` — so nothing failed and nobody read the log. This test does
the check without a build, in milliseconds, by slugifying every heading the way
``myst_heading_anchors`` does and looking each ``](page.md#anchor)`` up.

It covers ``conf.py`` too, because ``docs/signal-path.md`` is not a source file:
it is written at build time from a string literal in ``conf.py`` and is
gitignored, so a dead link inside it is invisible to a reader of ``docs/*.md``
and to ``git diff`` alike. That is exactly where the dead anchor was.

**The D range.** ``README.md`` advertised "Design decisions D1–D28" while
``DESIGN.md`` had reached D31 — the same failure ``test_readme_counts.py`` was
written to close for the test count, in a second place with no guard.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

#: ``myst_heading_anchors = 3`` in docs/conf.py -- only these levels get ids.
_MAX_ANCHOR_LEVEL = 3

#: A markdown link to another page with a fragment: ``](page.md#anchor)``.
_PAGE_ANCHOR = re.compile(r"\]\((?!https?:)([\w./-]+\.md)#([\w-]+)\)")

#: Pages that are generated rather than committed, and so are not on disk to
#: read headings from. Links INTO them cannot be checked; links FROM them can,
#: because their source text lives in conf.py.
_GENERATED = {"signal-path.md"}


def _slugs(text: str) -> set[str]:
    """The anchor a ``](page.md#fragment)`` link can actually reach.

    Two id schemes exist on every section and they genuinely disagree -- for
    ``Instrument (trunk order = graph order)`` myst emits a double hyphen where
    docutils emits one; for ``D17 - ... branch's own stage, and f_sky ...``
    docutils gives ``branch-s`` / ``f-sky`` where myst gives ``branchs`` /
    ``f_sky``. Only **myst's** is resolvable: its cross-reference resolver looks
    the fragment up in the anchors ``myst_heading_anchors`` registered, and
    reports ``local id not found in doc`` for anything else -- even when the
    docutils id is sitting right there in the emitted HTML. Measured: nine links
    written against docutils ids built cleanly into ``design.html`` and every one
    of them warned.

    The upstream function is imported rather than reimplemented, because a
    hand-rolled slugifier is a third scheme that agrees with neither.
    """
    from mdit_py_plugins.anchors.index import slugify

    return {slugify(re.sub(r"[`*]", "", text).strip())}


def _anchors_of(path: Path) -> set[str]:
    """Every anchor a page offers.

    ``{include}`` is deliberately NOT followed. ``docs/design.md`` and
    ``docs/changelog.md`` are two-line ``{include}`` stubs, and myst does not
    register heading anchors for included content: the built ``design.html``
    carries docutils ids for all 31 D-decisions and not one myst anchor, so
    ``design.md#d15-...`` cannot resolve however it is spelled. Following the
    include would make this test bless links that Sphinx rejects -- which is the
    mistake it caught the first time it was run. Link such pages without a
    fragment.
    """
    anchors: set[str] = set()
    in_fence = False
    for line in path.read_text().splitlines():
        if line.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        heading = re.match(r"^(#{1,6})\s+(.*)$", line)
        if heading and len(heading.group(1)) <= _MAX_ANCHOR_LEVEL:
            anchors |= _slugs(heading.group(2))
        target_line = re.match(r"^\((\S+)\)=\s*$", line)
        if target_line:
            anchors.add(target_line.group(1))
    return anchors


def _sources() -> list[tuple[str, str]]:
    """``(label, text)`` for every place a docs link can be written."""
    pages = [(p.name, p.read_text()) for p in sorted(DOCS.glob("*.md"))]
    pages.append(("conf.py", (DOCS / "conf.py").read_text()))
    return pages


def _link_targets() -> list[tuple[str, str, str]]:
    out = []
    for label, text in _sources():
        for page, anchor in _PAGE_ANCHOR.findall(text):
            out.append((label, page, anchor))
    return out


def test_there_are_links_to_check() -> None:
    """A regex that silently stops matching would make every check below pass."""
    targets = _link_targets()
    assert len(targets) >= 10, (
        f"Only {len(targets)} '](page.md#anchor)' links found across docs/*.md and "
        "conf.py; the link pattern has probably stopped matching."
    )


@pytest.mark.parametrize(("source", "page", "anchor"), _link_targets())
def test_cross_page_anchor_resolves(source: str, page: str, anchor: str) -> None:
    """Every ``](page.md#anchor)`` points at a heading that exists."""
    if page in _GENERATED:
        pytest.skip(f"{page} is written at build time by conf.py; not on disk to read")
    target = DOCS / page
    assert target.exists(), f"{source} links to {page}, which does not exist"

    anchors = _anchors_of(target)
    assert anchor in anchors, (
        f"{source} links to {page}#{anchor}, but that page has no such anchor.\n"
        f"myst generates anchors from heading text, so rewording a heading breaks "
        f"every inbound link to it with only a build warning.\n"
        f"Closest available: {sorted(a for a in anchors if anchor.split('-')[0] in a)[:5]}"
    )


def test_readme_d_range_matches_design() -> None:
    """``README.md``'s "D1-DN" must name the last decision ``DESIGN.md`` has."""
    decisions = re.findall(r"^### D(\d+)", (ROOT / "DESIGN.md").read_text(), re.MULTILINE)
    assert decisions, "No '### DNN' decision headings found in DESIGN.md"
    last = max(int(d) for d in decisions)

    quoted = re.search(r"Design decisions D1[-–]D(\d+)", (ROOT / "README.md").read_text())
    assert quoted, "README.md no longer states a 'Design decisions D1-DN' range"
    assert int(quoted.group(1)) == last, (
        f"README.md says the decisions run to D{quoted.group(1)}, but DESIGN.md's last "
        f"is D{last}. Update the README rather than loosening this: the count drifted "
        "by three before anything checked it."
    )
