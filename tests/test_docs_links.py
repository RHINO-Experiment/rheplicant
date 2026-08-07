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

It covers ``conf.py`` too, because links can be written there: ``conf.py``
generates the artefacts ``docs/signal-path.md`` embeds, and for a long time it
generated the whole page, which is exactly where the dead anchor was hiding.

**The D range.** ``README.md`` advertised "Design decisions D1–D28" while
``DESIGN.md`` had reached D31 — the same failure ``test_readme_counts.py`` was
written to close for the test count, in a second place with no guard.
"""

import re
import xml.dom.minidom
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

#: ``myst_heading_anchors = 3`` in docs/conf.py -- only these levels get ids.
_MAX_ANCHOR_LEVEL = 3

#: A markdown link to another page with a fragment: ``](page.md#anchor)``.
_PAGE_ANCHOR = re.compile(r"\]\((?!https?:)([\w./-]+\.md)#([\w-]+)\)")

#: Pages generated rather than committed. Empty, and that is the point:
#: ``signal-path.md`` used to be here, written wholesale by ``conf.py`` into a
#: gitignored file, so links into it were exempt from this check. Its prose is a
#: source file now and only the diagram is generated, so the exemption is gone
#: and ``signal-path.md#rhinos-template`` is checked like any other link.
_GENERATED: frozenset[str] = frozenset()


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

    Only heading slugs count. Two things that look like anchors are deliberately
    NOT collected, because neither satisfies a ``](page.md#fragment)`` link:

    * **Explicit MyST targets**, ``(label)=``. Measured: ``tour.md`` carried
      ``(graph-assembly)=`` above a reworded heading and the build still emitted
      three ``local id not found in doc 'tour': 'graph-assembly'`` warnings.
      They work for ``{ref}`` roles, not for fragments.
    * **``{include}``d content.** ``docs/design.md`` and ``docs/changelog.md``
      are two-line stubs, and myst registers no heading anchors for what they
      pull in: the built ``design.html`` carries docutils ids for all 31
      D-decisions and not one myst anchor, so ``design.md#d15-...`` cannot
      resolve however it is spelled.

    Collecting either would make this test bless links that Sphinx rejects,
    which is the failure it caught the first time it ran. Link such pages
    without a fragment.
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


def test_every_documentation_page_is_tracked_by_git() -> None:
    """No documentation page may be a build artefact.

    ``docs/signal-path.md`` was one: ``conf.py`` wrote the whole page from a
    string literal and ``.gitignore`` hid the result. It carried the composition
    model, the four node kinds, the "template, not the framework" argument and
    the custom-graph sketch -- and none of that was in ``git diff``, in a grep,
    or on GitHub, where ``README.md``'s link to the page 404'd. The
    ``except ImportError`` fallback replaced the entire page with a two-line
    stub, so a build machine without the package shipped documentation with no
    explanation of Cascade/Sum/Switch at all.

    Generated *artefacts* are fine and stay ignored -- the mermaid diagram of the
    live graph, the two assembly renders. A generated *page* is not.
    """
    import subprocess

    tracked = subprocess.run(
        ["git", "ls-files", "docs/*.md"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout.split()
    tracked_names = {Path(p).name for p in tracked}
    on_disk = {p.name for p in DOCS.glob("*.md")}

    untracked = on_disk - tracked_names
    assert not untracked, (
        f"{sorted(untracked)} exist in docs/ but are not tracked by git. A page "
        "that is generated and ignored cannot be diffed, grepped, read on GitHub, "
        "or edited where it is read -- generate the figures, author the prose."
    )


def test_every_example_script_is_mentioned_in_the_docs() -> None:
    """A demo nothing links to is a demo nobody runs.

    Six of the thirteen were unreachable when this was written, and two of those
    six had been reachable until a rewrite of ``tour.md`` replaced the paragraph
    that named them -- which is the failure mode: not that anyone decides to hide
    a script, but that the one sentence pointing at it is collateral in an edit
    about something else. ``docs/examples.md`` is the index; this is what keeps
    it complete.
    """
    scripts = sorted(p.name for p in (ROOT / "examples").glob("*.py"))
    assert scripts, "No example scripts found -- has examples/ moved?"

    prose = "\n".join(
        p.read_text() for p in list(DOCS.glob("*.md")) + [ROOT / "README.md"]
    )
    missing = [s for s in scripts if s not in prose]
    assert not missing, (
        f"{missing} exist in examples/ but are named nowhere in docs/*.md or "
        "README.md. Add them to docs/examples.md, or delete them."
    )


def test_the_examples_page_states_the_real_count() -> None:
    """``examples.md`` and ``README.md`` both count the scripts in words."""
    n = len(list((ROOT / "examples").glob("*.py")))
    words = {
        10: "Ten", 11: "Eleven", 12: "Twelve", 13: "Thirteen",
        14: "Fourteen", 15: "Fifteen", 16: "Sixteen",
    }
    assert n in words, f"{n} example scripts -- extend the number words above"
    word = words[n]
    for path in (DOCS / "examples.md", ROOT / "README.md"):
        text = path.read_text()
        # Case-insensitive: the count reads "Fourteen scripts" at the start of
        # a sentence and "fourteen runnable scripts" inside one, and both are
        # the same claim. A case-sensitive match called the second one absent.
        stated = re.search(r"\b(Ten|Eleven|Twelve|Thirteen|Fourteen|Fifteen|Sixteen)\b"
                           r"[^.\n|]*(runnable|scripts|demos)", text, re.IGNORECASE)
        assert stated, f"{path.name} no longer states how many examples there are"
        assert stated.group(1).capitalize() == word, (
            f"{path.name} says '{stated.group(1)}' examples; there are {n} ({word})."
        )


def test_every_committed_svg_is_well_formed_xml() -> None:
    """A browser serves ``.svg`` as XML, and declines to render a malformed one.

    This exists because the failure is invisible to every cheaper check. XML
    forbids ``--`` inside a comment; a generator that wrote
    ``<!-- edit this -- not that -->`` produced a file that opens fine in an
    editor, diffs clean, passes a checksum comparison against its own
    regeneration, and renders as a broken-image icon on the page. It shipped in
    a commit whose figure had been looked at -- before the comment was written.

    Parsing is the whole test. Anything a browser will not parse fails here
    first, with the line and column.
    """
    broken = []
    for path in sorted((DOCS / "_static").glob("*.svg")):
        try:
            xml.dom.minidom.parse(str(path))
        except Exception as exc:  # noqa: BLE001 - any parse failure is the bug
            broken.append(f"{path.name}: {exc}")
    assert not broken, (
        "These committed SVGs are not well-formed XML, so a browser will not "
        "render them at all:\n  " + "\n  ".join(broken)
    )


def test_the_sidebar_keeps_furo_s_own_components_and_adds_ours() -> None:
    """Naming ``html_sidebars`` REPLACES furo's default list, silently.

    Inserting one row means writing out the whole list, and anything left off
    disappears with no warning -- the first draft of this dropped
    ``ethical-ads.html``, Read the Docs' ad slot, purely as a side effect. This
    pins both directions: our row is present, and none of furo's are missing.
    """
    conf = (DOCS / "conf.py").read_text()
    listed = set(re.findall(r'"(sidebar/[a-z-]+\.html)"', conf))
    furo_default = {
        "sidebar/scroll-start.html",
        "sidebar/brand.html",
        "sidebar/search.html",
        "sidebar/navigation.html",
        "sidebar/ethical-ads.html",
        "sidebar/scroll-end.html",
    }
    assert furo_default <= listed, (
        f"conf.py's html_sidebars drops {sorted(furo_default - listed)} from "
        "furo's default list. Naming the list replaces it, so a component left "
        "off is simply gone."
    )
    assert "sidebar/github.html" in listed
    assert (DOCS / "_templates" / "sidebar" / "github.html").is_file()
