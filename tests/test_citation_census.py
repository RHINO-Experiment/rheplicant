"""How many ``file.py:<line>`` citations there are, and how many cannot be followed.

``tests/test_line_citations.py`` does the strong thing for three citations: it
pins a string the cited range must still contain, so the number has to reach
the claim. That is the right check and it does not scale to 729.

This is the census beside it. It counts, and it ratchets: the totals below may
FALL and may not rise. A citation migrated to ``path.py::qualname`` or to an
anchor lowers a number here and nothing else has to be edited; a new
``file.py:<line>`` written into the tree raises one and turns this red, which
is the moment to write the durable form instead.

**What is counted, and what was measured wrong three times before it was
right.** A citation names a path SUFFIX, and it must be resolved as one.
Matching on the basename alone says a citation of ``core/graph.py`` at line
350 points at ``config/graph.py``, which has 224 lines, and reports a
perfectly good citation as past end of file -- 46 of them, on the first run
of this scan.
Indexing only ``src/`` and ``tests/`` says the seventeen citations of
``examples/`` name files that do not exist. And an index built from
``ROOT.rglob`` picks up the checked-out worktree under ``.claude/`` and makes
every basename ambiguous. The numbers below are after all three were fixed;
the measurement, not the code, was the thing that kept being broken.

So the objective defects are:

* **unresolvable** -- no file in the tree has that path suffix. Zero today,
  and asserted at zero, because a new one is a plain mistake.
* **past end of file** -- the line does not exist. Zero today, same rule.
* **ambiguous** -- the suffix matches more than one file, so a reader cannot
  tell which is meant. 106 today, and this is the real debt: a bare
  ``noise.py`` matches five files in this tree.

Staleness in the stronger sense -- the line exists but no longer holds what
the citation claims -- is not decidable here, because only the citing sentence
knows what it claimed. That is what ``test_line_citations.py`` does by hand,
and why the migration target is a name rather than a number.

Historical documents under ``docs/superpowers/`` are excluded, on the same
grounds that file states: a dated plan said what it said, and rewriting line
numbers inside one would falsify the record rather than maintain it.
"""

from __future__ import annotations

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]

#: Trees whose text is LIVE -- maintained prose and code, where a citation is a
#: promise to a future reader.
CITING_ROOTS = ("src", "tests", "docs")
CITING_FILES = ("README.md", "DESIGN.md", "CLAUDE.md", "AGENTS.md")

#: Never scanned, and never used to resolve. ``.claude/`` holds a checked-out
#: worktree: a full second copy of the package, which makes every basename
#: ambiguous if it is indexed.
SKIP = (
    "docs/superpowers/", ".agents/", ".venv/", "node_modules/", "site/",
    "examples/TRIS/", "runs/", ".claude/", ".git/", "tools/",
)

#: Citations INTO a dependency. Their line numbers are that project's business
#: and move with its releases, not with this tree.
THIRD_PARTY = ("equinox/",)

CITATION = re.compile(r"\b([\w/]+\.py):(\d+)(?:-(\d+))?\b")

#: The ratchets. Measured 2026-09-20. Lower them when citations are migrated;
#: raising one is the change this file exists to make visible.
TOTAL_CEILING = 729
AMBIGUOUS_CEILING = 106


def _skipped(path: pathlib.Path) -> bool:
    text = str(path)
    return any(part in text for part in SKIP)


def _candidates() -> list[pathlib.Path]:
    """Every file a citation could be naming."""
    found = []
    for root in ("src", "tests", "examples"):
        for path in (ROOT / root).rglob("*.py"):
            if not _skipped(path):
                found.append(path)
    return found


def _citations() -> list[tuple[str, str, int, int]]:
    """``(citing file, cited path, first line, last line)`` over live text."""
    sources: list[pathlib.Path] = []
    for root in CITING_ROOTS:
        sources += [
            path for path in (ROOT / root).rglob("*")
            if path.is_file() and path.suffix in (".py", ".md")
        ]
    sources += [ROOT / name for name in CITING_FILES]

    found = []
    for path in sources:
        if _skipped(path) or not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for match in CITATION.finditer(text):
            target = match.group(1)
            if any(part in target for part in THIRD_PARTY):
                continue
            first = int(match.group(2))
            last = int(match.group(3)) if match.group(3) else first
            found.append((str(path.relative_to(ROOT)), target, first, last))
    return found


def _resolve(target: str, candidates: list[pathlib.Path]) -> list[pathlib.Path]:
    """Files whose path ends with the cited suffix.

    By SUFFIX, so ``core/graph.py`` cannot match ``config/graph.py``, and a
    bare ``graph.py`` matches both -- which is the ambiguity this file counts
    rather than resolves.
    """
    parts = tuple(target.split("/"))
    return [path for path in candidates if path.parts[-len(parts):] == parts]


def test_no_citation_names_a_file_that_is_not_there():
    """Zero today, and a new one is a plain mistake rather than a debt."""
    candidates = _candidates()
    missing = [
        (where, target)
        for where, target, _, _ in _citations()
        if not _resolve(target, candidates)
    ]
    assert not missing, (
        f"these citations name a path no file in the tree ends with: {missing}"
    )


def test_no_citation_points_past_the_end_of_its_file():
    """Also zero today. A line that does not exist cannot be followed at all."""
    candidates = _candidates()
    past = []
    for where, target, _, last in _citations():
        hits = _resolve(target, candidates)
        if len(hits) != 1:
            continue
        lines = len(hits[0].read_text(encoding="utf-8").splitlines())
        if last > lines:
            past.append((where, f"{target}:{last}", f"{hits[0]} has {lines} lines"))
    assert not past, f"these citations point past the end of the file: {past}"


def test_the_ambiguous_citation_count_only_ever_falls():
    """The real debt: a suffix matching several files names none of them.

    A bare ``noise.py`` matches five files in this tree, so a reader
    following a line number against it has to guess, and the guess is not
    recorded anywhere. The fix per citation
    is to lengthen the path until it is unique, or to migrate it to a name --
    both lower this number.
    """
    candidates = _candidates()
    ambiguous = [
        (where, target, len(hits))
        for where, target, _, _ in _citations()
        if len(hits := _resolve(target, candidates)) > 1
    ]
    assert len(ambiguous) <= AMBIGUOUS_CEILING, (
        f"{len(ambiguous)} citations are ambiguous and the recorded ceiling is "
        f"{AMBIGUOUS_CEILING}. A new one means a reader cannot follow it; "
        "lengthen the path or cite a name"
    )


def test_the_total_citation_count_only_ever_falls():
    """The migration ratchet.

    Line numbers rot on the next edit above them, which is why the target is
    ``path.py::qualname`` or an anchor. This number falling is the migration
    happening; it rising is a new citation written in the form being retired.
    """
    total = len(_citations())
    assert total <= TOTAL_CEILING, (
        f"{total} line citations, and the recorded ceiling is {TOTAL_CEILING}. "
        "A line number rots on the next edit above it -- cite "
        "`path.py::qualname` or a heading anchor instead"
    )


def test_the_ceilings_are_not_far_above_the_truth():
    """A ratchet nobody lowers stops being one.

    If a ceiling sits well above the measurement, the next twenty regressions
    fit underneath it and the guard reports nothing. This fails when either
    number has been left more than ten above what the tree actually has, which
    forces the ceiling down in the commit that does the migrating.
    """
    candidates = _candidates()
    citations = _citations()
    ambiguous = sum(1 for _, target, _, _ in citations
                    if len(_resolve(target, candidates)) > 1)
    assert TOTAL_CEILING - len(citations) <= 10, (
        f"TOTAL_CEILING is {TOTAL_CEILING} and the tree has {len(citations)}; "
        "lower the ceiling to the measurement"
    )
    assert AMBIGUOUS_CEILING - ambiguous <= 10, (
        f"AMBIGUOUS_CEILING is {AMBIGUOUS_CEILING} and the tree has "
        f"{ambiguous}; lower the ceiling to the measurement"
    )
