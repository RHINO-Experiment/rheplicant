"""How many ``file.py:<line>`` citations there are, and how many cannot be followed.

``tests/test_line_citations.py`` does the strong thing for three citations: it
pins a string the cited range must still contain, so the number has to reach
the claim. That is the right check and it does not scale to 729.

This is the census beside it. It counts, and it ratchets: the totals below may
FALL and may not rise. A citation migrated to ``path.py::qualname`` or to an
anchor lowers a number here and nothing else has to be edited; a new
``file.py:<line>`` written into the tree raises one and turns this red, which
is the moment to write the durable form instead.

**What is counted, and what was measured wrong four times before it was
right.** A citation names a path SUFFIX, and it must be resolved as one.
Matching on the basename alone says a citation of ``core/graph.py`` at line
350 points at ``config/graph.py``, which has 224 lines, and reports a
perfectly good citation as past end of file -- 46 of them, on the first run
of this scan.
Indexing only ``src/`` and ``tests/`` says the seventeen citations of
``examples/`` name files that do not exist. And an index built from
``ROOT.rglob`` picks up the checked-out worktree under ``.claude/`` and makes
every basename ambiguous. And ``SKIP`` was matched against the ABSOLUTE
path, while a worktree lives at ``.claude/worktrees/<name>/``: from one, all
698 sources and every candidate were skipped, and each check here passed over
an empty tree except the one with a floor. The numbers below are after all
four were fixed; the measurement, not the code, was the thing that kept being
broken.

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

import ast
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]

#: Trees whose text is LIVE -- maintained prose and code, where a citation is a
#: promise to a future reader.
CITING_ROOTS = ("src", "tests", "docs")
CITING_FILES = ("README.md", "DESIGN.md", "CLAUDE.md", "AGENTS.md")

#: Never scanned, and never used to resolve. ``.claude/`` holds a checked-out
#: worktree: a full second copy of the package, which makes every basename
#: ambiguous if it is indexed. Each entry is a run of directories below the
#: checkout's root; see :func:`_skipped`.
SKIP = (
    "docs/superpowers/",
    ".agents/",
    ".venv/",
    "node_modules/",
    "site/",
    "examples/TRIS/",
    "runs/",
    ".claude/",
    ".git/",
    "tools/",
)

_SKIP_RUNS = tuple(tuple(entry.strip("/").split("/")) for entry in SKIP)

#: Citations INTO a dependency. Their line numbers are that project's business
#: and move with its releases, not with this tree; so are their names.
THIRD_PARTY = ("equinox/", "utils/utils.py", "cal/utils/utils.py")

#: Names written to SHOW the citation form, in the docstrings of the guards
#: that enforce it. They are examples of a shape, not references, and a guard
#: that flagged its own illustration would be unwriteable.
PLACEHOLDERS = frozenset(
    {
        "file.py",
        "path.py",
        "module.py",
        "some_file.py",
    }
)

CITATION = re.compile(r"\b([\w/]+\.py):(\d+)(?:-(\d+))?\b")


def _prose_only(path: pathlib.Path, text: str) -> str:
    """The file with its RUNTIME strings blanked out.

    A citation inside a refusal message is not this file's business, and the
    distinction is not a convenience. A message is user-facing text pinned
    VERBATIM by ``test_config_preflight.py::TestNoMovedMessageWasReworded``,
    whose whole purpose is that nobody reword one casually; the 2026-09-20
    migration rewrote six of them and that guard caught it, correctly.

    So the two rules apply to two kinds of text. Prose -- docstrings, comments
    and markdown -- migrates to names and may carry no line numbers. A runtime
    message keeps what it shipped with, and changing one is a stop-and-ask.
    Twenty-one citations in eleven modules still carry a line number for that
    reason (counted 2026-09-20). Migrating them is a change to what a document
    author reads and is the user's call, not this file's.
    """
    if path.suffix != ".py":
        return text
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return text
    docstrings = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (
            isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and body
        ):
            first = body[0]
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                docstrings.add(id(first.value))
    lines = text.split("\n")
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        if id(node) in docstrings or node.end_lineno is None:
            continue
        for number in range(node.lineno - 1, node.end_lineno):
            lines[number] = " " * len(lines[number])
    return "\n".join(lines)


#: The ratchets. **Both reached ZERO on 2026-09-20**, which is what the
#: migration below was for; they stay as a floor, so a new ``file.py:<line>``
#: written into the tree turns this red rather than starting the debt again.
TOTAL_CEILING = 0
AMBIGUOUS_CEILING = 0

#: A citation in the form the migration produced: a path suffix, ``::``, and
#: the qualified name of what it points at.
NAME_CITATION = re.compile(r"\b([\w/]+\.py)::([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)")


def _skipped(path: pathlib.Path, root: pathlib.Path = ROOT) -> bool:
    """Whether ``path`` lies in a ``SKIP`` directory below ``root``.

    Only the part of the path below the root is read. The whole absolute path
    was read until 2026-09-24, so a checkout whose own location contained
    ``.claude/``, ``runs/``, ``site/`` or ``tools/`` skipped every file it
    had, and every worktree's location contains ``.claude/``. It is compared
    by directory, not by substring, because ``site/`` is also the end of
    ``composite/``.
    """
    folders = path.relative_to(root).parts[:-1]
    return any(
        folders[start : start + len(run)] == run
        for run in _SKIP_RUNS
        for start in range(len(folders) - len(run) + 1)
    )


def _candidates(root: pathlib.Path = ROOT) -> list[pathlib.Path]:
    """Every file a citation could be naming."""
    return [
        path
        for top in ("src", "tests", "examples")
        for path in (root / top).rglob("*.py")
        if not _skipped(path, root)
    ]


def _sources(root: pathlib.Path = ROOT) -> list[pathlib.Path]:
    """Every file whose text is live prose or code."""
    found = [
        path
        for top in CITING_ROOTS
        for path in (root / top).rglob("*")
        if path.is_file() and path.suffix in (".py", ".md")
    ]
    found += [root / name for name in CITING_FILES if (root / name).exists()]
    return [path for path in found if not _skipped(path, root)]


def _citations() -> list[tuple[str, str, int, int]]:
    """``(citing file, cited path, first line, last line)`` over live text."""
    found = []
    for path in _sources():
        text = _prose_only(path, path.read_text(encoding="utf-8", errors="replace"))
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
    return [path for path in candidates if path.parts[-len(parts) :] == parts]


def test_no_citation_names_a_file_that_is_not_there():
    """Zero today, and a new one is a plain mistake rather than a debt."""
    candidates = _candidates()
    missing = [
        (where, target) for where, target, _, _ in _citations() if not _resolve(target, candidates)
    ]
    assert not missing, f"these citations name a path no file in the tree ends with: {missing}"


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
    ambiguous = sum(1 for _, target, _, _ in citations if len(_resolve(target, candidates)) > 1)
    assert TOTAL_CEILING - len(citations) <= 10, (
        f"TOTAL_CEILING is {TOTAL_CEILING} and the tree has {len(citations)}; "
        "lower the ceiling to the measurement"
    )
    assert AMBIGUOUS_CEILING - ambiguous <= 10, (
        f"AMBIGUOUS_CEILING is {AMBIGUOUS_CEILING} and the tree has "
        f"{ambiguous}; lower the ceiling to the measurement"
    )


def _name_citations() -> list[tuple[str, str, str]]:
    """``(citing file, cited path, qualname)`` over live text."""
    found = []
    for path in _sources():
        text = _prose_only(path, path.read_text(encoding="utf-8", errors="replace"))
        for match in NAME_CITATION.finditer(text):
            target = match.group(1)
            if target in PLACEHOLDERS or any(part in target for part in THIRD_PARTY):
                continue
            found.append((str(path.relative_to(ROOT)), target, match.group(2)))
    return found


def _defined_names(path: pathlib.Path) -> set[str]:
    """Every qualified name a module defines, plus its module-level bindings."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return set()
    found: set[str] = set()

    def walk(node, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = f"{prefix}{child.name}"
                found.add(name)
                walk(child, f"{name}.")
            else:
                walk(child, prefix)

    walk(tree, "")
    # A LEAF name counts when the module has exactly one definition with it.
    # `file.py::TestThing.test_case` and `file.py::test_case` name the same
    # thing when nothing else in the file is called `test_case`, and the
    # shorter one is what fits on a line -- several of these citations are
    # over a hundred characters with the class in front.
    leaves: dict[str, int] = {}
    for name in list(found):
        leaves[name.split(".")[-1]] = leaves.get(name.split(".")[-1], 0) + 1
    found.update(leaf for leaf, count in leaves.items() if count == 1)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            found.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            found.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            # A name a module IMPORTS is a name a citation may mean. Several
            # sentences here say "`document.py` and `validation.py` import
            # `_same_value` from here", and the citation is about the name as
            # those modules use it.
            found.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
    return found


def test_the_tree_still_cites_by_name():
    """Guard the guard: the checks below would pass over an empty corpus."""
    assert len(_name_citations()) > 600, len(_name_citations())


def _elsewhere(tmp_path: pathlib.Path) -> pathlib.Path:
    """A checkout location whose own path passes through every ``SKIP`` entry."""
    return tmp_path.joinpath(*(part for run in _SKIP_RUNS for part in run), "checkout")


def test_a_skip_directory_above_the_root_skips_nothing(tmp_path):
    """Where the checkout sits is not part of the tree.

    A worktree lives at ``.claude/worktrees/<name>/``, and a checkout may sit
    under a ``runs/`` or ``tools/`` of its own. Below the root, an entry is
    skipped wherever it appears, and a directory that merely ends with one is
    not.
    """
    root = _elsewhere(tmp_path)
    for kept in ("README.md", "src/rheplicant/core/graph.py", "src/rheplicant/composite/graph.py"):
        assert not _skipped(root / kept, root), kept
    for entry in SKIP:
        assert _skipped(root / entry / "graph.py", root), entry
        assert _skipped(root / "src" / entry / "graph.py", root), entry


def test_the_census_reads_the_same_files_from_any_checkout(tmp_path):
    """The file set belongs to the tree, not to where it is checked out.

    The same checkout is reached a second time through a link whose own path
    passes through every ``SKIP`` entry. Both routes must find the same files,
    and some. From a worktree, the absolute-path rule found none, and every
    check in this file but the floor above passed over an empty tree.
    """
    elsewhere = _elsewhere(tmp_path)
    elsewhere.parent.mkdir(parents=True)
    elsewhere.symlink_to(ROOT, target_is_directory=True)
    for collect in (_sources, _candidates):
        here = {path.relative_to(ROOT) for path in collect()}
        there = {path.relative_to(elsewhere) for path in collect(elsewhere)}
        assert here, f"{collect.__name__} found nothing under {ROOT}"
        assert here == there, (collect.__name__, sorted(map(str, here ^ there))[:10])


def test_every_named_citation_resolves_to_one_file():
    """The ambiguity the line form could only COUNT, this form cannot have.

    A name is useless without a file, so the path half still has to be
    unambiguous -- and unlike a line number, a reader following one can tell
    whether they arrived.
    """
    candidates = _candidates()
    bad = [
        (where, f"{target}::{name}", len(_resolve(target, candidates)))
        for where, target, name in _name_citations()
        if len(_resolve(target, candidates)) != 1
    ]
    assert not bad, f"these citations name no file, or several: {bad}"


def test_every_named_citation_names_something_that_is_there():
    """**The check a line number could never support.**

    A line citation is true of any file long enough to have that line; there is no way to ask
    whether it still reaches what the sentence claimed, which is why 69 % of
    them were stale and nothing said so. ``file.py::qualname`` is a claim about
    the file's CONTENTS, so it can be checked on every run -- and a rename now
    fails here instead of quietly pointing at whatever moved into that line.
    """
    candidates = _candidates()
    missing = []
    for where, target, name in _name_citations():
        hits = _resolve(target, candidates)
        if len(hits) != 1:
            continue
        if name not in _defined_names(hits[0]):
            missing.append((where, f"{target}::{name}"))
    assert not missing, f"these citations name something their file does not define: {missing}"
