"""Prose that names something in the code must name something that is there.

Three documentation claims rotted in one session and nothing noticed. Two pages
said the whole-file loader had not shipped while ``load_document`` had, and
``CLAUDE.md`` told every reader to pass ``--no-cov`` long after that reversed.
``test_docs_links.py`` ran and passed throughout -- it reads links. The count
pins in ``test_config_surface.py`` and ``test_readme_counts.py`` ran and passed
too -- they read numbers. Nothing read a sentence.

Most of a sentence still cannot be read. What CAN be read is the part of it
that is a code name: ``Finding.check``, ``bias_tolerance=``,
``examples/gls_gcr.py``, ``tests/config/test_config_surface.py::TestX``. When
the code renames one of those, the sentence around it becomes false in a way
that is invisible to a reader and to every other guard here. That is the class
this module closes, and it is deliberately narrower than "the docs are true".

**Derived, never listed.** The vocabulary each check compares against is walked
out of ``src/`` and off the working tree at run time. There is no table of
approved names to fall behind the code, because a table is the thing that goes
stale next -- which is the defect this module exists to catch. Git is asked one
question only, and it is the right one for it: which pages SHIP. Whether a file
is THERE is asked of the disk, because a rename that has not been staged is
still a rename to the reader -- and a mutation run walked past the first draft,
which asked git both questions.

**AST, not text.** Members and parameters come from ``ast``. A regex over
``src/`` would match the name inside a docstring arguing that the name was
removed, which is how a first draft of ``test_coverage_instrument.py`` came to
match its own prose. Class bodies and ``def`` signatures are unambiguous;
prose about them is not.

**Every scan here asserts an ABSENCE**, so each has a companion asserting the
scan still matches something. A renamed directory, a fence marker that changed,
a regex that stopped firing -- any of those makes a scan for offenders pass by
finding nothing at all, and the guard would go quiet exactly when the docs
stopped being read.

**What this does NOT check, on purpose.**

*Bare class names.* ``ValueError`` is a builtin, ``ShapeDtypeStruct`` is jax's,
``SwitchCycle`` is ``rhino_cal_jax``'s, and ``BeamOperator`` is correctly
described in ``DESIGN.md`` as having been removed. Measured on this tree: 84 of
116 backticked CamelCase names resolve to a class in ``src/`` and 32 do not,
and separating the four reasons needs a hand-kept exemption list. The
``Class.member`` check below has no such problem: when the class is not ours it
simply does not fire, and that skip is derived rather than declared.

*Shipped-status prose.* "not yet", "ships today", "is planned" -- the claims
this module was commissioned over. Measured on this tree: 25 lines carry that
phrasing and, on reading, four are about the code's state at all; the rest are
about data ("after the recordings are gone") or about PyPI. Neither sentence
that actually rotted named the symbol it was wrong about -- ``load_document``
appears in neither -- so a guard pairing modality with a nearby name would have
been vacuous on the very instances that motivated it, while firing on ordinary
prose edits. It is not built here, and that is a finding rather than an
omission.

*``CHANGELOG.md``.* A changelog describes the past. A name that was right when
it was released stays right, so the file is excluded by that rule rather than
by being listed.

**The one cost this imposes.** The last check requires ``CLAUDE.md`` and
``AGENTS.md`` to stay byte-identical, so an edit to the working notes has to be
made twice. That is deliberate -- it is the price of noticing, and it was paid
silently and wrongly for four commits -- but it is a price, and the way to stop
paying it is to make one file the other's symlink rather than to weaken the
check.
"""

from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


# --------------------------------------------------------------------------
# The corpus: tracked markdown, minus the one file that is history by design.
# --------------------------------------------------------------------------

def _tracked() -> frozenset[str]:
    out = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files"],
        capture_output=True, text=True, check=True,
    ).stdout.split()
    return frozenset(out)


def _pages() -> list[Path]:
    """Every tracked ``.md`` except the changelog, which records the past."""
    return sorted(
        ROOT / name
        for name in _tracked()
        if name.endswith(".md") and Path(name).name != "CHANGELOG.md"
    )


#: An inline ``code span``. Fenced blocks are stripped first -- a fence is
#: executable and has its own guards (``test_tour_runs.py``); this module is
#: about the prose between them.
_SPAN = re.compile(r"`([^`\n]+)`")


def _prose_spans(path: Path):
    """``(line number, span text)`` for every code span outside a fence."""
    fenced = False
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        if line.lstrip().startswith("```"):
            fenced = not fenced
            continue
        if fenced:
            continue
        for span in _SPAN.findall(line):
            yield number, span.strip()


def _sites(pattern: re.Pattern[str]):
    """Every prose span matching ``pattern``, as ``(page, line, match)``."""
    for page in _pages():
        for number, span in _prose_spans(page):
            found = pattern.match(span)
            if found is not None:
                yield page.relative_to(ROOT), number, found


# --------------------------------------------------------------------------
# What src/ actually declares, walked once.
# --------------------------------------------------------------------------

def _class_table() -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """``(class -> members declared, class -> base names)`` over ``src/``.

    Members are methods, annotated fields and plain class-level assignments --
    the three ways this package writes something a caller can reach by dot.
    Bases are recorded by their trailing name (``eqx.Module`` -> ``Module``)
    so an inherited member can be found when the base is also ours.
    """
    members: dict[str, set[str]] = {}
    bases: dict[str, set[str]] = {}
    for path in sorted(SRC.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.ClassDef):
                continue
            here = members.setdefault(node.name, set())
            for statement in node.body:
                if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    here.add(statement.name)
                elif isinstance(statement, ast.AnnAssign) and isinstance(
                    statement.target, ast.Name
                ):
                    here.add(statement.target.id)
                elif isinstance(statement, ast.Assign):
                    here.update(t.id for t in statement.targets if isinstance(t, ast.Name))
            inherited = bases.setdefault(node.name, set())
            for base in node.bases:
                if isinstance(base, ast.Name):
                    inherited.add(base.id)
                elif isinstance(base, ast.Attribute):
                    inherited.add(base.attr)
    return members, bases


def _reachable(name: str, members, bases, seen: frozenset[str] = frozenset()) -> set[str]:
    """Members of ``name`` plus those of every base that is also ours."""
    if name in seen:
        return set()
    seen = seen | {name}
    found = set(members.get(name, ()))
    for base in bases.get(name, ()):
        if base in members:
            found |= _reachable(base, members, bases, seen)
    return found


def _parameter_names() -> set[str]:
    """Every name ``src/`` accepts as a keyword: parameters and fields.

    Deliberately flat. A documented ``foo=`` is checked against the whole
    package rather than against the one function the sentence is about,
    because prose rarely says which function it means and a guard that
    guessed would fire on correct sentences. What it still catches is the
    rename that removes a name from the package entirely -- which is what
    makes the prose false.
    """
    found: set[str] = set()
    for path in sorted(SRC.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                arguments = node.args
                found.update(
                    argument.arg
                    for argument in (
                        *arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs,
                    )
                )
                for slot in (arguments.vararg, arguments.kwarg):
                    if slot is not None:
                        found.add(slot.arg)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                found.add(node.target.id)
    return found


# --------------------------------------------------------------------------
# 1. `Class.member`
# --------------------------------------------------------------------------

#: ``Finding.check``, ``SwitchCycle.gather``, ``mmodes()`` is not this shape.
_MEMBER = re.compile(r"^([A-Z][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)(?:\(\))?$")


def _member_sites():
    members, bases = _class_table()
    for page, number, found in _sites(_MEMBER):
        owner, member = found.group(1), found.group(2)
        if owner not in members:
            continue          # not ours -- jax's, rhino-cal's, a builtin's
        yield page, number, owner, member, _reachable(owner, members, bases)


def test_every_class_member_the_docs_quote_still_exists() -> None:
    """A renamed method leaves the sentence about it silently false."""
    offenders = [
        f"{page}:{number}: `{owner}.{member}` -- {owner} has no {member}"
        for page, number, owner, member, reachable in _member_sites()
        if member not in reachable
    ]
    assert offenders == [], (
        "these name a member their class does not have:\n  "
        + "\n  ".join(offenders)
        + "\nIf the member is inherited from a base outside this package the "
        "walk cannot see it and the sentence is fine -- say so in the page "
        "rather than loosening this, since the skip for a class that is not "
        "ours is already automatic."
    )


def test_the_member_scan_still_reads_the_pages() -> None:
    """ANTI-VACUITY. The check above asserts an absence and so cannot prove
    on its own that it still reads anything: a changed fence marker, a moved
    page or a regex that stopped firing all make it pass by finding nothing.
    """
    checked = list(_member_sites())
    assert len(checked) >= 30, (
        f"only {len(checked)} `Class.member` spans were checked, against the "
        "55 on this tree when this was written. Either the pages stopped "
        "naming members or the scan stopped seeing them: "
        f"{[f'{p}:{n} {o}.{m}' for p, n, o, m, _ in checked[:5]]}"
    )


# --------------------------------------------------------------------------
# 2. `keyword=`
# --------------------------------------------------------------------------

#: ``bias_tolerance=``, ``noise_std=`` -- a span that is a keyword and an
#: equals sign and nothing else, which is how these pages write an argument.
_KEYWORD = re.compile(r"^([a-z_][a-z0-9_]*)=$")


def _keyword_sites():
    for page, number, found in _sites(_KEYWORD):
        yield page, number, found.group(1)


def test_every_keyword_argument_the_docs_quote_is_one() -> None:
    """``Pass bias_tolerance= to make it a refusal`` -- and if the parameter
    is renamed, the reader passes a name nothing accepts."""
    accepted = _parameter_names()
    offenders = [
        f"{page}:{number}: `{keyword}=`"
        for page, number, keyword in _keyword_sites()
        if keyword not in accepted
    ]
    assert offenders == [], (
        "no function or field in src/ takes these:\n  " + "\n  ".join(offenders)
        + "\nThe page tells a reader to pass an argument the package does not "
        "have. Rename it in the prose, or the parameter back."
    )


def test_the_keyword_scan_still_reads_the_pages() -> None:
    """ANTI-VACUITY -- see the member scan's companion."""
    checked = list(_keyword_sites())
    assert len(checked) >= 40, (
        f"only {len(checked)} `keyword=` spans were checked, against the 60 on "
        "this tree when this was written."
    )


# --------------------------------------------------------------------------
# 3. `directory/file.py`
# --------------------------------------------------------------------------

#: A path with at least one directory component. A bare ``setup.py`` is not
#: matched: ``DESIGN.md`` names one that another project does not have, which
#: is the point of the sentence.
_REPO_PATH = re.compile(r"^([\w][\w.-]*(?:/[\w.-]+)+\.py)$")


def _path_sites():
    """Only paths whose DIRECTORY exists here -- so a citation of another
    repository's file (``limTOD/tests/test_cstbeam.py``,
    ``tests/limtod_jax/test_horizon_partition.py``) is skipped by the tree's
    own shape rather than by being listed.

    Existence is asked of the WORKING TREE, not of ``git ls-files``. The first
    draft asked git, and a mutation run walked straight past it: renaming a
    cited module without staging the rename left the old name in the index, so
    the check said the file was there while the reader's editor said it was
    not. Git decides which pages SHIP, below; the disk decides what is
    actually there.
    """
    for page, number, found in _sites(_REPO_PATH):
        candidate = found.group(1)
        if not (ROOT / Path(candidate).parent).is_dir():
            continue
        yield page, number, candidate, (ROOT / candidate).is_file()


def test_every_repository_file_the_docs_cite_is_in_the_repository() -> None:
    """A moved test module leaves every page that cited it pointing at air."""
    offenders = [
        f"{page}:{number}: `{candidate}`"
        for page, number, candidate, present in _path_sites()
        if not present
    ]
    assert offenders == [], (
        "these cite a file this repository does not have, in a directory it "
        "does:\n  " + "\n  ".join(offenders)
        + "\nIf the file belongs to another project, its directory should not "
        "look like one of ours -- name the project in the path."
    )


def test_the_path_scan_still_reads_the_pages() -> None:
    """ANTI-VACUITY -- and the one most likely to matter here, since renaming
    a DIRECTORY makes every path under it skip rather than fail."""
    checked = list(_path_sites())
    assert len(checked) >= 35, (
        f"only {len(checked)} repository paths were checked, against the 47 on "
        "this tree when this was written. A renamed directory does not fail "
        "the check above -- it empties it."
    )


# --------------------------------------------------------------------------
# 4. `path.py::TestSomething`
# --------------------------------------------------------------------------

#: ``tests/config/test_config_surface.py::TestTheLayerBoundaryIsMechanical``.
_NODE_ID = re.compile(r"^([\w][\w./-]*\.py)::([\w:]+)$")


def _node_id_sites():
    for page, number, found in _sites(_NODE_ID):
        path, identifier = found.group(1), found.group(2)
        yield page, number, path, identifier, (ROOT / path).is_file()


def test_every_test_the_docs_name_by_id_can_be_found() -> None:
    """``CLAUDE.md`` sends agents to tests by node id. A renamed class turns
    that into a silent dead end -- pytest reports "no tests ran", not a typo.
    """
    offenders = []
    for page, number, path, identifier, present in _node_id_sites():
        if not present:
            offenders.append(f"{page}:{number}: `{path}::{identifier}` -- no such file")
            continue
        declared = {
            node.name
            for node in ast.walk(ast.parse((ROOT / path).read_text()))
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        }
        missing = [part for part in identifier.split("::") if part not in declared]
        if missing:
            offenders.append(
                f"{page}:{number}: `{path}::{identifier}` -- "
                f"{path} declares no {missing}"
            )
    assert offenders == [], (
        "these node ids select nothing:\n  " + "\n  ".join(offenders)
        + "\npytest exits 4 on an unmatched id and prints no tests ran, which "
        "reads like a passing selection to anyone skimming."
    )


def test_the_node_id_scan_still_reads_the_pages() -> None:
    """ANTI-VACUITY -- see above."""
    checked = list(_node_id_sites())
    assert len(checked) >= 3, (
        f"only {len(checked)} `file.py::Name` spans were checked, against the "
        "4 on this tree when this was written."
    )


# --------------------------------------------------------------------------
# 5. The working notes exist twice, and the copies drift
# --------------------------------------------------------------------------

def test_the_working_notes_are_one_document_and_not_two() -> None:
    """``CLAUDE.md`` and ``AGENTS.md`` are the same document for two tools.

    Same title, same opening paragraph, same sections. They are maintained by
    editing whichever one the current tool reads, and they drifted apart in
    OPPOSITE directions: ``AGENTS.md`` had the coverage section that moved the
    gate to its own job and ``CLAUDE.md`` did not; ``CLAUDE.md`` had the
    corrected config allowlist -- three files, not five -- and ``AGENTS.md``
    did not. Each was stale exactly where the other was current, and the file
    every new session reads FIRST was wrong in both readings at once.

    This is the same shape as the two pages that opened this module's
    docstring: ``config-values.md`` and ``config-resources.md`` carried one
    claim in two wordings, and fixing one would have left the other. A claim
    written twice is a claim that goes stale once.

    Byte equality is the whole check, deliberately. If these two ever have to
    differ, the answer is to generate one from the other or symlink them --
    not to soften this into a similarity threshold, which is how a diff of
    one paragraph becomes invisible again.
    """
    claude = (ROOT / "CLAUDE.md").read_text()
    agents = (ROOT / "AGENTS.md").read_text()
    if claude == agents:
        return
    import difflib
    diff = "\n".join(
        difflib.unified_diff(
            agents.splitlines(), claude.splitlines(),
            fromfile="AGENTS.md", tofile="CLAUDE.md", lineterm="", n=1,
        )
    )
    pytest.fail(
        "CLAUDE.md and AGENTS.md are two copies of one document and they "
        "disagree. Decide which reading is measured on THIS tree -- neither "
        "copy is authoritative by being newer -- and write it into both:\n"
        + diff
    )
