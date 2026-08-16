"""One rule, one binding -- as a command rather than as a review step.

Plan 3B moves eleven checks that already exist, in the right words, in the
wrong phase.  A hoist has exactly one legal shape: the refusal becomes a
module-level pure function where it already lives, the pass CALLS it, and the
section keeps calling it too.  **Copying the message into the pass and leaving
the original behind is forbidden**, because two validators for one property
drift -- that is the ``_number``-vs-``_whole`` divergence on the 2C ledger,
which shipped and was found by a reviewer's memory rather than by a run.

:func:`assert_bound_once` is the mechanical form of that rule: a refusal's
message literal appears in **exactly one module** under ``src/rheplicant/``.

**There is deliberately no shared table of literals.**  Each task
parametrizes a test in ITS OWN test module over ITS OWN literals.  A shared
table is the one resolution of this conflict whose failure mode is a *passing*
test: two tasks landing in parallel both edit it, the merge keeps one side,
and the rows that vanished are exactly the rows nobody is checking any more.
What is shared is this function.

**Why the harvest is ``ast`` and not ``grep``.**  A message in ``src/`` is
written as a dozen implicitly-concatenated pieces across a dozen lines, so a
raw substring search for a whole sentence finds nothing at all -- a walker
that reported "bound 0 times" for every literal would pass every
one-binding test that asserted ``>= 1`` and fail every one that asserted
``== 1``, and either way it would be measuring its own line wrapping.  CPython
folds adjacent literals in the parser and folds an f-string among them into a
single ``JoinedStr``, so the fold below cannot be fooled by a re-wrap.
Interpolations become :data:`_HOLE`, so ``f"got {value!r}"`` compares equal
across a change of variable name and unequal across a change of words.

**What this cannot see**, written down because a verification method with the
same blind spot as the code is this project's recorded failure mode:

* a message that survives as TEXT and stops being REACHED.  Nothing here runs
  a pass.  The per-check tests are what say a refusal still fires.
* a sentence duplicated with one word changed.  Two near-copies are two
  different literals and each is "bound once".  Only an exact duplicate is
  caught, which is the shape a copy-paste hoist actually produces.
* a duplicate outside ``src/rheplicant/`` -- a message written out in a test,
  or in ``docs/``.  Those are not bindings.
"""

import ast
import pathlib
import re

#: ``tests/config/`` -> the repository root -> the package.
_SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "rheplicant"

#: What an interpolation folds to.  The same character
#: ``test_config_preflight.py`` uses, so the two harvests agree on what a
#: message looks like.
_HOLE = "…"

#: Below this, a string is a key, a token or a format fragment rather than a
#: sentence, and the walk would report noise.
_MESSAGE_FLOOR = 40

#: Literals that legitimately live in TWO modules, with the reason.  **Not an
#: exemption list that may grow quietly**: every entry is measured, and
#: :func:`exempt_pairs_still_hold` is what fails the day one stops being true,
#: so the entry gets deleted rather than outliving its reason.
#:
#: ``coords.time is stored as`` is the measured one at ``e0e024a``: the time
#: axis has TWO guards, ``core/coordinates.py``'s and
#: ``radio/instrument/calibration.py``'s, and they differ in what they ask.
#: They are the plan's own named twin for C1, and collapsing them is a
#: decision that belongs to whichever task hoists that rule -- not something
#: this walker should force by going red.
_EXEMPT: dict[str, frozenset[str]] = {
    "coords.time is stored as": frozenset({
        "core/coordinates.py",
        "radio/instrument/calibration.py",
    }),
}


def _flat(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _message_texts(source: str) -> set[str]:
    """Every message-shaped string literal in ``source``, whitespace-flattened.

    DOCSTRINGS ARE EXCLUDED.  They are the layer's reasoning, they quote
    messages constantly and on purpose, and counting them would make every
    hoist's docstring a second "binding".
    """
    tree = ast.parse(source)
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            first = node.body[0] if node.body else None
            if (isinstance(first, ast.Expr)
                    and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)):
                docstrings.add(id(first.value))
    found = set()
    for node in ast.walk(tree):
        if id(node) in docstrings:
            continue
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            text = node.value
        elif isinstance(node, ast.JoinedStr):
            text = "".join(
                part.value if (isinstance(part, ast.Constant)
                               and isinstance(part.value, str)) else _HOLE
                for part in node.values)
        else:
            continue
        flat = _flat(text)
        if len(flat) >= _MESSAGE_FLOOR:
            found.add(flat)
    return found


def modules_carrying(literal: str) -> tuple[str, ...]:
    """Every module under ``src/rheplicant/`` that SAYS ``literal`` out loud.

    Matched as a substring of a harvested message, so a caller may pass a whole
    sentence or the distinctive clause of one.  Paths are returned relative to
    the package, so a failure names ``sections/noise.py`` rather than a
    machine-specific absolute path.
    """
    wanted = _flat(literal)
    return tuple(
        str(path.relative_to(_SRC))
        for path in sorted(_SRC.rglob("*.py"))
        if any(wanted in text for text in _message_texts(path.read_text()))
    )


def assert_bound_once(literal: str) -> None:
    """``literal`` is written in exactly one module under ``src/rheplicant/``.

    Raises:
        AssertionError: bound in no module (the check was deleted, or the
            literal in the test has drifted from the source), or bound in
            more than one (a hoist that copied rather than moved).
    """
    found = modules_carrying(literal)
    allowed = _EXEMPT.get(_flat(literal))
    if allowed is not None and set(found) == set(allowed):
        return
    assert len(found) == 1, (
        f"this message is bound {len(found)} times, in {list(found)}:\n\n"
        f"  {_flat(literal)}\n\n"
        "A rule has ONE binding. Two modules carrying one sentence is two "
        "validators for one property, and they drift -- that is the "
        "_number-vs-_whole divergence this walker exists to catch. A hoist "
        "extracts the refusal into a module-level pure function and CALLS it "
        "from the pass; it does not copy the words. Zero means the opposite: "
        "the literal in this test no longer matches any source, so the test "
        "is checking nothing."
    )


def exempt_pairs_still_hold() -> dict[str, tuple[tuple[str, ...], frozenset[str]]]:
    """Every :data:`_EXEMPT` entry, with what it claims and what is true now.

    The exemption must PROVE ITSELF.  A forgiveness list nobody re-measures is
    a list of duplications nobody is checking, which is the state this module
    exists to end -- one indirection along.  The test that reads this fails
    the day a pair becomes one module (delete the row) or grows a third
    (decide it).
    """
    return {literal: (modules_carrying(literal), allowed)
            for literal, allowed in _EXEMPT.items()}
