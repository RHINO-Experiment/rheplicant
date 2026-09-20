"""``json_schema()`` byte for byte, because a consumer reads the bytes.

``test_schema.py`` checks this object's PROPERTIES, and checks most of them
against the code's own tables rather than against literals -- the section list
is derived from pre-flight's tables, the exit list from the runs section's own
kind tuple. That is the stronger kind of test and it is not what a golden is
for.

What a property test cannot see is a change it does not have a property for: a
reason reworded, an operator added to the catalog, a unit removed from
``acceptedUnits``. Every one of those reaches `rheplicant-agent`, which calls
``json_schema()`` over its RPC surface and builds a document editor from the
answer. A change there is a change to somebody else's program.

So this pins the serialisation. The point is the DIFF: a reviewer sees exactly
what moved, and decides whether the ``schemaVersion`` beside it should move
too. The test cannot make that decision -- only a person can say whether a
change is additive -- so it does the half a test can do, which is refuse to
let the change pass unseen.

**Regenerating.** When a change is deliberate:

    .venv/bin/python -c "import json, pathlib; \\
      from rheplicant.config.schema import json_schema; \\
      pathlib.Path('tests/config/golden/json_schema.json').write_text( \\
        json.dumps(json_schema(), sort_keys=True, indent=2) + chr(10))"

Sorted keys and two-space indent, so the file is stable under a dict that
changes insertion order and the diff shows the change rather than the
reordering.
"""

from __future__ import annotations

import json
import pathlib

from rheplicant.config.schema import json_schema

GOLDEN = pathlib.Path(__file__).resolve().parent / "golden" / "json_schema.json"


def _canonical() -> str:
    return json.dumps(json_schema(), sort_keys=True, indent=2) + "\n"


def test_the_published_schema_matches_its_golden():
    assert GOLDEN.exists(), (
        f"{GOLDEN} is missing. The golden IS the record of what this package "
        "publishes; regenerate it with the command in this module's docstring"
    )
    live = _canonical()
    stored = GOLDEN.read_text(encoding="utf-8")
    if live == stored:
        return

    live_object, stored_object = json.loads(live), json.loads(stored)
    added = sorted(set(live_object) - set(stored_object))
    removed = sorted(set(stored_object) - set(live_object))
    changed = sorted(
        key
        for key in set(live_object) & set(stored_object)
        if live_object[key] != stored_object[key]
    )
    raise AssertionError(
        "json_schema() no longer matches its golden.\n"
        f"  top-level added:   {added}\n"
        f"  top-level removed: {removed}\n"
        f"  top-level changed: {changed}\n"
        "rheplicant-agent reads this over RPC and builds an editor from it, so "
        "a change here is a change to another program. If it is deliberate, "
        "regenerate the golden (command in this module's docstring) AND decide "
        "in the same commit whether schemaVersion should move: additive is "
        "safe, a removal or a rename is not."
    )


def test_the_golden_carries_the_version_the_code_publishes():
    """A golden regenerated without reading it would still pass the comparison.

    This is the one claim worth stating twice: the version in the file and the
    version in the code are the same string, so a regeneration cannot quietly
    ship a new shape under an old number -- the number is IN the bytes being
    compared, and this says which number it has to be.
    """
    stored = json.loads(GOLDEN.read_text(encoding="utf-8"))
    live = json_schema()["schemaVersion"]
    assert stored["schemaVersion"] == live, (
        f"the golden says schemaVersion {stored['schemaVersion']!r} and the "
        f"code publishes {live!r}. Regenerate the golden, or put the version "
        "back -- a consumer branches on this string"
    )
    assert isinstance(stored["schemaVersion"], str), (
        "schemaVersion is a STRING in this contract; a JSON number would be a "
        "different type on the consumer's side"
    )


def test_the_golden_is_stored_canonically():
    """Sorted, indented, newline-terminated.

    Not pedantry: a golden written with a different dump makes the next real
    diff unreadable, which is the whole value being protected.
    """
    stored = GOLDEN.read_text(encoding="utf-8")
    assert stored.endswith("\n"), "the golden must end with a newline"
    assert stored == json.dumps(json.loads(stored), sort_keys=True, indent=2) + "\n", (
        "the golden is not in canonical form; regenerate it with the command "
        "in this module's docstring rather than editing it by hand"
    )
