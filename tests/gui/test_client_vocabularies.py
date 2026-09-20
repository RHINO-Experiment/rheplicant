"""Every string-literal vocabulary in the client, against Python's own.

A10-4. ``types.ts`` restates vocabularies this package owns -- the node kinds,
the job kinds, a job's status, the output states, the finding severities, the
preview ids -- and the drift guard in ``form_catalog_finalize`` checked a
handful of interfaces, none of them these. A vocabulary that exists twice is
one that drifts, and the browser's copy drifts silently: TypeScript is happy
with any union it can see, and the value it rejects arrives from a server that
was right.

The census is derived from the file. Every union of two or more string
literals in ``types.ts`` must either be MATCHED against a live Python
vocabulary here, or be exempted by name with a reason. Both directions are
asserted, so a union added to the client fails until somebody says which it
is, and an entry naming a union that no longer exists fails too.
"""

from __future__ import annotations

import pathlib
import re

import pytest

CLIENT = (
    pathlib.Path(__file__).resolve().parents[2]
    / "src"
    / "rheplicant"
    / "gui"
    / "react"
    / "types.ts"
)

#: A union in ``types.ts`` -- either ``export type X =`` or a field ``x:`` --
#: whose members are two or more quoted strings.
_UNION = re.compile(
    r"(?:export type (?P<alias>\w+)\s*=|(?P<field>\w+)\??:)\s*"
    r'(?P<members>(?:\s*\|?\s*"[^"]+"\s*\|)+\s*"[^"]+")'
)


def client_vocabularies() -> dict[tuple[str, int], tuple[str, ...]]:
    """``(name, line) -> members``, for every string-literal union."""
    text = CLIENT.read_text(encoding="utf-8")
    found = {}
    for match in _UNION.finditer(text):
        name = match.group("alias") or match.group("field")
        members = tuple(re.findall(r'"([^"]+)"', match.group("members")))
        if len(members) >= 2:
            found[(name, text[: match.start()].count("\n") + 1)] = members
    return found


def live(name: str) -> tuple[str, ...]:
    """The Python vocabulary a client union must equal."""
    if name == "kind@node":
        from rheplicant.radio.graph import RADIO_GRAPH

        return tuple(sorted({node.kind for node in RADIO_GRAPH.nodes.values()}))
    if name == "segment":
        from rheplicant.radio.graph import RADIO_GRAPH

        return tuple(sorted({node.segment for node in RADIO_GRAPH.nodes.values()}))
    if name == "composition":
        from rheplicant.config.sections.compose import COMPOSITIONS

        return tuple(sorted(COMPOSITIONS))
    if name == "severity":
        from rheplicant.config.findings import SEVERITIES

        return tuple(sorted(SEVERITIES))
    if name == "change kind":
        from rheplicant.gui.validation import PresetChange

        return tuple(sorted(_literal_of(PresetChange, "kind")))
    if name == "configuration":
        from rheplicant.gui.document import NodeCard

        return tuple(sorted(_literal_of(NodeCard, "configuration")))
    if name == "JobKind":
        from rheplicant.gui.jobs import JobKind

        return tuple(sorted(_args(JobKind)))
    if name == "status":
        from rheplicant.gui.jobs import JobStatus

        return tuple(sorted(_args(JobStatus)))
    raise AssertionError(f"no live vocabulary is wired for {name!r}")


def _args(alias) -> tuple[str, ...]:
    import typing

    return tuple(typing.get_args(alias))


def _literal_of(cls, field: str) -> tuple[str, ...]:
    import dataclasses
    import typing

    for row in dataclasses.fields(cls):
        if row.name == field:
            annotation = row.type
            if isinstance(annotation, str):
                annotation = typing.get_type_hints(cls)[field]
            return tuple(typing.get_args(annotation))
    raise AssertionError(f"{cls.__name__} has no field {field!r}")


#: Client union -> the Python vocabularies it must equal, by the keys
#: ``live`` understands.
#:
#: A SET per name rather than one, because ``types.ts`` uses ``kind`` for two
#: unrelated vocabularies -- a graph node's kind and a document change's --
#: and a table keyed by name alone cannot tell them apart. That was the first
#: spelling here and it failed on the ambiguity rather than papering over it,
#: which is the right way round.
MATCHED = {
    "kind": ("kind@node", "change kind"),
    "segment": ("segment",),
    "composition": ("composition",),
    "configuration": ("configuration",),
    "severity": ("severity",),
    "JobKind": ("JobKind",),
    "status": ("status",),
}

#: Unions with no Python vocabulary to compare against, and why.
#:
#: Each of these is decided in a `Literal[...]` inline in one function rather
#: than bound to a name, so there is nothing to import. That is a smaller
#: problem than a silent copy and it is written down instead of hidden: a
#: reason here is a claim someone can disagree with, whereas an unchecked
#: union looks the same as a checked one.
EXEMPT = {
    "NodeFieldControl": (
        "the widget control names, decided by the catalog builder's own "
        "`widget=` arguments rather than by a bound vocabulary"
    ),
    "OutputState": (
        "bound as an inline Literal in `gui/outputs.py`'s projection and not "
        "exported; worth binding, and not while this census is being written"
    ),
    "preview_id": ("an inline Literal in `gui/previews.py`, not bound to a name there"),
    "cadence": ("an inline Literal in `gui/previews.py`, not bound to a name there"),
    "axis": ("an inline Literal in `gui/previews.py`, not bound to a name there"),
}


def test_the_client_declares_vocabularies_to_check():
    """Guard the guard: a regex that matched nothing would pass everything."""
    found = client_vocabularies()
    assert len(found) >= 9, sorted(name for name, _line in found)


def test_every_client_vocabulary_is_matched_or_exempted():
    """Both directions, so neither table can go stale quietly."""
    names = {name for name, _line in client_vocabularies()}
    classified = set(MATCHED) | set(EXEMPT)
    assert names <= classified, {"in types.ts and unclassified": sorted(names - classified)}
    assert classified <= names, {
        "classified here and gone from types.ts": sorted(classified - names)
    }


@pytest.mark.parametrize("name", sorted(MATCHED))
def test_a_matched_vocabulary_equals_the_python_one(name):
    found = {
        tuple(sorted(members))
        for (union, _line), members in client_vocabularies().items()
        if union == name
    }
    expected = {live(key) for key in MATCHED[name]}
    assert found == expected, {
        "in types.ts": sorted(found),
        "in this package": sorted(expected),
    }


def test_every_exemption_states_a_reason():
    """A label is not a reason, and an exemption without one is a hole."""
    for name, reason in EXEMPT.items():
        assert len(reason) > 40, f"{name}'s exemption is a label, not a reason"
