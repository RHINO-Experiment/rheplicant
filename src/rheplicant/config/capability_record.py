"""``capabilities.json``: the maturity of the physics one document places.

WHY THIS IS A FILE AND NOT A FIELD
----------------------------------
A published run should be able to say, without anyone reading its prose,
which of its physics was a stand-in. Check A53 already says it -- one
informational finding per document naming every non-maintained node -- but it
says it in a **message**, and a message is for a person. A consumer that wants
to filter an archive on "did anything placeholder feed this number" would have
to parse English.

The obvious place for the structured answer is ``provenance.json``, and that
is the place it must not go. ``provenance-v1`` is closed: every object sets
``additionalProperties: false`` and requires every property it declares, so a
new field is a ``format_version`` bump on a schema that ships in the wheel.
``_rheplicant_bootstrap.audit.provenance`` records that two of the three last
requests to add a field there were answered WITHOUT touching it -- the tree
became enumerable through ``integrity.json``, and presets became recoverable by
publishing their sources -- and says that is the route to prefer.

This is the third instance of the same answer. A separate published file, its
own ``format_version``, its own schema in the wheel, and ``integrity.json``
covers it like every other file in the tree. Nothing in the audit contract
moves.

**It is written on the success path only.** The record is a view of the
resolved layers, and a document refused before it resolves has none.

That is the motivating case rather than the whole of it, and the difference
is worth stating: the file is attached to the terminal SUCCESS publication, so
a run whose layers DID resolve and which then failed later -- building the
product bundle, say -- publishes a refused or error bundle without it. Every
such bundle still carries check A53's notice in ``diagnostics.json``, which is
the English of the same fact; what it does not carry is the machine-readable
form. Widening this means passing the already-built file into
``_publish_failure_once``, and it is a behaviour change to a published tree
rather than a repair, so it is written down here rather than made quietly.

WHAT IT RECORDS, AND WHAT IT DOES NOT
-------------------------------------
Every node the document places, with the level of the class that node
resolves to. Every node, not only the non-maintained ones: A53 is a notice and
says nothing when there is nothing to warn about, whereas a record that
omitted the maintained nodes would be indistinguishable from a record written
before the node existed.

The levels are READ, never decided here. They come off the ``maturity``
ClassVar through :func:`rheplicant.radio.capabilities`, the one walk every
view shares, so a node whose physics arrives changes this file in the same
commit that raises its level.

It does not record the whole registry. The installed version and commit are
already in ``provenance.json``, so the registry of the day is recoverable from
the software row; a copy here would be a second spelling that can disagree.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from _rheplicant_bootstrap.audit.json import canonical_json_bytes
from _rheplicant_bootstrap.audit.types import ResolvedLayerRecord
from _rheplicant_bootstrap.errors import ConfigError
from rheplicant.config.sections.compose import model_nodes
from rheplicant.config.sections.model import operator_table
from rheplicant.core.capability import Maturity
from rheplicant.radio import capabilities

CAPABILITIES_FORMAT_VERSION = 1

#: Why a node carries no level. Two entries, and there is deliberately no
#: third for "the class declares no maturity": every class addressable from a
#: document does declare one, and
#: ``tests/config/test_capability_record.py`` asserts it rather than leaving a
#: vocabulary entry nothing can produce.
#:
#: ``unresolved_type`` covers a ``python:`` node as well as a ``type:`` that
#: matches nothing: in both, the walk cannot settle the class without
#: importing. A ``python:`` node got this reason at every node id where the
#: table holds zero or several classes before 2026-09-23, when the one-class
#: case was brought into line. A separate entry for it would widen the
#: ``capabilities-v1`` enum that shipped in 0.9.0, which is a
#: ``format_version`` bump.
LEVEL_REASONS = ("spec_not_a_mapping", "unresolved_type")


@dataclass(frozen=True, slots=True)
class NodeLevel:
    """One node's maturity, or the reason there is none."""

    node_id: str
    type: str | None
    level: str | None
    level_reason: str | None


def node_levels(document: Mapping[str, Any]) -> tuple[NodeLevel, ...]:
    """Every node the document places, with the level it resolves to.

    In the document's own key order, because that order is a fact about the
    document and sorting would discard it.

    Resolution imports nothing, which is what lets a pre-flight check share
    it: a ``type:`` is matched against the classes already registered at that
    node id by :func:`~rheplicant.config.sections.model.operator_table`, and a
    node whose class cannot be settled that way gets ``level: null`` with a
    reason rather than being dropped. A dropped node and a node that does not
    exist look identical to a reader, and only one of them is true.

    A ``python:`` node never reaches the table. Its class is the import
    target's, which this walk does not import, so it gets ``level: null`` and
    ``type: null`` whatever the table holds at that node id.
    """
    specs = model_nodes(document)
    if not specs:
        return ()
    levels = capabilities()
    table = operator_table()
    rows: list[NodeLevel] = []
    for node_id, spec in specs.items():
        if not isinstance(spec, Mapping):
            rows.append(NodeLevel(node_id, None, None, "spec_not_a_mapping"))
            continue
        if "python" in spec:
            # The build dispatches on `python:` before it consults the table
            # (`sections/model.py::build_node_operator`), so a class registered
            # at this node id is one the document did not choose. Falling
            # through to the table published that class's level for a custom
            # operator whenever the node id held exactly one class, and A53
            # then named it. Settling the named class would mean importing it.
            rows.append(NodeLevel(node_id, None, None, "unresolved_type"))
            continue
        classes = table.get(node_id)
        declared = spec.get("type")
        if isinstance(declared, str):
            chosen = next((cls for cls in (classes or ()) if cls.__name__ == declared), None)
        elif classes and len(classes) == 1:
            chosen = classes[0]
        else:
            chosen = None
        if chosen is None or chosen.__name__ not in levels:
            # Report the name whichever way it was learned. Reading only
            # `declared` threw away a class this walk had already resolved --
            # the case where a node resolves unambiguously to one class that
            # `capabilities()` does not carry a level for, which is reachable
            # the moment `operator_table()` and `capability_classes()` walk
            # different populations. A row saying `type: null` then blames the
            # document for something the registry did.
            name = declared if isinstance(declared, str) else None
            if name is None and chosen is not None:
                name = chosen.__name__
            rows.append(NodeLevel(node_id, name, None, "unresolved_type"))
            continue
        rows.append(NodeLevel(node_id, chosen.__name__, levels[chosen.__name__].value, None))
    return tuple(rows)


def at_level(rows: tuple[NodeLevel, ...], level: Maturity) -> tuple[NodeLevel, ...]:
    """The rows at one level. A53 and the record read the same list."""
    return tuple(row for row in rows if row.level == level.value)


def capabilities_manifest(layers: Sequence[ResolvedLayerRecord]) -> bytes:
    """The canonical bytes published as ``capabilities.json``.

    **Per resolved layer, not per document.** A variant may place a different
    ``type:`` at a node than the base does, so one record for the document
    would answer for a layer the reader did not run. These are the same layers
    the resolved YAML artefacts come from -- ``config.resolved.yaml`` and
    ``variants/<n>/config.resolved.yaml`` -- so a row here has a published
    document beside it to check against.
    """
    rows = []
    for record in layers:
        if type(record) is not ResolvedLayerRecord:
            raise ConfigError("capabilities record requires exact resolved layers.")
        nodes = node_levels(record.effective_document)
        # Exactly one of `level` and `level_reason` is set, checked here as
        # `validate_software` checks its own pairs. The schema cannot say it:
        # both fields are independent nullable unions, so a row with neither
        # -- or with both -- is schema-VALID and incoherent, and would publish.
        # `node_levels` gets this right today; nothing downstream would notice
        # if it stopped.
        for row in nodes:
            if (row.level is None) == (row.level_reason is None):
                raise ConfigError(
                    f"capabilities record for node {row.node_id!r} sets "
                    f"level={row.level!r} and level_reason={row.level_reason!r}; "
                    "exactly one of the two carries the answer."
                )
        rows.append(
            {
                "layer": {"kind": record.layer.kind, "name": record.layer.name},
                "nodes": tuple(
                    {
                        "node_id": row.node_id,
                        "type": row.type,
                        "level": row.level,
                        "level_reason": row.level_reason,
                    }
                    for row in nodes
                ),
            }
        )
    return canonical_json_bytes(
        {"format_version": CAPABILITIES_FORMAT_VERSION, "layers": tuple(rows)}
    )


__all__ = [
    "CAPABILITIES_FORMAT_VERSION",
    "LEVEL_REASONS",
    "NodeLevel",
    "at_level",
    "capabilities_manifest",
    "node_levels",
]
