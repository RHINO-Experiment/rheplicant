"""Framework-free YAML projection for the config editor.

The browser never owns a second scientific state.  It sends YAML text here,
and every content edit receives a new YAML document plus a projection of the
live signal-path graph.  Selection and hover remain view state and therefore
do not invent config keys.

This module READS a document: it parses one and projects the graph, the node
cards and their typed fields. The commands that CHANGE one live beside it in
:mod:`rheplicant.gui.document_edits`, which imports from here and not the
other way round.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import yaml

from _rheplicant_bootstrap.errors import ConfigError
from _rheplicant_bootstrap.layering import apply_variant
from _rheplicant_bootstrap.yaml import safe_load_document
from rheplicant.core.graph import SignalGraph
from rheplicant.gui.forms import (
    FormCatalog,
    FormProjection,
    project_forms,
    widget_catalog,
)
from rheplicant.gui.node_forms import (
    NodeField,
    from_route_fields,
    project_compose_stages,
    project_node_fields,
    project_node_instances,
)
from rheplicant.gui.previews import PreviewProjection, project_previews
from rheplicant.gui.validation import ValidationProjection, validate_document

#: Re-exported under its historic private name: `outputs.py` and
#: `document_edits.py` import `_same_value` from here, and the point of
#: the move is one implementation, not one import path.
from rheplicant.gui.yaml_values import plain as _plain
from rheplicant.gui.yaml_values import same_value as _same_value
from rheplicant.radio.graph import RADIO_GRAPH

_COMPOSITION_KINDS = frozenset(("junction", "selector"))
_MODEL_SECTION_KEYS = frozenset(("kind", "acknowledge_double_count", "stages"))
_PROCESSING_NODES = tuple(
    node_id for node_id in RADIO_GRAPH._topo if RADIO_GRAPH.nodes[node_id].segment == "processing"
)
_PROCESSING_GRAPH = SignalGraph(
    "single-antenna-processing",
    {node_id: RADIO_GRAPH.nodes[node_id] for node_id in _PROCESSING_NODES},
    tuple(
        (source, target)
        for source, target in RADIO_GRAPH.edges
        if source in _PROCESSING_NODES and target in _PROCESSING_NODES
    ),
)
@dataclass(frozen=True, slots=True)
class NodeInstance:
    """One ordered instance at a ``many`` node.

    It carries the same typed view a single-slot :class:`NodeCard` does,
    because an instance IS one operator's settings -- the node above it is
    the list or the label mapping, which has no fields of its own.
    """

    instance_id: str
    label: str
    settings: object
    #: Where these settings live inside the node's own: ``("0",)`` for a list
    #: entry, ``("hot",)`` for a FAN label, ``("stages", "0")`` for a stage.
    #: Sent rather than re-derived, so the browser never has to decide again
    #: whether a key is an index or a label.
    slot: tuple[str, ...]
    typed_form: bool
    typed_form_reason: str | None
    type_choices: tuple[str, ...]
    selected_type: str | None
    fields: tuple[NodeField, ...]
    extra_keys: tuple[str, ...]
    removed_by_type: dict[str, tuple[str, ...]]


@dataclass(frozen=True, slots=True)
class GraphCounts:
    """Free structural counts derived from YAML without constructing a twin."""

    lit: int
    skipped: int
    reserved: int
    instances: int
    materialized: int


@dataclass(frozen=True, slots=True)
class NodeCard:
    """One live graph node projected for a frontend."""

    node_id: str
    label: str
    kind: str
    description: str
    explanation: str
    editable: bool
    reserved: bool
    many: bool
    segment: str
    lit: bool
    count: int
    configuration: Literal[
        "single",
        "sum",
        "fan",
        "chain",
        "compose",
        "region",
        "reserved",
        "junction",
        "selector",
    ]
    settings: object | None
    instances: tuple[NodeInstance, ...]
    stage_names: tuple[str, ...]
    #: A composed node's stages, and the keys a ``from:`` route takes. Both
    #: sit beside the node's refusal rather than replacing it: the node still
    #: has no field set of its own.
    stages: tuple[NodeInstance, ...]
    from_fields: tuple[NodeField, ...]
    #: The typed view of the same ``settings``. It rides on the card because
    #: the card is what a variant re-resolves: ``EditorSnapshot.forms``
    #: projects the BASE document only, so a typed form driven off that would
    #: show base numbers under a variant's name.
    typed_form: bool
    typed_form_reason: str | None
    type_choices: tuple[str, ...]
    selected_type: str | None
    fields: tuple[NodeField, ...]
    extra_keys: tuple[str, ...]
    removed_by_type: dict[str, tuple[str, ...]]
@dataclass(frozen=True, slots=True)
class GraphDiagram:
    """One base, backend, or resolved-variant graph projection."""

    name: str
    svg: str
    nodes: tuple[NodeCard, ...]
    walk_order: tuple[str, ...]
    counts: GraphCounts
    changed_nodes: tuple[str, ...] = ()
@dataclass(frozen=True, slots=True)
class EditorSnapshot:
    """The complete, serializable result of one document transition."""

    yaml_text: str
    svg: str
    nodes: tuple[NodeCard, ...]
    walk_order: tuple[str, ...]
    forms: FormProjection
    previews: PreviewProjection
    validation: ValidationProjection
    base_diagram: GraphDiagram
    backend_diagram: GraphDiagram
    variant_diagrams: tuple[GraphDiagram, ...]


def _load(yaml_text: str) -> Mapping[str, object]:
    if not isinstance(yaml_text, str):
        raise ConfigError("GUI YAML must be text.")
    try:
        payload = yaml_text.encode("utf-8", "strict")
    except UnicodeEncodeError:
        raise ConfigError("GUI YAML must contain valid UTF-8 text.") from None
    loaded = safe_load_document(payload, source_name="GUI document").value
    if not isinstance(loaded, Mapping):
        raise ConfigError("GUI document root must be a mapping.")
    return loaded


def _model(document: Mapping[str, object]) -> Mapping[str, object]:
    value = document.get("model", {})
    if not isinstance(value, Mapping):
        raise ConfigError("model: must be a mapping.")
    for node_id, settings in value.items():
        if node_id in _MODEL_SECTION_KEYS:
            continue
        node = RADIO_GRAPH.nodes.get(node_id)
        if node is not None and node.many:
            if node_id == "cal_loads":
                if not isinstance(settings, Mapping):
                    raise ConfigError("model.cal_loads: must be a label-keyed mapping (FAN).")
                entries = tuple(settings.values())
            else:
                if isinstance(settings, str | bytes) or not isinstance(settings, Sequence):
                    shape = "CHAIN" if node_id == "filters" else "SUM"
                    raise ConfigError(f"model.{node_id}: must be a list ({shape}).")
                entries = tuple(settings)
            if any(not isinstance(entry, Mapping) for entry in entries):
                raise ConfigError(f"model.{node_id}: every instance must be a mapping.")
        elif not isinstance(settings, Mapping):
            raise ConfigError(f"model.{node_id}: must be a mapping.")
    return value


def _dump(document: Mapping[str, object]) -> str:
    return yaml.safe_dump(
        _plain(document),
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=False,
    )


def _claims(model: Mapping[str, object], graph: SignalGraph) -> tuple[str, ...]:
    claimed: set[str] = set()
    for node_id, settings in model.items():
        if node_id not in graph.nodes or node_id in _MODEL_SECTION_KEYS:
            continue
        node = graph.nodes[node_id]
        if node.many or not isinstance(settings, Mapping) or "compose" in settings:
            claimed.add(node_id)
            continue
        at = settings.get("at")
        if "python" in settings and isinstance(at, str) and at in graph.nodes:
            claimed.add(at)
        elif (
            "python" in settings
            and not isinstance(at, str | bytes)
            and isinstance(at, Sequence)
            and at
            and all(isinstance(item, str) and item in graph.nodes for item in at)
        ):
            claimed.update(at)
        else:
            claimed.add(node_id)
    return tuple(node_id for node_id in graph._topo if node_id in claimed)


def _instances(
    node_id: str,
    value: object,
    catalog: FormCatalog | None = None,
    resources: Mapping[str, object] | None = None,
) -> tuple[NodeInstance, ...]:
    """The ordered instances of a ``many`` node, each with its typed view.

    The identities stay here and the field sets come from the projector, which
    walks the same settings in the same order -- so the two are zipped rather
    than each deciding separately what an instance is.

    ``value`` must already be PLAIN. ``NodeField.written`` carries what the
    document wrote, verbatim, and ``api.py`` returns the whole snapshot
    through ``dataclasses.asdict``, which DEEP-COPIES: a parser-owned
    ``mappingproxy`` arriving here failed the copy and took the entire session
    route down with a 500, for any document that configured a many node.
    """
    catalog = widget_catalog() if catalog is None else catalog
    projected = project_node_instances(node_id, value, catalog, resources=resources)
    if node_id == "cal_loads" and isinstance(value, Mapping):
        identities = [(str(label), str(label), settings) for label, settings in value.items()]
    elif not isinstance(value, str | bytes) and isinstance(value, Sequence):
        identities = [
            (f"{node_id}_{index}", f"{node_id.replace('_', ' ')} {index}", item)
            for index, item in enumerate(value, start=1)
        ]
    else:
        return ()
    if len(projected) != len(identities):
        return ()
    return tuple(
        NodeInstance(
            instance_id=instance_id,
            label=label,
            settings=_plain(settings),
            slot=typed.slot,
            typed_form=typed.typed_form,
            typed_form_reason=typed.typed_form_reason,
            type_choices=typed.type_choices,
            selected_type=typed.selected_type,
            fields=typed.fields,
            extra_keys=typed.extra_keys,
            removed_by_type=typed.removed_by_type,
        )
        for (instance_id, label, settings), typed in zip(identities, projected, strict=True)
    )


def _stages(
    node_id: str,
    value: object,
    catalog: FormCatalog,
    resources: Mapping[str, object] | None = None,
) -> tuple[NodeInstance, ...]:
    """A composed node's stages, each with the typed view of its operator.

    A stage is one operator's settings and carries a ``name:`` that addresses
    it in the path grammar, so the name is its label here rather than one of
    its fields.
    """
    projected = project_compose_stages(node_id, value, catalog, resources=resources)
    if not projected:
        return ()
    stages = value["stages"] if isinstance(value, Mapping) else []
    return tuple(
        NodeInstance(
            instance_id=f"{node_id}_stage_{index + 1}",
            label=(
                stage["name"]
                if isinstance(stage, Mapping) and isinstance(stage.get("name"), str)
                else f"stage {index + 1}"
            ),
            settings=_plain(stage),
            slot=typed.slot,
            typed_form=typed.typed_form,
            typed_form_reason=typed.typed_form_reason,
            type_choices=typed.type_choices,
            selected_type=typed.selected_type,
            fields=typed.fields,
            extra_keys=typed.extra_keys,
            removed_by_type=typed.removed_by_type,
        )
        for index, (stage, typed) in enumerate(zip(stages, projected, strict=True))
    )


def _configuration(
    node_id: str, value: object | None, *, kind: str, reserved: bool, many: bool
) -> str:
    if kind in _COMPOSITION_KINDS:
        return kind
    if reserved:
        return "reserved"
    if many:
        if node_id == "cal_loads":
            return "fan"
        return "chain" if node_id == "filters" else "sum"
    if isinstance(value, Mapping) and "compose" in value:
        return "compose"
    if (
        isinstance(value, Mapping)
        and not isinstance(value.get("at"), str | bytes)
        and isinstance(value.get("at"), Sequence)
        and len(value["at"]) > 1
    ):
        return "region"
    return "single"


def _explanation(node_id: str, *, kind: str, reserved: bool, many: bool, doc: str) -> str:
    if kind == "junction":
        return f"Automatic junction: it adds live branches and is not an operator slot. {doc}"
    if kind == "selector":
        return (
            "Automatic selector: observation.switching chooses one live branch per sample; "
            f"it is not an operator slot. {doc}"
        )
    if reserved:
        return f"Reserved graph slot with no shipped operator; configure it through python:. {doc}"
    if node_id == "cal_loads":
        return "FAN mapping: labels and order are locked to observation.switching.order[1:]."
    if many:
        shape = (
            "CHAIN list; order is execution order."
            if node_id == "filters"
            else "SUM list of independent contributions."
        )
        return f"{shape} {doc}"
    return doc


def _declared_resources(document: Mapping[str, object]) -> Mapping[str, object]:
    """The ``resources:`` block, or an empty one.

    A resource picker offers what the DOCUMENT declares, so a name it lists is
    always one the build can resolve. Read defensively: an invalid
    ``resources:`` is refused elsewhere, and a picker is not the place to
    raise about it.
    """
    found = document.get("resources")
    return found if isinstance(found, Mapping) else {}


def _node_cards(
    model: Mapping[str, object],
    graph: SignalGraph,
    catalog: FormCatalog | None = None,
    resources: Mapping[str, object] | None = None,
) -> tuple[NodeCard, ...]:
    lit = set(_claims(model, graph))
    # Built once for all 33 cards. It is the whole widget census -- about
    # 9 ms -- and one per node would spend a third of a second per diagram.
    catalog = widget_catalog() if catalog is None else catalog
    return tuple(
        NodeCard(
            node_id=node_id,
            label=node_id.replace("_", " "),
            kind=spec.kind,
            description=spec.doc,
            explanation=_explanation(
                node_id,
                kind=spec.kind,
                reserved=spec.reserved,
                many=spec.many,
                doc=spec.doc,
            ),
            editable=spec.kind not in _COMPOSITION_KINDS,
            reserved=spec.reserved,
            many=spec.many,
            segment=spec.segment,
            lit=node_id in lit,
            count=(
                len(model[node_id]) if spec.many and node_id in model else int(node_id in model)
            ),
            configuration=_configuration(
                node_id,
                model.get(node_id),
                kind=spec.kind,
                reserved=spec.reserved,
                many=spec.many,
            ),
            settings=settings,
            typed_form=typed.typed_form,
            typed_form_reason=typed.typed_form_reason,
            type_choices=typed.type_choices,
            selected_type=typed.selected_type,
            fields=typed.fields,
            extra_keys=typed.extra_keys,
            removed_by_type=typed.removed_by_type,
            instances=_instances(node_id, settings, catalog, resources),
            stages=_stages(node_id, settings, catalog, resources),
            from_fields=from_route_fields(
                node_id, settings, catalog, resources=resources
            ),
            stage_names=tuple(
                str(stage.get("name"))
                for stage in model.get(node_id, {}).get("stages", ())
                if isinstance(stage, Mapping) and isinstance(stage.get("name"), str)
            )
            if isinstance(model.get(node_id), Mapping)
            else (),
        )
        for node_id in graph._topo
        for spec in (graph.nodes[node_id],)
        for settings in (_plain(model[node_id]) if node_id in model else None,)
        for typed in (
            project_node_fields(node_id, settings, catalog, resources=resources),
        )
    )


def _graph_counts(graph: SignalGraph, nodes: tuple[NodeCard, ...]) -> GraphCounts:
    lit = {node.node_id for node in nodes if node.lit}
    live: dict[str, bool] = {}
    skipped = 0
    materialized = 0
    for node_id in graph._topo:
        spec = graph.nodes[node_id]
        upstream = sum(bool(live[parent]) for parent in graph._in[node_id])
        if spec.kind == "source":
            live[node_id] = node_id in lit
        elif spec.kind == "transform":
            live[node_id] = node_id in lit or upstream > 0
            if live[node_id] and node_id not in lit and upstream > 0:
                skipped += 1
        else:
            live[node_id] = upstream > 0
            if upstream == 1:
                skipped += 1
            elif upstream > 1:
                materialized += 1
    return GraphCounts(
        lit=len(lit),
        skipped=skipped,
        reserved=sum(node.reserved for node in nodes),
        instances=sum(node.count for node in nodes if node.lit),
        materialized=materialized,
    )


def _diagram(
    name: str,
    model: Mapping[str, object],
    graph: SignalGraph,
    *,
    changed_nodes: tuple[str, ...] = (),
    catalog: FormCatalog | None = None,
    resources: Mapping[str, object] | None = None,
) -> GraphDiagram:
    nodes = _node_cards(model, graph, catalog, resources)
    lit = tuple(node.node_id for node in nodes if node.lit)
    counts = {node.node_id: node.count for node in nodes if node.many and node.count > 1}
    return GraphDiagram(
        name=name,
        svg=graph.to_svg(
            lit=lit,
            title="Rheplicant config signal path" if name == "base" else name,
            counts=counts,
        ),
        nodes=nodes,
        walk_order=graph._topo,
        counts=_graph_counts(graph, nodes),
        changed_nodes=changed_nodes,
    )
_MISSING = object()


def _changed_nodes(base: Mapping[str, object], variant: Mapping[str, object]) -> tuple[str, ...]:
    return tuple(
        node_id
        for node_id in RADIO_GRAPH._topo
        if not _same_value(base.get(node_id, _MISSING), variant.get(node_id, _MISSING))
    )


def _variant_diagrams(
    document: Mapping[str, object],
    base_model: Mapping[str, object],
    catalog: FormCatalog | None = None,
    resources: Mapping[str, object] | None = None,
) -> tuple[GraphDiagram, ...]:
    variants = document.get("variants", {})
    if not isinstance(variants, Mapping):
        raise ConfigError("variants: must be a mapping of name to patch.")
    diagrams: list[GraphDiagram] = []
    for name in variants:
        if not isinstance(name, str):
            raise ConfigError("variants: names must be strings.")
        resolved = apply_variant(document, name)
        variant_model = _model(resolved)
        diagrams.append(
            _diagram(
                name,
                variant_model,
                RADIO_GRAPH,
                changed_nodes=_changed_nodes(base_model, variant_model),
                catalog=catalog,
                resources=_declared_resources(resolved),
            )
        )
    return tuple(diagrams)


def _project(yaml_text: str, document: Mapping[str, object]) -> EditorSnapshot:
    model = _model(document)
    # One census for every diagram this snapshot builds: base, backend and
    # one per variant.
    catalog = widget_catalog()
    # A variant may declare its own resources, so each diagram reads the
    # resources of the layer it was resolved against rather than the base's.
    resources = _declared_resources(document)
    base = _diagram("base", model, RADIO_GRAPH, catalog=catalog, resources=resources)
    backend = _diagram(
        "backend", model, _PROCESSING_GRAPH, catalog=catalog, resources=resources
    )
    forms = project_forms(document)
    return EditorSnapshot(
        yaml_text=yaml_text,
        svg=base.svg,
        nodes=base.nodes,
        walk_order=base.walk_order,
        forms=forms,
        previews=project_previews(document),
        validation=validate_document(yaml_text, document, forms),
        base_diagram=base,
        backend_diagram=backend,
        variant_diagrams=_variant_diagrams(document, model, catalog, resources),
    )


def snapshot(yaml_text: str) -> EditorSnapshot:
    """Parse ``yaml_text`` and project it without changing a byte."""
    document = _load(yaml_text)
    return _project(yaml_text, document)


def replace_yaml(yaml_text: str) -> EditorSnapshot:
    """Validate a bidirectional YAML-mirror replacement."""
    return snapshot(yaml_text)


__all__ = [
    "EditorSnapshot",
    "FormProjection",
    "GraphCounts",
    "GraphDiagram",
    "NodeCard",
    "NodeInstance",
    "replace_yaml",
    "snapshot",
]
