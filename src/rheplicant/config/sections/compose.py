"""``model:`` -> the assembled twin (schema §4.5.3-§4.5.6).

The section-level rules live here: which nodes exist, which are junction /
selector / reserved, the three ``many`` shapes (SUM list, FAN mapping in
switch order, CHAIN list), ``compose:`` at one node, ``at:`` placement for
``python:`` operators, ``snapshot_before:``, D-C13's acknowledgement, and the
non-graph ``kind: pipeline``. ``assemble()``'s own refusals (duplicate
placement, ``must_precede``, region validity) pass through untranslated --
those messages already name their remedy.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from rheplicant.config.context import ResolutionContext
from rheplicant.config.errors import ConfigError
from rheplicant.config.paths import refuse_misaddressed_region
from rheplicant.config.sections.model import build_node_operator

__all__ = ["build_model"]

_SECTION_KEYS = frozenset({"kind", "acknowledge_double_count"})
_COMPOSE_KEYS = frozenset({"compose", "stages"})


def _graph():
    from rheplicant.radio.graph import RADIO_GRAPH

    return RADIO_GRAPH


def _stage_operator(label: str, spec: Any, context: ResolutionContext,
                    node_id: str | None):
    """One ``stages:`` entry -> its operator."""
    from rheplicant.config.sections.model import _construct

    if not isinstance(spec, Mapping) or not isinstance(spec.get("name"), str):
        raise ConfigError(
            f"model.{label}: every stage is a mapping with a name: -- the "
            f"path grammar addresses stages by it; got {spec!r}."
        )
    spec = {key: value for key, value in spec.items() if key != "name"}
    if node_id is not None:
        return build_node_operator(node_id, spec, context)
    # kind: pipeline -- no node registry; the class is named directly.
    if "python" in spec:
        return build_node_operator(label, spec, context)
    declared = spec.get("type")
    if declared is None:
        raise ConfigError(
            f"model.{label}: a pipeline stage names its class -- type: "
            "(or python:) is required, because there is no graph node to "
            "choose it from."
        )
    import rheplicant.radio as radio
    from rheplicant.core.operator import AbstractOperator

    target = getattr(radio, declared, None)
    if not (isinstance(target, type) and issubclass(target, AbstractOperator)):
        raise ConfigError(
            f"model.{label}: type: {declared!r} is not an operator exported "
            "by rheplicant.radio."
        )
    remaining = {key: value for key, value in spec.items() if key != "type"}
    return _construct(label, target, remaining, context)


def _compose(node_id: str, spec: Mapping, context: ResolutionContext,
             node_kind: str):
    from rheplicant.core.combinators import SumOperator
    from rheplicant.core.graph import At
    from rheplicant.core.pipeline import Pipeline

    unknown = sorted(set(spec) - _COMPOSE_KEYS)
    if unknown:
        raise ConfigError(
            f"model.{node_id}: compose: takes stages: and nothing else; got "
            f"{unknown} too."
        )
    how = spec["compose"]
    if how not in ("cascade", "sum"):
        raise ConfigError(
            f"model.{node_id}: compose: is 'cascade' or 'sum'; got {how!r}."
        )
    if how == "cascade" and node_kind == "source":
        raise ConfigError(
            f"model.{node_id}: compose: cascade chains transforms, and this "
            "is a source node -- sources add, they do not chain; use "
            "compose: sum."
        )
    if how == "sum" and node_kind != "source":
        raise ConfigError(
            f"model.{node_id}: compose: sum adds source contributions, and "
            "this is a transform node -- transforms chain; use "
            "compose: cascade."
        )
    stages = spec.get("stages")
    if not isinstance(stages, list) or len(stages) < 2:
        raise ConfigError(
            f"model.{node_id}: compose: takes stages: -- a list of at least "
            "two named stage specs."
        )
    names = [entry.get("name") if isinstance(entry, Mapping) else None
             for entry in stages]
    operators = [
        _stage_operator(f"{node_id}.stages[{i}]", entry, context, node_id)
        for i, entry in enumerate(stages)
    ]
    if how == "cascade":
        return At(node_id, Pipeline(*operators, names=names))
    return At(node_id, SumOperator(*operators, names=names))


def _single(node_id: str, spec: Any, context: ResolutionContext,
            node_kind: str):
    from rheplicant.core.graph import At
    from rheplicant.core.operator import SnapshotOperator
    from rheplicant.core.pipeline import Pipeline

    if isinstance(spec, list):
        raise ConfigError(
            f"model.{node_id}: this node holds a single instance; a list is "
            "the shape of a many node (foregrounds, t_sys_extra, cal_loads, "
            "filters)."
        )
    if not isinstance(spec, Mapping):
        raise ConfigError(
            f"model.{node_id}: a node spec is a mapping; got "
            f"{type(spec).__name__} ({spec!r})."
        )
    if "compose" in spec:
        return _compose(node_id, spec, context, node_kind)
    spec = dict(spec)
    at_spec = spec.pop("at", None)
    snapshot = spec.pop("snapshot_before", None)
    if at_spec is not None and snapshot is not None:
        raise ConfigError(
            f"model.{node_id}: at: and snapshot_before: together are not a "
            "combination this layer writes -- relocate first, snapshot at "
            "the destination."
        )
    if at_spec is not None:
        if "python" not in spec:
            raise ConfigError(
                f"model.{node_id}: at: places an operator that declares no "
                "graph node of its own -- a python: operator. A shipped "
                "class already declares its node; assemble it there."
            )
        operator = build_node_operator(node_id, spec, context)
        if isinstance(at_spec, str):
            if at_spec != node_id:
                raise ConfigError(
                    f"model.{node_id}: at: {at_spec!r} disagrees with the "
                    "node key -- a single-node at: restates its own key."
                )
            return At(node_id, operator)
        if (not isinstance(at_spec, list)
                or not all(isinstance(n, str) for n in at_spec)):
            raise ConfigError(
                f"model.{node_id}: at: is a node id or a list of node ids; "
                f"got {at_spec!r}."
            )
        refuse_misaddressed_region(node_id, at_spec)
        return At(tuple(at_spec), operator)
    operator = build_node_operator(node_id, spec, context)
    if snapshot is not None:
        if not isinstance(snapshot, str) or not snapshot:
            raise ConfigError(
                f"model.{node_id}: snapshot_before: is the snapshot's name, "
                f"a non-empty string; got {snapshot!r}."
            )
        return At(node_id, Pipeline(SnapshotOperator(name=snapshot), operator,
                                    names=("snapshot", node_id)))
    return operator


def _many(node_id: str, spec: Any, context: ResolutionContext,
          switch_order: tuple[str, ...]):
    if node_id == "cal_loads":
        if not isinstance(spec, Mapping):
            raise ConfigError(
                "model.cal_loads: is a label-keyed mapping (FAN) -- the keys "
                "ARE observation.switching.order[1:], in that order; got "
                f"{type(spec).__name__}."
            )
        if not switch_order:
            raise ConfigError(
                "model.cal_loads: declared without an observation.switching "
                "order. The switch cycle is what gives each load its index -- "
                "declare switching: {mode: cycle, order: [antenna, ...]} "
                "(schema §4.1.5: mode: none means no loads at all)."
            )
        expected = list(switch_order[1:])
        got = list(spec)
        if got != expected:
            raise ConfigError(
                f"model.cal_loads: the keys are switching.order[1:] in that "
                f"order -- expected {expected}, got {got}. One list fixes the "
                "switch indices, the load order and the gamma_src rows."
            )
        return [build_node_operator("cal_loads", entry, context)
                for entry in spec.values()]
    # foregrounds / t_sys_extra (SUM) and filters (CHAIN): a list.
    if not isinstance(spec, list) or not spec:
        shape = "SUM" if node_id in ("foregrounds", "t_sys_extra") else "CHAIN"
        raise ConfigError(
            f"model.{node_id}: is a non-empty list ({shape}); got "
            f"{type(spec).__name__} ({spec!r})."
        )
    return [build_node_operator(node_id, entry, context) for entry in spec]


def _build_pipeline(section: Mapping, context: ResolutionContext):
    from rheplicant.core.pipeline import Pipeline

    unknown = sorted(set(section) - {"kind", "stages"})
    if unknown:
        raise ConfigError(
            f"model: kind: pipeline takes stages: and nothing else; got "
            f"{unknown} too."
        )
    stages = section.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ConfigError(
            "model: kind: pipeline requires stages: -- a non-empty list of "
            "named stage specs."
        )
    names = [entry.get("name") if isinstance(entry, Mapping) else None
             for entry in stages]
    operators = [
        _stage_operator(f"stages[{i}]", entry, context, None)
        for i, entry in enumerate(stages)
    ]
    return Pipeline(*operators, names=names)


def build_model(section: Any, context: ResolutionContext, *,
                switch_order: tuple[str, ...]):
    """The ``model:`` section -> an assembled twin (or a Pipeline)."""
    if not isinstance(section, Mapping):
        raise ConfigError(
            f"model: is a mapping; got {type(section).__name__} ({section!r})."
        )
    kind = section.get("kind", "graph")
    if kind == "pipeline":
        return _build_pipeline(section, context)
    if kind != "graph":
        raise ConfigError(
            f"model: kind: is 'graph' (the default) or 'pipeline'; got "
            f"{kind!r}."
        )
    graph = _graph()
    node_specs = {key: value for key, value in section.items()
                  if key not in _SECTION_KEYS}
    if not node_specs:
        raise ConfigError(
            "model: declares no nodes. A model lights at least one node of "
            "the signal path."
        )
    for node_id in node_specs:
        if node_id not in graph.nodes:
            raise ConfigError(
                f"model: {node_id!r} is not a node of graph "
                f"{graph.name!r}; known nodes: {list(graph.nodes)}."
            )
        node = graph.nodes[node_id]
        if node.kind in ("junction", "selector"):
            raise ConfigError(
                f"model.{node_id}: is a {node.kind} -- never an operator "
                "slot; it materializes automatically. The switch cycle is "
                "observation.switching."
            )
        if node.reserved and "type" in node_specs[node_id]:
            raise ConfigError(
                f"model.{node_id}: is reserved -- no shipped operator "
                "registers there; python: is the route."
            )
    if ("beam_spill" in node_specs and "ground_pickup" in node_specs
            and section.get("acknowledge_double_count") is not True):
        raise ConfigError(
            "model: beam_spill and ground_pickup both lit describe the "
            "ground twice (the spill term and the pickup term overlap); if "
            "that is deliberate, say so: acknowledge_double_count: true "
            "(check A32, decided as D-C13)."
        )
    operators: list[Any] = []
    for node_id, spec in node_specs.items():
        if graph.nodes[node_id].many:
            operators.extend(_many(node_id, spec, context, switch_order))
        else:
            operators.append(_single(node_id, spec, context,
                                     graph.nodes[node_id].kind))
    from rheplicant.radio import assemble

    return assemble(*operators)
