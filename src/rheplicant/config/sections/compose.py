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

from collections.abc import Container, Mapping
from typing import Any

from rheplicant.config.context import ResolutionContext
from rheplicant.config.errors import ConfigError
from rheplicant.config.paths import refuse_misaddressed_region
from rheplicant.config.sections.model import build_node_operator

__all__ = ["build_model", "cal_load_order_problem", "compose_shape_problem",
           "double_count_problem", "many_shape_problem",
           "node_placement_problems", "node_specs", "pipeline_shape_problem",
           "stage_shape_problem"]

_SECTION_KEYS = frozenset({"kind", "acknowledge_double_count"})
_COMPOSE_KEYS = frozenset({"compose", "stages"})


def _graph():
    from rheplicant.radio.graph import RADIO_GRAPH

    return RADIO_GRAPH


def node_specs(section: Mapping) -> dict[str, Any]:
    """The node keys of a ``model:`` section; everything else is section level.

    One binding, two callers: :func:`build_model` below, and
    ``config.preflight.model._nodes``.  The rule here is which keys are NOT
    nodes, and a second copy of it means that the day a third section-level
    key is added the pre-flight pass refuses it as an unknown node id while
    the build accepts it -- the two-validators-for-one-property shape this
    layer has paid for before.
    """
    return {key: value for key, value in section.items()
            if key not in _SECTION_KEYS}


def node_placement_problems(specs: Mapping[str, Any],
                            graph) -> list[tuple[str, str, str]]:
    """Checks A2, A3 and A4 over a ``model:`` mapping -- text and graph only.

    ``(check id, the document path to edit, the message)`` per problem, in the
    document's own key order and **at most one per node**: an id that is not a
    node of the graph has no ``NodeSpec`` to ask about its kind, so the walk
    stops at the first thing wrong with each key.  :func:`build_model` raises
    the first; ``preflight.model._graph_shape`` turns every one into a
    ``Finding``.

    A4 asks ``isinstance(spec, Mapping)`` before ``"type" in spec``, and that
    is a correction rather than a move.  Measured on the loop this function
    replaces: ``model: {beam: 3}`` left :func:`build_model` as a bare
    ``TypeError: argument of type 'int' is not iterable``, and ``model: {beam:
    'type'}`` earned A4 because ``'type' in 'type'`` is True -- an accident of
    Python, not a rule.  Both now reach the refusal that was always right for
    them (``build_node_operator``'s "a node spec is a mapping", and A6's
    "holds a single instance" for a list); in the pre-flight pass the
    ``TypeError`` was worse than a wrong message, because a check that raises
    aborts the whole pass and loses every other finding on the document.

    Nothing here resolves a value, constructs an operator or reads a file --
    which is precisely what lets the pre-flight pass ask these questions
    before ``build_resources`` reads a beam.
    """
    problems: list[tuple[str, str, str]] = []
    for node_id, spec in specs.items():
        if node_id not in graph.nodes:
            problems.append((
                "A2", "model",
                f"model: {node_id!r} is not a node of graph "
                f"{graph.name!r}; known nodes: {list(graph.nodes)}."))
            continue
        node = graph.nodes[node_id]
        if node.kind in ("junction", "selector"):
            problems.append((
                "A3", f"model.{node_id}",
                f"model.{node_id}: is a {node.kind} -- never an operator "
                "slot; it materializes automatically. The switch cycle is "
                "observation.switching."))
            continue
        if node.reserved and isinstance(spec, Mapping) and "type" in spec:
            problems.append((
                "A4", f"model.{node_id}",
                f"model.{node_id}: is reserved -- no shipped operator "
                "registers there; python: is the route."))
    return problems


def many_shape_problem(node_id: str, spec: Any, *, many: bool) -> str | None:
    """Check A6: the shape one node's spec must have, or None if it has it.

    A ``many`` node takes a non-empty list (SUM at ``foregrounds`` and
    ``t_sys_extra``, CHAIN at ``filters``) or, at ``cal_loads``, a label-keyed
    FAN mapping; a single node takes anything that is not a list.
    :func:`_single` and :func:`_many` raise this where they used to write it,
    and the pre-flight pass asks the same question of the raw text.

    **The FAN branch does not ask for non-emptiness and the list branch does**,
    which is deliberate rather than an oversight: an empty ``cal_loads: {}``
    is already refused by :func:`cal_load_order_problem` against
    ``switching.order[1:]``, with a message that names the labels it wanted,
    and a second "non-empty" sentence would be the vaguer of the two.  A list
    node has no such second reader.

    ``cal_loads``' key ORDER is that separate question, with a separate
    answer, and both callers ask this one first.
    """
    if not many:
        if isinstance(spec, list):
            return (f"model.{node_id}: this node holds a single instance; a "
                    "list is the shape of a many node (foregrounds, "
                    "t_sys_extra, cal_loads, filters).")
        return None
    if node_id == "cal_loads":
        if not isinstance(spec, Mapping):
            return ("model.cal_loads: is a label-keyed mapping (FAN) -- the "
                    "keys ARE observation.switching.order[1:], in that "
                    f"order; got {type(spec).__name__}.")
        return None
    if not isinstance(spec, list) or not spec:
        shape = "SUM" if node_id in ("foregrounds", "t_sys_extra") else "CHAIN"
        return (f"model.{node_id}: is a non-empty list ({shape}); got "
                f"{type(spec).__name__} ({spec!r}).")
    return None


def stage_shape_problem(label: str, spec: Any) -> str | None:
    """Return the builder's first refusal for one named stage, if any."""
    if not isinstance(spec, Mapping) or not isinstance(spec.get("name"), str):
        return (
            f"model.{label}: every stage is a mapping with a name: -- the "
            f"path grammar addresses stages by it; got {spec!r}."
        )
    return None


def pipeline_shape_problem(section: Mapping) -> str | None:
    """Return the pure refusal before a pipeline starts building stages."""
    unknown = sorted(set(section) - {"kind", "stages"})
    if unknown:
        return (
            "model: kind: pipeline takes stages: and nothing else; got "
            f"{unknown} too."
        )
    stages = section.get("stages")
    if not isinstance(stages, list) or not stages:
        return (
            "model: kind: pipeline requires stages: -- a non-empty list of "
            "named stage specs."
        )
    return None


def compose_shape_problem(
    node_id: str,
    spec: Mapping,
    node_kind: str,
) -> str | None:
    """Return the pure refusal before a composition starts building stages."""
    unknown = sorted(set(spec) - _COMPOSE_KEYS)
    if unknown:
        return (
            f"model.{node_id}: compose: takes stages: and nothing else; got "
            f"{unknown} too."
        )
    how = spec["compose"]
    if how not in ("cascade", "sum"):
        return (
            f"model.{node_id}: compose: is 'cascade' or 'sum'; got {how!r}."
        )
    if how == "cascade" and node_kind == "source":
        return (
            f"model.{node_id}: compose: cascade chains transforms, and this "
            "is a source node -- sources add, they do not chain; use "
            "compose: sum."
        )
    if how == "sum" and node_kind != "source":
        return (
            f"model.{node_id}: compose: sum adds source contributions, and "
            "this is a transform node -- transforms chain; use "
            "compose: cascade."
        )
    stages = spec.get("stages")
    if not isinstance(stages, list) or len(stages) < 2:
        return (
            f"model.{node_id}: compose: takes stages: -- a list of at least "
            "two named stage specs."
        )
    return None


def cal_load_order_problem(spec: Mapping,
                           switch_order: tuple[str, ...]) -> str | None:
    """Check A14's late leg: the FAN labels ARE ``switching.order[1:]``.

    One binding, two callers: :func:`_many` below, and
    ``config.preflight.model._a14_cal_load_keys``.  A14's other two legs --
    ``order[0]`` being the reserved literal ``antenna`` and a repeated label
    -- are ``switching.declared_order``'s, and measured they already precede
    the beam: ``document.py`` builds the observation before the resources.
    This one did not, because it runs inside ``build_model``, one call after
    a CST directory has been read.

    The spec's SHAPE is :func:`many_shape_problem`'s question and is asked
    first by both callers; this one assumes a mapping.
    """
    if not switch_order:
        return ("model.cal_loads: declared without an observation.switching "
                "order. The switch cycle is what gives each load its index -- "
                "declare switching: {mode: cycle, order: [antenna, ...]} "
                "(schema §4.1.5: mode: none means no loads at all).")
    expected = list(switch_order[1:])
    got = list(spec)
    if got != expected:
        return (f"model.cal_loads: the keys are switching.order[1:] in that "
                f"order -- expected {expected}, got {got}. One list fixes the "
                "switch indices, the load order and the gamma_src rows.")
    return None


def double_count_problem(node_ids: Container[str],
                         acknowledgement: Any) -> str | None:
    """Check A32, decided as D-C13: both ground terms lit, unacknowledged.

    ``is True`` rather than truthiness on purpose:
    ``tests/config/test_config_section_compose.py`` pins that the string
    ``'yes'`` is refused, and an acknowledgement that reads as true by
    accident is the one thing this key exists to prevent.

    The returned message already ends ``(check A32, decided as D-C13).``, so
    the ``Finding`` built from it appends no second citation.
    """
    if not ("beam_spill" in node_ids and "ground_pickup" in node_ids):
        return None
    if acknowledgement is True:
        return None
    return ("model: beam_spill and ground_pickup both lit describe the "
            "ground twice (the spill term and the pickup term overlap); if "
            "that is deliberate, say so: acknowledge_double_count: true "
            "(check A32, decided as D-C13).")


def _stage_operator(label: str, spec: Any, context: ResolutionContext,
                    node_id: str | None):
    """One ``stages:`` entry -> its operator."""
    from rheplicant.config.sections.model import _construct

    problem = stage_shape_problem(label, spec)
    if problem is not None:
        raise ConfigError(problem)
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

    problem = compose_shape_problem(node_id, spec, node_kind)
    if problem is not None:
        raise ConfigError(problem)
    how = spec["compose"]
    stages = spec.get("stages")
    assert isinstance(stages, list)
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

    problem = many_shape_problem(node_id, spec, many=False)
    if problem is not None:
        raise ConfigError(problem)
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
    # The shape first and the order second, in both branches and at both call
    # sites: a `cal_loads: []` reports the FAN shape rather than a key order
    # in a value that has no keys.
    problem = many_shape_problem(node_id, spec, many=True)
    if problem is not None:
        raise ConfigError(problem)
    if node_id == "cal_loads":
        problem = cal_load_order_problem(spec, switch_order)
        if problem is not None:
            raise ConfigError(problem)
        return [build_node_operator("cal_loads", entry, context)
                for entry in spec.values()]
    # foregrounds / t_sys_extra (SUM) and filters (CHAIN): a list.
    return [build_node_operator(node_id, entry, context) for entry in spec]


def _build_pipeline(section: Mapping, context: ResolutionContext):
    from rheplicant.core.pipeline import Pipeline

    problem = pipeline_shape_problem(section)
    if problem is not None:
        raise ConfigError(problem)
    stages = section.get("stages")
    assert isinstance(stages, list)
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
    # The local is `specs`, not `node_specs`: the shared function's name is
    # `node_specs`, and a local shadowing it here would still import, run and
    # pass -- there is no second call in this body to raise on.
    specs = node_specs(section)
    if not specs:
        raise ConfigError(
            "model: declares no nodes. A model lights at least one node of "
            "the signal path."
        )
    problems = node_placement_problems(specs, graph)
    if problems:
        raise ConfigError(problems[0][2])
    problem = double_count_problem(
        specs, section.get("acknowledge_double_count"))
    if problem is not None:
        raise ConfigError(problem)
    operators: list[Any] = []
    for node_id, spec in specs.items():
        if graph.nodes[node_id].many:
            operators.extend(_many(node_id, spec, context, switch_order))
        else:
            operators.append(_single(node_id, spec, context,
                                     graph.nodes[node_id].kind))
    from rheplicant.radio import assemble

    return assemble(*operators)
