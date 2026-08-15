"""``model:``, checked from the document's text (schema §6, A2-A8, A14, A30-A33).

Every check here is a second CALL SITE, never a second implementation: the
graph-shaped rules live in ``sections/compose.py`` and ``sections/model.py``
as module-level pure functions, this module hands them data read off the raw
document, and the build hands them the same data one phase later.  What moves
is the PHASE.  ``document.py`` builds the resources -- where a CST directory
is read and the spherical harmonic transform runs -- before it builds the
model, so every one of these refusals used to arrive after the beam was
analysed.

Three sources and no fourth (§2.4): the document mapping, ``RADIO_GRAPH``, and
operator classes resolved BY NAME off ``rheplicant.radio``.  No value node is
resolved, no operator is constructed, no file is read.  Measured (median of
2000 calls in one process), the three checks here answer a six-node document
in **1.8e-05 s**; the first call is **3.9e-04 s**, which is
``sections/switching``'s deferred import.  §0.1's budget for the whole pass is
0.05 s, against ``load_document``'s 1.536 s on a toy beam.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from rheplicant.config.errors import ConfigError
from rheplicant.config.findings import Finding, refuse
from rheplicant.config.preflight import register
from rheplicant.config.preflight.document import _task3_where
from rheplicant.config.sections.compose import (
    cal_load_order_problem,
    double_count_problem,
    many_shape_problem,
    node_placement_problems,
    node_specs,
)
from rheplicant.config.sections.model import (
    ambiguous_class_problem,
    operator_table,
)


def _t4_graph():
    """``RADIO_GRAPH``, imported where it is used.

    ``compose.py``'s own convention.  Measured, it costs nothing at call time
    -- ``import rheplicant.config`` already imports ``rheplicant.radio``
    through ``config/kinds/projectors.py:44``, a module-scope ``from
    rheplicant.radio import ...`` -- and deferring it means this module is not
    the one that pins that, should it ever stop being true.
    """
    from rheplicant.radio.graph import RADIO_GRAPH

    return RADIO_GRAPH


def _nodes(document: Mapping[str, Any]) -> dict[str, Any]:
    """The ``model:`` section's node specs, or ``{}``.

    EVERY model check in Tasks 4, 5 and 11 starts here, and none of them
    defines its own (§3.1, rule 1).  ``{}`` for the three shapes that declare
    no graph nodes at all: no ``model:`` at all (which ``_structural`` refuses
    before any check runs, but this is called directly too), a ``model:`` that
    is not a mapping (the build refuses it, with the type it got), and ``kind:
    pipeline``, which has no node registry -- reading a pipeline's ``stages:``
    as node ids would report every stage as an unknown node.
    """
    section = document.get("model")
    if not isinstance(section, Mapping):
        return {}
    if section.get("kind", "graph") != "graph":
        return {}
    return node_specs(section)


def _t4_at_nodes(spec: Any) -> tuple[str, ...]:
    """The node ids one spec's ``at:`` claims -- ``()`` when it claims none.

    Bound here because :func:`_lit` (this task) and Task 5's A5 must read
    ``at:`` the same way; a second reading of it is the collision shape §3.1
    names.  A malformed ``at:`` yields ``()`` and is refused at the build --
    ``_single`` answers "at: is a node id or a list of node ids" with the
    shape it got -- rather than being reinterpreted here.

    **This reads a SINGLE NODE's spec and nothing else**, which is the
    caller's job to know: ``_single`` is the only place ``at:`` is honoured,
    and it is reached only for a non-``many`` node with no ``compose:``.  A
    ``many`` node's entries go straight to ``build_node_operator``, where
    ``at`` is an unknown constructor field, and a ``compose:`` block is
    refused for the same key -- so an ``at:`` in either place places nothing.
    """
    if not isinstance(spec, Mapping):
        return ()
    at = spec.get("at")
    if isinstance(at, str):
        return (at,)
    if isinstance(at, list) and all(isinstance(node, str) for node in at):
        return tuple(at)
    return ()


def _lit(document: Mapping[str, Any]) -> frozenset[str]:
    """The node ids this document lights, as ``Assembly.lit`` will report them.

    The keys that are graph nodes, plus every node named in an ``at:`` claim.
    The keys alone are exact for every document that BUILDS -- a single-node
    ``at:`` must restate its own key (``_single``), a region is keyed by its
    last covered node (``paths.refuse_misaddressed_region``), and a shipped
    class always lands at the node its key names.  The two shapes where a key
    and its operator's node come apart are exactly what check A5 refuses
    (Task 5), so this reading is right wherever the document is.  The ``at:``
    union is what makes a region's INTERIOR nodes -- covered, lit, and never a
    key -- visible to A8's reachability question; measured, an ``at:
    [noise_wave, cw_tone]`` region reports both in ``Assembly.lit``, and
    ``test_lit_is_what_the_assembly_itself_reports`` is that comparison.

    **``at:`` is read only where ``_single`` reads it**: at a key that is a
    graph node, is not ``many``, and declares no ``compose:``.  Measured, a
    walk that read every spec lit nodes nothing lights -- ``cal_loads: {at:
    'noise_wave'}`` is a FAN mapping whose LABEL happens to be ``at`` and
    whose value is a load, and ``noise: {compose: cascade, stages: [...],
    at: [...]}`` is a block ``_compose`` refuses for the unknown key.  Both
    reported placements the assembly would never make, and §3.1 makes this
    the binding Tasks 5 and 11 start from, so a wrong answer here is wrong
    three times.
    """
    graph = _t4_graph()
    lit: set[str] = set()
    for key, spec in _nodes(document).items():
        if key not in graph.nodes:
            continue
        lit.add(key)
        if graph.nodes[key].many:
            continue
        if isinstance(spec, Mapping) and "compose" in spec:
            continue
        lit.update(node for node in _t4_at_nodes(spec) if node in graph.nodes)
    return frozenset(lit)


def _t4_entries(node_id: str, spec: Any, *,
                many: bool) -> list[tuple[str, Any]]:
    """``(the document path, the spec)`` per operator a node key declares.

    A single node declares one; a ``many`` node declares one per list entry or
    per FAN label, and the path is the entry's -- ``model.filters[1]`` -- so a
    three-filter chain sends the reader to the line to edit rather than to the
    node.

    ``compose:`` is the shape that has to be expanded rather than passed on:
    the stages are what reach ``build_node_operator``, and the composing
    mapping itself never does.  Measured, a ``noise`` composing two typed
    stages builds, so a walk that asked the composing mapping for a ``type:``
    would refuse a document the build accepts.
    """
    if many:
        if isinstance(spec, list):
            return [(f"{node_id}[{index}]", entry)
                    for index, entry in enumerate(spec)]
        if isinstance(spec, Mapping):
            return [(f"{node_id}.{label}", entry)
                    for label, entry in spec.items()]
        return []
    if isinstance(spec, Mapping) and "compose" in spec:
        stages = spec.get("stages")
        if not isinstance(stages, list):
            return []
        return [(f"{node_id}.stages[{index}]", entry)
                for index, entry in enumerate(stages)]
    return [(node_id, spec)]


@register("A2", "A3", "A4", "A6", "A7")
def _graph_shape(document: Mapping[str, Any]) -> Iterable[Finding]:
    """Checks A2, A3, A4, A6 and A7 -- the graph-shaped rules, from text.

    Registered under **all five ids it decides**, variadically (§3.1).  One
    function may carry several ids and ``preflight`` de-duplicates by function
    identity, so this still runs once; what the four extra slots buy is that
    ``register`` refuses a later function claiming A3, A4, A6 or A7 --
    measured before the change, all four were accepted, and a second A3 would
    have put two voices for one check in one report.  ``Report.checks()`` is
    unaffected either way: it reads the ids off the findings, not off the
    registry.  Run order is the order of FIRST binding, still ``"A2"``.

    One walk, because the questions are ordered -- an id that is not a node
    has no kind, a node whose shape is wrong has no entries to ask about a
    class.

    The ``node_id not in graph.nodes`` skip below **reads as redundant and is
    load-bearing**, which is worth a sentence because the redundant reading is
    the tempting one: :func:`node_placement_problems` has indeed already
    yielded A2 for such an id, but this loop walks the same ``specs`` mapping
    afterwards, so ``graph.nodes[node_id]`` on the next line is a ``KeyError``
    on the very first document that names a node that does not exist.
    ``preflight`` turns that into "check 'A2' RAISED KeyError" and loses every
    other finding on the document (§2.3's TRAP).  Measured: deleting the two
    lines turns eleven tests red across two modules, one of them
    ``test_preflight_document.py`` -- the skip is load-bearing for Task 3's
    checks as well as for this module's, because the crash takes the pass
    down and everything registered after it with it.
    """
    graph = _t4_graph()
    specs = _nodes(document)
    table = None
    for check, where, message in node_placement_problems(specs, graph):
        yield refuse(check, where, f"{message} (check {check}).")
    for node_id, spec in specs.items():
        if node_id not in graph.nodes:
            continue
        node = graph.nodes[node_id]
        if node.kind in ("junction", "selector"):
            continue
        problem = many_shape_problem(node_id, spec, many=node.many)
        if problem is not None:
            yield refuse("A6", f"model.{node_id}", f"{problem} (check A6).")
            continue
        if table is None:
            table = operator_table()
        classes = table.get(node_id)
        if not classes:
            continue
        for where, entry in _t4_entries(node_id, spec, many=node.many):
            if not isinstance(entry, Mapping):
                continue
            if "python" in entry or "from" in entry:
                # Neither route reaches `_pick_class`: the class is named
                # outright, or the node's `from:` route names it.  Asking
                # either for a `type:` refuses a document the build accepts.
                continue
            problem = ambiguous_class_problem(node_id, classes, entry)
            if problem is not None:
                # A FAN label is a name the user chose, and an un-spellable
                # `where` kills the pass from outside its per-check `try`.
                # Unreachable while `cal_loads` is the only FAN node and
                # registers one class; live the day a second one ships.
                yield refuse("A7", _task3_where(f"model.{where}"),
                             f"{problem} (check A7).")


@register("A32")
def _double_count(document: Mapping[str, Any]) -> Iterable[Finding]:
    """Check A32: ``beam_spill`` and ``ground_pickup`` both lit, unacknowledged.

    The message arrives with its own citation -- it ends ``(check A32, decided
    as D-C13).`` -- so nothing is appended to it.
    """
    section = document.get("model")
    if not isinstance(section, Mapping):
        return
    problem = double_count_problem(
        _nodes(document), section.get("acknowledge_double_count"))
    if problem is not None:
        yield refuse("A32", "model", problem)


def _t4_switch_order(document: Mapping[str, Any]) -> tuple[str, ...] | None:
    """``observation.switching``'s declared order, or ``None`` for "cannot say".

    ``()`` and ``None`` are different answers and the difference is the whole
    point: ``()`` is a document that declares no switch cycle, which is what
    makes ``model.cal_loads`` an error; ``None`` is a ``switching:`` block
    this layer cannot read, whose own refusal is
    ``switching.declared_order``'s or ``compile_switching``'s and already
    precedes the beam (``document.py`` builds the observation before the
    resources).  Guessing an order out of a malformed block would answer A14
    about a cycle the document does not declare.

    ``declared_order`` is the section's own reader and is CALLED, never
    re-implemented (§2.5).  The mode dispatch and the per-mode key sweep are
    not: they are read inline off ``compile_switching``'s private ``_KEYS``,
    which is a second READING of the grammar rather than a second call site,
    because ``compile_switching`` resolves value nodes and P-1 may not.  That
    is the seam to watch -- it is where the unhashable-``mode`` crash below
    lived -- and the ``_KEYS`` import is what keeps the key table itself from
    being copied.

    **``switching:`` has TWO grammars and reading it with one of them refuses
    a document that builds.**  Measured at ``f303af8``, and found by
    ``test_config_section_model.py``'s thermistor document rather than by
    reading: an INGESTED run (``observation.from_file``) declares ``order:``
    ALONE and no ``mode:`` at all, because the recording carries the cycle
    (``observation.py:336-348``), while every other run goes through
    ``compile_switching``, where a missing ``mode:`` means ``none``.  A
    single reading calls that run's three-label order "no switch cycle" and
    tells it to declare the cycle it has already declared.

    **A ``mode:`` that is not a string is nobody's refusal yet**, which is why
    it is rejected here by type rather than by lookup.  An unknown mode and a
    key the mode does not take are both ``compile_switching``'s, and both
    already precede the beam; ``mode: []`` is neither -- measured at
    ``f303af8`` it left ``compile_switching`` as a bare ``TypeError:
    unhashable type: 'list'``, and a ``_KEYS.get(mode)`` here inherits that
    one phase earlier, where it aborts the whole pass (§2.3's TRAP) and costs
    the document every other finding.
    """
    from rheplicant.config.sections.switching import _KEYS, declared_order

    observation = document.get("observation")
    if not isinstance(observation, Mapping):
        return None
    spec = observation.get("switching")
    if spec is None:
        return ()
    if not isinstance(spec, Mapping):
        return None
    if "from_file" in observation:
        if set(spec) - {"order"}:
            return None
    else:
        mode = spec.get("mode", "none")
        if not isinstance(mode, str):
            return None
        allowed = _KEYS.get(mode)
        if allowed is None or set(spec) - allowed:
            return None
        if mode == "none":
            return ()
    try:
        return declared_order(spec)
    except ConfigError:
        return None


@register("A14.cal_loads")
def _a14_cal_load_keys(document: Mapping[str, Any]) -> Iterable[Finding]:
    """A14's late leg: ``model.cal_loads``' keys ARE ``switching.order[1:]``.

    §3.2 (i)'s, and it reads as Task 6's.  Measured: A14's other two legs are
    ``declared_order``'s, which runs inside ``build_observation`` -- before
    ``build_resources`` -- so Task 6 hoists neither.  This one runs inside
    ``_many``, one call after a CST directory has been read, and ``_many``
    lives in this task's file.

    The slot is dotted (§3.2 (a)): Task 6 binds ``A14`` for the direction this
    one does not decide -- an order that names loads with no
    ``model.cal_loads`` at all -- and two functions cannot claim one slot.
    ``Finding.check`` stays the bare ``A14``.
    """
    spec = _nodes(document).get("cal_loads")
    if not isinstance(spec, Mapping):
        # Not declared, or declared with the wrong shape -- which is A6's
        # sentence, yielded by `_graph_shape`, and the build asks the shape
        # first too.  Two sentences about one key is one too many.
        return
    order = _t4_switch_order(document)
    if order is None:
        return
    problem = cal_load_order_problem(spec, order)
    if problem is not None:
        yield refuse("A14", "model.cal_loads", f"{problem} (check A14).")
