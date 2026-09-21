"""The one node walk the model checks share.

The ruling on this file was that its passes are "a dozen independent passes
sharing one node walk; the walk is the module and the passes are not". This
is the walk: the graph, the nodes an `at:` claims, the entries, the shape and
the maturity level of what a node resolves to. Every pass beside it reads
this and none of them reads each other.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from _rheplicant_bootstrap.path_syntax import longest_legal_prefix
from rheplicant.config.capability_record import at_level, node_levels
from rheplicant.config.findings import Finding, refuse, report
from rheplicant.config.preflight import register
from rheplicant.config.sections.compose import (
    double_count_problem,
    many_shape_problem,
    model_nodes,
    node_placement_problems,
)
from rheplicant.config.sections.model import (
    ambiguous_class_problem,
    operator_table,
)
from rheplicant.core.capability import Maturity


def _t4_graph():
    """``RADIO_GRAPH``, imported where it is used.

    ``compose.py``'s own convention.  Measured, it costs nothing at call time
    -- ``import rheplicant.config`` already imports ``rheplicant.radio``
    through ``projectors.py``, a module-scope ``from
    rheplicant.radio import ...`` -- and deferring it means this module is not
    the one that pins that, should it ever stop being true.
    """
    from rheplicant.radio.graph import RADIO_GRAPH

    return RADIO_GRAPH


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


def _t4_entries(node_id: str, spec: Any, *, many: bool) -> list[tuple[str, Any]]:
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
            return [(f"{node_id}[{index}]", entry) for index, entry in enumerate(spec)]
        if isinstance(spec, Mapping):
            return [(f"{node_id}.{label}", entry) for label, entry in spec.items()]
        return []
    if isinstance(spec, Mapping) and "compose" in spec:
        stages = spec.get("stages")
        if not isinstance(stages, list):
            return []
        return [(f"{node_id}.stages[{index}]", entry) for index, entry in enumerate(stages)]
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
    specs = model_nodes(document)
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
                yield refuse("A7", longest_legal_prefix(f"model.{where}"), f"{problem} (check A7).")


@register("A32")
def _double_count(document: Mapping[str, Any]) -> Iterable[Finding]:
    """Check A32: ``beam_spill`` and ``ground_pickup`` both lit, unacknowledged.

    The message arrives with its own citation -- it ends ``(check A32, decided
    as D-C13).`` -- so nothing is appended to it.
    """
    section = document.get("model")
    if not isinstance(section, Mapping):
        return
    problem = double_count_problem(model_nodes(document), section.get("acknowledge_double_count"))
    if problem is not None:
        yield refuse("A32", "model", problem)


@register("A53")
def _capability_level(document: Mapping[str, Any]) -> Iterable[Finding]:
    """Check A53: say so when a document's physics is not all maintained.

    **One finding per DOCUMENT, naming every node, not one per node.** The
    first version emitted one each and a four-node document earned four
    notices; seventeen of the twenty-nine shipped operators are placeholders,
    so a realistic document earned a column of them. A notice that repeats is
    a notice a reader learns to skip, which costs exactly the nodes it was
    written for -- and it would have rewritten `docs/config-validation.md`,
    the page that teaches people to read a report, into a list of them.

    Informational, never a refusal and never a warning. A placeholder's
    CONTRACT is real -- shapes, purity, PRNG consumption, ordering -- so a
    document that places one is doing nothing wrong; what it must not do is
    read the numbers as physics. ``report`` is the severity that says exactly
    that: "worth recording beside the run; not worth interrupting anyone
    over". This is its first producer.

    The levels are READ, never decided here. They come off the ``maturity``
    ClassVar through :func:`rheplicant.radio.capabilities`, the one walk every
    view shares, so a node whose physics arrives leaves this sentence in the
    same commit that raises its level.

    Nodes whose class cannot be resolved WITHOUT IMPORTING anything are
    skipped in silence, the same decline :func:`_t5_radio_class` documents; a
    decline can only lose a notice, never invent one. An ambiguous or
    misspelled ``type:`` is A7's to refuse and is not re-reported here.

    **The resolution itself moved out on 2026-09-20 and this check now reads
    it.** ``capabilities.json`` publishes the same node-to-level answer as a
    machine-readable record, and two copies of that resolution would be two
    things that can disagree about the same document -- a notice naming a node
    the published record calls maintained, with nothing rendering the two side
    by side. :func:`~rheplicant.config.capability_record.node_levels` is the
    one resolution; this check is a view of it that keeps its own editorial
    decisions, which are what levels are worth a sentence and how to word it.
    """
    rows = node_levels(document)
    if not rows:
        return
    placeholders = [f"{row.node_id} ({row.type})" for row in at_level(rows, Maturity.PLACEHOLDER)]
    experimental = [f"{row.node_id} ({row.type})" for row in at_level(rows, Maturity.EXPERIMENTAL)]
    if not placeholders and not experimental:
        return
    parts = []
    if placeholders:
        parts.append(
            f"placeholder physics at {', '.join(sorted(placeholders))} -- the "
            "contract is real and tested, the NUMBERS are a stand-in, so "
            "results through these nodes are not predictions"
        )
    if experimental:
        parts.append(
            f"experimental physics at {', '.join(sorted(experimental))} -- "
            "usable with stated limits and no general validity guarantee"
        )
    yield report("A53", "model", f"model: {'; '.join(parts)}. (check A53.)")
