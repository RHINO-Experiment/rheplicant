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
resolved, no operator is constructed, no file is read.  Re-measured with Task
5's three checks added (median of 2000 calls in one process, two runs): the
**six** checks here answer a **six**-node document -- ``preflight_helpers``'
base model, which has four, plus ``foregrounds:`` and ``bandpass:`` -- in
**2.5e-05 s**; the first call is **2.2e-04 s**.  §0.1's budget for the whole
pass is 0.05 s, against ``load_document``'s 1.536 s on a toy beam.

**That first call is not a deferred import**, which is what this file said
until Task 5 checked: pre-importing ``sections/switching``, ``core/fold`` and
``rheplicant.radio`` leaves it at 2.3e-04 s unchanged.  It is
``sections/model.operator_table()``, measured at 1.7e-04 s the first time and
2.3e-05 s every time after -- it is rebuilt per call rather than cached, which
is also most of the steady-state figure above.  Recorded rather than fixed:
the table is ``sections/model``'s and no task in this plan owns it.
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

    **The node an entry lands on, never the key it is written under.**  Those
    two come apart far more often than this function's first version claimed:
    it said "a shipped class always lands at the node its key names", and
    measured at ``1556ff8`` that is false for every ``python:`` relocation --
    ``{global_signal: …, noise: {python: 'rheplicant.radio:GainOperator'}}``
    assembles with ``lit == ('gain', 'global_signal')``, and a one-element
    ``at: ['gain']`` written under ``bandpass:`` does the same.  It also said
    the exceptions "are exactly what check A5 refuses", which is not so: A5
    refuses a relocation only when it COLLIDES, and a relocation onto an empty
    node builds.  The consequence was two live refusals of documents the
    package accepts -- A31 naming a source node whose operator had left it,
    and A8 asserting that a document "lights" a stage it does not.

    So the placement is resolved rather than assumed: :func:`_t5_claims` is
    the one reader of that, and this is its third caller.  A region answer
    (two or more ids) lights every node it covers plus the key -- measured, an
    ``at: [noise_wave, cw_tone]`` region reports both in ``Assembly.lit``, and
    that is what makes a region's INTERIOR nodes, covered and never a key,
    visible to A8's reachability question.  ``test_lit_is_what_the_assembly_
    itself_reports`` holds all six shapes against a real assembly.

    An entry that places NOTHING lights nothing: a ``python:`` target this
    layer will not import (measured, ``AssemblyError``: no ``graph_node`` and
    no ``At(...)``), a single-node ``at:`` disagreeing with its key, an ``at:``
    with no ``python:``, and a malformed ``at:`` are all documents the build
    refuses, and reporting a node for them is reporting a placement nobody
    makes.  §3.1 makes this the binding Tasks 5 and 11 start from, so a wrong
    answer here is wrong three times.
    """
    graph = _t4_graph()
    lit: set[str] = set()
    for key, spec in _nodes(document).items():
        if key not in graph.nodes:
            continue
        placed = _t5_claims(key, spec)
        if len(placed) >= 2:
            # A region is addressed by its LAST covered node
            # (`paths.refuse_misaddressed_region`), so for every region that
            # BUILDS the key is already one of the covered ids and this adds
            # nothing.  It is here for the region that does not build: the
            # document still names the node, and dropping it would make this
            # reader's answer depend on a refusal it is not making.
            lit.add(key)
        lit.update(node for node in placed if node in graph.nodes)
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


def _t5_radio_class(spec: Any):
    """The shipped class a spec's ``python:`` names, or ``None``.

    ``rheplicant.radio`` only, and BY NAME: §2.4 puts "operator classes
    resolved by name the way ``sections/model.py:49-54`` already does" in
    scope for P-1 and leaves importing an arbitrary module out of it -- an
    ``import_target`` here would run a user's module at pre-flight, which is
    both a file read and an unbounded cost.  A target this cannot resolve is
    not a failure: the caller treats it as an unknown placement and the
    assembly stays the backstop.

    Two clauses here are EQUIVALENT MUTANTS by construction rather than
    untested decisions, and both were measured that way:

    * ``attribute not in radio.__all__`` versus ``getattr(radio, attribute,
      None)``.  The membership test is what stops a bare ``AttributeError`` on
      a typo -- ``python: 'rheplicant.radio:Typo'`` -- from aborting the whole
      pass, and a test pins that.  Which of the two spellings does the
      stopping cannot be told apart: measured, no attribute of
      ``rheplicant.radio`` outside ``__all__`` is a class declaring a
      ``graph_node``, so the two never disagree about a placement.
    * ``target.count(":") != 1``.  Given the membership test it can change no
      answer: ``'rheplicant.radio:GainOperator:extra'`` partitions to the
      attribute ``'GainOperator:extra'`` and ``'rheplicant.radio'`` to ``''``,
      and neither is an exported name.  It is here so the shape is refused
      where it is read rather than three lines later.
    """
    if not isinstance(spec, Mapping):
        return None
    target = spec.get("python")
    if not isinstance(target, str) or target.count(":") != 1:
        return None
    module, _, attribute = target.partition(":")
    if module != "rheplicant.radio":
        return None
    import rheplicant.radio as radio

    if attribute not in radio.__all__:
        return None
    return getattr(radio, attribute)


def _t5_claims(key: Any, spec: Any) -> tuple[str, ...]:
    """The node ids one ``model:`` entry's operator actually lands on.

    **Three answers, and every caller must tell them apart:**

    * ``()`` -- nothing is placed, or text cannot say where.  NOT "nothing is
      wrong here": a ``python:`` target this layer will not import lands at
      its own class's node, so crediting the KEY would invent a collision on a
      document that assembles.
    * one id -- a single placement.  This is what A5 counts and what A8 asks
      its reachability question about.
    * two or more -- an ``At(...)`` REGION, in the document's own order.  A5
      and A8 both leave it alone (``_check_disjoint_claims`` and check A47 own
      a region's overlaps, in their own words); :func:`_lit` lights every node
      it covers.

    **This mirrors ``_single`` (``compose.py:270-327``) clause for clause**,
    and every clause below is pinned by a test that drives ``build_model`` as
    well as the pass.  The draft this replaces had none of them, and measured
    it invented a collision at ``gain`` on a ``snapshot_before:`` document
    that BUILDS, and answered A5 -- the wrong sentence, one phase early -- on
    a composing block and on a FAN label a user happened to spell ``at``:

    * a key that is not a graph node places nothing at all -- ``build_model``
      refuses it as A2 long before anything is constructed;
    * a ``many`` node's spec is a list or a FAN mapping, so its ``at`` is a
      switch LABEL and not a relocation (Task 4 measured the same trap in
      :func:`_lit`), and its entries all land at its own node;
    * ``compose:`` is dispatched BEFORE ``at:`` is popped, and ``_compose``
      refuses ``at`` as an unknown key, so a composing block lands at its key;
    * **``at:`` is honoured only where ``_single`` honours it**, and ``_single``
      refuses three shapes MORE PRECISELY than either check here could.
      Without ``python:`` it is *"at: places an operator that declares no
      graph node of its own"* (``compose.py:296-301``); beside
      ``snapshot_before:`` it is *"at: and snapshot_before: together are not a
      combination this layer writes"* (``compose.py:289-294``); and in the
      STRING spelling it must restate its own key
      (``compose.py:303-308``).  Answering ``()`` for those three is what
      stops A5 and A8 pre-empting the sentence that names the real fault with
      one that names a no-op -- ``cw_tone: {python: ...CWCalibrationOperator,
      at: 'noise'}`` would otherwise be told to give the tone its own node
      ``cw_tone``, which is the key it is already written under.  A LIST of
      one is NOT held to the restatement rule: ``refuse_misaddressed_region``
      returns early below two nodes (``paths.py:318-319``) and, measured,
      ``bandpass: {python: ...GainOperator, at: ['gain']}`` builds with the
      operator at ``gain``;
    * ``at: null`` is no ``at:`` at all -- ``_single`` pops it and tests ``is
      not None`` -- and so is ``snapshot_before: null``.  Measured, a
      ``snapshot_before: null`` beside ``at: ['gain']`` really does place at
      ``gain``: the assembly answers "Two operators provided for node 'gain'";
    * ``snapshot_before:`` wraps the operator in ``At(node_id, ...)``
      (``compose.py:319-326``), which OVERRIDES a ``python:`` class's own
      ``graph_node`` -- measured, ``noise: {python: ...GainOperator,
      snapshot_before: tap}`` builds with the operator at ``noise``.
    """
    graph = _t4_graph()
    node = graph.nodes.get(key) if isinstance(key, str) else None
    if node is None:
        return ()
    if node.many:
        # A list and a FAN mapping both place AT the key, so a `many` node
        # answers before the shape is asked about -- which is also why this
        # clause comes before the `Mapping` test below rather than sharing it.
        return (key,)
    if not isinstance(spec, Mapping):
        # `_single` refuses a single node's non-mapping spec outright ("a node
        # spec is a mapping; got ..."), so nothing is placed and the key fills
        # nothing.  Sharing the `many` clause's `return (key,)` here is what a
        # mutation round caught: it made A5 tell a reader that `model.gain:
        # null` "already fills" the node a relocation had landed on.
        return ()
    if "compose" in spec:
        return (key,)
    snapshot = spec.get("snapshot_before") is not None
    if spec.get("at") is not None:
        at = _t4_at_nodes(spec)
        if not at or snapshot or "python" not in spec:
            return ()
        if isinstance(spec["at"], str) and at[0] != key:
            return ()
        return at
    if snapshot:
        return (key,)
    if "python" in spec:
        home = getattr(_t5_radio_class(spec), "graph_node", None)
        return (home,) if isinstance(home, str) else ()
    return (key,)


def _t5_downstream(graph, node: str) -> set[str]:
    """Every node the signal leaving ``node`` reaches.

    ``core/fold._descendants`` itself, not a second breadth-first walk:
    reachability IS ``_check_ordering``'s rule (``fold.py:290``) and a copy of
    it here is a second definition of "after" that can drift from the one the
    assembly enforces.
    """
    from rheplicant.core.fold import _descendants

    return _descendants(graph, node)


@register("A5")
def _two_at_one_node(document: Mapping[str, Any]) -> Iterable[Finding]:
    """Check A5: two operators claim one node that holds a single instance.

    ``compose:`` is the document's spelling of the ``At(...)`` route
    ``core/graph.py:910-917`` names, so this message ends where that one does.
    A ``many`` node is skipped: several operators there is what ``many``
    MEANS, and so is a REGION -- an entry claiming two or more nodes is
    ``_check_disjoint_claims``' and check A47's, in their own words.

    **Blame is read off the placements, not off document order.**  The
    OCCUPANT is the entry written under the node it fills; every other
    claimant relocated onto it, and each of those gets its own sentence at its
    own ``where``.  Measured before this: naming ``keys[1]`` made the same
    collision blame ``model.noise`` or ``model.gain`` depending on which key
    the user happened to type first, and the second of those sends the reader
    to delete the entry that was at its own node -- the exact inversion
    ``test_the_message_names_both_keys_and_the_node`` says it exists to stop.
    One finding per intruder rather than one per node, so a node claimed three
    times names all the entries that have to move rather than costing the
    reader a round trip each.
    """
    graph = _t4_graph()
    claims: dict[str, list[str]] = {}
    for key, spec in _nodes(document).items():
        placed = _t5_claims(key, spec)
        if len(placed) != 1:
            continue
        claims.setdefault(placed[0], []).append(key)
    for node, keys in claims.items():
        # No `len(keys) < 2` here: the intruder list below IS the cardinality
        # rule, and a second expression of it is two clauses that can come to
        # disagree.  Measured -- with the guard present, weakening it to
        # `< 1` changed nothing, because a lone claimant is its own occupant
        # and leaves the list empty.
        if node not in graph.nodes or graph.nodes[node].many:
            continue
        occupant = node if node in keys else keys[0]
        for intruder in [key for key in keys if key != occupant]:
            yield refuse(
                "A5", f"model.{intruder}",
                f"model.{intruder}: puts a second operator at node {node!r}, "
                f"which model.{occupant} already fills, and this node accepts "
                "a single instance. Compose them under one key instead -- "
                "compose: cascade at a transform node, compose: sum at a "
                "source node, which is how this document spells At(...) "
                "(check A5).")


@register("A8")
def _tone_placement(document: Mapping[str, Any]) -> Iterable[Finding]:
    """Check A8: a ``cw_tone`` relocated at, or downstream of, what it tracks.

    The constraint is READ OFF THE CLASS -- ``must_precede`` and
    ``must_precede_because`` -- rather than written here, so the message says
    what the operator says.  Measured at ``48b359d``, ``CWCalibrationOperator``
    is the only class in ``rheplicant.radio.__all__`` (58 names) that declares
    one, and ``test_the_tone_is_the_only_shipped_class_with_an_ordering_
    constraint`` pins that: the day a second ships, that test goes red and
    someone decides whether this check's wording generalises.  The direct
    ``must_precede_because`` read below is sound for the same reason.

    **The only document route to a RELOCATED tone is ``python:``.**  Measured:
    ``at:`` on an entry with no ``python:`` is refused first, at
    ``compose.py:296-301``, so a tone written the ordinary way -- a
    ``cw_tone:`` key and no ``python:`` -- cannot be moved at all.  It does
    NOT follow that ``model.cw_tone.at`` is a key no document can contain, and
    an earlier draft of this paragraph said so and was wrong: measured,
    ``cw_tone: {python: ...CWCalibrationOperator, at: ['noise']}`` is a legal
    spelling that this check answers about, and ``at: 'noise'`` under the same
    key is ``_single``'s refusal rather than nothing.  What follows is only
    that keying on a NAME would be wrong in both directions -- the tone can be
    written under any key, and its own key can carry a relocation -- so the
    trigger is the declared ``must_precede`` and nothing else.

    Two refusals, because they name two fixes.  A tone placed where a LIT
    target is unreachable is the assembly's own rule, arriving earlier.  A
    tone placed AT a target is not a violation the assembly can see -- it
    skips ``target in path``, and ``fold.py:271-274`` and
    ``pipeline.py:94-102`` argue at length that an absent stage is nothing to
    pass through.  That reasoning is right and this does not touch it: what
    the document has and the assembly does not is the key, written down,
    saying the stage is there.
    """
    graph = _t4_graph()
    lit = _lit(document)
    for key, spec in _nodes(document).items():
        cls = _t5_radio_class(spec)
        required = tuple(getattr(cls, "must_precede", ()) or ())
        if not required:
            continue
        placed = _t5_claims(key, spec)
        # `len(placed) != 1` rather than `not placed`, and the two are the
        # same TODAY: :func:`_t5_claims` answers `()` for a region, so no
        # entry ever claims two nodes.  Written the longer way on purpose --
        # this is the clause that keeps A8 from answering about a region in a
        # voice that names one node, the day that reader changes.  Recorded as
        # an equivalent mutant rather than left as an untested decision.
        if len(placed) != 1 or placed[0] not in graph.nodes:
            continue
        node = placed[0]
        if node in required:
            yield refuse(
                "A8", f"model.{key}",
                f"model.{key}: puts {cls.__name__} IN the {node!r} slot, so "
                f"this document declares no {node!r} operator for it to pass "
                f"through -- it replaced the stage it is there to track. "
                f"{cls.must_precede_because} assemble() cannot say this: it "
                f"sees one placement, and an absent stage is deliberately no "
                f"violation there (core/fold.py:271-274), while the document "
                f"still has the key and the operator apart. Give it its own "
                f"node, {cls.graph_node!r} (check A8).")
            continue
        blocked = [target for target in required
                   if target in lit
                   and target not in _t5_downstream(graph, node)]
        if blocked:
            yield refuse(
                "A8", f"model.{key}",
                f"model.{key}: places {cls.__name__} at {node!r}, from which "
                f"{blocked} cannot be reached -- and this document lights "
                f"{'them' if len(blocked) > 1 else 'it'}, so nothing this "
                f"operator contributes ever passes through. "
                f"{cls.must_precede_because} Place it upstream: its own node "
                f"is {cls.graph_node!r} (check A8).")


@register("A31")
def _data_with_sources(document: Mapping[str, Any]) -> Iterable[Finding]:
    """Check A31: ``observation.data`` while ``model`` lights a source node.

    The predicate is ``Assembly.has_source``' own -- node KINDS, not operators
    (``core/graph.py:411-417``) -- evaluated statically over the lit ids.
    ``test_the_static_source_predicate_is_the_assemblys_own`` holds the
    package's own ``has_source`` against that static reading over six models
    and they agree in all six, so this pins the PACKAGE's predicate rather
    than this check's restatement of it.

    Two shapes stand down, both measured:

    * ``data:`` written EMPTY is no data at all -- ``_data`` returns ``None``
      for a ``None`` node (``observation.py:277-278``) and
      ``Assembly.__call__`` refuses only ``state.data is not None`` -- so
      ``"data" in section`` would refuse a document the package runs;
    * ``from_file:`` beside ``data:`` is ``build_observation``'s own refusal
      ("the recording IS the data", ``observation.py:315-326``), and it
      already precedes the beam (``document.py:72`` against ``:75``).  A31
      answering first would offer "the twin makes it", which is true of a
      simulating document and wrong about a recording.

    ``observation.from_file`` ALONE puts the recording in ``state.data`` and
    is the same defect; it is NOT refused here.  See §3.2 (e) 1 and
    ``test_the_recording_route_is_recorded_and_not_yet_refused``, which
    carries the exact three-line widening: closing it turns
    ``tests/config/test_config_document.py:167-179`` red, and that test pins
    the refusal as the assembly's on purpose.
    """
    section = document.get("observation")
    if not isinstance(section, Mapping):
        return
    if section.get("data") is None or "from_file" in section:
        return
    graph = _t4_graph()
    sources = sorted(node for node in _lit(document)
                     if graph.nodes[node].kind == "source")
    if not sources:
        return
    yield refuse(
        "A31", "observation.data",
        "observation.data: is the data a transform chain acts ON, and this "
        f"model lights the source nodes {sources} -- an assembly with sources "
        "GENERATES its own data, so the array declared here would be "
        "discarded rather than fitted. Drop observation.data (the twin makes "
        "it), or drop the sources to leave a transform chain (check A31).")
