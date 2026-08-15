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
11's two checks added (median of 2000 calls, in two separate processes so the
first-call figure is not warmed by the loop): the **eight** checks here answer
a **six**-node document -- ``preflight_helpers``' base model, which has four,
plus ``foregrounds:`` and ``bandpass:`` -- in **3.1e-05 s**; the first call is
**2.5e-04 s**.  §0.1's budget for the whole pass is 0.05 s, against
``load_document``'s 1.536 s on a toy beam, and the whole pass on a Task 11
document measures **3.5e-04 s**.

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
from rheplicant.config.paths import parse_path
from rheplicant.config.preflight import register
from rheplicant.config.preflight.document import _task3_where
from rheplicant.config.preflight.fitting import _kinds, _latents
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
from rheplicant.config.sections.transforms import _NAMED
from rheplicant.core.contract import RANDOMNESS


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


# ---------------------------------------------------------------------------
# Task 11 -- A30 (a stochastic stage the fit twin keeps) and A33 (a bandpass
# left free beside a gain).
# ---------------------------------------------------------------------------

#: The two exits that build neither a ``ParameterSpace`` over the fit twin nor
#: a forward function from it, so a stochastic stage is legal under them.
#: Everything else in ``sections/runs._KINDS`` does -- measured one kind at a
#: time through ``run_document`` on a ``twin: {without: []}`` document:
#: ``forward`` RUNS (``_run_forward`` is ``return built.twin(built.state)``,
#: ``exits.py:38-40``); thirteen kinds raise ``ParameterSpaceError`` naming
#: *"NoiseOperator at 'noise', which declares 'key' in requires"*; ``predict``
#: reaches ``space.forward_fn`` (``diagnostics.py:766``) and its ``reuse:``
#: can only name a kind that refuses first (``_DRAW_SOURCES``,
#: ``diagnostics.py:676-682``); and ``_run_mmodes`` (``diagnostics.py:
#: 594-684``) contains no ``_space(``, no ``forward_fn``, no
#: ``build_forward_fn`` and no ``fit_twin`` at all.
#:
#: Written as the COMPLEMENT on purpose (§3.2 (e) 2): a kind added to
#: ``_KINDS`` defaults to FITTING, so a new exit inherits the check rather
#: than escaping it.  A wrong refusal is loud; a lost check is silent.
#: ``test_the_complement_is_a_subset_of_the_declared_kinds`` is what makes the
#: day a genuinely non-fitting kind ships a day someone looks.
_A30_NOT_FITTING: frozenset[str] = frozenset({"forward", "mmodes"})

#: The registry name of the identifiability convention A33 advises.
#: ``transforms._NAMED`` holds it (``transforms.py:36``) and it resolves to
#: ``radio.instrument.receiver.unit_mean_bandpass`` (``:175-178``).  Advising
#: a word the registry does not hold would send a reader to a refusal quoting
#: a vocabulary without it, so a test pins this against ``_NAMED``.
_A33_CONVENTION: str = "unit_mean_bandpass"


def _a30_stochastic(node_id: Any, spec: Any, table: Mapping) -> str | None:
    """The operator class name, IF this entry's text says it draws randomness.

    ``None`` where it does not, and ``None`` where the text cannot say -- the
    two are deliberately not distinguished, because A30 refuses only a certain
    "yes".

    The declaration is ``RANDOMNESS in cls.requires``, read off the **class**.
    §2.5 names ``stages_requiring(pipeline, RANDOMNESS)`` instead, which takes
    a CONSTRUCTED ``AbstractOperator`` and reads ``stage.requires`` off
    instances -- and §3.2 (f) forbids constructing one.  They reconcile
    because ``requires`` is a ``ClassVar`` (``core/operator.py:85``), so the
    same capability predicate applied to the class satisfies §2.5's intent
    (it still catches any operator that declares ``'key'``, and never goes
    stale on the next one) and §3.2 (f)'s no-construction rule at once.  Only
    §2.5's literal function CALL is dropped.

    ``node_id`` is the node the entry LANDS ON, which is not always the key it
    is written under -- see :func:`_a30_placements`.

    Five routes, each measured at ``0263e0f`` through ``load_document``:

    * a ``python:`` target is resolved BY NAME through Task 5's
      :func:`_t5_radio_class` -- §2.4 item 3 puts exactly that in scope.  A
      target outside ``rheplicant.radio``, or one it does not export, answers
      ``None``: the class such a node builds is not knowable here, and
      guessing in the stochastic direction refuses a legal document.
      **Standing down on every ``python:`` spec would be a lost check**:
      measured, ``{emi: {python: 'rheplicant.radio:NoiseOperator', sigma:
      ...}}`` builds, lands ``NoiseOperator`` at ``noise``, and its fit twin
      keeps the draw.
    * ``from:`` derives the operator from another node, which is
      CONSTRUCTION; nothing in the text names a class.
    * ``compose:`` is expanded STAGE BY STAGE rather than asked about its
      node.  Measured: a block composing two ``python:`` ``GainOperator``
      stages at ``noise`` builds with no randomness anywhere, so the
      unanimity clause below applied to the composing mapping would refuse a
      document the package runs.
    * a ``type:`` is looked up among the node's own classes, which is
      ``_pick_class``' vocabulary; a name that is not one of them is check
      A7's refusal, in its own words.
    * with no ``type:``, the verdict is UNANIMITY over the node's classes.
      Measured, the three multi-class nodes are ``noise`` (both classes draw),
      ``flagging`` and ``filters`` (neither of theirs does), so no shipped
      node is mixed and the mixed branch is unreachable today.
      ``test_no_shipped_node_mixes_stochastic_and_deterministic_classes`` is
      what says so, and what will fail on the day a mixed node ships rather
      than this function guessing.

    Two clauses here are EQUIVALENT MUTANTS by construction rather than
    untested decisions, and both were measured that way against a 27-row
    battery:

    * ``all(...)`` versus ``any(...)`` in the unanimity clause.  They can
      differ only at a node whose classes DISAGREE about randomness, and no
      shipped node does -- which is precisely what the test named above
      asserts, so the day the two spellings can disagree is the day that test
      goes red.
    * ``not isinstance(declared, str)``.  Given the loop below it changes no
      answer: a non-string ``type:`` matches no ``cls.__name__``, so the loop
      falls to its own ``return None``.  It is here so the shape is refused
      where it is read.  What is NOT equivalent, and is pinned, is testing
      ``"type" in spec`` rather than ``spec.get("type") is not None``.
    """
    if isinstance(spec, (list, tuple)):
        # A `many` node's entries, and a `compose:` block's stages.
        for one in spec:
            found = _a30_stochastic(node_id, one, table)
            if found is not None:
                return found
        return None
    if not isinstance(spec, Mapping):
        return None
    if "python" in spec:
        cls = _t5_radio_class(spec)
        if cls is None or RANDOMNESS not in getattr(cls, "requires", ()):
            return None
        return cls.__name__
    if "from" in spec:
        return None
    if "compose" in spec:
        stages = spec.get("stages")
        if not isinstance(stages, (list, tuple)):
            # `_compose` refuses the shape in its own words; a mapping read as
            # a node spec here would reach the unanimity clause and call a
            # malformed block stochastic on the strength of its NODE.
            return None
        return _a30_stochastic(node_id, stages, table)
    classes = table.get(node_id) if isinstance(node_id, str) else None
    if not classes:
        return None
    if "type" in spec:
        # PRESENCE, not truthiness: `type: null` is a spec that named a class
        # and failed to, so falling through to the unanimity clause below
        # would answer about a class the document never chose -- and at
        # ``noise``, where both classes draw, that answer is a refusal.
        declared = spec["type"]
        if not isinstance(declared, str):
            return None
        for cls in classes:
            if cls.__name__ == declared:
                return cls.__name__ if RANDOMNESS in cls.requires else None
        return None
    if all(RANDOMNESS in cls.requires for cls in classes):
        return classes[0].__name__
    return None


def _a30_placements(document: Mapping[str, Any],
                    table: Mapping) -> dict[str, tuple[str, str]]:
    """node id -> ``(the site that put it there, the class name)``.

    **Keyed by NODE, not by the model key**, and that is the whole reason this
    is a function rather than a dict comprehension over :func:`_nodes`:
    ``inference.twin.without:`` names node ids -- ``Assembly.without``
    (``twin.py:59-60``) -- and a ``python:`` entry lands where its class
    declares rather than under the key it is written beneath.  Measured at
    ``0263e0f``: ``{emi: {python: 'rheplicant.radio:NoiseOperator', ...}}``
    with ``without: [noise]`` BUILDS and its fit twin is clean, while
    ``without: [emi]`` is the one the assembly itself refuses.  A reader
    keyed on the model key gets both of those backwards -- one a false
    refusal, one a lost check.

    :func:`_t5_claims` is the one placement reader in this file (§2.2), and a
    REGION's operator sits at the LAST node it covers -- ``paths.
    refuse_misaddressed_region`` says so and the assembly agrees (measured:
    ``at: ['noise', 'emi']`` reports the stage at ``emi``).  ``placed[-1]``
    is therefore right for a single placement and for a region alike.

    First claimant wins where two entries land on one node; that document is
    check A5's refusal, and A30's fix -- drop the node -- is the same
    whichever key it names.
    """
    placements: dict[str, tuple[str, str]] = {}
    for key, spec in _nodes(document).items():
        placed = _t5_claims(key, spec)
        if not placed:
            continue
        node_id = placed[-1]
        found = _a30_stochastic(node_id, spec, table)
        if found is not None:
            placements.setdefault(node_id, (f"model.{key}", found))
    return placements


def stochastic_nodes(document: Mapping[str, Any]) -> frozenset[str]:
    """The node ids whose declared operator class declares ``RANDOMNESS``.

    §3.2 (f)'s shared name: Task 11 (A30) binds it and Task 12 (A42) imports
    it, because two private predicates for one property is the collision
    §3.2 (f) calls "the likeliest remaining collision in the plan".  Public
    for that reason -- it crosses a module boundary, the precedent
    ``sections/observed.py:23`` sets.

    Read off the CLASS (``operator_table()``, measured at 0.2 ms on its first
    call and 0.02 ms after, importing nothing on §0's forbidden list) --
    never off a constructed operator, which is out of scope.

    The MODEL's nodes, before any ``inference.twin`` repair: A42 asks whether
    a node the observed data was simulated through has left the fit twin, so
    it needs the unrepaired answer, and A30 applies the repair itself.
    """
    return frozenset(_a30_placements(document, operator_table()))


@register("A30")
def _stochastic_in_fit_twin(document: Mapping[str, Any]) -> Iterable[Finding]:
    """Check A30: a stochastic stage the fit twin keeps, under an exit that
    closes over ONE template state.

    ``inference.twin`` is applied the way ``build_fit_twin`` applies it
    (``twin.py:59-69``): ``without:`` first, then ``replace:``.  Reading
    ``replace:`` is what keeps this honest on the twin route -- measured, a
    document whose only ``twin:`` key is ``replace: {noise:
    RadiometerNoiseOperator}`` builds a fit twin that still draws, and a check
    reading ``model:`` alone would name the wrong class about it.

    A ``replace:`` on a node ``without:`` has already dropped is skipped:
    measured, ``Assembly.replace_node`` raises ``KeyError("No node named
    'noise' in this assembly")`` there, so the document is refused whatever
    this says, and saying "write without: [noise]" about a document that
    contains that line names a fix it already has.

    A malformed ``twin:`` -- not a mapping, or a ``without:`` that is not a
    list -- reads as NO repair rather than as a stand-down.  ``build_fit_twin``
    refuses both shapes in its own words one phase later, and A30's advice
    (the list form, spelled out) is the right thing to say first.  What it may
    never do is RAISE: a ``without:`` entry that is a list is unhashable, and
    popping by it would abort the pass and discard every other finding
    (§2.3's TRAP).

    **A document that declares no latent stands the whole check down**, and
    that is Task 5's "do not pre-empt a more specific refusal" rather than a
    convenience.  A fitting exit with no ``inference.parameters`` fits
    nothing, and the package already says exactly that: measured at
    ``0263e0f``, ``fisher``, ``identifiability`` and ``score_directions`` on a
    document whose ``inference:`` is ``{}`` each refuse naming
    ``inference.parameters``, and three tests in ``tests/config/`` pin that
    sentence on purpose.  Without this clause A30 displaces all three --
    which also makes the task body's claim that this check "refuses nothing in
    the shipped suite" false, measured.  Repairing a twin for a fit the
    document does not declare is the wrong fix named first.
    """
    fitting = sorted(_kinds(document) - _A30_NOT_FITTING)
    if not fitting or not _latents(document):
        return ()
    table = operator_table()
    placements = _a30_placements(document, table)
    section = document.get("inference")
    section = section if isinstance(section, Mapping) else {}
    twin = section.get("twin")
    twin = twin if isinstance(twin, Mapping) else {}

    dropped = twin.get("without")
    dropped = tuple(one for one in dropped
                    if isinstance(one, str)) if isinstance(
                        dropped, (list, tuple)) else ()
    for node_id in dropped:
        placements.pop(node_id, None)
    replace = twin.get("replace")
    if isinstance(replace, Mapping):
        for node_id, spec in replace.items():
            if not isinstance(node_id, str) or node_id in dropped:
                continue
            found = _a30_stochastic(node_id, spec, table)
            if found is None:
                placements.pop(node_id, None)
            else:
                placements[node_id] = (
                    f"inference.twin.replace.{node_id}", found)
    if not placements:
        return ()

    named = " / ".join(f"kind: {kind}" for kind in fitting)
    findings: list[Finding] = []
    # Sorted by NODE id, so blame order is a property of the graph rather than
    # of the order the user happened to type two nodes in (Task 5's lesson).
    for node_id in sorted(placements):
        site, operator = placements[node_id]
        findings.append(refuse("A30", "inference.twin.without", (
            f"{site} puts {operator} at node {node_id!r}, which draws its own "
            f"randomness -- {operator} declares {RANDOMNESS!r} in requires "
            "-- and inference.twin.without: does not drop it. This document "
            f"declares {named}, and every exit but forward and mmodes closes "
            "the fit twin over ONE template state, so that draw would be the "
            "SAME realisation added to every prediction alike: a bias that is "
            "exactly affine and full rank, which is why no shape check, no "
            "linearity check and no rank test sees it. Write "
            f"inference.twin.without: [{node_id}] -- kind: forward keeps the "
            "node, and simulating with it is what it is for (check A30).")))
    return tuple(findings)


def _t11_bindings(
        document: Mapping[str, Any]
) -> tuple[tuple[str, frozenset[str], tuple[str, ...], Any], ...]:
    """``(document path, latent NAMES, into-path HEADS, transform)`` per binding.

    BOTH spellings, because ``build_space`` walks two loops over one meaning:
    ``inference.parameters.<n>.into`` (``transforms.py:344-361``) and
    ``inference.bindings[i].into`` (``:362-399``).  Both carry ``transform:``.
    A check that read one is 2C's shape 4 -- a hole closed on one route and
    left open on its twin -- in the one place this layer has an actual twin.

    The latent NAMES travel with the binding because A33's question is about
    two DIFFERENT parameters.  One latent written into both leaves (``into:
    [bandpass.bandpass, gain.gain]``, or two ``bindings`` entries naming the
    same latent) is one degree of freedom and no null direction at all, and a
    check that compared only path heads refuses it.

    A binding whose latent names cannot be read stands for its own ``where``,
    which no latent name can equal -- so it is never mistaken for the latent
    on the other node.  The malformed ``latents:`` itself is
    ``transforms.py:369-374``'s refusal.

    ``parse_path`` RAISES on a malformed path (measured: ``''``, ``'a..b'``,
    ``None`` and ``['a']`` all give ``ConfigError``), and a check that raises
    aborts the pass and hides every later finding (§2.3's TRAP).  So an
    unparseable ``into:`` is skipped here and left to
    ``_selectors``/``parse_path`` at build time, which already names it.
    """
    section = document.get("inference")
    section = section if isinstance(section, Mapping) else {}
    written: list[tuple[str, tuple[str, ...], Any, Any]] = [
        (f"inference.parameters.{name}", (name,), spec.get("into"),
         spec.get("transform"))
        for name, spec in _latents(document).items()
        if spec.get("into") is not None
    ]
    bindings = section.get("bindings")
    if isinstance(bindings, (list, tuple)):
        for index, entry in enumerate(bindings):
            if not isinstance(entry, Mapping) or entry.get("into") is None:
                continue
            latents = entry.get("latents")
            latents = (latents,) if isinstance(latents, str) else latents
            names = tuple(one for one in latents if isinstance(one, str)) \
                if isinstance(latents, (list, tuple)) else ()
            written.append((f"inference.bindings[{index}]", names,
                            entry.get("into"), entry.get("transform")))

    out: list[tuple[str, frozenset[str], tuple[str, ...], Any]] = []
    for where, names, into, transform in written:
        paths = [into] if isinstance(into, str) else into
        if not isinstance(paths, (list, tuple)):
            continue
        heads: list[str] = []
        for path in paths:
            if not isinstance(path, str):
                continue
            try:
                head = parse_path(path)[0]
            except ConfigError:
                continue
            if isinstance(head, str):
                heads.append(head)
        out.append((where, frozenset(names or (where,)), tuple(heads),
                    transform))
    return tuple(out)


def _a33_convention(transform: Any) -> bool | None:
    """Is ``transform`` the bandpass's identifiability convention?

    Three answers.  ``True`` -- it is :data:`_A33_CONVENTION`.  ``False`` --
    it is a registered NAME that is not: ``identity`` binds the leaf
    unchanged, ``exp`` and ``log`` are elementwise, ``sum`` reduces and
    ``split_rows`` re-shapes, and none of the five moves the mean, so the
    product of bandpass and gain stays the only constrained combination.
    ``None`` -- text cannot say: a MAPPING transform is an arbitrary callable
    (``{python: ...}``) or an affine map whose operands may be value nodes,
    and an unregistered name is ``parse_transform``'s own refusal
    (``transforms.py:179-182``), in its own words.

    The ``None`` answers stand A33 down rather than refusing, because the
    reader HAS declared a transform and telling them to declare one names a
    fix they have already applied -- Task 5's "do not pre-empt a more
    specific refusal", in the direction where the more specific sentence is
    the value grammar's.
    """
    if transform is None:
        return False
    if transform == _A33_CONVENTION:
        return True
    if isinstance(transform, str) and transform in _NAMED:
        return False
    return None


@register("A33")
def _bandpass_and_gain(document: Mapping[str, Any]) -> Iterable[Finding]:
    """Check A33: a latent free into ``bandpass`` beside a DIFFERENT one free
    into ``gain``, with no identifiability convention on the bandpass.

    Every declared latent is free -- ``_LATENT_KEYS`` (``parameters.py:
    27-29``) has no freeze -- so "both free" is "both bound".

    Pure text: the head of each ``into:`` path.  ``parse_path(path)[0]`` gives
    it (``config/paths.py:38``) -- measured, ``parse_path('bandpass.taps[0]')``
    is ``('bandpass', 'taps', 0)``, so a deeper path still counts by its node.

    The SHAPE half of A33 -- that ``unit_mean_bandpass`` maps ``(n,)`` to
    ``(n-1,)``, so a latent bound through it is one channel shorter than the
    node it writes -- is check C17 and needs a resolved shape.  That is Plan
    3C's; this half is two path heads, two latent names and a transform.

    **Measured absent today** at ``0263e0f``: a document lighting ``bandpass``
    and ``gain`` with a free latent into each and no transform builds with no
    refusal, and no document in ``tests/config/`` or ``docs/`` binds a latent
    into a ``bandpass.*`` path at all -- so this check refuses nothing that
    exists and its own tests are the only coverage it will have.
    ``exit_helpers.py:521-524`` calls its ``gain`` + ``global_signal.depth``
    pair *"the schema's A33 shape"*; that is an analogy about degeneracy and
    not an A33 document, because neither latent goes into ``bandpass``.
    """
    bindings = _t11_bindings(document)
    on_bandpass = [one for one in bindings if "bandpass" in one[2]]
    if not on_bandpass:
        return ()
    on_bandpass_latents = frozenset().union(
        *(names for _, names, _, _ in on_bandpass))
    on_gain_latents = frozenset().union(
        *(names for _, names, heads, _ in bindings if "gain" in heads))
    # The DIFFERENCE, not the presence: a latent written into both leaves is
    # one degree of freedom and the product IS constrained, so there is
    # nothing to trade and nothing to refuse.
    if not on_gain_latents - on_bandpass_latents:
        return ()
    verdicts = [_a33_convention(transform) for _, _, _, transform
                in on_bandpass]
    if any(verdict is not False for verdict in verdicts):
        return ()
    where = on_bandpass[0][0]
    return (refuse("A33", _task3_where(f"{where}.transform"), (
        f"{where} is free into bandpass and this document also frees a "
        "latent into gain. The receiver's bandpass and the gain multiply the "
        "same prediction, so only their PRODUCT is constrained: the fit has "
        "one exactly null direction and returns a finite, correctly-shaped "
        "answer in which the two have traded an arbitrary constant. Declare "
        f"transform: {_A33_CONVENTION} on the bandpass binding -- it divides "
        "out the mean, which is the convention that makes the pair "
        "identifiable (check A33).")),)
