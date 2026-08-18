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
plus ``foregrounds:`` and ``bandpass:`` -- in **3.3e-05 s**; the first call is
**2.3e-04 s**.  §0.1's budget for the whole pass is 0.05 s, against
``load_document``'s 1.536 s on a toy beam, and the whole pass on a Task 11
document measures **4.2e-04 s**.  (Re-measured after A30 and A33 each grew a
:func:`_lit` call at the review: **3.1e-05 -> 3.3e-05 s** here and 3.5e-04 ->
4.2e-04 s for the pass.)

**That first call is not a deferred import**, which is what this file said
until Task 5 checked: pre-importing ``sections/switching``, ``core/fold`` and
``rheplicant.radio`` leaves it at 2.3e-04 s unchanged.  It is
``sections/model.operator_table()``, measured at 1.7e-04 s the first time and
2.3e-05 s every time after -- it is rebuilt per call rather than cached, which
is also most of the steady-state figure above.  Recorded rather than fixed:
the table is ``sections/model``'s and no task in this plan owns it.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable, Mapping
from typing import Any

from _rheplicant_bootstrap.path_syntax import longest_legal_prefix
from rheplicant.config.errors import ConfigError
from rheplicant.config.findings import Finding, refuse
from rheplicant.config.paths import parse_path
from rheplicant.config.preflight import register
from rheplicant.config.preflight.fitting import _kinds, _latents, _runs
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

    An entry that places NOTHING lights nothing: a single-node ``at:``
    disagreeing with its key, an ``at:`` with no ``python:``, and a malformed
    ``at:`` are all documents the build refuses, and reporting a node for them
    is reporting a placement nobody makes.  §3.1 makes this the binding Tasks
    5 and 11 start from, so a wrong answer here is wrong three times.

    **The fourth shape this paragraph used to name is NOT one of them, and
    saying so was a false claim about the build.**  It read *"a ``python:``
    target this layer will not import (measured, ``AssemblyError``: no
    ``graph_node`` and no ``At(...)``)"*, which conflates two different
    documents: a target whose CLASS declares no ``graph_node`` really is the
    ``AssemblyError`` (nothing placed), while a target this pass merely
    declines to NAME is placed by the build wherever its class declares.
    Measured at the commit that widened :func:`_t5_radio_class`: the same
    class under two spellings gave ``('cal_loads',)`` and ``()``, and
    ``_t5_claims``' own docstring says ``()`` means "nothing placed **or**
    text cannot say where" -- the opposite of what this said, in the same file
    and the same commit.  This function still lights nothing for such an entry,
    because crediting the KEY would invent a placement the build does not
    make; :func:`_t5_placement` is where the distinction is available to a
    caller that needs it.
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
                yield refuse("A7", longest_legal_prefix(f"model.{where}"),
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

    **The class, not the spelling.**  Until this commit the module test was
    ``module != "rheplicant.radio"``, and the build resolves the same target
    through ``hatch.import_target`` (``sections/model.py:260``), which imports
    **any** module -- so the two spellings of ONE class object diverged, and
    six checks read the divergence.  Measured: ``import_target
    ('rheplicant.radio.instrument.calibration:CalLoadOperator') is
    CalLoadOperator`` is True, while the exported spelling gave
    :func:`_t5_claims` ``('cal_loads',)`` and the submodule spelling ``()`` --
    which A14 refuses **on absence**, so a document that BUILDS was refused,
    and A5, A15, A31, A52 and A30's ``replace:`` gate each lost their subject
    on the same document.  Task 12 had recorded the hole for A42 alone
    (``preflight/values.py:283-303``: *"two different resolvers, and only one
    of them is P-1's"*); it was never A42's.

    **The widening imports NOTHING**, which is what keeps §2.4 and §0 intact.
    A target resolves here only when its attribute is a name
    ``rheplicant.radio`` exports AND the named module -- already in
    ``sys.modules`` -- carries that same object.  Measured over all 58 names
    in ``rheplicant.radio.__all__``: 55 resolve by their own defining module's
    spelling to the identical object with **no new module imported**, and the
    other three (``RADIO_GRAPH``, ``rhino_to_state`` and one more) are names
    ``import_target`` also refuses at that spelling, so the two agree there
    too.  ``rheplicant.radio`` is imported by ``import rheplicant.config``
    already (``config/kinds/projectors.py:44``), so every module defining an
    exported class is in ``sys.modules`` before the pass runs.

    A module ``sys.modules`` does not carry is a **decline**, never a guess:
    importing it would run a user's module at pre-flight, which is both a file
    read and an unbounded cost.  The decline can only lose a check, never
    invent one -- to answer at all, the object found must BE the shipped class
    the build would construct -- and :func:`_t5_placement` is what tells a
    caller that a decline happened, because for A14 "no placement" and "no
    placement I can see" are opposite answers.

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
    import rheplicant.radio as radio

    if attribute not in radio.__all__:
        return None
    shipped = getattr(radio, attribute)
    if module == "rheplicant.radio":
        return shipped
    # `sys.modules.get` and never `import_module`: this reads what the process
    # already holds and imports nothing.  The identity test is what makes the
    # widening exact rather than a guess -- a module that binds this name to
    # something else, or does not bind it at all, is a decline.
    return shipped if getattr(sys.modules.get(module), attribute,
                              None) is shipped else None


def _t5_claims(key: Any, spec: Any) -> tuple[str, ...]:
    """The node ids one ``model:`` entry's operator actually lands on.

    **Three answers, and every caller must tell them apart:**

    * ``()`` -- nothing is placed, or text cannot say where.  NOT "nothing is
      wrong here": a ``python:`` target this layer will not import lands at
      its own class's node, so crediting the KEY would invent a collision on a
      document that assembles.  **A caller that refuses on ABSENCE must call
      :func:`_t5_placement` instead**, which separates the two -- reading this
      ``()`` as "nothing is placed" is what made A14 refuse a document that
      builds.
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
    return _t5_placement(key, spec) or ()


def _t5_placement(key: Any, spec: Any) -> tuple[str, ...] | None:
    """:func:`_t5_claims`, with "cannot say" told apart from "nothing placed".

    ``None`` -- and only for a ``python:`` target :func:`_t5_radio_class`
    declines.  Such an entry places its class wherever that class declares,
    which the build resolves and this pass cannot: the answer is *unknown*,
    not *empty*.  Every other ``()`` below is a placement the build itself
    makes nowhere, and stays ``()``.

    Two answers, two polarities, and collapsing them is a defect in whichever
    direction the caller refuses:

    * a check that refuses on ABSENCE (A14: *"this model places no
      calibration load at all"*) must read ``None`` as a stand-down, or it
      refuses a document that builds -- measured, and live until this commit;
    * a check that refuses on PRESENCE (A5, A8, A31, A52, A15, A30's
      ``replace:`` gate) loses its subject on a ``None`` and says nothing,
      which is a lost check rather than a false one.  Recorded rather than
      closed: naming the class would mean importing the module, which is out
      of P-1 (§2.4).

    :func:`_t5_radio_class`'s widening is what makes ``None`` rare -- every
    class ``rheplicant.radio`` exports now resolves under any spelling of its
    own module -- so what is left is a genuinely foreign class, which is the
    one case where the two answers still have to be told apart.
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
        cls = _t5_radio_class(spec)
        if cls is None:
            # NOT `()`.  The build imports this target and places the class
            # where it declares; this pass declines to import, so the answer
            # is unknown.  Returning `()` here is what made A14 refuse a
            # document that assembles.
            return None
        home = getattr(cls, "graph_node", None)
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

    **The remedy is CONDITIONAL on the intruder's own ordering declaration**,
    and offering it unconditionally was the worst shape the whole-branch
    review found.  Measured: ``cw_tone: {python: ...CWCalibrationOperator, at:
    ['gain']}`` beside ``model.gain`` earns A5 **and** A8 on one node, with
    opposite advice -- *"Compose them under one key"* against *"Give it its
    own node"* -- and ``raise_if_refused`` quotes A5, the first-registered, so
    the reader never sees the finding that forbids its fix.  Applying it
    verbatim (``gain: {compose: cascade, stages: [<the gain>, <the tone>]}``)
    then produced a **clean report** with the tone still inside the gain slot,
    which is exactly A8's physical objection: advice that silences a check
    without fixing what it was about.  A8 now reads composed stages, so that
    document is no longer silent -- and this clause is what stops the reader
    being sent there in the first place.
    """
    graph = _t4_graph()
    specs = _nodes(document)
    claims: dict[str, list[str]] = {}
    for key, spec in specs.items():
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
                "a single instance. "
                f"{_a5_remedy(node, specs.get(intruder))} (check A5).")


def _a5_remedy(node: str, spec: Any) -> str:
    """A5's fix clause -- the compose one, unless check A8 forbids it.

    ``compose:`` is the right answer for two operators that merely want the
    same node.  It is the WRONG answer for an operator whose class declares
    ``must_precede`` naming that node, because composing puts it inside the
    stage it has to come before -- and A8 co-fires on precisely that document
    saying so, in the opposite direction.  Measured, A5 is quoted first.

    The declaration is read off the CLASS, the way A8 reads it, so this clause
    says what the operator says rather than naming ``cw_tone`` here: the day a
    second class ships a ``must_precede``, both checks generalise together.
    """
    cls = _t5_radio_class(spec)
    required = tuple(getattr(cls, "must_precede", ()) or ())
    home = getattr(cls, "graph_node", None)
    if node not in required or not isinstance(home, str):
        return ("Compose them under one key instead -- compose: cascade at a "
                "transform node, compose: sum at a source node, which is how "
                "this document spells At(...)")
    return (f"Give it its own node, {home!r}: {cls.__name__} declares "
            f"must_precede={list(required)}, so it has to come BEFORE the "
            f"{node!r} operator and composing the two under {node!r} puts it "
            f"inside the stage it is there to track -- check A8, which fires "
            "on this same node, says what that costs")


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

    **A COMPOSED stage is read as well as a single one, and it was the hole
    check A5 sent readers into.**  ``_t5_claims`` answers ``(key,)`` for a
    ``compose:`` block and this function used to ask the composing MAPPING for
    a ``python:``, which a composing mapping never carries -- so ``gain:
    {compose: cascade, stages: [<the gain>, <the tone>]}`` was silent.  That
    document is A5's own advice applied verbatim (*"Compose them under one
    key"*), and measured, it produced a **clean report** with the tone still
    inside the gain slot: the remedy silenced the check rather than fixing
    what it was about.  :func:`_t4_entries` is the expansion, the one Task 4
    already wrote for A7.

    **Stage 0 of a cascade is not a violation**, and that is the rule rather
    than a concession: ``compose: cascade`` builds ``Pipeline(*stages)``
    (``compose.py:265``), which applies them in order at one node, so an
    operator written first has every other stage at that node downstream of
    it -- the node's own included.  ``check_stage_ordering``
    (``core/pipeline.py:129-131``) enforces the same relation one phase later
    and only between stages the document gave a ``name:`` to, which is the
    second backstop a cascade naming none of them escapes.
    """
    graph = _t4_graph()
    lit = _lit(document)
    for key, spec in _nodes(document).items():
        if key not in graph.nodes:
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
        # A `many` node's entries are separate operators at one node, NOT a
        # sequence, so stage 0 excuses nothing there -- the concession below
        # is `cascade`'s alone.
        composing = isinstance(spec, Mapping) and "compose" in spec
        cascade = composing and spec.get("compose") == "cascade"
        if composing and not cascade:
            # `compose: sum` at a transform node, and every other spelling,
            # are `_compose`'s own refusals in their own words
            # (``compose.py:237-252``) -- and this check's sentence in front
            # of one would name a fix that is not the fault (Task 5's rule).
            # A `sum` has no order for "stage 0" to mean anything about
            # either: its branches run in parallel on the same input, which
            # is why ``check_stage_ordering`` deliberately says nothing about
            # a ``SumOperator`` (``core/pipeline.py:117-120``).
            continue
        for index, (where, entry) in enumerate(
                _t4_entries(key, spec, many=graph.nodes[key].many)):
            cls = _t5_radio_class(entry)
            required = tuple(getattr(cls, "must_precede", ()) or ())
            if not required:
                continue
            if node in required:
                if cascade and index == 0:
                    continue
                if cascade:
                    yield refuse(
                        "A8", longest_legal_prefix(f"model.{where}"),
                        f"model.{where}: puts {cls.__name__} at node {node!r} "
                        f"-- the node it declares it must precede -- as stage "
                        f"{index} of a compose: cascade, which applies its "
                        f"stages in order at ONE node, so this operator is "
                        f"injected after stage {index - 1} rather than before "
                        f"the {node!r} operator. {cls.must_precede_because} "
                        f"Neither backstop sees it: assemble() sees one "
                        f"placement at {node!r} (core/fold.py:271-274), and "
                        f"check_stage_ordering compares only stages the "
                        f"document gave a name: to "
                        f"(core/pipeline.py:129-131). Make it stage 0 of the "
                        f"cascade, or give it its own node, "
                        f"{cls.graph_node!r} (check A8).")
                    continue
                yield refuse(
                    # `_task3_where` on every leg: a FAN label is a name the
                    # user chose, and an un-spellable `where` raises OUTSIDE
                    # the per-check `try` and kills the whole pass.  Measured
                    # identity on a node id, which is what the single-entry
                    # leg passes it.
                    "A8", longest_legal_prefix(f"model.{where}"),
                    f"model.{where}: puts {cls.__name__} IN the {node!r} "
                    f"slot, so this document declares no {node!r} operator "
                    f"for it to pass through -- it replaced the stage it is "
                    f"there to track. {cls.must_precede_because} assemble() "
                    f"cannot say this: it sees one placement, and an absent "
                    f"stage is deliberately no violation there "
                    f"(core/fold.py:271-274), while the document still has "
                    f"the key and the operator apart. Give it its own node, "
                    f"{cls.graph_node!r} (check A8).")
                continue
            blocked = [target for target in required
                       if target in lit
                       and target not in _t5_downstream(graph, node)]
            if blocked:
                yield refuse(
                    "A8", longest_legal_prefix(f"model.{where}"),
                    f"model.{where}: places {cls.__name__} at {node!r}, from "
                    f"which {blocked} cannot be reached -- and this document "
                    f"lights {'them' if len(blocked) > 1 else 'it'}, so "
                    f"nothing this operator contributes ever passes through. "
                    f"{cls.must_precede_because} Place it upstream: its own "
                    f"node is {cls.graph_node!r} (check A8).")


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
#: 594-660``, its ``def`` and ``end_lineno`` by AST) contains no ``_space(``,
#: no ``forward_fn``, no ``build_forward_fn`` and no ``fit_twin`` at all.
#:
#: Written as the COMPLEMENT on purpose (§3.2 (e) 2): a kind added to
#: ``_KINDS`` defaults to FITTING, so a new exit inherits the check rather
#: than escaping it.  A wrong refusal is loud; a lost check is silent.
#:
#: **Two tests, because they catch different things and the subset alone
#: catches almost nothing.**  ``test_the_complement_is_a_subset_of_the_
#: declared_kinds`` catches a TYPO here -- ``"forwards"`` would never match
#: and every forward document would become a refusal.  It cannot catch a new
#: kind: measured, adding one to ``_KINDS`` leaves a proper subset even more
#: proper and that test exits 0.  ``test_every_declared_kind_is_classified``
#: pins ``_KINDS`` by MEMBERSHIP and is what makes the day a genuinely
#: non-fitting kind ships a day someone looks; its failure message is the
#: instruction to classify the new one.
_A30_NOT_FITTING: frozenset[str] = frozenset({"forward", "mmodes"})

def _a30_exits(document: Mapping[str, Any]) -> tuple[str, ...]:
    """The declared kinds whose exit A30 is actually about, sorted.

    :func:`~rheplicant.config.preflight.fitting._kinds` is the set of kinds
    the document declares and it filters **nothing** -- not a kind the run
    grammar refuses, not a run whose refusal is the point of writing it.  A30
    was its one production caller and inherited both holes; this narrows it
    with the two facts ``_kinds`` deliberately does not carry, and both were
    measured.

    **A kind ``runs._KINDS`` does not contain is not an exit.**  Measured,
    ``runs: [{kind: banana}]`` earned A30 the sentence *"This document
    declares kind: banana, and every exit but forward and mmodes closes the
    fit twin over ONE template state"* -- a claim about the closure behaviour
    of a kind that does not exist.  ``parse_runs`` (``runs.py:87-90``) names
    the real fault, and on the ``load_document`` path nothing names it at
    all, which makes an invented claim worse rather than harmless.  §3.2 (e)
    2's rule survives intact: a kind ADDED to ``_KINDS`` still defaults to
    fitting, because the complement is still what classifies it, and
    ``test_every_declared_kind_is_classified`` still dates that.

    **A run declaring ``expect: refuse`` is not counted**, which is
    ``_blocks``' and ``_prior_gates``' clause reaching the check that never
    considered it -- ``grep -n expect preflight/model.py`` had no match before
    this commit.  A24/A25 and A27/A28 deliberately do NOT stand down there,
    and their reason is that no document in this repository expects their
    refusal; A30 has both the document and something stronger.  Measured with
    A30 bypassed: ``kind: fisher, expect: refuse`` over a fit twin that keeps
    ``noise`` captures *"ParameterSpaceError: ... NoiseOperator at 'noise',
    which declares 'key' in requires"* -- the very refusal A30 hoists, so the
    assertion the run exists to make is A30's own subject.  And A30's advice
    applied to it (``inference.twin.without: [noise]``) gives ``ConfigError:
    runs['fisher']: expect: refuse, and kind: fisher SUCCEEDED``: the fix
    trades one refusal for another, which is the advice-loop shape this
    commit exists to close.

    The gate is per RUN, so a ``kind: fisher`` that expects nothing still
    earns A30 beside an ``expect: refuse`` sibling of the same kind.
    """
    from rheplicant.config.sections.runs import _KINDS

    live = {run["kind"] for run in _runs(document)
            if isinstance(run.get("kind"), str)
            and run.get("expect") != "refuse"}
    return tuple(sorted(
        (_kinds(document) & live & frozenset(_KINDS)) - _A30_NOT_FITTING))


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
    ``inference.twin.without:`` names node ids -- it calls
    ``Assembly.without`` (``core/graph.py:526``, from ``twin.py:59-60``) --
    and a ``python:`` entry lands where its class declares rather than under
    the key it is written beneath.  Measured at ``0263e0f``: ``{emi: {python:
    'rheplicant.radio:NoiseOperator', ...}}`` with ``without: [noise]`` BUILDS
    and its fit twin is clean, while ``without: [emi]`` is the one the
    assembly itself refuses.  A reader keyed on the model key gets both of
    those backwards -- one a false refusal, one a lost check.

    :func:`_t5_claims` is the one placement reader in this file (§2.2), and a
    REGION's operator sits at the LAST node it covers -- ``paths.
    refuse_misaddressed_region`` says so and the assembly agrees (measured:
    ``at: ['noise', 'emi']`` reports the stage at ``emi``).  ``placed[-1]``
    is therefore right for a single placement and for a region alike.

    **Where two entries land on one node, the OCCUPANT is named** -- the entry
    written under the node it fills, and failing that the alphabetically first
    key.  ``_two_at_one_node`` picks the same way (``occupant = node if node
    in keys else keys[0]``, over a list built in document order), and the
    reason is Task 5's: naming whichever key the user happened to type first
    makes the same collision blame a different line when the document is
    reordered.  That document is check A5's refusal as well, and A30's fix --
    drop the node -- is the same whichever key it names, but the blame still
    has to be stable.
    """
    claims: dict[str, list[tuple[str, str]]] = {}
    for key, spec in _nodes(document).items():
        placed = _t5_claims(key, spec)
        if not placed:
            continue
        node_id = placed[-1]
        found = _a30_stochastic(node_id, spec, table)
        if found is not None:
            claims.setdefault(node_id, []).append((key, found))
    placements: dict[str, tuple[str, str]] = {}
    for node_id, entries in claims.items():
        key, found = next(
            (one for one in entries if one[0] == node_id), min(entries))
        placements[node_id] = (f"model.{key}", found)
    return placements


def stochastic_nodes(document: Mapping[str, Any]) -> frozenset[str]:
    """The node ids whose declared operator class declares ``RANDOMNESS``.

    §3.2 (f)'s shared name: Task 11 (A30) binds it and Task 12 (A42) imports
    it (``preflight/values.py:70``), because two private predicates for one
    property is the collision §3.2 (f) calls "the likeliest remaining
    collision in the plan".  **Public because it crosses a module boundary,
    and §3.2 (f) says so outright** -- an earlier draft of this sentence
    offered ``sections/observed.py:23`` as the precedent, which is
    ``from rheplicant.config.draws import _seed_name, seed_for``: a PRIVATE
    name pulled across a boundary, so it is evidence against the rule rather
    than for it.  (That module's ``__all__`` is at ``:28``.)

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

    **A ``replace:`` is read only for a node the model actually lights**, and
    the gate is one predicate for two shapes that produce the identical
    ``KeyError``.  ``Assembly.replace_node`` (``core/graph.py:473``) looks the
    node up in the repaired assembly, so it raises *"No node named 'X' in this
    assembly"* both when ``without:`` has just dropped X and when ``model:``
    never lit it -- and ``kind: pipeline`` is the third face of the same
    thing, an assembly with no nodes at all, which ``build_fit_twin`` refuses
    in its own words (*"inference.twin: repairs a graph assembly, and this
    model is kind: pipeline ... declare the fit pipeline as its own
    variant"*).  Measured at ``36b7e54``, all three reached A30, and on the
    unlit one the fix A30 named was itself an error: *"Write
    inference.twin.without: [rfi_field]"* gives ``AssemblyError: without
    ('rfi_field'): no operator sits at 'rfi_field' in this assembly``.  A
    refusal whose advice errors is worse than no refusal.

    :func:`_lit` is the predicate, and it is one place LOOSER than the
    assembly: a REGION lights every node it covers while its operator occupies
    only the last, so a ``replace:`` naming an interior node of a region would
    still be read.  Recorded rather than closed -- the tighter reader would be
    a second definition of "where does this operator sit", which §2.2 forbids.

    A malformed ``twin:`` -- not a mapping, or a ``without:`` that is not a
    list -- reads as NO repair rather than as a stand-down.  ``build_fit_twin``
    refuses both shapes in its own words one phase later, and A30's advice
    (the list form, spelled out) is the right thing to say first.  What it may
    never do is RAISE: a ``without:`` entry that is a list is unhashable, and
    popping by it would abort the pass and discard every other finding
    (§2.3's TRAP).

    **The kinds come from :func:`_a30_exits`, not from ``_kinds`` directly**,
    and the difference is two live defects: this check used to make a claim
    about the closure behaviour of ``kind: banana``, and to refuse a document
    whose only fitting run declared ``expect: refuse`` -- destroying the
    assertion that run exists to make, with advice whose own effect is
    ``expect: refuse, and kind: fisher SUCCEEDED``.  Both measured; see that
    function.

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
    fitting = _a30_exits(document)
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
        lit = _lit(document)
        for node_id, spec in replace.items():
            # The `isinstance` test is SUBSUMED by the `lit` test below --
            # every member of `_lit` is a RADIO_GRAPH node id, so a YAML
            # mapping key that is not a string can never be in it, and
            # measured, dropping the test changes no answer.  Kept because
            # what it protects against is `sorted(placements)` raising on
            # mixed key types, which is a property of THIS loop rather than
            # of the gate that happens to cover it today.
            if not isinstance(node_id, str) or node_id in dropped:
                continue
            if node_id not in lit:
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
        # `where` is the SITE and not the constant `inference.twin.without`:
        # two stochastic nodes give two findings, and a constant makes
        # `raise_if_refused`'s tail locate the second one by a path that names
        # nothing.  The line to ADD is spelled out in the message instead.
        findings.append(refuse("A30", site, (
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

    **A binding whose latent names cannot be read is DROPPED**, not carried
    with a stand-in, and a name ``inference.parameters`` does not DECLARE is
    dropped by the same rule.  Its own refusal is more specific and arrives
    with the value the user wrote -- *"inference.bindings[0]: latents: is a
    non-empty list of latent names; got 7."* (``transforms.py:369-374``), and
    *"inference.bindings[0]: 'ghost' is not a declared latent;
    inference.parameters declares ['g']."* -- and A33 answering first hands
    that reader a degeneracy lecture instead.  The membership half was
    measured LIVE: ``bindings: [{latents: ['ghost'], into:
    'bandpass.bandpass'}]`` earned A33 and was told to declare ``transform:
    unit_mean_bandpass``, which cannot help a latent that does not exist.
    This is ``_a23_prior_free``'s rule (``fitting.py:726-732``: *"``names``
    must already be names the document DECLARES WELL"*) applied on the side
    A33 left open -- A33's own docstring gates both path HEADS against
    ``_lit`` for exactly this shape and left the NAME ungated, which is 2C's
    "closed on one route, open on its twin" inside one function.
    ``inference.parameters.<n>`` entries need no filter: their name IS the
    declaration.  An
    earlier draft substituted the binding's ``where`` for its latent set, on
    the reasoning that no latent name can equal it; measured at ``36b7e54``
    that is true and beside the point, because it made A33 decide *"these are
    two different parameters"* from a value it had just failed to read, and
    it did so ASYMMETRICALLY -- the substitute counted as a difference on the
    gain side and as no difference on the bandpass side.

    ``inference.parameters.<n>`` entries are unaffected: their name is the
    mapping key, so it is always readable.

    ``parse_path`` RAISES on a malformed path (measured: ``''``, ``'a..b'``,
    ``None`` and ``['a']`` all give ``ConfigError``), and a check that raises
    aborts the pass and hides every later finding (§2.3's TRAP).  So an
    unparseable ``into:`` is skipped here and left to
    ``_selectors``/``parse_path`` at build time, which already names it.
    """
    section = document.get("inference")
    section = section if isinstance(section, Mapping) else {}
    declared = _latents(document)
    written: list[tuple[str, tuple[str, ...], Any, Any]] = [
        (f"inference.parameters.{name}", (name,), spec.get("into"),
         spec.get("transform"))
        for name, spec in declared.items()
        if spec.get("into") is not None
    ]
    bindings = section.get("bindings")
    if isinstance(bindings, (list, tuple)):
        for index, entry in enumerate(bindings):
            if not isinstance(entry, Mapping) or entry.get("into") is None:
                continue
            latents = entry.get("latents")
            latents = (latents,) if isinstance(latents, str) else latents
            names = tuple(one for one in latents
                          if isinstance(one, str) and one in declared) \
                if isinstance(latents, (list, tuple)) else ()
            if not names:
                continue
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
        out.append((where, frozenset(names), tuple(heads), transform))
    return tuple(out)


def _a33_convention(transform: Any) -> bool | None:
    """Is ``transform`` the bandpass's identifiability convention?

    Three answers.  ``True`` -- it is :data:`_A33_CONVENTION`.  ``False`` --
    it is absent, or a registered NAME that is not the convention:
    ``identity`` binds the leaf unchanged, ``exp`` and ``log`` are
    elementwise, ``sum`` reduces and ``split_rows`` re-shapes, and none of the
    five moves the mean, so the product of bandpass and gain stays the only
    constrained combination.  ``None`` -- **anything else**, which is text
    this pass cannot decide: a MAPPING transform is an arbitrary callable
    (``{python: ...}``) or an affine map whose operands may be value nodes; an
    unregistered NAME is ``parse_transform``'s own refusal
    (``transforms.py:179-182``); and every other type -- ``7``, ``[]``,
    ``True``, ``0``, ``""`` -- is *"is a name or a mapping; got ..."*
    (``transforms.py:183-184``).  All of them land here, not only the two the
    sentence used to name.

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
    node it writes -- is check C17, shipped in place by Plan 3B inside
    ``config/sections/`` (``sections/transforms.py``, ``sections/observed.py``)
    rather than left for a later plan to build; this half is two path heads,
    two latent names and a transform.

    **Measured absent today** at ``0263e0f``: a document lighting ``bandpass``
    and ``gain`` with a free latent into each and no transform builds with no
    refusal, and no document in ``tests/config/`` or ``docs/`` binds a latent
    into a ``bandpass.*`` path at all -- so this check refuses nothing that
    exists and its own tests are the only coverage it will have.
    ``exit_helpers.py:521-524`` calls its ``gain`` + ``global_signal.depth``
    pair *"the schema's A33 shape"*; that is an analogy about degeneracy and
    not an A33 document, because neither latent goes into ``bandpass``.

    **The MODEL is consulted as well as the bindings**, and that is a live
    correction rather than belt and braces.  An ``into:`` head is a node the
    user TYPED, not a node the document lights: measured at ``36b7e54``, a
    document whose model has no ``bandpass`` node at all earned A33 for
    ``into: bandpass.bandpass``, while the package's own sentence is *"Path
    'bandpass.bandpass' could not be walked against this twin: No node named
    'bandpass' in this assembly"*.  A typo'd head was answered with a
    degeneracy lecture and told to declare ``transform: unit_mean_bandpass``,
    which cannot help -- the path still resolves to nothing.  Both heads are
    gated, because a free ``gain`` the model does not light is the same
    mistake on the other side.
    """
    lit = _lit(document)
    if "bandpass" not in lit or "gain" not in lit:
        return ()
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
    return (refuse("A33", longest_legal_prefix(f"{where}.transform"), (
        f"{where} is free into bandpass and this document also frees a "
        "latent into gain. The receiver's bandpass and the gain multiply the "
        "same prediction, so only their PRODUCT is constrained: the fit has "
        "one exactly null direction and returns a finite, correctly-shaped "
        "answer in which the two have traded an arbitrary constant. Declare "
        f"transform: {_A33_CONVENTION} on the bandpass binding -- it divides "
        "out the mean, which is the convention that makes the pair "
        "identifiable (check A33).")),)
