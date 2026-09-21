"""Passes about the radio graph: what a node is, and where it may sit.

Task 5's class, claim and placement checks, the switch order, and the two
refusals that catch a node claimed twice. `_t5_claims` and `_t5_placement`
call each other, so they are one module by necessity rather than by taste,
and `_lit` is here because it reads the claims.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable, Mapping
from typing import Any

from rheplicant.config.errors import ConfigError
from rheplicant.config.findings import Finding, refuse
from rheplicant.config.preflight import register
from rheplicant.config.sections.compose import (
    cal_load_order_problem,
    model_nodes,
)

from .model_walk import (
    _t4_at_nodes,
    _t4_graph,
)


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
    for key, spec in model_nodes(document).items():
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
    (``observation.py::_data``), while every other run goes through
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
    spec = model_nodes(document).get("cal_loads")
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
    through ``hatch.import_target`` (``sections/model.py::_c7_beam_spill``), which imports
    **any** module -- so the two spellings of ONE class object diverged, and
    six checks read the divergence.  Measured: ``import_target
    ('rheplicant.radio.instrument.calibration:CalLoadOperator') is
    CalLoadOperator`` is True, while the exported spelling gave
    :func:`_t5_claims` ``('cal_loads',)`` and the submodule spelling ``()`` --
    which A14 refuses **on absence**, so a document that BUILDS was refused,
    and A5, A15, A31, A52 and A30's ``replace:`` gate each lost their subject
    on the same document.  Task 12 had recorded the hole for A42 alone
    (``preflight/values.py::_a42_removed``: *"two different resolvers, and only one
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
    already (``projectors.py``), so every module defining an
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
    return shipped if getattr(sys.modules.get(module), attribute, None) is shipped else None


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

    **This mirrors ``_single`` (``compose.py::compose_shape_problem``) clause for clause**,
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
      graph node of its own"* (``compose.py::cal_load_order_problem``); beside
      ``snapshot_before:`` it is *"at: and snapshot_before: together are not a
      combination this layer writes"* (``compose.py::cal_load_order_problem``); and in the
      STRING spelling it must restate its own key
      (``compose.py::cal_load_order_problem``).  Answering ``()`` for those three is what
      stops A5 and A8 pre-empting the sentence that names the real fault with
      one that names a no-op -- ``cw_tone: {python: ...CWCalibrationOperator,
      at: 'noise'}`` would otherwise be told to give the tone its own node
      ``cw_tone``, which is the key it is already written under.  A LIST of
      one is NOT held to the restatement rule: ``refuse_misaddressed_region``
      returns early below two nodes (``config/paths.py::refuse_misaddressed_region``) and, measured,
      ``bandpass: {python: ...GainOperator, at: ['gain']}`` builds with the
      operator at ``gain``;
    * ``at: null`` is no ``at:`` at all -- ``_single`` pops it and tests ``is
      not None`` -- and so is ``snapshot_before: null``.  Measured, a
      ``snapshot_before: null`` beside ``at: ['gain']`` really does place at
      ``gain``: the assembly answers "Two operators provided for node 'gain'";
    * ``snapshot_before:`` wraps the operator in ``At(node_id, ...)``
      (``compose.py::double_count_problem``), which OVERRIDES a ``python:`` class's own
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
    reachability IS ``_check_ordering``'s rule (``fold.py::_check_ordering``) and a copy of
    it here is a second definition of "after" that can drift from the one the
    assembly enforces.
    """
    from rheplicant.core.fold import _descendants

    return _descendants(graph, node)


@register("A5")
def _two_at_one_node(document: Mapping[str, Any]) -> Iterable[Finding]:
    """Check A5: two operators claim one node that holds a single instance.

    ``compose:`` is the document's spelling of the ``At(...)`` route
    ``core/graph_placement.py::_place_at_node`` names, so this message ends where that one does.
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
    specs = model_nodes(document)
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
                "A5",
                f"model.{intruder}",
                f"model.{intruder}: puts a second operator at node {node!r}, "
                f"which model.{occupant} already fills, and this node accepts "
                "a single instance. "
                f"{_a5_remedy(node, specs.get(intruder))} (check A5).",
            )


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
        return (
            "Compose them under one key instead -- compose: cascade at a "
            "transform node, compose: sum at a source node, which is how "
            "this document spells At(...)"
        )
    return (
        f"Give it its own node, {home!r}: {cls.__name__} declares "
        f"must_precede={list(required)}, so it has to come BEFORE the "
        f"{node!r} operator and composing the two under {node!r} puts it "
        f"inside the stage it is there to track -- check A8, which fires "
        "on this same node, says what that costs"
    )
