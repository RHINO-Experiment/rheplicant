"""Passes about where a run's data comes from.

Tone placement, data declared with sources, and the refusal for a run that
declares neither. These read the walk and the radio passes and are read by
nothing, which is what makes them separable.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from _rheplicant_bootstrap.path_syntax import longest_legal_prefix
from rheplicant.config.findings import Finding, refuse
from rheplicant.config.preflight import register
from rheplicant.config.sections.compose import (
    model_nodes,
    node_placement_problems,
)

from .model_radio import (
    _lit,
    _t5_claims,
    _t5_downstream,
    _t5_placement,
    _t5_radio_class,
)
from .model_walk import (
    _t4_entries,
    _t4_graph,
)


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
    ``compose.py::cal_load_order_problem``, so a tone written the ordinary way -- a
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
    skips ``target in path``, and ``fold.py::_check_ordering`` and
    ``pipeline.py::check_stage_ordering`` argue at length that an absent stage is nothing to
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
    (``compose.py::compose_shape_problem``), which applies them in order at one node, so an
    operator written first has every other stage at that node downstream of
    it -- the node's own included.  ``check_stage_ordering``
    (``pipeline.py::check_stage_ordering``) enforces the same relation one phase later
    and only between stages the document gave a ``name:`` to, which is the
    second backstop a cascade naming none of them escapes.
    """
    graph = _t4_graph()
    lit = _lit(document)
    for key, spec in model_nodes(document).items():
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
            # (``compose.py::composition_problem``) -- and this check's sentence in front
            # of one would name a fix that is not the fault (Task 5's rule).
            # A `sum` has no order for "stage 0" to mean anything about
            # either: its branches run in parallel on the same input, which
            # is why ``check_stage_ordering`` deliberately says nothing about
            # a ``SumOperator`` (``pipeline.py::check_stage_ordering``).
            continue
        for index, (where, entry) in enumerate(_t4_entries(key, spec, many=graph.nodes[key].many)):
            cls = _t5_radio_class(entry)
            required = tuple(getattr(cls, "must_precede", ()) or ())
            if not required:
                continue
            if node in required:
                if cascade and index == 0:
                    continue
                if cascade:
                    yield refuse(
                        "A8",
                        longest_legal_prefix(f"model.{where}"),
                        f"model.{where}: puts {cls.__name__} at node {node!r} "
                        f"-- the node it declares it must precede -- as stage "
                        f"{index} of a compose: cascade, which applies its "
                        f"stages in order at ONE node, so this operator is "
                        f"injected after stage {index - 1} rather than before "
                        f"the {node!r} operator. {cls.must_precede_because} "
                        f"Neither backstop sees it: assemble() sees one "
                        f"placement at {node!r} (core/fold.py:271), and "
                        f"check_stage_ordering compares only stages the "
                        f"document gave a name: to "
                        f"(core/pipeline.py:129). Make it stage 0 of the "
                        f"cascade, or give it its own node, "
                        f"{cls.graph_node!r} (check A8).",
                    )
                    continue
                yield refuse(
                    # `_task3_where` on every leg: a FAN label is a name the
                    # user chose, and an un-spellable `where` raises OUTSIDE
                    # the per-check `try` and kills the whole pass.  Measured
                    # identity on a node id, which is what the single-entry
                    # leg passes it.
                    "A8",
                    longest_legal_prefix(f"model.{where}"),
                    f"model.{where}: puts {cls.__name__} IN the {node!r} "
                    f"slot, so this document declares no {node!r} operator "
                    f"for it to pass through -- it replaced the stage it is "
                    f"there to track. {cls.must_precede_because} assemble() "
                    f"cannot say this: it sees one placement, and an absent "
                    f"stage is deliberately no violation there "
                    f"(core/fold.py:271), while the document still has "
                    f"the key and the operator apart. Give it its own node, "
                    f"{cls.graph_node!r} (check A8).",
                )
                continue
            blocked = [
                target
                for target in required
                if target in lit and target not in _t5_downstream(graph, node)
            ]
            if blocked:
                yield refuse(
                    "A8",
                    longest_legal_prefix(f"model.{where}"),
                    f"model.{where}: places {cls.__name__} at {node!r}, from "
                    f"which {blocked} cannot be reached -- and this document "
                    f"lights {'them' if len(blocked) > 1 else 'it'}, so "
                    f"nothing this operator contributes ever passes through. "
                    f"{cls.must_precede_because} Place it upstream: its own "
                    f"node is {cls.graph_node!r} (check A8).",
                )


@register("A31")
def _data_with_sources(document: Mapping[str, Any]) -> Iterable[Finding]:
    """Check A31: ``observation.data`` while ``model`` lights a source node.

    The predicate is ``Assembly.has_source``' own -- node KINDS, not operators
    (``core/graph.py::Assembly``) -- evaluated statically over the lit ids.
    ``test_the_static_source_predicate_is_the_assemblys_own`` holds the
    package's own ``has_source`` against that static reading over six models
    and they agree in all six, so this pins the PACKAGE's predicate rather
    than this check's restatement of it.

    Two shapes stand down, both measured:

    * ``data:`` written EMPTY is no data at all -- ``_data`` returns ``None``
      for a ``None`` node (``observation.py::_extra``) and
      ``Assembly.__call__`` refuses only ``state.data is not None`` -- so
      ``"data" in section`` would refuse a document the package runs;
    * ``from_file:`` beside ``data:`` is ``build_observation``'s own refusal
      ("the recording IS the data", ``observation.py::_aux``), and it
      already precedes the beam (``preflight/document.py`` against ``preflight/document.py``).  A31
      answering first would offer "the twin makes it", which is true of a
      simulating document and wrong about a recording.

    ``observation.from_file`` ALONE puts the recording in ``state.data`` and
    is the same defect; it is NOT refused here.  See §3.2 (e) 1 and
    ``test_the_recording_route_is_recorded_and_not_yet_refused``, which
    carries the exact three-line widening: closing it turns
    ``test_config_document.py::TestLaterErrorsCarryTheCompletedReport.test_a_builder_error_carries_the_completed_passes_report.explode``
    red, and that test pins
    the refusal as the assembly's on purpose.
    """
    section = document.get("observation")
    if not isinstance(section, Mapping):
        return
    if section.get("data") is None or "from_file" in section:
        return
    graph = _t4_graph()
    sources = sorted(node for node in _lit(document) if graph.nodes[node].kind == "source")
    if not sources:
        return
    yield refuse(
        "A31",
        "observation.data",
        "observation.data: is the data a transform chain acts ON, and this "
        f"model lights the source nodes {sources} -- an assembly with sources "
        "GENERATES its own data, so the array declared here would be "
        "discarded rather than fitted. Drop observation.data (the twin makes "
        "it), or drop the sources to leave a transform chain (check A31).",
    )


@register("A31.no_source")
def _no_source_and_no_data(document: Mapping[str, Any]) -> Iterable[Finding]:
    """A31's other half: a model lighting no source, with no data to act on.

    ``Assembly.__call__`` refuses both mismatches between ``has_source`` and
    ``state.data``.  :func:`_data_with_sources` decides the first in text;
    this decides the second, which used to reach the user as an
    ``AssemblyError`` traceback from inside the first run -- after
    ``rheplicant validate`` had printed "configuration valid".

    The predicate is again the assembly's own, node KINDS over the placed
    ids, and it refuses on ABSENCE, so :func:`_t5_placement`'s two
    stand-downs are read as such rather than as "nothing placed":

    * ``None`` -- a ``python:`` class this pass will not import lands where
      that class declares, which may be a source;
    * ``()`` -- an entry the build refuses in its own words (a disagreeing
      ``at:``, a non-mapping spec, a class with no ``graph_node``), which a
      "no source" sentence would pre-empt.

    A model A2, A3 or A4 already refuses stands down too: a misspelled
    ``uniform_skyy:`` is one fault, and "lights no source" would send the
    reader after a second.  ``kind: pipeline`` is not an assembly and has no
    ``has_source``; ``model_nodes`` returns ``{}`` for it.

    Data is declared by a non-null ``observation.data`` or by
    ``observation.from_file``, whose recording becomes ``state.data``.

    **It does not read ``runs:``, and that is a declared false positive.**
    ``mmodes`` and ``compare`` never evaluate the twin (``_run_mmodes``
    reads resources and ``built.state.coords``; ``_run_compare`` reads
    earlier runs' products), so a transform-only model declaring only those
    kinds, and no ``inference.observed: {from: simulation}``, would run and
    is refused here.  Scoping the check needs a per-kind "evaluates the
    twin" property, and the exit registry (``sections/exit_support.
    register``) carries none: adding one is a fifth atomically bound table
    and a measured classification of all 18 registrations across seven
    modules, and a run-kind property alone would still miss
    ``inference.observed: {from: simulation}``, which evaluates the twin
    while the document is built.  ``test_the_run_kinds_are_not_read`` pins
    this scope, so narrowing it is a decision a test records.
    """
    section = document.get("observation")
    if not isinstance(section, Mapping):
        return
    if section.get("data") is not None or "from_file" in section:
        return
    specs = model_nodes(document)
    graph = _t4_graph()
    if not specs or node_placement_problems(specs, graph):
        return
    placed: set[str] = set()
    for key, spec in specs.items():
        nodes = _t5_placement(key, spec)
        if not nodes:
            return
        placed.update(nodes)
    if any(graph.nodes[node].kind == "source" for node in placed):
        return
    antenna_sources = [
        node
        for node, spec in graph.nodes.items()
        if spec.kind == "source"
        and not spec.reserved
        and "antenna_loss" in _t5_downstream(graph, node)
    ]
    choices = ", ".join(antenna_sources[:-1]) + f" or {antenna_sources[-1]}"
    lit = [node for node in graph.nodes if node in placed]
    yield refuse(
        "A31",
        "model",
        f"model: lights {lit} and no source node, so its twin is a pure "
        "transform chain, and observation declares no data for it to act on. "
        "Every run that evaluates the twin would stop with 'This assembly is "
        "a pure transform chain (no source operators)'. Light a source on "
        f"the antenna branch ({choices}), or declare the data the chain "
        "transforms: observation.data, or observation.from_file for a "
        "recording (check A31).",
    )
