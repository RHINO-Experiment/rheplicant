"""Passes about stochastic nodes and the fit twin.

A30's exits and placements, A33's convention, and the check that a stochastic
node the likelihood accounts for is actually in the twin that simulated the
data. `stochastic_nodes` is the walk these share.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from rheplicant.config.findings import Finding, refuse
from rheplicant.config.preflight import register
from rheplicant.config.preflight.fitting import _kinds, _latents, _runs
from rheplicant.config.sections.compose import (
    model_nodes,
)
from rheplicant.config.sections.model import (
    operator_table,
)
from rheplicant.config.sections.transforms import _NAMED
from rheplicant.core.contract import RANDOMNESS

from .model_radio import (
    _lit,
    _t5_claims,
    _t5_radio_class,
)

#: The two exits that build neither a ``ParameterSpace`` over the fit twin nor
#: a forward function from it, so a stochastic stage is legal under them.
#: Everything else in ``sections/runs._KINDS`` does -- measured one kind at a
#: time through ``run_document`` on a ``twin: {without: []}`` document:
#: ``forward`` RUNS (``_run_forward`` is ``return built.twin(built.state)``,
#: ``exits.py``); thirteen kinds raise ``ParameterSpaceError`` naming
#: *"NoiseOperator at 'noise', which declares 'key' in requires"*; ``predict``
#: reaches ``space.forward_fn`` (``sections/diagnostics.py``) and its ``reuse:``
#: can only name a kind that refuses first (``_DRAW_SOURCES``,
#: ``sections/diagnostics.py::_parse_mmodes``); and ``_run_mmodes`` (``diagnostics.py:
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
_A30_NOT_FITTING: frozenset[str] = frozenset({"forward", "mmodes", "compare", "benchmark"})


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
    declares kind: banana, and these fitting exits close the fit twin over
    ONE template state"* -- a claim about the closure behaviour
    of a kind that does not exist.  ``parse_runs`` (``runs.py::_one``) names
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

    live = {
        run["kind"]
        for run in _runs(document)
        if isinstance(run.get("kind"), str) and run.get("expect") != "refuse"
    }
    return tuple(sorted((_kinds(document) & live & frozenset(_KINDS)) - _A30_NOT_FITTING))


#: The registry name of the identifiability convention A33 advises.
#: ``transforms._NAMED`` holds it (``transforms.py``) and it resolves to
#: ``radio.instrument.receiver.unit_mean_bandpass`` (``transforms.py::_beam_analysis``).  Advising
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
    because ``requires`` is a ``ClassVar`` (``operator.py::AbstractOperator``), so the
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


def _a30_placements(document: Mapping[str, Any], table: Mapping) -> dict[str, tuple[str, str]]:
    """node id -> ``(the site that put it there, the class name)``.

    **Keyed by NODE, not by the model key**, and that is the whole reason this
    is a function rather than a dict comprehension over :func:`model_nodes`:
    ``inference.twin.without:`` names node ids -- it calls
    ``Assembly.without`` (``core/graph.py::Assembly.without``, from ``inflight/twin.py``) --
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
    for key, spec in model_nodes(document).items():
        placed = _t5_claims(key, spec)
        if not placed:
            continue
        node_id = placed[-1]
        found = _a30_stochastic(node_id, spec, table)
        if found is not None:
            claims.setdefault(node_id, []).append((key, found))
    placements: dict[str, tuple[str, str]] = {}
    for node_id, entries in claims.items():
        key, found = next((one for one in entries if one[0] == node_id), min(entries))
        placements[node_id] = (f"model.{key}", found)
    return placements


def stochastic_nodes(document: Mapping[str, Any]) -> frozenset[str]:
    """The node ids whose declared operator class declares ``RANDOMNESS``.

    §3.2 (f)'s shared name: Task 11 (A30) binds it and Task 12 (A42) imports
    it (``preflight/values.py``), because two private predicates for one
    property is the collision §3.2 (f) calls "the likeliest remaining
    collision in the plan".  **Public because it crosses a module boundary,
    and §3.2 (f) says so outright** -- an earlier draft of this sentence
    offered ``observed.py`` as the precedent, which is
    ``from rheplicant.config.draws import _seed_name, seed_for``: a PRIVATE
    name pulled across a boundary, so it is evidence against the rule rather
    than for it.  (That module's ``__all__`` is at ``observed.py``.)

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
    (``inflight/twin.py``): ``without:`` first, then ``replace:``.  Reading
    ``replace:`` is what keeps this honest on the twin route -- measured, a
    document whose only ``twin:`` key is ``replace: {noise:
    RadiometerNoiseOperator}`` builds a fit twin that still draws, and a check
    reading ``model:`` alone would name the wrong class about it.

    **A ``replace:`` is read only for a node the model actually lights**, and
    the gate is one predicate for two shapes that produce the identical
    ``KeyError``.  ``Assembly.replace_node`` (``core/graph.py::Assembly.replace_node``) looks the
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
    dropped = (
        tuple(one for one in dropped if isinstance(one, str))
        if isinstance(dropped, (list, tuple))
        else ()
    )
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
                placements[node_id] = (f"inference.twin.replace.{node_id}", found)
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
        findings.append(
            refuse(
                "A30",
                site,
                (
                    f"{site} puts {operator} at node {node_id!r}, which draws its own "
                    f"randomness -- {operator} declares {RANDOMNESS!r} in requires "
                    "-- and inference.twin.without: does not drop it. This document "
                    f"declares {named}, and these fitting exits close "
                    "the fit twin over ONE template state, so that draw would be the "
                    "SAME realisation added to every prediction alike: a bias that is "
                    "exactly affine and full rank, which is why no shape check, no "
                    "linearity check and no rank test sees it. Write "
                    f"inference.twin.without: [{node_id}] -- kind: forward keeps the "
                    "node, and simulating with it is what it is for (check A30)."
                ),
            )
        )
    return tuple(findings)


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
    (``transforms.py::_beam_analysis``); and every other type -- ``7``, ``[]``,
    ``True``, ``0``, ``""`` -- is *"is a name or a mapping; got ..."*
    (``transforms.py::_beam_analysis``).  All of them land here, not only the two the
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
