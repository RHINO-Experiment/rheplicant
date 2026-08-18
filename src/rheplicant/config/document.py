"""One document -> one configured run (schema §3, the build order of §2).

The sequence is load-bearing: variants apply first (they patch the raw
document); the runtime facts decide the dtype; the observation is built next
because its grids and switch order ARE the resolution context; resources
resolve against that context; the model resolves against the context plus the
resources; and only then can an ingested State be finished, because
``to_state``'s ``source_order`` is read off the assembled twin's own switch
order (``rhino.py:607-611``).

Before any of it, the pre-flight pass: every check decidable from the
document's text, run on the variant-applied mapping and before
``build_runtime``.  That position is the whole point -- ``build_resources`` is
where a CST directory is read and a spherical harmonic transform runs, and
measured at ``be2027b`` a missing beam file out-ranked an unknown ``model:``
node, a junction given an operator, a ``flagging`` with no ``type:`` and two
capability-reserved keys, five for five.

**FOUR passes run here, and every one of their reports is kept.**  Pre-flight
and the axes pass inside :func:`_assemble`; the built pass and the post-flight
pass in the :func:`_through_built`/:func:`_through_priced` hooks, the second
of them after ``build_inference`` because its checks evaluate the twin.  Each
raises before it warns and each concatenates onto ``ConfiguredRun.report``, in
run order -- which is the only place a ``mode: report`` finding can live,
since it is neither a refusal nor a warning and every pass used to drop its
report on the floor.

**Plan 4A Task 10: the four hooks are the per-layer stages of
:mod:`rheplicant.config.orchestration`.**  :func:`load_document` asks
``prepare_document(scope="selected")`` for the one layer it has always built;
``run_document`` prepares all layers and executes the base schedule.  The
hooks themselves stay here -- the raise/warn pair per pass, in this file, is a
pinned census (``tests/config/test_config_inflight.py``) -- and the stage
stays honest: ``_build_with_axes`` is exactly ``_assemble``'s post-preflight
body, so the helper suites that drive ``_assemble`` read the same build the
orchestration runs.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, NamedTuple

from rheplicant.config.context import ResolutionContext
from rheplicant.config.errors import ConfigError
from rheplicant.config.findings import Report
from rheplicant.config.gating import gates
from rheplicant.config.inflight import Axes, Built, axes, built
from rheplicant.config.layering import apply_variant
from rheplicant.config.postflight import Priced, priced
from rheplicant.config.preflight import preflight
from rheplicant.config.resources import BuiltResources, build_resources
from rheplicant.config.sections.compose import build_model
from rheplicant.config.sections.inference import build_inference
from rheplicant.config.sections.observation import build_observation
from rheplicant.config.sections.runtime import (
    RuntimeFacts,
    build_runtime,
    state_key,
)
from rheplicant.core.coordinates import Coordinates
from rheplicant.core.state import State

if TYPE_CHECKING:
    from _rheplicant_bootstrap.types import LayerIdentity, TraceSink

__all__ = ["ConfiguredRun", "load_document", "run_forward"]


class ConfiguredRun(NamedTuple):
    """A document, built: the state, the twin, and everything that made them."""

    document: dict[str, Any]
    runtime: RuntimeFacts
    state: State
    twin: Any
    inference: Any
    resources: BuiltResources
    context: ResolutionContext
    #: Every finding this document earned, over all FOUR passes and in the
    #: order they ran: pre-flight, axes, built, post-flight.  A ``mode:
    #: report`` finding has nowhere else to go -- it is not a refusal and not
    #: a warning -- so without this field the whole ``report:`` half of schema
    #: §4.7.8 is computed and thrown away.  The record is in memory, on the
    #: object the caller already holds; serialising it is Plan 4's.
    #:
    #: DEFAULTED so that a caller constructing one by keyword need not know
    #: about it, and LAST because :class:`~rheplicant.config.inflight.Built`
    #: is spliced out of this tuple positionally (``Built(*run)``).
    #:
    #: Named ``report`` and not ``checks``: ``run.inference.checks`` is the
    #: DECLARATION -- what the document asked for -- and keeping both under
    #: one word is how a reader ends up asserting on the wrong one.
    report: Report = Report()


def _attach(report: Report, call, *args, **kwargs):
    """Run ``call``; if it refuses with a bare ``ConfigError``, attach the
    completed boundaries' report first.

    The attach is IN PLACE, and that is the contract of spec §8's "preserving
    args, cause, traceback, and any already attached report": the object the
    caller catches is the object the builder raised, with its own message,
    its own ``__cause__`` and its own traceback -- only ``report`` is filled,
    and only when the builder left it ``None``.  A refusal that already
    carries a report (a pass's own ``raise_if_refused``) is untouched.
    """
    try:
        return call(*args, **kwargs)
    except ConfigError as error:
        if error.report is None:
            error.report = report
        raise


def _complete_report_boundary(
    previous: Report,
    current: Report,
    *,
    stage,
    layer: LayerIdentity | None,
    trace: TraceSink | None,
) -> Report:
    """One boundary's accumulation and trace rows -> the cumulative report.

    The pass has ALREADY returned ``current`` (a callable that raised before
    returning never reaches here, and appends neither row).  The findings and
    the boundary are appended to the trace BEFORE the caller's
    ``raise_if_refused``, because the pass is complete once its report
    exists -- including when that report then refuses.  The raise and the
    warning emission stay at the call site: the four hook pairs in this file
    are a pinned census, and a helper that swallowed them would empty it.
    """
    cumulative = Report(findings=previous.findings + current.findings)
    if trace is not None:
        trace.record_findings(
            stage, layer,
            tuple(dataclasses.asdict(row) for row in current.findings))
        trace.boundary_completed(stage, layer)
    return cumulative


def _assemble(document: Mapping, *, variant: str | None = None,
              base_dir: str | None = None) -> ConfiguredRun:
    """:func:`load_document`'s body WITHOUT the built or the post-flight hook.

    Variants, the pre-flight pass, the axes hook, and every builder through
    ``build_inference``.  **Split out for one reason, and it is a testability
    one that is load-bearing rather than cosmetic:** those two hooks call
    ``raise_if_refused()`` before :func:`load_document` returns, so
    ``Built(*load_document(d))`` can never observe a built-slot REFUSAL -- the
    document that earns one never comes back.  Most built-slot rows are
    refusals, and so are most priced ones, so a test helper built on
    ``load_document`` could only ever exercise the passing half of the slot it
    exists to test.

    ``tests/config/inflight_helpers.built_run`` and ``priced_run`` both call
    THIS; a test about a HOOK -- that a refusal actually stops the load --
    calls :func:`load_document` under ``pytest.raises``.  **A later task that
    moves either hook down here re-arms exactly the failure this split was
    created to prevent**, for every built-slot and post-flight test at once.

    The report it returns carries the two passes it has run: pre-flight, then
    axes.  :func:`load_document` appends the two later ones.
    """
    if not isinstance(document, Mapping):
        raise ConfigError(
            f"A document is a mapping of sections; got "
            f"{type(document).__name__} ({document!r})."
        )
    doc = dict(document)
    if variant is not None:
        doc = apply_variant(doc, variant)
    # P-1, and its position is the point: `build_resources` below is where a
    # CST directory is read and a spherical harmonic transform runs.  Refuse
    # before you warn -- a document that is about to be refused should not
    # also spray warnings about lines the user is on their way to change.
    report = preflight(doc)
    report.raise_if_refused()
    report.emit_warnings()
    return _build_with_axes(doc, base_dir=base_dir, previous=report)


def _build_with_axes(document: Mapping, *, base_dir: str | None = None,
                     previous: Report, layer: LayerIdentity | None = None,
                     trace: TraceSink | None = None) -> ConfiguredRun:
    """The axes boundary, then every builder through ``build_inference``.

    This is ``_assemble``'s post-preflight body and the orchestration's
    per-layer build: one function, so the two routes cannot read the document
    differently.  ``previous`` is the already-completed text pre-flight
    report; a builder that refuses after the axes boundary has it attached
    (``_attach``), which is spec §8's rule that a later ``ConfigError``
    carries the completed boundaries' findings.
    """
    runtime = _attach(previous, build_runtime, document["runtime"])
    observation, context = _attach(previous, build_observation,
                                   document["observation"],
                                   runtime=runtime, base_dir=base_dir)
    # P-0.5, and its position is the point: `build_resources` on the next line
    # is the 90.9 % of this function's wall time that a bad time axis or a
    # non-uniform lst_deg has nothing to do with.  Raise before you warn, for
    # the same reason P-1 does; across slots that ordering cannot be global
    # (this pass has already returned when `build_resources` runs) and
    # `inflight/__init__.py` records why that is correct rather than tolerated.
    axis_report = _attach(previous, axes, Axes(document=document,
                                               runtime=runtime,
                                               observation=observation,
                                               context=context))
    cumulative = _complete_report_boundary(previous, axis_report,
                                           stage="axes", layer=layer,
                                           trace=trace)
    axis_report.raise_if_refused(cumulative=cumulative)
    axis_report.emit_warnings()
    resources = _attach(cumulative, build_resources,
                        document.get("resources") or {}, context)
    context = dataclasses.replace(context,
                                  resources=dict(resources.resources),
                                  ingest=observation.ingest)
    twin = _attach(cumulative, build_model, document["model"], context,
                   switch_order=observation.switch_order)

    if observation.ingest is not None:
        from rheplicant.radio.rhino import to_state

        source_order = list(observation.switch_order) or ["antenna"]
        state = _attach(cumulative, to_state, observation.ingest,
                        source_order=source_order)
        state = state.replace(
            coords=state.coords.replace(
                pointing=observation.pointing,
                extra={**state.coords.extra, **observation.extra},
            ),
            env=observation.env,
            key=state_key(runtime),
            meta=state.meta | observation.meta,
        )
    else:
        state = State(
            data=observation.data,
            coords=Coordinates(time=observation.time_s,
                               freq=observation.freq_hz,
                               pointing=observation.pointing,
                               extra=observation.extra),
            env=observation.env,
            aux=observation.aux,
            key=state_key(runtime),
            meta=observation.meta,
        )
    inference = _attach(cumulative, build_inference, document.get("inference"),
                        twin=twin, state=state, observation=observation,
                        context=context)
    # The two reports this function raised on and emitted are KEPT rather
    # than dropped: the carried report is the cumulative one, in pass order.
    return ConfiguredRun(document=document, runtime=runtime, state=state,
                         twin=twin, inference=inference, resources=resources,
                         context=context, report=cumulative)


def _carrying(run: ConfiguredRun, found: Report) -> ConfiguredRun:
    """``run`` with ``found`` appended to the report it already carries.

    One binding for the concatenation, because ``Report`` has no ``+``: two
    hand-written ``Report(findings=a.findings + b.findings)`` in one function
    is two chances to write ``found.findings + run.report.findings`` and put
    a later pass's findings in front of an earlier one's.  The staged hooks
    below use it, and ``tests/config/inflight_helpers.priced_run`` drives it
    directly.
    """
    return run._replace(
        report=Report(findings=run.report.findings + found.findings))


def _priced_payload(run: ConfiguredRun) -> Priced:
    """The post-flight payload for a run every earlier pass has finished with.

    Here rather than inside ``postflight`` so that ``priced(run)`` mirrors
    ``axes(facts)`` and ``built(run)`` -- an entry point that takes its
    payload -- and here rather than inline in :func:`_through_priced` so that
    ``tests/config/inflight_helpers.priced_run`` reaches the SAME reading of
    the document.  A second reading is how "which section the gates come
    from" ends up answered twice and differently.

    ``inference:`` may be absent, and this pass still runs (§3.2 (b)): a
    document with no latents has ``space is None`` and every gate stands down,
    but C16 needs no space and is exactly the check a ``kind: forward``
    document wants.  ``gating.gates`` returns all three gates with their
    defaults for a ``None`` section, so there is no branch here.
    """
    declared = run.document.get("inference")
    section = declared.get("checks") if isinstance(declared, Mapping) else None
    return Priced(run=run, gates=gates(section))


def _through_built(run: ConfiguredRun, *, layer: LayerIdentity | None = None,
                   trace: TraceSink | None = None) -> ConfiguredRun:
    """The built-pass hook: raise, then warn, then carry the findings.

    P-1.5, immediately before the parse stage and the return: after
    ``build_inference``, so one payload carries the raw twin, the fit twin
    and the space.  Being late costs nothing (``build_inference`` is 2.4 % of
    a warm load) and buys ``Built`` mirroring ``ConfiguredRun`` field for
    field.  **This slot saves no money** -- the beam is long since read --
    and the sentence "before any beam is analysed" is false about every check
    registered here.
    """
    built_report = _attach(run.report, built, Built(*run))
    cumulative = _complete_report_boundary(run.report, built_report,
                                           stage="built", layer=layer,
                                           trace=trace)
    built_report.raise_if_refused(cumulative=cumulative)
    built_report.emit_warnings()
    return _carrying(run, built_report)


def _through_priced(run: ConfiguredRun, *,
                    layer: LayerIdentity | None = None,
                    trace: TraceSink | None = None) -> ConfiguredRun:
    """The post-flight hook: the checks that have to RUN the thing.

    P-2.5, after the parse stage and before the return: a forward pass per
    linear claim, a ``jacfwd`` plus an SVD, two Newton solves.  It saves
    nothing and buys that they run at all, in this layer's voice, rather than
    detonating inside a fit.
    """
    priced_report = _attach(run.report, priced, _priced_payload(run))
    cumulative = _complete_report_boundary(run.report, priced_report,
                                           stage="postflight", layer=layer,
                                           trace=trace)
    priced_report.raise_if_refused(cumulative=cumulative)
    priced_report.emit_warnings()
    return _carrying(run, priced_report)


def load_document(document: Mapping, *, variant: str | None = None,
                  base_dir: str | None = None) -> ConfiguredRun:
    """Build the objects a document declares, in the order they feed each other.

    The selected layer of the canonical preparation: the text pre-flight fans
    over every declared layer (an unselected variant's text fault still
    refuses), the selected layer alone is built, and its effective schedule
    is handler-parsed once against its build for validation.  Returns only a
    complete ``ConfiguredRun`` -- there is no partial union and no
    non-raising replacement API (spec §8).
    """
    from rheplicant.config.orchestration import prepare_document

    prepared = prepare_document(document, scope="selected", variant=variant,
                                base_dir=base_dir)
    return prepared.layers[0].configured


def run_forward(run: ConfiguredRun | Mapping, *, variant: str | None = None,
                base_dir: str | None = None) -> State:
    """Evaluate the twin on the state -- 2A's one exit.

    Plain evaluation, deliberately: jit is the caller's choice
    (``eqx.filter_jit(run.twin)(run.state)``), and the Assembly's own guards
    speak for a data/source mismatch.  The runs: section's forward exit calls
    this; a bare mapping must now declare runs: like any document.
    """
    if not isinstance(run, ConfiguredRun):
        run = load_document(run, variant=variant, base_dir=base_dir)
    return run.twin(run.state)
