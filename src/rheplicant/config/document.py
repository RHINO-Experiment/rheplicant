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
pass in :func:`load_document`, the second of them after ``build_inference``
because its checks evaluate the twin.  Each raises before it warns and each
concatenates onto ``ConfiguredRun.report``, in run order -- which is the only
place a ``mode: report`` finding can live, since it is neither a refusal nor a
warning and every pass used to drop its report on the floor.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any, NamedTuple

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
from rheplicant.config.sections.runtime import RuntimeFacts, build_runtime, state_key
from rheplicant.core.coordinates import Coordinates
from rheplicant.core.state import State

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
    runtime = build_runtime(doc["runtime"])
    observation, context = build_observation(doc["observation"],
                                             runtime=runtime,
                                             base_dir=base_dir)
    # P-0.5, and its position is the point: `build_resources` on the next line
    # is the 90.9 % of this function's wall time that a bad time axis or a
    # non-uniform lst_deg has nothing to do with.  Raise before you warn, for
    # the same reason P-1 does; across slots that ordering cannot be global
    # (this pass has already returned when `build_resources` runs) and
    # `inflight/__init__.py` records why that is correct rather than tolerated.
    axis_report = axes(Axes(document=doc, runtime=runtime,
                            observation=observation, context=context))
    axis_report.raise_if_refused()
    axis_report.emit_warnings()
    resources = build_resources(doc.get("resources") or {}, context)
    context = dataclasses.replace(context,
                                  resources=dict(resources.resources),
                                  ingest=observation.ingest)
    twin = build_model(doc["model"], context,
                       switch_order=observation.switch_order)

    if observation.ingest is not None:
        from rheplicant.radio.rhino import to_state

        source_order = list(observation.switch_order) or ["antenna"]
        state = to_state(observation.ingest, source_order=source_order)
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
    inference = build_inference(doc.get("inference"), twin=twin, state=state,
                                observation=observation, context=context)
    # The two reports this function raised on and emitted are KEPT rather
    # than dropped, and this is the only place they can be: both are locals
    # of this function, and `load_document` -- which appends the two later
    # passes -- never sees either.  `Report` has no `+`, no `merge` and no
    # `extend`; concatenating the tuples is the spelling everywhere.
    return ConfiguredRun(document=doc, runtime=runtime, state=state,
                         twin=twin, inference=inference, resources=resources,
                         context=context,
                         report=Report(findings=report.findings
                                       + axis_report.findings))


def _carrying(run: ConfiguredRun, found: Report) -> ConfiguredRun:
    """``run`` with ``found`` appended to the report it already carries.

    One binding for the concatenation, because ``Report`` has no ``+``: two
    hand-written ``Report(findings=a.findings + b.findings)`` in one function
    is two chances to write ``found.findings + run.report.findings`` and put
    a later pass's findings in front of an earlier one's.
    """
    return run._replace(
        report=Report(findings=run.report.findings + found.findings))


def _priced_payload(run: ConfiguredRun) -> Priced:
    """The post-flight payload for a run every earlier pass has finished with.

    Here rather than inside ``postflight`` so that ``priced(run)`` mirrors
    ``axes(facts)`` and ``built(run)`` -- an entry point that takes its
    payload -- and here rather than inline in :func:`load_document` so that
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


def load_document(document: Mapping, *, variant: str | None = None,
                  base_dir: str | None = None) -> ConfiguredRun:
    """Build the objects a document declares, in the order they feed each other."""
    run = _assemble(document, variant=variant, base_dir=base_dir)
    # P-1.5, immediately before the return: after `build_inference`, so one
    # payload carries the raw twin, the fit twin and the space.  Being late
    # costs nothing (`build_inference` is 2.4 % of a warm load) and buys
    # `Built` mirroring `ConfiguredRun` field for field.  **This slot saves no
    # money** -- the beam is long since read -- and the sentence "before any
    # beam is analysed" is false about every check registered here.
    built_report = built(Built(*run))
    built_report.raise_if_refused()
    built_report.emit_warnings()
    run = _carrying(run, built_report)
    # P-2.5, after `build_inference` and before the return: the checks that
    # have to RUN the thing -- a forward pass per linear claim, a `jacfwd`
    # plus an SVD, two Newton solves.  It saves nothing and buys that they
    # run at all, in this layer's voice, rather than detonating inside a fit.
    #
    # **In `load_document` and NOT in `_assemble`, and that is load-bearing.**
    # `tests/config/inflight_helpers.built_run` calls `_assemble` precisely so
    # a raising hook cannot hide a slot's refusing half; a `raise_if_refused`
    # here-but-one-frame-lower re-arms that for every built-slot test and for
    # every post-flight one.  The receiver is `priced_report` and not `priced`
    # or `built`, both of which are entry points bound at module scope: a
    # local of either name shadows the function into an `UnboundLocalError`.
    priced_report = priced(_priced_payload(run))
    priced_report.raise_if_refused()
    priced_report.emit_warnings()
    return _carrying(run, priced_report)


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
