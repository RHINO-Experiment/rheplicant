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
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any, NamedTuple

from rheplicant.config.context import ResolutionContext
from rheplicant.config.errors import ConfigError
from rheplicant.config.layering import apply_variant
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


def load_document(document: Mapping, *, variant: str | None = None,
                  base_dir: str | None = None) -> ConfiguredRun:
    """Build the objects a document declares, in the order they feed each other."""
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
    return ConfiguredRun(document=doc, runtime=runtime, state=state,
                         twin=twin, inference=inference, resources=resources,
                         context=context)


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
