"""One document -> one configured run (schema §3, the build order of §2).

The sequence is load-bearing: variants apply first (they patch the raw
document); the runtime facts decide the dtype; the observation is built next
because its grids and switch order ARE the resolution context; resources
resolve against that context; the model resolves against the context plus the
resources; and only then can an ingested State be finished, because
``to_state``'s ``source_order`` is read off the assembled twin's own switch
order (``rhino.py:607-611``).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any, NamedTuple

from rheplicant.config.context import ResolutionContext
from rheplicant.config.errors import ConfigError
from rheplicant.config.layering import apply_variant
from rheplicant.config.resources import BuiltResources, build_resources
from rheplicant.config.sections.compose import build_model
from rheplicant.config.sections.inference import build_inference
from rheplicant.config.sections.observation import build_observation
from rheplicant.config.sections.runtime import RuntimeFacts, build_runtime, state_key
from rheplicant.core.coordinates import Coordinates
from rheplicant.core.state import State

__all__ = ["ConfiguredRun", "load_document", "run_forward"]

_SECTIONS = ("schema_version", "defaults", "plugins", "runtime", "observation",
             "resources", "model", "variants", "inference", "runs", "outputs",
             "campaign")
_NOT_YET = {
    "runs": "Plan 2B (the exits; run_forward is this layer's forward exit)",
    "outputs": "Plan 4 (outputs, provenance, the CLI)",
    "defaults": "Plan 4 (presets are YAML files, and the CLI is where YAML "
                "first comes off disk)",
    "plugins": "Plan 4 (plugin import belongs to the process entry point)",
}
_REQUIRED = ("runtime", "observation", "model")


class ConfiguredRun(NamedTuple):
    """A document, built: the state, the twin, and everything that made them."""

    document: dict[str, Any]
    runtime: RuntimeFacts
    state: State
    twin: Any
    inference: Any
    resources: BuiltResources
    context: ResolutionContext


def _sweep(document: Mapping) -> None:
    unknown = sorted(set(document) - set(_SECTIONS))
    if unknown:
        raise ConfigError(
            f"This document declares {unknown}; the sections are "
            f"{list(_SECTIONS)}."
        )
    if "campaign" in document:
        raise ConfigError(
            "campaign: is reserved with capability 4 (streaming evidence, "
            "schema §8.2) and refused in v1."
        )
    for section, route in _NOT_YET.items():
        if section in document:
            raise ConfigError(
                f"{section}: is not read by this layer yet -- it arrives with "
                f"{route}."
            )
    version = document.get("schema_version")
    if version != 1 or isinstance(version, bool):
        raise ConfigError(
            f"schema_version: 1 is required (got {version!r}); it is what "
            "lets a later loader read an older document on purpose rather "
            "than by luck."
        )
    missing = [section for section in _REQUIRED if section not in document]
    if missing:
        raise ConfigError(
            f"This document is missing {missing}; schema_version, runtime, "
            "observation and model are required."
        )


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
    _sweep(doc)
    runtime = build_runtime(doc["runtime"])
    observation, context = build_observation(doc["observation"],
                                             runtime=runtime,
                                             base_dir=base_dir)
    resources = build_resources(doc.get("resources") or {}, context)
    context = dataclasses.replace(context,
                                  resources=dict(resources.resources))
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
    speak for a data/source mismatch.
    """
    if not isinstance(run, ConfiguredRun):
        run = load_document(run, variant=variant, base_dir=base_dir)
    return run.twin(run.state)
