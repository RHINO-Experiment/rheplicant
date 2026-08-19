"""inference: the section orchestrator (schema §4.7, sequence of §2 of the
2B plan).

The order is load-bearing: the twin is repaired first (paths and bindings
resolve against the FIT twin); the space next; a non-frozen noise model
before ``observed:`` (``realise: {kind: from_model}`` draws with it); the
frozen sigma after (``source: observed`` reads the data just built); truth,
checks and trainable last.  ``checks:`` is grammar + record here -- its
gating (C12/C13) is Plan 3's validate, recorded in the handover's ledger.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NamedTuple

import jax.numpy as jnp

from _rheplicant_bootstrap.types import DestinationDescriptor
from rheplicant.config.context import ResolutionContext
from rheplicant.config.delivery import record_resolved_delivery
from rheplicant.config.errors import ConfigError
from rheplicant.config.gating import CHECK_NAMES, MODES, check_gates
from rheplicant.config.paths import resolve_path_on
from rheplicant.config.resources import check_unknown_keys
from rheplicant.config.sections.noise import (
    NoiseBuild,
    build_noise,
    freeze_sigma,
    freeze_sigmas,
)
from rheplicant.config.sections.npe import NpeSpec, parse_npe
from rheplicant.config.sections.observed import ObservedBuild, build_observed
from rheplicant.config.sections.parameters import parse_latents
from rheplicant.config.sections.transforms import _c17_validate_space, build_space
from rheplicant.config.sections.twin import build_fit_twin
from rheplicant.config.values import resolve_value

__all__ = ["CheckSpec", "InferenceBuild", "build_inference"]

_INFERENCE_KEYS = frozenset({"twin", "observed", "parameters", "bindings",
                             "joint_prior", "trainable", "noise", "truth",
                             "checks", "npe"})
#: The check names and the four mode words, ONE binding each, in
#: :mod:`rheplicant.config.gating` -- where ``preflight/gated.py`` can reach
#: them without importing this module and the builders behind it.  Kept under
#: their old private names because they are read from outside (``findings.py``
#: pins ``set(SEVERITIES) < set(_MODES)``, which is the argument for why there
#: are three severities and four modes).
_CHECK_NAMES = CHECK_NAMES
_MODES = MODES
_TRAINABLE_KEYS = frozenset({"all", "nodes", "leaves"})


class CheckSpec(NamedTuple):
    """One entry of ``inference.checks``, parsed and recorded."""

    mode: str
    report: bool
    reason: str | None
    rtol: float | None


class InferenceBuild(NamedTuple):
    """Everything ``inference:`` produced, ready for the exits."""

    fit_twin: Any
    space: Any
    noise: NoiseBuild
    observed: ObservedBuild | None
    truth: dict[str, Any]
    truth_omitted: dict[str, str]
    checks: dict[str, CheckSpec]
    trainable: Any
    replaced: tuple[str, ...]
    npe: NpeSpec | None = None
    #: ``{latent name: its ref: as a jnp array in context.dtype}``, for
    #: every latent that declares one.  ``ParsedLatent.ref`` reaches no
    #: other build field -- ``build_space`` keeps ``entry.latent`` and drops
    #: the rest (``transforms.py:402``) -- and ``kind: nuts``'s ``init: ref``
    #: is its first consumer since Plan 2B parsed it.  Populated HERE, where
    #: the latents are already parsed, rather than by a second
    #: ``parse_latents`` inside the executor: two validators for one grammar
    #: is a shape this effort has paid for once already.
    refs: dict[str, Any] | None = None


def _checks(section: Any) -> dict[str, CheckSpec]:
    """Grammar, then the record.  The grammar itself lives in
    :func:`~rheplicant.config.gating.check_gates`.

    The refusal a caller sees is unchanged, character for character: this
    raises the FIRST finding's message, and ``check_gates`` decides in the
    order this function used to and yields at most one finding per entry.
    There is no second copy of the grammar here -- two validators for one
    property is the divergence this layer has already paid for once.  What
    stays is the RECORD: ``CheckSpec`` is what the document DECLARED, and the
    effective gate (defaults applied) is ``gating.gates``' answer, asked by the
    pass rather than by the builder.
    """
    found = check_gates(section)
    if found:
        raise ConfigError(found[0].message)
    if section is None:
        return {}
    parsed: dict[str, CheckSpec] = {}
    for name, spec in section.items():
        rtol = spec.get("rtol")
        parsed[name] = CheckSpec(mode=spec.get("mode"),
                                 report=bool(spec.get("report", False)),
                                 reason=spec.get("reason"),
                                 rtol=float(rtol) if rtol is not None
                                 else None)
    return parsed


def _trainable(section: Any, fit_twin: Any) -> Any:
    if section is None:
        return None
    import equinox as eqx
    import jax

    if not isinstance(section, Mapping):
        raise ConfigError(f"inference.trainable: is a mapping; got "
                          f"{section!r}.")
    check_unknown_keys("inference.trainable", dict(section), _TRAINABLE_KEYS,
                       label="trainable:")
    everything = section.get("all", False)
    if not isinstance(everything, bool):
        raise ConfigError(f"inference.trainable.all: is a bool; got "
                          f"{everything!r}.")
    nodes = tuple(section.get("nodes") or ())
    leaves = tuple(section.get("leaves") or ())
    if everything:
        if nodes or leaves:
            raise ConfigError(
                "inference.trainable: all: true already takes every inexact "
                "leaf in the twin (build_forward_fn's own default); naming "
                "nodes: or leaves: beside it says two different things."
            )
        return eqx.is_inexact_array
    if not nodes and not leaves:
        raise ConfigError(
            "inference.trainable: declares nothing -- all: true, nodes: or "
            "leaves:."
        )
    spec = jax.tree.map(lambda _: False, fit_twin)
    for node in nodes:
        sub = fit_twin[node]
        spec = eqx.tree_at(lambda p, _n=node: p[_n], spec,
                           replace=jax.tree.map(eqx.is_inexact_array, sub))
    for path in leaves:
        spec = eqx.tree_at(resolve_path_on(path, fit_twin).selector, spec,
                           replace=True)
    return spec


def _derive_truth(parsed: Any, observed: ObservedBuild | None, twin: Any,
                  fit_twin: Any) -> tuple[dict, dict]:
    truth: dict[str, Any] = {}
    omitted: dict[str, str] = {}
    if parsed is None:
        return truth, omitted
    primary = observed.primary if observed is not None else None
    record = observed.records.get(primary, {}) if primary else {}
    if record.get("from") != "simulation":
        return truth, omitted
    at = observed.at.get(primary, {})
    base = twin if record.get("twin") == "full" else fit_twin
    for name, entry in parsed.items():
        if name in at:
            truth[name] = at[name]
        elif entry.into and len(entry.into) == 1 and entry.transform in (
                None, "identity"):
            truth[name] = resolve_path_on(entry.into[0], base).leaf
        elif entry.into and entry.transform in (None, "identity"):
            omitted[name] = ("one latent feeds several leaves; no single "
                             "leaf holds its truth; declare observed.at or "
                             "truth:")
        elif entry.into:
            omitted[name] = ("reached through transform "
                             f"{entry.transform!r}, which is not invertible "
                             "from the leaf; declare observed.at or truth:")
        else:
            omitted[name] = ("bound through inference.bindings; declare "
                             "observed.at or truth:")
    return truth, omitted


def build_inference(section: Any, *, twin: Any, state: Any, observation: Any,
                    context: ResolutionContext) -> InferenceBuild:
    """``inference:`` -> an :class:`InferenceBuild`; absent -> the defaults."""
    if section is None:
        section = {}
    if not isinstance(section, Mapping):
        raise ConfigError(f"inference: is a mapping of subsections; got "
                          f"{section!r}.")
    check_unknown_keys("inference", dict(section), _INFERENCE_KEYS,
                       label="inference:")
    npe = parse_npe(section["npe"], context) if "npe" in section else None
    fit_twin, replaced = build_fit_twin(section.get("twin"), twin, context)
    parsed = (parse_latents(section["parameters"], context)
              if "parameters" in section else None)
    bindings = section.get("bindings")
    if bindings is not None and not isinstance(bindings, (list, tuple)):
        raise ConfigError(
            f"inference.bindings: is a LIST of binding mappings; got "
            f"{type(bindings).__name__} ({bindings!r}). A single binding "
            "still takes its dash."
        )
    space = build_space(parsed, bindings,
                        section.get("joint_prior"), fit_twin=fit_twin,
                        replaced=replaced, context=context)
    # The first moment both the space and the fit twin exist, and BEFORE the
    # two builders below that run a real forward pass (`build_noise`'s
    # `source: prediction_at_init` and `build_observed`'s simulation branch).
    # A space that does not fit its twin is refused here rather than after
    # the layer has paid for a prediction it is about to throw away.
    _c17_validate_space(space, fit_twin, parsed, bindings)
    noise = build_noise(section.get("noise"), observation=observation,
                        context=context)
    observed = build_observed(section.get("observed"), twin=twin,
                              fit_twin=fit_twin, space=space, noise=noise,
                              state=state, observation=observation,
                              context=context)
    if noise.frozen is not None:
        if noise.frozen["source"] == "observed":
            if observed is None or observed.primary is None:
                raise ConfigError(
                    "inference.noise: source: observed decides the sigma "
                    "from the primary observed data, and this document "
                    "declares none (or several with no primary)."
                )
            noise = freeze_sigmas(noise, observed.entries,
                                  primary=observed.primary)
        else:
            # source: prediction_at_init reads the TWIN, not the data, so it
            # is ONE sigma however many observations the document declares
            # and there is nothing per-observation to fan.  Fanning it would
            # move the sigma onto the data behind the document's back.
            bound = (space.bind(fit_twin, dict(space.initial_values()))
                     if space is not None else fit_twin)
            noise = freeze_sigma(noise, bound(state).data)
    truth, omitted = _derive_truth(parsed, observed, twin, fit_twin)
    for name, node in (section.get("truth") or {}).items():
        if parsed is None or name not in parsed:
            raise ConfigError(
                f"inference.truth: {name!r} is not a declared latent; "
                f"inference.parameters declares "
                f"{sorted(parsed) if parsed else []}."
            )
        destination = DestinationDescriptor(
            f"inference.truth.{name}", "config_path", "inference.truth.*"
        )
        resolved = resolve_value(node, context, destination=destination)
        truth[name] = jnp.asarray(resolved.value, dtype=context.dtype)
        record_resolved_delivery(context, destination, resolved.unit)
        omitted.pop(name, None)
    return InferenceBuild(fit_twin=fit_twin, space=space, noise=noise,
                          observed=observed, truth=truth,
                          truth_omitted=omitted, checks=_checks(
                              section.get("checks")),
                          trainable=_trainable(section.get("trainable"),
                                               fit_twin),
                          replaced=replaced, npe=npe,
                          refs={name: entry.ref
                                for name, entry in (parsed or {}).items()
                                if entry.ref is not None})
