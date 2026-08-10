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

from rheplicant.config.context import ResolutionContext
from rheplicant.config.errors import ConfigError
from rheplicant.config.paths import resolve_path_on
from rheplicant.config.resources import check_unknown_keys
from rheplicant.config.sections.noise import NoiseBuild, build_noise, freeze_sigma
from rheplicant.config.sections.observed import ObservedBuild, build_observed
from rheplicant.config.sections.parameters import parse_latents
from rheplicant.config.sections.transforms import build_space
from rheplicant.config.sections.twin import build_fit_twin
from rheplicant.config.values import resolve_value

__all__ = ["CheckSpec", "InferenceBuild", "build_inference"]

_INFERENCE_KEYS = frozenset({"twin", "observed", "parameters", "bindings",
                             "joint_prior", "trainable", "noise", "truth",
                             "checks", "npe"})
_CHECK_NAMES = frozenset({"identifiability", "linearity",
                          "prior_sensitivity"})
_MODES = ("refuse", "warn", "report", "skip")
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


def _checks(section: Any) -> dict[str, CheckSpec]:
    if section is None:
        return {}
    if not isinstance(section, Mapping):
        raise ConfigError(f"inference.checks: is a mapping; got {section!r}.")
    parsed: dict[str, CheckSpec] = {}
    for name, spec in section.items():
        where = f"inference.checks.{name}"
        if name not in _CHECK_NAMES:
            raise ConfigError(
                f"{where}: {name!r} is not a check; v1 knows "
                f"{sorted(_CHECK_NAMES)}."
            )
        if not isinstance(spec, Mapping):
            raise ConfigError(f"{where}: is a mapping with mode:; got "
                              f"{spec!r}.")
        allowed = frozenset({"mode", "report", "reason"}) | (
            frozenset({"rtol"}) if name == "identifiability" else frozenset())
        check_unknown_keys(where, dict(spec), allowed, label="a check:")
        mode = spec.get("mode")
        if mode not in _MODES:
            raise ConfigError(f"{where}.mode: is one of {list(_MODES)}; "
                              f"got {mode!r}.")
        reason = spec.get("reason")
        if mode == "skip" and not isinstance(reason, str):
            raise ConfigError(
                f"{where}: mode: skip carries its own reason: (check A37) -- "
                "three unrelated skips sharing one sentence was v0's mistake."
            )
        if mode != "skip" and reason is not None:
            raise ConfigError(f"{where}: reason: belongs to mode: skip "
                              "alone.")
        rtol = spec.get("rtol")
        parsed[name] = CheckSpec(mode=mode, report=bool(spec.get("report",
                                                                 False)),
                                 reason=reason,
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
    if "npe" in section:
        raise ConfigError(
            "inference.npe: arrives with Plan 2C, alongside the npe exit."
        )
    check_unknown_keys("inference", dict(section), _INFERENCE_KEYS,
                       label="inference:")
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
            reference = observed.entries[observed.primary]
        else:
            bound = (space.bind(fit_twin, dict(space.initial_values()))
                     if space is not None else fit_twin)
            reference = bound(state).data
        noise = freeze_sigma(noise, reference)
    truth, omitted = _derive_truth(parsed, observed, twin, fit_twin)
    for name, node in (section.get("truth") or {}).items():
        if parsed is None or name not in parsed:
            raise ConfigError(
                f"inference.truth: {name!r} is not a declared latent; "
                f"inference.parameters declares "
                f"{sorted(parsed) if parsed else []}."
            )
        truth[name] = jnp.asarray(resolve_value(node, context).value,
                                  dtype=context.dtype)
        omitted.pop(name, None)
    return InferenceBuild(fit_twin=fit_twin, space=space, noise=noise,
                          observed=observed, truth=truth,
                          truth_omitted=omitted, checks=_checks(
                              section.get("checks")),
                          trainable=_trainable(section.get("trainable"),
                                               fit_twin),
                          replaced=replaced)
