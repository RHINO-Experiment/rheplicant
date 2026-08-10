"""inference.observed: what the fit compares against (schema §4.7.1).

Three forms: a simulation (``from: simulation`` -- the FULL twin by default,
truth injected through ``at:``, scatter added by ``realise:``), a file, and a
mapping of named observations.  ``realise:`` draws with the package's own
``NoiseModel.realise`` so the generator and the likelihood cannot disagree --
that seam is exactly why Plan 0 shipped ``realise`` on the Protocol.

Every drawing ``realise:`` names its seed as ``{from: runtime.seeds.<name>}``;
``seed_for`` (``config/draws.py:46``) resolves a declared name or derives one
by blake2s fold-in, and refuses ``runtime.seed: null``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from rheplicant.config.context import ResolutionContext
from rheplicant.config.draws import _seed_name, seed_for
from rheplicant.config.errors import ConfigError
from rheplicant.config.resources import check_unknown_keys
from rheplicant.config.values import resolve_value

__all__ = ["ObservedBuild", "build_observed"]

_SIM_KEYS = frozenset({"from", "twin", "at", "realise"})
_FORM_KEYS = _SIM_KEYS | {"file"}
_REALISE_KINDS = ("none", "homoscedastic", "radiometer", "from_model")


class ObservedBuild(NamedTuple):
    """Named observed arrays, the primary among them, and their provenance."""

    entries: dict[str, Any]
    primary: str | None
    at: dict[str, dict[str, Any]]
    records: dict[str, dict[str, Any]]


def _shape(context: ResolutionContext) -> tuple[int, int]:
    return (int(context.time.shape[0]), int(context.freq.shape[0]))


def _realise(name: str, spec: Any, prediction: Any, *, noise: Any,
             observation: Any, context: ResolutionContext):
    where = f"inference.observed.{name}.realise"
    if spec is None:
        return prediction, None, None
    if not isinstance(spec, Mapping):
        raise ConfigError(f"{where}: is a mapping with kind:; got {spec!r}.")
    kind = spec.get("kind")
    if kind not in _REALISE_KINDS:
        raise ConfigError(f"{where}.kind: {kind!r} is not one of "
                          f"{list(_REALISE_KINDS)}.")
    if kind == "none":
        check_unknown_keys(where, dict(spec), frozenset({"kind"}),
                           label="kind: none")
        return prediction, None, kind
    seed = seed_for(_seed_name(dict(spec), where), context)
    key = jax.random.key(seed)
    if kind == "homoscedastic":
        from rheplicant.inference import HomoscedasticNoise

        check_unknown_keys(where, dict(spec),
                           frozenset({"kind", "sigma", "seed"}),
                           label="kind: homoscedastic")
        if "sigma" not in spec:
            raise ConfigError(f"{where}: kind: homoscedastic requires "
                              "sigma: -- the scatter that goes INTO the data.")
        sigma = jnp.asarray(resolve_value(spec["sigma"], context).value,
                            dtype=context.dtype)
        return (HomoscedasticNoise(sigma).realise(prediction, key=key),
                seed, kind)
    if kind == "radiometer":
        from rheplicant.inference import RadiometerNoise

        check_unknown_keys(where, dict(spec), frozenset({"kind", "seed"}),
                           label="kind: radiometer")
        width = observation.channel_width_hz
        tau = observation.integration_time_s
        if width is None or tau is None:
            raise ConfigError(
                f"{where}: kind: radiometer draws d -> d(1 + fw) with f = "
                "1/sqrt(channel_width * integration_time), read from "
                "observation.time -- and this document does not declare both."
            )
        return (RadiometerNoise(float(width), float(tau)).realise(prediction,
                                                                  key=key),
                seed, kind)
    # from_model
    check_unknown_keys(where, dict(spec), frozenset({"kind", "seed"}),
                       label="kind: from_model")
    if noise is None or noise.model is None:
        declared = "none" if noise is None else noise.kind
        raise ConfigError(
            f"{where}: kind: from_model draws with inference.noise's own "
            f"model, and this document declares kind: {declared} -- none has "
            "nothing to draw with, and a frozen sigma is decided FROM the "
            "data, so the data cannot be drawn from it."
        )
    return noise.model.realise(prediction, key=key), seed, kind


def _one(name: str, spec: Mapping, *, twin: Any, fit_twin: Any, space: Any,
         noise: Any, state: Any, observation: Any,
         context: ResolutionContext):
    where = f"inference.observed.{name}"
    if "file" in spec:
        check_unknown_keys(where, dict(spec), frozenset({"file"}),
                           label="the file form")
        data = jnp.asarray(resolve_value({"file": dict(spec["file"])},
                                         context).value)
        wanted = _shape(context)
        if tuple(data.shape) != wanted:
            raise ConfigError(
                f"{where}: the file holds shape {tuple(data.shape)}; this "
                f"run's grids say {wanted}. Exactly -- broadcast-compatible "
                "is the dangerous case (check C11)."
            )
        return data, {}, {"from": "file", "twin": None, "realise": None,
                          "seed": None}
    if spec.get("from") != "simulation":
        raise ConfigError(
            f"{where}: is {{from: simulation, ...}} or {{file: {{...}}}}; "
            f"got {dict(spec)!r}."
        )
    check_unknown_keys(where, dict(spec), _SIM_KEYS, label="a simulation")
    choice = spec.get("twin", "full")
    if choice not in ("full", "fit"):
        raise ConfigError(f"{where}.twin: is full or fit (default full); "
                          f"got {choice!r}.")
    base = twin if choice == "full" else fit_twin
    at_spec = spec.get("at") or {}
    if not isinstance(at_spec, Mapping):
        raise ConfigError(f"{where}.at: is a mapping of latent -> truth "
                          f"value; got {at_spec!r}.")
    if at_spec and space is None:
        raise ConfigError(
            f"{where}.at: names latents, and this document declares no "
            "inference.parameters."
        )
    at_values: dict[str, Any] = {}
    for latent, node in at_spec.items():
        if latent not in space.names:
            raise ConfigError(
                f"{where}.at: {latent!r} is not a declared latent; "
                f"inference.parameters declares {list(space.names)}."
            )
        at_values[latent] = jnp.asarray(resolve_value(node, context).value,
                                        dtype=context.dtype)
    if space is not None:
        values = dict(space.initial_values())
        values.update(at_values)
        bound = space.bind(base, values)
    else:
        bound = base
    prediction = bound(state).data
    data, seed, kind = _realise(name, spec.get("realise"), prediction,
                                noise=noise, observation=observation,
                                context=context)
    return data, at_values, {"from": "simulation", "twin": choice,
                             "realise": kind, "seed": seed}


def build_observed(section: Any, *, twin: Any, fit_twin: Any, space: Any,
                   noise: Any, state: Any, observation: Any,
                   context: ResolutionContext) -> ObservedBuild | None:
    """``inference.observed`` -> named arrays, or None when undeclared."""
    if section is None:
        return None
    if not isinstance(section, Mapping):
        raise ConfigError(f"inference.observed: is a mapping; got "
                          f"{section!r}.")
    if "from" in section or "file" in section:
        named: dict[str, Mapping] = {"primary": section}
    else:
        named = {}
        for name, spec in section.items():
            if not isinstance(name, str) or name in _FORM_KEYS:
                raise ConfigError(
                    f"inference.observed: {name!r} is not usable as an "
                    f"observation name -- the grammar owns "
                    f"{sorted(_FORM_KEYS)}."
                )
            if not isinstance(spec, Mapping) or not (
                    "from" in spec or "file" in spec):
                raise ConfigError(
                    f"inference.observed.{name}: is {{from: simulation, ...}} "
                    f"or {{file: {{...}}}}; got {spec!r}."
                )
            named[name] = spec
        if not named:
            raise ConfigError("inference.observed: declares no observation.")
    entries: dict[str, Any] = {}
    at: dict[str, dict[str, Any]] = {}
    records: dict[str, dict[str, Any]] = {}
    for name, spec in named.items():
        entries[name], at[name], records[name] = _one(
            name, spec, twin=twin, fit_twin=fit_twin, space=space,
            noise=noise, state=state, observation=observation,
            context=context)
    if "primary" in entries:
        primary: str | None = "primary"
    elif len(entries) == 1:
        primary = next(iter(entries))
    else:
        primary = None
    return ObservedBuild(entries=entries, primary=primary, at=at,
                         records=records)
