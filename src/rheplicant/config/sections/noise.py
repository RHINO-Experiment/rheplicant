"""inference.noise: the likelihood's sigma (schema §4.7.6).

Two different things are called "noise model" in this package; this section
builds only the likelihood's (``rheplicant.inference.noise``), never the graph
node (``model.noise`` -- and D-C17's two-sigma cross-check between them is
Plan 3's, per ``RadiometerNoiseOperator``'s own docstring).

``radiometer_frozen`` exists nowhere in src/ on purpose: it is this layer's
construct.  The sigma is DECIDED into an array -- the one form the conjugate
seam accepts (``linear.py:1031``) -- from ``|observed|`` or the starting
prediction, once Task 6 has either in hand (:func:`freeze_sigma`).

``include_logdet`` is parsed and checked here (A49, both directions) and
RECORDED on the build; its first consumer is 2C's likelihood-carrying exits
(``NoiseModelLikelihood.include_logdet``, ``inference/noise.py:372``).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NamedTuple

import jax.numpy as jnp

from rheplicant.config.context import ResolutionContext
from rheplicant.config.errors import ConfigError
from rheplicant.config.resources import check_unknown_keys
from rheplicant.config.sections.observation import ObservationBuild, _dimensioned
from rheplicant.config.values import resolve_value

__all__ = ["NoiseBuild", "build_noise", "decided_noise", "freeze_sigma",
           "freeze_sigmas"]

_KIND_KEYS = {
    "none": frozenset({"kind"}),
    "homoscedastic": frozenset({"kind", "sigma", "axis", "flags"}),
    "radiometer": frozenset({"kind", "channel_width", "integration_time",
                             "floor", "include_logdet", "flags"}),
    "radiometer_frozen": frozenset({"kind", "source", "channel_width",
                                    "integration_time", "floor"}),
}


class NoiseBuild(NamedTuple):
    """What ``inference.noise`` declared: a model, or a sigma to be decided.

    ``by_observation`` is the frozen sigma of EVERY observation, keyed by
    its name, and is None for every kind that has nothing to fan -- which is
    all of them but ``radiometer_frozen`` with ``source: observed``, the one
    kind whose sigma is decided from the DATA.  ``sigma`` stays the
    primary's, and on a one-observation document it is the same object under
    the one key, so nothing that reads ``sigma`` moved.

    The name is ``by_observation`` and not ``sigmas`` on purpose: one letter
    from ``sigma`` is one typo from a test that reads the wrong field and
    passes.
    """

    kind: str
    model: Any = None
    sigma: Any = None
    include_logdet: bool | None = None
    frozen: dict[str, float] | None = None
    by_observation: dict[str, Any] | None = None


def _fact(where: str, node: Any, observation: ObservationBuild,
          attribute: str, declared_at: str, context: ResolutionContext,
          dimension: str, what: str) -> float:
    if isinstance(node, Mapping) and dict(node) == {"from": "observation"}:
        value = getattr(observation, attribute)
        if value is None:
            raise ConfigError(
                f"{where}: {{from: observation}} reads {declared_at}, and "
                "this document does not declare it."
            )
        return float(value)
    return float(_dimensioned(where, node, context, dimension=dimension,
                              what=what).value)


def _wrap_flags(where: str, node: Any, observation: ObservationBuild,
                model: Any) -> Any:
    from rheplicant.inference import FlaggedNoise

    if not (isinstance(node, Mapping) and dict(node) == {"from": "observation"}):
        raise ConfigError(
            f"{where}: flags is {{from: observation}} -- the one place a "
            f"flag mask lives is observation.aux.flags; got {node!r}."
        )
    flags = (observation.aux or {}).get("flags")
    if flags is None:
        raise ConfigError(
            f"{where}: {{from: observation}} reads observation.aux.flags, "
            "and this document does not declare it."
        )
    return FlaggedNoise(model, flags)


def build_noise(section: Any, *, observation: ObservationBuild,
                context: ResolutionContext) -> NoiseBuild:
    """``inference.noise`` -> a :class:`NoiseBuild` (sigma still undecided
    for ``radiometer_frozen`` -- see :func:`freeze_sigma`)."""
    from rheplicant.inference import HomoscedasticNoise, RadiometerNoise

    if section is None:
        return NoiseBuild(kind="none")
    if not isinstance(section, Mapping):
        raise ConfigError(f"inference.noise: is a mapping with kind:; got "
                          f"{section!r}.")
    kind = section.get("kind")
    if kind not in _KIND_KEYS:
        raise ConfigError(
            f"inference.noise.kind: {kind!r} is not one of "
            f"{sorted(_KIND_KEYS)}."
        )
    check_unknown_keys("inference.noise", dict(section), _KIND_KEYS[kind],
                       label=f"kind: {kind}",
                       hints={"include_logdet":
                              "include_logdet is required exactly when the "
                              "sigma depends on the prediction (kind: "
                              "radiometer) and refused otherwise -- for a "
                              "constant sigma it changes nothing (A49)"})
    if kind == "none":
        return NoiseBuild(kind="none")
    if kind == "homoscedastic":
        if "sigma" not in section:
            raise ConfigError("inference.noise: kind: homoscedastic requires "
                              "sigma: -- a value node.")
        sigma = jnp.asarray(resolve_value(section["sigma"], context).value,
                            dtype=context.dtype)
        axis = section.get("axis", "none")
        if axis not in ("none", "time", "freq"):
            raise ConfigError(f"inference.noise.axis: is none, time or freq; "
                              f"got {axis!r}.")
        if sigma.ndim == 1:
            if axis == "none":
                raise ConfigError(
                    "inference.noise.sigma: is 1-D, and a 1-D sigma reads "
                    "equally well along either axis of (n_time, n_freq) "
                    "data; declare axis: time or axis: freq (check A26)."
                )
            sigma = sigma[:, None] if axis == "time" else sigma[None, :]
        elif axis != "none":
            raise ConfigError(
                "inference.noise.axis: says how to read a 1-D sigma; this "
                f"one has shape {tuple(sigma.shape)}."
            )
        model: Any = HomoscedasticNoise(sigma)
        if "flags" in section:
            model = _wrap_flags("inference.noise", section["flags"],
                                observation, model)
        return NoiseBuild(kind=kind, model=model)
    width = _fact("inference.noise.channel_width",
                  section.get("channel_width", {"from": "observation"}),
                  observation, "channel_width_hz",
                  "observation.time.channel_width", context,
                  dimension="frequency", what="a bandwidth")
    tau = _fact("inference.noise.integration_time",
                section.get("integration_time", {"from": "observation"}),
                observation, "integration_time_s",
                "observation.time.integration_time", context,
                dimension="time", what="a duration")
    floor = 0.0
    if "floor" in section:
        floor = float(_dimensioned("inference.noise.floor", section["floor"],
                                   context, dimension="temperature",
                                   what="a noise floor").value)
    if kind == "radiometer":
        include = section.get("include_logdet")
        if not isinstance(include, bool):
            raise ConfigError(
                "inference.noise.include_logdet: is required for a "
                "prediction-dependent noise model and has no default. False "
                "is the documented GLS variant -- a DIFFERENT estimator, "
                "biased high by (1 + f^2) (inference/noise.py:56-68) -- and "
                "a lost declaration comes back True with no error."
            )
        model = RadiometerNoise(width, tau, floor)
        if "flags" in section:
            model = _wrap_flags("inference.noise", section["flags"],
                                observation, model)
        return NoiseBuild(kind=kind, model=model, include_logdet=include)
    source = section.get("source")
    if source not in ("observed", "prediction_at_init"):
        raise ConfigError(
            "inference.noise.source: radiometer_frozen decides its sigma "
            "from 'observed' or 'prediction_at_init'; got "
            f"{source!r}."
        )
    return NoiseBuild(kind=kind,
                      frozen={"source": source, "channel_width_hz": width,
                              "integration_time_s": tau, "floor_k": floor})


def freeze_sigma(build: NoiseBuild, reference: Any) -> NoiseBuild:
    """Decide the frozen sigma from its reference array.

    ``reference`` is the observed data (``source: observed``) or one forward
    evaluation at the declared inits (``source: prediction_at_init``); Task
    6's orchestrator supplies whichever the document named.
    """
    facts = build.frozen or {}
    base = jnp.abs(jnp.asarray(reference))
    if facts.get("floor_k", 0.0) > 0.0:
        base = jnp.maximum(base, facts["floor_k"])
    fractional = 1.0 / (facts["channel_width_hz"]
                        * facts["integration_time_s"]) ** 0.5
    return build._replace(sigma=base * fractional)


def freeze_sigmas(build: NoiseBuild, references: Mapping[str, Any], *,
                  primary: str) -> NoiseBuild:
    """One frozen sigma per observation, and the primary's as the default.

    ``source: observed`` decides the sigma FROM the data, so a document with
    several observations has several sigmas -- and a run's ``on:`` says
    which of them weighs its residuals.  Freezing once off the primary and
    handing that array to every run is what this replaces: measured on a
    two-observation document whose second entry is twice the first, a run on
    the second was weighed with HALF the sigma it should have been, and the
    only thing that changed was a number nobody could see.

    ``sigma`` stays the primary's, so :func:`decided_noise` -- which takes
    no run -- is unchanged, and on one observation this returns that same
    array under that same name.  The arithmetic is :func:`freeze_sigma`'s,
    called once per reference rather than reimplemented: a floor that
    clipped in one function and not the other is the shape this avoids.
    """
    sigmas = {name: freeze_sigma(build, reference).sigma
              for name, reference in references.items()}
    return build._replace(sigma=sigmas[primary], by_observation=sigmas)


def decided_noise(build: NoiseBuild) -> Any:
    """What an exit passes as ``noise=``/``noise_std=``: model, sigma or None."""
    if build.kind == "none":
        return None
    if build.model is not None:
        return build.model
    if build.sigma is None:
        raise ConfigError(
            "inference.noise: this radiometer_frozen sigma was never frozen "
            "-- build_inference decides it once observed data exists. This "
            "is a config-layer sequencing bug, not a document error."
        )
    return build.sigma
