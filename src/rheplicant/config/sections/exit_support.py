"""What every exit executor shares: the sweep, the accessors, the registry.

An executor is ``(run, built, *, results=None) -> product``.  It is registered
under its ``runs[].kind`` by the :func:`register` decorator, and
:data:`EXECUTORS` is the one table :func:`execute_run` dispatches through.
The leaf modules (``exits``, ``conjugate``, ``diagnostics``) import from here
and never from each other, so the registration is a one-way import.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from rheplicant.config.errors import ConfigError
from rheplicant.config.sections.noise import decided_noise

__all__ = ["EXECUTORS", "register", "reuse_of"]

EXECUTORS: dict[str, Callable[..., Any]] = {}


def register(kind: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Bind an executor to its ``runs[].kind``.

    Registering the same kind twice is a programming error, not a
    configuration one, so it asserts rather than raising ConfigError.
    """

    def bind(fn: Callable[..., Any]) -> Callable[..., Any]:
        assert kind not in EXECUTORS, f"{kind} is already registered"
        EXECUTORS[kind] = fn
        return fn

    return bind


def _sweep(run: Any, allowed: frozenset[str]) -> None:
    unknown = sorted(set(run.options) - allowed)
    if unknown:
        raise ConfigError(
            f"runs[{run.name!r}]: kind: {run.kind} does not take {unknown}; "
            f"it takes {sorted(allowed)}."
        )


def _number(run: Any, key: str, value: Any, *, kind: type,
            minimum: float | None = None) -> Any:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(
            f"runs[{run.name!r}]: {key}: is a number; got {value!r}."
        )
    if minimum is not None and not value >= minimum:
        raise ConfigError(
            f"runs[{run.name!r}]: {key}: must be >= {minimum:g}; got "
            f"{value!r}."
        )
    return kind(value)


def _space(run: Any, built: Any) -> Any:
    space = built.inference.space
    if space is None:
        raise ConfigError(
            f"runs[{run.name!r}]: kind: {run.kind} fits latents, and this "
            "document declares no inference.parameters."
        )
    return space


def _noise(run: Any, built: Any) -> Any:
    noise = decided_noise(built.inference.noise)
    if noise is None:
        raise ConfigError(
            f"runs[{run.name!r}]: kind: {run.kind} weighs residuals with "
            "inference.noise, and this document declares kind: none -- "
            "legal only for forward and optimize."
        )
    return noise


def _observed(run: Any, built: Any) -> Any:
    observed = built.inference.observed
    if observed is None:
        raise ConfigError(
            f"runs[{run.name!r}]: kind: {run.kind} compares against "
            "inference.observed, and this document declares none."
        )
    name = run.on
    if name == "primary" and observed.primary is not None:
        name = observed.primary
    if name not in observed.entries:
        raise ConfigError(
            f"runs[{run.name!r}]: on: {run.on!r} names no observation; this "
            f"document declares {sorted(observed.entries)}."
        )
    return observed.entries[name]


def _passthrough(options: Mapping, keys: tuple[str, ...]) -> dict:
    return {key: options[key] for key in keys if key in options}


def _decided_sigma(run: Any, built: Any) -> Any:
    """The DECIDED sigma array the conjugate seam takes as ``noise_std=``.

    ``wiener_solve``, ``gcr_sample`` and ``condition_estimate`` compute
    ``1/sigma**2`` directly and refuse a NoiseModel outright
    (``linear.py:1031``).  A constant-sigma model is decided here -- its
    ``std`` ignores the prediction by contract
    (``depends_on_prediction`` is False), so evaluating it on the run's own
    grid gives the full-shaped array, which is also the one shape
    ``check_noise_std_axis`` never has to guess an axis for.  A
    prediction-dependent one cannot be decided at all, and that is check A27.

    Takes no ``observed``: the shape comes from ``built.state.coords``, so a
    document with no ``inference.observed`` still decides a sigma (which is
    what ``condition`` needs).
    """
    import jax.numpy as jnp

    from rheplicant.inference import NoiseModel

    decided = _noise(run, built)
    if not isinstance(decided, NoiseModel):
        return decided
    if decided.depends_on_prediction:
        raise ConfigError(
            f"runs[{run.name!r}]: kind: {run.kind} takes a DECIDED sigma "
            "array, and inference.noise.kind: "
            f"{built.inference.noise.kind} makes sigma a function of the "
            "prediction -- which a conjugate solve has not got, because the "
            "prediction is what it solves for (linear.py:1031, check A27). "
            "Two routes run this noise: kind: conjugate.gls iterates the "
            "covariance it implies, or inference.noise.kind: "
            "radiometer_frozen decides the sigma once and keeps this exit."
        )
    # Only the SHAPE is load-bearing.  A constant-sigma model's std() ignores
    # its argument's VALUES by contract and returns the dtype of its own
    # sigma, not the probe's (measured: a float32 sigma against a float64
    # prediction comes back float32).  So no dtype= here: passing
    # built.context.dtype would read as enforcing the document's dtype on the
    # result, which it does not do -- the document's dtype already reached
    # this sigma when build_noise resolved it.
    coords = built.state.coords
    shape = (int(coords.time.size), int(coords.freq.size))
    return decided.std(jnp.zeros(shape))


def _decided_model(run: Any, built: Any) -> Any:
    """The noise MODEL an exit that iterates a covariance needs (check A28).

    The mirror of :func:`_decided_sigma`.  ``decided_noise`` returns either a
    NoiseModel or a frozen sigma array, and the two are not interchangeable
    at the conjugate seam: ``iterative_gls`` takes ``noise=`` (the RULE,
    ``gls.py:181-196``) where the three conjugate solves take ``noise_std=``
    (a decided array), and passing either one where the other belongs is a
    hard ParameterSpaceError in both directions.  The refusal names the exit
    that wants the other shape rather than merely rejecting this one.
    """
    from rheplicant.inference import NoiseModel

    noise = _noise(run, built)
    if isinstance(noise, NoiseModel):
        return noise
    raise ConfigError(
        f"runs[{run.name!r}]: kind: {run.kind} solves for the covariance "
        "a PREDICTION-DEPENDENT sigma implies, so it reads inference.noise "
        "as a model; inference.noise.kind: "
        f"{built.inference.noise.kind} decides its sigma into an array "
        "before any run sees it, and a decided array has no fixed point to "
        "iterate (check A28). Declare inference.noise.kind: radiometer to "
        "iterate the rule, or run kind: conjugate.wiener, which is what a "
        "decided sigma wants."
    )


def reuse_of(run: Any, results: Mapping[str, Any] | None) -> Any:
    """The RunResult an exit's ``reuse:`` names, or a refusal saying why not.

    Runs execute in declaration order, so a reuse may only look backwards --
    naming a later run reads exactly like naming a missing one, and the
    message says so.
    """
    where = f"runs[{run.name!r}]"
    if run.reuse is None:
        raise ConfigError(
            f"{where}: kind: {run.kind} reads an earlier run's product, so "
            "reuse: <run name> is required."
        )
    results = results or {}
    if run.reuse not in results:
        raise ConfigError(
            f"{where}: reuse: {run.reuse!r} names no earlier run; runs "
            f"execute in declaration order and by now {sorted(results)} have "
            "run."
        )
    earlier = results[run.reuse]
    if earlier.error is not None:
        raise ConfigError(
            f"{where}: reuse: {run.reuse!r} refused ({earlier.error}), so it "
            "has no product to read."
        )
    return earlier
