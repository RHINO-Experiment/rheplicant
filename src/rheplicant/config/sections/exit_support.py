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
