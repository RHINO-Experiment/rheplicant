"""The exit executors: one function per runs[].kind (schema §4.7.9).

Each executor sweeps its own kind-specific keys, reads what it needs off the
ConfiguredRun's InferenceBuild, and drives the package's documented entry
point.  Package refusals -- stochastic stages, priors, shapes -- speak for
themselves; the config layer adds only what the grammar can see.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from rheplicant.config.errors import ConfigError
from rheplicant.config.sections.noise import decided_noise
from rheplicant.config.sections.runs import RunResult, RunSpec

__all__ = ["execute_run"]

_EXECUTORS: dict[str, Callable[[RunSpec, Any], Any]] = {}


def _sweep(run: RunSpec, allowed: frozenset[str]) -> None:
    unknown = sorted(set(run.options) - allowed)
    if unknown:
        raise ConfigError(
            f"runs[{run.name!r}]: kind: {run.kind} does not take {unknown}; "
            f"it takes {sorted(allowed)}."
        )


def _space(run: RunSpec, built: Any) -> Any:
    space = built.inference.space
    if space is None:
        raise ConfigError(
            f"runs[{run.name!r}]: kind: {run.kind} fits latents, and this "
            "document declares no inference.parameters."
        )
    return space


def _noise(run: RunSpec, built: Any) -> Any:
    noise = decided_noise(built.inference.noise)
    if noise is None:
        raise ConfigError(
            f"runs[{run.name!r}]: kind: {run.kind} weighs residuals with "
            "inference.noise, and this document declares kind: none -- "
            "legal only for forward and optimize."
        )
    return noise


def _observed(run: RunSpec, built: Any) -> Any:
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


def _run_forward(run: RunSpec, built: Any) -> Any:
    _sweep(run, frozenset())
    return built.twin(built.state)


_EXECUTORS["forward"] = _run_forward


def execute_run(run: RunSpec, built: Any) -> RunResult:
    """One run entry against its ConfiguredRun -> a RunResult."""
    executor = _EXECUTORS[run.kind]
    if run.expect == "refuse":
        try:
            executor(run, built)
        except Exception as error:  # noqa: BLE001 -- run-and-capture is the point
            return RunResult(name=run.name, kind=run.kind, product=None,
                             error=error)
        raise ConfigError(
            f"runs[{run.name!r}]: expect: refuse, and kind: {run.kind} "
            "SUCCEEDED -- the assertion this run makes about the design no "
            "longer holds."
        )
    return RunResult(name=run.name, kind=run.kind,
                     product=executor(run, built), error=None)
