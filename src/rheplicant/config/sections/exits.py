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


def _number(run: RunSpec, key: str, value: Any, *, kind: type,
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


_OPTIMIZE_KEYS = frozenset({"optimizer", "learning_rate", "n_steps", "beta1",
                            "beta2", "eps", "loss"})
_ADAM_ONLY = ("beta1", "beta2", "eps")


def _run_fisher(run: RunSpec, built: Any) -> Any:
    from rheplicant.inference import fisher_information, parameter_covariance

    _sweep(run, frozenset({"space", "jitter"}))
    inference = built.inference
    space = _space(run, built)
    noise = _noise(run, built)
    forward, values = space.forward_fn(inference.fit_twin, built.state)
    use_space = run.options.get("space", False)
    if not isinstance(use_space, bool):
        raise ConfigError(f"runs[{run.name!r}]: space: is a bool; got "
                          f"{use_space!r}.")
    fisher = fisher_information(forward, values, noise,
                                space=space if use_space else None)
    jitter = _number(run, "jitter", run.options.get("jitter", 0.0),
                     kind=float, minimum=0.0)
    return {"fisher": fisher,
            "covariance": parameter_covariance(fisher, jitter=jitter)}


_EXECUTORS["fisher"] = _run_fisher


def _loss_fn(run: RunSpec) -> Any:
    from rheplicant.inference import mean_squared_error

    loss = run.options.get("loss", "mse")
    if loss == "mse":
        return mean_squared_error
    if isinstance(loss, dict) and set(loss) == {"python"}:
        from rheplicant.config.hatch import import_target

        return import_target(loss["python"])
    raise ConfigError(
        f"runs[{run.name!r}]: loss: is 'mse' or {{python: 'mod:fn'}}; got "
        f"{loss!r}."
    )


def _run_optimize(run: RunSpec, built: Any) -> Any:
    from rheplicant.inference import AdamCalibrator, GradientCalibrator, build_forward_fn

    _sweep(run, _OPTIMIZE_KEYS)
    inference = built.inference
    observed = _observed(run, built)
    optimizer = run.options.get("optimizer")
    if optimizer not in ("gradient", "adam"):
        raise ConfigError(
            f"runs[{run.name!r}]: optimizer: gradient or adam is required; "
            f"got {optimizer!r}. The two are different algorithms behind an "
            "identical .fit."
        )
    for key in ("learning_rate", "n_steps"):
        if key not in run.options:
            raise ConfigError(
                f"runs[{run.name!r}]: {key}: is required -- the shipped "
                "default (1e-2) sits five orders of magnitude from what a "
                "real fit has needed (examples/radio_digital_twin.py:112)."
            )
    if optimizer == "gradient":
        for key in _ADAM_ONLY:
            if key in run.options:
                raise ConfigError(
                    f"runs[{run.name!r}]: {key}: belongs to optimizer: adam."
                )
        calibrator = GradientCalibrator(
            learning_rate=_number(run, "learning_rate",
                                  run.options["learning_rate"], kind=float),
            n_steps=_number(run, "n_steps", run.options["n_steps"], kind=int))
    else:
        knobs = {key: _number(run, key, run.options[key], kind=float)
                 for key in _ADAM_ONLY if key in run.options}
        calibrator = AdamCalibrator(
            learning_rate=_number(run, "learning_rate",
                                  run.options["learning_rate"], kind=float),
            n_steps=_number(run, "n_steps", run.options["n_steps"], kind=int),
            **knobs)
    trainable = inference.trainable
    space = inference.space
    if trainable is not None and space is not None:
        raise ConfigError(
            f"runs[{run.name!r}]: inference.trainable and "
            "inference.parameters are both declared, and optimize cannot "
            "serve two masters -- trainable is the calibrator-only route "
            "(schema §4.7.5); drop one."
        )
    if trainable is not None:
        forward, params0 = build_forward_fn(inference.fit_twin, built.state,
                                            trainable)
    elif space is not None:
        forward, params0 = space.forward_fn(inference.fit_twin, built.state)
    else:
        raise ConfigError(
            f"runs[{run.name!r}]: optimize needs inference.trainable or "
            "inference.parameters -- something must be free to move."
        )
    params_fit, losses = calibrator.fit(forward, params0, observed,
                                        loss_fn=_loss_fn(run))
    return {"params": params_fit, "losses": losses}


_EXECUTORS["optimize"] = _run_optimize


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
