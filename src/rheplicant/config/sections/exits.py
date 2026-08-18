"""The exit executors: one handler per runs[].kind (schema §4.7.9).

Each base kind's PARSER sweeps the entry's kind-specific keys, validates
them, and normalizes one selected set of defaults into the two frozen views
of a ``ParsedOptions`` -- reading only static facts off the configured
build, never the twin.  The EXECUTOR then reads nothing but the parsed
execution view and the ConfiguredRun's InferenceBuild, and spends itself on
the package's documented entry point.  Package refusals -- stochastic
stages, priors, shapes -- speak for themselves; the config layer adds only
what the grammar can see.  Every executor keeps its one textual ``_sweep``
as its first statement: until Task 10 retires the mirror, preflight's P-1
table derives each kind's allowed keys from that call site, and on a parsed
run it can only ever fire on a mis-bound registry.

This module holds Plan 2B's five exits and the dispatcher.  The shared
machinery is in ``exit_support``; the conjugate family and the diagnostics are
in their own modules, imported at the foot of this one so that importing
``exits`` leaves every kind registered.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from _rheplicant_bootstrap.variants import LayerRef
from rheplicant.config.errors import ConfigError
from rheplicant.config.sections.exit_support import (
    _PROBE,
    EXECUTORS,
    ParsedRun,
    _binds,
    _noise,
    _number,
    _observed,
    _passthrough,
    _space,
    _sweep,
    handler_for,
    parse_run,
    parsed_options,
    register,
)
from rheplicant.config.sections.runs import RunResult, RunSpec

__all__ = ["execute_run"]


def _parse_forward(options, context):
    """``forward`` takes no keys; the sweep is the whole grammar."""
    _sweep(context.spec, frozenset())
    return parsed_options(options, resolved=options)


@register("forward", parse=_parse_forward)
def _run_forward(run: ParsedRun, built: Any, previous: Any = None) -> Any:
    _sweep(run, frozenset())
    return built.twin(built.state)


_OPTIMIZE_KEYS = frozenset({"optimizer", "learning_rate", "n_steps", "beta1",
                            "beta2", "eps", "loss"})
_ADAM_ONLY = ("beta1", "beta2", "eps")
#: Measured against ``AdamCalibrator``'s own static defaults, so passing them
#: explicitly is byte-identical to today's omission.
_ADAM_DEFAULTS = {"beta1": 0.9, "beta2": 0.999, "eps": 1e-8}


def _parse_fisher(options, context):
    """``space:``/``jitter:`` validated and defaulted; no Fisher math here."""
    spec = context.spec
    built = context.configured_run
    _sweep(spec, frozenset({"space", "jitter"}))
    _space(spec, built)
    _noise(spec, built)
    use_space = options.get("space", False)
    if not isinstance(use_space, bool):
        raise ConfigError(f"runs[{spec.name!r}]: space: is a bool; got "
                          f"{use_space!r}.")
    jitter = _number(spec, "jitter", options.get("jitter", 0.0),
                     kind=float, minimum=0.0)
    normalized = {"space": use_space, "jitter": jitter}
    return parsed_options(normalized, resolved=normalized)


@register("fisher", parse=_parse_fisher)
def _run_fisher(run: ParsedRun, built: Any, previous: Any = None) -> Any:
    from rheplicant.inference import fisher_information, parameter_covariance

    _sweep(run, frozenset({"space", "jitter"}))
    inference = built.inference
    space = _space(run, built)
    noise = _noise(run, built)
    forward, values = space.forward_fn(inference.fit_twin, built.state)
    fisher = fisher_information(
        forward, values, noise,
        space=space if run.options["space"] else None)
    return {"fisher": fisher,
            "covariance": parameter_covariance(
                fisher, jitter=run.options["jitter"])}


def _loss_fn(run: RunSpec) -> Any:
    from rheplicant.inference import mean_squared_error

    loss = run.options.get("loss", "mse")
    if loss == "mse":
        return mean_squared_error
    if isinstance(loss, dict) and set(loss) == {"python"}:
        from rheplicant.config.hatch import import_target

        scoring = import_target(loss["python"])
        # The sibling of kind: gradient's objective: check, and named by its
        # refusal.  Without it the natural mistake -- `def my_loss(prediction)`
        # -- reaches the user as a raw `TypeError: ... takes exactly one
        # argument (2 given)` naming no run, which the layer's own
        # single-ConfigError grep cannot see, because it escapes through a
        # CALL rather than a raise.  mean_squared_error(prediction, observed)
        # is the shape every loss must take.
        binds, signature = _binds(scoring, _PROBE, _PROBE)
        if not binds:
            raise ConfigError(
                f"runs[{run.name!r}]: loss: {loss['python']!r} cannot be "
                f"called as (prediction, observed) -- its signature is "
                f"{signature}. A loss takes BOTH, the same shape "
                "kind: gradient's objective: takes, even when it reads only "
                "the first."
            )
        return scoring
    raise ConfigError(
        f"runs[{run.name!r}]: loss: is 'mse' or {{python: 'mod:fn'}}; got "
        f"{loss!r}."
    )


def _parse_optimize(options, context):
    """The whole optimize grammar: keys, requireds, routes, loss, defaults.

    Every refusal below is the one the executor used to raise mid-run, in
    the same order; the executor is left with the calibrator and the fit.
    """
    spec = context.spec
    built = context.configured_run
    _sweep(spec, _OPTIMIZE_KEYS)
    inference = built.inference
    _observed(spec, built)
    optimizer = options.get("optimizer")
    if optimizer not in ("gradient", "adam"):
        raise ConfigError(
            f"runs[{spec.name!r}]: optimizer: gradient or adam is required; "
            f"got {optimizer!r}. The two are different algorithms behind an "
            "identical .fit."
        )
    for key in ("learning_rate", "n_steps"):
        if key not in options:
            raise ConfigError(
                f"runs[{spec.name!r}]: {key}: is required -- the shipped "
                "default (1e-2) sits five orders of magnitude from what a "
                "real fit has needed (examples/radio_digital_twin.py:112)."
            )
    if optimizer == "gradient":
        for key in _ADAM_ONLY:
            if key in options:
                raise ConfigError(
                    f"runs[{spec.name!r}]: {key}: belongs to optimizer: "
                    "adam."
                )
    execution = {"optimizer": optimizer}
    resolved = {"optimizer": optimizer}
    execution["learning_rate"] = resolved["learning_rate"] = _number(
        spec, "learning_rate", options["learning_rate"], kind=float)
    execution["n_steps"] = resolved["n_steps"] = _number(
        spec, "n_steps", options["n_steps"], kind=int)
    if optimizer == "adam":
        for key in _ADAM_ONLY:
            execution[key] = resolved[key] = _number(
                spec, key, options.get(key, _ADAM_DEFAULTS[key]), kind=float)
    trainable = inference.trainable
    space = inference.space
    if trainable is not None and space is not None:
        raise ConfigError(
            f"runs[{spec.name!r}]: inference.trainable and "
            "inference.parameters are both declared, and optimize cannot "
            "serve two masters -- trainable is the calibrator-only route "
            "(schema §4.7.5); drop one."
        )
    if trainable is None and space is None:
        raise ConfigError(
            f"runs[{spec.name!r}]: optimize needs inference.trainable or "
            "inference.parameters -- something must be free to move."
        )
    execution["loss"] = _loss_fn(spec)
    resolved["loss"] = options.get("loss", "mse")
    return parsed_options(execution, resolved=resolved)


@register("optimize", parse=_parse_optimize)
def _run_optimize(run: ParsedRun, built: Any, previous: Any = None) -> Any:
    from rheplicant.inference import AdamCalibrator, GradientCalibrator, build_forward_fn

    _sweep(run, _OPTIMIZE_KEYS)
    inference = built.inference
    observed = _observed(run, built)
    options = run.options
    if options["optimizer"] == "gradient":
        calibrator = GradientCalibrator(
            learning_rate=options["learning_rate"],
            n_steps=options["n_steps"])
    else:
        calibrator = AdamCalibrator(
            learning_rate=options["learning_rate"],
            n_steps=options["n_steps"], beta1=options["beta1"],
            beta2=options["beta2"], eps=options["eps"])
    if inference.trainable is not None:
        forward, params0 = build_forward_fn(inference.fit_twin, built.state,
                                            inference.trainable)
    else:
        forward, params0 = inference.space.forward_fn(inference.fit_twin,
                                                      built.state)
    params_fit, losses = calibrator.fit(forward, params0, observed,
                                        loss_fn=options["loss"])
    return {"params": params_fit, "losses": losses}


_BLOCK_KEYS = frozenset({"names", "steps", "engine", "learning_rate"})
_ESTIMATE_KEYS = frozenset({"blocks", "max_iter", "tol", "min_sweeps",
                            "check_identifiability", "solve_tol",
                            "solve_guard"})
_SAMPLE_KEYS = frozenset({"blocks", "seed", "n_sweeps", "warmup", "rhat_max",
                          "warm_start", "check_identifiability", "solve_tol",
                          "solve_guard"})
_WARM_KEYS = frozenset({"kind", "blocks", "max_iter", "tol", "min_sweeps",
                        "move", "check_identifiability", "solve_tol",
                        "solve_guard"})
_ESTIMATE_PASSTHROUGH = ("max_iter", "tol", "min_sweeps",
                         "check_identifiability", "solve_tol", "solve_guard")
_SAMPLE_PASSTHROUGH = ("warmup", "rhat_max", "check_identifiability",
                       "solve_tol", "solve_guard")
#: The parser-injected defaults behind the passthrough tuples, measured
#: against ``SamplingPlan``'s own signatures (``DEFAULT_MAX_ITER``,
#: ``DEFAULT_CHI2_TOL``, ``MIN_SWEEPS``, ``CHECK_ONCE``, ``solve_tol=1e-6``,
#: ``solve_guard=1e-3``; ``warmup=None``, ``DEFAULT_RHAT_MAX``) -- an
#: explicit keyword is byte-identical to today's omission.  A warm start
#: gets none: its passthrough has always been declared-only.
_ESTIMATE_DEFAULTS = {"max_iter": 100, "tol": 1e-8, "min_sweeps": 3,
                      "check_identifiability": "once", "solve_tol": 1e-6,
                      "solve_guard": 0.001}
_SAMPLE_DEFAULTS = {"warmup": None, "rhat_max": 1.05,
                    "check_identifiability": "once", "solve_tol": 1e-6,
                    "solve_guard": 0.001}


def _blocks(where: str, node: Any) -> tuple[Any, ...]:
    from rheplicant.inference import Block

    if not isinstance(node, list) or not node:
        raise ConfigError(f"{where}: blocks: is a non-empty list of block "
                          f"mappings; got {node!r}.")
    built = []
    for index, entry in enumerate(node):
        if not isinstance(entry, dict):
            raise ConfigError(f"{where}: blocks[{index}] is a mapping; got "
                              f"{entry!r}.")
        unknown = sorted(set(entry) - _BLOCK_KEYS)
        if unknown:
            raise ConfigError(
                f"{where}: blocks[{index}] does not take {unknown}; a block "
                f"takes {sorted(_BLOCK_KEYS)}."
            )
        names = entry.get("names")
        if not isinstance(names, list) or not names or not all(
                isinstance(name, str) for name in names):
            raise ConfigError(f"{where}: blocks[{index}].names is a "
                              f"non-empty list of latent names; got "
                              f"{entry.get('names')!r}.")
        knobs = {key: entry[key] for key in ("steps", "engine",
                                             "learning_rate") if key in entry}
        built.append(Block(*names, **knobs))
    return tuple(built)


def _a29_estimate_takes_no_seed(where: str, options: Mapping[str, Any]) -> None:
    """``plan.estimate`` refuses a seed; ``plan.sample`` requires one.

    Module-level and taking plain data so that ``preflight/fitting.py`` calls
    the SAME refusal from the raw document (plan §2.2: one name, one binding,
    two call sites).  A copy of this string in the pass is the
    ``_number``-vs-``_whole`` divergence with a new name.

    ``warm_start:`` is deliberately not a second caller.  It builds a
    ``plan.estimate`` of its own (``:287-288``, and ``kind:`` there may be
    nothing else -- ``:268-271``), but ``_WARM_KEYS`` (``:172``) carries no
    ``seed`` at all, so the sweep at ``:264-267`` already refuses one by name
    -- a more specific sentence than this asymmetry.
    """
    if "seed" in options:
        raise ConfigError(
            f"{where}: plan.estimate refuses a seed -- the asymmetry is the "
            "package's own (sample takes key=, estimate has no key "
            "parameter; check A29). Drop it, or make this run plan.sample."
        )


def _parse_plan(options, context):
    """Both plan kinds' grammar: blocks, knobs, seed, warm start -- no draws.

    One parser for the pair, branching on ``spec.kind`` exactly where the
    executor used to.  The warm start is validated and normalized here --
    its ``plan.estimate`` grammar is this same one -- but the estimate
    itself is scientific execution and stays in the executor.
    """
    from rheplicant.config.draws import _seed_name, seed_for

    spec = context.spec
    built = context.configured_run
    estimate = spec.kind == "plan.estimate"
    where = f"runs[{spec.name!r}]"
    if estimate:
        _a29_estimate_takes_no_seed(where, options)
    _sweep(spec, _ESTIMATE_KEYS if estimate else _SAMPLE_KEYS)
    space = _space(spec, built)
    _noise(spec, built)
    _observed(spec, built)
    execution = {"blocks": _blocks(where, options.get("blocks"))}
    resolved = {"blocks": options["blocks"]}
    if estimate:
        for key in _ESTIMATE_PASSTHROUGH:
            execution[key] = resolved[key] = options.get(
                key, _ESTIMATE_DEFAULTS[key])
        return parsed_options(execution, resolved=resolved)
    if "n_sweeps" not in options:
        raise ConfigError(f"{where}: n_sweeps: is required for plan.sample.")
    execution["seed"] = resolved["seed"] = seed_for(
        _seed_name(dict(options), where), built.context)
    warm = options.get("warm_start")
    warm_execution = None
    if warm is not None:
        if not isinstance(warm, dict):
            raise ConfigError(f"{where}: warm_start: is a mapping; got "
                              f"{warm!r}.")
        unknown = sorted(set(warm) - _WARM_KEYS)
        if unknown:
            raise ConfigError(f"{where}: warm_start does not take "
                              f"{unknown}; it takes {sorted(_WARM_KEYS)}.")
        if warm.get("kind") != "plan.estimate":
            raise ConfigError(
                f"{where}: warm_start.kind: plan.estimate is the one warm "
                f"start there is; got {warm.get('kind')!r}."
            )
        move = warm.get("move")
        if not isinstance(move, list) or not move or not all(
                isinstance(name, str) for name in move):
            raise ConfigError(
                f"{where}: warm_start.move: is required -- the latents "
                "whose inits the warm start moves; the rest stay declared."
            )
        missing = sorted(set(move) - set(space.names))
        if missing:
            raise ConfigError(
                f"{where}: warm_start.move names {missing}, which "
                f"inference.parameters does not declare; it declares "
                f"{list(space.names)}."
            )
        warm_execution = {"kind": "plan.estimate",
                          "blocks": _blocks(f"{where}: warm_start",
                                            warm.get("blocks")),
                          "move": tuple(move)}
        warm_execution.update(_passthrough(warm, _ESTIMATE_PASSTHROUGH))
    execution["n_sweeps"] = resolved["n_sweeps"] = _number(
        spec, "n_sweeps", options["n_sweeps"], kind=int)
    for key in _SAMPLE_PASSTHROUGH:
        execution[key] = resolved[key] = options.get(
            key, _SAMPLE_DEFAULTS[key])
    if warm_execution is not None:
        execution["warm_start"] = warm_execution
        resolved["warm_start"] = warm
    return parsed_options(execution, resolved=resolved)


@register("plan.estimate", parse=_parse_plan)
@register("plan.sample", parse=_parse_plan)
def _run_plan(run: ParsedRun, built: Any, previous: Any = None) -> Any:
    import equinox as eqx
    import jax

    from rheplicant.inference import SamplingPlan

    estimate = run.kind == "plan.estimate"
    where = f"runs[{run.name!r}]"
    if estimate:
        _a29_estimate_takes_no_seed(where, run.options)
    _sweep(run, _ESTIMATE_KEYS if estimate else _SAMPLE_KEYS)
    inference = built.inference
    space = _space(run, built)
    noise = _noise(run, built)
    observed = _observed(run, built)
    options = run.options
    if estimate:
        return SamplingPlan(space, *options["blocks"]).estimate(
            inference.fit_twin, built.state, observed, noise=noise,
            **_passthrough(options, _ESTIMATE_PASSTHROUGH))
    key = jax.random.key(options["seed"])
    warm = options.get("warm_start")
    if warm is not None:
        est = SamplingPlan(space, *warm["blocks"]).estimate(
            inference.fit_twin, built.state, observed, noise=noise,
            **_passthrough(warm, _ESTIMATE_PASSTHROUGH))
        space = eqx.tree_at(
            lambda s: [s.latent(name).init for name in warm["move"]], space,
            [est.values[name] for name in warm["move"]])
    return SamplingPlan(space, *options["blocks"]).sample(
        inference.fit_twin, built.state, observed, noise=noise, key=key,
        n_sweeps=options["n_sweeps"],
        **_passthrough(options, _SAMPLE_PASSTHROUGH))


#: The compatibility path has no canonical layer and no schedule position:
#: ``run_document`` executes the base schedule through ``execute_run``
#: without a trace, and Task 10's orchestration is what installs the real
#: target layers and indices.  The shim carries an EMPTY document rather
#: than ``built.document`` -- freezing a caller's document here would cost a
#: deep evidence pass per run and could refuse programmatic documents the
#: layer accepts today, neither of which the compatibility wrapper may do.
_COMPATIBILITY_LAYER = LayerRef(kind="base", name=None, prefix="",
                                document={}, declared_runs=None)


def execute_run(run: RunSpec, built: Any,
                results: Any = None) -> RunResult:
    """One run entry against its ConfiguredRun -> a RunResult.

    The wrapper keeps its exact signature and its legacy unknown-kind
    refusal byte-for-byte, then routes through the registry: parse once,
    pre-execute, execute.  The ``expect: refuse`` capture wraps the whole
    parse/pre-execute/execute triple: the grammar refusals the parser now
    raises ran INSIDE the bare executor call before the parse/execute
    split, so wrapping the parse is what preserves today's capture scope,
    not a widening of it.
    """
    if run.kind not in EXECUTORS:
        raise ConfigError(
            f"runs[{run.name!r}]: kind: {run.kind} has no executor. Every "
            f"kind this layer declares must register one; it knows "
            f"{sorted(EXECUTORS)}."
        )
    previous = results if results is not None else MappingProxyType({})

    def attempt():
        parsed = parse_run(run, built, index=0, layer=_COMPATIBILITY_LAYER)
        handler = handler_for(run.kind)
        handler.pre_execute(parsed, built, previous)
        return handler.execute(parsed, built, previous)

    if run.expect == "refuse":
        try:
            attempt()
        except Exception as error:  # noqa: BLE001 -- run-and-capture is the point
            return RunResult(name=run.name, kind=run.kind, product=None,
                             error=error, variant=run.variant)
        raise ConfigError(
            f"runs[{run.name!r}]: expect: refuse, and kind: {run.kind} "
            "SUCCEEDED -- the assertion this run makes about the design no "
            "longer holds."
        )
    return RunResult(name=run.name, kind=run.kind, product=attempt(),
                     error=None, variant=run.variant)


# Importing the leaf modules is what registers their kinds.  The import sits
# at the foot rather than the head because they import the shared support
# this module also imports; the completeness test in
# tests/config/test_config_exit_support.py is what keeps it from rotting.
from rheplicant.config.sections import conjugate, diagnostics, npe, nuts  # noqa: E402,F401
