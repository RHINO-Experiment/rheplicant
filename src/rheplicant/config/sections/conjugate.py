"""The conjugate family: its shared machine, and the exits that ride it.

Four exits live here -- ``conjugate.wiener``, ``conjugate.gcr``,
``conjugate.gls`` and ``condition`` (schema §4.7.9).  Every one of them starts
by turning ``names:`` into a :class:`~rheplicant.inference.linear.LinearBlock`
and ``prior_std:``/``prior_mean:`` into the per-member mappings the grouped
solves take, so both live here rather than four times over.

The module is in two halves.  The first is that shared machine --
:func:`_conjugate_block` and :func:`_prior_kwargs`, which every exit opens
with.  The second is the executors: :func:`_run_conjugate`, registered for
``conjugate.wiener``, is where a document's run actually reaches
``wiener_solve``, and it owns the ``width:`` vocabulary and the solver-knob
table that go with it.

Two rules this module exists to keep:

* **The block is always the GROUPED spelling.**  ``linear_operator`` takes
  ``name=`` OR ``names=`` and they are not interchangeable: the first returns
  a bare array, the second ``{latent: array}``, and six downstream consumers
  raise on the bare form (``linear.py:184-215``).  The config layer compiles
  to ``names=`` even for a block of one.
* **A grouped block's prior is per member.**  ``S`` is block-diagonal, not a
  multiple of the identity, so ``_per_member`` (``linear.py:963-973``) refuses
  a scalar outright rather than broadcasting it.  A scalar in a document is
  therefore broadcast HERE, and only when the block names exactly one latent
  -- check A51.

``check:`` belongs to ``linear_operator`` alone; none of the four solves takes
it.  It is passed only when the document declares it, so the package's own
default stands otherwise.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from rheplicant.config.errors import ConfigError
from rheplicant.config.sections.exit_support import (
    _decided_sigma,
    _number,
    _observed,
    _space,
    _sweep,
    register,
)
from rheplicant.config.sections.runs import RunSpec

# --- One block, so Tasks 4-6 have one place to add to ----------------------

# The pieces Tasks 3-6 COMPOSE each exit's own _sweep set from.  Not a
# wholesale union for all four: see _SOLVE_KEYS below for which exit takes
# which, and why `condition` takes only part of _PRIOR_KEYS.
_BLOCK_KEYS = frozenset({"names", "check"})
_PRIOR_KEYS = frozenset({"prior_std", "prior_mean"})
#: The two kinds that ALWAYS solve at a DECLARED sigma, so _conjugate_block
#: resolves it for them and check A27 fires there.  conjugate.gcr may take
#: GLSResult.noise_std instead (``noise_from: gls``) and conjugate.gls takes
#: the noise RULE through ``_decided_model``: resolving a decided sigma for
#: either would fire A27 on exactly the document that exit exists to serve.
_DECIDES_SIGMA_HERE = frozenset({"conjugate.wiener", "condition"})
#: key, coercion, floor, and whether the package spells "off" as None.
_SOLVER_KNOBS = (("tol", float, 0.0, False),
                 ("maxiter", int, 1, True),
                 ("require_convergence", float, 0.0, True))
#: The same three, as names: the CG knobs every conjugate SOLVE forwards.
#: Derived rather than retyped so the two can never drift.
_SOLVE_PASSTHROUGH = tuple(key for key, _cast, _floor, _null in _SOLVER_KNOBS)
#: What the three conjugate SOLVES take -- ``conjugate.wiener`` (this task),
#: ``conjugate.gcr`` (Task 4) and ``conjugate.gls`` (Task 5), which union
#: their own keys onto this.
#:
#: NOT what every exit in this module takes.  ``condition`` is the fourth, and
#: measured against the venv its signature is
#: ``condition_estimate(block, *, noise_std, prior_std=None, iterations=12,
#: key=None)`` -- so of these seven it can use exactly three: ``names`` and
#: ``check`` (which build the block) and ``prior_std``.  It takes no
#: ``prior_mean`` and none of the three CG knobs, because it runs power
#: iteration rather than CG.  Task 6 therefore builds its own set from
#: ``_BLOCK_KEYS | {"prior_std", "iterations"}`` rather than unioning onto
#: this one, and it must NOT call :func:`_prior_kwargs`, which emits
#: ``prior_mean`` whenever the document declares it and would reach
#: ``condition_estimate`` as a TypeError.
_SOLVE_KEYS = _BLOCK_KEYS | _PRIOR_KEYS | frozenset(_SOLVE_PASSTHROUGH)
_WIENER_KEYS = _SOLVE_KEYS | frozenset({"width"})
#: All three names, including the one that is refused: an unknown ``width:``
#: must still be told the whole vocabulary, or a reader who wanted draws is
#: told the word does not exist rather than where it lives.
_WIDTHS = ("none", "draws", "fisher")


def _selected(run: Any, where: str) -> tuple[str, ...]:
    """``names:`` -> the latents this block groups, in the declared order."""
    names = run.options.get("names")
    if isinstance(names, str):
        return (names,)
    if (isinstance(names, list) and names
            and all(isinstance(one, str) for one in names)):
        return tuple(names)
    raise ConfigError(
        f"{where}: names: is required -- one latent name, or a list of "
        "them. The conjugate exits always build the GROUPED operator "
        "(linear_operator(names=)), whose solution comes back keyed by "
        f"latent, so which latents the block holds is not guessed; got "
        f"{names!r}."
    )


def _conjugate_block(run: Any, built: Any, where: str, *,
                     needs_observed: bool = True) -> tuple[Any, Any, Any]:
    """``(block, sigma, observed)`` -- everything a conjugate solve opens with.

    Three things come back together because no executor may hold one without
    having decided the others: the grouped ``LinearBlock``, the decided sigma
    (with check A27), and the observation.

    ``needs_observed`` says whether this exit's solve reads data:
    ``condition`` estimates kappa from the operator alone
    (``linear.py:1337`` takes no ``observed``), while the three solves do.
    Where it is True the missing-observation refusal fires BEFORE the
    operator is built, so a document with no ``inference.observed`` hears
    about the data it did not declare rather than about its latents; where it
    is False ``observed`` is None and ``_observed`` is never reached.

    ``sigma`` is :func:`_decided_sigma` for the kinds in
    :data:`_DECIDES_SIGMA_HERE` and None for the other two, which find their
    own -- see that constant's note.

    ``where`` is the ``exits.py`` spelling ``f"runs[{run.name!r}]"`` and is the
    THIRD POSITIONAL argument, never a keyword (plan section 3.1).
    """
    from rheplicant.inference import linear_operator

    space = _space(run, built)
    observed = _observed(run, built) if needs_observed else None
    sigma = (_decided_sigma(run, built)
             if run.kind in _DECIDES_SIGMA_HERE else None)
    knobs: dict[str, Any] = {}
    if "check" in run.options:
        check = run.options["check"]
        if not isinstance(check, bool):
            raise ConfigError(f"{where}: check: is a bool; got {check!r}.")
        knobs["check"] = check
    block = linear_operator(space, built.inference.fit_twin, built.state,
                            names=_selected(run, where), **knobs)
    return block, sigma, observed


def _one_prior(run: Any, where: str, key: str, value: Any, block: Any,
               space: Any) -> dict[str, Any]:
    """One of ``prior_std``/``prior_mean`` -> the per-member mapping."""
    minimum = 0.0 if key == "prior_std" else None
    if isinstance(value, Mapping):
        if set(value) != set(block.names):
            declared = [name for name in block.names
                        if space.latent(name).prior is not None]
            raise ConfigError(
                f"{where}: {key}: names {sorted(value)}, and this block "
                f"groups {list(block.names)}; S is block-diagonal, so a "
                "grouped block takes one entry per latent. Name every "
                f"member, or drop {key}: and let each latent's own prior: "
                f"drive the solve ({declared} declare one)."
            )
        return {name: _number(run, f"{key}.{name}", value[name], kind=float,
                              minimum=minimum)
                for name in block.names}
    number = _number(run, key, value, kind=float, minimum=minimum)
    if len(block.names) == 1:
        return {block.names[0]: number}
    raise ConfigError(
        f"{where}: {key}: {value!r} is one number for a block grouping "
        f"{list(block.names)}, and S is block-diagonal rather than a "
        "multiple of the identity: their widths differ by orders of "
        "magnitude and a block-diagonal S returns a finite, "
        "correctly-shaped, wrongly-regularised answer with no residual "
        f"signature (check A51). Write one entry per latent -- {key}: "
        f"{{{block.names[0]}: ...}} -- or drop the key and let each "
        "latent's own prior: drive the solve."
    )


def _prior_kwargs(run: Any, built: Any, block: Any,
                  where: str) -> dict[str, Any]:
    """The ``prior_std=``/``prior_mean=`` keywords the solve should take.

    Absent keys are absent from the result: the package then reads each
    latent's own ``Latent(prior=...)`` (``linear.py:928-931``), which is the
    standing decision that config never restates a package default.

    The ParameterSpace is derived HERE, from ``built``.  Callers pass
    ``built`` -- never a space, and never the ``where`` string, which binds
    silently and then breaks only inside the refusal branch that reads
    ``space.latent(name).prior`` (plan section 3.1).
    """
    space = _space(run, built)
    return {key: _one_prior(run, where, key, run.options[key], block, space)
            for key in ("prior_std", "prior_mean") if key in run.options}


def _knobs(run: Any, specs: tuple) -> dict[str, Any]:
    """The knobs among ``specs`` this document declared, coerced.

    A knob the document omits is omitted from the call, so the package's own
    default stands -- this plan's standing decision.  ``None`` passes through
    where the package spells "off" that way (``maxiter: null`` is no cap,
    ``require_convergence: null`` is no guard); everywhere else a non-number
    is a ConfigError here rather than a bare TypeError from inside a trace.

    Tasks 4 and 5 call this too, with their own spec tuples, so that
    ``maxiter: "many"`` is refused identically whichever exit was asked for.
    """
    resolved: dict[str, Any] = {}
    for key, cast, floor, nullable in specs:
        if key not in run.options:
            continue                  # the package's own default stands
        value = run.options[key]
        if value is None and nullable:
            resolved[key] = None      # "no cap" / "no guard", as the package
            continue                  # spells them
        resolved[key] = _number(run, key, value, kind=cast, minimum=floor)
    return resolved


def _width(run: Any, where: str) -> str:
    """``width:`` -> one of :data:`_WIDTHS`, or the refusal that names them.

    ``draws`` is in the vocabulary and refused by name: it would be a second
    spelling of ``kind: conjugate.gcr`` -- the same solver, the same seed, the
    same draws -- and check A29 (a seed is required for gcr and refused for
    the deterministic conjugate exits) would need a branch to survive it.
    """
    width = run.options.get("width")
    if width not in _WIDTHS:
        raise ConfigError(
            f"{where}: width: is required and has no default; it is one of "
            f"{list(_WIDTHS)} and this run declares {width!r}. wiener_solve "
            "returns the posterior MEAN alone, and a mean with no error bar "
            "is not a posterior -- width: none is the deliberate admission "
            "that this run wants only the mean."
        )
    if width == "draws":
        raise ConfigError(
            f"{where}: width: draws draws the posterior with gcr_sample, "
            "which needs a named seed and a draw count -- and n_draws: and "
            "seed: are kind: conjugate.gcr's own keys, not this exit's. "
            "Declare a run with kind: conjugate.gcr over the same names:, or "
            "width: fisher for the Gaussian width around this mean."
        )
    return width


def _require_the_whole_space(where: str, space: Any, block: Any) -> None:
    """Refuse ``width: fisher`` over anything less than the whole space.

    ``fisher_information(space=...)`` inverts the posterior precision of every
    latent the space declares, so over a sub-block it MARGINALISES the ones
    the solve held fixed -- and reports the wider number under this run's
    name, where it reads as this block's own width.
    """
    outside = sorted(set(space.names) - set(block.names))
    if not outside:
        return
    raise ConfigError(
        f"{where}: width: fisher inverts the posterior precision of the "
        "WHOLE parameter space, so it is this block's width only when the "
        f"block IS the space; this block groups {list(block.names)} and "
        f"inference.parameters also declares {outside}, which the solve held "
        "fixed and the covariance would marginalise. Widen names: to the "
        "whole space, or take the width from a kind: conjugate.gcr run over "
        "this block."
    )


def _gaussian_width(built: Any, space: Any, sigma: Any,
                    mean: Any) -> dict[str, Any]:
    """``width: fisher`` -> the Fisher and the posterior covariance.

    ``space=`` is what makes it the POSTERIOR precision rather than the
    likelihood one -- the width of a fit, not of the solve just computed.
    Measured on the tight-prior document: sigma 0.00476898 with it, 0.01587247
    without, and the returned matrix calls itself ``covariance`` rather than
    ``posterior_covariance``.

    The Fisher is taken AT the solved mean, which is where a width belongs.
    ``mean`` alone, not merged over ``forward_fn``'s starting values:
    :func:`_require_the_whole_space` has already established that the block IS
    the space, and ``mean`` is keyed by the block's members, so a merge would
    be shadowed key for key and read as a defence that is not defending.

    No test pins the CHOICE of point, and honestly so: the block this exit
    solves is affine in its latents by construction (``check:`` defaults True
    and ``check_linearity`` enforces it), so J -- and therefore the Fisher --
    does not depend on it.  Measured over the three documents where it could
    differ: the two non-affine ones are degenerate (sigma comes back at the
    prior width, or NaN) and the affine one agrees to every digit.  The mean
    is here because it is the right point once ``check: false`` lets a curved
    latent through.
    """
    from rheplicant.inference import fisher_information, parameter_covariance

    forward, _ = space.forward_fn(built.inference.fit_twin, built.state)
    fisher = fisher_information(forward, mean, sigma, space=space)
    return {"fisher": fisher, "covariance": parameter_covariance(fisher)}


@register("conjugate.wiener")
def _run_conjugate(run: RunSpec, built: Any, *, results: Any = None) -> Any:
    """linear_operator + wiener_solve, and the width the solve does not give.

    ``wiener_solve`` returns ``(x_hat, relative_residual)`` -- the posterior
    MEAN and a CG diagnostic, no covariance and no draw.  ``width:`` is
    therefore required and has no default: it is the run's statement about
    what it wants in place of the error bar it is not getting.

    This is the family's one executor: Task 4 registers ``conjugate.gcr`` on
    it as well and branches on ``run.kind``, exactly as ``exits.py``'s
    ``_run_plan`` carries ``plan.estimate`` and ``plan.sample``.  The three
    lines that are ``conjugate.wiener``'s alone and become that branch are the
    sweep set, :func:`_width`, and the solver call; everything else --
    :func:`_knobs`, :func:`_conjugate_block`, :func:`_prior_kwargs` and the
    product's ``mean``/``residual`` -- is already shared.  ``conjugate.gls``
    is a SEPARATE executor: its sigma spelling, its product and its
    convergence gate share nothing with this body.
    """
    from rheplicant.inference import wiener_solve

    _sweep(run, _WIENER_KEYS)
    where = f"runs[{run.name!r}]"
    width = _width(run, where)
    solver = _knobs(run, _SOLVER_KNOBS)
    space = _space(run, built)
    # Three together: no executor may hold one without having decided the
    # others.  `check:` went to linear_operator in there and reaches no solve
    # -- 'check' in inspect.signature(wiener_solve).parameters is False.
    block, sigma, observed = _conjugate_block(run, built, where)
    if width == "fisher":
        _require_the_whole_space(where, space, block)
    solution, residual = wiener_solve(
        block, observed, noise_std=sigma,
        **_prior_kwargs(run, built, block, where), **solver)
    # as_dict is the idempotent wrap (linear.py:184): the product's shape is
    # the mapping six downstream consumers read, whichever spelling built the
    # block.  Over a GROUPED block -- the only one this layer compiles -- it
    # is a no-op, so no test can distinguish it from `solution`; it is here
    # because it is what makes the product's shape independent of that.
    # float() because the residual is a scalar jax.Array and these products
    # end up in reports and in expect: refuse messages.
    product = {"mean": block.as_dict(solution),
               "residual": float(residual),
               "width": width}
    if width == "none":
        return product
    return {**product, **_gaussian_width(built, space, sigma,
                                         product["mean"])}
