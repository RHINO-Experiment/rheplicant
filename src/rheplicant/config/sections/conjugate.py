"""The conjugate family: its shared machine, and the exits that ride it.

Four exits live here -- ``conjugate.wiener``, ``conjugate.gcr``,
``conjugate.gls`` and ``condition`` (schema §4.7.9).  Every one of them starts
by turning ``names:`` into a :class:`~rheplicant.inference.linear.LinearBlock`
and ``prior_std:``/``prior_mean:`` into the per-member mappings the grouped
solves take, so both live here rather than four times over.

The module is in two halves.  The first is that shared machine --
:func:`_conjugate_block` and :func:`_prior_kwargs`, which every exit opens
with, plus :func:`_gls_result`, the one ``iterative_gls`` call in the layer.
The second is the executors: :func:`_run_conjugate` is registered for BOTH
``conjugate.wiener`` and ``conjugate.gcr`` and branches on ``run.kind``,
because the two decide their block, their prior and their CG knobs
identically and differ only in what they do with them -- the posterior mean
(:func:`_wiener_product`, which owns the ``width:`` vocabulary) or a draw from
the posterior itself (:func:`_gcr_product`, which owns ``n_draws:``,
``seed:`` and ``noise_from:``).

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
    _decided_model,
    _decided_sigma,
    _noise,
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
#: reweight knob, coercion, floor, and whether the package spells "off" as
#: None -- the same 4-tuple shape as :data:`_SOLVER_KNOBS`, so one
#: :func:`_knobs` call serves both.  None of the three is nullable:
#: ``iterative_gls`` derives ``reweight_tol`` from ``tol`` when it is absent,
#: which is what absent means.
_GLS_KNOB_SPECS = (("reweight_tol", float, 0.0, False),
                   ("min_reweights", int, 1, False),
                   ("max_reweights", int, 1, False))
#: The same three, as names: what the sweep allows and what the
#: ``noise_from: declared`` staleness check looks for.  Derived rather than
#: retyped, so a knob cannot be swept in and coerced by nothing.
_GLS_KNOBS = tuple(key for key, _cast, _floor, _null in _GLS_KNOB_SPECS)
#: Where ``conjugate.gcr`` may take its ``noise_std=`` from.  ``declared`` is
#: :func:`_decided_sigma`; ``gls`` is ``GLSResult.noise_std``, which is the
#: only route open to a document check A27 refuses.
_NOISE_FROM = ("declared", "gls")
#: ``conjugate.gcr``'s own set, built from the SHARED :data:`_SOLVE_KEYS` and
#: not from :data:`_WIENER_KEYS`: ``width:`` asks a draw for the error bar it
#: already is.
_GCR_KEYS = _SOLVE_KEYS | frozenset(
    {"n_draws", "seed", "noise_from", "acknowledge_unconverged_covariance",
     *_GLS_KNOBS})


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


def _gls_result(run: Any, built: Any, *, block: Any, observed: Any,
                prior: dict, solve: dict, where: str) -> Any:
    """``iterative_gls`` at this document's noise model -> a ``GLSResult``.

    Shared by ``conjugate.gcr``'s ``noise_from: gls`` and, from Task 5, by
    ``conjugate.gls`` itself: whichever runs, the covariance is found the same
    way, the knobs are coerced the same way, and the same acknowledgement is
    demanded of a fixed point that was never reached.  One condition, one
    message, whichever exit reached it.

    ``prior`` is :func:`_prior_kwargs`' output and ``solve`` is
    :func:`_knobs`' over :data:`_SOLVER_KNOBS` -- both compiled ONCE by the
    executor and passed in, so the covariance is found at the same prior and
    the same CG tolerance the draw is then taken at.  **A caller must not pass
    ``solve`` a second time**; the reweight knobs are this function's own and
    are coerced here.  Every knob travels through :func:`_knobs`, so one the
    document does not declare never reaches ``iterative_gls`` and the
    package's default stands -- and ``maxiter: "many"`` is a ConfigError here
    exactly as it is on ``conjugate.wiener``.

    Everything after ``built`` is KEYWORD-ONLY on purpose.  Every parameter
    here is typed ``Any``, so a positional call written by analogy with a
    neighbouring helper would bind ``block`` to ``observed`` and ``where`` to
    ``prior`` in silence, and break only inside a refusal branch a passing
    test need never reach -- which is the exact defect plan section 3.1 was
    written to prevent.

    The ``{delta:.4g}`` in the refusal below is **pinned by two tasks** (this
    one and Task 5's ``conjugate.gls``) and is therefore API: at ``:.3g`` both
    pins fail, so a reword is a cross-task change rather than a local one.
    """
    from rheplicant.inference import iterative_gls

    acknowledged = run.options.get("acknowledge_unconverged_covariance", False)
    if not isinstance(acknowledged, bool):
        raise ConfigError(
            f"{where}: acknowledge_unconverged_covariance: is a bool; got "
            f"{acknowledged!r}."
        )
    found = iterative_gls(block, observed, noise=_decided_model(run, built),
                          **prior, **solve,
                          **_knobs(run, _GLS_KNOB_SPECS))
    if not bool(found.converged) and not acknowledged:
        raise ConfigError(
            f"{where}: iterative_gls stopped after {int(found.iterations)} "
            f"reweights at a relative step of {float(found.delta):.4g}, which "
            "never fell below reweight_tol -- so the noise_std it returns is "
            "NOT a fixed point, and everything conditioned on it inherits "
            "that (gls.py:89-92). Raise max_reweights, loosen reweight_tol, "
            "or declare acknowledge_unconverged_covariance: true to take the "
            "covariance as it stands."
        )
    return found


def _gls_record(found: Any) -> dict:
    """GLSResult's diagnostics as the Python scalars a product can carry.

    ``iterations``, ``delta`` and ``converged`` are jax.Arrays -- the casts
    are ``examples/gls_gcr.py:150-152``'s, and without them a product that
    says ``converged`` carries a truthy-but-untyped array into a report.  The
    three keys are exactly the GLSResult fields of the same names (measured:
    ``GLSResult._fields == ('noise_std', 'solution', 'residual', 'iterations',
    'delta', 'converged')``).
    """
    return {"iterations": int(found.iterations),
            "delta": float(found.delta),
            "converged": bool(found.converged)}


def _wiener_plan(run: Any, *, where: str) -> dict:
    """``conjugate.wiener``'s own grammar, decided before anything is built.

    One dict lookup against a three-word vocabulary, and it runs BEFORE
    :func:`_conjugate_block` for the reason :func:`_gcr_plan` gives.

    A dict of one, not the bare string, so that BOTH plans are the same type:
    the caller binds them to one name, and a wiener plan that came back a
    string turns any later ``plan["..."]`` on that path into
    ``TypeError: string indices must be integers`` -- a package-shaped failure
    from a layer whose job is to not produce those.
    """
    return {"width": _width(run, where)}


def _gcr_plan(run: Any, *, where: str) -> dict:
    """``conjugate.gcr``'s own grammar, decided before anything is built.

    Every check here is a dict lookup and a type test, and every one of them
    runs BEFORE :func:`_conjugate_block` -- which builds the operator and runs
    ``check_linearity``.  A document that is wrong in the grammar AND wrong in
    the model must hear about the grammar: the alternative is that a missing
    ``seed:`` surfaces as a ``ParameterSpaceError`` from inside the package,
    which is the seam this layer exists to close.
    """
    if "seed" not in run.options:
        raise ConfigError(
            f"{where}: conjugate.gcr draws from the posterior, so seed: is "
            "required and has no default -- gcr_sample's key= has none either "
            "(check A29). The deterministic conjugate exits, conjugate.wiener "
            "and conjugate.gls, refuse one instead."
        )
    noise_from = run.options.get("noise_from", "declared")
    if noise_from not in _NOISE_FROM:
        raise ConfigError(
            f"{where}: noise_from: is declared or gls; got {noise_from!r}."
        )
    if noise_from != "gls":
        # acknowledge_unconverged_covariance joins the three reweight knobs
        # here: every one is a member of _GCR_KEYS, so the sweep accepts them,
        # and under noise_from: declared _gls_result never runs to consult
        # them -- a declared key that reaches nothing, which is what this
        # block stops.  Computed only on the branch that can refuse: on the
        # gls route every one of them is read.
        stale = sorted(key for key in (*_GLS_KNOBS,
                                       "acknowledge_unconverged_covariance")
                       if key in run.options)
        if stale:
            raise ConfigError(
                f"{where}: {stale} are iterative_gls' own knobs and this run "
                "says noise_from: declared, which runs no GLS -- they would "
                "be read by nothing. Declare noise_from: gls, or drop them."
            )
    return {"noise_from": noise_from,
            "n_draws": _number(run, "n_draws", run.options.get("n_draws", 1),
                               kind=int, minimum=1)}


def _draw_sigma(run: Any, built: Any, *, block: Any, observed: Any,
                prior: dict, solve: dict, noise_from: str,
                where: str) -> tuple[Any, dict | None]:
    """``(noise_std, gls record)`` for a draw: declared, or the one GLS finds.

    ``noise_from: gls`` exists to serve exactly the document check A27
    refuses: a prediction-dependent sigma has no value until there is a
    prediction, and a draw's prediction is what it draws.  So this is the one
    place in the family where A27 has a third answer -- run the fixed-point
    iteration first, and draw at the covariance it converges to.

    ``noise_from`` arrives already validated, from :func:`_gcr_plan`.
    Keyword-only for the reason :func:`_gls_result` records.
    """
    if noise_from == "gls":
        found = _gls_result(run, built, block=block, observed=observed,
                            prior=prior, solve=solve, where=where)
        return found.noise_std, _gls_record(found)
    if getattr(_noise(run, built), "depends_on_prediction", False):
        raise ConfigError(
            f"{where}: inference.noise.kind: {built.inference.noise.kind} has "
            "a sigma that depends on the prediction, and a conjugate draw has "
            "no prediction to evaluate it at -- the prediction is what it "
            "draws. Declare noise_from: gls, which runs iterative_gls first "
            "and draws at the covariance it converges to, or "
            "inference.noise.kind: radiometer_frozen, which decides one sigma "
            "array up front (check A27)."
        )
    return _decided_sigma(run, built), None


def _wiener_product(run: Any, built: Any, *, block: Any, sigma: Any,
                    observed: Any, prior: dict, solve: dict, width: str,
                    where: str) -> dict:
    """``wiener_solve``, and the width the solve does not give.

    ``wiener_solve`` returns ``(x_hat, relative_residual)`` -- the posterior
    MEAN and a CG diagnostic, no covariance and no draw.  ``width:`` is
    therefore required and has no default: it is the run's statement about
    what it wants in place of the error bar it is not getting, and it arrives
    already decided, from :func:`_wiener_plan`.

    Keyword-only for the reason :func:`_gls_result` records.
    """
    from rheplicant.inference import wiener_solve

    space = _space(run, built)
    if width == "fisher":
        _require_the_whole_space(where, space, block)
    solution, residual = wiener_solve(block, observed, noise_std=sigma,
                                      **prior, **solve)
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


def _gcr_product(run: Any, built: Any, *, block: Any, observed: Any,
                 prior: dict, solve: dict, plan: dict, where: str) -> dict:
    """``gcr_sample``, ``n_draws`` times, from the seed the document names.

    ``gcr_sample`` returns ONE draw and has no ``n_draws`` parameter -- its
    own docstring says to ``vmap`` over split keys -- so ``n_draws:`` is this
    layer's own vocabulary rather than a package default restated, and the
    vmap over :func:`jax.random.split` is the whole of it.

    The residual is kept per draw rather than thrown away, which is what makes
    a badly conditioned block visible in the product instead of only in an
    ``eqx.error_if``.

    ``plan`` is :func:`_gcr_plan`'s output -- this run's ``noise_from`` and
    ``n_draws``, both already validated.  Keyword-only for the reason
    :func:`_gls_result` records.
    """
    import jax

    from rheplicant.config.draws import _seed_name, seed_for
    from rheplicant.inference import gcr_sample

    sigma, gls = _draw_sigma(run, built, block=block, observed=observed,
                             prior=prior, solve=solve,
                             noise_from=plan["noise_from"], where=where)
    # The NAME is resolved from the run and looked up in built.context, which
    # is the widened context ConfiguredRun carries; seed_for is what makes the
    # integer the draw consumes the same integer provenance.json reports.
    keys = jax.random.split(
        jax.random.key(seed_for(_seed_name(dict(run.options), where),
                                built.context)),
        plan["n_draws"])
    draws, residual = jax.vmap(
        lambda one: gcr_sample(block, observed, noise_std=sigma, key=one,
                               **prior, **solve))(keys)
    # as_dict is the idempotent wrap (linear.py:184), here for the same reason
    # as in :func:`_wiener_product` and just as unobservable: over a GROUPED
    # block -- the only one this layer compiles -- gcr_sample already returns
    # the mapping, so no test can distinguish this from `draws` (measured: the
    # whole suite is green with it removed).  It is what makes the product's
    # shape independent of which spelling built the block.
    return {"draws": block.as_dict(draws), "residual": residual,
            "noise_std": sigma, "gls": gls}


@register("conjugate.gcr")
@register("conjugate.wiener")
def _run_conjugate(run: RunSpec, built: Any, *, results: Any = None) -> Any:
    """The two conjugate exits that solve the same block: the mean, or a draw.

    One executor for both kinds, exactly as ``exits.py``'s ``_run_plan``
    carries ``plan.estimate`` and ``plan.sample``: the block, the prior and
    the CG knobs are decided identically, and only what is done with them
    differs.  ``conjugate.gls`` is a SEPARATE executor -- its sigma spelling,
    its product and its convergence gate share nothing with either half.

    **The two products deliberately share no key but** ``residual``:

    * ``conjugate.wiener`` -> ``{"mean", "residual", "width"}``, plus
      ``{"fisher", "covariance"}`` when ``width: fisher``;
    * ``conjugate.gcr`` -> ``{"draws", "residual", "noise_std", "gls"}``,
      always all four, ``gls`` being None under ``noise_from: declared``.

    So a ``predict`` or a report reading one must branch on
    ``RunResult.kind``, which is the rule section 2.4 records for every other
    product in this layer.
    """
    where = f"runs[{run.name!r}]"
    drawing = run.kind == "conjugate.gcr"
    _sweep(run, _GCR_KEYS if drawing else _WIENER_KEYS)
    # CHEAP FIRST, and the order is load-bearing.  Everything above
    # _conjugate_block is a dict lookup or a type test; _conjugate_block
    # builds the operator and runs check_linearity, which probes the
    # prediction at three scales per latent.  A document broken in its grammar
    # AND in its model must hear about the grammar, or a missing `width:`
    # surfaces as a ParameterSpaceError from inside the package -- exactly the
    # substitution this layer exists to prevent.  `plan` is the per-kind half
    # of that: `width:` is conjugate.wiener's alone and must not run for a
    # draw, and seed:/noise_from:/n_draws: are conjugate.gcr's.
    plan = (_gcr_plan if drawing else _wiener_plan)(run, where=where)
    solve = _knobs(run, _SOLVER_KNOBS)
    # Three together: no executor may hold one without having decided the
    # others.  `check:` went to linear_operator in there and reaches no solve
    # -- 'check' in inspect.signature(wiener_solve).parameters is False.
    # `sigma` comes back None for conjugate.gcr, which resolves its own below.
    block, sigma, observed = _conjugate_block(run, built, where)
    prior = _prior_kwargs(run, built, block, where)
    if drawing:
        return _gcr_product(run, built, block=block, observed=observed,
                            prior=prior, solve=solve, plan=plan, where=where)
    return _wiener_product(run, built, block=block, sigma=sigma,
                           observed=observed, prior=prior, solve=solve,
                           width=plan["width"], where=where)
