"""The conjugate family's executors: one exit each, over one shared opening.

Four exits live here -- ``conjugate.wiener``, ``conjugate.gcr``,
``conjugate.gls`` and ``condition`` (schema §4.7.9).  What they SHARE -- the
grouped block, the per-member prior, the coerced CG knobs -- is
``conjugate_support.py``; what each of them then does with it is here.

:func:`_run_conjugate` is registered for BOTH ``conjugate.wiener`` and
``conjugate.gcr`` and branches on ``run.kind``, because the two decide their
block, their prior and their CG knobs identically and differ only in what they
do with them -- the posterior mean (:func:`_wiener_product`, which owns the
``width:`` vocabulary) or a draw from the posterior itself
(:func:`_gcr_product`, which owns ``n_draws:``, ``seed:`` and ``noise_from:``).
:func:`_run_gls` is a SEPARATE executor for the reason its own docstring
gives, and the two meet again at :func:`_gls_result` -- the one
``iterative_gls`` call in the layer, which ``conjugate.gcr``'s
``noise_from: gls`` route and ``conjugate.gls`` itself both reach, so that one
condition has one message whichever exit found it.

:func:`_run_condition` is separate again, and shares less with the three than
they share with each other: ``condition_estimate`` reads no data, takes no
``prior_mean`` and none of the CG knobs, and answers with a bare scalar.  What
it does share is the opening -- the grouped block and the decided sigma -- so
it rides :func:`_conjugate_block` with ``needs_observed=False`` rather than
growing a second block builder.

The import runs one way: this module imports ``conjugate_support`` and
``exit_support``, and neither imports it back.  Nothing here is re-exported
from either of those, because two importable spellings of one function is how
the two drift apart.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from rheplicant.config.errors import ConfigError
from rheplicant.config.sections.conjugate_support import (
    _BLOCK_KEYS,
    _SOLVE_KEYS,
    _SOLVER_KNOBS,
    _conjugate_block,
    _knobs,
    _one_prior,
    _prior_kwargs,
)
from rheplicant.config.sections.exit_support import (
    _decided_model,
    _decided_sigma,
    _noise,
    _number,
    _space,
    _sweep,
    register,
)
from rheplicant.config.sections.runs import RunSpec

# --- One block, so each exit's vocabulary has one place to be read ---------

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
#: ``conjugate.gls``'s own set, built from the SHARED :data:`_SOLVE_KEYS` for
#: the same reason :data:`_GCR_KEYS` is: ``width:`` is ``conjugate.wiener``'s
#: alone.  It adds ``iterative_gls``' reweight knobs and the acknowledgement
#: its convergence gate demands, and nothing else -- no ``seed:`` (check A29
#: refuses one for the deterministic conjugate exits, and this exit returns a
#: point estimate and its covariance), and none of ``conjugate.gcr``'s draw
#: vocabulary.
_GLS_KEYS = _SOLVE_KEYS | frozenset(
    {"acknowledge_unconverged_covariance", *_GLS_KNOBS})
#: ``condition``'s own set, and the one member of the family built from
#: :data:`~rheplicant.config.sections.conjugate_support._BLOCK_KEYS` rather
#: than from :data:`_SOLVE_KEYS`.  Measured against the venv,
#: ``condition_estimate(block, *, noise_std, prior_std=None, iterations=12,
#: key=None)`` takes none of the three CG knobs and no ``prior_mean``: it runs
#: power iteration, not CG, so of _SOLVE_KEYS' seven it can use exactly three
#: (``names`` and ``check``, which build the operator, and ``prior_std``).
#: ``seed`` is its own -- optional here, where ``conjugate.gcr`` requires one,
#: because ``condition_estimate``'s ``key`` has a default and ``gcr_sample``'s
#: has none (schema §4.7.9 lists all four; A29's row names ``plan.estimate``
#: as the exit that REFUSES a seed).
_CONDITION_KEYS = _BLOCK_KEYS | frozenset({"prior_std", "iterations", "seed"})
#: The one knob it forwards, in :data:`_SOLVER_KNOBS`' 4-tuple shape so that
#: one :func:`_knobs` call serves it too.  NOT nullable: ``maxiter: null`` and
#: ``require_convergence: null`` are how the package spells "no cap" and "no
#: guard", and power iteration has no such spelling -- ``iterations: null``
#: would reach ``range(None)`` inside ``extreme_eigenvalues``.
_CONDITION_KNOBS = (("iterations", int, 1, False),)


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


#: What ``conjugate.gls`` tells ``_decided_model`` it wants the noise RULE
#: for, and what it offers a document that decided its sigma into an array
#: (check A28).  ``_decided_model`` takes both keyword-only and REQUIRED --
#: it wrote ONE sentence for both its callers until Plan 3A, and ``npe.py``
#: inherited this one, which is false of ``kind: npe`` in both halves.
#:
#: Bound here, at the caller's own module scope, rather than written inline
#: at the call site: the only document that reaches the raise below is one
#: the pre-flight pass now refuses at P-1, so ``conjugate.py``'s clause is
#: unreachable through ``run_document`` and a copy of it in a test would pin
#: nothing.  ``test_config_conjugate_shared.py``'s four direct
#: ``_decided_model`` calls spread THIS mapping, and
#: ``test_the_pass_writes_the_clause_its_caller_supplies``
#: (``test_preflight_fitting.py``) holds the pass's own A28 sentence to it.
#:
#: ``reads`` and ``because`` are ``be2027b``'s own words, to the character.
#: A two-clause form of :func:`~rheplicant.config.sections.exit_support.
#: _decided_model` templated *"as a RULE"* and *"is not a rule"* as fixed
#: text and so reworded this sentence -- which plan §2.3 does not license: the
#: four A39 messages are the only CORRECTED ones and every other check this
#: plan touches is a MOVE that keeps its words.
_A28_GLS_CLAUSES: dict[str, str] = {
    "wants": "solves for the covariance a PREDICTION-DEPENDENT sigma implies",
    "reads": "a model",
    "because": "has no fixed point to iterate",
    "instead": ("Declare inference.noise.kind: radiometer to iterate the "
                "rule, or run kind: conjugate.wiener, which is what a "
                "decided sigma wants."),
}

#: The same two clauses for the OTHER exit that reaches :func:`_gls_result`,
#: and it is not the same advice.  Measured at this task: ``kind:
#: conjugate.gcr`` under ``noise_from: gls`` reaches ``_decided_model``
#: through this function, so a frozen sigma there earned
#: :data:`_A28_GLS_CLAUSES` -- a gcr user told to *"run kind:
#: conjugate.wiener"* when ``noise_from: declared`` is one key and, measured,
#: RUNS on exactly that document.  That is the defect this plan's Task 10
#: exists to fix, on the sibling route neither §6 nor the task's own table
#: noticed; ``_gls_result`` now takes ``clauses`` keyword-only and REQUIRED
#: for the same reason ``_decided_model`` does -- a third caller has no
#: default left to inherit.
#:
#: The last clause is not decoration.  Check A27's own gcr sentence offers
#: ``noise_from: gls`` OR ``inference.noise.kind: radiometer_frozen``, so a
#: user who takes both lands precisely here, and a refusal that did not say
#: so would be a fix clause pointing at what its sibling check refuses.
#:
#: ``reads`` and ``because`` are ``_A28_GLS_CLAUSES``' -- this route INHERITED
#: that whole sentence at ``be2027b``, so its two shared fragments are a move
#: and only the two clauses above are the fix.
_A28_GCR_CLAUSES: dict[str, str] = {
    "wants": ("under noise_from: gls runs iterative_gls first and draws at "
              "the covariance it converges to"),
    "reads": "a model",
    "because": "has no fixed point to iterate",
    "instead": ("Drop noise_from: gls: the declared route draws at that "
                "array directly, which is what a frozen sigma is for -- and "
                "noise_from: gls is check A27's answer for "
                "inference.noise.kind: radiometer, so declaring both asks a "
                "reweighting to find a fixed point in a number that is "
                "already fixed."),
}


def _gls_result(run: Any, built: Any, *, block: Any, observed: Any,
                prior: dict, solve: dict, where: str,
                clauses: dict[str, str]) -> Any:
    """``iterative_gls`` at this document's noise model -> a ``GLSResult``.

    Shared by ``conjugate.gcr``'s ``noise_from: gls`` and by
    :func:`_run_gls`, ``conjugate.gls``'s own executor: whichever runs, the
    covariance is found the same
    way, the knobs are coerced the same way, and the same acknowledgement is
    demanded of a fixed point that was never reached.  One condition, one
    message, whichever exit reached it.

    ``clauses`` is the ONE thing the two callers must not share, and it is
    REQUIRED and keyword-only for that reason.  This function is where a
    ``conjugate.gcr`` run reaches :func:`_decided_model` (A28), and until
    Plan 3A it handed that accessor ``conjugate.gls``'s sentence -- so a gcr
    document with a frozen sigma was told to *"run kind: conjugate.wiener"*
    when ``noise_from: declared`` is one key away and, measured, runs.  The
    two mappings are :data:`_A28_GLS_CLAUSES` and :data:`_A28_GCR_CLAUSES`.

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

    The ``{delta:.4g}`` in the refusal below is **pinned by two test
    modules** -- ``test_config_exits_gcr.py`` on the draw's route and
    ``test_config_exits_gls.py`` on ``conjugate.gls``' own -- and is therefore
    API: at ``:.3g`` both pins fail, so a reword is a cross-module change
    rather than a local one.
    """
    from rheplicant.inference import iterative_gls

    acknowledged = run.options.get("acknowledge_unconverged_covariance", False)
    if not isinstance(acknowledged, bool):
        raise ConfigError(
            f"{where}: acknowledge_unconverged_covariance: is a bool; got "
            f"{acknowledged!r}."
        )
    found = iterative_gls(block, observed,
                          noise=_decided_model(run, built, **clauses),
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


def _a29_gcr_needs_a_seed(where: str, options: Mapping[str, Any]) -> None:
    """``conjugate.gcr`` draws, so it needs a seed; its two siblings refuse one.

    Module-level and taking plain data for plan §2.2's reason: the pre-flight
    pass (``preflight/fitting.py``) calls THIS function from the raw document,
    so the sentence a user reads before the beam and the one the executor
    raises are the same object rather than two copies.

    PRESENCE only.  The FORM of the declaration -- a literal, or a name
    outside ``runtime.seeds.`` -- is ``draws._seed_name``'s, which
    :func:`_gcr_product` reaches at ``:503`` and which the pass calls
    separately.  Splitting them that way is what lets a seedless run be
    described once, in this voice, rather than twice in two.
    """
    if "seed" not in options:
        raise ConfigError(
            f"{where}: conjugate.gcr draws from the posterior, so seed: is "
            "required and has no default -- gcr_sample's key= has none either "
            "(check A29). The deterministic conjugate exits, conjugate.wiener "
            "and conjugate.gls, refuse one instead."
        )


def _gcr_plan(run: Any, *, where: str) -> dict:
    """``conjugate.gcr``'s own grammar, decided before anything is built.

    Every check here is a dict lookup and a type test, and every one of them
    runs BEFORE :func:`_conjugate_block` -- which builds the operator and runs
    ``check_linearity``.  A document that is wrong in the grammar AND wrong in
    the model must hear about the grammar: the alternative is that a missing
    ``seed:`` surfaces as a ``ParameterSpaceError`` from inside the package,
    which is the seam this layer exists to close.
    """
    _a29_gcr_needs_a_seed(where, run.options)
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
                            prior=prior, solve=solve, where=where,
                            clauses=_A28_GCR_CLAUSES)
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


@register("conjugate.gls")
def _run_gls(run: RunSpec, built: Any, *, results: Any = None) -> Any:
    """``kind: conjugate.gls`` -> a GLSResult whose diagnostics are scalars.

    A separate executor rather than a third ``@register`` on
    :func:`_run_conjugate` (plan section 3.1): the sigma spelling (``noise=``
    against ``noise_std=``), the product (a ``GLSResult`` against a 2-tuple)
    and the convergence gate share nothing with that body.  What it DOES share
    -- the block, the prior, the CG knobs, the reweighted solve and its
    refusal -- it shares by calling the same functions, so
    ``conjugate.gcr``'s ``noise_from: gls`` route and this exit find the same
    covariance the same way and refuse the same non-fixed-point in the same
    words.

    :func:`_conjugate_block` hands back ``sigma`` as None here: this exit
    takes the noise RULE, which :func:`_gls_result` fetches through
    ``_decided_model`` (check A28).  Resolving a decided sigma instead would
    fire check A27 on exactly the document this exit exists to serve, which is
    why ``conjugate.gls`` is not in ``_DECIDES_SIGMA_HERE``.

    **No staleness check, unlike** :func:`_gcr_plan`'s.  That one exists
    because ``noise_from: declared`` accepts the reweight knobs in its sweep
    and then runs no loop to consult them; here every member of
    :data:`_GLS_KEYS` is read on every run -- the loop is what this exit IS --
    so there is no branch on which a declared key reaches nothing.
    """
    where = f"runs[{run.name!r}]"
    _sweep(run, _GLS_KEYS)
    # `sigma` is None here by design (see above) and is dropped rather than
    # bound: the one thing it must never do is reach a solve.
    block, _, observed = _conjugate_block(run, built, where)
    # `solve` is compiled HERE and passed in; _gls_result coerces the reweight
    # knobs itself and must not be handed those a second time (TypeError: got
    # multiple values).  Everything after `built` is keyword-only there, and
    # deliberately: every parameter is typed Any, so a positional call written
    # by analogy would bind `where` to `block` in silence and break only
    # inside a refusal branch a passing test need never reach.
    found = _gls_result(run, built, block=block, observed=observed,
                        prior=_prior_kwargs(run, built, block, where),
                        solve=_knobs(run, _SOLVER_KNOBS), where=where,
                        clauses=_A28_GLS_CLAUSES)
    # iterations/delta/converged are jax.Arrays on the way out (gls.py:97-99);
    # _gls_record casts all three, so neither a report nor diagnostics.json
    # ever sees a traced value -- examples/gls_gcr.py:150-152 is the idiom.
    # as_dict is the idempotent wrap (linear.py:184), here for the same reason
    # as in _wiener_product and just as unobservable: over a GROUPED block --
    # the only one this layer compiles -- the solution already IS the mapping,
    # so no test can distinguish this from `found.solution` (measured: the
    # whole suite is green with it removed).
    return found._replace(solution=block.as_dict(found.solution),
                          **_gls_record(found))


@register("condition")
def _run_condition(run: RunSpec, built: Any, *, results: Any = None) -> Any:
    """kappa for the block a conjugate exit would solve -- and no data at all.

    ``condition_estimate`` (linear.py:1337) takes no ``observed`` and no
    ``prior_mean``, and never calls ``_check_solve_arguments``: a block and a
    decided sigma are the whole input, which is why :func:`_conjugate_block`
    runs with ``needs_observed=False`` -- this is its one caller that does.  It
    DOES run both noise refusals and ``_require_prior_std``, so the A27 gate
    reaches it through :func:`_conjugate_block` and A51's mapping rule through
    :func:`_one_prior`.

    :func:`_prior_kwargs` is deliberately NOT called -- as defence in depth,
    and the depth is worth stating honestly.  It emits ``prior_mean`` whenever
    the document declares one, and ``condition_estimate`` has no such
    parameter; but the refusal above runs BEFORE the sweep and before the
    block, so on this route ``_prior_kwargs`` could never see a ``prior_mean``
    to emit.  Measured: splatting it here instead leaves the whole suite
    green, and no document can tell the two apart.  What shaping
    ``prior_std`` alone through :func:`_one_prior` buys is independence from
    that ordering -- if a later task relaxes the centre refusal, this exit
    still cannot hand ``condition_estimate`` a keyword it does not take.

    Its ``key`` is ``jax.Array | None = None``, defaulting internally to
    ``jax.random.key(0)`` -- the opposite of ``gcr_sample``'s required
    keyword-only key.  So ``seed:`` is OPTIONAL here and A29 does not make it
    required: this is the one exit where that asymmetry runs the other way,
    and an unseeded run is reproducible rather than arbitrary.

    Returns the package's own scalar array, not a tuple and not a dict.
    """
    import jax

    from rheplicant.config.draws import _seed_name, seed_for
    from rheplicant.inference import condition_estimate

    where = f"runs[{run.name!r}]"
    # BEFORE the sweep on purpose: the sweep would fire first with the generic
    # "does not take ['prior_mean']" and the reader would fix the symptom by
    # deleting a key they had good reason to write.
    if "prior_mean" in run.options:
        raise ConfigError(
            f"{where}: kind: condition does not take prior_mean: -- kappa is "
            "the conditioning of A^T N^-1 A + S^-1, which the prior's WIDTH "
            "sets and its centre does not, and condition_estimate has no "
            "prior_mean parameter to pass it to. Drop it; prior_std: is the "
            "knob that moves kappa. To see what the centre does, solve: "
            "kind: conjugate.wiener takes prior_mean."
        )
    _sweep(run, _CONDITION_KEYS)
    # `observed` is None here and is dropped rather than bound: this exit has
    # no parameter to pass it to, and `on:` therefore decides nothing.
    block, sigma, _ = _conjugate_block(run, built, where,
                                       needs_observed=False)
    # iterations travels through the family's one coercion path, so
    # `iterations: "twelve"` is refused in the same words `maxiter: "many"` is;
    # a knob the document omits is omitted from the call, and POWER_ITERATIONS
    # (12) stands as the package's own default.
    kwargs: dict[str, Any] = _knobs(run, _CONDITION_KNOBS)
    if "prior_std" in run.options:
        kwargs["prior_std"] = _one_prior(run, where, "prior_std",
                                         run.options["prior_std"], block,
                                         _space(run, built))
    if "seed" in run.options:
        # The NAME is resolved from the run and looked up in built.context, as
        # _gcr_product does it: seed_for is what makes the integer the estimate
        # consumes the same integer provenance.json reports.
        kwargs["key"] = jax.random.key(
            seed_for(_seed_name(dict(run.options), where), built.context))
    return condition_estimate(block, noise_std=sigma, **kwargs)
