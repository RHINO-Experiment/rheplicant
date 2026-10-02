"""The solves: Wiener, conditioned, and the GCR sample.

Everything that takes a block and priors and returns numbers, plus the
argument checking they share and the far-domain conversion they go through.
This is the third job the ruling named, and the one whose tests are numeric
rather than structural.
"""

from typing import Any

import jax
import jax.numpy as jnp
from bayesmith.exact.solve import condition_bound as _far_condition_bound
from bayesmith.exact.solve import condition_estimate as _far_condition_estimate
from bayesmith.exact.solve import gcr_sample as _far_gcr_sample
from bayesmith.exact.solve import wiener_solve as _far_wiener_solve

from rheplicant.core.conditioning import POWER_ITERATIONS
from rheplicant.core.errors import ParameterSpaceError
from rheplicant.inference.likelihood import check_observed_shape
from rheplicant.inference.noise import check_noise_std_axis

from .linear_block import (
    LinearBlock,
    _domain_centre,
)
from .linear_priors import (
    _refuse_a_noise_model_at_the_conjugate_seam,
    _require_prior_std,
    _resolve_prior,
)

#: The observed node this adapter presents to bayesmith. Upstream a
#:  :class:`~rheplicant.inference.linear_block.LinearBlock` predicts ONE array and the data arrives
#: as an argument
#: to the solve; downstream a block is keyed by observed-node name, because
#: there a block is cut out of a graph that may carry several. One name is
#: therefore enough, and it never reaches a caller: every public exit in this
#: module takes and returns the upstream spelling.
_OBSERVED = "observed"


def _as_far_block(
    block: LinearBlock,
    *,
    observed: jax.Array | None,
    prior_mean: Any,
    prior_std: Any,
) -> Any:
    """This block, plus the solve's own arguments, as bayesmith's ``LinearBlock``.

    The two dataclasses hold the same linear algebra in different shapes, and
    every difference is about WHERE something lives rather than what it is:

    ==================  ===========================  =========================
    ..                  here                         there
    ==================  ===========================  =========================
    the latents         ``name: str | tuple``        ``names: tuple``
    the domain          ``shape``/``dtype``, or dict ``{name: ...}`` always
    the prediction      ``offset: Array``            ``offset: {node: Array}``
    the data            an ARGUMENT to the solve     ``data`` on the block
    the prior           ``prior``, plus keywords     ``prior_mean``/``prior_std``
    ==================  ===========================  =========================

    So a group converts by relabelling and a single-latent block converts by
    wrapping. **No number changes**: ``probe_17_linear_solve_seam.py`` §0 pins
    the posterior mean bit-identical across this conversion, which is the
    precondition that makes the refusal comparisons in the rest of that probe
    mean anything -- without it they would be comparing two different models.

    That the data and the prior are ARGUMENTS here and FIELDS there is also
    the whole reason this module keeps its refusals rather than delegating
    them. A refusal that reads ``noise_std`` has nothing to read once the
    sigma has become a :class:`~bayesmith.exact.precision.Precision`, so it
    has to fire before the conversion or not at all -- D48's rule, and
    ``check_noise_std_axis`` is the case where not-at-all costs an answer
    rather than a message: measured, the far side accepts an ambiguous
    ``(n,)`` sigma against an ``(n, n)`` prediction and returns a finite,
    correctly shaped array that differs from the intended reading by 2.5e-03.

    Args:
        block: the upstream block.
        observed: the data. ``None`` for :func:`condition_bound` and
            :func:`condition_estimate`, which have none to pass --
            conditioning is a property of ``AᵀN⁻¹A + S⁻¹`` alone. They get a
            zero of the prediction's shape, which the far side never reads.
        prior_mean: already resolved against the declaration; ``None``, or
            ``None`` per member, means a zero-centred prior.
        prior_std: already resolved, and already checked non-``None`` by
            :func:`_require_prior_std`.

    Returns:
        A :class:`bayesmith.exact.block.LinearBlock` over the same domain.
    """
    from bayesmith.exact.block import LinearBlock as _FarBlock

    # `_domain_centre` takes a group's centre per member, and its only previous
    # caller was `_conjugate_solve`, which always received `prior_mean` already
    # normalized by `_resolve_prior`. `condition_bound` and `condition_estimate`
    # have no prior mean to resolve -- conditioning does not depend on one --
    # so the bare `None` is spread over the members here rather than being
    # allowed to reach a subscript.
    if prior_mean is None and block.grouped:
        prior_mean = dict.fromkeys(block.names)
    centre = _domain_centre(block, prior_mean)
    data = jnp.zeros_like(block.offset) if observed is None else observed

    if block.grouped:
        names = block.names
        shape = {member: block.shape[member] for member in names}
        dtype = {member: block.dtype[member] for member in names}
        std = {member: prior_std[member] for member in names}
        mean = {member: centre[member] for member in names}

        def forward(domain: dict[str, Any]) -> dict[str, jax.Array]:
            return {_OBSERVED: block.forward(domain)}

        def adjoint(codomain: dict[str, jax.Array]) -> dict[str, Any]:
            return dict(block.adjoint(codomain[_OBSERVED]))

    else:
        name = block.name
        names = (name,)
        shape = {name: block.shape}
        dtype = {name: block.dtype}
        std = {name: prior_std}
        mean = {name: centre}

        def forward(domain: dict[str, Any]) -> dict[str, jax.Array]:
            return {_OBSERVED: block.forward(domain[name])}

        def adjoint(codomain: dict[str, jax.Array]) -> dict[str, Any]:
            return {name: block.adjoint(codomain[_OBSERVED])}

    return _FarBlock(
        names=names,
        shape=shape,
        dtype=dtype,
        offset={_OBSERVED: block.offset},
        forward=forward,
        adjoint=adjoint,
        data={_OBSERVED: data},
        prior_mean=mean,
        prior_std=std,
    )


def _from_far_domain(block: LinearBlock, solution: dict[str, Any]) -> Any:
    """The far side's ``{name: array}`` back in this block's own spelling.

    A ``name=`` block's answer is a bare array and a ``names=`` group's is a
    dict, and the two are not interchangeable --
    :meth:`~rheplicant.inference.linear_block.LinearBlock.as_dict`
    exists because six downstream consumers index by latent name and all six
    raise on the bare form. The far side only has the dict, so this is where
    the distinction is restored.
    """
    if block.grouped:
        return {member: solution[member] for member in block.names}
    return solution[block.name]


def _far_precision(noise_std: Any) -> Any:
    """``sigma`` as the far side's ``{node: Precision}``.

    ``diagonal_from`` is bayesmith's own bridge from a decided sigma, so the
    weight this produces is ``1 / sigma**2`` computed by the same code the
    correlated path uses. It is NOT
    :func:`~rheplicant.inference.noise.inverse_variance`, and the difference
    is the one :func:`_check_solve_arguments` documents at length: on a NaN
    sigma this propagates, where ``inverse_variance`` would map it to weight
    zero and silently drop the sample.
    """
    from bayesmith.exact.precision import diagonal_from

    return diagonal_from({_OBSERVED: jnp.asarray(noise_std)})


def _check_solve_arguments(
    block: LinearBlock,
    observed: jax.Array,
    prior_mean: Any,
    prior_std: Any,
    caller: str,
    *,
    noise_std: Any = None,
) -> tuple[Any, Any]:
    """Shared preconditions for the mean and the draw, plus the resolved prior.

    Returns the ``(prior_mean, prior_std)`` the solve should actually use: the
    keywords when they were given, the latent's declaration when they were not,
    and an exception when the two disagree.

    ``noise_std`` is checked here for the axis contract
    (:func:`~rheplicant.inference.noise.check_noise_std_axis`). Both exits pass
    it — :func:`wiener_solve` and :func:`gcr_sample` — and
    ``tests/inference/test_noise_std_axis.py`` asserts they refuse the same
    inputs, because a rule enforced on the mean and not on the draw is worse
    than no rule: it teaches that the argument is checked.

    **Two homes, and it stays that way.** Every other exit reaches that rule
    through :func:`~rheplicant.inference.uncertainty.as_noise_model`, the one
    place a ``noise_std`` argument is normalized. This module does not call it:
    :func:`_conjugate_solve` and :func:`condition_estimate` take the bare array
    to ``1 / sigma**2`` directly. Routing them through ``as_noise_model`` would
    leave one home, and it was assessed and rejected for two measured reasons.

    **The weight formulas disagree on NaN, in the dangerous direction.**
    ``1 / sigma**2`` and
    :func:`~rheplicant.inference.noise.inverse_variance` agree on every finite
    sigma, on ``inf`` (both give exactly ``0``), on ``0`` and on a negative
    sigma. They differ on ``nan``: this module propagates it, so the solution
    comes back NaN and the caller knows; ``inverse_variance`` maps it to weight
    ``0.0``, which *means* "unobserved". Switching would turn "your sigma array
    has a NaN in it" from a loud failure into a silently dropped sample — and
    in a conjugate solve a dropped sample moves the posterior WIDTH, not only
    the point, with nothing reporting how many went.

    **A conjugate solve has no prediction to give it.**
    ``inverse_variance(noise, prediction)`` needs one, and for a
    ``depends_on_prediction`` model it genuinely matters: a
    :class:`~rheplicant.inference.noise.RadiometerNoise`'s weights move 9x
    between a 100 K and a 300 K prediction. But the prediction is what the
    solve is *for*. So the solves would have to freeze ``N`` at some arbitrary
    point, and :mod:`rheplicant.inference.plan` already documents what freezing
    costs: an exact draw from a linear-Gaussian conditional *at that
    covariance*, which is not the full model's conditional. Accepting a
    ``NoiseModel`` at this seam would invite precisely that mistake by making
    it type-check.

    The duplication is therefore deliberate rather than owed.
    ``tests/inference/test_noise_std_axis.py::TestWhyTheRuleHasTwoHomes`` pins
    both measurements, so an author who unifies them anyway meets the two
    consequences rather than rediscovering them. The keyword stays optional so
    an internal caller that has already normalized need not pay for it twice.
    """
    check_observed_shape(jnp.shape(block.offset), observed)
    if noise_std is not None:
        check_noise_std_axis(noise_std, jnp.shape(block.offset), caller)
        _refuse_a_noise_model_at_the_conjugate_seam(noise_std, caller)
    prior_mean, prior_std = _resolve_prior(block, prior_mean, prior_std, caller)
    _require_prior_std(block, prior_std, caller)
    if jnp.issubdtype(jnp.asarray(block.offset).dtype, jnp.complexfloating):
        raise ParameterSpaceError(
            f"{caller} expects a real-valued prediction; this block's offset is complex."
        )
    return prior_mean, prior_std


def wiener_solve(
    block: LinearBlock,
    observed: jax.Array,
    *,
    noise_std: Any,
    prior_std: Any = None,
    prior_mean: Any = None,
    tol: float = 1e-6,
    maxiter: int | None = None,
    require_convergence: float | None = None,
) -> tuple[Any, jax.Array]:
    """Posterior mean of a linear-Gaussian block — the Wiener filter, by CG.

    With ``d = A x + offset + n``, ``n ~ N(0, N)`` and ``x ~ N(m, S)``::

        x̂ = (AᵀN⁻¹A + S⁻¹)⁻¹ [AᵀN⁻¹ (d - offset) + S⁻¹m]

    solved with conjugate gradients, so the normal operator is only ever
    *applied*, never formed. Each iteration costs one JVP and one VJP through
    the forward model — which is why a block with 10⁶ degrees of freedom is
    tractable at all.

    The normal operator and the right-hand side are both obtained as gradients
    of the objective itself rather than assembled from ``A`` and ``Aᵀ`` by
    hand. That is not a shortcut: it makes the operator symmetric positive
    definite *by construction* over the real degrees of freedom, with no
    adjoint-convention arithmetic left to get wrong for complex latents.

    This is the posterior **mean**, not a sample. For a draw, see
    :func:`gcr_sample`, which adds a fluctuation term to this same right-hand
    side and costs exactly the same solve.

    Args:
        block: from :func:`~rheplicant.inference.linear.linear_operator`.
        observed: the data, shaped like ``block.offset``.
        noise_std: noise standard deviation — a scalar, or an array whose
            SHAPE says which axis of the data it runs along: ``(n_time, 1)``
            for a per-time sigma, ``(1, n_freq)`` for a per-channel one. A bare
            1-D vector is accepted only where its length matches a single axis
            of the data; on a square grid it matches two, both readings are
            legitimate, and the one broadcasting picks is not the one most
            callers mean — so that case raises rather than being resolved by
            trailing-axis alignment. See
            :func:`~rheplicant.inference.noise.check_noise_std_axis`.

            **An array, never a** :class:`~rheplicant.inference.noise.NoiseModel`,
            and the keyword name is the signal rather than an accident of
            history. ``noise_std=`` is a sigma that has already been decided;
            ``noise=`` — on :func:`~rheplicant.inference.gls.iterative_gls` and
            on :class:`~rheplicant.inference.plan.SamplingPlan` — is the *rule*
            that decides one. A conjugate solve has no prediction to evaluate a
            rule at, the prediction being what it solves for, so a model is
            refused here by name rather than quietly frozen at some arbitrary
            point: see ``_refuse_a_noise_model_at_the_conjugate_seam`` for
            the message and ``_check_solve_arguments`` for the two measured
            reasons this seam is not routed through ``as_noise_model``. Freeze
            it yourself — ``noise.std(prediction)`` — and pass that array.
        prior_std: prior standard deviation on the latent — scalar or
            broadcastable to it. **Defaults to the latent's declared prior**;
            required only when there is none, because without a prior the
            normal operator can be singular and CG would return a finite,
            arbitrary answer instead of complaining. Passing a value that
            contradicts the declaration raises rather than one silently
            winning — see the note below.
        prior_mean: centre of the prior. Defaults to the declared prior's
            location, and to zero when nothing is declared — which is wrong for
            most physical quantities, a noise-wave temperature sitting near
            250 K. Equivalent to an affine binding that adds the same offset,
            but says what it means.
        tol: CG tolerance — a bound on the relative RESIDUAL, which is not the
            same as accuracy. See the note on conditioning below.
        maxiter: CG iteration cap. ``None`` lets JAX choose.
        require_convergence: raise unless the relative ERROR can be bounded by
            this. Off by default (``None``), which returns whatever CG
            produced; pass a target such as ``1e-3`` to turn the guard on.
            jax's ``cg`` reports no convergence status, so without the guard
            an unconverged solve comes back looking like a converged one. The
            Note below says why it is off by default.

            The bound is ``κ · relative_residual``, with ``κ`` bounded by
            :func:`condition_bound`. Guarding on the residual alone would
            certify nothing in the regime that matters — see below — so this
            costs ``POWER_ITERATIONS`` extra operator applications. That is not
            free: on a well-conditioned block, where CG itself converges in a
            few iterations, it is a real fraction of the solve. In a Gibbs sweep,
            where the conditioning barely moves from sweep to sweep, call
            :func:`condition_bound` once outside the loop, choose ``tol``
            from it, and leave the guard off inside — the same
            bargain :func:`~rheplicant.inference.linear.linear_operator`'s ``check`` offers.

    Returns:
        ``(x̂, relative_residual)``, the residual being ``‖M x̂ - b‖ / ‖b‖``
        over the real degrees of freedom. Note that this is the residual, not
        the error; multiply by :func:`condition_estimate` for the error bound.

        ``x̂`` is the block's own domain, so its shape follows the spelling that
        built the block: a ``{name: array}`` dict for ``names=``, and a **bare
        array** for ``name=``. The bare form is not what anything downstream
        reads — ``space.forward_fn``'s ``forward``,
        :meth:`~rheplicant.inference.parameters.ParameterSpace.bind`,
        :func:`~rheplicant.inference.uncertainty.fisher_information`,
        :func:`~rheplicant.inference.identifiability.identifiability`'s ``at=``,
        :func:`~rheplicant.inference.linear.linear_operator`'s own ``at=`` and
        :func:`~rheplicant.inference.engines.conditional_potential` all index by
        latent name and all six raise on it. Wrap it as ``{block.name: x̂}``
        first; :meth:`~rheplicant.inference.linear_block.LinearBlock.as_dict` is that call, and does
        nothing to the
        grouped form, so it is correct either way.

    Note:
        **Conditioning, and why ``tol`` is not accuracy.** Residual and error
        differ by the condition number of ``M = AᵀN⁻¹A + S⁻¹``::

            ‖x̂ - x*‖ / ‖x*‖  ≤  κ(M) · ‖M x̂ - b‖ / ‖b‖

        For a block the data does not fully identify — one calibration load
        against three unknowns, a flagged channel, a short integration — the
        prior is the only thing holding the blind directions down, so
        ``λ_min(M)`` is exactly ``1/prior_std²`` and ``κ ≈ ‖AᵀN⁻¹A‖ · prior_std²``
        runs to 1e6 and beyond. At κ=1e7 the default ``tol=1e-6`` bounds the
        relative error by 10: no digits at all. CG stops on a residual that
        looks converged, having left the prior-dominated directions at their
        starting value, and the draw comes back with far too little scatter.

        This is exactly the regime these solvers exist for, so the accuracy
        target is stated as an error and not a residual. To solve rather than
        refuse, pass ``tol ≈ require_convergence / κ`` with a ``maxiter`` to
        match. Past ``κ · eps`` no tolerance helps and only precision does; the
        guard says so in its own words.

        **It is OFF by default, and that is a recent, deliberate retreat.** The
        guard shipped on, against a κ that ``_condition_estimate`` has since
        been shown to under-report by up to a factor of 700 — so what was on by
        default was a promise to bound the error that did not bound it. The κ
        here is now a rigorous UPPER bound, and the same measurement that made
        it sound made it conservative: on a block the data DOES identify in
        every direction it can read five orders of magnitude high (1.44e+06
        measured against a true κ under 10, because λ_min is then set by the
        data and not by the prior, which is all the bound knows about). On by
        default, that refuses correct solves wholesale. So the choice is
        yours to make per solve, and when you make it the answer means
        something: a solve this guard passes has its error bounded, and one it
        refuses may still be fine — the bound could not prove it.

    Note:
        **Where S comes from.** ``Latent(prior=dist.Normal(m, s))`` is the
        package's one statement of what a latent is a priori, and it is the
        statement ``to_numpyro_model`` reads. So it is the statement this solve
        reads too: declare it once and both exits target the same posterior.
        The keywords remain, for a prior-free latent and for overriding a
        declaration you are deliberately solving away from — but a keyword that
        *contradicts* a declaration raises, because the alternative is one of
        the two silently winning and the two exits quietly disagreeing. A
        declared prior with no conjugate Gaussian form (a Half-Normal, a
        Uniform) raises here as well; NUTS is where that space belongs.
    """
    prior_mean, prior_std = _check_solve_arguments(
        block, observed, prior_mean, prior_std, "wiener_solve", noise_std=noise_std
    )
    solution, residual = _far_wiener_solve(
        _as_far_block(block, observed=observed, prior_mean=prior_mean, prior_std=prior_std),
        precision=_far_precision(noise_std),
        tol=tol,
        maxiter=maxiter,
        require_convergence=require_convergence,
    )
    return _from_far_domain(block, solution), residual


def condition_bound(
    block: LinearBlock,
    *,
    noise_std: Any,
    prior_std: Any = None,
    iterations: int = POWER_ITERATIONS,
    key: jax.Array | None = None,
) -> jax.Array:
    """An UPPER BOUND on the conditioning of the system this block is solved with.

    ``κ(AᵀN⁻¹A + S⁻¹)`` says how much a solver's residual understates its
    error: for a solution ``x`` with relative residual ``r``,

        ‖x - x*‖ / ‖x*‖  ≤  κ · r

    so a residual of 1e-6 against κ=1e7 certifies nothing. **This is the number
    to divide an accuracy target by** — for a target relative accuracy ``a``,
    ask :func:`wiener_solve` or :func:`gcr_sample` for roughly ``tol = a /
    condition_bound(...)`` — and it is the number ``require_convergence``
    itself reads. :func:`condition_estimate` measures κ instead and is biased
    LOW; a tolerance chosen from it is too loose by that bias.

    ``λ_max · max(prior_variance)``. ``AᵀN⁻¹A`` is positive semi-definite, so
    ``λ_min ≥ 1/max(prior_variance)`` exactly. ``λ_max`` is approached from
    BELOW and geometrically, so the estimate can only make the bound smaller.

    A large bound is not a defect; it is what the bound is entitled to say. On
    a block whose data constrains every direction, λ_min is set by the data
    rather than by the prior and this reads five decades high — measured, 1.44e+06
    against a true κ under 10. It costs iterations, not correctness.

    Costs ``iterations`` applications of the normal operator, half what
    :func:`condition_estimate` costs, and forms no matrix.

    Args:
        block: from :func:`~rheplicant.inference.linear.linear_operator`.
        noise_std: as for :func:`condition_estimate`, with the same refusals.
        prior_std: as for :func:`condition_estimate`.
        iterations: power-iteration steps for ``λ_max``.
        key: PRNG key for the starting vector. Fixed by default.

    Returns:
        The bound, as a scalar array.
    """
    check_noise_std_axis(noise_std, jnp.shape(block.offset), "condition_bound")
    _refuse_a_noise_model_at_the_conjugate_seam(noise_std, "condition_bound")
    _, prior_std = _resolve_prior(block, None, prior_std, "condition_bound")
    _require_prior_std(block, prior_std, "condition_bound")
    return _far_condition_bound(
        _as_far_block(block, observed=None, prior_mean=None, prior_std=prior_std),
        precision=_far_precision(noise_std),
        iterations=iterations,
        key=key,
    )


def condition_estimate(
    block: LinearBlock,
    *,
    noise_std: Any,
    prior_std: Any = None,
    iterations: int = POWER_ITERATIONS,
    key: jax.Array | None = None,
) -> jax.Array:
    """The MEASURED conditioning of the system this block is solved with.

    ``κ(AᵀN⁻¹A + S⁻¹)`` is the number that says how much a solver's residual
    understates its error: for a solution ``x`` with relative residual ``r``,

        ‖x - x*‖ / ‖x*‖  ≤  κ · r

    so a residual of 1e-6 against κ=1e7 certifies nothing at all.

    **Do not divide an accuracy target by this number.**
    :func:`condition_bound` is the one to divide by, and it is what
    ``require_convergence`` itself reads. This one measures ``λ_min`` by a
    second power iteration, whose leading eigenvalues crowd against ``λ_max``
    with vanishing gaps on a graded spectrum, so the ``λ_min`` it returns is
    too LARGE and this κ too SMALL — measured on a 20-point geometric
    spectrum at a true κ of 1e4, λ_min came back 33.9× high and κ 33.9× low;
    at 1e7 over 50 points the factor was ~700 and 2000 iterations did not
    close it. A ``tol`` computed from it is too LOOSE by that factor, which is
    the direction that certifies an answer it should have refused. See
    ``_condition_estimate`` for where those numbers came from.

    **What it is good for is the thing a bound cannot do: it can SEE a
    degeneracy.** A near-degenerate partition shows up entirely in ``λ_min``,
    which the bound replaces with the prior's floor and therefore cannot
    report. Measured in
    ``tests/inference/test_linear_groups.py::TestGroupedVsAlternating``: the
    joint operator's κ exceeds its members' by orders of magnitude here, and
    by a factor of 1.7 under the bound. Read it as a diagnostic — "how badly
    conditioned is this partition?" — and never as a certificate.

    Large κ is not a defect here, it is the design: for a block the data does
    not fully identify, ``λ_min`` is exactly ``1/prior_std²`` while ``λ_max``
    is set by the data, so κ grows with how much better the data constrains
    one direction than the prior constrains another.

    Costs ``2 · iterations`` applications of the normal operator — each the
    same JVP-plus-VJP a CG iteration costs — and no matrix is ever formed.
    :func:`condition_bound` costs half that, measuring only the top.

    Args:
        block: from :func:`~rheplicant.inference.linear.linear_operator`.
        noise_std: the same decided sigma array those solves take, and a
            :class:`~rheplicant.inference.noise.NoiseModel` is as wrong here as
            it is there — a κ is the conditioning of one particular normal
            operator, so it needs the covariance settled, not a rule for
            producing one. Both refusals the solves apply run here too: a model
            is refused by name, and a 1-D sigma whose axis the prediction
            cannot settle is refused by
            :func:`~rheplicant.inference.noise.check_noise_std_axis`. They have
            to, because a diagnostic is only about the system it describes: a κ
            computed under a different reading of the same array answers a
            different question than the solve it was computed for, and would
            report the conditioning of an operator nobody builds.
        prior_std: as for :func:`wiener_solve` — it defaults to the latent's
            declared prior, so the κ reported here is the κ of the system those
            solves will build rather than of a system nobody solves.
        iterations: power-iteration steps per end of the spectrum. The
            default is comfortable; the estimate typically settles within
            three.
        key: PRNG key for the starting vectors. Fixed by default, so the
            estimate is reproducible.

    Returns:
        The measured condition number, as a scalar array.
    """
    check_noise_std_axis(noise_std, jnp.shape(block.offset), "condition_estimate")
    _refuse_a_noise_model_at_the_conjugate_seam(noise_std, "condition_estimate")
    _, prior_std = _resolve_prior(block, None, prior_std, "condition_estimate")
    _require_prior_std(block, prior_std, "condition_estimate")
    return _far_condition_estimate(
        _as_far_block(block, observed=None, prior_mean=None, prior_std=prior_std),
        precision=_far_precision(noise_std),
        iterations=iterations,
        key=key,
    )


def gcr_sample(
    block: LinearBlock,
    observed: jax.Array,
    *,
    noise_std: Any,
    prior_std: Any = None,
    key: jax.Array,
    prior_mean: Any = None,
    tol: float = 1e-6,
    maxiter: int | None = None,
    require_convergence: float | None = None,
) -> tuple[Any, jax.Array]:
    """Draw an EXACT posterior sample of a linear-Gaussian block.

    The constrained-realization (GCR) identity: solve the same system
    :func:`wiener_solve` does, but with two white-noise terms added to the
    right-hand side::

        (AᵀN⁻¹A + S⁻¹) x = AᵀN⁻¹(d - offset) + S⁻¹m + AᵀN⁻¹ᐟ² ω₁ + S⁻¹ᐟ² ω₂

    with ``ω₁``, ``ω₂`` standard normal on the data and on the latent. The
    right-hand side then has the posterior-mean numerator as its mean and
    covariance ``AᵀN⁻¹A + S⁻¹`` — the operator itself — so ``x = M⁻¹b`` has the posterior
    mean and covariance ``M⁻¹M M⁻¹ = M⁻¹`` exactly. Not an approximation and
    not a Markov chain: every call is an independent draw, with no burn-in and
    nothing to diagnose for convergence.

    This is what makes a 10⁶-dimensional block samplable at all. It costs one
    CG solve — the same as the mean — because the fluctuation enters the
    right-hand side, never the operator.

    In a Gibbs scheme, this draws the linear block conditional on the nonlinear
    parameters; rebuild the block with
    :func:`linear_operator(..., check=False)` each sweep, having checked the
    linearity claim once outside the loop. The conditioning guard is worth
    hoisting the same way: :func:`condition_estimate` once to fix ``tol``, then
    ``require_convergence=None`` in the loop. What you must NOT do is leave
    ``tol`` at its default and the guard off — that is the combination this
    module returned a silently over-confident posterior for.

    Args:
        block: from :func:`~rheplicant.inference.linear.linear_operator`.
        observed: the data, shaped like ``block.offset``.
        noise_std: noise standard deviation, exactly as for
            :func:`wiener_solve` — the same axis contract on a 1-D sigma
            (both exits share ``_check_solve_arguments``, so a shape one refuses
            the other refuses), and the same refusal of a
            :class:`~rheplicant.inference.noise.NoiseModel` at this seam. The
            keyword is the signal: ``noise_std=`` takes a decided sigma,
            ``noise=`` takes the rule that decides one, and a draw has no
            prediction to evaluate a rule at any more than the mean does.
        prior_std: prior standard deviation on the latent. Defaults to the
            latent's declared prior, as for :func:`wiener_solve`, and required
            only when there is none. For a complex latent this is the width of
            the real and imaginary parts independently.
        key: PRNG key. ``vmap`` over split keys for many independent draws.
        prior_mean: centre of the prior; defaults to the declared prior's
            location, and to zero when nothing is declared. With uninformative
            data the draws fall back to ``N(prior_mean, prior_std²)``, which is
            the check that it is wired in correctly.
        tol: CG tolerance — a bound on the residual, not on the accuracy.
        maxiter: CG iteration cap.
        require_convergence: as for :func:`wiener_solve`, including the
            conditioning note there, which a draw is MORE exposed to than the
            mean. The fluctuation term ``S⁻¹ᐟ²ω₂`` puts weight on every
            direction of the latent by construction, including the ones the
            data is blind to — so a draw always has something to resolve where
            the operator is worst conditioned, whereas the mean does only when
            ``prior_mean`` is nonzero.

    Returns:
        ``(x, relative_residual)``. An unconverged CG returns a draw from the
        WRONG distribution — and a distribution that is too NARROW, since the
        directions left unresolved are the prior-dominated ones that should
        have carried the most scatter. ``require_convergence`` is worth passing
        here more than anywhere; it is off by default for the reason
        :func:`wiener_solve` gives, which is about the bound's conservatism and
        not about this risk being small.

        ``x`` carries the block's domain, dict or bare array, exactly as
        :func:`wiener_solve`'s does; see the note there, and
        :meth:`~rheplicant.inference.linear_block.LinearBlock.as_dict` for the wrap.

    Note:
        ``S`` is read off ``Latent(prior=...)`` when the keywords are omitted;
        see the corresponding note on :func:`wiener_solve` for what that does
        and does not permit. It matters more here than for the mean: with a
        declared prior ignored, the fluctuation term ``S⁻¹ᐟ²ω₂`` is drawn at
        the wrong width, so every draw is wrong in the one direction the mean
        can be right in.
    """
    prior_mean, prior_std = _check_solve_arguments(
        block, observed, prior_mean, prior_std, "gcr_sample", noise_std=noise_std
    )
    solution, residual = _far_gcr_sample(
        _as_far_block(block, observed=observed, prior_mean=prior_mean, prior_std=prior_std),
        precision=_far_precision(noise_std),
        key=key,
        tol=tol,
        maxiter=maxiter,
        require_convergence=require_convergence,
    )
    return _from_far_domain(block, solution), residual
