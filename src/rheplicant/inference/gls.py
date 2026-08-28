"""Iteratively reweighted least squares: finding the covariance to solve at.

:func:`~rheplicant.inference.linear.gcr_sample` is a linear sampler *given* a
covariance, and :func:`~rheplicant.inference.linear.wiener_solve` is the
corresponding mean. Both take ``noise_std`` and neither cares where it came
from. Under :class:`~rheplicant.inference.noise.HomoscedasticNoise` it comes
from the caller and there is nothing more to say.

Under the default :class:`~rheplicant.inference.noise.RadiometerNoise` there
is. Sigma tracks the prediction::

    sigma_i = |A x + offset|_i / sqrt(delta_nu * tau)

so the weights depend on the solution and the solution depends on the weights.
Neither is available first. This module supplies the missing half — the
covariance — and changes nothing about the two solvers::

    found = iterative_gls(block, observed, noise=RadiometerNoise(dnu, tau),
                          prior_std=PRIOR)
    draw, _ = gcr_sample(block, observed, noise_std=found.noise_std,
                         prior_std=PRIOR, key=key)

The algorithm is a fixed-point iteration: solve with the current weights,
recompute the weights at the new prediction, repeat. It is the same
iteratively-reweighted GLS as hydra-tod's
``hydra_tod.linear_sampler.iterative_gls``, but **matrix-free** — hydra-tod
forms a dense design matrix ``U`` and a dense ``N_inv``, while here the same
algorithm runs on the :class:`~rheplicant.inference.linear.LinearBlock`'s JVP
and VJP, which is what makes a block with 10^6 degrees of freedom possible at
all.

**What this estimator is, and is not.** Freezing sigma inside each solve is
what makes each step a linear-Gaussian problem, and it is also what makes the
converged answer *generalized least squares* rather than the maximum of the
full Gaussian likelihood: the log-determinant's dependence on the solution is
held fixed rather than differentiated. The two differ, in a known direction —
see :mod:`rheplicant.inference.noise`. GLS is the right estimator to condition
a constrained realization on, because a GCR draw is exactly a draw from a
linear-Gaussian posterior at a *given* covariance; if you want the full
likelihood's mode or posterior, that is a job for a gradient sampler
(:mod:`rheplicant.inference.numpyro_bridge`), not for this.
"""

from typing import Any, NamedTuple

import jax
from bayesmith.exact.gls import iterative_gls as _far_iterative_gls

from rheplicant.core.errors import ParameterSpaceError
from rheplicant.inference.linear import (
    _OBSERVED,
    LinearBlock,
    _as_far_block,
    _check_solve_arguments,
    _from_far_domain,
)
from rheplicant.inference.noise import NoiseModel

#: Reweighting steps taken before the convergence test is consulted. Matches
#: hydra-tod's default: the first steps of a fixed-point iteration can be
#: nearly stationary without being near the fixed point.
MIN_REWEIGHTS: int = 5

#: Cap on reweighting steps, so a non-contracting problem terminates and says
#: so through ``converged`` rather than spinning.
MAX_REWEIGHTS: int = 100

#: Multiple of the working precision's epsilon used as the default
#: ``reweight_tol``. See :func:`iterative_gls` for why the default cannot be a
#: fixed number.
REWEIGHT_TOL_EPS: float = 8.0


class GLSResult(NamedTuple):
    """What a reweighting run produced.

    A NamedTuple, so it is a pytree and survives ``jit`` unchanged.

    * ``noise_std`` — the converged sigma: **the covariance**, and the whole
      point of the exercise. Feed it to
      :func:`~rheplicant.inference.linear.gcr_sample` or
      :func:`~rheplicant.inference.linear.wiener_solve` as ``noise_std=``.
    * ``solution`` — the GLS point estimate at that covariance, shaped like the
      latent.
    * ``residual`` — relative CG residual of the final solve. Not an accuracy;
      see :func:`~rheplicant.inference.linear.wiener_solve`.
    * ``iterations`` — reweighting steps taken, the first solve included.
    * ``delta`` — relative change of the last step, ``‖x_new - x‖ / ‖x_new‖``.
    * ``converged`` — whether ``delta`` fell below ``reweight_tol`` within
      ``max_reweights``. **False here means the returned covariance is not a
      fixed point**, and everything conditioned on it inherits that.
    """

    noise_std: jax.Array
    solution: Any
    residual: jax.Array
    iterations: jax.Array
    delta: jax.Array
    converged: jax.Array


def iterative_gls(
    block: LinearBlock,
    observed: jax.Array,
    *,
    noise: NoiseModel,
    prior_std: Any = None,
    prior_mean: Any = None,
    tol: float = 1e-6,
    maxiter: int | None = None,
    reweight_tol: float | None = None,
    min_reweights: int = MIN_REWEIGHTS,
    max_reweights: int = MAX_REWEIGHTS,
    require_convergence: float | None = None,
) -> GLSResult:
    """Find the covariance a prediction-dependent noise model implies.

    Repeats: solve at the current sigma, recompute sigma at the new prediction.
    When ``noise.depends_on_prediction`` is ``False`` there is nothing to
    repeat and this is a single :func:`wiener_solve`.

    Args:
        block: from :func:`~rheplicant.inference.linear.linear_operator`.
        observed: the data, shaped like ``block.offset``.
        noise: the noise model — supplies sigma at each prediction.
        prior_std: prior standard deviation on the latent. Defaults to the
            latent's declared prior, and required only when there is none, for
            the reason :func:`wiener_solve` requires it. Resolved once here and
            passed down explicitly, so every inner solve sees the same ``S``.
        prior_mean: centre of the prior; defaults to the declared prior's
            location, and to zero when nothing is declared.
        tol: CG tolerance for each inner solve.
        maxiter: CG iteration cap for each inner solve.
        reweight_tol: stop when the latent's relative change falls below this.
            **The default cannot be a fixed number**, because two independent
            floors bound how small a step is measurable at all, and it defaults
            to ``max(8 * eps, tol)``:

            * the arithmetic's own epsilon — a relative step below it is
              rounding, not a measurement. float32's is ``1.2e-7``, so a
              plausible-looking ``1e-8`` is exactly this trap;
            * **the inner solver's tolerance** ``tol`` — consecutive solves
              differ by roughly their own CG residual no matter what the outer
              iteration is doing, so a step smaller than ``tol`` measures CG,
              not the fixed point. This is the binding floor in float64, where
              a tight ``tol=1e-10`` sits five orders of magnitude above ``eps``.

            Ask for less than either and the run does not fail quietly: it
            spends ``max_reweights`` steps and reports ``converged=False`` for a
            fixed point it had in fact reached.
        min_reweights: steps taken before the test is consulted.
        max_reweights: cap on steps.
        require_convergence: bound on the relative error of the **final** solve,
            as for :func:`wiener_solve`. Deliberately applied once, at the
            converged covariance, and not inside the loop: the guard costs
            ``POWER_ITERATIONS`` extra operator applications, which is the
            same bargain :func:`wiener_solve`'s own docstring recommends for a
            Gibbs sweep. It bounds the error of what is returned; it says
            nothing about the intermediate steps, which do not need it.

    Returns:
        A :class:`GLSResult`. **Check ``converged``** — a covariance that is not
        a fixed point is still a number, and a draw conditioned on it is still
        a draw.

    Note:
        The iteration starts from sigma evaluated at the **data** rather than
        from hydra-tod's unweighted least squares. It is the natural first
        guess (the data is an estimate of the prediction), it costs one solve
        less, and it honours flags from the first step where unit weights would
        not. A fixed point does not depend on where the iteration started, so
        the two agree where either converges.

        Built on ``lax.while_loop``, so it is jittable but **not** reverse-mode
        differentiable. That is not the limitation it looks like: the result is
        a fixed point, so implicit differentiation — not unrolling — is the
        right way to take a gradient through it.
    """
    if not isinstance(noise, NoiseModel):
        # A bare sigma array used to reach `noise.depends_on_prediction` and
        # come back as `AttributeError: 'ArrayImpl' object has no attribute
        # 'depends_on_prediction'` -- an attribute name the caller never wrote,
        # from a layer they were not thinking about. This exit is the ONLY one
        # in the package that requires a model rather than accepting one: its
        # whole subject is the fixed point a prediction-dependent sigma
        # implies, and a constant sigma has no fixed point to find.
        #
        # It stays HERE rather than delegating, and for the same reason its
        # mirror in `linear` does: bayesmith takes a `sigma_of` CALLABLE and
        # has never heard of `NoiseModel`, so past the seam there is nothing
        # left to recognise. `linear` refuses a model where a decided sigma
        # belongs; this exit requires one where a rule belongs. Two opposite
        # refusals, one reason -- the evidence for both is erased by the same
        # conversion.
        raise ParameterSpaceError(
            f"iterative_gls needs a NoiseModel, not a bare sigma; got "
            f"{type(noise).__name__}. It solves for the covariance a "
            f"PREDICTION-DEPENDENT sigma implies, so a decided array leaves it "
            f"nothing to iterate. Wrap it -- HomoscedasticNoise(sigma) -- and "
            f"it will return after one step with converged=True, or pass the "
            f"array to wiener_solve, which is what a constant sigma wants."
        )
    prior_mean, prior_std = _check_solve_arguments(
        block, observed, prior_mean, prior_std, "iterative_gls"
    )
    if not 1 <= min_reweights <= max_reweights:
        # Kept here although the far side carries this check WORD FOR WORD,
        # because it does not carry the CLASS: it raises `GraphError`, and its
        # own comment calls that a misfit for what is a bad keyword argument.
        # Iron law 1 keeps exception identity, and pre-positioning is cheaper
        # than translating one class at one site.
        raise ParameterSpaceError(
            f"iterative_gls needs 1 <= min_reweights <= max_reweights, got "
            f"{min_reweights} and {max_reweights}. The loop caps at "
            "max_reweights either way, so this configuration would silently "
            "get fewer steps than it asked for."
        )

    def sigma_of(domain: Any) -> dict[str, Any]:
        """``{observed: sigma}`` at the prediction this latent implies.

        The far side's reweighting rule. It receives the block's DOMAIN in
        bayesmith's dict spelling, so it is unwrapped back to this package's
        before `noise.std` sees it -- `noise` is rheplicant's own object and
        speaks arrays.
        """
        latent = _from_far_domain(block, domain)
        return {_OBSERVED: noise.std(block.forward(latent) + block.offset)}

    far = _far_iterative_gls(
        _as_far_block(
            block, observed=observed, prior_mean=prior_mean, prior_std=prior_std
        ),
        sigma_of,
        depends_on_prediction=noise.depends_on_prediction,
        tol=tol,
        maxiter=maxiter,
        reweight_tol=reweight_tol,
        min_reweights=min_reweights,
        max_reweights=max_reweights,
        require_convergence=require_convergence,
    )
    # Back to this package's container. The two GLSResults hold ONE covariance
    # each and spell it differently: the far side stores `precision` and
    # derives `noise_std`, this one stores the sigma. Iron law 1 keeps this
    # layout, so the derivation runs here -- and for a diagonal precision it
    # returns the very array `sigma_of` produced, bitwise.
    return GLSResult(
        noise_std=far.noise_std[_OBSERVED],
        solution=_from_far_domain(block, far.solution),
        residual=far.residual,
        iterations=far.iterations,
        delta=far.delta,
        converged=far.converged,
    )
