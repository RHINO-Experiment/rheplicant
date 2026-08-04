"""The two engines a Gibbs block can be updated by, and the conditioning they share.

:mod:`rheplicant.inference.plan` decides *which* latents go in a block and
*which* engine that block takes; this module is what an engine actually does to
one block, given everything outside it held fixed.

There are exactly two, and the split is not a taxonomy — it is the same split
the rest of the package already makes. A block whose latents are all declared
``linear=True`` is a linear-Gaussian conditional, so its point estimate is
:func:`~rheplicant.inference.linear.wiener_solve` and its draw is
:func:`~rheplicant.inference.linear.gcr_sample`, which is an **exact**
conditional draw. Anything else has only a gradient, so its point estimate is a
descent and its draw is NUTS.

**Conditioning is one function, used by both.** :class:`Conditioning` closes
over the model once and hands out three things: the joint prediction, the joint
sigma, and the joint chi-squared. Every engine reads the block's neighbours out
of the same ``values`` dict, so "condition on everything outside this block" is
never written by a caller and never written twice here.

**The two exits share this file, not their signatures.** Each engine exposes an
``..._estimate`` and a ``..._draw``, and for the conjugate engine they differ by
one argument to one private function — ``_conjugate_solve``'s ``key=None | k``,
which is where this package already said that a point estimate and a posterior
sample are two exits from one workflow. That argument stays private: what a
caller says is :meth:`~rheplicant.inference.plan.SamplingPlan.estimate` or
:meth:`~rheplicant.inference.plan.SamplingPlan.sample`.

**Where sigma is evaluated.** Immediately before each block's update, at the
current joint values. For a constant noise model that is a no-op. For a
prediction-dependent one
(:class:`~rheplicant.inference.noise.RadiometerNoise`) it makes the sweep
itself the reweighting iteration that
:func:`~rheplicant.inference.gls.iterative_gls` performs for a single block —
which is why a plan does **not** nest ``iterative_gls`` inside a block. The two
loops would be the same loop, one inside the other, converging to the same
fixed point at the product of their costs.
"""

import dataclasses
from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax

from rheplicant.core.errors import ParameterSpaceError
from rheplicant.core.operator import AbstractOperator
from rheplicant.core.state import State
from rheplicant.inference.linear import (
    _magnitude,
    gcr_sample,
    linear_operator,
    wiener_solve,
)
from rheplicant.inference.noise import NoiseModel
from rheplicant.inference.parameters import ParameterSpace

#: The engine for a block whose latents are ALL declared ``linear=True``.
CONJUGATE: str = "conjugate"

#: The engine for anything else — and the legitimate downgrade for a block the
#: caller wants stepped by gradient even though it could be solved.
GRADIENT: str = "gradient"

#: The two, in the order a message should list them.
ENGINES: tuple[str, ...] = (CONJUGATE, GRADIENT)

#: Inner steps a gradient block takes when ``Block(..., steps=)`` says nothing.
#: Small on purpose: within a Gibbs sweep the block is revisited every sweep, so
#: the useful quantity is steps x sweeps, and a large inner count spends its
#: effort exploring a conditional that is about to move.
DEFAULT_GRADIENT_STEPS: int = 25

#: Adam step size for a gradient block's point estimate, as a FRACTION of the
#: latent's own magnitude (``max|init|``, falling back to 1.0). A single
#: absolute step cannot serve a beam width near 12 degrees and a log-gain near
#: 0.1 at once; a relative one can.
DEFAULT_LEARNING_RATE: float = 1e-2


@dataclasses.dataclass(frozen=True)
class Conditioning:
    """The model, closed over once, with everything a block update needs.

    Deliberately a plain frozen dataclass and not an ``eqx.Module``, for the
    reason :class:`~rheplicant.inference.linear.LinearBlock` is: it holds a
    traced closure (``forward``) and is built where it is needed rather than
    carried around as a differentiable pytree.

    ``forward`` is
    :meth:`~rheplicant.inference.parameters.ParameterSpace.forward_fn`'s, built
    ONCE per run. That matters: ``forward_fn`` validates the space against the
    pipeline, which costs several abstract traces, and a sweep loop that rebuilt
    it per block per sweep would pay that hundreds of times for an answer that
    cannot change.

    Attributes:
        space, pipeline, state_template: the model under inference.
        observed: the data. Shaped like the prediction — checked at the exits.
        noise: the noise model, normalized (a bare sigma is already wrapped).
        forward: ``{name: value} -> prediction``.
    """

    space: ParameterSpace
    pipeline: AbstractOperator
    state_template: State
    observed: jax.Array
    noise: NoiseModel
    forward: Callable[[dict[str, jax.Array]], jax.Array]

    def sigma(self, values: dict[str, jax.Array]) -> jax.Array:
        """Noise sigma at the current joint prediction, shaped like the data."""
        return self.noise.std(self.forward(values))

    def chi2(self, values: dict[str, jax.Array]) -> jax.Array:
        """The JOINT chi-squared at the current parameter tuple.

        The quantity a plan monitors for convergence, and the reason it is here
        rather than in a block: it is computed from the whole parameter tuple
        against the whole data set, so it is the one number in a Gibbs scheme
        that no partition can hide anything from. A per-block CG residual read
        ~1e-7 on an answer thousands of kelvin wrong — and read the SAME
        ~1e-7 on the run that was right, which is the sharper complaint.
        It was not lying; it simply cannot see across the partition it is
        computed inside (``tests/inference/test_degenerate_partition.py``).

        An unobserved sample (infinite sigma, from
        :class:`~rheplicant.inference.noise.FlaggedNoise`) contributes exactly
        zero rather than ``0 * inf``.
        """
        prediction = self.forward(values)
        sigma = self.noise.std(prediction)
        seen = jnp.isfinite(sigma)
        safe = jnp.where(seen, sigma, 1.0)
        residual = jnp.where(seen, (self.observed - prediction) / safe, 0.0)
        return jnp.sum(residual**2)


def _log_prior(space: ParameterSpace, names: Sequence[str], x: dict[str, jax.Array]):
    """Total log prior density of the block's latents, zero where undeclared.

    A prior-free latent contributes nothing, which is exactly what "free
    parameter" means in this package: usable by a point estimate, and refused
    outright by a draw — see :func:`gradient_draw`.

    ``prior.log_prob`` is called on the latent's own value, so the gradient
    engine samples in the coordinates the latent was DECLARED in. A prior with
    bounded support (``Uniform``, ``HalfNormal``) therefore has to be
    reparameterized — declare ``log_x`` with a ``Normal`` prior and put the
    ``exp`` in the ``Bind``, which is the transformation this package already
    recommends for a positive quantity.
    """
    total = jnp.zeros(())
    for name in names:
        prior = space.latent(name).prior
        if prior is not None:
            total = total + jnp.sum(prior.log_prob(x[name]))
    return total


def conditional_potential(
    cond: Conditioning, names: Sequence[str], values: dict[str, jax.Array]
) -> Callable[[dict[str, jax.Array]], jax.Array]:
    """``x -> -log p(x | everything else)``, up to a constant.

    ``0.5 chi2 - log prior``, with ``x`` a ``{name: array}`` dict over the
    block's members and every other latent frozen at ``values``. This is the
    objective the gradient engine descends and the potential NUTS samples, so
    the two exits of that engine cannot target different distributions.
    """
    others = {key: value for key, value in values.items() if key not in names}

    def potential(x: dict[str, jax.Array]) -> jax.Array:
        return 0.5 * cond.chi2({**others, **x}) - _log_prior(cond.space, names, x)

    return potential


# ------------------------------------------------------- the conjugate engine --


def _conjugate_update(
    cond: Conditioning,
    names: tuple[str, ...],
    values: dict[str, jax.Array],
    *,
    key: jax.Array | None,
    tol: float,
    maxiter: int | None,
    require_convergence: float | None,
) -> tuple[dict[str, jax.Array], jax.Array]:
    """One conjugate-Gaussian block update. ``key=None`` is the mean, ``k`` a draw.

    The block is exported with ``names=`` even when it holds one latent, so the
    answer always comes back as a ``{name: array}`` dict and a partition can mix
    one-latent and many-latent blocks with no special case anywhere above.

    ``check=False`` because the linearity claim is checked once per run, before
    the loop — the bargain
    :func:`~rheplicant.inference.linear.gcr_sample`'s own docstring recommends
    for a Gibbs sweep.
    """
    sigma = cond.sigma(values)
    block = linear_operator(
        cond.space, cond.pipeline, cond.state_template,
        names=names, at=values, check=False,
    )
    solve = wiener_solve if key is None else gcr_sample
    extra = {} if key is None else {"key": key}
    solved, residual = solve(
        block, cond.observed, noise_std=sigma, tol=tol, maxiter=maxiter,
        require_convergence=require_convergence, **extra,
    )
    return {**values, **solved}, residual


def conjugate_estimate(cond, names, values, **kwargs):
    """The block's conditional posterior MEAN, by :func:`wiener_solve`."""
    return _conjugate_update(cond, names, values, key=None, **kwargs)


def conjugate_draw(cond, names, values, *, key, **kwargs):
    """An EXACT conditional draw, by :func:`gcr_sample`.

    Exact is the operative word and it is what makes a plan of conjugate blocks
    an exact Gibbs sampler: no accept step, no inner step count, nothing tuned.
    See :func:`gradient_draw` for what changes the moment a block is not one of
    these.
    """
    return _conjugate_update(cond, names, values, key=key, **kwargs)


# -------------------------------------------------------- the gradient engine --


def _adam(
    potential: Callable[[dict[str, jax.Array]], jax.Array],
    x0: dict[str, jax.Array],
    steps: int,
    step_sizes: dict[str, float],
) -> dict[str, jax.Array]:
    """Adam on a conditional POSTERIOR, per-latent step sizes, no dependencies.

    Not :class:`~rheplicant.inference.calibrate.AdamCalibrator`, and the
    difference is not stylistic: that class minimizes ``loss_fn(forward(params),
    observed)``, an objective that sees only the prediction and the data, and a
    conditional posterior is the sum of that with a prior term the interface has
    no way to pass. Dropping the prior would make a gradient block's point
    estimate and its draw target different distributions.

    ``step_sizes`` is per latent, in the latent's own units. See
    :data:`DEFAULT_LEARNING_RATE`.
    """
    beta1, beta2, floor = 0.9, 0.999, 1e-8
    zeros = jax.tree.map(jnp.zeros_like, x0)

    def step(carry, index):
        x, first, second = carry
        grads = jax.grad(potential)(x)
        first = jax.tree.map(lambda a, g: beta1 * a + (1 - beta1) * g, first, grads)
        second = jax.tree.map(lambda a, g: beta2 * a + (1 - beta2) * g**2, second, grads)
        count = index + 1
        x = {
            name: x[name]
            - step_sizes[name]
            * (first[name] / (1 - beta1**count))
            / (jnp.sqrt(second[name] / (1 - beta2**count)) + floor)
            for name in x
        }
        return (x, first, second), potential(x)

    (fitted, _, _), _ = lax.scan(step, (x0, zeros, zeros), jnp.arange(steps))
    return fitted


def gradient_estimate(
    cond: Conditioning,
    names: tuple[str, ...],
    values: dict[str, jax.Array],
    *,
    steps: int,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    **_ignored,
) -> tuple[dict[str, jax.Array], jax.Array]:
    """Descend the block's conditional potential for ``steps`` Adam steps.

    Returns the updated values and the potential reached, which stands in the
    residual's place in the conjugate engine's return — a number to record,
    never a convergence verdict. The verdict is the joint chi-squared, one level
    up.
    """
    potential = conditional_potential(cond, names, values)
    x0 = {name: values[name] for name in names}
    step_sizes = {
        name: learning_rate * _magnitude(cond.space.latent(name)) for name in names
    }
    fitted = _adam(potential, x0, steps, step_sizes)
    return {**values, **fitted}, potential(fitted)


def _require_numpyro():
    try:
        import numpyro.infer  # noqa: F401
    except ImportError as exc:  # pragma: no cover - numpyro is an optional extra
        raise ImportError(
            "A gradient block's DRAW is NUTS, which needs numpyro: "
            "pip install 'rheplicant[numpyro]'. A gradient block's point estimate "
            "needs nothing extra, and neither does a conjugate block at either exit."
        ) from exc


def require_priors(space: ParameterSpace, names: Sequence[str], where: str) -> None:
    """A gradient DRAW needs a prior on every member; a point estimate does not.

    Same rule, and the same reason, as
    :func:`~rheplicant.inference.numpyro_bridge.to_numpyro_model`'s: a
    prior-free latent is a free parameter, which is meaningful to an optimizer
    and meaningless in a posterior. Stated here rather than left to NUTS because
    without it the potential is flat in that latent and the chain wanders off to
    wherever the geometry lets it, reporting nothing wrong.
    """
    missing = [name for name in names if space.latent(name).prior is None]
    if missing:
        raise ParameterSpaceError(
            f"Block {where} is stepped by the gradient engine, and sampling it needs a "
            f"prior on every member; {missing} have none. A prior-free latent is a free "
            "parameter: fine for plan.estimate(), meaningless in a draw — the potential "
            "is flat in it and the chain wanders without any diagnostic saying so. "
            "Declare Latent(prior=...), or use plan.estimate()."
        )


def gradient_draw(
    cond: Conditioning,
    names: tuple[str, ...],
    values: dict[str, jax.Array],
    *,
    key: jax.Array,
    steps: int,
    tuning: Any = None,
    adapt: bool = True,
    **_ignored,
) -> tuple[dict[str, jax.Array], Any]:
    """``steps`` NUTS steps on the block's conditional potential, last state kept.

    **This is where a plan stops being an exact sampler.** A conjugate block's
    draw is an exact conditional draw, so Gibbs over conjugate blocks is exact.
    A finite number of NUTS steps is not a draw from the conditional — it is a
    Markov transition that *leaves the conditional invariant*, which makes the
    whole scheme Metropolis-within-Gibbs. That is still a valid sampler with the
    right stationary distribution, and it is not the same thing: the inner step
    count now affects mixing, and ``steps=20`` is a statistical assumption
    wearing the clothes of a performance knob.

    Args:
        cond, names, values: the conditioning, as everywhere in this module.
        key: PRNG key for this block at this sweep.
        steps: inner NUTS steps.
        tuning: ``(step_size, inverse_mass_matrix)`` carried from the previous
            sweep, or ``None`` to adapt from scratch.
        adapt: whether NUTS may adapt during this call. True through the plan's
            warmup and **False afterwards**: adapting a kernel from the states it
            is visiting destroys the reversibility the transition's validity
            rests on, so the tuning is frozen for every sweep whose draws are
            kept.

    Returns:
        ``(values, tuning)`` — the block updated to the chain's last state, and
        the tuning to hand to the next sweep.
    """
    _require_numpyro()
    from numpyro.infer import MCMC, NUTS

    potential = conditional_potential(cond, names, values)
    x0 = {name: values[name] for name in names}

    if adapt or tuning is None:
        kernel = NUTS(potential_fn=potential)
        warmup = steps
    else:
        step_size, inverse_mass_matrix = tuning
        kernel = NUTS(
            potential_fn=potential,
            step_size=step_size,
            inverse_mass_matrix=inverse_mass_matrix,
            adapt_step_size=False,
            adapt_mass_matrix=False,
        )
        warmup = 0

    mcmc = MCMC(kernel, num_warmup=warmup, num_samples=steps, progress_bar=False)
    mcmc.run(key, init_params=x0)
    drawn = mcmc.get_samples()
    adapted = mcmc.last_state.adapt_state
    return (
        {**values, **{name: drawn[name][-1] for name in names}},
        (adapted.step_size, adapted.inverse_mass_matrix),
    )


__all__ = [
    "CONJUGATE",
    "DEFAULT_GRADIENT_STEPS",
    "DEFAULT_LEARNING_RATE",
    "ENGINES",
    "GRADIENT",
    "Conditioning",
    "conditional_potential",
    "conjugate_draw",
    "conjugate_estimate",
    "gradient_draw",
    "gradient_estimate",
    "require_priors",
]
