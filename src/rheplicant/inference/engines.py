"""The engines a Gibbs block can be updated by, and the conditioning they share.

:mod:`rheplicant.inference.plan` decides *which* latents go in a block and
*which* engine that block takes; this module is what an engine actually does to
one block, given everything outside it held fixed.

There are three, and the split is not a taxonomy — it is the same split the rest
of the package already makes. A block whose latents are all declared
``linear=True`` is a linear-Gaussian conditional, so its point estimate is
:func:`~rheplicant.inference.linear.wiener_solve` and its draw is
:func:`~rheplicant.inference.linear.gcr_sample`, which is an **exact**
conditional draw. A block whose prediction is ``exp`` of an affine map is the
same thing once the DATA is taken to logs, under the multiplicative noise the
radiometer equation gives — :data:`LOG_CONJUGATE`, and
:mod:`rheplicant.inference.loglinear` is where that transform and its
first-order caveat live. Anything else has only a gradient, so its point
estimate is a descent and its draw is NUTS.

The first two share every line of their update but one, so they share a
function: :func:`_conjugate_update` takes which transition to build, and the
difference between them is exactly that a log block does not re-evaluate sigma.

**Conditioning is one function, used by both.** :class:`Conditioning` closes
over the model once and hands out three things: the joint prediction, the joint
sigma, and the joint chi-squared. Every engine reads the block's neighbours out
of the same ``values`` dict, so "condition on everything outside this block" is
never written by a caller and never written twice here.

**The two exits share this file, not their signatures.** Each engine exposes an
``..._estimate`` and a ``..._draw``, and for the conjugate engine they differ by
one argument to one private function — ``bayesmith.exact.solve._conjugate_solve``'s
``key=None | k``,
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

import equinox as eqx
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

#: The engine for a block that is conjugate once the DATA is taken to logs —
#: the prediction is ``exp(affine)``, so ``linear_operator`` refuses it while
#: :func:`~rheplicant.inference.loglinear.log_linear_operator` does not. There
#: is no ``Latent(log_linear=True)`` to derive this from, deliberately: it is
#: discovered by probing (:func:`~rheplicant.inference.partition.auto_blocks`)
#: or asked for explicitly, never inferred from a declaration that does not
#: exist.
LOG_CONJUGATE: str = "log_conjugate"

#: The engine for anything else — and the legitimate downgrade for a block the
#: caller wants stepped by gradient even though it could be solved.
GRADIENT: str = "gradient"

#: The three, in the order a message should list them.
ENGINES: tuple[str, ...] = (CONJUGATE, LOG_CONJUGATE, GRADIENT)

#: The engines that solve in closed form, and so have no inner step count and
#: no step size. Named once because three places refuse those arguments.
CLOSED_FORM: tuple[str, ...] = (CONJUGATE, LOG_CONJUGATE)

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
        log_observed, log_sigma: the data and sigma a
            :data:`LOG_CONJUGATE` block solves against, from
            :func:`~rheplicant.inference.loglinear.to_log_space`. ``None`` when
            the partition has no such block.

            Computed ONCE per run rather than per sweep, and that is not an
            optimization — it is the log-space claim showing up in the code.
            ``log_sigma`` does not depend on the prediction, so unlike
            :meth:`sigma` there is nothing for a sweep to re-evaluate; and
            ``to_log_space`` refuses a non-positive sample eagerly, which a
            jitted transition could not do at all.
    """

    space: ParameterSpace
    pipeline: AbstractOperator
    state_template: State
    observed: jax.Array
    noise: NoiseModel
    forward: Callable[[dict[str, jax.Array]], jax.Array]
    log_observed: jax.Array | None = None
    log_sigma: jax.Array | None = None

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


def _potential_of(
    cond: Conditioning, names: Sequence[str]
) -> Callable[[dict[str, jax.Array], dict[str, jax.Array]], jax.Array]:
    """The same objective, with the neighbours as an ARGUMENT rather than a closure.

    This is the whole of the compilation fix. :func:`conditional_potential`
    rebuilds its closure every sweep because ``others`` changes, so every
    ``jax.jit`` below it is keyed on a function object that never repeats: 41
    XLA compilations over 20 sweeps, 121 over 60, against 0.3 ms of leapfrog per
    call. Lifting ``others`` to a traced argument leaves ONE program per block
    per branch for the life of a run.

    Kept separate from :func:`conditional_potential`, which stays the
    single-argument objective the estimator and the residual reporting take.
    """

    def potential(
        others: dict[str, jax.Array], x: dict[str, jax.Array]
    ) -> jax.Array:
        return 0.5 * cond.chi2({**others, **x}) - _log_prior(cond.space, names, x)

    return potential


def _gradient_transition(
    cond: Conditioning, names: Sequence[str], *, steps: int, adapt: bool
) -> Callable[..., tuple[dict[str, jax.Array], tuple[jax.Array, jax.Array]]]:
    """One jittable NUTS transition: ``(others, x0, key, tuning) -> (x, tuning)``.

    Reproduces ``MCMC(kernel, num_warmup=w, num_samples=steps).run(...)`` exactly
    -- measured to 8.9e-16, one ulp -- by scanning ``kernel.sample`` the number
    of times ``MCMC`` would have applied it: ``2 * steps`` while adapting
    (``num_warmup=steps`` AND ``num_samples=steps``) and ``steps`` when frozen.
    Getting that count wrong is the failure mode with the most room to hide, and
    it does not hide: ``steps`` instead of ``2 * steps`` disagrees at 1e-1.

    ``kernel.init`` is called rather than the ``hmc()`` primitives beneath it
    because ``init`` performs its own ``random.split``, exactly as ``MCMC.run``
    does. Skipping that split yields a valid but DIFFERENT chain -- measured, a
    sweep-0 draw off by 3.8e-4, which is the size of the posterior sigma.

    ``eqx.filter_jit`` and not ``jax.jit``: measured at 0.143 ms against
    0.115 ms, so the choice is free, and it is what preserves an
    ``EquinoxRuntimeError`` from a guard inside the traced region rather than
    degrading it to a bare ``JaxRuntimeError`` with the message buried.
    """
    from numpyro.infer import NUTS

    length = 2 * steps if adapt else steps
    potential_of = _potential_of(cond, names)

    @eqx.filter_jit
    def transition(others, x0, key, tuning):
        def potential(x):
            return potential_of(others, x)

        if adapt:
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
        start = kernel.init(key, warmup, init_params=x0)
        final, _ = jax.lax.scan(
            lambda carry, _: (kernel.sample(carry, (), {}), None),
            start,
            None,
            length=length,
        )
        adapted = final.adapt_state
        return final.z, (adapted.step_size, adapted.inverse_mass_matrix)

    return transition


# ------------------------------------------------------- the conjugate engine --


def _conjugate_transition(
    cond: Conditioning,
    names: tuple[str, ...],
    *,
    draw: bool,
    tol: float,
    maxiter: int | None,
    require_convergence: float | None,
) -> Callable[..., tuple[dict[str, jax.Array], jax.Array]]:
    """One jittable conjugate update: ``(values, key) -> (solved, residual)``.

    The same closure-identity fix :func:`_gradient_transition` applies to the
    gradient block. ``linear_operator`` returns an operator whose matvec closes
    over ``at=values``, so an unjitted sweep hands the CG solver a Python
    function it has never seen, and XLA rebuilds the program: measured, one
    compilation per sweep, 21 at 20 sweeps against 61 at 60.

    ``eqx.filter_jit`` and **not** ``jax.jit``, and here that is a correctness
    requirement rather than a preference. ``gcr_sample``'s convergence guard is
    an :func:`equinox.error_if`; under ``filter_jit`` it surfaces as an
    ``EquinoxRuntimeError`` carrying the message that names the remedy, and
    under ``jax.jit`` it degrades to a bare ``JaxRuntimeError``. A guard whose
    diagnosis is buried is most of a guard thrown away, which is why
    ``test_the_conjugate_convergence_guard_still_raises_equinox`` asserts on the
    exception TYPE and not only on the text.

    One consequence worth stating: ``cond.sigma(values)`` is now traced, so a
    noise or forward model with Python control flow on a parameter's *value*
    raises where it used to run. That was already true of the gradient path,
    which NumPyro has always traced.
    """

    @eqx.filter_jit
    def transition(values, key):
        sigma = cond.sigma(values)
        block = linear_operator(
            cond.space, cond.pipeline, cond.state_template,
            names=names, at=values, check=False,
        )
        extra = {"key": key} if draw else {}
        return (gcr_sample if draw else wiener_solve)(
            block, cond.observed, noise_std=sigma, tol=tol, maxiter=maxiter,
            require_convergence=require_convergence, **extra,
        )

    return transition


def _log_conjugate_transition(
    cond: Conditioning,
    names: tuple[str, ...],
    *,
    draw: bool,
    tol: float,
    maxiter: int | None,
    require_convergence: float | None,
) -> Callable[..., tuple[dict[str, jax.Array], jax.Array]]:
    """The same update, against ``log`` of the data. Note what is NOT here.

    :func:`_conjugate_transition` opens with ``sigma = cond.sigma(values)``,
    because under a prediction-dependent noise model the covariance moves every
    time the block's neighbours do. In log space it does not:
    ``Var[log(1 + f w)]`` is a function of ``f`` alone, so ``cond.log_sigma`` is
    computed once for the run and simply used. The reweighting fixed point
    :func:`~rheplicant.inference.gls.iterative_gls` performs has nothing to
    iterate on here, and the absence of that one line is what says so.

    **What this costs, stated where it is incurred.** The log-space Gaussian is
    the multiplicative model only to first order, so a plan mixing
    :data:`LOG_CONJUGATE` and :data:`CONJUGATE` blocks draws its conditionals
    from two likelihoods that agree to ``O(f^2)`` rather than exactly. That is
    the same class of statement as
    :func:`gradient_draw`'s Metropolis-within-Gibbs downgrade: still a usable
    sampler, no longer the exact one a pure-conjugate plan is.
    :data:`~rheplicant.inference.loglinear.FIRST_ORDER_MAX_FRACTIONAL` bounds
    the discrepancy — at the ``f = 4.05e-3`` of a 61 kHz channel at 1 s the
    variance the two disagree on is 4e-5 of itself.
    """
    from rheplicant.inference.loglinear import log_linear_operator

    @eqx.filter_jit
    def transition(values, key):
        block = log_linear_operator(
            cond.space, cond.pipeline, cond.state_template,
            names=names, at=values, check=False,
        )
        extra = {"key": key} if draw else {}
        return (gcr_sample if draw else wiener_solve)(
            block, cond.log_observed, noise_std=cond.log_sigma, tol=tol,
            maxiter=maxiter, require_convergence=require_convergence, **extra,
        )

    return transition


def _conjugate_update(
    cond: Conditioning,
    names: tuple[str, ...],
    values: dict[str, jax.Array],
    *,
    key: jax.Array | None,
    tol: float,
    maxiter: int | None,
    require_convergence: float | None,
    programs: dict[Any, Any] | None = None,
    engine: str = CONJUGATE,
) -> tuple[dict[str, jax.Array], jax.Array]:
    """One conjugate-Gaussian block update. ``key=None`` is the mean, ``k`` a draw.

    The block is exported with ``names=`` even when it holds one latent, so the
    answer always comes back as a ``{name: array}`` dict and a partition can mix
    one-latent and many-latent blocks with no special case anywhere above.

    ``check=False`` because the linearity claim is checked once per run, before
    the loop — the bargain
    :func:`~rheplicant.inference.linear.gcr_sample`'s own docstring recommends
    for a Gibbs sweep.

    ``programs`` is the caller's compiled-transition cache; see
    :func:`gradient_draw` for why it is the caller's and why the key excludes
    the conditioning. Everything in the key here is part of the compiled
    program: ``tol`` and ``maxiter`` reach ``jax.scipy.sparse.linalg.cg`` as
    static arguments, and ``require_convergence`` decides whether the guard is
    traced into the graph at all.
    """
    draw = key is not None
    # `engine` is IN the cache key: the two transitions differ in which space
    # they solve, and serving one from the other's slot would draw a log block
    # against linear-space data with every guard still green.
    key_for = (names, draw, tol, maxiter, require_convergence, engine)
    transition = None if programs is None else programs.get(key_for)
    if transition is None:
        build = (
            _log_conjugate_transition
            if engine == LOG_CONJUGATE
            else _conjugate_transition
        )
        transition = build(
            cond, names, draw=draw, tol=tol, maxiter=maxiter,
            require_convergence=require_convergence,
        )
        if programs is not None:
            programs[key_for] = transition
    solved, residual = transition(values, key)
    return {**values, **solved}, residual


def conjugate_estimate(cond, names, values, **kwargs):
    """The block's conditional posterior MEAN, by :func:`wiener_solve`."""
    return _conjugate_update(cond, names, values, key=None, **kwargs)


def log_conjugate_estimate(cond, names, values, **kwargs):
    """The same mean, solved in log space. See :func:`_log_conjugate_transition`."""
    return _conjugate_update(
        cond, names, values, key=None, engine=LOG_CONJUGATE, **kwargs
    )


def log_conjugate_draw(cond, names, values, *, key, **kwargs):
    """An exact draw from the LOG-SPACE conditional.

    Exact for that conditional, which is the multiplicative model's own only to
    first order in ``f`` — the one qualification a
    :data:`CONJUGATE` block does not carry. See
    :func:`_log_conjugate_transition` for the size of it.
    """
    return _conjugate_update(
        cond, names, values, key=key, engine=LOG_CONJUGATE, **kwargs
    )


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
    programs: dict[Any, Any] | None = None,
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
        programs: a caller-owned cache of compiled transitions, keyed on
            ``(names, steps, adapting)``. A plan threads one dict through its
            whole run so the block compiles once instead of once per sweep;
            ``None`` compiles fresh, which is correct but pays 300 ms.

            The key deliberately does **not** include the conditioning. Keying
            on ``id(cond)`` would be the obvious shortcut and is a trap: CPython
            reuses ids after collection, so a stale program could be served
            against a different ``observed`` — a confident wrong answer with
            every guard still green. Callers own the cache and therefore own its
            lifetime; a new conditioning gets a new dict.

    Returns:
        ``(values, tuning)`` — the block updated to the chain's last state, and
        the tuning to hand to the next sweep.
    """
    _require_numpyro()

    adapting = adapt or tuning is None
    key_for = (tuple(names), steps, adapting)
    if programs is None:
        transition = _gradient_transition(cond, names, steps=steps, adapt=adapting)
    else:
        transition = programs.get(key_for)
        if transition is None:
            transition = _gradient_transition(
                cond, names, steps=steps, adapt=adapting
            )
            programs[key_for] = transition

    others = {key: value for key, value in values.items() if key not in names}
    x0 = {name: values[name] for name in names}
    drawn, adapted = transition(others, x0, key, tuning)
    return {**values, **drawn}, adapted


__all__ = [
    "CLOSED_FORM",
    "CONJUGATE",
    "DEFAULT_GRADIENT_STEPS",
    "DEFAULT_LEARNING_RATE",
    "ENGINES",
    "GRADIENT",
    "LOG_CONJUGATE",
    "Conditioning",
    "conditional_potential",
    "conjugate_draw",
    "conjugate_estimate",
    "log_conjugate_draw",
    "log_conjugate_estimate",
    "gradient_draw",
    "gradient_estimate",
    "require_priors",
]
