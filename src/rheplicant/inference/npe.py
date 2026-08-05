"""Amortized neural posterior estimation: inference that never writes a likelihood.

Every other engine in this package evaluates a likelihood. NUTS does it once
per leapfrog step; :func:`~rheplicant.inference.linear.gcr_sample` exploits its
conjugate form. Simulation-based inference does not evaluate one at all: it
draws pairs ``(theta, x)`` from the prior and the simulator, fits a conditional
density ``q(theta | x)`` to them, and then reads the posterior off ``q`` at the
data actually observed.

Two properties follow, and they are why this exists alongside the exact
solvers rather than instead of them:

* **Amortized.** The training cost is paid once. Evaluating the posterior for a
  new observation is a forward pass — no chain, no burn-in, no re-solving. For
  a nightly re-calibration over many observations that is the difference
  between hours and milliseconds.
* **Likelihood-free.** Nothing here needs the noise to be Gaussian, or the
  forward model to be differentiable, or a normalization to be tractable. It
  needs a simulator, which the twin already is.

The price is that the answer is only as good as the fit, and an approximate
posterior has **no internal notion of being wrong** — a badly-trained ``q``
returns a confident, smooth, incorrect distribution and reports nothing amiss.
So this module is deliberately built to be checkable: on a linear-Gaussian
problem the exact posterior is available from
:func:`~rheplicant.inference.linear.gcr_sample`, and the package's tests hold
the estimator to it. Validate on a case you can solve before trusting one you
cannot.

The density is a **conditional Gaussian mixture** — an MLP mapping a summary of
the data to the weights, means and scales of a mixture over the latent vector.
A normalizing flow is more expressive; a mixture is a few dozen lines, is exact
for a Gaussian posterior at one component, and keeps the failure modes legible.
Adam is hand-rolled here for the same reason it is in
:mod:`rheplicant.inference.calibrate`: no optax dependency.

Usage::

    thetas, bank = simulate_pairs(twin, state, space, noise=noise,
                                  key=jax.random.key(0), n_simulations=20_000)
    q = NeuralPosterior.create(thetas, bank, key=jax.random.key(1))
    q, losses = train_posterior(q, thetas, bank, key=jax.random.key(2))
    draws = q.sample(observed, key=jax.random.key(3), n_samples=4000)
"""

from collections.abc import Callable
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp

from rheplicant.core.errors import ParameterSpaceError, StateValidationError
from rheplicant.core.operator import AbstractOperator
from rheplicant.core.state import State
from rheplicant.inference.noise import NoiseModel
from rheplicant.inference.parameters import ParameterSpace

#: Floor on a mixture component's scale, as a fraction of the standardized
#: latent's unit width. Without it a component can collapse onto a single
#: training point and take the log-density to infinity.
MIN_SCALE: float = 1e-3


def _standardize(values: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Per-feature mean and a strictly positive scale."""
    mean = jnp.mean(values, axis=0)
    scale = jnp.std(values, axis=0)
    return mean, jnp.where(scale > 0.0, scale, 1.0)


def simulate_pairs(
    pipeline: AbstractOperator,
    state_template: State,
    space: ParameterSpace,
    *,
    noise: NoiseModel,
    key: jax.Array,
    n_simulations: int,
) -> tuple[jax.Array, jax.Array]:
    """Draw ``(theta, x)`` pairs from the prior and the simulator.

    The "simulation" in simulation-based inference, and it is the twin doing
    it: priors from the :class:`~rheplicant.inference.parameters.ParameterSpace`,
    the forward model from the pipeline, the scatter from the noise model — the
    same three objects every other engine uses.

    Args:
        pipeline: the deterministic forward model.
        state_template: the input state it is evaluated on.
        space: what is inferred. Every latent needs a prior, since the prior is
            what is being sampled from.
        noise: supplies sigma at each simulated prediction. Under
            :class:`~rheplicant.inference.noise.RadiometerNoise` the scatter is
            multiplicative, exactly as in the data.
        key: PRNG key.
        n_simulations: how many pairs.

    Returns:
        ``(thetas, data)`` — ``thetas`` is ``(n_simulations, n_latent_values)``
        with each latent's values raveled and concatenated in ``space.names``
        order; ``data`` is ``(n_simulations, *data.shape)``.

    Note:
        A sample the noise model reports as unobserved (infinite sigma, from
        :class:`~rheplicant.inference.noise.FlaggedNoise`) is simulated without
        scatter. It carries no information either way, and the right place to
        remove it is the estimator's ``embed`` — leaving it in the input feeds
        the network a constant.
    """
    missing = [latent.name for latent in space.latents if latent.prior is None]
    if missing:
        raise ParameterSpaceError(
            f"Latent(s) {missing} have no prior, so there is nothing to simulate "
            "from. Simulation-based inference draws theta FROM the prior; a "
            "prior-free latent has no distribution to draw."
        )
    if n_simulations < 1:
        raise StateValidationError(
            f"n_simulations must be positive, got {n_simulations}."
        )
    space.validate(pipeline)

    names = list(space.names)
    prior_key, noise_key = jax.random.split(key)
    prior_keys = dict(zip(names, jax.random.split(prior_key, len(names)), strict=True))

    def one(index_key: jax.Array, draw_index: jax.Array):
        values = {
            name: space.latent(name).prior.sample(
                jax.random.fold_in(prior_keys[name], draw_index)
            )
            for name in names
        }
        prediction = space.bind(pipeline, values)(state_template).data
        sigma = noise.std(prediction)
        scatter = jnp.where(jnp.isfinite(sigma), sigma, 0.0)
        observed = prediction + scatter * jax.random.normal(
            index_key, jnp.shape(prediction)
        )
        flat = jnp.concatenate([jnp.ravel(jnp.asarray(values[n])) for n in names])
        return flat, observed

    indices = jnp.arange(n_simulations)
    keys = jax.random.split(noise_key, n_simulations)
    return jax.vmap(one)(keys, indices)


class NeuralPosterior(eqx.Module):
    """``q(theta | x)``: a conditional Gaussian mixture over the latent vector.

    An ``eqx.nn.MLP`` maps ``embed(x)`` to the mixture's log-weights, means and
    log-scales. Both ``theta`` and the embedded data are standardized using
    statistics taken from the training bank, which is why :meth:`create` needs
    the bank rather than only its shapes: an unstandardized network trained on
    kelvin-scale data and unit-scale parameters does not converge, and reports
    that as a mediocre fit rather than as an error.

    Attributes:
        net: the MLP.
        embed: ``x -> feature vector`` for ONE datum (static). Defaults to
            ``jnp.ravel``. This is where a large observation is summarized and
            where flagged samples are dropped.
        n_components: mixture components (static). One is exact for a Gaussian
            posterior, which is what makes the linear-Gaussian check sharp.
        n_params: length of the latent vector (static).
        theta_mean, theta_scale, data_mean, data_scale: standardization,
            derived from the bank.
        min_scale: floor on a component's standardized scale (static).
    """

    net: eqx.nn.MLP
    embed: Callable[[jax.Array], jax.Array] = eqx.field(static=True)
    n_components: int = eqx.field(static=True)
    n_params: int = eqx.field(static=True)
    theta_mean: jax.Array
    theta_scale: jax.Array
    data_mean: jax.Array
    data_scale: jax.Array
    min_scale: float = eqx.field(static=True, default=MIN_SCALE)

    @classmethod
    def create(
        cls,
        thetas: jax.Array,
        data: jax.Array,
        *,
        key: jax.Array,
        embed: Callable[[jax.Array], jax.Array] = jnp.ravel,
        n_components: int = 4,
        width: int = 64,
        depth: int = 3,
        min_scale: float = MIN_SCALE,
    ) -> "NeuralPosterior":
        """Build an untrained estimator sized and standardized to a bank."""
        if thetas.ndim != 2:
            raise StateValidationError(
                f"thetas must be (n_simulations, n_params), got shape {thetas.shape}."
            )
        if thetas.shape[0] != data.shape[0]:
            raise StateValidationError(
                f"thetas has {thetas.shape[0]} simulations but data has "
                f"{data.shape[0]}; they must be the same pairs."
            )
        if not n_components >= 1:  # `not >=` so a NaN count is refused too
            raise StateValidationError(
                f"n_components must be positive, got {n_components}."
            )
        features = jax.vmap(embed)(data)
        n_params = thetas.shape[1]
        theta_mean, theta_scale = _standardize(thetas)
        data_mean, data_scale = _standardize(features)
        net = eqx.nn.MLP(
            in_size=features.shape[1],
            out_size=n_components * (1 + 2 * n_params),
            width_size=width,
            depth=depth,
            key=key,
        )
        return cls(
            net=net, embed=embed, n_components=n_components, n_params=n_params,
            theta_mean=theta_mean, theta_scale=theta_scale,
            data_mean=data_mean, data_scale=data_scale, min_scale=min_scale,
        )

    def _mixture(self, datum: jax.Array):
        """``(log_weights, means, scales)`` in STANDARDIZED latent space."""
        features = (self.embed(datum) - self.data_mean) / self.data_scale
        raw = self.net(features)
        k, d = self.n_components, self.n_params
        logits = raw[:k]
        means = raw[k : k + k * d].reshape(k, d)
        # softplus, not exp: a large positive output cannot overflow the scale,
        # and the floor keeps a component from collapsing onto one training
        # point and taking the log-density to infinity.
        scales = jax.nn.softplus(raw[k + k * d :].reshape(k, d)) + self.min_scale
        return jax.nn.log_softmax(logits), means, scales

    def log_prob(self, theta: jax.Array, datum: jax.Array) -> jax.Array:
        """``log q(theta | x)`` for one pair, in the latent's own units."""
        log_weights, means, scales = self._mixture(datum)
        z = (jnp.ravel(theta) - self.theta_mean) / self.theta_scale
        per_component = jnp.sum(
            -0.5 * ((z - means) / scales) ** 2
            - jnp.log(scales)
            - 0.5 * jnp.log(2.0 * jnp.pi),
            axis=-1,
        )
        # The standardization is a change of variables, so its Jacobian belongs
        # in the density: without it log_prob is off by a constant that depends
        # on the training bank, which is exactly the kind of error that trains
        # away invisibly and then breaks any comparison against a real density.
        return jax.nn.logsumexp(log_weights + per_component) - jnp.sum(
            jnp.log(self.theta_scale)
        )

    def sample(
        self, datum: jax.Array, key: jax.Array, n_samples: int
    ) -> jax.Array:
        """Draw ``(n_samples, n_params)`` from ``q(theta | x)``."""
        log_weights, means, scales = self._mixture(datum)

        def one(subkey):
            pick_key, draw_key = jax.random.split(subkey)
            component = jax.random.categorical(pick_key, log_weights)
            z = means[component] + scales[component] * jax.random.normal(
                draw_key, (self.n_params,)
            )
            return self.theta_mean + self.theta_scale * z

        return jax.vmap(one)(jax.random.split(key, n_samples))


class TrainingHistory(NamedTuple):
    """Per-step losses from :func:`train_posterior`.

    * ``train`` — mean negative log-density on the minibatch, ``(n_steps,)``.
    * ``validation`` — the same on the held-out split, ``(n_steps,)``, or an
      empty array when ``validation_fraction`` is zero.
    * ``best_step`` — the step whose validation loss was lowest, and the
      parameters that were returned.

    **The validation curve is the only instrument that reports over-fitting.**
    The training loss falls monotonically past the point where the fit stops
    being a posterior; what over-fitting does to ``q`` is make it too NARROW,
    so the failure presents as an unusually confident answer rather than a
    visibly bad one.
    """

    train: jax.Array
    validation: jax.Array
    best_step: jax.Array


def train_posterior(
    posterior: NeuralPosterior,
    thetas: jax.Array,
    data: jax.Array,
    *,
    key: jax.Array,
    n_steps: int = 3000,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    validation_fraction: float = 0.1,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[NeuralPosterior, TrainingHistory]:
    """Maximize the mean ``log q(theta | x)`` over the simulation bank.

    That objective is what makes ``q`` a posterior rather than a fit to
    anything else: its population optimum is the true ``p(theta | x)``, because
    the pairs are drawn from the joint.

    Adam, hand-rolled, matching
    :class:`~rheplicant.inference.calibrate.AdamCalibrator` — the package does
    not depend on optax.

    Args:
        posterior: from :meth:`NeuralPosterior.create` (or a partly trained one;
            training resumes from wherever it is).
        thetas, data: the bank, as returned by :func:`simulate_pairs`.
        key: PRNG key for the split and the minibatching.
        n_steps: Adam steps.
        batch_size: simulations per step; capped at the training split.
        learning_rate, beta1, beta2, eps: Adam settings.
        validation_fraction: share of the bank held out. **Defaults to 0.1, and
            the returned estimator is the one from the best validation step,
            not the last one** — see the note below. Set to ``0.0`` to train on
            everything and return the final parameters, which is the faster and
            more dangerous option.

    Returns:
        ``(posterior, history)`` — the estimator at its best validation step and
        a :class:`TrainingHistory`.

    Note:
        **Over-fitting an NPE makes it over-confident, which is the failure that
        does not look like one.** Measured on this package's own linear-Gaussian
        test problem, with a bank of 8192 simulations: at 1500 steps the fitted
        posterior width is 0.88 of the exact one; at 4000 steps with four
        mixture components it is 0.60. The training loss improves throughout.
        Nothing about the resulting density looks wrong — it is smooth, it
        integrates to one, and it is centred correctly. Holding out a split and
        returning the best step is what turns that into something visible, and
        it is the default for that reason.

        Relatedly, prefer few components. A Gaussian posterior is exact at
        ``n_components=1``, and extra components mostly buy capacity to memorize
        the bank.
    """
    if n_steps < 1:
        raise StateValidationError(f"n_steps must be positive, got {n_steps}.")
    if not 0.0 <= validation_fraction < 1.0:
        raise StateValidationError(
            f"validation_fraction must be in [0, 1), got {validation_fraction}."
        )
    n_bank = thetas.shape[0]
    n_validation = int(round(validation_fraction * n_bank))
    if validation_fraction > 0.0 and n_validation < 1:
        raise StateValidationError(
            f"validation_fraction={validation_fraction} holds out zero of "
            f"{n_bank} simulations. Enlarge the bank or pass 0.0 explicitly to "
            "train without a held-out split."
        )

    split_key, train_key = jax.random.split(key)
    shuffled = jax.random.permutation(split_key, n_bank)
    validation_index = shuffled[:n_validation]
    train_index = shuffled[n_validation:]
    batch = min(batch_size, int(train_index.shape[0]))

    params, static = eqx.partition(posterior, eqx.is_inexact_array)

    def loss(free: Any, index: jax.Array) -> jax.Array:
        model = eqx.combine(free, static)
        return -jnp.mean(jax.vmap(model.log_prob)(thetas[index], data[index]))

    zeros = jax.tree.map(jnp.zeros_like, params)

    def step(carry, step_key):
        current, m, v, count, best, best_loss, best_step = carry
        picked = train_index[
            jax.random.choice(step_key, train_index.shape[0], (batch,), replace=False)
        ]
        value, grads = jax.value_and_grad(loss)(current, picked)
        m = jax.tree.map(lambda a, g: beta1 * a + (1 - beta1) * g, m, grads)
        v = jax.tree.map(lambda a, g: beta2 * a + (1 - beta2) * g**2, v, grads)
        t = count + 1
        current = jax.tree.map(
            lambda p, mm, vv: p
            - learning_rate
            * (mm / (1 - beta1**t))
            / (jnp.sqrt(vv / (1 - beta2**t)) + eps),
            current, m, v,
        )
        if n_validation:
            held_out = loss(current, validation_index)
            improved = held_out < best_loss
            best = jax.tree.map(
                lambda old, new: jnp.where(improved, new, old), best, current
            )
            best_loss = jnp.where(improved, held_out, best_loss)
            best_step = jnp.where(improved, t, best_step)
        else:
            held_out = jnp.asarray(jnp.nan)
            best, best_step = current, t
        return (current, m, v, t, best, best_loss, best_step), (value, held_out)

    init = (params, zeros, zeros, 0, params, jnp.asarray(jnp.inf), jnp.asarray(0))
    (_, _, _, _, best, _, best_step), (train_losses, validation_losses) = jax.lax.scan(
        step, init, jax.random.split(train_key, n_steps)
    )
    history = TrainingHistory(
        train=train_losses,
        validation=validation_losses if n_validation else jnp.zeros((0,)),
        best_step=best_step,
    )
    return eqx.combine(best, static), history
