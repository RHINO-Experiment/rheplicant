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

import jax
import jax.numpy as jnp
from bayesmith.amortize import NeuralPosterior as _FarNeuralPosterior
from bayesmith.amortize import TrainingHistory as _FarTrainingHistory
from bayesmith.amortize import train_posterior as _far_train_posterior
from bayesmith.errors import StructureError as _FarStructureError

from rheplicant.core.errors import ParameterSpaceError, StateValidationError
from rheplicant.core.operator import AbstractOperator
from rheplicant.core.state import State
from rheplicant.inference.noise import NoiseModel
from rheplicant.inference.parameters import ParameterSpace

#: Floor on a mixture component's scale, as a fraction of the standardized
#: latent's unit width. A collapsed component -- one sitting on a single
#: training point -- would take the log-density to infinity, and this keeps a
#: component's width away from that.
#:
#: **Measured 2026-08-29: the floor guards a LIMIT, not a reachable value.**
#: The scale is ``softplus(raw) + min_scale`` (``NeuralPosterior._mixture``,
#: below), and ``softplus`` is strictly positive -- over ``raw`` in
#: ``[-80, 80]`` its minimum is ``1.8e-35``, never zero. So ``min_scale = 0``
#: does not give a zero scale: a deliberately collapsible bank (eight distinct
#: thetas, eight components) returns a finite ``log_prob`` at ``0.0``
#: (-3.6740) as at the default (-3.6689). This comment previously said the
#: collapse happens "without it", which is not what the arithmetic does, and
#: the config layer's refusal of ``min_scale: 0`` cites this line for that
#: claim -- see ``config/sections/npe.py::_positive``. bayesmith's
#: ``amortize.MIN_SCALE`` records the same measurement independently and
#: refuses only a NEGATIVE floor.
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
            multiplicative, exactly as in the data -- because the draw is
            taken with the model's own ``realise``, so the simulator and the
            likelihood cannot disagree about the law.
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
        # The scatter law comes from the noise model's OWN generator, not from
        # a `prediction + std * normal` written here. They are not the same
        # draw: RadiometerNoise realises `d (1 + f w)`, whose sign follows the
        # prediction, while `std` returns `|d| f` and applies `floor` -- an
        # absolute value and a reweighting remedy that a generator must not
        # have. Simulating with one and weighting with the other is exactly the
        # disagreement `realise` exists to make impossible.
        sigma = noise.std(prediction)
        observed = jnp.where(
            jnp.isfinite(sigma),
            noise.realise(prediction, key=index_key),
            prediction,
        )
        flat = jnp.concatenate([jnp.ravel(jnp.asarray(values[n])) for n in names])
        return flat, observed

    indices = jnp.arange(n_simulations)
    keys = jax.random.split(noise_key, n_simulations)
    return jax.vmap(one)(keys, indices)


class NeuralPosterior(_FarNeuralPosterior):
    """``q(theta | x)``: a conditional Gaussian mixture over the latent vector.

    **Subclasses :class:`bayesmith.amortize.NeuralPosterior` as of the Wave C
    `npe` switch** (migration ledger **D10**, owner-authorised 2026-08-27,
    scoped by **D42**). The network, the standardization, the mixture and both
    exits are inherited; the whole of this class is the ``create`` override
    below, and the whole of that override is one exception translation.

    **Why a subclass and not a re-export.** Class identity is not pinned
    anywhere -- 27 references across 7 test files, zero ``isinstance`` or
    ``type() is`` assertions -- so a re-export would preserve every name. What
    it would not preserve is the **exception class**: seven guard tests in
    ``tests/inference/test_inference_construction_guards.py`` expect
    :class:`~rheplicant.core.errors.StateValidationError`, and the far side
    raises ``StructureError`` throughout. That is what D10(3)'s "thin wrapper"
    is for, and it is measured rather than assumed.

    **Why a subclass and not a held instance.** D12 found that subclassing
    cannot translate an exception raised in ``__check_init__``, because the
    base class raises during construction before a subclass can intervene.
    These three refusals are in :meth:`create`, a classmethod, and both sides
    end theirs with ``return cls(...)`` -- so the override is clean. Holding a
    far-side instance instead would nest the pytree; inheriting keeps the nine
    field names in order and the fourteen leaves flat, which the config layer's
    reading of ``create``'s signature and ``sample``'s positional contract
    depends on.

    Measured across the seam before the near-side implementation was deleted,
    and it cannot be re-measured now: untrained ``log_prob``, ``log_prob``
    after 200 training steps, and all three ``TrainingHistory`` fields agreed
    at ``max|delta| = 0.0``. See
    ``docs/superpowers/specs/2026-08-29-wave-C-npe-opening.md`` in bayesmith.

    Attributes:
        Inherited unchanged, nine of them in the order the far side declares:
        ``net``, ``embed``, ``n_components``, ``n_params``, ``theta_mean``,
        ``theta_scale``, ``data_mean``, ``data_scale``, ``min_scale``.
    """

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
        """The far side's, with its refusals raised as this package's class.

        **The parameter list is restated rather than forwarded as
        ``*args, **kwargs``, and that is not a style choice.** The config layer
        DERIVES its ``npe:`` grammar from this signature -- which keys exist,
        which are optional, which default to what, and that ``embed`` belongs
        to ``create`` and not to ``train`` -- so a signature of ``*args`` is a
        grammar of nothing. Measured: forwarding took seven tests in
        ``tests/config/test_config_section_npe.py::
        TestTheGrammarMatchesTheSignatures`` down at once.

        Restating creates the drift risk that forwarding was meant to avoid,
        so it is guarded rather than hoped: ``test_npe_signatures_match_the_far
        _side`` asserts this list equals
        :meth:`bayesmith.amortize.NeuralPosterior.create`'s, parameter for
        parameter and default for default.
        """
        try:
            return super().create(
                thetas,
                data,
                key=key,
                embed=embed,
                n_components=n_components,
                width=width,
                depth=depth,
                min_scale=min_scale,
            )
        except _FarStructureError as exc:
            raise StateValidationError(str(exc)) from None


#: Re-exported rather than wrapped: it carries no refusals to translate, its
#: three fields (`train`, `validation`, `best_step`) match the far side's by
#: name and order, and nothing pins its identity.
TrainingHistory = _FarTrainingHistory


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
    """Train ``posterior`` on ``(thetas, data)``, returning the best step.

    **Delegates to :func:`bayesmith.amortize.train_posterior`** (D10). The
    three refusals it carries -- a non-positive ``n_steps``, a
    ``validation_fraction`` outside ``[0, 1)``, and a fraction that rounds to
    zero held-out simulations -- are the far side's, re-raised here as
    :class:`~rheplicant.core.errors.StateValidationError`. Their text is
    unchanged: each of the three is pinned upstream, and every pin was checked
    by running its pattern against the far side's real message rather than by
    reading the two side by side.

    Returns:
        ``(posterior, history)`` -- the estimator at its best validation step
        and a :class:`TrainingHistory`. The returned estimator is whatever the
        far side built, so it is a plain
        ``bayesmith.amortize.NeuralPosterior`` rather than the subclass above.
        That is invisible to every caller here: no test asserts the class, and
        the subclass adds no field and no method beyond ``create``'s
        translation.

    Note:
        **Over-fitting an NPE makes it over-confident, which is the failure
        that does not look like one.** Measured on this package's own
        linear-Gaussian test problem, with a bank of 8192 simulations: at 1500
        steps the fitted posterior width is 0.88 of the exact one; at 4000
        steps with four mixture components it is 0.60. The training loss
        improves throughout. Nothing about the resulting density looks wrong --
        it is smooth, it integrates to one, and it is centred correctly.
        Holding out a split and returning the best step is what turns that into
        something visible, and it is the default for that reason.

        Relatedly, prefer few components. A Gaussian posterior is exact at
        ``n_components=1``, and extra components mostly buy capacity to
        memorize the bank.
    """
    try:
        return _far_train_posterior(
            posterior,
            thetas,
            data,
            key=key,
            n_steps=n_steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            validation_fraction=validation_fraction,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
        )
    except _FarStructureError as exc:
        raise StateValidationError(str(exc)) from None
