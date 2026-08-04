"""Bridge a Pipeline/Assembly to a NumPyro probabilistic model.

Bayesian inference through the same seam as everything else (D7): a
:class:`~rheplicant.inference.parameters.ParameterSpace` says what is
inferred and how it enters the model, and this module turns it into priors
plus a likelihood::

    import numpyro.distributions as dist
    from rheplicant.inference import ParameterSpace, to_numpyro_model

    space = ParameterSpace.direct(
        "log_gain", init=0.0, into=lambda p: p["gain"].gain, fn=jnp.exp,
        prior=dist.Normal(0.0, 0.2),
    )
    model = to_numpyro_model(twin, state_template, space, noise_std=0.5)
    mcmc = numpyro.infer.MCMC(numpyro.infer.NUTS(model), num_warmup=500,
                              num_samples=500)
    mcmc.run(jax.random.key(0), observed=observation.data)

Sample sites are named by their **latents**, so NUTS explores — and the
samples come back keyed by — the coordinates the model was declared in.
``log_gain`` above is one site, even though its value reaches a pipeline leaf
called ``gain``, and it would remain one site if it drove five stages at once.

IMPORTANT — stochastic operators: in a Bayesian model the noise lives in the
LIKELIHOOD, not in the forward model. Build the pipeline you hand to
``to_numpyro_model`` *without* NoiseOperator/RFIOperator draws (the framework
already separates them as their own stages), or their fixed-key draws would
be treated as deterministic signal.

Posterior predictive / pushforward: :func:`predict_from_samples` runs the
pipeline over MCMC samples (pairs with
:mod:`rheplicant.inference.uncertainty`'s summaries).
"""

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from rheplicant.core.errors import ParameterSpaceError, StateValidationError
from rheplicant.core.operator import AbstractOperator
from rheplicant.core.state import State
from rheplicant.inference.likelihood import check_observed_shape
from rheplicant.inference.noise import HomoscedasticNoise
from rheplicant.inference.parameters import ParameterSpace
from rheplicant.inference.uncertainty import as_noise_model


def _require_numpyro():
    try:
        import numpyro  # noqa: F401
        import numpyro.distributions  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "This feature needs numpyro: pip install 'rheplicant[numpyro]'."
        ) from exc


def _require_priors(space: ParameterSpace) -> None:
    """Every latent must have a prior — a free parameter has no posterior."""
    missing = [latent.name for latent in space.latents if latent.prior is None]
    if missing:
        raise ParameterSpaceError(
            f"Latent(s) {missing} have no prior, so they cannot be sampled. A prior-free "
            "latent is a free parameter: fine for the calibrators, meaningless in a "
            "posterior. Give it a prior, or drop it from the space."
        )


def to_numpyro_model(
    pipeline: AbstractOperator,
    state_template: State,
    space: ParameterSpace,
    noise_std: Any,
    flags: jax.Array | None = None,
    obs_name: str = "obs",
):
    """Build a NumPyro model: priors -> bound pipeline -> Gaussian likelihood.

    Args:
        pipeline: the (deterministic) forward model.
        state_template: input state the model is evaluated on (closed over).
        space: what to infer and how it binds. Every latent needs a prior.
        noise_std: how noisy the data is. Three forms:

            * a scalar or array standard deviation;
            * a :class:`~rheplicant.inference.noise.NoiseModel` — in particular
              :class:`~rheplicant.inference.noise.RadiometerNoise`, whose sigma
              tracks the prediction and therefore the sampled parameters;
            * a NumPyro distribution, to infer a constant sigma (sampled at
              site ``"noise_std"``).
        flags: optional boolean mask (True = flagged); flagged samples are
            excluded from the likelihood (RFI flags -> noise covariance).
            Equivalent to wrapping ``noise_std`` in
            :class:`~rheplicant.inference.noise.FlaggedNoise`.
        obs_name: name of the observed sample site.

    Returns:
        A NumPyro model ``model(observed=None)`` — condition by passing
        ``observed=data``; run without it for prior-predictive checks. The
        noiseless prediction is recorded at the deterministic site
        ``"prediction"``. A conditioning ``observed`` whose shape is not
        exactly the prediction's is refused
        (:func:`~rheplicant.inference.likelihood.check_observed_shape`) while
        the model is traced, before any sample is drawn: broadcasting it would
        give NUTS a different posterior to explore, and it would explore it
        successfully.

    Note:
        **A prediction-dependent sigma brings its log-determinant with it, and
        that is the point of routing it through here.** ``Normal(loc,
        scale).log_prob`` contains ``-log scale``, so when ``scale`` is a
        function of the sampled parameters the term is part of the potential
        automatically — this is the full Gaussian density, not the generalized
        least squares that :func:`~rheplicant.inference.gls.iterative_gls`
        converges to (which freezes that dependence in order to keep each step
        linear). The two answers differ, in the direction
        :mod:`rheplicant.inference.noise` gives in closed form.

        An unobserved sample — infinite sigma, from
        :class:`~rheplicant.inference.noise.FlaggedNoise` or ``flags`` — is
        masked out rather than given an infinite scale, which would send the
        whole potential to ``-inf``. Masking is the limit that exists.
    """
    _require_numpyro()
    import numpyro
    import numpyro.distributions as dist

    _require_priors(space)
    space.validate(pipeline)

    def model(observed: jax.Array | None = None):
        values = {
            latent.name: numpyro.sample(latent.name, latent.prior)
            for latent in space.latents
        }
        prediction = space.bind(pipeline, values)(state_template).data
        # Trace time, not run time: both shapes are static, so this compiles
        # away entirely and NUTS refuses before it draws a single sample.
        # `observed=None` is the prior-predictive call, not a mismatch.
        if observed is not None:
            check_observed_shape(
                jnp.shape(prediction), observed, predictor="this model"
            )
        numpyro.deterministic("prediction", prediction)

        noise = as_noise_model(
            HomoscedasticNoise(numpyro.sample("noise_std", noise_std))
            if isinstance(noise_std, dist.Distribution)
            else noise_std,
            flags,
        )
        sigma = noise.std(prediction)

        # A sample with infinite sigma was not observed. Handing that scale to
        # Normal sends its log_prob -- and so the whole potential -- to -inf,
        # because r^2/sigma^2 vanishes but log sigma does not. Masking is the
        # limit that exists.
        seen = jnp.isfinite(sigma)
        site = dist.Normal(prediction, jnp.where(seen, sigma, 1.0))
        with numpyro.handlers.mask(mask=seen):
            numpyro.sample(obs_name, site, obs=observed)

    return model


def init_to_declared(space: ParameterSpace):
    """A NumPyro init strategy that starts where the ``ParameterSpace`` says.

    ``Latent(..., init=...)`` already states where the model starts, and the
    calibrators and :func:`~rheplicant.inference.linear.check_linearity` both
    use it. NUTS does not: a kernel built without an ``init_strategy`` uses
    NumPyro's default ``init_to_uniform``, which draws in the *unconstrained*
    space with no knowledge of the declaration. Pass this instead::

        kernel = numpyro.infer.NUTS(model, init_strategy=init_to_declared(space))

    **This is not a tuning knob.** On the ring toy of
    ``examples/tutorial_nuts.py`` — 1024 samples constraining three beam
    parameters — the default initialization gives ``r_hat = 840`` and an
    effective sample size of **2** out of 8000 draws, while the identical model
    started here gives ``r_hat = 1.002`` and ``n_eff = 1327``. Neither
    tightening the priors nor tripling the warmup moved those numbers at all;
    only the starting point did.

    The mechanism is ordinary and worth recognising: a posterior far narrower
    than its prior is a needle, ``init_to_uniform`` lands in the haystack, and
    warmup adapts a step size for wherever it landed. The declared ``init`` does
    not have to be good — the one in that tutorial is deliberately mis-set, some
    11000 nats below the peak — it only has to be somewhere a gradient can be
    followed.

    Args:
        space: the space the model was built from.

    Returns:
        A NumPyro init strategy, ready for ``NUTS(model, init_strategy=...)``.
    """
    _require_numpyro()
    import numpyro

    return numpyro.infer.init_to_value(values=space.initial_values())


def predict_from_samples(
    pipeline: AbstractOperator,
    state_template: State,
    space: ParameterSpace,
    samples: dict[str, jax.Array],
) -> jax.Array:
    """Posterior predictive: run the pipeline over MCMC samples.

    Args:
        pipeline, state_template, space: as given to :func:`to_numpyro_model`.
        samples: ``mcmc.get_samples()`` — latent name -> ``(n_samples, ...)``.

    Returns:
        ``(n_samples, *data.shape)`` noiseless predictions (add likelihood
        noise separately if you want the full predictive).
    """
    _require_numpyro()
    space.validate(pipeline)
    for name in space.names:
        if name not in samples:
            raise StateValidationError(
                f"samples is missing site {name!r}; available: {sorted(samples)}"
            )
        # Checking only the NAME lets a wrong-shaped stack broadcast into the
        # leaf and return a finite, correctly-shaped, wrong predictive.
        expected = space.latent(name).init.shape
        got = jnp.shape(samples[name])[1:]
        if got != expected:
            raise StateValidationError(
                f"samples[{name!r}] has per-sample shape {got}, but the latent is "
                f"{expected}. The leading axis must be the sample axis."
            )
    lengths = {jnp.shape(samples[name])[0] for name in space.names}
    if len(lengths) != 1:
        raise StateValidationError(
            f"samples have differing numbers of draws across sites: {sorted(lengths)}."
        )
    stacked = {name: samples[name] for name in space.names}

    def run(values: dict[str, jax.Array]) -> jax.Array:
        return space.bind(pipeline, values)(state_template).data

    return eqx.filter_vmap(run)(stacked)
