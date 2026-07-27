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
from rheplicant.inference.parameters import ParameterSpace


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
        noise_std: likelihood noise standard deviation — a scalar/array, or a
            NumPyro distribution to infer it (sampled at site ``"noise_std"``).
        flags: optional boolean mask (True = flagged); flagged samples are
            excluded from the likelihood (RFI flags -> noise covariance).
        obs_name: name of the observed sample site.

    Returns:
        A NumPyro model ``model(observed=None)`` — condition by passing
        ``observed=data``; run without it for prior-predictive checks. The
        noiseless prediction is recorded at the deterministic site
        ``"prediction"``.
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
        numpyro.deterministic("prediction", prediction)

        scale = (
            numpyro.sample("noise_std", noise_std)
            if isinstance(noise_std, dist.Distribution)
            else noise_std
        )
        site = dist.Normal(prediction, scale)
        if flags is not None:
            with numpyro.handlers.mask(mask=~flags):
                numpyro.sample(obs_name, site, obs=observed)
        else:
            numpyro.sample(obs_name, site, obs=observed)

    return model


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
