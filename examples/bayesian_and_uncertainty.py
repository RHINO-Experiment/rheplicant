"""Bayesian calibration + uncertainty propagation — end-to-end demo.

One twin, three uncertainty routes through the same seam:

1. NumPyro NUTS posterior over (gain, sky amplitude);
2. Fisher forecast (Cramer-Rao) — should match the posterior widths here,
   since the model is nearly linear in the parameters;
3. delta-method prediction band + Monte Carlo posterior predictive.

Run:  uv run python examples/bayesian_and_uncertainty.py
(requires the numpyro extra: pip install 'rheplicant[numpyro]')
"""

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from rheplicant import Coordinates, State
from rheplicant.inference import (
    ParameterSpace,
    fisher_information,
    parameter_covariance,
    predict_from_samples,
    propagate_covariance,
    to_numpyro_model,
)
from rheplicant.radio import GainOperator, SkyOperator, assemble

TRUE_GAIN, TRUE_SKY, SIGMA = 1.1, 100.0, 0.5

state = State(
    coords=Coordinates(time=jnp.linspace(0.0, 60.0, 32),
                       freq=jnp.linspace(60e6, 85e6, 8)),
    meta={"telescope": "generic-dish"},
)

truth = assemble(
    SkyOperator(amplitude=jnp.array(TRUE_SKY)),
    GainOperator(gain=jnp.array(TRUE_GAIN)),
)
observed = truth(state).data + SIGMA * jax.random.normal(
    jax.random.key(99), (32, 8)
)

# ----------------------------------------------------------- NumPyro NUTS --
twin = assemble(
    SkyOperator(amplitude=jnp.array(TRUE_SKY)),   # fixed (no prior attached)
    GainOperator(gain=jnp.array(1.0)),
)
space = ParameterSpace.direct(
    "gain", init=1.0, into=lambda p: p["gain"].gain, prior=dist.Normal(1.0, 0.3),
)
model = to_numpyro_model(twin, state, space, noise_std=SIGMA)

mcmc = numpyro.infer.MCMC(
    numpyro.infer.NUTS(model), num_warmup=300, num_samples=300,
    progress_bar=False,
)
mcmc.run(jax.random.key(0), observed=observed)
samples = mcmc.get_samples()
gain_mean = float(samples["gain"].mean())
gain_std = float(samples["gain"].std())
print(f"NUTS posterior:   gain = {gain_mean:.4f} +/- {gain_std:.4f}  "
      f"(truth {TRUE_GAIN})")

# --------------------------------------------------------- Fisher forecast --
forward, params0 = space.forward_fn(twin, state)

F = fisher_information(forward, params0, noise_std=SIGMA)
cov = parameter_covariance(F)
print(f"Fisher forecast:  sigma(gain) = {float(jnp.sqrt(cov.matrix[0, 0])):.4f}  "
      "(Cramer-Rao — matches the posterior width for this near-linear model)")

# ------------------------------------------------- prediction uncertainty --
band = propagate_covariance(forward, params0, cov)
predictive = predict_from_samples(twin, state, space, samples)
print(f"delta-method prediction std (first channel): {float(band[0, 0]):.4f}")
print(f"posterior-predictive std   (first channel): "
      f"{float(predictive[:, 0, 0].std()):.4f}")
