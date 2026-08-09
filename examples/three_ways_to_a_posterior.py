"""One posterior, three engines: exact, gradient-sampled, and likelihood-free.

The same twin, the same ParameterSpace and the same noise model, handed to:

  1. gcr_sample  -- EXACT. The model is linear in the gain with a Gaussian
     prior and Gaussian noise, so the posterior is conjugate and every draw is
     independent. No chain, no burn-in, nothing to diagnose.
  2. NUTS        -- gradient MCMC through the NumPyro bridge. Makes no use of
     the conjugacy, so it must reproduce (1) if both are right.
  3. NPE         -- amortized neural posterior estimation. Never writes a
     likelihood at all: it fits q(theta | x) to pairs drawn from the prior and
     the simulator.

The order is the point. (1) is the reference; (2) is checked against it; (3)
is checked against both. An approximate posterior has no internal notion of
being wrong -- a badly-trained estimator returns a smooth, confident,
correctly-centred, incorrect distribution -- so validating on a problem with a
closed-form answer is not a formality, it is the only thing that catches it.

Run:  .venv/bin/python examples/three_ways_to_a_posterior.py
"""

import time

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from rheplicant import Coordinates, Environment, State
from rheplicant.core.pipeline import Pipeline
from rheplicant.inference import (
    HomoscedasticNoise,
    NeuralPosterior,
    ParameterSpace,
    gcr_sample,
    linear_operator,
    simulate_pairs,
    to_numpyro_model,
    train_posterior,
    wiener_solve,
)
from rheplicant.radio import GainOperator, SkyOperator

N_TIME, N_FREQ = 8, 4
SKY, TRUE_GAIN, SIGMA = 100.0, 1.1, 5.0
PRIOR_MEAN, PRIOR_STD = 1.0, 0.05
N_SIMULATIONS, N_STEPS = 32768, 2000

state = State(
    coords=Coordinates(
        time=jnp.linspace(0.0, 7.0, N_TIME), freq=jnp.linspace(60e6, 85e6, N_FREQ)
    ),
    env=Environment(temperature=jnp.array(280.0)),
    key=jax.random.key(0),
    meta={"telescope": "RHINO"},
)


def twin_at(gain):
    return Pipeline(
        SkyOperator(amplitude=jnp.array(SKY)),
        GainOperator(gain=jnp.array(gain)),
        names=("sky", "gain"),
    )


twin = twin_at(1.0)  # the model starts mis-calibrated
space = ParameterSpace.direct(
    "gain", init=1.0, into=lambda p: p["gain"].gain,
    prior=dist.Normal(PRIOR_MEAN, PRIOR_STD), linear=True,
)
noise = HomoscedasticNoise(jnp.asarray(SIGMA))

truth = twin_at(TRUE_GAIN)(state).data
observed = truth + SIGMA * jax.random.normal(jax.random.key(99), truth.shape)
print(f"truth  gain = {TRUE_GAIN}    prior  N({PRIOR_MEAN}, {PRIOR_STD})")
print(f"data   {observed.shape}, sigma = {SIGMA} K\n")


def report(name, mean, std, seconds, reference=None):
    line = f"{name:<26s} {mean:9.5f}  +/- {std:8.5f}   {seconds:6.2f} s"
    if reference is not None:
        ref_mean, ref_std = reference
        line += (f"   mean {abs(mean - ref_mean) / ref_std:5.2f} sigma, "
                 f"width {std / ref_std:5.2f}x")
    print(line)


print(f"{'engine':<26s} {'mean':>9s}  {'width':>12s}   {'time':>8s}   vs exact")

# ---------------------------------------------------------------- 1. exact
start = time.perf_counter()
block = linear_operator(space, twin, state)
exact_mean, _ = wiener_solve(
    block, observed, noise_std=SIGMA, prior_std=PRIOR_STD, prior_mean=PRIOR_MEAN
)
exact_draws = jax.vmap(
    lambda k: gcr_sample(
        block, observed, noise_std=SIGMA, prior_std=PRIOR_STD,
        prior_mean=PRIOR_MEAN, key=k,
    )[0]
)(jax.random.split(jax.random.key(4), 4000))
exact = (float(exact_mean), float(exact_draws.std()))
report("1. gcr_sample (exact)", *exact, time.perf_counter() - start)

# ------------------------------------------------------------------ 2. NUTS
start = time.perf_counter()
model = to_numpyro_model(twin, state, space, noise_std=noise)
mcmc = numpyro.infer.MCMC(
    numpyro.infer.NUTS(model), num_warmup=800, num_samples=3000, progress_bar=False
)
mcmc.run(jax.random.key(0), observed=observed)
chain = mcmc.get_samples()["gain"]
report("2. NUTS (gradient MCMC)", float(chain.mean()), float(chain.std()),
       time.perf_counter() - start, exact)

# ------------------------------------------------------------------- 3. NPE
start = time.perf_counter()
thetas, bank = simulate_pairs(
    twin, state, space, noise=noise,
    key=jax.random.key(0), n_simulations=N_SIMULATIONS,
)
q = NeuralPosterior.create(
    thetas, bank, key=jax.random.key(1), n_components=1, width=64, depth=2
)
q, history = train_posterior(
    q, thetas, bank, key=jax.random.key(2),
    n_steps=N_STEPS, batch_size=512, learning_rate=2e-3,
)
train_seconds = time.perf_counter() - start
npe_draws = q.sample(observed, jax.random.key(3), 4000)
report("3. NPE (likelihood-free)", float(npe_draws.mean()), float(npe_draws.std()),
       train_seconds, exact)

print(f"\nNPE training: {N_SIMULATIONS} simulations, {N_STEPS} steps, best "
      f"validation at step {int(history.best_step)}")
print(f"   train loss {float(history.train[-100:].mean()):8.3f}   "
      f"validation {float(history.validation[-100:].mean()):8.3f}")

# --------------------------------------------------------- what amortized means
# A second observation costs one forward pass through the trained network. The
# other two engines start from scratch: another CG solve, another chain.
other = twin_at(1.05)(state).data
other = other + SIGMA * jax.random.normal(jax.random.key(7), other.shape)

start = time.perf_counter()
again = q.sample(other, jax.random.key(8), 4000)
npe_seconds = time.perf_counter() - start

start = time.perf_counter()
reference, _ = wiener_solve(
    block, other, noise_std=SIGMA, prior_std=PRIOR_STD, prior_mean=PRIOR_MEAN
)
exact_seconds = time.perf_counter() - start

print("\nA SECOND observation (truth 1.05), no retraining:")
print(f"   NPE        {float(again.mean()):.5f}  in {npe_seconds * 1e3:6.1f} ms")
print(f"   exact      {float(reference):.5f}  in {exact_seconds * 1e3:6.1f} ms")
print("\nThat is what amortization buys, and what it costs: the training above")
print("is paid once, but the answer is only ever as good as the fit -- and the")
print("fit reports nothing when it is wrong. Hence the exact column.")
