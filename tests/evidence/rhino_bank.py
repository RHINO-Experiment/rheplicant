"""A RHINO-like band, its prior bank, and the noise that weights it.

Not a test file -- imported by six of them, which is why the repo root has to be
on sys.path (`python -m pytest`, never bare `pytest`).

The numbers are the ones section 5 is about. A ~200 K foreground three orders of
magnitude above a ~0.2 K trough is what makes a plain SVD order the science
direction last; a sigma that varies across the band by a measured factor of
2.389 is what makes the metric a real choice rather than a formality. Under a
constant sigma every test built on this fixture would hold vacuously.

The latents are declared in a NON-alphabetical order on purpose. `jax` sorts a
dict's keys when it flattens, so anything that builds a named matrix in declared
order and labels it in flatten order is wrong by a permutation -- and that
permutation is the identity for an alphabetical fixture, which is how the same
bug survived Plan A's `BayesMemory.fisher()` tests.
"""

import jax
import jax.numpy as jnp

from rheplicant.core.coordinates import Coordinates
from rheplicant.core.operator import AbstractOperator
from rheplicant.core.state import State
from rheplicant.inference import Bind, Latent, ParameterSpace
from rheplicant.inference.noise import RadiometerNoise

N_FREQ = 128
FREQ_LO, FREQ_HI = 60e6, 85e6
NU_REF = 75e6
TROUGH_CENTRE, TROUGH_WIDTH = 78e6, 3.0e6

#: RHINO's channel width and per-sample integration, from the spec's section 3
#: worked example. Their product sets f = 1/sqrt(dnu tau) = 3.7722e-4, so f^2 =
#: 1.4230e-7 -- D21's multiplicative GLS bias factor for this band.
CHANNEL_WIDTH, INTEGRATION_TIME = 24.4e3, 288.0

TRUTH = {
    "t21_depth": -0.2,
    "index": -2.5,
    "amplitude": 200.0,
    "running": 0.0,
}
PRIOR_STD = {
    "t21_depth": 0.1,
    "index": 0.05,
    "amplitude": 20.0,
    "running": 0.05,
}


class _Normal:
    """Minimal duck-typed prior: only ``log_prob`` is read downstream."""

    def __init__(self, loc, scale):
        self.loc, self.scale = loc, scale

    def log_prob(self, x):
        return -0.5 * (
            ((x - self.loc) / self.scale) ** 2 + jnp.log(2 * jnp.pi * self.scale**2)
        )


class GlobalSkyOperator(AbstractOperator):
    """``T(nu) = A (nu/nu0)^(beta + gamma ln(nu/nu0)) + depth * trough(nu)``.

    A running spectral index, so the model is genuinely nonlinear in three of
    its four latents -- which is the regime T2 cannot serve and T1 exists for.
    """

    t21_depth: jax.Array
    index: jax.Array
    amplitude: jax.Array
    running: jax.Array

    def __call__(self, state: State) -> State:
        log_nu = jnp.log(state.coords.freq / NU_REF)
        foreground = self.amplitude * jnp.exp(
            (self.index + self.running * log_nu) * log_nu
        )
        trough = jnp.exp(
            -0.5 * ((state.coords.freq - TROUGH_CENTRE) / TROUGH_WIDTH) ** 2
        )
        return state.with_data(foreground + self.t21_depth * trough)


def pipeline(**overrides) -> GlobalSkyOperator:
    values = {**TRUTH, **overrides}
    return GlobalSkyOperator(**{k: jnp.asarray(v) for k, v in values.items()})


def state() -> State:
    return State(
        coords=Coordinates(freq=jnp.linspace(FREQ_LO, FREQ_HI, N_FREQ)),
        key=jax.random.key(0),
        meta={"telescope": "RHINO", "obs_id": "bank"},
    )


def space() -> ParameterSpace:
    latents = tuple(
        Latent(
            name,
            init=jnp.asarray(TRUTH[name]),
            prior=_Normal(TRUTH[name], PRIOR_STD[name]),
        )
        # NOT sorted: see the module docstring.
        for name in ("t21_depth", "index", "amplitude", "running")
    )
    return ParameterSpace(
        latents=latents,
        bindings=tuple(
            Bind(latent.name, into=lambda p, n=latent.name: getattr(p, n))
            for latent in latents
        ),
    )


def noise() -> RadiometerNoise:
    return RadiometerNoise(
        channel_width=CHANNEL_WIDTH, integration_time=INTEGRATION_TIME
    )


def forward():
    """``(forward(values) -> (N_FREQ,), values_at_truth)``."""
    return space().forward_fn(pipeline(), state())


def bank(key, n_draws=400):
    """``(n_draws, N_FREQ)`` predictions at draws from the declared priors."""
    predict, values = forward()
    names = tuple(values)
    draws = jax.random.normal(key, (n_draws, len(names)))
    scaled = {
        name: TRUTH[name] + PRIOR_STD[name] * draws[:, i]
        for i, name in enumerate(names)
    }
    return jax.vmap(lambda i: predict({n: scaled[n][i] for n in names}))(
        jnp.arange(n_draws)
    )


def weight():
    """``1/sigma`` at the truth -- the reference metric for every basis below."""
    predict, values = forward()
    sigma = noise().std(predict(values))
    # Select on isfinite BEFORE dividing, never divide and hope: `1/inf` is 0.0
    # but `1/0` is inf, and one inf in the weight makes every projection NaN.
    finite = jnp.isfinite(sigma) & (sigma > 0.0)
    return jnp.where(finite, 1.0 / jnp.where(finite, sigma, 1.0), 0.0)


def support():
    """The box `bank()` populates: truth +/- 3 prior sigma, per latent."""
    return {
        name: (TRUTH[name] - 3 * PRIOR_STD[name], TRUTH[name] + 3 * PRIOR_STD[name])
        for name in TRUTH
    }


def observed(key, **overrides):
    """One epoch of data: the truth (or an override) plus radiometer noise."""
    predict, values = forward()
    truth = predict({**values, **{k: jnp.asarray(v) for k, v in overrides.items()}})
    sigma = noise().std(truth)
    return truth + sigma * jax.random.normal(key, truth.shape)
