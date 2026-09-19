"""In float32, a point estimate is within 0.1 sigma of the MAP or refused.

The float32 half of the T-002 reviews. The relative change test certified two
collinear conjugate blocks 1.6 posterior sigma from the MAP at ``N = 1e4``
and 16.5 sigma at ``N = 1e6`` (first review, float32). The gap certificate
that replaced it refused good float32 answers instead, because it could only
certify a decrease the arithmetic resolves, and its resolution was 50 to 1500
times coarser than the distance it stood for (second review). The Newton
decrement measures the distance at the returned point, so a float32 run
certifies whenever its gradient is resolved, however little the objective
moves between sweeps; it refuses where the gradient itself is rounding, and
names float64 as the remedy.

This module runs in the suite's default float32. The references are the MAP
of the data the run actually sees (their float32 rounding), computed in
float64: from :mod:`test_estimate_reaches_map`'s closed form for the linear
model, and by NumPy Newton for the power law, since this module cannot turn
on x64.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_estimate_reaches_map import (
    A_PRIOR,
    A_TRUE,
    BETA_PRIOR,
    BETA_TRUE,
    LARGE_N,
    MAHALANOBIS_MAX,
    REF_FREQ,
    TEMPLATE_SIGMA,
    _grid_state,
    _mahalanobis,
    _template_case,
    _template_map,
    _template_plan,
)

from rheplicant import Coordinates, State
from rheplicant.core.errors import ParameterSpaceError
from rheplicant.core.pipeline import Pipeline
from rheplicant.inference import (
    Bind,
    Block,
    HomoscedasticNoise,
    Latent,
    ParameterSpace,
    SamplingPlan,
)
from rheplicant.radio import ForegroundOperator

dist = pytest.importorskip("numpyro.distributions", reason="numpyro not installed")


@pytest.mark.parametrize("shape", LARGE_N, ids=["N=1e4", "N=1e5"])
@pytest.mark.parametrize("eps", [1.0, 0.5, 0.2], ids=["r=0.86", "r=0.96", "r=0.993"])
def test_a_float32_collinear_pair_is_certified_within_a_tenth_of_a_sigma(shape, eps):
    """Every cell converges, within 0.1 sigma. Under the gap certificate
    ``N = 1e5, r = 0.993`` refused on resolution."""
    assert not jax.config.read("jax_enable_x64"), "this module's premise is float32"
    observed, *_ = _template_case(eps, 3.0, 21, shape=shape)
    data = np.asarray(observed, np.float32)
    _, exact, precision = _template_map(data, eps, 3.0)
    sd = np.sqrt(np.diag(np.linalg.inv(precision)))
    plan, pipeline = _template_plan(eps, 3.0, exact + 20.0 * sd, dtype=jnp.float32)
    estimate = plan.estimate(
        pipeline, _grid_state(*shape), jnp.asarray(data),
        noise=HomoscedasticNoise(sigma=jnp.array(TEMPLATE_SIGMA, jnp.float32)),
        max_iter=1000,
    )
    got = [float(estimate.values["a"]), float(estimate.values["b"])]
    distance = _mahalanobis(got, exact, precision)
    assert estimate.diagnostics.converged is True
    assert distance < MAHALANOBIS_MAX, (distance, estimate.diagnostics.sweeps)


def _power_law_map(freq, observed, sigma):
    """The power law's joint MAP and precision, NumPy Newton in float64.

    ``mu = A x**-beta`` with ``x = freq / REF_FREQ`` and the Normal priors of
    :mod:`test_estimate_reaches_map`; the Hessian keeps the residual terms,
    so it is the objective's own and not Gauss-Newton.
    """
    x = np.asarray(freq, np.float64) / REF_FREQ
    data = np.asarray(observed, np.float64).ravel()
    log_x = np.log(x)

    def derivatives(params):
        amplitude, beta = params
        power = x ** (-beta)
        residual = (data - amplitude * power) / sigma
        d_amplitude, d_beta = power, -amplitude * log_x * power
        gradient = np.array([
            -(residual @ d_amplitude) / sigma + (amplitude - A_PRIOR[0]) / A_PRIOR[1] ** 2,
            -(residual @ d_beta) / sigma + (beta - BETA_PRIOR[0]) / BETA_PRIOR[1] ** 2,
        ])
        cross = d_amplitude @ d_beta / sigma**2 + (residual @ (log_x * power)) / sigma
        curvature = (d_beta @ d_beta / sigma**2
                     - (residual @ (amplitude * log_x**2 * power)) / sigma
                     + 1.0 / BETA_PRIOR[1] ** 2)
        hessian = np.array([
            [d_amplitude @ d_amplitude / sigma**2 + 1.0 / A_PRIOR[1] ** 2, cross],
            [cross, curvature],
        ])
        return gradient, hessian

    params = np.array([A_TRUE, BETA_TRUE])
    for _ in range(60):
        gradient, hessian = derivatives(params)
        params = params - np.linalg.solve(hessian, gradient)
    gradient, hessian = derivatives(params)
    step = np.linalg.solve(hessian, gradient)
    assert np.sqrt(step @ hessian @ step) < 1e-8, "the reference Newton solve stalled"
    return params, hessian


def _power_law_run(n_freq, sigma):
    """A run of the power law in float32, and the float64 reference."""
    freq = np.linspace(60e6, 85e6, n_freq)
    rng = np.random.default_rng(2026)
    data = A_TRUE * (freq / REF_FREQ) ** (-BETA_TRUE) + sigma * rng.standard_normal(n_freq)
    data = data.astype(np.float32)
    exact, precision = _power_law_map(freq, data, sigma)

    def f32(value):
        return jnp.array(value, jnp.float32)

    state = State(coords=Coordinates(time=jnp.arange(1.0),
                                     freq=jnp.asarray(freq, jnp.float32)))
    pipeline = Pipeline(
        ForegroundOperator(amplitude=f32(A_TRUE), spectral_index=f32(BETA_TRUE),
                           ref_freq=REF_FREQ),
        names=("fg",),
    )
    space = ParameterSpace(
        latents=[
            Latent("A", init=f32(900.0), prior=dist.Normal(*A_PRIOR), linear=True),
            Latent("beta", init=f32(2.5), prior=dist.Normal(*BETA_PRIOR)),
        ],
        bindings=[
            Bind("A", into=lambda p: p["fg"].amplitude),
            Bind("beta", into=lambda p: p["fg"].spectral_index),
        ],
    )
    plan = SamplingPlan(space, Block("A"), Block("beta"))

    def run():
        return plan.estimate(
            pipeline, state, jnp.asarray(data[None, :]),
            noise=HomoscedasticNoise(sigma=f32(sigma)), max_iter=400,
        )

    return run, exact, precision


@pytest.mark.parametrize("n_freq", [512, 4096])
@pytest.mark.parametrize("sigma", [1.0, 0.1])
def test_a_float32_power_law_converges_within_a_tenth_of_a_sigma(n_freq, sigma):
    """The second review's MEDIUM: these float32 runs are correct and the gap
    certificate refused them, because it could not resolve their decreases.
    The decrement certifies them where they stand."""
    run, exact, precision = _power_law_run(n_freq, sigma)
    estimate = run()
    got = [float(estimate.values["A"]), float(estimate.values["beta"])]
    distance = _mahalanobis(got, exact, precision)
    assert estimate.diagnostics.converged is True
    assert distance < MAHALANOBIS_MAX, (distance, estimate.diagnostics.sweeps)


def test_a_float32_run_below_its_rounding_is_refused_and_told_float64():
    """At 0.01 K a posterior sigma of the spectral index is 4.7e-6, 20 float32
    ulps of 2.55. The sweep does not settle: the objective moves by rounding,
    rises included, at every sweep, so no sweep is a candidate. With the
    review's data (seed 2027) it settles 0.44 sigma from the MAP instead and
    the decrement refuses every candidate. Either way the refusal names
    float64."""
    run, _, _ = _power_law_run(512, 0.01)
    with pytest.raises(ParameterSpaceError) as refused:
        run()
    message = str(refused.value)
    assert "did not converge" in message and "JAX_ENABLE_X64=1" in message
