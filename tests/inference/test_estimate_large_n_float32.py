"""In float32, a large-N point estimate is either within 0.1 sigma or refused.

The float32 half of the T-002 review's HIGH. The relative change test
certified two collinear conjugate blocks 1.6 posterior sigma from the MAP at
``N = 1e4`` and 16.5 sigma at ``N = 1e6`` (reviewer's measurements, float32).
The gap certificate is in nats, and a decrease is only resolved to about
``eps * sqrt(sum term**2)``; when that is coarser than a 0.1-sigma
certificate needs at the measured contraction, the run refuses at
``max_iter`` with a message naming the resolution and float64 as the remedy.

This module runs in the suite's default float32; the float64 half, where
every cell must converge, is in ``test_estimate_reaches_map.py``, which
supplies the model and its closed-form MAP.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_estimate_reaches_map import (
    LARGE_N,
    MAHALANOBIS_MAX,
    TEMPLATE_SIGMA,
    _grid_state,
    _mahalanobis,
    _template_case,
    _template_map,
    _template_plan,
)

from rheplicant.core.errors import ParameterSpaceError
from rheplicant.inference import HomoscedasticNoise

pytest.importorskip("numpyro.distributions", reason="numpyro not installed")

#: Measured on this grid: the cells that must return (and did, within 0.04
#: sigma) and the cell that must refuse on resolution. The rest may do either,
#: which is the contract; these two keep the test from passing by always
#: refusing or by never refusing.
MUST_CONVERGE = {((100, 100), 1.0), ((250, 400), 1.0)}
MUST_REFUSE_ON_RESOLUTION = {((250, 400), 0.2)}


@pytest.mark.parametrize("shape", LARGE_N, ids=["N=1e4", "N=1e5"])
@pytest.mark.parametrize("eps", [1.0, 0.5, 0.2], ids=["r=0.86", "r=0.96", "r=0.993"])
def test_a_float32_estimate_is_within_a_tenth_of_a_sigma_or_refused(shape, eps):
    assert not jax.config.read("jax_enable_x64"), "this module's premise is float32"
    observed, *_ = _template_case(eps, 3.0, 21, shape=shape)
    data = np.asarray(observed, np.float32)
    # the MAP of the data the run actually sees: the float32 rounding of it
    _, exact, precision = _template_map(data, eps, 3.0)
    sd = np.sqrt(np.diag(np.linalg.inv(precision)))
    plan, pipeline = _template_plan(eps, 3.0, exact + 20.0 * sd, dtype=jnp.float32)
    try:
        estimate = plan.estimate(
            pipeline, _grid_state(*shape), jnp.asarray(data),
            noise=HomoscedasticNoise(sigma=jnp.array(TEMPLATE_SIGMA, jnp.float32)),
            max_iter=1000,
        )
    except ParameterSpaceError as refusal:
        message = str(refusal)
        assert "did not converge" in message, message
        assert (shape, eps) not in MUST_CONVERGE, message
        if (shape, eps) in MUST_REFUSE_ON_RESOLUTION:
            assert "resolved only to" in message and "JAX_ENABLE_X64=1" in message
        return
    assert (shape, eps) not in MUST_REFUSE_ON_RESOLUTION
    got = [float(estimate.values["a"]), float(estimate.values["b"])]
    distance = _mahalanobis(got, exact, precision)
    assert estimate.diagnostics.converged is True
    assert distance < MAHALANOBIS_MAX, (distance, estimate.diagnostics.sweeps)

