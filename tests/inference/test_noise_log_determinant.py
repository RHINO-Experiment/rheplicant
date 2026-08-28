"""``log_determinant`` and ``NoiseModelLikelihood`` must not drift apart.

``sum log sigma`` is now spelled twice in this package -- once inside
:class:`~rheplicant.inference.noise.NoiseModelLikelihood`, which carries the
full Gaussian normalization, and once in
:func:`~rheplicant.inference.noise.log_determinant`, which drops the constant
``0.5 n log 2 pi`` so that a gradient block's potential keeps its magnitude
small (see that function for the measured cost of not dropping it).

Two spellings of one rule is the arrangement this codebase's notes single out
as the one that goes stale, because nothing renders them side by side. This
module is what renders them side by side: the exact relation between the two
is asserted, so a change to either that is not matched in the other goes red
here rather than in whichever of the two happened to be exercised.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rheplicant.inference.noise import (
    FlaggedNoise,
    HomoscedasticNoise,
    NoiseModelLikelihood,
    RadiometerNoise,
    log_determinant,
)

N_TIME, N_FREQ = 4, 10
N = N_TIME * N_FREQ

PREDICTION = jnp.abs(
    2.0 + jax.random.normal(jax.random.key(1), (N_TIME, N_FREQ))
) + 0.5
OTHER = PREDICTION * 1.7
OBSERVED = PREDICTION + 0.1 * jax.random.normal(jax.random.key(2), (N_TIME, N_FREQ))

FLAGS = jnp.zeros((N_TIME, N_FREQ), dtype=bool).at[0, ::3].set(True)

CONSTANT = HomoscedasticNoise(0.3)
VARYING = RadiometerNoise(channel_width=4.0, integration_time=1.0)

MODELS = [
    pytest.param(CONSTANT, N, id="homoscedastic"),
    pytest.param(VARYING, N, id="radiometer"),
    pytest.param(FlaggedNoise(CONSTANT, FLAGS), N - 4, id="flagged-homoscedastic"),
    pytest.param(FlaggedNoise(VARYING, FLAGS), N - 4, id="flagged-radiometer"),
]


@pytest.mark.parametrize("noise, n_seen", MODELS)
def test_it_is_the_likelihoods_own_logdet_without_the_2pi(noise, n_seen):
    """The exact relation, not an approximation of it.

    ``include_logdet`` toggles ``0.5 sum log 2 pi sigma^2`` inside the
    likelihood, so the difference between the two settings is that term with a
    sign, and ``log_determinant`` is the same term less ``0.5 n log 2 pi``
    over the OBSERVED samples.
    """
    full = float(NoiseModelLikelihood(noise)(PREDICTION, OBSERVED))
    gls = float(NoiseModelLikelihood(noise, include_logdet=False)(PREDICTION, OBSERVED))
    expected = -(full - gls) - 0.5 * n_seen * np.log(2.0 * np.pi)
    assert float(log_determinant(noise, PREDICTION)) == pytest.approx(
        expected, rel=1e-5, abs=1e-5
    )


@pytest.mark.parametrize("noise, n_seen", MODELS)
def test_the_observed_count_the_relation_assumes_is_the_real_one(noise, n_seen):
    """Anti-vacuity for the case above: ``n_seen`` is asserted, not assumed.

    The relation carries ``0.5 n log 2 pi``, so a wrong ``n`` would be absorbed
    into the tolerance for a small enough error and would hide a masking bug.
    """
    assert int(jnp.sum(jnp.isfinite(noise.std(PREDICTION)))) == n_seen


class TestWhetherItMovesWithThePrediction:
    """The property that decides whether dropping it changes an estimator.

    Constant under a prediction-independent sigma -- so invisible to any
    gradient, which is why B1 could hide -- and NOT constant otherwise, which
    is the anti-vacuity twin without which the first case says nothing.
    """

    def test_a_constant_sigma_gives_a_constant(self):
        here = float(log_determinant(CONSTANT, PREDICTION))
        there = float(log_determinant(CONSTANT, OTHER))
        assert here == pytest.approx(there, rel=1e-6)

    def test_a_prediction_dependent_sigma_does_not(self):
        here = float(log_determinant(VARYING, PREDICTION))
        there = float(log_determinant(VARYING, OTHER))
        assert abs(here - there) > 1.0, (
            f"the two predictions differ by only {abs(here - there)} nats, so "
            "this fixture cannot tell a dropped log-determinant from a kept one"
        )
        # ...and by the analytic amount: sigma scales with the prediction.
        assert there - here == pytest.approx(N * np.log(1.7), rel=1e-4)


def test_a_flagged_sample_contributes_nothing_rather_than_infinity():
    """``log(inf)`` is ``inf``; one flagged channel must not take the sum."""
    flagged = FlaggedNoise(VARYING, FLAGS)
    value = log_determinant(flagged, PREDICTION)
    assert jnp.isfinite(value)
    kept = jnp.sum(jnp.where(FLAGS, 0.0, jnp.log(VARYING.std(PREDICTION))))
    assert float(value) == pytest.approx(float(kept), rel=1e-5)
