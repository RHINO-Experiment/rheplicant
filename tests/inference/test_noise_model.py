"""Tests for the noise-model seam: sigma, weights, and the log-determinant.

The analytic content of this file is the last class. When sigma depends on the
prediction, the Gaussian log-density's normalization is no longer a constant,
and dropping it — which is exactly what generalized least squares does — moves
the estimator. For the multiplicative model these tests use, both estimators
are available in closed form, so the difference is pinned rather than measured.
"""

import importlib.util

import jax
import jax.numpy as jnp
import pytest

from rheplicant.core.errors import StateValidationError
from rheplicant.inference.likelihood import (
    GaussianLikelihood,
    MaskedGaussianLikelihood,
)
from rheplicant.inference.noise import (
    FlaggedNoise,
    HomoscedasticNoise,
    NoiseModel,
    NoiseModelLikelihood,
    RadiometerNoise,
    inverse_variance,
)

N_DATA = 32
CHANNEL_WIDTH, INTEGRATION_TIME = 50.0, 2.0  # Hz, s -> fractional 0.1


@pytest.fixture
def prediction():
    return jnp.linspace(200.0, 400.0, N_DATA)


@pytest.fixture
def observed(prediction):
    noise = 0.05 * prediction * jax.random.normal(jax.random.key(3), prediction.shape)
    return prediction + noise


class TestHomoscedasticNoise:
    def test_std_is_the_sigma_broadcast_to_the_prediction(self, prediction):
        noise = HomoscedasticNoise(jnp.asarray(0.7))
        assert noise.std(prediction).shape == prediction.shape
        assert jnp.allclose(noise.std(prediction), 0.7)

    def test_it_does_not_depend_on_the_prediction(self, prediction):
        noise = HomoscedasticNoise(jnp.asarray(0.7))
        assert noise.depends_on_prediction is False
        assert jnp.allclose(noise.std(prediction), noise.std(2.0 * prediction))

    def test_a_per_sample_sigma_is_allowed(self, prediction):
        sigma = jnp.linspace(0.1, 1.0, N_DATA)
        assert jnp.allclose(HomoscedasticNoise(sigma).std(prediction), sigma)

    def test_it_satisfies_the_protocol(self):
        assert isinstance(HomoscedasticNoise(jnp.asarray(1.0)), NoiseModel)

    def test_an_array_does_not_satisfy_the_protocol(self):
        """jax and numpy arrays both HAVE a ``.std`` method, so a protocol
        keyed on that alone would swallow a bare sigma as a noise model and
        call it with the prediction. ``depends_on_prediction`` is the marker
        that keeps ``noise_std=`` polymorphic without being ambiguous."""
        assert not isinstance(jnp.ones(4), NoiseModel)
        assert not isinstance(0.5, NoiseModel)


class TestRadiometerNoise:
    def test_fractional_is_one_over_root_bandwidth_time(self):
        noise = RadiometerNoise(CHANNEL_WIDTH, INTEGRATION_TIME)
        assert jnp.allclose(noise.fractional, 1.0 / jnp.sqrt(50.0 * 2.0))

    def test_std_is_proportional_to_the_prediction(self, prediction):
        noise = RadiometerNoise(CHANNEL_WIDTH, INTEGRATION_TIME)
        assert jnp.allclose(noise.std(prediction), prediction * noise.fractional)

    def test_it_depends_on_the_prediction(self, prediction):
        noise = RadiometerNoise(CHANNEL_WIDTH, INTEGRATION_TIME)
        assert noise.depends_on_prediction is True
        doubled = noise.std(2.0 * prediction)
        assert jnp.allclose(doubled, 2.0 * noise.std(prediction))

    def test_a_negative_prediction_still_gives_a_positive_sigma(self):
        # An IRLS iterate can cross zero even where the physics cannot; a
        # negative sigma would flip the sign of every weight silently.
        noise = RadiometerNoise(CHANNEL_WIDTH, INTEGRATION_TIME)
        assert jnp.all(noise.std(jnp.array([-100.0, -1.0, 1.0])) > 0.0)

    def test_the_floor_bounds_sigma_away_from_zero(self):
        noise = RadiometerNoise(CHANNEL_WIDTH, INTEGRATION_TIME, floor=10.0)
        assert jnp.allclose(noise.std(jnp.array([0.0])), 10.0 * noise.fractional)

    def test_zero_prediction_without_a_floor_gives_zero_sigma(self):
        # Documented, not guarded: 1/sigma^2 is then infinite, which is the
        # loud failure. The floor exists for callers that need the quiet one.
        noise = RadiometerNoise(CHANNEL_WIDTH, INTEGRATION_TIME)
        assert jnp.allclose(noise.std(jnp.array([0.0])), 0.0)

    def test_a_nonpositive_bandwidth_is_refused(self):
        with pytest.raises(StateValidationError, match="positive"):
            RadiometerNoise(0.0, INTEGRATION_TIME)
        with pytest.raises(StateValidationError, match="positive"):
            RadiometerNoise(CHANNEL_WIDTH, -1.0)

    @pytest.mark.skipif(
        importlib.util.find_spec("rhino_cal_jax") is None,
        reason="rhino_cal_jax not installed",
    )
    def test_sigma_agrees_with_the_simulation_side(self, prediction):
        """The statistics here and the physics there must use the same sigma.

        ``rhino_cal_jax.power.add_radiometer_noise`` is what *generates* the
        data; this model is what *scores* it. A mismatch would show up as a
        perfectly plausible, wrongly-weighted fit.
        """
        import numpy as np
        from rhino_cal_jax.power import add_radiometer_noise

        noise = RadiometerNoise(CHANNEL_WIDTH, INTEGRATION_TIME)
        keys = jax.random.split(jax.random.key(0), 4000)
        draws = jax.vmap(
            lambda k: add_radiometer_noise(
                prediction, k, t_int=INTEGRATION_TIME, delta_nu=CHANNEL_WIDTH
            )
        )(keys)
        empirical = draws.std(axis=0)
        np.testing.assert_allclose(empirical, noise.std(prediction), rtol=0.05)


class TestFlaggedNoise:
    def test_flagged_samples_get_infinite_sigma(self, prediction):
        flags = jnp.arange(N_DATA) % 4 == 0
        noise = FlaggedNoise(HomoscedasticNoise(jnp.asarray(0.5)), flags)
        sigma = noise.std(prediction)
        assert jnp.all(jnp.isinf(sigma[flags]))
        assert jnp.allclose(sigma[~flags], 0.5)

    def test_it_delegates_prediction_dependence(self, prediction):
        flags = jnp.zeros(N_DATA, dtype=bool)
        base = RadiometerNoise(CHANNEL_WIDTH, INTEGRATION_TIME)
        assert FlaggedNoise(base, flags).depends_on_prediction is True
        assert (
            FlaggedNoise(HomoscedasticNoise(jnp.asarray(1.0)), flags)
            .depends_on_prediction
            is False
        )

    def test_wrapping_twice_takes_the_union(self, prediction):
        first = jnp.arange(N_DATA) % 4 == 0
        second = jnp.arange(N_DATA) % 3 == 0
        noise = FlaggedNoise(
            FlaggedNoise(HomoscedasticNoise(jnp.asarray(0.5)), first), second
        )
        bad = jnp.isinf(noise.std(prediction))
        assert jnp.array_equal(bad, first | second)

    def test_a_mismatched_flag_shape_is_refused(self, prediction):
        noise = FlaggedNoise(HomoscedasticNoise(jnp.asarray(0.5)), jnp.ones(5, bool))
        with pytest.raises(StateValidationError, match="shape"):
            noise.std(prediction)


class TestInverseVariance:
    def test_it_is_one_over_sigma_squared(self, prediction):
        noise = HomoscedasticNoise(jnp.asarray(0.5))
        assert jnp.allclose(inverse_variance(noise, prediction), 1.0 / 0.25)

    def test_an_unobserved_sample_carries_zero_weight(self, prediction):
        flags = jnp.arange(N_DATA) % 4 == 0
        noise = FlaggedNoise(HomoscedasticNoise(jnp.asarray(0.5)), flags)
        weights = inverse_variance(noise, prediction)
        assert jnp.all(weights[flags] == 0.0)
        assert jnp.all(jnp.isfinite(weights))


class TestNoiseModelLikelihood:
    def test_homoscedastic_reproduces_the_existing_gaussian_exactly(
        self, prediction, observed
    ):
        old = GaussianLikelihood(jnp.asarray(0.5))(prediction, observed)
        new = NoiseModelLikelihood(HomoscedasticNoise(jnp.asarray(0.5)))(
            prediction, observed
        )
        assert jnp.allclose(old, new, rtol=1e-6)

    def test_flagged_reproduces_the_existing_masked_gaussian_exactly(
        self, prediction, observed
    ):
        flags = jnp.arange(N_DATA) % 4 == 0
        old = MaskedGaussianLikelihood(jnp.asarray(0.5), flags)(prediction, observed)
        new = NoiseModelLikelihood(
            FlaggedNoise(HomoscedasticNoise(jnp.asarray(0.5)), flags)
        )(prediction, observed)
        assert jnp.allclose(old, new, rtol=1e-6)

    def test_an_unobserved_sample_cannot_poison_the_total(self, prediction):
        """inf * 0 is NaN; a masked sample must contribute a clean zero."""
        flags = jnp.arange(N_DATA) % 4 == 0
        observed = prediction.at[0].set(jnp.inf)  # arbitrarily bad, and flagged
        noise = FlaggedNoise(HomoscedasticNoise(jnp.asarray(0.5)), flags)
        assert jnp.isfinite(NoiseModelLikelihood(noise)(prediction, observed))

    def test_dropping_the_logdet_changes_nothing_for_constant_sigma(self, prediction):
        # Residuals of order sigma, deliberately: with the fixture's 5% noise
        # on a ~300 K prediction the chi-squared term is ~1e4, and taking the
        # difference of two float32 numbers that size loses the ~7 nat answer
        # to cancellation long before the implementation gets a say.
        noise = HomoscedasticNoise(jnp.asarray(0.5))
        observed = prediction + 0.5 * jnp.sin(jnp.arange(N_DATA, dtype=float))
        full = NoiseModelLikelihood(noise)(prediction, observed)
        gls = NoiseModelLikelihood(noise, include_logdet=False)(prediction, observed)
        assert jnp.allclose(full - gls, -0.5 * N_DATA * jnp.log(2 * jnp.pi * 0.25))

    def test_it_is_jittable_and_differentiable(self, prediction, observed):
        noise = RadiometerNoise(CHANNEL_WIDTH, INTEGRATION_TIME)
        like = NoiseModelLikelihood(noise)
        grad = jax.jit(jax.grad(lambda p: like(p, observed)))(prediction)
        assert grad.shape == prediction.shape
        assert jnp.all(jnp.isfinite(grad))


class TestTheLogDetIsNotAConstant:
    """The whole reason the seam exists, in closed form.

    For ``d_i = theta (1 + w_i)`` with ``w ~ N(0, f^2)`` the prediction is the
    scalar ``theta`` repeated, and ``sigma_i = theta f``. Both estimators are
    then solvable by hand:

    * GLS (log-det dropped) minimizes ``sum (d_i - theta)^2 / (theta f)^2``,
      whose stationary point is ``theta = S2 / S1`` — independent of ``f``, and
      with expectation ``theta_true (1 + f^2)``: biased HIGH.
    * The full Gaussian adds ``2 n log theta``, giving
      ``n f^2 theta^2 + S1 theta - S2 = 0``. Substituting the expectations
      ``E[S1] = n theta_t`` and ``E[S2] = n theta_t^2 (1 + f^2)`` satisfies it
      exactly at ``theta = theta_t``: asymptotically UNBIASED.

    So the term GLS discards as a normalization constant is precisely the term
    that removes the bias. These tests assert the implemented log-density has a
    vanishing gradient at each closed form, which checks the implementation
    without ever running an optimizer.
    """

    FRACTIONAL = 0.1  # 1/sqrt(50 * 2); large enough that (1 + f^2) is visible
    THETA_TRUE = 300.0
    N = 4096

    @pytest.fixture
    def data(self):
        w = self.FRACTIONAL * jax.random.normal(jax.random.key(11), (self.N,))
        return self.THETA_TRUE * (1.0 + w)

    @staticmethod
    def _predict(theta, n):
        return jnp.full((n,), theta)

    def _logp(self, theta, d, *, include_logdet):
        like = NoiseModelLikelihood(
            RadiometerNoise(CHANNEL_WIDTH, INTEGRATION_TIME),
            include_logdet=include_logdet,
        )
        return like(self._predict(theta, d.shape[0]), d)

    def test_the_gls_optimum_is_the_ratio_of_the_first_two_moments(self, data):
        theta_gls = jnp.sum(data**2) / jnp.sum(data)
        slope = jax.grad(lambda t: self._logp(t, data, include_logdet=False))(theta_gls)
        assert abs(float(slope)) < 1e-3 * self.N

    def test_the_full_optimum_solves_the_quadratic_the_logdet_adds(self, data):
        s1, s2 = jnp.sum(data), jnp.sum(data**2)
        a = data.shape[0] * self.FRACTIONAL**2
        theta_full = (-s1 + jnp.sqrt(s1**2 + 4 * a * s2)) / (2 * a)
        slope = jax.grad(lambda t: self._logp(t, data, include_logdet=True))(theta_full)
        assert abs(float(slope)) < 1e-3 * self.N

    def test_the_logdet_pulls_the_estimate_down(self, data):
        s1, s2 = jnp.sum(data), jnp.sum(data**2)
        a = data.shape[0] * self.FRACTIONAL**2
        theta_gls = s2 / s1
        theta_full = (-s1 + jnp.sqrt(s1**2 + 4 * a * s2)) / (2 * a)
        assert theta_full < theta_gls

    def test_gls_is_biased_high_by_one_plus_f_squared_and_the_full_one_is_not(self):
        """The bias is a factor, so it survives averaging over realizations."""
        f, n = self.FRACTIONAL, 1024
        a = n * f**2

        def estimates(key):
            d = self.THETA_TRUE * (1.0 + f * jax.random.normal(key, (n,)))
            s1, s2 = jnp.sum(d), jnp.sum(d**2)
            return s2 / s1, (-s1 + jnp.sqrt(s1**2 + 4 * a * s2)) / (2 * a)

        gls, full = jax.vmap(estimates)(jax.random.split(jax.random.key(5), 400))
        assert jnp.allclose(gls.mean() / self.THETA_TRUE, 1.0 + f**2, rtol=0.02)
        assert jnp.allclose(full.mean() / self.THETA_TRUE, 1.0, rtol=0.005)
