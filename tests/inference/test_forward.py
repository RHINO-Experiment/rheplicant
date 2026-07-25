"""Tests for the inference layer: forward-fn builder, likelihood, calibrator."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from replicant import Pipeline
from replicant.inference import (
    GaussianLikelihood,
    GradientCalibrator,
    build_forward_fn,
    mean_squared_error,
)
from replicant.radio import GainOperator, NoiseOperator, SkyOperator


@pytest.fixture
def pipeline():
    return Pipeline(
        SkyOperator(amplitude=jnp.array(1.0)),
        GainOperator(gain=jnp.array(1.5)),
        NoiseOperator(sigma=jnp.array(0.1)),
        names=("sky", "gain", "noise"),
    )


class TestBuildForwardFn:
    def test_matches_direct_run(self, pipeline, template_state):
        forward, params0 = build_forward_fn(pipeline, template_state)
        direct = pipeline(template_state).data
        assert jnp.array_equal(forward(params0), direct)

    def test_params_are_only_inexact_arrays(self, pipeline, template_state):
        _, params0 = build_forward_fn(pipeline, template_state)
        leaves = jax.tree.leaves(params0)
        assert len(leaves) == 3  # amplitude, gain, sigma
        assert all(jnp.issubdtype(leaf.dtype, jnp.inexact) for leaf in leaves)

    def test_grad_is_finite_and_nonzero(self, pipeline, template_state):
        forward, params0 = build_forward_fn(pipeline, template_state)
        observed = forward(params0)

        def loss(params):
            return mean_squared_error(forward(params), observed * 1.1)

        grads = jax.grad(loss)(params0)
        for leaf in jax.tree.leaves(grads):
            assert jnp.all(jnp.isfinite(leaf))
        assert any(jnp.any(leaf != 0) for leaf in jax.tree.leaves(grads))

    def test_custom_filter_spec_restricts_params(self, pipeline, template_state):
        """Only the gain should be trainable under a targeted filter_spec."""
        spec = jax.tree.map(lambda _: False, pipeline)
        spec = eqx.tree_at(lambda p: p["gain"].gain, spec, replace=True)
        forward, params0 = build_forward_fn(pipeline, template_state, filter_spec=spec)
        assert len(jax.tree.leaves(params0)) == 1
        # forward still runs: static side supplies the frozen leaves
        assert forward(params0).shape == pipeline(template_state).data.shape


class TestGaussianLikelihood:
    def test_matches_formula(self):
        lik = GaussianLikelihood(noise_std=jnp.array(2.0))
        pred, obs = jnp.zeros(3), jnp.array([1.0, -1.0, 2.0])
        expected = -0.5 * jnp.sum((obs / 2.0) ** 2 + jnp.log(2 * jnp.pi * 4.0))
        assert jnp.allclose(lik(pred, obs), expected)

    def test_perfect_prediction_is_most_likely(self):
        lik = GaussianLikelihood(noise_std=jnp.array(1.0))
        obs = jnp.array([1.0, 2.0])
        assert lik(obs, obs) > lik(obs + 0.5, obs)


class TestGradientCalibrator:
    def test_recovers_known_gain(self, template_state):
        """Noiseless toy problem: infer gain=2 starting from gain=1."""
        true = Pipeline(
            SkyOperator(amplitude=jnp.array(1.0)),
            GainOperator(gain=jnp.array(2.0)),
            names=("sky", "gain"),
        )
        observed = true(template_state).data

        model = true.replace_stage("gain", GainOperator(gain=jnp.array(1.0)))
        spec = jax.tree.map(lambda _: False, model)
        spec = eqx.tree_at(lambda p: p["gain"].gain, spec, replace=True)
        forward, params0 = build_forward_fn(model, template_state, filter_spec=spec)

        calibrator = GradientCalibrator(learning_rate=0.1, n_steps=100)
        params_fit, losses = calibrator.fit(forward, params0, observed)

        fitted_gain = jax.tree.leaves(params_fit)[0]
        assert jnp.allclose(fitted_gain, 2.0, atol=1e-3)
        assert losses[-1] < losses[0]
        assert jnp.all(jnp.diff(losses) <= 1e-12)  # loss non-increasing

    def test_invalid_config_rejected(self):
        with pytest.raises(ValueError):
            GradientCalibrator(learning_rate=-1.0)
        with pytest.raises(ValueError):
            GradientCalibrator(n_steps=0)


# The NumPyro bridge has its own suite: tests/inference/test_numpyro_bridge.py
