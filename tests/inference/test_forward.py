"""Tests for the inference layer: forward-fn builder, likelihood, calibrator."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from rheplicant import Pipeline
from rheplicant.core.errors import ParameterSpaceError
from rheplicant.inference import (
    GaussianLikelihood,
    GradientCalibrator,
    build_forward_fn,
    mean_squared_error,
)
from rheplicant.radio import GainOperator, NoiseOperator, SkyOperator


@pytest.fixture
def pipeline():
    """Deterministic, and 3 != 5 so a stage swap is a different number."""
    return Pipeline(
        SkyOperator(amplitude=jnp.array(3.0)),
        GainOperator(gain=jnp.array(5.0)),
        names=("sky", "gain"),
    )


@pytest.fixture
def stochastic_pipeline(pipeline):
    return Pipeline(
        *pipeline.stages, NoiseOperator(sigma=jnp.array(0.1)), names=("sky", "gain", "noise")
    )


class TestBuildForwardFn:
    def test_matches_direct_run(self, pipeline, template_state):
        forward, params0 = build_forward_fn(pipeline, template_state)
        direct = pipeline(template_state).data
        assert jnp.array_equal(forward(params0), direct)
        assert jnp.allclose(direct, 15.0)

    def test_params_are_only_inexact_arrays(self, pipeline, template_state):
        _, params0 = build_forward_fn(pipeline, template_state)
        leaves = jax.tree.leaves(params0)
        assert len(leaves) == 2  # amplitude, gain
        assert all(jnp.issubdtype(leaf.dtype, jnp.inexact) for leaf in leaves)

    def test_a_stochastic_stage_is_refused_by_name(
        self, stochastic_pipeline, template_state
    ):
        """The closure would freeze one noise draw and fit against it forever."""
        with pytest.raises(ParameterSpaceError, match="NoiseOperator at 'noise'"):
            build_forward_fn(stochastic_pipeline, template_state)

    def test_the_stochastic_pipeline_still_runs_as_a_simulator(
        self, stochastic_pipeline, template_state
    ):
        """Refusing a FIT TARGET must not refuse the twin that generates data."""
        first = stochastic_pipeline(template_state).data
        other = stochastic_pipeline(template_state.replace(key=jax.random.key(3))).data
        assert jnp.all(jnp.isfinite(first)) and not jnp.allclose(first, other)

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
