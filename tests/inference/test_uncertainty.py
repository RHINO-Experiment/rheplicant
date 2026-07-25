"""Tests for uncertainty propagation: Fisher, delta method, pushforward."""

import jax
import jax.numpy as jnp
import pytest

from replicant.core.errors import StateValidationError
from replicant.inference.uncertainty import (
    fisher_information,
    parameter_covariance,
    propagate_covariance,
    push_forward,
)

N_DATA, N_PAR = 12, 3


@pytest.fixture
def linear_problem():
    """forward(theta) = A @ theta — Fisher and delta method are EXACT here."""
    key = jax.random.key(0)
    A = jax.random.normal(key, (N_DATA, N_PAR))
    theta0 = jnp.array([1.0, -2.0, 0.5])

    def forward(theta):
        return A @ theta

    return forward, theta0, A


class TestFisher:
    def test_matches_analytic_linear(self, linear_problem):
        forward, theta0, A = linear_problem
        sigma = 0.3
        F = fisher_information(forward, theta0, noise_std=sigma)
        expected = A.T @ A / sigma**2
        assert jnp.allclose(F.matrix, expected, rtol=1e-5)

    def test_flags_zero_out_samples(self, linear_problem):
        forward, theta0, A = linear_problem
        flags = jnp.zeros(N_DATA, bool).at[:4].set(True)
        F = fisher_information(forward, theta0, noise_std=1.0, flags=flags)
        expected = A[4:].T @ A[4:]
        assert jnp.allclose(F.matrix, expected, rtol=1e-5)

    def test_flags_shape_mismatch_raises(self, linear_problem):
        forward, theta0, _ = linear_problem
        with pytest.raises(StateValidationError, match="flags"):
            fisher_information(forward, theta0, 1.0, flags=jnp.zeros(3, bool))

    def test_pytree_params(self):
        """Fisher works on structured (pytree) parameter sets."""

        def forward(p):
            return p["gain"] * jnp.arange(1.0, 5.0) + p["offset"]

        params = {"gain": jnp.array(2.0), "offset": jnp.array(0.1)}
        F = fisher_information(forward, params, noise_std=1.0)
        assert F.matrix.shape == (2, 2)
        assert jnp.all(jnp.linalg.eigvalsh(F.matrix) > 0)

    def test_empty_params_rejected(self):
        with pytest.raises(StateValidationError, match="no trainable"):
            fisher_information(lambda p: jnp.zeros(3), {}, 1.0)


class TestCovariancePropagation:
    def test_cramer_rao_roundtrip(self, linear_problem):
        forward, theta0, A = linear_problem
        F = fisher_information(forward, theta0, noise_std=0.5)
        cov = parameter_covariance(F)
        assert jnp.allclose(F.matrix @ cov.matrix, jnp.eye(N_PAR), atol=1e-4)

    def test_delta_method_exact_for_linear(self, linear_problem):
        """For y = A theta: std(y) = sqrt(diag(A Sigma A^T)) exactly."""
        forward, theta0, A = linear_problem
        cov = jnp.diag(jnp.array([0.04, 0.01, 0.09]))
        std = propagate_covariance(forward, theta0, cov)
        expected = jnp.sqrt(jnp.diag(A @ cov @ A.T))
        assert jnp.allclose(std, expected, rtol=1e-5)

    def test_delta_method_matches_monte_carlo(self, linear_problem):
        forward, theta0, _ = linear_problem
        cov = 0.02 * jnp.eye(N_PAR)
        std_delta = propagate_covariance(forward, theta0, cov)

        chol = jnp.linalg.cholesky(cov)
        draws = theta0 + jax.random.normal(jax.random.key(1), (20000, N_PAR)) @ chol.T
        std_mc = push_forward(forward, draws).std(axis=0)
        assert jnp.allclose(std_delta, std_mc, rtol=0.05)

    def test_cov_shape_mismatch_raises(self, linear_problem):
        forward, theta0, _ = linear_problem
        with pytest.raises(StateValidationError, match="param_cov"):
            propagate_covariance(forward, theta0, jnp.eye(N_PAR + 1))

    def test_structure_mismatch_rejected(self, linear_problem):
        """Regression: a covariance from a DIFFERENT parameterization must
        not be silently applied — same size, different flattening order."""
        forward, theta0, _ = linear_problem
        other_params = {"alpha": jnp.zeros(2), "beta": jnp.zeros(1)}
        F_other = fisher_information(
            lambda p: jnp.concatenate([p["alpha"], p["beta"]]) * jnp.ones(3),
            other_params, noise_std=1.0,
        )
        cov_other = parameter_covariance(F_other, jitter=1e-6)
        with pytest.raises(StateValidationError, match="structure"):
            propagate_covariance(forward, theta0, cov_other)


class TestPushForward:
    def test_matches_python_loop(self, linear_problem):
        forward, theta0, _ = linear_problem
        samples = theta0 + 0.1 * jax.random.normal(jax.random.key(2), (5, N_PAR))
        stacked = push_forward(forward, samples)
        assert stacked.shape == (5, N_DATA)
        for i in range(5):
            assert jnp.allclose(stacked[i], forward(samples[i]))

    def test_pytree_samples(self):
        def forward(p):
            return p["a"] * jnp.ones(3)

        samples = {"a": jnp.arange(4.0)}
        out = push_forward(forward, samples)
        assert out.shape == (4, 3)
        assert jnp.array_equal(out[:, 0], jnp.arange(4.0))


class TestEndToEndWithPipeline:
    def test_fisher_through_real_forward_fn(self, template_state):
        """The seam works: Fisher of the gain through a real assembled twin."""
        import equinox as eqx

        from replicant.inference import build_forward_fn
        from replicant.radio import GainOperator, SkyOperator, assemble

        twin = assemble(
            SkyOperator(amplitude=jnp.array(100.0)),
            GainOperator(gain=jnp.array(1.1)),
        )
        spec = jax.tree.map(lambda _: False, twin)
        spec = eqx.tree_at(lambda p: p["gain"].gain, spec, replace=True)
        forward, params0 = build_forward_fn(twin, template_state, filter_spec=spec)

        F = fisher_information(forward, params0, noise_std=0.5)
        # d(pred)/d(gain) = 100 per sample; F = n_samples * 100^2 / 0.25
        n_samples = template_state.coords.time.shape[0] * template_state.coords.freq.shape[0]
        assert jnp.allclose(F.matrix[0, 0], n_samples * 1e4 / 0.25, rtol=1e-4)
        sigma_gain = jnp.sqrt(parameter_covariance(F).matrix[0, 0])
        assert sigma_gain < 1e-2  # sub-percent gain forecast
