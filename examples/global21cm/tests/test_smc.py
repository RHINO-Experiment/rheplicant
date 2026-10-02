"""The tempered SMC against a closed-form Gaussian posterior.

Prior u ~ Normal(0, I_7); likelihood exp(-|A u - b|^2 / 2). The posterior
is Normal(mu, S) with S = (I + A^T A)^-1, mu = S A^T b, and
log Z = -|b|^2/2 + b^T A S A^T b / 2 + log det(S) / 2.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from global21cm import smc


@pytest.fixture(scope="module")
def problem():
    rng = np.random.default_rng(0)
    a = np.diag([3.0, 1.0, 0.5, 2.0, 0.2, 4.0, 1.5]) + 0.1 * rng.normal(size=(7, 7))
    b = rng.normal(size=7)
    cov = np.linalg.inv(np.eye(7) + a.T @ a)
    mean = cov @ a.T @ b
    log_z = -0.5 * b @ b + 0.5 * b @ a @ cov @ a.T @ b + 0.5 * np.linalg.slogdet(cov)[1]
    loglik = lambda u: -0.5 * jnp.sum((jnp.asarray(u) @ a.T - b) ** 2, axis=1)  # noqa: E731
    return loglik, mean, cov, log_z


@pytest.fixture(scope="module")
def result(problem):
    return smc.run(problem[0], seed=3, n=8000, n_move=20)


def test_log_evidence(problem, result):
    assert result["log_z"] == pytest.approx(problem[3], abs=0.2)


def test_the_likelihood_is_reached_in_several_tempering_steps(result):
    # One step would be importance sampling from the prior; this problem is easy
    # enough that its log Z would still pass the tolerance above.
    assert result["steps"] > 1


def test_increment_keeps_the_target_ess():
    # Each step is the largest whose incremental weights keep ESS = ess_frac N.
    loglik = -50.0 * np.random.default_rng(0).random(1000)
    step = smc._increment(loglik, 0.0, 0.6, 1000)
    w = np.exp(step * (loglik - loglik.max()))
    assert 0.0 < step < 1.0
    assert w.sum() ** 2 / (w**2).sum() == pytest.approx(600.0, rel=1e-6)


def test_increment_takes_the_rest_when_the_target_is_kept():
    # A flat likelihood never lowers the ESS: the whole remaining way in one step.
    assert smc._increment(np.zeros(100), 0.25, 0.6, 100) == 0.75


def test_mean_and_covariance(problem, result):
    _, mean, cov, _ = problem
    se = np.sqrt(np.diag(cov) / 8000) * 5  # a generous allowance for particle correlation
    assert np.all(np.abs(result["u"].mean(axis=0) - mean) < 4 * se)
    np.testing.assert_allclose(np.cov(result["u"], rowvar=False).diagonal(), cov.diagonal(), rtol=0.1)


def test_minus_infinity_likelihood_is_never_accepted():
    # Half the space forbidden: a proposal with log L = -inf gives log_a = -inf, and
    # log(uniform) < -inf is False, so it is rejected; NaN would compare False too.
    loglik = lambda u: jnp.where(jnp.asarray(u)[:, 0] > 0, 0.0, -jnp.inf)  # noqa: E731
    out = smc.run(loglik, seed=1, n=4000, n_move=10)
    assert np.all(out["u"][:, 0] > 0)
    assert out["log_z"] == pytest.approx(np.log(0.5), abs=0.05)
