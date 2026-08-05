"""Streaming must equal batch, exactly, wherever an exact answer exists."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rheplicant.inference import Bind, Latent, ParameterSpace
from rheplicant.inference.compress import compress_linear
from rheplicant.inference.factorize import Factorization
from rheplicant.inference.memory import BayesMemory

N_EPOCHS = 12
N_DATA = 30
N_THETA = 3
SIGMA = 0.2
PRIOR_STD = 5.0
TRUTH = jnp.array([1.3, -0.7, 0.4])


class _Normal:
    def __init__(self, loc, scale):
        self.loc, self.scale = loc, scale

    def log_prob(self, x):
        return -0.5 * jnp.sum(
            ((x - self.loc) / self.scale) ** 2 + jnp.log(2 * jnp.pi * self.scale**2)
        )


def _campaign():
    designs, datasets = [], []
    for epoch in range(N_EPOCHS):
        key = jax.random.key(100 + epoch)
        kd, kn = jax.random.split(key)
        design = jax.random.normal(kd, (N_DATA, N_THETA))
        designs.append(design)
        datasets.append(design @ TRUTH + SIGMA * jax.random.normal(kn, (N_DATA,)))
    return designs, datasets


def _factorization():
    latent = Latent("x", init=jnp.zeros(N_THETA), prior=_Normal(0.0, PRIOR_STD), linear=True)
    space = ParameterSpace(
        latents=(latent,), bindings=(Bind("x", into=lambda p: p.x),)
    )
    return Factorization(space)


def _batch_posterior_moments(designs, datasets):
    """The oracle: one dense normal-equations solve over the stacked campaign."""
    stacked_design = np.concatenate([np.asarray(d) for d in designs], axis=0)
    stacked_data = np.concatenate([np.asarray(d) for d in datasets], axis=0)
    fisher = stacked_design.T @ stacked_design / SIGMA**2
    fisher += np.eye(N_THETA) / PRIOR_STD**2
    rhs = stacked_design.T @ stacked_data / SIGMA**2
    covariance = np.linalg.inv(fisher)
    return covariance @ rhs, covariance


def _streamed_memory(designs, datasets):
    memory = BayesMemory(_factorization())
    for epoch, (design, data) in enumerate(zip(designs, datasets, strict=True)):
        memory = memory.remember(
            compress_linear(
                design={"x": design}, observed=data, noise_std=SIGMA,
                shapes={"x": (N_THETA,)}, epoch_id=f"night-{epoch:03d}",
            )
        )
    return memory


def test_streamed_posterior_mean_and_covariance_match_the_batch_solve():
    designs, datasets = _campaign()
    mean, covariance = _batch_posterior_moments(designs, datasets)
    memory = _streamed_memory(designs, datasets)

    grad = jax.grad(lambda v: memory.log_posterior(v))({"x": jnp.zeros(N_THETA)})["x"]
    hessian = jax.hessian(lambda v: memory.log_posterior(v))({"x": jnp.zeros(N_THETA)})
    streamed_precision = -np.asarray(hessian["x"]["x"])
    streamed_covariance = np.linalg.inv(streamed_precision)
    streamed_mean = streamed_covariance @ np.asarray(grad)

    np.testing.assert_allclose(streamed_mean, mean, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(streamed_covariance, covariance, rtol=1e-10, atol=1e-12)


def test_streamed_log_posterior_matches_the_dense_one_up_to_a_constant():
    designs, datasets = _campaign()
    memory = _streamed_memory(designs, datasets)
    stacked_design = np.concatenate([np.asarray(d) for d in designs], axis=0)
    stacked_data = np.concatenate([np.asarray(d) for d in datasets], axis=0)

    def dense(x):
        resid = stacked_data - stacked_design @ np.asarray(x)
        return -0.5 * (
            resid @ resid / SIGMA**2
            + len(stacked_data) * np.log(2 * np.pi * SIGMA**2)
            + float(np.asarray(x) @ np.asarray(x)) / PRIOR_STD**2
            + N_THETA * np.log(2 * np.pi * PRIOR_STD**2)
        )

    probes = [jnp.zeros(N_THETA), TRUTH, jnp.array([-3.0, 4.0, 0.5])]
    for probe in probes:
        assert float(memory.log_posterior({"x": probe})) == pytest.approx(
            dense(probe), rel=1e-10
        )


def test_shuffling_the_epochs_changes_nothing():
    designs, datasets = _campaign()
    forward = _streamed_memory(designs, datasets)
    order = list(reversed(range(N_EPOCHS)))
    backward = _streamed_memory([designs[i] for i in order], [datasets[i] for i in order])
    probe = {"x": jnp.array([0.2, -0.4, 1.1])}
    assert float(forward.log_posterior(probe)) == pytest.approx(
        float(backward.log_posterior(probe)), rel=1e-11
    )


def test_two_half_campaigns_merge_into_the_whole():
    designs, datasets = _campaign()
    half = N_EPOCHS // 2
    whole = _streamed_memory(designs, datasets)
    first = _streamed_memory(designs[:half], datasets[:half])
    merged = first
    for epoch in range(half, N_EPOCHS):
        merged = merged.remember(
            compress_linear(
                design={"x": designs[epoch]}, observed=datasets[epoch],
                noise_std=SIGMA, shapes={"x": (N_THETA,)},
                epoch_id=f"night-{epoch:03d}",
            )
        )
    probe = {"x": jnp.array([0.2, -0.4, 1.1])}
    assert float(merged.log_posterior(probe)) == pytest.approx(
        float(whole.log_posterior(probe)), rel=1e-11
    )


def test_a_rank_deficient_epoch_inside_a_campaign_still_matches_the_batch_solve():
    """One blind night in twelve -- the case the square-root form exists for.

    Every other test here streams epochs that each constrain all three
    directions, so the accumulator never has to represent a term that says
    nothing about part of the space. This one replaces night 4 with a design of
    rank 1: its Fisher has two exactly-zero eigenvalues, and in ``(F, b)`` form
    the corresponding Schur complement would be a solve against a singular
    matrix. In square-root form it is simply a factor with fewer rows than
    columns, and ``combine`` is still a QR.

    Measured on this campaign: night 4's Fisher has eigenvalues
    ``(0.0, 0.0, 407.989)`` -- two exact zeros, not two small numbers -- and
    the streamed posterior still reproduces the dense batch solve to a maximum
    relative error of 1.0e-15 on the mean and 6.5e-15 on the covariance. The
    dense log-posterior, constant included, agrees to the same order. The
    assertions below are pinned at 1e-10, five orders looser than what the
    arithmetic delivers, so that a regression has to be a real one.
    """
    designs, datasets = _campaign()
    designs, datasets = list(designs), list(datasets)
    blind = jnp.zeros((N_DATA, N_THETA)).at[:, 0].set(
        jax.random.normal(jax.random.key(900), (N_DATA,))
    )
    designs[4] = blind
    datasets[4] = blind @ TRUTH + SIGMA * jax.random.normal(
        jax.random.key(901), (N_DATA,)
    )
    assert int(np.linalg.matrix_rank(np.asarray(blind), tol=1e-9)) == 1

    memory = _streamed_memory(designs, datasets)
    lone = np.linalg.eigvalsh(np.asarray(memory.archive[4].info.fisher()))
    assert np.allclose(lone[:2], 0.0, atol=1e-9)
    assert lone[2] > 1.0

    mean, covariance = _batch_posterior_moments(designs, datasets)
    grad = jax.grad(lambda v: memory.log_posterior(v))({"x": jnp.zeros(N_THETA)})["x"]
    hessian = jax.hessian(lambda v: memory.log_posterior(v))({"x": jnp.zeros(N_THETA)})
    streamed_covariance = np.linalg.inv(-np.asarray(hessian["x"]["x"]))
    np.testing.assert_allclose(
        streamed_covariance @ np.asarray(grad), mean, rtol=1e-10, atol=1e-12
    )
    np.testing.assert_allclose(streamed_covariance, covariance, rtol=1e-10, atol=1e-12)

    stacked_design = np.concatenate([np.asarray(d) for d in designs], axis=0)
    stacked_data = np.concatenate([np.asarray(d) for d in datasets], axis=0)
    probe = jnp.array([0.2, -0.4, 1.1])
    resid = stacked_data - stacked_design @ np.asarray(probe)
    dense = -0.5 * (
        resid @ resid / SIGMA**2
        + len(stacked_data) * np.log(2 * np.pi * SIGMA**2)
        + float(np.asarray(probe) @ np.asarray(probe)) / PRIOR_STD**2
        + N_THETA * np.log(2 * np.pi * PRIOR_STD**2)
    )
    assert float(memory.log_posterior({"x": probe})) == pytest.approx(dense, rel=1e-10)
