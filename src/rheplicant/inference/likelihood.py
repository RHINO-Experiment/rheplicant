"""Likelihoods: score a forward-model prediction against observed data.

A likelihood is any callable ``(prediction, observed) -> scalar log-prob``.
The Protocol below documents the contract; :class:`GaussianLikelihood` is the
minimal concrete instance. Real instrument likelihoods (radiometer-equation
noise, 1/f covariance, Toeplitz solvers ported from hydra-tod/comat) will
implement the same contract.
"""

from typing import Protocol, runtime_checkable

import equinox as eqx
import jax
import jax.numpy as jnp


@runtime_checkable
class Likelihood(Protocol):
    """Contract: ``logp = likelihood(prediction, observed)`` (scalar)."""

    def __call__(self, prediction: jax.Array, observed: jax.Array) -> jax.Array: ...


class GaussianLikelihood(eqx.Module):
    """Independent Gaussian likelihood with fixed noise level.

    Attributes:
        noise_std: noise standard deviation — scalar or broadcastable to the
            data shape; a differentiable leaf (so it can itself be inferred).
    """

    noise_std: jax.Array

    def __call__(self, prediction: jax.Array, observed: jax.Array) -> jax.Array:
        resid = (observed - prediction) / self.noise_std
        return -0.5 * jnp.sum(resid**2 + jnp.log(2.0 * jnp.pi * self.noise_std**2))


class MaskedGaussianLikelihood(eqx.Module):
    """Gaussian likelihood that ignores flagged samples.

    The seam where RFI flags inform the noise covariance (GCR draft): pass
    ``flags`` from ``state.aux["flags"]`` (True = flagged/bad); flagged
    samples contribute zero to the log-probability, equivalent to infinite
    noise variance on those samples.

    Attributes:
        noise_std: noise standard deviation — scalar or broadcastable.
        flags: boolean mask, True = excluded; ``None`` behaves exactly like
            :class:`GaussianLikelihood`.
    """

    noise_std: jax.Array
    flags: jax.Array | None = None

    def __call__(self, prediction: jax.Array, observed: jax.Array) -> jax.Array:
        resid = (observed - prediction) / self.noise_std
        per_sample = resid**2 + jnp.log(2.0 * jnp.pi * self.noise_std**2)
        if self.flags is not None:
            per_sample = jnp.where(self.flags, 0.0, per_sample)
        return -0.5 * jnp.sum(per_sample)


def mean_squared_error(prediction: jax.Array, observed: jax.Array) -> jax.Array:
    """Plain MSE — the default loss for quick gradient calibration."""
    return jnp.mean((prediction - observed) ** 2)
