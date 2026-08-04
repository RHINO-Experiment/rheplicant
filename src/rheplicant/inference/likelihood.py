"""Likelihoods: score a forward-model prediction against observed data.

A likelihood is any callable ``(prediction, observed) -> scalar log-prob``.
The Protocol below documents the contract; :class:`GaussianLikelihood` is the
minimal concrete instance. Real instrument likelihoods (radiometer-equation
noise, 1/f covariance, Toeplitz solvers ported from hydra-tod/comat) will
implement the same contract.

The contract's one unwritten precondition — that the two arguments describe the
same data — is written here, as :func:`check_observed_shape`. It lives at this
seam because this is where prediction meets observed, and every inference route
that consumes ``observed`` (the calibrators, the NumPyro observation site, the
conjugate-Gaussian solves) calls it at its own entry point rather than
re-deriving the refusal.
"""

from typing import Any, Protocol, runtime_checkable

import equinox as eqx
import jax
import jax.numpy as jnp

from rheplicant.core.errors import ParameterSpaceError


@runtime_checkable
class Likelihood(Protocol):
    """Contract: ``logp = likelihood(prediction, observed)`` (scalar)."""

    def __call__(self, prediction: jax.Array, observed: jax.Array) -> jax.Array: ...


def check_observed_shape(
    prediction_shape: Any,
    observed: Any,
    *,
    predictor: str = "this block",
) -> None:
    """Refuse an ``observed`` the prediction would have to broadcast against.

    Every scoring rule in this package subtracts the two, and NumPy
    broadcasting makes ``(24, 8) - (8,)`` a legal, finite, wrong residual. The
    failure has no symptom: the loss converges, the Fisher matrix inverts, NUTS
    reports a healthy ``r_hat``, and every recovered parameter is the average of
    a problem nobody posed. So the mismatch is an error at entry, not a warning
    later.

    Shapes are static, so this costs nothing at run time — call it once where
    the arguments arrive, never inside a jitted step or a gradient evaluation.

    Args:
        prediction_shape: the shape the model predicts. A tuple, from
            ``jnp.shape`` or ``jax.eval_shape`` — no array need be computed.
        observed: the data. Only its shape is read.
        predictor: how the caller names the thing that predicts, for the
            message ("this block", "this forward model", ...).

    Raises:
        ParameterSpaceError: if the shapes differ at all. Broadcast-compatible
            is not the same as equal, and it is exactly the compatible cases
            that are dangerous.
    """
    if jnp.shape(observed) != tuple(prediction_shape):
        raise ParameterSpaceError(
            f"observed has shape {jnp.shape(observed)} but {predictor} predicts "
            f"{tuple(prediction_shape)}. Broadcasting these would solve a different "
            "problem and return a perfectly finite answer."
        )


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
