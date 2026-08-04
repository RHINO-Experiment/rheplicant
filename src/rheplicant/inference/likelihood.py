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

from typing import Any, ClassVar, Protocol, runtime_checkable

import equinox as eqx
import jax
import jax.numpy as jnp

from rheplicant.core.errors import ParameterSpaceError


@runtime_checkable
class Likelihood(Protocol):
    """Contract: ``logp = likelihood(prediction, observed)`` (scalar).

    The Protocol cannot express the one thing a caller most needs to know
    about a scoring function — whether it is to be **maximized** or
    **minimized** — because both senses have exactly this signature.
    ``isinstance(mean_squared_error, Likelihood)`` is ``True``, and so is
    ``isinstance(GaussianLikelihood(1.0), Likelihood)``, while handing the
    second to a minimizer walks a log-density unbounded below and reports a
    beautifully improving loss the whole way down (measured: g = -30.7 against
    a truth of 1.0, loss -3.2e+07 -> -1.3e+11).

    So the sense is carried as an attribute instead. ``sense`` is
    ``"maximize"`` on every likelihood in this package and absent on plain
    error functions, which default to ``"minimize"``. It is advisory — a
    caller may declare nothing — which is why
    :mod:`rheplicant.inference.calibrate` also *measures* the sense at entry
    rather than trusting the declaration alone.
    """

    #: ``"maximize"`` for a log-density, ``"minimize"`` for an error. Optional;
    #: an object that does not declare it is read as ``"minimize"``.
    sense: str

    def __call__(self, prediction: jax.Array, observed: jax.Array) -> jax.Array: ...


#: Read from a scoring function that does not declare :attr:`Likelihood.sense`.
#: ``"minimize"`` because the un-annotated case is a plain error function --
#: a bare ``lambda p, o: jnp.mean((p - o) ** 2)`` -- and because defaulting the
#: other way would refuse every such lambda in every example in the package.
DEFAULT_SENSE = "minimize"
MAXIMIZE, MINIMIZE = "maximize", "minimize"


def sense_of(scoring_function: Any) -> str:
    """The declared sense of a scoring function, or the default.

    Args:
        scoring_function: any callable ``(prediction, observed) -> scalar``.

    Returns:
        ``"maximize"`` or ``"minimize"``.

    Raises:
        ParameterSpaceError: if ``sense`` is present but is neither. A typo in
            a declaration must not silently read as the default, which is the
            permissive direction and the one that loses a fit.
    """
    declared = getattr(scoring_function, "sense", DEFAULT_SENSE)
    if declared not in (MAXIMIZE, MINIMIZE):
        raise ParameterSpaceError(
            f"{type(scoring_function).__name__}.sense is {declared!r}; it must be "
            f"{MAXIMIZE!r} (a log-density) or {MINIMIZE!r} (an error)."
        )
    return declared


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
        sense: ``"maximize"``. This is a log-density: it is unbounded below,
            so a minimizer handed this object walks away from the truth and
            reports an improving loss the entire way.
    """

    sense: ClassVar[str] = MAXIMIZE

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
        sense: ``"maximize"`` — see :class:`GaussianLikelihood`.
    """

    sense: ClassVar[str] = MAXIMIZE

    noise_std: jax.Array
    flags: jax.Array | None = None

    def __call__(self, prediction: jax.Array, observed: jax.Array) -> jax.Array:
        resid = (observed - prediction) / self.noise_std
        per_sample = resid**2 + jnp.log(2.0 * jnp.pi * self.noise_std**2)
        if self.flags is not None:
            per_sample = jnp.where(self.flags, 0.0, per_sample)
        return -0.5 * jnp.sum(per_sample)


def mean_squared_error(prediction: jax.Array, observed: jax.Array) -> jax.Array:
    """Plain MSE — the default loss for quick gradient calibration.

    Declares no ``sense``, and so reads as ``"minimize"``: the un-annotated
    case is a plain error function, which is what a minimizer wants.
    """
    return jnp.mean((prediction - observed) ** 2)
