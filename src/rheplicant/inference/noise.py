"""Noise models: the one object every inference route asks for sigma.

A noise model answers a single question — *given this prediction, how noisy is
the datum?* — and everything statistical follows from the answer: the
likelihood, the loss, the weights a Wiener solve or a GCR draw uses, the
Fisher matrix, the scale of a NumPyro observation site.

Before this module those five routes each took a bare ``noise_std`` argument,
which quietly assumed the answer is *given* and *constant*. For a radiometer
it is neither::

    sigma(d) = |d| / sqrt(delta_nu * tau)

Sigma is a function of the very thing being inferred. That single fact is what
forces iteratively-reweighted least squares (:mod:`rheplicant.inference.gls`),
what puts a log-determinant term in a gradient-sampled posterior, and what RFI
flags flow into.

Three models, composed rather than configured:

.. list-table::
   :header-rows: 1

   * - Model
     - sigma
     - ``depends_on_prediction``
   * - :class:`HomoscedasticNoise`
     - a constant, per-sample or scalar
     - ``False``
   * - :class:`RadiometerNoise`
     - ``|prediction| / sqrt(delta_nu * tau)``
     - ``True``
   * - :class:`FlaggedNoise`
     - the wrapped model, or ``inf`` where flagged
     - inherited

``FlaggedNoise`` is how flags reach the covariance: by *wrapping a noise
model*, not by bolting a ``flags=`` keyword onto five separate functions. An
infinite sigma is a self-describing encoding of "this sample was not
observed", and :class:`NoiseModelLikelihood` and :func:`inverse_variance` both
give it a clean zero rather than a NaN.

``depends_on_prediction`` is the property downstream code branches on: ``False``
means one solve, ``True`` means a loop.

**On the log-determinant.** The Gaussian log-density is::

    log p = -1/2 sum_i [ r_i^2 / sigma_i(theta)^2  +  log 2 pi sigma_i(theta)^2 ]

When sigma is constant the second term is an additive constant and dropping it
changes nothing. When sigma depends on the prediction it does not, and dropping
it — which is precisely what generalized least squares does — gives a
*different estimator*, one with no penalty for shrinking the prediction to make
the variance small. For the multiplicative model both are solvable in closed
form: GLS returns ``sum d^2 / sum d``, biased high by ``(1 + f^2)``, while the
full density is asymptotically unbiased. So :class:`NoiseModelLikelihood`
keeps the term by default, and ``include_logdet=False`` is the explicit,
documented GLS variant rather than an oversight.
"""

from typing import ClassVar, Protocol, runtime_checkable

import equinox as eqx
import jax
import jax.numpy as jnp

from rheplicant.core.errors import StateValidationError


@runtime_checkable
class NoiseModel(Protocol):
    """Contract: ``sigma = noise.std(prediction)``, shaped like the prediction.

    Attributes:
        depends_on_prediction: whether ``std`` actually reads its argument.
            ``False`` lets a solver skip the reweighting loop entirely, so it
            is a claim about the model, not a hint.
    """

    depends_on_prediction: bool

    def std(self, prediction: jax.Array) -> jax.Array: ...


class HomoscedasticNoise(eqx.Module):
    """Constant noise: sigma independent of what the model predicts.

    The behaviour every ``noise_std=`` argument had before this module
    existed, named so that it is a choice rather than a default.

    Attributes:
        sigma: standard deviation — scalar or broadcastable to the data, and a
            differentiable leaf, so it can itself be inferred.
    """

    depends_on_prediction: ClassVar[bool] = False

    sigma: jax.Array

    def std(self, prediction: jax.Array) -> jax.Array:
        return jnp.broadcast_to(jnp.asarray(self.sigma), jnp.shape(prediction))


class RadiometerNoise(eqx.Module):
    """The radiometer equation: sigma proportional to the prediction itself.

    ``sigma = |prediction| * f`` with the fractional level
    ``f = 1 / sqrt(delta_nu * tau)`` — the multiplicative form, ``d -> d(1+w)``,
    which is what the radiometer equation actually says and what the noise-wave
    draft's Eq. 8 writes. This is rheplicant's default noise model.

    Because sigma tracks the prediction, a solve for the prediction and a
    weighting by its noise are the same problem: see
    :func:`~rheplicant.inference.gls.iterative_gls`.

    Attributes:
        channel_width: channel bandwidth ``delta_nu`` [Hz] (static — instrument
            metadata, known rather than fitted).
        integration_time: per-sample integration time ``tau`` [s] (static).
        floor: lower bound applied to ``|prediction|`` before scaling [K].
            Defaults to ``0.0``, i.e. the exact physics: a prediction that
            passes through zero then has zero sigma and infinite weight, which
            is a loud failure. A reweighting iterate can cross zero where the
            physics cannot, and a floor is the remedy there.
    """

    depends_on_prediction: ClassVar[bool] = True

    channel_width: float = eqx.field(static=True)
    integration_time: float = eqx.field(static=True)
    floor: float = eqx.field(static=True, default=0.0)

    def __check_init__(self):
        if self.channel_width <= 0.0 or self.integration_time <= 0.0:
            raise StateValidationError(
                "RadiometerNoise needs a positive channel_width [Hz] and "
                f"integration_time [s], got {self.channel_width} and "
                f"{self.integration_time}. Their product sets the fractional "
                "noise 1/sqrt(delta_nu * tau)."
            )
        if self.floor < 0.0:
            raise StateValidationError(f"floor must be >= 0, got {self.floor}.")

    @property
    def fractional(self) -> float:
        """``1 / sqrt(delta_nu * tau)`` — the fractional noise per sample."""
        return 1.0 / (self.channel_width * self.integration_time) ** 0.5

    def std(self, prediction: jax.Array) -> jax.Array:
        # abs, not the raw prediction: a negative iterate would otherwise flip
        # the sign of every weight it touches without any of them being NaN.
        magnitude = jnp.abs(jnp.asarray(prediction))
        if self.floor > 0.0:
            magnitude = jnp.maximum(magnitude, self.floor)
        return magnitude * self.fractional


class FlaggedNoise(eqx.Module):
    """Wrap a noise model so flagged samples carry infinite variance.

    The seam where RFI flagging meets the noise covariance: a flagged sample
    was not observed, so it must inform nothing. Encoding that as ``sigma =
    inf`` keeps the fact inside the noise model, where every consumer already
    looks, instead of as a parallel ``flags=`` argument each of them has to
    remember to honour.

    ``inf`` is only ever an encoding — :class:`NoiseModelLikelihood` and
    :func:`inverse_variance` both turn it into a clean zero contribution rather
    than letting ``inf * 0`` become NaN.

    Attributes:
        base: the noise model in force on unflagged samples.
        flags: boolean array shaped like the data; ``True`` = flagged.
    """

    base: NoiseModel
    flags: jax.Array

    @property
    def depends_on_prediction(self) -> bool:
        return self.base.depends_on_prediction

    def std(self, prediction: jax.Array) -> jax.Array:
        sigma = self.base.std(prediction)
        if jnp.shape(self.flags) != jnp.shape(sigma):
            raise StateValidationError(
                f"flags shape {jnp.shape(self.flags)} does not match the "
                f"prediction shape {jnp.shape(sigma)}."
            )
        return jnp.where(self.flags, jnp.inf, sigma)


def inverse_variance(noise: NoiseModel, prediction: jax.Array) -> jax.Array:
    """Per-sample weights ``1 / sigma^2``, with a clean zero where unobserved.

    The quantity every weighted solve wants. An infinite sigma (see
    :class:`FlaggedNoise`) becomes exactly ``0.0`` rather than an underflowed
    denominator, so the weight array is finite by construction.
    """
    sigma = noise.std(prediction)
    observed = jnp.isfinite(sigma)
    safe = jnp.where(observed, sigma, 1.0)
    return jnp.where(observed, 1.0 / safe**2, 0.0)


class NoiseModelLikelihood(eqx.Module):
    """Gaussian log-density under a :class:`NoiseModel`.

    Generalizes :class:`~rheplicant.inference.likelihood.GaussianLikelihood`
    (which is this with :class:`HomoscedasticNoise`) and
    :class:`~rheplicant.inference.likelihood.MaskedGaussianLikelihood` (this
    with :class:`FlaggedNoise` around it); both remain, and both agree with
    this to roundoff.

    Attributes:
        noise: the noise model — supplies sigma at the prediction.
        include_logdet: keep the ``log 2 pi sigma^2`` normalization (default).
            Setting it ``False`` gives generalized least squares. That is only
            the same objective when sigma does not depend on the prediction;
            when it does, GLS is a *different estimator* — see this module's
            docstring for the closed forms and which way the bias runs. Static:
            it selects an objective, not a value.
    """

    noise: NoiseModel
    include_logdet: bool = eqx.field(static=True, default=True)

    def __call__(self, prediction: jax.Array, observed: jax.Array) -> jax.Array:
        sigma = self.noise.std(prediction)
        # An unobserved sample contributes nothing at all. Taking the limit
        # instead would not work: r^2/sigma^2 -> 0 but log sigma^2 -> inf, so
        # a single flagged channel would send the whole log-density to -inf.
        seen = jnp.isfinite(sigma)
        safe = jnp.where(seen, sigma, 1.0)
        per_sample = ((observed - prediction) / safe) ** 2
        if self.include_logdet:
            per_sample = per_sample + jnp.log(2.0 * jnp.pi * safe**2)
        return -0.5 * jnp.sum(jnp.where(seen, per_sample, 0.0))
