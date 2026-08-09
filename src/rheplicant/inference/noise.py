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

**On which axis a sigma vector runs along.** "Scalar or broadcastable to the
data" was the whole contract until :func:`check_noise_std_axis`, and it is not
one: against a square ``(n_time, n_freq)`` grid a length-n sigma vector reads
equally well as one sigma per time sample and as one per frequency channel.
NumPy settles the tie by aligning trailing axes, so a per-time vector is applied
per-frequency and the resulting error bar is flat where the sigma it was built
from spans two orders of magnitude. Both readings are legitimate, which is
precisely why neither may be assumed — the ambiguous vector is refused and the
caller writes ``(n, 1)`` or ``(1, n)``.

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

from typing import Any, ClassVar, Protocol, runtime_checkable

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

    def realise(self, prediction: jax.Array, *, key: jax.Array) -> jax.Array:
        """Draw one noisy observation of ``prediction`` under this model.

        The generator to ``std``'s assumption. A caller that draws with this
        and weights with ``std`` cannot have the two disagree, which
        is the failure mode of every hand-written ``data + sigma * normal``
        line beside a likelihood carrying its own sigma.
        """
        ...


class HomoscedasticNoise(eqx.Module):
    """Constant noise: sigma independent of what the model predicts.

    The behaviour every ``noise_std=`` argument had before this module
    existed, named so that it is a choice rather than a default.

    Attributes:
        sigma: standard deviation — a scalar, or an array shaped so that which
            axis it runs along is written down: ``(n_time, 1)`` for a per-time
            sigma, ``(1, n_freq)`` for a per-channel one. A bare 1-D vector is
            fine only where its length matches a single axis of the prediction;
            where it matches more than one the reading is ambiguous and
            :func:`check_noise_std_axis` refuses it. Either way a differentiable
            leaf, so sigma can itself be inferred.
    """

    depends_on_prediction: ClassVar[bool] = False

    sigma: jax.Array

    def std(self, prediction: jax.Array) -> jax.Array:
        return jnp.broadcast_to(jnp.asarray(self.sigma), jnp.shape(prediction))

    def realise(self, prediction: jax.Array, *, key: jax.Array) -> jax.Array:
        """Additive: ``d + sigma * w``, ``w ~ N(0, 1)``."""
        draw = jax.random.normal(key, jnp.shape(prediction))
        return prediction + self.std(prediction) * draw


class RadiometerNoise(eqx.Module):
    """The radiometer equation: sigma proportional to the prediction itself.

    ``sigma = |prediction| * f`` with the fractional level
    ``f = 1 / sqrt(delta_nu * tau)`` — the multiplicative form, ``d -> d(1+w)``,
    which is what the radiometer equation actually says and what the noise-wave
    radiometer equation writes. This is rheplicant's default noise model.

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
        if not (self.channel_width > 0.0 and self.integration_time > 0.0):
            raise StateValidationError(
                "RadiometerNoise needs a positive channel_width [Hz] and "
                f"integration_time [s], got {self.channel_width} and "
                f"{self.integration_time}. Their product sets the fractional "
                "noise 1/sqrt(delta_nu * tau)."
            )
        if not self.floor >= 0.0:  # `not >=` so a NaN floor is refused too
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

    def realise(self, prediction: jax.Array, *, key: jax.Array) -> jax.Array:
        """Multiplicative: ``d (1 + f w)``, ``f = 1/sqrt(delta_nu tau)``.

        The multiplicative form, not ``d + sigma(d) w`` -- because
        ``sigma = |prediction| * f`` uses an absolute value that a *generator*
        must not, and the two forms differ in sign wherever the prediction
        does. ``floor`` is deliberately not applied here: it is a remedy for a
        reweighting iterate crossing zero, and a generator has no iterate.
        """
        draw = jax.random.normal(key, jnp.shape(prediction))
        return prediction * (1.0 + self.fractional * draw)


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

    def realise(self, prediction: jax.Array, *, key: jax.Array) -> jax.Array:
        """The wrapped model's draw, unchanged.

        Flags say a sample was not OBSERVED, not that it had no true value, so
        they belong to the likelihood's covariance and not to the generator.
        ``std`` puts ``inf`` at the flagged samples; drawing at that sigma
        would produce a data set no instrument could record, and every
        consumer that turns ``inf`` into a clean zero weight expects the datum
        underneath to be finite.
        """
        return self.base.realise(prediction, key=key)


def _constant_sigma(noise_std: Any) -> Any | None:
    """The FIXED sigma array inside a ``noise_std`` argument, if it holds one.

    ``None`` for a model whose sigma is computed from the prediction: its shape
    is the prediction's by construction, so there is no free axis to misread.
    ``FlaggedNoise`` is unwrapped because flagging changes which samples count,
    not which axis the sigma underneath runs along.
    """
    if isinstance(noise_std, FlaggedNoise):
        return _constant_sigma(noise_std.base)
    if isinstance(noise_std, HomoscedasticNoise):
        return noise_std.sigma
    if isinstance(noise_std, NoiseModel):
        return None
    return noise_std


def check_noise_std_axis(
    noise_std: Any, prediction_shape: Any, caller: str
) -> None:
    """Refuse a 1-D ``noise_std`` whose axis the prediction cannot settle.

    "Scalar or broadcastable to the data" is not a contract. Against a square
    grid — ``(n_time, n_freq)`` with ``n_time == n_freq``, which is not exotic
    but the shape a test rig and a single scan block both land on — a length-n
    sigma vector reads equally well as one sigma per time sample and as one per
    frequency channel. NumPy settles it by aligning trailing axes, so the
    per-time reading is applied per-frequency; every downstream number is
    finite, correctly shaped, and answers a question nobody asked.

    Measured on an 8x8 grid with a per-time gain latent and
    ``sigma = linspace(0.01, 1.0, 8)``: the explicit ``(8, 1)`` gives an error
    bar spanning 0.00004 to 0.00354, and the bare ``(8,)`` gives a flat
    0.00010 — the ~90x structure the sigma vector describes, averaged away
    without a word. So the vector is refused and the caller says which axis it
    meant by giving it one.

    Only SHAPES are read, which is the point rather than an economy. A NaN
    defeats every comparison-based guard — ``nan < 0`` and ``nan != nan``
    both come back the answer that lets it through — so a guard that reached
    for the values would be exactly the one a poisoned sigma sails past.
    Shapes are integers; there is nothing here for a NaN to defeat, and
    nothing that has to be re-checked inside a trace.

    Args:
        noise_std: the argument as the caller passed it — a scalar, an array,
            or a :class:`NoiseModel`. Wrapped constant sigmas are unwrapped;
            a prediction-dependent model is exempt by construction.
        prediction_shape: the shape the model predicts, from ``jnp.shape`` or
            ``jax.eval_shape``. No array need be computed.
        caller: the exit to name in the message.

    Raises:
        StateValidationError: if ``noise_std`` is 1-D and its length matches
            more than one axis of the prediction.
    """
    sigma = _constant_sigma(noise_std)
    if sigma is None:
        return
    shape = jnp.shape(sigma)
    if len(shape) != 1:
        return
    (length,) = shape
    if length == 1:
        # A one-element vector is a scalar wearing an axis: every reading
        # broadcasts the same single number to every sample, so the readings
        # coincide and there is nothing to disambiguate. Without this a (1, 1)
        # prediction would refuse a perfectly unambiguous sigma.
        return
    prediction_shape = tuple(prediction_shape)
    axes = [axis for axis, size in enumerate(prediction_shape) if size == length]
    if len(axes) < 2:
        return

    def explicit(axis: int) -> tuple[int, ...]:
        return tuple(length if i == axis else 1 for i in range(len(prediction_shape)))

    first, last = axes[0], axes[-1]
    raise StateValidationError(
        f"{caller} was given a noise_std of shape ({length},) for a prediction of "
        f"shape {prediction_shape}, whose axes {axes} are ALL of length {length} — so "
        "this vector has more than one legitimate reading and they are different noise "
        f"models. Broadcasting would silently pick axis {last} (trailing axes align), "
        f"weighting one sigma per axis-{last} index and repeating it across every "
        f"other axis; read along axis {first} the same numbers weight an entirely "
        "different set of samples. Both come back finite and correctly shaped, so "
        "nothing downstream can tell them apart. Say which you mean by giving the "
        f"axis a shape: reshape to {explicit(first)} for axis {first}, or to "
        f"{explicit(last)} for axis {last}."
    )


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
