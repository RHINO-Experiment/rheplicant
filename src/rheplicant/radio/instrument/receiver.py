"""Receiver bandpass — PLACEHOLDER.

Real physics to come: a frequency-dependent bandpass from measurement or an
instrument model. **Not** the reflection/impedance-mismatch effects this
docstring once promised — those arrived at the ``noise_wave`` node, where
:class:`~rheplicant.radio.instrument.noise_wave.NoiseWaveOperator` carries the
source and receiver ``Gamma`` and is real physics rather than a stand-in.
Sky-side
additive temperatures (atmosphere, ground spill) are *not* receiver
business — they are branches of the antenna-temperature sum
(:class:`~rheplicant.radio.environment.atmosphere.AtmosphericEmissionOperator`,
:class:`~rheplicant.radio.environment.ground.GroundPickupOperator`), entering
before the reflection/noise-wave terms and therefore seeing the
``(1-|Gamma|^2)`` loss. The receiver temperature itself enters after the
reflection, as the noise-wave ``T_0`` (see
:class:`~rheplicant.radio.instrument.noise_wave.NoiseWaveOperator`) and the
post-gain thermal noise ``T_n``.

THE BANDPASS/GAIN SCALE IS NOT IDENTIFIABLE, and the convention that fixes it
lives here. The prediction depends on ``b(nu)`` and ``g(t)`` only through their
product, so ``b -> c*b, g -> g/c`` leaves every predicted sample bit-for-bit
unchanged for any scalar ``c``. Free both and the model has one exactly null
direction — measured, not asserted::

    b free (5 ch) + g free (6 samples), T_ant known
        n_par=11 rank=10 nullity=1     null singular value 8.4e-17 of s_max
        participation                  {'bandpass': 0.50, 'gain': 0.50}
        direction(0)['bandpass'] / b   +0.2206 in every channel
        direction(0)['gain']     / g   -0.2206 in every sample

That last pair IS the trade, read straight off
:meth:`~rheplicant.inference.identifiability.IdentifiabilityReport.direction`:
a perturbation proportional to ``+b`` matched by one proportional to ``-g``.

**The convention: the bandpass carries only SHAPE (mean 1), the gain carries
the level.** Build a bandpass latent with :func:`unit_mean_bandpass` and the
null direction is gone — ``n_par=10 rank=10 nullity=0``, weakest identified
direction 0.41.

Why unit mean rather than pinning a reference channel to 1, the other obvious
convention? Both remove exactly one parameter and both work. Unit mean is the
better one for a real instrument on two counts: the reference is an average
over the band, so noise on it falls as ``sqrt(n_freq)`` instead of being
whatever one channel happened to do; and that channel can be flagged. RHINO
flags channels — it is a radio telescope — and a convention anchored to a
single channel makes the entire absolute gain scale hostage to the one channel
RFI happens to sit in. A band average degrades gracefully instead.

**Normalising inside the binding is NOT enough, and this is the trap.** Binding
a full ``(n_freq,)`` latent through ``fn=lambda b: b / mean(b)`` looks like the
same convention and is not: the prediction is now blind to the scale of the
RAW latent, so the null direction survives — measured ``n_par=11 rank=10
nullity=1``, with ``participation`` now ``{'bandpass': 1.00, 'gain': 0.00}``.
The degeneracy moved out of the b/g trade and into the bandpass latent's own
scale ray; it did not go away. Removing a degeneracy means removing a
parameter, which is what :func:`unit_mean_bandpass` does — it takes
``n_freq - 1`` free values.
"""

from typing import ClassVar

import jax
import jax.numpy as jnp

from rheplicant.core.errors import StateValidationError
from rheplicant.core.operator import AbstractOperator
from rheplicant.core.state import State


def unit_mean_bandpass(free: jax.Array) -> jax.Array:
    """Expand ``n_freq - 1`` free values into an ``n_freq`` bandpass of mean 1.

    The convention that makes a jointly-free bandpass and gain identifiable —
    see the module docstring for the measurement. Use it as a binding's ``fn``::

        Bind("bandpass_shape",
             into=lambda p: p["bandpass"].bandpass,
             fn=unit_mean_bandpass)

    with ``Latent("bandpass_shape", init=unit_mean_free(b_estimate))``.

    The last channel is the dependent one, which is a property of these
    COORDINATES and not of the convention: the image of this map is the whole
    mean-1 hyperplane, so no channel is privileged in the model. It only means
    the flat parameter vector's last entry is not "the last channel".

    Raises:
        StateValidationError: if ``free`` is not 1-D. A ``(1, n)`` array would
            otherwise concatenate along the wrong axis and return a bandpass of
            the wrong length, which ``ReceiverOperator`` would then reject with
            a channel-count message pointing at the wrong thing.
    """
    if free.ndim != 1:
        raise StateValidationError(
            f"unit_mean_bandpass expects the (n_freq - 1,) free values, got "
            f"ndim={free.ndim}. One channel is determined by the mean-1 "
            "convention, so this array is one SHORTER than the bandpass."
        )
    n_freq = free.size + 1
    return jnp.concatenate([free, (n_freq - jnp.sum(free))[None]])


def unit_mean_free(bandpass: jax.Array) -> jax.Array:
    """The ``unit_mean_bandpass`` coordinates of an existing bandpass estimate.

    The inverse of :func:`unit_mean_bandpass` on the mean-1 hyperplane:
    ``unit_mean_bandpass(unit_mean_free(b))`` is ``b / mean(b)``. Use it to turn
    a measured or modelled bandpass into a ``Latent``'s ``init``, so the
    starting point is the estimate itself rather than a flat guess.

    The overall level of ``bandpass`` is discarded, by construction — that is
    the gain's to carry. Give the gain latent an init scaled by the discarded
    ``mean(bandpass)`` if the product is meant to be preserved.

    Raises:
        StateValidationError: if ``bandpass`` is not 1-D.
    """
    if bandpass.ndim != 1:
        raise StateValidationError(
            f"unit_mean_free expects a (n_freq,) bandpass, got ndim={bandpass.ndim}."
        )
    return (bandpass / jnp.mean(bandpass))[:-1]


class ReceiverOperator(AbstractOperator):
    """Apply a frequency-dependent bandpass to ``state.data`` (placeholder).

    When this bandpass and a :class:`~rheplicant.radio.instrument.gain.
    GainOperator` gain are inferred together, declare the bandpass through
    :func:`unit_mean_bandpass`: free as-is, the two share one exactly null
    direction (the module docstring measures it).

    Attributes:
        bandpass: ``(n_freq,)`` dimensionless bandpass — differentiable.
    """

    requires: ClassVar[tuple[str, ...]] = ("data",)
    provides: ClassVar[tuple[str, ...]] = ("data",)
    graph_node: ClassVar[str] = "bandpass"

    bandpass: jax.Array

    def __call__(self, state: State) -> State:
        if self.bandpass.ndim == 0:
            return state.with_data(state.data * self.bandpass)
        if self.bandpass.ndim == 1:
            if self.bandpass.shape[-1] != state.data.shape[-1]:
                raise StateValidationError(
                    f"bandpass has {self.bandpass.shape[-1]} channels but data has "
                    f"{state.data.shape[-1]}."
                )
            return state.with_data(state.data * self.bandpass[None, :])
        raise StateValidationError(
            f"bandpass must be scalar or (n_freq,), got shape "
            f"{tuple(self.bandpass.shape)} (ndim={self.bandpass.ndim}). A 2-D (or "
            "higher) array whose last axis happens to match n_freq would otherwise "
            "pass the channel check silently and broadcast state.data up to "
            f"{self.bandpass.ndim + 1}-D via bandpass[None, :], carrying the extra "
            "axis downstream instead of being refused here."
        )
