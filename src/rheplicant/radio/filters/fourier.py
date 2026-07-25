"""FourierBandFilter: project onto a Fourier band along one data axis.

One class covers the classic waterfall filters:

- ``axis=0`` (time)      -> fringe-rate filtering,
- ``axis=1`` (frequency) -> delay filtering,

and a high-pass along time (limTOD's ``HP_filter_TOD``) is
``FourierBandFilter(axis=0, low=cutoff, high=0.5, mode="extract")``.

The band is specified in cycles/sample, ``0 <= low < high <= 0.5`` (Nyquist).
Projection: FFT along the axis, zero everything outside ``low <= |f| < high``
(DC survives only if ``low == 0``), inverse FFT.
"""

from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp

from rheplicant.core.errors import StateValidationError
from rheplicant.core.state import State
from rheplicant.radio.filters.base import AbstractLinearFilter


class FourierBandFilter(AbstractLinearFilter):
    """Band projection in fringe-rate (axis=0) or delay (axis=1) space.

    Attributes:
        axis: data axis to transform (static; 0=time, 1=frequency).
        low: band lower edge, cycles/sample (static).
        high: band upper edge, cycles/sample (static; up to 0.5 = Nyquist).
        mode: ``"extract"`` (keep band) or ``"remove"`` (notch band).
    """

    requires: ClassVar[tuple[str, ...]] = ("data",)
    provides: ClassVar[tuple[str, ...]] = ("data",)

    axis: int = eqx.field(static=True)
    low: float = eqx.field(static=True)
    high: float = eqx.field(static=True)
    mode: str = eqx.field(static=True, default="remove")

    def __check_init__(self):
        if self.axis not in (0, 1):
            raise StateValidationError(f"axis must be 0 (time) or 1 (freq), got {self.axis!r}.")
        if not 0.0 <= self.low < self.high <= 0.5:
            raise StateValidationError(
                f"Band must satisfy 0 <= low < high <= 0.5, got low={self.low}, high={self.high}."
            )

    def project(self, data: jax.Array, state: State) -> jax.Array:
        n = data.shape[self.axis]
        f = jnp.abs(jnp.fft.fftfreq(n))
        in_band = (f >= self.low) & (f < self.high)
        shape = [1] * data.ndim
        shape[self.axis] = n
        spectrum = jnp.fft.fft(data, axis=self.axis)
        return jnp.real(jnp.fft.ifft(spectrum * in_band.reshape(shape), axis=self.axis))
