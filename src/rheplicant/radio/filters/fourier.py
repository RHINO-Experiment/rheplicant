"""FourierBandFilter: project onto a Fourier band along one data axis.

One class covers the classic waterfall filters:

- ``axis=0`` (time)      -> fringe-rate filtering,
- ``axis=1`` (frequency) -> delay filtering,

and a high-pass along time (limTOD's ``HP_filter_TOD``) is
``FourierBandFilter(axis=0, low=cutoff, high=0.5, mode="extract")``.

The band is specified in cycles/sample, ``0 <= low < high <= 0.5`` (Nyquist).
Projection: FFT along the axis, zero everything outside the band, inverse FFT.

The band is half-open on the right, ``low <= |f| < high``, *except* when
``high`` is exactly ``0.5`` (Nyquist), in which case the right edge is closed:
``low <= |f| <= 0.5``. This is deliberate, not an inconsistency: a half-open
right edge is what lets adjacent bands ``[a, b)`` and ``[b, c)`` partition the
axis without either one claiming the bin at ``f = b`` twice. But nothing lies
beyond Nyquist for a closed edge to double-count against, and a strict
``< 0.5`` would instead make the Nyquist bin unreachable by *any* band, even
one explicitly written to include it (``high=0.5``). Do not generalise the
closed edge to interior boundaries -- that would reintroduce double-counting
at ordinary band boundaries.

For even ``n``, ``jnp.fft.fftfreq(n)`` has an exact bin at Nyquist (returned
as ``-0.5``, hence the ``jnp.abs`` before the band test). For odd ``n`` there
is no bin exactly at 0.5, so the closed-edge case is simply a no-op there
(DC survives only if ``low == 0``, as always).
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

    The band membership test is half-open on the right, ``[low, high)``,
    *except* at ``high == 0.5`` where it is closed, ``[low, 0.5]`` -- see the
    module docstring for why. ``high`` is a static (Python-level) field, so
    this is a plain Python branch resolved at trace time, not a runtime
    ``jnp.where``.
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
        # Right edge is closed only at Nyquist (high == 0.5), so a band
        # written to reach Nyquist actually reaches it; every interior edge
        # stays half-open so abutting bands don't double-count their shared
        # boundary bin. `self.high` is static, so this branches in Python at
        # trace time, not via `jnp.where`. See the module/class docstrings.
        upper = (f <= self.high) if self.high == 0.5 else (f < self.high)
        in_band = (f >= self.low) & upper
        shape = [1] * data.ndim
        shape[self.axis] = n
        spectrum = jnp.fft.fft(data, axis=self.axis)
        return jnp.real(jnp.fft.ifft(spectrum * in_band.reshape(shape), axis=self.axis))
