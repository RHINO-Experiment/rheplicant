"""Sky projectors: how a sky representation is SEEN as antenna temperature.

The second half of the modular sky abstraction (see
:mod:`~rheplicant.radio.sky.model`). A projector maps sky maps to the
``(n_time, n_freq)`` time-ordered antenna temperature, given the observation
coordinates. Swapping the projector swaps the observation engine without
touching the sky model — and *linear* projectors additionally expose
``adjoint``, which :class:`~rheplicant.radio.filters.SkySpaceFilter` reuses for
map-making / sky-space filtering.

Three engines. The two that compute the physics live in sibling modules,
named for the observation geometry they serve — look there first:

- :class:`~rheplicant.radio.sky.general_pointing.GeneralPointingProjector` —
  pure JAX, any pointing, differentiable in sky *and* beam. The default.
- :class:`~rheplicant.radio.sky.driftscan.DriftScanProjector` — the same
  physics for a drift scan (fixed pointing, only LST advancing) at
  ``O(lmax³ + n_time·lmax)`` instead of ``O(n_time·lmax³)``. Equal to the
  general engine to float64 roundoff — an optimization, not an
  approximation — so on RHINO's static zenith pointing it is simply the
  right engine.

and one that takes the projection as data, defined here:

- :class:`MatrixProjector` — a precomputed sky->TOD matrix (e.g. from
  ``limTOD.simulator.generate_sky2sys_projection``). Fully differentiable
  TODAY: the matrix is built offline once, the JAX side is pure einsum,
  and it needs no optional dependency.
"""

import abc

import equinox as eqx
import jax
import jax.numpy as jnp

from rheplicant.core.coordinates import Coordinates
from rheplicant.core.errors import StateValidationError


class AbstractSkyProjector(eqx.Module):
    """Sky representation ``(n_freq, n_pix)`` -> antenna temperature ``(n_time, n_freq)``."""

    @abc.abstractmethod
    def forward(self, sky: jax.Array, coords: Coordinates) -> jax.Array:
        """Observe the sky: ``(n_freq, n_pix) -> (n_time, n_freq)``."""

    def adjoint(self, tod: jax.Array, coords: Coordinates) -> jax.Array:
        """Adjoint map ``(n_time, n_freq) -> (n_freq, n_pix)`` (linear projectors only).

        Required by sky-space filtering / map-making. Every shipped engine is
        linear and implements it; a nonlinear one may leave it unimplemented.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement adjoint; sky-space filtering "
            "needs a linear projector: GeneralPointingProjector (general pointing), "
            "DriftScanProjector (drift scans, much cheaper), or MatrixProjector "
            "(precomputed matrix)."
        )


class MatrixProjector(AbstractSkyProjector):
    """Linear projection by a precomputed sky->TOD matrix.

    The matrix is exactly what ``limTOD.simulator.generate_sky2sys_projection``
    produces (beam-weighted pointing rows over selected sky pixels): build it
    once offline with the existing numpy limTOD, load it here, and the whole
    sky term is differentiable (w.r.t. the *sky*) with zero porting work.
    Valid while pointing and beam are fixed.

    For a drift scan specifically, prefer
    :class:`~rheplicant.radio.sky.driftscan.DriftScanProjector`: no offline
    matrix to build or store, differentiable in the beam as well as the sky,
    and it derives the projection on the fly for less than the matrix costs
    to apply.

    Attributes:
        matrix: ``(n_time, n_pix)`` shared across frequency (achromatic beam),
            or ``(n_freq, n_time, n_pix)`` for a chromatic beam.
    """

    matrix: jax.Array

    def __check_init__(self):
        if self.matrix.ndim not in (2, 3):
            raise StateValidationError(
                f"matrix must be (n_time, n_pix) or (n_freq, n_time, n_pix), "
                f"got ndim={self.matrix.ndim}."
            )

    def _check_pix(self, sky: jax.Array):
        if sky.shape[-1] != self.matrix.shape[-1]:
            raise StateValidationError(
                f"sky has {sky.shape[-1]} pixels but the projection matrix has "
                f"{self.matrix.shape[-1]}."
            )

    def forward(self, sky: jax.Array, coords: Coordinates) -> jax.Array:
        self._check_pix(sky)
        if self.matrix.ndim == 2:
            return jnp.einsum("tp,fp->tf", self.matrix, sky)
        return jnp.einsum("ftp,fp->tf", self.matrix, sky)

    def adjoint(self, tod: jax.Array, coords: Coordinates) -> jax.Array:
        if self.matrix.ndim == 2:
            return jnp.einsum("tp,tf->fp", self.matrix, tod)
        return jnp.einsum("ftp,tf->fp", self.matrix, tod)
