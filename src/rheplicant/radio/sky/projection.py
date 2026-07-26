"""Sky projectors: how a sky representation is SEEN as antenna temperature.

The second half of the modular sky abstraction (see
:mod:`~rheplicant.radio.sky.model`). A projector maps sky maps to the
``(n_time, n_freq)`` time-ordered antenna temperature, given the observation
coordinates. Swapping the projector swaps the observation engine without
touching the sky model — and *linear* projectors additionally expose
``adjoint``, which :class:`~rheplicant.radio.filters.SkySpaceFilter` reuses for
map-making / sky-space filtering.

Five engines. The two REAL ones live in sibling modules — look there first:

- :class:`~rheplicant.radio.sky.native.NativeLimTODProjector` — pure JAX,
  general pointing, differentiable in sky *and* beam. The default choice.
- :class:`~rheplicant.radio.sky.driftscan.DriftScanProjector` — the same
  physics for a drift scan (fixed pointing, only LST advancing) at
  ``O(lmax³ + n_time·lmax)`` instead of ``O(n_time·lmax³)``. Equal to the
  general engine to float64 roundoff — an optimization, not an
  approximation — so on RHINO's static zenith pointing it is simply the
  right engine.

and three supporting ones, defined here:

- :class:`MatrixProjector` — a precomputed sky->TOD matrix (e.g. from
  ``limTOD.simulator.generate_sky2sys_projection``). Fully differentiable
  TODAY: the matrix is built offline once, the JAX side is pure einsum.
- :class:`LimTODProjector` — oracle bridge to numpy limTOD via
  ``jax.pure_callback``: jit-compatible, NOT differentiable. For forward
  simulation and validation of the native JAX port.
- :class:`MModeProjector` — PLACEHOLDER m-mode transfer matrices, kept as
  the minimal statement of the m-mode contract;
  :class:`~rheplicant.radio.sky.driftscan.DriftScanProjector` supersedes it
  for real work.
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

        Required by sky-space filtering / map-making. Nonlinear or oracle
        projectors may leave this unimplemented.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement adjoint; sky-space filtering "
            "needs a linear projector: NativeLimTODProjector (general pointing), "
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


class MModeProjector(AbstractSkyProjector):
    """PLACEHOLDER m-mode projection for drift-scan observations.

    For a periodic drift scan the antenna temperature is a Fourier series in
    LST: ``T(lst, f) = Re sum_m [B_m(f, :) . sky(f, :)] e^(i m lst)``. The
    real version — :class:`~rheplicant.radio.sky.driftscan.DriftScanProjector`
    — derives the m-mode projection from the beam alms on the fly (via
    ``limtod_jax.driftscan``); this placeholder takes transfer matrices as a
    given complex array and demonstrates the contract — including an exact
    adjoint (unitary FFT, ``norm="ortho"``), verified by dot-product tests.

    Attributes:
        transfer: ``(n_freq, n_m, n_pix)`` complex m-mode transfer matrices;
            ``n_m`` must equal ``n_time`` (full-FFT convention).
    """

    transfer: jax.Array

    def __check_init__(self):
        if self.transfer.ndim != 3:
            raise StateValidationError(
                f"transfer must be (n_freq, n_m, n_pix), got ndim={self.transfer.ndim}."
            )

    def _check_shapes(self, sky: jax.Array, n_time: int):
        if sky.shape[-1] != self.transfer.shape[-1] or sky.shape[0] != self.transfer.shape[0]:
            raise StateValidationError(
                f"sky shape {sky.shape} does not match transfer "
                f"(n_freq={self.transfer.shape[0]}, n_pix={self.transfer.shape[-1]})."
            )
        if self.transfer.shape[1] != n_time:
            raise StateValidationError(
                f"transfer has n_m={self.transfer.shape[1]} but coords.time has "
                f"{n_time} samples (full-FFT convention requires n_m == n_time)."
            )

    def forward(self, sky: jax.Array, coords: Coordinates) -> jax.Array:
        if coords is None or coords.time is None:
            raise StateValidationError("MModeProjector requires coords.time.")
        self._check_shapes(sky, coords.time.shape[0])
        modes = jnp.einsum("fmp,fp->fm", self.transfer, sky.astype(self.transfer.dtype))
        return jnp.real(jnp.fft.ifft(modes, axis=1, norm="ortho")).T

    def adjoint(self, tod: jax.Array, coords: Coordinates) -> jax.Array:
        spectra = jnp.fft.fft(tod.T.astype(self.transfer.dtype), axis=1, norm="ortho")
        return jnp.real(jnp.einsum("fmp,fm->fp", jnp.conj(self.transfer), spectra))


class LimTODProjector(AbstractSkyProjector):
    """Oracle bridge to numpy limTOD (``generate_TOD_sky``) via ``jax.pure_callback``.

    Jit-compatible but NOT differentiable and not vmappable — use for forward
    simulation and as the ground-truth oracle that the native JAX port
    (:class:`~rheplicant.radio.sky.native.NativeLimTODProjector`) is tested
    against. Requires the ``limTOD`` package to be importable.

    Coordinate conventions (degrees, per the RHINO family):

        * ``coords.extra["lst_deg"]`` — ``(n_time,)`` local sidereal times.
        * ``coords.pointing`` — ``(n_time, 2)`` azimuth/elevation [deg].
        * ``coords.extra["selfrot_deg"]`` — optional ``(n_time,)`` self-rotation
          (defaults to zero).

    Attributes:
        beam_maps: ``(n_freq, n_pix)`` HEALPix beam maps (traced array; passed
            through the callback).
        lat_deg: site latitude [deg] (static configuration).
        normalize_beam: forwarded to ``generate_TOD_sky`` (static).
    """

    beam_maps: jax.Array
    lat_deg: float = eqx.field(static=True)
    normalize_beam: bool = eqx.field(static=True, default=False)

    def forward(self, sky: jax.Array, coords: Coordinates) -> jax.Array:
        if coords is None or coords.pointing is None:
            raise StateValidationError(
                "LimTODProjector requires coords.pointing (n_time, 2) az/el in degrees."
            )
        if coords.extra.get("lst_deg") is None:
            raise StateValidationError(
                'LimTODProjector requires coords.extra["lst_deg"] (n_time,) in degrees.'
            )
        n_time = coords.pointing.shape[0]
        selfrot = coords.extra.get("selfrot_deg", jnp.zeros(n_time))
        dtype = jnp.result_type(sky.dtype, jnp.float32)
        out_spec = jax.ShapeDtypeStruct((n_time, sky.shape[0]), dtype)

        def host(sky_np, beams_np, lst_np, pointing_np, selfrot_np):  # numpy land
            import numpy as np

            try:
                from limTOD.simulator import generate_TOD_sky
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "LimTODProjector needs the limTOD package "
                    "(pip install -e <path-to-limTOD>); for a differentiable "
                    "alternative use MatrixProjector with a precomputed "
                    "generate_sky2sys_projection matrix."
                ) from exc

            columns = [
                generate_TOD_sky(
                    np.asarray(beams_np[f]),
                    np.asarray(sky_np[f]),
                    np.asarray(lst_np),
                    self.lat_deg,
                    np.asarray(pointing_np[:, 0]),
                    np.asarray(pointing_np[:, 1]),
                    np.asarray(selfrot_np),
                    normalize_beam=self.normalize_beam,
                )
                for f in range(sky_np.shape[0])
            ]
            return np.stack(columns, axis=1).astype(dtype)

        return jax.pure_callback(
            host, out_spec, sky, self.beam_maps, coords.extra["lst_deg"],
            coords.pointing, selfrot,
        )
