"""The general-pointing sky engine: any pointing, one rotation per sample.

:class:`GeneralPointingProjector` assumes nothing about the observation —
azimuth, elevation and self-rotation are per-sample DATA, so tracking,
raster scans and transits all work — and pays for that generality with one
O(lmax³) Wigner rotation per time sample. Pure JAX, differentiable w.r.t.
BOTH the sky maps and the beam alms, with the exact transpose that
:class:`~rheplicant.radio.filters.SkySpaceFilter` map-making requires.

Its sibling :class:`~rheplicant.radio.sky.driftscan.DriftScanProjector`
covers the one geometry this engine cannot exploit: a drift scan, where the
pointing never moves. There the rotation happens ONCE for the whole scan,
reproducing this projector to float64 roundoff for a fraction of the cost.
Naming follows that split — the two real engines are named for the
observation geometry they serve, not for the package they were ported from
(both come from ``limtod_jax``).

The heavy lifting lives in the ``limtod_jax`` package (shipped with the
limTOD repo: ``pip install "limTOD[jax]"``); this adapter only wires it to
the :class:`~rheplicant.radio.sky.projection.AbstractSkyProjector` seam. It is
imported lazily so rheplicant's dependencies are unchanged.

Semantics: per frequency, ``forward`` equals numpy
``limTOD.simulator.generate_TOD_sky(..., truncate_frac_thres=0.0)`` — the
LINEAR chain (the default ``1e-10`` truncation is a nonlinear cleanup
outside the port contract) — to float64 roundoff when x64 is enabled.

PRECISION: enable ``jax_enable_x64`` for quantitative work. The map<->alm
steps (s2fft healpix transforms, Price-McEwen recursion) carry O(10%)
errors in float32 even at small lmax; the Wigner rotation core is
float32-stable, but the projector as a whole inherits the transform error
(see ``limtod_jax.hpx``).
"""

import equinox as eqx
import jax
import jax.numpy as jnp

from rheplicant.core.coordinates import Coordinates
from rheplicant.core.errors import StateValidationError
from rheplicant.radio.sky.projection import AbstractSkyProjector


def _limtod_jax():
    try:
        import limtod_jax
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "GeneralPointingProjector needs the limtod_jax package: install the "
            "limTOD repo with its jax extra (pip install -e '<limTOD>[jax]'). "
            "Without it, MatrixProjector takes a precomputed sky->TOD matrix "
            "and needs no optional dependency (fixed pointing and beam only)."
        ) from exc
    return limtod_jax


class GeneralPointingProjector(AbstractSkyProjector):
    """Pure-JAX limTOD sky projector: jit/vmap/grad-safe with exact adjoint.

    Handles ARBITRARY pointing, one Wigner rotation per time sample. If the
    observation is a drift scan — fixed azimuth/elevation/self-rotation, only
    LST advancing — use
    :class:`~rheplicant.radio.sky.driftscan.DriftScanProjector` instead: it
    returns the same numbers to float64 roundoff for a single rotation over
    the whole scan (``O(lmax³ + n_time·lmax)`` vs ``O(n_time·lmax³)``).

    Coordinate conventions (degrees, per the RHINO family):

        * ``coords.extra["lst_deg"]`` — ``(n_time,)`` local sidereal times.
        * ``coords.pointing`` — ``(n_time, 2)`` azimuth/elevation [deg].
        * ``coords.extra["selfrot_deg"]`` — optional ``(n_time,)``
          self-rotation (defaults to zero).

    Attributes:
        beam_alms: ``(n_freq, n_alm)`` packed healpy beam alms (traced —
            beam parameters are differentiable). Compute them as numpy
            limTOD does (``hp.map2alm(beam_map, lmax=lmax)``) for oracle
            equivalence. Must be VALID real-field alms (m=0 coefficients
            real — automatic for map2alm output); forward/adjoint are exact
            transposes on that subspace.
        lat_deg: site latitude [deg] (static).
        lmax: harmonic band-limit; must match ``beam_alms`` length (static).
        nside: HEALPix nside of the sky maps, RING ordering (static).
        normalize_beam: numpy limTOD's ``normalize_beam`` semantics — divide
            each sample by the rotated beam's pixel sum (static).
    """

    beam_alms: jax.Array
    lat_deg: float = eqx.field(static=True)
    lmax: int = eqx.field(static=True)
    nside: int = eqx.field(static=True)
    normalize_beam: bool = eqx.field(static=True, default=False)

    def __check_init__(self):
        # Deliberately inline (== limtod_jax.alm.nalm_of_lmax) so that shape
        # validation works even when the optional limtod_jax isn't installed.
        n_alm = (self.lmax + 1) * (self.lmax + 2) // 2
        if self.beam_alms.ndim != 2 or self.beam_alms.shape[-1] != n_alm:
            raise StateValidationError(
                f"beam_alms must be (n_freq, n_alm={n_alm}) packed alms for "
                f"lmax={self.lmax}, got shape {self.beam_alms.shape}."
            )

    # ------------------------------------------------------------------ utils
    def _validate_coords(self, coords: Coordinates) -> None:
        if coords is None or coords.pointing is None:
            raise StateValidationError(
                "GeneralPointingProjector requires coords.pointing (n_time, 2) "
                "az/el in degrees."
            )
        if coords.extra.get("lst_deg") is None:
            raise StateValidationError(
                'GeneralPointingProjector requires coords.extra["lst_deg"] '
                "(n_time,) in degrees."
            )

    def _zyz(self, ltj, coords: Coordinates) -> jax.Array:
        assert coords.pointing is not None  # _validate_coords ran first
        n_time = coords.pointing.shape[0]
        selfrot = coords.extra.get("selfrot_deg", jnp.zeros(n_time))
        psi, theta, phi = ltj.zyz_of_pointing(
            coords.extra["lst_deg"],
            self.lat_deg,
            coords.pointing[:, 0],
            coords.pointing[:, 1],
            selfrot,
        )
        return jnp.stack([psi, theta, phi], axis=-1)

    def _ones_alm(self, ltj) -> jax.Array | None:
        """Quadrature alms of the ones map — the normalization denominator.

        A pure function of the static ``(nside, lmax)``, but XLA does NOT
        constant-fold it (an earlier version of this comment claimed it did):
        it is a full s2fft analysis, too large for the folding budget. Traced,
        at nside 64 / lmax 191 in x64, it costs 64 ms and 10 MB per call and
        7700 lines of HLO; hoisted into the JAX constant-evaluation context,
        0.04 ms, 0.15 MB, 48 lines, bitwise identical.
        """
        if not self.normalize_beam:
            return None
        with jax.ensure_compile_time_eval():
            return ltj.ones_quadrature_alm(nside=self.nside, lmax=self.lmax)

    # ------------------------------------------------------------- interface
    def forward(self, sky: jax.Array, coords: Coordinates) -> jax.Array:
        self._validate_coords(coords)
        n_pix = 12 * self.nside**2
        if sky.shape[-1] != n_pix or sky.shape[0] != self.beam_alms.shape[0]:
            raise StateValidationError(
                f"sky must be (n_freq={self.beam_alms.shape[0]}, "
                f"n_pix={n_pix}) for nside={self.nside}, got {sky.shape}."
            )
        ltj = _limtod_jax()
        angles = self._zyz(ltj, coords)
        ones_alm = self._ones_alm(ltj)

        def one_freq(beam_alm, sky_map):
            sky_alm = ltj.map2alm_quad(sky_map, nside=self.nside, lmax=self.lmax)
            return ltj.generate_tod_sky(
                beam_alm, sky_alm, angles,
                lmax=self.lmax, normalize=self.normalize_beam, ones_alm=ones_alm,
            )

        return jax.vmap(one_freq)(self.beam_alms, sky).T

    def adjoint(self, tod: jax.Array, coords: Coordinates) -> jax.Array:
        self._validate_coords(coords)
        assert coords.pointing is not None  # narrowed by _validate_coords
        n_time, n_freq = coords.pointing.shape[0], self.beam_alms.shape[0]
        if tod.ndim != 2 or tod.shape[0] != n_time or tod.shape[1] != n_freq:
            raise StateValidationError(
                f"tod must be (n_time={n_time}, n_freq={n_freq}), "
                f"got {tod.shape}."
            )
        ltj = _limtod_jax()
        angles = self._zyz(ltj, coords)
        ones_alm = self._ones_alm(ltj)

        def one_freq(beam_alm, tod_t):
            alm = ltj.generate_tod_sky_adjoint(
                tod_t, beam_alm, angles,
                lmax=self.lmax, normalize=self.normalize_beam, ones_alm=ones_alm,
            )
            return ltj.alm2map(alm, nside=self.nside, lmax=self.lmax)

        return jax.vmap(one_freq)(self.beam_alms, tod.T)
