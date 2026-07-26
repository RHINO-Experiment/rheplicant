"""Drift-scan m-mode projector — the fast path for RHINO's actual geometry.

:class:`DriftScanProjector` is the "real version" that the
:class:`~rheplicant.radio.sky.projection.MModeProjector` placeholder
promised: instead of taking precomputed transfer matrices, it derives the
m-mode projection on the fly from the beam alms via
``limtod_jax.driftscan`` (one Wigner rotation for the whole scan plus
per-m phases). For a genuine drift scan — fixed azimuth/elevation/
self-rotation, only LST advancing — it reproduces
:class:`~rheplicant.radio.sky.native.NativeLimTODProjector` to roundoff
at O(lmax³ + n_time·lmax) instead of O(n_time·lmax³), with the same
exact sky-slot adjoint. Tracking or scanning strategies still need the
general projector.

Where the general projectors read the pointing from ``coords.pointing``
per call, here the pointing IS the projector configuration: az/el/selfrot
are static fields, and ``coords`` only supplies
``coords.extra["lst_deg"]``. ``coords.pointing``, if present, is ignored
by design — a drift scan that needs a per-sample pointing is not a drift
scan.

The optional horizon mask (``horizon_mask=True``) applies the physical
below-ground cut to the beam in the horizontal frame before projecting,
with cosine apodization (``apod_deg``) to tame the Gibbs ringing of a
hard cut at finite band-limit — see the ringing study in the limTOD docs
(``docs/driftscan.md``): narrow beams never need it; wide low-elevation
beams need it *and* 2–5° of apodization.

PRECISION: enable ``jax_enable_x64`` for quantitative work (the map<->alm
steps inherit s2fft's float32 limitation; see ``limtod_jax.hpx``).
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
            "DriftScanProjector needs the limtod_jax package with drift-scan "
            "support (limTOD >= 1.6): install the limTOD repo with its jax "
            "extra (pip install -e '<limTOD>[jax]')."
        ) from exc
    if not hasattr(limtod_jax, "driftscan"):  # pragma: no cover
        raise ImportError(
            "The installed limtod_jax predates the drift-scan m-mode path "
            "(limtod_jax.driftscan); upgrade the limTOD installation."
        )
    return limtod_jax


class DriftScanProjector(AbstractSkyProjector):
    """m-mode sky projector for drift scans: one beam rotation, per-m phases.

    Equal to :class:`~rheplicant.radio.sky.native.NativeLimTODProjector`
    with constant pointing, to float64 roundoff — but the whole scan costs
    a single Wigner rotation. Pure JAX (jit/vmap/grad-safe), exact
    transpose in :meth:`adjoint`, and :meth:`mmodes` exposes the m-mode
    coefficients (the Fourier series of the sidereal-day TOD) directly
    for m-mode analyses.

    Coordinate conventions (degrees, per the RHINO family):

        * ``coords.extra["lst_deg"]`` — ``(n_time,)`` local sidereal times.
        * ``coords.pointing`` is IGNORED: the drift pointing is projector
          configuration (static fields below), not per-sample data.

    Attributes:
        beam_alms: ``(n_freq, n_alm)`` packed healpy beam alms in the
            BEAM-LOCAL frame (traced — beam parameters are differentiable).
            Compute them as numpy limTOD does (``hp.map2alm(beam_map)``);
            valid real-field alms (m = 0 coefficients real).
        lat_deg: site latitude [deg] (static).
        az_deg / el_deg / selfrot_deg: the fixed drift-scan pointing [deg]
            (static configuration).
        lmax: harmonic band-limit matching ``beam_alms`` (static).
        nside: HEALPix nside of the sky maps, RING ordering (static).
        normalize_beam: numpy limTOD's ``normalize_beam`` semantics
            (static).
        horizon_mask: apply the below-horizon cut to the beam in the
            horizontal frame before projecting (static; default off,
            matching numpy limTOD).
        apod_deg: cosine-apodization width of the horizon cut [deg of
            elevation] (static; only used with ``horizon_mask``).
        mask_iterations: healpy-equivalent ``map2alm`` iterations in the
            mask re-analysis (static).
        lst_ref_deg: reference LST [deg] of the m-mode expansion (static);
            ``None`` uses the first sample of ``coords.extra["lst_deg"]``.
            Any value gives the same TOD — it only re-anchors the phases.
    """

    beam_alms: jax.Array
    lat_deg: float = eqx.field(static=True)
    az_deg: float = eqx.field(static=True)
    el_deg: float = eqx.field(static=True)
    lmax: int = eqx.field(static=True)
    nside: int = eqx.field(static=True)
    selfrot_deg: float = eqx.field(static=True, default=0.0)
    normalize_beam: bool = eqx.field(static=True, default=False)
    horizon_mask: bool = eqx.field(static=True, default=False)
    apod_deg: float = eqx.field(static=True, default=0.0)
    mask_iterations: int = eqx.field(static=True, default=3)
    lst_ref_deg: float | None = eqx.field(static=True, default=None)

    def __check_init__(self):
        n_alm = (self.lmax + 1) * (self.lmax + 2) // 2
        if self.beam_alms.ndim != 2 or self.beam_alms.shape[-1] != n_alm:
            raise StateValidationError(
                f"beam_alms must be (n_freq, n_alm={n_alm}) packed alms for "
                f"lmax={self.lmax}, got shape {self.beam_alms.shape}."
            )

    # ------------------------------------------------------------------ utils
    def _validate_coords(self, coords: Coordinates) -> None:
        if coords is None or coords.extra.get("lst_deg") is None:
            raise StateValidationError(
                'DriftScanProjector requires coords.extra["lst_deg"] '
                "(n_time,) in degrees. (coords.pointing is ignored: the "
                "drift pointing lives on the projector.)"
            )

    def _dphi_and_ref(self, coords: Coordinates):
        lst = coords.extra["lst_deg"]
        ref = self.lst_ref_deg if self.lst_ref_deg is not None else lst[0]
        return jnp.deg2rad(lst - ref), ref

    def _beam_ref_alms(self, ltj, ref) -> jax.Array:
        """Per-frequency celestial-frame beam alms at the reference LST."""
        beam = self.beam_alms
        if self.horizon_mask:
            beam = jax.vmap(
                lambda b: ltj.horizon_masked_beam_alm(
                    b,
                    self.az_deg,
                    self.el_deg,
                    self.selfrot_deg,
                    nside=self.nside,
                    lmax=self.lmax,
                    apod_deg=self.apod_deg,
                    iterations=self.mask_iterations,
                )
            )(beam)
        return jax.vmap(
            lambda b: ltj.beam_alm_at_reference(
                b,
                ref,
                self.lat_deg,
                self.az_deg,
                self.el_deg,
                self.selfrot_deg,
                lmax=self.lmax,
            )
        )(beam)

    def _ones_alm(self, ltj) -> jax.Array | None:
        if not self.normalize_beam:
            return None
        # Pure function of static (nside, lmax): a constant subgraph under jit.
        return ltj.ones_quadrature_alm(nside=self.nside, lmax=self.lmax)

    def _validate_sky(self, sky: jax.Array) -> None:
        n_pix = 12 * self.nside**2
        if sky.shape[-1] != n_pix or sky.shape[0] != self.beam_alms.shape[0]:
            raise StateValidationError(
                f"sky must be (n_freq={self.beam_alms.shape[0]}, "
                f"n_pix={n_pix}) for nside={self.nside}, got {sky.shape}."
            )

    # ------------------------------------------------------------- interface
    def forward(self, sky: jax.Array, coords: Coordinates) -> jax.Array:
        self._validate_coords(coords)
        self._validate_sky(sky)
        ltj = _limtod_jax()
        dphi, ref = self._dphi_and_ref(coords)
        beam_refs = self._beam_ref_alms(ltj, ref)
        ones_alm = self._ones_alm(ltj)

        def one_freq(beam_ref, sky_map):
            sky_alm = ltj.map2alm_quad(sky_map, nside=self.nside, lmax=self.lmax)
            return ltj.driftscan_tod(
                beam_ref, sky_alm, dphi,
                lmax=self.lmax, normalize=self.normalize_beam, ones_alm=ones_alm,
            )

        return jax.vmap(one_freq)(beam_refs, sky).T

    def adjoint(self, tod: jax.Array, coords: Coordinates) -> jax.Array:
        self._validate_coords(coords)
        n_time, n_freq = coords.extra["lst_deg"].shape[0], self.beam_alms.shape[0]
        if tod.ndim != 2 or tod.shape[0] != n_time or tod.shape[1] != n_freq:
            raise StateValidationError(
                f"tod must be (n_time={n_time}, n_freq={n_freq}), got {tod.shape}."
            )
        ltj = _limtod_jax()
        dphi, ref = self._dphi_and_ref(coords)
        beam_refs = self._beam_ref_alms(ltj, ref)
        ones_alm = self._ones_alm(ltj)

        def one_freq(beam_ref, tod_t):
            alm = ltj.driftscan_tod_adjoint(
                tod_t, beam_ref, dphi,
                lmax=self.lmax, normalize=self.normalize_beam, ones_alm=ones_alm,
            )
            return ltj.alm2map(alm, nside=self.nside, lmax=self.lmax)

        return jax.vmap(one_freq)(beam_refs, tod.T)

    def mmodes(self, sky: jax.Array, coords: Coordinates) -> jax.Array:
        """m-modes ``Ṽ_m`` of the drift-scan TOD, per frequency.

        Returns a complex ``(n_freq, lmax+1)`` array — the Fourier
        coefficients of the (sidereal-day-periodic) TOD, m ≥ 0 (real
        fields make the negative-m half redundant). ``coords`` supplies
        the reference LST anchoring the phases (first sample unless
        ``lst_ref_deg`` is set); the coefficients' magnitudes are
        sampling-independent.
        """
        self._validate_coords(coords)
        self._validate_sky(sky)
        ltj = _limtod_jax()
        _, ref = self._dphi_and_ref(coords)
        beam_refs = self._beam_ref_alms(ltj, ref)

        def one_freq(beam_ref, sky_map):
            sky_alm = ltj.map2alm_quad(sky_map, nside=self.nside, lmax=self.lmax)
            return ltj.mmodes_from_sky(beam_ref, sky_alm, lmax=self.lmax)

        return jax.vmap(one_freq)(beam_refs, sky)
