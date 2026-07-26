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

Two static opt-ins turn the projector from "correct" into "fast for
inference", both preserving full jit/vmap/grad behaviour:

* :meth:`DriftScanProjector.to_reference_frame` pays the O(lmax³) Wigner
  rotation ONCE and returns an equivalent projector that skips it on every
  later call — the difference between rotating once and rotating per
  likelihood evaluation;
* ``uniform_sampling=True`` routes the time synthesis (and its adjoint)
  through real FFTs, O(n_time·log n_time) independent of lmax, when the LST
  grid is uniform over a full sidereal turn.

The optional horizon mask (``horizon_mask=True``) applies the physical
below-ground cut to the beam in the horizontal frame before projecting,
with cosine apodization (``apod_deg``) to tame the Gibbs ringing of a
hard cut at finite band-limit — see the ringing study in the limTOD docs
(``docs/driftscan.md``): narrow beams never need it; wide low-elevation
beams need it *and* 2–5° of apodization.

PRECISION: enable ``jax_enable_x64`` for quantitative work (the map<->alm
steps inherit s2fft's float32 limitation; see ``limtod_jax.hpx``).
"""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp

from rheplicant.core.coordinates import Coordinates
from rheplicant.core.errors import StateValidationError
from rheplicant.radio.sky.projection import AbstractSkyProjector


def _limtod_jax(uniform: bool = False):
    """The limtod_jax module, gated on the symbols this adapter actually calls.

    Two feature levels, so an outdated install fails with a clear message at
    the boundary instead of an AttributeError deep inside a traced call:
    the drift-scan path needs limTOD >= 1.6, and ``uniform_sampling`` needs
    the FFT fast path plus the public grid check added in 1.7.
    """
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
    if uniform and not hasattr(limtod_jax, "check_uniform_grid"):
        raise ImportError(
            "uniform_sampling=True needs the uniform-grid FFT fast path added "
            "in limTOD 1.7 (limtod_jax.check_uniform_grid and the uniform= "
            "argument); the installed limtod_jax is older. Upgrade limTOD, or "
            "drop the flag to use the exact direct sum."
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
        beam_frame: ``"local"`` (default) — ``beam_alms`` are beam-local and
            the reference rotation happens on every call, keeping gradients
            w.r.t. the beam-local alms; ``"reference"`` — they are already
            the celestial-frame alms at ``lst_ref_deg``, so the rotation is
            skipped. Build the latter with :meth:`to_reference_frame`
            (static).
        uniform_sampling: use the FFT synthesis/adjoint (static). Requires
            ``coords.extra["lst_deg"]`` to be a uniform grid over a full
            sidereal turn with ``2·lmax < n_time``; validated by
            ``limtod_jax`` whenever the values are concrete. Gradients w.r.t.
            the LST grid then live on a one-parameter family (a global
            shift), so ``dphi``'s Jacobian is a single column — exact for any
            timing parameter that keeps the grid a uniform full turn, and
            undefined for per-sample perturbations, which are rejected or
            NaN-poisoned rather than fitted.
        beam_ref_lst_deg: set only by :meth:`to_reference_frame` — the LST the
            cached beam was actually rotated to (static). In
            ``"reference"`` mode it must equal ``lst_ref_deg``; the pair is
            what makes an attempt to re-anchor the phases against a stale
            cached rotation fail loudly instead of silently.
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
    beam_frame: str = eqx.field(static=True, default="local")
    uniform_sampling: bool = eqx.field(static=True, default=False)
    beam_ref_lst_deg: float | None = eqx.field(static=True, default=None)

    def __check_init__(self):
        n_alm = (self.lmax + 1) * (self.lmax + 2) // 2
        if self.beam_alms.ndim != 2 or self.beam_alms.shape[-1] != n_alm:
            raise StateValidationError(
                f"beam_alms must be (n_freq, n_alm={n_alm}) packed alms for "
                f"lmax={self.lmax}, got shape {self.beam_alms.shape}."
            )
        if self.beam_frame not in ("local", "reference"):
            raise StateValidationError(
                f'beam_frame must be "local" or "reference", got '
                f"{self.beam_frame!r}."
            )
        if self.beam_frame == "reference":
            if self.lst_ref_deg is None:
                raise StateValidationError(
                    'beam_frame="reference" requires an explicit lst_ref_deg: '
                    "the cached beam was rotated to ONE specific reference LST, "
                    "which must not silently depend on the coords passed later. "
                    "Use to_reference_frame() to build this state correctly."
                )
            # beam_ref_lst_deg records the LST the cached beam was ACTUALLY
            # rotated to, and only to_reference_frame() sets it. Without the
            # pair, dataclasses.replace(cached, lst_ref_deg=...) would re-anchor
            # the phases against a stale rotation and return a silently wrong
            # TOD — the invariant is what makes that combination loud.
            if self.beam_ref_lst_deg != self.lst_ref_deg:
                raise StateValidationError(
                    f"cached beam is anchored at lst_ref_deg="
                    f"{self.beam_ref_lst_deg}, but lst_ref_deg is now "
                    f"{self.lst_ref_deg}: the phases would be measured from a "
                    f"reference the cached rotation does not correspond to. "
                    f"Re-anchor from the beam-local projector via "
                    f"to_reference_frame(lst_ref_deg=...) instead of editing "
                    f"the field."
                )
            if self.horizon_mask:
                raise StateValidationError(
                    'beam_frame="reference" is incompatible with '
                    "horizon_mask=True: the mask must be applied in the "
                    "horizontal frame, i.e. BEFORE the reference rotation. "
                    "Build the projector with beam_frame='local' and "
                    "horizon_mask=True, then call to_reference_frame()."
                )

    # ------------------------------------------------------------------ utils
    def _validate_coords(self, coords: Coordinates) -> None:
        if coords is None or coords.extra.get("lst_deg") is None:
            raise StateValidationError(
                'DriftScanProjector requires coords.extra["lst_deg"] '
                "(n_time,) in degrees. (coords.pointing is ignored: the "
                "drift pointing lives on the projector.)"
            )
        if self.uniform_sampling:
            self._validate_uniform_grid(coords)

    def _validate_uniform_grid(self, coords: Coordinates) -> None:
        """Check the FFT contract at THIS boundary, where it can still be seen.

        The opt-in lives on the projector but the grid arrives per call, so
        this is the only place a clear error is possible. It matters that the
        check happens on the RAW ``lst_deg``: once inside a ``jit`` trace,
        deriving ``dphi`` (``jnp.deg2rad(lst - ref)``) yields a tracer even
        when the grid is a compile-time constant, so limtod_jax's own eager
        check would silently skip. The raw closure constant is usually still
        concrete here, which turns "silently NaN at runtime" into "clear
        ValueError at trace time". When it genuinely is traced, limtod_jax's
        pure-JAX guard still poisons a violated contract with NaN.
        """
        import numpy as np

        ltj = _limtod_jax(self.uniform_sampling)
        lst = coords.extra["lst_deg"]
        try:
            raw = np.asarray(lst)
        except jax.errors.TracerArrayConversionError:
            return  # genuinely traced: the NaN guard downstream covers it
        ref = self.lst_ref_deg if self.lst_ref_deg is not None else float(raw[0])
        # Check at the grid's NATIVE dtype. Upcasting to float64 first would
        # hide the real precision, and limtod_jax's tolerance is dtype-scaled:
        # a legitimate float32 degree grid deviates ~3e-7 rad, which the f64
        # bound (~9e-14) rejects outright.
        ltj.check_uniform_grid(np.deg2rad(raw - np.asarray(ref, raw.dtype)))

    def _dphi_and_ref(self, coords: Coordinates):
        lst = coords.extra["lst_deg"]
        ref = self.lst_ref_deg if self.lst_ref_deg is not None else lst[0]
        return jnp.deg2rad(lst - ref), ref

    def _beam_ref_alms(self, ltj, ref) -> jax.Array:
        """Per-frequency celestial-frame beam alms at the reference LST.

        With ``beam_frame="reference"`` the stored alms already ARE that, so
        the O(lmax³) rotation is skipped entirely — the point of
        :meth:`to_reference_frame` for repeated (inference-loop) evaluation.
        """
        if self.beam_frame == "reference":
            return self.beam_alms
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
        ltj = _limtod_jax(self.uniform_sampling)
        dphi, ref = self._dphi_and_ref(coords)
        beam_refs = self._beam_ref_alms(ltj, ref)
        ones_alm = self._ones_alm(ltj)

        def one_freq(beam_ref, sky_map):
            sky_alm = ltj.map2alm_quad(sky_map, nside=self.nside, lmax=self.lmax)
            return ltj.driftscan_tod(
                beam_ref, sky_alm, dphi,
                lmax=self.lmax, normalize=self.normalize_beam, ones_alm=ones_alm,
                uniform=self.uniform_sampling,
            )

        return jax.vmap(one_freq)(beam_refs, sky).T

    def adjoint(self, tod: jax.Array, coords: Coordinates) -> jax.Array:
        self._validate_coords(coords)
        n_time, n_freq = coords.extra["lst_deg"].shape[0], self.beam_alms.shape[0]
        if tod.ndim != 2 or tod.shape[0] != n_time or tod.shape[1] != n_freq:
            raise StateValidationError(
                f"tod must be (n_time={n_time}, n_freq={n_freq}), got {tod.shape}."
            )
        ltj = _limtod_jax(self.uniform_sampling)
        dphi, ref = self._dphi_and_ref(coords)
        beam_refs = self._beam_ref_alms(ltj, ref)
        ones_alm = self._ones_alm(ltj)

        def one_freq(beam_ref, tod_t):
            alm = ltj.driftscan_tod_adjoint(
                tod_t, beam_ref, dphi,
                lmax=self.lmax, normalize=self.normalize_beam, ones_alm=ones_alm,
                uniform=self.uniform_sampling,
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

        Requires ``normalize_beam=False``: normalization divides the TOD by
        the ones-map denominator, which is not part of the m-mode expansion,
        so these coefficients would no longer be the spectrum of what
        :meth:`forward` returns (measured ~18x off). Rejected rather than
        silently mismatched — the same policy as the ``"reference"``/mask
        combination above.
        """
        if self.normalize_beam:
            raise StateValidationError(
                "mmodes() is defined for the UN-normalized sky TOD, but this "
                "projector has normalize_beam=True: forward() additionally "
                "divides by the ones-map denominator (the beam integral), "
                "which is not representable in the m-mode expansion, so the "
                "returned coefficients would not be the spectrum of "
                "forward(). Use a normalize_beam=False projector for m-mode "
                "analysis (the denominator is constant along a drift, so it "
                "is an overall scale you can divide out yourself)."
            )
        self._validate_coords(coords)
        self._validate_sky(sky)
        ltj = _limtod_jax(self.uniform_sampling)
        _, ref = self._dphi_and_ref(coords)
        beam_refs = self._beam_ref_alms(ltj, ref)

        def one_freq(beam_ref, sky_map):
            sky_alm = ltj.map2alm_quad(sky_map, nside=self.nside, lmax=self.lmax)
            return ltj.mmodes_from_sky(beam_ref, sky_alm, lmax=self.lmax)

        return jax.vmap(one_freq)(beam_refs, sky)

    def to_reference_frame(
        self, lst_ref_deg: float | None = None
    ) -> "DriftScanProjector":
        """Precompute the beam rotation once; return an equivalent projector.

        The returned projector holds the celestial-frame beam alms at
        ``lst_ref_deg`` (mask already applied if it was configured) and
        skips the O(lmax³) Wigner rotation on every subsequent
        ``forward``/``adjoint``/``mmodes`` — the difference between paying
        the rotation once and paying it per likelihood evaluation. Call it
        OUTSIDE the inference loop.

        Fully functional: ``self`` is unchanged, and this is pure JAX, so it
        is itself differentiable and jit-safe. Gradients through the RESULT
        are with respect to the reference-frame alms; if you need gradients
        w.r.t. the beam-local alms (or w.r.t. pointing), keep the
        ``"local"`` projector — which is exactly the compute-vs-flexibility
        trade this method exposes rather than hides.

        Args:
            lst_ref_deg: reference LST [deg]; defaults to this projector's
                ``lst_ref_deg``, which must then be set (a cached rotation
                cannot depend on coords supplied later).
        """
        if self.beam_frame == "reference":
            if lst_ref_deg is not None and lst_ref_deg != self.lst_ref_deg:
                raise StateValidationError(
                    f"this projector's beam is already cached at "
                    f"lst_ref_deg={self.lst_ref_deg}; re-anchoring to "
                    f"{lst_ref_deg} would need the beam-local alms. Rebuild "
                    'from the "local" projector instead.'
                )
            return self
        ref = lst_ref_deg if lst_ref_deg is not None else self.lst_ref_deg
        if ref is None:
            raise StateValidationError(
                "to_reference_frame() needs a reference LST: pass "
                "lst_ref_deg=... or set it on the projector. It cannot "
                "default to coords.extra['lst_deg'][0] here, because the "
                "rotation is baked in before any coords are seen."
            )
        ltj = _limtod_jax(self.uniform_sampling)
        return dataclasses.replace(
            self,
            beam_alms=self._beam_ref_alms(ltj, ref),
            lst_ref_deg=float(ref),
            beam_frame="reference",
            horizon_mask=False,  # already applied into the cached alms
            beam_ref_lst_deg=float(ref),  # the invariant __check_init__ checks
        )
