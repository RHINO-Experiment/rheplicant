"""The chromatic beam and the drift-scan projection, through limtod_jax.

Beams: an analytic chromatic Gaussian (main scenario; FWHM 30 deg at 70 MHz
scaling as ``1/nu``) or RHINO's simulated horn (stress scenario;
``HornWet{f}.fits``, NSIDE 512, one file per 0.5 MHz). Both have their
boresight at the north pole of the beam-local frame, limtod_jax's convention.

The projection is limtod_jax's m-mode drift scan
(``limtod_jax.driftscan.DriftScanMmode``): the beam is rotated to the
celestial frame once, and each LST sample is a per-m phase. The beam enters
as true alms (``healpy.map2alm``), the sky as quadrature alms
(``map2alm_quad``), and ``normalize=True`` divides by the beam's pixel sum
so the output is an antenna temperature in K.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
from limtod_jax import DriftScanMmode, map2alm_quad

from global21cm import scenario


def gaussian_beam_maps(freqs_mhz, nside: int) -> np.ndarray:
    """Chromatic Gaussian beam maps ``(n_freq, n_pix)``, peak 1, beam-local frame."""
    import healpy as hp

    theta, _ = hp.pix2ang(nside, np.arange(12 * nside**2))
    fwhm = np.deg2rad(scenario.GAUSSIAN_FWHM70_DEG) * 70.0 / np.asarray(freqs_mhz)
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return np.exp(-0.5 * (theta[None, :] / sigma[:, None]) ** 2)


def hornwet_beam_maps(freqs_mhz, nside: int) -> np.ndarray:
    """HornWet maps ``(n_freq, n_pix)`` at ``nside``, each divided by its peak."""
    import healpy as hp

    directory = scenario.hornwet_dir()
    maps = []
    for nu in np.asarray(freqs_mhz):
        path = directory / f"HornWet{nu:.1f}.fits"
        if not path.is_file():
            raise FileNotFoundError(f"{path} is missing; the band needs one file per channel.")
        beam = hp.ud_grade(hp.read_map(str(path)), nside)
        maps.append(beam / beam.max())
    return np.stack(maps)


def beam_maps(case: scenario.Scenario, nside: int) -> np.ndarray:
    """The scenario's beam at ``nside``."""
    if case.beam == "gaussian":
        return gaussian_beam_maps(case.freqs_mhz(), nside)
    if case.beam == "hornwet":
        return hornwet_beam_maps(case.freqs_mhz(), nside)
    raise ValueError(f"unknown beam {case.beam!r}")


def beam_alms(maps, lmax: int) -> jax.Array:
    """True beam alms ``(n_freq, n_alm)``, from ``healpy.map2alm(iter=3)``.

    healpy rather than ``limtod_jax.map2alm_iter`` because the fit analyses a
    finer beam map at a band-limit below s2fft's ``lmax >= 2 nside - 1``
    floor; limtod_jax documents ``hp.map2alm`` as the reference for beam alms.
    """
    import healpy as hp

    return jnp.stack([jnp.asarray(hp.map2alm(m, lmax=lmax, iter=3)) for m in np.asarray(maps)])


def _operator(alm, nside: int, lmax: int, lst_deg) -> DriftScanMmode:
    return DriftScanMmode.from_pointing(
        alm,
        jnp.asarray(lst_deg),
        scenario.LAT_DEG,
        scenario.AZ_DEG,
        scenario.EL_DEG,
        lmax=lmax,
        normalize=True,
        nside=nside,
        uniform_sampling=2 * lmax < len(lst_deg),
    )


def drift_waterfall(sky_maps, alms, nside: int, lmax: int, lst_deg) -> np.ndarray:
    """Project sky maps ``(n_freq, n_pix)`` to a waterfall ``(n_time, n_freq)`` in K."""
    columns = []
    for sky, alm in zip(jnp.asarray(sky_maps), alms):
        operator = _operator(alm, nside, lmax, lst_deg)
        columns.append(operator(map2alm_quad(sky, nside=nside, lmax=lmax)))
    return np.asarray(jnp.stack(columns, axis=1))


@functools.lru_cache(maxsize=8)
def _response_rows(nside: int, lmax: int, lst: tuple[float, ...]):
    """One jitted ``beam alm -> (n_time, n_pix)`` Jacobian, reused for every channel."""

    def rows(alm):
        operator = _operator(alm, nside, lmax, np.asarray(lst))
        return jax.jacrev(lambda m: operator(map2alm_quad(m, nside=nside, lmax=lmax)))(jnp.zeros(12 * nside**2))

    return jax.jit(rows)


def drift_response(alms, nside: int, lmax: int, lst_deg) -> np.ndarray:
    """The projection as a matrix, ``(n_freq, n_time, n_pix)``.

    Row ``t`` of each channel is ``d TOD_t / d map``, built by reverse mode:
    one vector-Jacobian product per LST sample rather than one forward
    projection per pixel. The projection is linear in the sky, so the matrix
    is the same operator as :func:`drift_waterfall`. Measured by the round-3
    reviewer at NSIDE 16 / lmax 47: 3.6 s and 0.9 GB against 22.3 s and 2.1 GB
    for the unit-map build it replaces, equal to 5.7e-16 relative.
    """
    rows = _response_rows(nside, lmax, tuple(float(v) for v in lst_deg))
    return np.asarray(jnp.stack([rows(jnp.asarray(alm)) for alm in alms]))
