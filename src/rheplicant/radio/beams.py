"""Measured/simulated beam patterns as HEALPix maps for the sky engines.

**Thin adapters, deliberately.** How a measured beam becomes a beam map, where
the horizon falls in it and what share of its solid angle survives are limTOD's
subject (D20, D25) — exactly as the noise-wave data model is
``rhino_cal_jax``'s. This package's job is to *place* the result on a signal
path. Everything here is a pass-through whose only added value is at the seam:
frequencies in Hz, because that is what ``Coordinates.freq`` carries, and
``nside`` inferred from maps that already know it.

The physics and its conventions live upstream:

* :mod:`limTOD.cstbeam` — CST Studio far-field exports onto HEALPix maps
  (``read_cst_farfield``, ``cst_frequency_table``, ``cst_beam_maps``), plus a
  ``cst_beam_func`` for limTOD's own simulator. **Read that module's
  conventions before trusting a beam**, in particular that the CST azimuth's
  offset and handedness are facts about the as-built horn which the export does
  not contain: ``phi0_deg`` and ``phi_sense`` are assumptions to check, not
  results, and for RHINO's horn — 30-60 % azimuthal structure around the
  ``theta = 30`` deg ring — the handedness is not a detail.
* :func:`limtod_jax.horizon_truncated_beam` — the horizon cut and the surviving
  sky fraction.

Needs ``healpy`` and ``scipy``, both already required by ``limTOD``, so the
``rheplicant[limtod]`` extra covers this module.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from rheplicant.core.errors import StateValidationError

#: Frequencies cross this seam in Hz, because that is what ``Coordinates.freq``
#: carries; limTOD's beam APIs are in MHz, as they are throughout that package.
_HZ_PER_MHZ = 1e6


def _require_limtod_jax(feature: str):
    """The limtod_jax module, gated on the symbol this adapter actually calls.

    Floored at the release that owns the horizon physics, so an outdated
    install says so at the boundary instead of raising AttributeError midway.
    """
    try:
        import limtod_jax
    except ImportError as exc:  # pragma: no cover - exercised by the import guard
        raise ImportError(
            f"rheplicant.radio.beams.{feature} needs limTOD's JAX package: "
            'pip install "rheplicant[limtod]"'
        ) from exc
    if not hasattr(limtod_jax, feature):
        raise ImportError(
            f"rheplicant.radio.beams.{feature} needs limTOD >= 1.9, which is "
            f"where {feature}() lives; the installed limtod_jax "
            f"({getattr(limtod_jax, '__version__', 'unknown')}) does not have it."
        )
    return limtod_jax


def _require_cstbeam():
    """``limTOD.cstbeam``, gated on the module rather than on a version.

    The CST reader moved upstream after limTOD 1.9, so a version floor cannot
    yet name it; the import is the check, and it says what is missing rather
    than failing on an AttributeError three calls deeper.
    """
    try:
        from limTOD import cstbeam
    except ImportError as exc:  # pragma: no cover - exercised by the import guard
        raise ImportError(
            "rheplicant.radio.beams needs limTOD.cstbeam, which is where the CST "
            "far-field reader lives. The installed limTOD does not have it "
            "(it arrived after 1.9). Install limTOD from source, or "
            'pip install "rheplicant[limtod]" if you have none at all.'
        ) from exc
    return cstbeam


def read_cst_farfield(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read one CST far-field export — a pass-through to limTOD.

    See :func:`limTOD.cstbeam.read_cst_farfield`, which owns the format and the
    conventions.

    Returns:
        ``(theta_deg, phi_deg, directivity)``; ``directivity`` is linear power
        (``10 ** (dBi / 10)``), not dB.
    """
    return _require_cstbeam().read_cst_farfield(path)


def cst_frequency_table(directory: str | Path, *, suffix: str = ".txt") -> dict[float, Path]:
    """Map frequency **[Hz]** to file for a directory of CST exports.

    A pass-through to :func:`limTOD.cstbeam.cst_frequency_table`, whose keys are
    in MHz; the conversion is this seam's whole contribution.
    """
    table = _require_cstbeam().cst_frequency_table(directory, suffix=suffix)
    return {mhz * _HZ_PER_MHZ: path for mhz, path in table.items()}


def cst_beam_maps(
    directory: str | Path,
    freq_hz,
    *,
    nside: int,
    suffix: str = ".txt",
    phi0_deg: float = 0.0,
    phi_sense: str = "ccw",
) -> np.ndarray:
    """Sample a directory of CST exports onto HEALPix maps — pass-through.

    :func:`limTOD.cstbeam.cst_beam_maps` does the work and documents the
    conventions; this takes ``freq_hz`` in Hz to match ``Coordinates.freq``.

    Args:
        directory: directory of per-frequency CST exports.
        freq_hz: ``(n_freq,)`` output frequencies [Hz].
        nside: HEALPix resolution of the output maps (RING ordering).
        suffix: file extension of the exports.
        phi0_deg: CST azimuth landing on the beam-map ``phi = 0`` meridian.
        phi_sense: ``"ccw"`` or ``"cw"``. A fact about the horn, not the file —
            see :mod:`limTOD.cstbeam`.

    Returns:
        ``(n_freq, 12 * nside ** 2)`` linear-power beam maps, unnormalized.
        Pass ``normalize_beam=True`` to the projector and let it divide by its
        own quadrature, which is the only way the band limit cancels exactly
        (see ``docs/sky-engines.md``).
    """
    freq_mhz = np.atleast_1d(np.asarray(freq_hz, dtype=float)) / _HZ_PER_MHZ
    return _require_cstbeam().cst_beam_maps(
        directory, freq_mhz, nside=nside, suffix=suffix,
        phi0_deg=phi0_deg, phi_sense=phi_sense,
    )


def horizon_truncated_beam(beam_maps, *, el_deg: float = 90.0, apod_deg: float = 0.0):
    """Cut beam maps at the horizon — a thin pass-through to limTOD.

    The physics, the conventions and their numerical locks live in
    :func:`limtod_jax.horizon_truncated_beam` (limTOD >= 1.9); this exists only
    so that ``nside`` need not be repeated when the maps already carry it.

    Args:
        beam_maps: ``(n_freq, npix)`` or ``(npix,)`` HEALPix RING beam maps in
            the beam-local frame.
        el_deg: boresight elevation [deg]; only 90 is supported — see limTOD.
        apod_deg: cosine-apodization width of the cut [deg of elevation].

    Returns:
        ``(truncated_maps, sky_fraction)``, shapes ``(n_freq, npix)`` and
        ``(n_freq,)``. Hand the fraction straight to
        :class:`~rheplicant.radio.instrument.beam_spill.BeamSpillOperator`.

    Raises:
        StateValidationError: if the maps are not a valid HEALPix length.
    """
    ltj = _require_limtod_jax("horizon_truncated_beam")
    maps = np.atleast_2d(np.asarray(beam_maps, dtype=float))
    n_pix = maps.shape[-1]
    nside = int(round(math.sqrt(n_pix / 12.0)))
    if 12 * nside**2 != n_pix:
        raise StateValidationError(
            f"beam_maps has {n_pix} pixels, which is not a valid HEALPix map "
            "length (12*nside**2)."
        )
    truncated, fraction = ltj.horizon_truncated_beam(
        maps, nside=nside, el_deg=el_deg, apod_deg=apod_deg
    )
    return np.asarray(truncated), np.asarray(fraction)
