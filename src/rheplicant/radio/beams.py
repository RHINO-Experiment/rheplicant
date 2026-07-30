"""Measured/simulated beam patterns as HEALPix maps for the sky engines.

The sky projectors take a beam as HEALPix maps in limTOD's beam-local
convention (boresight at the pole, RING ordering). This module reads the one
format RHINO actually ships its horn in — CST Studio far-field ASCII exports,
one file per frequency — and produces exactly that.

Conventions, stated because getting one wrong returns a finite, correctly
shaped, wrong beam:

* CST's ``Theta`` is measured from the model's ``+z`` axis and maps directly
  onto the HEALPix colatitude: the boresight sits at the pole, which is what
  :class:`~rheplicant.radio.sky.driftscan.DriftScanProjector` and
  :class:`~rheplicant.radio.sky.general_pointing.GeneralPointingProjector`
  expect of a beam-local map.
* ``Abs(Dir.)`` is total directivity in dBi — a POWER quantity. The maps come
  back as ``10 ** (dBi / 10)``, which is the ``B`` of ``int(B T) / int(B)``.
  Nothing is normalized here: pass ``normalize_beam=True`` to the projector and
  let it divide by its own quadrature ``int(B)``, which is the only way the
  band-limit cancels exactly (see ``docs/sky-engines.md``).
* CST's ``Phi`` is measured from the model's ``+x`` axis; limTOD's beam-map
  ``phi = 0`` is carried to the direction of increasing elevation. Which
  physical direction the CST ``+x`` axis points is a fact about how the horn
  was built and oriented, and is NOT in the file — so it cannot be derived
  here. ``phi0_deg`` and ``phi_sense`` expose the two degrees of freedom (an
  offset and a handedness); the defaults are the identity mapping, which is an
  assumption to check against the as-built horn, not a result. For a beam with
  real azimuthal structure — RHINO's horn varies by 30-60 % around the
  ``theta = 30`` deg ring — the handedness is not a detail.

Needs ``healpy`` and ``scipy``, both already required by ``limTOD``, so the
``rheplicant[limtod]`` extra covers this module too.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from rheplicant.core.errors import StateValidationError

#: Column index of ``Abs(Dir.)[dBi]`` in a CST far-field ASCII export.
_CST_DIRECTIVITY_COLUMN = 2
#: Trailing frequency in MHz of a CST filename, e.g. ``HornDry70.5.txt``.
_FREQ_IN_NAME = re.compile(r"([0-9]+(?:\.[0-9]+)?)$")


def _require_healpy():
    try:
        import healpy
    except ImportError as exc:  # pragma: no cover - exercised by the import guard
        raise ImportError(
            "rheplicant.radio.beams needs healpy (and scipy). Both come with the "
            'sky-engine extra: pip install "rheplicant[limtod]"'
        ) from exc
    return healpy


def read_cst_farfield(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read one CST far-field export into a regular ``(theta, phi)`` grid.

    Args:
        path: a CST Studio far-field ASCII export (two header lines, then
            ``Theta Phi Abs(Dir.) ...`` rows on a regular grid).

    Returns:
        ``(theta_deg, phi_deg, directivity)`` with shapes ``(n_theta,)``,
        ``(n_phi,)`` and ``(n_theta, n_phi)``. ``directivity`` is linear power
        (``10 ** (dBi / 10)``), not dB.

    Raises:
        StateValidationError: if the rows do not fill a complete regular grid —
            an incomplete export would otherwise reshape into a plausible-looking
            beam with the samples in the wrong places.
    """
    table = np.loadtxt(path, skiprows=2)
    if table.ndim != 2 or table.shape[1] <= _CST_DIRECTIVITY_COLUMN:
        raise StateValidationError(
            f"{path}: expected a CST far-field table with at least "
            f"{_CST_DIRECTIVITY_COLUMN + 1} columns, got shape {table.shape}."
        )
    theta_deg = np.unique(table[:, 0])
    phi_deg = np.unique(table[:, 1])
    if theta_deg.size * phi_deg.size != table.shape[0]:
        raise StateValidationError(
            f"{path}: {table.shape[0]} rows do not fill the "
            f"{theta_deg.size} x {phi_deg.size} (theta, phi) grid they span; "
            "the export is incomplete or not on a regular grid."
        )
    # Rows run theta-fastest within each phi block, so (n_phi, n_theta) is the
    # natural reshape; transpose to the (theta, phi) the interpolator wants.
    directivity = 10.0 ** (
        table[:, _CST_DIRECTIVITY_COLUMN].reshape(phi_deg.size, theta_deg.size).T / 10.0
    )
    return theta_deg, phi_deg, directivity


def cst_frequency_table(directory: str | Path, *, suffix: str = ".txt") -> dict[float, Path]:
    """Map frequency [Hz] to file for a directory of CST exports.

    The frequency is read from the trailing number of the stem, in MHz —
    ``HornDry70.5.txt`` is 70.5 MHz. Files whose stem does not end in a number
    are ignored.

    Args:
        directory: a directory of per-frequency CST exports.
        suffix: file extension to consider.

    Raises:
        StateValidationError: if the directory holds no matching file.
    """
    directory = Path(directory).expanduser()
    table: dict[float, Path] = {}
    for path in sorted(directory.glob(f"*{suffix}")):
        match = _FREQ_IN_NAME.search(path.stem)
        if match is not None:
            table[float(match.group(1)) * 1e6] = path
    if not table:
        raise StateValidationError(
            f"No CST exports found in {directory} (looked for '*{suffix}' whose "
            "stem ends in a frequency in MHz, e.g. 'HornDry70.5.txt')."
        )
    return table


def cst_beam_maps(
    directory: str | Path,
    freq_hz,
    *,
    nside: int,
    suffix: str = ".txt",
    phi0_deg: float = 0.0,
    phi_sense: str = "ccw",
) -> np.ndarray:
    """Sample a directory of CST exports onto HEALPix maps at given frequencies.

    Linearly interpolates the linear-power beam between the two bracketing CST
    files. Extrapolation is refused: a beam invented outside the simulated band
    is not a beam.

    Args:
        directory: directory of per-frequency CST exports (see
            :func:`cst_frequency_table`).
        freq_hz: ``(n_freq,)`` output frequencies [Hz].
        nside: HEALPix resolution of the output maps (RING ordering).
        suffix: file extension of the exports.
        phi0_deg: CST azimuth that lands on the beam-map ``phi = 0`` meridian.
        phi_sense: ``"ccw"`` if CST azimuth increases with beam-map ``phi``,
            ``"cw"`` if it decreases. See the module docstring — this is a fact
            about the horn, not about the file.

    Returns:
        ``(n_freq, 12 * nside ** 2)`` linear-power beam maps, unnormalized.

    Raises:
        StateValidationError: on an unknown ``phi_sense``, or a requested
            frequency outside the range the directory covers.
    """
    hp = _require_healpy()
    from scipy.interpolate import RegularGridInterpolator

    if phi_sense not in ("ccw", "cw"):
        raise StateValidationError(
            f"phi_sense must be 'ccw' or 'cw', got {phi_sense!r}."
        )
    freq_hz = np.atleast_1d(np.asarray(freq_hz, dtype=float))
    table = cst_frequency_table(directory, suffix=suffix)
    available = np.array(sorted(table))
    if freq_hz.min() < available[0] or freq_hz.max() > available[-1]:
        raise StateValidationError(
            f"Requested {freq_hz.min() / 1e6:.3f}-{freq_hz.max() / 1e6:.3f} MHz but "
            f"{Path(directory).expanduser()} covers only "
            f"{available[0] / 1e6:.3f}-{available[-1] / 1e6:.3f} MHz. Extrapolating "
            "a beam outside its simulated band would return a plausible, "
            "unsupported answer."
        )

    # Only the files actually bracketing a requested frequency are read: the
    # RHINO directories hold 61 of them, each a 65k-row parse.
    needed = sorted({
        available[index]
        for f in freq_hz
        for index in (
            max(int(np.searchsorted(available, f, side="right")) - 1, 0),
            min(int(np.searchsorted(available, f, side="left")), available.size - 1),
        )
    })
    theta_hp, phi_hp = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    sign = 1.0 if phi_sense == "ccw" else -1.0
    phi_cst = np.rad2deg(sign * phi_hp) + phi0_deg

    sampled = {}
    for f in needed:
        theta_deg, phi_deg, directivity = read_cst_farfield(table[f])
        # Close the azimuth circle so the interpolator can wrap instead of
        # clamping the last degree onto a boundary value.
        phi_closed = np.append(phi_deg, phi_deg[0] + 360.0)
        grid = np.concatenate([directivity, directivity[:, :1]], axis=1)
        interp = RegularGridInterpolator(
            (theta_deg, phi_closed), grid, method="linear", bounds_error=False,
            fill_value=None,
        )
        sampled[f] = interp(
            np.stack([np.rad2deg(theta_hp),
                      np.mod(phi_cst - phi_deg[0], 360.0) + phi_deg[0]], axis=-1)
        )

    grid = np.asarray(needed)
    stack = np.stack([sampled[f] for f in needed])
    return np.stack([_interp_frequency(f, grid, stack) for f in freq_hz])


def _interp_frequency(f: float, grid: np.ndarray, maps: np.ndarray) -> np.ndarray:
    """Linear interpolation of ``maps`` (n_grid, npix) at frequency ``f``."""
    if grid.size == 1:
        return maps[0]
    upper = int(np.clip(np.searchsorted(grid, f, side="left"), 1, grid.size - 1))
    lower = upper - 1
    span = grid[upper] - grid[lower]
    weight = 0.0 if span == 0.0 else (f - grid[lower]) / span
    return (1.0 - weight) * maps[lower] + weight * maps[upper]
