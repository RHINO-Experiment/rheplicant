"""DriftScanProjector refuses an (nside, lmax) its transforms cannot run.

s2fft's HEALPix transforms need ``nside >= 2`` and ``lmax + 1 >= 2 * nside``.
Below either edge the projector used to construct and then crash on the first
call, inside s2fft: ``ValueError: Need at least one array to stack`` at
nside 1, a ``TypeError`` from the forward FFT and a bare ``AssertionError``
from the adjoint for ``lmax < 2 * nside - 1``. Measured on this package's
forward, adjoint, mmodes, normalised-beam, horizon-mask and ``from_beam_maps``
paths for nside 1-4 and 8 (forward and adjoint at 16 and 32): every path
failed below the edge and every path ran at and above it, with no upper edge
up to ``lmax = 64`` at nside 2. The config layer's ``beam_analysis`` transform already refuses the
same domain (``tests/config/test_config_transforms.py``).
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rheplicant import Coordinates
from rheplicant.core.errors import StateValidationError
from rheplicant.radio.sky import DriftScanProjector

pytest.importorskip("limtod_jax")

N_TIME = 9
LAT, AZ, EL, SELFROT = 53.2, 30.0, 50.0, 10.0


def _alms(lmax):
    n_alm = (lmax + 1) * (lmax + 2) // 2
    re, im = jax.random.normal(jax.random.key(0), (2, 1, n_alm))
    return re + 1j * im.at[:, : lmax + 1].set(0.0)


def _projector(nside, lmax, **kwargs):
    return DriftScanProjector(
        beam_alms=_alms(lmax), lat_deg=LAT, az_deg=AZ, el_deg=EL,
        lmax=lmax, nside=nside, selfrot_deg=SELFROT, **kwargs,
    )


def _coords():
    return Coordinates(
        time=jnp.arange(float(N_TIME)), freq=jnp.asarray([60e6]),
        extra={"lst_deg": jnp.linspace(0.0, 40.0, N_TIME)},
    )


#: Below the edge: nside 1 at any lmax, and ``lmax = 2 * nside - 2`` (one
#: under) or far under at nside 2..64.
REFUSED = [(1, 0), (1, 1), (1, 5), (1, 40), (2, 0), (2, 2), (3, 4), (4, 0),
           (4, 6), (8, 14), (16, 30), (32, 62), (64, 0), (64, 126)]
#: At the edge, ``lmax = 2 * nside - 1``, and far above it.
ACCEPTED = [(2, 3), (3, 5), (4, 7), (8, 15), (16, 31), (32, 63), (2, 64)]


@pytest.mark.parametrize(("nside", "lmax"), REFUSED)
def test_the_crash_domain_is_refused_at_construction(nside, lmax):
    with pytest.raises(StateValidationError) as excinfo:
        _projector(nside, lmax)
    message = str(excinfo.value)
    assert "nside >= 2" in message and "lmax >= 2 * nside - 1" in message
    assert f"nside={nside}" in message and f"lmax={lmax}" in message


@pytest.mark.parametrize(("nside", "lmax"), ACCEPTED)
def test_the_valid_side_constructs_and_both_s2fft_transforms_run(nside, lmax):
    """The other side of the edge, through the two s2fft calls that crashed:
    the analysis ``sky_to_alms`` and the synthesis the adjoint ends in."""
    import limtod_jax as ltj

    sky = jax.random.uniform(jax.random.key(1), (1, 12 * nside**2))
    alms = _projector(nside, lmax).sky_to_alms(sky)
    synthesised = ltj.alm2map(alms[0], nside=nside, lmax=lmax)
    assert np.isfinite(np.asarray(alms)).all()
    assert np.isfinite(np.asarray(synthesised)).all()


@pytest.mark.parametrize(("nside", "lmax"), [(2, 3), (4, 7)])
def test_at_the_edge_the_projector_itself_runs(nside, lmax):
    """The whole forward and adjoint at the edge, not only the transforms --
    at the two smallest nside, because the Wigner rotation dominates the cost
    and is not what fails."""
    coords = _coords()
    sky = jax.random.uniform(jax.random.key(1), (1, 12 * nside**2))
    tod = jax.random.normal(jax.random.key(2), (N_TIME, 1))
    projector = _projector(nside, lmax)
    assert np.isfinite(np.asarray(projector.forward(sky, coords))).all()
    assert np.isfinite(np.asarray(projector.adjoint(tod, coords))).all()


@pytest.mark.parametrize(("nside", "lmax"), [(1, 5), (4, 6)])
def test_from_beam_maps_refuses_before_the_transform(nside, lmax):
    """``from_beam_maps`` runs ``map2alm_iter`` before it constructs, so the
    refusal has to come first there too, or s2fft's error arrives instead."""
    maps = jnp.ones((1, 12 * nside**2))
    with pytest.raises(StateValidationError, match="lmax >= 2 \\* nside - 1"):
        DriftScanProjector.from_beam_maps(maps, lat_deg=LAT, az_deg=AZ, el_deg=EL, lmax=lmax)


def test_from_beam_maps_accepts_the_edge():
    maps = jnp.ones((1, 12 * 4**2))
    built = DriftScanProjector.from_beam_maps(maps, lat_deg=LAT, az_deg=AZ, el_deg=EL, lmax=7)
    assert built.lmax == 7 and built.nside == 4


def test_a_field_edit_into_the_crash_domain_is_refused():
    """``dataclasses.replace`` re-runs the check, as every functional update does."""
    with pytest.raises(StateValidationError, match="nside >= 2"):
        dataclasses.replace(_projector(4, 11), nside=8)
