"""The MERS synchrotron law in jax, and the sky maps built from it.

MERS (``MERS/fg_model.py``, class ``SynchrotronExtrapolator``) extrapolates
the Haslam 408 MHz map with a per-pixel spectral index and one global
curvature,

    T(p, nu) = (T408(p) - 8.9) (nu / 408)^{beta(p) + c ln(nu / nu_c)},

with ``c = -0.10`` and ``nu_c = 11704 MHz`` hard-coded (``fg_model.py``
lines 63-77 and 120-142). MERS is numpy and healpy and is not installable,
so this module ports the few lines needed rather than importing it;
``tests/test_ports.py`` checks the port against MERS itself
when MERS is on disk.

Demo B fits the same law with its amplitude referred to ``nu0`` instead of
408 MHz, which keeps the amplitude prior on the scale of the data:

    T(p, nu) = A(p) exp[ beta(p) ln(nu / nu0) + c (Q(nu) - Q(nu0)) ],
    Q(nu)    = ln(nu / 408) ln(nu / nu_c),
    A(p)     = (T408(p) - 8.9) (nu0 / 408)^{beta(p) + c ln(nu0 / nu_c)}.

The two forms are the same function of frequency, which the tests check.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from global21cm import scenario


def mers_synchrotron(t408, beta, freqs_mhz, curvature=scenario.CURVATURE_TRUE):
    """MERS's extrapolation, ``(n_pix,) -> (n_pix, n_freq)`` in K."""
    t408 = jnp.asarray(t408)
    beta = jnp.asarray(beta)
    nu = jnp.asarray(freqs_mhz)
    index = beta[:, None] + curvature * jnp.log(nu / scenario.NU_CURVATURE_MHZ)[None, :]
    pure = (t408 - scenario.MONOPOLE_OFFSET_K)[:, None]
    return pure * (nu / scenario.NU_HASLAM_MHZ)[None, :] ** index


def _q(nu):
    return jnp.log(nu / scenario.NU_HASLAM_MHZ) * jnp.log(nu / scenario.NU_CURVATURE_MHZ)


def spectral_design(freqs_mhz, nu0_mhz=scenario.NU0_MHZ):
    """``(ln(nu/nu0), Q(nu) - Q(nu0))``, the two frequency vectors of the law."""
    nu = jnp.asarray(freqs_mhz)
    return jnp.log(nu / nu0_mhz), _q(nu) - _q(jnp.asarray(nu0_mhz))


def sky_from_amplitude(amplitude, beta, curvature, freqs_mhz, nu0_mhz=scenario.NU0_MHZ):
    """Demo B's sky, ``(n_pix,) -> (n_freq, n_pix)`` in K; linear in ``amplitude``."""
    log_ratio, q_shift = spectral_design(freqs_mhz, nu0_mhz)
    exponent = beta[None, :] * log_ratio[:, None] + curvature * q_shift[:, None]
    return amplitude[None, :] * jnp.exp(exponent)


def amplitude_at_nu0(t408, beta, curvature=scenario.CURVATURE_TRUE, nu0_mhz=scenario.NU0_MHZ):
    """``A(p)`` of the module docstring: the MERS map evaluated at ``nu0``."""
    return mers_synchrotron(t408, beta, np.array([nu0_mhz]), curvature)[:, 0]


def load_mers_maps(nside_out: int):
    """Haslam and the CNN index, rotated to equatorial and degraded to ``nside_out``.

    MERS's maps are Galactic; the drift scan's sky is equatorial (RA/Dec).
    The rotation is done by pixel interpolation at the native NSIDE 512,
    then both maps are averaged down with ``healpy.ud_grade``, which is what
    MERS's own ``change_nside`` does (``fg_model.py:30``).
    """
    import healpy as hp

    data = scenario.mers_data_dir()
    t408 = hp.read_map(str(data / scenario.HASLAM_FILE))
    beta = np.load(data / scenario.BETA_FILE)
    rotator = hp.Rotator(coord=["G", "C"])
    t408_eq = rotator.rotate_map_pixel(t408)
    beta_eq = rotator.rotate_map_pixel(beta)
    return hp.ud_grade(t408_eq, nside_out), hp.ud_grade(beta_eq, nside_out)
