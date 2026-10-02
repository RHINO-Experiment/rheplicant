"""The jax ports against MERS itself.

MERS is not installable; point ``$MERS_DIR`` at its checkout or at the
package directory inside it that holds ``fg_model.py`` and ``SEDfitting.py``.
Without it these tests skip, and say so.
"""

import os
import sys

import numpy as np
import pytest

from global21cm import foreground_physical as phys
from global21cm import scenario
from global21cm.foreground_beamconv import moment_basis, orthonormal_span

MERS_DIR = os.environ.get("MERS_DIR", "")
if MERS_DIR and not os.path.isfile(os.path.join(MERS_DIR, "fg_model.py")):
    MERS_DIR = os.path.join(MERS_DIR, "MERS")  # the checkout root rather than the package
if not os.path.isfile(os.path.join(MERS_DIR, "fg_model.py")):
    pytest.skip("$MERS_DIR names neither MERS's checkout nor its package directory",
                allow_module_level=True)
sys.path.insert(0, MERS_DIR)
fg_model = pytest.importorskip("fg_model")
SEDfitting = pytest.importorskip("SEDfitting")

FREQS = scenario.MAIN.freqs_mhz()
NSIDE = 4


def _maps(seed=0):
    rng = np.random.default_rng(seed)
    npix = 12 * NSIDE**2
    t408 = 15.0 + 200.0 * rng.random(npix)
    beta = -3.1 + 0.4 * rng.random(npix)
    return t408, beta


class TestSynchrotronPort:
    def test_matches_synchrotron_extrapolator(self):
        t408, beta = _maps()
        mers = fg_model.SynchrotronExtrapolator(
            reference_map=t408, spectral_index_map=beta, nside=NSIDE
        ).map(FREQS)
        ours = np.asarray(phys.mers_synchrotron(t408, beta, FREQS))
        np.testing.assert_allclose(ours, mers, rtol=1e-12)

    @pytest.mark.parametrize("freq", [FREQS[0], 70.0, FREQS[-1], 408.0, 1000.0])
    def test_single_frequencies_including_the_reference(self, freq):
        t408, beta = _maps(1)
        mers = fg_model.SynchrotronExtrapolator(
            reference_map=t408, spectral_index_map=beta, nside=NSIDE
        ).map(freq)
        ours = np.asarray(phys.mers_synchrotron(t408, beta, np.array([freq])))[:, 0]
        np.testing.assert_allclose(ours, mers, rtol=1e-12)

    def test_the_nu0_form_is_the_same_function(self):
        t408, beta = _maps(2)
        amplitude = phys.amplitude_at_nu0(t408, beta)
        ours = np.asarray(phys.sky_from_amplitude(amplitude, beta, scenario.CURVATURE_TRUE, FREQS))
        mers = fg_model.SynchrotronExtrapolator(
            reference_map=t408, spectral_index_map=beta, nside=NSIDE
        ).map(FREQS)
        np.testing.assert_allclose(ours.T, mers, rtol=1e-12)


class TestMomentBasis:
    @pytest.mark.parametrize("n_moments", [1, 3, 7])
    @pytest.mark.parametrize("beta0", [-2.55, -3.2, 0.0])
    def test_matches_fg_moment_basis(self, n_moments, beta0):
        mers = SEDfitting.fg_moment_basis(FREQS, nu_ref=70.0).basis(beta0, n_moments - 1)
        np.testing.assert_allclose(moment_basis(FREQS, beta0, n_moments, 70.0), mers, rtol=1e-12)

    @pytest.mark.parametrize("n_beam", [1, 4])
    def test_matches_the_bcsvd_tensor_product(self, n_beam):
        rng = np.random.default_rng(3)
        spectra = np.linalg.qr(rng.normal(size=(FREQS.size, n_beam)))[0]
        mers = SEDfitting.fg_moment_basis(FREQS, nu_ref=70.0).basis(-2.55, 4, BCSVD=spectra)
        ours = moment_basis(FREQS, -2.55, 5, 70.0, spectra)
        np.testing.assert_allclose(ours, mers, rtol=1e-12)

    def test_orthonormalising_keeps_the_span(self):
        basis = moment_basis(FREQS, -2.55, 6, 70.0)
        q = orthonormal_span(basis)
        np.testing.assert_allclose(q.T @ q, np.eye(6), atol=1e-12)
        projected = q @ (q.T @ basis)
        np.testing.assert_allclose(projected, basis, atol=1e-10)
