"""The plugin's forward models against the simulator, where the two coincide.

Demo B coincides with the simulator when the fit resolution equals the
truth resolution and the spectral factors are the true ones: then the
operator is the same limtod_jax projection of the same MERS sky. Demo A
coincides when the beam is achromatic and the sky is one power law, because
every LST's spectrum is then exactly the zeroth moment.

Small grids (NSIDE 4, lmax 11, 24 LSTs, 5 channels) keep these fast.
"""

import healpy as hp
import jax.numpy as jnp
import numpy as np
import pytest
from rheplicant import Coordinates, State

from global21cm import foreground_physical as phys
from global21cm import instrument, scenario, signal21
from global21cm.foreground_beamconv import MomentForeground, moment_basis
from global21cm.physical_operator import PhysicalSkyForeground, moment_jacobian

NSIDE, LMAX, N_TIME = 4, 11, 24
FREQS = np.array([55.0, 62.0, 70.0, 78.0, 85.0])
LST = 360.0 * np.arange(N_TIME) / N_TIME


def _state():
    coords = Coordinates(time=jnp.arange(N_TIME) * 3600.0, freq=jnp.asarray(FREQS) * 1e6)
    return State(coords=coords)


def _truth_maps(seed=0):
    rng = np.random.default_rng(seed)
    npix = 12 * NSIDE**2
    return 20.0 + 300.0 * rng.random(npix), -2.9 + 0.3 * rng.random(npix)


def _beam_alms(freqs):
    return instrument.beam_alms(instrument.gaussian_beam_maps(freqs, NSIDE), LMAX)


class TestDemoB:
    def test_operator_reproduces_the_simulator_at_the_truth(self):
        t408, beta = _truth_maps()
        alms = _beam_alms(FREQS)
        sky = np.asarray(phys.mers_synchrotron(t408, beta, FREQS)).T
        simulated = instrument.drift_waterfall(sky, alms, NSIDE, LMAX, LST)

        amplitude = np.asarray(phys.amplitude_at_nu0(t408, beta))
        log_ratio, q_shift = (np.asarray(v) for v in phys.spectral_design(FREQS))
        spectra = np.exp(beta[None, :] * log_ratio[:, None] + scenario.CURVATURE_TRUE * q_shift[:, None])
        npix = amplitude.size
        operator = PhysicalSkyForeground(
            coords=jnp.zeros(1),
            offset=jnp.concatenate([jnp.asarray(amplitude), jnp.zeros(npix)]),
            design=jnp.zeros((2 * npix, 1)),
            spectra=jnp.asarray(spectra),
            response=jnp.asarray(instrument.drift_response(alms, NSIDE, LMAX, LST)),
        )
        predicted = np.asarray(operator(_state()).data)
        np.testing.assert_allclose(predicted, simulated, rtol=1e-9)
        assert np.ptp(simulated) > 1.0  # the comparison is not between two constants

    def test_the_b_map_moves_the_spectral_index(self):
        t408, beta = _truth_maps(3)
        shift = 1e-3
        alms = _beam_alms(FREQS)
        sky = np.asarray(phys.mers_synchrotron(t408, beta + shift, FREQS)).T
        simulated = instrument.drift_waterfall(sky, alms, NSIDE, LMAX, LST)
        amplitude = np.asarray(phys.amplitude_at_nu0(t408, beta + shift))
        log_ratio, q_shift = (np.asarray(v) for v in phys.spectral_design(FREQS))
        spectra = np.exp(beta[None, :] * log_ratio[:, None] + scenario.CURVATURE_TRUE * q_shift[:, None])
        operator = PhysicalSkyForeground(
            coords=jnp.zeros(1),
            offset=jnp.concatenate([jnp.asarray(amplitude), jnp.asarray(shift * amplitude)]),
            design=jnp.zeros((2 * amplitude.size, 1)),
            spectra=jnp.asarray(spectra),
            response=jnp.asarray(instrument.drift_response(alms, NSIDE, LMAX, LST)),
        )
        predicted = np.asarray(operator(_state()).data)
        # Second order: (1e-3 * 0.24)^2 / 2 = 3e-8 relative at the band edge.
        np.testing.assert_allclose(predicted, simulated, rtol=5e-8)
        without_b = operator.maps()[0]
        assert not np.allclose(np.asarray(without_b), np.asarray(operator.offset[amplitude.size :]))

    def test_moment_jacobian_is_the_operator(self):
        rng = np.random.default_rng(6)
        npix = 12 * NSIDE**2
        response = rng.random((FREQS.size, N_TIME, npix))
        spectra = 1.0 + rng.random((FREQS.size, npix))
        maps = rng.normal(size=2 * npix)
        operator = PhysicalSkyForeground(
            coords=jnp.zeros(1), offset=jnp.asarray(maps), design=jnp.zeros((2 * npix, 1)),
            spectra=jnp.asarray(spectra), response=jnp.asarray(response),
        )  # fmt: skip
        jac = moment_jacobian(response, spectra, np.log(FREQS / scenario.NU0_MHZ))
        waterfall = np.asarray(operator(_state()).data)
        np.testing.assert_allclose((jac @ maps).reshape(FREQS.size, N_TIME).T, waterfall, rtol=1e-10)

    def test_first_moment_is_the_linearised_index(self):
        t408, beta = _truth_maps(1)
        amplitude = np.asarray(phys.amplitude_at_nu0(t408, beta))
        shift = 1e-4
        exact = np.asarray(
            phys.sky_from_amplitude(amplitude, beta + shift, scenario.CURVATURE_TRUE, FREQS)
        )
        base = np.asarray(phys.sky_from_amplitude(amplitude, beta, scenario.CURVATURE_TRUE, FREQS))
        linear = base * (1.0 + shift * np.log(FREQS / scenario.NU0_MHZ))[:, None]
        # Second order in shift * ln(nu / nu0): below 1e-9 relative here.
        np.testing.assert_allclose(linear, exact, rtol=1e-9)


class TestDemoA:
    def test_zeroth_moment_reproduces_an_achromatic_power_law_sky(self):
        t408, _ = _truth_maps(2)
        alms = _beam_alms(np.full(FREQS.size, 70.0))  # the same beam at every channel
        power = (FREQS / 70.0) ** scenario.BETA0_MOMENT
        sky = t408[None, :] * power[:, None]
        simulated = instrument.drift_waterfall(sky, alms, NSIDE, LMAX, LST)

        basis = moment_basis(FREQS, scenario.BETA0_MOMENT, 1, 70.0)
        coefficients, *_ = np.linalg.lstsq(basis, simulated.T, rcond=None)
        operator = MomentForeground(coefficients=jnp.asarray(coefficients.T), basis=jnp.asarray(basis))
        predicted = np.asarray(operator(_state()).data)
        np.testing.assert_allclose(predicted, simulated, rtol=1e-10)


class TestSignalAndOracle:
    def test_signal_operator_matches_the_injected_curve(self):
        from global21cm.simulate import signal_21cm

        theta = jnp.asarray(signal21.theta_log_true())
        predicted = np.asarray(signal21.Emulated21cmSignal(theta=theta)(_state()).data)
        expected = signal_21cm(FREQS)
        np.testing.assert_allclose(predicted, np.broadcast_to(expected, predicted.shape), rtol=1e-6)
        assert expected.min() < -0.05

    def test_unit_normal_latents_map_onto_the_box(self):
        low, high = (np.asarray(v) for v in signal21.prior_box())
        np.testing.assert_allclose(signal21.stack_theta(*signal21.u_true()),
                                   signal21.theta_log_true(), rtol=1e-12)  # fmt: skip
        far = np.asarray(signal21.stack_theta(*([40.0] * 7)))
        assert np.all(far <= high) and np.all(np.asarray(signal21.stack_theta(*([-40.0] * 7))) >= low)

    def test_compressed_signal_is_the_design_times_the_curve(self):
        theta = jnp.asarray(signal21.theta_log_true())
        design = np.random.default_rng(5).normal(size=(FREQS.size, FREQS.size))
        state = State(coords=Coordinates(time=jnp.zeros(1), freq=jnp.asarray(FREQS) * 1e6))
        out = np.asarray(signal21.CompressedSignal(theta=theta, design=jnp.asarray(design))(state).data)
        expected = design @ np.asarray(signal21.curve_kelvin(theta, FREQS))
        np.testing.assert_allclose(out, expected[None, :], rtol=1e-10)

    def test_compressed_signal_refuses_a_waterfall_grid(self):
        from rheplicant.core.errors import StateValidationError

        operator = signal21.CompressedSignal(theta=jnp.zeros(7), design=jnp.eye(FREQS.size))
        with pytest.raises(StateValidationError):
            operator(_state())


def test_rotation_keeps_the_monopole():
    """The Galactic-to-equatorial rotation used on the MERS maps preserves the mean."""
    rng = np.random.default_rng(4)
    m = 10.0 + rng.random(12 * 16**2)
    rotated = hp.Rotator(coord=["G", "C"]).rotate_map_pixel(m)
    assert abs(rotated.mean() - m.mean()) < 1e-2 * m.std()


def test_reverse_mode_response_equals_unit_map_projections():
    alms = _beam_alms(FREQS[:2])
    response = instrument.drift_response(alms, NSIDE, LMAX, LST)
    for pixel in (0, 57, 12 * NSIDE**2 - 1):
        unit = np.zeros((2, 12 * NSIDE**2))
        unit[:, pixel] = 1.0
        column = instrument.drift_waterfall(unit, alms, NSIDE, LMAX, LST)  # (n_time, 2)
        np.testing.assert_allclose(response[:, :, pixel], column.T, rtol=1e-10, atol=1e-14)
