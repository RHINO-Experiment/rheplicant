"""Contract tests for the taxonomy component operators (elements coverage)."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from rheplicant import AbstractOperator, Environment, Pipeline, State, SumOperator
from rheplicant.core.errors import StateValidationError
from rheplicant.radio import (
    CWCalibrationOperator,
    EMIOperator,
    FlaggingOperator,
    ForegroundOperator,
    GlobalSignalOperator,
    GroundPickupOperator,
    IonosphereOperator,
    NoiseWaveOperator,
    PointSourceOperator,
    RFIOperator,
)

# Mirrors tests/conftest.py fixture dimensions.
N_TIME = 8
N_FREQ = 4


@pytest.fixture
def data_state(template_state):
    return template_state.with_data(jnp.full((N_TIME, N_FREQ), 10.0))


NEW_OPERATOR_CLASSES = [
    GlobalSignalOperator,
    ForegroundOperator,
    PointSourceOperator,
    IonosphereOperator,
    GroundPickupOperator,
    RFIOperator,
    NoiseWaveOperator,
    CWCalibrationOperator,
    EMIOperator,
    FlaggingOperator,
]


@pytest.mark.parametrize("cls", NEW_OPERATOR_CLASSES)
def test_contract_declared(cls):
    assert issubclass(cls, AbstractOperator)
    assert isinstance(cls.requires, tuple) and cls.requires
    assert isinstance(cls.provides, tuple) and cls.provides


class TestSkyComponents:
    def test_global_signal_trough(self, template_state):
        op = GlobalSignalOperator(
            depth=jnp.array(0.2), centre=jnp.array(72e6), width=jnp.array(5e6)
        )
        out = op(template_state)
        assert out.data.shape == (N_TIME, N_FREQ)
        assert jnp.all(out.data <= 0)  # absorption
        assert jnp.all(out.data >= -0.2)
        # constant in time
        assert jnp.array_equal(out.data[0], out.data[-1])

    def test_foreground_power_law(self, template_state):
        op = ForegroundOperator(
            amplitude=jnp.array(1e3), spectral_index=jnp.array(2.5), ref_freq=70e6
        )
        out = op(template_state)
        freq = template_state.coords.freq
        expected = 1e3 * (freq / 70e6) ** (-2.5)
        assert jnp.allclose(out.data[3], expected)
        # decreasing with frequency for positive index
        assert jnp.all(jnp.diff(out.data[0]) < 0)

    def test_foreground_rejects_bad_ref_freq(self):
        with pytest.raises(StateValidationError, match="ref_freq"):
            ForegroundOperator(
                amplitude=jnp.array(1.0), spectral_index=jnp.array(2.5), ref_freq=0.0
            )

    def test_point_sources_level(self, template_state):
        out = PointSourceOperator(level=jnp.array(3.0))(template_state)
        assert jnp.all(out.data == 3.0)

    def test_requires_coords(self):
        with pytest.raises(StateValidationError, match="coords"):
            PointSourceOperator(level=jnp.array(1.0))(State())


class TestEnvironmentComponents:
    def test_ionosphere_chromatic_multiplier(self, data_state):
        op = IonosphereOperator(delta=jnp.array(0.01), ref_freq=70e6)
        out = op(data_state)
        factor = 1.0 + 0.01 * (data_state.coords.freq / 70e6) ** (-2.0)
        assert jnp.allclose(out.data, 10.0 * factor[None, :])
        # stronger distortion at lower frequency
        assert out.data[0, 0] > out.data[0, -1]

    def test_ground_pickup_uses_env_temperature(self, template_state):
        op = GroundPickupOperator(coupling=jnp.array(0.01), t_ground=jnp.array(999.0))
        out = op(template_state)  # env.temperature = 280.0 in fixture
        assert jnp.allclose(out.data, 0.01 * 280.0)

    def test_ground_pickup_fallback_without_env(self, coords):
        op = GroundPickupOperator(coupling=jnp.array(0.01), t_ground=jnp.array(300.0))
        out = op(State(coords=coords))
        assert jnp.allclose(out.data, 3.0)

    def test_ground_pickup_per_time_temperature(self, coords):
        temps = jnp.linspace(270.0, 290.0, N_TIME)
        s = State(coords=coords, env=Environment(temperature=temps))
        out = GroundPickupOperator(coupling=jnp.array(1.0), t_ground=jnp.array(0.0))(s)
        assert jnp.allclose(out.data[:, 0], temps)

    def test_rfi_sparse_and_reproducible(self, template_state):
        op = RFIOperator(amplitude=jnp.array(100.0), occupancy=0.5)
        a, b = op(template_state), op(template_state)
        assert jnp.array_equal(a.data, b.data)  # same seed
        assert set(jnp.unique(a.data).tolist()) <= {0.0, 100.0}
        # key consumed
        assert not jnp.array_equal(
            jax.random.key_data(a.key), jax.random.key_data(template_state.key)
        )

    def test_rfi_occupancy_validated(self):
        with pytest.raises(StateValidationError, match="occupancy"):
            RFIOperator(amplitude=jnp.array(1.0), occupancy=1.5)


class TestInstrumentComponents:
    def test_noise_wave_linear_in_parameters(self, data_state):
        """The GCR-critical property: output is linear in (T_unc, T_cos, T_sin, T_rx)."""
        assert data_state.coords.freq.shape[0] == N_FREQ

        def build(t_unc, t_cos, t_sin, t_rx):
            return NoiseWaveOperator(
                t_unc=jnp.array(t_unc),
                t_cos=jnp.array(t_cos),
                t_sin=jnp.array(t_sin),
                t_rx=jnp.array(t_rx),
                gamma_src_re=jnp.full((1, N_FREQ), 0.1),
                gamma_src_im=jnp.full((1, N_FREQ), 0.05),
                gamma_rec_re=jnp.zeros(N_FREQ),
                gamma_rec_im=jnp.zeros(N_FREQ),
            )

        # One source, so no coords.extra["receiver_input"] switch array is needed.
        base = build(0.0, 0.0, 0.0, 0.0)(data_state).data
        # doubling each T doubles its (isolated) contribution
        for i, one in enumerate([(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)]):
            d1 = build(*one)(data_state).data - base
            d2 = build(*(2 * x for x in one))(data_state).data - base
            assert jnp.allclose(d2, 2.0 * d1), f"nonlinear in T component {i}"

    def test_noise_wave_reflection_loss(self, data_state):
        """|Gamma_src|^2 = 0.25 with a matched receiver (Gamma_rec = 0).

        Gamma_rec = 0 gives the reflection factor F = 1 exactly (draft Eq. 3),
        which is the placeholder's old F -> 1 limit — so the expected 7.5
        survives as a special case of the real model rather than an assumption
        it makes.
        """
        op = NoiseWaveOperator(
            t_unc=jnp.array(0.0), t_cos=jnp.array(0.0), t_sin=jnp.array(0.0),
            t_rx=jnp.array(0.0),
            gamma_src_re=jnp.full((1, N_FREQ), 0.3),
            gamma_src_im=jnp.full((1, N_FREQ), 0.4),
            gamma_rec_re=jnp.zeros(N_FREQ),
            gamma_rec_im=jnp.zeros(N_FREQ),
        )
        out = op(data_state)
        assert jnp.allclose(out.data, 10.0 * (1 - 0.25))  # |Gamma_src|^2 = 0.25

    def test_cw_tone_hits_nearest_channel(self, data_state):
        freq = data_state.coords.freq  # linspace(60e6, 85e6, 4)
        op = CWCalibrationOperator(
            amplitude=jnp.array(50.0),
            tone_freq=float(freq[2]),
            line_width=float(freq[1] - freq[0]),
        )
        out = op(data_state)
        assert jnp.all(out.data[:, 2] == 60.0)
        assert jnp.all(out.data[:, [0, 1, 3]] == 10.0)

    @pytest.mark.parametrize("tone_freq", [-1.0, 0.0])
    def test_cw_tone_freq_validated(self, tone_freq):
        """Both sides of a CLOSED boundary: 0 Hz is not a frequency either, and
        it is the value a missing setting defaults to."""
        with pytest.raises(StateValidationError, match="tone_freq"):
            CWCalibrationOperator(
                amplitude=jnp.array(1.0), tone_freq=tone_freq, line_width=1e6
            )

    def test_emi_comb(self, data_state):
        out = EMIOperator(amplitude=jnp.array(5.0), period=2)(data_state)
        assert jnp.all(out.data[:, 0] == 15.0) and jnp.all(out.data[:, 2] == 15.0)
        assert jnp.all(out.data[:, 1] == 10.0) and jnp.all(out.data[:, 3] == 10.0)

    def test_emi_period_validated(self):
        with pytest.raises(StateValidationError, match="period"):
            EMIOperator(amplitude=jnp.array(1.0), period=0)


class TestBackendComponents:
    def test_flagging_stores_mask_in_aux(self, data_state):
        spiked = data_state.with_data(data_state.data.at[0, 0].set(1e6))
        out = FlaggingOperator(threshold=100.0)(spiked)
        assert out.aux["flags"].shape == (N_TIME, N_FREQ)
        assert bool(out.aux["flags"][0, 0]) is True
        assert int(out.aux["flags"].sum()) == 1
        # data untouched; input state untouched
        assert jnp.array_equal(out.data, spiked.data)
        assert "flags" not in data_state.aux


class TestTaxonomyComposition:
    """The elements-structured forward model composes and differentiates."""

    @pytest.fixture
    def antenna_temperature(self):
        astro = Pipeline(
            SumOperator(
                GlobalSignalOperator(
                    depth=jnp.array(0.2), centre=jnp.array(72e6), width=jnp.array(5e6)
                ),
                ForegroundOperator(
                    amplitude=jnp.array(1e3), spectral_index=jnp.array(2.5), ref_freq=70e6
                ),
                PointSourceOperator(level=jnp.array(2.0)),
                names=("signal", "foregrounds", "point_sources"),
            ),
            IonosphereOperator(delta=jnp.array(0.01), ref_freq=70e6),
            names=("sky", "ionosphere"),
        )
        return SumOperator(
            astro,
            GroundPickupOperator(coupling=jnp.array(0.01), t_ground=jnp.array(300.0)),
            RFIOperator(amplitude=jnp.array(100.0), occupancy=0.05),
            names=("astro", "ground", "rfi"),
        )

    def test_composes_and_runs_under_jit(self, antenna_temperature, template_state):
        out = eqx.filter_jit(antenna_temperature)(template_state)
        assert out.data.shape == (N_TIME, N_FREQ)
        assert jnp.all(jnp.isfinite(out.data))

    def test_grad_reaches_science_parameter(self, antenna_temperature, template_state):
        """d(loss)/d(signal depth) must flow through the whole assembly."""

        def loss(op):
            return jnp.mean(op(template_state).data ** 2)

        grads = eqx.filter_grad(loss)(antenna_temperature)
        depth_grad = grads["astro"]["sky"]["signal"].depth
        assert jnp.isfinite(depth_grad) and depth_grad != 0
