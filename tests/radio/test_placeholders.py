"""Contract tests for the radio placeholder operators.

The math here is intentionally trivial (placeholders), but the *contracts* —
shapes, purity, PRNG consumption, coordinate updates — are the real thing and
must survive when real physics replaces the bodies.
"""

import jax
import jax.numpy as jnp
import pytest

from rheplicant import AbstractOperator, State
from rheplicant.core.errors import StateValidationError
from rheplicant.radio import (
    ADCOperator,
    AtmosphericEmissionOperator,
    BackendOperator,
    GainOperator,
    NoiseOperator,
    ReceiverOperator,
    SkyOperator,
)

# Mirrors tests/conftest.py fixture dimensions.
N_TIME = 8
N_FREQ = 4


@pytest.fixture
def data_state(template_state):
    """Template state with a known (n_time, n_freq) data payload."""
    return template_state.with_data(jnp.full((N_TIME, N_FREQ), 10.0))


ALL_OPERATOR_CLASSES = [
    SkyOperator,
    AtmosphericEmissionOperator,
    ReceiverOperator,
    GainOperator,
    NoiseOperator,
    ADCOperator,
    BackendOperator,
]


@pytest.mark.parametrize("cls", ALL_OPERATOR_CLASSES)
def test_contract_declared(cls):
    assert issubclass(cls, AbstractOperator)
    assert isinstance(cls.requires, tuple) and isinstance(cls.provides, tuple)
    assert "data" in cls.provides


class TestSkyOperator:
    def test_fills_data_from_coords(self, template_state):
        out = SkyOperator(amplitude=jnp.array(1e3))(template_state)
        assert out.data.shape == (N_TIME, N_FREQ)
        assert jnp.all(out.data == 1e3)

    def test_requires_coords(self):
        with pytest.raises(StateValidationError, match="coords"):
            SkyOperator(amplitude=jnp.array(1.0))(State(key=jax.random.key(0)))


class TestSimpleArithmetic:
    def test_atmosphere_emits_scalar(self, template_state):
        # Source-type: a t_ant_sum branch producing its own contribution.
        out = AtmosphericEmissionOperator(t_atm=jnp.array(150.0))(template_state)
        assert out.data.shape == (N_TIME, N_FREQ)
        assert jnp.all(out.data == 150.0)

    def test_atmosphere_emits_per_freq(self, template_state):
        t_atm = jnp.arange(float(N_FREQ))
        out = AtmosphericEmissionOperator(t_atm=t_atm)(template_state)
        assert out.data.shape == (N_TIME, N_FREQ)
        assert jnp.array_equal(out.data[0], t_atm)

    def test_atmosphere_channel_mismatch_raises(self, template_state):
        with pytest.raises(StateValidationError, match="t_atm"):
            AtmosphericEmissionOperator(t_atm=jnp.ones(N_FREQ + 1))(template_state)

    def test_receiver_applies_bandpass(self, data_state):
        bandpass = jnp.arange(1.0, N_FREQ + 1.0)
        out = ReceiverOperator(bandpass=bandpass)(data_state)
        assert jnp.array_equal(out.data[3], 10.0 * bandpass)

    def test_receiver_shape_mismatch_raises(self, data_state):
        with pytest.raises(StateValidationError, match="bandpass"):
            ReceiverOperator(bandpass=jnp.ones(N_FREQ + 1))(data_state)

    def test_receiver_scalar_bandpass(self, data_state):
        out = ReceiverOperator(bandpass=jnp.array(2.0))(data_state)
        assert jnp.array_equal(out.data, 20.0 * jnp.ones((N_TIME, N_FREQ)))

    @pytest.mark.parametrize(
        "shape",
        [(1, N_FREQ), (2, N_FREQ), (N_TIME, N_FREQ), (1, 1, N_FREQ)],
        ids=["leading-1", "leading-2", "leading-n_time", "3d"],
    )
    def test_receiver_ndim_too_high_raises(self, data_state, shape):
        # Every one of these has a LAST axis that matches n_freq, so it would
        # broadcast silently via bandpass[None, :] instead of failing the
        # existing (n_freq,) length check — the guard must catch ndim itself.
        with pytest.raises(StateValidationError, match="bandpass"):
            ReceiverOperator(bandpass=jnp.ones(shape))(data_state)

    def test_gain_scalar(self, data_state):
        out = GainOperator(gain=jnp.array(2.0))(data_state)
        assert jnp.all(out.data == 20.0)

    def test_gain_per_time(self, data_state):
        gain = jnp.arange(1.0, N_TIME + 1.0)
        out = GainOperator(gain=gain)(data_state)
        assert jnp.array_equal(out.data[:, 0], 10.0 * gain)

    def test_input_state_untouched(self, data_state):
        GainOperator(gain=jnp.array(2.0))(data_state)
        assert jnp.all(data_state.data == 10.0)


class TestNoiseOperator:
    def test_consumes_key(self, data_state):
        out = NoiseOperator(sigma=jnp.array(0.1))(data_state)
        assert not jnp.array_equal(
            jax.random.key_data(out.key), jax.random.key_data(data_state.key)
        )

    def test_seed_reproducible(self, data_state):
        op = NoiseOperator(sigma=jnp.array(0.1))
        assert jnp.array_equal(op(data_state).data, op(data_state).data)

    def test_successive_draws_differ(self, data_state):
        op = NoiseOperator(sigma=jnp.array(0.1))
        once = op(data_state)
        twice = op(once)  # advanced key -> different realisation
        assert not jnp.array_equal(once.data, twice.data)

    def test_noise_magnitude_scales_with_sigma(self, data_state):
        quiet = NoiseOperator(sigma=jnp.array(1e-6))(data_state)
        assert jnp.allclose(quiet.data, 10.0, atol=1e-4)


class TestADCOperator:
    def test_scales_and_clips(self, data_state):
        out = ADCOperator(scale=jnp.array(1.0), n_bits=4)(data_state)  # limit = 8
        assert jnp.all(out.data == 8.0)

    def test_within_range_passes_through(self, data_state):
        out = ADCOperator(scale=jnp.array(0.1), n_bits=14)(data_state)
        assert jnp.allclose(out.data, 1.0)

    def test_invalid_bits_rejected(self):
        with pytest.raises(StateValidationError, match="n_bits"):
            ADCOperator(scale=jnp.array(1.0), n_bits=0)


class TestBackendOperator:
    def test_averages_time_chunks(self, data_state):
        out = BackendOperator(n_chunk=4)(data_state)
        assert out.data.shape == (N_TIME // 4, N_FREQ)
        assert jnp.all(out.data == 10.0)

    def test_updates_time_coordinate(self, data_state):
        out = BackendOperator(n_chunk=4)(data_state)
        assert out.coords.time.shape == (N_TIME // 4,)
        # mean of [0,1,2,3] and [4,5,6,7] on linspace(0,7,8)
        assert jnp.array_equal(out.coords.time, jnp.array([1.5, 5.5]))

    def test_indivisible_chunk_raises(self, data_state):
        with pytest.raises(StateValidationError, match="divisible"):
            BackendOperator(n_chunk=3)(data_state)

    def test_invalid_chunk_rejected(self):
        with pytest.raises(StateValidationError, match="n_chunk"):
            BackendOperator(n_chunk=0)
