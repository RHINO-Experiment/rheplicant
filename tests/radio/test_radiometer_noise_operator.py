"""RadiometerNoiseOperator: the generator half of the radiometer equation."""

import jax
import jax.numpy as jnp
import pytest

from rheplicant.core import RANDOMNESS
from rheplicant.core.coordinates import Coordinates
from rheplicant.core.errors import StateValidationError
from rheplicant.core.state import State
from rheplicant.radio import RadiometerNoiseOperator


def make_state(data, seed=7):
    return State(
        data=data,
        coords=Coordinates(time=jnp.arange(float(data.shape[0])),
                           freq=jnp.linspace(60e6, 85e6, data.shape[1])),
        key=jax.random.key(seed),
    )


class TestTheDeclaration:
    def test_it_draws_and_says_so(self):
        assert RANDOMNESS in RadiometerNoiseOperator.requires
        assert RadiometerNoiseOperator.graph_node == "noise"

    def test_the_statics_mirror_the_likelihood_rule(self):
        op = RadiometerNoiseOperator(channel_width=3.125e6, integration_time=2.0)
        assert op.fractional == pytest.approx(1.0 / (3.125e6 * 2.0) ** 0.5)

    def test_nonpositive_widths_are_refused(self):
        with pytest.raises(StateValidationError, match="channel_width"):
            RadiometerNoiseOperator(channel_width=0.0, integration_time=2.0)
        with pytest.raises(StateValidationError, match="integration_time"):
            RadiometerNoiseOperator(channel_width=1e6, integration_time=-1.0)


class TestTheDraw:
    def test_the_form_is_multiplicative_at_a_negative_prediction(self):
        """d(1+fw), NOT d + |d| f w -- the two agree to rounding at d > 0 and
        differ in sign wherever the prediction is negative (the Plan 0
        lesson: sample where candidate implementations disagree)."""
        op = RadiometerNoiseOperator(channel_width=1e6, integration_time=1.0)
        data = jnp.full((4, 8), -100.0)
        state = make_state(data)
        subkey, _ = state.next_key()
        w = jax.random.normal(subkey, data.shape)
        out = op(state)
        expected = data * (1.0 + op.fractional * w)
        assert jnp.allclose(out.data, expected)
        wrong = data + jnp.abs(data) * op.fractional * w
        assert not jnp.allclose(out.data, wrong)

    def test_the_key_advances(self):
        op = RadiometerNoiseOperator(channel_width=1e6, integration_time=1.0)
        state = make_state(jnp.ones((4, 8)))
        once = op(state)
        assert not jnp.array_equal(once.key, state.key)
        twice = op(once)
        assert not jnp.allclose(once.data, twice.data)

    def test_no_key_is_a_loud_failure(self):
        from rheplicant.core.state import MissingKeyError

        op = RadiometerNoiseOperator(channel_width=1e6, integration_time=1.0)
        state = State(data=jnp.ones((2, 2)))
        with pytest.raises(MissingKeyError):
            op(state)
