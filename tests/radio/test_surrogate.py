"""Tests for NeuralOperator: hybrid physics + ML as an ordinary operator."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from rheplicant import State
from rheplicant.core.errors import StateValidationError
from rheplicant.core.graph import At
from rheplicant.inference import AdamCalibrator, build_forward_fn
from rheplicant.radio import (
    GainOperator,
    NeuralOperator,
    ReceiverOperator,
    SkyOperator,
    assemble,
)

F_MIN, F_MAX = 60e6, 85e6
N_TIME, N_FREQ = 8, 4


@pytest.fixture
def surrogate():
    return NeuralOperator.create(jax.random.key(0), f_min=F_MIN, f_max=F_MAX)


class TestNeuralOperator:
    def test_response_positive_and_near_unity_at_init(self, surrogate, coords):
        response = surrogate.response(coords.freq)
        assert response.shape == (N_FREQ,)
        assert jnp.all(response > 0)
        assert jnp.allclose(response, 1.0, atol=0.5)  # fresh MLP ~ identity

    def test_applies_response_to_data(self, surrogate, template_state):
        s = template_state.with_data(jnp.ones((N_TIME, N_FREQ)))
        out = surrogate(s)
        assert jnp.allclose(out.data, surrogate.response(s.coords.freq)[None, :])

    def test_requires_freq(self, surrogate):
        with pytest.raises(StateValidationError, match="freq"):
            surrogate(State(data=jnp.ones((2, 2))))

    def test_window_validated(self):
        with pytest.raises(StateValidationError, match="f_max"):
            NeuralOperator.create(jax.random.key(0), f_min=1.0, f_max=1.0)

    def test_weights_are_differentiable_leaves(self, surrogate, template_state):
        s = template_state.with_data(jnp.ones((N_TIME, N_FREQ)))

        def loss(op):
            return jnp.sum(op(s).data ** 2)

        grads = eqx.filter_grad(loss)(surrogate)
        leaves = jax.tree.leaves(eqx.filter(grads, eqx.is_inexact_array))
        assert leaves and any(jnp.any(g != 0) for g in leaves)

    def test_placeable_via_at(self, template_state, surrogate):
        asm = assemble(
            SkyOperator(amplitude=jnp.array(1.0)),
            At("bandpass", surrogate),
        )
        assert "bandpass" in asm.lit
        assert jnp.all(jnp.isfinite(asm(template_state).data))


class TestHybridTraining:
    def test_surrogate_learns_structured_bandpass(self, template_state):
        """The headline hybrid demo: physics chain + neural stage recovers
        a rippled bandpass by gradient descent on the MLP weights only."""
        freq = template_state.coords.freq
        ripple = 1.0 + 0.3 * jnp.sin(
            2 * jnp.pi * 3 * (freq - F_MIN) / (F_MAX - F_MIN)
        )
        truth = assemble(
            SkyOperator(amplitude=jnp.array(1.0)),
            ReceiverOperator(bandpass=ripple),
            GainOperator(gain=jnp.array(1.0)),
        )
        observed = truth(template_state).data

        surrogate_twin = assemble(
            SkyOperator(amplitude=jnp.array(1.0)),
            At("bandpass", NeuralOperator.create(jax.random.key(1), F_MIN, F_MAX)),
            GainOperator(gain=jnp.array(1.0)),
        )
        # train ONLY the MLP weights (is_inexact_array skips the activation fn)
        spec = jax.tree.map(lambda _: False, surrogate_twin)
        spec = eqx.tree_at(
            lambda p: p["bandpass"], spec,
            jax.tree.map(eqx.is_inexact_array, surrogate_twin["bandpass"]),
        )
        forward, params0 = build_forward_fn(
            surrogate_twin, template_state, filter_spec=spec
        )
        calibrator = AdamCalibrator(learning_rate=1e-2, n_steps=1500)
        params_fit, losses = calibrator.fit(forward, params0, observed)

        assert losses[-1] < losses[0] / 100  # loss drops by > 2 orders
        fitted = eqx.combine(
            params_fit, eqx.partition(surrogate_twin, spec)[1]
        )["bandpass"]
        recovered = fitted.response(freq)
        rel_err = jnp.max(jnp.abs(recovered - ripple) / ripple)
        assert rel_err < 0.05  # bandpass recovered to < 5%
