"""Hybrid physics + ML: a neural surrogate learns the receiver bandpass.

The truth twin carries a rippled physical bandpass; the surrogate twin
replaces that single stage with ``NeuralOperator`` (an eqx.nn.MLP) placed at
the same graph node via ``At("bandpass", ...)``. Training only the MLP
weights through the standard calibration seam recovers the bandpass shape —
no ML-specific machinery anywhere: network weights are ordinary
differentiable leaves.

Run:  uv run python examples/neural_surrogate.py
"""

import equinox as eqx
import jax
import jax.numpy as jnp

from replicant import At, Coordinates, State
from replicant.inference import AdamCalibrator, build_forward_fn
from replicant.radio import (
    GainOperator,
    NeuralOperator,
    ReceiverOperator,
    SkyOperator,
    assemble,
)

F_MIN, F_MAX = 60e6, 85e6
N_TIME, N_FREQ = 16, 32

state = State(
    coords=Coordinates(time=jnp.linspace(0.0, 60.0, N_TIME),
                       freq=jnp.linspace(F_MIN, F_MAX, N_FREQ)),
)
freq = state.coords.freq

# Truth: a physically structured (rippled) bandpass.
ripple = 1.0 + 0.3 * jnp.sin(2 * jnp.pi * 3 * (freq - F_MIN) / (F_MAX - F_MIN))
truth = assemble(
    SkyOperator(amplitude=jnp.array(1.0)),
    ReceiverOperator(bandpass=ripple),
    GainOperator(gain=jnp.array(1.0)),
)
observed = truth(state).data

# Surrogate twin: same chain, the bandpass stage is now a neural network.
surrogate_twin = assemble(
    SkyOperator(amplitude=jnp.array(1.0)),
    At("bandpass", NeuralOperator.create(jax.random.key(1), F_MIN, F_MAX)),
    GainOperator(gain=jnp.array(1.0)),
)

# Train ONLY the MLP weights (is_inexact_array skips the activation function).
spec = jax.tree.map(lambda _: False, surrogate_twin)
spec = eqx.tree_at(
    lambda p: p["bandpass"], spec,
    jax.tree.map(eqx.is_inexact_array, surrogate_twin["bandpass"]),
)
forward, params0 = build_forward_fn(surrogate_twin, state, filter_spec=spec)

calibrator = AdamCalibrator(learning_rate=1e-2, n_steps=3000)
params_fit, losses = calibrator.fit(forward, params0, observed)

fitted = eqx.combine(params_fit, eqx.partition(surrogate_twin, spec)[1])
recovered = fitted["bandpass"].response(freq)
rel_err = float(jnp.max(jnp.abs(recovered - ripple) / ripple))
print(f"training loss: {float(losses[0]):.3e} -> {float(losses[-1]):.3e}")
print(f"bandpass recovered by the neural surrogate: max relative error {rel_err:.2%}")
