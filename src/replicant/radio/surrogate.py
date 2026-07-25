"""Neural surrogate operators: hybrid physics + ML in the same formalism.

A neural network is just another operator: an ``eqx.nn.MLP``'s weights are
ordinary traced leaves, so :class:`NeuralOperator` plugs into pipelines,
graph assembly, ``build_forward_fn``, gradient calibration, Fisher forecasts,
and the NumPyro bridge exactly like a physical parameter set — no special
machinery anywhere.

:class:`NeuralOperator` learns a positive multiplicative spectral response
``data * exp(MLP(freq))`` — a drop-in surrogate for any smooth
frequency-dependent stage (bandpass, beam chromaticity correction, ...).
It deliberately has NO default graph node: a surrogate's placement is a
modelling decision, so provide it explicitly, e.g.
``At("bandpass", NeuralOperator.create(...))`` to replace the physical
bandpass with a learned one (see ``examples/neural_surrogate.py``).
"""

from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp

from replicant.core.errors import StateValidationError
from replicant.core.operator import AbstractOperator
from replicant.core.state import State


class NeuralOperator(AbstractOperator):
    """Learned positive spectral response: ``data * exp(MLP(freq_normalized))``.

    The MLP maps a normalized frequency in ``[-1, 1]`` to a log-correction,
    so the response is positive by construction and initializes near unity
    (fresh MLPs output ~0). Weights are differentiable leaves.

    Attributes:
        mlp: ``eqx.nn.MLP`` with ``in_size=1, out_size=1``.
        f_min: lower edge of the frequency normalization window [Hz] (static).
        f_max: upper edge [Hz] (static).
    """

    requires: ClassVar[tuple[str, ...]] = ("data", "coords.freq")
    provides: ClassVar[tuple[str, ...]] = ("data",)

    mlp: eqx.nn.MLP
    f_min: float = eqx.field(static=True)
    f_max: float = eqx.field(static=True)

    def __check_init__(self):
        if not self.f_max > self.f_min:
            raise StateValidationError(
                f"Need f_max > f_min, got f_min={self.f_min}, f_max={self.f_max}."
            )

    @classmethod
    def create(
        cls,
        key: jax.Array,
        f_min: float,
        f_max: float,
        width: int = 16,
        depth: int = 2,
    ) -> "NeuralOperator":
        """Build a fresh surrogate (near-identity response) for a frequency window."""
        mlp = eqx.nn.MLP(
            in_size=1, out_size=1, width_size=width, depth=depth,
            activation=jax.nn.tanh, key=key,
        )
        return cls(mlp=mlp, f_min=f_min, f_max=f_max)

    def response(self, freq: jax.Array) -> jax.Array:
        """The learned spectral response, shape ``(n_freq,)`` (positive).

        The log-correction is clipped to ±20 before exponentiation, so a
        wild training step cannot overflow to inf/NaN and permanently poison
        an optimizer's moment estimates — the response stays finite in
        (~2e-9, ~5e8) and gradients keep flowing back toward sanity.
        """
        features = 2.0 * (freq - self.f_min) / (self.f_max - self.f_min) - 1.0
        log_correction = jax.vmap(self.mlp)(features[:, None])[:, 0]
        return jnp.exp(jnp.clip(log_correction, -20.0, 20.0))

    def __call__(self, state: State) -> State:
        if state.coords is None or state.coords.freq is None:
            raise StateValidationError("NeuralOperator requires state.coords.freq.")
        return state.with_data(state.data * self.response(state.coords.freq)[None, :])
