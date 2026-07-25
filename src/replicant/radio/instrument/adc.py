"""ADCOperator — PLACEHOLDER digitization.

Real physics to come: true quantization (round to 2**n_bits levels) has zero
gradient almost everywhere, so the differentiable version will need a
straight-through estimator (identity gradient through the rounding) or a
smooth surrogate. This placeholder applies scale + clip, which is
differentiable almost everywhere and preserves the saturation behaviour.
"""

from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp

from replicant.core.errors import StateValidationError
from replicant.core.operator import AbstractOperator
from replicant.core.state import State


class ADCOperator(AbstractOperator):
    """Scale and clip ``state.data`` to the ADC dynamic range (placeholder).

    Attributes:
        scale: pre-digitization scaling — differentiable scalar.
        n_bits: ADC bit depth (static configuration; clip limit is 2**(n_bits-1)).
    """

    requires: ClassVar[tuple[str, ...]] = ("data",)
    provides: ClassVar[tuple[str, ...]] = ("data",)
    graph_node: ClassVar[str] = "adc"

    scale: jax.Array
    n_bits: int = eqx.field(static=True)

    def __check_init__(self):
        if not isinstance(self.n_bits, int) or self.n_bits < 1:
            raise StateValidationError(f"n_bits must be a positive int, got {self.n_bits!r}.")

    def __call__(self, state: State) -> State:
        limit = 2.0 ** (self.n_bits - 1)
        return state.with_data(jnp.clip(state.data * self.scale, -limit, limit))
