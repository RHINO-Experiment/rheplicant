"""SkyOperator — PLACEHOLDER sky model.

Real physics to come (port of limTOD's sky-TOD step): sky maps with spectral
models, observed along ``coords.pointing``. This placeholder only establishes
the contract: *the sky operator creates ``data`` on the (time, frequency)
grid defined by the coordinates.*
"""

from typing import ClassVar

import jax
import jax.numpy as jnp

from rheplicant.core.errors import StateValidationError
from rheplicant.core.operator import AbstractOperator
from rheplicant.core.state import State


class SkyOperator(AbstractOperator):
    """Fill ``state.data`` with a uniform sky brightness [K] (placeholder).

    Attributes:
        amplitude: sky brightness temperature [K] — a differentiable scalar.
    """

    requires: ClassVar[tuple[str, ...]] = ("coords.time", "coords.freq")
    provides: ClassVar[tuple[str, ...]] = ("data",)
    graph_node: ClassVar[str] = "uniform_sky"

    amplitude: jax.Array

    def __call__(self, state: State) -> State:
        if state.coords is None or state.coords.time is None or state.coords.freq is None:
            raise StateValidationError(
                "SkyOperator requires state.coords with both time and freq axes."
            )
        n_time = state.coords.time.shape[0]
        n_freq = state.coords.freq.shape[0]
        sky = self.amplitude * jnp.ones((n_time, n_freq))
        return state.with_data(sky)
