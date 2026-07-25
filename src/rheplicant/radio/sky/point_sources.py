"""PointSourceOperator — PLACEHOLDER bright point sources.

Element: "Bright point sources (these are mostly diluted by the beam, but
still there at a low level)".

Real physics to come: a catalogue of bright sources with spectra, entering
through the (sidelobe-weighted) beam as the sky drifts. The placeholder is a
constant low-level contribution.
"""

from typing import ClassVar

import jax
import jax.numpy as jnp

from rheplicant.core.errors import StateValidationError
from rheplicant.core.operator import AbstractOperator
from rheplicant.core.state import State


class PointSourceOperator(AbstractOperator):
    """Produce a constant beam-diluted point-source level (placeholder).

    Attributes:
        level: effective contribution [K] — differentiable scalar.
    """

    requires: ClassVar[tuple[str, ...]] = ("coords.time", "coords.freq")
    provides: ClassVar[tuple[str, ...]] = ("data",)
    graph_node: ClassVar[str] = "point_sources"

    level: jax.Array

    def __call__(self, state: State) -> State:
        if state.coords is None or state.coords.time is None or state.coords.freq is None:
            raise StateValidationError(
                "PointSourceOperator requires state.coords with time and freq axes."
            )
        shape = (state.coords.time.shape[0], state.coords.freq.shape[0])
        return state.with_data(self.level * jnp.ones(shape))
