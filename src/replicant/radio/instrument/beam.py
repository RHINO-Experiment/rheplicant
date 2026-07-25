"""BeamOperator — PLACEHOLDER beam response.

Real physics to come (port of limTOD's beam handling): a measured or
EM-simulated primary beam convolved with the sky (limTOD does this via
harmonic-space alm rotation, ZYZ Euler convention). This placeholder reduces
the beam to a single coupling factor.
"""

from typing import ClassVar

import jax

from replicant.core.operator import AbstractOperator
from replicant.core.state import State


class BeamOperator(AbstractOperator):
    """Scale ``state.data`` by an effective beam solid-angle factor (placeholder).

    Attributes:
        solid_angle: dimensionless beam coupling factor — differentiable scalar.
    """

    requires: ClassVar[tuple[str, ...]] = ("data",)
    provides: ClassVar[tuple[str, ...]] = ("data",)
    graph_node: ClassVar[str] = "beam"

    solid_angle: jax.Array

    def __call__(self, state: State) -> State:
        return state.with_data(state.data * self.solid_angle)
