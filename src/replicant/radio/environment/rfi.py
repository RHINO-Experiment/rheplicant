"""RFIOperator — PLACEHOLDER radio-frequency interference.

Element: "RFI (very complicated, mix of narrow and wideband signals with
temporal structure on a variety of scales, some bright and some low-level)".
Pain point: "For low-level RFI, we can perhaps consider a stochastic process
model that fits unknown added variance based on night to night variations."

Real physics to come: a stochastic-process model for low-level unflagged RFI
(the hardest pain point — no reasonable model exists yet), plus bright
narrow/wideband transmitters. The placeholder draws a sparse random mask of
constant-amplitude spikes via the State PRNG protocol.
"""

from typing import ClassVar

import equinox as eqx
import jax

from replicant.core.errors import StateValidationError
from replicant.core.operator import AbstractOperator
from replicant.core.state import State


class RFIOperator(AbstractOperator):
    """Produce sparse random RFI spikes (placeholder).

    Contribution: ``amplitude * mask`` where ``mask ~ Bernoulli(occupancy)``
    per (time, freq) cell, drawn through the State PRNG protocol.

    Attributes:
        amplitude: spike amplitude [K] — differentiable scalar.
        occupancy: probability a cell hosts RFI (static configuration;
            the mask draw is not differentiable anyway).
    """

    requires: ClassVar[tuple[str, ...]] = ("coords.time", "coords.freq", "key")
    provides: ClassVar[tuple[str, ...]] = ("data",)
    graph_node: ClassVar[str] = "rfi_field"

    amplitude: jax.Array
    occupancy: float = eqx.field(static=True)

    def __check_init__(self):
        if not 0.0 <= self.occupancy <= 1.0:
            raise StateValidationError(f"occupancy must be in [0, 1], got {self.occupancy}.")

    def __call__(self, state: State) -> State:
        if state.coords is None or state.coords.time is None or state.coords.freq is None:
            raise StateValidationError(
                "RFIOperator requires state.coords with time and freq axes."
            )
        shape = (state.coords.time.shape[0], state.coords.freq.shape[0])
        subkey, state = state.next_key()
        mask = jax.random.bernoulli(subkey, self.occupancy, shape)
        # bool mask promotes to the amplitude's dtype — no hardcoded precision
        return state.with_data(self.amplitude * mask)
