"""EMIOperator — PLACEHOLDER self-generated interference.

Element: "Self-generated EMI (e.g. due to power supply fluctuations,
switched-mode power sources and some control signals from the Arduino control
board, Odroid computer, or the SDR itself). Mostly looks like RFI."

Real physics to come: characterised spectral lines of the system's own
electronics (switching harmonics form comb-like structures). The placeholder
adds a constant-amplitude frequency comb.
"""

from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp

from replicant.core.errors import StateValidationError
from replicant.core.operator import AbstractOperator
from replicant.core.state import State


class EMIOperator(AbstractOperator):
    """Add a frequency comb of self-generated EMI lines (placeholder).

    Every ``period``-th channel receives an extra ``amplitude``.

    Attributes:
        amplitude: line amplitude [K-equivalent] — differentiable scalar.
        period: channel spacing of the comb (static configuration).
    """

    requires: ClassVar[tuple[str, ...]] = ("data",)
    provides: ClassVar[tuple[str, ...]] = ("data",)
    graph_node: ClassVar[str] = "emi"

    amplitude: jax.Array
    period: int = eqx.field(static=True)

    def __check_init__(self):
        if not isinstance(self.period, int) or self.period < 1:
            raise StateValidationError(f"period must be a positive int, got {self.period!r}.")

    def __call__(self, state: State) -> State:
        n_freq = state.data.shape[-1]
        comb = (jnp.arange(n_freq) % self.period == 0).astype(state.data.dtype)
        return state.with_data(state.data + self.amplitude * comb[None, :])
