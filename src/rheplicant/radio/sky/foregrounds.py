"""ForegroundOperator — PLACEHOLDER diffuse foregrounds.

Element: "Multiple diffuse FG components (variable in LST and freq.)".

Real physics to come (port of limTOD's sky handling): foreground maps with
uncertain spectral-index structure (Anstey-style) improved via the moment
expansion — the identified pain point where "reasonable" few-percent models
must reach 0.1% accuracy. The placeholder is a single power law, constant in
time (the real component varies in LST as the Galaxy transits).
"""

from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp

from rheplicant.core.errors import StateValidationError
from rheplicant.core.operator import AbstractOperator
from rheplicant.core.state import State


class ForegroundOperator(AbstractOperator):
    """Produce a power-law foreground spectrum on the (time, freq) grid (placeholder).

    Contribution: ``amplitude * (freq / ref_freq) ** (-spectral_index)``,
    constant in time. Amplitude and spectral index are differentiable leaves.

    Attributes:
        amplitude: brightness temperature at ``ref_freq`` [K].
        spectral_index: power-law index (synchrotron-like ~2.5).
        ref_freq: reference frequency [Hz] (static configuration).
    """

    requires: ClassVar[tuple[str, ...]] = ("coords.time", "coords.freq")
    provides: ClassVar[tuple[str, ...]] = ("data",)
    graph_node: ClassVar[str] = "foregrounds"

    amplitude: jax.Array
    spectral_index: jax.Array
    ref_freq: float = eqx.field(static=True)

    def __check_init__(self):
        if not self.ref_freq > 0:  # `not >` so a NaN ref_freq is refused too
            raise StateValidationError(f"ref_freq must be > 0, got {self.ref_freq}.")

    def __call__(self, state: State) -> State:
        if state.coords is None or state.coords.time is None or state.coords.freq is None:
            raise StateValidationError(
                "ForegroundOperator requires state.coords with time and freq axes."
            )
        freq = state.coords.freq
        profile = self.amplitude * (freq / self.ref_freq) ** (-self.spectral_index)
        n_time = state.coords.time.shape[0]
        return state.with_data(jnp.broadcast_to(profile[None, :], (n_time, freq.shape[0])))
