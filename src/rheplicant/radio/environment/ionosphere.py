"""IonosphereOperator — PLACEHOLDER ionospheric distortion.

Element: "Ionosphere (complicated, varying in time and frequency, distorting
the astrophysical signal)".

Real physics to come: time-variable ionospheric absorption and refraction
(chromatic, roughly ~ freq^-2), applied to the astrophysical signal only —
which is why in a forward model this operator sits *after* the astrophysical
sum but *before* terrestrial contributions are added. The placeholder is a
static chromatic scaling.
"""

from typing import ClassVar

import equinox as eqx
import jax

from rheplicant.core.errors import StateValidationError
from rheplicant.core.operator import AbstractOperator
from rheplicant.core.state import State


class IonosphereOperator(AbstractOperator):
    """Apply a chromatic ~freq^-2 distortion to existing data (placeholder).

    ``data * (1 + delta * (freq / ref_freq)^-2)`` — ``delta`` is a
    differentiable leaf controlling the distortion amplitude.

    Attributes:
        delta: fractional distortion at ``ref_freq``.
        ref_freq: reference frequency [Hz] (static configuration).
    """

    requires: ClassVar[tuple[str, ...]] = ("data", "coords.freq")
    provides: ClassVar[tuple[str, ...]] = ("data",)
    graph_node: ClassVar[str] = "ionosphere"

    delta: jax.Array
    ref_freq: float = eqx.field(static=True)

    def __check_init__(self):
        if self.ref_freq <= 0:
            raise StateValidationError(f"ref_freq must be > 0, got {self.ref_freq}.")

    def __call__(self, state: State) -> State:
        if state.coords is None or state.coords.freq is None:
            raise StateValidationError("IonosphereOperator requires state.coords.freq.")
        factor = 1.0 + self.delta * (state.coords.freq / self.ref_freq) ** (-2.0)
        return state.with_data(state.data * factor[None, :])
