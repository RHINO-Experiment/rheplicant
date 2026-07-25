"""BackendOperator — PLACEHOLDER backend processing.

Real physics to come: correlator/spectrometer integration, RFI flagging
(MomentRFI integration point), frequency rebinning, waterfall product
generation. This placeholder averages time chunks — and, importantly,
demonstrates the contract that *shape-changing operators must update the
coordinates along with the data*.
"""

from typing import ClassVar

import equinox as eqx

from replicant.core.errors import StateValidationError
from replicant.core.operator import AbstractOperator
from replicant.core.state import State


class BackendOperator(AbstractOperator):
    """Average ``state.data`` over time chunks of ``n_chunk`` samples (placeholder).

    Updates ``coords.time`` to the per-chunk mean times, keeping data and
    coordinates consistent.

    Attributes:
        n_chunk: samples per integration chunk (static configuration).
    """

    requires: ClassVar[tuple[str, ...]] = ("data", "coords.time")
    provides: ClassVar[tuple[str, ...]] = ("data", "coords.time")
    graph_node: ClassVar[str] = "averaging"

    n_chunk: int = eqx.field(static=True)

    def __check_init__(self):
        if not isinstance(self.n_chunk, int) or self.n_chunk < 1:
            raise StateValidationError(f"n_chunk must be a positive int, got {self.n_chunk!r}.")

    def __call__(self, state: State) -> State:
        n_time = state.data.shape[0]
        if n_time % self.n_chunk != 0:
            raise StateValidationError(
                f"n_time={n_time} is not divisible by n_chunk={self.n_chunk}."
            )
        n_out = n_time // self.n_chunk
        data = state.data.reshape(n_out, self.n_chunk, *state.data.shape[1:]).mean(axis=1)

        coords = state.coords
        if coords is not None and coords.time is not None:
            coords = coords.replace(time=coords.time.reshape(n_out, self.n_chunk).mean(axis=1))
        return state.replace(data=data, coords=coords)
