"""BackendOperator — PLACEHOLDER backend processing.

Real physics to come: correlator/spectrometer integration, RFI flagging
(MomentRFI integration point), frequency rebinning, waterfall product
generation. This placeholder averages time chunks — and, importantly,
demonstrates the contract that *shape-changing operators must update the
coordinates along with the data*.
"""

from typing import ClassVar

import equinox as eqx

from rheplicant.core.errors import StateValidationError
from rheplicant.core.operator import AbstractOperator
from rheplicant.core.state import State


class BackendOperator(AbstractOperator):
    """Average ``state.data`` over time chunks of ``n_chunk`` samples (placeholder).

    Updates ``coords.time`` to the per-chunk mean times, keeping data and
    coordinates consistent.

    This is one of exactly two places in the package that does ARITHMETIC on
    ``coords.time`` values rather than reading its length (the other is
    :class:`~rheplicant.radio.instrument.calibration.CWCalibrationOperator`'s
    drift), and it is where a time axis the stored dtype cannot carry surfaces
    as a wrong number rather than as an exception. Measured on 8 samples 100 s
    apart on a unix-epoch axis, ``n_chunk=2``, before
    :class:`~rheplicant.core.coordinates.Coordinates` guarded its own store::

        chunk times  [1750000128, 1750000256, 1750000384, 1750000640]
        float64 truth[1750000050, 1750000250, 1750000450, 1750000650]
        error [s]    [       +78,         +6,        -66,        -10]

    Wrong by 78 s of a 100 s cadence, with two of the eight input samples
    already merged before the mean ran. Nothing here can detect that -- the
    merged values arrive indistinguishable from real ones -- so the check lives
    at the store, in ``Coordinates``, and this operator inherits it: the
    ``replace`` below re-runs it on the chunk-mean axis it produces.

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
