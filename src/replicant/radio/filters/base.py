"""AbstractLinearFilter: the common shape of data filters.

Sidereal-repeat extraction, sky-space (map-making) filtering, and
fringe-rate/delay filtering are all *linear projections* — they differ only
in which subspace they project onto. The base class fixes the shared
semantics; concrete filters implement ``project`` and declare a static
``mode`` field:

    mode="extract" ->  P d        (keep the projected component)
    mode="remove"  ->  d - P d    (subtract it)

Filters are ordinary operators, so analysis chains are ordinary Pipelines —
and because filters are differentiable, the signal loss they induce (the
transfer function) can itself be marginalised in inference later.

Filters typically run on *calibrated* data (see
:class:`~replicant.radio.instrument.calibration.ApplyCalibrationOperator`);
preserve the raw data first with ``SnapshotOperator``.
"""

import abc
from typing import ClassVar

import jax

from replicant.core.errors import StateValidationError
from replicant.core.operator import AbstractOperator
from replicant.core.state import State

_MODES = ("extract", "remove")


class AbstractLinearFilter(AbstractOperator):
    """Base for projection filters. Concrete classes declare a static ``mode`` field."""

    graph_node: ClassVar[str] = "filters"

    def __check_init__(self):
        mode = getattr(self, "mode", None)
        if mode not in _MODES:
            raise StateValidationError(
                f"{type(self).__name__}.mode must be one of {_MODES}, got {mode!r}."
            )

    @abc.abstractmethod
    def project(self, data: jax.Array, state: State) -> jax.Array:
        """Return the projected component ``P d`` (same shape as ``data``)."""

    def __call__(self, state: State) -> State:
        projected = self.project(state.data, state)
        if self.mode == "extract":
            return state.with_data(projected)
        return state.with_data(state.data - projected)
