"""SiderealFilter: project onto the day-repeating (sky-locked) subspace.

Structure that repeats every sidereal day is sky-locked (the drifting sky
through a fixed beam); everything else is instrument/environment. Averaging
the same LST bin across days is the orthogonal projection onto that
subspace — ``extract`` returns the repeating sidereal structure, ``remove``
returns the non-repeating residual.

Run this on *calibrated* data: uncorrected gain drifts are not sky-locked and
would leak into the day-average. Real version: irregular/gapped LST sampling
via binning; the placeholder assumes the time axis is exactly ``n_days``
concatenated identical LST grids (day-major ordering).
"""

from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp

from replicant.core.errors import StateValidationError
from replicant.core.state import State
from replicant.radio.filters.base import AbstractLinearFilter


class SiderealFilter(AbstractLinearFilter):
    """Per-LST mean across days, tiled back to full length.

    Attributes:
        n_days: number of sidereal days concatenated along the time axis
            (static; ``n_time`` must be divisible by it).
        mode: ``"extract"`` (repeating structure) or ``"remove"`` (residual).
    """

    requires: ClassVar[tuple[str, ...]] = ("data",)
    provides: ClassVar[tuple[str, ...]] = ("data",)

    n_days: int = eqx.field(static=True)
    mode: str = eqx.field(static=True, default="remove")

    def __check_init__(self):
        if not isinstance(self.n_days, int) or self.n_days < 2:
            raise StateValidationError(
                f"n_days must be an int >= 2 (repetition needs at least two days), "
                f"got {self.n_days!r}."
            )

    def project(self, data: jax.Array, state: State) -> jax.Array:
        n_time = data.shape[0]
        if n_time % self.n_days != 0:
            raise StateValidationError(
                f"n_time={n_time} is not divisible by n_days={self.n_days}."
            )
        n_lst = n_time // self.n_days
        per_day = data.reshape(self.n_days, n_lst, *data.shape[1:])
        template = per_day.mean(axis=0)
        return jnp.tile(template, (self.n_days,) + (1,) * (data.ndim - 1))
