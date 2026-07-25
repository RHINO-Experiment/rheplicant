"""GlobalSignalOperator — PLACEHOLDER 21 cm global signal.

Element: "21cm global signal (roughly const. in LST, smooth in freq.)".

Real physics to come: physical global-signal models (e.g. parametrized
absorption troughs, ARES/21cmFAST-style physical parameters) whose recovery
is the science target. The placeholder is a Gaussian absorption feature —
constant in time, smooth in frequency — with differentiable depth, centre,
and width, so signal-recovery experiments work end-to-end already.
"""

from typing import ClassVar

import jax
import jax.numpy as jnp

from replicant.core.errors import StateValidationError
from replicant.core.operator import AbstractOperator
from replicant.core.state import State


class GlobalSignalOperator(AbstractOperator):
    """Produce a Gaussian absorption trough on the (time, freq) grid (placeholder).

    Contribution: ``-depth * exp(-0.5 ((freq - centre) / width)^2)``,
    constant in time. All three parameters are differentiable leaves.

    Attributes:
        depth: trough depth [K] (positive number gives absorption).
        centre: trough centre frequency [Hz].
        width: trough Gaussian width [Hz].
    """

    requires: ClassVar[tuple[str, ...]] = ("coords.time", "coords.freq")
    provides: ClassVar[tuple[str, ...]] = ("data",)
    graph_node: ClassVar[str] = "global_signal"

    depth: jax.Array
    centre: jax.Array
    width: jax.Array

    def __call__(self, state: State) -> State:
        if state.coords is None or state.coords.time is None or state.coords.freq is None:
            raise StateValidationError(
                "GlobalSignalOperator requires state.coords with time and freq axes."
            )
        freq = state.coords.freq
        profile = -self.depth * jnp.exp(-0.5 * ((freq - self.centre) / self.width) ** 2)
        n_time = state.coords.time.shape[0]
        return state.with_data(jnp.broadcast_to(profile[None, :], (n_time, freq.shape[0])))
