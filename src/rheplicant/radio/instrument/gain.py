"""GainOperator — PLACEHOLDER time-dependent gain.

Real physics to come (port of limTOD / hydra-tod gain models): multiplicative
gain g(t) with 1/f flicker fluctuations. The gain is the primary calibration
target — it must stay a differentiable leaf so gradient-based and Bayesian
calibration can infer it.

THIS GAIN CARRIES THE ABSOLUTE LEVEL. Inferred jointly with a free bandpass,
``g(t)`` and ``b(nu)`` are separately unidentifiable: only their product enters
the prediction, so ``b -> c*b, g -> g/c`` changes nothing and the pair has one
exactly null direction. The package's convention puts the whole scale here and
leaves the bandpass as pure shape — declare the bandpass latent with
:func:`~rheplicant.radio.instrument.receiver.unit_mean_bandpass`, which is
where the measurement and the reasoning are written down.
"""

from typing import ClassVar

import jax

from rheplicant.core.errors import StateValidationError
from rheplicant.core.operator import AbstractOperator
from rheplicant.core.state import State


class GainOperator(AbstractOperator):
    """Multiply ``state.data`` by a gain (placeholder).

    Attributes:
        gain: differentiable scalar (constant gain) or ``(n_time,)`` array
            (per-sample gain, broadcast across frequency).
    """

    requires: ClassVar[tuple[str, ...]] = ("data",)
    provides: ClassVar[tuple[str, ...]] = ("data",)
    graph_node: ClassVar[str] = "gain"

    gain: jax.Array

    def __call__(self, state: State) -> State:
        if self.gain.ndim == 0:
            return state.with_data(state.data * self.gain)
        if self.gain.ndim == 1:
            if self.gain.shape[0] != state.data.shape[0]:
                raise StateValidationError(
                    f"gain has {self.gain.shape[0]} samples but data has "
                    f"{state.data.shape[0]} time samples."
                )
            return state.with_data(state.data * self.gain[:, None])
        raise StateValidationError(f"gain must be scalar or 1D, got ndim={self.gain.ndim}.")
