"""Receiver bandpass — PLACEHOLDER.

Real physics to come: frequency-dependent bandpass, and eventually
reflection/impedance-mismatch effects along the signal chain. Sky-side
additive temperatures (atmosphere, ground spill) are *not* receiver
business — they are branches of the antenna-temperature sum
(:class:`~replicant.radio.environment.atmosphere.AtmosphericEmissionOperator`,
:class:`~replicant.radio.environment.ground.GroundPickupOperator`), entering
before the reflection/noise-wave terms and therefore seeing the
``(1-|Gamma|^2)`` loss. The receiver temperature itself enters after the
reflection, as the noise-wave ``T_0`` (see
:class:`~replicant.radio.instrument.noise_wave.NoiseWaveOperator`) and the
post-gain thermal noise ``T_n``.
"""

from typing import ClassVar

import jax

from replicant.core.errors import StateValidationError
from replicant.core.operator import AbstractOperator
from replicant.core.state import State


class ReceiverOperator(AbstractOperator):
    """Apply a frequency-dependent bandpass to ``state.data`` (placeholder).

    Attributes:
        bandpass: ``(n_freq,)`` dimensionless bandpass — differentiable.
    """

    requires: ClassVar[tuple[str, ...]] = ("data",)
    provides: ClassVar[tuple[str, ...]] = ("data",)
    graph_node: ClassVar[str] = "bandpass"

    bandpass: jax.Array

    def __call__(self, state: State) -> State:
        if self.bandpass.shape[-1] != state.data.shape[-1]:
            raise StateValidationError(
                f"bandpass has {self.bandpass.shape[-1]} channels but data has "
                f"{state.data.shape[-1]}."
            )
        return state.with_data(state.data * self.bandpass[None, :])
