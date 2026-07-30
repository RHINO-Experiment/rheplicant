"""AntennaLossOperator — the antenna's own ohmic loss, before the receiver.

A real antenna is not a lossless collector. Conductor and dielectric loss
dissipate a fraction of everything it gathers, and — by Kirchhoff, since the
lossy structure is a passive element in thermal equilibrium — re-emit that
fraction as thermal noise at the antenna's own physical temperature::

    T_ant = eta * T_collected + (1 - eta) * T_phys

``eta`` is the radiation efficiency. This is a *different* loss from the
``(1 - |Gamma|^2)|F|^2`` factor inside
:class:`~rheplicant.radio.instrument.noise_wave.NoiseWaveOperator`: that one is
the impedance MISMATCH at the antenna-receiver interface, this one is
dissipation INSIDE the antenna. They multiply, and neither substitutes for the
other. An efficiency folded into the noise-wave couplings would be
indistinguishable from a mismatch in the fit while being wrong about the added
``(1 - eta) T_phys`` term, which a mismatch does not produce.

Placement (graph v1.3). The node sits on the trunk between ``t_ant_sum`` and
the ``receiver_input`` switch, and the position is the physics:

* AFTER ``t_ant_sum``, and applying to the WHOLE sum — sky, ground spill,
  atmospheric emission, everything the beam collected passes through the same
  lossy conductor. This is exactly the argument that D13 found *false* for the
  atmosphere (opacity must not attenuate ground pickup, which never crosses the
  atmosphere) and that holds here for the opposite reason: ohmic loss acts
  after collection, so provenance no longer matters.
* BEFORE ``receiver_input`` — the calibration loads connect at the receiver
  input, downstream of the antenna, so they never see this loss. Putting it
  after the switch would attenuate the loads too and bias every noise-wave
  solution that uses them.

``efficiency`` is physically in ``[0, 1]`` and ``t_physical`` is an ambient
temperature in kelvin, but neither is range-checked: both are differentiable
leaves that a calibrator may hold as tracers, so a value check would either fail
under ``jit`` or be skipped there — the two failure modes this codebase refuses.
Shapes are checked; values are the caller's to keep physical.
"""

from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp

from rheplicant.core.errors import StateValidationError
from rheplicant.core.operator import AbstractOperator
from rheplicant.core.state import State


class AntennaLossOperator(AbstractOperator):
    """Attenuate by the radiation efficiency and add the antenna's own emission.

    Attributes:
        efficiency: radiation efficiency ``eta``; differentiable scalar or
            ``(n_freq,)`` spectrum. ``1.0`` is a lossless antenna and makes the
            operator an exact identity.
        t_physical: the antenna structure's physical temperature [K];
            differentiable scalar or ``(n_freq,)``.
    """

    requires: ClassVar[tuple[str, ...]] = ("data",)
    provides: ClassVar[tuple[str, ...]] = ("data",)
    graph_node: ClassVar[str] = "antenna_loss"

    efficiency: jax.Array = eqx.field(converter=jnp.asarray)
    t_physical: jax.Array = eqx.field(converter=jnp.asarray)

    def __check_init__(self):
        for name, value in (("efficiency", self.efficiency),
                            ("t_physical", self.t_physical)):
            if value.ndim > 1:
                raise StateValidationError(
                    f"{name} must be scalar or (n_freq,), got ndim={value.ndim}."
                )
        if (
            self.efficiency.ndim == 1
            and self.t_physical.ndim == 1
            and self.efficiency.shape != self.t_physical.shape
        ):
            raise StateValidationError(
                f"efficiency has {self.efficiency.shape[0]} channels but "
                f"t_physical has {self.t_physical.shape[0]}."
            )

    def _check_channels(self, name: str, value: jax.Array, n_freq: int) -> None:
        if value.ndim == 1 and value.shape[0] != n_freq:
            raise StateValidationError(
                f"{name} has {value.shape[0]} channels but data has {n_freq}. "
                "Broadcasting would silently apply the wrong loss per channel."
            )

    def __call__(self, state: State) -> State:
        if state.data is None or jnp.asarray(state.data).ndim != 2:
            got = None if state.data is None else jnp.asarray(state.data).shape
            raise StateValidationError(
                f"AntennaLossOperator expects (n_time, n_freq) data; got {got}."
            )
        n_freq = state.data.shape[1]
        self._check_channels("efficiency", self.efficiency, n_freq)
        self._check_channels("t_physical", self.t_physical, n_freq)
        return state.with_data(
            self.efficiency * state.data + (1.0 - self.efficiency) * self.t_physical
        )
