"""AtmosphericEmissionOperator — PLACEHOLDER beam-averaged atmospheric emission.

The atmosphere enters the graph twice (equivalent-entry pair, like ground
spill):

- ``atmosphere`` (this operator): the beam-averaged *emission* as an additive
  effective temperature — a branch of the antenna-temperature sum, parallel
  to ``ground_pickup`` and ``t_sys_extra``. It sits before the
  ``receiver_input`` switch (calibration loads do not see the sky) and before
  the noise-wave stage (atmospheric emission arrives through the antenna, so
  it suffers the ``(1-|Gamma|^2)`` reflection loss).
- ``atmosphere_field`` (reserved, no shipped operator): strict radiative
  transfer on the astro branch *before* the beam —
  ``e^(-tau sec z) T_sky + T_atm (1 - e^(-tau sec z))`` inside the beam
  integral. Opacity must act on the astro sky alone: applied after the
  antenna-temperature sum it would wrongly attenuate ground pickup, which
  never crosses the atmosphere.

Real physics to come: emission temperature from an atmospheric model
(opacity x ambient temperature, beam-weighted airmass), slowly varying in
time and frequency for a zenith-pointing drift scan.
"""

from typing import ClassVar

import jax
import jax.numpy as jnp

from rheplicant.core.errors import StateValidationError
from rheplicant.core.operator import AbstractOperator
from rheplicant.core.state import State


class AtmosphericEmissionOperator(AbstractOperator):
    """Produce the beam-averaged atmospheric emission contribution [K].

    Source-type: a branch of the antenna-temperature ``SumOperator``,
    producing its own ``(n_time, n_freq)`` contribution on the shared grid.

    Attributes:
        t_atm: emission temperature — differentiable scalar or ``(n_freq,)``.
    """

    requires: ClassVar[tuple[str, ...]] = ("coords.time", "coords.freq")
    provides: ClassVar[tuple[str, ...]] = ("data",)
    graph_node: ClassVar[str] = "atmosphere"

    t_atm: jax.Array

    def __call__(self, state: State) -> State:
        if state.coords is None or state.coords.time is None or state.coords.freq is None:
            raise StateValidationError(
                "AtmosphericEmissionOperator requires state.coords with time and freq axes."
            )
        n_time = state.coords.time.shape[0]
        n_freq = state.coords.freq.shape[0]
        row = jnp.atleast_1d(self.t_atm)
        if row.shape[-1] not in (1, n_freq):
            raise StateValidationError(
                f"t_atm has {row.shape[-1]} channels but coords.freq has {n_freq}."
            )
        return state.with_data(jnp.broadcast_to(row, (n_time, n_freq)))
