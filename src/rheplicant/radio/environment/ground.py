"""GroundPickupOperator — PLACEHOLDER ground spill-over.

Element: "Ground pickup via the beam sidelobes (depends on ambient
temperature as well)". Pain point: "a simple low-order spatial model might be
worth trying, i.e. use the existing topographic template and allow its
outline and internal structure to be modulated by relatively smooth functions
of alt/az. This will couple into the beam effects however."

Real physics to come: sidelobe-weighted topographic template with smooth
alt/az modulation (coupled to the beam model). The placeholder demonstrates
the *environment coupling contract*: the contribution is
``coupling * T_ambient`` with the ambient temperature read from the traced
``state.env`` when available — exactly why Environment is a traced (and thus
differentiable) part of State.
"""

from typing import ClassVar

import jax
import jax.numpy as jnp

from rheplicant.core.errors import StateValidationError
from rheplicant.core.operator import AbstractOperator
from rheplicant.core.state import State


class GroundPickupOperator(AbstractOperator):
    """Produce a ground-pickup contribution coupled to ambient temperature.

    The BODY is a placeholder: one scalar coupling times one temperature, with
    no dependence on where the sidelobes actually point. The contract is not.
    What is real here is the environment-coupling seam -- an ambient
    temperature read from the traced state with a declared parameter fallback,
    and a placement in the antenna-temperature sum parallel to atmospheric
    emission -- and that seam does not change when the physics arrives.

    Real physics to come: a sidelobe-weighted topographic template modulated by
    smooth functions of alt/az, which couples this operator to the beam model
    and makes ``coupling`` a field rather than a scalar.

    Contribution: ``coupling * T_amb`` where ``T_amb`` comes from
    ``state.env.temperature`` (scalar or per-time) if present, else from the
    ``t_ground`` fallback parameter.

    Attributes:
        coupling: sidelobe coupling fraction — differentiable scalar.
        t_ground: fallback ground temperature [K] — differentiable scalar.
    """

    requires: ClassVar[tuple[str, ...]] = ("coords.time", "coords.freq", "env.temperature")
    provides: ClassVar[tuple[str, ...]] = ("data",)
    graph_node: ClassVar[str] = "ground_pickup"

    coupling: jax.Array
    t_ground: jax.Array

    def __call__(self, state: State) -> State:
        if state.coords is None or state.coords.time is None or state.coords.freq is None:
            raise StateValidationError(
                "GroundPickupOperator requires state.coords with time and freq axes."
            )
        n_time = state.coords.time.shape[0]
        n_freq = state.coords.freq.shape[0]

        if state.env is not None and state.env.temperature is not None:
            t_amb = state.env.temperature
        else:
            t_amb = self.t_ground
        t_amb = jnp.atleast_1d(t_amb)
        if t_amb.ndim == 1 and t_amb.shape[0] not in (1, n_time):
            raise StateValidationError(
                f"env.temperature has {t_amb.shape[0]} samples but coords.time has {n_time}."
            )
        column = jnp.broadcast_to(t_amb, (n_time,))
        return state.with_data(self.coupling * column[:, None] * jnp.ones((1, n_freq)))
