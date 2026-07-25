"""Environment: traced environmental telemetry riding along with a State.

These fields typically do NOT enter the forward model — they are preserved for
diagnostics, correlation studies, sensitivity analysis and reproducibility.
Because they are traced pytree leaves (not static), they can also be promoted
into the forward model later (e.g. temperature-dependent gain) without any
structural change.

Non-numeric descriptors (weather strings, site names...) belong in
``State.meta``, not here.
"""

import dataclasses
from typing import Any

import equinox as eqx
import jax

from rheplicant.core.coordinates import as_array_or_none
from rheplicant.core.errors import StateValidationError


class Environment(eqx.Module):
    """Numeric environmental telemetry.

    Attributes:
        temperature: ambient temperature(s) [K] — scalar or ``(n_time,)``.
        humidity: relative humidity — scalar or ``(n_time,)``.
        extra: dict of additional *traced* telemetry arrays
            (wind speed, soil moisture, receiver enclosure temperature, ...).
    """

    temperature: jax.Array | None = eqx.field(default=None, converter=as_array_or_none)
    humidity: jax.Array | None = eqx.field(default=None, converter=as_array_or_none)
    extra: dict[str, Any] = eqx.field(default_factory=dict, converter=dict)

    def __check_init__(self):
        if not all(isinstance(k, str) for k in self.extra):
            raise StateValidationError("env.extra keys must be strings")

    def replace(self, **changes: Any) -> "Environment":
        """Functional update: return a new Environment with ``changes`` applied."""
        return dataclasses.replace(self, **changes)
