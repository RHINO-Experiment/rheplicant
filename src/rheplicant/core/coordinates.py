"""Coordinates: the traced coordinate container flowing with a State.

All fields are optional and traced (they are pytree leaves, so they can be
jitted / vmapped / differentiated through). Structural validation (ndim only,
never values) runs at construction and again on every functional ``replace``.

Angle convention (RHINO family): degrees in public-facing APIs, radians
internally. This module stores whatever it is given — the convention is
enforced by operators, not by the container.
"""

import dataclasses
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from rheplicant.core.errors import StateValidationError


def as_array_or_none(value: Any) -> jax.Array | None:
    """Converter: pass ``None`` through, coerce everything else to a jax array."""
    return None if value is None else jnp.asarray(value)


class Coordinates(eqx.Module):
    """Coordinate axes of the data flowing through a pipeline.

    Attributes:
        time: ``(n_time,)`` sample times [seconds].
        freq: ``(n_freq,)`` channel frequencies [Hz].
        pointing: ``(n_time, k)`` pointing coordinates (e.g. alt/az pairs, k=2).
        extra: dict of additional *traced* coordinate arrays (e.g. spatial grids).
    """

    time: jax.Array | None = eqx.field(default=None, converter=as_array_or_none)
    freq: jax.Array | None = eqx.field(default=None, converter=as_array_or_none)
    pointing: jax.Array | None = eqx.field(default=None, converter=as_array_or_none)
    extra: dict[str, Any] = eqx.field(default_factory=dict, converter=dict)

    def __check_init__(self):
        # Structural (shape-rank) checks only: jit-safe, value-independent.
        if self.time is not None and self.time.ndim != 1:
            raise StateValidationError(f"coords.time must be 1D, got ndim={self.time.ndim}")
        if self.freq is not None and self.freq.ndim != 1:
            raise StateValidationError(f"coords.freq must be 1D, got ndim={self.freq.ndim}")
        if self.pointing is not None and self.pointing.ndim != 2:
            raise StateValidationError(
                f"coords.pointing must be 2D (n_time, k), got ndim={self.pointing.ndim}"
            )
        if not all(isinstance(k, str) for k in self.extra):
            raise StateValidationError("coords.extra keys must be strings")

    def replace(self, **changes: Any) -> "Coordinates":
        """Functional update: return a new Coordinates with ``changes`` applied.

        Re-runs converters and validation (unlike raw ``eqx.tree_at``).
        """
        return dataclasses.replace(self, **changes)
