"""State: the immutable scientific context flowing through a Pipeline.

A State is a JAX pytree (an ``equinox.Module``), so it can be passed through
``jit`` / ``grad`` / ``vmap`` / ``scan`` directly. It carries *everything* a
pipeline might need — signal data, coordinates, environment telemetry,
metadata, randomness — even though not every field enters the forward model:
metadata and environment ride along for diagnostics, correlation studies,
sensitivity analysis and reproducibility.

Field taxonomy
--------------
Traced (pytree leaves — differentiable, vmappable):
    ``data``, ``coords``, ``env``, ``aux``, ``key``
Static (part of the treedef — participates in the jit cache key):
    ``meta``

Rule of thumb: strings/labels/settings go in ``meta`` (changing them triggers
recompilation, by design); numbers and arrays go in ``aux``/``env``/``coords``.

PRNG protocol
-------------
Operators that need randomness must consume it functionally::

    subkey, state = state.next_key()      # never reuse state.key directly
    noise = jax.random.normal(subkey, shape)

and return the *advanced* state, so a single seed reproduces the entire run
and keys are never reused across operators.
"""

import dataclasses
from typing import Any

import equinox as eqx
import jax

from rheplicant.core.coordinates import Coordinates
from rheplicant.core.environment import Environment
from rheplicant.core.errors import MissingKeyError, StateValidationError
from rheplicant.core.frozen import FrozenMapping


class State(eqx.Module):
    """Immutable data/state container for differentiable pipelines.

    Attributes:
        data: the signal/data payload — any pytree of arrays (a single array,
            a tuple, or a dict of named streams). Convention is set by the
            operators acting on it, not by State itself.
        coords: :class:`Coordinates` (time / freq / pointing / extra), or None.
        env: :class:`Environment` (numeric telemetry), or None.
        aux: dict of additional *traced* user arrays (weights, masks, caches).
        key: a typed JAX PRNG key (``jax.random.key(seed)``), or None.
        meta: static, hashable metadata (:class:`FrozenMapping`) — experiment
            info, hardware labels, observation IDs, arbitrary user fields.

    Example::

        state = State(
            coords=Coordinates(time=t, freq=f),
            key=jax.random.key(0),
            meta={"telescope": "RHINO", "obs_id": "demo-001"},
        )
        out = pipeline(state)
    """

    data: Any = None
    coords: Coordinates | None = None
    env: Environment | None = None
    aux: dict[str, Any] = eqx.field(default_factory=dict, converter=dict)
    key: jax.Array | None = None
    meta: FrozenMapping = eqx.field(
        static=True, default_factory=FrozenMapping, converter=FrozenMapping
    )

    def __check_init__(self):
        # Structural checks only (types, dtypes) — never traced values: jit-safe.
        if self.key is not None:
            is_array = isinstance(self.key, jax.Array)
            if not (is_array and jax.numpy.issubdtype(self.key.dtype, jax.dtypes.prng_key)):
                detail = f" with dtype {self.key.dtype}" if is_array else ""
                raise StateValidationError(
                    "State.key must be a typed PRNG key made with jax.random.key(seed) "
                    f"(got {type(self.key).__name__}{detail})."
                )
        if self.coords is not None and not isinstance(self.coords, Coordinates):
            raise StateValidationError(
                f"State.coords must be a Coordinates instance, got {type(self.coords).__name__}"
            )
        if self.env is not None and not isinstance(self.env, Environment):
            raise StateValidationError(
                f"State.env must be an Environment instance, got {type(self.env).__name__}"
            )
        if not all(isinstance(k, str) for k in self.aux):
            raise StateValidationError("State.aux keys must be strings")

    # -- functional updates --------------------------------------------------

    def replace(self, **changes: Any) -> "State":
        """Return a new State with ``changes`` applied; the original is untouched.

        Built on ``dataclasses.replace``, so converters and validation re-run
        on every update. For surgical edits deep inside nested pytrees,
        ``equinox.tree_at`` remains the right tool.
        """
        return dataclasses.replace(self, **changes)

    def with_data(self, data: Any) -> "State":
        """Shorthand for ``state.replace(data=data)`` — the most common update."""
        return self.replace(data=data)

    def checkpoint(self, name: str = "raw") -> "State":
        """Snapshot the current data into ``aux["snapshot/<name>"]``.

        Zero-copy (JAX arrays are immutable — the snapshot is a reference), so
        checkpointing raw data before destructive processing (calibration,
        filtering) costs nothing. Retrieve with
        ``state.aux["snapshot/raw"]``.

        Raises:
            StateValidationError: if there is no data to snapshot.
        """
        if self.data is None:
            raise StateValidationError(
                f"Cannot checkpoint {name!r}: state.data is None (nothing to snapshot)."
            )
        return self.replace(aux={**self.aux, f"snapshot/{name}": self.data})

    # -- PRNG protocol -------------------------------------------------------

    def next_key(self) -> tuple[jax.Array, "State"]:
        """Split the PRNG key: return ``(subkey, state_with_advanced_key)``.

        Raises:
            MissingKeyError: if this State carries no key.
        """
        if self.key is None:
            raise MissingKeyError(
                "This operator needs randomness but State.key is None; "
                "construct the state with key=jax.random.key(seed)."
            )
        subkey, carry = jax.random.split(self.key)
        return subkey, self.replace(key=carry)
