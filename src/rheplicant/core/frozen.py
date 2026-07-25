"""FrozenMapping: an immutable, hashable mapping for static metadata.

``State.meta`` is a *static* pytree component: it lives in the treedef, not in
the traced leaves, so under ``jax.jit`` it participates in the compilation
cache key. That requires it to be hashable and support ``==`` — which plain
``dict`` does not. FrozenMapping provides exactly that, and validates at
construction (the system boundary) that every value is itself hashable.

Rule of thumb for users: strings/labels/settings go in ``meta`` (changing them
recompiles); numbers and arrays go in ``State.aux`` / ``env`` / ``coords``
(traced, differentiable).
"""

from collections.abc import Iterable, Mapping
from typing import Any


class FrozenMapping(Mapping):
    """Immutable, hashable ``Mapping[str, Hashable]``.

    Functional updates return new instances::

        meta = FrozenMapping(telescope="RHINO")
        meta2 = meta.set(obs_id="demo-001")   # meta unchanged
        meta3 = meta2.remove("obs_id")
        merged = meta | {"band": "low"}
    """

    __slots__ = ("_items", "_hash")

    def __init__(
        self,
        data: "Mapping[str, Any] | Iterable[tuple[str, Any]] | None" = None,
        **kwargs: Any,
    ):
        if isinstance(data, FrozenMapping):
            src: dict[str, Any] = dict(data._items)
        elif data is None:
            src = {}
        else:
            src = dict(data)
        src.update(kwargs)

        for k, v in src.items():
            if not isinstance(k, str):
                raise TypeError(
                    f"FrozenMapping keys must be strings, got {type(k).__name__}: {k!r}"
                )
            try:
                hash(v)
            except TypeError:
                raise TypeError(
                    f"FrozenMapping value for {k!r} must be hashable (got {type(v).__name__}). "
                    "Static metadata must be hashable for jit-cache correctness; "
                    "put arrays or other unhashable data in State.aux / env / coords instead."
                ) from None

        object.__setattr__(self, "_items", src)
        object.__setattr__(self, "_hash", hash(frozenset(src.items())))

    # -- Mapping interface ---------------------------------------------------

    def __getitem__(self, key: str) -> Any:
        return self._items[key]

    def __iter__(self):
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    # -- hashing / equality (consistent, order-independent) ------------------

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FrozenMapping):
            return NotImplemented
        return self._items == other._items

    # -- immutability --------------------------------------------------------

    def __setattr__(self, name: str, value: Any):
        raise AttributeError("FrozenMapping is immutable")

    def __delattr__(self, name: str):
        raise AttributeError("FrozenMapping is immutable")

    # -- functional updates --------------------------------------------------

    def set(self, **kwargs: Any) -> "FrozenMapping":
        """Return a new FrozenMapping with ``kwargs`` added/overridden."""
        return FrozenMapping(self, **kwargs)

    def remove(self, key: str) -> "FrozenMapping":
        """Return a new FrozenMapping without ``key`` (KeyError if absent)."""
        items = dict(self._items)
        del items[key]
        return FrozenMapping(items)

    def __or__(self, other: "Mapping[str, Any]") -> "FrozenMapping":
        return FrozenMapping({**self._items, **dict(other)})

    def __repr__(self) -> str:
        return f"FrozenMapping({self._items!r})"
