"""LiveNames: a registry's key set, read at the moment it is asked for.

Four registries in this layer are filled by more than one module -- value
forms, file readers, derivations, resource kinds -- and every "unknown X"
refusal lists what is available. A module-level ``tuple(_TABLE)`` would freeze
that list at import time and quietly go short as later modules register, so
the message would name a set the loader does not actually have. This is the
same discipline ``core/graph.py:350``'s ``get_graph`` applies by listing
``list(_GRAPHS)`` inside the refusal rather than beside it.
"""

from collections.abc import Collection, Mapping
from typing import Any


class LiveNames(Collection):
    """A sorted, always-current view of a registry's keys.

    Iterating, sizing, ``in`` and ``repr`` all read the underlying mapping at
    the moment they are called, so a name registered after this object was
    built is still reported.
    """

    __slots__ = ("_table",)

    def __init__(self, table: Mapping[str, Any]):
        self._table = table

    def __iter__(self):
        return iter(sorted(self._table))

    def __len__(self) -> int:
        return len(self._table)

    def __contains__(self, name: object) -> bool:
        return name in self._table

    def __repr__(self) -> str:
        return repr(sorted(self._table))
