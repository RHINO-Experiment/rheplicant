"""Small recursive immutable containers for bootstrap records."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType


def freeze(value: object) -> object:
    """Copy nested mappings and sequences into immutable built-in containers."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: freeze(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(freeze(item) for item in value)
    return value


def thaw(value: object) -> object:
    """Copy frozen bootstrap containers back into independently mutable values."""
    if isinstance(value, Mapping):
        return {key: thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw(item) for item in value]
    return value
