"""Small recursive immutable containers for bootstrap records."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType

from _rheplicant_bootstrap.errors import ConfigError


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


def freeze_evidence(value: object, *, where: str) -> object:
    """Detach and strictly freeze one tree safe to retain as audit evidence."""
    active: set[int] = set()

    def freeze_one(item: object) -> object:
        if item is None or isinstance(item, bool | int | float | str | bytes):
            return item
        if isinstance(item, bytearray | memoryview):
            return bytes(item)
        if isinstance(item, Mapping):
            identity = id(item)
            if identity in active:
                raise ConfigError(f"{where}: cyclic evidence mapping is not allowed.")
            active.add(identity)
            try:
                frozen_mapping: dict[object, object] = {}
                for key, child in item.items():
                    frozen_key = freeze_one(key)
                    try:
                        duplicate = frozen_key in frozen_mapping
                    except TypeError as exc:
                        raise ConfigError(
                            f"{where}: evidence mapping key {key!r} is not hashable."
                        ) from exc
                    if duplicate:
                        raise ConfigError(
                            f"{where}: evidence mapping keys collide after freezing: "
                            f"{key!r}."
                        )
                    frozen_mapping[frozen_key] = freeze_one(child)
                return MappingProxyType(frozen_mapping)
            finally:
                active.remove(identity)
        if isinstance(item, Sequence):
            identity = id(item)
            if identity in active:
                raise ConfigError(f"{where}: cyclic evidence sequence is not allowed.")
            active.add(identity)
            try:
                return tuple(freeze_one(child) for child in item)
            finally:
                active.remove(identity)
        raise ConfigError(
            f"{where}: unsupported evidence leaf {type(item).__name__} ({item!r})."
        )

    return freeze_one(value)


__all__ = ["freeze", "freeze_evidence", "thaw"]
