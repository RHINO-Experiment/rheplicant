"""Small recursive immutable containers for bootstrap records."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType

from _rheplicant_bootstrap.errors import ConfigError

_EVIDENCE_DEPTH_LIMIT = 100
_EVIDENCE_NODE_LIMIT = 250_000
_SCALAR_TYPES = (type(None), bool, int, float, str, bytes)


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
    completed: dict[int, object] = {}
    seen: set[int] = set()
    current_type = type(value).__name__

    def canonical_scalar(item: object) -> object | None:
        item_type = type(item)
        if item_type in _SCALAR_TYPES:
            return item
        if isinstance(item, str):
            return str.__str__(item)
        if isinstance(item, bool):
            return bool(item)
        if isinstance(item, int):
            return int(item)
        if isinstance(item, float):
            return float(item)
        if isinstance(item, bytes):
            return bytes(item)
        if isinstance(item, bytearray | memoryview):
            return bytes(item)
        return None

    def register(item: object, depth: int) -> None:
        nonlocal current_type
        current_type = type(item).__name__
        if depth > _EVIDENCE_DEPTH_LIMIT:
            raise ConfigError(
                f"{where}: evidence depth {depth} exceeds limit "
                f"{_EVIDENCE_DEPTH_LIMIT} at type {current_type}."
            )
        identity = id(item)
        if identity in seen:
            return
        seen.add(identity)
        if len(seen) > _EVIDENCE_NODE_LIMIT:
            raise ConfigError(
                f"{where}: evidence unique node count {len(seen)} exceeds limit "
                f"{_EVIDENCE_NODE_LIMIT} at type {current_type}."
            )

    def freeze_one(item: object, depth: int) -> object:
        register(item, depth)
        scalar = canonical_scalar(item)
        if scalar is not None or item is None:
            return scalar
        identity = id(item)
        if identity in completed:
            return completed[identity]
        if isinstance(item, Mapping):
            if identity in active:
                raise ConfigError(
                    f"{where}: cyclic evidence container of type "
                    f"{type(item).__name__} is not allowed."
                )
            active.add(identity)
            try:
                frozen_mapping: dict[object, object] = {}
                for key, child in item.items():
                    register(key, depth + 1)
                    frozen_key = canonical_scalar(key)
                    if frozen_key is None and key is not None:
                        raise ConfigError(
                            f"{where}: unsupported evidence mapping key type "
                            f"{type(key).__name__}."
                        )
                    try:
                        duplicate = frozen_key in frozen_mapping
                    except TypeError as exc:
                        raise ConfigError(
                            f"{where}: evidence mapping key type "
                            f"{type(key).__name__} is not hashable."
                        ) from exc
                    if duplicate:
                        raise ConfigError(
                            f"{where}: evidence mapping keys collide after freezing "
                            f"type {type(key).__name__}."
                        )
                    frozen_mapping[frozen_key] = freeze_one(child, depth + 1)
                result = MappingProxyType(frozen_mapping)
                completed[identity] = result
                return result
            finally:
                active.remove(identity)
        if isinstance(item, Sequence):
            if identity in active:
                raise ConfigError(
                    f"{where}: cyclic evidence container of type "
                    f"{type(item).__name__} is not allowed."
                )
            active.add(identity)
            try:
                result = tuple(freeze_one(child, depth + 1) for child in item)
                completed[identity] = result
                return result
            finally:
                active.remove(identity)
        raise ConfigError(
            f"{where}: unsupported evidence leaf type {type(item).__name__}."
        )

    try:
        return freeze_one(value, 1)
    except RecursionError as exc:
        raise ConfigError(
            f"{where}: evidence recursion exceeded at type {current_type}."
        ) from exc


__all__ = ["freeze", "freeze_evidence", "thaw"]
