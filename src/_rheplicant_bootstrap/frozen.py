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
    completed: dict[int, object] = {}

    def thaw_one(item: object) -> object:
        identity = id(item)
        if isinstance(item, Mapping):
            if identity in completed:
                return completed[identity]
            result: dict[object, object] = {}
            completed[identity] = result
            result.update(
                (thaw_one(key), thaw_one(child))
                for key, child in item.items()
            )
            return result
        if isinstance(item, tuple):
            if tuple.__len__(item) == 0:
                return []
            if identity in completed:
                return completed[identity]
            sequence: list[object] = []
            completed[identity] = sequence
            sequence.extend(thaw_one(child) for child in item)
            return sequence
        return item

    return thaw_one(value)


def freeze_evidence(value: object, *, where: str) -> object:
    """Detach and strictly freeze one tree safe to retain as audit evidence."""
    active: set[int] = set()
    completed: dict[int, tuple[object, int, str]] = {}
    seen: set[int] = set()
    current_type = type(value).__name__
    unsupported = object()

    def canonical_scalar(item: object) -> object:
        item_type = type(item)
        if item_type in _SCALAR_TYPES:
            return item
        if isinstance(item, str):
            return str.__str__(item)
        if isinstance(item, int):
            return int.__int__(item)
        if isinstance(item, float):
            return float.__float__(item)
        if isinstance(item, bytes):
            return bytes.__bytes__(item)
        if isinstance(item, bytearray):
            copied = bytearray.__getitem__(item, slice(None))
            return bytes.__new__(bytes, copied)
        if isinstance(item, memoryview):
            return memoryview.tobytes(item)
        return unsupported

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

    def freeze_one(item: object, depth: int) -> tuple[object, int, str]:
        register(item, depth)
        scalar = canonical_scalar(item)
        if scalar is not unsupported:
            return scalar, 1, type(item).__name__
        identity = id(item)
        if identity in completed:
            result, height, deepest_type = completed[identity]
            deepest_depth = depth + height - 1
            if deepest_depth > _EVIDENCE_DEPTH_LIMIT:
                raise ConfigError(
                    f"{where}: evidence depth {deepest_depth} exceeds limit "
                    f"{_EVIDENCE_DEPTH_LIMIT} at type {deepest_type}."
                )
            return result, height, deepest_type
        if isinstance(item, Mapping):
            if identity in active:
                raise ConfigError(
                    f"{where}: cyclic evidence container of type "
                    f"{type(item).__name__} is not allowed."
                )
            active.add(identity)
            try:
                frozen_mapping: dict[str, object] = {}
                maximum_child_height = 0
                deepest_type = type(item).__name__
                for key, child in item.items():
                    register(key, depth + 1)
                    if not isinstance(key, str):
                        raise ConfigError(
                            f"{where}: unsupported evidence mapping key type "
                            f"{type(key).__name__}."
                        )
                    frozen_key = str.__str__(key)
                    if frozen_key in frozen_mapping:
                        raise ConfigError(
                            f"{where}: evidence mapping keys collide after freezing "
                            f"type {type(key).__name__}."
                        )
                    frozen_child, child_height, child_deepest_type = freeze_one(
                        child, depth + 1
                    )
                    frozen_mapping[frozen_key] = frozen_child
                    if child_height > maximum_child_height:
                        maximum_child_height = child_height
                        deepest_type = child_deepest_type
                result = MappingProxyType(frozen_mapping)
                height = 1 + maximum_child_height
                record = (result, height, deepest_type)
                completed[identity] = record
                return record
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
                frozen_children: list[object] = []
                maximum_child_height = 0
                deepest_type = type(item).__name__
                for child in item:
                    frozen_child, child_height, child_deepest_type = freeze_one(
                        child, depth + 1
                    )
                    frozen_children.append(frozen_child)
                    if child_height > maximum_child_height:
                        maximum_child_height = child_height
                        deepest_type = child_deepest_type
                result = tuple(frozen_children)
                height = 1 + maximum_child_height
                record = (result, height, deepest_type)
                completed[identity] = record
                return record
            finally:
                active.remove(identity)
        raise ConfigError(
            f"{where}: unsupported evidence leaf type {type(item).__name__}."
        )

    try:
        return freeze_one(value, 1)[0]
    except ConfigError:
        raise
    except RecursionError as exc:
        raise ConfigError(
            f"{where}: evidence recursion exceeded at type {current_type}."
        ) from exc
    except Exception:
        raise ConfigError(
            f"{where}: evidence protocol failed at type {current_type}."
        ) from None


__all__ = ["freeze", "freeze_evidence", "thaw"]
