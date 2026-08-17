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
    completed: dict[int, list[tuple[object, object]]] = {}

    def completed_value(item: object) -> object | None:
        for source, result in completed.get(id(item), ()):
            if source is item:
                return result
        return None

    def remember(item: object, result: object) -> None:
        completed.setdefault(id(item), []).append((item, result))

    def thaw_one(item: object) -> object:
        if isinstance(item, Mapping):
            cached = completed_value(item)
            if cached is not None:
                return cached
            result: dict[object, object] = {}
            remember(item, result)
            result.update(
                (thaw_one(key), thaw_one(child))
                for key, child in item.items()
            )
            return result
        if isinstance(item, tuple):
            if tuple.__len__(item) == 0:
                return []
            cached = completed_value(item)
            if cached is not None:
                return cached
            sequence: list[object] = []
            remember(item, sequence)
            sequence.extend(thaw_one(child) for child in item)
            return sequence
        return item

    return thaw_one(value)


def freeze_evidence(value: object, *, where: str) -> object:
    """Detach and strictly freeze one tree safe to retain as audit evidence."""
    active: dict[int, list[object]] = {}
    completed: dict[int, list[tuple[object, object, int, str]]] = {}
    seen: dict[int, list[object]] = {}
    node_count = 0
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
        nonlocal current_type, node_count
        current_type = type(item).__name__
        if depth > _EVIDENCE_DEPTH_LIMIT:
            raise ConfigError(
                f"{where}: evidence depth {depth} exceeds limit "
                f"{_EVIDENCE_DEPTH_LIMIT} at type {current_type}."
            )
        identity = id(item)
        bucket = seen.setdefault(identity, [])
        if any(retained is item for retained in bucket):
            return
        bucket.append(item)
        node_count += 1
        if node_count > _EVIDENCE_NODE_LIMIT:
            raise ConfigError(
                f"{where}: evidence unique node count {node_count} exceeds limit "
                f"{_EVIDENCE_NODE_LIMIT} at type {current_type}."
            )

    def mapping_pairs(item: Mapping, container_type: str):
        try:
            pairs = item.items()
        except Exception:
            raise ConfigError(
                f"{where}: evidence protocol failed at type {container_type}."
            ) from None
        try:
            iterator = iter(pairs)
        except Exception:
            raise ConfigError(
                f"{where}: evidence protocol failed at type {container_type}."
            ) from None
        while True:
            try:
                pair = next(iterator)
            except StopIteration:
                return
            except Exception:
                raise ConfigError(
                    f"{where}: evidence protocol failed at type {container_type}."
                ) from None
            try:
                key, child = pair
            except Exception:
                raise ConfigError(
                    f"{where}: evidence protocol failed at type {container_type}."
                ) from None
            yield key, child

    def sequence_items(item: Sequence, container_type: str):
        try:
            iterator = iter(item)
        except Exception:
            raise ConfigError(
                f"{where}: evidence protocol failed at type {container_type}."
            ) from None
        while True:
            try:
                yield next(iterator)
            except StopIteration:
                return
            except Exception:
                raise ConfigError(
                    f"{where}: evidence protocol failed at type {container_type}."
                ) from None

    def freeze_one(item: object, depth: int) -> tuple[object, int, str]:
        register(item, depth)
        scalar = canonical_scalar(item)
        if scalar is not unsupported:
            return scalar, 1, type(item).__name__
        identity = id(item)
        cached = next(
            (
                record
                for record in completed.get(identity, ())
                if record[0] is item
            ),
            None,
        )
        if cached is not None:
            _, result, height, deepest_type = cached
            deepest_depth = depth + height - 1
            if deepest_depth > _EVIDENCE_DEPTH_LIMIT:
                raise ConfigError(
                    f"{where}: evidence depth {deepest_depth} exceeds limit "
                    f"{_EVIDENCE_DEPTH_LIMIT} at type {deepest_type}."
                )
            return result, height, deepest_type
        if isinstance(item, Mapping):
            active_bucket = active.setdefault(identity, [])
            if any(retained is item for retained in active_bucket):
                raise ConfigError(
                    f"{where}: cyclic evidence container of type "
                    f"{type(item).__name__} is not allowed."
                )
            active_bucket.append(item)
            try:
                frozen_mapping: dict[str, object] = {}
                maximum_child_height = 0
                deepest_type = type(item).__name__
                container_type = type(item).__name__
                for key, child in mapping_pairs(item, container_type):
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
                record = (item, result, height, deepest_type)
                completed.setdefault(identity, []).append(record)
                return result, height, deepest_type
            finally:
                assert active_bucket.pop() is item
                if not active_bucket:
                    del active[identity]
        if isinstance(item, Sequence):
            active_bucket = active.setdefault(identity, [])
            if any(retained is item for retained in active_bucket):
                raise ConfigError(
                    f"{where}: cyclic evidence container of type "
                    f"{type(item).__name__} is not allowed."
                )
            active_bucket.append(item)
            try:
                frozen_children: list[object] = []
                maximum_child_height = 0
                deepest_type = type(item).__name__
                container_type = type(item).__name__
                for child in sequence_items(item, container_type):
                    frozen_child, child_height, child_deepest_type = freeze_one(
                        child, depth + 1
                    )
                    frozen_children.append(frozen_child)
                    if child_height > maximum_child_height:
                        maximum_child_height = child_height
                        deepest_type = child_deepest_type
                result = tuple(frozen_children)
                height = 1 + maximum_child_height
                record = (item, result, height, deepest_type)
                completed.setdefault(identity, []).append(record)
                return result, height, deepest_type
            finally:
                assert active_bucket.pop() is item
                if not active_bucket:
                    del active[identity]
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
