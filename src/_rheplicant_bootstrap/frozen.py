"""Small recursive immutable containers for bootstrap records."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import cast

from _rheplicant_bootstrap.errors import ConfigError

_EVIDENCE_DEPTH_LIMIT = 100
_EVIDENCE_NODE_LIMIT = 250_000
_EVIDENCE_EDGE_LIMIT = 250_000
_SCALAR_TYPES = (type(None), bool, int, float, str, bytes)


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class _FrozenConcat(Sequence[object]):
    """Private persistent frozen sequence with a shared prefix."""

    _parent: tuple[object, ...] | _FrozenConcat
    _suffix: tuple[object, ...]
    _length: int

    def __init__(
        self,
        parent: tuple[object, ...] | _FrozenConcat,
        suffix: tuple[object, ...],
    ) -> None:
        if type(parent) is not tuple and type(parent) is not _FrozenConcat:
            raise ConfigError(
                "frozen concat parent must be an exact tuple or concat."
            )
        if type(suffix) is not tuple:
            raise ConfigError("frozen concat suffix must be an exact tuple.")
        object.__setattr__(self, "_parent", parent)
        object.__setattr__(self, "_suffix", suffix)
        object.__setattr__(self, "_length", len(parent) + len(suffix))

    def extend(self, suffix: tuple[object, ...]) -> _FrozenConcat:
        if type(suffix) is tuple and not suffix:
            return self
        return _FrozenConcat(self, suffix)

    def __len__(self) -> int:
        return self._length

    def __iter__(self):
        pending: list[tuple[object, ...] | _FrozenConcat] = [self]
        while pending:
            current = pending.pop()
            if type(current) is _FrozenConcat:
                pending.append(current._suffix)
                pending.append(current._parent)
                continue
            yield from tuple.__iter__(current)

    def __getitem__(self, index: int | slice):
        if type(index) is slice:
            if any(
                bound is not None and type(bound) is not int
                for bound in (index.start, index.stop, index.step)
            ):
                raise TypeError(
                    "frozen concat slice bounds must be exact int or null."
                )
            return tuple(self)[index]
        if type(index) is not int:
            raise TypeError(
                "frozen concat indices must be exact int or slice."
            )
        position = index
        if position < 0:
            position += self._length
        if position < 0 or position >= self._length:
            raise IndexError("tuple index out of range")
        current: tuple[object, ...] | _FrozenConcat = self
        while type(current) is _FrozenConcat:
            parent_length = len(current._parent)
            if position < parent_length:
                current = current._parent
                continue
            return current._suffix[position - parent_length]
        return tuple.__getitem__(cast(tuple[object, ...], current), position)

    def __eq__(self, other: object) -> bool:
        if type(other) is not tuple and type(other) is not _FrozenConcat:
            return NotImplemented
        if len(self) != len(other):
            return False
        return all(left == right for left, right in zip(self, other, strict=True))

    __hash__ = None  # type: ignore[assignment]


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
        if type(item) is _FrozenConcat:
            cached = completed_value(item)
            if cached is not None:
                return cached
            sequence: list[object] = []
            remember(item, sequence)
            sequence.extend(thaw_one(child) for child in item)
            return sequence
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
    edge_count = 0
    current_type = type(value).__name__
    unsupported = object()

    def canonical_scalar(item: object) -> object:
        item_type = type(item)
        if any(item_type is scalar_type for scalar_type in _SCALAR_TYPES):
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
            charge_edge(container_type)
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
                child = next(iterator)
            except StopIteration:
                return
            except Exception:
                raise ConfigError(
                    f"{where}: evidence protocol failed at type {container_type}."
                ) from None
            charge_edge(container_type)
            yield child

    def charge_edge(container_type: str) -> None:
        nonlocal edge_count
        edge_count += 1
        if edge_count > _EVIDENCE_EDGE_LIMIT:
            raise ConfigError(
                f"{where}: evidence protocol emission count {edge_count} "
                f"exceeds limit {_EVIDENCE_EDGE_LIMIT} at type {container_type}."
            )

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
                removed = active_bucket.pop()
                assert removed is item
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
                removed = active_bucket.pop()
                assert removed is item
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
