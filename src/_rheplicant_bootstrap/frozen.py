"""Small recursive immutable containers for bootstrap records."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import (
    BuiltinFunctionType,
    ClassMethodDescriptorType,
    FunctionType,
    MappingProxyType,
    MethodDescriptorType,
    WrapperDescriptorType,
)
from typing import cast

from _rheplicant_bootstrap.errors import ConfigError

_EVIDENCE_DEPTH_LIMIT = 100
_EVIDENCE_NODE_LIMIT = 250_000
_EVIDENCE_EDGE_LIMIT = 250_000
_SCALAR_TYPES = (type(None), bool, int, float, str, bytes)
_STATIC_CLASS_TEXT_LIMIT = 256
_STATIC_MISSING = object()
_TRUSTED_PROTOCOL_MEMBER_TYPES = (
    BuiltinFunctionType,
    ClassMethodDescriptorType,
    FunctionType,
    MethodDescriptorType,
    WrapperDescriptorType,
)
_CLASSMETHOD_FUNCTION_DESCRIPTOR = classmethod.__dict__["__func__"]
_STATICMETHOD_FUNCTION_DESCRIPTOR = staticmethod.__dict__["__func__"]
_TYPE_TEXT_DESCRIPTORS = {
    field: type.__dict__[field]
    for field in ("__name__", "__module__", "__qualname__")
}
_TYPE_MRO_DESCRIPTOR = type.__dict__["__mro__"]
_TYPE_NAMESPACE_DESCRIPTOR = type.__dict__["__dict__"]


class _EvidenceRoots:
    __slots__ = (
        "consume",
        "integer_bit_limit",
        "json_only",
        "text_limit",
        "values",
    )

    def __init__(
        self,
        values: list[object],
        text_limit: int | None,
        json_only: bool,
        integer_bit_limit: int | None,
        consume: Callable[[], None] | None,
    ) -> None:
        self.values = values
        self.text_limit = text_limit
        self.json_only = json_only
        self.integer_bit_limit = integer_bit_limit
        self.consume = consume


def _require_utf8_text(value: str, *, where: str) -> str:
    try:
        str.encode(value, "utf-8", "strict")
    except UnicodeEncodeError:
        raise ConfigError(
            f"{where}: evidence text must contain only valid UTF-8."
        ) from None
    return value


def _require_utf8_text_limited(
    value: str,
    *,
    where: str,
    byte_limit: int,
) -> str:
    if str.__len__(value) > byte_limit:
        raise ConfigError(
            f"{where} scalar exceeds the {byte_limit}-byte limit."
        )
    try:
        encoded = str.encode(value, "utf-8", "strict")
    except UnicodeEncodeError:
        raise ConfigError(
            f"{where}: evidence text must contain only valid UTF-8."
        ) from None
    if bytes.__len__(encoded) > byte_limit:
        raise ConfigError(
            f"{where} scalar exceeds the {byte_limit}-byte limit."
        )
    return value


def static_class_text(
    actual_type: type,
    field: str,
    *,
    fallback: str,
) -> str:
    """Read one class text field through ``type``'s trusted descriptor."""
    descriptor = _TYPE_TEXT_DESCRIPTORS.get(field)
    if descriptor is None:
        return fallback
    try:
        value = descriptor.__get__(actual_type, type(actual_type))
    except Exception:
        return fallback
    if (
        type(value) is not str
        or not value
        or str.__len__(value) > _STATIC_CLASS_TEXT_LIMIT
    ):
        return fallback
    try:
        str.encode(value, "utf-8", "strict")
    except UnicodeEncodeError:
        return fallback
    return value


def static_type_name(value: object) -> str:
    """Return the real runtime type name without binding metaclass hooks."""
    return static_class_text(type(value), "__name__", fallback="unknown")


def static_class_mro(value: object) -> tuple[type, ...]:
    """Return a class object's real MRO, or empty for a non-class value."""
    try:
        result = _TYPE_MRO_DESCRIPTOR.__get__(value, type(value))
    except Exception:
        return ()
    if type(result) is not tuple:
        return ()
    return result


def _static_protocol_member(actual_type: type, name: str) -> bool | None:
    """Classify one protocol member without binding foreign descriptors."""
    member = static_class_attribute(actual_type, name, _STATIC_MISSING)
    if member is _STATIC_MISSING:
        return None
    member_type = type(member)
    if any(
        member_type is trusted_type
        for trusted_type in _TRUSTED_PROTOCOL_MEMBER_TYPES
    ):
        return True
    if member_type is classmethod or member_type is staticmethod:
        descriptor = (
            _CLASSMETHOD_FUNCTION_DESCRIPTOR
            if member_type is classmethod
            else _STATICMETHOD_FUNCTION_DESCRIPTOR
        )
        try:
            wrapped = descriptor.__get__(member, member_type)
        except Exception:
            return False
        return (
            type(wrapped) is FunctionType
            or type(wrapped) is BuiltinFunctionType
        )
    if (
        static_class_attribute(member_type, "__get__", _STATIC_MISSING)
        is not _STATIC_MISSING
    ):
        return False
    return callable(member)


def _static_mapping_protocol(actual_type: type) -> bool:
    return _static_protocol_member(actual_type, "items") is True


def _static_sequence_protocol(actual_type: type) -> bool:
    if _static_protocol_member(actual_type, "__getitem__") is not True:
        return False
    if _static_protocol_member(actual_type, "__len__") is not True:
        return False
    iterator = _static_protocol_member(actual_type, "__iter__")
    return iterator is None or iterator


def static_isinstance(
    value: object,
    expected: type | tuple[type, ...],
) -> bool:
    """Check nominal scalars or static container protocol shape."""
    actual_type = type(value)
    actual_mro = static_class_mro(actual_type)
    candidates = expected if type(expected) is tuple else (expected,)
    for candidate in tuple.__iter__(candidates):
        if candidate is Mapping:
            if _static_mapping_protocol(actual_type):
                return True
            continue
        if candidate is Sequence:
            nominal_sequence = any(
                base is Sequence for base in actual_mro
            )
            if nominal_sequence:
                if _static_sequence_protocol(actual_type):
                    return True
                continue
            if _static_mapping_protocol(actual_type):
                continue
            if _static_sequence_protocol(actual_type):
                return True
            continue
        if any(base is candidate for base in actual_mro):
            return True
    return False


def static_class_attribute(
    actual_type: type,
    name: str,
    default: object = None,
) -> object:
    """Read an unbound class/MRO namespace entry without metaclass lookup."""
    for base in static_class_mro(actual_type):
        try:
            namespace = _TYPE_NAMESPACE_DESCRIPTOR.__get__(base, type(base))
        except Exception:
            return default
        if name in namespace:
            return namespace[name]
    return default


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
    if static_isinstance(value, Mapping):
        return MappingProxyType({key: freeze(item) for key, item in value.items()})
    if static_isinstance(value, (list, tuple)):
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
        if static_isinstance(item, Mapping):
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
        if static_isinstance(item, tuple):
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
    if type(value) is _EvidenceRoots:
        root_values = object.__getattribute__(value, "values")
        text_limit = object.__getattribute__(value, "text_limit")
        json_only = object.__getattribute__(value, "json_only")
        integer_bit_limit = object.__getattribute__(
            value, "integer_bit_limit"
        )
        shared_consume = object.__getattribute__(value, "consume")
    else:
        root_values = None
        text_limit = None
        json_only = False
        integer_bit_limit = None
        shared_consume = None
    active: dict[int, list[object]] = {}
    completed: dict[int, list[tuple[object, object, int, str]]] = {}
    seen: dict[int, list[object]] = {}
    node_count = 0
    edge_count = 0
    current_type = static_type_name(value)
    unsupported = object()

    def canonical_scalar(item: object) -> object:
        if json_only and static_isinstance(
            item, (bytes, bytearray, memoryview)
        ):
            raise ConfigError(f"{where} contains a value that is not JSON.")
        if (
            json_only
            and static_isinstance(item, int)
            and not static_isinstance(item, bool)
            and integer_bit_limit is not None
            and int.bit_length(item) > integer_bit_limit
        ):
            raise ConfigError(
                f"{where} integer exceeds the {integer_bit_limit}-bit limit."
            )
        item_type = type(item)
        if item_type is str:
            text = cast(str, item)
            if text_limit is not None:
                return _require_utf8_text_limited(
                    text,
                    where=where,
                    byte_limit=text_limit,
                )
            return _require_utf8_text(text, where=where)
        if any(item_type is scalar_type for scalar_type in _SCALAR_TYPES):
            return item
        if static_isinstance(item, str):
            if (
                text_limit is not None
                and str.__len__(item) > text_limit
            ):
                raise ConfigError(
                    f"{where} scalar exceeds the {text_limit}-byte limit."
                )
            text = str.__str__(item)
            if text_limit is not None:
                return _require_utf8_text_limited(
                    text,
                    where=where,
                    byte_limit=text_limit,
                )
            return _require_utf8_text(text, where=where)
        if static_isinstance(item, int):
            return int.__int__(item)
        if static_isinstance(item, float):
            return float.__float__(item)
        if static_isinstance(item, bytes):
            return bytes.__bytes__(item)
        if static_isinstance(item, bytearray):
            copied = bytearray.__getitem__(item, slice(None))
            return bytes.__new__(bytes, copied)
        if static_isinstance(item, memoryview):
            return memoryview.tobytes(item)
        return unsupported

    def register(item: object, depth: int) -> None:
        nonlocal current_type, node_count
        current_type = static_type_name(item)
        if depth > _EVIDENCE_DEPTH_LIMIT:
            raise ConfigError(
                f"{where}: evidence depth {depth} exceeds limit "
                f"{_EVIDENCE_DEPTH_LIMIT} at type {current_type}."
            )
        identity = id(item)
        bucket = seen.setdefault(identity, [])
        if any(retained is item for retained in bucket):
            return
        if shared_consume is not None:
            shared_consume()
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
        if shared_consume is not None:
            shared_consume()
        edge_count += 1
        if edge_count > _EVIDENCE_EDGE_LIMIT:
            raise ConfigError(
                f"{where}: evidence protocol emission count {edge_count} "
                f"exceeds limit {_EVIDENCE_EDGE_LIMIT} at type {container_type}."
            )

    def freeze_one(item: object, depth: int) -> tuple[object, int, str]:
        register(item, depth)
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
        scalar = canonical_scalar(item)
        if scalar is not unsupported:
            scalar_type = static_type_name(item)
            completed.setdefault(identity, []).append(
                (item, scalar, 1, scalar_type)
            )
            return scalar, 1, scalar_type
        if static_isinstance(item, Mapping):
            active_bucket = active.setdefault(identity, [])
            if any(retained is item for retained in active_bucket):
                raise ConfigError(
                    f"{where}: cyclic evidence container of type "
                    f"{static_type_name(item)} is not allowed."
                )
            active_bucket.append(item)
            try:
                frozen_mapping: dict[str, object] = {}
                maximum_child_height = 0
                deepest_type = static_type_name(item)
                container_type = static_type_name(item)
                for key, child in mapping_pairs(item, container_type):
                    if not static_isinstance(key, str):
                        register(key, depth + 1)
                        raise ConfigError(
                            f"{where}: unsupported evidence mapping key type "
                            f"{static_type_name(key)}."
                        )
                    frozen_key_value, _, _ = freeze_one(key, depth + 1)
                    if type(frozen_key_value) is not str:
                        raise ConfigError(
                            f"{where}: unsupported evidence mapping key type "
                            f"{static_type_name(key)}."
                        )
                    frozen_key = cast(str, frozen_key_value)
                    if frozen_key in frozen_mapping:
                        raise ConfigError(
                            f"{where}: evidence mapping keys collide after freezing "
                            f"type {static_type_name(key)}."
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
        if static_isinstance(item, Sequence):
            active_bucket = active.setdefault(identity, [])
            if any(retained is item for retained in active_bucket):
                raise ConfigError(
                    f"{where}: cyclic evidence container of type "
                    f"{static_type_name(item)} is not allowed."
                )
            active_bucket.append(item)
            try:
                frozen_children: list[object] = []
                maximum_child_height = 0
                deepest_type = static_type_name(item)
                container_type = static_type_name(item)
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
            f"{where}: unsupported evidence leaf type {static_type_name(item)}."
        )

    try:
        if root_values is not None:
            if list.__len__(root_values) > _EVIDENCE_EDGE_LIMIT:
                raise ConfigError(
                    f"{where}: evidence root count exceeds limit "
                    f"{_EVIDENCE_EDGE_LIMIT}."
                )
            frozen_roots: list[object] = []
            for root in list.__iter__(root_values):
                frozen_roots.append(freeze_one(root, 1)[0])
            return tuple(frozen_roots)
        return freeze_one(value, 1)[0]
    except ConfigError:
        raise
    except RecursionError:
        raise ConfigError(
            f"{where}: evidence recursion exceeded at type {current_type}."
        ) from None
    except Exception:
        raise ConfigError(
            f"{where}: evidence protocol failed at type {current_type}."
        ) from None


def _freeze_evidence_roots(
    values: list[object],
    *,
    where: str,
    text_limit: int | None = None,
    json_only: bool = False,
    integer_bit_limit: int | None = None,
    consume: Callable[[], None] | None = None,
) -> tuple[object, ...]:
    """Freeze trusted root references with shared identity and no depth wrapper."""
    if type(values) is not list:
        raise ConfigError(f"{where}: evidence roots must be an exact list.")
    if text_limit is not None and (
        type(text_limit) is not int or text_limit < 0
    ):
        raise ConfigError(f"{where}: evidence text limit is invalid.")
    if type(json_only) is not bool:
        raise ConfigError(f"{where}: JSON-only evidence policy is invalid.")
    if integer_bit_limit is not None and (
        type(integer_bit_limit) is not int or integer_bit_limit < 0
    ):
        raise ConfigError(f"{where}: evidence integer limit is invalid.")
    if json_only and integer_bit_limit is None:
        raise ConfigError(f"{where}: JSON integer limit is required.")
    if consume is not None and not callable(consume):
        raise ConfigError(f"{where}: evidence consumer is invalid.")
    frozen = freeze_evidence(
        _EvidenceRoots(
            values,
            text_limit,
            json_only,
            integer_bit_limit,
            consume,
        ),
        where=where,
    )
    if type(frozen) is not tuple:
        raise ConfigError(f"{where}: frozen evidence roots must be an exact tuple.")
    return frozen


__all__ = [
    "freeze",
    "freeze_evidence",
    "static_class_mro",
    "static_class_attribute",
    "static_class_text",
    "static_isinstance",
    "static_type_name",
    "thaw",
]
