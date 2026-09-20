"""Reading an object the caller supplied, without trusting any of it.

``merge_extends`` is handed arbitrary mappings, and every question it wants to
ask -- what type is this, is it a mapping, what are its items, have I seen it
before -- can be answered by a lie if it is asked through the object's own
protocol. So it is asked through ``type``, through the raw MRO descriptor, and
through iteration that treats every step as able to raise.

These are the primitives. :mod:`_rheplicant_bootstrap.layering_ownership` and
:mod:`_rheplicant_bootstrap.layering_copy` both need them, and they live here
so that neither has to import the other (A1, section 3.2).
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from types import (
    BuiltinFunctionType,
    CodeType,
    FunctionType,
    MappingProxyType,
)
from weakref import ReferenceType

from _rheplicant_bootstrap.errors import ConfigError


def _mapping_pairs(mapping: Mapping, *, failure: str):
    try:
        pairs = mapping.items()
    except Exception:
        raise ConfigError(failure) from None
    try:
        iterator = iter(pairs)
    except Exception:
        raise ConfigError(failure) from None
    while True:
        try:
            pair = next(iterator)
        except StopIteration:
            return
        except Exception:
            raise ConfigError(failure) from None
        try:
            key, value = pair
        except Exception:
            raise ConfigError(failure) from None
        yield key, value


_COMPATIBILITY_FAILURE = "merge_extends: compatibility traversal or deepcopy failed."


_COMPATIBILITY_ATOMIC_TYPES = (
    type(None),
    bool,
    int,
    float,
    complex,
    str,
    bytes,
    bytearray,
    memoryview,
)


_COMPATIBILITY_OWNERSHIP_SCALAR_TYPES = (
    type(None),
    type(Ellipsis),
    type(NotImplemented),
    bool,
    int,
    float,
    complex,
    str,
    bytes,
    CodeType,
    type,
    range,
)


_COMPATIBILITY_OWNERSHIP_COPY_ATOMIC_TYPES = (
    *_COMPATIBILITY_OWNERSHIP_SCALAR_TYPES,
    FunctionType,
    BuiltinFunctionType,
    ReferenceType,
    property,
)


def _compatibility_has_exact_type(value: object, candidates: tuple[type, ...]) -> bool:
    value_type = type(value)
    return any(value_type is candidate for candidate in candidates)


_COMPATIBILITY_MISSING = object()


_COMPATIBILITY_UNSAFE_LAZY_ANNOTATIONS = object()


def _compatibility_raw_mro_descriptor(value_type: type, name: str) -> object:
    bases = type.__getattribute__(value_type, "__mro__")
    for base in bases:
        namespace = type.__getattribute__(base, "__dict__")
        try:
            return namespace[name]
        except KeyError:
            continue
    return _COMPATIBILITY_MISSING


def _compatibility_has_mro_identity(value: object, candidates: tuple[type, ...]) -> bool:
    try:
        bases = type.__getattribute__(type(value), "__mro__")
    except Exception:
        raise ConfigError(_COMPATIBILITY_FAILURE) from None
    return any(base is candidate for base in bases for candidate in candidates)


def _compatibility_type_name(value: object) -> str:
    try:
        name = type.__getattribute__(type(value), "__name__")
        if type(name) is not str:
            raise TypeError
        return name
    except Exception:
        raise ConfigError(_COMPATIBILITY_FAILURE) from None


def _compatibility_has_mro_base(value: object, candidates: tuple[type, ...]) -> bool:
    value_type = type(value)
    if _compatibility_has_mro_identity(value, candidates):
        return True
    try:
        for name in ("__getattribute__", "__class__"):
            resolved = _compatibility_raw_mro_descriptor(value_type, name)
            expected = _compatibility_raw_mro_descriptor(object, name)
            if resolved is not expected:
                return False
        metaclass = type(value_type)
        for name in ("__eq__", "__hash__"):
            resolved = _compatibility_raw_mro_descriptor(metaclass, name)
            expected = _compatibility_raw_mro_descriptor(object, name)
            if resolved is not expected:
                return False
        return isinstance(value, candidates)
    except Exception:
        raise ConfigError(_COMPATIBILITY_FAILURE) from None


def _compatibility_is_mapping(value: object) -> bool:
    if type(value) is MappingProxyType:
        return True
    return _compatibility_has_mro_base(value, (dict, Mapping))


def _compatibility_is_mutable_mapping(value: object) -> bool:
    return _compatibility_has_mro_base(value, (dict, MutableMapping))


def _compatibility_pairs(
    mapping: Mapping,
) -> dict[str, object]:
    canonical: dict[str, object] = {}
    for key, value in _mapping_pairs(mapping, failure=_COMPATIBILITY_FAILURE):
        if not _compatibility_has_mro_identity(key, (str,)):
            raise ConfigError(
                f"merge_extends: keys are strings; got {_compatibility_type_name(key)}."
            )
        exact_key = str.__str__(key)
        if exact_key in canonical:
            raise ConfigError("merge_extends: keys collide after canonicalization.")
        canonical[exact_key] = value
    return canonical


def _compatibility_add_identity(buckets: dict[int, list[object]], item: object) -> bool:
    identity = id(item)
    bucket = buckets.setdefault(identity, [])
    if any(source is item for source in bucket):
        return False
    bucket.append(item)
    return True


def _compatibility_has_identity(buckets: dict[int, list[object]], item: object) -> bool:
    return any(source is item for source in buckets.get(id(item), ()))


def _compatibility_increment_occurrence(
    occurrences: dict[int, list[tuple[object, int]]], item: object
) -> None:
    identity = id(item)
    bucket = occurrences.setdefault(identity, [])
    for index, (source, count) in enumerate(bucket):
        if source is item:
            bucket[index] = (source, count + 1)
            return
    bucket.append((item, 1))


def _compatibility_occurrence_count(
    occurrences: dict[int, list[tuple[object, int]]], item: object
) -> int:
    for source, count in occurrences.get(id(item), ()):
        if source is item:
            return count
    return 0


def _compatibility_iter_values(container: object):
    if _compatibility_is_mapping(container):
        yield from _compatibility_pairs(container).values()
        return
    try:
        if _compatibility_has_mro_identity(container, (list,)):
            iterator = list.__iter__(container)
        elif _compatibility_has_mro_identity(container, (tuple,)):
            iterator = tuple.__iter__(container)
        elif _compatibility_has_mro_identity(container, (set,)):
            iterator = set.__iter__(container)
        elif _compatibility_has_mro_identity(container, (frozenset,)):
            iterator = frozenset.__iter__(container)
        else:
            raise TypeError
    except Exception:
        raise ConfigError(_COMPATIBILITY_FAILURE) from None
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            return
        except Exception:
            raise ConfigError(_COMPATIBILITY_FAILURE) from None
