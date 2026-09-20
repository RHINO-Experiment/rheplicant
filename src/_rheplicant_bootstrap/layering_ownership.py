"""Does this object own mutable state that a deepcopy must not share?

``merge_extends`` copies its inputs and then has to prove the copy is one.
Proving it means walking an arbitrary object and deciding which of its parts
are its own mutable state and which are shared, interned or immutable -- a
function's closure cells, a descriptor's value, a weakref's referent, a class
whose ``__dict__`` is a public view rather than the real mapping.

That question is separable from the merge itself, which is why it is here
(A1, section 3.2): :mod:`_rheplicant_bootstrap.layering_copy` asks it, and
this module is the whole of the answer.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from types import (
    BuiltinFunctionType,
    CellType,
    FunctionType,
    GetSetDescriptorType,
    MemberDescriptorType,
    ModuleType,
)
from weakref import ReferenceType

from _rheplicant_bootstrap.errors import ConfigError
from _rheplicant_bootstrap.layering_probe import (
    _COMPATIBILITY_FAILURE,
    _COMPATIBILITY_MISSING,
    _COMPATIBILITY_OWNERSHIP_COPY_ATOMIC_TYPES,
    _COMPATIBILITY_OWNERSHIP_SCALAR_TYPES,
    _COMPATIBILITY_UNSAFE_LAZY_ANNOTATIONS,
    _compatibility_add_identity,
    _compatibility_has_exact_type,
    _compatibility_has_mro_identity,
    _compatibility_is_mapping,
    _compatibility_iter_values,
    _compatibility_raw_mro_descriptor,
    _mapping_pairs,
)


def _compatibility_builtin_descriptor_value(value: object, owner: type, name: str) -> object:
    try:
        descriptor = type.__getattribute__(owner, "__dict__")[name]
        descriptor_type = type(descriptor)
        if not any(
            descriptor_type is candidate
            for candidate in (GetSetDescriptorType, MemberDescriptorType)
        ):
            raise TypeError
        return descriptor_type.__get__(descriptor, value, type(value))
    except (AttributeError, KeyError, ValueError):
        return _COMPATIBILITY_MISSING
    except Exception:
        raise ConfigError(_COMPATIBILITY_FAILURE) from None


def _compatibility_builtin_descriptor_values(
    value: object, owner: type, names: tuple[str, ...]
) -> list[object]:
    state_values: list[object] = []
    for name in names:
        state = _compatibility_builtin_descriptor_value(value, owner, name)
        if state is not _COMPATIBILITY_MISSING and state is not None:
            state_values.append(state)
    return state_values


def _compatibility_function_state(
    value: object,
) -> tuple[list[object], list[dict]]:
    if type(value) is BuiltinFunctionType:
        return (
            _compatibility_builtin_descriptor_values(
                value,
                BuiltinFunctionType,
                ("__self__", "__module__"),
            ),
            [],
        )

    state_values: list[object] = []
    state_mappings: list[dict] = []
    function_state = _compatibility_builtin_descriptor_value(value, FunctionType, "__dict__")
    if type(function_state) is not dict:
        raise ConfigError(_COMPATIBILITY_FAILURE)
    state_mappings.append(function_state)

    closure = _compatibility_builtin_descriptor_value(value, FunctionType, "__closure__")
    if closure is not None:
        if type(closure) is not tuple:
            raise ConfigError(_COMPATIBILITY_FAILURE)
        for cell in closure:
            if type(cell) is not CellType:
                raise ConfigError(_COMPATIBILITY_FAILURE)
            state_values.append(cell)
            contents = _compatibility_builtin_descriptor_value(cell, CellType, "cell_contents")
            if contents is not _COMPATIBILITY_MISSING:
                state_values.append(contents)

    defaults = _compatibility_builtin_descriptor_value(value, FunctionType, "__defaults__")
    if defaults is not None:
        if type(defaults) is not tuple:
            raise ConfigError(_COMPATIBILITY_FAILURE)
        state_values.append(defaults)

    keyword_defaults = _compatibility_builtin_descriptor_value(
        value, FunctionType, "__kwdefaults__"
    )
    if keyword_defaults is not None:
        if type(keyword_defaults) is not dict:
            raise ConfigError(_COMPATIBILITY_FAILURE)
        state_mappings.append(keyword_defaults)

    annotate = _compatibility_builtin_descriptor_value(value, FunctionType, "__annotate__")
    if annotate is _COMPATIBILITY_MISSING or annotate is None:
        annotations = _compatibility_builtin_descriptor_value(
            value, FunctionType, "__annotations__"
        )
        if annotations is not None:
            if type(annotations) is not dict:
                raise ConfigError(_COMPATIBILITY_FAILURE)
            state_mappings.append(annotations)
    else:
        state_values.append(_COMPATIBILITY_UNSAFE_LAZY_ANNOTATIONS)

    for name in ("__doc__", "__module__"):
        metadata = _compatibility_builtin_descriptor_value(value, FunctionType, name)
        if metadata is not _COMPATIBILITY_MISSING and metadata is not None:
            state_values.append(metadata)

    type_parameters = _compatibility_builtin_descriptor_value(
        value, FunctionType, "__type_params__"
    )
    if type_parameters is not _COMPATIBILITY_MISSING and type_parameters is not None:
        if type(type_parameters) is not tuple:
            raise ConfigError(_COMPATIBILITY_FAILURE)
        state_values.append(type_parameters)
    return state_values, state_mappings


def _compatibility_referential_atomic_state(value: object) -> list[object]:
    if type(value) is property:
        return _compatibility_builtin_descriptor_values(
            value,
            property,
            ("fget", "fset", "fdel", "__doc__", "__name__"),
        )

    try:
        referent = ReferenceType.__call__(value)
    except Exception:
        raise ConfigError(_COMPATIBILITY_FAILURE) from None
    callback = _compatibility_builtin_descriptor_value(value, ReferenceType, "__callback__")
    state_values = []
    if referent is not None:
        state_values.append(referent)
    if callback is not _COMPATIBILITY_MISSING and callback is not None:
        state_values.append(callback)
    return state_values


def _compatibility_ownership_state(
    value: object,
) -> tuple[list[object], list[dict], bool, bool]:
    if _compatibility_has_exact_type(value, _COMPATIBILITY_OWNERSHIP_SCALAR_TYPES):
        return [], [], False, True
    if _compatibility_has_exact_type(value, (FunctionType, BuiltinFunctionType)):
        state_values, state_mappings = _compatibility_function_state(value)
        return state_values, state_mappings, True, True
    if _compatibility_has_exact_type(value, (ReferenceType, property)):
        return (
            _compatibility_referential_atomic_state(value),
            [],
            True,
            True,
        )
    if type(value) is CellType:
        contents = _compatibility_builtin_descriptor_value(value, CellType, "cell_contents")
        if contents is _COMPATIBILITY_MISSING:
            return [], [], True, True
        return [contents], [], True, True
    if _compatibility_has_mro_identity(value, (type, ModuleType)):
        return [], [], False, True
    state_values: list[object] = []
    state_mappings: list[dict] = []
    state_bearing = False
    state_is_complete = True
    try:
        value_type = type(value)
        bases = type.__getattribute__(value_type, "__mro__")
    except Exception:
        raise ConfigError(_COMPATIBILITY_FAILURE) from None

    try:
        dict_descriptor = _compatibility_raw_mro_descriptor(value_type, "__dict__")
    except Exception:
        raise ConfigError(_COMPATIBILITY_FAILURE) from None
    descriptor_type = type(dict_descriptor)
    if any(
        descriptor_type is candidate for candidate in (GetSetDescriptorType, MemberDescriptorType)
    ):
        state_bearing = True
        try:
            instance_state = descriptor_type.__get__(dict_descriptor, value, value_type)
        except AttributeError:
            pass
        except Exception:
            raise ConfigError(_COMPATIBILITY_FAILURE) from None
        else:
            if type(instance_state) is not dict:
                state_is_complete = False
            else:
                state_mappings.append(instance_state)
    elif dict_descriptor is not _COMPATIBILITY_MISSING:
        state_is_complete = False

    for base in bases:
        try:
            namespace = type.__getattribute__(base, "__dict__")
        except Exception:
            raise ConfigError(_COMPATIBILITY_FAILURE) from None
        for name, descriptor in namespace.items():
            if name == "__dict__":
                continue
            if type(descriptor) is not MemberDescriptorType:
                continue
            state_bearing = True
            try:
                state = MemberDescriptorType.__get__(descriptor, value, value_type)
            except AttributeError:
                continue
            except Exception:
                raise ConfigError(_COMPATIBILITY_FAILURE) from None
            state_values.append(state)
    if _compatibility_has_mro_identity(value, (memoryview,)):
        try:
            descriptor = type.__getattribute__(memoryview, "__dict__")["obj"]
            backing = GetSetDescriptorType.__get__(descriptor, value, value_type)
        except Exception:
            raise ConfigError(_COMPATIBILITY_FAILURE) from None
        state_bearing = True
        state_values.append(backing)
    if _compatibility_has_mro_identity(value, (deque,)):
        try:
            iterator = deque.__iter__(value)
        except Exception:
            raise ConfigError(_COMPATIBILITY_FAILURE) from None
        while True:
            try:
                state_values.append(next(iterator))
            except StopIteration:
                break
            except Exception:
                raise ConfigError(_COMPATIBILITY_FAILURE) from None
        state_bearing = True
    if _compatibility_has_mro_identity(value, (ReferenceType,)):
        state_values.extend(_compatibility_referential_atomic_state(value))
        state_bearing = True
    return state_values, state_mappings, state_bearing, state_is_complete


def _compatibility_instance_state_values(value: object) -> list[object]:
    state_values, state_mappings, _, state_is_complete = _compatibility_ownership_state(value)
    if not state_is_complete:
        raise ConfigError(_COMPATIBILITY_FAILURE)
    return [*state_values, *state_mappings]


def _compatibility_uncanonicalized_mapping_values(
    mapping: Mapping,
) -> list[object]:
    return [value for _, value in _mapping_pairs(mapping, failure=_COMPATIBILITY_FAILURE)]


def _compatibility_state_mapping_is_public_view(
    state_mapping: object, public_mapping: dict[str, object]
) -> bool:
    if type(state_mapping) is not dict:
        return False
    if dict.__len__(state_mapping) != dict.__len__(public_mapping):
        return False
    for key, value in dict.items(state_mapping):
        if not _compatibility_has_mro_identity(key, (str,)):
            return False
        exact_key = str.__str__(key)
        try:
            public_value = dict.__getitem__(public_mapping, exact_key)
        except KeyError:
            return False
        if public_value is not value:
            return False
    return True


def _compatibility_original_mutables(
    roots: tuple[Mapping, Mapping],
) -> dict[int, list[object]]:
    mutables: dict[int, list[object]] = {}
    seen: dict[int, list[object]] = {}
    pending: list[object] = [*roots]
    while pending:
        item = pending.pop()
        if _compatibility_has_exact_type(item, _COMPATIBILITY_OWNERSHIP_SCALAR_TYPES):
            continue
        if not _compatibility_add_identity(seen, item):
            continue
        is_mapping = _compatibility_is_mapping(item)
        is_copy_atomic = _compatibility_has_exact_type(
            item, _COMPATIBILITY_OWNERSHIP_COPY_ATOMIC_TYPES
        )
        is_immutable_container = _compatibility_has_mro_identity(item, (tuple, frozenset))
        is_static_namespace = _compatibility_has_mro_identity(item, (type, ModuleType))
        state_values, state_mappings, _, state_is_complete = _compatibility_ownership_state(item)
        if not (is_copy_atomic or is_immutable_container or is_static_namespace):
            _compatibility_add_identity(mutables, item)
        if state_is_complete:
            pending.extend(state_values)
            pending.extend(state_mappings)
        if is_mapping:
            pending.extend(_compatibility_uncanonicalized_mapping_values(item))
        elif _compatibility_has_mro_identity(item, (list, tuple, set, frozenset)):
            pending.extend(_compatibility_iter_values(item))
    return mutables
