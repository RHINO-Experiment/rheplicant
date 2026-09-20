"""The defensive copy ``merge_extends`` takes before it merges anything.

``merge_extends`` has to hand a caller a document it can mutate without
touching the one it was given, and the inputs are arbitrary mappings: a
``UserDict``, a ``MappingProxyType``, something whose ``__getitem__`` lies.
Everything here exists to make that copy and then to CHECK it -- that the
result shares no mutable state with its source, that reading it went through
the protocols it claims to support, and that nothing was quietly skipped.

It is separated from :mod:`_rheplicant_bootstrap.layering` because it is a
different subject with a different failure mode. Layering is about which
value wins; this is about whether a copy is a copy. Every name here is
private to that question and is used by ``merge_extends`` alone, which is why
the split is a move rather than a redesign (A1, section 3.2).
"""

from __future__ import annotations

import copy
from collections import UserDict, defaultdict
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from types import (
    MappingProxyType,
)
from weakref import CallableProxyType, ProxyType

from _rheplicant_bootstrap.errors import ConfigError
from _rheplicant_bootstrap.layering_keys import _deletion_target
from _rheplicant_bootstrap.layering_ownership import (
    _compatibility_instance_state_values,
    _compatibility_original_mutables,
    _compatibility_ownership_state,
    _compatibility_state_mapping_is_public_view,
    _compatibility_uncanonicalized_mapping_values,
)
from _rheplicant_bootstrap.layering_probe import (
    _COMPATIBILITY_FAILURE,
    _COMPATIBILITY_OWNERSHIP_SCALAR_TYPES,
    _COMPATIBILITY_UNSAFE_LAZY_ANNOTATIONS,
    _compatibility_add_identity,
    _compatibility_has_exact_type,
    _compatibility_has_identity,
    _compatibility_has_mro_identity,
    _compatibility_increment_occurrence,
    _compatibility_is_mapping,
    _compatibility_is_mutable_mapping,
    _compatibility_iter_values,
    _compatibility_occurrence_count,
    _compatibility_pairs,
    _compatibility_type_name,
)

#: Iterating a mapping the caller supplied, without trusting any step of it.
#: Shared with :mod:`_rheplicant_bootstrap.layering`, and it lives here
#: because defensive iteration is this module's subject.


def _compatibility_deepcopy_roots(
    parent: Mapping, child: Mapping
) -> tuple[Mapping, Mapping, dict[int, list[object]]]:
    """Detach both roots in one deepcopy graph, materializing only frozen views."""
    memo: dict[int, object] = {}
    mapping_shells: list[tuple[Mapping, dict[str, object]]] = []
    seen: dict[int, list[object]] = {}

    def protocol_values(value: object):
        try:
            iterator = iter(value)  # type: ignore[arg-type]
        except Exception:
            raise ConfigError(_COMPATIBILITY_FAILURE) from None
        while True:
            try:
                yield next(iterator)
            except StopIteration:
                return
            except Exception:
                raise ConfigError(_COMPATIBILITY_FAILURE) from None

    def discover(item: object) -> None:
        identity = id(item)
        bucket = seen.setdefault(identity, [])
        if any(source is item for source in bucket):
            return
        bucket.append(item)
        is_mapping = _compatibility_is_mapping(item)
        if _compatibility_has_exact_type(item, (dict, MappingProxyType)):
            shell: dict[str, object] = {}
            memo[identity] = shell
            mapping_shells.append((item, shell))
            canonical = _compatibility_pairs(item)
            for nested in canonical.values():
                discover(nested)
        elif is_mapping:
            try:
                deepcopy_protocol = getattr(type(item), "__deepcopy__", None)
            except Exception:
                raise ConfigError(_COMPATIBILITY_FAILURE) from None
            if deepcopy_protocol is not None:
                return
            canonical = _compatibility_pairs(item)
            for nested in canonical.values():
                discover(nested)
        elif _compatibility_has_mro_identity(item, (list, tuple, set, frozenset)):
            for nested in protocol_values(item):
                discover(nested)

    # Validate the two control mappings before copy protocols can touch their keys.
    _compatibility_pairs(parent)
    _compatibility_pairs(child)
    original_mutables = _compatibility_original_mutables((parent, child))
    discover((parent, child))

    def detached(value: object) -> object:
        try:
            return copy.deepcopy(value, memo)
        except Exception:
            raise ConfigError(_COMPATIBILITY_FAILURE) from None

    for original, shell in mapping_shells:
        canonical = _compatibility_pairs(original)
        for key, value in canonical.items():
            shell[key] = detached(value)
    roots = detached((parent, child))
    assert type(roots) is tuple and len(roots) == 2
    detached_parent, detached_child = roots
    if not _compatibility_is_mapping(detached_parent) or not _compatibility_is_mapping(
        detached_child
    ):
        raise ConfigError(_COMPATIBILITY_FAILURE)
    return detached_parent, detached_child, original_mutables


def _compatibility_reaches_mapping(
    values: object,
    target: Mapping,
) -> bool:
    """Detect whether a shallow COW shell would break a cycle to ``target``."""
    return _compatibility_reaches_forbidden_mapping(
        values,
        target=target,
        forbidden=None,
        ignored=None,
        opaque_is_failure=True,
        inspect_object_state=False,
        state_mappings=None,
        public_mapping=None,
    )


def _compatibility_reaches_forbidden_mapping(
    values: object,
    *,
    target: Mapping,
    forbidden: dict[int, list[object]] | None,
    ignored: object | None,
    opaque_is_failure: bool,
    inspect_object_state: bool,
    state_mappings: object | None,
    public_mapping: dict[str, object] | None,
) -> bool:
    field_role = 0
    content_role = 1
    public_role = 2
    try:
        pending = [(item, field_role, 0) for item in values]
        if state_mappings is not None:
            pending.extend((item, field_role, 2) for item in state_mappings)
    except Exception:
        raise ConfigError(_COMPATIBILITY_FAILURE) from None
    public_identities: dict[int, list[object]] = {}
    if public_mapping is not None:
        try:
            for item in dict.values(public_mapping):
                _compatibility_add_identity(public_identities, item)
        except Exception:
            raise ConfigError(_COMPATIBILITY_FAILURE) from None
    seen_by_mode: tuple[dict[int, list[object]], ...] = ({}, {}, {})
    while pending:
        item, role, state_mapping_kind = pending.pop()
        if role == content_role and _compatibility_has_identity(public_identities, item):
            role = public_role
        if role == public_role:
            continue
        if item is _COMPATIBILITY_UNSAFE_LAZY_ANNOTATIONS:
            raise ConfigError(_COMPATIBILITY_FAILURE)
        if inspect_object_state and _compatibility_has_exact_type(
            item, (ProxyType, CallableProxyType)
        ):
            raise ConfigError(_COMPATIBILITY_FAILURE)
        if ignored is not None and item is ignored:
            continue
        if item is target or (
            forbidden is not None and _compatibility_has_identity(forbidden, item)
        ):
            return True
        if _compatibility_has_exact_type(item, _COMPATIBILITY_OWNERSHIP_SCALAR_TYPES):
            continue
        is_known_container = _compatibility_is_mapping(item) or _compatibility_has_mro_identity(
            item, (list, tuple, set, frozenset)
        )
        if not is_known_container and not inspect_object_state:
            if opaque_is_failure:
                raise ConfigError(_COMPATIBILITY_FAILURE)
            continue
        if state_mapping_kind:
            mode = 2
        elif role == public_role:
            mode = 0
        else:
            mode = 1
        if not _compatibility_add_identity(seen_by_mode[mode], item):
            continue
        if inspect_object_state and role != public_role:
            (
                state_values,
                nested_state_mappings,
                _,
                state_is_complete,
            ) = _compatibility_ownership_state(item)
            if state_is_complete:
                pending.extend((state, field_role, 0) for state in state_values)
                pending.extend((state, field_role, 1) for state in nested_state_mappings)
        if not is_known_container:
            continue
        if _compatibility_is_mapping(item):
            if inspect_object_state:
                nested_values = _compatibility_uncanonicalized_mapping_values(item)
            else:
                nested_values = _compatibility_pairs(item).values()
            child_role = (
                content_role
                if state_mapping_kind == 2
                and public_mapping is not None
                and _compatibility_state_mapping_is_public_view(item, public_mapping)
                else field_role
                if state_mapping_kind
                else public_role
                if role == public_role
                else content_role
            )
            pending.extend((nested, child_role, 0) for nested in nested_values)
            continue
        try:
            iterator = iter(item)
        except Exception:
            raise ConfigError(_COMPATIBILITY_FAILURE) from None
        while True:
            try:
                nested = next(iterator)
            except StopIteration:
                break
            except Exception:
                raise ConfigError(_COMPATIBILITY_FAILURE) from None
            child_role = public_role if role == public_role else content_role
            pending.append((nested, child_role, 0))
    return False


def _compatibility_reset_mapping(mapping: MutableMapping, values: Mapping[str, object]) -> None:
    try:
        if _compatibility_has_mro_identity(mapping, (dict,)):
            dict.clear(mapping)
            dict.update(mapping, values)
        else:
            MutableMapping.clear(mapping)
            MutableMapping.update(mapping, values)
    except Exception:
        raise ConfigError(_COMPATIBILITY_FAILURE) from None


def _compatibility_mapping_get(mapping: Mapping, key: str, default: object = None) -> object:
    try:
        if _compatibility_has_mro_identity(mapping, (dict,)):
            return dict.get(mapping, key, default)
        return Mapping.get(mapping, key, default)
    except Exception:
        raise ConfigError(_COMPATIBILITY_FAILURE) from None


def _compatibility_mapping_pop(mapping: MutableMapping, key: str) -> None:
    try:
        if _compatibility_has_mro_identity(mapping, (dict,)):
            dict.pop(mapping, key, None)
        else:
            MutableMapping.pop(mapping, key, None)
    except Exception:
        raise ConfigError(_COMPATIBILITY_FAILURE) from None


def _compatibility_mapping_set(mapping: MutableMapping, key: str, value: object) -> None:
    try:
        if _compatibility_has_mro_identity(mapping, (dict,)):
            dict.__setitem__(mapping, key, value)
        else:
            mapping[key] = value
    except Exception:
        raise ConfigError(_COMPATIBILITY_FAILURE) from None


def _compatibility_mapping_occurrences(
    roots: tuple[Mapping, Mapping],
) -> dict[int, list[tuple[object, int]]]:
    occurrences: dict[int, list[tuple[object, int]]] = {}
    expanded: dict[int, list[object]] = {}
    pending: list[object] = [*roots]
    while pending:
        item = pending.pop()
        if _compatibility_is_mapping(item):
            _compatibility_increment_occurrence(occurrences, item)
        if not (
            _compatibility_is_mapping(item)
            or _compatibility_has_mro_identity(item, (list, tuple, set, frozenset))
        ):
            continue
        if not _compatibility_add_identity(expanded, item):
            continue
        pending.extend(_compatibility_iter_values(item))
    return occurrences


@dataclass(slots=True)
class _CompatibilityMergeContext:
    active_children: dict[int, list[object]]
    mapping_occurrences: dict[int, list[tuple[object, int]]]
    must_split_mappings: dict[int, list[object]]
    original_mutables: dict[int, list[object]]


def _compatibility_mark_mapping_descendants(
    values: object, *, context: _CompatibilityMergeContext
) -> None:
    try:
        pending = list(values)  # values is an exact dict view at call sites
    except Exception:
        raise ConfigError(_COMPATIBILITY_FAILURE) from None
    expanded: dict[int, list[object]] = {}
    while pending:
        item = pending.pop()
        if not (
            _compatibility_is_mapping(item)
            or _compatibility_has_mro_identity(item, (list, tuple, set, frozenset))
        ):
            continue
        if not _compatibility_add_identity(expanded, item):
            continue
        if _compatibility_is_mapping(item):
            _compatibility_add_identity(context.must_split_mappings, item)
        pending.extend(_compatibility_iter_values(item))


def _compatibility_validate_split_state(
    copied: MutableMapping,
    parent: Mapping,
    *,
    context: _CompatibilityMergeContext,
) -> None:
    copied_data: object | None = None
    if type(parent) is UserDict:
        try:
            parent_data = object.__getattribute__(parent, "data")
            copied_data = object.__getattribute__(copied, "data")
        except Exception:
            raise ConfigError(_COMPATIBILITY_FAILURE) from None
        if (
            type(parent_data) is not dict
            or type(copied_data) is not dict
            or copied_data is parent_data
        ):
            raise ConfigError(_COMPATIBILITY_FAILURE)
    state_values = _compatibility_instance_state_values(copied)
    if copied_data is not None:
        state_values = [state for state in state_values if state is not copied_data]
    if _compatibility_reaches_forbidden_mapping(
        state_values,
        target=parent,
        forbidden=context.must_split_mappings,
        ignored=copied,
        opaque_is_failure=True,
        inspect_object_state=False,
        state_mappings=None,
        public_mapping=None,
    ):
        raise ConfigError(_COMPATIBILITY_FAILURE)


def _compatibility_validate_mapping_mutation_target(
    mapping: MutableMapping,
    *,
    context: _CompatibilityMergeContext,
    public_mapping: dict[str, object],
) -> None:
    if _compatibility_has_identity(context.original_mutables, mapping):
        raise ConfigError(_COMPATIBILITY_FAILURE)
    if _compatibility_has_mro_identity(mapping, (dict,)):
        return
    copied_data: object | None = None
    if type(mapping) is UserDict:
        try:
            copied_data = object.__getattribute__(mapping, "data")
        except Exception:
            raise ConfigError(_COMPATIBILITY_FAILURE) from None
        if type(copied_data) is not dict or _compatibility_has_identity(
            context.original_mutables, copied_data
        ):
            raise ConfigError(_COMPATIBILITY_FAILURE)
    state_values, state_mappings, _, state_is_complete = _compatibility_ownership_state(mapping)
    if not state_is_complete:
        raise ConfigError(_COMPATIBILITY_FAILURE)
    if _compatibility_reaches_forbidden_mapping(
        state_values,
        target=mapping,
        forbidden=context.original_mutables,
        ignored=mapping,
        opaque_is_failure=False,
        inspect_object_state=True,
        state_mappings=state_mappings,
        public_mapping=public_mapping,
    ):
        raise ConfigError(_COMPATIBILITY_FAILURE)


def _compatibility_validate_dict_read_protocols(parent: dict) -> None:
    protocols = (
        ("__getattribute__", (dict, defaultdict)),
        ("__getitem__", (dict,)),
        ("__iter__", (dict,)),
        ("__len__", (dict,)),
        ("items", (dict,)),
    )
    try:
        bases = type.__getattribute__(type(parent), "__mro__")
        for name, expected_owners in protocols:
            expected = tuple(
                type.__getattribute__(owner, "__dict__")[name] for owner in expected_owners
            )
            resolved = None
            for base in bases:
                namespace = type.__getattribute__(base, "__dict__")
                try:
                    resolved = namespace[name]
                except KeyError:
                    continue
                break
            if not any(resolved is candidate for candidate in expected):
                raise ConfigError(_COMPATIBILITY_FAILURE)
    except ConfigError:
        raise
    except Exception:
        raise ConfigError(_COMPATIBILITY_FAILURE) from None


def _compatibility_split_mapping(
    parent: Mapping,
    parent_values: Mapping[str, object],
    *,
    context: _CompatibilityMergeContext,
) -> MutableMapping:
    if type(parent) is dict:
        return dict(parent_values)
    if not _compatibility_has_mro_identity(parent, (dict,)) and type(parent) is not UserDict:
        raise ConfigError(_COMPATIBILITY_FAILURE)
    try:
        merged = copy.copy(parent)
    except Exception:
        raise ConfigError(_COMPATIBILITY_FAILURE) from None
    if (
        type(merged) is not type(parent)
        or merged is parent
        or not _compatibility_is_mutable_mapping(merged)
    ):
        raise ConfigError(_COMPATIBILITY_FAILURE)
    _compatibility_validate_split_state(merged, parent, context=context)
    return merged


def _merge_extends_compat(
    child: Mapping,
    parent: Mapping,
    *,
    context: _CompatibilityMergeContext,
    reuse_parent: bool,
) -> MutableMapping:
    child_identity = id(child)
    active_bucket = context.active_children.setdefault(child_identity, [])
    if any(source is child for source in active_bucket):
        raise ConfigError("merge_extends: overlapping cyclic mappings cannot be merged.")
    active_bucket.append(child)
    try:
        parent_values = _compatibility_pairs(parent)
        if type(parent) is not dict and _compatibility_has_mro_identity(parent, (dict,)):
            _compatibility_validate_dict_read_protocols(parent)
        requires_split = not reuse_parent and (
            _compatibility_occurrence_count(context.mapping_occurrences, parent) > 1
            or _compatibility_has_identity(context.must_split_mappings, parent)
        )
        if requires_split:
            if _compatibility_reaches_mapping(parent_values.values(), parent):
                raise ConfigError("merge_extends: overlapping cyclic mappings cannot be merged.")
            _compatibility_mark_mapping_descendants(parent_values.values(), context=context)
        if reuse_parent and _compatibility_has_mro_identity(parent, (dict,)):
            merged = parent
        elif reuse_parent:
            merged = dict(parent_values)
        elif requires_split:
            merged = _compatibility_split_mapping(parent, parent_values, context=context)
        elif _compatibility_is_mutable_mapping(parent):
            merged = parent
        else:
            raise ConfigError(_COMPATIBILITY_FAILURE)
        _compatibility_validate_mapping_mutation_target(
            merged,
            context=context,
            public_mapping=parent_values,
        )
        _compatibility_reset_mapping(merged, parent_values)
        child_values = _compatibility_pairs(child)
        for key, value in child_values.items():
            if str.startswith(key, "~"):
                target = _deletion_target(key)
                if value is not None:
                    raise ConfigError(f"{key!r}: deletion value must be null.")
                _compatibility_mapping_pop(merged, target)
                continue
            value_mapping = (
                _compatibility_pairs(value) if _compatibility_is_mapping(value) else None
            )
            if value_mapping is not None and "append" in value_mapping:
                if set(value_mapping) != {"append"}:
                    siblings = sorted(set(value_mapping) - {"append"})
                    raise ConfigError(
                        f"{key!r}: append must be the only key when extending a list; "
                        f"got the sibling keys {siblings}."
                    )
                appended = value_mapping["append"]
                if not _compatibility_has_mro_identity(appended, (list, tuple)):
                    raise ConfigError(
                        f"{key!r}: append is a sequence; got {_compatibility_type_name(appended)}."
                    )
                inherited = _compatibility_mapping_get(merged, key, [])
                if not _compatibility_has_mro_identity(inherited, (list,)):
                    raise ConfigError(
                        f"{key!r} is extended with {{append: ...}} but the inherited "
                        f"value is {_compatibility_type_name(inherited)}, not a list."
                    )
                if _compatibility_has_identity(context.original_mutables, inherited):
                    raise ConfigError(_COMPATIBILITY_FAILURE)
                try:
                    list.extend(inherited, appended)
                except Exception:
                    raise ConfigError(_COMPATIBILITY_FAILURE) from None
                _compatibility_mapping_set(merged, key, inherited)
                continue
            inherited = _compatibility_mapping_get(merged, key)
            if _compatibility_is_mapping(value) and _compatibility_is_mapping(inherited):
                _compatibility_mapping_set(
                    merged,
                    key,
                    _merge_extends_compat(
                        value,
                        inherited,
                        context=context,
                        reuse_parent=False,
                    ),
                )
                continue
            _compatibility_mapping_set(merged, key, value)
        return merged
    finally:
        removed = active_bucket.pop()
        assert removed is child
        if not active_bucket:
            del context.active_children[child_identity]
