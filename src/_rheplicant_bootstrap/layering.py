"""JAX-free document layering with immutable per-value origin evidence."""

from __future__ import annotations

import copy
from collections import UserDict, defaultdict, deque
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from types import (
    BuiltinFunctionType,
    CellType,
    CodeType,
    FunctionType,
    GetSetDescriptorType,
    MappingProxyType,
    MemberDescriptorType,
    ModuleType,
)
from typing import TypeAlias
from weakref import CallableProxyType, ProxyType, ReferenceType

from _rheplicant_bootstrap.errors import ConfigError
from _rheplicant_bootstrap.frozen import freeze_evidence
from _rheplicant_bootstrap.presets import PresetRequest, PresetSnapshot
from _rheplicant_bootstrap.types import Origin

OriginSegment: TypeAlias = str | int


def _canonical_origin(value: object, *, where: str) -> Origin:
    if not isinstance(value, Origin):
        raise ConfigError(f"{where} must be an Origin; got {type(value).__name__}.")
    if not isinstance(value.kind, str):
        raise ConfigError(
            f"{where} kind must be a string; got {type(value.kind).__name__}."
        )
    kind = str.__str__(value.kind)
    if value.name is None:
        name = None
    elif isinstance(value.name, str):
        name = str.__str__(value.name)
    else:
        raise ConfigError(
            f"{where} name must be a string or null; got "
            f"{type(value.name).__name__}."
        )
    try:
        canonical = Origin(kind, name)
    except ValueError as exc:
        raise ConfigError(f"{where} is invalid: {exc}") from exc
    if type(value) is Origin and type(value.kind) is str and (
        value.name is None or type(value.name) is str
    ):
        return value
    return canonical


def _canonical_segment(value: object, *, where: str) -> OriginSegment:
    if isinstance(value, bool):
        raise ConfigError(f"{where} must be str or int; got bool.")
    if isinstance(value, str):
        return str.__str__(value)
    if isinstance(value, int):
        return int.__int__(value)
    raise ConfigError(f"{where} must be str or int; got {type(value).__name__}.")


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


@dataclass(frozen=True, slots=True)
class OriginNode:
    """One value/container in the origin tree; the document root has no origin."""

    origin: Origin | None
    children: Mapping[OriginSegment, OriginNode]

    def __post_init__(self) -> None:
        rebuilt = _detach_origin_tree(self, public_origin_node=True)
        object.__setattr__(self, "origin", rebuilt.origin)
        object.__setattr__(self, "children", rebuilt.children)


def _trusted_origin_node(
    origin: Origin | None,
    children: Mapping[OriginSegment, OriginNode],
) -> OriginNode:
    node = object.__new__(OriginNode)
    object.__setattr__(node, "origin", origin)
    object.__setattr__(node, "children", children)
    return node


def _detach_origin_tree(
    root: OriginNode, *, public_origin_node: bool
) -> OriginNode:
    completed: dict[int, list[tuple[object, OriginNode]]] = {}
    active: dict[int, list[object]] = {}
    traversal_failure = (
        "origin children mapping traversal failed."
        if public_origin_node
        else "merge result origin tree children traversal failed."
    )

    def rebuild(node: object) -> OriginNode:
        if not isinstance(node, OriginNode):
            message = (
                "origin child value must be an OriginNode; got "
                if public_origin_node
                else "merge result origin tree children must be OriginNode values; got "
            )
            raise ConfigError(f"{message}{type(node).__name__}.")
        identity = id(node)
        for source, result in completed.get(identity, ()):
            if source is node:
                return result
        bucket = active.setdefault(identity, [])
        if any(source is node for source in bucket):
            raise ConfigError(
                "origin tree must be acyclic."
                if public_origin_node
                else "merge result origin tree must be acyclic."
            )
        bucket.append(node)
        try:
            origin = (
                None
                if node.origin is None
                else _canonical_origin(
                    node.origin,
                    where=(
                        "origin node origin"
                        if public_origin_node
                        else "merge result origin tree origin"
                    ),
                )
            )
            if not isinstance(node.children, Mapping):
                raise ConfigError(
                    "origin children must be a mapping."
                    if public_origin_node
                    else "merge result origin tree children must be a mapping."
                )
            canonical: dict[OriginSegment, OriginNode] = {}
            for segment, child in _mapping_pairs(
                node.children, failure=traversal_failure
            ):
                exact_segment = _canonical_segment(
                    segment,
                    where=(
                        "origin child segment"
                        if public_origin_node
                        else "merge result origin tree segment"
                    ),
                )
                if exact_segment in canonical:
                    raise ConfigError(
                        "origin child segments collide after canonicalization."
                        if public_origin_node
                        else "merge result origin tree segments collide after "
                        "canonicalization."
                    )
                canonical[exact_segment] = rebuild(child)
            result = _trusted_origin_node(
                origin, MappingProxyType(canonical)
            )
            completed.setdefault(identity, []).append((node, result))
            return result
        finally:
            removed = bucket.pop()
            assert removed is node
            if not bucket:
                del active[identity]

    try:
        return rebuild(root)
    except ConfigError:
        raise
    except RecursionError:
        raise ConfigError(
            "origin tree recursion exceeds the supported depth."
            if public_origin_node
            else "merge result origin tree recursion exceeds the supported depth."
        ) from None
    except Exception:
        raise ConfigError(
            "origin tree protocol failed."
            if public_origin_node
            else "merge result origin tree protocol failed."
        ) from None


@dataclass(frozen=True, slots=True)
class DeletionRecord:
    path: Sequence[OriginSegment]
    origin: Origin

    def __post_init__(self) -> None:
        if isinstance(self.path, str | bytes) or not isinstance(self.path, Sequence):
            raise ConfigError("deletion path must be a sequence of segments.")
        try:
            given = tuple(self.path)
        except Exception:
            raise ConfigError("deletion path sequence traversal failed.") from None
        if not given:
            raise ConfigError("deletion path must be non-empty.")
        canonical = tuple(
            _canonical_segment(segment, where="deletion path segment")
            for segment in given
        )
        origin = _canonical_origin(self.origin, where="deletion origin")
        object.__setattr__(self, "path", canonical)
        object.__setattr__(self, "origin", origin)


def _canonicalize_origin_tree(root: OriginNode) -> OriginNode:
    return _detach_origin_tree(root, public_origin_node=False)


@dataclass(frozen=True, slots=True)
class MergeResult:
    document: Mapping[str, object]
    origins: OriginNode
    deletions: Sequence[DeletionRecord]

    def __post_init__(self) -> None:
        if not isinstance(self.document, Mapping):
            raise ConfigError("merge result document must be a mapping.")
        if not isinstance(self.origins, OriginNode):
            raise ConfigError("merge result origins must be an OriginNode.")
        if isinstance(self.deletions, str | bytes) or not isinstance(
            self.deletions, Sequence
        ):
            raise ConfigError("merge result deletions must be a sequence.")
        try:
            deletions = tuple(self.deletions)
        except Exception:
            raise ConfigError(
                "merge result deletions sequence traversal failed."
            ) from None
        if any(not isinstance(item, DeletionRecord) for item in deletions):
            raise ConfigError("merge result deletions must contain DeletionRecord values.")
        deletions = tuple(
            DeletionRecord(item.path, item.origin) for item in deletions
        )
        origins = _canonicalize_origin_tree(self.origins)
        document = freeze_evidence(self.document, where="merge result document")
        assert isinstance(document, Mapping)
        _validate_parallel_origin_tree(document, origins)
        object.__setattr__(self, "document", document)
        object.__setattr__(self, "origins", origins)
        object.__setattr__(self, "deletions", deletions)


def _trusted_merge_result(
    document: Mapping[str, object],
    origins: OriginNode,
    deletions: tuple[DeletionRecord, ...],
) -> MergeResult:
    """Build an internal result from lockstep COW fragments already validated."""
    assert type(document) is MappingProxyType
    assert type(origins) is OriginNode
    assert origins.origin is None
    assert type(origins.children) is MappingProxyType
    assert len(document) == len(origins.children)
    result = object.__new__(MergeResult)
    object.__setattr__(result, "document", document)
    object.__setattr__(result, "origins", origins)
    object.__setattr__(result, "deletions", deletions)
    return result


def _validate_parallel_origin_tree(
    document: Mapping[str, object], origins: OriginNode
) -> None:
    pending: list[tuple[object, OriginNode, bool]] = [(document, origins, True)]
    seen_pairs: dict[
        tuple[int, int], list[tuple[object, OriginNode]]
    ] = {}
    document_origins: dict[int, list[tuple[object, OriginNode]]] = {}
    while pending:
        value, node, is_root = pending.pop()
        if not isinstance(node, OriginNode):
            raise ConfigError(
                "merge result origin tree children must be OriginNode values."
            )
        if is_root:
            if node.origin is not None:
                raise ConfigError("merge result origin tree root origin must be null.")
        elif node.origin is None:
            raise ConfigError(
                "merge result origin tree descendants must have concrete origins."
            )
        else:
            _canonical_origin(
                node.origin, where="merge result origin tree descendant origin"
            )

        is_alias_container = isinstance(value, Mapping) or (
            isinstance(value, tuple) and tuple.__len__(value) != 0
        )
        if is_alias_container:
            document_identity = id(value)
            origin_identity = id(node)
            origins_bucket = document_origins.setdefault(document_identity, [])
            prior = next(
                (
                    prior_node
                    for prior_value, prior_node in origins_bucket
                    if prior_value is value
                ),
                None,
            )
            if prior is not None and prior is not node:
                raise ConfigError(
                    "merge result origin tree assigns divergent origins to "
                    "one shared document container."
                )
            if prior is None:
                origins_bucket.append((value, node))
            pair = (document_identity, origin_identity)
            pair_bucket = seen_pairs.setdefault(pair, [])
            if any(
                prior_value is value and prior_node is node
                for prior_value, prior_node in pair_bucket
            ):
                continue
            pair_bucket.append((value, node))

        children = node.children
        try:
            given_segments = tuple(children)
        except Exception:
            raise ConfigError(
                "merge result origin tree children traversal failed."
            ) from None
        if isinstance(value, Mapping):
            expected = tuple(value)
            child_values = value
        elif isinstance(value, tuple):
            expected = tuple(range(len(value)))
            child_values = None
        else:
            expected = ()
            child_values = None
        expected_type = str if isinstance(value, Mapping) else int
        if (
            len(given_segments) != len(expected)
            or any(type(segment) is not expected_type for segment in given_segments)
        ):
            raise ConfigError(
                "merge result origin tree children must exactly match the document."
            )
        for segment in expected:
            try:
                child_node = children[segment]
            except (KeyError, TypeError):
                raise ConfigError(
                    "merge result origin tree children must exactly match the document."
                ) from None
            except Exception:
                raise ConfigError(
                    "merge result origin tree child lookup failed."
                ) from None
            child_value = (
                child_values[segment]
                if child_values is not None
                else value[segment]
            )
            pending.append((child_value, child_node, False))


def _frozen_children(
    children: Mapping[OriginSegment, OriginNode],
) -> Mapping[OriginSegment, OriginNode]:
    return MappingProxyType(dict(children))


def _origin_node(
    value: object,
    origin: Origin,
    memo: dict[int, list[tuple[object, OriginNode]]] | None = None,
) -> OriginNode:
    if memo is None:
        memo = {}
    identity = id(value)
    memoized = isinstance(value, Mapping) or (
        isinstance(value, tuple) and tuple.__len__(value) != 0
    )
    if memoized:
        for source, node in memo.get(identity, ()):
            if source is value:
                return node
    if isinstance(value, Mapping):
        children = {
            key: _origin_node(item, origin, memo) for key, item in value.items()
        }
    elif isinstance(value, list | tuple):
        children = {
            index: _origin_node(item, origin, memo)
            for index, item in enumerate(value)
        }
    else:
        children = {}
    result = _trusted_origin_node(
        origin=origin, children=_frozen_children(children)
    )
    if memoized:
        memo.setdefault(identity, []).append((value, result))
    return result


def _root_node(document: Mapping[str, object], origin: Origin) -> OriginNode:
    memo: dict[int, list[tuple[object, OriginNode]]] = {}
    return _trusted_origin_node(
        origin=None,
        children=_frozen_children(
            {
                key: _origin_node(value, origin, memo)
                for key, value in document.items()
            }
        ),
    )


def initial_merge(document: Mapping[str, object], *, origin: Origin) -> MergeResult:
    if not isinstance(document, Mapping):
        raise ConfigError(
            "initial_merge: document is a mapping; got "
            f"{type(document).__name__}."
        )
    origin = _canonical_origin(origin, where="initial_merge origin")
    evidence = freeze_evidence(document, where="initial_merge document")
    assert isinstance(evidence, Mapping)
    return _trusted_merge_result(
        document=evidence,
        origins=_root_node(evidence, origin),
        deletions=(),
    )


def _append_value(
    *,
    key: str,
    inherited: object,
    inherited_origin: OriginNode,
    given: Mapping[object, object],
    origin: Origin,
    context: _MergeContext,
) -> tuple[tuple[object, ...], OriginNode]:
    cache_key = (id(inherited), id(inherited_origin), id(given))
    for (
        cached_inherited,
        cached_origin,
        cached_given,
        cached_result,
    ) in context.appends.get(cache_key, ()):
        if (
            cached_inherited is inherited
            and cached_origin is inherited_origin
            and cached_given is given
        ):
            return cached_result
    if set(given) != {"append"}:
        siblings = sorted(str(item) for item in set(given) - {"append"})
        raise ConfigError(
            f"{key!r}: append must be the only key when extending a list; "
            f"got the sibling keys {siblings}."
        )
    appended = given["append"]
    if not isinstance(appended, tuple):
        raise ConfigError(
            f"{key!r}: append is a sequence; got "
            f"{type(appended).__name__} ({appended!r})."
        )
    if not isinstance(inherited, tuple):
        raise ConfigError(
            f"{key!r} is extended with {{append: ...}} but the inherited value "
            f"is {type(inherited).__name__}, not a list."
        )
    if tuple.__len__(appended) == 0:
        result = (inherited, inherited_origin)
        context.appends.setdefault(cache_key, []).append(
            (inherited, inherited_origin, given, result)
        )
        return result
    values = tuple([*inherited, *appended])
    inherited_children = dict(inherited_origin.children)
    offset = len(inherited)
    inherited_children.update(
        {
            offset + index: _origin_node(item, origin, context.origin_nodes)
            for index, item in enumerate(appended)
        }
    )
    result = (
        values,
        _trusted_origin_node(
            origin=inherited_origin.origin,
            children=_frozen_children(inherited_children),
        ),
    )
    context.appends.setdefault(cache_key, []).append(
        (inherited, inherited_origin, given, result)
    )
    return result


def _deletion_target(key: str) -> str:
    target = key[1:]
    if not target:
        raise ConfigError("layering deletion key must name a value after '~'.")
    return target


class _MergeContext:
    __slots__ = ("appends", "fragments", "origin_nodes")

    def __init__(self) -> None:
        self.appends: dict[
            tuple[int, int, int],
            list[
                tuple[
                    object,
                    OriginNode,
                    Mapping[object, object],
                    tuple[tuple[object, ...], OriginNode],
                ]
            ],
        ] = {}
        self.fragments: dict[
            tuple[int, int, int],
            list[
                tuple[
                    Mapping[str, object],
                    OriginNode,
                    Mapping[str, object],
                    tuple[
                        Mapping[str, object],
                        OriginNode,
                        tuple[tuple[OriginSegment, ...], ...],
                    ],
                ]
            ],
        ] = {}
        self.origin_nodes: dict[
            int, list[tuple[object, OriginNode]]
        ] = {}


def _merge_mapping(
    base: Mapping[str, object],
    base_origins: OriginNode,
    patch: Mapping[str, object],
    *,
    origin: Origin,
    context: _MergeContext,
    diagnostic_prefix: tuple[OriginSegment, ...],
) -> tuple[
    Mapping[str, object], OriginNode, tuple[tuple[OriginSegment, ...], ...]
]:
    cache_key = (id(base), id(base_origins), id(patch))
    for (
        cached_base,
        cached_origins,
        cached_patch,
        cached_result,
    ) in context.fragments.get(cache_key, ()):
        if (
            cached_base is base
            and cached_origins is base_origins
            and cached_patch is patch
        ):
            return cached_result
    merged = dict(base)
    children = dict(base_origins.children)
    deletions: list[tuple[OriginSegment, ...]] = []
    for key, value in patch.items():
        if not isinstance(key, str):
            raise ConfigError(
                f"layering key at {diagnostic_prefix or ('<root>',)!r} must "
                f"be a string; got {type(key).__name__}."
            )
        if key.startswith("~"):
            target = _deletion_target(key)
            if value is not None:
                rendered = ".".join(map(str, (*diagnostic_prefix, key)))
                raise ConfigError(f"{rendered}: deletion value must be null.")
            merged.pop(target, None)
            children.pop(target, None)
            deletions.append((target,))
            continue
        inherited = merged.get(key)
        inherited_origin = children.get(key)
        if isinstance(value, Mapping) and "append" in value:
            if inherited_origin is None:
                inherited = ()
                inherited_origin = _origin_node(
                    inherited, origin, context.origin_nodes
                )
            merged[key], children[key] = _append_value(
                key=key,
                inherited=inherited,
                inherited_origin=inherited_origin,
                given=value,
                origin=origin,
                context=context,
            )
            continue
        if (
            isinstance(value, Mapping)
            and isinstance(inherited, Mapping)
            and inherited_origin is not None
        ):
            child_document, child_origin, child_deletions = _merge_mapping(
                inherited,
                inherited_origin,
                value,
                origin=origin,
                context=context,
                diagnostic_prefix=(*diagnostic_prefix, key),
            )
            merged[key] = child_document
            children[key] = child_origin
            deletions.extend((key, *path) for path in child_deletions)
            continue
        merged[key] = value
        children[key] = _origin_node(value, origin, context.origin_nodes)
    result = (
        MappingProxyType(merged),
        _trusted_origin_node(
            origin=base_origins.origin,
            children=_frozen_children(children),
        ),
        tuple(deletions),
    )
    context.fragments.setdefault(cache_key, []).append(
        (base, base_origins, patch, result)
    )
    return result


def merge_with_origins(
    parent: MergeResult,
    patch: Mapping[str, object],
    *,
    origin: Origin,
) -> MergeResult:
    if not isinstance(parent, MergeResult):
        raise ConfigError(
            "merge_with_origins: parent is a MergeResult; got "
            f"{type(parent).__name__}."
        )
    if not isinstance(patch, Mapping):
        raise ConfigError(
            "merge_with_origins: patch is a mapping; got "
            f"{type(patch).__name__}."
        )
    origin = _canonical_origin(origin, where="merge_with_origins origin")
    frozen_patch = freeze_evidence(patch, where="merge_with_origins patch")
    assert isinstance(frozen_patch, Mapping)
    document, origins, relative_deletions = _merge_mapping(
        parent.document,
        parent.origins,
        frozen_patch,
        origin=origin,
        context=_MergeContext(),
        diagnostic_prefix=(),
    )
    deletions = (*parent.deletions, *(DeletionRecord(path, origin) for path in relative_deletions))
    return _trusted_merge_result(
        document=document,
        origins=origins,
        deletions=tuple(deletions),
    )


def origins_at(origins: OriginNode, path: Sequence[OriginSegment]) -> Origin:
    node = origins
    traversed: list[OriginSegment] = []
    try:
        segments = tuple(path)
    except Exception:
        raise ConfigError("origin path sequence traversal failed.") from None
    for segment in segments:
        exact_segment = _canonical_segment(
            segment, where="origin path segment"
        )
        traversed.append(exact_segment)
        try:
            node = node.children[exact_segment]
        except KeyError:
            raise ConfigError(f"origin path {tuple(traversed)!r} does not exist.") from None
        except Exception:
            raise ConfigError("origin child lookup failed.") from None
    if node.origin is None:
        raise ConfigError("origin path must identify a document value, not the root.")
    return node.origin


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


def _compatibility_has_exact_type(
    value: object, candidates: tuple[type, ...]
) -> bool:
    value_type = type(value)
    return any(value_type is candidate for candidate in candidates)


_COMPATIBILITY_MISSING = object()
_COMPATIBILITY_UNSAFE_LAZY_ANNOTATIONS = object()


def _compatibility_raw_mro_descriptor(
    value_type: type, name: str
) -> object:
    bases = type.__getattribute__(value_type, "__mro__")
    for base in bases:
        namespace = type.__getattribute__(base, "__dict__")
        try:
            return namespace[name]
        except KeyError:
            continue
    return _COMPATIBILITY_MISSING


def _compatibility_has_mro_identity(
    value: object, candidates: tuple[type, ...]
) -> bool:
    try:
        bases = type.__getattribute__(type(value), "__mro__")
    except Exception:
        raise ConfigError(_COMPATIBILITY_FAILURE) from None
    return any(
        base is candidate for base in bases for candidate in candidates
    )


def _compatibility_type_name(value: object) -> str:
    try:
        name = type.__getattribute__(type(value), "__name__")
        if type(name) is not str:
            raise TypeError
        return name
    except Exception:
        raise ConfigError(_COMPATIBILITY_FAILURE) from None


def _compatibility_has_mro_base(
    value: object, candidates: tuple[type, ...]
) -> bool:
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
    return _compatibility_has_mro_base(
        value, (dict, MutableMapping)
    )


def _compatibility_pairs(
    mapping: Mapping,
) -> dict[str, object]:
    canonical: dict[str, object] = {}
    for key, value in _mapping_pairs(
        mapping, failure=_COMPATIBILITY_FAILURE
    ):
        if not _compatibility_has_mro_identity(key, (str,)):
            raise ConfigError(
                "merge_extends: keys are strings; got "
                f"{_compatibility_type_name(key)}."
            )
        exact_key = str.__str__(key)
        if exact_key in canonical:
            raise ConfigError(
                "merge_extends: keys collide after canonicalization."
            )
        canonical[exact_key] = value
    return canonical


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
                deepcopy_protocol = getattr(
                    type(item), "__deepcopy__", None
                )
            except Exception:
                raise ConfigError(_COMPATIBILITY_FAILURE) from None
            if deepcopy_protocol is not None:
                return
            canonical = _compatibility_pairs(item)
            for nested in canonical.values():
                discover(nested)
        elif _compatibility_has_mro_identity(
            item, (list, tuple, set, frozenset)
        ):
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
    if not _compatibility_is_mapping(
        detached_parent
    ) or not _compatibility_is_mapping(detached_child):
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
            pending.extend(
                (item, field_role, 2) for item in state_mappings
            )
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
        if role == content_role and _compatibility_has_identity(
            public_identities, item
        ):
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
            forbidden is not None
            and _compatibility_has_identity(forbidden, item)
        ):
            return True
        if _compatibility_has_exact_type(
            item, _COMPATIBILITY_OWNERSHIP_SCALAR_TYPES
        ):
            continue
        is_known_container = (
            _compatibility_is_mapping(item)
            or _compatibility_has_mro_identity(
                item, (list, tuple, set, frozenset)
            )
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
            ) = (
                _compatibility_ownership_state(item)
            )
            if state_is_complete:
                pending.extend(
                    (state, field_role, 0) for state in state_values
                )
                pending.extend(
                    (state, field_role, 1)
                    for state in nested_state_mappings
                )
        if not is_known_container:
            continue
        if _compatibility_is_mapping(item):
            if inspect_object_state:
                nested_values = (
                    _compatibility_uncanonicalized_mapping_values(item)
                )
            else:
                nested_values = _compatibility_pairs(item).values()
            child_role = (
                content_role
                if state_mapping_kind == 2
                and public_mapping is not None
                and _compatibility_state_mapping_is_public_view(
                    item, public_mapping
                )
                else field_role
                if state_mapping_kind
                else public_role
                if role == public_role
                else content_role
            )
            pending.extend(
                (nested, child_role, 0) for nested in nested_values
            )
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
            child_role = (
                public_role if role == public_role else content_role
            )
            pending.append((nested, child_role, 0))
    return False


def _compatibility_reset_mapping(
    mapping: MutableMapping, values: Mapping[str, object]
) -> None:
    try:
        if _compatibility_has_mro_identity(mapping, (dict,)):
            dict.clear(mapping)
            dict.update(mapping, values)
        else:
            MutableMapping.clear(mapping)
            MutableMapping.update(mapping, values)
    except Exception:
        raise ConfigError(_COMPATIBILITY_FAILURE) from None


def _compatibility_mapping_get(
    mapping: Mapping, key: str, default: object = None
) -> object:
    try:
        if _compatibility_has_mro_identity(mapping, (dict,)):
            return dict.get(mapping, key, default)
        return Mapping.get(mapping, key, default)
    except Exception:
        raise ConfigError(_COMPATIBILITY_FAILURE) from None


def _compatibility_mapping_pop(
    mapping: MutableMapping, key: str
) -> None:
    try:
        if _compatibility_has_mro_identity(mapping, (dict,)):
            dict.pop(mapping, key, None)
        else:
            MutableMapping.pop(mapping, key, None)
    except Exception:
        raise ConfigError(_COMPATIBILITY_FAILURE) from None


def _compatibility_mapping_set(
    mapping: MutableMapping, key: str, value: object
) -> None:
    try:
        if _compatibility_has_mro_identity(mapping, (dict,)):
            dict.__setitem__(mapping, key, value)
        else:
            mapping[key] = value
    except Exception:
        raise ConfigError(_COMPATIBILITY_FAILURE) from None


def _compatibility_add_identity(
    buckets: dict[int, list[object]], item: object
) -> bool:
    identity = id(item)
    bucket = buckets.setdefault(identity, [])
    if any(source is item for source in bucket):
        return False
    bucket.append(item)
    return True


def _compatibility_has_identity(
    buckets: dict[int, list[object]], item: object
) -> bool:
    return any(
        source is item for source in buckets.get(id(item), ())
    )


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
            or _compatibility_has_mro_identity(
                item, (list, tuple, set, frozenset)
            )
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
            or _compatibility_has_mro_identity(
                item, (list, tuple, set, frozenset)
            )
        ):
            continue
        if not _compatibility_add_identity(expanded, item):
            continue
        if _compatibility_is_mapping(item):
            _compatibility_add_identity(context.must_split_mappings, item)
        pending.extend(_compatibility_iter_values(item))


def _compatibility_builtin_descriptor_value(
    value: object, owner: type, name: str
) -> object:
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
    function_state = _compatibility_builtin_descriptor_value(
        value, FunctionType, "__dict__"
    )
    if type(function_state) is not dict:
        raise ConfigError(_COMPATIBILITY_FAILURE)
    state_mappings.append(function_state)

    closure = _compatibility_builtin_descriptor_value(
        value, FunctionType, "__closure__"
    )
    if closure is not None:
        if type(closure) is not tuple:
            raise ConfigError(_COMPATIBILITY_FAILURE)
        for cell in closure:
            if type(cell) is not CellType:
                raise ConfigError(_COMPATIBILITY_FAILURE)
            state_values.append(cell)
            contents = _compatibility_builtin_descriptor_value(
                cell, CellType, "cell_contents"
            )
            if contents is not _COMPATIBILITY_MISSING:
                state_values.append(contents)

    defaults = _compatibility_builtin_descriptor_value(
        value, FunctionType, "__defaults__"
    )
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

    annotate = _compatibility_builtin_descriptor_value(
        value, FunctionType, "__annotate__"
    )
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
        metadata = _compatibility_builtin_descriptor_value(
            value, FunctionType, name
        )
        if metadata is not _COMPATIBILITY_MISSING and metadata is not None:
            state_values.append(metadata)

    type_parameters = _compatibility_builtin_descriptor_value(
        value, FunctionType, "__type_params__"
    )
    if (
        type_parameters is not _COMPATIBILITY_MISSING
        and type_parameters is not None
    ):
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
    callback = _compatibility_builtin_descriptor_value(
        value, ReferenceType, "__callback__"
    )
    state_values = []
    if referent is not None:
        state_values.append(referent)
    if callback is not _COMPATIBILITY_MISSING and callback is not None:
        state_values.append(callback)
    return state_values


def _compatibility_ownership_state(
    value: object,
) -> tuple[list[object], list[dict], bool, bool]:
    if _compatibility_has_exact_type(
        value, _COMPATIBILITY_OWNERSHIP_SCALAR_TYPES
    ):
        return [], [], False, True
    if _compatibility_has_exact_type(
        value, (FunctionType, BuiltinFunctionType)
    ):
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
        contents = _compatibility_builtin_descriptor_value(
            value, CellType, "cell_contents"
        )
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
        dict_descriptor = _compatibility_raw_mro_descriptor(
            value_type, "__dict__"
        )
    except Exception:
        raise ConfigError(_COMPATIBILITY_FAILURE) from None
    descriptor_type = type(dict_descriptor)
    if any(
        descriptor_type is candidate
        for candidate in (GetSetDescriptorType, MemberDescriptorType)
    ):
        state_bearing = True
        try:
            instance_state = descriptor_type.__get__(
                dict_descriptor, value, value_type
            )
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
                state = MemberDescriptorType.__get__(
                    descriptor, value, value_type
                )
            except AttributeError:
                continue
            except Exception:
                raise ConfigError(_COMPATIBILITY_FAILURE) from None
            state_values.append(state)
    if _compatibility_has_mro_identity(value, (memoryview,)):
        try:
            descriptor = type.__getattribute__(memoryview, "__dict__")[
                "obj"
            ]
            backing = GetSetDescriptorType.__get__(
                descriptor, value, value_type
            )
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
        state_values.extend(
            _compatibility_referential_atomic_state(value)
        )
        state_bearing = True
    return state_values, state_mappings, state_bearing, state_is_complete


def _compatibility_instance_state_values(value: object) -> list[object]:
    state_values, state_mappings, _, state_is_complete = (
        _compatibility_ownership_state(value)
    )
    if not state_is_complete:
        raise ConfigError(_COMPATIBILITY_FAILURE)
    return [*state_values, *state_mappings]


def _compatibility_uncanonicalized_mapping_values(
    mapping: Mapping,
) -> list[object]:
    return [
        value
        for _, value in _mapping_pairs(
            mapping, failure=_COMPATIBILITY_FAILURE
        )
    ]


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
        if _compatibility_has_exact_type(
            item, _COMPATIBILITY_OWNERSHIP_SCALAR_TYPES
        ):
            continue
        if not _compatibility_add_identity(seen, item):
            continue
        is_mapping = _compatibility_is_mapping(item)
        is_copy_atomic = _compatibility_has_exact_type(
            item, _COMPATIBILITY_OWNERSHIP_COPY_ATOMIC_TYPES
        )
        is_immutable_container = _compatibility_has_mro_identity(
            item, (tuple, frozenset)
        )
        is_static_namespace = _compatibility_has_mro_identity(
            item, (type, ModuleType)
        )
        state_values, state_mappings, _, state_is_complete = (
            _compatibility_ownership_state(item)
        )
        if not (
            is_copy_atomic
            or is_immutable_container
            or is_static_namespace
        ):
            _compatibility_add_identity(mutables, item)
        if state_is_complete:
            pending.extend(state_values)
            pending.extend(state_mappings)
        if is_mapping:
            pending.extend(
                _compatibility_uncanonicalized_mapping_values(item)
            )
        elif _compatibility_has_mro_identity(
            item, (list, tuple, set, frozenset)
        ):
            pending.extend(_compatibility_iter_values(item))
    return mutables


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
        state_values = [
            state for state in state_values if state is not copied_data
        ]
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
        if (
            type(copied_data) is not dict
            or _compatibility_has_identity(
                context.original_mutables, copied_data
            )
        ):
            raise ConfigError(_COMPATIBILITY_FAILURE)
    state_values, state_mappings, _, state_is_complete = (
        _compatibility_ownership_state(mapping)
    )
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
                type.__getattribute__(owner, "__dict__")[name]
                for owner in expected_owners
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
    if not _compatibility_has_mro_identity(
        parent, (dict,)
    ) and type(parent) is not UserDict:
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
    _compatibility_validate_split_state(
        merged, parent, context=context
    )
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
        raise ConfigError(
            "merge_extends: overlapping cyclic mappings cannot be merged."
        )
    active_bucket.append(child)
    try:
        parent_values = _compatibility_pairs(parent)
        if type(parent) is not dict and _compatibility_has_mro_identity(
            parent, (dict,)
        ):
            _compatibility_validate_dict_read_protocols(parent)
        requires_split = not reuse_parent and (
            _compatibility_occurrence_count(
                context.mapping_occurrences, parent
            )
            > 1
            or _compatibility_has_identity(
                context.must_split_mappings, parent
            )
        )
        if requires_split:
            if _compatibility_reaches_mapping(
                parent_values.values(), parent
            ):
                raise ConfigError(
                    "merge_extends: overlapping cyclic mappings cannot be merged."
                )
            _compatibility_mark_mapping_descendants(
                parent_values.values(), context=context
            )
        if reuse_parent and _compatibility_has_mro_identity(
            parent, (dict,)
        ):
            merged = parent
        elif reuse_parent:
            merged = dict(parent_values)
        elif requires_split:
            merged = _compatibility_split_mapping(
                parent, parent_values, context=context
            )
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
                _compatibility_pairs(value)
                if _compatibility_is_mapping(value)
                else None
            )
            if value_mapping is not None and "append" in value_mapping:
                if set(value_mapping) != {"append"}:
                    siblings = sorted(set(value_mapping) - {"append"})
                    raise ConfigError(
                        f"{key!r}: append must be the only key when extending a list; "
                        f"got the sibling keys {siblings}."
                    )
                appended = value_mapping["append"]
                if not _compatibility_has_mro_identity(
                    appended, (list, tuple)
                ):
                    raise ConfigError(
                        f"{key!r}: append is a sequence; got "
                        f"{_compatibility_type_name(appended)}."
                    )
                inherited = _compatibility_mapping_get(merged, key, [])
                if not _compatibility_has_mro_identity(
                    inherited, (list,)
                ):
                    raise ConfigError(
                        f"{key!r} is extended with {{append: ...}} but the inherited "
                        f"value is {_compatibility_type_name(inherited)}, not a list."
                    )
                if _compatibility_has_identity(
                    context.original_mutables, inherited
                ):
                    raise ConfigError(_COMPATIBILITY_FAILURE)
                try:
                    list.extend(inherited, appended)
                except Exception:
                    raise ConfigError(_COMPATIBILITY_FAILURE) from None
                _compatibility_mapping_set(merged, key, inherited)
                continue
            inherited = _compatibility_mapping_get(merged, key)
            if _compatibility_is_mapping(
                value
            ) and _compatibility_is_mapping(inherited):
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


def merge_extends(child: dict, parent: dict) -> dict:
    """Deep-merge ``child`` over ``parent`` on the lossless public boundary."""
    for label, given in (("child", child), ("parent", parent)):
        if not _compatibility_is_mapping(given):
            raise ConfigError(
                f"merge_extends: {label} is a mapping; got "
                f"{_compatibility_type_name(given)}."
            )
    try:
        (
            detached_parent,
            detached_child,
            original_mutables,
        ) = _compatibility_deepcopy_roots(parent, child)
        context = _CompatibilityMergeContext(
            active_children={},
            mapping_occurrences=_compatibility_mapping_occurrences(
                (detached_parent, detached_child)
            ),
            must_split_mappings={},
            original_mutables=original_mutables,
        )
        return _merge_extends_compat(
            detached_child,
            detached_parent,
            context=context,
            reuse_parent=True,
        )
    except ConfigError:
        raise
    except RecursionError as exc:
        raise ConfigError(
            "merge_extends: value graph recursion exceeds the supported depth."
        ) from exc


def recursive_update(base: Mapping, patch: Mapping) -> dict:
    """Return ``patch`` deep-merged over ``base`` without mutating either."""
    for label, given in (("base", base), ("patch", patch)):
        if not _compatibility_is_mapping(given):
            raise ConfigError(
                f"recursive_update: {label} is a mapping; got "
                f"{_compatibility_type_name(given)}."
            )
    return merge_extends(patch, base)


def apply_variant(document: Mapping, name: str) -> dict:
    """Return the document with one named one-level variant patch applied."""
    if not isinstance(document, Mapping):
        raise ConfigError(
            "variant document is a mapping; got "
            f"{type(document).__name__}."
        )
    if not isinstance(name, str):
        raise ConfigError(
            "variant name is a string; got " f"{type(name).__name__}."
        )
    name = str.__str__(name)

    def mapping_values(mapping: Mapping) -> dict[str, object]:
        canonical: dict[str, object] = {}
        for key, value in _mapping_pairs(
            mapping, failure="apply_variant: mapping traversal failed."
        ):
            if not isinstance(key, str):
                raise ConfigError(
                    "apply_variant: mapping keys are strings; got "
                    f"{type(key).__name__}."
                )
            exact_key = str.__str__(key)
            if exact_key in canonical:
                raise ConfigError(
                    "apply_variant: mapping keys collide after canonicalization."
                )
            canonical[exact_key] = value
        return canonical

    document_values = mapping_values(document)
    variants = document_values.get("variants", {})
    if not isinstance(variants, Mapping):
        raise ConfigError(
            f"variants: is a mapping of name -> patch; got "
            f"{type(variants).__name__}."
        )
    variant_values = mapping_values(variants)
    if not variant_values:
        raise ConfigError(
            f"variant {name!r} was requested but this document declares no variants."
        )
    if name not in variant_values:
        raise ConfigError(
            f"variant {name!r} is not declared; this document declares "
            f"{sorted(variant_values)}."
        )
    patch = variant_values[name]
    if not isinstance(patch, Mapping):
        raise ConfigError(
            f"variant {name!r}: the patch is a mapping of sections; got "
            f"{type(patch).__name__}."
        )
    patch_values = mapping_values(patch)
    for key in ("variants", "~variants"):
        if key in patch_values:
            raise ConfigError(
                f"variant {name!r} declares {key!r}. Layering is one level "
                "deep by design: there is no ordering between variants and "
                "no variant builds on another, so a comparison's halves "
                "cannot drift apart through a chain."
            )
    for key in ("schema_version", "~schema_version"):
        if key in patch_values:
            raise ConfigError(
                f"variant {name!r} touches {key!r}. The version belongs to "
                "the document; a patch that changes -- or deletes -- how the "
                "document is read is not a patch."
            )
    return recursive_update(document, patch)


def parse_default(raw: object) -> PresetRequest:
    if isinstance(raw, str):
        return PresetRequest(name=raw, only=None)
    if not isinstance(raw, Mapping):
        raise ConfigError(
            "defaults: each entry is a preset name or a {from:, only:} mapping."
        )
    canonical: dict[str, object] = {}
    for key, value in _mapping_pairs(
        raw, failure="defaults: preset entry mapping traversal failed."
    ):
        if not isinstance(key, str):
            raise ConfigError(
                "defaults: preset entry keys are strings; got "
                f"{type(key).__name__}."
            )
        exact_key = str.__str__(key)
        if exact_key in canonical:
            raise ConfigError(
                "defaults: preset entry keys collide after canonicalization."
            )
        canonical[exact_key] = value
    unknown = sorted(set(canonical) - {"from", "only"})
    if unknown:
        raise ConfigError(f"defaults: preset entry has unknown keys {unknown}.")
    if "from" not in canonical:
        raise ConfigError("defaults: preset entry requires from:.")
    name = canonical["from"]
    if "only" not in canonical:
        return PresetRequest(name=name, only=None)  # type: ignore[arg-type]
    only = canonical["only"]
    if not isinstance(only, list | tuple) or isinstance(only, str):
        raise ConfigError("defaults: only: is a sequence of dotted paths.")
    return PresetRequest(name=name, only=only)  # type: ignore[arg-type]


def _select_only(
    request: PresetRequest, document: Mapping[str, object]
) -> dict[str, object]:
    if request.only is None:
        return dict(document)
    paths = [tuple(str.split(path, ".")) for path in request.only]
    terminal = object()
    trie: dict[object, object] = {}
    for parts in paths:
        rendered = ".".join(parts)
        if parts[0] == "model" and len(parts) != 1:
            raise ConfigError(
                f"defaults preset {request.name!r}: select model as a whole; "
                f"{rendered!r} is a partial model selection."
            )
        node = trie
        for part in parts:
            prior = node.get(terminal)
            if isinstance(prior, str):
                raise ConfigError(
                    f"defaults preset {request.name!r}: only paths "
                    f"{prior!r} and {rendered!r} overlap."
                )
            child = node.get(part)
            if child is None:
                child = {}
                node[part] = child
            assert isinstance(child, dict)
            node = child
        if terminal in node:
            raise ConfigError(
                f"defaults preset {request.name!r}: only path {rendered!r} is duplicate."
            )
        if node:
            descendant = node
            while terminal not in descendant:
                descendant = next(
                    child
                    for key, child in descendant.items()
                    if key is not terminal
                )
                assert isinstance(descendant, dict)
            prior = descendant[terminal]
            assert isinstance(prior, str)
            raise ConfigError(
                f"defaults preset {request.name!r}: only paths "
                f"{rendered!r} and {prior!r} overlap."
            )
        node[terminal] = rendered

    selected: dict[str, object] = {}
    for parts in paths:
        value: object = document
        for part in parts:
            if not isinstance(value, Mapping) or part not in value:
                raise ConfigError(
                    f"defaults preset {request.name!r}: only path "
                    f"{'.'.join(parts)!r} does not exist."
                )
            value = value[part]
        cursor = selected
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})  # type: ignore[assignment]
        cursor[parts[-1]] = value
    return selected


def _without_key(result: MergeResult, key: str) -> MergeResult:
    document = dict(result.document)
    document.pop(key, None)
    children = dict(result.origins.children)
    children.pop(key, None)
    return _trusted_merge_result(
        document=MappingProxyType(document),
        origins=_trusted_origin_node(None, _frozen_children(children)),
        deletions=tuple(result.deletions),
    )


def _replace_key(
    result: MergeResult,
    key: str,
    value: object,
    *,
    origin: Origin,
) -> MergeResult:
    return merge_with_origins(
        _without_key(result, key), {key: value}, origin=origin
    )


def _replace_key_with_node(
    result: MergeResult, key: str, value: object, node: OriginNode
) -> MergeResult:
    without = _without_key(result, key)
    document = dict(without.document)
    document[key] = value
    children = dict(without.origins.children)
    children[key] = node
    return _trusted_merge_result(
        document=MappingProxyType(document),
        origins=_trusted_origin_node(None, _frozen_children(children)),
        deletions=tuple(result.deletions),
    )


def _apply_user_model(
    result: MergeResult, user_model: object, *, origin: Origin
) -> MergeResult:
    if user_model is None:
        return _replace_key(result, "model", None, origin=origin)
    if not isinstance(user_model, Mapping):
        raise ConfigError("model: is a mapping when package presets are layered.")
    if "inherit" not in user_model:
        return _replace_key(result, "model", user_model, origin=origin)
    inherit = user_model["inherit"]
    if not isinstance(inherit, list | tuple) or isinstance(inherit, str):
        raise ConfigError("model.inherit: is a sequence of model node names.")
    candidate = result.document.get("model")
    candidate_node = result.origins.children.get("model")
    if not isinstance(candidate, Mapping) or candidate_node is None:
        candidate = {}
        candidate_node = _trusted_origin_node(
            origin=None, children=_frozen_children({})
        )
    inherited: dict[str, object] = {}
    inherited_children: dict[str, OriginNode] = {}
    seen: set[str] = set()
    for name in inherit:
        if not isinstance(name, str) or not name:
            raise ConfigError(f"model.inherit: node name {name!r} is not a non-empty string.")
        if name in seen:
            raise ConfigError(f"model.inherit: node {name!r} is repeated.")
        seen.add(name)
        if name not in candidate:
            raise ConfigError(f"model.inherit: candidate node {name!r} is absent.")
        inherited[name] = candidate[name]
        inherited_children[name] = candidate_node.children[name]
    declared = {key: value for key, value in user_model.items() if key != "inherit"}
    model_origin = candidate_node.origin if inherited else origin
    seed_node = _trusted_origin_node(
        origin=model_origin,
        children=_frozen_children(inherited_children),
    )
    seeded = _replace_key_with_node(result, "model", inherited, seed_node)
    return merge_with_origins(seeded, {"model": declared}, origin=origin)


def layer_presets(
    document: Mapping[str, object],
    requests: Sequence[PresetRequest],
    *,
    preset_provider: Callable[[str], PresetSnapshot],
) -> tuple[MergeResult, Sequence[tuple[PresetRequest, PresetSnapshot]]]:
    """Layer selected presets in order and the user document last."""
    if not isinstance(document, Mapping):
        raise ConfigError("configuration document is a mapping before preset layering.")
    user_evidence = freeze_evidence(document, where="layer_presets document")
    assert isinstance(user_evidence, Mapping)
    try:
        requests = tuple(requests)
    except Exception:
        raise ConfigError(
            "defaults: request sequence traversal failed."
        ) from None
    result = initial_merge({}, origin=Origin("rheplicant-default"))
    selected: list[tuple[PresetRequest, PresetSnapshot]] = []
    names: set[str] = set()
    for request in requests:
        if not isinstance(request, PresetRequest):
            raise ConfigError(
                "defaults: requests must contain PresetRequest values; got "
                f"{type(request).__name__}."
            )
        if request.name in names:
            raise ConfigError(
                f"defaults: package preset {request.name!r} appears more than once."
            )
        names.add(request.name)
        try:
            provided = preset_provider(request.name)
        except ConfigError:
            raise
        except Exception:
            raise ConfigError(
                f"defaults: preset provider failed for {request.name!r}."
            ) from None
        if type(provided) is not PresetSnapshot:
            raise ConfigError(
                "defaults: preset provider returned "
                f"{type(provided).__name__}; expected PresetSnapshot."
            )
        try:
            snapshot = PresetSnapshot(
                name=provided.name,
                resource=provided.resource,
                input_bytes=provided.input_bytes,
                sha256=provided.sha256,
                document=provided.document,
                expanded_nodes=provided.expanded_nodes,
            )
        except ConfigError:
            raise
        except Exception:
            raise ConfigError(
                f"defaults: preset provider snapshot for {request.name!r} "
                "could not be validated."
            ) from None
        if snapshot.name != request.name:
            raise ConfigError(
                f"defaults: preset provider returned {snapshot.name!r} for "
                f"request {request.name!r}."
            )
        chosen = _select_only(request, snapshot.document)
        missing_model = object()
        model = chosen.pop("model", missing_model)
        preset_origin = Origin("preset", request.name)
        result = merge_with_origins(result, chosen, origin=preset_origin)
        if model is not missing_model:
            result = _replace_key(result, "model", model, origin=preset_origin)
        selected.append((request, snapshot))

    user = dict(user_evidence)
    has_user_model = "model" in user
    user_model = user.pop("model", None)
    result = merge_with_origins(result, user, origin=Origin("user"))
    if has_user_model:
        result = _apply_user_model(result, user_model, origin=Origin("user"))
    return result, tuple(selected)


__all__ = [
    "DeletionRecord",
    "MergeResult",
    "OriginNode",
    "apply_variant",
    "initial_merge",
    "layer_presets",
    "merge_extends",
    "merge_with_origins",
    "origins_at",
    "parse_default",
    "recursive_update",
]
