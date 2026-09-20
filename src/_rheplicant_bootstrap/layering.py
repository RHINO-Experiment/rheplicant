"""JAX-free document layering with immutable per-value origin evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import (
    MappingProxyType,
)
from typing import TypeAlias, cast

from _rheplicant_bootstrap.errors import ConfigError
from _rheplicant_bootstrap.frozen import _FrozenConcat, freeze_evidence
from _rheplicant_bootstrap.layering_copy import (
    _compatibility_deepcopy_roots,
    _compatibility_mapping_occurrences,
    _CompatibilityMergeContext,
    _merge_extends_compat,
)
from _rheplicant_bootstrap.layering_keys import _deletion_target
from _rheplicant_bootstrap.layering_origins import (
    DeletionRecord,
    OriginNode,
    _canonical_origin,
    _canonical_segment,
    _canonicalize_origin_tree,
    _DeletionLedger,
    _trusted_origin_node,
)
from _rheplicant_bootstrap.layering_overlay import (
    _OverlayBuilder,
    _OverlayMapping,
    _trusted_overlay_root,
)
from _rheplicant_bootstrap.layering_probe import (
    _compatibility_is_mapping,
    _compatibility_type_name,
)
from _rheplicant_bootstrap.types import Origin

OriginSegment: TypeAlias = str | int


def _is_frozen_sequence(value: object) -> bool:
    return type(value) is tuple or type(value) is _FrozenConcat


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
        if isinstance(self.deletions, str | bytes) or not isinstance(self.deletions, Sequence):
            raise ConfigError("merge result deletions must be a sequence.")
        try:
            deletions = tuple(self.deletions)
        except Exception:
            raise ConfigError("merge result deletions sequence traversal failed.") from None
        if any(not isinstance(item, DeletionRecord) for item in deletions):
            raise ConfigError("merge result deletions must contain DeletionRecord values.")
        deletions = tuple(DeletionRecord(item.path, item.origin) for item in deletions)
        origins = _canonicalize_origin_tree(self.origins)
        document = freeze_evidence(self.document, where="merge result document")
        assert isinstance(document, Mapping)
        _validate_parallel_origin_tree(document, origins)
        object.__setattr__(self, "document", document)
        object.__setattr__(self, "origins", origins)
        object.__setattr__(self, "deletions", deletions)


_CANONICAL_FRESH = object()
_CANONICAL_RUNNING = object()
_CANONICAL_COMPLETED = object()
_CANONICAL_FAILED = object()
_CANONICAL_CONSUMED = object()


class _CanonicalVariantDocument(Mapping[str, object]):
    """Private one-shot bridge from compatibility grammar to origin evidence."""

    __slots__ = ("_name", "_parent", "_patch", "_payload", "_state")

    def __init__(self, parent: MergeResult, name: str, patch: object) -> None:
        if type(parent) is not MergeResult:
            raise ConfigError("canonical variant parent must be an exact MergeResult.")
        if type(parent.deletions) is not _DeletionLedger:
            raise ConfigError("canonical variant parent must carry the trusted deletion ledger.")
        if not isinstance(name, str):
            raise ConfigError("canonical variant name must be a string.")
        exact_name = str.__str__(name)
        if not exact_name:
            raise ConfigError("canonical variant name must be non-empty.")
        self._parent = parent
        self._name = exact_name
        self._patch = patch
        self._payload: tuple[MergeResult, dict[str, object]] | None = None
        self._state = _CANONICAL_FRESH

    def __getitem__(self, key: str) -> object:
        return self._parent.document[key]

    def __iter__(self):
        return iter(self._parent.document)

    def __len__(self) -> int:
        return len(self._parent.document)


def _canonical_variant_document(
    parent: MergeResult,
    name: str,
    patch: object,
) -> Mapping[str, object]:
    return _CanonicalVariantDocument(parent, name, patch)


def _take_canonical_variant_result(
    document: Mapping[str, object],
    returned: object,
) -> MergeResult:
    if type(document) is not _CanonicalVariantDocument:
        raise ConfigError("canonical variant result requires the exact private document.")
    if document._state is _CANONICAL_CONSUMED:
        raise ConfigError("canonical variant result was already taken.")
    if document._state is not _CANONICAL_COMPLETED or document._payload is None:
        raise ConfigError("canonical variant result was not produced.")
    result, expected_return = document._payload
    if returned is not expected_return:
        document._payload = None
        document._state = _CANONICAL_FAILED
        raise ConfigError("canonical variant return identity did not match.")
    document._payload = None
    document._state = _CANONICAL_CONSUMED
    return result


def _trusted_merge_result(
    document: Mapping[str, object],
    origins: OriginNode,
    deletions: Sequence[DeletionRecord],
) -> MergeResult:
    """Build an internal result from lockstep COW fragments already validated."""
    if type(deletions) is not tuple and type(deletions) is not _DeletionLedger:
        raise ConfigError("trusted merge deletions must be an exact tuple or ledger.")
    trusted = type(deletions) is _DeletionLedger
    allowed_mapping_types = (_OverlayMapping,) if trusted else (MappingProxyType,)
    if not any(type(document) is candidate for candidate in allowed_mapping_types):
        raise ConfigError("trusted merge document has an invalid exact type.")
    if type(origins) is not OriginNode or origins.origin is not None:
        raise ConfigError("trusted merge origins must be an exact root node.")
    if not any(type(origins.children) is candidate for candidate in allowed_mapping_types):
        raise ConfigError("trusted merge origin children have an invalid exact type.")
    if len(document) != len(origins.children):
        raise ConfigError("trusted merge document and origin roots must have equal length.")
    result = object.__new__(MergeResult)
    object.__setattr__(result, "document", document)
    object.__setattr__(result, "origins", origins)
    object.__setattr__(result, "deletions", deletions)
    return result


def _canonical_variant_parent(parent: MergeResult) -> MergeResult:
    if type(parent) is not MergeResult or type(parent.deletions) is not tuple:
        raise ConfigError("canonical variant parent must have exact public merge evidence.")
    document = _trusted_overlay_root(parent.document)
    origin_children = _trusted_overlay_root(parent.origins.children)
    return _trusted_merge_result(
        document,
        _trusted_origin_node(None, origin_children),
        _DeletionLedger(parent.deletions, ()),
    )


def _validate_parallel_origin_tree(document: Mapping[str, object], origins: OriginNode) -> None:
    pending: list[tuple[object, OriginNode, bool]] = [(document, origins, True)]
    seen_pairs: dict[tuple[int, int], list[tuple[object, OriginNode]]] = {}
    document_origins: dict[int, list[tuple[object, OriginNode]]] = {}
    while pending:
        value, node, is_root = pending.pop()
        if not isinstance(node, OriginNode):
            raise ConfigError("merge result origin tree children must be OriginNode values.")
        if is_root:
            if node.origin is not None:
                raise ConfigError("merge result origin tree root origin must be null.")
        elif node.origin is None:
            raise ConfigError("merge result origin tree descendants must have concrete origins.")
        else:
            _canonical_origin(node.origin, where="merge result origin tree descendant origin")

        is_alias_container = isinstance(value, Mapping) or (
            _is_frozen_sequence(value) and len(value) != 0
        )
        if is_alias_container:
            document_identity = id(value)
            origin_identity = id(node)
            origins_bucket = document_origins.setdefault(document_identity, [])
            prior = next(
                (prior_node for prior_value, prior_node in origins_bucket if prior_value is value),
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
            raise ConfigError("merge result origin tree children traversal failed.") from None
        if isinstance(value, Mapping):
            expected = tuple(value)
            child_values = value
        elif _is_frozen_sequence(value):
            expected = tuple(range(len(value)))
            child_values = None
        else:
            expected = ()
            child_values = None
        expected_type = str if isinstance(value, Mapping) else int
        if len(given_segments) != len(expected) or any(
            type(segment) is not expected_type for segment in given_segments
        ):
            raise ConfigError("merge result origin tree children must exactly match the document.")
        for segment in expected:
            try:
                child_node = children[segment]
            except (KeyError, TypeError):
                raise ConfigError(
                    "merge result origin tree children must exactly match the document."
                ) from None
            except Exception:
                raise ConfigError("merge result origin tree child lookup failed.") from None
            child_value = child_values[segment] if child_values is not None else value[segment]
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
    memoized = isinstance(value, Mapping) or (_is_frozen_sequence(value) and len(value) != 0)
    if memoized:
        for source, node in memo.get(identity, ()):
            if source is value:
                return node
    if isinstance(value, Mapping):
        children = {key: _origin_node(item, origin, memo) for key, item in value.items()}
    elif isinstance(value, list | tuple) or type(value) is _FrozenConcat:
        children = {index: _origin_node(item, origin, memo) for index, item in enumerate(value)}
    else:
        children = {}
    result = _trusted_origin_node(origin=origin, children=_frozen_children(children))
    if memoized:
        memo.setdefault(identity, []).append((value, result))
    return result


def _root_node(document: Mapping[str, object], origin: Origin) -> OriginNode:
    memo: dict[int, list[tuple[object, OriginNode]]] = {}
    return _trusted_origin_node(
        origin=None,
        children=_frozen_children(
            {key: _origin_node(value, origin, memo) for key, value in document.items()}
        ),
    )


def initial_merge(document: Mapping[str, object], *, origin: Origin) -> MergeResult:
    if not isinstance(document, Mapping):
        raise ConfigError(f"initial_merge: document is a mapping; got {type(document).__name__}.")
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
) -> tuple[Sequence[object], OriginNode]:
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
            f"{key!r}: append is a sequence; got {type(appended).__name__} ({appended!r})."
        )
    if not _is_frozen_sequence(inherited):
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
    values: Sequence[object]
    if context.trusted:
        if type(inherited) is tuple:
            values = _FrozenConcat(inherited, appended)
        else:
            values = cast(_FrozenConcat, inherited).extend(appended)
        inherited_children: dict[OriginSegment, OriginNode] | _OverlayBuilder = _OverlayBuilder(
            context.overlay_base(inherited_origin.children)
        )
    else:
        values = tuple([*inherited, *appended])
        inherited_children = dict(inherited_origin.children)
    offset = len(inherited)
    for index, item in enumerate(appended):
        inherited_children[offset + index] = _origin_node(item, origin, context.origin_nodes)
    origin_children = (
        inherited_children.publish()
        if type(inherited_children) is _OverlayBuilder
        else _frozen_children(inherited_children)
    )
    result = (
        values,
        _trusted_origin_node(
            origin=inherited_origin.origin,
            children=origin_children,
        ),
    )
    context.appends.setdefault(cache_key, []).append((inherited, inherited_origin, given, result))
    return result


_ORDINARY_STORAGE = object()
_TRUSTED_STORAGE = object()


class _MergeContext:
    __slots__ = ("appends", "fragments", "origin_nodes", "storage")

    def __init__(self, *, storage: object) -> None:
        if storage is not _ORDINARY_STORAGE and storage is not _TRUSTED_STORAGE:
            raise ConfigError("merge storage policy marker is invalid.")
        self.storage = storage
        self.appends: dict[
            tuple[int, int, int],
            list[
                tuple[
                    object,
                    OriginNode,
                    Mapping[object, object],
                    tuple[Sequence[object], OriginNode],
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
        self.origin_nodes: dict[int, list[tuple[object, OriginNode]]] = {}

    @property
    def trusted(self) -> bool:
        return self.storage is _TRUSTED_STORAGE

    def overlay_base(self, base: Mapping[OriginSegment, object]) -> _OverlayMapping:
        if not self.trusted:
            raise ConfigError("ordinary merge cannot request trusted storage.")
        if type(base) is _OverlayMapping:
            return base
        if type(base) is MappingProxyType:
            return _trusted_overlay_root(base)
        raise ConfigError("trusted merge base must be an exact mapping proxy or overlay.")


def _merge_mapping(
    base: Mapping[str, object],
    base_origins: OriginNode,
    patch: Mapping[str, object],
    *,
    origin: Origin,
    context: _MergeContext,
    diagnostic_prefix: tuple[OriginSegment, ...],
) -> tuple[Mapping[str, object], OriginNode, tuple[tuple[OriginSegment, ...], ...]]:
    cache_key = (id(base), id(base_origins), id(patch))
    for (
        cached_base,
        cached_origins,
        cached_patch,
        cached_result,
    ) in context.fragments.get(cache_key, ()):
        if cached_base is base and cached_origins is base_origins and cached_patch is patch:
            return cached_result
    if context.trusted:
        trusted_base = context.overlay_base(base)
        trusted_children = context.overlay_base(base_origins.children)
        merged: dict[str, object] | _OverlayBuilder = _OverlayBuilder(trusted_base)
        children: dict[OriginSegment, OriginNode] | _OverlayBuilder = _OverlayBuilder(
            trusted_children
        )
    else:
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
                inherited_origin = _origin_node(inherited, origin, context.origin_nodes)
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
    merged_document = (
        cast(_OverlayBuilder, merged).publish()
        if type(merged) is _OverlayBuilder
        else MappingProxyType(merged)
    )
    merged_children = (
        cast(_OverlayBuilder, children).publish()
        if type(children) is _OverlayBuilder
        else _frozen_children(children)
    )
    result = (
        cast(Mapping[str, object], merged_document),
        _trusted_origin_node(
            origin=base_origins.origin,
            children=cast(Mapping[OriginSegment, OriginNode], merged_children),
        ),
        tuple(deletions),
    )
    context.fragments.setdefault(cache_key, []).append((base, base_origins, patch, result))
    return result


def merge_with_origins(
    parent: MergeResult,
    patch: Mapping[str, object],
    *,
    origin: Origin,
) -> MergeResult:
    if not isinstance(parent, MergeResult):
        raise ConfigError(
            f"merge_with_origins: parent is a MergeResult; got {type(parent).__name__}."
        )
    if not isinstance(patch, Mapping):
        raise ConfigError(f"merge_with_origins: patch is a mapping; got {type(patch).__name__}.")
    if type(parent.deletions) is _DeletionLedger and (
        type(parent.document) is not _OverlayMapping
        or type(parent.origins) is not OriginNode
        or parent.origins.origin is not None
        or type(parent.origins.children) is not _OverlayMapping
    ):
        raise ConfigError("trusted merge parent must carry exact overlay roots.")
    origin = _canonical_origin(origin, where="merge_with_origins origin")
    frozen_patch = freeze_evidence(patch, where="merge_with_origins patch")
    assert isinstance(frozen_patch, Mapping)
    document, origins, relative_deletions = _merge_mapping(
        parent.document,
        parent.origins,
        frozen_patch,
        origin=origin,
        context=_MergeContext(
            storage=(
                _TRUSTED_STORAGE if type(parent.deletions) is _DeletionLedger else _ORDINARY_STORAGE
            )
        ),
        diagnostic_prefix=(),
    )
    suffix = tuple(DeletionRecord(path, origin) for path in relative_deletions)
    if type(parent.deletions) is _DeletionLedger:
        deletions: Sequence[DeletionRecord] = parent.deletions.extend(suffix)
    else:
        deletions = (*parent.deletions, *suffix)
    return _trusted_merge_result(
        document=document,
        origins=origins,
        deletions=deletions,
    )


def origins_at(origins: OriginNode, path: Sequence[OriginSegment]) -> Origin:
    node = origins
    traversed: list[OriginSegment] = []
    try:
        segments = tuple(path)
    except Exception:
        raise ConfigError("origin path sequence traversal failed.") from None
    for segment in segments:
        exact_segment = _canonical_segment(segment, where="origin path segment")
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


def merge_extends(child: dict, parent: dict) -> dict:
    """Deep-merge ``child`` over ``parent`` on the lossless public boundary."""
    for label, given in (("child", child), ("parent", parent)):
        if not _compatibility_is_mapping(given):
            raise ConfigError(
                f"merge_extends: {label} is a mapping; got {_compatibility_type_name(given)}."
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
                f"recursive_update: {label} is a mapping; got {_compatibility_type_name(given)}."
            )
    return merge_extends(patch, base)


#: ``apply_variant``, ``layer_presets`` and ``parse_default`` moved to
#: :mod:`_rheplicant_bootstrap.layering_variants` with the variant grammar
#: (A1, section 3.2). They are NOT re-exported here: this module would have to
#: import the one that imports it, and a cycle held open by import order is a
#: worse thing to leave behind than six updated import lines.
__all__ = [
    "DeletionRecord",
    "MergeResult",
    "OriginNode",
    "initial_merge",
    "merge_extends",
    "merge_with_origins",
    "origins_at",
    "recursive_update",
]
