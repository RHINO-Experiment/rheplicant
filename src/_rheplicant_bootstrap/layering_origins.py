"""Per-value origin evidence: which layer a value in the merged document came from.

A1 named this one directly -- "the origin record alone is a module". It is the
half of layering that answers "where did this come from", against the half
that answers "which one wins": an immutable tree parallel to the document, a
node per value, and the ledger of deletions that the tree has to stay
consistent with.

The separation is not cosmetic. A merge that produced the wrong VALUE is
caught by the document; a merge that produced the right value with the wrong
ORIGIN is caught by nothing except the checks in here, which is why they are
this defensive.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TypeAlias, cast

from _rheplicant_bootstrap.errors import ConfigError
from _rheplicant_bootstrap.layering_probe import _mapping_pairs
from _rheplicant_bootstrap.types import Origin

OriginSegment: TypeAlias = str | int


def _canonical_origin(value: object, *, where: str) -> Origin:
    if not isinstance(value, Origin):
        raise ConfigError(f"{where} must be an Origin; got {type(value).__name__}.")
    if not isinstance(value.kind, str):
        raise ConfigError(f"{where} kind must be a string; got {type(value.kind).__name__}.")
    kind = str.__str__(value.kind)
    if value.name is None:
        name = None
    elif isinstance(value.name, str):
        name = str.__str__(value.name)
    else:
        raise ConfigError(
            f"{where} name must be a string or null; got {type(value.name).__name__}."
        )
    try:
        canonical = Origin(kind, name)
    except ValueError as exc:
        raise ConfigError(f"{where} is invalid: {exc}") from exc
    if (
        type(value) is Origin
        and type(value.kind) is str
        and (value.name is None or type(value.name) is str)
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


def _detach_origin_tree(root: OriginNode, *, public_origin_node: bool) -> OriginNode:
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
            for segment, child in _mapping_pairs(node.children, failure=traversal_failure):
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
                        else "merge result origin tree segments collide after canonicalization."
                    )
                canonical[exact_segment] = rebuild(child)
            result = _trusted_origin_node(origin, MappingProxyType(canonical))
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
            _canonical_segment(segment, where="deletion path segment") for segment in given
        )
        origin = _canonical_origin(self.origin, where="deletion origin")
        object.__setattr__(self, "path", canonical)
        object.__setattr__(self, "origin", origin)


def _validate_deletion_ledger_chunk(rows: object, *, where: str) -> tuple[DeletionRecord, ...]:
    if type(rows) is not tuple:
        raise ConfigError(f"canonical deletion ledger {where} must be an exact tuple.")
    if any(type(row) is not DeletionRecord for row in rows):
        raise ConfigError(
            f"canonical deletion ledger {where} must contain exact DeletionRecord values."
        )
    return rows


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class _DeletionLedger(Sequence[DeletionRecord]):
    """Persistent private deletion evidence: one shared root plus suffixes."""

    _parent: tuple[DeletionRecord, ...] | _DeletionLedger
    _suffix: tuple[DeletionRecord, ...]
    _length: int

    def __init__(
        self,
        parent: tuple[DeletionRecord, ...] | _DeletionLedger,
        suffix: tuple[DeletionRecord, ...],
    ) -> None:
        canonical_parent: tuple[DeletionRecord, ...] | _DeletionLedger
        if type(parent) is tuple:
            canonical_parent = _validate_deletion_ledger_chunk(parent, where="parent")
        elif type(parent) is _DeletionLedger:
            canonical_parent = parent
        else:
            raise ConfigError("canonical deletion ledger parent must be an exact tuple or ledger.")
        canonical_suffix = _validate_deletion_ledger_chunk(suffix, where="suffix")
        object.__setattr__(self, "_parent", canonical_parent)
        object.__setattr__(self, "_suffix", canonical_suffix)
        object.__setattr__(
            self,
            "_length",
            len(canonical_parent) + len(canonical_suffix),
        )

    def extend(self, suffix: tuple[DeletionRecord, ...]) -> _DeletionLedger:
        if type(suffix) is tuple and not suffix:
            return self
        return _DeletionLedger(self, suffix)

    def __len__(self) -> int:
        return self._length

    def __iter__(self):
        pending: list[tuple[DeletionRecord, ...] | _DeletionLedger] = [self]
        while pending:
            current = pending.pop()
            if type(current) is _DeletionLedger:
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
                raise TypeError("canonical deletion ledger slice bounds must be exact int or null.")
            return tuple(self)[index]
        if type(index) is not int:
            raise TypeError("canonical deletion ledger indices must be exact int or slice.")
        position = index
        if position < 0:
            position += self._length
        if position < 0 or position >= self._length:
            raise IndexError("tuple index out of range")
        current: tuple[DeletionRecord, ...] | _DeletionLedger = self
        while type(current) is _DeletionLedger:
            parent_length = len(current._parent)
            if position < parent_length:
                current = current._parent
                continue
            return current._suffix[position - parent_length]
        return tuple.__getitem__(cast(tuple[DeletionRecord, ...], current), position)

    def __eq__(self, other: object) -> bool:
        if type(other) is not tuple and type(other) is not _DeletionLedger:
            return NotImplemented
        if len(self) != len(other):
            return False
        return all(left == right for left, right in zip(self, other, strict=True))

    __hash__ = None  # type: ignore[assignment]


def _canonicalize_origin_tree(root: OriginNode) -> OriginNode:
    return _detach_origin_tree(root, public_origin_node=False)
