"""The read-only overlay a merge hands out instead of the document itself.

A merged document is built once and read many times, and the readers are not
all trusted -- a pre-flight check, a plugin, a template. This is the proxy
they get: a mapping that answers from the merged layers without copying them
and without letting anything write.

Separated from the merge for the same reason as the rest of the split (A1,
section 3.2): building the overlay and deciding which value wins are different
questions, and only the second one is about layering at all.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import cast

from _rheplicant_bootstrap.errors import ConfigError
from _rheplicant_bootstrap.layering_origins import OriginSegment


def _is_exact_overlay_key(value: object) -> bool:
    return type(value) is str or type(value) is int


def _overlay_keys(
    mapping: Mapping[OriginSegment, object], *, where: str
) -> tuple[OriginSegment, ...]:
    try:
        keys = tuple(mapping)
    except Exception:
        raise ConfigError(f"overlay {where} traversal failed.") from None
    if any(not _is_exact_overlay_key(key) for key in keys):
        raise ConfigError("overlay mapping keys must be exact string or integer values.")
    return keys


def _snapshot_overlay_proxy(mapping: MappingProxyType, *, where: str) -> MappingProxyType:
    """Copy one untrusted proxy traversal before validating its snapshot."""
    try:
        iterator = iter(mapping)
    except Exception:
        raise ConfigError(f"overlay {where} protocol failed.") from None
    snapshot: dict[OriginSegment, object] = {}
    while True:
        try:
            key = next(iterator)
        except StopIteration:
            break
        except Exception:
            raise ConfigError(f"overlay {where} protocol failed.") from None
        if not _is_exact_overlay_key(key):
            raise ConfigError("overlay mapping keys must be exact string or integer values.")
        if key in snapshot:
            raise ConfigError(f"overlay {where} traversal emitted a duplicate key.")
        try:
            snapshot[key] = mapping[key]
        except Exception:
            raise ConfigError(f"overlay {where} protocol failed.") from None
    return MappingProxyType(snapshot)


def _validate_overlay_parts(
    base: Mapping[OriginSegment, object],
    delta: MappingProxyType,
    hidden: frozenset[OriginSegment],
    end_keys: tuple[OriginSegment, ...],
    *,
    validate_base: bool,
) -> int:
    if type(base) is not MappingProxyType and type(base) is not _OverlayMapping:
        raise ConfigError("overlay base must be an exact mapping proxy or overlay.")
    if type(delta) is not MappingProxyType:
        raise ConfigError("overlay delta must be an exact mapping proxy.")
    if type(hidden) is not frozenset:
        raise ConfigError("overlay hidden keys must be an exact frozenset.")
    if type(end_keys) is not tuple:
        raise ConfigError("overlay end keys must be an exact tuple.")
    if validate_base and type(base) is MappingProxyType:
        _overlay_keys(base, where="base")
    delta_keys = _overlay_keys(delta, where="delta")
    if any(not _is_exact_overlay_key(key) for key in hidden):
        raise ConfigError("overlay mapping keys must be exact string or integer values.")
    if any(not _is_exact_overlay_key(key) for key in end_keys):
        raise ConfigError("overlay mapping keys must be exact string or integer values.")
    try:
        if len(set(end_keys)) != len(end_keys):
            raise ConfigError("overlay end keys must be unique.")
        end_set = set(end_keys)
        if any(key not in base for key in hidden):
            raise ConfigError("overlay hidden keys must exist in the base.")
        if any(key not in delta for key in end_keys):
            raise ConfigError("overlay end keys must identify delta values.")
        for key in end_keys:
            if key in base and key not in hidden:
                raise ConfigError("overlay end keys must be new or re-added values.")
        for key in delta_keys:
            if key in end_set:
                continue
            if key not in base or key in hidden:
                raise ConfigError("overlay delta replacements must retain a live base key.")
        return len(base) - len(hidden) + len(end_keys)
    except ConfigError:
        raise
    except Exception:
        raise ConfigError("overlay storage protocol failed.") from None


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class _OverlayMapping(Mapping[OriginSegment, object]):
    """Private immutable mapping overlay with final ``dict`` order."""

    _base: MappingProxyType | _OverlayMapping
    _delta: MappingProxyType
    _hidden: frozenset[OriginSegment]
    _end_keys: tuple[OriginSegment, ...]
    _length: int

    def __init__(
        self,
        base: Mapping[OriginSegment, object],
        delta: MappingProxyType,
        hidden: frozenset[OriginSegment],
        end_keys: tuple[OriginSegment, ...],
    ) -> None:
        if type(base) is not MappingProxyType and type(base) is not _OverlayMapping:
            raise ConfigError("overlay base must be an exact mapping proxy or overlay.")
        if type(delta) is not MappingProxyType:
            raise ConfigError("overlay delta must be an exact mapping proxy.")
        if type(hidden) is not frozenset:
            raise ConfigError("overlay hidden keys must be an exact frozenset.")
        if type(end_keys) is not tuple:
            raise ConfigError("overlay end keys must be an exact tuple.")
        canonical_base: MappingProxyType | _OverlayMapping = (
            _snapshot_overlay_proxy(base, where="base")
            if type(base) is MappingProxyType
            else cast(_OverlayMapping, base)
        )
        canonical_delta = _snapshot_overlay_proxy(delta, where="delta")
        length = _validate_overlay_parts(
            canonical_base,
            canonical_delta,
            hidden,
            end_keys,
            validate_base=False,
        )
        object.__setattr__(self, "_base", canonical_base)
        object.__setattr__(self, "_delta", canonical_delta)
        object.__setattr__(self, "_hidden", hidden)
        object.__setattr__(self, "_end_keys", end_keys)
        object.__setattr__(self, "_length", length)

    def __len__(self) -> int:
        return self._length

    def __iter__(self):
        for key in self._base:
            if key not in self._hidden:
                yield key
        yield from tuple.__iter__(self._end_keys)

    def __getitem__(self, key: OriginSegment) -> object:
        if not _is_exact_overlay_key(key):
            raise KeyError("overlay keys are exact strings or integers")
        if key in self._delta:
            return self._delta[key]
        if key in self._hidden:
            raise KeyError(key)
        return self._base[key]

    __hash__ = None  # type: ignore[assignment]


def _trusted_overlay(
    base: Mapping[OriginSegment, object],
    delta: MappingProxyType,
    hidden: frozenset[OriginSegment],
    end_keys: tuple[OriginSegment, ...],
) -> Mapping[OriginSegment, object]:
    if type(base) is not _OverlayMapping:
        raise ConfigError("trusted overlay base must be the exact private marker.")
    if type(delta) is not MappingProxyType:
        raise ConfigError("overlay delta must be an exact mapping proxy.")
    if type(hidden) is not frozenset:
        raise ConfigError("overlay hidden keys must be an exact frozenset.")
    if type(end_keys) is not tuple:
        raise ConfigError("overlay end keys must be an exact tuple.")
    canonical_delta = _snapshot_overlay_proxy(delta, where="delta")
    if len(canonical_delta) == 0 and len(hidden) == 0 and len(end_keys) == 0:
        return base
    length = _validate_overlay_parts(
        base,
        canonical_delta,
        hidden,
        end_keys,
        validate_base=False,
    )
    overlay = object.__new__(_OverlayMapping)
    object.__setattr__(overlay, "_base", base)
    object.__setattr__(overlay, "_delta", canonical_delta)
    object.__setattr__(overlay, "_hidden", hidden)
    object.__setattr__(overlay, "_end_keys", end_keys)
    object.__setattr__(overlay, "_length", length)
    return overlay


def _trusted_overlay_root(
    base: Mapping[OriginSegment, object],
) -> _OverlayMapping:
    if type(base) is not MappingProxyType:
        raise ConfigError("trusted overlay root must wrap an exact mapping proxy.")
    root = object.__new__(_OverlayMapping)
    object.__setattr__(root, "_base", base)
    object.__setattr__(root, "_delta", MappingProxyType({}))
    object.__setattr__(root, "_hidden", frozenset())
    object.__setattr__(root, "_end_keys", ())
    object.__setattr__(root, "_length", len(base))
    return root


def _trusted_overlay_omit(
    base: Mapping[OriginSegment, object], key: OriginSegment
) -> Mapping[OriginSegment, object]:
    if not _is_exact_overlay_key(key):
        raise ConfigError("trusted overlay omission key must be an exact string or integer.")
    if type(base) is not _OverlayMapping:
        raise ConfigError("trusted overlay omission base must be the exact private marker.")
    if key not in base:
        return base
    return _trusted_overlay(
        base,
        MappingProxyType({}),
        frozenset({key}),
        (),
    )


class _OverlayBuilder:
    """Mutable one-merge builder that publishes one immutable overlay."""

    __slots__ = ("_base", "_delta", "_end_keys", "_hidden")

    def __init__(self, base: Mapping[OriginSegment, object]) -> None:
        if type(base) is not _OverlayMapping:
            raise ConfigError("trusted overlay builder base must be the exact private marker.")
        self._base = base
        self._delta: dict[OriginSegment, object] = {}
        self._hidden: set[OriginSegment] = set()
        self._end_keys: dict[OriginSegment, None] = {}

    def get(self, key: OriginSegment, default: object = None) -> object:
        try:
            return self[key]
        except KeyError:
            return default

    def __getitem__(self, key: OriginSegment) -> object:
        if not _is_exact_overlay_key(key):
            raise KeyError("trusted overlay keys must be exact strings or integers.")
        if key in self._delta:
            return self._delta[key]
        if key in self._hidden:
            raise KeyError(key)
        return self._base[key]

    def __setitem__(self, key: OriginSegment, value: object) -> None:
        if not _is_exact_overlay_key(key):
            raise ConfigError("trusted overlay keys must be exact strings or integers.")
        if key in self._delta:
            self._delta[key] = value
            return
        if key in self._hidden:
            self._delta[key] = value
            self._end_keys[key] = None
            return
        if key in self._base:
            self._delta[key] = value
            return
        self._delta[key] = value
        self._end_keys[key] = None

    def pop(self, key: OriginSegment, default: object = None) -> object:
        try:
            previous = self[key]
        except KeyError:
            return default
        if key in self._delta:
            del self._delta[key]
            if key in self._end_keys:
                del self._end_keys[key]
                if key in self._base:
                    self._hidden.add(key)
            else:
                self._hidden.add(key)
        else:
            self._hidden.add(key)
        return previous

    def publish(self) -> Mapping[OriginSegment, object]:
        return _trusted_overlay(
            self._base,
            MappingProxyType(dict(self._delta)),
            frozenset(self._hidden),
            tuple(self._end_keys),
        )
