"""JAX-free audited plugin imports and their closed JSON projection."""

from __future__ import annotations

import hashlib
import importlib
import importlib.machinery
import inspect
import json
import keyword
import math
import os
import re
import stat
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path, PurePosixPath
from types import (
    GetSetDescriptorType,
    MappingProxyType,
    MemberDescriptorType,
    ModuleType,
)
from typing import Literal, TypeAlias, cast

from _rheplicant_bootstrap.errors import ConfigError
from _rheplicant_bootstrap.frozen import (
    _freeze_evidence_roots,
    freeze_evidence,
    static_class_attribute,
    static_class_mro,
    static_class_text,
    static_isinstance,
    static_type_name,
)
from _rheplicant_bootstrap.types import JsonValue

PluginOriginReason: TypeAlias = Literal["no_origin", "namespace_package"]
PluginLoaderReason: TypeAlias = Literal[
    "no_origin",
    "namespace_package",
    "generated_module",
]
PluginPathReason: TypeAlias = Literal[
    "no_origin",
    "namespace_package",
    "generated_module",
    "not_regular_file",
    "unreadable",
]
PluginHashReason: TypeAlias = Literal[
    "no_origin",
    "namespace_package",
    "generated_module",
    "not_regular_file",
    "unreadable",
    "extension_module",
]
PluginVersionReason: TypeAlias = Literal["not_installed", "unreadable"]
PluginDirectUrlReason: TypeAlias = Literal[
    "not_installed",
    "missing_direct_url",
    "unreadable",
]

PLUGIN_DISTRIBUTION_ROW_KEYS = (
    "name",
    "version",
    "version_reason",
    "direct_url",
    "direct_url_reason",
)
PLUGIN_ROW_KEYS = (
    "name",
    "already_imported",
    "origin",
    "origin_reason",
    "loader_type",
    "loader_type_reason",
    "resolved_path",
    "resolved_path_reason",
    "distributions",
    "distributions_reason",
    "code_hash",
    "code_hash_reason",
    "unobserved_io",
)

_ORIGIN_REASONS = frozenset(("no_origin", "namespace_package"))
_LOADER_REASONS = frozenset(
    ("no_origin", "namespace_package", "generated_module")
)
_PATH_REASONS = frozenset(
    (
        "no_origin",
        "namespace_package",
        "generated_module",
        "not_regular_file",
        "unreadable",
    )
)
_HASH_REASONS = frozenset((*_PATH_REASONS, "extension_module"))
_VERSION_REASONS = frozenset(("not_installed", "unreadable"))
_DIRECT_URL_REASONS = frozenset(
    ("not_installed", "missing_direct_url", "unreadable")
)
_PEP_503_RUN = re.compile(r"[-_.]+")
_LOWER_SHA256 = re.compile(r"[0-9a-f]{64}")
_GENERATED_ORIGINS = frozenset(("built-in", "frozen"))
_EXTENSION_SUFFIXES = tuple(
    str.__str__(suffix)
    for suffix in tuple(importlib.machinery.EXTENSION_SUFFIXES)
    if isinstance(suffix, str)
)
_EXTENSION_FILE_LOADER = importlib.machinery.ExtensionFileLoader
_MISSING = object()
_METADATA_EVIDENCE_LIMIT = 250_000
_DIRECT_URL_TEXT_LIMIT = 1024 * 1024
_DIRECT_URL_INTEGER_BIT_LIMIT = math.ceil(
    _DIRECT_URL_TEXT_LIMIT * math.log2(10)
)
_DIRECT_URL_DEPTH_LIMIT = 100
_FOREIGN_EXCEPTION_DETAIL_LIMIT = 1024


class _MetadataBudgetExceeded(ConfigError):
    pass


class _MetadataBudget:
    __slots__ = ("_limit", "_used")

    def __init__(self) -> None:
        limit = _METADATA_EVIDENCE_LIMIT
        if type(limit) is not int or limit < 1:
            raise ConfigError(
                "plugins: distribution metadata evidence budget is invalid."
            )
        self._limit = limit
        self._used = 0

    def consume(self) -> None:
        used = self._used + 1
        if used > self._limit:
            raise _MetadataBudgetExceeded(
                "plugins: distribution metadata evidence budget exceeds limit "
                f"{self._limit}."
            )
        self._used = used

    def ensure_remaining(self, count: int) -> None:
        if count > self._limit - self._used:
            raise _MetadataBudgetExceeded(
                "plugins: distribution metadata evidence budget exceeds limit "
                f"{self._limit}."
            )


def _validate_utf8_text(value: str, *, where: str) -> None:
    try:
        str.encode(value, "utf-8", "strict")
    except UnicodeEncodeError:
        raise ConfigError(f"{where} must contain only valid UTF-8 text.") from None


def _is_utf8_text(value: str) -> bool:
    try:
        str.encode(value, "utf-8", "strict")
    except UnicodeEncodeError:
        return False
    return True


def _canonical_text(value: object, *, where: str, nonempty: bool = True) -> str:
    if not static_isinstance(value, str):
        raise ConfigError(f"{where} must be a string.")
    canonical = str.__str__(value)
    if nonempty and not canonical:
        raise ConfigError(f"{where} must be a non-empty string.")
    _validate_utf8_text(canonical, where=where)
    return canonical


def _canonical_module_name(value: object) -> str:
    if not static_isinstance(value, str):
        raise ConfigError("plugins: entries must be dot-separated Python module names.")
    canonical = str.__str__(value)
    _validate_utf8_text(canonical, where="plugin module name")
    parts = str.split(canonical, ".")
    if not canonical or any(
        not part or not str.isidentifier(part) or keyword.iskeyword(part)
        for part in parts
    ):
        raise ConfigError(
            "plugins: entries must be non-empty dot-separated Python module names."
        )
    return canonical


def _normalize_distribution_name(value: object, *, require_normalized: bool) -> str:
    canonical = _canonical_text(value, where="plugin distribution name")
    normalized = _PEP_503_RUN.sub("-", str.lower(canonical))
    if not normalized:
        raise ConfigError("plugin distribution name must be non-empty.")
    if require_normalized and canonical != normalized:
        raise ConfigError(
            "plugin distribution name must be PEP-503-normalized."
        )
    return normalized


def _canonical_reason(
    value: object,
    *,
    where: str,
    allowed: frozenset[str],
) -> str | None:
    if value is None:
        return None
    if not static_isinstance(value, str):
        raise ConfigError(f"{where} reason is invalid.")
    canonical = str.__str__(value)
    if canonical not in allowed:
        raise ConfigError(f"{where} reason is invalid.")
    return canonical


def _value_reason(
    value: object,
    reason: object,
    *,
    where: str,
    allowed: frozenset[str],
) -> tuple[object, str | None]:
    canonical_reason = _canonical_reason(reason, where=where, allowed=allowed)
    if (value is None) is (canonical_reason is None):
        raise ConfigError(
            f"plugin {where} value and reason must be exact complements."
        )
    return value, canonical_reason


def _validate_json_text(value: str, *, where: str) -> None:
    if len(value) > _DIRECT_URL_TEXT_LIMIT:
        raise ConfigError(
            f"{where} scalar exceeds the {_DIRECT_URL_TEXT_LIMIT}-byte limit."
        )
    try:
        encoded = str.encode(value, "utf-8", "strict")
    except UnicodeEncodeError:
        raise ConfigError(f"{where} must contain only valid UTF-8 text.") from None
    if len(encoded) > _DIRECT_URL_TEXT_LIMIT:
        raise ConfigError(
            f"{where} scalar exceeds the {_DIRECT_URL_TEXT_LIMIT}-byte limit."
        )


def _validate_frozen_json(
    value: object,
    *,
    where: str,
    budget: _MetadataBudget | None = None,
    visited_nodes: set[int] | None = None,
    require_frozen: bool = False,
) -> None:
    """Validate the exact output of ``freeze_evidence`` as finite JSON."""
    visited = set() if visited_nodes is None else visited_nodes
    active_containers: set[int] = set()

    def validate(item: object, depth: int) -> None:
        identity = id(item)
        if identity in active_containers:
            raise ConfigError(f"{where} contains cyclic JSON evidence.")
        if identity in visited:
            return
        if depth > _DIRECT_URL_DEPTH_LIMIT:
            raise ConfigError(
                f"{where} depth exceeds limit {_DIRECT_URL_DEPTH_LIMIT}."
            )
        if budget is not None:
            budget.consume()
        item_type = type(item)
        if item is None or item_type is bool:
            visited.add(identity)
            return
        if item_type is str:
            _validate_json_text(cast(str, item), where=where)
            visited.add(identity)
            return
        if item_type is int:
            visited.add(identity)
            return
        if item_type is float:
            if not math.isfinite(cast(float, item)):
                raise ConfigError(f"{where} must contain only finite JSON numbers.")
            visited.add(identity)
            return
        is_mapping = (
            item_type is MappingProxyType
            if require_frozen
            else static_isinstance(item, Mapping)
        )
        if is_mapping:
            active_containers.add(identity)
            try:
                iterator = iter(item.items())
            except Exception:
                active_containers.remove(identity)
                raise ConfigError(f"{where} JSON mapping traversal failed.") from None
            try:
                while True:
                    try:
                        pair = next(iterator)
                    except StopIteration:
                        break
                    except Exception:
                        raise ConfigError(
                            f"{where} JSON mapping traversal failed."
                        ) from None
                    if budget is not None:
                        budget.consume()
                    try:
                        key, child = pair
                    except Exception:
                        raise ConfigError(
                            f"{where} JSON mapping traversal failed."
                        ) from None
                    if type(key) is not str:
                        raise ConfigError(
                            f"{where} must have string JSON object keys."
                        )
                    validate(key, depth + 1)
                    validate(child, depth + 1)
            finally:
                active_containers.remove(identity)
            visited.add(identity)
            return
        if type(item) is tuple:
            active_containers.add(identity)
            try:
                for child in tuple.__iter__(item):
                    if budget is not None:
                        budget.consume()
                    validate(child, depth + 1)
            finally:
                active_containers.remove(identity)
            visited.add(identity)
            return
        raise ConfigError(f"{where} contains a value that is not JSON.")

    validate(value, 0)


def _freeze_direct_url(
    value: object,
    *,
    where: str,
    budget: _MetadataBudget | None = None,
) -> Mapping[str, JsonValue]:
    if not static_isinstance(value, Mapping):
        raise ConfigError(f"{where} must be a JSON object.")
    frozen_roots = _freeze_evidence_roots(
        [value],
        where=where,
        text_limit=_DIRECT_URL_TEXT_LIMIT,
        json_only=True,
        integer_bit_limit=_DIRECT_URL_INTEGER_BIT_LIMIT,
        consume=None if budget is None else budget.consume,
    )
    frozen = frozen_roots[0]
    if not static_isinstance(frozen, Mapping):
        raise ConfigError(f"{where} must be a JSON object.")
    _validate_frozen_json(frozen, where=where)
    return cast(Mapping[str, JsonValue], frozen)


@dataclass(frozen=True, slots=True)
class PluginDistributionRecord:
    name: str
    version: str | None
    version_reason: PluginVersionReason | None
    direct_url: Mapping[str, JsonValue] | None
    direct_url_reason: PluginDirectUrlReason | None

    def __post_init__(self) -> None:
        name = _normalize_distribution_name(self.name, require_normalized=True)
        raw_version, version_reason = _value_reason(
            self.version,
            self.version_reason,
            where="distribution version",
            allowed=_VERSION_REASONS,
        )
        version = (
            None
            if raw_version is None
            else _canonical_text(
                raw_version, where="plugin distribution version"
            )
        )
        raw_direct_url, direct_url_reason = _value_reason(
            self.direct_url,
            self.direct_url_reason,
            where="distribution direct_url",
            allowed=_DIRECT_URL_REASONS,
        )
        direct_url = (
            None
            if raw_direct_url is None
            else _freeze_direct_url(
                raw_direct_url, where="plugin distribution direct_url"
            )
        )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "version_reason", version_reason)
        object.__setattr__(self, "direct_url", direct_url)
        object.__setattr__(self, "direct_url_reason", direct_url_reason)


@dataclass(frozen=True, slots=True)
class _DistributionSnapshot:
    name: object
    version: object
    version_reason: object
    direct_url: object
    direct_url_reason: object


def _snapshot_distribution_record(value: object) -> _DistributionSnapshot:
    if type(value) is not PluginDistributionRecord:
        raise ConfigError(
            "plugin distributions must contain exact PluginDistributionRecord values."
        )
    try:
        return _DistributionSnapshot(
            name=object.__getattribute__(value, "name"),
            version=object.__getattribute__(value, "version"),
            version_reason=object.__getattribute__(value, "version_reason"),
            direct_url=object.__getattribute__(value, "direct_url"),
            direct_url_reason=object.__getattribute__(value, "direct_url_reason"),
        )
    except Exception:
        raise ConfigError("plugin distribution record is malformed.") from None


def _copy_distribution_record(
    value: object,
    *,
    direct_url_copy: object = _MISSING,
) -> PluginDistributionRecord:
    if type(value) is not _DistributionSnapshot:
        raise ConfigError("plugin distribution snapshot is malformed.")
    try:
        raw_direct_url = object.__getattribute__(value, "direct_url")
        raw_direct_url_reason = object.__getattribute__(
            value, "direct_url_reason"
        )
        direct_url_reason = (
            raw_direct_url_reason
            if raw_direct_url is None
            else "missing_direct_url"
        )
        if raw_direct_url is not None:
            _value_reason(
                raw_direct_url,
                raw_direct_url_reason,
                where="distribution direct_url",
                allowed=_DIRECT_URL_REASONS,
            )
        copied = PluginDistributionRecord(
            name=object.__getattribute__(value, "name"),
            version=object.__getattribute__(value, "version"),
            version_reason=object.__getattribute__(value, "version_reason"),
            direct_url=None,
            direct_url_reason=direct_url_reason,
        )
        if raw_direct_url is not None:
            retained_direct_url = (
                raw_direct_url
                if direct_url_copy is _MISSING
                else direct_url_copy
            )
            object.__setattr__(copied, "direct_url", retained_direct_url)
            object.__setattr__(copied, "direct_url_reason", None)
        return copied
    except ConfigError:
        raise
    except Exception:
        raise ConfigError("plugin distribution record is malformed.") from None


@dataclass(frozen=True, slots=True)
class PluginRecord:
    name: str
    already_imported: bool
    origin: str | None
    origin_reason: PluginOriginReason | None
    loader_type: str | None
    loader_type_reason: PluginLoaderReason | None
    resolved_path: str | None
    resolved_path_reason: PluginPathReason | None
    distributions: tuple[PluginDistributionRecord, ...]
    distributions_reason: Literal["no_distribution"] | None
    code_hash: str | None
    code_hash_reason: PluginHashReason | None
    unobserved_io: Literal[True]

    def __post_init__(self) -> None:
        try:
            raw_name = object.__getattribute__(self, "name")
            raw_already_imported = object.__getattribute__(
                self, "already_imported"
            )
            raw_origin_field = object.__getattribute__(self, "origin")
            raw_origin_reason_field = object.__getattribute__(
                self, "origin_reason"
            )
            raw_loader_type_field = object.__getattribute__(
                self, "loader_type"
            )
            raw_loader_type_reason_field = object.__getattribute__(
                self, "loader_type_reason"
            )
            raw_resolved_path_field = object.__getattribute__(
                self, "resolved_path"
            )
            raw_resolved_path_reason_field = object.__getattribute__(
                self, "resolved_path_reason"
            )
            raw_distributions = object.__getattribute__(self, "distributions")
            raw_distributions_reason = object.__getattribute__(
                self, "distributions_reason"
            )
            raw_code_hash_field = object.__getattribute__(self, "code_hash")
            raw_code_hash_reason_field = object.__getattribute__(
                self, "code_hash_reason"
            )
            raw_unobserved_io = object.__getattribute__(
                self, "unobserved_io"
            )
        except Exception:
            raise ConfigError("plugin record is malformed.") from None

        name = _canonical_module_name(raw_name)
        if type(raw_already_imported) is not bool:
            raise ConfigError("plugin already_imported must be a bool.")

        raw_origin, origin_reason = _value_reason(
            raw_origin_field,
            raw_origin_reason_field,
            where="origin",
            allowed=_ORIGIN_REASONS,
        )
        origin = (
            None
            if raw_origin is None
            else _canonical_text(raw_origin, where="plugin origin")
        )
        raw_loader_type, loader_type_reason = _value_reason(
            raw_loader_type_field,
            raw_loader_type_reason_field,
            where="loader_type",
            allowed=_LOADER_REASONS,
        )
        loader_type = (
            None
            if raw_loader_type is None
            else _canonical_text(raw_loader_type, where="plugin loader_type")
        )
        if loader_type is not None:
            loader_module, separator, loader_name = loader_type.rpartition(".")
            if not separator or not loader_module or not loader_name:
                raise ConfigError("plugin loader_type must be fully qualified.")
        raw_resolved_path, resolved_path_reason = _value_reason(
            raw_resolved_path_field,
            raw_resolved_path_reason_field,
            where="resolved_path",
            allowed=_PATH_REASONS,
        )
        resolved_path = (
            None
            if raw_resolved_path is None
            else _canonical_text(
                raw_resolved_path, where="plugin resolved_path"
            )
        )
        if resolved_path is not None and not os.path.isabs(resolved_path):
            raise ConfigError("plugin resolved_path must be absolute.")

        if type(raw_distributions) is not tuple:
            raise ConfigError("plugin distributions must be an exact tuple.")
        distribution_budget = _MetadataBudget()
        distribution_budget.ensure_remaining(tuple.__len__(raw_distributions))
        distribution_snapshots: list[_DistributionSnapshot] = []
        direct_url_roots: list[object] = []
        for raw_distribution in tuple.__iter__(raw_distributions):
            distribution_budget.consume()
            snapshot = _snapshot_distribution_record(raw_distribution)
            distribution_snapshots.append(snapshot)
            raw_direct_url = snapshot.direct_url
            if raw_direct_url is not None:
                if type(raw_direct_url) is not MappingProxyType:
                    raise ConfigError(
                        "plugin distribution direct_url must be recursively frozen."
                    )
                direct_url_roots.append(raw_direct_url)

        frozen_direct_urls: tuple[object, ...] = ()
        if direct_url_roots:
            frozen_roots = _freeze_evidence_roots(
                direct_url_roots,
                where="plugin distribution direct_urls",
                text_limit=_DIRECT_URL_TEXT_LIMIT,
                json_only=True,
                integer_bit_limit=_DIRECT_URL_INTEGER_BIT_LIMIT,
                consume=distribution_budget.consume,
            )
            frozen_direct_urls = frozen_roots
            if tuple.__len__(frozen_direct_urls) != len(direct_url_roots):
                raise ConfigError(
                    "plugin distribution direct_url snapshot is malformed."
                )
            validated_snapshot_nodes: set[int] = set()
            for frozen_direct_url in tuple.__iter__(frozen_direct_urls):
                if type(frozen_direct_url) is not MappingProxyType:
                    raise ConfigError(
                        "plugin distribution direct_url must be a JSON object."
                    )
                _validate_frozen_json(
                    frozen_direct_url,
                    where="plugin distribution direct_url",
                    visited_nodes=validated_snapshot_nodes,
                    require_frozen=True,
                )

        copied_distributions: list[PluginDistributionRecord] = []
        previous_name: str | None = None
        frozen_index = 0
        for snapshot in distribution_snapshots:
            raw_direct_url = snapshot.direct_url
            direct_url_copy: object = _MISSING
            if raw_direct_url is not None:
                direct_url_copy = frozen_direct_urls[frozen_index]
                frozen_index += 1
            distribution = _copy_distribution_record(
                snapshot,
                direct_url_copy=direct_url_copy,
            )
            if previous_name is not None and distribution.name <= previous_name:
                raise ConfigError(
                    "plugin distributions must be normalized, unique, and sorted."
                )
            copied_distributions.append(distribution)
            previous_name = distribution.name
        distributions = tuple(copied_distributions)
        distributions_reason = _canonical_reason(
            raw_distributions_reason,
            where="distributions",
            allowed=frozenset(("no_distribution",)),
        )
        if distributions:
            if distributions_reason is not None:
                raise ConfigError(
                    "plugin distributions must have no reason when present."
                )
        elif distributions_reason != "no_distribution":
            raise ConfigError(
                "plugin distributions must use no_distribution when empty."
            )

        raw_code_hash, code_hash_reason = _value_reason(
            raw_code_hash_field,
            raw_code_hash_reason_field,
            where="code_hash",
            allowed=_HASH_REASONS,
        )
        code_hash = (
            None
            if raw_code_hash is None
            else _canonical_text(raw_code_hash, where="plugin code_hash")
        )
        if code_hash is not None and _LOWER_SHA256.fullmatch(code_hash) is None:
            raise ConfigError(
                "plugin code_hash must be a lowercase 64-character SHA-256."
            )
        if raw_unobserved_io is not True:
            raise ConfigError("plugin unobserved_io must be exactly true.")

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "already_imported", raw_already_imported)
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "origin_reason", origin_reason)
        object.__setattr__(self, "loader_type", loader_type)
        object.__setattr__(self, "loader_type_reason", loader_type_reason)
        object.__setattr__(self, "resolved_path", resolved_path)
        object.__setattr__(self, "resolved_path_reason", resolved_path_reason)
        object.__setattr__(self, "distributions", distributions)
        object.__setattr__(self, "distributions_reason", distributions_reason)
        object.__setattr__(self, "code_hash", code_hash)
        object.__setattr__(self, "code_hash_reason", code_hash_reason)
        object.__setattr__(self, "unobserved_io", raw_unobserved_io)


def _protocol_value(operation, *, where: str):
    try:
        return operation()
    except Exception:
        raise ConfigError(f"plugins: distribution metadata {where} failed.") from None


def _protocol_items(
    value: object,
    *,
    where: str,
    budget: _MetadataBudget,
):
    iterator = _protocol_value(lambda: iter(value), where=where)
    while True:
        try:
            item = next(iterator)
        except StopIteration:
            return
        except Exception:
            raise ConfigError(
                f"plugins: distribution metadata {where} failed."
            ) from None
        budget.consume()
        yield item


def _ordered_metadata_sequence(value: object, *, where: str) -> object:
    if (
        static_isinstance(value, (str, bytes))
        or static_isinstance(value, Mapping)
        or not static_isinstance(value, Sequence)
    ):
        raise ConfigError(
            f"plugins: distribution metadata {where} must be an ordered sequence."
        )
    return value


def _top_level_distribution_names(
    top_level: str, budget: _MetadataBudget
) -> set[str]:
    top_map = _protocol_value(
        metadata.packages_distributions,
        where="packages_distributions inspection",
    )
    if not static_isinstance(top_map, Mapping):
        raise ConfigError(
            "plugins: distribution metadata packages_distributions is not a mapping."
        )
    raw_pairs = _protocol_value(
        lambda: top_map.items(),
        where="packages_distributions traversal",
    )
    raw_names: object | None = None
    found = False
    seen_keys: set[str] = set()
    for pair in _protocol_items(
        raw_pairs,
        where="packages_distributions traversal",
        budget=budget,
    ):
        try:
            raw_key, raw_value = pair
        except Exception:
            raise ConfigError(
                "plugins: distribution metadata packages_distributions "
                "traversal failed."
            ) from None
        key = _canonical_text(
            raw_key,
            where="plugins: distribution metadata top-level key",
        )
        if key in seen_keys:
            raise ConfigError(
                "plugins: distribution metadata top-level keys collide "
                "after canonicalization."
            )
        seen_keys.add(key)
        if key == top_level:
            raw_names = raw_value
            found = True
    if not found:
        return set()
    raw_names = _ordered_metadata_sequence(
        raw_names, where="candidate names"
    )
    names: set[str] = set()
    for raw_name in _protocol_items(
        raw_names, where="candidate traversal", budget=budget
    ):
        names.add(
            _normalize_distribution_name(raw_name, require_normalized=False)
        )
    return names


def _record_text_top_level(
    text: str, budget: _MetadataBudget
) -> str | None:
    if not text or str.startswith(text, "/"):
        return None
    top_start: int | None = None
    top_end: int | None = None
    start = 0
    length = str.__len__(text)
    while start <= length:
        budget.consume()
        end = str.find(text, "/", start)
        if end < 0:
            end = length
        if end > start:
            component_length = end - start
            if component_length == 1 and text[start] == ".":
                pass
            else:
                if (
                    component_length == 2
                    and text[start] == "."
                    and text[start + 1] == "."
                ):
                    return None
                if any(
                    str.find(text, forbidden, start, end) >= 0
                    for forbidden in ("\x00", "\\", ":")
                ):
                    return None
                if top_start is None:
                    top_start = start
                    top_end = end
        if end == length:
            break
        start = end + 1
    if top_start is None or top_end is None:
        return None
    return text[top_start:top_end]


def _record_parts(entry: object, budget: _MetadataBudget) -> str | None:
    if static_isinstance(entry, str):
        return _record_text_top_level(str.__str__(entry), budget)
    try:
        raw_parts = entry.parts
        raw_is_absolute = entry.is_absolute()
    except Exception:
        raise ConfigError(
            "plugins: distribution metadata RECORD path inspection failed."
        ) from None
    if type(raw_is_absolute) is not bool:
        raise ConfigError(
            "plugins: distribution metadata RECORD path inspection failed."
        )
    raw_parts = _ordered_metadata_sequence(
        raw_parts, where="RECORD path components"
    )
    top_name: str | None = None
    for part in _protocol_items(
        raw_parts, where="RECORD path component traversal", budget=budget
    ):
        if not static_isinstance(part, str):
            return None
        exact = str.__str__(part)
        if (
            exact in ("", ".", "..", "/")
            or "\x00" in exact
            or "\\" in exact
            or ":" in exact
        ):
            return None
        if top_name is None:
            top_name = exact
    if raw_is_absolute or top_name is None:
        return None
    return top_name


def _lexical_path(value: object) -> Path:
    try:
        given = Path(value)
        return Path(os.path.abspath(os.fspath(given)))
    except Exception:
        raise ConfigError(
            "plugins: distribution metadata artifact-root inspection failed."
        ) from None


def _distribution_name(distribution: object) -> str:
    try:
        raw_metadata = distribution.metadata
        raw_name = raw_metadata["Name"]
    except Exception:
        raise ConfigError(
            "plugins: distribution metadata name inspection failed."
        ) from None
    return _normalize_distribution_name(raw_name, require_normalized=False)


def _artifact_distribution_candidates(
    resolved_path: str | None,
    budget: _MetadataBudget,
) -> dict[str, object]:
    if resolved_path is None:
        return {}
    raw_distributions = _protocol_value(
        metadata.distributions, where="installed-distribution enumeration"
    )
    candidates: dict[str, object] = {}
    module_path = Path(resolved_path)
    for distribution in _protocol_items(
        raw_distributions,
        where="installed-distribution traversal",
        budget=budget,
    ):
        try:
            files = distribution.files
        except Exception:
            raise ConfigError(
                "plugins: distribution metadata RECORD inspection failed."
            ) from None
        if files is None:
            continue
        files = _ordered_metadata_sequence(files, where="RECORD files")
        claimed = False
        base: tuple[Path, Path] | None = None
        rejected_base = False
        roots: dict[str, Path | None] = {}
        for entry in _protocol_items(
            files, where="RECORD traversal", budget=budget
        ):
            top_name = _record_parts(entry, budget)
            if top_name is None:
                continue
            if rejected_base:
                break
            if base is None:
                try:
                    lexical_base = _lexical_path(
                        distribution.locate_file(PurePosixPath())
                    )
                    resolved_base = lexical_base.resolve(strict=True)
                except Exception:
                    raise ConfigError(
                        "plugins: distribution metadata artifact-root "
                        "inspection failed."
                    ) from None
                if (
                    lexical_base == Path(lexical_base.anchor)
                    or resolved_base == Path(resolved_base.anchor)
                ):
                    rejected_base = True
                    break
                base = (lexical_base, resolved_base)

            if top_name not in roots:
                lexical_base, resolved_base = base
                try:
                    lexical_root = _lexical_path(
                        distribution.locate_file(PurePosixPath(top_name))
                    )
                    if (
                        lexical_root == Path(lexical_root.anchor)
                        or lexical_root != lexical_base / top_name
                    ):
                        roots[top_name] = None
                        continue
                    artifact_root = lexical_root.resolve(strict=True)
                    if artifact_root == resolved_base:
                        roots[top_name] = None
                        continue
                    artifact_root.relative_to(resolved_base)
                except ValueError:
                    roots[top_name] = None
                    continue
                except Exception:
                    raise ConfigError(
                        "plugins: distribution metadata artifact-root "
                        "inspection failed."
                    ) from None
                roots[top_name] = artifact_root

            artifact_root = roots[top_name]
            if artifact_root is None:
                continue
            try:
                module_path.relative_to(artifact_root)
            except ValueError:
                continue
            claimed = True
            break
        if claimed:
            name = _distribution_name(distribution)
            candidates.setdefault(name, distribution)
    return candidates


def _safe_exception_text(exc: Exception) -> str:
    name = static_type_name(exc)
    args = BaseException.args.__get__(exc, BaseException)
    if type(args) is tuple and len(args) == 1:
        detail = args[0]
        if (
            type(detail) is str
            and str.__len__(detail) <= _FOREIGN_EXCEPTION_DETAIL_LIMIT
            and _is_utf8_text(detail)
        ):
            return f"{name}: {detail}"
    return f"{name}: details unavailable"


def _loader_type(loader: object) -> str:
    selected = _loader_class(loader)
    exact_module = static_class_text(
        selected, "__module__", fallback="builtins"
    )
    exact_qualname = static_class_text(
        selected, "__qualname__", fallback="unknown"
    )
    return f"{exact_module}.{exact_qualname}"


def _loader_class(loader: object) -> type:
    if static_class_mro(loader):
        return cast(type, loader)
    return type(loader)


def _module_spec(module: object) -> object | None:
    if not any(base is ModuleType for base in static_class_mro(type(module))):
        raise ConfigError("plugins: import did not return a module.")
    try:
        spec = inspect.getattr_static(module, "__spec__", _MISSING)
    except Exception:
        raise ConfigError("plugins: module specification inspection failed.") from None
    if spec is _MISSING:
        return None
    return spec


def _origin_and_loader(
    module: object,
) -> tuple[
    str | None,
    PluginOriginReason | None,
    str | None,
    PluginLoaderReason | None,
    object | None,
]:
    spec = _module_spec(module)
    if spec is None:
        return None, "no_origin", None, "no_origin", None
    raw_origin = _static_spec_field(spec, "origin")
    loader = _static_spec_field(spec, "loader")
    locations = _static_spec_field(spec, "submodule_search_locations")

    if raw_origin is None:
        origin = None
        origin_reason: PluginOriginReason = (
            "namespace_package" if locations is not None else "no_origin"
        )
    elif static_isinstance(raw_origin, str):
        origin = str.__str__(raw_origin)
        if not origin:
            raise ConfigError("plugins: module origin must be non-empty or null.")
        origin_reason = None
    else:
        raise ConfigError("plugins: module origin must be a string or null.")

    if loader is None:
        if origin_reason is not None:
            loader_reason: PluginLoaderReason = origin_reason
        else:
            loader_reason = "generated_module"
        loader_type = None
    else:
        loader_type = _loader_type(loader)
        loader_reason = None
    return origin, origin_reason, loader_type, loader_reason, loader


def _static_spec_field(spec: object, field: str) -> object:
    try:
        raw = inspect.getattr_static(spec, field, _MISSING)
    except Exception:
        raise ConfigError(
            "plugins: module specification inspection failed."
        ) from None
    if raw is _MISSING:
        raise ConfigError("plugins: module specification is incomplete.")
    class_value = static_class_attribute(type(spec), field, _MISSING)
    if raw is class_value:
        if type(raw) is MemberDescriptorType or type(raw) is GetSetDescriptorType:
            try:
                return raw.__get__(spec, type(spec))
            except AttributeError:
                raise ConfigError(
                    "plugins: module specification is incomplete."
                ) from None
            except Exception:
                raise ConfigError(
                    "plugins: module specification inspection failed."
                ) from None
        descriptor_get = static_class_attribute(type(raw), "__get__", _MISSING)
        if descriptor_get is not _MISSING:
            raise ConfigError(
                "plugins: module specification inspection failed."
            )
    return raw


def _generated_origin(origin: str) -> bool:
    return origin in _GENERATED_ORIGINS or (
        str.startswith(origin, "<") and str.endswith(origin, ">")
    )


def _resolved_plugin_path(
    origin: str | None,
    origin_reason: PluginOriginReason | None,
) -> tuple[str | None, PluginPathReason | None, os.stat_result | None]:
    if origin is None:
        return None, cast(PluginPathReason, origin_reason), None
    if _generated_origin(origin):
        return None, "generated_module", None
    try:
        resolved = Path(origin).resolve(strict=True)
        initial_stat = resolved.stat()
    except (FileNotFoundError, NotADirectoryError):
        return None, "not_regular_file", None
    except Exception:
        return None, "unreadable", None
    if not stat.S_ISREG(initial_stat.st_mode):
        return None, "not_regular_file", None
    return str(resolved), None, initial_stat


def _is_extension_loader(
    loader: object | None,
    loader_type: str | None,
    resolved_path: str | None,
) -> bool:
    if resolved_path is not None and any(
        str.endswith(resolved_path, suffix)
        for suffix in _EXTENSION_SUFFIXES
    ):
        return True
    if loader is None or loader_type is None:
        return False
    return any(
        base is _EXTENSION_FILE_LOADER
        for base in static_class_mro(_loader_class(loader))
    )


def _hash_plugin_artifact(
    resolved_path: str | None,
    resolved_path_reason: PluginPathReason | None,
    initial_stat: os.stat_result | None,
    *,
    extension: bool,
) -> tuple[str | None, PluginHashReason | None]:
    if resolved_path is None:
        return None, cast(PluginHashReason, resolved_path_reason)
    if extension:
        return None, "extension_module"
    if initial_stat is None:
        return None, "unreadable"
    descriptor: int | None = None
    close_failed = False
    outcome: tuple[str | None, PluginHashReason | None] = (None, "unreadable")
    try:
        try:
            digest = hashlib.sha256()
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
                os, "O_NONBLOCK", 0
            )
            flags |= getattr(os, "O_NOFOLLOW", 0)
            path_before = os.lstat(resolved_path)
            descriptor = os.open(resolved_path, flags)
            target_before = os.fstat(descriptor)
            if not stat.S_ISREG(target_before.st_mode):
                outcome = (None, "not_regular_file")
            elif not (
                _same_snapshot(initial_stat, path_before)
                and _same_snapshot(path_before, target_before)
            ):
                outcome = (None, "unreadable")
            else:
                remaining = target_before.st_size
                complete = True
                while remaining:
                    requested_size = min(remaining, 1024 * 1024)
                    chunk = os.read(descriptor, requested_size)
                    if (
                        type(chunk) is not bytes
                        or not chunk
                        or len(chunk) > requested_size
                    ):
                        complete = False
                        break
                    remaining -= len(chunk)
                    digest.update(chunk)
                sentinel = b""
                if complete:
                    sentinel = os.read(descriptor, 1)
                    if type(sentinel) is not bytes or len(sentinel) > 1:
                        complete = False
                target_after = os.fstat(descriptor)
                path_after = os.lstat(resolved_path)
                if not (
                    complete
                    and sentinel == b""
                    and _same_snapshot(target_before, target_after)
                    and _same_snapshot(target_after, path_after)
                ):
                    outcome = (None, "unreadable")
                else:
                    outcome = (digest.hexdigest(), None)
        except Exception:
            outcome = (None, "unreadable")
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except Exception:
                close_failed = True
    if close_failed:
        return None, "unreadable"
    return outcome


def _same_snapshot(left: os.stat_result, right: os.stat_result) -> bool:
    return left.st_dev == right.st_dev and left.st_ino == right.st_ino and (
        left.st_mode,
        left.st_size,
        left.st_mtime_ns,
        left.st_ctime_ns,
    ) == (
        right.st_mode,
        right.st_size,
        right.st_mtime_ns,
        right.st_ctime_ns,
    )


def _strict_json_object(
    text: str, budget: _MetadataBudget
) -> Mapping[str, JsonValue]:
    def refuse_constant(_value: str):
        raise ValueError("non-finite JSON number")

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON object key")
            result[key] = value
        return result

    parsed = json.loads(
        text,
        parse_constant=refuse_constant,
        object_pairs_hook=unique_object,
    )
    return _freeze_direct_url(
        parsed,
        where="plugin distribution direct_url",
        budget=budget,
    )


def _distribution_record(
    name: str,
    distribution: object | None,
    *,
    not_installed: bool,
    budget: _MetadataBudget,
) -> PluginDistributionRecord:
    if not_installed:
        return PluginDistributionRecord(
            name=name,
            version=None,
            version_reason="not_installed",
            direct_url=None,
            direct_url_reason="not_installed",
        )
    if distribution is None:
        return PluginDistributionRecord(
            name=name,
            version=None,
            version_reason="unreadable",
            direct_url=None,
            direct_url_reason="unreadable",
        )

    try:
        raw_version = distribution.version
        version = _canonical_text(
            raw_version, where="plugin distribution version"
        )
        version_reason: PluginVersionReason | None = None
    except Exception:
        version = None
        version_reason = "unreadable"

    try:
        raw_direct_url = distribution.read_text("direct_url.json")
    except Exception:
        direct_url = None
        direct_url_reason: PluginDirectUrlReason | None = "unreadable"
    else:
        if raw_direct_url is None:
            direct_url = None
            direct_url_reason = "missing_direct_url"
        elif static_isinstance(raw_direct_url, str):
            try:
                if str.__len__(raw_direct_url) > _DIRECT_URL_TEXT_LIMIT:
                    raise ConfigError(
                        "plugin distribution direct_url scalar exceeds the "
                        f"{_DIRECT_URL_TEXT_LIMIT}-byte limit."
                    )
                exact_direct_url = str.__str__(raw_direct_url)
                _validate_json_text(
                    exact_direct_url,
                    where="plugin distribution direct_url",
                )
                direct_url = _strict_json_object(exact_direct_url, budget)
            except _MetadataBudgetExceeded:
                raise
            except Exception:
                direct_url = None
                direct_url_reason = "unreadable"
            else:
                direct_url_reason = None
        else:
            direct_url = None
            direct_url_reason = "unreadable"

    return PluginDistributionRecord(
        name=name,
        version=version,
        version_reason=version_reason,
        direct_url=direct_url,
        direct_url_reason=direct_url_reason,
    )


def _distribution_records(
    module_name: str, resolved_path: str | None
) -> tuple[PluginDistributionRecord, ...]:
    budget = _MetadataBudget()
    top_level = str.split(module_name, ".", 1)[0]
    names = _top_level_distribution_names(top_level, budget)
    artifact_candidates = _artifact_distribution_candidates(
        resolved_path, budget
    )
    names.update(artifact_candidates)
    for _name in names:
        budget.consume()
    rows: list[PluginDistributionRecord] = []
    for name in sorted(names):
        distribution = artifact_candidates.get(name)
        missing = False
        if distribution is None:
            try:
                distribution = metadata.distribution(name)
            except metadata.PackageNotFoundError:
                missing = True
            except Exception:
                distribution = None
        rows.append(
            _distribution_record(
                name,
                distribution,
                not_installed=missing,
                budget=budget,
            )
        )
    return tuple(rows)


def import_plugin(name: str) -> PluginRecord:
    """Import one trusted module and capture only independently auditable facts."""
    canonical_name = _canonical_module_name(name)
    already_imported = canonical_name in sys.modules
    try:
        module = importlib.import_module(canonical_name)
    except Exception as exc:
        raise ConfigError(
            f"plugins: importing {canonical_name!r} raised "
            f"{_safe_exception_text(exc)}."
        ) from None

    origin, origin_reason, loader_type, loader_reason, loader = (
        _origin_and_loader(module)
    )
    resolved_path, resolved_path_reason, initial_stat = _resolved_plugin_path(
        origin, origin_reason
    )
    code_hash, code_hash_reason = _hash_plugin_artifact(
        resolved_path,
        resolved_path_reason,
        initial_stat,
        extension=_is_extension_loader(loader, loader_type, resolved_path),
    )
    distributions = _distribution_records(canonical_name, resolved_path)
    return PluginRecord(
        name=canonical_name,
        already_imported=already_imported,
        origin=origin,
        origin_reason=origin_reason,
        loader_type=loader_type,
        loader_type_reason=loader_reason,
        resolved_path=resolved_path,
        resolved_path_reason=resolved_path_reason,
        distributions=distributions,
        distributions_reason=None if distributions else "no_distribution",
        code_hash=code_hash,
        code_hash_reason=code_hash_reason,
        unobserved_io=True,
    )


def _copy_plugin_record(value: object) -> PluginRecord:
    if type(value) is not PluginRecord:
        raise ConfigError("plugin audit row requires an exact PluginRecord.")
    try:
        return PluginRecord(
            name=object.__getattribute__(value, "name"),
            already_imported=object.__getattribute__(value, "already_imported"),
            origin=object.__getattribute__(value, "origin"),
            origin_reason=object.__getattribute__(value, "origin_reason"),
            loader_type=object.__getattribute__(value, "loader_type"),
            loader_type_reason=object.__getattribute__(value, "loader_type_reason"),
            resolved_path=object.__getattribute__(value, "resolved_path"),
            resolved_path_reason=object.__getattribute__(
                value, "resolved_path_reason"
            ),
            distributions=object.__getattribute__(value, "distributions"),
            distributions_reason=object.__getattribute__(
                value, "distributions_reason"
            ),
            code_hash=object.__getattribute__(value, "code_hash"),
            code_hash_reason=object.__getattribute__(value, "code_hash_reason"),
            unobserved_io=object.__getattribute__(value, "unobserved_io"),
        )
    except ConfigError:
        raise
    except Exception:
        raise ConfigError("plugin audit record is malformed.") from None


def plugin_audit_row(record: PluginRecord) -> Mapping[str, JsonValue]:
    """Validate and close the sole JSON projection of plugin facts."""
    canonical = _copy_plugin_record(record)
    distributions = tuple(
        {
            "name": item.name,
            "version": item.version,
            "version_reason": item.version_reason,
            "direct_url": item.direct_url,
            "direct_url_reason": item.direct_url_reason,
        }
        for item in canonical.distributions
    )
    projected = {
        "name": canonical.name,
        "already_imported": canonical.already_imported,
        "origin": canonical.origin,
        "origin_reason": canonical.origin_reason,
        "loader_type": canonical.loader_type,
        "loader_type_reason": canonical.loader_type_reason,
        "resolved_path": canonical.resolved_path,
        "resolved_path_reason": canonical.resolved_path_reason,
        "distributions": distributions,
        "distributions_reason": canonical.distributions_reason,
        "code_hash": canonical.code_hash,
        "code_hash_reason": canonical.code_hash_reason,
        "unobserved_io": canonical.unobserved_io,
    }
    frozen = freeze_evidence(projected, where="plugin audit row")
    if not static_isinstance(frozen, Mapping):
        raise ConfigError("plugin audit row must be a mapping.")
    return cast(Mapping[str, JsonValue], frozen)


__all__ = [
    "PLUGIN_DISTRIBUTION_ROW_KEYS",
    "PLUGIN_ROW_KEYS",
    "PluginDirectUrlReason",
    "PluginDistributionRecord",
    "PluginHashReason",
    "PluginLoaderReason",
    "PluginOriginReason",
    "PluginPathReason",
    "PluginRecord",
    "PluginVersionReason",
    "import_plugin",
    "plugin_audit_row",
]
