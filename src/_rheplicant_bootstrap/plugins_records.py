"""The frozen records an audited import produces.

What a plugin and its distribution look like once they have been resolved and
projected, plus the defensive copies. A record is the boundary between the
import that built it and the audit row that serialises it.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import (
    MappingProxyType,
)
from typing import Literal

from _rheplicant_bootstrap.errors import ConfigError
from _rheplicant_bootstrap.frozen import (
    _freeze_evidence_roots,
)
from _rheplicant_bootstrap.types import JsonValue

from .plugins_projection import (
    _DIRECT_URL_INTEGER_BIT_LIMIT,
    _DIRECT_URL_TEXT_LIMIT,
    _canonical_module_name,
    _canonical_text,
    _freeze_direct_url,
    _MetadataBudget,
    _normalize_distribution_name,
    _validate_frozen_json,
)
from .plugins_vocabulary import (
    _DIRECT_URL_REASONS,
    _HASH_REASONS,
    _LOADER_REASONS,
    _MISSING,
    _ORIGIN_REASONS,
    _PATH_REASONS,
    _VERSION_REASONS,
    PluginDirectUrlReason,
    PluginHashReason,
    PluginLoaderReason,
    PluginOriginReason,
    PluginPathReason,
    PluginVersionReason,
    _canonical_reason,
    _value_reason,
)

#: Used only by PluginRecord's code-hash validation.
_LOWER_SHA256 = re.compile(r"[0-9a-f]{64}")


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
            else _canonical_text(raw_version, where="plugin distribution version")
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
            else _freeze_direct_url(raw_direct_url, where="plugin distribution direct_url")
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
        raw_direct_url_reason = object.__getattribute__(value, "direct_url_reason")
        direct_url_reason = (
            raw_direct_url_reason if raw_direct_url is None else "missing_direct_url"
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
            retained_direct_url = raw_direct_url if direct_url_copy is _MISSING else direct_url_copy
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
            raw_already_imported = object.__getattribute__(self, "already_imported")
            raw_origin_field = object.__getattribute__(self, "origin")
            raw_origin_reason_field = object.__getattribute__(self, "origin_reason")
            raw_loader_type_field = object.__getattribute__(self, "loader_type")
            raw_loader_type_reason_field = object.__getattribute__(self, "loader_type_reason")
            raw_resolved_path_field = object.__getattribute__(self, "resolved_path")
            raw_resolved_path_reason_field = object.__getattribute__(self, "resolved_path_reason")
            raw_distributions = object.__getattribute__(self, "distributions")
            raw_distributions_reason = object.__getattribute__(self, "distributions_reason")
            raw_code_hash_field = object.__getattribute__(self, "code_hash")
            raw_code_hash_reason_field = object.__getattribute__(self, "code_hash_reason")
            raw_unobserved_io = object.__getattribute__(self, "unobserved_io")
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
        origin = None if raw_origin is None else _canonical_text(raw_origin, where="plugin origin")
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
            else _canonical_text(raw_resolved_path, where="plugin resolved_path")
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
                    raise ConfigError("plugin distribution direct_url must be recursively frozen.")
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
                raise ConfigError("plugin distribution direct_url snapshot is malformed.")
            validated_snapshot_nodes: set[int] = set()
            for frozen_direct_url in tuple.__iter__(frozen_direct_urls):
                if type(frozen_direct_url) is not MappingProxyType:
                    raise ConfigError("plugin distribution direct_url must be a JSON object.")
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
                raise ConfigError("plugin distributions must be normalized, unique, and sorted.")
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
                raise ConfigError("plugin distributions must have no reason when present.")
        elif distributions_reason != "no_distribution":
            raise ConfigError("plugin distributions must use no_distribution when empty.")

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
            raise ConfigError("plugin code_hash must be a lowercase 64-character SHA-256.")
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
            resolved_path_reason=object.__getattribute__(value, "resolved_path_reason"),
            distributions=object.__getattribute__(value, "distributions"),
            distributions_reason=object.__getattribute__(value, "distributions_reason"),
            code_hash=object.__getattribute__(value, "code_hash"),
            code_hash_reason=object.__getattribute__(value, "code_hash_reason"),
            unobserved_io=object.__getattribute__(value, "unobserved_io"),
        )
    except ConfigError:
        raise
    except Exception:
        raise ConfigError("plugin audit record is malformed.") from None
