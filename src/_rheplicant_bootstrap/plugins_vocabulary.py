"""The words a plugin refusal is allowed to use.

Every reason a plugin can be refused for is one of these Literals, and the two
row-key tuples say which fields the audit projection carries. They are here,
apart from the code that raises them, because a reader checking whether a
refusal is spelled the way the schema expects should not have to read the
importer to find out.
"""

from __future__ import annotations

from typing import Literal, TypeAlias

from _rheplicant_bootstrap.errors import ConfigError
from _rheplicant_bootstrap.frozen import (
    static_isinstance,
)

#: One sentinel. Defined in the module neither half imports back, so
#: that identity tests on both sides of the split see one object.
_MISSING = object()


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

_LOADER_REASONS = frozenset(("no_origin", "namespace_package", "generated_module"))

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

_DIRECT_URL_REASONS = frozenset(("not_installed", "missing_direct_url", "unreadable"))


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
        raise ConfigError(f"plugin {where} value and reason must be exact complements.")
    return value, canonical_reason
