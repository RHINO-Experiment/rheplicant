"""JAX-free audited plugin imports and their closed JSON projection."""

from __future__ import annotations

import hashlib
import importlib
import importlib.machinery
import inspect
import os
import stat
import sys
from collections.abc import Mapping
from importlib import metadata
from pathlib import Path, PurePosixPath
from types import (
    GetSetDescriptorType,
    MemberDescriptorType,
    ModuleType,
)
from typing import cast

from _rheplicant_bootstrap.errors import ConfigError
from _rheplicant_bootstrap.frozen import (
    freeze_evidence,
    static_class_attribute,
    static_class_mro,
    static_class_text,
    static_isinstance,
    static_type_name,
)
from _rheplicant_bootstrap.types import JsonValue

from .plugins_projection import (
    _DIRECT_URL_TEXT_LIMIT,
    _canonical_module_name,
    _canonical_text,
    _is_utf8_text,
    _MetadataBudget,
    _MetadataBudgetExceeded,
    _normalize_distribution_name,
    _ordered_metadata_sequence,
    _protocol_items,
    _protocol_value,
    _strict_json_object,
    _validate_json_text,
)
from .plugins_records import (
    PluginDistributionRecord,
    PluginRecord,
    _copy_plugin_record,
)
from .plugins_vocabulary import (  # noqa: F401  # some are re-exports: this module's __all__ still names them  # noqa: F401  # re-exports: __all__ still names them, TypeAlias assignments
    _MISSING,
    PLUGIN_DISTRIBUTION_ROW_KEYS,
    PLUGIN_ROW_KEYS,
    PluginDirectUrlReason,
    PluginHashReason,
    PluginLoaderReason,
    PluginOriginReason,
    PluginPathReason,
    PluginVersionReason,
)

_GENERATED_ORIGINS = frozenset(("built-in", "frozen"))
_EXTENSION_SUFFIXES = tuple(
    str.__str__(suffix)
    for suffix in tuple(importlib.machinery.EXTENSION_SUFFIXES)
    if isinstance(suffix, str)
)
_EXTENSION_FILE_LOADER = importlib.machinery.ExtensionFileLoader
_FOREIGN_EXCEPTION_DETAIL_LIMIT = 1024


def _top_level_distribution_names(top_level: str, budget: _MetadataBudget) -> set[str]:
    top_map = _protocol_value(
        metadata.packages_distributions,
        where="packages_distributions inspection",
    )
    if not static_isinstance(top_map, Mapping):
        raise ConfigError("plugins: distribution metadata packages_distributions is not a mapping.")
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
                "plugins: distribution metadata packages_distributions traversal failed."
            ) from None
        key = _canonical_text(
            raw_key,
            where="plugins: distribution metadata top-level key",
        )
        if key in seen_keys:
            raise ConfigError(
                "plugins: distribution metadata top-level keys collide after canonicalization."
            )
        seen_keys.add(key)
        if key == top_level:
            raw_names = raw_value
            found = True
    if not found:
        return set()
    raw_names = _ordered_metadata_sequence(raw_names, where="candidate names")
    names: set[str] = set()
    for raw_name in _protocol_items(raw_names, where="candidate traversal", budget=budget):
        names.add(_normalize_distribution_name(raw_name, require_normalized=False))
    return names


def _record_text_top_level(text: str, budget: _MetadataBudget) -> str | None:
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
                if component_length == 2 and text[start] == "." and text[start + 1] == ".":
                    return None
                if any(
                    str.find(text, forbidden, start, end) >= 0 for forbidden in ("\x00", "\\", ":")
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
        raise ConfigError("plugins: distribution metadata RECORD path inspection failed.") from None
    if type(raw_is_absolute) is not bool:
        raise ConfigError("plugins: distribution metadata RECORD path inspection failed.")
    raw_parts = _ordered_metadata_sequence(raw_parts, where="RECORD path components")
    top_name: str | None = None
    for part in _protocol_items(raw_parts, where="RECORD path component traversal", budget=budget):
        if not static_isinstance(part, str):
            return None
        exact = str.__str__(part)
        if exact in ("", ".", "..", "/") or "\x00" in exact or "\\" in exact or ":" in exact:
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
        raise ConfigError("plugins: distribution metadata name inspection failed.") from None
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
            raise ConfigError("plugins: distribution metadata RECORD inspection failed.") from None
        if files is None:
            continue
        files = _ordered_metadata_sequence(files, where="RECORD files")
        claimed = False
        base: tuple[Path, Path] | None = None
        rejected_base = False
        roots: dict[str, Path | None] = {}
        for entry in _protocol_items(files, where="RECORD traversal", budget=budget):
            top_name = _record_parts(entry, budget)
            if top_name is None:
                continue
            if rejected_base:
                break
            if base is None:
                try:
                    lexical_base = _lexical_path(distribution.locate_file(PurePosixPath()))
                    resolved_base = lexical_base.resolve(strict=True)
                except Exception:
                    raise ConfigError(
                        "plugins: distribution metadata artifact-root inspection failed."
                    ) from None
                if lexical_base == Path(lexical_base.anchor) or resolved_base == Path(
                    resolved_base.anchor
                ):
                    rejected_base = True
                    break
                base = (lexical_base, resolved_base)

            if top_name not in roots:
                lexical_base, resolved_base = base
                try:
                    lexical_root = _lexical_path(distribution.locate_file(PurePosixPath(top_name)))
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
                        "plugins: distribution metadata artifact-root inspection failed."
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
    exact_module = static_class_text(selected, "__module__", fallback="builtins")
    exact_qualname = static_class_text(selected, "__qualname__", fallback="unknown")
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
        raise ConfigError("plugins: module specification inspection failed.") from None
    if raw is _MISSING:
        raise ConfigError("plugins: module specification is incomplete.")
    class_value = static_class_attribute(type(spec), field, _MISSING)
    if raw is class_value:
        if type(raw) is MemberDescriptorType or type(raw) is GetSetDescriptorType:
            try:
                return raw.__get__(spec, type(spec))
            except AttributeError:
                raise ConfigError("plugins: module specification is incomplete.") from None
            except Exception:
                raise ConfigError("plugins: module specification inspection failed.") from None
        descriptor_get = static_class_attribute(type(raw), "__get__", _MISSING)
        if descriptor_get is not _MISSING:
            raise ConfigError("plugins: module specification inspection failed.")
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
        str.endswith(resolved_path, suffix) for suffix in _EXTENSION_SUFFIXES
    ):
        return True
    if loader is None or loader_type is None:
        return False
    return any(base is _EXTENSION_FILE_LOADER for base in static_class_mro(_loader_class(loader)))


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
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
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
                    if type(chunk) is not bytes or not chunk or len(chunk) > requested_size:
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
    return (
        left.st_dev == right.st_dev
        and left.st_ino == right.st_ino
        and (
            left.st_mode,
            left.st_size,
            left.st_mtime_ns,
            left.st_ctime_ns,
        )
        == (
            right.st_mode,
            right.st_size,
            right.st_mtime_ns,
            right.st_ctime_ns,
        )
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
        version = _canonical_text(raw_version, where="plugin distribution version")
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
    artifact_candidates = _artifact_distribution_candidates(resolved_path, budget)
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
            f"plugins: importing {canonical_name!r} raised {_safe_exception_text(exc)}."
        ) from None

    origin, origin_reason, loader_type, loader_reason, loader = _origin_and_loader(module)
    resolved_path, resolved_path_reason, initial_stat = _resolved_plugin_path(origin, origin_reason)
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
