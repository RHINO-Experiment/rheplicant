"""JAX-free package-preset discovery and exact immutable snapshots."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from _rheplicant_bootstrap.errors import ConfigError
from _rheplicant_bootstrap.frozen import freeze
from _rheplicant_bootstrap.source import read_stable_regular_bytes
from _rheplicant_bootstrap.yaml import safe_load_document

_PRESET_RESOURCES = {
    "rhino_v1": "rheplicant/config/presets/rhino_v1.yaml",
}
_PRESET_SECTIONS = frozenset(
    {"runtime", "observation", "resources", "model", "inference"}
)


@dataclass(frozen=True, slots=True)
class PresetRequest:
    name: str
    only: Sequence[str] | None


@dataclass(frozen=True, slots=True)
class PresetSnapshot:
    name: str
    resource: str
    input_bytes: bytes
    sha256: str
    document: Mapping[str, object]
    expanded_nodes: int


def validate_preset_name(name: object) -> str:
    if not isinstance(name, str) or re.fullmatch(
        r"[A-Za-z][A-Za-z0-9_-]*", name
    ) is None:
        raise ConfigError(f"defaults: invalid package preset name {name!r}.")
    return name


def validate_preset_document(name: str, loaded: object) -> dict[str, object]:
    if not isinstance(loaded, Mapping):
        raise ConfigError(
            f"preset:{name}: document is a mapping; got "
            f"{type(loaded).__name__} ({loaded!r})."
        )
    invalid_keys = [key for key in loaded if not isinstance(key, str)]
    if invalid_keys:
        key = invalid_keys[0]
        raise ConfigError(
            f"preset:{name}: top-level key is a string; got "
            f"{type(key).__name__} ({key!r})."
        )
    forbidden = sorted(set(loaded) - _PRESET_SECTIONS)
    if forbidden:
        raise ConfigError(
            f"preset:{name}: package presets may contain only scientific base "
            f"sections {sorted(_PRESET_SECTIONS)}; got {forbidden}."
        )
    return dict(loaded)


def read_installed_preset(name: str) -> PresetSnapshot:
    name = validate_preset_name(name)
    try:
        resource = _PRESET_RESOURCES[name]
    except KeyError:
        raise ConfigError(f"defaults: unknown package preset {name!r}.") from None

    try:
        distribution = importlib.metadata.distribution("rheplicant")
        recorded = {item.as_posix(): item for item in (distribution.files or ())}
        if resource in recorded:
            path = Path(recorded[resource].locate())
        else:
            direct_url_text = distribution.read_text("direct_url.json")
            direct_url = json.loads(direct_url_text) if direct_url_text else {}
            if not isinstance(direct_url, Mapping):
                raise TypeError("direct_url.json is not a mapping")
            dir_info = direct_url.get("dir_info", {})
            if not isinstance(dir_info, Mapping):
                raise TypeError("direct_url.json dir_info is not a mapping")
            editable = dir_info.get("editable") is True
            if not editable:
                raise ConfigError(
                    f"defaults: installed distribution does not contain {resource!r}."
                )
            spec = importlib.util.find_spec("rheplicant")
            locations = (
                ()
                if spec is None
                else tuple(spec.submodule_search_locations or ())
            )
            if len(locations) != 1:
                raise ConfigError(
                    "defaults: editable rheplicant package root is not unique."
                )
            relative = Path(resource).relative_to("rheplicant")
            path = Path(locations[0]) / relative
    except ConfigError:
        raise
    except (AttributeError, ImportError, OSError, TypeError, ValueError) as exc:
        raise ConfigError(f"defaults: cannot discover package preset {name!r}: {exc}") from exc

    raw = read_stable_regular_bytes(
        path, maximum=16 * 1024 * 1024, source_name=f"preset:{name}"
    )
    loaded = safe_load_document(raw, source_name=f"preset:{name}")
    document = validate_preset_document(name, loaded.value)
    frozen_document = freeze(document)
    assert isinstance(frozen_document, Mapping)
    return PresetSnapshot(
        name=name,
        resource=resource,
        input_bytes=raw,
        sha256=hashlib.sha256(raw).hexdigest(),
        document=frozen_document,
        expanded_nodes=loaded.expanded_nodes,
    )


__all__ = [
    "PresetRequest",
    "PresetSnapshot",
    "read_installed_preset",
    "validate_preset_document",
    "validate_preset_name",
]
