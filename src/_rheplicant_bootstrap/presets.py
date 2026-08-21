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
from _rheplicant_bootstrap.frozen import freeze_evidence
from _rheplicant_bootstrap.source import read_stable_regular_bytes
from _rheplicant_bootstrap.yaml import safe_load_document

_PRESET_RESOURCES = {
    "rhino_v1": "rheplicant/config/presets/rhino_v1.yaml",
}
_PRESET_SECTIONS = frozenset(
    {"runtime", "observation", "resources", "model", "inference"}
)
_MAXIMUM_PRESET_BYTES = 16 * 1024 * 1024
_MAXIMUM_EXPANDED_NODES = 250_000


def _canonical_bytes(value: object, *, where: str, maximum: int) -> bytes:
    try:
        if isinstance(value, bytes):
            observed = bytes.__len__(value)
            if observed > maximum:
                raise ConfigError(
                    f"{where} byte count {observed} exceeds limit {maximum}."
                )
            return bytes.__bytes__(value)
        if isinstance(value, bytearray):
            observed = bytearray.__len__(value)
            if observed > maximum:
                raise ConfigError(
                    f"{where} byte count {observed} exceeds limit {maximum}."
                )
            copied = bytearray.__getitem__(value, slice(None))
            return bytes.__new__(bytes, copied)
        if isinstance(value, memoryview):
            observed = memoryview.nbytes.__get__(value)
            if observed > maximum:
                raise ConfigError(
                    f"{where} byte count {observed} exceeds limit {maximum}."
                )
            return memoryview.tobytes(value)
    except ConfigError:
        raise
    except Exception:
        raise ConfigError(
            f"{where} must be a usable byte buffer; got {type(value).__name__}."
        ) from None
    raise ConfigError(
        f"{where} must be a byte buffer; got {type(value).__name__}."
    )


@dataclass(frozen=True, slots=True)
class PresetRequest:
    name: str
    only: Sequence[str] | None

    def __post_init__(self) -> None:
        name = validate_preset_name(self.name)
        object.__setattr__(self, "name", name)
        if self.only is None:
            return
        if isinstance(self.only, str | bytes) or not isinstance(self.only, Sequence):
            raise ConfigError("defaults: only: is a sequence of dotted paths.")
        try:
            only = tuple(self.only)
        except Exception:
            raise ConfigError(
                "defaults: only: sequence traversal failed."
            ) from None
        if not only:
            raise ConfigError("defaults: only: must select at least one path.")
        canonical: list[str] = []
        for path in only:
            if not isinstance(path, str):
                raise ConfigError(
                    "defaults: only: document paths are strings; got "
                    f"{type(path).__name__}."
                )
            exact_path = str.__str__(path)
            if not exact_path or any(
                not part for part in str.split(exact_path, ".")
            ):
                raise ConfigError(
                    f"defaults: only: has invalid document path {exact_path!r}."
                )
            canonical.append(exact_path)
        object.__setattr__(self, "only", tuple(canonical))


@dataclass(frozen=True, slots=True)
class PresetSnapshot:
    name: str
    resource: str
    input_bytes: bytes
    sha256: str
    document: Mapping[str, object]
    expanded_nodes: int

    def __post_init__(self) -> None:
        name = validate_preset_name(self.name)
        if not isinstance(self.resource, str):
            raise ConfigError(
                f"preset:{name}: resource must be a non-empty string; got "
                f"{type(self.resource).__name__}."
            )
        resource = str.__str__(self.resource)
        if not resource:
            raise ConfigError(f"preset:{name}: resource must be a non-empty string.")
        input_bytes = _canonical_bytes(
            self.input_bytes,
            where=f"preset:{name}: input_bytes",
            maximum=_MAXIMUM_PRESET_BYTES,
        )
        if not isinstance(self.sha256, str):
            raise ConfigError(
                f"preset:{name}: sha256 must be a lowercase hexadecimal digest."
            )
        sha256 = str.__str__(self.sha256)
        if re.fullmatch(r"[0-9a-f]{64}", sha256) is None:
            raise ConfigError(
                f"preset:{name}: sha256 must be a lowercase hexadecimal digest."
            )
        if isinstance(self.expanded_nodes, bool) or not isinstance(
            self.expanded_nodes, int
        ):
            raise ConfigError(
                f"preset:{name}: expanded_nodes must be a non-negative integer."
            )
        expanded_nodes = int.__int__(self.expanded_nodes)
        if expanded_nodes < 0 or expanded_nodes > _MAXIMUM_EXPANDED_NODES:
            raise ConfigError(
                f"preset:{name}: expanded_nodes {expanded_nodes} must be between "
                f"0 and {_MAXIMUM_EXPANDED_NODES}."
            )
        observed_sha256 = hashlib.sha256(input_bytes).hexdigest()
        if sha256 != observed_sha256:
            raise ConfigError(
                f"preset:{name}: sha256 does not match input_bytes."
            )
        if not isinstance(self.document, Mapping):
            raise ConfigError(f"preset:{name}: snapshot document is a mapping.")
        frozen_document = freeze_evidence(
            self.document, where=f"preset:{name}.document"
        )
        assert isinstance(frozen_document, Mapping)
        validate_preset_document(name, frozen_document)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "resource", resource)
        object.__setattr__(self, "input_bytes", input_bytes)
        object.__setattr__(self, "sha256", sha256)
        object.__setattr__(self, "document", frozen_document)
        object.__setattr__(self, "expanded_nodes", expanded_nodes)


def validate_preset_name(name: object) -> str:
    if not isinstance(name, str):
        raise ConfigError(
            "defaults: invalid package preset name of type "
            f"{type(name).__name__}."
        )
    canonical = str.__str__(name)
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", canonical) is None:
        raise ConfigError(f"defaults: invalid package preset name {canonical!r}.")
    return canonical


def validate_preset_document(name: str, loaded: object) -> dict[str, object]:
    name = validate_preset_name(name)
    if not isinstance(loaded, Mapping):
        raise ConfigError(
            f"preset:{name}: document is a mapping; got "
            f"{type(loaded).__name__}."
        )
    canonical: dict[str, object] = {}
    try:
        pairs = loaded.items()
        iterator = iter(pairs)
    except Exception:
        raise ConfigError(
            f"preset:{name}: document mapping traversal failed."
        ) from None
    while True:
        try:
            pair = next(iterator)
        except StopIteration:
            break
        except Exception:
            raise ConfigError(
                f"preset:{name}: document mapping traversal failed."
            ) from None
        try:
            key, value = pair
        except Exception:
            raise ConfigError(
                f"preset:{name}: document mapping traversal failed."
            ) from None
        if not isinstance(key, str):
            raise ConfigError(
                f"preset:{name}: top-level key is a string; got "
                f"{type(key).__name__}."
            )
        exact_key = str.__str__(key)
        if exact_key in canonical:
            raise ConfigError(
                f"preset:{name}: top-level keys collide after "
                "canonicalization."
            )
        canonical[exact_key] = value
    forbidden = sorted(set(canonical) - _PRESET_SECTIONS)
    if forbidden:
        raise ConfigError(
            f"preset:{name}: package presets may contain only scientific base "
            f"sections {sorted(_PRESET_SECTIONS)}; got {forbidden}."
        )
    return canonical


def read_installed_preset(name: str) -> PresetSnapshot:
    name = validate_preset_name(name)
    try:
        resource = _PRESET_RESOURCES[name]
    except KeyError:
        raise ConfigError(f"defaults: unknown package preset {name!r}.") from None

    failure = f"defaults: cannot discover package preset {name!r}."

    def protocol(operation):
        try:
            return operation()
        except Exception:
            raise ConfigError(failure) from None

    def protocol_values(value):
        iterator = protocol(lambda: iter(value))
        while True:
            try:
                yield next(iterator)
            except StopIteration:
                return
            except Exception:
                raise ConfigError(failure) from None

    distribution = protocol(
        lambda: importlib.metadata.distribution("rheplicant")
    )
    files = protocol(lambda: distribution.files)
    recorded: dict[str, object] = {}
    if files is not None:
        for item in protocol_values(files):
            recorded_name = protocol(lambda item=item: item.as_posix())
            if not isinstance(recorded_name, str):
                raise ConfigError(failure)
            recorded[str.__str__(recorded_name)] = item
    if resource in recorded:
        located = protocol(lambda: recorded[resource].locate())
        path = protocol(lambda: Path(located))
    else:
        direct_url_text = protocol(
            lambda: distribution.read_text("direct_url.json")
        )
        if direct_url_text is None:
            exact_direct_url_text = ""
        elif isinstance(direct_url_text, str):
            exact_direct_url_text = str.__str__(direct_url_text)
        else:
            raise ConfigError(failure)
        try:
            direct_url = (
                json.loads(exact_direct_url_text)
                if exact_direct_url_text
                else {}
            )
        except Exception:
            raise ConfigError(failure) from None
        if not isinstance(direct_url, Mapping):
            raise ConfigError(failure)
        dir_info = protocol(lambda: direct_url.get("dir_info", {}))
        if not isinstance(dir_info, Mapping):
            raise ConfigError(failure)
        editable = protocol(lambda: dir_info.get("editable")) is True
        if not editable:
            raise ConfigError(
                f"defaults: installed distribution does not contain {resource!r}."
            )
        spec = protocol(lambda: importlib.util.find_spec("rheplicant"))
        if spec is None:
            locations = ()
        else:
            given_locations = protocol(
                lambda: spec.submodule_search_locations
            )
            locations = (
                ()
                if given_locations is None
                else tuple(protocol_values(given_locations))
            )
        if len(locations) != 1:
            raise ConfigError(
                "defaults: editable rheplicant package root is not unique."
            )
        relative = Path(resource).relative_to("rheplicant")
        root = protocol(lambda: Path(locations[0]))
        path = root / relative

    if not isinstance(path, Path):
        raise ConfigError(
            f"defaults: cannot discover package preset {name!r}."
        )

    raw = read_stable_regular_bytes(
        path, maximum=_MAXIMUM_PRESET_BYTES, source_name=f"preset:{name}"
    )
    loaded = safe_load_document(raw, source_name=f"preset:{name}")
    document = validate_preset_document(name, loaded.value)
    return PresetSnapshot(
        name=name,
        resource=resource,
        input_bytes=raw,
        sha256=hashlib.sha256(raw).hexdigest(),
        document=document,
        expanded_nodes=loaded.expanded_nodes,
    )


__all__ = [
    "PresetRequest",
    "PresetSnapshot",
    "read_installed_preset",
    "validate_preset_document",
    "validate_preset_name",
]
