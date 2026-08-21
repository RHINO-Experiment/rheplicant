"""The sole Config Plan 4A raw/preset/effective preparation pipeline."""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from _rheplicant_bootstrap.errors import ConfigError
from _rheplicant_bootstrap.frozen import freeze_evidence
from _rheplicant_bootstrap.layering import (
    DeletionRecord,
    OriginNode,
    _DeletionLedger,
    _OverlayMapping,
    _validate_parallel_origin_tree,
    layer_presets,
)
from _rheplicant_bootstrap.presets import PresetRequest, PresetSnapshot
from _rheplicant_bootstrap.process import (
    EffectiveProcessEntry,
    OutputGrammarParser,
    RawProcessEntry,
    parse_effective_process_mapping,
    parse_raw_process_mapping,
)
from _rheplicant_bootstrap.types import LayerIdentity, SourceInput, Stage
from _rheplicant_bootstrap.variants import (
    LayerEnumeration,
    LayerRef,
    enumerate_layers_once,
)
from _rheplicant_bootstrap.yaml import YamlLimits, safe_load_document


@dataclass(frozen=True, slots=True)
class SelectedPreset:
    request: PresetRequest
    snapshot: PresetSnapshot

    def __post_init__(self) -> None:
        if type(self.request) is not PresetRequest:
            raise ConfigError("selected preset request must be a PresetRequest.")
        if type(self.snapshot) is not PresetSnapshot:
            raise ConfigError("selected preset snapshot must be a PresetSnapshot.")
        if self.request.name != self.snapshot.name:
            raise ConfigError(
                "selected preset request and snapshot names must match."
            )
        object.__setattr__(
            self, "request", PresetRequest(self.request.name, self.request.only)
        )


def _canonical_text(value: object, *, where: str, nullable: bool = False):
    if nullable and value is None:
        return None
    if not isinstance(value, str):
        suffix = " or null" if nullable else ""
        raise ConfigError(
            f"{where} must be a string{suffix}; got {type(value).__name__}."
        )
    text = str.__str__(value)
    if not text:
        raise ConfigError(f"{where} must be a non-empty string.")
    return text


def _canonical_input_bytes(value: object) -> bytes:
    try:
        if isinstance(value, bytes):
            return bytes.__bytes__(value)
        if isinstance(value, bytearray):
            return bytes.__new__(bytes, bytearray.__getitem__(value, slice(None)))
        if isinstance(value, memoryview):
            return memoryview.tobytes(value)
    except Exception:
        raise ConfigError(
            "config source input_bytes must be a usable byte buffer."
        ) from None
    raise ConfigError(
        "config source input_bytes must be a byte buffer; got "
        f"{type(value).__name__}."
    )


def _canonical_launch_mode(value: object, *, where: str) -> str:
    if not isinstance(value, str):
        raise ConfigError(f"{where} launch_mode must be 'cli' or 'embedded'.")
    launch_mode = str.__str__(value)
    if launch_mode not in ("cli", "embedded"):
        raise ConfigError(f"{where} launch_mode must be 'cli' or 'embedded'.")
    return launch_mode


def _validate_source_shape(
    *,
    source_path: str,
    source_realpath: str | None,
    source_name: str,
    base_dir: str,
    where: str,
) -> None:
    def is_normalized_absolute(path: str) -> bool:
        return os.path.isabs(path) and os.path.normpath(path) == path

    if not is_normalized_absolute(base_dir):
        raise ConfigError(
            f"{where} base_dir must be an absolute normalized path."
        )
    if source_path == "<stdin>" or source_name == "<stdin>":
        if source_path != "<stdin>" or source_name != "<stdin>":
            raise ConfigError(
                f"{where} stdin source_path and source_name must both be '<stdin>'."
            )
        if source_realpath is not None:
            raise ConfigError(f"{where} stdin source_realpath must be null.")
        return
    if source_name != source_path:
        raise ConfigError(f"{where} source_name must match source_path.")
    if not is_normalized_absolute(source_path):
        raise ConfigError(
            f"{where} source_path must be an absolute normalized path."
        )
    if source_realpath is None:
        raise ConfigError(f"{where} file source_realpath must not be null.")
    if not is_normalized_absolute(source_realpath):
        raise ConfigError(
            f"{where} source_realpath must be an absolute normalized path."
        )


def _canonical_source_input(source: object) -> SourceInput:
    if type(source) is not SourceInput:
        raise ConfigError("source must be a SourceInput record.")
    input_bytes = _canonical_input_bytes(source.input_bytes)
    source_path = _canonical_text(
        source.source_path, where="source source_path"
    )
    source_realpath = _canonical_text(
        source.source_realpath,
        where="source source_realpath",
        nullable=True,
    )
    source_name = _canonical_text(
        source.source_name, where="source source_name"
    )
    base_dir = _canonical_text(source.base_dir, where="source base_dir")
    launch_mode = _canonical_launch_mode(source.launch_mode, where="source")
    _validate_source_shape(
        source_path=source_path,
        source_realpath=source_realpath,
        source_name=source_name,
        base_dir=base_dir,
        where="source",
    )
    return SourceInput(
        input_bytes=input_bytes,
        source_path=source_path,
        source_realpath=source_realpath,
        source_name=source_name,
        base_dir=base_dir,
        launch_mode=launch_mode,  # type: ignore[arg-type]
    )


@dataclass(frozen=True, slots=True)
class BootstrapManifest:
    protocol_version: int
    launch_mode: Literal["cli", "embedded"]
    input_sha256: str
    presets: Sequence[SelectedPreset]
    source_name: str
    source_path: str
    source_realpath: str | None
    base_dir: str

    def __post_init__(self) -> None:
        if isinstance(self.protocol_version, bool) or not isinstance(
            self.protocol_version, int
        ) or int.__int__(self.protocol_version) != 1:
            raise ConfigError("bootstrap protocol_version must be 1.")
        launch_mode = _canonical_launch_mode(
            self.launch_mode, where="bootstrap"
        )
        if not isinstance(self.input_sha256, str):
            raise ConfigError(
                "bootstrap input_sha256 must be a lowercase hexadecimal digest."
            )
        input_sha256 = str.__str__(self.input_sha256)
        if re.fullmatch(r"[0-9a-f]{64}", input_sha256) is None:
            raise ConfigError(
                "bootstrap input_sha256 must be a lowercase hexadecimal digest."
            )
        if isinstance(self.presets, str | bytes) or not isinstance(
            self.presets, Sequence
        ):
            raise ConfigError("bootstrap presets must be a sequence.")
        try:
            presets = tuple(self.presets)
        except Exception:
            raise ConfigError("bootstrap presets sequence traversal failed.") from None
        if any(type(item) is not SelectedPreset for item in presets):
            raise ConfigError(
                "bootstrap presets must contain SelectedPreset values."
            )
        canonical_presets = tuple(
            SelectedPreset(item.request, item.snapshot) for item in presets
        )
        if len({item.request.name for item in canonical_presets}) != len(
            canonical_presets
        ):
            raise ConfigError(
                "bootstrap presets contain a duplicate preset name."
            )
        source_name = _canonical_text(
            self.source_name, where="bootstrap source_name"
        )
        source_path = _canonical_text(
            self.source_path, where="bootstrap source_path"
        )
        source_realpath = _canonical_text(
            self.source_realpath,
            where="bootstrap source_realpath",
            nullable=True,
        )
        base_dir = _canonical_text(self.base_dir, where="bootstrap base_dir")
        _validate_source_shape(
            source_path=source_path,
            source_realpath=source_realpath,
            source_name=source_name,
            base_dir=base_dir,
            where="bootstrap",
        )
        object.__setattr__(self, "protocol_version", 1)
        object.__setattr__(self, "launch_mode", launch_mode)
        object.__setattr__(self, "input_sha256", input_sha256)
        object.__setattr__(self, "presets", canonical_presets)
        object.__setattr__(self, "source_name", source_name)
        object.__setattr__(self, "source_path", source_path)
        object.__setattr__(self, "source_realpath", source_realpath)
        object.__setattr__(self, "base_dir", base_dir)


@dataclass(frozen=True, slots=True)
class ConfigSource:
    input_bytes: bytes
    source_path: str
    source_realpath: str | None
    source_name: str
    base_dir: str
    parsed_document: Mapping[str, object]
    layered_document: Mapping[str, object]
    origins: OriginNode
    bootstrap_manifest: BootstrapManifest

    def __post_init__(self) -> None:
        input_bytes = _canonical_input_bytes(self.input_bytes)
        source_path = _canonical_text(
            self.source_path, where="config source source_path"
        )
        source_realpath = _canonical_text(
            self.source_realpath,
            where="config source source_realpath",
            nullable=True,
        )
        source_name = _canonical_text(
            self.source_name, where="config source source_name"
        )
        base_dir = _canonical_text(self.base_dir, where="config source base_dir")
        _validate_source_shape(
            source_path=source_path,
            source_realpath=source_realpath,
            source_name=source_name,
            base_dir=base_dir,
            where="config source",
        )
        if type(self.origins) is not OriginNode:
            raise ConfigError("config source origins must be an OriginNode.")
        if self.origins.origin is not None:
            raise ConfigError("config source origins root must have null origin.")
        if type(self.bootstrap_manifest) is not BootstrapManifest:
            raise ConfigError(
                "config source bootstrap_manifest must be a BootstrapManifest."
            )
        manifest = self.bootstrap_manifest
        if hashlib.sha256(input_bytes).hexdigest() != manifest.input_sha256:
            raise ConfigError(
                "config source input_bytes do not match bootstrap input_sha256."
            )
        if (
            source_path,
            source_realpath,
            source_name,
            base_dir,
        ) != (
            manifest.source_path,
            manifest.source_realpath,
            manifest.source_name,
            manifest.base_dir,
        ):
            raise ConfigError(
                "config source fields do not match the bootstrap manifest."
            )
        if not isinstance(self.parsed_document, Mapping):
            raise ConfigError("config source parsed_document must be a mapping.")
        if not isinstance(self.layered_document, Mapping):
            raise ConfigError("config source layered_document must be a mapping.")
        parsed = freeze_evidence(self.parsed_document, where="parsed document")
        layered = freeze_evidence(self.layered_document, where="layered document")
        if not isinstance(parsed, Mapping):
            raise ConfigError("config source parsed_document must be a mapping.")
        if not isinstance(layered, Mapping):
            raise ConfigError("config source layered_document must be a mapping.")
        _validate_parallel_origin_tree(layered, self.origins)
        object.__setattr__(self, "input_bytes", input_bytes)
        object.__setattr__(self, "source_path", source_path)
        object.__setattr__(self, "source_realpath", source_realpath)
        object.__setattr__(self, "source_name", source_name)
        object.__setattr__(self, "base_dir", base_dir)
        object.__setattr__(self, "parsed_document", parsed)
        object.__setattr__(self, "layered_document", layered)


@dataclass(frozen=True, slots=True)
class PreparedConfig:
    source: ConfigSource
    process: EffectiveProcessEntry
    layers: Sequence[LayerRef]
    layer_origins: Mapping[LayerIdentity, OriginNode]
    layer_deletions: Mapping[LayerIdentity, Sequence[DeletionRecord]]

    def __post_init__(self) -> None:
        if type(self.source) is not ConfigSource:
            raise ConfigError("prepared source must be a ConfigSource.")
        if type(self.process) is not EffectiveProcessEntry:
            raise ConfigError(
                "prepared process must be an EffectiveProcessEntry."
            )
        enumeration = LayerEnumeration(
            self.layers,
            self.layer_origins,
            self.layer_deletions,
        )
        object.__setattr__(self, "layers", enumeration.layers)
        object.__setattr__(self, "layer_origins", enumeration.origins)
        object.__setattr__(self, "layer_deletions", enumeration.deletions)


def _trusted_prepared_config(
    source: ConfigSource,
    process: EffectiveProcessEntry,
    enumeration: LayerEnumeration,
) -> PreparedConfig:
    if type(source) is not ConfigSource:
        raise ConfigError("trusted prepared source must be a ConfigSource.")
    if type(process) is not EffectiveProcessEntry:
        raise ConfigError(
            "trusted prepared process must be an EffectiveProcessEntry."
        )
    if type(enumeration) is not LayerEnumeration:
        raise ConfigError(
            "trusted prepared enumeration must carry deletion ledgers."
        )
    layers = enumeration.layers
    origins = enumeration.origins
    deletions = enumeration.deletions
    if (
        type(layers) is not tuple
        or not layers
        or type(origins) is not MappingProxyType
        or type(deletions) is not MappingProxyType
    ):
        raise ConfigError("trusted prepared enumeration has invalid storage.")
    try:
        origin_items = tuple(origins.items())
        deletion_items = tuple(deletions.items())
    except Exception:
        raise ConfigError(
            "trusted prepared enumeration evidence traversal failed."
        ) from None
    if len(origin_items) != len(layers) or len(deletion_items) != len(layers):
        raise ConfigError(
            "trusted prepared enumeration evidence must match every layer."
        )

    def matches_identity(value: object, layer: LayerRef) -> bool:
        return (
            type(value) is LayerIdentity
            and type(value.kind) is str
            and (value.name is None or type(value.name) is str)
            and value.kind == layer.kind
            and value.name == layer.name
        )

    for index, layer in enumerate(layers):
        if (
            type(layer) is not LayerRef
            or type(layer.document) is not _OverlayMapping
        ):
            raise ConfigError(
                "trusted prepared enumeration documents are invalid."
            )
        if (
            type(layer.kind) is not str
            or type(layer.prefix) is not str
            or (layer.name is not None and type(layer.name) is not str)
        ):
            raise ConfigError(
                "trusted prepared enumeration identities are invalid."
            )
        if (
            layer.kind == "base"
            and (layer.name is not None or layer.prefix != "")
        ) or (
            layer.kind == "variant"
            and (
                not layer.name
                or layer.prefix != f"variants.{layer.name}"
            )
        ) or layer.kind not in ("base", "variant") or (
            index == 0 and layer.kind != "base"
        ) or (index != 0 and layer.kind != "variant"):
            raise ConfigError(
                "trusted prepared enumeration identities are invalid."
            )
        try:
            origin_identity, root = origin_items[index]
            deletion_identity, rows = deletion_items[index]
        except Exception:
            raise ConfigError(
                "trusted prepared enumeration evidence traversal failed."
            ) from None
        if (
            not matches_identity(origin_identity, layer)
            or not matches_identity(deletion_identity, layer)
        ):
            raise ConfigError(
                "trusted prepared enumeration evidence must match every layer."
            )
        if (
            type(root) is not OriginNode
            or root.origin is not None
            or type(root.children) is not _OverlayMapping
            or len(layer.document) != len(root.children)
            or type(rows) is not _DeletionLedger
        ):
            raise ConfigError(
                "trusted prepared enumeration evidence values are invalid."
            )
    prepared = object.__new__(PreparedConfig)
    object.__setattr__(prepared, "source", source)
    object.__setattr__(prepared, "process", process)
    object.__setattr__(prepared, "layers", enumeration.layers)
    object.__setattr__(prepared, "layer_origins", enumeration.origins)
    object.__setattr__(prepared, "layer_deletions", enumeration.deletions)
    return prepared


def parse_raw_process_entry(
    document: Mapping[str, object],
    *,
    parse_outputs: OutputGrammarParser,
) -> RawProcessEntry:
    return parse_raw_process_mapping(document, parse_outputs=parse_outputs)


def parse_effective_process_entry(
    document: Mapping[str, object],
    layers: Sequence[LayerRef],
    *,
    raw: RawProcessEntry,
    parse_outputs: OutputGrammarParser,
) -> EffectiveProcessEntry:
    return parse_effective_process_mapping(
        document, layers, raw=raw, parse_outputs=parse_outputs
    )


def _aggregate_limit(
    *, name: str, observed: int, maximum: int
) -> ConfigError:
    return ConfigError(
        f"defaults: aggregate YAML {name} observed {observed} exceeds "
        f"limit {maximum}."
    )


def prepare_config_pipeline(
    source: SourceInput,
    *,
    preset_provider: Callable[[str], PresetSnapshot],
    parse_outputs: OutputGrammarParser,
    boundary_completed: Callable[[Stage, LayerIdentity | None], None]
    | None = None,
) -> PreparedConfig:
    """Parse once, layer presets once, and enumerate every layer once."""
    source = _canonical_source_input(source)
    loaded = safe_load_document(
        source.input_bytes, source_name=source.source_name
    )
    if boundary_completed is not None:
        boundary_completed("source", None)
    if not isinstance(loaded.value, Mapping):
        raise ConfigError(
            "document: configuration root must be a mapping; got "
            f"{type(loaded.value).__name__}."
        )
    parsed_document = loaded.value
    raw = parse_raw_process_entry(
        parsed_document, parse_outputs=parse_outputs
    )
    if boundary_completed is not None:
        boundary_completed("raw_process_entry", None)

    limits = YamlLimits()
    aggregate_bytes = len(source.input_bytes)
    aggregate_nodes = loaded.expanded_nodes

    def budgeted_provider(name: str) -> PresetSnapshot:
        nonlocal aggregate_bytes, aggregate_nodes
        snapshot = preset_provider(name)
        if type(snapshot) is not PresetSnapshot:
            raise ConfigError(
                f"defaults: preset provider for {name!r} must return "
                "PresetSnapshot."
            )
        next_bytes = aggregate_bytes + len(snapshot.input_bytes)
        if next_bytes > limits.input_bytes:
            raise _aggregate_limit(
                name="byte count", observed=next_bytes, maximum=limits.input_bytes
            )
        next_nodes = aggregate_nodes + snapshot.expanded_nodes
        if next_nodes > limits.expanded_nodes:
            raise _aggregate_limit(
                name="expanded nodes",
                observed=next_nodes,
                maximum=limits.expanded_nodes,
            )
        aggregate_bytes = next_bytes
        aggregate_nodes = next_nodes
        return snapshot

    merged, selected_pairs = layer_presets(
        parsed_document,
        raw.defaults,
        preset_provider=budgeted_provider,
    )
    if boundary_completed is not None:
        boundary_completed("preset_layering", None)
    enumeration = enumerate_layers_once(
        merged.document, merged.origins, merged.deletions
    )
    effective = parse_effective_process_entry(
        merged.document,
        enumeration.layers,
        raw=raw,
        parse_outputs=parse_outputs,
    )
    if boundary_completed is not None:
        boundary_completed("effective_process_entry", None)

    selected = tuple(
        SelectedPreset(request=request, snapshot=snapshot)
        for request, snapshot in selected_pairs
    )
    manifest = BootstrapManifest(
        protocol_version=1,
        launch_mode=source.launch_mode,
        input_sha256=hashlib.sha256(source.input_bytes).hexdigest(),
        presets=selected,
        source_name=source.source_name,
        source_path=source.source_path,
        source_realpath=source.source_realpath,
        base_dir=source.base_dir,
    )
    config_source = ConfigSource(
        input_bytes=source.input_bytes,
        source_path=source.source_path,
        source_realpath=source.source_realpath,
        source_name=source.source_name,
        base_dir=source.base_dir,
        parsed_document=parsed_document,
        layered_document=merged.document,
        origins=merged.origins,
        bootstrap_manifest=manifest,
    )
    return _trusted_prepared_config(
        config_source,
        effective,
        enumeration,
    )


def prepare_config(
    source: SourceInput,
    *,
    preset_provider: Callable[[str], PresetSnapshot],
    parse_outputs: OutputGrammarParser,
    boundary_completed: Callable[[Stage, LayerIdentity | None], None]
    | None = None,
) -> PreparedConfig:
    return prepare_config_pipeline(
        source,
        preset_provider=preset_provider,
        parse_outputs=parse_outputs,
        boundary_completed=boundary_completed,
    )


__all__ = [
    "BootstrapManifest",
    "ConfigSource",
    "PreparedConfig",
    "SelectedPreset",
    "parse_effective_process_entry",
    "parse_raw_process_entry",
    "prepare_config",
    "prepare_config_pipeline",
]
