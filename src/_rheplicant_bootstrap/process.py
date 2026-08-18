"""JAX-free grammar for process-entry and runtime sections."""

from __future__ import annotations

import keyword
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Protocol

from _rheplicant_bootstrap.errors import ConfigError
from _rheplicant_bootstrap.frozen import freeze_evidence
from _rheplicant_bootstrap.layering import parse_default
from _rheplicant_bootstrap.presets import PresetRequest

_RUNTIME_KEYS = frozenset({"jax_enable_x64", "platform", "seed", "seeds"})
_PLATFORMS = ("auto", "cpu", "gpu", "tpu")
_PROCESS_SECTIONS = ("defaults", "plugins", "outputs")
_DEFAULTS_FORM_ERROR = (
    "defaults: must be a list whose entries are preset names or mappings "
    "with exactly 'from' and optional 'only'."
)


class OutputGrammarParser(Protocol):
    def __call__(self, raw_outputs: object, /) -> object:
        """Validate/freeze only the outputs section grammar."""


@dataclass(frozen=True, slots=True)
class RawProcessEntry:
    schema_version: int
    defaults: Sequence[PresetRequest]
    plugins: Sequence[str]
    raw_runtime: Mapping[str, object] | None
    outputs: object

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or int.__int__(self.schema_version) != 1
        ):
            raise ConfigError("schema_version: 1 is required at the process entry point.")
        object.__setattr__(self, "schema_version", 1)
        if isinstance(self.defaults, str | bytes) or not isinstance(
            self.defaults, Sequence
        ):
            raise ConfigError(_DEFAULTS_FORM_ERROR)
        try:
            defaults = tuple(self.defaults)
        except Exception:
            raise ConfigError(_DEFAULTS_FORM_ERROR) from None
        if any(type(item) is not PresetRequest for item in defaults):
            raise ConfigError(_DEFAULTS_FORM_ERROR)
        canonical_defaults = tuple(
            PresetRequest(item.name, item.only) for item in defaults
        )
        if len({item.name for item in canonical_defaults}) != len(
            canonical_defaults
        ):
            raise ConfigError("defaults: duplicate package preset.")
        plugins = _parse_plugins(self.plugins)
        object.__setattr__(self, "defaults", canonical_defaults)
        object.__setattr__(self, "plugins", plugins)
        if self.raw_runtime is not None:
            if not isinstance(self.raw_runtime, Mapping):
                raise ConfigError("raw process runtime must be a mapping or null.")
            frozen_runtime = freeze_evidence(
                self.raw_runtime, where="raw process runtime"
            )
            if not isinstance(frozen_runtime, Mapping):
                raise ConfigError("raw process runtime must be a mapping or null.")
            object.__setattr__(self, "raw_runtime", frozen_runtime)
        object.__setattr__(self, "outputs", _freeze_output(self.outputs))


@dataclass(frozen=True, slots=True)
class RuntimeSpec:
    jax_enable_x64: bool
    platform: Literal["auto", "cpu", "gpu", "tpu"]
    seed: int | None
    seeds: Mapping[str, int]

    def __post_init__(self) -> None:
        if not isinstance(self.jax_enable_x64, bool):
            raise ConfigError("runtime.jax_enable_x64 is a bool.")
        if not isinstance(self.platform, str):
            raise ConfigError(f"runtime.platform is one of {list(_PLATFORMS)}.")
        platform = str.__str__(self.platform)
        if platform not in _PLATFORMS:
            raise ConfigError(f"runtime.platform is one of {list(_PLATFORMS)}.")
        if self.seed is not None and (
            isinstance(self.seed, bool) or not isinstance(self.seed, int)
        ):
            raise ConfigError("runtime.seed is an int or null.")
        frozen_seeds = freeze_evidence(self.seeds, where="runtime seeds")
        if not isinstance(frozen_seeds, Mapping):
            raise ConfigError("runtime.seeds is a mapping of name -> int.")
        canonical_seeds: dict[str, int] = {}
        for name, value in frozen_seeds.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise ConfigError(f"runtime.seeds.{name} is an int.")
            canonical_seeds[name] = int.__int__(value)
        object.__setattr__(self, "platform", platform)
        object.__setattr__(
            self, "seed", None if self.seed is None else int.__int__(self.seed)
        )
        object.__setattr__(self, "seeds", MappingProxyType(canonical_seeds))


@dataclass(frozen=True, slots=True)
class EffectiveProcessEntry:
    runtime: RuntimeSpec
    plugins: Sequence[str]
    outputs: object

    def __post_init__(self) -> None:
        if type(self.runtime) is not RuntimeSpec:
            raise ConfigError("effective runtime must be a RuntimeSpec.")
        object.__setattr__(self, "plugins", _parse_plugins(self.plugins))
        object.__setattr__(self, "outputs", _freeze_output(self.outputs))


def _freeze_output(value: object) -> object:
    if isinstance(
        value,
        Mapping
        | Sequence
        | str
        | int
        | float
        | bytes
        | bytearray
        | memoryview,
    ):
        return freeze_evidence(value, where="outputs")
    return value


def _mapping_pairs(mapping: Mapping, *, failure: str):
    try:
        iterator = iter(mapping.items())
    except Exception:
        raise ConfigError(failure) from None
    while True:
        try:
            pair = next(iterator)
        except StopIteration:
            return
        except Exception:
            raise ConfigError(failure) from None
        try:
            key, value = pair
        except Exception:
            raise ConfigError(failure) from None
        yield key, value


def _top_level(document: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(document, Mapping):
        raise ConfigError(
            "document: configuration root must be a mapping; got "
            f"{type(document).__name__}."
        )
    result: dict[str, object] = {}
    for position, (key, value) in enumerate(
        _mapping_pairs(
            document, failure="document: top-level mapping traversal failed."
        ),
        start=1,
    ):
        if not isinstance(key, str):
            detail = (
                f" {int.__int__(key)}"
                if isinstance(key, int) and not isinstance(key, bool)
                else ""
            )
            raise ConfigError(
                f"document: top-level key {position} must be a string; got "
                f"{type(key).__name__}{detail}."
            )
        exact = str.__str__(key)
        if exact in result:
            raise ConfigError(
                "document: top-level keys collide after canonicalization."
            )
        result[exact] = value
    return result


def _parse_defaults(raw: object) -> tuple[PresetRequest, ...]:
    if isinstance(raw, str | bytes) or not isinstance(raw, Sequence):
        raise ConfigError(_DEFAULTS_FORM_ERROR)
    try:
        entries = tuple(raw)
    except Exception:
        raise ConfigError(_DEFAULTS_FORM_ERROR) from None
    parsed: list[PresetRequest] = []
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        try:
            request = parse_default(entry)
        except ConfigError:
            raise ConfigError(_DEFAULTS_FORM_ERROR) from None
        if request.name in seen:
            raise ConfigError(
                f"defaults[{index}]: duplicate package preset {request.name!r}."
            )
        seen.add(request.name)
        parsed.append(request)
    return tuple(parsed)


def _module_name(raw: object, *, index: int) -> str:
    if not isinstance(raw, str):
        raise ConfigError(
            f"plugins[{index}]: module name must be a string; got "
            f"{type(raw).__name__}."
        )
    name = str.__str__(raw)
    pieces = str.split(name, ".")
    if (
        not name
        or any(
            not piece
            or not str.isidentifier(piece)
            or keyword.iskeyword(piece)
            for piece in pieces
        )
    ):
        raise ConfigError(
            f"plugins[{index}]: {name!r} is not a dot-separated Python "
            "module name."
        )
    return name


def _parse_plugins(raw: object) -> tuple[str, ...]:
    if isinstance(raw, str | bytes) or not isinstance(raw, Sequence):
        raise ConfigError("plugins: must be a list of Python module names.")
    try:
        entries = tuple(raw)
    except Exception:
        raise ConfigError("plugins: sequence traversal failed.") from None
    result: list[str] = []
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        name = _module_name(entry, index=index)
        if name in seen:
            raise ConfigError(f"plugins[{index}]: duplicate module name {name!r}.")
        seen.add(name)
        result.append(name)
    return tuple(result)


def _call_output_parser(
    parser: OutputGrammarParser, raw_outputs: object
) -> object:
    if not callable(parser):
        raise ConfigError("parse_outputs must be callable.")
    try:
        return parser(raw_outputs)
    except ConfigError:
        raise
    except Exception as error:
        raise ConfigError(
            "outputs: grammar parser raised " f"{type(error).__name__}."
        ) from error


def parse_raw_process_mapping(
    document: Mapping[str, object],
    *,
    parse_outputs: OutputGrammarParser,
) -> RawProcessEntry:
    """Validate only facts needed before package preset layering."""
    top = _top_level(document)
    version = top.get("schema_version")
    if (
        isinstance(version, bool)
        or not isinstance(version, int)
        or int.__int__(version) != 1
    ):
        raise ConfigError("schema_version: 1 is required at the process entry point.")
    defaults = _parse_defaults(top["defaults"]) if "defaults" in top else ()
    plugins = _parse_plugins(top["plugins"]) if "plugins" in top else ()
    if "runtime" in top:
        raw_runtime = top["runtime"]
        if not isinstance(raw_runtime, Mapping):
            raise ConfigError("runtime: must be a mapping when present.")
        frozen_runtime = freeze_evidence(raw_runtime, where="runtime")
        if not isinstance(frozen_runtime, Mapping):
            raise ConfigError("runtime: must be a mapping when present.")
        raw_runtime = frozen_runtime
    else:
        raw_runtime = None
    outputs = _call_output_parser(parse_outputs, top.get("outputs", {}))
    return RawProcessEntry(
        schema_version=1,
        defaults=defaults,
        plugins=plugins,
        raw_runtime=raw_runtime,
        outputs=outputs,
    )


def parse_runtime(section: object) -> RuntimeSpec:
    """Normalize ``runtime:`` without importing or consulting JAX."""
    if not isinstance(section, Mapping):
        raise ConfigError(f"runtime: is a mapping; got {type(section).__name__}.")
    canonical: dict[str, object] = {}
    for key, value in _mapping_pairs(
        section, failure="runtime: mapping traversal failed."
    ):
        if not isinstance(key, str):
            raise ConfigError(
                "runtime: keys are strings; got " f"{type(key).__name__}."
            )
        exact_key = str.__str__(key)
        if exact_key in canonical:
            raise ConfigError(
                "runtime: keys collide after canonicalization."
            )
        canonical[exact_key] = value
    raw_seeds_source = canonical.get("seeds", MappingProxyType({}))
    frozen = freeze_evidence(canonical, where="runtime")
    if not isinstance(frozen, Mapping):
        raise ConfigError("runtime: is a mapping.")
    unknown = sorted(set(frozen) - _RUNTIME_KEYS)
    if unknown:
        raise ConfigError(f"runtime: unknown keys {unknown}.")

    x64 = frozen.get("jax_enable_x64", False)
    if not isinstance(x64, bool):
        raise ConfigError(
            "runtime.jax_enable_x64 is a bool; got " f"{type(x64).__name__}."
        )
    platform = frozen.get("platform", "auto")
    if not isinstance(platform, str):
        raise ConfigError(
            f"runtime.platform is one of {list(_PLATFORMS)}; got "
            f"{type(platform).__name__}."
        )
    exact_platform = str.__str__(platform)
    if exact_platform not in _PLATFORMS:
        raise ConfigError(
            f"runtime.platform is one of {list(_PLATFORMS)}; got "
            f"{exact_platform!r}."
        )

    seed = frozen.get("seed")
    if seed is not None and (
        isinstance(seed, bool) or not isinstance(seed, int)
    ):
        raise ConfigError(
            f"runtime.seed is an int or null; got {type(seed).__name__}."
        )
    canonical_seed = None if seed is None else int.__int__(seed)

    raw_seeds = frozen.get("seeds", MappingProxyType({}))
    if not isinstance(raw_seeds, Mapping):
        raise ConfigError(
            "runtime.seeds is a mapping of name -> int; got "
            f"{type(raw_seeds_source).__name__}."
        )
    seeds: dict[str, int] = {}
    for name, value in raw_seeds.items():
        if not isinstance(name, str):
            raise ConfigError(
                "runtime.seeds keys are strings; got "
                f"{type(name).__name__}."
            )
        exact_name = str.__str__(name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(
                f"runtime.seeds.{exact_name} is an int; got "
                f"{type(value).__name__}."
            )
        seeds[exact_name] = int.__int__(value)
    return RuntimeSpec(
        jax_enable_x64=bool(x64),
        platform=exact_platform,  # type: ignore[arg-type]
        seed=canonical_seed,
        seeds=MappingProxyType(seeds),
    )


def _variant_pairs(document: Mapping[str, object]):
    if "variants" not in document:
        return ()
    variants = document["variants"]
    if not isinstance(variants, Mapping):
        raise ConfigError(
            "variants: is a mapping of name -> patch; got "
            f"{type(variants).__name__}."
        )
    return tuple(
        _mapping_pairs(
            variants, failure="variants: mapping traversal failed."
        )
    )


def validate_variant_process_sections(
    document: Mapping[str, object],
) -> None:
    """Refuse a variant touching document-global process-entry sections."""
    for raw_name, patch in _variant_pairs(document):
        if not isinstance(raw_name, str):
            raise ConfigError(
                "variants: names must be non-empty strings; got "
                f"{type(raw_name).__name__}."
            )
        name = str.__str__(raw_name)
        if not name:
            raise ConfigError("variants: names must be non-empty strings; got ''.")
        if not isinstance(patch, Mapping):
            continue
        keys: set[str] = set()
        for raw_key, _ in _mapping_pairs(
            patch, failure=f"variants.{name}: patch mapping traversal failed."
        ):
            if not isinstance(raw_key, str):
                continue
            keys.add(str.__str__(raw_key))
        for section in _PROCESS_SECTIONS:
            if section in keys or f"~{section}" in keys:
                raise ConfigError(
                    f"variants.{name}.{section}: variants may not change "
                    f"process-entry section {section!r}; edit the base "
                    f"{section} section."
                )


def _render_runtime_value(field: str, value: object) -> str:
    if field == "jax_enable_x64":
        return "true" if value is True else "false"
    return repr(value)


def parse_effective_process_mapping(
    document: Mapping[str, object],
    layers: Sequence[object],
    *,
    raw: RawProcessEntry,
    parse_outputs: OutputGrammarParser,
) -> EffectiveProcessEntry:
    """Validate effective runtime/process state after preset layering."""
    if type(raw) is not RawProcessEntry:
        raise ConfigError("effective raw process entry must be a RawProcessEntry.")
    top = _top_level(document)
    validate_variant_process_sections(top)
    if "runtime" not in top:
        raise ConfigError("runtime: is required after package presets are layered.")
    base_runtime = parse_runtime(top["runtime"])
    effective_plugins = (
        _parse_plugins(top["plugins"]) if "plugins" in top else ()
    )
    if effective_plugins != tuple(raw.plugins):
        raise ConfigError(
            "plugins: effective process-entry list differs from the raw "
            "process entry."
        )
    if isinstance(layers, str | bytes) or not isinstance(layers, Sequence):
        raise ConfigError("effective layer sequence must be a sequence.")
    try:
        frozen_layers = tuple(layers)
    except Exception:
        raise ConfigError(
            "effective layer sequence traversal failed."
        ) from None
    from _rheplicant_bootstrap.variants import LayerRef

    if any(type(layer) is not LayerRef for layer in frozen_layers):
        raise ConfigError(
            "effective layer sequence must contain LayerRef values."
        )

    mismatches: list[str] = []
    for layer in frozen_layers:
        if layer.kind != "variant":
            continue
        name = layer.name
        layer_document = layer.document
        if not isinstance(name, str) or not isinstance(layer_document, Mapping):
            raise ConfigError("effective layer record is invalid.")
        try:
            runtime = parse_runtime(layer_document.get("runtime"))
        except ConfigError as error:
            message = str(error)
            if message.startswith("runtime"):
                message = f"variants.{name}.runtime{message[len('runtime'):]}"
            else:
                message = f"variants.{name}.runtime: {message}"
            raise ConfigError(message) from None
        for field in ("jax_enable_x64", "platform"):
            base_value = getattr(base_runtime, field)
            variant_value = getattr(runtime, field)
            if variant_value != base_value:
                mismatches.append(
                    f"runtime.{field}={_render_runtime_value(field, base_value)}, "
                    f"variants.{name}.runtime.{field}="
                    f"{_render_runtime_value(field, variant_value)}"
                )
    if mismatches:
        raise ConfigError(
            "runtime: process-global settings disagree: "
            + "; ".join(mismatches)
            + "."
        )

    outputs = _call_output_parser(parse_outputs, top.get("outputs", {}))
    return EffectiveProcessEntry(
        runtime=base_runtime,
        plugins=effective_plugins,
        outputs=outputs,
    )


__all__ = [
    "EffectiveProcessEntry",
    "OutputGrammarParser",
    "RawProcessEntry",
    "RuntimeSpec",
    "parse_effective_process_mapping",
    "parse_raw_process_mapping",
    "parse_runtime",
    "validate_variant_process_sections",
]
