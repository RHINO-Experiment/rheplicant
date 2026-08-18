"""JAX-free grammar for process-entry and runtime sections."""

from __future__ import annotations

import keyword
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Protocol

from _rheplicant_bootstrap.errors import ConfigError
from _rheplicant_bootstrap.frozen import (
    freeze_evidence,
    static_isinstance,
    static_type_name,
)
from _rheplicant_bootstrap.layering import parse_default
from _rheplicant_bootstrap.presets import PresetRequest

_RUNTIME_KEYS = frozenset({"jax_enable_x64", "platform", "seed", "seeds"})
_PLATFORMS = ("auto", "cpu", "gpu", "tpu")
_PROCESS_SECTIONS = ("defaults", "plugins", "outputs")
_DEFAULTS_FORM_ERROR = (
    "defaults: must be a list whose entries are preset names or mappings "
    "with exactly 'from' and optional 'only'."
)
_PROCESS_MAPPING_ENTRY_LIMIT = 250_000
_PROCESS_SEQUENCE_ENTRY_LIMIT = 250_000
_RUNTIME_MAPPING_ENTRY_LIMIT = 250_000
_RUNTIME_RENDER_DEPTH_LIMIT = 8
_RUNTIME_RENDER_NODE_LIMIT = 64
_RUNTIME_RENDER_TEXT_LIMIT = 120
_RUNTIME_RENDER_INTEGER_BIT_LIMIT = _RUNTIME_RENDER_TEXT_LIMIT * 4
_MISSING = object()


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
        try:
            raw_schema_version = object.__getattribute__(
                self, "schema_version"
            )
            raw_defaults = object.__getattribute__(self, "defaults")
            raw_plugins = object.__getattribute__(self, "plugins")
            raw_runtime = object.__getattribute__(self, "raw_runtime")
            raw_outputs = object.__getattribute__(self, "outputs")
        except Exception:
            raise ConfigError("raw process entry is malformed.") from None
        if (
            static_isinstance(raw_schema_version, bool)
            or not static_isinstance(raw_schema_version, int)
            or int.__int__(raw_schema_version) != 1
        ):
            raise ConfigError("schema_version: 1 is required at the process entry point.")
        if static_isinstance(raw_defaults, (str, bytes)) or not static_isinstance(
            raw_defaults, Sequence
        ):
            raise ConfigError(_DEFAULTS_FORM_ERROR)
        defaults = tuple(
            _sequence_items(
                raw_defaults,
                failure=_DEFAULTS_FORM_ERROR,
                limit_message=(
                    "defaults: sequence entry count exceeds limit "
                    f"{_PROCESS_SEQUENCE_ENTRY_LIMIT}."
                ),
            )
        )
        if any(type(item) is not PresetRequest for item in defaults):
            raise ConfigError(_DEFAULTS_FORM_ERROR)
        canonical_defaults = tuple(
            PresetRequest(item.name, item.only) for item in defaults
        )
        if len({item.name for item in canonical_defaults}) != len(
            canonical_defaults
        ):
            raise ConfigError("defaults: duplicate package preset.")
        plugins = _parse_plugins(raw_plugins)
        frozen_runtime = raw_runtime
        if raw_runtime is not None:
            if not static_isinstance(raw_runtime, Mapping):
                raise ConfigError("raw process runtime must be a mapping or null.")
            frozen_runtime = freeze_evidence(
                raw_runtime, where="raw process runtime"
            )
            if not static_isinstance(frozen_runtime, Mapping):
                raise ConfigError("raw process runtime must be a mapping or null.")
        frozen_outputs = _freeze_output(raw_outputs)
        object.__setattr__(self, "schema_version", 1)
        object.__setattr__(self, "defaults", canonical_defaults)
        object.__setattr__(self, "plugins", plugins)
        object.__setattr__(self, "raw_runtime", frozen_runtime)
        object.__setattr__(self, "outputs", frozen_outputs)


@dataclass(frozen=True, slots=True)
class RuntimeSpec:
    jax_enable_x64: bool
    platform: Literal["auto", "cpu", "gpu", "tpu"]
    seed: int | None
    seeds: Mapping[str, int]

    def __post_init__(self) -> None:
        try:
            raw_x64 = object.__getattribute__(self, "jax_enable_x64")
            raw_platform = object.__getattribute__(self, "platform")
            raw_seed = object.__getattribute__(self, "seed")
            raw_seeds = object.__getattribute__(self, "seeds")
        except Exception:
            raise ConfigError("runtime specification is malformed.") from None
        if not static_isinstance(raw_x64, bool):
            raise ConfigError("runtime.jax_enable_x64 is a bool.")
        if not static_isinstance(raw_platform, str):
            raise ConfigError(f"runtime.platform is one of {list(_PLATFORMS)}.")
        platform = str.__str__(raw_platform)
        if platform not in _PLATFORMS:
            raise ConfigError(f"runtime.platform is one of {list(_PLATFORMS)}.")
        if raw_seed is not None and (
            static_isinstance(raw_seed, bool)
            or not static_isinstance(raw_seed, int)
        ):
            raise ConfigError("runtime.seed is an int or null.")
        frozen_seeds = freeze_evidence(raw_seeds, where="runtime seeds")
        if not static_isinstance(frozen_seeds, Mapping):
            raise ConfigError("runtime.seeds is a mapping of name -> int.")
        canonical_seeds: dict[str, int] = {}
        for name, value in frozen_seeds.items():
            if static_isinstance(value, bool) or not static_isinstance(value, int):
                raise ConfigError(f"runtime.seeds.{name} is an int.")
            canonical_seeds[name] = int.__int__(value)
        object.__setattr__(self, "jax_enable_x64", raw_x64)
        object.__setattr__(self, "platform", platform)
        object.__setattr__(
            self, "seed", None if raw_seed is None else int.__int__(raw_seed)
        )
        object.__setattr__(self, "seeds", MappingProxyType(canonical_seeds))


@dataclass(frozen=True, slots=True)
class EffectiveProcessEntry:
    runtime: RuntimeSpec
    plugins: Sequence[str]
    outputs: object

    def __post_init__(self) -> None:
        try:
            raw_runtime = object.__getattribute__(self, "runtime")
            raw_plugins = object.__getattribute__(self, "plugins")
            raw_outputs = object.__getattribute__(self, "outputs")
        except Exception:
            raise ConfigError("effective process entry is malformed.") from None
        if type(raw_runtime) is not RuntimeSpec:
            raise ConfigError("effective runtime must be a RuntimeSpec.")
        runtime = _copy_runtime_spec(raw_runtime)
        plugins = _parse_plugins(raw_plugins)
        outputs = _freeze_output(raw_outputs)
        object.__setattr__(self, "runtime", runtime)
        object.__setattr__(self, "plugins", plugins)
        object.__setattr__(self, "outputs", outputs)


def _copy_runtime_spec(value: RuntimeSpec) -> RuntimeSpec:
    try:
        x64 = object.__getattribute__(value, "jax_enable_x64")
        platform = object.__getattribute__(value, "platform")
        seed = object.__getattribute__(value, "seed")
        seeds = object.__getattribute__(value, "seeds")
    except Exception:
        raise ConfigError("runtime specification is malformed.") from None
    return RuntimeSpec(x64, platform, seed, seeds)


def _freeze_output(value: object) -> object:
    if static_isinstance(
        value,
        (Mapping, Sequence, str, int, float, bytes, bytearray, memoryview),
    ):
        return freeze_evidence(value, where="outputs")
    return value


def _mapping_pairs(
    mapping: Mapping,
    *,
    failure: str,
    limit: int | None = None,
    limit_message: str | None = None,
):
    try:
        iterator = iter(mapping.items())
    except Exception:
        raise ConfigError(failure) from None
    position = 0
    while True:
        try:
            pair = next(iterator)
        except StopIteration:
            return
        except Exception:
            raise ConfigError(failure) from None
        position += 1
        if limit is not None and position > limit:
            if limit_message is None:
                raise ConfigError(failure)
            raise ConfigError(limit_message)
        try:
            key, value = pair
        except Exception:
            raise ConfigError(failure) from None
        yield key, value


def _sequence_items(
    sequence: object,
    *,
    failure: str,
    limit_message: str,
):
    try:
        iterator = iter(sequence)
    except Exception:
        raise ConfigError(failure) from None
    position = 0
    while True:
        try:
            item = next(iterator)
        except StopIteration:
            return
        except Exception:
            raise ConfigError(failure) from None
        position += 1
        if position > _PROCESS_SEQUENCE_ENTRY_LIMIT:
            raise ConfigError(limit_message)
        yield item


def _top_level(document: Mapping[str, object]) -> dict[str, object]:
    if not static_isinstance(document, Mapping):
        raise ConfigError(
            "document: configuration root must be a mapping; got "
            f"{static_type_name(document)}."
        )
    result: dict[str, object] = {}
    for position, (key, value) in enumerate(
        _mapping_pairs(
            document,
            failure="document: top-level mapping traversal failed.",
            limit=_PROCESS_MAPPING_ENTRY_LIMIT,
            limit_message=(
                "document: top-level mapping entry count exceeds limit "
                f"{_PROCESS_MAPPING_ENTRY_LIMIT}."
            ),
        ),
        start=1,
    ):
        if not static_isinstance(key, str):
            detail = (
                f" {int.__int__(key)}"
                if static_isinstance(key, int)
                and not static_isinstance(key, bool)
                else ""
            )
            raise ConfigError(
                f"document: top-level key {position} must be a string; got "
                f"{static_type_name(key)}{detail}."
            )
        exact = str.__str__(key)
        if exact in result:
            raise ConfigError(
                "document: top-level keys collide after canonicalization."
            )
        result[exact] = value
    return result


def _parse_defaults(raw: object) -> tuple[PresetRequest, ...]:
    if static_isinstance(raw, (str, bytes)) or not static_isinstance(
        raw, Sequence
    ):
        raise ConfigError(_DEFAULTS_FORM_ERROR)
    entries = tuple(
        _sequence_items(
            raw,
            failure=_DEFAULTS_FORM_ERROR,
            limit_message=(
                "defaults: sequence entry count exceeds limit "
                f"{_PROCESS_SEQUENCE_ENTRY_LIMIT}."
            ),
        )
    )
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
    if not static_isinstance(raw, str):
        raise ConfigError(
            f"plugins[{index}]: module name must be a string; got "
            f"{static_type_name(raw)}."
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
    if static_isinstance(raw, (str, bytes)) or not static_isinstance(
        raw, Sequence
    ):
        raise ConfigError("plugins: must be a list of Python module names.")
    entries = tuple(
        _sequence_items(
            raw,
            failure="plugins: sequence traversal failed.",
            limit_message=(
                "plugins: sequence entry count exceeds limit "
                f"{_PROCESS_SEQUENCE_ENTRY_LIMIT}."
            ),
        )
    )
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
            "outputs: grammar parser raised " f"{static_type_name(error)}."
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
        static_isinstance(version, bool)
        or not static_isinstance(version, int)
        or int.__int__(version) != 1
    ):
        raise ConfigError("schema_version: 1 is required at the process entry point.")
    defaults = _parse_defaults(top["defaults"]) if "defaults" in top else ()
    plugins = _parse_plugins(top["plugins"]) if "plugins" in top else ()
    if "runtime" in top:
        raw_runtime = top["runtime"]
        if not static_isinstance(raw_runtime, Mapping):
            raise ConfigError("runtime: must be a mapping when present.")
        frozen_runtime = freeze_evidence(raw_runtime, where="runtime")
        if not static_isinstance(frozen_runtime, Mapping):
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
    if not static_isinstance(section, Mapping):
        raise ConfigError(
            "runtime: is a mapping; got " f"{_runtime_description(section)}."
        )
    canonical: dict[str, object] = {}
    for key, value in _mapping_pairs(
        section,
        failure="runtime: mapping traversal failed.",
        limit=_RUNTIME_MAPPING_ENTRY_LIMIT,
        limit_message=(
            "runtime: mapping entry count exceeds limit "
            f"{_RUNTIME_MAPPING_ENTRY_LIMIT}."
        ),
    ):
        if not static_isinstance(key, str):
            raise ConfigError(
                "runtime: keys are strings; got " f"{static_type_name(key)}."
            )
        exact_key = str.__str__(key)
        if exact_key in canonical:
            raise ConfigError(
                "runtime: keys collide after canonicalization."
            )
        canonical[exact_key] = value
    unknown = sorted(set(canonical) - _RUNTIME_KEYS)
    if unknown:
        message = (
            f"runtime: the runtime section does not take {unknown}; it takes "
            f"{sorted(_RUNTIME_KEYS)}."
        )
        if "x64_required_by" in unknown:
            message += (
                " x64_required_by is emitted by the loader into the resolved "
                "record, never written by hand."
            )
        raise ConfigError(message)

    x64 = canonical.get("jax_enable_x64", False)
    if not static_isinstance(x64, bool):
        x64_type, x64_rendered = _runtime_description_parts(x64)
        if x64_rendered is not None:
            raise ConfigError(
                f"runtime.jax_enable_x64 is a bool; got {x64_type} "
                f"({x64_rendered})."
            )
        raise ConfigError(f"runtime.jax_enable_x64 is a bool; got {x64_type}.")
    platform = canonical.get("platform", "auto")
    if not static_isinstance(platform, str):
        raise ConfigError(
            f"runtime.platform is one of {list(_PLATFORMS)}; got "
            f"{_runtime_render(platform)}."
        )
    exact_platform = str.__str__(platform)
    if exact_platform not in _PLATFORMS:
        raise ConfigError(
            f"runtime.platform is one of {list(_PLATFORMS)}; got "
            f"{exact_platform!r}."
        )

    seed = canonical.get("seed")
    if seed is not None and (
        static_isinstance(seed, bool) or not static_isinstance(seed, int)
    ):
        seed_type, seed_rendered = _runtime_description_parts(seed)
        if seed_rendered is not None:
            raise ConfigError(
                f"runtime.seed is an int or null; got {seed_type} "
                f"({seed_rendered}). null is legal and recorded -- it means "
                "State.key = None, and a run that realises randomness will "
                "say so when it asks for the key."
            )
        raise ConfigError(
            f"runtime.seed is an int or null; got {seed_type}. null is legal "
            "and recorded -- it "
            "means State.key = None, and a run that realises randomness will "
            "say so when it asks for the key."
        )
    canonical_seed = None if seed is None else int.__int__(seed)

    raw_seeds = canonical.get("seeds", MappingProxyType({}))
    if not static_isinstance(raw_seeds, Mapping):
        seeds_type, seeds_rendered = _runtime_description_parts(raw_seeds)
        if seeds_rendered is not None:
            raise ConfigError(
                f"runtime.seeds is a mapping of name -> int; got {seeds_type} "
                f"({seeds_rendered})."
            )
        raise ConfigError(
            f"runtime.seeds is a mapping of name -> int; got {seeds_type}."
        )
    seeds: dict[str, int] = {}
    for name, value in _mapping_pairs(
        raw_seeds,
        failure="runtime.seeds: mapping traversal failed.",
        limit=_RUNTIME_MAPPING_ENTRY_LIMIT,
        limit_message=(
            "runtime.seeds: mapping entry count exceeds limit "
            f"{_RUNTIME_MAPPING_ENTRY_LIMIT}."
        ),
    ):
        if not static_isinstance(name, str):
            raise ConfigError(
                "runtime.seeds keys are strings; got " f"{_runtime_render(name)}."
            )
        exact_name = str.__str__(name)
        if exact_name in seeds:
            raise ConfigError(
                "runtime.seeds keys collide after canonicalization."
            )
        if static_isinstance(value, bool) or not static_isinstance(value, int):
            raise ConfigError(
                f"runtime.seeds.{exact_name} is an int; got "
                f"{_runtime_description(value)}."
            )
        seeds[exact_name] = int.__int__(value)
    return RuntimeSpec(
        jax_enable_x64=bool(x64),
        platform=exact_platform,  # type: ignore[arg-type]
        seed=canonical_seed,
        seeds=MappingProxyType(seeds),
    )


def _runtime_render(value: object) -> str:
    """Render a small exact-built-in tree without invoking foreign hooks."""
    active: set[int] = set()
    nodes = 0

    def render(item: object, depth: int) -> str | None:
        nonlocal nodes
        nodes += 1
        if (
            nodes > _RUNTIME_RENDER_NODE_LIMIT
            or depth > _RUNTIME_RENDER_DEPTH_LIMIT
        ):
            return None
        item_type = type(item)
        if item is None:
            return "None"
        if item_type is bool:
            return "True" if item else "False"
        if item_type is int:
            if int.bit_length(item) > _RUNTIME_RENDER_INTEGER_BIT_LIMIT:
                return None
            try:
                result = int.__repr__(item)
            except Exception:
                return None
            return result if len(result) <= _RUNTIME_RENDER_TEXT_LIMIT else None
        if item_type is float:
            try:
                result = float.__repr__(item)
            except Exception:
                return None
            return result if len(result) <= _RUNTIME_RENDER_TEXT_LIMIT else None
        if item_type is str:
            if str.__len__(item) > _RUNTIME_RENDER_TEXT_LIMIT:
                return None
            try:
                result = repr(str.__str__(item))
            except Exception:
                return None
            return result if len(result) <= _RUNTIME_RENDER_TEXT_LIMIT else None
        if item_type is not list and item_type is not tuple and item_type is not dict:
            return None
        identity = id(item)
        if identity in active:
            return None
        active.add(identity)
        try:
            children: list[str] = []
            if item_type is dict:
                try:
                    iterator = iter(dict.items(item))
                    for key, child in iterator:
                        rendered_key = render(key, depth + 1)
                        rendered_child = render(child, depth + 1)
                        if rendered_key is None or rendered_child is None:
                            return None
                        children.append(f"{rendered_key}: {rendered_child}")
                        if sum(map(len, children)) > _RUNTIME_RENDER_TEXT_LIMIT:
                            return None
                except Exception:
                    return None
                result = "{" + ", ".join(children) + "}"
            else:
                iterator = (
                    list.__iter__(item)
                    if item_type is list
                    else tuple.__iter__(item)
                )
                for child in iterator:
                    rendered_child = render(child, depth + 1)
                    if rendered_child is None:
                        return None
                    children.append(rendered_child)
                    if sum(map(len, children)) > _RUNTIME_RENDER_TEXT_LIMIT:
                        return None
            if item_type is list:
                result = "[" + ", ".join(children) + "]"
            elif item_type is tuple:
                suffix = "," if len(children) == 1 else ""
                result = "(" + ", ".join(children) + suffix + ")"
            if len(result) > _RUNTIME_RENDER_TEXT_LIMIT:
                return None
            return result
        finally:
            active.remove(identity)

    rendered = render(value, 0)
    if rendered is not None:
        return rendered
    return static_type_name(value)


def _runtime_description(value: object) -> str:
    name, rendered = _runtime_description_parts(value)
    if rendered is None:
        return name
    return f"{name} ({rendered})"


def _runtime_description_parts(value: object) -> tuple[str, str | None]:
    name = static_type_name(value)
    rendered = _runtime_render(value)
    if rendered == name:
        return name, None
    return name, rendered


def _variant_pairs(document: dict[str, object]):
    variants = document.get("variants", _MISSING)
    if variants is _MISSING:
        return ()
    if not static_isinstance(variants, Mapping):
        raise ConfigError(
            "variants: is a mapping of name -> patch; got "
            f"{static_type_name(variants)}."
        )
    return _mapping_pairs(
        variants,
        failure="variants: mapping traversal failed.",
        limit=_PROCESS_MAPPING_ENTRY_LIMIT,
        limit_message=(
            "variants: mapping entry count exceeds limit "
            f"{_PROCESS_MAPPING_ENTRY_LIMIT}."
        ),
    )


def validate_variant_process_sections(
    document: Mapping[str, object],
) -> None:
    """Refuse a variant touching document-global process-entry sections."""
    top = _top_level(document)
    for raw_name, patch in _variant_pairs(top):
        if not static_isinstance(raw_name, str):
            raise ConfigError(
                "variants: names must be non-empty strings; got "
                f"{static_type_name(raw_name)}."
            )
        name = str.__str__(raw_name)
        if not name:
            raise ConfigError("variants: names must be non-empty strings; got ''.")
        if not static_isinstance(patch, Mapping):
            continue
        keys: set[str] = set()
        for raw_key, _ in _mapping_pairs(
            patch,
            failure=f"variants.{name}: patch mapping traversal failed.",
            limit=_PROCESS_MAPPING_ENTRY_LIMIT,
            limit_message=(
                f"variants.{name}: patch mapping entry count exceeds limit "
                f"{_PROCESS_MAPPING_ENTRY_LIMIT}."
            ),
        ):
            if not static_isinstance(raw_key, str):
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
    try:
        raw_plugins_value = object.__getattribute__(raw, "plugins")
    except Exception:
        raise ConfigError("effective raw process entry is invalid.") from None
    raw_plugins = _parse_plugins(raw_plugins_value)
    top = _top_level(document)
    validate_variant_process_sections(top)
    if "runtime" not in top:
        raise ConfigError("runtime: is required after package presets are layered.")
    base_runtime = parse_runtime(top["runtime"])
    effective_plugins = (
        _parse_plugins(top["plugins"]) if "plugins" in top else ()
    )
    if effective_plugins != raw_plugins:
        raise ConfigError(
            "plugins: effective process-entry list differs from the raw "
            "process entry."
        )
    if static_isinstance(layers, (str, bytes)) or not static_isinstance(
        layers, Sequence
    ):
        raise ConfigError("effective layer sequence must be a sequence.")
    frozen_layers = tuple(
        _sequence_items(
            layers,
            failure="effective layer sequence traversal failed.",
            limit_message=(
                "effective layer sequence entry count exceeds limit "
                f"{_PROCESS_SEQUENCE_ENTRY_LIMIT}."
            ),
        )
    )
    from _rheplicant_bootstrap.variants import LayerRef

    if any(type(layer) is not LayerRef for layer in frozen_layers):
        raise ConfigError(
            "effective layer sequence must contain LayerRef values."
        )

    layer_snapshots: list[tuple[str, str | None, Mapping]] = []
    for layer in frozen_layers:
        try:
            raw_kind = object.__getattribute__(layer, "kind")
            raw_name = object.__getattribute__(layer, "name")
            layer_document = object.__getattribute__(layer, "document")
        except Exception:
            raise ConfigError("effective layer record is invalid.") from None
        if not static_isinstance(raw_kind, str):
            raise ConfigError("effective layer record is invalid.")
        kind = str.__str__(raw_kind)
        if kind == "base":
            if raw_name is not None:
                raise ConfigError("effective layer record is invalid.")
            name = None
        elif kind == "variant":
            if not static_isinstance(raw_name, str):
                raise ConfigError("effective layer record is invalid.")
            name = str.__str__(raw_name)
            if not name:
                raise ConfigError("effective layer record is invalid.")
        else:
            raise ConfigError("effective layer record is invalid.")
        if not static_isinstance(layer_document, Mapping):
            raise ConfigError("effective layer record is invalid.")
        layer_snapshots.append((kind, name, layer_document))

    mismatches: list[str] = []
    for kind, name, layer_document in layer_snapshots:
        if kind != "variant":
            continue
        if name is None:
            raise ConfigError("effective layer record is invalid.")
        try:
            layer_runtime = None
            layer_keys: set[str] = set()
            for raw_key, raw_value in _mapping_pairs(
                layer_document,
                failure="effective layer mapping traversal failed.",
                limit=_RUNTIME_MAPPING_ENTRY_LIMIT,
                limit_message=(
                    "effective layer mapping entry count exceeds limit "
                    f"{_RUNTIME_MAPPING_ENTRY_LIMIT}."
                ),
            ):
                if not static_isinstance(raw_key, str):
                    raise ConfigError("effective layer record is invalid.")
                key = str.__str__(raw_key)
                if key in layer_keys:
                    raise ConfigError("effective layer record is invalid.")
                layer_keys.add(key)
                if key == "runtime":
                    layer_runtime = raw_value
            runtime = parse_runtime(layer_runtime)
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
