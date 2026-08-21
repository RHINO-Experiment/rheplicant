"""Build complete resolved documents and their deterministic YAML artefacts."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from _rheplicant_bootstrap.errors import ConfigError
from _rheplicant_bootstrap.frozen import freeze_evidence, static_isinstance, thaw
from _rheplicant_bootstrap.layering import OriginNode
from _rheplicant_bootstrap.types import JsonValue, LayerIdentity, Origin

from .names import validate_encoded_names
from .types import PathEncoding, ResolvedLayerRecord
from .yaml import dump_resolved_yaml


@dataclass(frozen=True, slots=True)
class ResolvedArtefact:
    layer: LayerIdentity
    relative_path: str
    payload: bytes
    sha256: str


@dataclass(frozen=True, slots=True)
class ResolvedArtefactSet:
    files: Sequence[ResolvedArtefact]
    path_encodings: Sequence[PathEncoding]


@dataclass(slots=True)
class _MutableOrigin:
    origin: Origin | None
    children: dict[str | int, _MutableOrigin]


_PATH_PART = re.compile(r"([^\[\]]+)(\[\])?")
_SCHEMA_ROOTS = frozenset(("runtime", "observation", "resources", "model", "inference", "runs"))


def _mutable_json(value: object, *, where: str) -> object:
    frozen = freeze_evidence(value, where=where)
    copied = thaw(frozen)
    if isinstance(copied, dict):
        return copied
    if isinstance(copied, list):
        return copied
    if copied is None or type(copied) in (bool, int, float, str):
        return copied
    raise ConfigError(f"{where} is not plain JSON.")


def _copy_origins(document: object, node: OriginNode, *, path: str) -> _MutableOrigin:
    if type(node) is not OriginNode:
        raise ConfigError(f"resolved origin node is invalid at {path or '<root>'}.")
    if isinstance(document, Mapping):
        keys = tuple(document)
        if tuple(node.children) != keys:
            raise ConfigError(f"resolved document/origin shape differs at {path or '<root>'}.")
        return _MutableOrigin(
            node.origin,
            {
                key: _copy_origins(
                    document[key], node.children[key], path=key if not path else f"{path}.{key}"
                )
                for key in keys
            },
        )
    if isinstance(document, list | tuple):
        indexes = tuple(range(len(document)))
        if tuple(node.children) != indexes:
            raise ConfigError(f"resolved document/origin shape differs at {path or '<root>'}.")
        return _MutableOrigin(
            node.origin,
            {
                index: _copy_origins(document[index], node.children[index], path=f"{path}[{index}]")
                for index in indexes
            },
        )
    if node.children:
        raise ConfigError(f"resolved scalar has child origins at {path or '<root>'}.")
    return _MutableOrigin(node.origin, {})


def _default_origin(value: object) -> _MutableOrigin:
    origin = Origin("rheplicant-default")
    if isinstance(value, dict):
        return _MutableOrigin(origin, {key: _default_origin(child) for key, child in value.items()})
    if isinstance(value, list | tuple):
        return _MutableOrigin(
            origin,
            {index: _default_origin(child) for index, child in enumerate(value)},
        )
    return _MutableOrigin(origin, {})


def _freeze_origins(node: _MutableOrigin) -> OriginNode:
    return OriginNode(
        node.origin,
        {key: _freeze_origins(child) for key, child in node.children.items()},
    )


def _tokens(path: str) -> tuple[str, ...]:
    tokens: list[str] = []
    for part in path.split("."):
        match = _PATH_PART.fullmatch(part)
        if match is None:
            return ()
        tokens.append(match.group(1))
        if match.group(2) is not None:
            tokens.append("[]")
    return tuple(tokens)


def _apply_default(
    document: dict[str, object],
    origins: _MutableOrigin,
    *,
    path: str,
    value: object,
) -> None:
    tokens = _tokens(path)
    if not tokens:
        raise ConfigError(f"resolved default path {path!r} is invalid.")
    if tokens[0] not in document and (len(tokens) > 1 and tokens[0] not in _SCHEMA_ROOTS):
        return

    def apply(current: object, node: _MutableOrigin, offset: int) -> None:
        token = tokens[offset]
        if token == "[]":
            if isinstance(current, dict):
                pairs = tuple(current.items())
            elif isinstance(current, list):
                pairs = tuple(enumerate(current))
            else:
                return
            for key, child in pairs:
                apply(child, node.children[key], offset + 1)
            return
        if not isinstance(current, dict):
            return
        final = offset == len(tokens) - 1
        if final:
            if token not in current:
                copied = _mutable_json(value, where=f"resolved default {path}")
                current[token] = copied
                node.children[token] = _default_origin(copied)
            return
        if token not in current:
            if "[]" in tokens[offset + 1 :]:
                return
            current[token] = {}
            node.children[token] = _default_origin({})
        apply(current[token], node.children[token], offset + 1)

    apply(document, origins, 0)


def _sequence(value: object, *, where: str) -> tuple[Mapping[str, object], ...]:
    if type(value) is not tuple:
        raise ConfigError(f"resolved layer audit {where} must be a sequence.")
    if any(not static_isinstance(row, Mapping) for row in value):
        raise ConfigError(f"resolved layer audit {where} rows must be mappings.")
    return cast(tuple[Mapping[str, object], ...], value)


def _descriptor(row: object) -> Mapping[str, JsonValue]:
    if not static_isinstance(row, Mapping):
        raise ConfigError("resolved run descriptor must be a mapping.")
    keys = ("index", "name", "kind", "variant")
    if set(row) != set(keys):
        raise ConfigError("resolved run descriptor has the wrong fields.")
    return cast(Mapping[str, JsonValue], freeze_evidence(row, where="run descriptor"))


def _namespace(record: ResolvedLayerRecord) -> Mapping[str, JsonValue]:
    audit = record.audit
    required = (
        "presets",
        "defaults",
        "deliveries",
        "inputs",
        "findings",
        "python_targets",
        "seeds",
        "resources",
        "gates",
        "parsed_runs",
        "deletions",
        "switch_map",
        "truth",
        "truth_omissions",
        "x64_required_by",
    )
    if set(audit) != set(required):
        raise ConfigError("resolved layer audit has the wrong fields.")
    deliveries = _sequence(audit["deliveries"], where="deliveries")
    numeric = {}
    for row in deliveries:
        origin = row["origin"]
        if not static_isinstance(origin, Mapping):
            raise ConfigError("resolved numeric origin must be a mapping.")
        numeric[row["document_path"]] = {
            "dtype": row["dtype"],
            "origin": Origin(origin["kind"], origin["name"]).render(),
        }
    inputs = tuple(
        {
            "document_path": row["document_path"],
            "path": row["path"],
            "format": row["format"],
            "sha256": row["sha256"],
        }
        for row in _sequence(audit["inputs"], where="inputs")
    )
    resources = _sequence(audit["resources"], where="resources")
    resource = {"build_order": (), "shared_objects": {}} if not resources else resources[-1]
    gates = {
        row["name"]: {
            "schema_id": row["schema_id"],
            "declared_mode": row["declared_mode"],
            "effective_state": row["effective_state"],
            "reason": row["reason"],
        }
        for row in _sequence(audit["gates"], where="gates")
    }
    deletions = tuple(
        {
            "path": row["path"],
            "origin": Origin(row["origin"]["kind"], row["origin"]["name"]).render(),
        }
        for row in _sequence(audit["deletions"], where="deletions")
    )
    return cast(
        Mapping[str, JsonValue],
        freeze_evidence(
            {
                "format_version": 1,
                "layer": {"kind": record.layer.kind, "name": record.layer.name},
                "presets": audit["presets"],
                "numeric": numeric,
                "inputs": inputs,
                "resources": resource,
                "observation": {"switch_map": audit["switch_map"]},
                "inference": {
                    "truth": audit["truth"],
                    "truth_omissions": audit["truth_omissions"],
                    "effective_gates": gates,
                },
                "runs": {
                    "declared_runs_in_layer": tuple(
                        {
                            "index": row.index,
                            "name": row.name,
                            "kind": row.kind,
                            "variant": row.variant,
                        }
                        for row in record.declared_runs
                    ),
                    "execution_runs": tuple(
                        {
                            "index": row.index,
                            "name": row.name,
                            "kind": row.kind,
                            "variant": row.variant,
                        }
                        for row in record.execution_runs
                    ),
                },
                "runtime": {"x64_required_by": audit["x64_required_by"]},
                "deletions": deletions,
            },
            where="resolved namespace",
        ),
    )


def build_resolved_document(
    record: ResolvedLayerRecord,
) -> tuple[Mapping[str, JsonValue], OriginNode]:
    """Return detached parallel document and origin trees."""
    if type(record) is not ResolvedLayerRecord:
        raise ConfigError("resolved document requires an exact ResolvedLayerRecord.")
    if "_rheplicant_resolved" in record.effective_document:
        raise ConfigError("input document contains emitted-only _rheplicant_resolved.")
    document = _mutable_json(record.effective_document, where="effective document")
    if not isinstance(document, dict):
        raise ConfigError("effective document must be a mapping.")
    origins = _copy_origins(document, record.origins, path="")
    namespace_row = _namespace(record)
    defaults = _sequence(record.audit["defaults"], where="defaults")
    for row in defaults:
        if set(row) != {"path", "value"} or type(row["path"]) is not str:
            raise ConfigError("resolved default row is malformed.")
        _apply_default(document, origins, path=row["path"], value=row["value"])
    namespace = _mutable_json(namespace_row, where="resolved namespace")
    document["_rheplicant_resolved"] = namespace
    origins.children["_rheplicant_resolved"] = _default_origin(namespace)
    frozen = freeze_evidence(document, where="resolved document")
    if not static_isinstance(frozen, Mapping):
        raise ConfigError("resolved document must remain a mapping.")
    return cast(Mapping[str, JsonValue], frozen), _freeze_origins(origins)


def build_resolved_artefacts(
    records: Sequence[ResolvedLayerRecord],
    *,
    run_names: Sequence[str],
    variant_names: Sequence[str],
    component_limit: int,
) -> ResolvedArtefactSet:
    """Validate both name spaces, then emit only completed layers."""
    run_encodings = validate_encoded_names("run", run_names, component_limit=component_limit)
    variant_encodings = validate_encoded_names(
        "variant", variant_names, component_limit=component_limit
    )
    variant_by_name = {row.document_name: row.encoded_name for row in variant_encodings}
    files: list[ResolvedArtefact] = []
    seen: set[LayerIdentity] = set()
    for record in records:
        if type(record) is not ResolvedLayerRecord:
            raise ConfigError("resolved artefact records must be exact typed rows.")
        if record.layer in seen:
            raise ConfigError("resolved artefact layer is duplicated.")
        seen.add(record.layer)
        if record.layer.kind == "base":
            relative_path = "config.resolved.yaml"
        else:
            name = record.layer.name
            if name not in variant_by_name:
                raise ConfigError("resolved variant layer was not declared.")
            relative_path = f"variants/{variant_by_name[name]}/config.resolved.yaml"
        document, origins = build_resolved_document(record)
        payload = dump_resolved_yaml(document, origins)
        files.append(
            ResolvedArtefact(
                record.layer,
                relative_path,
                payload,
                hashlib.sha256(payload).hexdigest(),
            )
        )
    return ResolvedArtefactSet(
        files=tuple(files),
        path_encodings=(*run_encodings, *variant_encodings),
    )


__all__ = [
    "ResolvedArtefact",
    "ResolvedArtefactSet",
    "build_resolved_artefacts",
    "build_resolved_document",
]
