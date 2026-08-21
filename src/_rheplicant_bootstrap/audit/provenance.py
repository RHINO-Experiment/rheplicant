"""Strict projection of an audit snapshot into provenance JSON."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import cast

from _rheplicant_bootstrap.audit.types import (
    ArtefactRecord,
    ArtefactTable,
    AuditSnapshot,
    InputRecord,
    PathEncoding,
    PythonTargetRecord,
    ResourceRecord,
    RunOutcomeRecord,
    SeedRecord,
    VariantRecord,
)
from _rheplicant_bootstrap.errors import ConfigError
from _rheplicant_bootstrap.plugins import (
    PLUGIN_DISTRIBUTION_ROW_KEYS,
    PLUGIN_ROW_KEYS,
)
from _rheplicant_bootstrap.runtime import (
    RUNTIME_ACTUAL_ROW_KEYS,
    RUNTIME_PRIOR_ENVIRONMENT_ROW_KEYS,
    RUNTIME_REQUESTED_ROW_KEYS,
    RUNTIME_ROW_KEYS,
)
from _rheplicant_bootstrap.types import (
    CompletedBoundary,
    JsonValue,
    LayerIdentity,
    RunDescriptor,
    Status,
)

from .software import validate_software
from .trace import LAYER_ROW_KEYS, STAGES

STATUSES = ("ok", "refused", "error")
ARTEFACT_REASONS = (
    "metadata_envelope",
    "transaction_not_reached",
    "boundary_not_reached",
    "layer_not_complete",
    "not_applicable_to_status",
)
PROVENANCE_KEYS = (
    "format_version",
    "status",
    "completed_boundaries",
    "bootstrap",
    "software",
    "runtime",
    "inputs",
    "plugins",
    "python_targets",
    "seeds",
    "variants",
    "resources",
    "runs",
    "path_encodings",
    "artefacts",
)
ARTEFACT_TABLE_KEYS = (
    "marker",
    "lock",
    "journal",
    "input",
    "resolved_base",
    "resolved_variants",
    "provenance",
    "diagnostics",
)
ARTEFACT_RECORD_KEYS = ("relative_path", "written", "bytes", "sha256", "reason")
_BOOTSTRAP_KEYS = (
    "protocol_version",
    "launch_mode",
    "input_sha256",
    "presets",
    "source_name",
    "source_path",
    "source_realpath",
    "base_dir",
)
_METADATA = frozenset(("marker", "lock", "journal", "provenance", "diagnostics"))
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _layer(layer: LayerIdentity | None) -> Mapping[str, JsonValue] | None:
    if layer is None:
        return None
    if type(layer) is not LayerIdentity:
        raise ConfigError("audit envelope layer is invalid.")
    return {"kind": layer.kind, "name": layer.name}


def _descriptor(row) -> Mapping[str, JsonValue]:
    if type(row) is not RunDescriptor:
        raise ConfigError("audit run descriptor is not exact.")
    return {
        "index": row.index,
        "name": row.name,
        "kind": row.kind,
        "variant": row.variant,
    }


def completed_boundaries(snapshot: AuditSnapshot) -> Sequence[Mapping[str, JsonValue]]:
    rows = []
    for boundary in snapshot.completed_boundaries:
        if type(boundary) is not CompletedBoundary:
            raise ConfigError("audit snapshot boundary is not exact.")
        if boundary.stage not in STAGES:
            raise ConfigError("audit snapshot contains an unknown boundary.")
        rows.append({"stage": boundary.stage, "layer": _layer(boundary.layer)})
    return tuple(rows)


def _exact_mapping(value: object, keys: Sequence[str], where: str) -> Mapping[str, JsonValue]:
    if not isinstance(value, Mapping) or tuple(value) != tuple(keys):
        raise ConfigError(f"{where} has the wrong fields.")
    return cast(Mapping[str, JsonValue], value)


def _artefact_record(row: object, *, where: str, metadata: bool) -> Mapping[str, JsonValue]:
    if type(row) is not ArtefactRecord:
        raise ConfigError(f"{where} is not an exact ArtefactRecord.")
    if type(row.relative_path) is not str or not row.relative_path:
        raise ConfigError(f"{where} has an invalid relative path.")
    if type(row.written) is not bool:
        raise ConfigError(f"{where} has an invalid written flag.")
    if row.reason is not None and row.reason not in ARTEFACT_REASONS:
        raise ConfigError(f"{where} has an unknown reason.")
    if metadata:
        if row.bytes is not None or row.sha256 is not None:
            raise ConfigError(f"{where} metadata cannot claim bytes or a digest.")
        if row.written and row.reason != "metadata_envelope":
            raise ConfigError(f"{where} written metadata needs metadata_envelope.")
        if not row.written and row.reason in (None, "metadata_envelope"):
            raise ConfigError(f"{where} omitted metadata needs an omission reason.")
    elif row.written:
        if (
            type(row.bytes) is not int
            or row.bytes < 0
            or type(row.sha256) is not str
            or _SHA256.fullmatch(row.sha256) is None
            or row.reason is not None
        ):
            raise ConfigError(f"{where} written content row is invalid.")
    elif row.bytes is not None or row.sha256 is not None or row.reason is None:
        raise ConfigError(f"{where} omitted content row is invalid.")
    return {
        "relative_path": row.relative_path,
        "written": row.written,
        "bytes": row.bytes,
        "sha256": row.sha256,
        "reason": row.reason,
    }


def artefact_table(snapshot: AuditSnapshot) -> Mapping[str, JsonValue]:
    table = snapshot.artefacts
    if type(table) is not ArtefactTable:
        raise ConfigError("audit snapshot artefacts are not an exact ArtefactTable.")
    expected_variants = sum(1 for row in snapshot.path_encodings if row.kind == "variant")
    if len(table.resolved_variants) != expected_variants:
        raise ConfigError("artefact resolved variant row count is inconsistent.")
    rows = {}
    paths = []
    for name in ARTEFACT_TABLE_KEYS:
        value = getattr(table, name)
        if name == "resolved_variants":
            rows[name] = tuple(
                _artefact_record(row, where=f"artefacts.{name}[{index}]", metadata=False)
                for index, row in enumerate(value)
            )
            paths.extend(row.relative_path for row in value)
        else:
            projected = _artefact_record(
                value,
                where=f"artefacts.{name}",
                metadata=name in _METADATA,
            )
            rows[name] = projected
            paths.append(value.relative_path)
    if len(paths) != len(set(paths)):
        raise ConfigError("artefact paths must be unique.")
    return rows


def _runtime(value: object) -> Mapping[str, JsonValue] | None:
    if value is None:
        return None
    row = _exact_mapping(value, RUNTIME_ROW_KEYS, "runtime")
    _exact_mapping(row["requested"], RUNTIME_REQUESTED_ROW_KEYS, "runtime.requested")
    _exact_mapping(row["actual"], RUNTIME_ACTUAL_ROW_KEYS, "runtime.actual")
    _exact_mapping(
        row["prior_environment"],
        RUNTIME_PRIOR_ENVIRONMENT_ROW_KEYS,
        "runtime.prior_environment",
    )
    return row


def _plugins(rows) -> Sequence[Mapping[str, JsonValue]]:
    projected = []
    for index, row in enumerate(rows):
        exact = _exact_mapping(row, PLUGIN_ROW_KEYS, f"plugins[{index}]")
        distributions = exact["distributions"]
        if type(distributions) is not tuple:
            raise ConfigError("plugin distributions must be a sequence.")
        for candidate_index, candidate in enumerate(distributions):
            _exact_mapping(
                candidate,
                PLUGIN_DISTRIBUTION_ROW_KEYS,
                f"plugins[{index}].distributions[{candidate_index}]",
            )
        projected.append(exact)
    return tuple(projected)


def build_provenance(
    snapshot: AuditSnapshot,
    *,
    status: Status,
) -> Mapping[str, JsonValue]:
    """Project the exact closed provenance tree."""
    if type(snapshot) is not AuditSnapshot or status not in STATUSES:
        raise ConfigError("provenance requires an exact snapshot and status.")
    bootstrap = snapshot.bootstrap
    if bootstrap is not None:
        bootstrap = _exact_mapping(bootstrap, _BOOTSTRAP_KEYS, "bootstrap")
    software = None if snapshot.software is None else validate_software(snapshot.software)
    inputs = []
    for index, row in enumerate(snapshot.inputs):
        if type(row) is not InputRecord:
            raise ConfigError(f"inputs[{index}] is not an exact InputRecord.")
        captured = _exact_mapping(
            row.captured,
            LAYER_ROW_KEYS["input"],
            f"inputs[{index}]",
        )
        inputs.append({"layer": _layer(row.layer), **dict(captured)})
    for index, row in enumerate(snapshot.python_targets):
        if type(row) is not PythonTargetRecord:
            raise ConfigError(f"python_targets[{index}] is not exact.")
    targets = tuple(
        {
            "layer": _layer(row.layer),
            "document_path": row.document_path,
            "target": row.target,
            "code_hash": row.code_hash,
            "unobserved_io": row.unobserved_io,
        }
        for row in snapshot.python_targets
    )
    for index, row in enumerate(snapshot.seeds):
        if type(row) is not SeedRecord:
            raise ConfigError(f"seeds[{index}] is not exact.")
    seeds = tuple(
        {"layer": _layer(row.layer), "root": row.root, "named": row.named}
        for row in snapshot.seeds
    )
    for index, row in enumerate(snapshot.variants):
        if type(row) is not VariantRecord:
            raise ConfigError(f"variants[{index}] is not exact.")
    variants = tuple(
        {
            "layer": _layer(row.layer),
            "encoded_name": row.encoded_name,
            "status": row.status,
            "resolved_sha256": row.resolved_sha256,
        }
        for row in snapshot.variants
    )
    for index, row in enumerate(snapshot.resources):
        if type(row) is not ResourceRecord:
            raise ConfigError(f"resources[{index}] is not exact.")
    resources = tuple(
        {
            "layer": _layer(row.layer),
            "build_order": row.build_order,
            "shared_objects": row.shared_objects,
        }
        for row in snapshot.resources
    )
    for index, row in enumerate(snapshot.run_outcomes):
        if type(row) is not RunOutcomeRecord:
            raise ConfigError(f"runs[{index}] is not exact.")
    runs = tuple(
        {
            "layer": _layer(row.layer),
            **_descriptor(row.descriptor),
            "status": row.status,
            "wall_time_ns": row.wall_time_ns,
        }
        for row in snapshot.run_outcomes
    )
    for index, row in enumerate(snapshot.path_encodings):
        if type(row) is not PathEncoding:
            raise ConfigError(f"path_encodings[{index}] is not exact.")
    encodings = tuple(
        {
            "kind": row.kind,
            "document_name": row.document_name,
            "encoded_name": row.encoded_name,
        }
        for row in snapshot.path_encodings
    )
    return {
        "format_version": 1,
        "status": status,
        "completed_boundaries": completed_boundaries(snapshot),
        "bootstrap": bootstrap,
        "software": software,
        "runtime": _runtime(snapshot.runtime),
        "inputs": tuple(inputs),
        "plugins": _plugins(snapshot.plugins),
        "python_targets": targets,
        "seeds": seeds,
        "variants": variants,
        "resources": resources,
        "runs": runs,
        "path_encodings": encodings,
        "artefacts": artefact_table(snapshot),
    }


__all__ = [
    "ARTEFACT_REASONS",
    "ARTEFACT_RECORD_KEYS",
    "ARTEFACT_TABLE_KEYS",
    "PROVENANCE_KEYS",
    "STATUSES",
    "artefact_table",
    "build_provenance",
    "completed_boundaries",
]
