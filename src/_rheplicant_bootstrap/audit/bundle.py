"""Pure validation and serialization of the mandatory audit bundle."""

from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from _rheplicant_bootstrap.audit.resolved import ResolvedArtefact
from _rheplicant_bootstrap.audit.types import AuditSnapshot, ResolvedLayerRecord
from _rheplicant_bootstrap.errors import ConfigError
from _rheplicant_bootstrap.types import CompletedBoundary, Status

from .diagnostics import build_diagnostics
from .json import canonical_json_bytes
from .provenance import artefact_table, build_provenance


@dataclass(frozen=True, slots=True)
class AuditBundle:
    input: bytes
    resolved: Sequence[ResolvedArtefact]
    provenance: bytes
    diagnostics: bytes
    files: Mapping[str, bytes]


def candidate_serialization_snapshot(snapshot: AuditSnapshot) -> AuditSnapshot:
    """Purely append a candidate document-level serialization boundary."""
    if type(snapshot) is not AuditSnapshot:
        raise ConfigError("serialization candidate requires an exact AuditSnapshot.")
    if any(row.stage == "serialization" for row in snapshot.completed_boundaries):
        raise ConfigError("serialization boundary is already present.")
    return dataclasses.replace(
        snapshot,
        completed_boundaries=(
            *snapshot.completed_boundaries,
            CompletedBoundary("serialization", None),
        ),
    )


def terminal_reserialization_snapshot(snapshot: AuditSnapshot) -> AuditSnapshot:
    """Detach a snapshot whose one serialization boundary is already final."""
    if type(snapshot) is not AuditSnapshot:
        raise ConfigError("terminal serialization requires an exact AuditSnapshot.")
    rows = tuple(
        index
        for index, row in enumerate(snapshot.completed_boundaries)
        if row.stage == "serialization"
    )
    if rows != (len(snapshot.completed_boundaries) - 1,):
        raise ConfigError("terminal serialization requires one final boundary.")
    if snapshot.completed_boundaries[-1].layer is not None:
        raise ConfigError("serialization boundary must be document-scoped.")
    return dataclasses.replace(snapshot, completed_boundaries=tuple(snapshot.completed_boundaries))


def _validate_candidate(snapshot: AuditSnapshot) -> None:
    rows = tuple(row for row in snapshot.completed_boundaries if row.stage == "serialization")
    if len(rows) != 1 or snapshot.completed_boundaries[-1] != CompletedBoundary(
        "serialization", None
    ):
        raise ConfigError("bundle requires one final document-level serialization boundary.")


def _validate_resolved(
    snapshot: AuditSnapshot,
    rows: Sequence[ResolvedArtefact],
) -> tuple[ResolvedArtefact, ...]:
    if type(rows) not in (tuple, list):
        raise ConfigError("resolved artefacts must be a sequence.")
    copied = tuple(rows)
    if any(type(row) is not ResolvedArtefact for row in copied):
        raise ConfigError("resolved artefacts must be exact typed rows.")
    snapshot_rows = tuple(snapshot.resolved_layers)
    if any(type(row) is not ResolvedLayerRecord for row in snapshot_rows):
        raise ConfigError("snapshot resolved layers must be exact typed rows.")
    paths = [row.relative_path for row in copied]
    if len(paths) != len(set(paths)):
        raise ConfigError("resolved artefact paths are duplicated.")
    if tuple(row.layer for row in copied) != tuple(row.layer for row in snapshot_rows):
        raise ConfigError("resolved artefacts do not match the completed snapshot layers.")
    for row in copied:
        if type(row.payload) is not bytes:
            raise ConfigError("resolved artefact payload must be exact bytes.")
        if hashlib.sha256(row.payload).hexdigest() != row.sha256:
            raise ConfigError("resolved artefact digest does not match its payload.")
        if row.layer.kind == "base" and row.relative_path != "config.resolved.yaml":
            raise ConfigError("base resolved artefact has the wrong path.")
        if row.layer.kind == "variant" and not (
            row.relative_path.startswith("variants/")
            and row.relative_path.endswith("/config.resolved.yaml")
        ):
            raise ConfigError("variant resolved artefact has the wrong path.")
    return copied


def _validate_materialized_content(
    snapshot: AuditSnapshot,
    *,
    input_bytes: bytes,
    resolved: Sequence[ResolvedArtefact],
) -> None:
    # First validate the whole fixed table, including metadata/content invariants.
    artefact_table(snapshot)
    table = snapshot.artefacts
    if table.input.relative_path != "config.input.yaml":
        raise ConfigError("input artefact has the wrong path.")
    if table.provenance.relative_path != "provenance.json":
        raise ConfigError("provenance artefact has the wrong path.")
    if table.diagnostics.relative_path != "diagnostics.json":
        raise ConfigError("diagnostics artefact has the wrong path.")

    def validate_written(row, payload: bytes | None, *, where: str) -> None:
        if not row.written:
            return
        if payload is None:
            raise ConfigError(f"{where} claims materialization without bundle bytes.")
        if row.bytes != len(payload) or row.sha256 != hashlib.sha256(payload).hexdigest():
            raise ConfigError(f"{where} materialization contradicts bundle bytes.")

    validate_written(table.input, input_bytes, where="input artefact")
    payloads = {row.relative_path: row.payload for row in resolved}
    validate_written(
        table.resolved_base,
        payloads.get(table.resolved_base.relative_path),
        where="resolved base artefact",
    )
    for index, row in enumerate(table.resolved_variants):
        validate_written(
            row,
            payloads.get(row.relative_path),
            where=f"resolved variant artefact[{index}]",
        )


def serialize_bundle(
    candidate: AuditSnapshot,
    *,
    status: Status,
    input_bytes: bytes,
    resolved: Sequence[ResolvedArtefact],
) -> AuditBundle:
    """Validate and emit bytes without mutating trace or external state."""
    if type(candidate) is not AuditSnapshot or type(input_bytes) is not bytes:
        raise ConfigError("bundle inputs must be exact immutable audit values.")
    _validate_candidate(candidate)
    if candidate.bootstrap is None:
        raise ConfigError("bundle requires earned bootstrap facts.")
    if hashlib.sha256(input_bytes).hexdigest() != candidate.bootstrap["input_sha256"]:
        raise ConfigError("input bytes do not match the bootstrap digest.")
    resolved_rows = _validate_resolved(candidate, resolved)
    _validate_materialized_content(
        candidate,
        input_bytes=input_bytes,
        resolved=resolved_rows,
    )
    provenance = canonical_json_bytes(build_provenance(candidate, status=status))
    diagnostics = canonical_json_bytes(build_diagnostics(candidate, status=status))
    files = {"config.input.yaml": input_bytes}
    for row in resolved_rows:
        if row.relative_path in files:
            raise ConfigError("bundle file paths are duplicated.")
        files[row.relative_path] = row.payload
    files["provenance.json"] = provenance
    files["diagnostics.json"] = diagnostics
    return AuditBundle(
        input=input_bytes,
        resolved=resolved_rows,
        provenance=provenance,
        diagnostics=diagnostics,
        files=MappingProxyType(files),
    )


__all__ = [
    "AuditBundle",
    "candidate_serialization_snapshot",
    "serialize_bundle",
    "terminal_reserialization_snapshot",
]
