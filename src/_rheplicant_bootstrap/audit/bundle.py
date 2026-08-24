"""Pure validation and serialization of the mandatory audit bundle."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from _rheplicant_bootstrap.audit.integrity import INTEGRITY_NAME, integrity_bytes
from _rheplicant_bootstrap.audit.resolved import ResolvedArtefact
from _rheplicant_bootstrap.audit.types import AuditSnapshot, ResolvedLayerRecord
from _rheplicant_bootstrap.errors import ConfigError
from _rheplicant_bootstrap.types import CompletedBoundary, Status

from .diagnostics import DIAGNOSTICS_KEYS, build_diagnostics
from .json import canonical_json_bytes
from .provenance import PROVENANCE_KEYS, artefact_table, build_provenance


@dataclass(frozen=True, slots=True)
class AuditBundle:
    input: bytes
    resolved: Sequence[ResolvedArtefact]
    provenance: bytes
    diagnostics: bytes
    files: Mapping[str, bytes]


def _validate_file_pair(relative_path: object, payload: object) -> None:
    if type(relative_path) is not str or type(payload) is not bytes:
        raise ConfigError("bundle files must be exact text/bytes pairs.")
    try:
        relative_path.encode("utf-8")
    except UnicodeEncodeError:
        raise ConfigError("bundle file path must be valid UTF-8 text.") from None
    if relative_path.startswith("/") or "\0" in relative_path or "\\" in relative_path:
        raise ConfigError("bundle file path is not a portable relative path.")
    parts = relative_path.split("/")
    if not parts or any(part in ("", ".", "..") for part in parts):
        raise ConfigError("bundle file path is not a portable relative path.")


#: Paths a detached bundle file may never claim. All three are written by the
#: transaction after everything else is final, and all three are replaced
#: together in ``replace_staged_metadata`` -- so a merged file sitting on one
#: of these names would be overwritten without warning.
#:
#: ``integrity.json`` joined this tuple after review rather than with the
#: feature: it was added to the tail of the bundle and to the replaceable set
#: in the transaction, and NOT to the one list that stops something else
#: claiming the name. A scientific product called ``integrity.json`` would have
#: merged cleanly here and then failed inside :func:`with_integrity` with
#: "already present in the bundle", which names neither the product nor the
#: collision.
RESERVED_BUNDLE_PATHS = ("provenance.json", "diagnostics.json", INTEGRITY_NAME)


def with_integrity(bundle: AuditBundle) -> AuditBundle:
    """Append ``integrity.json``, covering every other file in the bundle.

    Placed immediately before the two metadata files, so the tail of a staged
    tree reads ``integrity.json, provenance.json, diagnostics.json`` -- the
    three that can only be written once everything else is final.

    The payload written here describes the bundle as it stands NOW, which for a
    staged bundle means the provenance and diagnostics of the staging pass.
    ``replace_staged_metadata`` recomputes and replaces it once those two are
    final; this call exists so the file has a slot to be replaced in, since a
    transaction refuses to add a path it did not stage.

    Calling it twice is refused rather than tolerated: a second manifest would
    silently cover the first, and "the digest list covers everything except
    itself" would stop being true.
    """
    validate_serialized_bundle(bundle)
    rows = tuple(bundle.files.items())
    if any(name == INTEGRITY_NAME for name, _payload in rows):
        raise ConfigError(f"{INTEGRITY_NAME} is already present in the bundle.")
    covered = rows[:-2]
    files = dict(covered)
    files[INTEGRITY_NAME] = integrity_bytes(rows)
    files["provenance.json"] = bundle.provenance
    files["diagnostics.json"] = bundle.diagnostics
    merged = dataclasses.replace(bundle, files=MappingProxyType(files))
    validate_serialized_bundle(merged)
    return merged


def merge_bundle_files(
    bundle: AuditBundle,
    additional: Mapping[str, bytes],
) -> AuditBundle:
    """Insert detached files before the two fixed audit metadata files."""
    validate_serialized_bundle(bundle)
    if not isinstance(additional, Mapping):
        raise ConfigError("additional bundle files must be a mapping.")
    rows = tuple(bundle.files.items())
    files = dict(rows[:-2])
    try:
        additional_rows = tuple(additional.items())
    except Exception:
        raise ConfigError("additional bundle file traversal failed.") from None
    for relative_path, payload in additional_rows:
        _validate_file_pair(relative_path, payload)
        if relative_path in files or relative_path in RESERVED_BUNDLE_PATHS:
            raise ConfigError(f"bundle file path {relative_path!r} is reserved or duplicated.")
        files[relative_path] = payload
    files["provenance.json"] = bundle.provenance
    files["diagnostics.json"] = bundle.diagnostics
    merged = dataclasses.replace(bundle, files=MappingProxyType(files))
    validate_serialized_bundle(merged)
    return merged


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


def validate_serialized_bundle(bundle: AuditBundle) -> None:
    """Validate detached bundle bytes before a transaction writes them."""
    if type(bundle) is not AuditBundle or not isinstance(bundle.files, Mapping):
        raise ConfigError("serialized audit bundle is not exact.")
    try:
        provenance = json.loads(bundle.provenance)
        diagnostics = json.loads(bundle.diagnostics)
    except (UnicodeError, json.JSONDecodeError):
        raise ConfigError("serialized audit metadata is not JSON.") from None
    if (
        type(provenance) is not dict
        or tuple(provenance) != tuple(sorted(PROVENANCE_KEYS))
        or type(diagnostics) is not dict
        or tuple(diagnostics) != tuple(sorted(DIAGNOSTICS_KEYS))
    ):
        raise ConfigError("serialized audit metadata has the wrong top-level fields.")
    if canonical_json_bytes(provenance) != bundle.provenance:
        raise ConfigError("serialized provenance is not canonical JSON.")
    if canonical_json_bytes(diagnostics) != bundle.diagnostics:
        raise ConfigError("serialized diagnostics is not canonical JSON.")
    if provenance["status"] != diagnostics["status"]:
        raise ConfigError("serialized audit metadata statuses disagree.")
    if provenance["artefacts"] != diagnostics["artefacts"]:
        raise ConfigError("serialized audit artefact tables disagree.")
    rows = tuple(bundle.files.items())
    for relative_path, payload in rows:
        _validate_file_pair(relative_path, payload)
    if not rows or rows[0] != ("config.input.yaml", bundle.input):
        raise ConfigError("serialized audit bundle has inconsistent input bytes.")
    if rows[-2:] != (
        ("provenance.json", bundle.provenance),
        ("diagnostics.json", bundle.diagnostics),
    ):
        raise ConfigError("serialized audit bundle has inconsistent metadata bytes.")


__all__ = [
    "AuditBundle",
    "candidate_serialization_snapshot",
    "merge_bundle_files",
    "serialize_bundle",
    "terminal_reserialization_snapshot",
    "validate_serialized_bundle",
]
