"""Frozen, JAX-free records stored by the configuration audit trace."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from _rheplicant_bootstrap.layering import OriginNode
from _rheplicant_bootstrap.types import (
    CompletedBoundary,
    DestinationDescriptor,
    JsonValue,
    LayerIdentity,
    Origin,
    RunDescriptor,
    Stage,
)


@dataclass(frozen=True, slots=True)
class DefaultRecord:
    layer: LayerIdentity
    path: str
    value: JsonValue


@dataclass(frozen=True, slots=True)
class DeliveryRecord:
    layer: LayerIdentity
    destination: DestinationDescriptor
    dtype: str
    origin: Origin
    unit: str | None


@dataclass(frozen=True, slots=True)
class InputRecord:
    layer: LayerIdentity
    captured: Mapping[str, JsonValue]


@dataclass(frozen=True, slots=True)
class FindingRecord:
    layer: LayerIdentity
    stage: Stage
    check: str
    severity: Literal["refuse", "warn", "report"]
    where: str
    message: str


@dataclass(frozen=True, slots=True)
class PythonTargetRecord:
    layer: LayerIdentity
    document_path: str
    target: str
    code_hash: None
    unobserved_io: Literal[True]


@dataclass(frozen=True, slots=True)
class SeedRecord:
    layer: LayerIdentity
    root: int | None
    named: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class VariantRecord:
    layer: LayerIdentity
    encoded_name: str | None
    status: Literal["ok", "refused", "error", "not_reached"]
    resolved_sha256: str | None


@dataclass(frozen=True, slots=True)
class ResourceRecord:
    layer: LayerIdentity
    build_order: Sequence[str]
    shared_objects: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class GateRecord:
    layer: LayerIdentity
    name: str
    schema_id: str
    declared_mode: str
    effective_state: str
    reason: str | None


@dataclass(frozen=True, slots=True)
class DeferredValidationRecord:
    layer: LayerIdentity
    descriptor: RunDescriptor
    checks: Sequence[str]


@dataclass(frozen=True, slots=True)
class ParsedRunRecord:
    layer: LayerIdentity
    descriptor: RunDescriptor
    resolved_options: Mapping[str, JsonValue]
    deferred_checks: Sequence[str]


@dataclass(frozen=True, slots=True)
class RunOutcomeRecord:
    layer: LayerIdentity
    descriptor: RunDescriptor
    status: Literal["ok", "expected_refusal", "refused", "error"]
    wall_time_ns: int
    exception_type: str | None
    exception_message: str | None
    capture_scope: Literal["arbitrary_exception"] | None
    is_dirt_error: bool | None


@dataclass(frozen=True, slots=True)
class DeletionAuditRecord:
    layer: LayerIdentity
    path: Sequence[str | int]
    origin: Origin


@dataclass(frozen=True, slots=True)
class ResolvedLayerRecord:
    layer: LayerIdentity
    effective_document: Mapping[str, JsonValue]
    origins: OriginNode
    declared_runs: Sequence[RunDescriptor]
    execution_runs: Sequence[RunDescriptor]
    audit: Mapping[str, JsonValue]


@dataclass(frozen=True, slots=True)
class PathEncoding:
    kind: Literal["run", "variant"]
    document_name: str
    encoded_name: str


ArtefactReason = Literal[
    "metadata_envelope",
    "transaction_not_reached",
    "boundary_not_reached",
    "layer_not_complete",
    "not_applicable_to_status",
]


@dataclass(frozen=True, slots=True)
class ArtefactRecord:
    relative_path: str
    written: bool
    bytes: int | None
    sha256: str | None
    reason: ArtefactReason | None


@dataclass(frozen=True, slots=True)
class ArtefactTable:
    marker: ArtefactRecord
    lock: ArtefactRecord
    journal: ArtefactRecord
    input: ArtefactRecord
    resolved_base: ArtefactRecord
    resolved_variants: Sequence[ArtefactRecord]
    provenance: ArtefactRecord
    diagnostics: ArtefactRecord


@dataclass(frozen=True, slots=True)
class ArtefactMaterialization:
    slot: Literal[
        "marker",
        "lock",
        "journal",
        "input",
        "resolved_base",
        "resolved_variant",
        "provenance",
        "diagnostics",
    ]
    variant_index: int | None
    relative_path: str
    bytes: int | None
    sha256: str | None


@dataclass(frozen=True, slots=True)
class ErrorRecord:
    exception_type: str
    message: str


@dataclass(frozen=True, slots=True)
class AuditSnapshot:
    bootstrap: Mapping[str, JsonValue] | None
    completed_boundaries: Sequence[CompletedBoundary]
    defaults: Sequence[DefaultRecord]
    deliveries: Sequence[DeliveryRecord]
    inputs: Sequence[InputRecord]
    findings: Sequence[FindingRecord]
    plugins: Sequence[Mapping[str, JsonValue]]
    python_targets: Sequence[PythonTargetRecord]
    runtime: Mapping[str, JsonValue] | None
    seeds: Sequence[SeedRecord]
    variants: Sequence[VariantRecord]
    resources: Sequence[ResourceRecord]
    gates: Sequence[GateRecord]
    deferred_validations: Sequence[DeferredValidationRecord]
    parsed_runs: Sequence[ParsedRunRecord]
    run_outcomes: Sequence[RunOutcomeRecord]
    deletions: Sequence[DeletionAuditRecord]
    resolved_layers: Sequence[ResolvedLayerRecord]
    path_encodings: Sequence[PathEncoding]
    artefacts: ArtefactTable
    error: ErrorRecord | None


def empty_artefact_table() -> ArtefactTable:
    def row(path: str, reason: ArtefactReason) -> ArtefactRecord:
        return ArtefactRecord(path, False, None, None, reason)

    return ArtefactTable(
        marker=row(".rheplicant-results.json", "boundary_not_reached"),
        lock=row(".rheplicant-lock", "transaction_not_reached"),
        journal=row(".rheplicant-journal", "transaction_not_reached"),
        input=row("config.input.yaml", "boundary_not_reached"),
        resolved_base=row("config.resolved.yaml", "layer_not_complete"),
        resolved_variants=(),
        provenance=row("provenance.json", "boundary_not_reached"),
        diagnostics=row("diagnostics.json", "boundary_not_reached"),
    )
