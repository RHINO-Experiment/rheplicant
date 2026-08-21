"""Strict projection of an audit snapshot into diagnostics JSON."""

from __future__ import annotations

from collections.abc import Mapping

from _rheplicant_bootstrap.audit.types import (
    AuditSnapshot,
    DeferredValidationRecord,
    FindingRecord,
    GateRecord,
    RunOutcomeRecord,
)
from _rheplicant_bootstrap.errors import ConfigError
from _rheplicant_bootstrap.types import JsonValue, LayerIdentity, RunDescriptor, Status

from .provenance import STATUSES, artefact_table, completed_boundaries

RUN_STATUSES = ("ok", "expected_refusal", "refused", "error")
CAPTURE_SCOPES = ("arbitrary_exception",)
DIAGNOSTICS_KEYS = (
    "format_version",
    "status",
    "completed_boundaries",
    "findings",
    "error",
    "gates",
    "deferred_validations",
    "runs",
    "artefacts",
)


def _layer(layer) -> Mapping[str, JsonValue]:
    if type(layer) is not LayerIdentity:
        raise ConfigError("diagnostics layer is not exact.")
    return {"kind": layer.kind, "name": layer.name}


def _descriptor(row) -> Mapping[str, JsonValue]:
    if type(row) is not RunDescriptor:
        raise ConfigError("diagnostics descriptor is not exact.")
    return {
        "index": row.index,
        "name": row.name,
        "kind": row.kind,
        "variant": row.variant,
    }


def build_diagnostics(
    snapshot: AuditSnapshot,
    *,
    status: Status,
) -> Mapping[str, JsonValue]:
    """Project the exact closed diagnostics tree."""
    if type(snapshot) is not AuditSnapshot or status not in STATUSES:
        raise ConfigError("diagnostics requires an exact snapshot and status.")
    if status == "ok" and snapshot.error is not None:
        raise ConfigError("successful diagnostics cannot carry an error.")
    if status != "ok" and snapshot.error is None:
        raise ConfigError("failed diagnostics require an earned error record.")
    findings = []
    for index, row in enumerate(snapshot.findings):
        if type(row) is not FindingRecord:
            raise ConfigError(f"findings[{index}] is not exact.")
        findings.append(
            {
                "check": row.check,
                "severity": row.severity,
                "where": row.where,
                "message": row.message,
            }
        )
    error = (
        None
        if snapshot.error is None
        else {
            "exception_type": snapshot.error.exception_type,
            "message": snapshot.error.message,
        }
    )
    gates = []
    for index, row in enumerate(snapshot.gates):
        if type(row) is not GateRecord:
            raise ConfigError(f"gates[{index}] is not exact.")
        gates.append(
            {
                "layer": _layer(row.layer),
                "name": row.name,
                "schema_id": row.schema_id,
                "declared_mode": row.declared_mode,
                "effective_state": row.effective_state,
                "reason": row.reason,
            }
        )
    deferred = []
    for index, row in enumerate(snapshot.deferred_validations):
        if type(row) is not DeferredValidationRecord:
            raise ConfigError(f"deferred_validations[{index}] is not exact.")
        deferred.append(
            {
                "layer": _layer(row.layer),
                "descriptor": _descriptor(row.descriptor),
                "checks": row.checks,
            }
        )
    runs = []
    for index, row in enumerate(snapshot.run_outcomes):
        if type(row) is not RunOutcomeRecord:
            raise ConfigError(f"runs[{index}] is not exact.")
        if row.status not in RUN_STATUSES:
            raise ConfigError("diagnostics contains an unknown run status.")
        if row.status == "expected_refusal":
            if row.capture_scope not in CAPTURE_SCOPES or type(row.is_dirt_error) is not bool:
                raise ConfigError("expected-refusal diagnostics are incomplete.")
        elif row.capture_scope is not None:
            raise ConfigError("only expected refusals carry a capture scope.")
        runs.append(
            {
                "layer": _layer(row.layer),
                **_descriptor(row.descriptor),
                "status": row.status,
                "wall_time_ns": row.wall_time_ns,
                "exception_type": row.exception_type,
                "exception_message": row.exception_message,
                "capture_scope": row.capture_scope,
                "is_dirt_error": row.is_dirt_error,
                "phases": (),
            }
        )
    return {
        "format_version": 1,
        "status": status,
        "completed_boundaries": completed_boundaries(snapshot),
        "findings": tuple(findings),
        "error": error,
        "gates": tuple(gates),
        "deferred_validations": tuple(deferred),
        "runs": tuple(runs),
        "artefacts": artefact_table(snapshot),
    }


__all__ = [
    "CAPTURE_SCOPES",
    "DIAGNOSTICS_KEYS",
    "RUN_STATUSES",
    "build_diagnostics",
]
