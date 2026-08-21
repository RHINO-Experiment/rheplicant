"""Layer-bound projection of resolved defaults and derived audit facts."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, TypeVar, cast

from _rheplicant_bootstrap.audit.types import AuditSnapshot
from _rheplicant_bootstrap.frozen import freeze_evidence, static_isinstance
from _rheplicant_bootstrap.layering import OriginNode
from _rheplicant_bootstrap.types import (
    JsonValue,
    LayerIdentity,
    TraceSink,
)
from rheplicant.config.errors import ConfigError

T = TypeVar("T")


def to_json_value(value: object) -> JsonValue:
    """Detach an already-resolved plain value, refusing live runtime objects."""
    frozen = freeze_evidence(value, where="resolved audit value")
    pending = [frozen]
    while pending:
        item = pending.pop()
        if item is None or type(item) in (bool, int, str):
            continue
        if type(item) is float:
            if not math.isfinite(item):
                raise ConfigError("resolved audit values must be finite JSON.")
            continue
        if static_isinstance(item, Mapping):
            for key, child in item.items():
                if type(key) is not str:
                    raise ConfigError("resolved audit mapping keys must be strings.")
                pending.append(child)
            continue
        if type(item) is tuple:
            pending.extend(item)
            continue
        raise ConfigError("resolved audit values must be plain JSON.")
    return cast(JsonValue, frozen)


class ResolutionAuditSink(TraceSink, Protocol):
    def snapshot(self) -> AuditSnapshot: ...

    def record_resource(self, layer: LayerIdentity, row: Mapping[str, JsonValue]) -> None: ...

    def record_seed(self, layer: LayerIdentity, row: Mapping[str, JsonValue]) -> None: ...

    def record_gate(self, layer: LayerIdentity, row: Mapping[str, JsonValue]) -> None: ...

    def record_python_target(self, layer: LayerIdentity, row: Mapping[str, JsonValue]) -> None: ...

    def record_deletion(self, layer: LayerIdentity, row: Mapping[str, JsonValue]) -> None: ...


@dataclass(frozen=True, slots=True)
class ResolutionAudit:
    layer: LayerIdentity
    trace: ResolutionAuditSink

    def use_default(self, path: str, value: T) -> T:
        self.trace.record_default(self.layer, path, to_json_value(value))
        return value

    def resource(self, build_order: Sequence[str], shared_objects: Mapping[str, str]) -> None:
        self.trace.record_resource(
            self.layer,
            {
                "build_order": tuple(build_order),
                "shared_objects": dict(shared_objects),
            },
        )

    def seed(self, root: int | None, named: Mapping[str, int]) -> None:
        self.trace.record_seed(
            self.layer,
            {"root": root, "named": dict(named)},
        )

    def gate(
        self,
        name: str,
        schema_id: str,
        declared_mode: str,
        effective_state: str,
        reason: str | None,
    ) -> None:
        self.trace.record_gate(
            self.layer,
            {
                "name": name,
                "schema_id": schema_id,
                "declared_mode": declared_mode,
                "effective_state": effective_state,
                "reason": reason,
            },
        )

    def python_target(self, document_path: str, target: str) -> None:
        self.trace.record_python_target(
            self.layer,
            {
                "document_path": document_path,
                "target": target,
                "code_hash": None,
                "unobserved_io": True,
            },
        )


def parallel_origins(document: object, origins: OriginNode) -> OriginNode:
    """Copy only origin nodes that have a matching effective-document value."""
    if type(origins) is not OriginNode:
        raise ConfigError("resolved origins must be an exact OriginNode.")
    if static_isinstance(document, Mapping):
        children = {}
        for key, child in document.items():
            if key not in origins.children:
                raise ConfigError(f"resolved origins are missing {key!r}.")
            children[key] = parallel_origins(child, origins.children[key])
        return OriginNode(origins.origin, children)
    if type(document) in (tuple, list):
        children = {}
        for index, child in enumerate(document):
            if index not in origins.children:
                raise ConfigError(f"resolved origins are missing index {index}.")
            children[index] = parallel_origins(child, origins.children[index])
        return OriginNode(origins.origin, children)
    if origins.children:
        raise ConfigError("resolved scalar has child origins.")
    return OriginNode(origins.origin, {})


def _runtime_json(value: object) -> JsonValue:
    """Project an already-earned runtime scalar/array into detached JSON."""
    if static_isinstance(value, Mapping):
        return to_json_value(
            {str(key): _runtime_json(child) for key, child in value.items()}
        )
    if type(value) in (list, tuple):
        return to_json_value(tuple(_runtime_json(child) for child in value))
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return _runtime_json(tolist())
        except (TypeError, ValueError, OverflowError):
            raise ConfigError("resolved runtime value cannot be converted to JSON.") from None
    return to_json_value(value)


def layer_audit_row(
    trace: ResolutionAuditSink,
    layer: LayerIdentity,
    *,
    configured: object,
) -> Mapping[str, JsonValue]:
    """Project only append-only facts already earned by one exact layer."""
    snapshot = trace.snapshot()

    def descriptor(value):
        return {
            "index": value.index,
            "name": value.name,
            "kind": value.kind,
            "variant": value.variant,
        }

    bootstrap = snapshot.bootstrap
    presets = ()
    if bootstrap is not None and "presets" in bootstrap:
        presets = bootstrap["presets"]
    context = configured.context
    inference = configured.inference
    switch_order = tuple(context.switch_order)
    truth = _runtime_json(inference.truth)
    truth_omitted = inference.truth_omitted
    truth_omissions = tuple(
        {"name": name, "reason": reason}
        for name, reason in truth_omitted.items()
    )
    x64_required_by = tuple(
        row.destination.document_path
        for row in snapshot.deliveries
        if row.layer == layer and row.dtype == "float64"
    )
    projected = {
        "presets": presets,
        "defaults": tuple(
            {"path": row.path, "value": row.value}
            for row in snapshot.defaults
            if row.layer == layer
        ),
        "deliveries": tuple(
            {
                "document_path": row.destination.document_path,
                "domain": row.destination.domain,
                "selector": row.destination.selector,
                "dtype": row.dtype,
                "origin": {"kind": row.origin.kind, "name": row.origin.name},
                "unit": row.unit,
            }
            for row in snapshot.deliveries
            if row.layer == layer
        ),
        "inputs": tuple(row.captured for row in snapshot.inputs if row.layer == layer),
        "findings": tuple(
            {
                "stage": row.stage,
                "check": row.check,
                "severity": row.severity,
                "where": row.where,
                "message": row.message,
            }
            for row in snapshot.findings
            if row.layer == layer
        ),
        "python_targets": tuple(
            {
                "document_path": row.document_path,
                "target": row.target,
                "code_hash": row.code_hash,
                "unobserved_io": row.unobserved_io,
            }
            for row in snapshot.python_targets
            if row.layer == layer
        ),
        "seeds": tuple(
            {"root": row.root, "named": row.named} for row in snapshot.seeds if row.layer == layer
        ),
        "resources": tuple(
            {
                "build_order": row.build_order,
                "shared_objects": row.shared_objects,
            }
            for row in snapshot.resources
            if row.layer == layer
        ),
        "gates": tuple(
            {
                "name": row.name,
                "schema_id": row.schema_id,
                "declared_mode": row.declared_mode,
                "effective_state": row.effective_state,
                "reason": row.reason,
            }
            for row in snapshot.gates
            if row.layer == layer
        ),
        "parsed_runs": tuple(
            {
                "descriptor": descriptor(row.descriptor),
                "resolved_options": row.resolved_options,
                "deferred_checks": row.deferred_checks,
            }
            for row in snapshot.parsed_runs
            if row.layer == layer
        ),
        "deletions": tuple(
            {
                "path": row.path,
                "origin": {"kind": row.origin.kind, "name": row.origin.name},
            }
            for row in snapshot.deletions
            if row.layer == layer
        ),
        "switch_map": {name: index for index, name in enumerate(switch_order)},
        "truth": truth,
        "truth_omissions": truth_omissions,
        "x64_required_by": x64_required_by,
    }
    frozen = to_json_value(projected)
    if not static_isinstance(frozen, Mapping):
        raise ConfigError("resolved layer audit projection must be a mapping.")
    return cast(Mapping[str, JsonValue], frozen)


__all__ = [
    "ResolutionAudit",
    "ResolutionAuditSink",
    "layer_audit_row",
    "parallel_origins",
    "to_json_value",
]
