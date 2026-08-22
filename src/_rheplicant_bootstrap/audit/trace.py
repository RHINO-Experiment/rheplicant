"""Thread-safe append-only storage for configuration audit facts."""

from __future__ import annotations

import math
import re
import threading
from collections.abc import Mapping, Sequence
from typing import cast

from _rheplicant_bootstrap.errors import ConfigError
from _rheplicant_bootstrap.frozen import (
    freeze_evidence,
    static_isinstance,
)
from _rheplicant_bootstrap.layering import OriginNode
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
    DestinationDescriptor,
    JsonValue,
    LayerIdentity,
    Origin,
    RunDescriptor,
    Stage,
)

from .names import encode_name
from .types import (
    ArtefactMaterialization,
    ArtefactRecord,
    ArtefactTable,
    AuditSnapshot,
    DefaultRecord,
    DeferredValidationRecord,
    DeletionAuditRecord,
    DeliveryRecord,
    ErrorRecord,
    FindingRecord,
    GateRecord,
    InputRecord,
    ParsedRunRecord,
    PathEncoding,
    PythonTargetRecord,
    ResolvedLayerRecord,
    ResourceRecord,
    RunOutcomeRecord,
    SeedRecord,
    VariantRecord,
    empty_artefact_table,
)

STAGES = (
    "source",
    "raw_process_entry",
    "preset_layering",
    "effective_process_entry",
    "output_preflight",
    "runtime",
    "plugins",
    "preflight",
    "axes",
    "built",
    "run_parse",
    "postflight",
    "execution",
    "serialization",
)

LAYER_ROW_KEYS = {
    "input": (
        "document_path",
        "path",
        "realpath",
        "format",
        "kind",
        "sha256",
        "members",
    ),
    "python_target": (
        "document_path",
        "target",
        "code_hash",
        "unobserved_io",
    ),
    "seed": ("root", "named"),
    "variant": ("encoded_name", "status", "resolved_sha256"),
    "resource": ("build_order", "shared_objects"),
    "gate": (
        "name",
        "schema_id",
        "declared_mode",
        "effective_state",
        "reason",
    ),
    "deferred_validation": ("descriptor", "checks"),
    "parsed_run": ("descriptor", "resolved_options", "deferred_checks"),
    "run_outcome": (
        "descriptor",
        "status",
        "wall_time_ns",
        "exception_type",
        "exception_message",
        "capture_scope",
        "is_dirt_error",
    ),
    "deletion": ("path", "origin"),
    "resolved_layer": (
        "effective_document",
        "origins",
        "declared_runs",
        "execution_runs",
        "audit",
    ),
}

_BOOTSTRAP_ROW_KEYS = (
    "protocol_version",
    "launch_mode",
    "input_sha256",
    "presets",
    "source_name",
    "source_path",
    "source_realpath",
    "base_dir",
    "invocation_outputs_dir",
)
_PATH_ENCODING_KEYS = ("kind", "document_name", "encoded_name")
_ERROR_KEYS = ("exception_type", "message")
_FINDING_KEYS = ("check", "severity", "where", "message")
_DESCRIPTOR_KEYS = ("index", "name", "kind", "variant")
_ORIGIN_KEYS = ("kind", "name")
_CAPTURE_MEMBER_KEYS = ("relative_path", "path", "realpath", "sha256")
_LOWER_SHA256 = re.compile(r"[0-9a-f]{64}")
_ARTEFACT_REASONS = frozenset(
    (
        "metadata_envelope",
        "transaction_not_reached",
        "boundary_not_reached",
        "layer_not_complete",
        "not_applicable_to_status",
    )
)
_METADATA_SLOTS = frozenset(("marker", "lock", "journal", "provenance", "diagnostics"))
_CONTENT_SLOTS = frozenset(("input", "resolved_base", "resolved_variant"))
_PLUGIN_REASON_SETS = {
    "origin": frozenset(("no_origin", "namespace_package")),
    "loader_type": frozenset(("no_origin", "namespace_package", "generated_module")),
    "resolved_path": frozenset(
        (
            "no_origin",
            "namespace_package",
            "generated_module",
            "not_regular_file",
            "unreadable",
        )
    ),
    "code_hash": frozenset(
        (
            "no_origin",
            "namespace_package",
            "generated_module",
            "not_regular_file",
            "unreadable",
            "extension_module",
        )
    ),
    "version": frozenset(("not_installed", "unreadable")),
    "direct_url": frozenset(("not_installed", "missing_direct_url", "unreadable")),
}
_PEP_503_RUN = re.compile(r"[-_.]+")


def _text(value: object, *, where: str, empty: bool = False) -> str:
    if type(value) is not str or (not empty and not value):
        qualifier = "a string" if empty else "a non-empty string"
        raise ConfigError(f"{where} must be {qualifier}.")
    try:
        value.encode("utf-8", "strict")
    except UnicodeEncodeError:
        raise ConfigError(f"{where} must contain valid UTF-8 text.") from None
    return value


def _optional_text(value: object, *, where: str) -> str | None:
    return None if value is None else _text(value, where=where)


def _sha256(value: object, *, where: str, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if type(value) is not str or _LOWER_SHA256.fullmatch(value) is None:
        raise ConfigError(f"{where} must be a lowercase SHA-256 digest.")
    return value


def _freeze_json(value: object, *, where: str) -> JsonValue:
    frozen = freeze_evidence(value, where=where)
    pending = [frozen]
    while pending:
        item = pending.pop()
        if item is None or type(item) in (bool, int, str):
            continue
        if type(item) is float:
            if not math.isfinite(item):
                raise ConfigError(f"{where} contains a non-finite number.")
            continue
        if static_isinstance(item, Mapping):
            for key, child in item.items():
                if type(key) is not str:
                    raise ConfigError(f"{where} mapping keys must be strings.")
                pending.append(child)
            continue
        if type(item) is tuple:
            pending.extend(item)
            continue
        raise ConfigError(f"{where} contains a value that is not JSON.")
    return cast(JsonValue, frozen)


def _mapping(value: object, *, keys: Sequence[str], where: str) -> Mapping[str, object]:
    if not static_isinstance(value, Mapping):
        raise ConfigError(f"{where} must be a mapping.")
    frozen = freeze_evidence(value, where=where)
    if not static_isinstance(frozen, Mapping):
        raise ConfigError(f"{where} must be a mapping.")
    actual = tuple(frozen)
    if len(actual) != len(keys) or set(actual) != set(keys):
        raise ConfigError(f"{where} must have exactly keys {tuple(keys)!r}.")
    if any(type(key) is not str for key in actual):
        raise ConfigError(f"{where} keys must be strings.")
    return cast(Mapping[str, object], frozen)


def _raw_mapping(value: object, *, keys: Sequence[str], where: str) -> Mapping[str, object]:
    """Validate keys without freezing a permitted typed bootstrap child."""
    if not static_isinstance(value, Mapping):
        raise ConfigError(f"{where} must be a mapping.")
    try:
        actual = tuple(value)
    except Exception:
        raise ConfigError(f"{where} traversal failed.") from None
    if (
        len(actual) != len(keys)
        or any(type(key) is not str for key in actual)
        or set(actual) != set(keys)
    ):
        raise ConfigError(f"{where} must have exactly keys {tuple(keys)!r}.")
    return cast(Mapping[str, object], value)


def _layer(value: object) -> LayerIdentity:
    if type(value) is not LayerIdentity:
        raise ConfigError("audit layer must be an exact LayerIdentity.")
    kind = object.__getattribute__(value, "kind")
    name = object.__getattribute__(value, "name")
    if kind == "base" and name is None:
        return LayerIdentity("base", None)
    if kind == "variant" and type(name) is str and name:
        return LayerIdentity("variant", _text(name, where="variant layer name"))
    raise ConfigError("audit layer identity is invalid.")


def _stage(value: object) -> Stage:
    if type(value) is not str or value not in STAGES:
        raise ConfigError(f"unknown audit stage {value!r}.")
    return cast(Stage, value)


def _descriptor(value: object, *, where: str) -> RunDescriptor:
    row = _mapping(value, keys=_DESCRIPTOR_KEYS, where=where)
    index = row["index"]
    if type(index) is not int or index < 0:
        raise ConfigError(f"{where}.index must be a non-negative integer.")
    return RunDescriptor(
        index=index,
        name=_text(row["name"], where=f"{where}.name"),
        kind=_text(row["kind"], where=f"{where}.kind"),
        variant=_optional_text(row["variant"], where=f"{where}.variant"),
    )


def _origin(value: object, *, where: str) -> Origin:
    row = _mapping(value, keys=_ORIGIN_KEYS, where=where)
    kind = row["kind"]
    name = row["name"]
    try:
        return Origin(cast(str, kind), cast(str | None, name))
    except (TypeError, ValueError):
        raise ConfigError(f"{where} is invalid.") from None


def _typed_origin(value: object, *, where: str) -> Origin:
    if type(value) is not Origin:
        raise ConfigError(f"{where} must be an exact Origin.")
    try:
        return Origin(
            object.__getattribute__(value, "kind"),
            object.__getattribute__(value, "name"),
        )
    except (TypeError, ValueError):
        raise ConfigError(f"{where} is invalid.") from None


def _destination(value: object) -> DestinationDescriptor:
    if type(value) is not DestinationDescriptor:
        raise ConfigError("audit destination must be an exact DestinationDescriptor.")
    try:
        return DestinationDescriptor(
            _text(
                object.__getattribute__(value, "document_path"),
                where="destination.document_path",
            ),
            object.__getattribute__(value, "domain"),
            _text(
                object.__getattribute__(value, "selector"),
                where="destination.selector",
            ),
        )
    except ValueError:
        raise ConfigError("audit destination is invalid.") from None


def _string_sequence(value: object, *, where: str) -> tuple[str, ...]:
    frozen = _freeze_json(value, where=where)
    if type(frozen) is not tuple:
        raise ConfigError(f"{where} must be a sequence.")
    return tuple(_text(item, where=f"{where} item") for item in frozen)


def _descriptor_sequence(value: object, *, where: str) -> tuple[RunDescriptor, ...]:
    if type(value) not in (tuple, list):
        frozen = freeze_evidence(value, where=where)
        if type(frozen) is not tuple:
            raise ConfigError(f"{where} must be a sequence.")
        value = frozen
    return tuple(_descriptor(item, where=f"{where}[{index}]") for index, item in enumerate(value))


def _artefact_record(value: object, *, where: str, metadata: bool) -> ArtefactRecord:
    if type(value) is not ArtefactRecord:
        raise ConfigError(f"{where} must be an exact ArtefactRecord.")
    path = _text(object.__getattribute__(value, "relative_path"), where=f"{where}.relative_path")
    written = object.__getattribute__(value, "written")
    size = object.__getattribute__(value, "bytes")
    digest = object.__getattribute__(value, "sha256")
    reason = object.__getattribute__(value, "reason")
    if type(written) is not bool:
        raise ConfigError(f"{where}.written must be a bool.")
    if reason is not None and (type(reason) is not str or reason not in _ARTEFACT_REASONS):
        raise ConfigError(f"{where}.reason is invalid.")
    if metadata:
        if size is not None or digest is not None:
            raise ConfigError(f"{where} metadata must not carry bytes or a hash.")
        if written and reason != "metadata_envelope":
            raise ConfigError(f"{where} written metadata needs metadata_envelope.")
        if not written and (reason is None or reason == "metadata_envelope"):
            raise ConfigError(f"{where} unwritten metadata needs an omission reason.")
    else:
        if written:
            if type(size) is not int or type(size) is bool or size < 0:
                raise ConfigError(f"{where}.bytes must be non-negative.")
            _sha256(digest, where=f"{where}.sha256")
            if reason is not None:
                raise ConfigError(f"{where} written content cannot have a reason.")
        elif size is not None or digest is not None or reason is None:
            raise ConfigError(f"{where} unwritten content is malformed.")
    return ArtefactRecord(path, written, size, digest, reason)


def _artefact_table(value: object) -> ArtefactTable:
    if type(value) is not ArtefactTable:
        raise ConfigError("artefacts must be an exact ArtefactTable.")
    variants = object.__getattribute__(value, "resolved_variants")
    if type(variants) not in (tuple, list):
        raise ConfigError("artefacts.resolved_variants must be a sequence.")
    return ArtefactTable(
        marker=_artefact_record(value.marker, where="artefacts.marker", metadata=True),
        lock=_artefact_record(value.lock, where="artefacts.lock", metadata=True),
        journal=_artefact_record(value.journal, where="artefacts.journal", metadata=True),
        input=_artefact_record(value.input, where="artefacts.input", metadata=False),
        resolved_base=_artefact_record(
            value.resolved_base, where="artefacts.resolved_base", metadata=False
        ),
        resolved_variants=tuple(
            _artefact_record(row, where=f"artefacts.resolved_variants[{index}]", metadata=False)
            for index, row in enumerate(variants)
        ),
        provenance=_artefact_record(value.provenance, where="artefacts.provenance", metadata=True),
        diagnostics=_artefact_record(
            value.diagnostics, where="artefacts.diagnostics", metadata=True
        ),
    )


class AuditTrace:
    """Own one lock and an append-only ledger of validated frozen facts."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._boundaries: list[CompletedBoundary] = []
        self._defaults: list[DefaultRecord] = []
        self._deliveries: list[DeliveryRecord] = []
        self._inputs: list[InputRecord] = []
        self._findings: list[FindingRecord] = []
        self._plugins: list[Mapping[str, JsonValue]] = []
        self._python_targets: list[PythonTargetRecord] = []
        self._seeds: list[SeedRecord] = []
        self._variants: list[VariantRecord] = []
        self._resources: list[ResourceRecord] = []
        self._gates: list[GateRecord] = []
        self._deferred: list[DeferredValidationRecord] = []
        self._parsed: list[ParsedRunRecord] = []
        self._outcomes: list[RunOutcomeRecord] = []
        self._deletions: list[DeletionAuditRecord] = []
        self._resolved: list[ResolvedLayerRecord] = []
        self._encodings: list[PathEncoding] = []
        self._bootstrap: Mapping[str, JsonValue] | None = None
        self._software: Mapping[str, JsonValue] | None = None
        self._runtime: Mapping[str, JsonValue] | None = None
        self._error: ErrorRecord | None = None
        self._artefacts: ArtefactTable | None = None
        self._materializations: list[ArtefactMaterialization] = []
        self._seed_layers: set[LayerIdentity] = set()
        self._variant_layers: set[LayerIdentity] = set()
        self._resource_layers: set[LayerIdentity] = set()
        self._parsed_keys: set[tuple[LayerIdentity, RunDescriptor]] = set()
        self._outcome_keys: set[tuple[LayerIdentity, RunDescriptor]] = set()
        self._resolved_layers: set[LayerIdentity] = set()
        self._materialized_keys: set[tuple[str, int | None]] = set()

    def boundary_completed(self, stage: Stage, layer: LayerIdentity | None = None) -> None:
        exact_stage = _stage(stage)
        exact_layer = None if layer is None else _layer(layer)
        row = CompletedBoundary(exact_stage, exact_layer)
        with self._lock:
            if row in self._boundaries:
                raise ConfigError(
                    f"audit boundary {exact_stage!r} is already completed for {exact_layer!r}."
                )
            self._boundaries.append(row)

    def record_findings(
        self,
        stage: Stage,
        layer: LayerIdentity,
        findings: Sequence[Mapping[str, str]],
    ) -> None:
        exact_stage = _stage(stage)
        exact_layer = _layer(layer)
        if type(findings) not in (tuple, list):
            raise ConfigError("audit findings must be a sequence.")
        rows: list[FindingRecord] = []
        for index, finding in enumerate(findings):
            item = _mapping(
                finding,
                keys=_FINDING_KEYS,
                where=f"audit finding[{index}]",
            )
            severity = item["severity"]
            if severity not in ("refuse", "warn", "report"):
                raise ConfigError(f"unknown finding severity {severity!r}.")
            rows.append(
                FindingRecord(
                    exact_layer,
                    exact_stage,
                    _text(item["check"], where="finding.check"),
                    cast(str, severity),
                    _text(item["where"], where="finding.where", empty=True),
                    _text(item["message"], where="finding.message", empty=True),
                )
            )
        with self._lock:
            self._findings.extend(rows)

    def record_default(self, layer: LayerIdentity, path: str, value: JsonValue) -> None:
        row = DefaultRecord(
            _layer(layer),
            _text(path, where="default path"),
            _freeze_json(value, where="default value"),
        )
        with self._lock:
            self._defaults.append(row)

    def record_delivery(
        self,
        layer: LayerIdentity,
        destination: DestinationDescriptor,
        *,
        dtype: str,
        origin: Origin,
        unit: str | None,
    ) -> None:
        row = DeliveryRecord(
            _layer(layer),
            _destination(destination),
            _text(dtype, where="delivery dtype"),
            _typed_origin(origin, where="delivery origin"),
            _optional_text(unit, where="delivery unit"),
        )
        with self._lock:
            self._deliveries.append(row)

    def record_input(self, layer: LayerIdentity, row: Mapping[str, JsonValue]) -> None:
        exact_layer = _layer(layer)
        captured = _mapping(row, keys=LAYER_ROW_KEYS["input"], where="audit input")
        for key in ("document_path", "path", "realpath", "format"):
            _text(captured[key], where=f"audit input.{key}")
        if captured["kind"] not in ("file", "directory"):
            raise ConfigError("audit input.kind is invalid.")
        _sha256(captured["sha256"], where="audit input.sha256")
        members = captured["members"]
        if type(members) is not tuple:
            raise ConfigError("audit input.members must be a sequence.")
        member_names: list[str] = []
        for index, member in enumerate(members):
            item = _mapping(
                member,
                keys=_CAPTURE_MEMBER_KEYS,
                where=f"audit input.members[{index}]",
            )
            for key in ("relative_path", "path", "realpath"):
                _text(item[key], where=f"audit input member.{key}")
            _sha256(item["sha256"], where="audit input member.sha256")
            member_names.append(cast(str, item["relative_path"]))
        if member_names != sorted(member_names) or len(member_names) != len(set(member_names)):
            raise ConfigError("audit input members must be unique and sorted.")
        if captured["kind"] == "file" and members:
            raise ConfigError("audit file input cannot contain members.")
        frozen = cast(Mapping[str, JsonValue], _freeze_json(captured, where="audit input"))
        with self._lock:
            self._inputs.append(InputRecord(exact_layer, frozen))

    def record_python_target(self, layer: LayerIdentity, row: Mapping[str, JsonValue]) -> None:
        item = _mapping(row, keys=LAYER_ROW_KEYS["python_target"], where="python target")
        if item["code_hash"] is not None or item["unobserved_io"] is not True:
            raise ConfigError("python target trust-boundary fields are invalid.")
        record = PythonTargetRecord(
            _layer(layer),
            _text(item["document_path"], where="python target.document_path"),
            _text(item["target"], where="python target.target"),
            None,
            True,
        )
        with self._lock:
            self._python_targets.append(record)

    def record_seed(self, layer: LayerIdentity, row: Mapping[str, JsonValue]) -> None:
        exact_layer = _layer(layer)
        item = _mapping(row, keys=LAYER_ROW_KEYS["seed"], where="seed")
        root = item["root"]
        if root is not None and (type(root) is not int or root < 0):
            raise ConfigError("seed.root must be a non-negative integer or null.")
        named = _freeze_json(item["named"], where="seed.named")
        if not static_isinstance(named, Mapping):
            raise ConfigError("seed.named must be a mapping.")
        for name, value in named.items():
            _text(name, where="seed name")
            if type(value) is not int or value < 0:
                raise ConfigError("named seeds must be non-negative integers.")
        record = SeedRecord(exact_layer, root, cast(Mapping[str, int], named))
        with self._lock:
            if exact_layer in self._seed_layers:
                raise ConfigError(f"seed facts for {exact_layer!r} are already recorded.")
            self._seed_layers.add(exact_layer)
            self._seeds.append(record)

    def record_variant(self, layer: LayerIdentity, row: Mapping[str, JsonValue]) -> None:
        exact_layer = _layer(layer)
        item = _mapping(row, keys=LAYER_ROW_KEYS["variant"], where="variant")
        status = item["status"]
        if status not in ("ok", "refused", "error", "not_reached"):
            raise ConfigError(f"unknown variant status {status!r}.")
        record = VariantRecord(
            exact_layer,
            _optional_text(item["encoded_name"], where="variant.encoded_name"),
            cast(str, status),
            _sha256(item["resolved_sha256"], where="variant.resolved_sha256", optional=True),
        )
        with self._lock:
            if exact_layer in self._variant_layers:
                raise ConfigError(f"variant facts for {exact_layer!r} are already recorded.")
            self._variant_layers.add(exact_layer)
            self._variants.append(record)

    def record_resource(self, layer: LayerIdentity, row: Mapping[str, JsonValue]) -> None:
        exact_layer = _layer(layer)
        item = _mapping(row, keys=LAYER_ROW_KEYS["resource"], where="resource")
        order = _string_sequence(item["build_order"], where="resource.build_order")
        shared = _freeze_json(item["shared_objects"], where="resource.shared_objects")
        if not static_isinstance(shared, Mapping):
            raise ConfigError("resource.shared_objects must be a mapping.")
        copied_shared = {
            _text(key, where="resource name"): _text(value, where="resource shared object")
            for key, value in shared.items()
        }
        frozen_shared = cast(
            Mapping[str, str], freeze_evidence(copied_shared, where="resource.shared_objects")
        )
        record = ResourceRecord(exact_layer, order, frozen_shared)
        with self._lock:
            if exact_layer in self._resource_layers:
                raise ConfigError(f"resource facts for {exact_layer!r} are already recorded.")
            self._resource_layers.add(exact_layer)
            self._resources.append(record)

    def record_gate(self, layer: LayerIdentity, row: Mapping[str, JsonValue]) -> None:
        item = _mapping(row, keys=LAYER_ROW_KEYS["gate"], where="gate")
        record = GateRecord(
            _layer(layer),
            _text(item["name"], where="gate.name"),
            _text(item["schema_id"], where="gate.schema_id"),
            _text(item["declared_mode"], where="gate.declared_mode"),
            _text(item["effective_state"], where="gate.effective_state"),
            _optional_text(item["reason"], where="gate.reason"),
        )
        with self._lock:
            self._gates.append(record)

    def record_deferred_validation(
        self, layer: LayerIdentity, row: Mapping[str, JsonValue]
    ) -> None:
        item = _mapping(
            row,
            keys=LAYER_ROW_KEYS["deferred_validation"],
            where="deferred validation",
        )
        record = DeferredValidationRecord(
            _layer(layer),
            _descriptor(item["descriptor"], where="deferred validation.descriptor"),
            _string_sequence(item["checks"], where="deferred validation.checks"),
        )
        with self._lock:
            self._deferred.append(record)

    def record_parsed_run(self, layer: LayerIdentity, row: Mapping[str, JsonValue]) -> None:
        exact_layer = _layer(layer)
        item = _mapping(row, keys=LAYER_ROW_KEYS["parsed_run"], where="parsed run")
        descriptor = _descriptor(item["descriptor"], where="parsed run.descriptor")
        options = _freeze_json(item["resolved_options"], where="parsed run.resolved_options")
        if not static_isinstance(options, Mapping):
            raise ConfigError("parsed run.resolved_options must be a mapping.")
        record = ParsedRunRecord(
            exact_layer,
            descriptor,
            cast(Mapping[str, JsonValue], options),
            _string_sequence(item["deferred_checks"], where="parsed run.deferred_checks"),
        )
        key = (exact_layer, descriptor)
        with self._lock:
            if key in self._parsed_keys:
                raise ConfigError("parsed run is already recorded.")
            self._parsed_keys.add(key)
            self._parsed.append(record)

    def record_run_outcome(self, layer: LayerIdentity, row: Mapping[str, JsonValue]) -> None:
        exact_layer = _layer(layer)
        item = _mapping(row, keys=LAYER_ROW_KEYS["run_outcome"], where="run outcome")
        descriptor = _descriptor(item["descriptor"], where="run outcome.descriptor")
        status = item["status"]
        if status not in ("ok", "expected_refusal", "refused", "error"):
            raise ConfigError(f"unknown run outcome status {status!r}.")
        wall_time = item["wall_time_ns"]
        if type(wall_time) is not int or wall_time < 0:
            raise ConfigError("run outcome.wall_time_ns must be non-negative.")
        capture_scope = item["capture_scope"]
        if capture_scope not in (None, "arbitrary_exception"):
            raise ConfigError("run outcome.capture_scope is invalid.")
        dirt = item["is_dirt_error"]
        if dirt is not None and type(dirt) is not bool:
            raise ConfigError("run outcome.is_dirt_error must be bool or null.")
        record = RunOutcomeRecord(
            exact_layer,
            descriptor,
            cast(str, status),
            wall_time,
            _optional_text(item["exception_type"], where="run outcome.exception_type"),
            _optional_text(item["exception_message"], where="run outcome.exception_message"),
            cast(str | None, capture_scope),
            cast(bool | None, dirt),
        )
        key = (exact_layer, descriptor)
        with self._lock:
            if key not in self._parsed_keys:
                raise ConfigError("run outcome requires its matching parsed run.")
            if key in self._outcome_keys:
                raise ConfigError("run outcome is already recorded.")
            self._outcome_keys.add(key)
            self._outcomes.append(record)

    def freeze_layer(self, layer: LayerIdentity, row: Mapping[str, JsonValue]) -> None:
        exact_layer = _layer(layer)
        item = _raw_mapping(row, keys=LAYER_ROW_KEYS["resolved_layer"], where="resolved layer")
        document = _freeze_json(
            item["effective_document"], where="resolved layer.effective_document"
        )
        if not static_isinstance(document, Mapping):
            raise ConfigError("resolved layer.effective_document must be a mapping.")
        origins = item["origins"]
        if type(origins) is not OriginNode:
            raise ConfigError("resolved layer.origins must be an exact OriginNode.")
        copied_origins = OriginNode(origins.origin, origins.children)
        declared = _descriptor_sequence(item["declared_runs"], where="resolved layer.declared_runs")
        execution = _descriptor_sequence(
            item["execution_runs"], where="resolved layer.execution_runs"
        )
        audit = _freeze_json(item["audit"], where="resolved layer.audit")
        if not static_isinstance(audit, Mapping):
            raise ConfigError("resolved layer.audit must be a mapping.")
        record = ResolvedLayerRecord(
            exact_layer,
            cast(Mapping[str, JsonValue], document),
            copied_origins,
            declared,
            execution,
            cast(Mapping[str, JsonValue], audit),
        )
        with self._lock:
            if CompletedBoundary("postflight", exact_layer) not in self._boundaries:
                raise ConfigError("resolved layer requires completed postflight.")
            missing = [
                descriptor
                for descriptor in declared
                if (exact_layer, descriptor) not in self._parsed_keys
            ]
            if missing:
                raise ConfigError("resolved layer requires every matching parsed run.")
            if exact_layer in self._resolved_layers:
                raise ConfigError(f"resolved layer {exact_layer!r} is already frozen.")
            self._resolved_layers.add(exact_layer)
            self._resolved.append(record)

    def record_deletion(self, layer: LayerIdentity, row: Mapping[str, JsonValue]) -> None:
        item = _mapping(row, keys=LAYER_ROW_KEYS["deletion"], where="deletion")
        path = _freeze_json(item["path"], where="deletion.path")
        if type(path) is not tuple or any(
            type(part) not in (str, int) or type(part) is bool for part in path
        ):
            raise ConfigError("deletion.path must contain only strings and integers.")
        record = DeletionAuditRecord(
            _layer(layer),
            cast(tuple[str | int, ...], path),
            _origin(item["origin"], where="deletion.origin"),
        )
        with self._lock:
            self._deletions.append(record)

    def record_bootstrap(self, row: Mapping[str, JsonValue]) -> None:
        item = _mapping(row, keys=_BOOTSTRAP_ROW_KEYS, where="bootstrap")
        if type(item["protocol_version"]) is not int or item["protocol_version"] < 1:
            raise ConfigError("bootstrap.protocol_version must be a positive integer.")
        if item["launch_mode"] not in ("cli", "embedded"):
            raise ConfigError("bootstrap.launch_mode is invalid.")
        _sha256(item["input_sha256"], where="bootstrap.input_sha256")
        presets = _string_sequence(item["presets"], where="bootstrap.presets")
        for key in ("source_name", "source_path", "base_dir"):
            _text(item[key], where=f"bootstrap.{key}")
        if item["source_realpath"] is not None:
            _text(item["source_realpath"], where="bootstrap.source_realpath")
        if item["invocation_outputs_dir"] is not None:
            _text(item["invocation_outputs_dir"], where="bootstrap.invocation_outputs_dir")
        projected = dict(item)
        projected["presets"] = presets
        frozen = cast(Mapping[str, JsonValue], _freeze_json(projected, where="bootstrap"))
        with self._lock:
            if self._bootstrap is not None:
                raise ConfigError("bootstrap facts are already recorded.")
            self._bootstrap = frozen

    def record_runtime(self, row: Mapping[str, JsonValue]) -> None:
        item = _mapping(row, keys=RUNTIME_ROW_KEYS, where="runtime")
        requested = _mapping(
            item["requested"], keys=RUNTIME_REQUESTED_ROW_KEYS, where="runtime.requested"
        )
        actual = _mapping(item["actual"], keys=RUNTIME_ACTUAL_ROW_KEYS, where="runtime.actual")
        prior = _mapping(
            item["prior_environment"],
            keys=RUNTIME_PRIOR_ENVIRONMENT_ROW_KEYS,
            where="runtime.prior_environment",
        )
        if type(requested["jax_enable_x64"]) is not bool or requested["platform"] not in (
            "auto",
            "cpu",
            "gpu",
            "tpu",
        ):
            raise ConfigError("runtime requested state is invalid.")
        if type(actual["jax_enable_x64"]) is not bool:
            raise ConfigError("runtime actual jax_enable_x64 must be a bool.")
        _text(actual["backend"], where="runtime.actual.backend")
        for key in RUNTIME_PRIOR_ENVIRONMENT_ROW_KEYS:
            if prior[key] is not None:
                _text(prior[key], where=f"runtime.prior_environment.{key}", empty=True)
        if actual["jax_enable_x64"] is not requested["jax_enable_x64"]:
            raise ConfigError("runtime actual x64 contradicts requested state.")
        if requested["platform"] != "auto" and actual["backend"] != requested["platform"]:
            raise ConfigError("runtime actual backend contradicts requested state.")
        frozen = cast(Mapping[str, JsonValue], _freeze_json(item, where="runtime"))
        with self._lock:
            if self._runtime is not None:
                raise ConfigError("runtime facts are already recorded.")
            self._runtime = frozen

    def record_software(self, row: Mapping[str, JsonValue]) -> None:
        """Bind one detached software acquisition row before serialization."""
        frozen = _freeze_json(row, where="software")
        if not static_isinstance(frozen, Mapping):
            raise ConfigError("software facts must be a mapping.")
        with self._lock:
            if self._software is not None:
                raise ConfigError("software facts are already recorded.")
            self._software = cast(Mapping[str, JsonValue], frozen)

    @staticmethod
    def _plugin_value_reason(item: Mapping[str, object], field: str, *, where: str) -> None:
        value = item[field]
        reason = item[f"{field}_reason"]
        allowed = _PLUGIN_REASON_SETS[field]
        if reason is not None and (type(reason) is not str or reason not in allowed):
            raise ConfigError(f"{where} reason is invalid.")
        if (value is None) == (reason is None):
            raise ConfigError(f"{where} value and reason must be exact complements.")

    def record_plugin(self, row: Mapping[str, JsonValue]) -> None:
        item = _mapping(row, keys=PLUGIN_ROW_KEYS, where="plugin")
        _text(item["name"], where="plugin.name")
        if type(item["already_imported"]) is not bool or item["unobserved_io"] is not True:
            raise ConfigError("plugin boolean fields are invalid.")
        for field in ("origin", "loader_type", "resolved_path", "code_hash"):
            self._plugin_value_reason(item, field, where=f"plugin.{field}")
            if item[field] is not None:
                _text(item[field], where=f"plugin.{field}")
        if item["code_hash"] is not None:
            _sha256(item["code_hash"], where="plugin.code_hash")
        distributions = item["distributions"]
        if type(distributions) is not tuple:
            raise ConfigError("plugin.distributions must be a sequence.")
        distribution_reason = item["distributions_reason"]
        if distributions:
            if distribution_reason is not None:
                raise ConfigError("plugin distributions cannot have a reason.")
        elif distribution_reason != "no_distribution":
            raise ConfigError("empty plugin distributions require no_distribution.")
        names: list[str] = []
        for index, candidate in enumerate(distributions):
            distribution = _mapping(
                candidate,
                keys=PLUGIN_DISTRIBUTION_ROW_KEYS,
                where=f"plugin.distributions[{index}]",
            )
            name = _text(distribution["name"], where="plugin distribution.name")
            normalized = _PEP_503_RUN.sub("-", name.lower())
            if name != normalized:
                raise ConfigError("plugin distribution names must be normalized.")
            names.append(name)
            for field in ("version", "direct_url"):
                self._plugin_value_reason(distribution, field, where=f"plugin distribution.{field}")
            if distribution["version"] is not None:
                _text(distribution["version"], where="plugin distribution.version")
            if distribution["direct_url"] is not None:
                direct = _freeze_json(
                    distribution["direct_url"], where="plugin distribution.direct_url"
                )
                if not static_isinstance(direct, Mapping):
                    raise ConfigError("plugin distribution.direct_url must be a mapping.")
        if names != sorted(names) or len(names) != len(set(names)):
            raise ConfigError("plugin distributions must be unique and sorted.")
        frozen = cast(Mapping[str, JsonValue], _freeze_json(item, where="plugin"))
        with self._lock:
            self._plugins.append(frozen)

    def record_path_encoding(self, row: Mapping[str, JsonValue]) -> None:
        item = _mapping(row, keys=_PATH_ENCODING_KEYS, where="path encoding")
        kind = item["kind"]
        if kind not in ("run", "variant"):
            raise ConfigError(f"unknown encoded-name kind {kind!r}.")
        document_name = _text(item["document_name"], where="path encoding.document_name")
        encoded_name = _text(item["encoded_name"], where="path encoding.encoded_name")
        if encoded_name != encode_name(document_name):
            raise ConfigError("path encoding does not match the document name.")
        record = PathEncoding(cast(str, kind), document_name, encoded_name)
        with self._lock:
            if any(
                old.kind == record.kind and old.document_name == record.document_name
                for old in self._encodings
            ):
                raise ConfigError("path encoding is already recorded.")
            self._encodings.append(record)

    def record_error(self, row: Mapping[str, JsonValue]) -> None:
        item = _mapping(row, keys=_ERROR_KEYS, where="error")
        record = ErrorRecord(
            _text(item["exception_type"], where="error.exception_type"),
            _text(item["message"], where="error.message", empty=True),
        )
        with self._lock:
            if self._error is not None:
                raise ConfigError("error is already recorded.")
            self._error = record

    def configure_artefacts(self, table: ArtefactTable) -> None:
        copied = _artefact_table(table)
        all_rows = (
            copied.marker,
            copied.lock,
            copied.journal,
            copied.input,
            copied.resolved_base,
            *copied.resolved_variants,
            copied.provenance,
            copied.diagnostics,
        )
        if any(row.written for row in all_rows):
            raise ConfigError("configured artefacts must all start unwritten.")
        paths = [row.relative_path for row in all_rows]
        if len(paths) != len(set(paths)):
            raise ConfigError("configured artefact paths must be unique.")
        with self._lock:
            if self._artefacts is not None:
                raise ConfigError("artefacts are already configured.")
            self._artefacts = copied

    def record_artefact_materialized(self, row: ArtefactMaterialization) -> None:
        if type(row) is not ArtefactMaterialization:
            raise ConfigError("materialization must be an exact ArtefactMaterialization.")
        slot = object.__getattribute__(row, "slot")
        index = object.__getattribute__(row, "variant_index")
        path = _text(
            object.__getattribute__(row, "relative_path"), where="materialization.relative_path"
        )
        size = object.__getattribute__(row, "bytes")
        digest = object.__getattribute__(row, "sha256")
        if slot not in _METADATA_SLOTS | _CONTENT_SLOTS:
            raise ConfigError(f"unknown artefact slot {slot!r}.")
        if slot == "resolved_variant":
            if type(index) is not int or index < 0:
                raise ConfigError("resolved variant materialization needs an index.")
        elif index is not None:
            raise ConfigError("scalar artefact materialization cannot have an index.")
        if slot in _METADATA_SLOTS:
            if size is not None or digest is not None:
                raise ConfigError("metadata materialization cannot carry bytes/hash.")
        else:
            if type(size) is not int or type(size) is bool or size < 0:
                raise ConfigError("content materialization needs non-negative bytes.")
            _sha256(digest, where="materialization.sha256")
        copied = ArtefactMaterialization(slot, index, path, size, digest)
        key = (slot, index)
        with self._lock:
            if self._artefacts is None:
                raise ConfigError("artefacts must be configured before materialization.")
            if slot == "resolved_variant":
                assert type(index) is int
                if index >= len(self._artefacts.resolved_variants):
                    raise ConfigError("resolved variant materialization index is out of range.")
                expected = self._artefacts.resolved_variants[index]
            else:
                expected = getattr(self._artefacts, slot)
            if path != expected.relative_path:
                raise ConfigError("artefact materialization path does not match its slot.")
            if key in self._materialized_keys:
                raise ConfigError("artefact slot is already materialized.")
            self._materialized_keys.add(key)
            self._materializations.append(copied)

    def _fold_artefacts(self) -> ArtefactTable:
        table = self._artefacts if self._artefacts is not None else empty_artefact_table()
        scalar = {
            "marker": table.marker,
            "lock": table.lock,
            "journal": table.journal,
            "input": table.input,
            "resolved_base": table.resolved_base,
            "provenance": table.provenance,
            "diagnostics": table.diagnostics,
        }
        variants = list(table.resolved_variants)
        for event in self._materializations:
            reason = "metadata_envelope" if event.slot in _METADATA_SLOTS else None
            materialized = ArtefactRecord(
                event.relative_path, True, event.bytes, event.sha256, reason
            )
            if event.slot == "resolved_variant":
                assert event.variant_index is not None
                variants[event.variant_index] = materialized
            else:
                scalar[event.slot] = materialized
        return ArtefactTable(
            marker=scalar["marker"],
            lock=scalar["lock"],
            journal=scalar["journal"],
            input=scalar["input"],
            resolved_base=scalar["resolved_base"],
            resolved_variants=tuple(variants),
            provenance=scalar["provenance"],
            diagnostics=scalar["diagnostics"],
        )

    def snapshot(self) -> AuditSnapshot:
        with self._lock:
            return AuditSnapshot(
                bootstrap=self._bootstrap,
                software=self._software,
                completed_boundaries=tuple(self._boundaries),
                defaults=tuple(self._defaults),
                deliveries=tuple(self._deliveries),
                inputs=tuple(self._inputs),
                findings=tuple(self._findings),
                plugins=tuple(self._plugins),
                python_targets=tuple(self._python_targets),
                runtime=self._runtime,
                seeds=tuple(self._seeds),
                variants=tuple(self._variants),
                resources=tuple(self._resources),
                gates=tuple(self._gates),
                deferred_validations=tuple(self._deferred),
                parsed_runs=tuple(self._parsed),
                run_outcomes=tuple(self._outcomes),
                deletions=tuple(self._deletions),
                resolved_layers=tuple(self._resolved),
                path_encodings=tuple(self._encodings),
                artefacts=self._fold_artefacts(),
                error=self._error,
            )

    def completed_boundaries(self) -> Sequence[CompletedBoundary]:
        with self._lock:
            return tuple(self._boundaries)
