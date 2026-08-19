from __future__ import annotations

import dataclasses
import inspect
import threading

import pytest

from _rheplicant_bootstrap.audit import AuditTrace
from _rheplicant_bootstrap.audit.types import (
    ArtefactMaterialization,
    ArtefactRecord,
    ArtefactTable,
    AuditSnapshot,
    DefaultRecord,
    DeferredValidationRecord,
    DeletionAuditRecord,
    DeliveryRecord,
    FindingRecord,
    GateRecord,
    InputRecord,
    ParsedRunRecord,
    PythonTargetRecord,
    ResolvedLayerRecord,
    ResourceRecord,
    RunOutcomeRecord,
    SeedRecord,
    VariantRecord,
)
from _rheplicant_bootstrap.errors import ConfigError
from _rheplicant_bootstrap.layering import OriginNode
from _rheplicant_bootstrap.types import LayerIdentity, Origin, TraceSink

SHA256_4 = "88d4266fd4e6338d13b845fcf289579d209c897823b9217da3e161936f031589"


@pytest.fixture
def trace():
    return AuditTrace()


@pytest.fixture
def table():
    def metadata(path, reason):
        return ArtefactRecord(path, False, None, None, reason)

    def content(path, reason="layer_not_complete"):
        return ArtefactRecord(path, False, None, None, reason)

    return ArtefactTable(
        marker=metadata(".rheplicant-results.json", "boundary_not_reached"),
        lock=metadata(".rheplicant-lock", "transaction_not_reached"),
        journal=metadata(".rheplicant-transaction.json", "transaction_not_reached"),
        input=content("config.input.yaml", "boundary_not_reached"),
        resolved_base=content("config.resolved.yaml"),
        resolved_variants=(
            content("variants/n-61/config.resolved.yaml"),
            content("variants/n-62/config.resolved.yaml"),
        ),
        provenance=metadata("provenance.json", "boundary_not_reached"),
        diagnostics=metadata("diagnostics.json", "boundary_not_reached"),
    )


def test_record_and_snapshot_field_contracts():
    layer_rows = (
        DefaultRecord,
        DeliveryRecord,
        InputRecord,
        FindingRecord,
        PythonTargetRecord,
        SeedRecord,
        VariantRecord,
        ResourceRecord,
        GateRecord,
        DeferredValidationRecord,
        ParsedRunRecord,
        RunOutcomeRecord,
        DeletionAuditRecord,
        ResolvedLayerRecord,
    )
    assert all(dataclasses.fields(row)[0].name == "layer" for row in layer_rows)
    assert tuple(field.name for field in dataclasses.fields(AuditSnapshot)) == (
        "bootstrap",
        "completed_boundaries",
        "defaults",
        "deliveries",
        "inputs",
        "findings",
        "plugins",
        "python_targets",
        "runtime",
        "seeds",
        "variants",
        "resources",
        "gates",
        "deferred_validations",
        "parsed_runs",
        "run_outcomes",
        "deletions",
        "resolved_layers",
        "path_encodings",
        "artefacts",
        "error",
    )
    for method in (
        "boundary_completed",
        "record_findings",
        "record_default",
        "record_delivery",
        "record_input",
        "record_parsed_run",
        "record_run_outcome",
        "freeze_layer",
        "completed_boundaries",
    ):
        assert tuple(inspect.signature(getattr(AuditTrace, method)).parameters) == tuple(
            inspect.signature(getattr(TraceSink, method)).parameters
        )


def test_trace_freezes_at_append_and_snapshot_is_detached():
    trace = AuditTrace()
    layer = LayerIdentity("base", None)
    value = {"x": [1]}
    trace.record_default(layer, "model.x", value)
    value["x"][0] = 9
    first = trace.snapshot()
    assert first.defaults[0].value["x"] == (1,)
    trace.record_default(layer, "model.y", 2)
    assert len(first.defaults) == 1
    assert len(trace.snapshot().defaults) == 2


def test_boundaries_are_ordered_unique_completions():
    trace = AuditTrace()
    trace.boundary_completed("source")
    trace.boundary_completed("preflight", LayerIdentity("base", None))
    with pytest.raises(ConfigError, match="already completed"):
        trace.boundary_completed("preflight", LayerIdentity("base", None))


def test_same_path_in_two_layers_never_coalesces(trace):
    trace.record_default(LayerIdentity("base", None), "model.x", 1)
    trace.record_default(LayerIdentity("variant", "v"), "model.x", 2)
    assert [(row.layer, row.value) for row in trace.snapshot().defaults] == [
        (LayerIdentity("base", None), 1),
        (LayerIdentity("variant", "v"), 2),
    ]


def test_findings_and_deletions_keep_stage_layer_and_order(trace):
    layer = LayerIdentity("variant", "v")
    trace.record_findings(
        "preflight",
        layer,
        [
            {
                "check": "A30",
                "severity": "warn",
                "where": "model.x",
                "message": "first",
            },
            {
                "check": "A31",
                "severity": "report",
                "where": "runs[0]",
                "message": "second",
            },
        ],
    )
    trace.record_deletion(
        layer,
        {"path": ["model", "gone"], "origin": {"kind": "variant", "name": "v"}},
    )
    snapshot = trace.snapshot()
    assert [row.message for row in snapshot.findings] == ["first", "second"]
    assert snapshot.deletions[0].layer == layer


def test_outcomes_and_layer_freeze_require_prior_facts(trace):
    layer = LayerIdentity("base", None)
    descriptor = {"index": 0, "name": "fit", "kind": "nuts", "variant": None}
    outcome = {
        "descriptor": descriptor,
        "status": "ok",
        "wall_time_ns": 1,
        "exception_type": None,
        "exception_message": None,
        "capture_scope": None,
        "is_dirt_error": None,
    }
    with pytest.raises(ConfigError, match="matching parsed"):
        trace.record_run_outcome(layer, outcome)
    trace.record_parsed_run(
        layer,
        {"descriptor": descriptor, "resolved_options": {}, "deferred_checks": []},
    )
    trace.record_run_outcome(layer, outcome)
    resolved = {
        "effective_document": {},
        "origins": OriginNode(None, {}),
        "declared_runs": [descriptor],
        "execution_runs": [descriptor],
        "audit": {},
    }
    with pytest.raises(ConfigError, match="postflight"):
        trace.freeze_layer(layer, resolved)
    trace.boundary_completed("postflight", layer)
    trace.freeze_layer(layer, resolved)
    with pytest.raises(ConfigError, match="already frozen"):
        trace.freeze_layer(layer, resolved)


def test_artefact_events_fold_monotonically_without_sparse_keys(trace, table):
    trace.configure_artefacts(table)
    trace.record_artefact_materialized(
        ArtefactMaterialization("input", None, "config.input.yaml", 4, SHA256_4)
    )
    trace.record_artefact_materialized(
        ArtefactMaterialization("journal", None, ".rheplicant-transaction.json", None, None)
    )
    artefacts = trace.snapshot().artefacts
    assert dataclasses.fields(artefacts) == dataclasses.fields(ArtefactTable)
    assert (artefacts.input.written, artefacts.input.bytes, artefacts.input.reason) == (
        True,
        4,
        None,
    )
    assert (
        artefacts.journal.written,
        artefacts.journal.bytes,
        artefacts.journal.sha256,
        artefacts.journal.reason,
    ) == (True, None, None, "metadata_envelope")
    with pytest.raises(ConfigError, match="already materialized"):
        trace.record_artefact_materialized(
            ArtefactMaterialization("input", None, "config.input.yaml", 4, SHA256_4)
        )


def test_concurrent_appends_are_complete_and_unique(trace):
    layer = LayerIdentity("base", None)
    threads = [
        threading.Thread(target=trace.record_default, args=(layer, f"x.{i}", i)) for i in range(50)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    rows = trace.snapshot().defaults
    assert len(rows) == 50
    assert {row.path for row in rows} == {f"x.{i}" for i in range(50)}


@pytest.mark.parametrize("bad", [object(), b"bytes", {"x": object()}])
def test_non_json_default_is_refused(trace, bad):
    with pytest.raises(ConfigError, match="JSON|evidence"):
        trace.record_default(LayerIdentity("base", None), "x", bad)


def test_delivery_copies_typed_origin(trace):
    from _rheplicant_bootstrap.types import DestinationDescriptor

    trace.record_delivery(
        LayerIdentity("base", None),
        DestinationDescriptor("model.x", "config_path", "configured.x"),
        dtype="float64",
        origin=Origin("user"),
        unit="m",
    )
    assert trace.snapshot().deliveries[0].origin == Origin("user")
