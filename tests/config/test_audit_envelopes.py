from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

import _rheplicant_bootstrap.audit.bundle as audit_bundle
from _rheplicant_bootstrap.audit.integrity import INTEGRITY_NAME
import _rheplicant_bootstrap.audit.software as audit_software
from _rheplicant_bootstrap.audit import AuditTrace
from _rheplicant_bootstrap.audit.bundle import (
    candidate_serialization_snapshot,
    merge_bundle_files,
    serialize_bundle,
    with_integrity,
    terminal_reserialization_snapshot,
    validate_serialized_bundle,
)
from _rheplicant_bootstrap.audit.diagnostics import (
    CAPTURE_SCOPES,
    DIAGNOSTICS_KEYS,
    RUN_STATUSES,
)
from _rheplicant_bootstrap.audit.json import canonical_json_bytes
from _rheplicant_bootstrap.audit.provenance import (
    ARTEFACT_REASONS,
    ARTEFACT_RECORD_KEYS,
    ARTEFACT_TABLE_KEYS,
    PROVENANCE_KEYS,
    STATUSES,
)
from _rheplicant_bootstrap.audit.resolved import ResolvedArtefact
from _rheplicant_bootstrap.audit.software import collect_software
from _rheplicant_bootstrap.audit.trace import STAGES
from _rheplicant_bootstrap.audit.types import (
    ArtefactRecord,
    ArtefactTable,
    DeferredValidationRecord,
    DeletionAuditRecord,
    FindingRecord,
    GateRecord,
    InputRecord,
    PathEncoding,
    ResolvedLayerRecord,
    RunOutcomeRecord,
)
from _rheplicant_bootstrap.errors import ConfigError
from _rheplicant_bootstrap.layering import OriginNode
from _rheplicant_bootstrap.types import (
    CompletedBoundary,
    LayerIdentity,
    Origin,
    RunDescriptor,
)

INPUT = b"schema_version: 1\n"
INPUT_SHA = hashlib.sha256(INPUT).hexdigest()
GOLDEN = Path(__file__).with_name("golden")


def software_row():
    fact = {"version": "1.0", "reason": None}
    return {
        "rheplicant": {
            "version": "1.0",
            "version_reason": None,
            "source_root": "/src",
            "source_root_reason": None,
            "git_commit": "a" * 40,
            "git_commit_reason": None,
            "dirty": False,
            "dirty_reason": None,
            "tracked_diff_sha256": "0" * 64,
            "tracked_diff_sha256_reason": None,
        },
        "dependencies": {
            name: dict(fact) for name in ("jax", "jaxlib", "equinox", "numpy", "numpyro")
        },
        "python": {
            "version": "3.12.0",
            "implementation": "CPython",
            "executable": "/python",
        },
        "host": {"system": "TestOS", "release": "1", "machine": "test"},
    }


def snapshot(status="ok"):
    trace = AuditTrace()
    trace.record_bootstrap(
        {
            "protocol_version": 1,
            "launch_mode": "cli",
            "input_sha256": INPUT_SHA,
            "presets": (),
            "source_name": "config.yaml",
            "source_path": "/work/config.yaml",
            "source_realpath": "/work/config.yaml",
            "base_dir": "/work",
            "invocation_outputs_dir": None,
            "invocation_outputs_write": None,
        }
    )
    trace.record_software(software_row())
    trace.record_runtime(
        {
            "requested": {"jax_enable_x64": False, "platform": "cpu"},
            "actual": {"jax_enable_x64": False, "backend": "cpu"},
            "prior_environment": {"jax_enable_x64": None, "jax_platforms": None},
        }
    )
    trace.boundary_completed("source")
    if status != "ok":
        trace.record_error(
            {
                "exception_type": (
                    "rheplicant.config.errors.ConfigError"
                    if status == "refused"
                    else "builtins.RuntimeError"
                ),
                "message": "configuration refused" if status == "refused" else "internal error",
            }
        )
    return trace, trace.snapshot()


def test_canonical_json_contract():
    value = {"雪": [1, -0.0], "a": True}
    assert canonical_json_bytes(value) == (
        b'{\n  "a": true,\n  "\xe9\x9b\xaa": [\n    1,\n    -0.0\n  ]\n}\n'
    )
    for value in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ConfigError, match="non-finite"):
            canonical_json_bytes({"bad": value})


@pytest.mark.parametrize("status", STATUSES)
def test_envelopes_match_exact_goldens(status):
    _trace, initial = snapshot(status)
    bundle = serialize_bundle(
        candidate_serialization_snapshot(initial),
        status=status,
        input_bytes=INPUT,
        resolved=(),
    )
    assert bundle.provenance == (GOLDEN / f"provenance-{status}.json").read_bytes()
    assert bundle.diagnostics == (GOLDEN / f"diagnostics-{status}.json").read_bytes()
    assert tuple(bundle.files) == (
        "config.input.yaml",
        "provenance.json",
        "diagnostics.json",
    )
    assert bundle.files["config.input.yaml"] is bundle.input
    assert bundle.files["provenance.json"] is bundle.provenance
    assert bundle.files["diagnostics.json"] is bundle.diagnostics


def test_candidate_and_bundle_are_pure():
    trace, initial = snapshot()
    before = trace.snapshot()
    candidate = candidate_serialization_snapshot(initial)
    first = serialize_bundle(candidate, status="ok", input_bytes=INPUT, resolved=())
    second = serialize_bundle(candidate, status="ok", input_bytes=INPUT, resolved=())
    assert first == second
    assert trace.snapshot() == before
    assert candidate.completed_boundaries[-1] == CompletedBoundary("serialization", None)
    terminal = terminal_reserialization_snapshot(candidate)
    assert terminal == candidate and terminal is not candidate


def test_bundle_does_not_reacquire_software(monkeypatch):
    def explode():
        raise AssertionError("serialization reacquired external software facts")

    monkeypatch.setattr(audit_bundle, "collect_software", explode, raising=False)
    monkeypatch.setattr(audit_software, "collect_software", explode)
    _trace, initial = snapshot()
    serialize_bundle(
        candidate_serialization_snapshot(initial),
        status="ok",
        input_bytes=INPUT,
        resolved=(),
    )


def test_serialized_bundle_validation_refuses_noncanonical_or_disagreeing_metadata():
    _trace, initial = snapshot()
    bundle = serialize_bundle(
        candidate_serialization_snapshot(initial),
        status="ok",
        input_bytes=INPUT,
        resolved=(),
    )
    validate_serialized_bundle(bundle)
    with pytest.raises(ConfigError, match="not JSON"):
        validate_serialized_bundle(dataclasses.replace(bundle, provenance=b"{"))
    diagnostics = json.loads(bundle.diagnostics)
    diagnostics["status"] = "refused"
    forged = canonical_json_bytes(diagnostics)
    files = dict(bundle.files)
    files["diagnostics.json"] = forged
    with pytest.raises(ConfigError, match="statuses disagree"):
        validate_serialized_bundle(
            dataclasses.replace(bundle, diagnostics=forged, files=files)
        )


def test_every_path_the_transaction_writes_is_reserved_against_merged_files():
    """Found in review rather than by a test, which is why it exists.

    ``integrity.json`` was added to the bundle's tail and to the set of files
    ``replace_staged_metadata`` rewrites, and NOT to the list that stops a
    merged file claiming the same name. A scientific product called
    ``integrity.json`` would have merged cleanly here and then failed inside
    ``with_integrity`` with "already present in the bundle" -- a message naming
    neither the product nor the collision, raised from a function whoever named
    the product never called.

    The reserved set is read from ``RESERVED_BUNDLE_PATHS`` rather than listed
    again, so a fourth file written by the transaction is covered the moment it
    is added there.
    """
    from _rheplicant_bootstrap.audit.bundle import RESERVED_BUNDLE_PATHS

    assert INTEGRITY_NAME in RESERVED_BUNDLE_PATHS, RESERVED_BUNDLE_PATHS

    _trace, initial = snapshot()
    bundle = serialize_bundle(
        candidate_serialization_snapshot(initial),
        status="ok",
        input_bytes=INPUT,
        resolved=(),
    )
    for reserved in RESERVED_BUNDLE_PATHS:
        with pytest.raises(ConfigError, match="reserved or duplicated"):
            merge_bundle_files(bundle, {reserved: b"science"})


def test_with_integrity_covers_every_other_file_and_refuses_to_run_twice():
    """The manifest cannot contain itself, and a second one would silently
    cover the first -- so the second call is refused rather than tolerated."""
    _trace, initial = snapshot()
    bundle = serialize_bundle(
        candidate_serialization_snapshot(initial),
        status="ok",
        input_bytes=INPUT,
        resolved=(),
    )
    merged = merge_bundle_files(bundle, {"runs/x/arrays.npz": b"science"})
    anchored = with_integrity(merged)

    assert tuple(anchored.files)[-3:] == (
        INTEGRITY_NAME,
        "provenance.json",
        "diagnostics.json",
    )
    manifest = json.loads(anchored.files[INTEGRITY_NAME])
    listed = {row["relative_path"] for row in manifest["files"]}
    assert listed == set(merged.files), "the manifest and the bundle disagree"
    assert INTEGRITY_NAME not in listed, "a file cannot digest itself"

    with pytest.raises(ConfigError, match="already present"):
        with_integrity(anchored)


def test_scientific_files_are_inserted_before_the_two_fixed_metadata_files():
    _trace, initial = snapshot()
    bundle = serialize_bundle(
        candidate_serialization_snapshot(initial),
        status="ok",
        input_bytes=INPUT,
        resolved=(),
    )
    merged = merge_bundle_files(
        bundle,
        {
            "runs/n-666f7277617264/arrays.npz": b"science",
            "products.json": b"manifest",
        },
    )
    assert tuple(merged.files) == (
        "config.input.yaml",
        "runs/n-666f7277617264/arrays.npz",
        "products.json",
        "provenance.json",
        "diagnostics.json",
    )
    assert tuple(bundle.files) == (
        "config.input.yaml",
        "provenance.json",
        "diagnostics.json",
    )
    validate_serialized_bundle(merged)


@pytest.mark.parametrize(
    "path",
    (
        "config.input.yaml",
        "provenance.json",
        "diagnostics.json",
        "/absolute",
        "../escape",
        "bad//path",
        "bad\\path",
        "nul\0path",
    ),
)
def test_scientific_file_merge_refuses_reserved_or_nonportable_paths(path):
    _trace, initial = snapshot()
    bundle = serialize_bundle(
        candidate_serialization_snapshot(initial),
        status="ok",
        input_bytes=INPUT,
        resolved=(),
    )
    with pytest.raises(ConfigError, match="bundle file"):
        merge_bundle_files(bundle, {path: b"science"})


def test_scientific_file_merge_requires_exact_immutable_pairs():
    _trace, initial = snapshot()
    bundle = serialize_bundle(
        candidate_serialization_snapshot(initial),
        status="ok",
        input_bytes=INPUT,
        resolved=(),
    )
    with pytest.raises(ConfigError, match="exact text/bytes"):
        merge_bundle_files(bundle, {"products.json": bytearray(b"mutable")})


def test_serialization_boundary_is_exactly_one_final_document_row():
    _trace, initial = snapshot()
    candidate = candidate_serialization_snapshot(initial)
    with pytest.raises(ConfigError, match="already present"):
        candidate_serialization_snapshot(candidate)
    for bad in (
        dataclasses.replace(candidate, completed_boundaries=()),
        dataclasses.replace(
            candidate,
            completed_boundaries=(
                CompletedBoundary("serialization", None),
                CompletedBoundary("source", None),
            ),
        ),
        dataclasses.replace(
            candidate,
            completed_boundaries=(CompletedBoundary("serialization", LayerIdentity("base", None)),),
        ),
    ):
        with pytest.raises(ConfigError, match="serialization"):
            serialize_bundle(bad, status="ok", input_bytes=INPUT, resolved=())


def test_bundle_refuses_inconsistent_input_and_resolved_evidence():
    _trace, initial = snapshot()
    candidate = candidate_serialization_snapshot(initial)
    with pytest.raises(ConfigError, match="input bytes"):
        serialize_bundle(candidate, status="ok", input_bytes=b"different", resolved=())
    rogue = ResolvedArtefact(
        LayerIdentity("base", None),
        "config.resolved.yaml",
        b"payload",
        hashlib.sha256(b"payload").hexdigest(),
    )
    with pytest.raises(ConfigError, match="completed snapshot layers"):
        serialize_bundle(candidate, status="ok", input_bytes=INPUT, resolved=(rogue,))
    bad_hash = dataclasses.replace(rogue, sha256="0" * 64)
    resolved = ResolvedLayerRecord(
        LayerIdentity("base", None), {}, OriginNode(None, {}), (), (), {}
    )
    forged = dataclasses.replace(candidate, resolved_layers=(resolved,))
    with pytest.raises(ConfigError, match="digest"):
        serialize_bundle(forged, status="ok", input_bytes=INPUT, resolved=(bad_hash,))


def test_closed_vocabularies_and_top_level_shapes():
    assert STATUSES == ("ok", "refused", "error")
    assert RUN_STATUSES == ("ok", "expected_refusal", "refused", "error")
    assert STAGES == (
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
    assert ARTEFACT_REASONS == (
        "metadata_envelope",
        "transaction_not_reached",
        "boundary_not_reached",
        "layer_not_complete",
        "not_applicable_to_status",
    )
    assert CAPTURE_SCOPES == ("arbitrary_exception",)
    assert PROVENANCE_KEYS == (
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
    assert DIAGNOSTICS_KEYS == (
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
    assert ARTEFACT_TABLE_KEYS == (
        "marker",
        "lock",
        "journal",
        "input",
        "resolved_base",
        "resolved_variants",
        "provenance",
        "diagnostics",
    )
    assert ARTEFACT_RECORD_KEYS == ("relative_path", "written", "bytes", "sha256", "reason")


def test_collect_software_does_not_import_dependencies():
    code = (
        "import json,sys; "
        "from _rheplicant_bootstrap.audit.software import collect_software; "
        "before=set(sys.modules); row=collect_software(); "
        "print(json.dumps(sorted((set(sys.modules)-before)&{'jax','jaxlib','equinox','numpy','numpyro'})))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == []
    assert tuple(collect_software()) == ("rheplicant", "dependencies", "python", "host")


def test_semantic_rows_preserve_layers_order_and_closed_diagnostics():
    _trace, initial = snapshot()
    base = LayerIdentity("base", None)
    variant = LayerIdentity("variant", "later")
    captured = {
        "document_path": "data.source",
        "path": "/work/data.csv",
        "realpath": "/work/data.csv",
        "format": "csv",
        "kind": "file",
        "sha256": "1" * 64,
        "members": (),
    }
    descriptor = RunDescriptor(0, "predict", "predict", None)
    checks = (
        "predict.reuse_available",
        "predict.reuse_succeeded",
        "predict.variant_matches",
        "predict.product_supported",
        "predict.draw_count_available",
    )
    def unwritten(path, reason):
        return ArtefactRecord(path, False, None, None, reason)

    def metadata(path):
        return ArtefactRecord(path, True, None, None, "metadata_envelope")

    artefacts = ArtefactTable(
        marker=metadata(".rheplicant-results.json"),
        lock=metadata(".rheplicant-lock"),
        journal=metadata(".rheplicant-journal"),
        input=ArtefactRecord("config.input.yaml", True, len(INPUT), INPUT_SHA, None),
        resolved_base=unwritten("config.resolved.yaml", "layer_not_complete"),
        resolved_variants=(
            unwritten("variants/n-6c61746572/config.resolved.yaml", "layer_not_complete"),
        ),
        provenance=metadata("provenance.json"),
        diagnostics=metadata("diagnostics.json"),
    )
    enriched = dataclasses.replace(
        initial,
        inputs=(InputRecord(base, captured), InputRecord(variant, captured)),
        findings=(
            FindingRecord(base, "preflight", "first", "warn", "runs[0]", "one"),
            FindingRecord(variant, "postflight", "second", "report", "", "two"),
        ),
        gates=(GateRecord(base, "units", "gate-v1", "warn", "enabled", None),),
        deferred_validations=(DeferredValidationRecord(base, descriptor, checks),),
        run_outcomes=(
            RunOutcomeRecord(
                variant,
                descriptor,
                "expected_refusal",
                7,
                "rheplicant.config.errors.ConfigError",
                "expected",
                "arbitrary_exception",
                False,
            ),
        ),
        deletions=(DeletionAuditRecord(variant, ("model", "old"), Origin("user")),),
        path_encodings=(PathEncoding("variant", "later", "n-6c61746572"),),
        artefacts=artefacts,
    )
    bundle = serialize_bundle(
        candidate_serialization_snapshot(enriched),
        status="ok",
        input_bytes=INPUT,
        resolved=(),
    )
    provenance = json.loads(bundle.provenance)
    diagnostics = json.loads(bundle.diagnostics)

    assert [row["layer"] for row in provenance["inputs"]] == [
        {"kind": "base", "name": None},
        {"kind": "variant", "name": "later"},
    ]
    assert [row["check"] for row in diagnostics["findings"]] == ["first", "second"]
    assert tuple(diagnostics["findings"][0]) == ("check", "message", "severity", "where")
    assert diagnostics["deferred_validations"][0]["checks"] == list(checks)
    assert diagnostics["runs"][0] == {
        "capture_scope": "arbitrary_exception",
        "exception_message": "expected",
        "exception_type": "rheplicant.config.errors.ConfigError",
        "index": 0,
        "is_dirt_error": False,
        "kind": "predict",
        "layer": {"kind": "variant", "name": "later"},
        "name": "predict",
        "phases": [],
        "status": "expected_refusal",
        "variant": None,
        "wall_time_ns": 7,
    }
    assert provenance["artefacts"] == diagnostics["artefacts"]
    assert provenance["artefacts"]["lock"]["written"] is True
    assert provenance["artefacts"]["journal"]["written"] is True
    assert provenance["artefacts"]["resolved_variants"][0]["reason"] == (
        "layer_not_complete"
    )
    assert "deletions" not in provenance and "deletions" not in diagnostics


def test_envelope_refuses_invalid_typed_rows_and_content_digest():
    _trace, initial = snapshot()
    candidate = candidate_serialization_snapshot(initial)
    with pytest.raises(ConfigError, match=r"inputs\[0\].*exact"):
        serialize_bundle(
            dataclasses.replace(candidate, inputs=(object(),)),
            status="ok",
            input_bytes=INPUT,
            resolved=(),
        )
    with pytest.raises(ConfigError, match="exact ArtefactTable"):
        serialize_bundle(
            dataclasses.replace(candidate, artefacts={}),
            status="ok",
            input_bytes=INPUT,
            resolved=(),
        )
    bad_metadata = ArtefactRecord(
        "provenance.json", True, 4, "0" * 64, "metadata_envelope"
    )
    with pytest.raises(ConfigError, match="metadata cannot claim bytes"):
        serialize_bundle(
            dataclasses.replace(
                candidate,
                artefacts=dataclasses.replace(
                    candidate.artefacts,
                    provenance=bad_metadata,
                ),
            ),
            status="ok",
            input_bytes=INPUT,
            resolved=(),
        )
    bad_input = ArtefactRecord("config.input.yaml", True, len(INPUT), "x" * 64, None)
    with pytest.raises(ConfigError, match="artefacts.input.*invalid"):
        serialize_bundle(
            dataclasses.replace(
                candidate,
                artefacts=dataclasses.replace(candidate.artefacts, input=bad_input),
            ),
            status="ok",
            input_bytes=INPUT,
            resolved=(),
        )
    wrong_claim = ArtefactRecord("config.input.yaml", True, len(INPUT), "0" * 64, None)
    with pytest.raises(ConfigError, match="contradicts bundle bytes"):
        serialize_bundle(
            dataclasses.replace(
                candidate,
                artefacts=dataclasses.replace(candidate.artefacts, input=wrong_claim),
            ),
            status="ok",
            input_bytes=INPUT,
            resolved=(),
        )
