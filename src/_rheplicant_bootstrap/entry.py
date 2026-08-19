"""One ordered entry pipeline shared by the CLI and generated programs."""

from __future__ import annotations

import base64
import hashlib
import importlib
import os
import tempfile
import traceback
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Literal, TextIO, cast

from _rheplicant_bootstrap.audit import AuditTrace
from _rheplicant_bootstrap.audit.bundle import (
    candidate_serialization_snapshot,
    merge_bundle_files,
    serialize_bundle,
    terminal_reserialization_snapshot,
)
from _rheplicant_bootstrap.audit.resolved import (
    ResolvedArtefact,
    build_resolved_artefacts,
)
from _rheplicant_bootstrap.audit.software import collect_software
from _rheplicant_bootstrap.audit.types import (
    ArtefactMaterialization,
    ArtefactReason,
    ArtefactRecord,
    ArtefactTable,
    AuditSnapshot,
)
from _rheplicant_bootstrap.capture import (
    CapturedInput,
    CaptureService,
    captured_input_json,
)
from _rheplicant_bootstrap.errors import ConfigError
from _rheplicant_bootstrap.output.manager import (
    acquire_output_lease,
    close_output_lease,
    inspect_output_path,
    parse_output_grammar,
    resolve_output_request,
    verify_a34_under_lease,
    verify_publication_under_lease,
)
from _rheplicant_bootstrap.output.paths import journal_name, lock_name
from _rheplicant_bootstrap.output.platform import OutputPlatform, platform_adapter
from _rheplicant_bootstrap.output.transaction import (
    TransactionInterrupted,
    publish_failure,
    publish_success,
    recover_transaction,
    replace_staged_metadata,
    stage_bundle,
)
from _rheplicant_bootstrap.output.types import (
    OutputLease,
    OutputPathInspection,
    OutputRequest,
    ParsedOutputSection,
    PublicationLease,
    VerifiedOutputLease,
)
from _rheplicant_bootstrap.plugins import import_plugin, plugin_audit_row
from _rheplicant_bootstrap.prepare import PreparedConfig, prepare_config
from _rheplicant_bootstrap.presets import (
    PresetSnapshot,
    read_installed_preset,
    validate_preset_document,
)
from _rheplicant_bootstrap.runtime import establish_runtime, runtime_audit_row
from _rheplicant_bootstrap.types import SourceInput, Status
from _rheplicant_bootstrap.yaml import safe_load_document

_TRUSTED_CODE_WARNING = (
    "warning: trusted plugin/python code may perform unobserved filesystem I/O\n"
)


def _exception_row(error: BaseException) -> dict[str, str]:
    return {
        "exception_type": f"{type(error).__module__}.{type(error).__qualname__}",
        "message": str(error),
    }


def _record_error_once(trace: AuditTrace, error: BaseException) -> None:
    if trace.snapshot().error is None:
        trace.record_error(_exception_row(error))


def _bootstrap_row(prepared: PreparedConfig) -> dict[str, object]:
    manifest = prepared.source.bootstrap_manifest
    return {
        "protocol_version": manifest.protocol_version,
        "launch_mode": manifest.launch_mode,
        "input_sha256": manifest.input_sha256,
        "presets": tuple(row.request.name for row in manifest.presets),
        "source_name": manifest.source_name,
        "source_path": manifest.source_path,
        "source_realpath": manifest.source_realpath,
        "base_dir": manifest.base_dir,
    }


def _declared_run_names(document: Mapping[str, object]) -> tuple[str, ...]:
    """Project only names needed for the pre-JAX filesystem codec gate."""
    raw = document.get("runs", ())
    if isinstance(raw, Mapping):
        rows: Sequence[object] = (raw,)
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        rows = raw
    else:
        return ()
    names: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        kind = row.get("kind")
        name = row.get("name", kind)
        if isinstance(name, str):
            names.append(str.__str__(name))
    return tuple(names)


def _declared_variant_names(prepared: PreparedConfig) -> tuple[str, ...]:
    return tuple(
        cast(str, layer.name)
        for layer in prepared.layers
        if layer.kind == "variant"
    )


def _metadata(path: str, reason: ArtefactReason) -> ArtefactRecord:
    return ArtefactRecord(path, False, None, None, reason)


def _content(path: str, reason: ArtefactReason) -> ArtefactRecord:
    return ArtefactRecord(path, False, None, None, reason)


def _configure_artefacts(
    trace: AuditTrace,
    *,
    request: OutputRequest,
    variant_paths: Sequence[str],
) -> None:
    if request.target_path is None:
        raise ConfigError("run output request has no target path.")
    trace.configure_artefacts(
        ArtefactTable(
            marker=_metadata(
                ".rheplicant-results.json", "boundary_not_reached"
            ),
            lock=_metadata(
                lock_name(request.target_path), "transaction_not_reached"
            ),
            journal=_metadata(
                journal_name(request.target_path), "transaction_not_reached"
            ),
            input=_content("config.input.yaml", "boundary_not_reached"),
            resolved_base=_content(
                "config.resolved.yaml", "layer_not_complete"
            ),
            resolved_variants=tuple(
                _content(path, "layer_not_complete") for path in variant_paths
            ),
            provenance=_metadata("provenance.json", "boundary_not_reached"),
            diagnostics=_metadata("diagnostics.json", "boundary_not_reached"),
        )
    )


def _materialization_is_written(
    snapshot: AuditSnapshot, row: ArtefactMaterialization
) -> bool:
    table = snapshot.artefacts
    if row.slot == "resolved_variant":
        if row.variant_index is None:
            return False
        return table.resolved_variants[row.variant_index].written
    return cast(ArtefactRecord, getattr(table, row.slot)).written


def _record_materializations(
    trace: AuditTrace, rows: Sequence[ArtefactMaterialization]
) -> None:
    for row in rows:
        if not _materialization_is_written(trace.snapshot(), row):
            trace.record_artefact_materialized(row)


def _candidate(trace: AuditTrace) -> tuple[AuditSnapshot, bool]:
    snapshot = trace.snapshot()
    serializations = tuple(
        row for row in snapshot.completed_boundaries if row.stage == "serialization"
    )
    if not serializations:
        return candidate_serialization_snapshot(snapshot), False
    return terminal_reserialization_snapshot(snapshot), True


def _resolved(
    trace: AuditTrace,
    *,
    run_names: Sequence[str],
    variant_names: Sequence[str],
    component_limit: int,
) -> tuple[ResolvedArtefact, ...]:
    return tuple(
        build_resolved_artefacts(
            trace.snapshot().resolved_layers,
            run_names=run_names,
            variant_names=variant_names,
            component_limit=component_limit,
        ).files
    )


def _publish_transaction(
    authorization: VerifiedOutputLease | PublicationLease,
    *,
    publication: Literal["success", "refused", "error"],
    status: Status,
    trace: AuditTrace,
    input_bytes: bytes,
    run_names: Sequence[str],
    variant_names: Sequence[str],
    component_limit: int,
    platform: OutputPlatform,
    additional_files: Mapping[str, bytes] | None = None,
) -> str:
    candidate, already_serialized = _candidate(trace)
    resolved = _resolved(
        trace,
        run_names=run_names,
        variant_names=variant_names,
        component_limit=component_limit,
    )
    initial = serialize_bundle(
        candidate,
        status=status,
        input_bytes=input_bytes,
        resolved=resolved,
    )
    if additional_files is not None:
        initial = merge_bundle_files(initial, additional_files)
    try:
        handle, materialized = stage_bundle(
            authorization,
            initial,
            platform,
            publication=publication,
        )
        _record_materializations(trace, materialized)
        final_candidate, final_already_serialized = _candidate(trace)
        if final_already_serialized is not already_serialized:
            raise RuntimeError("serialization boundary changed during staging")
        final_resolved = _resolved(
            trace,
            run_names=run_names,
            variant_names=variant_names,
            component_limit=component_limit,
        )
        final = serialize_bundle(
            final_candidate,
            status=status,
            input_bytes=input_bytes,
            resolved=final_resolved,
        )
        if additional_files is not None:
            final = merge_bundle_files(final, additional_files)
        replace_staged_metadata(handle, final, platform)
        if not already_serialized:
            trace.boundary_completed("serialization")
        if tuple(trace.snapshot().completed_boundaries) != tuple(
            final_candidate.completed_boundaries
        ):
            raise RuntimeError("live audit boundaries disagree with final metadata")
        if publication == "success":
            return publish_success(handle, platform)
        return publish_failure(handle, platform)
    except TransactionInterrupted as error:
        _record_materializations(
            trace, error.state.unreported_materializations
        )
        raise


def _recover_publication(
    lease: OutputLease,
    platform: OutputPlatform,
    original: BaseException,
) -> PublicationLease | None:
    try:
        recover_transaction(lease, platform)
        return verify_publication_under_lease(lease, platform)
    except Exception as recovery_error:
        try:
            original.add_note(f"output recovery failed: {recovery_error}")
        except Exception:
            pass
        return None


def _report_failure_audit(
    stream: TextIO,
    *,
    status: Status,
    path: str,
) -> None:
    label = "refused" if status == "refused" else "error"
    stream.write(f"{label} audit: {path}\n")
    stream.flush()


def _publish_failure_once(
    original: BaseException,
    *,
    status: Literal["refused", "error"],
    publication: PublicationLease,
    lease: OutputLease,
    trace: AuditTrace,
    prepared: PreparedConfig,
    run_names: Sequence[str],
    variant_names: Sequence[str],
    platform: OutputPlatform,
    stderr: TextIO,
) -> None:
    """Publish one terminal sibling; recover it once and never recurse."""
    _record_error_once(trace, original)
    try:
        path = _publish_transaction(
            publication,
            publication=status,
            status=status,
            trace=trace,
            input_bytes=prepared.source.input_bytes,
            run_names=run_names,
            variant_names=variant_names,
            component_limit=publication.component_limit,
            platform=platform,
        )
    except Exception as transaction_error:
        if isinstance(transaction_error, TransactionInterrupted):
            _record_materializations(
                trace, transaction_error.state.unreported_materializations
            )
        _recover_publication(lease, platform, transaction_error)
        if transaction_error is not original:
            try:
                original.add_note(
                    f"terminal {status} audit failed: {transaction_error}"
                )
            except Exception:
                pass
        raise original from transaction_error
    _report_failure_audit(stderr, status=status, path=path)
    raise original


def _publish_error_after_transaction_failure(
    original: BaseException,
    *,
    lease: OutputLease,
    trace: AuditTrace,
    prepared: PreparedConfig,
    run_names: Sequence[str],
    variant_names: Sequence[str],
    platform: OutputPlatform,
    stderr: TextIO,
) -> None:
    """Recover a failed success transaction, then try one error sibling."""
    if isinstance(original, TransactionInterrupted):
        _record_materializations(
            trace, original.state.unreported_materializations
        )
    _record_error_once(trace, original)
    publication = _recover_publication(lease, platform, original)
    if publication is None:
        raise original
    _publish_failure_once(
        original,
        status="error",
        publication=publication,
        lease=lease,
        trace=trace,
        prepared=prepared,
        run_names=run_names,
        variant_names=variant_names,
        platform=platform,
        stderr=stderr,
    )


def _validate_explicit_output(
    inspection: OutputPathInspection,
    platform: OutputPlatform,
) -> None:
    """Perform the read-only portion of recovery/A34 for validate."""
    if inspection.recovery.requires_recovery:
        detail = inspection.recovery.reason or "transaction recovery is required"
        names = (inspection.target_name, *inspection.recovery.update_temp_names)
        raise ConfigError(f"ambiguous output recovery ({detail}); preserved: " + ", ".join(names))
    access = inspection.access
    if not access.reliable:
        raise ConfigError(access.reason or "cannot verify output access control.")
    if access.owner_uid != access.effective_uid:
        raise ConfigError("output parent has the wrong effective uid owner.")
    if access.mode & 0o022:
        raise ConfigError("output parent is group or other writable.")
    if not access.access_acl_is_trivial or not access.default_acl_is_trivial:
        raise ConfigError(access.reason or "output parent has non-trivial access control.")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    nearest_fd = os.open(inspection.nearest_existing_ancestor, flags)
    try:
        platform.verify_rename_noreplace_available(nearest_fd)
    finally:
        os.close(nearest_fd)
    if inspection.missing_components or not inspection.target.exists:
        return
    if not inspection.request.clobber:
        raise ConfigError("output target exists and outputs.clobber is false.")
    parent_fd = os.open(inspection.parent_path, flags)
    try:
        from _rheplicant_bootstrap.output.manager import _read_owned_marker

        _read_owned_marker(parent_fd, inspection.target_name)
    finally:
        os.close(parent_fd)


def _prepare_document(
    prepared: PreparedConfig,
    *,
    trace: AuditTrace,
    stderr: TextIO,
    warning_written: bool,
):
    session, orchestration = establish_runtime(
        prepared.process.runtime,
        import_main=lambda: importlib.import_module(
            "rheplicant.config.orchestration"
        ),
    )
    trace.record_runtime(runtime_audit_row(session))
    trace.boundary_completed("runtime")
    if prepared.process.plugins and not warning_written:
        stderr.write(_TRUSTED_CODE_WARNING)
        stderr.flush()
        warning_written = True
    for name in prepared.process.plugins:
        record = import_plugin(name)
        trace.record_plugin(plugin_audit_row(record))
        session.verify(boundary=f"plugin {name!r}")
    trace.boundary_completed("plugins")

    root = Path(tempfile.mkdtemp(prefix="rheplicant-capture-"))

    def on_verified(layer, row: CapturedInput) -> None:
        trace.record_input(layer, captured_input_json(row))

    capture = CaptureService(root, on_verified=on_verified)
    try:
        document = orchestration.prepare_document(
            prepared.source.layered_document,
            scope="all_layers",
            base_dir=prepared.source.base_dir,
            layers=prepared.layers,
            layer_origins=prepared.layer_origins,
            layer_deletions=prepared.layer_deletions,
            trace=trace,
            capture=capture,
        )
        if trace.snapshot().python_targets and not warning_written:
            stderr.write(_TRUSTED_CODE_WARNING)
            stderr.flush()
            warning_written = True
        return session, orchestration, document, capture, warning_written
    except BaseException:
        capture.close()
        raise


def dispatch_request(
    command: str,
    source: SourceInput,
    *,
    stdout: TextIO,
    stderr: TextIO,
    preset_provider: Callable[[str], PresetSnapshot] = read_installed_preset,
    platform: OutputPlatform | None = None,
) -> int:
    """Execute validate/run in the sole normative order."""
    if command not in ("validate", "run"):
        raise ConfigError(f"unknown dispatcher command {command!r}.")
    chosen_platform = platform_adapter() if platform is None else platform
    trace = AuditTrace()
    trace.record_software(collect_software())
    prepared = prepare_config(
        source,
        preset_provider=preset_provider,
        parse_outputs=parse_output_grammar,
        boundary_completed=trace.boundary_completed,
    )
    trace.record_bootstrap(_bootstrap_row(prepared))
    outputs = prepared.process.outputs
    if type(outputs) is not ParsedOutputSection:
        raise ConfigError("effective outputs are not a parsed output section.")
    request = resolve_output_request(outputs, source=source, command=command)
    run_names = _declared_run_names(prepared.source.layered_document)
    variant_names = _declared_variant_names(prepared)
    warning_written = False

    if command == "validate":
        if request.target_path is not None:
            inspection = inspect_output_path(request, chosen_platform)
            _validate_explicit_output(inspection, chosen_platform)
            encoded = build_resolved_artefacts(
                (),
                run_names=run_names,
                variant_names=variant_names,
                component_limit=inspection.component_limit,
            )
            for row in encoded.path_encodings:
                trace.record_path_encoding(
                    {
                        "kind": row.kind,
                        "document_name": row.document_name,
                        "encoded_name": row.encoded_name,
                    }
                )
        trace.boundary_completed("output_preflight")
        _session, _orchestration, _document, capture, warning_written = _prepare_document(
            prepared,
            trace=trace,
            stderr=stderr,
            warning_written=warning_written,
        )
        try:
            if outputs.stdout != "none":
                stdout.write(
                    f"configuration valid: base + {len(variant_names)} variants\n"
                )
                stdout.flush()
            return 0
        finally:
            capture.close()

    if request.target_path is None:
        raise ConfigError("run output request has no target path.")
    inspection = inspect_output_path(request, chosen_platform)
    encoded = build_resolved_artefacts(
        (),
        run_names=run_names,
        variant_names=variant_names,
        component_limit=inspection.component_limit,
    )
    lease: OutputLease | None = None
    publication: PublicationLease | None = None
    capture: CaptureService | None = None
    try:
        lease = acquire_output_lease(inspection, chosen_platform)
        variant_paths = tuple(
            f"variants/{row.encoded_name}/config.resolved.yaml"
            for row in encoded.path_encodings
            if row.kind == "variant"
        )
        _configure_artefacts(trace, request=request, variant_paths=variant_paths)
        for row in encoded.path_encodings:
            trace.record_path_encoding(
                {
                    "kind": row.kind,
                    "document_name": row.document_name,
                    "encoded_name": row.encoded_name,
                }
            )
        trace.record_artefact_materialized(
            ArtefactMaterialization("lock", None, lease.lock_name, None, None)
        )
        recover_transaction(lease, chosen_platform)
        publication = verify_publication_under_lease(lease, chosen_platform)
        try:
            verified = verify_a34_under_lease(publication, chosen_platform)
        except ConfigError as error:
            _publish_failure_once(
                error,
                status="refused",
                publication=publication,
                lease=lease,
                trace=trace,
                prepared=prepared,
                run_names=run_names,
                variant_names=variant_names,
                platform=chosen_platform,
                stderr=stderr,
            )
        trace.boundary_completed("output_preflight")
        try:
            _session, orchestration, document, capture, warning_written = _prepare_document(
                prepared,
                trace=trace,
                stderr=stderr,
                warning_written=warning_written,
            )
            execution = orchestration.execute_prepared(document, trace=trace)
            trace.boundary_completed("execution")
        except Exception as original:
            failure_status: Literal["refused", "error"] = (
                "refused" if isinstance(original, ConfigError) else "error"
            )
            _publish_failure_once(
                original,
                status=failure_status,
                publication=publication,
                lease=lease,
                trace=trace,
                prepared=prepared,
                run_names=run_names,
                variant_names=variant_names,
                platform=chosen_platform,
                stderr=stderr,
            )

        status: Status = execution.status
        terminal = execution.error
        if status != "ok":
            if terminal is None:
                terminal = RuntimeError("failed execution has no terminal error")
                status = "error"
            _publish_failure_once(
                terminal,
                status=status,
                publication=publication,
                lease=lease,
                trace=trace,
                prepared=prepared,
                run_names=run_names,
                variant_names=variant_names,
                platform=chosen_platform,
                stderr=stderr,
            )
        additional_files: dict[str, bytes] | None = None
        if request.products or request.report is not None:
            try:
                scientific = orchestration.build_product_bundle(
                    execution,
                    requests=request.products,
                    report=request.report,
                    component_limit=publication.component_limit,
                )
                additional_files = {
                    row.relative_path: row.payload for row in scientific.files
                }
                if len(additional_files) != len(scientific.files):
                    raise ConfigError("scientific product paths are duplicated.")
                if "products.json" in additional_files:
                    raise ConfigError("scientific product path 'products.json' is reserved.")
                additional_files["products.json"] = scientific.manifest
            except Exception as original:
                failure_status = (
                    "refused" if isinstance(original, ConfigError) else "error"
                )
                _publish_failure_once(
                    original,
                    status=failure_status,
                    publication=publication,
                    lease=lease,
                    trace=trace,
                    prepared=prepared,
                    run_names=run_names,
                    variant_names=variant_names,
                    platform=chosen_platform,
                    stderr=stderr,
                )
        try:
            path = _publish_transaction(
                verified,
                publication="success",
                status="ok",
                trace=trace,
                input_bytes=prepared.source.input_bytes,
                run_names=run_names,
                variant_names=variant_names,
                component_limit=publication.component_limit,
                platform=chosen_platform,
                additional_files=additional_files,
            )
        except Exception as transaction_error:
            _publish_error_after_transaction_failure(
                transaction_error,
                lease=lease,
                trace=trace,
                prepared=prepared,
                run_names=run_names,
                variant_names=variant_names,
                platform=chosen_platform,
                stderr=stderr,
            )
        if outputs.stdout != "none":
            stdout.write(f"configuration run complete: {path}\n")
            stdout.flush()
        return 0
    finally:
        if capture is not None:
            capture.close()
        if lease is not None:
            close_output_lease(lease)


def _decode_bytes(value: object, *, where: str) -> bytes:
    if type(value) is bytes:
        return value
    if type(value) is not str:
        raise ConfigError(f"{where} must be base64 text or exact bytes.")
    try:
        return base64.b64decode(value, validate=True)
    except Exception:
        raise ConfigError(f"{where} is not canonical base64.") from None


def _embedded_presets(rows: Sequence[Mapping[str, object]]) -> dict[str, PresetSnapshot]:
    snapshots: dict[str, PresetSnapshot] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ConfigError(f"embedded preset[{index}] must be a mapping.")
        allowed = {
            "expanded_nodes",
            "input_bytes",
            "input_bytes_b64",
            "name",
            "resource",
            "sha256",
        }
        if not set(row) <= allowed or not {
            "expanded_nodes",
            "name",
            "resource",
            "sha256",
        } <= set(row):
            raise ConfigError(f"embedded preset[{index}] has the wrong fields.")
        if ("input_bytes" in row) == ("input_bytes_b64" in row):
            raise ConfigError(
                f"embedded preset[{index}] requires exactly one byte field."
            )
        raw = row.get("input_bytes", row.get("input_bytes_b64"))
        payload = _decode_bytes(raw, where=f"embedded preset[{index}] bytes")
        name = row["name"]
        resource = row["resource"]
        digest = row["sha256"]
        expanded_nodes = row["expanded_nodes"]
        if type(name) is not str or type(resource) is not str or type(digest) is not str:
            raise ConfigError(f"embedded preset[{index}] text fields are invalid.")
        loaded = safe_load_document(payload, source_name=f"preset:{name}")
        document = validate_preset_document(name, loaded.value)
        if loaded.expanded_nodes != expanded_nodes:
            raise ConfigError(f"embedded preset {name!r} expanded-node count changed.")
        if hashlib.sha256(payload).hexdigest() != digest:
            raise ConfigError(f"embedded preset {name!r} digest does not match bytes.")
        snapshot = PresetSnapshot(
            name,
            resource,
            payload,
            digest,
            document,
            cast(int, expanded_nodes),
        )
        if name in snapshots:
            raise ConfigError(f"embedded preset {name!r} is duplicated.")
        snapshots[name] = snapshot
    return snapshots


def _render_exception(error: Exception, stderr: TextIO, *, traceback_error: bool) -> int:
    report = getattr(error, "report", None)
    first_refusal = None
    findings = getattr(report, "findings", ()) if report is not None else ()
    for finding in findings:
        severity = getattr(finding, "severity", "finding")
        where = getattr(finding, "where", "document")
        message = str(getattr(finding, "message", finding))
        stderr.write(f"{severity}: {where}: {message}\n")
        if first_refusal is None and severity == "refuse":
            first_refusal = message
    message = str(error)
    if message and message != first_refusal:
        stderr.write(message + "\n")
    if traceback_error:
        traceback.print_exception(type(error), error, error.__traceback__, file=stderr)
    stderr.flush()
    return 1 if traceback_error else 2


def run_embedded_config(
    *,
    source_path: str,
    source_realpath: str | None,
    source_name: str,
    base_dir: str,
    presets: Sequence[Mapping[str, object]],
    input_bytes_b64: str | None = None,
    input_bytes: bytes | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Rebuild immutable embedded snapshots and enter the ordinary run path."""
    import sys

    chosen_stdout = sys.stdout if stdout is None else stdout
    chosen_stderr = sys.stderr if stderr is None else stderr
    if (input_bytes is None) == (input_bytes_b64 is None):
        return _render_exception(
            ConfigError("embedded input requires exactly one byte field."),
            chosen_stderr,
            traceback_error=False,
        )
    try:
        payload = _decode_bytes(
            input_bytes if input_bytes is not None else input_bytes_b64,
            where="embedded input",
        )
        snapshots = _embedded_presets(presets)

        def provider(name: str) -> PresetSnapshot:
            try:
                return snapshots[name]
            except KeyError:
                raise ConfigError(
                    f"embedded source does not contain preset {name!r}."
                ) from None

        source = SourceInput(
            payload,
            source_path,
            source_realpath,
            source_name,
            base_dir,
            "embedded",
        )
        return dispatch_request(
            "run",
            source,
            stdout=chosen_stdout,
            stderr=chosen_stderr,
            preset_provider=provider,
        )
    except ConfigError as error:
        return _render_exception(error, chosen_stderr, traceback_error=False)
    except Exception as error:
        return _render_exception(error, chosen_stderr, traceback_error=True)


__all__ = ["dispatch_request", "run_embedded_config"]
