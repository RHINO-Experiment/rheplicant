"""Shared ordered preparation for scientific configuration execution."""

from __future__ import annotations

import importlib
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from _rheplicant_bootstrap.audit import AuditTrace
from _rheplicant_bootstrap.capture import (
    CapturedInput,
    CaptureService,
    captured_input_json,
)
from _rheplicant_bootstrap.plugins import import_plugin, plugin_audit_row
from _rheplicant_bootstrap.prepare import PreparedConfig
from _rheplicant_bootstrap.runtime import (
    RuntimeSession,
    establish_runtime,
    runtime_audit_row,
)
from _rheplicant_bootstrap.types import LayerIdentity

TRUSTED_CODE_WARNING = (
    "warning: trusted plugin/python code may perform unobserved filesystem I/O\n"
)


@dataclass(slots=True)
class PreparedExecution:
    runtime: RuntimeSession
    orchestration: Any
    document: object
    capture: CaptureService
    trace: AuditTrace
    warning_written: bool
    _closed: bool = False

    def close(self) -> None:
        if not self._closed:
            self.capture.close()
            self._closed = True


def _capture_service(trace: AuditTrace) -> CaptureService:
    root = Path(tempfile.mkdtemp(prefix="rheplicant-capture-"))

    def on_verified(layer: LayerIdentity, row: CapturedInput) -> None:
        trace.record_input(layer, captured_input_json(row))

    return CaptureService(root, on_verified=on_verified)


def prepare_execution_environment(
    prepared: PreparedConfig,
    *,
    trace: AuditTrace,
    stderr: TextIO,
    warning_written: bool,
) -> PreparedExecution:
    session, orchestration = establish_runtime(
        prepared.process.runtime,
        import_main=lambda: importlib.import_module(
            "rheplicant.config.orchestration"
        ),
    )
    trace.record_runtime(runtime_audit_row(session))
    trace.boundary_completed("runtime")
    if prepared.process.plugins and not warning_written:
        stderr.write(TRUSTED_CODE_WARNING)
        stderr.flush()
        warning_written = True
    for name in prepared.process.plugins:
        record = import_plugin(name)
        trace.record_plugin(plugin_audit_row(record))
        session.verify(boundary=f"plugin {name!r}")
    trace.boundary_completed("plugins")
    capture = _capture_service(trace)
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
            stderr.write(TRUSTED_CODE_WARNING)
            stderr.flush()
            warning_written = True
        return PreparedExecution(
            session,
            orchestration,
            document,
            capture,
            trace,
            warning_written,
        )
    except BaseException:
        capture.close()
        raise
