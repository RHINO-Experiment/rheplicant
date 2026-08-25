"""Shared ordered preparation for scientific configuration execution."""

from __future__ import annotations

import importlib
import shutil
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
    """A capture service on a fresh temporary root.

    The directory exists before the object that owns it does, so anything
    raised in between leaks it: nothing else knows the path, and the process
    that could have cleaned it up is the one unwinding. The window is small
    -- one closure definition and one constructor -- and small is the reason
    to close it here rather than to argue about how it could be entered.
    """
    root = Path(tempfile.mkdtemp(prefix="rheplicant-capture-"))
    try:

        def on_verified(layer: LayerIdentity, row: CapturedInput) -> None:
            trace.record_input(layer, captured_input_json(row))

        return CaptureService(root, on_verified=on_verified)
    except BaseException:
        # ignore_errors: this path is already failing, and a cleanup that
        # raises here would replace the failure the caller needs to see --
        # the same trade `prepare_execution_environment` makes explicitly
        # below, where there is an exception object to hang a note on.
        shutil.rmtree(root, ignore_errors=True)
        raise


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
    except BaseException as error:
        # `close()` removes every captured file and then rmtree's the root,
        # and both can raise: `_remove` only forgives FileNotFoundError, and
        # rmtree forgives nothing. Called bare in an except block, a failure
        # there REPLACES the exception being handled, so a permissions
        # problem during cleanup would be reported in place of the real
        # failure -- the one the user has to act on.
        #
        # The note keeps both: the original propagates, and the cleanup
        # failure travels with it instead of vanishing, which is the half a
        # bare `suppress` would lose.
        try:
            capture.close()
        except BaseException as cleanup:  # noqa: BLE001 - reported, not handled
            error.add_note(
                f"capture cleanup also failed and was not the original fault: "
                f"{type(cleanup).__name__}: {cleanup}"
            )
        raise
