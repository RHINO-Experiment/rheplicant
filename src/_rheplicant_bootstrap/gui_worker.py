"""Fresh-interpreter scientific worker for explicit GUI jobs.

This process isolates runtime establishment from the long-lived GUI API. It is
not a sandbox: trusted plugins, Python targets and server paths retain the
worker account's authority.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence

from _rheplicant_bootstrap.audit import AuditTrace
from _rheplicant_bootstrap.errors import ConfigError
from _rheplicant_bootstrap.execution_environment import (
    prepare_execution_environment,
)
from _rheplicant_bootstrap.output.manager import parse_output_grammar
from _rheplicant_bootstrap.prepare import PreparedConfig, prepare_config
from _rheplicant_bootstrap.presets import read_installed_preset
from _rheplicant_bootstrap.types import SourceInput

_FRAME_PREFIX = b"\x1eRHEPLICANT_GUI_JOB "


def _source(yaml_text: str) -> SourceInput:
    path = "/rheplicant-gui/session.yaml"
    return SourceInput(
        yaml_text.encode("utf-8", "strict"),
        path,
        path,
        path,
        "/rheplicant-gui",
        "embedded",
    )


def _prepared_config(yaml_text: str) -> PreparedConfig:
    return prepare_config(
        _source(yaml_text),
        preset_provider=read_installed_preset,
        parse_outputs=parse_output_grammar,
    )


def _finding(row: object, layer: str) -> dict[str, object]:
    return {
        "check": getattr(row, "check", ""),
        "severity": getattr(row, "severity", "report"),
        "where": getattr(row, "where", "document"),
        "message": str(getattr(row, "message", row)),
        "layer": layer,
    }


def _run_validation(yaml_text: str) -> dict[str, object]:
    prepared = _prepared_config(yaml_text)
    execution = prepare_execution_environment(
        prepared,
        trace=AuditTrace(),
        stderr=sys.stderr,
        warning_written=False,
    )
    try:
        findings = [
            _finding(row, layer.layer.prefix or "base")
            for layer in execution.document.layers
            for row in layer.configured.report.findings
        ]
        return {
            "findings": findings,
            "layers": len(execution.document.layers),
        }
    finally:
        execution.close()


def _write_frame(frame: Mapping[str, object]) -> None:
    sys.stdout.flush()
    encoded = json.dumps(frame, sort_keys=True).encode("utf-8", "strict")
    sys.stdout.buffer.write(_FRAME_PREFIX + encoded + b"\n")
    sys.stdout.buffer.flush()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "kind",
        choices=("validate", "preview_forward", "run", "compare", "benchmark"),
    )
    kind = parser.parse_args(argv).kind
    try:
        yaml_text = sys.stdin.buffer.read().decode("utf-8", "strict")
        if kind != "validate":
            raise ConfigError(f"GUI worker kind {kind!r} is not implemented yet")
        frame = {"status": "ok", "result": _run_validation(yaml_text)}
    except ConfigError as error:
        frame = {"status": "refused", "message": str(error)}
    except Exception as error:  # noqa: BLE001 -- one bounded terminal frame
        frame = {
            "status": "error",
            "exception_type": type(error).__name__,
            "message": str(error),
        }
    _write_frame(frame)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
