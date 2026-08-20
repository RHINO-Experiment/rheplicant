"""Fresh-interpreter scientific worker for explicit GUI jobs.

This process isolates runtime establishment from the long-lived GUI API. It is
not a sandbox: trusted plugins, Python targets and server paths retain the
worker account's authority.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from io import StringIO

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


def _array_summary(value: object, *, include_values: bool) -> dict[str, object]:
    import numpy as np

    array = np.asarray(value)
    summary: dict[str, object] = {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }
    measured = np.abs(array) if np.iscomplexobj(array) else array
    if np.iscomplexobj(array):
        summary["statistic"] = "magnitude"
    if array.size:
        summary.update(
            minimum=float(np.nanmin(measured)),
            maximum=float(np.nanmax(measured)),
            mean=float(np.nanmean(measured)),
        )
    if include_values and array.ndim == 2 and not np.iscomplexobj(array):
        row_step = max(1, math.ceil(array.shape[0] / 64))
        column_step = max(1, math.ceil(array.shape[1] / 64))
        summary["values"] = array[::row_step, ::column_step].tolist()
    return summary


def _uniform_sky(configured: object) -> dict[str, float]:
    import jax.numpy as jnp
    import numpy as np

    context = configured.context
    state = configured.state
    n_freq = int(state.coords.freq.shape[0])
    found: dict[str, float] = {}
    for name, resource in context.resources.items():
        forward = getattr(resource, "forward", None)
        nside = getattr(resource, "nside", None)
        if callable(forward) and isinstance(nside, int) and nside > 0:
            sky = jnp.full((n_freq, 12 * nside * nside), 200.0)
            found[name] = float(np.asarray(forward(sky, state.coords)).mean())
    return found


def _bounded_preview_result(
    record: object,
    *,
    configured: object,
    adc: object,
) -> dict[str, object]:
    if record.status != "ok":
        if isinstance(record.error, BaseException):
            raise record.error
        raise RuntimeError("forward preview failed without a terminal error")
    result = record.results["preview-forward"].product
    data = getattr(result, "data", None)
    if data is None:
        raise ConfigError("forward preview produced no waterfall data.")
    aux = getattr(result, "aux", {})
    taps = (
        {
            str(name): _array_summary(value, include_values=False)
            for name, value in aux.items()
        }
        if isinstance(aux, Mapping)
        else {}
    )
    n_bits = adc.get("n_bits") if isinstance(adc, Mapping) else None
    saturated_fraction = None
    if isinstance(n_bits, int) and not isinstance(n_bits, bool) and n_bits > 0:
        import numpy as np

        array = np.asarray(data)
        saturated_fraction = float(np.mean(np.abs(array) >= 2 ** (n_bits - 1)))
    return {
        "waterfall": _array_summary(data, include_values=True),
        "taps": taps,
        "saturated_fraction": saturated_fraction,
        "uniform_sky_mean": _uniform_sky(configured),
    }


def _run_forward_preview(yaml_text: str) -> dict[str, object]:
    prepared = _prepared_config(yaml_text)
    execution = prepare_execution_environment(
        prepared,
        trace=AuditTrace(),
        stderr=sys.stderr,
        warning_written=False,
    )
    try:
        record = execution.orchestration.execute_prepared(
            execution.document,
            trace=execution.trace,
        )
        source_document = prepared.source.layered_document
        model = source_document.get("model", {})
        adc = model.get("adc") if isinstance(model, Mapping) else None
        configured = execution.document.layers[0].configured
        return _bounded_preview_result(
            record,
            configured=configured,
            adc=adc,
        )
    finally:
        execution.close()


def _run_formal(yaml_text: str) -> dict[str, object]:
    from _rheplicant_bootstrap.entry import dispatch_request

    stdout = StringIO()
    stderr = StringIO()
    exit_code = dispatch_request(
        "run",
        _source(yaml_text),
        stdout=stdout,
        stderr=stderr,
    )
    return {
        "exit_code": exit_code,
        "stdout": stdout.getvalue(),
        "stderr": stderr.getvalue(),
    }


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
        if kind == "validate":
            result = _run_validation(yaml_text)
        elif kind == "preview_forward":
            result = _run_forward_preview(yaml_text)
        else:
            result = _run_formal(yaml_text)
        frame = {"status": "ok", "result": result}
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
