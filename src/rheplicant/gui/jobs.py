"""Explicit, content-bound GUI jobs over Plan 4's execution surfaces."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from hashlib import sha256
from io import StringIO
from threading import RLock
from typing import Literal
from uuid import uuid4

import yaml

from _rheplicant_bootstrap.errors import ConfigError
from _rheplicant_bootstrap.types import SourceInput
from _rheplicant_bootstrap.yaml import safe_load_document

JobKind = Literal["validate", "preview_forward", "run", "compare", "benchmark"]
JobStatus = Literal["queued", "running", "succeeded", "refused", "error"]
JobRunner = Callable[[JobKind, str], Mapping[str, object]]


@dataclass(frozen=True, slots=True)
class JobRecord:
    """One immutable job state bound to the exact submitted YAML bytes."""

    job_id: str
    session_id: str
    kind: JobKind
    revision: int
    yaml_digest: str
    status: JobStatus
    result: object | None = None
    message: str | None = None


@dataclass(frozen=True, slots=True)
class JobProjection:
    """A job plus its staleness relative to the session's current YAML."""

    job_id: str
    session_id: str
    kind: JobKind
    revision: int
    yaml_digest: str
    status: JobStatus
    result: object | None
    message: str | None
    stale: bool


def yaml_digest(yaml_text: str) -> str:
    """Hash the exact authoritative YAML bytes."""
    return sha256(yaml_text.encode("utf-8", "strict")).hexdigest()


class JobStore:
    """Thread-safe in-memory lifecycle storage for explicit GUI jobs."""

    def __init__(self, *, id_factory: Callable[[], str] | None = None) -> None:
        self._jobs: dict[str, JobRecord] = {}
        self._yaml: dict[str, str] = {}
        self._lock = RLock()
        self._id_factory = id_factory or (lambda: uuid4().hex)

    def submit(
        self,
        session_id: str,
        kind: JobKind,
        revision: int,
        yaml_text: str,
    ) -> JobRecord:
        job_id = self._id_factory()
        if not isinstance(job_id, str) or not job_id:
            raise RuntimeError("job id factory returned no usable id")
        row = JobRecord(
            job_id,
            session_id,
            kind,
            revision,
            yaml_digest(yaml_text),
            "queued",
        )
        with self._lock:
            if job_id in self._jobs:
                raise RuntimeError(f"job id {job_id!r} was generated twice")
            self._jobs[job_id] = row
            self._yaml[job_id] = yaml_text
        return row

    def get(self, job_id: str) -> JobRecord:
        with self._lock:
            try:
                return self._jobs[job_id]
            except KeyError:
                raise KeyError(job_id) from None

    def run(self, job_id: str, runner: JobRunner) -> None:
        with self._lock:
            current = self.get(job_id)
            if current.status != "queued":
                raise RuntimeError(f"job {job_id!r} is already {current.status}")
            running = replace(current, status="running")
            self._jobs[job_id] = running
            source = self._yaml[job_id]
        try:
            result = runner(running.kind, source)
        except ConfigError as error:
            finished = replace(
                running,
                status="refused",
                result=_error_result(error),
                message=str(error),
            )
        except Exception as error:  # noqa: BLE001 -- the job records terminal errors
            finished = replace(
                running,
                status="error",
                message=f"{type(error).__name__}: {error}",
            )
        else:
            finished = replace(running, status="succeeded", result=_plain(result))
        with self._lock:
            self._jobs[job_id] = finished
            self._yaml.pop(job_id, None)

    def project(self, session_id: str, current_digest: str) -> tuple[JobProjection, ...]:
        with self._lock:
            rows = tuple(row for row in self._jobs.values() if row.session_id == session_id)
        return tuple(
            JobProjection(**dataclasses.asdict(row), stale=row.yaml_digest != current_digest)
            for row in rows
        )


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_plain(item) for item in value]
    return value


def _error_result(error: ConfigError) -> object | None:
    report = getattr(error, "report", None)
    findings = getattr(report, "findings", None)
    if findings is None:
        return None
    return {"findings": [_finding(row, "unknown") for row in findings]}


def _document(yaml_text: str) -> dict[str, object]:
    loaded = safe_load_document(
        yaml_text.encode("utf-8", "strict"),
        source_name="GUI job document",
    ).value
    if not isinstance(loaded, Mapping):
        raise ConfigError("GUI job document root must be a mapping.")
    return dict(_plain(loaded))


def _declared_kinds(yaml_text: str) -> tuple[str, ...]:
    runs = _document(yaml_text).get("runs", ())
    if isinstance(runs, Mapping):
        runs = (runs,)
    if isinstance(runs, str | bytes) or not isinstance(runs, Sequence):
        return ()
    return tuple(
        kind
        for row in runs
        if isinstance(row, Mapping)
        for kind in (row.get("kind"),)
        if isinstance(kind, str)
    )


def forward_preview_document(yaml_text: str) -> str:
    """Create a detached one-forward schedule; fitting exits cannot survive."""
    document = _document(yaml_text)
    document["runs"] = [{"name": "preview-forward", "kind": "forward"}]
    return yaml.safe_dump(document, allow_unicode=True, sort_keys=False)


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


def _prepared_mapping(yaml_text: str) -> tuple[dict[str, object], str]:
    from _rheplicant_bootstrap.output.manager import parse_output_grammar
    from _rheplicant_bootstrap.prepare import prepare_config
    from _rheplicant_bootstrap.presets import read_installed_preset

    prepared = prepare_config(
        _source(yaml_text),
        preset_provider=read_installed_preset,
        parse_outputs=parse_output_grammar,
    )
    return dict(_plain(prepared.source.layered_document)), prepared.source.base_dir


def _finding(row: object, layer: str) -> dict[str, object]:
    return {
        "check": getattr(row, "check", ""),
        "severity": getattr(row, "severity", "report"),
        "where": getattr(row, "where", "document"),
        "message": str(getattr(row, "message", row)),
        "layer": layer,
    }


def run_priced_validation(yaml_text: str) -> Mapping[str, object]:
    """Run the public Plan 4 orchestration through all priced boundaries."""
    from rheplicant.config.orchestration import prepare_document

    document, base_dir = _prepared_mapping(yaml_text)
    prepared = prepare_document(document, scope="all_layers", base_dir=base_dir)
    findings = [
        _finding(row, layer.layer.prefix or "base")
        for layer in prepared.layers
        for row in layer.configured.report.findings
    ]
    return {"findings": findings, "layers": len(prepared.layers)}


def _array_summary(value: object, *, include_values: bool) -> dict[str, object]:
    import numpy as np

    array = np.asarray(value)
    summary: dict[str, object] = {"shape": list(array.shape), "dtype": str(array.dtype)}
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


def run_forward_preview(yaml_text: str) -> Mapping[str, object]:
    """Execute one synthesized forward exit and return bounded render data."""
    from rheplicant.config.orchestration import execute_prepared, prepare_document

    document, base_dir = _prepared_mapping(yaml_text)
    prepared = prepare_document(document, scope="all_layers", base_dir=base_dir)
    record = execute_prepared(prepared)
    if record.status != "ok":
        if isinstance(record.error, BaseException):
            raise record.error
        raise RuntimeError("forward preview failed without a terminal error")
    result = record.results["preview-forward"].product
    data = getattr(result, "data", None)
    if data is None:
        raise ConfigError("forward preview produced no waterfall data.")
    aux = getattr(result, "aux", {})
    taps = {
        str(name): _array_summary(value, include_values=False)
        for name, value in aux.items()
    } if isinstance(aux, Mapping) else {}
    model = document.get("model", {})
    adc = model.get("adc") if isinstance(model, Mapping) else None
    n_bits = adc.get("n_bits") if isinstance(adc, Mapping) else None
    saturated_fraction = None
    if isinstance(n_bits, int) and not isinstance(n_bits, bool) and n_bits > 0:
        import numpy as np

        array = np.asarray(data)
        saturated_fraction = float(np.mean(np.abs(array) >= 2 ** (n_bits - 1)))
    configured = prepared.layers[0].configured
    return {
        "waterfall": _array_summary(data, include_values=True),
        "taps": taps,
        "saturated_fraction": saturated_fraction,
        "uniform_sky_mean": _uniform_sky(configured),
    }


def execute_job(
    kind: JobKind,
    yaml_text: str,
    *,
    validator: Callable[[str], Mapping[str, object]] = run_priced_validation,
    forwarder: Callable[[str], Mapping[str, object]] = run_forward_preview,
    dispatcher: Callable[..., int] | None = None,
) -> Mapping[str, object]:
    """Execute one explicit action without conflating fitting with preview."""
    if kind == "validate":
        return validator(yaml_text)
    if kind == "preview_forward":
        return forwarder(forward_preview_document(yaml_text))
    if kind not in ("run", "compare", "benchmark"):
        raise ConfigError(f"unknown GUI job kind {kind!r}.")
    if kind != "run" and kind not in _declared_kinds(yaml_text):
        raise ConfigError(f"The document declares no {kind!r} exit to run.")
    if dispatcher is None:
        from _rheplicant_bootstrap.entry import dispatch_request

        dispatcher = dispatch_request
    stdout = StringIO()
    stderr = StringIO()
    exit_code = dispatcher("run", _source(yaml_text), stdout=stdout, stderr=stderr)
    result = {
        "exit_code": exit_code,
        "stdout": stdout.getvalue(),
        "stderr": stderr.getvalue(),
    }
    if exit_code == 2:
        raise ConfigError(stderr.getvalue().strip() or f"{kind} job was refused.")
    if exit_code != 0:
        raise RuntimeError(stderr.getvalue().strip() or f"{kind} job failed.")
    return result


__all__ = [
    "JobKind",
    "JobProjection",
    "JobRecord",
    "JobRunner",
    "JobStatus",
    "JobStore",
    "execute_job",
    "forward_preview_document",
    "run_forward_preview",
    "run_priced_validation",
    "yaml_digest",
]
