"""Explicit, content-bound GUI jobs over Plan 4's execution surfaces."""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from hashlib import sha256
from io import StringIO
from threading import RLock
from typing import Literal, cast
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
            output = getattr(error, "gui_output", None)
            exception_type = getattr(
                error, "job_exception_type", type(error).__name__
            )
            finished = replace(
                running,
                status="error",
                result=None if output is None else {"output": output},
                message=f"{exception_type}: {error}",
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


def _unsafe_formal_refusal(yaml_text: str) -> ConfigError | None:
    from rheplicant.gui.outputs import project_output_workflow

    try:
        projection = project_output_workflow(yaml_text)
    except Exception:  # noqa: BLE001 -- optional preflight cannot replace job truth
        return None
    if projection.state != "blocked_unsafe":
        return None
    output: dict[str, object] = {
        "state": projection.state,
        "state_message": projection.state_message,
    }
    if isinstance(projection.target_path, str):
        output["target_path"] = projection.target_path
    error = ConfigError(projection.state_message)
    error.gui_output = output
    return error


def _error_result(error: ConfigError) -> object | None:
    output = getattr(error, "gui_output", None)
    report = getattr(error, "report", None)
    findings = getattr(report, "findings", None)
    if findings is None:
        return None if output is None else {"output": output}
    result = {"findings": [_finding(row, "unknown") for row in findings]}
    if output is not None:
        result["output"] = output
    return result


def _failure_audit(stderr: str) -> dict[str, object] | None:
    from rheplicant.gui.outputs import output_summary_at_path

    for line in reversed(stderr.splitlines()):
        for prefix in ("refused audit: ", "error audit: "):
            if line.startswith(prefix):
                summary = output_summary_at_path(line.removeprefix(prefix))
                return summary if summary.get("marker_id") is not None else None
    return None


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


def _finding(row: object, layer: str) -> dict[str, object]:
    return {
        "check": getattr(row, "check", ""),
        "severity": getattr(row, "severity", "report"),
        "where": getattr(row, "where", "document"),
        "message": str(getattr(row, "message", row)),
        "layer": layer,
    }


_FRAME_PREFIX = b"\x1eRHEPLICANT_GUI_JOB "


def _last_worker_frame(stdout: bytes) -> Mapping[str, object]:
    at = stdout.rfind(_FRAME_PREFIX)
    if at < 0:
        raise RuntimeError("GUI scientific worker returned no result frame")
    payload = stdout[at + len(_FRAME_PREFIX) :].split(b"\n", 1)[0]
    frame = json.loads(payload.decode("utf-8", "strict"))
    if not isinstance(frame, Mapping) or frame.get("status") not in {
        "ok",
        "refused",
        "error",
    }:
        raise RuntimeError(
            "GUI scientific worker returned an invalid result frame"
        )
    status = frame["status"]
    if status == "ok" and not isinstance(frame.get("result"), Mapping):
        raise RuntimeError("GUI scientific worker result must be a mapping")
    if status == "refused" and not isinstance(frame.get("message"), str):
        raise RuntimeError(
            "GUI scientific worker refusal must carry a message"
        )
    if status == "error" and (
        not isinstance(frame.get("exception_type"), str)
        or not isinstance(frame.get("message"), str)
    ):
        raise RuntimeError(
            "GUI scientific worker error must carry a type and message"
        )
    return frame


def _run_isolated_job(
    kind: JobKind, yaml_text: str
) -> Mapping[str, object]:
    completed = subprocess.run(
        [sys.executable, "-m", "_rheplicant_bootstrap.gui_worker", kind],
        input=yaml_text.encode("utf-8", "strict"),
        capture_output=True,
        check=False,
    )
    stderr = completed.stderr.decode("utf-8", "replace")[-4000:]
    if completed.returncode != 0:
        raise RuntimeError(
            f"GUI scientific worker exited {completed.returncode}: {stderr}"
        )
    try:
        frame = _last_worker_frame(completed.stdout)
    except (RuntimeError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{error}; worker stderr: {stderr}") from None
    if frame["status"] == "refused":
        raise ConfigError(str(frame["message"]))
    if frame["status"] == "error":
        error = RuntimeError(str(frame["message"]))
        error.job_exception_type = str(frame["exception_type"])
        raise error
    return cast(Mapping[str, object], frame["result"])


def run_priced_validation(yaml_text: str) -> Mapping[str, object]:
    """Run priced validation in a fresh Plan 4 runtime environment."""
    return _run_isolated_job("validate", yaml_text)


def run_forward_preview(yaml_text: str) -> Mapping[str, object]:
    """Execute one synthesized Forward exit in a fresh Plan 4 runtime."""
    return _run_isolated_job("preview_forward", yaml_text)


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
    unsafe_refusal = _unsafe_formal_refusal(yaml_text)
    if unsafe_refusal is not None:
        raise unsafe_refusal
    if dispatcher is None:
        result = dict(_run_isolated_job(kind, yaml_text))
        exit_code = result.get("exit_code")
        stderr_text = str(result.get("stderr", ""))
    else:
        stdout = StringIO()
        stderr = StringIO()
        exit_code = dispatcher(
            "run",
            _source(yaml_text),
            stdout=stdout,
            stderr=stderr,
        )
        stderr_text = stderr.getvalue()
        result = {
            "exit_code": exit_code,
            "stdout": stdout.getvalue(),
            "stderr": stderr_text,
        }
    if exit_code == 2:
        error = ConfigError(stderr_text.strip() or f"{kind} job was refused.")
        output = _failure_audit(stderr_text)
        if output is not None:
            error.gui_output = output
        raise error
    if exit_code != 0:
        error = RuntimeError(stderr_text.strip() or f"{kind} job failed.")
        output = _failure_audit(stderr_text)
        if output is not None:
            error.gui_output = output
        raise error
    from rheplicant.gui.outputs import completed_output_summary

    result["output"] = completed_output_summary(yaml_text)
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
