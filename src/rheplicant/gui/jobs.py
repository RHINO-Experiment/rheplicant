"""Explicit, content-bound GUI jobs over Plan 4's execution surfaces."""

from __future__ import annotations

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
from _rheplicant_bootstrap.gui_child import (
    DRAIN_SECONDS as _DRAIN_SECONDS,
)
from _rheplicant_bootstrap.gui_child import (
    FrameScan as _FrameScan,
)
from _rheplicant_bootstrap.gui_child import (
    StreamTail as _StreamTail,
)
from _rheplicant_bootstrap.gui_child import (
    drained_run as _drained_run,
)
from _rheplicant_bootstrap.gui_limits import (
    MAX_CHILD_STREAM_BYTES,
    MAX_FRAME_BYTES,
    MAX_RETAINED_JOBS,
    MAX_STREAM_BYTES,
    MAX_WORKER_SECONDS,
    bounded_findings,
    bounded_result,
    bounded_stream_bytes,
    bounded_stream_text,
    bounded_text,
    bounded_worker_result,
)
from _rheplicant_bootstrap.types import SourceInput
from _rheplicant_bootstrap.yaml import safe_load_document

JobKind = Literal["validate", "preview_forward", "run", "compare", "benchmark"]
JobStatus = Literal["queued", "running", "succeeded", "refused", "error"]
JobRunner = Callable[[JobKind, str], Mapping[str, object]]

# A job that has not reached a terminal status still owns its action.  Once it
# is terminal, an identical re-run is a new question and must be allowed.
_ACTIVE_STATUSES: frozenset[str] = frozenset({"queued", "running"})


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
    """Thread-safe in-memory lifecycle storage for explicit GUI jobs.

    Bounded storage.  ``MAX_RESULT_BYTES`` bounds one result and bounded
    nothing about a process that kept every result it had ever produced: what
    a server costs is the product of that budget and a count, and until
    ``MAX_RETAINED_JOBS`` there was no second factor.  Finished jobs are
    therefore retired here, oldest first, and only finished ones -- a
    ``queued`` or ``running`` record owns its action, is read by
    ``_active_duplicate``, and has a result nobody has collected yet.

    Bounding the mapping also bounds the two scans that walk it: the duplicate
    check on every submit, and the per-session projection on every poll.  Both
    used to grow without limit for the life of the process, under the lock.

    Retiring a job is visible from outside, and deliberately so: an audit
    link the GUI handed out names a ``job_id``, and once that job is retired
    the link cannot resolve.  :meth:`retired` is what lets the caller answer
    "this existed and is no longer served" rather than "this never existed",
    which are different claims and only one of them is true.  The memory of
    retirement is itself bounded -- ``MAX_RETAINED_JOBS`` ids, the same window
    again -- so a link older than two full windows degrades to the ordinary
    missing-job answer, which is still bounded and still honest.
    """

    def __init__(self, *, id_factory: Callable[[], str] | None = None) -> None:
        self._jobs: dict[str, JobRecord] = {}
        self._yaml: dict[str, str] = {}
        self._retired: dict[str, None] = {}
        self._lock = RLock()
        self._id_factory = id_factory or (lambda: uuid4().hex)

    def submit(
        self,
        session_id: str,
        kind: JobKind,
        revision: int,
        yaml_text: str,
    ) -> JobRecord:
        """Record one queued job, refusing a live twin of the same submission.

        Suppression is per ``session_id`` by construction: two sessions
        holding byte-identical YAML may still both launch a run against the
        same target, because a session owns its own actions and nothing else.
        Deciding who may write one output directory is the output lease
        layer's invariant, and this store must not be read as enforcing it.
        """
        # The id factory is caller-supplied, so it runs before the lock and
        # never inside the store's critical section.
        job_id = self._id_factory()
        if not isinstance(job_id, str) or not job_id:
            raise RuntimeError("job id factory returned no usable id")
        digest = yaml_digest(yaml_text)
        row = JobRecord(
            job_id,
            session_id,
            kind,
            revision,
            digest,
            "queued",
        )
        with self._lock:
            if job_id in self._jobs:
                raise RuntimeError(f"job id {job_id!r} was generated twice")
            active = self._active_duplicate(session_id, kind, revision, digest)
            if active is not None:
                raise ConfigError(
                    f"A {kind!r} job for revision {revision} of document "
                    f"{digest[:12]} is already {active.status}; wait for it to "
                    f"finish before submitting it again."
                )
            self._jobs[job_id] = row
            self._yaml[job_id] = yaml_text
            self._evict()
        return row

    def _evict(self) -> None:
        """Retire finished jobs, oldest first, until the store fits its bound.

        The caller holds ``_lock``; this is not a public entry point for the
        same reason ``_active_duplicate`` is not, and for a sharper one: it
        deletes.  Insertion order IS submission order for a ``dict``, so the
        front of the mapping is the oldest job and the scan needs no clock.

        Active rows are skipped rather than counted out, so a store whose live
        jobs alone exceed the bound stays above it and nothing running is lost.
        That is the deliberate asymmetry ``MAX_RETAINED_JOB_BYTES`` describes:
        the budget is for results that have been produced, and an active job
        has not produced one.
        """
        over = len(self._jobs) - MAX_RETAINED_JOBS
        if over <= 0:
            return
        for job_id, row in list(self._jobs.items()):
            if over <= 0:
                return
            if row.status in _ACTIVE_STATUSES:
                continue
            del self._jobs[job_id]
            self._yaml.pop(job_id, None)
            self._remember_retired(job_id)
            over -= 1

    def _remember_retired(self, job_id: str) -> None:
        """Keep one retired id for as long as one more window of jobs.

        Bounded, because a set of ids that grew for the life of the process
        would be the very defect this eviction exists to close, wearing a
        smaller hat.
        """
        self._retired[job_id] = None
        while len(self._retired) > MAX_RETAINED_JOBS:
            del self._retired[next(iter(self._retired))]

    def retired(self, job_id: str) -> bool:
        """Whether this store issued ``job_id`` and has since retired it.

        ``False`` means only that this store cannot say otherwise: an id it
        never issued and an id it has forgotten answer alike, which is why the
        caller's refusal for ``False`` must be the one that claims least.
        """
        with self._lock:
            return job_id in self._retired

    def _active_duplicate(
        self,
        session_id: str,
        kind: JobKind,
        revision: int,
        digest: str,
    ) -> JobRecord | None:
        """Find a live twin of one submission; the caller holds ``_lock``.

        This is deliberately not a public entry point.  Reading it outside the
        lock that inserts would make it advice rather than an invariant: two
        submissions could both read "no twin" and both insert.
        """
        for row in self._jobs.values():
            if (
                row.status in _ACTIVE_STATUSES
                and row.session_id == session_id
                and row.kind == kind
                and row.revision == revision
                and row.yaml_digest == digest
            ):
                return row
        return None

    def get(self, job_id: str) -> JobRecord:
        with self._lock:
            try:
                return self._jobs[job_id]
            except KeyError:
                raise KeyError(job_id) from None

    def run(self, job_id: str, runner: JobRunner) -> None:
        """Run one queued job and write exactly one terminal record for it.

        Every exit from here is terminal, including the exits that are
        themselves failures.  A record left ``running`` is not cosmetic:
        ``_active_duplicate`` reads it, nothing cancels or deletes a job, so
        one stranded record refuses every identical resubmission for the life
        of the process.  Building the terminal record is therefore inside the
        guarantee, not after it -- bounding a refusal's findings can raise,
        and so can a ``BaseException`` from the runner, which is re-raised
        here with the record already terminal.
        """
        with self._lock:
            # Read the row directly rather than through ``get``: this holds
            # ``_lock`` already, and only its being an ``RLock`` makes the
            # public entry point safe from here -- an invariant that should
            # not depend on a lock's flavour.
            try:
                current = self._jobs[job_id]
            except KeyError:
                raise KeyError(job_id) from None
            if current.status != "queued":
                raise RuntimeError(f"job {job_id!r} is already {current.status}")
            running = replace(current, status="running")
            self._jobs[job_id] = running
            source = self._yaml[job_id]
        finished: JobRecord | None = None
        try:
            try:
                result = runner(running.kind, source)
            except ConfigError as error:
                finished = replace(
                    running,
                    status="refused",
                    result=_error_result(error),
                    message=bounded_text(error),
                )
            except Exception as error:  # noqa: BLE001 -- the job records terminal errors
                output = getattr(error, "gui_output", None)
                exception_type = getattr(
                    error, "job_exception_type", type(error).__name__
                )
                # Each part is bounded before the join, so no unbounded string
                # is ever built here, and the join is bounded again after it.
                finished = replace(
                    running,
                    status="error",
                    result=(
                        None if output is None else {"output": bounded_result(output)}
                    ),
                    message=bounded_text(
                        f"{bounded_text(exception_type)}: {bounded_text(error)}"
                    ),
                )
            else:
                finished = replace(
                    running, status="succeeded", result=bounded_result(result)
                )
        finally:
            if finished is None:
                finished = _stranded(running, sys.exc_info()[1])
            with self._lock:
                self._jobs[job_id] = finished
                self._yaml.pop(job_id, None)
                # This row only became evictable on this line, so the store
                # can be over its bound with nothing that was retirable when
                # the row was inserted.
                self._evict()

    def abandon(self, job_id: str, failure: BaseException | None) -> None:
        """Write a terminal record for a queued job that will never run.

        ``submit`` inserts the row and the caller registers what runs it as a
        separate statement, so everything between the two can fail.  ``queued``
        is one of ``_ACTIVE_STATUSES`` and no route cancels or deletes a job,
        so a row that never reaches :meth:`run` refuses every identical
        resubmission for the life of the process -- the same defect ``run``
        already closes for itself, on the half only the caller can close.

        A row that has already left ``queued`` is left exactly as it stands:
        this supplies a missing outcome and never overwrites a recorded one.
        Unknown ids are silently accepted for the same reason -- the caller is
        unwinding, and a second failure here would strand what it came to free.
        """
        with self._lock:
            current = self._jobs.get(job_id)
            if current is None or current.status != "queued":
                return
            self._jobs[job_id] = _terminal(
                current, "the job was never started", failure
            )
            self._yaml.pop(job_id, None)
            self._evict()

    def project(self, session_id: str, current_digest: str) -> tuple[JobProjection, ...]:
        """Project this session's jobs, sharing each result rather than copying it.

        ``dataclasses.asdict`` deep-copies every value it walks, so building a
        projection through it copied each stored result in full -- on every
        poll, and ``useJobPolling`` polls continuously while any job is active.
        Sharing is safe because there is nothing to protect: ``JobRecord`` is
        frozen, a terminal record is replaced rather than mutated, and the
        result it carries is a plain structure ``bounded_result`` built once
        and nothing writes to afterwards.
        """
        with self._lock:
            rows = tuple(row for row in self._jobs.values() if row.session_id == session_id)
        return tuple(
            JobProjection(
                row.job_id,
                row.session_id,
                row.kind,
                row.revision,
                row.yaml_digest,
                row.status,
                row.result,
                row.message,
                row.yaml_digest != current_digest,
            )
            for row in rows
        )


def projection_body(row: JobProjection) -> dict[str, object]:
    """One JSON-ready mapping for a projected job, built rather than copied.

    The poll path used to deep-copy twice: once inside :meth:`JobStore.project`
    and once more here, where ``dataclasses.asdict`` walked the projection it
    had just been handed.  Naming the fields costs a dict per job and copies
    nothing, and the key order is the projection's own declaration order, so
    the body is byte-for-byte what ``asdict`` produced.
    """
    return {
        "job_id": row.job_id,
        "session_id": row.session_id,
        "kind": row.kind,
        "revision": row.revision,
        "yaml_digest": row.yaml_digest,
        "status": row.status,
        "result": row.result,
        "message": row.message,
        "stale": row.stale,
    }


def _stranded(running: JobRecord, failure: BaseException | None) -> JobRecord:
    """One terminal record for a job whose own failure handling failed."""
    return _terminal(running, "the job recorded no terminal result", failure)


def _terminal(
    row: JobRecord, summary: str, failure: BaseException | None
) -> JobRecord:
    """One terminal record for a job that will never produce its own.

    Describing the failure must not become a second way to strand the job, so
    every step here has a constant answer to fall back on.
    """
    try:
        detail = (
            "no exception was recorded"
            if failure is None
            else f"{type(failure).__name__}: {bounded_text(failure)}"
        )
    except Exception:  # noqa: BLE001 -- a message never outranks a terminal record
        detail = "the failure could not be described"
    return replace(
        row,
        status="error",
        result=None,
        message=bounded_text(f"{summary}: {detail}"),
    )


def _plain(value: object) -> object:
    """Normalise a parsed document. Documents are re-serialised, not reported,
    so this stays unbounded; job results go through ``bounded_result``."""
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
        return None if output is None else {"output": bounded_result(output)}
    result: dict[str, object] = {
        "findings": bounded_findings([_finding(row, "unknown") for row in findings])
    }
    if output is not None:
        result["output"] = bounded_result(output)
    return result


_AUDIT_PREFIXES = ("refused audit: ", "error audit: ")


def _audit_path(stderr: str) -> str | None:
    """Read the published bundle's path out of one *whole* stream: a FALLBACK.

    ``entry`` writes this line when it publishes the failure sibling, before
    the failure unwinds and prints itself, so the line sits at the head of a
    stream whose tail is what any bounded excerpt keeps.  Scanning an excerpt
    for it loses the ``/artifacts/`` links exactly when a run failed loudly.

    This is in-band signalling and it is not preferred anywhere it can be
    avoided.  A stream is written by whoever holds the descriptor, the rule
    here is that the LAST match wins, and what wins becomes a directory this
    long-lived parent then opens and serves artefacts out of.  A plugin that
    prints ``error audit: /somewhere/else`` gets to choose that directory.

    So the worker's framed ``failure_audit`` field is preferred wherever there
    is a frame: the worker reads its OWN whole stream, inside its own process,
    before anything else can have written to it, and the value crosses as a
    field of a result frame rather than as a line anybody may print.  This
    scan answers only where no frame exists at all -- the injected-dispatcher
    path, which runs in-process and has no worker to have framed anything --
    and as a last resort where the frame carried no field.
    """
    for line in reversed(stderr.splitlines()):
        for prefix in _AUDIT_PREFIXES:
            if line.startswith(prefix):
                return line.removeprefix(prefix)
    return None


def _failure_audit(path: str | None) -> dict[str, object] | None:
    """Project the published failure bundle at ``path``, if it is one."""
    if path is None:
        return None
    from rheplicant.gui.outputs import output_summary_at_path

    summary = output_summary_at_path(path)
    return summary if summary.get("marker_id") is not None else None


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


def _worker_frame(scan: _FrameScan) -> Mapping[str, object]:
    if scan.oversized:
        raise RuntimeError(
            "GUI scientific worker result frame is larger than the "
            f"{MAX_FRAME_BYTES}-byte limit"
        )
    if scan.payload is None:
        raise RuntimeError("GUI scientific worker returned no result frame")
    frame = json.loads(scan.payload.decode("utf-8", "strict"))
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


def _child_stream(
    answered: bytes | None, tail: _StreamTail, limit: int
) -> bytes:
    """Take one child stream's bounded tail without ever holding all of it.

    ``subprocess.run`` reports ``None`` for a stream it redirected, which is
    the drained path taken here; the parent's peak memory is ``limit`` rather
    than whatever the child chose to print.  A substituted runner that answers
    with the bytes itself is still honoured, bounded to the same tail.
    """
    return tail.tail() if answered is None else answered[-limit:]


def _child_frames(answered: bytes | None, scan: _FrameScan) -> _FrameScan:
    """Scan a substituted runner's answer the way a live stream is scanned."""
    if answered is None:
        return scan
    replacement = _FrameScan()
    replacement.feed(answered)
    replacement.close()
    return replacement


def _run_isolated_job(
    kind: JobKind, yaml_text: str
) -> Mapping[str, object]:
    frames = _FrameScan()
    errors = _StreamTail(limit=MAX_STREAM_BYTES)
    try:
        completed = _drained_run(
            [sys.executable, "-m", "_rheplicant_bootstrap.gui_worker", kind],
            input_bytes=yaml_text.encode("utf-8", "strict"),
            stdout=frames,
            stderr=errors,
            # Read from this module at call time rather than closed over, so
            # both bounds stay one constant that a caller can still shorten.
            timeout=MAX_WORKER_SECONDS,
            drain_seconds=_DRAIN_SECONDS,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            "GUI scientific worker did not finish within "
            f"{MAX_WORKER_SECONDS} seconds and was ended"
        ) from None
    stderr = bounded_stream_bytes(
        _child_stream(completed.stderr, errors, MAX_STREAM_BYTES)
    )
    for name, sink in (("stdout", frames), ("stderr", errors)):
        if sink.overflowed:
            raise RuntimeError(
                bounded_text(
                    f"GUI scientific worker wrote more than "
                    f"{MAX_CHILD_STREAM_BYTES} bytes to {name}: {stderr}"
                )
            )
        if sink.read_error is not None:
            raise RuntimeError(
                bounded_text(
                    f"GUI scientific worker {name} could not be read: "
                    f"{sink.read_error}"
                )
            )
    if completed.returncode != 0:
        raise RuntimeError(
            f"GUI scientific worker exited {completed.returncode}: {stderr}"
        )
    try:
        frame = _worker_frame(_child_frames(completed.stdout, frames))
    except (RuntimeError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"{bounded_text(error)}; worker stderr: {stderr}"
        ) from None
    if frame["status"] == "refused":
        raise ConfigError(bounded_text(frame["message"]))
    if frame["status"] == "error":
        error = RuntimeError(bounded_text(frame["message"]))
        error.job_exception_type = bounded_text(frame["exception_type"])
        raise error
    # The worker bounded this before it framed it, and the job store bounds
    # whatever a runner returns.  Bounding it twice is what rewrites the count
    # in a truncation notice and escapes the reserved key out of the channel,
    # so the result is marked as already bounded rather than bounded again.
    return cast(Mapping[str, object], bounded_worker_result(frame["result"]))


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
        stderr_text = bounded_stream_text(result.get("stderr", ""))
        # The framed field first, always: the worker read its own whole stream
        # inside its own process and carried the line the tail no longer holds.
        # The excerpt scan answers only when the frame carried no field, and
        # ``_audit_path`` says what that concession costs.
        carried = result.get("failure_audit")
        audit = carried if isinstance(carried, str) else _audit_path(stderr_text)
    else:
        stdout = StringIO()
        stderr = StringIO()
        exit_code = dispatcher(
            "run",
            _source(yaml_text),
            stdout=stdout,
            stderr=stderr,
        )
        # The tail is kept for the reader, because a failure lands at the end;
        # the audit line is taken from the whole stream, because it is written
        # before the traceback that then pushes it out of the tail.  There is
        # no frame on this path -- the dispatcher is called in-process and
        # returns an exit code -- so the scan is the only channel there is.
        recorded = stderr.getvalue()
        stderr_text = bounded_stream_text(recorded)
        audit = _audit_path(recorded)
        result = {
            "exit_code": exit_code,
            "stdout": bounded_stream_text(stdout.getvalue()),
            "stderr": stderr_text,
        }
    if exit_code == 2:
        error = ConfigError(
            bounded_text(stderr_text.strip() or f"{kind} job was refused.")
        )
        output = _failure_audit(audit)
        if output is not None:
            error.gui_output = output
        raise error
    if exit_code != 0:
        error = RuntimeError(
            bounded_text(stderr_text.strip() or f"{kind} job failed.")
        )
        output = _failure_audit(audit)
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
    "projection_body",
    "run_forward_preview",
    "run_priced_validation",
    "yaml_digest",
]
