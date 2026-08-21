"""Fresh-interpreter scientific worker for explicit GUI jobs.

This process isolates runtime establishment from the long-lived GUI API. It is
not a sandbox: trusted plugins, Python targets and server paths retain the
worker account's authority.

It also does not depend on that API surviving.  The parent sweeps this
worker's process group on every path it controls, and ``SIGKILL`` is not one
of them: a server killed mid-job runs no ``finally``, and because the worker
deliberately sits OUTSIDE the server's own group, a group signal aimed at the
server never reaches it either.  Measured, before the watch below existed: the
server dies, the group anchor follows it on stdin EOF, and the worker and
whatever science it spawned run on with a JAX or GPU context held, for as long
as the machine stays up.  So the worker watches for its own orphaning and ends
the whole job group itself.
"""

from __future__ import annotations

import argparse
import math
import os
import signal
import sys
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from io import StringIO

from _rheplicant_bootstrap.audit import AuditTrace
from _rheplicant_bootstrap.errors import ConfigError
from _rheplicant_bootstrap.execution_environment import (
    prepare_execution_environment,
)
from _rheplicant_bootstrap.gui_limits import (
    bounded_findings,
    bounded_frame,
    bounded_result,
    bounded_stream_text,
    bounded_text,
)
from _rheplicant_bootstrap.output.manager import parse_output_grammar
from _rheplicant_bootstrap.prepare import PreparedConfig, prepare_config
from _rheplicant_bootstrap.presets import read_installed_preset
from _rheplicant_bootstrap.types import SourceInput

_FRAME_PREFIX = b"\x1eRHEPLICANT_GUI_JOB "

_PARENT_POLL_SECONDS = 0.5
"""How often the worker asks whether the parent that started it still exists.

Polling rather than ``PR_SET_PDEATHSIG``, which is Linux-only and this runs on
darwin too.  Nor stdin: :func:`main` reads the whole job document to EOF before
any science begins, so by the time there is anything to protect the one signal
that stream could have carried has already been spent.  A reparented process
is the portable fact left, and reading it costs one syscall.

Irrelevant against ``MAX_WORKER_SECONDS``, and short enough that an orphaned
GPU context is measured in seconds rather than in uptime.
"""

_ORPHANED_EXIT = 3
"""Exit status of a worker that ended itself because its parent had gone.

Only reachable where there is no group to end instead, and never read by the
parent -- there is no parent by then, which is the point.
"""


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
        findings = bounded_findings(
            [
                _finding(row, layer.layer.prefix or "base")
                for layer in execution.document.layers
                for row in layer.configured.report.findings
            ]
        )
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


_AUDIT_PREFIXES = ("refused audit: ", "error audit: ")


def _published_audit(stderr_text: str) -> str | None:
    """Find the failure bundle a run published, before the tail is taken.

    ``entry`` writes this line when it publishes the sibling and only then
    lets the failure unwind and print itself, so a long traceback pushes the
    one line the parent's ``/artifacts/`` links come from out of any bounded
    excerpt.  Read it from the whole stream while the whole stream exists.
    """
    for line in reversed(stderr_text.splitlines()):
        for prefix in _AUDIT_PREFIXES:
            if line.startswith(prefix):
                return bounded_text(line.removeprefix(prefix))
    return None


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
    # Bounded here rather than in the parent: the frame this feeds must be
    # bounded before it is written, not after it has been received.
    stderr_text = stderr.getvalue()
    result: dict[str, object] = {
        "exit_code": exit_code,
        "stdout": bounded_stream_text(stdout.getvalue()),
        "stderr": bounded_stream_text(stderr_text),
    }
    audit = _published_audit(stderr_text)
    if audit is not None:
        result["failure_audit"] = audit
    return result


def _job_group(parent: int) -> int | None:
    """The process group the parent opened FOR this worker, or ``None``.

    A worker started through ``gui_child.drained_run`` runs in a group its
    parent created and named -- the anchor's -- which is precisely a group
    that is NOT the parent's own.  A worker started any other way inherits its
    parent's group, and that group holds whatever else the parent lives among:
    a shell's job, a supervisor's children, the rest of a test session.

    So the identity is the certificate, exactly as it is in
    ``gui_child.process_group``, read from the other side: our group differs
    from our parent's ONLY if somebody put us in a group made for us, and the
    only process that could have is the parent, at the moment it started us.
    Read while the parent is still alive, because that is the only time it can
    be read at all.

    ``None`` -- inherited, or unknowable -- means the sweep is declined and
    this process ends alone.  A group that was not opened for us is not ours
    to ``SIGKILL``, whatever is holding it.
    """
    try:
        group = os.getpgrp()
        return None if group == os.getpgid(parent) else group
    except (OSError, AttributeError):  # pragma: no cover - stated POSIX floor
        return None


def _reclaim(group: int | None) -> None:
    """End this orphaned job: the whole group where there is one, else itself.

    ``SIGKILL`` and not the parent's TERM-then-KILL escalation.  Surviving a
    TERM aimed at our own group in order to escalate would mean changing this
    process's disposition, which only the main thread may do and which would
    also make every ordinary parent-side sweep wait out its full grace period.
    There is nothing to be gentle for by this point regardless: the process
    that would have read a graceful result is the one that has died.

    The group takes the anchor, this worker and any science it spawned, so the
    orphan is reclaimed rather than renamed.  ``os._exit`` is the fallback for
    a group that could not be signalled and for the declined-sweep case, and
    is deliberately not an exception: an orphaned worker must not be able to
    write one more frame on its way out.
    """
    if group is not None:
        try:
            os.killpg(group, signal.SIGKILL)
        except (OSError, AttributeError):  # pragma: no cover - stated POSIX floor
            pass
    os._exit(_ORPHANED_EXIT)


def _holds(pid: int) -> bool:
    """Whether ``pid`` still names a process; signal 0 sends nothing.

    ``EPERM`` means it exists and is simply not ours, which is still "there",
    and is the answer that keeps this from reporting a death it cannot see.
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    return True


def _orphaned(parent: int, group: int | None) -> bool:
    """Whether the parent that started this worker has gone.

    Two facts rather than one, because the obvious one has a blind spot at
    exactly the moment it is most likely to be needed.

    ``os.getppid`` changing is the portable signal, and it changes once and
    for one reason -- this process was reparented to init -- so it cannot fire
    spuriously.  What it cannot see is a parent that died BEFORE the watch was
    armed: the value captured then is already init's, it never changes again,
    and the loop is inert for the life of the job.  Measured while building
    the orphaning test: a server killed during the worker's interpreter
    start-up left the worker running with a watch that could never fire.

    So the group's leader is read as well.  ``_job_group`` has already
    certified that this group was opened for this worker by its parent, which
    makes the leader a process the parent created and holds open -- the anchor
    in ``gui_child._group_anchor``, which dies on the EOF its parent's death
    delivers.  The leader's pid cannot be recycled underneath this check while
    this process is still a member of the group it names.  And the parent
    never closes that group down while the worker is alive: it sweeps only
    after the child it is waiting on has gone.

    ``None`` for the group means there is no such leader to read, and the
    ppid alone answers -- which is right, because a worker in a group it
    merely inherited has no anchor of its own to have lost.
    """
    if os.getppid() != parent:
        return True
    return group is not None and not _holds(group)


def _watch_parent(parent: int, group: int | None, stop: threading.Event) -> None:
    """Poll until the job ends or this worker turns out to have been orphaned."""
    while not stop.wait(_PARENT_POLL_SECONDS):
        if _orphaned(parent, group):
            _reclaim(group)


@contextmanager
def _watched_for_orphaning() -> Iterator[None]:
    """Watch for the parent's death for exactly as long as the job runs.

    Armed and disarmed with the job rather than with the process: stopping is
    setting an event, so a normal exit is not delayed by a millisecond, and an
    in-process caller of :func:`main` carries the watch for the length of one
    call and no longer.  The thread is a daemon so that nothing joins it, and
    it never touches stdout, so the bounded frame protocol cannot see it.
    """
    parent = os.getppid()
    stop = threading.Event()
    watch = threading.Thread(
        target=_watch_parent,
        args=(parent, _job_group(parent), stop),
        name="rheplicant-gui-worker-orphan-watch",
        daemon=True,
    )
    watch.start()
    try:
        yield
    finally:
        stop.set()


def _write_frame(frame: Mapping[str, object]) -> None:
    sys.stdout.flush()
    sys.stdout.buffer.write(_FRAME_PREFIX + bounded_frame(frame) + b"\n")
    sys.stdout.buffer.flush()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "kind",
        choices=("validate", "preview_forward", "run", "compare", "benchmark"),
    )
    kind = parser.parse_args(argv).kind
    with _watched_for_orphaning():
        return _framed_job(kind)


def _framed_job(kind: str) -> int:
    """Run one job and write exactly one frame for it, whatever it did."""
    try:
        yaml_text = sys.stdin.buffer.read().decode("utf-8", "strict")
        if kind == "validate":
            result = _run_validation(yaml_text)
        elif kind == "preview_forward":
            result = _run_forward_preview(yaml_text)
        else:
            result = _run_formal(yaml_text)
        frame = {"status": "ok", "result": bounded_result(result)}
    except ConfigError as error:
        frame = {"status": "refused", "message": bounded_text(error)}
    except Exception as error:  # noqa: BLE001 -- one bounded terminal frame
        frame = {
            "status": "error",
            "exception_type": bounded_text(type(error).__name__),
            "message": bounded_text(error),
        }
    try:
        _write_frame(frame)
    except Exception as error:  # noqa: BLE001 -- a frame is written regardless
        # Writing is part of the job, not something after it: a worker that
        # returns without a frame is read by the parent as a job that
        # finished with no result, which is the one thing it did not do.
        _write_frame(
            {
                "status": "error",
                "exception_type": "GuiFrameUnwritable",
                "message": bounded_text(error),
            }
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
