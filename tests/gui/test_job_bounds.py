"""The bounded-diagnostic contract for GUI job results and messages.

Every limit here is a constant in ``_rheplicant_bootstrap.gui_limits`` so the
fresh-interpreter worker and the long-lived parent bound the same things by
the same numbers.  Truncating late bounds a message but not the memory that
held it, so these tests measure where the bound actually bites.
"""

from __future__ import annotations

import contextlib
import json
import os
import pathlib
import re
import signal
import subprocess
import sys
import threading
import time
import tracemalloc
from io import BytesIO
from types import SimpleNamespace

import pytest
import yaml

import _rheplicant_bootstrap.gui_worker as gui_worker
from _rheplicant_bootstrap import gui_child, gui_limits
from _rheplicant_bootstrap.errors import ConfigError
from _rheplicant_bootstrap.gui_limits import (
    MAX_CHILD_STREAM_BYTES,
    MAX_COLLECTION_LENGTH,
    MAX_FINDING_COUNT,
    MAX_FRAME_BYTES,
    MAX_FRAME_TAIL_BYTES,
    MAX_NESTING_DEPTH,
    MAX_RESULT_BYTES,
    MAX_STREAM_BYTES,
    MAX_TEXT_CHARACTERS,
    MAX_WORKER_SECONDS,
    TRUNCATION_KEY,
    TRUNCATION_MARKER,
    bounded_findings,
    bounded_frame,
    bounded_result,
    bounded_stream_text,
    bounded_text,
    bounded_worker_result,
)
from rheplicant.gui import jobs
from rheplicant.gui.jobs import JobStore, execute_job, yaml_digest
from tests.config.test_config_document import synthetic_document

from .test_jobs import YAML

_FRAME_PREFIX = b"\x1eRHEPLICANT_GUI_JOB "

# Pinned as a literal, never as the imported constant: asserting a value is
# present using the constant itself passes vacuously if the constant is
# emptied, which is exactly the regression these tests exist to catch.
_MARKER = "…[truncated]"

# A stream of one repeated character cannot distinguish keeping the head from
# keeping the tail, so every stream fixture below is written between these two
# sentinels and asserts which of them survived.
_HEAD = "HEAD-SENTINEL-DROP-ME"
_TAIL = "TAIL-SENTINEL-KEEP-ME"


def _preview_document():
    document = synthetic_document()
    document["observation"]["freq"]["grid"]["linspace"]["num"] = 64
    document["observation"]["time"]["grid"]["arange"]["num"] = 64
    return yaml.safe_dump(document, sort_keys=False)


# --- the contract itself -------------------------------------------------


def test_every_limit_is_declared_once_and_leaves_room_for_the_science():
    assert MAX_COLLECTION_LENGTH >= 64, "a 64x64 preview grid must survive"
    assert MAX_NESTING_DEPTH >= 6
    assert MAX_FINDING_COUNT >= 1
    assert TRUNCATION_MARKER == _MARKER
    assert MAX_TEXT_CHARACTERS >= len(TRUNCATION_MARKER) + 1
    assert MAX_FRAME_TAIL_BYTES > MAX_FRAME_BYTES
    assert MAX_FRAME_BYTES >= MAX_RESULT_BYTES
    assert MAX_STREAM_BYTES >= MAX_TEXT_CHARACTERS
    assert MAX_CHILD_STREAM_BYTES > MAX_FRAME_TAIL_BYTES
    assert 0 < MAX_WORKER_SECONDS <= 24 * 60 * 60, "finite, and not a day"
    # The reserved key is one escape long, so no escaped payload key can be
    # it; see gui_limits for why that makes the channel unforgeable.
    assert TRUNCATION_KEY.startswith("\x00")
    assert not TRUNCATION_KEY.removeprefix("\x00").startswith("\x00")


def test_neither_side_of_the_worker_boundary_redefines_a_limit():
    """One source of truth: every file imports, none assigns.

    ``gui_child`` is on this list because it is a third holder of these
    constants -- it decides when a child stream is over the cap and how large
    a frame may be -- and a limit re-declared there would be a limit that
    disagreed with the one the worker framed against.
    """
    names = (
        "MAX_CHILD_STREAM_BYTES",
        "MAX_COLLECTION_LENGTH",
        "MAX_FINDING_COUNT",
        "MAX_FRAME_BYTES",
        "MAX_FRAME_TAIL_BYTES",
        "MAX_NESTING_DEPTH",
        "MAX_RESULT_BYTES",
        "MAX_STREAM_BYTES",
        "MAX_TEXT_CHARACTERS",
        "MAX_WORKER_SECONDS",
        "TRUNCATION_KEY",
        "TRUNCATION_MARKER",
    )
    for module in (jobs, gui_worker, gui_child):
        source = pathlib.Path(module.__file__).read_text(encoding="utf-8")
        assert "from _rheplicant_bootstrap.gui_limits import" in source
        for name in names:
            assert f"{name} =" not in source, (module.__file__, name)


# --- the worker bounds before it frames ----------------------------------


def _drive_worker(monkeypatch, kind, runner):
    written = BytesIO()
    monkeypatch.setattr(
        gui_worker.sys,
        "stdin",
        SimpleNamespace(buffer=BytesIO(b"schema_version: 1\n")),
    )
    monkeypatch.setattr(
        gui_worker.sys,
        "stdout",
        SimpleNamespace(buffer=written, flush=lambda: None),
    )
    monkeypatch.setattr(gui_worker, "_run_validation", runner)
    assert gui_worker.main([kind]) == 0
    encoded = written.getvalue().rsplit(_FRAME_PREFIX, 1)[1].split(b"\n", 1)[0]
    return encoded, json.loads(encoded.decode("utf-8", "strict"))


def test_the_worker_bounds_a_five_megabyte_result_before_it_frames_it(monkeypatch):
    encoded, frame = _drive_worker(
        monkeypatch,
        "validate",
        lambda _text: {"findings": [], "stdout": "x" * 5_000_000},
    )

    assert len(encoded) <= MAX_FRAME_BYTES
    assert frame["status"] == "ok"
    assert len(frame["result"]["stdout"]) <= MAX_TEXT_CHARACTERS
    assert _MARKER in frame["result"]["stdout"]


def test_the_worker_bounds_a_megabyte_exception_before_it_frames_it(monkeypatch):
    encoded, frame = _drive_worker(
        monkeypatch,
        "validate",
        lambda _text: (_ for _ in ()).throw(RuntimeError("b" * 1_500_000)),
    )

    assert len(encoded) <= MAX_FRAME_BYTES
    assert frame["status"] == "error"
    assert frame["exception_type"] == "RuntimeError"
    assert len(frame["message"]) <= MAX_TEXT_CHARACTERS
    assert _MARKER in frame["message"]


def test_the_worker_bounds_a_megabyte_refusal_before_it_frames_it(monkeypatch):
    encoded, frame = _drive_worker(
        monkeypatch,
        "validate",
        lambda _text: (_ for _ in ()).throw(ConfigError("r" * 1_500_000)),
    )

    assert len(encoded) <= MAX_FRAME_BYTES
    assert frame["status"] == "refused"
    assert len(frame["message"]) <= MAX_TEXT_CHARACTERS
    assert _MARKER in frame["message"]


def test_the_worker_caps_the_finding_count_with_a_visible_marker(monkeypatch):
    rows = [
        SimpleNamespace(
            check=f"C{index}",
            severity="report",
            where="document",
            # Only the first message is long.  The per-finding character cap
            # is still exercised, while every row near the end is short, so a
            # marker on the LAST row can only have come from the synthetic
            # count finding -- with uniformly long messages ``bounded_text``
            # stamps the last kept row and the assertion below passes whether
            # or not the count was ever reported.
            message="m" * 20_000 if index == 0 else f"finding {index}",
        )
        for index in range(5_000)
    ]
    layers = SimpleNamespace(
        layers=(
            SimpleNamespace(
                layer=SimpleNamespace(prefix=""),
                configured=SimpleNamespace(report=SimpleNamespace(findings=rows)),
            ),
        )
    )
    execution = SimpleNamespace(document=layers, close=lambda: None)
    monkeypatch.setattr(gui_worker, "_prepared_config", lambda _text: object())
    monkeypatch.setattr(
        gui_worker,
        "prepare_execution_environment",
        lambda *_args, **_kwargs: execution,
    )

    found = gui_worker._run_validation("schema_version: 1\n")

    assert len(found["findings"]) <= MAX_FINDING_COUNT
    assert all(
        len(row["message"]) <= MAX_TEXT_CHARACTERS for row in found["findings"]
    )
    assert _MARKER in found["findings"][-1]["message"]
    assert len(found["findings"]) == MAX_FINDING_COUNT
    assert _MARKER in found["findings"][0]["message"], "the long row was cut"
    dropped = found["findings"][-1]
    assert dropped["check"] == "gui.diagnostics.truncated"
    assert dropped["severity"] == "report"
    assert dropped["message"] == (
        f"{5_000 - (MAX_FINDING_COUNT - 1)} further findings were dropped "
        f"{_MARKER}"
    )


def test_the_worker_bounds_the_formal_streams_it_returns(monkeypatch):
    from _rheplicant_bootstrap import entry

    def dispatcher(_command, _source, *, stdout, stderr):
        # Distinguishable ends: a single repeated character cannot tell
        # head-keeping from tail-keeping, and the end of a stream is where a
        # failure -- and the audit line the parent links from -- lands.
        stdout.write(_HEAD + "o" * 5_000_000 + _TAIL)
        stderr.write(_HEAD + "e" * 5_000_000 + _TAIL)
        return 0

    monkeypatch.setattr(entry, "dispatch_request", dispatcher)

    found = gui_worker._run_formal("schema_version: 1\n")

    assert len(found["stdout"]) <= MAX_TEXT_CHARACTERS
    assert len(found["stderr"]) <= MAX_TEXT_CHARACTERS
    assert _MARKER in found["stdout"]
    assert _MARKER in found["stderr"]
    assert found["stdout"].startswith(_MARKER)
    assert found["stderr"].startswith(_MARKER)
    assert found["stdout"].endswith(_TAIL)
    assert found["stderr"].endswith(_TAIL)
    assert _HEAD not in found["stdout"]
    assert _HEAD not in found["stderr"]


# --- the parent never buffers the child's output -------------------------


_NOISY_CHILD = (
    "import sys, json;"
    "sys.stdin.buffer.read();"
    "block = b'n' * 65536;"
    "written = [sys.stdout.buffer.write(block) for _ in range(512)];"
    "frame = json.dumps({'status': 'ok', 'result': {'layers': 2}});"
    "sys.stdout.buffer.write(b'\\x1eRHEPLICANT_GUI_JOB ' + frame.encode() + b'\\n');"
    "sys.stdout.buffer.flush()"
)


def test_the_parent_never_holds_the_whole_child_stream_in_memory(monkeypatch):
    """The child writes 32 MiB; the parent's peak must be a constant."""
    real_run = subprocess.run

    def fake_run(_arguments, **kwargs):
        return real_run([sys.executable, "-c", _NOISY_CHILD], **kwargs)

    monkeypatch.setattr(jobs.subprocess, "run", fake_run)
    tracemalloc.start()
    try:
        found = jobs._run_isolated_job("validate", "schema_version: 1\n")
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    assert found == {"layers": 2}
    assert peak < 8 * 1024 * 1024, peak


_FRAME_LINE = (
    "frame = json.dumps({'status': 'ok', 'result': {'layers': 2}});"
    "sys.stdout.buffer.write(b'\\x1eRHEPLICANT_GUI_JOB ' + frame.encode() + b'\\n');"
)

_LATE_NOISE_CHILD = (
    "import sys, json;"
    "sys.stdin.buffer.read();"
    "block = b'n' * 65536;"
    "before = [sys.stdout.buffer.write(block) for _ in range(512)];"
    + _FRAME_LINE
    + "after = [sys.stdout.buffer.write(block) for _ in range(64)];"
    "sys.stdout.buffer.flush()"
)

_FLOOD_MEGABYTES = MAX_CHILD_STREAM_BYTES // (1024 * 1024) + 1

_FLOOD_CHILD = (
    "import sys;"
    "sys.stdin.buffer.read();"
    "block = b'n' * 1048576;"
    f"written = [sys.stdout.buffer.write(block) for _ in range({_FLOOD_MEGABYTES})];"
    "sys.stdout.buffer.flush()"
)


def _child_runner(monkeypatch, source):
    """Run ``source`` as the worker, keeping every stream argument intact."""
    real_run = subprocess.run

    def fake_run(_arguments, **kwargs):
        return real_run([sys.executable, "-c", source], **kwargs)

    monkeypatch.setattr(jobs.subprocess, "run", fake_run)


def test_a_frame_is_not_lost_to_whatever_the_child_prints_after_it(monkeypatch):
    """Four megabytes of shutdown chatter must not erase a finished result.

    The worker writes its frame last, but ``atexit`` handlers and native
    libraries write to the same descriptor afterwards, so a result frame is
    not the last thing on stdout and a fixed tail window can hold only noise.
    """
    _child_runner(monkeypatch, _LATE_NOISE_CHILD)
    tracemalloc.start()
    try:
        found = jobs._run_isolated_job("validate", "schema_version: 1\n")
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    assert found == {"layers": 2}
    assert peak < 8 * 1024 * 1024, peak


def test_a_child_stream_beyond_the_cap_ends_the_job_rather_than_the_volume(
    monkeypatch,
):
    """The bound has to bite on what the child writes, not on what is kept.

    Redirecting the stream to a temporary file bounds the parent's memory and
    nothing else: the file grows to whatever the child printed.  Nothing is
    stored here, and past the cap the job is refused outright.
    """
    _child_runner(monkeypatch, _FLOOD_CHILD)
    tracemalloc.start()
    try:
        with pytest.raises(RuntimeError) as failure:
            jobs._run_isolated_job("validate", "schema_version: 1\n")
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    message = str(failure.value)
    assert f"more than {MAX_CHILD_STREAM_BYTES} bytes" in message
    assert len(message) <= MAX_TEXT_CHARACTERS
    assert peak < 8 * 1024 * 1024, peak


def test_the_parent_bounds_how_long_a_worker_may_run(monkeypatch):
    captured: dict[str, object] = {}

    def fake_run(arguments, **kwargs):
        captured.update(kwargs)
        raise subprocess.TimeoutExpired(arguments, kwargs["timeout"])

    monkeypatch.setattr(jobs.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="did not finish within"):
        jobs._run_isolated_job("validate", "schema_version: 1\n")

    assert captured["timeout"] == MAX_WORKER_SECONDS


def test_a_worker_that_never_exits_leaves_a_bounded_terminal_job(monkeypatch):
    """A child that never returns must not hold its job -- or its twin.

    A record stuck in ``running`` is what ``_active_duplicate`` reads, and no
    route cancels a job, so an unbounded child would refuse every identical
    resubmission for the life of the process.
    """
    _child_runner(
        monkeypatch, "import sys, time; sys.stdin.buffer.read(); time.sleep(30)"
    )
    monkeypatch.setattr(jobs, "MAX_WORKER_SECONDS", 0.5)
    store = JobStore(id_factory=iter(("job-a", "job-b")).__next__)
    row = store.submit("session-1", "validate", 0, YAML)

    store.run(row.job_id, jobs._run_isolated_job)

    finished = store.get(row.job_id)
    assert finished.status == "error"
    assert "did not finish within" in finished.message
    assert len(finished.message) <= MAX_TEXT_CHARACTERS
    assert store.submit("session-1", "validate", 0, YAML).status == "queued"


def test_the_parent_refuses_a_frame_payload_beyond_the_frame_cap(monkeypatch):
    payload = b'{"status":"ok","result":{"pad":"' + b"p" * MAX_FRAME_BYTES + b'"}}'

    def fake_run(arguments, **_kwargs):
        return subprocess.CompletedProcess(
            arguments,
            0,
            stdout=_FRAME_PREFIX + payload + b"\n",
            stderr=b"worker detail",
        )

    monkeypatch.setattr(jobs.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="result frame is larger"):
        jobs._run_isolated_job("validate", "schema_version: 1\n")


def test_the_parent_bounds_the_child_stderr_it_reports(monkeypatch):
    real_run = subprocess.run
    child = (
        "import sys;"
        "sys.stdin.buffer.read();"
        f"sys.stderr.buffer.write({_HEAD.encode()!r});"
        "sys.stderr.buffer.write(b'e' * 5_000_000);"
        f"sys.stderr.buffer.write({_TAIL.encode()!r});"
        "sys.stderr.buffer.flush();"
        "raise SystemExit(7)"
    )

    def fake_run(_arguments, **kwargs):
        return real_run([sys.executable, "-c", child], **kwargs)

    monkeypatch.setattr(jobs.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError) as failure:
        jobs._run_isolated_job("validate", "schema_version: 1\n")

    message = str(failure.value)
    assert "exited 7" in message
    assert len(message) <= MAX_TEXT_CHARACTERS + 100
    assert _MARKER in message
    # The child's failure is at the end of what it printed; keeping the head
    # of five megabytes of padding would report nothing about it.
    assert message.endswith(_TAIL)
    assert _HEAD not in message


# --- the store bounds every terminal record ------------------------------


def test_a_megabyte_terminal_exception_is_bounded_in_the_store():
    store = JobStore(id_factory=lambda: "job-error")
    row = store.submit("session-1", "validate", 0, YAML)

    store.run(
        row.job_id,
        lambda _kind, _yaml: (_ for _ in ()).throw(RuntimeError("x" * 1_500_000)),
    )

    finished = store.get(row.job_id)
    assert finished.status == "error"
    assert len(finished.message) <= MAX_TEXT_CHARACTERS
    assert finished.message.startswith("RuntimeError: ")
    assert _MARKER in finished.message


def test_a_megabyte_refusal_is_bounded_in_the_store():
    store = JobStore(id_factory=lambda: "job-refused")
    row = store.submit("session-1", "validate", 0, YAML)

    store.run(
        row.job_id,
        lambda _kind, _yaml: (_ for _ in ()).throw(ConfigError("y" * 1_500_000)),
    )

    finished = store.get(row.job_id)
    assert finished.status == "refused"
    assert len(finished.message) <= MAX_TEXT_CHARACTERS
    assert _MARKER in finished.message


def test_a_hostile_success_result_is_bounded_in_length_depth_and_strings():
    # The leaf is short and the chain is uniform, so nothing but the nesting
    # cap can explain a marker part-way down it: a long leaf would be stamped
    # by ``bounded_text`` no matter how deep the depth limit was raised.
    deep: object = "leaf"
    for _ in range(60):
        deep = {"nested": deep}
    store = JobStore(id_factory=lambda: "job-hostile")
    row = store.submit("session-1", "validate", 0, YAML)

    store.run(
        row.job_id,
        lambda _kind, _yaml: {
            "wide": list(range(50_000)),
            "deep": deep,
            "text": "t" * 900_000,
        },
    )

    finished = store.get(row.job_id)
    encoded = json.dumps(finished.result, ensure_ascii=False)
    assert finished.status == "succeeded"
    assert len(finished.result["wide"]) <= MAX_COLLECTION_LENGTH + 1
    assert len(finished.result["text"]) <= MAX_TEXT_CHARACTERS
    assert len(encoded) <= MAX_RESULT_BYTES
    assert _MARKER in encoded
    # Walk the chain: the marker must stand at exactly the declared cap, and
    # the sixty authored levels below it must be gone.
    node = finished.result["deep"]
    depth = 1  # ``result["deep"]`` is already one level inside the result
    while isinstance(node, dict):
        node = node["nested"]
        depth += 1
    assert node == _MARKER
    assert depth == MAX_NESTING_DEPTH


def test_refused_findings_are_capped_with_one_marker_finding():
    rows = [
        SimpleNamespace(
            check=f"C{index}",
            severity="report",
            where="document",
            # Long only where the character cap is under test; short near the
            # end, so the marker on the last row cannot be a stamped message
            # standing in for the count finding that must be there.
            message="m" * 50_000 if index == 0 else f"finding {index}",
        )
        for index in range(5_000)
    ]
    error = ConfigError("priced refusal")
    error.report = SimpleNamespace(findings=rows)
    store = JobStore(id_factory=lambda: "job-findings")
    row = store.submit("session-1", "validate", 0, YAML)

    store.run(row.job_id, lambda _kind, _yaml: (_ for _ in ()).throw(error))

    finished = store.get(row.job_id)
    findings = finished.result["findings"]
    assert finished.status == "refused"
    assert len(findings) <= MAX_FINDING_COUNT
    assert all(len(item["message"]) <= MAX_TEXT_CHARACTERS for item in findings)
    assert _MARKER in findings[-1]["message"]
    assert len(findings) == MAX_FINDING_COUNT
    assert _MARKER in findings[0]["message"], "the long row was cut"
    assert findings[-1]["check"] == "gui.diagnostics.truncated"
    assert findings[-1]["message"] == (
        f"{5_000 - (MAX_FINDING_COUNT - 1)} further findings were dropped "
        f"{_MARKER}"
    )


def test_a_legal_closed_result_passes_through_untouched():
    result = {
        "exit_code": 0,
        "stdout": "wrote 3 products",
        "stderr": "",
        "output": {
            "state": "ready_new",
            "target_path": "/srv/results/run-1",
            "marker_id": "12345678-1234-4123-8123-123456789abc",
            "audit_files": ["config.resolved.yaml", "audit.json"],
            "target_device": 16777232,
            "target_inode": 4242,
        },
    }
    store = JobStore(id_factory=lambda: "job-clean")
    row = store.submit("session-1", "run", 0, YAML)

    store.run(row.job_id, lambda _kind, _yaml: result)

    finished = store.get(row.job_id)
    assert finished.status == "succeeded"
    assert finished.result == result
    assert _MARKER not in json.dumps(finished.result, ensure_ascii=False)


# --- the dispatcher path bounds its own streams --------------------------


def test_the_dispatcher_path_bounds_the_streams_it_records():
    def noisy(_command, _source, *, stdout, stderr):
        stdout.write(_HEAD + "o" * 5_000_000 + _TAIL)
        stderr.write(_HEAD + "e" * 5_000_000 + _TAIL)
        return 0

    found = execute_job("run", YAML, dispatcher=noisy)

    assert len(found["stdout"]) <= MAX_TEXT_CHARACTERS
    assert len(found["stderr"]) <= MAX_TEXT_CHARACTERS
    assert _MARKER in found["stdout"]
    assert _MARKER in found["stderr"]
    # What a run says last is what a reader needs; keeping the head would
    # report five megabytes of padding and none of the outcome.
    assert found["stdout"].endswith(_TAIL)
    assert found["stderr"].endswith(_TAIL)
    assert _HEAD not in found["stdout"]
    assert _HEAD not in found["stderr"]


def _published_target(tmp_path):
    """One published failure bundle the artifacts route can serve."""
    target = tmp_path / "result.refused"
    target.mkdir(mode=0o700)
    marker_id = "12345678-1234-4123-8123-123456789abc"
    marker = target / ".rheplicant-results.json"
    marker.write_text(
        json.dumps({"format_version": 1, "run_directory_id": marker_id}),
        encoding="utf-8",
    )
    marker.chmod(0o600)
    resolved = target / "config.resolved.yaml"
    resolved.write_text("schema_version: 1\n", encoding="utf-8")
    resolved.chmod(0o600)
    return target, marker_id


_TRACEBACK = "".join(
    f'  File "plugin.py", line {index}, in step\n    raise ConfigError(row)\n'
    for index in range(500)
)


def test_a_published_audit_line_survives_the_traceback_that_follows_it(tmp_path):
    """The audit line is printed before the traceback, not after it.

    ``entry._publish_failure_once`` writes ``refused audit: <path>`` and only
    then does the refusal unwind and print its findings, so a long traceback
    pushes the one line the ``/artifacts/`` route needs out of any bounded
    window.  The link must be taken from the stream, not from its excerpt.
    """
    target, marker_id = _published_target(tmp_path)

    def dispatcher(_command, _source, *, stdout, stderr):
        stderr.write(f"refused: priced refusal\nrefused audit: {target}\n")
        stderr.write(_TRACEBACK)
        return 2

    with pytest.raises(ConfigError) as refusal:
        execute_job("run", YAML, dispatcher=dispatcher)

    output = getattr(refusal.value, "gui_output", None)
    assert len(_TRACEBACK) > MAX_TEXT_CHARACTERS, "the fixture must overflow"
    assert output is not None, "the audit bundle links went with the traceback"
    assert output["target_path"] == str(target)
    assert output["marker_id"] == marker_id
    assert output["audit_files"] == ["config.resolved.yaml"]
    assert len(str(refusal.value)) <= MAX_TEXT_CHARACTERS
    assert _MARKER in str(refusal.value)


def test_the_worker_carries_the_audit_line_its_stream_cap_removes(monkeypatch):
    from _rheplicant_bootstrap import entry

    def dispatcher(_command, _source, *, stdout, stderr):
        stderr.write("refused audit: /srv/results/run-1\n")
        stderr.write(_TRACEBACK * 20)
        return 2

    monkeypatch.setattr(entry, "dispatch_request", dispatcher)

    found = gui_worker._run_formal("schema_version: 1\n")

    assert len(found["stderr"]) <= MAX_TEXT_CHARACTERS
    assert "refused audit: " not in found["stderr"], "the tail is what is kept"
    assert found["failure_audit"] == "/srv/results/run-1"


def test_the_parent_links_the_audit_bundle_the_worker_carried(
    tmp_path, monkeypatch
):
    target, marker_id = _published_target(tmp_path)
    monkeypatch.setattr(
        jobs,
        "_run_isolated_job",
        lambda _kind, _text: {
            "exit_code": 2,
            "stdout": "",
            "stderr": f"{_MARKER}the bounded tail of a long traceback",
            "failure_audit": str(target),
        },
    )

    with pytest.raises(ConfigError) as refusal:
        execute_job("run", YAML)

    output = getattr(refusal.value, "gui_output", None)
    assert output is not None
    assert output["target_path"] == str(target)
    assert output["marker_id"] == marker_id


@pytest.mark.parametrize(
    ("exit_code", "failure"), [(2, ConfigError), (9, RuntimeError)]
)
def test_the_dispatcher_path_bounds_the_failure_message(exit_code, failure):
    def noisy(_command, _source, *, stdout, stderr):
        stderr.write("e" * 5_000_000)
        return exit_code

    with pytest.raises(failure) as raised:
        execute_job("run", YAML, dispatcher=noisy)

    assert len(str(raised.value)) <= MAX_TEXT_CHARACTERS
    assert _MARKER in str(raised.value)


# --- the science survives the caps ---------------------------------------


def test_a_real_sixty_four_by_sixty_four_preview_survives_every_cap():
    text = _preview_document()
    store = JobStore(id_factory=lambda: "job-preview")
    row = store.submit("session-1", "preview_forward", 0, text)

    store.run(row.job_id, execute_job)

    finished = store.get(row.job_id)
    assert finished.status == "succeeded", finished.message
    waterfall = finished.result["waterfall"]
    assert waterfall["shape"] == [64, 64]
    assert len(waterfall["values"]) == 64
    assert all(len(line) == 64 for line in waterfall["values"])
    assert all(isinstance(cell, float) for line in waterfall["values"] for cell in line)
    assert isinstance(waterfall["dtype"], str)
    for key in ("minimum", "maximum", "mean"):
        assert isinstance(waterfall[key], float)
    assert isinstance(finished.result["taps"], dict)
    assert isinstance(finished.result["uniform_sky_mean"], dict)
    assert _MARKER not in json.dumps(finished.result, ensure_ascii=False)
    assert finished.yaml_digest == yaml_digest(text)


# --- the truncation signal is out of the payload's reach -----------------


def _claims(value):
    """Every out-of-band truncation notice inside one bounded result."""
    if isinstance(value, dict):
        here = [value[TRUNCATION_KEY]] if TRUNCATION_KEY in value else []
        return here + [
            claim for item in value.values() for claim in _claims(item)
        ]
    if isinstance(value, list):
        return [claim for item in value for claim in _claims(item)]
    return []


def test_payload_content_cannot_destroy_the_truncation_signal():
    """A real entry named like the channel keeps its value, not the count.

    ``uniform_sky_mean`` keys are resource names and ``taps`` keys are aux
    names: both are authored in the user's YAML, so both can be spelled to
    collide with anything a truncation notice is written under.
    """
    wanted = "the real value the user cares about"
    payload = {TRUNCATION_KEY: wanted, _MARKER: "a marker-named entry too"}
    payload.update({f"entry-{index}": index for index in range(400)})

    bounded = bounded_result(payload)

    assert bounded["\x00" + TRUNCATION_KEY] == wanted
    assert bounded[_MARKER] == "a marker-named entry too"
    assert len(_claims(bounded)) == 1
    assert "further entries were dropped" in _claims(bounded)[0]

    intact = bounded_result({TRUNCATION_KEY: wanted})
    assert intact == {"\x00" + TRUNCATION_KEY: wanted}
    assert _claims(intact) == []


def test_payload_content_cannot_forge_a_truncation_that_never_happened():
    """Nothing was over a cap here, so nothing may read as though it was."""
    payload = {
        # The reviewer's case: names authored in the user's YAML, spelled to
        # look like the signal a reader scans for.
        "uniform_sky_mean": {f"probe {_MARKER}": 1.0},
        "taps": {"gain": f"a name ending in {_MARKER}"},
        "rows": [f"7 further entries were dropped {_MARKER}"],
        # And the strongest case: a name that *is* the channel.
        "resources": {TRUNCATION_KEY: "a resource named like the channel"},
    }

    bounded = bounded_result(payload)

    assert _claims(bounded) == []
    assert bounded["uniform_sky_mean"] == payload["uniform_sky_mean"]
    assert bounded["taps"] == payload["taps"]
    assert bounded["rows"] == payload["rows"]
    assert bounded["resources"] == {
        "\x00" + TRUNCATION_KEY: "a resource named like the channel"
    }


def test_a_truncated_list_reports_its_loss_in_the_same_channel():
    bounded = bounded_result({"wide": list(range(MAX_COLLECTION_LENGTH + 40))})

    kept = bounded["wide"]
    assert len(kept) == MAX_COLLECTION_LENGTH + 1
    assert kept[:MAX_COLLECTION_LENGTH] == list(range(MAX_COLLECTION_LENGTH))
    assert _claims(bounded) == [f"40 further entries were dropped {_MARKER}"]


# --- degrading beats failing to answer at all ----------------------------


def test_bounded_text_never_exceeds_a_limit_too_small_for_its_marker():
    for limit in range(len(_MARKER) + 2):
        assert len(bounded_text("x" * 100, limit=limit)) <= limit
        assert len(bounded_stream_text("x" * 100, limit=limit)) <= limit
    assert bounded_text("x" * 100, limit=5) == _MARKER[:5]
    assert bounded_stream_text("x" * 100, limit=5) == _MARKER[:5]
    assert bounded_text("x" * 100, limit=0) == ""
    assert bounded_text("short", limit=len("short")) == "short"


def test_a_value_json_cannot_carry_degrades_to_its_text():
    bounded = bounded_result({"raw": b"raw", "count": 3, "fine": "text"})

    assert bounded == {"raw": "b'raw'", "count": 3, "fine": "text"}
    assert json.dumps(bounded)


def test_an_unencodable_frame_degrades_to_a_terminal_error_frame():
    encoded = bounded_frame({"status": "ok", "result": {"raw": b"raw"}})

    frame = json.loads(encoded.decode("utf-8", "strict"))
    assert frame["status"] == "error"
    assert frame["exception_type"] == "GuiFrameUnencodable"
    assert len(frame["message"]) <= MAX_TEXT_CHARACTERS


def test_the_worker_frames_a_result_json_cannot_carry(monkeypatch):
    """A worker that raises while writing writes nothing, and nothing reads
    as a finished job with no result."""
    encoded, frame = _drive_worker(
        monkeypatch,
        "validate",
        lambda _text: {"findings": [], "raw": b"raw"},
    )

    assert len(encoded) <= MAX_FRAME_BYTES
    assert frame["status"] == "ok"
    assert frame["result"]["raw"] == "b'raw'"


# --- the frame is recognised wherever a read happens to land -------------


_GOOD_FRAME = b'{"status":"ok","result":{"layers":2}}'


def _scanned(stream, size):
    scan = jobs._FrameScan()
    for at in range(0, len(stream), size):
        scan.feed(stream[at : at + size])
    scan.close()
    return scan


@pytest.mark.parametrize("size", [1, 2, 7, 19, 20, 21, 64, 4096, 1 << 20])
@pytest.mark.parametrize(
    ("before", "after"),
    [
        (b"", b""),
        (b"noise without a newline", b"\nlate noise\n"),
        (b"lines\nof\nnoise\n", b"more noise, no newline"),
        (b"\x1eRHEPLICANT", b"\x1eRHEPLI"),
    ],
    ids=["bare", "unterminated-head", "trailing-noise", "partial-prefixes"],
)
def test_a_frame_split_across_reads_is_still_one_frame(size, before, after):
    """A read boundary is not a stream boundary.

    Reads land wherever the child flushed and the pipe filled, so the prefix,
    the payload and the newline can each be split; none of those splits may
    change what the parent decides the worker said.
    """
    scan = _scanned(before + _FRAME_PREFIX + _GOOD_FRAME + b"\n" + after, size)

    assert scan.payload == _GOOD_FRAME
    assert scan.oversized == 0


@pytest.mark.parametrize("size", [1, 20, 4096])
def test_the_last_frame_wins_however_the_reads_fall(size):
    old = b'{"status":"refused","message":"old"}'
    stream = (
        _FRAME_PREFIX + old + b"\nplugin noise\n" + _FRAME_PREFIX + _GOOD_FRAME
    )

    assert _scanned(stream, size).payload == _GOOD_FRAME
    # Even with no newline between them, a second prefix ends the first frame.
    assert _scanned(_FRAME_PREFIX + old + _FRAME_PREFIX + _GOOD_FRAME, size).payload == (
        _GOOD_FRAME
    )


@pytest.mark.parametrize("size", [1, 20, 65536])
def test_an_oversized_frame_is_reported_rather_than_half_read(size):
    stream = (
        _FRAME_PREFIX + b'{"pad":"' + b"p" * (MAX_FRAME_BYTES + 8) + b'"}\n'
    )

    scan = _scanned(stream, size)

    assert scan.payload is None
    assert scan.oversized > MAX_FRAME_BYTES


# --- the truncation channel survives the boundary it was built for -------


def _crossed(value):
    """Exactly what a frame does to a bounded result: JSON out, JSON in."""
    return json.loads(json.dumps(value, ensure_ascii=False))


def test_the_two_bounding_passes_compose_without_rewriting_the_notice():
    """The worker bounds, frames and ships; the parent stores what it gets.

    Both sides call the same bounding, so every real result is bounded twice,
    and the second pass reads the FIRST pass's notice as one more payload
    entry: it counts it against the length cap and rewrites the number, or
    escapes the reserved key and removes the notice from the channel
    altogether.  ``gui_limits`` claims the key's presence means this module
    put it there; that claim has to survive the crossing it was written for.
    """
    payload = {
        f"entry-{index}": index
        for index in range(MAX_COLLECTION_LENGTH + 44)
    }
    worker_side = bounded_result(payload)
    assert _claims(worker_side) == [f"44 further entries were dropped {_MARKER}"]

    parent_side = bounded_result(bounded_worker_result(_crossed(worker_side)))

    assert _claims(parent_side) == [f"44 further entries were dropped {_MARKER}"]
    assert parent_side == worker_side
    assert TRUNCATION_KEY in parent_side
    assert "\x00" + TRUNCATION_KEY not in parent_side


def test_a_twice_bounded_list_keeps_the_count_it_reported_the_first_time():
    payload = {"wide": list(range(MAX_COLLECTION_LENGTH + 138))}
    worker_side = bounded_result(payload)
    assert _claims(worker_side) == [f"138 further entries were dropped {_MARKER}"]

    parent_side = bounded_result(bounded_worker_result(_crossed(worker_side)))

    assert _claims(parent_side) == [f"138 further entries were dropped {_MARKER}"]
    assert parent_side["wide"][:MAX_COLLECTION_LENGTH] == list(
        range(MAX_COLLECTION_LENGTH)
    )
    assert len(parent_side["wide"]) == MAX_COLLECTION_LENGTH + 1


def test_a_worker_bounded_result_reaches_the_store_with_its_count_intact(
    monkeypatch,
):
    """The whole crossing, through the seam it actually happens on."""
    payload = {
        f"entry-{index}": index
        for index in range(MAX_COLLECTION_LENGTH + 44)
    }
    # Exactly what ``gui_worker.main`` writes: bound, then frame.
    frame = bounded_frame({"status": "ok", "result": bounded_result(payload)})

    def fake_run(arguments, **_kwargs):
        return subprocess.CompletedProcess(
            arguments, 0, stdout=_FRAME_PREFIX + frame + b"\n", stderr=b""
        )

    monkeypatch.setattr(jobs.subprocess, "run", fake_run)
    store = JobStore(id_factory=lambda: "job-crossed")
    row = store.submit("session-1", "validate", 0, YAML)

    store.run(row.job_id, jobs.execute_job)

    finished = store.get(row.job_id)
    assert finished.status == "succeeded"
    assert _claims(finished.result) == [
        f"44 further entries were dropped {_MARKER}"
    ]
    assert json.dumps(finished.result, ensure_ascii=False)


def test_a_result_larger_than_the_parents_budget_is_bounded_for_real():
    """Provenance is not a licence: the frame cap sits above the result cap."""
    oversized = {"pad": "p" * (MAX_RESULT_BYTES + 4096)}

    bounded = bounded_worker_result(oversized)

    assert len(bounded["pad"]) <= MAX_TEXT_CHARACTERS
    assert len(json.dumps(bounded, ensure_ascii=False)) <= MAX_RESULT_BYTES


# --- both reserved channels are reserved by escaping, not by hope --------


def test_the_stored_key_escape_is_injective():
    """Two payload keys must never be stored under one name.

    A rule that escapes only the keys that EQUAL the channel still collides:
    it maps ``TRUNCATION_KEY`` onto the escaped form of itself, which is
    exactly what an already-escaped payload key is stored as.
    """
    names = {
        "",
        "gui.truncated",
        TRUNCATION_KEY,
        "\x00" + TRUNCATION_KEY,
        "\x00\x00" + TRUNCATION_KEY,
        "\x00",
        "\x00\x00",
        _MARKER,
        f"probe {_MARKER}",
    }

    stored = {name: gui_limits._stored_key(name) for name in names}

    assert len(set(stored.values())) == len(names), stored
    assert TRUNCATION_KEY not in set(stored.values())


def test_a_payload_finding_cannot_announce_a_truncation_that_never_happened():
    """The findings marker is a channel too, and it was open in both ways."""
    forged = [
        {
            "check": "gui.diagnostics.truncated",
            "severity": "report",
            "where": "document",
            "message": f"9999 further findings were dropped {_MARKER}",
            "layer": "base",
        }
    ]

    kept = bounded_findings(forged)

    assert len(kept) == 1
    assert kept[0]["check"] != "gui.diagnostics.truncated"
    assert kept[0]["check"].endswith("gui.diagnostics.truncated")

    # And the other direction: a real marker still spells itself plainly.
    rows = [
        {"check": f"C{index}", "severity": "report", "where": "d",
         "message": f"m{index}", "layer": "base"}
        for index in range(MAX_FINDING_COUNT + 7)
    ]

    capped = bounded_findings(rows)

    assert len(capped) == MAX_FINDING_COUNT
    assert capped[-1]["check"] == "gui.diagnostics.truncated"
    assert [row["check"] for row in capped[:-1]] == [
        f"C{index}" for index in range(MAX_FINDING_COUNT - 1)
    ]


def test_a_finding_key_cannot_reach_the_reserved_mapping_channel():
    kept = bounded_findings([{TRUNCATION_KEY: "forged", "check": "c"}])

    assert TRUNCATION_KEY not in kept[0]
    assert kept[0]["\x00" + TRUNCATION_KEY] == "forged"


# --- the child is a process, not just a stream ---------------------------

_POSIX_ONLY = pytest.mark.skipif(
    os.name != "posix", reason="process groups and killpg are POSIX"
)

#: A descendant that inherits the worker's streams and simply will not let go.
_HOLDS_THE_STREAM = "import time\ntime.sleep(120)\n"

#: The same, except that it also writes a result frame of its own -- after the
#: worker has exited, so "the last frame wins" hands it the whole job.
_FRAMES_A_RESULT = (
    "import sys, time\n"
    "time.sleep(0.4)\n"
    "sys.stdout.buffer.write("
    "b'\\x1eRHEPLICANT_GUI_JOB {\"status\":\"ok\",\"result\":{\"layers\":666}}\\n')\n"
    "sys.stdout.buffer.flush()\n"
    "time.sleep(120)\n"
)

_WORKER_FRAME = (
    "sys.stdout.buffer.write("
    "b'\\x1eRHEPLICANT_GUI_JOB {\"status\":\"ok\",\"result\":{\"layers\":2}}\\n');"
    "sys.stdout.buffer.flush()"
)


def _worker_leaving(tmp_path, descendant_source):
    """A worker that spawns a descendant on its own streams, then exits.

    Exactly what a scientific worker does when the science it establishes
    spawns anything of its own: the grandchild inherits the pipes the parent
    is draining, and ``subprocess.run``'s timeout reaches only the worker.
    """
    script = tmp_path / "descendant.py"
    script.write_text(descendant_source, encoding="utf-8")
    record = tmp_path / "descendant.pid"
    source = (
        "import subprocess, sys;"
        "sys.stdin.buffer.read();"
        f"child = subprocess.Popen([sys.executable, {str(script)!r}]);"
        f"open({str(record)!r}, 'w').write(str(child.pid));"
        + _WORKER_FRAME
    )
    return source, record


def _open_descriptors():
    """How many descriptors this process holds, counted the same way twice."""
    return len(os.listdir("/dev/fd"))


def _settled(fds, threads, *, timeout=15.0):
    """Wait a bounded time for the reclaim, then report what is still out."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _open_descriptors() <= fds and threading.active_count() <= threads:
            return True
        time.sleep(0.05)
    return False


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@_POSIX_ONLY
def test_a_descendant_holding_the_pipe_leaks_no_descriptor_and_no_thread(
    monkeypatch, tmp_path
):
    """The leak that ends a long-lived server, measured rather than argued.

    A grandchild that inherited the worker's stdout keeps ``os.read`` from
    ever returning, so the join times out.  The descriptor used to be left
    open deliberately -- closing a live reader's number would hand it to the
    next file this process opens -- and the daemon thread was simply
    abandoned.  Nothing reclaimed either: four jobs cost four descriptors and
    four threads per stream, and the server dies at ``accept()`` with EMFILE
    long before any limit in ``gui_limits`` is reached.
    """
    source, _ = _worker_leaving(tmp_path, _HOLDS_THE_STREAM)
    _child_runner(monkeypatch, source)
    monkeypatch.setattr(jobs, "_DRAIN_SECONDS", 1.0)
    fds, threads = _open_descriptors(), threading.active_count()

    for _ in range(4):
        with pytest.raises(RuntimeError):
            jobs._run_isolated_job("validate", "schema_version: 1\n")

    assert _settled(fds, threads), (
        f"descriptors {fds} -> {_open_descriptors()}, "
        f"threads {threads} -> {threading.active_count()}"
    )


@_POSIX_ONLY
def test_the_worker_group_is_swept_so_no_descendant_outlives_the_job(
    monkeypatch, tmp_path
):
    """``subprocess.run``'s timeout ends the direct child and nothing else.

    A worker's descendant goes on holding its outputs, its memory and its CPU
    after the job is terminal, and there is no later moment that reaches it.
    The parent therefore runs the worker inside a process group it opened and
    can name, and takes the whole group back on the way out.
    """
    source, record = _worker_leaving(tmp_path, _HOLDS_THE_STREAM)
    _child_runner(monkeypatch, source)
    monkeypatch.setattr(jobs, "_DRAIN_SECONDS", 1.0)

    with pytest.raises(RuntimeError):
        jobs._run_isolated_job("validate", "schema_version: 1\n")

    descendant = int(record.read_text())
    deadline = time.monotonic() + 10.0
    while _alive(descendant) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not _alive(descendant), "the descendant outlived its job"


@_POSIX_ONLY
def test_a_frame_a_descendant_wrote_is_not_reported_as_the_workers_result(
    monkeypatch, tmp_path
):
    """The reviewer's case: ``{"layers": 666}`` reported as the science.

    The worker frames its own result and exits; a grandchild that inherited
    the same descriptor then frames a different one.  The scan's rule is that
    the last frame wins, which is right for a stream ONE process writes, so
    what has to hold is that a stream more than one process wrote is not read
    as that one process's answer at all.
    """
    source, _ = _worker_leaving(tmp_path, _FRAMES_A_RESULT)
    _child_runner(monkeypatch, source)
    monkeypatch.setattr(jobs, "_DRAIN_SECONDS", 1.5)

    with pytest.raises(RuntimeError) as failure:
        jobs._run_isolated_job("validate", "schema_version: 1\n")

    assert "did not end when the worker did" in str(failure.value)
    assert "666" not in str(failure.value)


def test_only_the_reader_thread_ever_touches_a_live_sink():
    """The other ordering of the same race, pinned where it can be seen.

    ``stdout.close()`` used to run on the parent's thread while the reader was
    still inside ``feed``: one ordering committed a descendant's frame, the
    other mutated a bytearray under itself.  Neither is reachable while the
    reader is the only thing that touches its sink.
    """
    touched: set[int] = set()

    class _Watched(gui_child.FrameScan):
        def feed(self, chunk):
            touched.add(threading.get_ident())
            super().feed(chunk)

        def close(self):
            touched.add(threading.get_ident())
            super().close()

    scan = _Watched()
    gui_child.drained_run(
        [
            sys.executable,
            "-c",
            "import sys; sys.stdin.buffer.read();"
            " sys.stdout.buffer.write(b'chatter\\n'); " + _WORKER_FRAME,
        ],
        input_bytes=b"",
        stdout=scan,
        stderr=gui_child.StreamTail(limit=1024),
        timeout=60,
    )

    assert scan.payload == b'{"status":"ok","result":{"layers":2}}'
    assert touched, "the sink was never fed at all"
    assert threading.get_ident() not in touched


def test_a_sink_that_fails_mid_stream_records_it_rather_than_losing_it():
    """``_drain`` caught ``OSError`` alone, and the sinks do not raise that.

    A bytearray disturbed while it is being sliced raises ``IndexError`` or
    ``BufferError``; both escaped the reader thread entirely, leaving a sink
    half-fed and a parent that read it as the child's whole answer.
    """

    class _Brittle(gui_child.ChildStream):
        def _consume(self, chunk):
            raise IndexError("the buffer moved under the reader")

    sink = _Brittle()
    gui_child.drained_run(
        [sys.executable, "-c", "import sys; sys.stdin.buffer.read();" + _WORKER_FRAME],
        input_bytes=b"",
        stdout=sink,
        stderr=gui_child.StreamTail(limit=1024),
        timeout=60,
    )

    assert isinstance(sink.read_error, IndexError)


@_POSIX_ONLY
def test_a_descriptor_a_live_reader_still_holds_is_never_handed_on():
    """The safeguard the old comment named and nothing tested.

    Closing the number while a reader is still inside ``os.read`` hands it to
    the next file this process opens, and the reader then copies that file
    into a job result.  The reader owns its descriptor for exactly that
    reason, and ``finish`` answering ``False`` must not be a licence to take
    it back.
    """
    release = threading.Event()

    class _Blocking(gui_child.StreamTail):
        def _consume(self, chunk):
            release.wait(10.0)
            super()._consume(chunk)

    read, write = os.pipe()
    reader = gui_child._StreamReader(read, _Blocking(limit=64))
    reader.start()
    try:
        os.write(write, b"data")
        assert reader.finish(0.5) is False, "a stuck reader reported success"

        probe = os.open(os.devnull, os.O_RDONLY)
        try:
            assert probe != read, "a live reader's descriptor was handed on"
        finally:
            os.close(probe)
    finally:
        release.set()
        os.close(write)
        reader.finish(10.0)


@pytest.mark.parametrize(
    ("written", "over"),
    [(15, False), (16, False), (17, True)],
    ids=["one-below", "exactly-at", "one-above"],
)
def test_the_child_stream_cap_bites_one_byte_past_the_agreement(
    monkeypatch, written, over
):
    """``MAX_CHILD_STREAM_BYTES`` is the largest total the parent ACCEPTS.

    Exactly that many bytes is inside the agreement; one more is outside it.
    Neither side of the boundary was tested, so ``>=`` read the same as ``>``.
    """
    monkeypatch.setattr(gui_child, "MAX_CHILD_STREAM_BYTES", 16)
    sink = gui_child.StreamTail(limit=8)

    sink.feed(b"x" * written)

    assert sink.overflowed is over
    assert sink.total == written
    # Past the cap nothing is stored at all, which is the point of the cap.
    assert (sink.tail() == b"") is over


def test_the_framed_audit_field_outranks_anything_printed_to_the_stream(
    tmp_path, monkeypatch
):
    """The frame is the channel; the stream scan is only the fallback.

    ``_audit_path`` takes the LAST matching line out of a stream anything
    holding the descriptor may write, and what it takes becomes a directory
    this long-lived parent opens and serves artefacts out of.  Where a frame
    exists it carries the value the worker read from its own whole stream
    inside its own process, and that must win outright -- not merely be
    consulted first and then overridden by a louder line.
    """
    target, marker_id = _published_target(tmp_path)
    forged = tmp_path / "somewhere-else"
    forged.mkdir()
    monkeypatch.setattr(
        jobs,
        "_run_isolated_job",
        lambda _kind, _text: {
            "exit_code": 2,
            "stdout": "",
            # A plugin printing the same line the parent scans for, last.
            "stderr": f"refused audit: {forged}\n",
            "failure_audit": str(target),
        },
    )

    with pytest.raises(ConfigError) as refusal:
        execute_job("run", YAML)

    output = getattr(refusal.value, "gui_output", None)
    assert output is not None
    assert output["target_path"] == str(target)
    assert output["marker_id"] == marker_id


def test_the_stream_scan_still_answers_where_no_frame_carried_a_field(
    tmp_path, monkeypatch
):
    """The fallback is a fallback, not dead code: the dispatcher path has no
    frame at all, and a worker frame may carry no field."""
    target, marker_id = _published_target(tmp_path)
    monkeypatch.setattr(
        jobs,
        "_run_isolated_job",
        lambda _kind, _text: {
            "exit_code": 2,
            "stdout": "",
            "stderr": f"refused audit: {target}\n",
        },
    )

    with pytest.raises(ConfigError) as refusal:
        execute_job("run", YAML)

    output = getattr(refusal.value, "gui_output", None)
    assert output is not None
    assert output["target_path"] == str(target)
    assert output["marker_id"] == marker_id


# --- the browser holds a copy of three of these limits -------------------

_RESULT_SUMMARY = (
    pathlib.Path(__file__).resolve().parents[2]
    / "src"
    / "rheplicant"
    / "gui"
    / "react"
    / "ResultSummary.tsx"
)


def _tsx_constant(source: str, name: str) -> str:
    found = re.search(rf"^const {name} = (.+);$", source, re.MULTILINE)
    assert found is not None, f"{name} is no longer declared in ResultSummary"
    return found.group(1)


def test_the_browser_copy_of_every_shared_limit_still_matches_this_one():
    """The renderer bounds again, in a language that cannot import from here.

    ``ResultSummary`` re-bounds every server string it renders, because a
    megabyte that reaches the DOM is a megabyte in the DOM whatever the server
    did.  Three of this module's constants are therefore hard-copied into
    TypeScript, and a copy nothing compares is a copy that drifts.  The marker
    matters most of the three: it is the only shared STRING, so a change to it
    makes the browser announce a truncation in one spelling while the server
    wrote another, and neither side fails.

    Read-only, deliberately: this asserts the two agree and never edits either.
    """
    source = _RESULT_SUMMARY.read_text(encoding="utf-8")

    assert json.loads(_tsx_constant(source, "TRUNCATION_MARKER")) == (
        TRUNCATION_MARKER
    )
    assert int(_tsx_constant(source, "MAX_RENDERED_CHARACTERS")) == (
        MAX_TEXT_CHARACTERS
    )
    assert int(_tsx_constant(source, "MAX_RENDERED_FINDINGS")) == (
        MAX_FINDING_COUNT
    )


#: A descendant that leaves the group before it takes the stream hostage, so
#: the sweep cannot reach it.  Not exotic: anything that daemonises itself does
#: exactly this, and the group is not the only thing that has to hold.
_ESCAPES_THE_GROUP = "import os, time\nos.setsid()\ntime.sleep(120)\n"


@_POSIX_ONLY
def test_a_descendant_outside_the_group_still_costs_no_descriptor_or_thread(
    monkeypatch, tmp_path
):
    """The two reclaims are independent, and each has to stand on its own.

    Sweeping the group ends the usual descendant, and the reader then reaches
    the end of its stream in the ordinary way.  A descendant that called
    ``setsid`` is in no group of ours, so nothing may be signalled to it -- and
    the descriptor and the thread must still come back, from the reader's own
    side, or the leak simply moves to the case the sweep cannot see.
    """
    source, record = _worker_leaving(tmp_path, _ESCAPES_THE_GROUP)
    _child_runner(monkeypatch, source)
    monkeypatch.setattr(jobs, "_DRAIN_SECONDS", 1.0)
    fds, threads = _open_descriptors(), threading.active_count()

    try:
        for _ in range(3):
            with pytest.raises(RuntimeError):
                jobs._run_isolated_job("validate", "schema_version: 1\n")

        assert _settled(fds, threads), (
            f"descriptors {fds} -> {_open_descriptors()}, "
            f"threads {threads} -> {threading.active_count()}"
        )
    finally:
        # It escaped the sweep on purpose, so this test owns ending it.
        with contextlib.suppress(OSError, ValueError):
            os.kill(int(record.read_text()), signal.SIGKILL)
