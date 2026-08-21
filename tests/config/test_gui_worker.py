from __future__ import annotations

import json
import os
import subprocess
import sys
from io import BytesIO
from types import SimpleNamespace

import pytest
import yaml

import _rheplicant_bootstrap.gui_worker as gui_worker
from _rheplicant_bootstrap.errors import ConfigError
from rheplicant.gui import jobs
from tests.config.test_config_document import synthetic_document


def _legal_outputs(tmp_path):
    return {
        "dir": str(tmp_path / "priced-preview"),
        "clobber": False,
        "write": {"arrays": {"format": "npz"}},
    }


def _priced_yaml(*, runtime=None, plugins=None, defaults=None, outputs=None):
    document = synthetic_document()
    if runtime is not None:
        document["runtime"] = {**document.get("runtime", {}), **runtime}
    if plugins is not None:
        document["plugins"] = plugins
    if defaults is not None:
        document["defaults"] = defaults
        document["observation"]["pointing"] = {"materialise": []}
    if outputs is not None:
        document["outputs"] = outputs
    return yaml.safe_dump(document, sort_keys=False)


def _invoke_worker(kind, yaml_text, *, trace=None):
    env = dict(os.environ)
    env.pop("JAX_ENABLE_X64", None)
    env.pop("JAX_PLATFORMS", None)
    if trace is not None:
        env["RHEPLICANT_TEST_PLUGIN_TRACE"] = str(trace)
    completed = subprocess.run(
        [sys.executable, "-m", "_rheplicant_bootstrap.gui_worker", kind],
        input=yaml_text.encode("utf-8", "strict"),
        capture_output=True,
        env=env,
        check=False,
    )
    prefix = b"\x1eRHEPLICANT_GUI_JOB "
    encoded = completed.stdout.rsplit(prefix, 1)[1].split(b"\n", 1)[0]
    return completed, json.loads(encoded.decode("utf-8", "strict"))


def test_validate_worker_establishes_runtime_before_plugins_in_order(tmp_path):
    trace = tmp_path / "plugins.txt"
    text = _priced_yaml(
        runtime={"jax_enable_x64": True},
        plugins=[
            "tests.config.gui_worker_plugin_a",
            "tests.config.gui_worker_plugin_b",
        ],
        defaults=["rhino_v1"],
        outputs=_legal_outputs(tmp_path),
    )

    completed, frame = _invoke_worker("validate", text, trace=trace)

    assert completed.returncode == 0
    assert frame == {
        "status": "ok",
        "result": {"findings": [], "layers": 2},
    }
    assert trace.read_text(encoding="utf-8").splitlines() == [
        "a:x64=True",
        "b:x64=True",
    ]
    assert completed.stderr.decode().count("trusted plugin/python code") == 1


def test_validate_worker_frames_plugin_refusal_without_protocol_noise(tmp_path):
    completed, frame = _invoke_worker(
        "validate",
        _priced_yaml(plugins=["does_not_exist.plan6a"]),
    )

    assert completed.returncode == 0
    assert frame["status"] == "refused"
    assert "does_not_exist.plan6a" in frame["message"]


def test_validate_worker_warns_once_for_a_python_target():
    document = synthetic_document()
    document["model"]["gain"] = {
        "python": "rheplicant.radio:GainOperator",
        "gain": {"value": 1.1, "unit": "dimensionless"},
    }

    completed, frame = _invoke_worker(
        "validate", yaml.safe_dump(document, sort_keys=False)
    )

    assert frame["status"] == "ok"
    assert completed.stderr.decode().count("trusted plugin/python code") == 1


def test_validation_worker_closes_environment_after_success(monkeypatch):
    class RecordingExecution:
        def __init__(self):
            self.document = SimpleNamespace(
                layers=(
                    SimpleNamespace(
                        layer=SimpleNamespace(prefix=""),
                        configured=SimpleNamespace(
                            report=SimpleNamespace(findings=())
                        ),
                    ),
                )
            )
            self.close_calls = 0

        def close(self):
            self.close_calls += 1

    execution = RecordingExecution()
    monkeypatch.setattr(gui_worker, "_prepared_config", lambda _text: object())
    monkeypatch.setattr(
        gui_worker,
        "prepare_execution_environment",
        lambda *_args, **_kwargs: execution,
    )

    assert gui_worker._run_validation("exact yaml") == {
        "findings": [],
        "layers": 1,
    }
    assert execution.close_calls == 1


def test_parent_worker_adapter_preserves_exact_job_bytes(monkeypatch):
    exact_yaml = "schema_version: 1\n# café and formatting stay exact\n"
    captured = []

    def fake_run(arguments, **kwargs):
        captured.append((arguments, kwargs["input"]))
        frame = {"status": "ok", "result": {"findings": [], "layers": 1}}
        return subprocess.CompletedProcess(
            arguments,
            0,
            stdout=(
                b"plugin stdout noise\n\x1eRHEPLICANT_GUI_JOB "
                + json.dumps(frame).encode("utf-8")
                + b"\n"
            ),
            stderr=b"",
        )

    monkeypatch.setattr(jobs.subprocess, "run", fake_run)

    assert jobs.run_priced_validation(exact_yaml) == {
        "findings": [],
        "layers": 1,
    }
    preview = jobs.forward_preview_document(exact_yaml)
    jobs.run_forward_preview(preview)
    jobs._run_isolated_job("run", exact_yaml)
    assert captured == [
        (
            [sys.executable, "-m", "_rheplicant_bootstrap.gui_worker", "validate"],
            exact_yaml.encode("utf-8", "strict"),
        ),
        (
            [
                sys.executable,
                "-m",
                "_rheplicant_bootstrap.gui_worker",
                "preview_forward",
            ],
            preview.encode("utf-8", "strict"),
        ),
        (
            [sys.executable, "-m", "_rheplicant_bootstrap.gui_worker", "run"],
            exact_yaml.encode("utf-8", "strict"),
        ),
    ]
    assert yaml.safe_load(preview)["runs"] == [
        {"name": "preview-forward", "kind": "forward"}
    ]


@pytest.mark.parametrize(
    "failure",
    [ConfigError("refused"), RuntimeError("boom")],
)
def test_priced_job_closes_environment_on_every_terminal_failure(
    monkeypatch, failure
):
    class RecordingExecution:
        def __init__(self):
            self.close_calls = 0

        @property
        def document(self):
            raise failure

        def close(self):
            self.close_calls += 1

    execution = RecordingExecution()
    monkeypatch.setattr(gui_worker, "_prepared_config", lambda _text: object())
    monkeypatch.setattr(
        gui_worker,
        "prepare_execution_environment",
        lambda *_args, **_kwargs: execution,
    )

    with pytest.raises(type(failure), match=str(failure)):
        gui_worker._run_validation("exact yaml")

    assert execution.close_calls == 1


@pytest.mark.parametrize(
    "failure",
    [None, ConfigError("refused"), RuntimeError("boom")],
)
def test_forward_preview_closes_environment_on_every_terminal_path(
    monkeypatch, failure
):
    class RecordingOrchestration:
        @staticmethod
        def execute_prepared(_document, *, trace):
            assert trace == "trace"
            return "record"

    class RecordingExecution:
        orchestration = RecordingOrchestration()
        document = SimpleNamespace(
            layers=(SimpleNamespace(configured="configured"),)
        )
        trace = "trace"

        def __init__(self):
            self.close_calls = 0

        def close(self):
            self.close_calls += 1

    prepared = SimpleNamespace(
        source=SimpleNamespace(
            layered_document={"model": {"adc": {"n_bits": 4}}}
        )
    )
    execution = RecordingExecution()
    monkeypatch.setattr(gui_worker, "_prepared_config", lambda _text: prepared)
    monkeypatch.setattr(
        gui_worker,
        "prepare_execution_environment",
        lambda *_args, **_kwargs: execution,
    )

    def bounded(record, *, configured, adc):
        assert (record, configured, adc) == (
            "record",
            "configured",
            {"n_bits": 4},
        )
        if failure is not None:
            raise failure
        return {"waterfall": {}}

    monkeypatch.setattr(gui_worker, "_bounded_preview_result", bounded)

    if failure is None:
        assert gui_worker._run_forward_preview("exact yaml") == {
            "waterfall": {}
        }
    else:
        with pytest.raises(type(failure), match=str(failure)):
            gui_worker._run_forward_preview("exact yaml")

    assert execution.close_calls == 1


@pytest.mark.parametrize(
    ("kind", "runner_name"),
    [
        ("validate", "validate"),
        ("preview_forward", "preview"),
        ("run", "formal"),
        ("compare", "formal"),
        ("benchmark", "formal"),
    ],
)
def test_worker_main_routes_every_job_kind(monkeypatch, kind, runner_name):
    calls = []

    def runner(name):
        def run(yaml_text):
            calls.append((name, yaml_text))
            return {"runner": name}

        return run

    monkeypatch.setattr(gui_worker, "_run_validation", runner("validate"))
    monkeypatch.setattr(gui_worker, "_run_forward_preview", runner("preview"))
    monkeypatch.setattr(gui_worker, "_run_formal", runner("formal"))
    monkeypatch.setattr(
        gui_worker,
        "_write_frame",
        lambda frame: calls.append(("frame", frame)),
    )
    monkeypatch.setattr(
        gui_worker.sys,
        "stdin",
        SimpleNamespace(buffer=BytesIO(b"exact yaml bytes")),
    )

    assert gui_worker.main([kind]) == 0
    assert calls == [
        (runner_name, "exact yaml bytes"),
        ("frame", {"status": "ok", "result": {"runner": runner_name}}),
    ]


def test_formal_worker_calls_plan4_dispatcher_with_exact_bytes(monkeypatch):
    from _rheplicant_bootstrap import entry

    calls = []

    def dispatcher(command, source, *, stdout, stderr):
        calls.append((command, source.input_bytes, stdout, stderr))
        stdout.write("done")
        stderr.write("detail")
        return 7

    monkeypatch.setattr(entry, "dispatch_request", dispatcher)

    assert gui_worker._run_formal("schema_version: 1\n# café\n") == {
        "exit_code": 7,
        "stdout": "done",
        "stderr": "detail",
    }
    assert calls[0][:2] == (
        "run",
        "schema_version: 1\n# café\n".encode("utf-8", "strict"),
    )


@pytest.mark.parametrize(
    "stdout",
    [
        b"no reserved prefix",
        b"\x1eRHEPLICANT_GUI_JOB \xff\n",
        b"\x1eRHEPLICANT_GUI_JOB {not json}\n",
        b'\x1eRHEPLICANT_GUI_JOB {"status":"unknown"}\n',
        b'\x1eRHEPLICANT_GUI_JOB {"status":"ok","result":[]}\n',
        b'\x1eRHEPLICANT_GUI_JOB {"status":"refused"}\n',
        b'\x1eRHEPLICANT_GUI_JOB {"status":"refused","message":1}\n',
        b'\x1eRHEPLICANT_GUI_JOB {"status":"error","message":"boom"}\n',
        b'\x1eRHEPLICANT_GUI_JOB {"status":"error",'
        b'"exception_type":1,"message":"boom"}\n',
    ],
)
def test_parent_rejects_malformed_worker_frames_with_stderr_context(
    monkeypatch, stdout
):
    def fake_run(arguments, **_kwargs):
        return subprocess.CompletedProcess(
            arguments,
            0,
            stdout=stdout,
            stderr=b"bounded worker detail",
        )

    monkeypatch.setattr(jobs.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="worker stderr: bounded worker detail"):
        jobs._run_isolated_job("validate", "schema_version: 1\n")


def test_parent_uses_the_last_raw_bytes_frame(monkeypatch):
    first = b'\x1eRHEPLICANT_GUI_JOB {"status":"refused","message":"old"}\n'
    last = b'\x1eRHEPLICANT_GUI_JOB {"status":"ok","result":{"layers":2}}\n'

    def fake_run(arguments, **_kwargs):
        return subprocess.CompletedProcess(
            arguments,
            0,
            stdout=first + b"plugin noise\n" + last,
            stderr=b"",
        )

    monkeypatch.setattr(jobs.subprocess, "run", fake_run)

    assert jobs._run_isolated_job("validate", "schema_version: 1\n") == {
        "layers": 2
    }


def test_parent_rejects_a_nonzero_worker_exit_before_parsing_frames(monkeypatch):
    def fake_run(arguments, **_kwargs):
        return subprocess.CompletedProcess(
            arguments,
            7,
            stdout=b'\x1eRHEPLICANT_GUI_JOB {"status":"ok","result":{}}\n',
            stderr=b"worker process failed",
        )

    monkeypatch.setattr(jobs.subprocess, "run", fake_run)

    with pytest.raises(
        RuntimeError, match="exited 7: worker process failed"
    ):
        jobs._run_isolated_job("validate", "schema_version: 1\n")


def test_parent_converts_a_valid_worker_refusal_to_config_error(monkeypatch):
    def fake_run(arguments, **_kwargs):
        return subprocess.CompletedProcess(
            arguments,
            0,
            stdout=(
                b'\x1eRHEPLICANT_GUI_JOB {"status":"refused",'
                b'"message":"scientific refusal"}\n'
            ),
            stderr=b"",
        )

    monkeypatch.setattr(jobs.subprocess, "run", fake_run)

    with pytest.raises(jobs.ConfigError, match="scientific refusal"):
        jobs._run_isolated_job("validate", "schema_version: 1\n")


def test_worker_error_type_reaches_the_job_store(monkeypatch):
    def fake_run(arguments, **_kwargs):
        return subprocess.CompletedProcess(
            arguments,
            0,
            stdout=(
                b'\x1eRHEPLICANT_GUI_JOB {"status":"error",'
                b'"exception_type":"WorkerValueError","message":"boom"}\n'
            ),
            stderr=b"",
        )

    monkeypatch.setattr(jobs.subprocess, "run", fake_run)
    store = jobs.JobStore(id_factory=lambda: "job-error")
    row = store.submit("session", "validate", 1, "schema_version: 1\n")

    store.run(row.job_id, jobs.execute_job)

    finished = store.get(row.job_id)
    assert finished.status == "error"
    assert finished.message == "WorkerValueError: boom"
