from __future__ import annotations

import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml

import _rheplicant_bootstrap.gui_worker as gui_worker
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


def test_parent_worker_adapter_passes_exact_utf8_bytes(monkeypatch):
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
    assert captured == [
        (
            [sys.executable, "-m", "_rheplicant_bootstrap.gui_worker", "validate"],
            exact_yaml.encode("utf-8", "strict"),
        )
    ]


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
