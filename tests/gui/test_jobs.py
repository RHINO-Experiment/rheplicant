from __future__ import annotations

from dataclasses import FrozenInstanceError
from io import StringIO
from types import SimpleNamespace

import pytest
import yaml

from _rheplicant_bootstrap.gui_worker import _array_summary
from rheplicant.config import ConfigError
from rheplicant.gui.jobs import (
    JobStore,
    execute_job,
    forward_preview_document,
    run_forward_preview,
    run_priced_validation,
    yaml_digest,
)
from tests.config.test_config_document import synthetic_document

YAML = """\
schema_version: 1
runtime: {jax_enable_x64: true}
model: {}
runs:
  - {name: fit, kind: optimize, n_steps: 2}
  - {name: comparison, kind: compare, variants: [base, alternate]}
  - {name: timing, kind: benchmark, variants: [base, alternate]}
outputs: {dir: results/example, stdout: none}
"""


def test_job_store_returns_opaque_ids_and_marks_bound_results_stale():
    store = JobStore(id_factory=iter(("job-a", "job-b")).__next__)
    first = store.submit("session-1", "validate", 3, YAML)
    second = store.submit("session-1", "preview_forward", 3, YAML)

    assert first.job_id == "job-a"
    assert second.job_id == "job-b"
    assert first.status == "queued"
    current = store.project("session-1", yaml_digest(YAML))
    assert [row.stale for row in current] == [False, False]
    changed = store.project("session-1", yaml_digest(YAML + "# edit\n"))
    assert [row.stale for row in changed] == [True, True]
    with pytest.raises(FrozenInstanceError):
        first.status = "succeeded"  # type: ignore[misc]


def test_runner_completion_and_refusal_are_retained_without_mutating_yaml():
    store = JobStore(id_factory=lambda: "job-a")
    row = store.submit("session-1", "validate", 0, YAML)
    store.run(row.job_id, lambda _kind, _yaml: {"findings": [{"check": "C13"}]})

    completed = store.get("job-a")
    assert completed.status == "succeeded"
    assert completed.result == {"findings": [{"check": "C13"}]}
    assert completed.yaml_digest == yaml_digest(YAML)

    refused_store = JobStore(id_factory=lambda: "job-r")
    refused = refused_store.submit("session-1", "validate", 0, YAML)
    refused_store.run(
        refused.job_id,
        lambda _kind, _yaml: (_ for _ in ()).throw(ConfigError("priced refusal")),
    )
    assert refused_store.get("job-r").status == "refused"
    assert refused_store.get("job-r").message == "priced refusal"
    assert refused_store.get("job-r").result is None


@pytest.mark.parametrize("kind", ["run", "compare", "benchmark"])
def test_unsafe_formal_refusal_projects_only_the_closed_output_state(
    kind,
    monkeypatch,
):
    from rheplicant.gui.outputs import OutputState

    monkeypatch.setattr(
        "rheplicant.gui.outputs._inspection_state",
        lambda _inspection: OutputState(
            "blocked_unsafe",
            "Closed unsafe output projection.",
        ),
    )

    dispatched = []

    def refused(_command, _source, *, stdout, stderr):
        dispatched.append(True)
        stderr.write("formal refusal without an output classification")
        return 2

    store = JobStore(id_factory=lambda: f"job-unsafe-{kind}")
    row = store.submit("session-1", kind, 0, YAML)
    store.run(
        row.job_id,
        lambda kind, text: execute_job(kind, text, dispatcher=refused),
    )

    finished = store.get(row.job_id)
    assert finished.status == "refused"
    assert finished.message == "Closed unsafe output projection."
    assert dispatched == []
    assert finished.result == {
        "output": {
            "state": "blocked_unsafe",
            "state_message": "Closed unsafe output projection.",
            "target_path": "/rheplicant-gui/results/example",
        }
    }


def test_projector_failure_cannot_escape_generic_refusal_or_retain_yaml(monkeypatch):
    monkeypatch.setattr(
        "rheplicant.gui.outputs.project_output_workflow",
        lambda _yaml: (_ for _ in ()).throw(RuntimeError("projection failed")),
    )
    store = JobStore(id_factory=lambda: "job-projector-error")
    row = store.submit("session-1", "run", 0, YAML)

    def refused(_command, _source, *, stdout, stderr):
        stderr.write("original refusal")
        return 2

    store.run(
        row.job_id,
        lambda kind, text: execute_job(kind, text, dispatcher=refused),
    )

    finished = store.get(row.job_id)
    assert finished.status == "refused"
    assert finished.message == "original refusal"
    assert finished.result is None
    assert row.job_id not in store._yaml


def test_undeclared_compare_is_not_relabelled_by_an_unsafe_output_projection(
    monkeypatch,
):
    calls = []

    def unsafe_projection(_yaml):
        calls.append(True)
        return SimpleNamespace(
            state="blocked_unsafe",
            state_message="Unrelated unsafe projection.",
            target_path="/unrelated",
        )

    monkeypatch.setattr(
        "rheplicant.gui.outputs.project_output_workflow",
        unsafe_projection,
    )
    document = yaml.safe_load(YAML)
    document["runs"] = [{"name": "fit", "kind": "optimize", "n_steps": 2}]
    run_only_yaml = yaml.safe_dump(document, sort_keys=False)
    store = JobStore(id_factory=lambda: "job-no-compare")
    row = store.submit("session-1", "compare", 0, run_only_yaml)
    store.run(row.job_id, execute_job)

    finished = store.get(row.job_id)
    assert calls == []
    assert finished.status == "refused"
    assert finished.message == "The document declares no 'compare' exit to run."
    assert finished.result is None


def test_non_unsafe_formal_projection_does_not_enrich_a_dispatcher_refusal(
    monkeypatch,
):
    monkeypatch.setattr(
        "rheplicant.gui.outputs.project_output_workflow",
        lambda _yaml: SimpleNamespace(
            state="ready_new",
            state_message="Ready.",
            target_path="/ready",
        ),
    )

    def refused(_command, _source, *, stdout, stderr):
        stderr.write("ordinary formal refusal")
        return 2

    store = JobStore(id_factory=lambda: "job-ready-refusal")
    row = store.submit("session-1", "run", 0, YAML)
    store.run(
        row.job_id,
        lambda kind, text: execute_job(kind, text, dispatcher=refused),
    )

    finished = store.get(row.job_id)
    assert finished.status == "refused"
    assert finished.message == "ordinary formal refusal"
    assert finished.result is None


@pytest.mark.parametrize("kind", ["validate", "preview_forward"])
def test_non_formal_jobs_never_enter_the_output_safety_bridge(kind, monkeypatch):
    calls = []

    def ready_projection(_yaml):
        calls.append(True)
        return SimpleNamespace(
            state="ready_new",
            state_message="Ready.",
            target_path="/ready",
        )

    monkeypatch.setattr(
        "rheplicant.gui.outputs.project_output_workflow",
        ready_projection,
    )

    def refused(_yaml):
        raise ConfigError("non-formal refusal")

    store = JobStore(id_factory=lambda: f"job-{kind}")
    row = store.submit("session-1", kind, 0, YAML)
    store.run(
        row.job_id,
        lambda selected, text: execute_job(
            selected,
            text,
            validator=refused,
            forwarder=refused,
        ),
    )

    finished = store.get(row.job_id)
    assert calls == []
    assert finished.status == "refused"
    assert finished.message == "non-formal refusal"
    assert finished.result is None


def test_runner_retains_a_worker_supplied_terminal_exception_name():
    store = JobStore(id_factory=lambda: "job-error")
    row = store.submit("session-1", "validate", 0, YAML)
    error = RuntimeError("worker failed")
    error.job_exception_type = "WorkerValueError"

    store.run(
        row.job_id,
        lambda _kind, _yaml: (_ for _ in ()).throw(error),
    )

    finished = store.get(row.job_id)
    assert finished.status == "error"
    assert finished.message == "WorkerValueError: worker failed"


def test_forward_preview_replaces_every_declared_exit_with_one_forward_only():
    preview = yaml.safe_load(forward_preview_document(YAML))

    assert preview["runs"] == [{"name": "preview-forward", "kind": "forward"}]
    assert all(
        row["kind"] not in {"optimize", "nuts", "plan.estimate", "plan.sample", "npe"}
        for row in preview["runs"]
    )
    assert yaml.safe_load(YAML)["runs"][0]["kind"] == "optimize"


@pytest.mark.parametrize("kind", ["run", "compare", "benchmark"])
def test_formal_jobs_use_the_plan4_dispatcher_with_exact_original_bytes(kind):
    calls = []

    def dispatcher(command, source, *, stdout, stderr):
        calls.append((command, source, stdout, stderr))
        stdout.write("done")
        return 0

    found = execute_job(kind, YAML, dispatcher=dispatcher)

    assert len(calls) == 1
    command, source, stdout, stderr = calls[0]
    assert command == "run"
    assert source.input_bytes == YAML.encode("utf-8")
    assert isinstance(stdout, StringIO)
    assert isinstance(stderr, StringIO)
    assert found["exit_code"] == 0
    assert found["stdout"] == "done"


@pytest.mark.parametrize("kind", ["compare", "benchmark"])
def test_named_scientific_job_refuses_when_the_document_declares_no_such_exit(kind):
    document = yaml.safe_load(YAML)
    document["runs"] = [{"name": "forward", "kind": "forward"}]
    calls = []

    with pytest.raises(ConfigError, match=f"no {kind!r} exit"):
        execute_job(
            kind,
            yaml.safe_dump(document, sort_keys=False),
            dispatcher=lambda *_args, **_kwargs: calls.append(True) or 0,
        )
    assert calls == []


def test_validate_and_preview_use_separate_priced_orchestration_seams():
    calls = []

    def validate(yaml_text):
        calls.append(("validate", yaml_text))
        return {"findings": []}

    def preview(yaml_text):
        calls.append(("preview", yaml.safe_load(yaml_text)["runs"]))
        return {"waterfall": {"shape": [8, 4]}}

    assert execute_job("validate", YAML, validator=validate) == {"findings": []}
    assert execute_job("preview_forward", YAML, forwarder=preview) == {
        "waterfall": {"shape": [8, 4]}
    }
    assert calls[0] == ("validate", YAML)
    assert calls[1] == (
        "preview",
        [{"name": "preview-forward", "kind": "forward"}],
    )


def test_job_identity_keeps_original_bytes_and_preview_detaches_only_runs():
    exact_yaml = YAML
    store = JobStore(id_factory=lambda: "job-1")

    submitted = store.submit("session", "preview_forward", 7, exact_yaml)

    assert submitted.yaml_digest == yaml_digest(exact_yaml)
    captured = []
    execute_job(
        "preview_forward",
        exact_yaml,
        forwarder=lambda text: captured.append(text) or {"waterfall": {}},
    )
    preview_document = yaml.safe_load(captured[0])
    assert preview_document["runs"] == [
        {"name": "preview-forward", "kind": "forward"}
    ]
    assert "kind: optimize" in exact_yaml

    execute_job(
        "validate",
        exact_yaml,
        validator=lambda text: captured.append(text) or {"findings": []},
    )
    assert captured[1] == exact_yaml


@pytest.mark.parametrize("kind", ["run", "compare", "benchmark"])
def test_default_formal_jobs_use_the_clean_worker_with_exact_bytes(
    monkeypatch, kind
):
    calls = []

    def isolated(worker_kind, yaml_text):
        calls.append((worker_kind, yaml_text))
        return {"exit_code": 0, "stdout": "done", "stderr": ""}

    monkeypatch.setattr("rheplicant.gui.jobs._run_isolated_job", isolated)

    found = execute_job(kind, YAML)

    assert calls == [(kind, YAML)]
    assert found["exit_code"] == 0
    assert found["stdout"] == "done"


def test_real_priced_validation_and_forward_preview_cross_plan4_orchestration():
    text = yaml.safe_dump(synthetic_document(), sort_keys=False)

    validated = run_priced_validation(text)
    previewed = run_forward_preview(forward_preview_document(text))

    assert validated == {"findings": [], "layers": 2}
    assert previewed["waterfall"]["shape"] == [16, 8]
    assert len(previewed["waterfall"]["values"]) == 16
    assert all(len(row) == 8 for row in previewed["waterfall"]["values"])
    assert previewed["saturated_fraction"] is None


def test_real_priced_validation_accepts_plan4_preset_and_outputs(tmp_path):
    document = synthetic_document()
    document["defaults"] = ["rhino_v1"]
    document["observation"]["pointing"] = {"materialise": []}
    document["outputs"] = {
        "dir": str(tmp_path / "priced-preview"),
        "clobber": False,
        "write": {"arrays": {"format": "npz"}},
    }
    text = yaml.safe_dump(document, sort_keys=False)

    assert run_priced_validation(text) == {"findings": [], "layers": 2}


def test_real_forward_preview_accepts_plan4_preset_and_outputs(tmp_path):
    document = synthetic_document()
    document["defaults"] = ["rhino_v1"]
    document["observation"]["pointing"] = {"materialise": []}
    document["outputs"] = {
        "dir": str(tmp_path / "priced-preview"),
        "clobber": False,
        "write": {"arrays": {"format": "npz"}},
    }
    text = yaml.safe_dump(document, sort_keys=False)

    found = run_forward_preview(forward_preview_document(text))

    assert found["waterfall"]["shape"] == [16, 8]
    assert len(found["waterfall"]["values"]) == 16


def test_complex_taps_are_summarised_by_magnitude_without_losing_the_dtype():
    import numpy as np

    found = _array_summary(np.array([3 + 4j, 5 + 12j]), include_values=False)

    assert found == {
        "shape": [2],
        "dtype": "complex128",
        "statistic": "magnitude",
        "minimum": 5.0,
        "maximum": 13.0,
        "mean": 9.0,
    }


def test_refused_job_retains_its_published_audit_bundle_links(tmp_path):
    import json

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

    def dispatcher(_command, _source, *, stdout, stderr):
        stderr.write(f"refused: priced refusal\nrefused audit: {target}\n")
        return 2

    store = JobStore(id_factory=lambda: "job-refused")
    row = store.submit("session-1", "run", 0, YAML)
    store.run(
        row.job_id,
        lambda kind, text: execute_job(kind, text, dispatcher=dispatcher),
    )

    finished = store.get(row.job_id)
    assert finished.status == "refused"
    assert finished.result["output"]["target_path"] == str(target)
    assert finished.result["output"]["marker_id"] == marker_id
    assert finished.result["output"]["target_device"] == target.stat().st_dev
    assert finished.result["output"]["target_inode"] == target.stat().st_ino
    assert finished.result["output"]["audit_files"] == ["config.resolved.yaml"]


# --- Finding 7: atomic duplicate job suppression -------------------------


def test_an_identical_active_job_is_refused_rather_than_queued_twice():
    """The React guard is a convenience; the backend owns the invariant."""
    store = JobStore(id_factory=iter(("job-a", "job-b")).__next__)
    first = store.submit("session-1", "validate", 3, YAML)

    with pytest.raises(ConfigError) as refusal:
        store.submit("session-1", "validate", 3, YAML)

    message = str(refusal.value)
    assert "'validate'" in message
    assert "revision 3" in message
    assert yaml_digest(YAML)[:12] in message
    assert yaml_digest(YAML) not in message
    assert "schema_version" not in message
    assert len(message) <= 200
    assert [row.job_id for row in store.project("session-1", yaml_digest(YAML))] == [
        first.job_id
    ]


def test_only_the_exact_kind_revision_document_and_session_is_a_duplicate():
    ids = iter(("job-a", "job-b", "job-c", "job-d", "job-e"))
    store = JobStore(id_factory=lambda: next(ids))

    store.submit("session-1", "validate", 3, YAML)
    store.submit("session-1", "preview_forward", 3, YAML)
    store.submit("session-1", "validate", 4, YAML)
    store.submit("session-1", "validate", 3, YAML + "# edit\n")
    store.submit("session-2", "validate", 3, YAML)

    assert len(store.project("session-1", yaml_digest(YAML))) == 4
    assert len(store.project("session-2", yaml_digest(YAML))) == 1


def test_a_running_job_still_blocks_an_identical_submission():
    ids = iter(("job-a", "job-b"))
    store = JobStore(id_factory=lambda: next(ids))
    row = store.submit("session-1", "validate", 3, YAML)
    refusals = []

    def runner(_kind, _yaml):
        try:
            store.submit("session-1", "validate", 3, YAML)
        except ConfigError as error:
            refusals.append(str(error))
        return {"findings": []}

    store.run(row.job_id, runner)

    assert len(refusals) == 1
    assert "already running" in refusals[0]
    assert len(store.project("session-1", yaml_digest(YAML))) == 1


@pytest.mark.parametrize(
    ("outcome", "runner"),
    [
        ("succeeded", lambda _kind, _yaml: {"findings": []}),
        (
            "refused",
            lambda _kind, _yaml: (_ for _ in ()).throw(ConfigError("priced refusal")),
        ),
        (
            "error",
            lambda _kind, _yaml: (_ for _ in ()).throw(RuntimeError("boom")),
        ),
    ],
)
def test_an_identical_re_run_is_allowed_once_the_first_job_is_terminal(
    outcome, runner
):
    ids = iter(("job-a", "job-b"))
    store = JobStore(id_factory=lambda: next(ids))
    first = store.submit("session-1", "validate", 3, YAML)
    store.run(first.job_id, runner)
    assert store.get("job-a").status == outcome

    second = store.submit("session-1", "validate", 3, YAML)

    assert second.job_id == "job-b"
    assert second.status == "queued"
    assert len(store.project("session-1", yaml_digest(YAML))) == 2


def test_concurrent_identical_submissions_create_exactly_one_job():
    """The check and the insert must be ONE lock acquisition.

    Passing is deterministic: while the check and the insert share a single
    acquisition, no interleaving can admit two identical jobs, so this test
    cannot fail unless that invariant is really broken.  Catching a check
    taken *outside* the insert's lock is not deterministic -- it turns on
    which thread wins the lock in the gap the released check opens -- so the
    store is loaded first to widen the scan held under the lock, the
    interpreter is told to change hands often, and the race is run many times.
    """
    import itertools
    import sys
    import threading

    workers, rounds, seeded = 24, 60, 1000
    counter = itertools.count()
    guard = threading.Lock()

    def identifier():
        with guard:
            return f"job-{next(counter)}"

    def one_race():
        store = JobStore(id_factory=identifier)
        for index in range(seeded):
            store.submit(f"other-{index}", "validate", 3, YAML)
        barrier = threading.Barrier(workers)
        accepted: list[object] = []
        unexpected: list[BaseException] = []

        def submit():
            try:
                barrier.wait(timeout=30)
                accepted.append(store.submit("session-1", "validate", 3, YAML))
            except ConfigError:
                pass
            except BaseException as error:  # noqa: BLE001 -- reported by the test
                unexpected.append(error)

        threads = [threading.Thread(target=submit) for _ in range(workers)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)
        assert unexpected == []
        return len(accepted), len(store.project("session-1", yaml_digest(YAML)))

    was = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        outcomes = [one_race() for _ in range(rounds)]
    finally:
        sys.setswitchinterval(was)

    assert outcomes == [(1, 1)] * rounds


# --- a job reaches a terminal status even when its failure fails ---------


def _stranding_refusal():
    """A refusal whose findings cannot be read: ``_error_result`` raises."""
    error = ConfigError("priced refusal")
    error.report = SimpleNamespace(findings=3)
    return error


class _UnreadableResult(dict):
    """A success whose mapping cannot be walked: ``bounded_result`` raises."""

    def items(self):
        raise TypeError("'int' object is not iterable")


@pytest.mark.parametrize(
    "runner",
    [
        pytest.param(
            lambda _kind, _yaml: (_ for _ in ()).throw(_stranding_refusal()),
            id="refusal-whose-findings-cannot-be-bounded",
        ),
        pytest.param(
            lambda _kind, _yaml: _UnreadableResult(),
            id="result-that-cannot-be-bounded",
        ),
    ],
)
def test_a_failure_inside_a_failure_still_leaves_the_job_terminal(runner):
    """Recording the outcome is inside the guarantee, not after it.

    Before duplicate suppression a stranded ``running`` row was cosmetic.
    Now ``_active_duplicate`` reads it and no route cancels or deletes a job,
    so a row left running refuses every identical resubmission for the life
    of the process.
    """
    ids = iter(("job-a", "job-b"))
    store = JobStore(id_factory=lambda: next(ids))
    row = store.submit("session-1", "validate", 3, YAML)

    with pytest.raises(TypeError):
        store.run(row.job_id, runner)

    finished = store.get("job-a")
    assert finished.status == "error"
    assert finished.message.startswith("the job recorded no terminal result: ")
    assert "TypeError" in finished.message
    resubmitted = store.submit("session-1", "validate", 3, YAML)
    assert resubmitted.status == "queued"


def test_a_base_exception_leaves_the_job_terminal_and_still_propagates():
    ids = iter(("job-a", "job-b"))
    store = JobStore(id_factory=lambda: next(ids))
    row = store.submit("session-1", "validate", 3, YAML)

    with pytest.raises(KeyboardInterrupt):
        store.run(
            row.job_id,
            lambda _kind, _yaml: (_ for _ in ()).throw(KeyboardInterrupt()),
        )

    finished = store.get("job-a")
    assert finished.status == "error"
    assert "KeyboardInterrupt" in finished.message
    assert store.submit("session-1", "validate", 3, YAML).status == "queued"


def test_a_terminal_record_is_written_once_and_keeps_the_first_outcome():
    """The guarantee adds a record; it must not add a second one."""
    store = JobStore(id_factory=lambda: "job-a")
    row = store.submit("session-1", "validate", 3, YAML)

    store.run(row.job_id, lambda _kind, _yaml: {"findings": []})

    finished = store.get("job-a")
    assert finished.status == "succeeded"
    assert finished.result == {"findings": []}
    assert finished.message is None
