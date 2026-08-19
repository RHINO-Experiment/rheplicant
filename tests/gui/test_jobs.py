from __future__ import annotations

from dataclasses import FrozenInstanceError
from io import StringIO

import pytest
import yaml

from rheplicant.config import ConfigError
from rheplicant.gui.jobs import (
    JobStore,
    _array_summary,
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


def test_real_priced_validation_and_forward_preview_cross_plan4_orchestration():
    text = yaml.safe_dump(synthetic_document(), sort_keys=False)

    validated = run_priced_validation(text)
    previewed = run_forward_preview(forward_preview_document(text))

    assert validated == {"findings": [], "layers": 2}
    assert previewed["waterfall"]["shape"] == [16, 8]
    assert len(previewed["waterfall"]["values"]) == 16
    assert all(len(row) == 8 for row in previewed["waterfall"]["values"])
    assert previewed["saturated_fraction"] is None


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
