"""What a long-lived job store keeps, and what it therefore has to give up.

``MAX_RESULT_BYTES`` bounds ONE job result.  Nothing bounded how many results
one process retained: measured, 200 finished jobs held 404.5 MB that was never
freed, and a single poll of a session with that history built a 200.5 MB body.
Both costs are the same product -- retained jobs times the per-result budget --
so both are closed by bounding the first factor and by not copying the results
that remain.

The bound has a price, and it is paid here rather than hidden: an audit-bundle
link into a job the store has retired stops resolving.  These tests pin the
refusal that replaces it, and pin that it is told apart from a job id that was
never issued at all.
"""

from __future__ import annotations

import json

import pytest
import yaml

from _rheplicant_bootstrap.gui_limits import MAX_RETAINED_JOBS
from rheplicant.gui.jobs import JobStore, yaml_digest
from tests.config.preflight_helpers import preflight_document

SESSION = "session-1"

YAML = """\
schema_version: 1
runtime: {jax_enable_x64: true}
model: {}
runs:
  - {name: fit, kind: optimize, n_steps: 2}
outputs: {dir: results/example, stdout: none}
"""


def _store() -> JobStore:
    counter = iter(f"job-{index:04d}" for index in range(10_000))
    return JobStore(id_factory=counter.__next__)


def _finish(store: JobStore, revision: int) -> str:
    """Submit one job that differs from every other, and run it to terminal."""
    row = store.submit(SESSION, "validate", revision, YAML)
    store.run(row.job_id, lambda _kind, _yaml, at=revision: {"revision": at})
    return row.job_id


def _held(store: JobStore) -> list[str]:
    return [row.job_id for row in store.project(SESSION, yaml_digest(YAML))]


def test_the_store_retains_no_more_terminal_jobs_than_the_bound():
    """The defect: ``_jobs`` only ever grew, for the life of the process."""
    store = _store()

    for revision in range(MAX_RETAINED_JOBS + 9):
        _finish(store, revision)

    assert len(_held(store)) == MAX_RETAINED_JOBS


def test_eviction_takes_the_oldest_terminal_record_first():
    """Oldest first, so what a reader is most likely to still want survives."""
    store = _store()
    issued = [_finish(store, revision) for revision in range(MAX_RETAINED_JOBS + 5)]

    assert _held(store) == issued[5:]
    for retired in issued[:5]:
        with pytest.raises(KeyError):
            store.get(retired)


def test_an_active_job_is_never_evicted_however_old_it_is():
    """The one record eviction must not reach, placed where it is reached first.

    A ``queued`` job owns its action: ``_active_duplicate`` reads it, and a
    job that vanishes before it runs is a job whose result nothing can ever
    collect.  Here it is the OLDEST row in the store, so an eviction that
    merely took the front of the mapping would take exactly this one.
    """
    store = _store()
    waiting = store.submit(SESSION, "run", -1, YAML).job_id
    for revision in range(MAX_RETAINED_JOBS + 4):
        _finish(store, revision)

    held = _held(store)
    assert waiting in held, "an active job was evicted"
    assert store.get(waiting).status == "queued"
    assert len(held) == MAX_RETAINED_JOBS


def test_a_projection_shares_the_stored_result_rather_than_copying_it():
    """One deep copy per poll per job, and the poll runs while a job is live."""
    store = _store()
    job_id = _finish(store, 0)

    projected = store.project(SESSION, yaml_digest(YAML))[0]

    assert projected.result is store.get(job_id).result


def _api():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx2")
    from fastapi.testclient import TestClient

    from rheplicant.gui.api import create_app

    return TestClient, create_app


def test_the_jobs_body_shares_the_stored_result_rather_than_copying_it():
    """``project`` copied, and then ``_jobs_body`` copied what it had copied.

    Identity is the only assertion that can tell one copy from none: a body
    built with ``dataclasses.asdict`` is equal to the stored result and is
    never the same object as it.
    """
    pytest.importorskip("fastapi")
    from rheplicant.gui import api

    store = api.SessionStore(job_store=_store())
    session_id, session = store.create(yaml.safe_dump(preflight_document(variants={})))
    row = store.jobs.submit(session_id, "validate", 0, session.yaml_text)
    store.jobs.run(row.job_id, lambda _kind, _yaml: {"findings": [{"check": "C13"}]})
    stored = store.jobs.get(row.job_id).result

    polled = api._jobs_body(store, session_id, session)
    whole = api._session_body(store, session_id, session)

    assert polled["jobs"][0]["result"] is stored
    assert whole["jobs"][0]["result"] is stored


def _audit_target(tmp_path):
    target = tmp_path / "result"
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
    identity = target.stat()
    return {
        "output": {
            "target_path": str(target),
            "marker_id": marker_id,
            "target_device": identity.st_dev,
            "target_inode": identity.st_ino,
            "audit_files": ["config.resolved.yaml"],
        }
    }


def test_a_retired_jobs_audit_link_is_refused_as_gone_rather_than_missing(tmp_path):
    """The price of the bound, charged where the user actually meets it.

    A link the GUI handed out keeps its shape after the job behind it has been
    retired, so the answer must say which of the two things happened.  ``410``
    says the job existed and this server no longer serves it; the bundle it
    names is still on disk, untouched.  A ``404`` would say the id was never
    issued, which is a different and false claim.
    """
    TestClient, create_app = _api()
    result = _audit_target(tmp_path)
    jobs = _store()
    browser = TestClient(create_app(job_store=jobs, job_runner=lambda _k, _y: result))
    created = browser.post(
        "/api/sessions",
        json={"yaml_text": yaml.safe_dump(preflight_document(variants={}))},
    ).json()
    session_id = created["session_id"]
    submitted = browser.post(
        f"/api/sessions/{session_id}/jobs",
        json={"expected_revision": 0, "kind": "run"},
    ).json()
    job_id = submitted["jobs"][0]["job_id"]
    link = f"/api/sessions/{session_id}/jobs/{job_id}/artifacts/config.resolved.yaml"
    assert browser.get(link).status_code == 200

    for revision in range(MAX_RETAINED_JOBS):
        row = jobs.submit(session_id, "validate", revision, YAML)
        jobs.run(row.job_id, lambda _kind, _yaml: {"findings": []})

    gone = browser.get(link)
    assert gone.status_code == 410
    assert "retired" in gone.json()["detail"]
    assert str(MAX_RETAINED_JOBS) in gone.json()["detail"]


def test_a_job_id_that_was_never_issued_is_still_refused_as_missing(tmp_path):
    """The other half of the same answer: absence is not retirement."""
    TestClient, create_app = _api()
    browser = TestClient(create_app(job_store=_store(), job_runner=lambda _k, _y: {}))
    created = browser.post(
        "/api/sessions",
        json={"yaml_text": yaml.safe_dump(preflight_document(variants={}))},
    ).json()
    session_id = created["session_id"]

    missing = browser.get(
        f"/api/sessions/{session_id}/jobs/never-issued/artifacts/config.resolved.yaml"
    )

    assert missing.status_code == 404
    assert "does not exist" in missing.json()["detail"]
