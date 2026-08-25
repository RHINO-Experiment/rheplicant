"""The submit response describes the snapshot the job was bound to.

A7.4 filed ``submit_job``'s response as "built from a stale snapshot" and
prescribed re-reading the session inside the same critical section, with the
instruction to *verify the race first*. Verified 2026-08-25, and the finding
does not hold: the response is not stale, it is ANCHORED, and re-reading
would introduce the inconsistency the item wants to prevent.

``SessionStore.submit_job`` takes ``_lock``, revision-checks, and binds the
job to ``current.yaml_text`` -- one immutable value, by its own docstring.
``_session_body`` then draws every session-derived field from that same
value: ``revision``, ``yaml_digest``, ``dirty``, ``validation_stale``,
``can_undo``, ``can_redo``, ``outputs``, ``document``. The one live read is
``store.jobs.project(session_id, digest)``, which returns every job of the
session and uses the digest for one thing -- the per-job ``stale`` flag,
``row.yaml_digest != current_digest``. Handing it the snapshot's digest
marks each job relative to the document the caller submitted against.

Re-reading the session under the lock would report a revision the caller's
job is NOT bound to, and would mark the job just accepted as **stale in the
response acknowledging it**. The client's next mutation carries the revision this response
returned, so if another actor advanced the session in between, the 409 is
the correct answer and is what optimistic concurrency is for.

These tests pin the anchoring, so the "fix" cannot be applied later without
something going red.
"""

from __future__ import annotations

import pytest

from rheplicant.gui.api import SessionStore, _session_body


@pytest.fixture
def submittable() -> str:
    """A document `submit_job` will actually accept.

    ``tests/gui/test_document.BASE`` carries text-level refusals, so
    ``submit_job`` answers "Execution jobs are disabled while text-level
    refusals exist." before any of this is reached -- measured, on the first
    run of these tests. The job routes use ``preflight_document`` for the
    same reason.
    """
    import yaml

    from tests.config.preflight_helpers import preflight_document

    return yaml.safe_dump(preflight_document(variants={}), sort_keys=False)


@pytest.fixture
def store():
    return SessionStore()


def _advance(store: SessionStore, session_id: str) -> None:
    """Move the stored session on, the way a concurrent edit would."""
    from rheplicant.gui.session import replace_session_yaml

    current = store.get(session_id)
    store.apply(
        session_id,
        lambda found: replace_session_yaml(
            found,
            found.yaml_text + "\n# a concurrent edit\n",
            expected_revision=found.revision,
        ),
    )
    assert store.get(session_id).revision != current.revision


def test_the_body_reports_the_snapshot_it_was_handed_not_the_stored_one(
    store, submittable
):
    """The whole of A7.4, as one assertion.

    The session advances between the submit and the body build -- exactly
    the window the item describes -- and the body still reports the
    submitted revision, because that is the revision the job is bound to.
    """
    session_id, submitted = store.create(submittable)
    _advance(store, session_id)

    body = _session_body(store, session_id, submitted)

    assert body["revision"] == submitted.revision
    assert body["revision"] != store.get(session_id).revision


def test_every_session_field_in_the_body_comes_from_the_same_snapshot(
    store, submittable
):
    """Anchoring is only worth having if it is TOTAL.

    A body that took its revision from the snapshot and its digest from the
    store would be internally inconsistent in a way no single-field test
    sees, so each field is compared against the value the snapshot implies
    rather than against a literal.
    """
    import dataclasses

    from rheplicant.gui.document import snapshot
    from rheplicant.gui.outputs import project_output_workflow

    session_id, submitted = store.create(submittable)
    _advance(store, session_id)

    body = _session_body(store, session_id, submitted)

    assert body["yaml_digest"] != _session_body(
        store, session_id, store.get(session_id)
    )["yaml_digest"]
    assert body["dirty"] == submitted.dirty
    assert body["can_undo"] == submitted.can_undo
    assert body["can_redo"] == submitted.can_redo
    assert body["document"]["yaml_text"] == submitted.yaml_text
    assert (
        body["outputs"]
        == dataclasses.asdict(project_output_workflow(submitted.yaml_text))
    )
    assert body["document"] == dataclasses.asdict(snapshot(submitted.yaml_text))


def test_the_snapshots_digest_decides_which_jobs_are_marked_stale(
    store, submittable
):
    """The one live read, and what the digest actually does with it.

    Not filtering -- measured. ``JobStore.project`` returns every job of the
    session and uses ``current_digest`` for one thing: the per-job flag
    ``row.yaml_digest != current_digest``, surfaced as ``"stale"``. The first
    version of this test asserted filtering, which was inferred rather than
    read, and it failed on the job it expected to be excluded.

    The correction makes the anchoring argument STRONGER rather than weaker.
    Passing the SNAPSHOT's digest marks each job stale-or-not relative to the
    document the caller submitted against. Re-read the session instead, as
    A7.4 prescribes, and the job the caller just submitted comes back marked
    **stale in the very response acknowledging it** -- because the newer
    session's digest is not the digest the job was bound to. That is not a
    subtle inconsistency; it is a visibly wrong badge on a job that was
    accepted a microsecond earlier.
    """
    session_id, submitted = store.create(submittable)
    store.submit_job(session_id, "validate", submitted.revision)
    _advance(store, session_id)

    anchored = _session_body(store, session_id, submitted)["jobs"]
    reread = _session_body(store, session_id, store.get(session_id))["jobs"]

    assert len(anchored) == len(reread) == 1, "one job, projected either way"
    assert anchored[0]["job_id"] == reread[0]["job_id"]
    # The whole of A7.4, in two booleans.
    assert anchored[0]["stale"] is False, (
        "the acknowledgement must not declare the job it is acknowledging stale"
    )
    assert reread[0]["stale"] is True, (
        "and re-reading the session -- what A7.4 prescribes -- is what would "
        "make it say so"
    )
