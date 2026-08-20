"""FastAPI routes for the selected React YAML editor.

The mutable object here is only an in-memory registry of immutable
``EditorSession`` records.  All scientific transitions remain in
``rheplicant.gui.document`` and ``rheplicant.gui.session`` so they are directly
testable without FastAPI, a browser, or file IO.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from threading import RLock
from typing import Literal
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, HTTPException, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from _rheplicant_bootstrap.errors import ConfigError
from rheplicant.gui.document import EditorSnapshot, set_node, snapshot
from rheplicant.gui.jobs import JobKind, JobRunner, JobStore, execute_job, yaml_digest
from rheplicant.gui.outputs import project_output_workflow, read_audit_artifact
from rheplicant.gui.session import (
    EditorSession,
    RevisionConflict,
    compose_session_node,
    edit_session_many_node,
    edit_session_node,
    load_session_yaml,
    mark_saved,
    move_session_node_instance,
    new_session,
    place_session_node,
    redo,
    replace_session_yaml,
    set_session_output_product,
    set_session_output_report,
    set_session_snapshot_before,
    undo,
)
from rheplicant.gui.starter import STARTER_YAML


class _ClosedModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class YamlPayload(_ClosedModel):
    yaml_text: str


class NodeEditPayload(YamlPayload):
    enabled: bool
    settings: dict[str, object] | list[dict[str, object]] | None = None
    variant: str | None = None


class RevisionPayload(_ClosedModel):
    expected_revision: int


class JobPayload(RevisionPayload):
    kind: Literal["validate", "preview_forward", "run", "compare", "benchmark"]


class SessionYamlPayload(RevisionPayload):
    yaml_text: str


class SessionNodeEditPayload(RevisionPayload):
    enabled: bool
    settings: dict[str, object] | list[dict[str, object]] | None = None
    variant: str | None = None


class SessionManyPayload(RevisionPayload):
    entries: dict[str, object] | list[dict[str, object]]
    variant: str | None = None


class SessionMovePayload(RevisionPayload):
    from_index: int
    to_index: int
    variant: str | None = None


class SessionComposePayload(RevisionPayload):
    compose: str
    stages: list[dict[str, object]]
    variant: str | None = None


class SessionPlacementPayload(RevisionPayload):
    at: str | list[str]
    settings: dict[str, object]
    variant: str | None = None


class SessionSnapshotPayload(RevisionPayload):
    snapshot_name: str
    variant: str | None = None


class SessionOutputProductPayload(RevisionPayload):
    enabled: bool
    format: str | None = None
    runs: list[str] = Field(default_factory=list)
    keys: list[str] = Field(default_factory=list)
    themes: list[str] = Field(default_factory=list)


class SessionOutputReportPayload(RevisionPayload):
    enabled: bool
    rows: list[str] = Field(default_factory=list)
    columns: list[str] = Field(default_factory=lambda: ["mean", "std", "seconds"])
    reference: str | None = None
    relative: list[str] = Field(default_factory=list)
    formats: list[str] = Field(default_factory=lambda: ["text"])


class SessionStore:
    """Thread-safe storage around immutable editor-session values."""

    def __init__(self, *, job_store: JobStore | None = None) -> None:
        self._sessions: dict[str, EditorSession] = {}
        self._lock = RLock()
        self.jobs = job_store if job_store is not None else JobStore()

    def create(self, yaml_text: str) -> tuple[str, EditorSession]:
        session = new_session(yaml_text)
        session_id = uuid4().hex
        with self._lock:
            self._sessions[session_id] = session
        return session_id, session

    def get(self, session_id: str) -> EditorSession:
        with self._lock:
            try:
                return self._sessions[session_id]
            except KeyError:
                raise KeyError(session_id) from None

    def apply(
        self,
        session_id: str,
        transition: Callable[[EditorSession], EditorSession],
    ) -> EditorSession:
        with self._lock:
            try:
                current = self._sessions[session_id]
            except KeyError:
                raise KeyError(session_id) from None
            updated = transition(current)
            self._sessions[session_id] = updated
            return updated

    def submit_job(
        self,
        session_id: str,
        kind: JobKind,
        expected_revision: int,
    ) -> tuple[EditorSession, str]:
        """Revision-check and bind a job to one immutable YAML snapshot."""
        with self._lock:
            try:
                current = self._sessions[session_id]
            except KeyError:
                raise KeyError(session_id) from None
            if type(expected_revision) is not int or expected_revision != current.revision:
                raise RevisionConflict(expected_revision, current.revision)
            if snapshot(current.yaml_text).validation.run_blocked:
                raise ConfigError(
                    "Execution jobs are disabled while text-level refusals exist."
                )
            row = self.jobs.submit(session_id, kind, current.revision, current.yaml_text)
            return current, row.job_id


def _snapshot_body(found: EditorSnapshot) -> dict[str, object]:
    return dataclasses.asdict(found)


def _session_body(
    store: SessionStore,
    session_id: str,
    session: EditorSession,
) -> dict[str, object]:
    return {
        "session_id": session_id,
        "revision": session.revision,
        "dirty": session.dirty,
        "validation_stale": session.validation_stale,
        "can_undo": session.can_undo,
        "can_redo": session.can_redo,
        "jobs": [
            dataclasses.asdict(row)
            for row in store.jobs.project(session_id, yaml_digest(session.yaml_text))
        ],
        "outputs": dataclasses.asdict(project_output_workflow(session.yaml_text)),
        "document": _snapshot_body(snapshot(session.yaml_text)),
    }


def _not_found(session_id: str) -> HTTPException:
    return HTTPException(
        status_code=404,
        detail=f"Editor session {session_id!r} does not exist.",
    )


def _apply(
    store: SessionStore,
    session_id: str,
    transition: Callable[[EditorSession], EditorSession],
) -> dict[str, object]:
    try:
        return _session_body(store, session_id, store.apply(session_id, transition))
    except KeyError:
        raise _not_found(session_id) from None
    except RevisionConflict as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except ConfigError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


def create_app(
    frontend_dir: Path | None = None,
    *,
    session_store: SessionStore | None = None,
    job_store: JobStore | None = None,
    job_runner: JobRunner = execute_job,
) -> FastAPI:
    """Build the selected-stack API independently of a live frontend."""
    app = FastAPI(title="Rheplicant YAML config editor")
    if session_store is not None and job_store is not None:
        raise ValueError("job_store belongs to SessionStore when both are supplied.")
    store = session_store if session_store is not None else SessionStore(job_store=job_store)

    @app.get("/api/starter")
    def get_starter() -> dict[str, str]:
        return {"yaml_text": STARTER_YAML}

    @app.post("/api/snapshot")
    def document_snapshot(payload: YamlPayload) -> dict[str, object]:
        try:
            return _snapshot_body(snapshot(payload.yaml_text))
        except ConfigError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.patch("/api/nodes/{node_id}")
    def edit_node(node_id: str, payload: NodeEditPayload) -> dict[str, object]:
        try:
            return _snapshot_body(
                set_node(
                    payload.yaml_text,
                    node_id,
                    enabled=payload.enabled,
                    settings=payload.settings,
                    variant=payload.variant,
                )
            )
        except ConfigError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.post("/api/sessions", status_code=201)
    def create_session(payload: YamlPayload) -> dict[str, object]:
        try:
            session_id, session = store.create(payload.yaml_text)
            return _session_body(store, session_id, session)
        except ConfigError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.get("/api/sessions/{session_id}")
    def get_session(session_id: str) -> dict[str, object]:
        try:
            return _session_body(store, session_id, store.get(session_id))
        except KeyError:
            raise _not_found(session_id) from None

    @app.put("/api/sessions/{session_id}/yaml")
    def replace_yaml_route(
        session_id: str,
        payload: SessionYamlPayload,
    ) -> dict[str, object]:
        return _apply(
            store,
            session_id,
            lambda current: replace_session_yaml(
                current,
                payload.yaml_text,
                expected_revision=payload.expected_revision,
            ),
        )

    @app.patch("/api/sessions/{session_id}/nodes/{node_id}")
    def edit_session_node_route(
        session_id: str,
        node_id: str,
        payload: SessionNodeEditPayload,
    ) -> dict[str, object]:
        return _apply(
            store,
            session_id,
            lambda current: edit_session_node(
                current,
                node_id,
                enabled=payload.enabled,
                settings=payload.settings,
                expected_revision=payload.expected_revision,
                variant=payload.variant,
            ),
        )

    @app.put("/api/sessions/{session_id}/nodes/{node_id}/many")
    def edit_session_many_route(
        session_id: str,
        node_id: str,
        payload: SessionManyPayload,
    ) -> dict[str, object]:
        return _apply(
            store,
            session_id,
            lambda current: edit_session_many_node(
                current,
                node_id,
                payload.entries,
                expected_revision=payload.expected_revision,
                variant=payload.variant,
            ),
        )

    @app.post("/api/sessions/{session_id}/nodes/{node_id}/move")
    def move_session_node_route(
        session_id: str,
        node_id: str,
        payload: SessionMovePayload,
    ) -> dict[str, object]:
        return _apply(
            store,
            session_id,
            lambda current: move_session_node_instance(
                current,
                node_id,
                payload.from_index,
                payload.to_index,
                expected_revision=payload.expected_revision,
                variant=payload.variant,
            ),
        )

    @app.put("/api/sessions/{session_id}/nodes/{node_id}/compose")
    def compose_session_node_route(
        session_id: str,
        node_id: str,
        payload: SessionComposePayload,
    ) -> dict[str, object]:
        return _apply(
            store,
            session_id,
            lambda current: compose_session_node(
                current,
                node_id,
                payload.compose,
                payload.stages,
                expected_revision=payload.expected_revision,
                variant=payload.variant,
            ),
        )

    @app.put("/api/sessions/{session_id}/nodes/{node_id}/placement")
    def place_session_node_route(
        session_id: str,
        node_id: str,
        payload: SessionPlacementPayload,
    ) -> dict[str, object]:
        return _apply(
            store,
            session_id,
            lambda current: place_session_node(
                current,
                node_id,
                payload.at,
                payload.settings,
                expected_revision=payload.expected_revision,
                variant=payload.variant,
            ),
        )

    @app.put("/api/sessions/{session_id}/nodes/{node_id}/snapshot-before")
    def snapshot_session_node_route(
        session_id: str,
        node_id: str,
        payload: SessionSnapshotPayload,
    ) -> dict[str, object]:
        return _apply(
            store,
            session_id,
            lambda current: set_session_snapshot_before(
                current,
                node_id,
                payload.snapshot_name,
                expected_revision=payload.expected_revision,
                variant=payload.variant,
            ),
        )

    @app.put("/api/sessions/{session_id}/outputs/products/{product_name}")
    def output_product_route(
        session_id: str,
        product_name: str,
        payload: SessionOutputProductPayload,
    ) -> dict[str, object]:
        return _apply(
            store,
            session_id,
            lambda current: set_session_output_product(
                current,
                product_name,
                enabled=payload.enabled,
                format=payload.format,
                runs=payload.runs,
                keys=payload.keys,
                themes=payload.themes,
                expected_revision=payload.expected_revision,
            ),
        )

    @app.put("/api/sessions/{session_id}/outputs/report")
    def output_report_route(
        session_id: str,
        payload: SessionOutputReportPayload,
    ) -> dict[str, object]:
        return _apply(
            store,
            session_id,
            lambda current: set_session_output_report(
                current,
                enabled=payload.enabled,
                rows=payload.rows,
                columns=payload.columns,
                reference=payload.reference,
                relative=payload.relative,
                formats=payload.formats,
                expected_revision=payload.expected_revision,
            ),
        )

    @app.post("/api/sessions/{session_id}/undo")
    def undo_route(
        session_id: str,
        payload: RevisionPayload,
    ) -> dict[str, object]:
        return _apply(
            store,
            session_id,
            lambda current: undo(
                current,
                expected_revision=payload.expected_revision,
            ),
        )

    @app.post("/api/sessions/{session_id}/redo")
    def redo_route(
        session_id: str,
        payload: RevisionPayload,
    ) -> dict[str, object]:
        return _apply(
            store,
            session_id,
            lambda current: redo(
                current,
                expected_revision=payload.expected_revision,
            ),
        )

    @app.post("/api/sessions/{session_id}/load")
    def load_route(
        session_id: str,
        payload: SessionYamlPayload,
    ) -> dict[str, object]:
        return _apply(
            store,
            session_id,
            lambda current: load_session_yaml(
                current,
                payload.yaml_text,
                expected_revision=payload.expected_revision,
            ),
        )

    @app.post("/api/sessions/{session_id}/save")
    def save_route(
        session_id: str,
        payload: RevisionPayload,
    ) -> dict[str, object]:
        return _apply(
            store,
            session_id,
            lambda current: mark_saved(
                current,
                expected_revision=payload.expected_revision,
            ),
        )

    @app.post("/api/sessions/{session_id}/jobs", status_code=202)
    def submit_job_route(
        session_id: str,
        payload: JobPayload,
        background_tasks: BackgroundTasks,
    ) -> dict[str, object]:
        try:
            session, job_id = store.submit_job(
                session_id,
                payload.kind,
                payload.expected_revision,
            )
        except KeyError:
            raise _not_found(session_id) from None
        except RevisionConflict as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except ConfigError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        body = _session_body(store, session_id, session)
        background_tasks.add_task(store.jobs.run, job_id, job_runner)
        return body

    @app.get("/api/sessions/{session_id}/jobs/{job_id}/artifacts/{artifact_name}")
    def audit_artifact_route(
        session_id: str,
        job_id: str,
        artifact_name: str,
    ) -> Response:
        try:
            job = store.jobs.get(job_id)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Job {job_id!r} does not exist.") from None
        if job.session_id != session_id:
            raise HTTPException(status_code=404, detail=f"Job {job_id!r} does not exist.")
        result = job.result
        output = result.get("output") if isinstance(result, Mapping) else None
        if not isinstance(output, Mapping):
            raise HTTPException(status_code=409, detail="The job has no completed audit bundle.")
        target = output.get("target_path")
        marker_id = output.get("marker_id")
        target_device = output.get("target_device")
        target_inode = output.get("target_inode")
        files = output.get("audit_files")
        if (
            not isinstance(target, str)
            or not isinstance(marker_id, str)
            or type(target_device) is not int
            or type(target_inode) is not int
            or isinstance(files, str | bytes)
            or not isinstance(files, Sequence)
            or artifact_name not in files
        ):
            raise HTTPException(
                status_code=409,
                detail="The audit artefact is not available for this job.",
            )
        try:
            artifact = read_audit_artifact(
                target,
                marker_id,
                artifact_name,
                target_device=target_device,
                target_inode=target_inode,
            )
        except ConfigError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        return Response(content=artifact.payload, media_type=artifact.media_type)

    if frontend_dir is not None:
        root = frontend_dir.resolve()
        if not (root / "index.html").is_file():
            raise ValueError(f"Frontend directory {root} has no index.html.")
        app.mount("/", StaticFiles(directory=root, html=True), name="frontend")
    return app


__all__ = ["SessionStore", "create_app"]
