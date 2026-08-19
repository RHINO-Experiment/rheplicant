"""FastAPI routes for the selected React YAML editor.

The mutable object here is only an in-memory registry of immutable
``EditorSession`` records.  All scientific transitions remain in
``rheplicant.gui.document`` and ``rheplicant.gui.session`` so they are directly
testable without FastAPI, a browser, or file IO.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from pathlib import Path
from threading import RLock
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict

from _rheplicant_bootstrap.errors import ConfigError
from rheplicant.gui.document import EditorSnapshot, set_node, snapshot
from rheplicant.gui.session import (
    EditorSession,
    RevisionConflict,
    edit_session_node,
    load_session_yaml,
    mark_saved,
    new_session,
    redo,
    replace_session_yaml,
    undo,
)


class _ClosedModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class YamlPayload(_ClosedModel):
    yaml_text: str


class NodeEditPayload(YamlPayload):
    enabled: bool
    settings: dict[str, object] | None = None


class RevisionPayload(_ClosedModel):
    expected_revision: int


class SessionYamlPayload(RevisionPayload):
    yaml_text: str


class SessionNodeEditPayload(RevisionPayload):
    enabled: bool
    settings: dict[str, object] | None = None


class SessionStore:
    """Thread-safe storage around immutable editor-session values."""

    def __init__(self) -> None:
        self._sessions: dict[str, EditorSession] = {}
        self._lock = RLock()

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


def _snapshot_body(found: EditorSnapshot) -> dict[str, object]:
    return dataclasses.asdict(found)


def _session_body(session_id: str, session: EditorSession) -> dict[str, object]:
    return {
        "session_id": session_id,
        "revision": session.revision,
        "dirty": session.dirty,
        "validation_stale": session.validation_stale,
        "can_undo": session.can_undo,
        "can_redo": session.can_redo,
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
        return _session_body(session_id, store.apply(session_id, transition))
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
) -> FastAPI:
    """Build the selected-stack API independently of a live frontend."""
    app = FastAPI(title="Rheplicant YAML config editor")
    store = session_store if session_store is not None else SessionStore()

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
                )
            )
        except ConfigError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.post("/api/sessions", status_code=201)
    def create_session(payload: YamlPayload) -> dict[str, object]:
        try:
            session_id, session = store.create(payload.yaml_text)
            return _session_body(session_id, session)
        except ConfigError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.get("/api/sessions/{session_id}")
    def get_session(session_id: str) -> dict[str, object]:
        try:
            return _session_body(session_id, store.get(session_id))
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

    if frontend_dir is not None:
        root = frontend_dir.resolve()
        if not (root / "index.html").is_file():
            raise ValueError(f"Frontend directory {root} has no index.html.")
        app.mount("/", StaticFiles(directory=root, html=True), name="frontend")
    return app


__all__ = ["SessionStore", "create_app"]
