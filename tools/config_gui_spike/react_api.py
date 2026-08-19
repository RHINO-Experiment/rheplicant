"""FastAPI boundary for the React/TypeScript half of the Plan 5 spike."""

from __future__ import annotations

import dataclasses
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict

from rheplicant.config import ConfigError
from rheplicant.gui import EditorSnapshot, set_node, snapshot


class YamlPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    yaml_text: str


class NodeEditPayload(YamlPayload):
    enabled: bool
    settings: dict[str, object] | None = None


def _body(found: EditorSnapshot) -> dict[str, object]:
    return dataclasses.asdict(found)


def create_app(frontend_dir: Path | None = None) -> FastAPI:
    """Build the API independently of a running frontend."""
    app = FastAPI(title="Rheplicant config editor — React spike")

    @app.post("/api/snapshot")
    def document_snapshot(payload: YamlPayload) -> dict[str, object]:
        try:
            return _body(snapshot(payload.yaml_text))
        except ConfigError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.patch("/api/nodes/{node_id}")
    def edit_node(node_id: str, payload: NodeEditPayload) -> dict[str, object]:
        try:
            return _body(
                set_node(
                    payload.yaml_text,
                    node_id,
                    enabled=payload.enabled,
                    settings=payload.settings,
                )
            )
        except ConfigError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    if frontend_dir is not None:
        root = frontend_dir.resolve()
        if not (root / "index.html").is_file():
            raise ValueError(f"Frontend directory {root} has no index.html.")
        app.mount("/", StaticFiles(directory=root, html=True), name="frontend")
    return app


app = create_app()

__all__ = ["create_app"]
