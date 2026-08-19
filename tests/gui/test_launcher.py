from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

from rheplicant.gui import launcher

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE = """\
runtime:
  jax_enable_x64: true
model:
  gain:
    type: GainOperator
    gain: 1.0
runs:
  - name: forward
    kind: forward
"""


def test_selected_gui_extra_and_console_launcher_are_public() -> None:
    project = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())["project"]

    assert project["optional-dependencies"]["gui"] == [
        "fastapi>=0.116,<1",
        "uvicorn>=0.35,<1",
    ]
    assert project["optional-dependencies"]["gui-react"] == [
        *project["optional-dependencies"]["gui"],
        "httpx2",
    ]
    assert project["scripts"]["rheplicant-gui"] == "rheplicant.gui.launcher:main"


def test_bundled_frontend_is_a_closed_production_build() -> None:
    root = launcher.frontend_directory()
    index = root / "index.html"
    assert index.is_file()
    markup = index.read_text()
    assert "Rheplicant YAML config editor" in markup

    references = re.findall(r'(?:src|href)="/([^"#?]+)"', markup)
    assert references
    assert any(reference.endswith(".js") for reference in references)
    assert any(reference.endswith(".css") for reference in references)
    assert all((root / reference).is_file() for reference in references)
    assert not tuple(root.rglob("*.map"))
    assert not tuple(root.rglob("*.tsx"))
    javascript = "\n".join(
        (root / reference).read_text()
        for reference in references
        if reference.endswith(".js")
    )
    assert "/api/sessions" in javascript
    assert "Rheplicant config editor" in javascript


def test_bundled_app_serves_the_editor_and_api_from_one_origin() -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx2")
    from fastapi.testclient import TestClient

    client = TestClient(launcher.create_editor_app())
    index = client.get("/")
    assert index.status_code == 200
    assert "Rheplicant YAML config editor" in index.text

    created = client.post("/api/sessions", json={"yaml_text": BASE})
    assert created.status_code == 201
    assert created.json()["document"]["yaml_text"] == BASE


def test_launcher_defaults_to_loopback_and_remote_binding_is_explicit(monkeypatch) -> None:
    calls: list[tuple[str, int, str]] = []

    def fake_serve(*, host: str, port: int, log_level: str) -> None:
        calls.append((host, port, log_level))

    monkeypatch.setattr(launcher, "serve", fake_serve)
    assert launcher.main(["--port", "9123", "--log-level", "error"]) == 0
    assert calls == [("127.0.0.1", 9123, "error")]

    with pytest.raises(SystemExit, match="--allow-remote"):
        launcher.main(["--host", "0.0.0.0"])
    assert len(calls) == 1

    assert launcher.main(["--host", "::1", "--port", "9124"]) == 0
    assert launcher.main(
        ["--host", "0.0.0.0", "--allow-remote", "--port", "9125"]
    ) == 0
