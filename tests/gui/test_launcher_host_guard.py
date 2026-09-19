"""The served editor refuses a foreign Host and a cross-origin write.

A page on ``attacker.example`` whose DNS record is re-pointed at 127.0.0.1
(DNS rebinding) talks to the editor with ``Host: attacker.example`` and a
matching ``Origin``. Before this guard every state-changing route accepted
that, including job submission, which runs a document's ``python:`` and
``plugins:`` targets. The guard is installed by the launcher, on the app it
serves, and not by ``create_app``: the in-process API tests address the app
as ``testserver`` and exercise the routes, not the deployment.

Every client here is Starlette's in-process TestClient; nothing binds a
socket. ``uvicorn.run`` is replaced so that ``serve`` hands over the exact app
it would have served.
"""

from __future__ import annotations

import pytest

from rheplicant.gui import launcher

pytest.importorskip("fastapi")
pytest.importorskip("httpx2")
pytest.importorskip("uvicorn")

import uvicorn  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from starlette.websockets import WebSocketDisconnect  # noqa: E402

from rheplicant.gui.starter import STARTER_YAML  # noqa: E402

LOOPBACK = "http://127.0.0.1:8000"
FOREIGN_ORIGIN = "http://attacker.example:8000"


def served_app(monkeypatch, **options):
    """Return the ASGI app ``serve`` would pass to uvicorn."""
    handed: list[object] = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **_: handed.append(app))
    launcher.serve(port=8000, log_level="info", **{"host": "127.0.0.1", **options})
    assert len(handed) == 1
    return handed[0]


@pytest.fixture
def client(monkeypatch):
    return TestClient(served_app(monkeypatch), base_url=LOOPBACK)


def new_session(client: TestClient) -> tuple[str, int]:
    created = client.post("/api/sessions", json={"yaml_text": STARTER_YAML})
    assert created.status_code == 201, created.text
    body = created.json()
    return body["session_id"], body["revision"]


@pytest.mark.parametrize(
    "host",
    [
        "attacker.example:8000",
        "attacker.example",
        "testserver",
        "127.0.0.1.attacker.example:8000",
        "localhost.attacker.example",
        "localhost.",
        "0.0.0.0:8000",
        "[::2]:8000",
        "127.0.0.2:8000",
        "::1",
        "[::1",
        "[::1]x",
        "127.0.0.1:80:80",
        "127.0.0.1:",
        "user@127.0.0.1",
        "",
    ],
)
def test_a_foreign_or_malformed_host_is_refused_for_reads(client, host) -> None:
    response = client.get("/api/starter", headers={"Host": host})
    assert response.status_code == 400
    assert response.text == "Invalid Host header."


def test_a_rebound_host_is_refused_for_every_kind_of_request(client) -> None:
    rebound = {"Host": "attacker.example:8000", "Origin": FOREIGN_ORIGIN}
    assert client.get("/", headers=rebound).status_code == 400
    assert client.head("/api/starter", headers=rebound).status_code == 400
    created = client.post(
        "/api/sessions", json={"yaml_text": STARTER_YAML}, headers=rebound
    )
    assert created.status_code == 400
    assert created.text == "Invalid Host header."
    session_id, revision = new_session(client)
    replaced = client.put(
        f"/api/sessions/{session_id}/yaml",
        json={"yaml_text": STARTER_YAML, "expected_revision": revision},
        headers=rebound,
    )
    assert replaced.status_code == 400
    submitted = client.post(
        f"/api/sessions/{session_id}/jobs",
        json={"kind": "validate", "expected_revision": revision},
        headers=rebound,
    )
    assert submitted.status_code == 400


@pytest.mark.parametrize(
    "host",
    [
        "127.0.0.1",
        "127.0.0.1:8000",
        "127.0.0.1:61234",
        "localhost",
        "localhost:8765",
        "LocalHost:8000",
        "[::1]",
        "[::1]:8765",
        "[0:0:0:0:0:0:0:1]:8765",
    ],
)
def test_loopback_hosts_are_accepted_with_any_port(client, host) -> None:
    assert client.get("/api/starter", headers={"Host": host}).status_code == 200
    created = client.post(
        "/api/sessions", json={"yaml_text": STARTER_YAML}, headers={"Host": host}
    )
    assert created.status_code == 201


@pytest.mark.parametrize(
    "origin",
    [
        FOREIGN_ORIGIN,
        "http://evil.example",
        "null",
        "http://[::1",
        "file://",
        "http://127.0.0.1:9999",
        "http://localhost:8000",
        "https://127.0.0.1:8000:1",
    ],
)
@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE"])
def test_a_foreign_origin_is_refused_on_state_changing_methods(
    client, method, origin
) -> None:
    session_id, revision = new_session(client)
    request = {
        "POST": ("/api/sessions", {"yaml_text": STARTER_YAML}),
        "PUT": (
            f"/api/sessions/{session_id}/yaml",
            {"yaml_text": STARTER_YAML, "expected_revision": revision},
        ),
        "PATCH": (
            f"/api/sessions/{session_id}/fields",
            {"path": "runtime.jax_enable_x64", "value": True, "expected_revision": revision},
        ),
        "DELETE": (f"/api/sessions/{session_id}", None),
    }
    path, body = request[method]
    response = client.request(method, path, json=body, headers={"Origin": origin})
    assert response.status_code == 403
    assert response.text == "Cross-origin request refused."
    unchanged = client.get(f"/api/sessions/{session_id}")
    assert unchanged.json()["revision"] == revision


def test_a_foreign_origin_does_not_block_safe_methods(client) -> None:
    headers = {"Origin": FOREIGN_ORIGIN}
    assert client.get("/api/starter", headers=headers).status_code == 200
    assert client.head("/", headers=headers).status_code == 200


@pytest.mark.parametrize(
    ("host", "origin"),
    [
        ("127.0.0.1:8000", "http://127.0.0.1:8000"),
        # The Vite dev server proxies /api without rewriting Host.
        ("localhost:5173", "http://localhost:5173"),
        ("[::1]:8765", "http://[::1]:8765"),
        ("[0:0::1]:8765", "http://[::1]:8765"),
        ("LocalHost:8000", "HTTP://LOCALHOST:8000"),
        ("127.0.0.1", "http://127.0.0.1:80"),
        ("127.0.0.1:80", "http://127.0.0.1"),
        ("127.0.0.1:8000", None),
    ],
)
def test_the_request_s_own_or_a_missing_origin_is_accepted_on_writes(
    client, host, origin
) -> None:
    headers = {"Host": host} if origin is None else {"Host": host, "Origin": origin}
    created = client.post(
        "/api/sessions", json={"yaml_text": STARTER_YAML}, headers=headers
    )
    assert created.status_code == 201
    body = created.json()
    replaced = client.put(
        f"/api/sessions/{body['session_id']}/yaml",
        json={"yaml_text": STARTER_YAML, "expected_revision": body["revision"]},
        headers=headers,
    )
    assert replaced.status_code == 200


@pytest.mark.parametrize(
    ("host", "origin"),
    [
        ("127.0.0.1:8000", "http://127.0.0.1:9999"),
        ("127.0.0.1:8000", "http://localhost:8000"),
        ("localhost:5173", "http://localhost:8000"),
        ("[::1]:8765", "http://[::1]:8000"),
        ("127.0.0.1", "https://127.0.0.1"),
        ("127.0.0.1:8000", "http://127.0.0.1"),
    ],
)
def test_an_origin_on_another_port_or_loopback_name_is_refused_on_writes(
    client, host, origin
) -> None:
    """A loopback Origin is not enough: another port on this machine is
    another origin, for example a second local web server."""
    session_id, revision = new_session(client)
    response = client.put(
        f"/api/sessions/{session_id}/yaml",
        json={"yaml_text": STARTER_YAML, "expected_revision": revision},
        headers={"Host": host, "Origin": origin},
    )
    assert response.status_code == 403
    assert response.text == "Cross-origin request refused."


def test_default_ports_follow_the_request_scheme(monkeypatch) -> None:
    client = TestClient(served_app(monkeypatch), base_url="https://127.0.0.1")
    for origin, status in (
        ("https://127.0.0.1", 201),
        ("https://127.0.0.1:443", 201),
        ("http://127.0.0.1", 403),
    ):
        created = client.post(
            "/api/sessions", json={"yaml_text": STARTER_YAML}, headers={"Origin": origin}
        )
        assert created.status_code == status, origin


def test_a_websocket_under_a_foreign_host_is_closed(client) -> None:
    with pytest.raises(WebSocketDisconnect) as excinfo:
        with client.websocket_connect("/ws", headers={"Host": "attacker.example"}):
            pass
    assert excinfo.value.code == 1008


def test_create_editor_app_carries_the_guard_itself() -> None:
    client = TestClient(launcher.create_editor_app(), base_url="http://testserver")
    assert client.get("/api/starter").status_code == 400
    loopback = TestClient(launcher.create_editor_app(), base_url=LOOPBACK)
    assert loopback.get("/api/starter").status_code == 200


def test_a_loopback_bind_address_is_itself_an_allowed_host(monkeypatch) -> None:
    app = served_app(monkeypatch, host="127.0.0.2")
    client = TestClient(app, base_url="http://127.0.0.2:8000")
    assert client.get("/api/starter").status_code == 200
    assert client.get("/api/starter", headers={"Host": "127.0.0.3"}).status_code == 400


def test_allow_remote_without_an_allowed_host_is_refused(monkeypatch) -> None:
    calls: list[dict] = []
    monkeypatch.setattr(launcher, "serve", lambda **kwargs: calls.append(kwargs))
    with pytest.raises(SystemExit) as excinfo:
        launcher.main(["--host", "0.0.0.0", "--allow-remote"])
    assert isinstance(excinfo.value.code, str)
    assert "--allowed-host" in excinfo.value.code
    assert calls == []


def test_serve_refuses_allow_remote_without_an_allowed_host(monkeypatch) -> None:
    handed: list[object] = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **_: handed.append(app))
    with pytest.raises(RuntimeError, match="--allowed-host"):
        launcher.serve(host="0.0.0.0", port=8000, log_level="info", allow_remote=True)
    assert handed == []


@pytest.mark.parametrize(
    "name", ["gui.example.org:8000", "http://gui.example.org", "", "gui example", "a/b"]
)
def test_a_malformed_allowed_host_is_refused(monkeypatch, name) -> None:
    calls: list[dict] = []
    monkeypatch.setattr(launcher, "serve", lambda **kwargs: calls.append(kwargs))
    with pytest.raises(SystemExit) as excinfo:
        launcher.main(["--host", "0.0.0.0", "--allow-remote", "--allowed-host", name])
    assert "--allowed-host" in str(excinfo.value.code)
    assert calls == []


def test_main_passes_every_allowed_host_to_serve(monkeypatch) -> None:
    calls: list[dict] = []
    monkeypatch.setattr(launcher, "serve", lambda **kwargs: calls.append(kwargs))
    assert (
        launcher.main(
            [
                "--host",
                "0.0.0.0",
                "--allow-remote",
                "--allowed-host",
                "gui.example.org",
                "--allowed-host",
                "10.0.0.5",
            ]
        )
        == 0
    )
    assert calls == [
        {
            "host": "0.0.0.0",
            "port": 8000,
            "log_level": "info",
            "allow_remote": True,
            "allowed_hosts": ("gui.example.org", "10.0.0.5"),
        }
    ]


def test_an_allowed_host_is_accepted_beside_loopback_and_nothing_else(
    monkeypatch,
) -> None:
    app = served_app(
        monkeypatch,
        host="0.0.0.0",
        allow_remote=True,
        allowed_hosts=("GUI.example.org", "fe80::1"),
    )
    client = TestClient(app, base_url="http://gui.example.org:8000")
    assert client.get("/api/starter").status_code == 200
    created = client.post(
        "/api/sessions",
        json={"yaml_text": STARTER_YAML},
        headers={"Origin": "http://gui.example.org:8000"},
    )
    assert created.status_code == 201
    for host in ("127.0.0.1:8000", "localhost", "[::1]:8000", "[fe80::1]:8000"):
        assert client.get("/api/starter", headers={"Host": host}).status_code == 200
    for host in ("other.example.org", "0.0.0.0:8000", "gui.example.org.evil"):
        assert client.get("/api/starter", headers={"Host": host}).status_code == 400
    refused = client.post(
        "/api/sessions",
        json={"yaml_text": STARTER_YAML},
        headers={"Origin": "http://other.example.org"},
    )
    assert refused.status_code == 403


def test_a_single_string_is_not_taken_for_a_list_of_host_names() -> None:
    with pytest.raises(TypeError, match="not one string"):
        launcher.create_editor_app(allowed_hosts="gui.example.org")
