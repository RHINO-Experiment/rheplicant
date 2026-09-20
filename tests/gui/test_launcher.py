from __future__ import annotations

import re
import socket
import subprocess
import sys
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


def refused(capsys, call, *, naming: str) -> None:
    """A refused invocation: exit ``REFUSAL_EXIT``, message on stderr.

    It used to be ``pytest.raises(SystemExit, match=...)``, which passed
    because ``SystemExit("message")`` carries the text as its CODE and exits
    **1**. ``docs/config-cli.md`` reserves 1 for an internal failure and 2 for
    a refusal, so this command was reporting a rejected invocation as a fault
    of its own. The message is the same; where it goes and what the process
    returns are not.
    """
    with pytest.raises(SystemExit) as excinfo:
        call()
    assert excinfo.value.code == launcher.REFUSAL_EXIT, excinfo.value.code
    assert naming in capsys.readouterr().err


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
        (root / reference).read_text() for reference in references if reference.endswith(".js")
    )
    assert "/api/sessions" in javascript
    assert "Rheplicant configuration workbench" in javascript


def test_bundled_app_serves_the_editor_and_api_from_one_origin() -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx2")
    from fastapi.testclient import TestClient

    # The launcher's app answers only to loopback host names, so the client
    # addresses it the way a browser on this machine would.
    client = TestClient(launcher.create_editor_app(), base_url="http://127.0.0.1:8000")
    index = client.get("/")
    assert index.status_code == 200
    assert "Rheplicant YAML config editor" in index.text

    created = client.post("/api/sessions", json={"yaml_text": BASE})
    assert created.status_code == 201
    assert created.json()["document"]["yaml_text"] == BASE


def test_launcher_defaults_to_loopback_and_remote_binding_is_explicit(monkeypatch, capsys) -> None:
    calls: list[tuple[str, int, str]] = []

    def fake_serve(
        *,
        host: str,
        port: int,
        log_level: str,
        allow_remote: bool = False,
        allowed_hosts: tuple[str, ...] = (),
    ) -> None:
        calls.append((host, port, log_level))

    monkeypatch.setattr(launcher, "serve", fake_serve)
    assert launcher.main(["--port", "9123", "--log-level", "error"]) == 0
    assert calls == [("127.0.0.1", 9123, "error")]

    refused(capsys, lambda: launcher.main(["--host", "0.0.0.0"]), naming="--allow-remote")
    assert len(calls) == 1

    assert launcher.main(["--host", "::1", "--port", "9124"]) == 0
    refused(
        capsys,
        lambda: launcher.main(["--host", "0.0.0.0", "--allow-remote", "--port", "9125"]),
        naming="--allowed-host",
    )
    assert len(calls) == 2
    assert (
        launcher.main(
            [
                "--host",
                "0.0.0.0",
                "--allow-remote",
                "--allowed-host",
                "gui.example.org",
                "--port",
                "9125",
            ]
        )
        == 0
    )
    assert calls[-1] == ("0.0.0.0", 9125, "info")


def test_serve_refuses_a_non_loopback_bind_without_acknowledgement(monkeypatch) -> None:
    """serve() is the programmatic entry point: a caller who imports it
    directly, bypassing main()'s own early check, must still be refused a
    non-loopback bind without acknowledgement. Before this test the guard
    lived only in main(), so `from rheplicant.gui.launcher import serve;
    serve(host="0.0.0.0", ...)` bound every interface with no authentication
    and no acknowledgement at all.

    uvicorn.run is monkeypatched, not serve() itself, so this exercises the
    real function body -- including the guard -- rather than a stand-in.
    """
    pytest.importorskip("uvicorn")
    import uvicorn

    calls: list[tuple[tuple, dict]] = []
    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: calls.append((args, kwargs)))

    with pytest.raises(RuntimeError) as excinfo:
        launcher.serve(host="0.0.0.0", port=8000, log_level="info")
    assert str(excinfo.value) == (
        "Refusing a non-loopback bind without --allow-remote: the editor "
        "has no authentication or multi-user isolation."
    )
    assert calls == []


def test_serve_allows_a_non_loopback_bind_when_remote_is_acknowledged(monkeypatch) -> None:
    """The acknowledgement is the `allow_remote=True` keyword, not merely the
    absence of an exception: serve() must actually reach uvicorn.run once a
    caller has explicitly accepted the risk.
    """
    pytest.importorskip("uvicorn")
    import uvicorn

    calls: list[tuple[tuple, dict]] = []
    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: calls.append((args, kwargs)))

    launcher.serve(
        host="0.0.0.0",
        port=8000,
        log_level="info",
        allow_remote=True,
        allowed_hosts=("gui.example.org",),
    )
    assert len(calls) == 1
    _, kwargs = calls[0]
    assert kwargs["host"] == "0.0.0.0"
    assert kwargs["port"] == 8000
    assert kwargs["log_level"] == "info"


def test_serve_allows_loopback_ipv4_without_acknowledgement(monkeypatch) -> None:
    """A loopback bind carries no remote-exposure risk, so serve() must not
    demand allow_remote for it.
    """
    pytest.importorskip("uvicorn")
    import uvicorn

    calls: list[tuple[tuple, dict]] = []
    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: calls.append((args, kwargs)))

    launcher.serve(host="127.0.0.1", port=8000, log_level="info")
    assert len(calls) == 1


def test_serve_allows_localhost_without_acknowledgement(monkeypatch) -> None:
    """ "localhost" is not a literal loopback address but `_is_loopback`
    treats it as one; serve() must honour the same exemption rather than
    only recognising numeric loopback addresses.
    """
    pytest.importorskip("uvicorn")
    import uvicorn

    calls: list[tuple[tuple, dict]] = []
    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: calls.append((args, kwargs)))

    launcher.serve(host="localhost", port=8000, log_level="info")
    assert len(calls) == 1


GUI_SERVER_MISSING = "The GUI server is not installed. Install `rheplicant[gui]`."
GUI_DEPENDENCIES_MISSING = "The GUI dependencies are not installed. Install `rheplicant[gui]`."


def test_main_exits_with_the_message_when_the_server_is_not_installed(monkeypatch) -> None:
    """Without the `gui` extra the console script used to end in a RuntimeError
    traceback. The message was already right; the exit is now a SystemExit
    carrying it, which Python prints without a traceback."""
    monkeypatch.setitem(sys.modules, "uvicorn", None)
    with pytest.raises(SystemExit) as excinfo:
        launcher.main([])
    assert excinfo.value.code == GUI_SERVER_MISSING


def test_main_exits_with_the_message_when_the_api_stack_is_not_installed(monkeypatch) -> None:
    pytest.importorskip("uvicorn")
    import uvicorn

    calls: list[object] = []
    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: calls.append(args))
    monkeypatch.setitem(sys.modules, "fastapi", None)
    monkeypatch.delitem(sys.modules, "rheplicant.gui.api", raising=False)
    with pytest.raises(SystemExit) as excinfo:
        launcher.main([])
    assert excinfo.value.code == GUI_DEPENDENCIES_MISSING
    assert calls == []


def test_the_console_script_prints_no_traceback_without_the_gui_extra() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.modules['uvicorn'] = None; "
            "from rheplicant.gui.launcher import main; raise SystemExit(main([]))",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 1
    assert "Traceback" not in completed.stderr
    assert completed.stderr.strip() == GUI_SERVER_MISSING


def _resolves_to(monkeypatch, *addresses: str) -> None:
    def fake_getaddrinfo(host, port, *args, **kwargs):
        assert host.casefold() == "localhost"
        return [
            (
                socket.AF_INET6 if ":" in address else socket.AF_INET,
                socket.SOCK_STREAM,
                6,
                "",
                (address, 0),
            )
            for address in addresses
        ]

    monkeypatch.setattr(launcher.socket, "getaddrinfo", fake_getaddrinfo)


def test_localhost_is_a_loopback_bind_only_when_every_address_is_loopback(
    monkeypatch, capsys
) -> None:
    """ "localhost" used to be accepted by name. It is now resolved, and a
    hosts file that points it anywhere else needs --allow-remote like any
    other non-loopback bind."""
    pytest.importorskip("uvicorn")
    import uvicorn

    calls: list[object] = []
    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: calls.append(args))

    _resolves_to(monkeypatch, "127.0.0.1", "203.0.113.7")
    with pytest.raises(RuntimeError, match="--allow-remote"):
        launcher.serve(host="localhost", port=8000, log_level="info")
    refused(capsys, lambda: launcher.main(["--host", "LOCALHOST"]), naming="--allow-remote")

    _resolves_to(monkeypatch)
    with pytest.raises(RuntimeError, match="--allow-remote"):
        launcher.serve(host="localhost", port=8000, log_level="info")
    assert calls == []

    _resolves_to(monkeypatch, "127.0.0.1", "::1")
    launcher.serve(host="localhost", port=8000, log_level="info")
    assert len(calls) == 1


def test_localhost_that_does_not_resolve_is_not_a_loopback_bind(monkeypatch, capsys) -> None:
    pytest.importorskip("uvicorn")
    import uvicorn

    calls: list[object] = []
    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: calls.append(args))

    def unresolvable(*args, **kwargs):
        raise socket.gaierror(socket.EAI_NONAME, "nodename nor servname provided")

    monkeypatch.setattr(launcher.socket, "getaddrinfo", unresolvable)
    refused(capsys, lambda: launcher.main(["--host", "localhost"]), naming="--allow-remote")
    assert calls == []


def test_the_console_script_exits_two_for_a_refused_invocation() -> None:
    """The contract at the process level, which is where a caller reads it.

    In-process tests see ``SystemExit``; a shell sees the number. Both halves
    are asserted because they were both wrong together: the message was the
    exit CODE, so the process returned 1 and printed the text -- indisputably
    a refusal reported as an internal failure.
    """
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from rheplicant.gui.launcher import main; "
            "raise SystemExit(main(['--host', '0.0.0.0']))",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    # NOT compared against `launcher.REFUSAL_EXIT`, which is what this
    # assertion said first and what made it unable to fail: reading the
    # constant means lowering the constant back to 1 keeps the test green,
    # and "the two commands agree" was the whole claim. The number is
    # measured from `rheplicant` refusing something, in this same test.
    refusal = subprocess.run(
        [
            sys.executable,
            "-c",
            "from _rheplicant_bootstrap.cli import main; "
            "raise SystemExit(main(['run', '/nonexistent/nothing.yaml']))",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert refusal.returncode not in (0, 1), refusal.stderr
    assert completed.returncode == refusal.returncode
    assert "--allow-remote" in completed.stderr
    assert "Traceback" not in completed.stderr
