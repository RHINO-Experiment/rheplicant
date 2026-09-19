"""Local launcher for the packaged YAML configuration editor.

The editor is deliberately a single-process, same-origin application.  It has
no authentication or tenant boundary, so non-loopback binding requires an
explicit acknowledgement at the command line.

Binding to loopback does not by itself keep a browser page out.  A page on a
domain the attacker controls can re-point that domain at 127.0.0.1 (DNS
rebinding) and then reach the editor as a same-origin client, under its own
host name.  The served app therefore answers only to loopback host names and
to the names given with ``--allowed-host``, and refuses a state-changing
request whose ``Origin`` names any other host.
"""

from __future__ import annotations

import argparse
import ipaddress
import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from urllib.parse import urlsplit

#: Host names the served app always answers to, in their canonical spelling.
LOOPBACK_HOST_NAMES = ("127.0.0.1", "localhost", "[::1]")

#: Methods no route of the editor uses to change state.  Any other method is
#: refused when it carries an ``Origin`` outside the allowed hosts.
_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
_HOST_NAME = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?")
_PORT_SUFFIX = re.compile(r"(?::[0-9]{1,5})?")


def frontend_directory() -> Path:
    """Return the installed, immutable production frontend directory."""
    root = Path(__file__).resolve().with_name("static")
    if not (root / "index.html").is_file():
        raise RuntimeError(
            "The packaged GUI frontend is missing. Reinstall rheplicant from a "
            "wheel or rebuild the frontend assets."
        )
    return root


def _canonical_host(name: str) -> str | None:
    """Return the spelling a host name is compared in, or None if malformed.

    IP literals are compared as addresses, so ``[0:0::1]`` and ``[::1]`` are
    one host; an IPv6 address is bracketed, as it appears in a URL.  Other
    names are compared case-insensitively and must be plain DNS labels.
    """
    bracketed = name.startswith("[") and name.endswith("]")
    try:
        address = ipaddress.ip_address(name[1:-1] if bracketed else name)
    except ValueError:
        if bracketed or _HOST_NAME.fullmatch(name) is None:
            return None
        return name.casefold()
    if address.version == 6:
        return f"[{address.compressed}]"
    return address.compressed


def _host_header_name(value: str) -> str | None:
    """Return the canonical host of a ``Host`` header value, port removed."""
    if value.startswith("["):
        end = value.find("]") + 1
        if end == 0:
            return None
        name, port = value[:end], value[end:]
    else:
        name, colon, rest = value.partition(":")
        port = colon + rest
    if _PORT_SUFFIX.fullmatch(port) is None:
        return None
    return _canonical_host(name)


def _origin_host_name(value: str) -> str | None:
    """Return the canonical host an ``Origin`` header names, if it names one."""
    try:
        name = urlsplit(value).hostname
    except ValueError:
        return None
    return None if name is None else _canonical_host(name)


def _allowed_host_names(names: Iterable[str]) -> frozenset[str]:
    """Return loopback plus ``names`` in canonical spelling, or refuse one."""
    if isinstance(names, str):
        # A lone string would iterate as one-letter host names, all valid.
        raise TypeError("allowed_hosts is a collection of host names, not one string.")
    allowed = set(LOOPBACK_HOST_NAMES)
    for name in names:
        canonical = _canonical_host(name)
        if canonical is None:
            raise ValueError(
                f"--allowed-host {name!r} is not a host name or IP address; "
                "give the name alone, without a scheme or port."
            )
        allowed.add(canonical)
    return frozenset(allowed)


class _HostGuard:
    """Refuse a foreign ``Host`` and a cross-origin state-changing request.

    A plain ASGI middleware rather than Starlette's ``TrustedHostMiddleware``:
    that one splits the header at the first colon, so it refuses
    ``[::1]:8765`` and with it the documented ``--host ::1`` launch.  A
    missing ``Origin`` is accepted, because only browsers send one and
    command-line clients cannot be rebound.
    """

    def __init__(self, app, *, allowed: frozenset[str]) -> None:
        self.app = app
        self.allowed = allowed

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] not in {"http", "websocket"}:
            await self.app(scope, receive, send)
            return
        hosts = _header_values(scope, b"host")
        if len(hosts) != 1 or _host_header_name(hosts[0]) not in self.allowed:
            await _refuse(scope, send, 400, "Invalid Host header.")
            return
        if scope.get("method", "GET") not in _SAFE_METHODS and any(
            _origin_host_name(origin) not in self.allowed
            for origin in _header_values(scope, b"origin")
        ):
            await _refuse(scope, send, 403, "Cross-origin request refused.")
            return
        await self.app(scope, receive, send)


def _header_values(scope, name: bytes) -> list[str]:
    return [
        value.decode("latin-1")
        for key, value in scope.get("headers", ())
        if key.lower() == name
    ]


async def _refuse(scope, send, status: int, message: str) -> None:
    if scope["type"] == "websocket":
        # Closing before accepting is how ASGI refuses a handshake.
        await send({"type": "websocket.close", "code": 1008})
        return
    body = message.encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"text/plain; charset=utf-8"),
                (b"content-length", str(len(body)).encode("ascii")),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


def create_editor_app(*, allowed_hosts: Iterable[str] = ()):
    """Create the same-origin API and bundled frontend application.

    The app answers only to loopback host names plus ``allowed_hosts``; see
    the module docstring for why.
    """
    allowed = _allowed_host_names(allowed_hosts)
    try:
        from rheplicant.gui.api import create_app
    except ModuleNotFoundError as error:
        if error.name in {"fastapi", "pydantic", "starlette"}:
            raise RuntimeError(
                "The GUI dependencies are not installed. Install `rheplicant[gui]`."
            ) from error
        raise
    app = create_app(frontend_directory())
    app.add_middleware(_HostGuard, allowed=allowed)
    return app


def _is_loopback(host: str) -> bool:
    candidate = host.strip().removeprefix("[").removesuffix("]")
    if candidate.casefold() == "localhost":
        return True
    try:
        return ipaddress.ip_address(candidate).is_loopback
    except ValueError:
        return False


def _assert_bind(
    host: str, *, allow_remote: bool, allowed_hosts: Sequence[str] = ()
) -> None:
    """Refuse a bind the caller has not acknowledged or named hosts for.

    This is the one home for the bind rules. Both `serve` (the
    programmatic entry point) and `main` (the CLI entry point) call it, so
    the rules cannot be bypassed by importing and calling `serve` directly --
    only *where* they are checked from differs, not the predicate or message.
    """
    if not allow_remote and not _is_loopback(host):
        raise RuntimeError(
            "Refusing a non-loopback bind without --allow-remote: the editor "
            "has no authentication or multi-user isolation."
        )
    if allow_remote and not allowed_hosts:
        raise RuntimeError(
            "Refusing --allow-remote without --allowed-host: name each host "
            "the browser will use to reach the editor. Every other host name "
            "is refused, which is what stops a DNS-rebinding page."
        )
    try:
        _allowed_host_names(allowed_hosts)
    except ValueError as error:
        raise RuntimeError(str(error)) from None


def _bind_host_names(host: str) -> tuple[str, ...]:
    """Name a loopback bind address as a host too, so it answers to itself."""
    canonical = _canonical_host(host.strip())
    if canonical is None or canonical == "localhost" or not _is_loopback(host):
        return ()
    return (canonical,)


def serve(
    *,
    host: str,
    port: int,
    log_level: str,
    allow_remote: bool = False,
    allowed_hosts: Sequence[str] = (),
) -> None:
    """Run the editor until interrupted."""
    _assert_bind(host, allow_remote=allow_remote, allowed_hosts=allowed_hosts)
    try:
        import uvicorn
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "The GUI server is not installed. Install `rheplicant[gui]`."
        ) from error
    app = create_editor_app(allowed_hosts=(*allowed_hosts, *_bind_host_names(host)))
    uvicorn.run(app, host=host, port=port, log_level=log_level)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rheplicant-gui",
        description="Start the local YAML-as-truth rheplicant config editor.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="bind host (default: loopback)")
    parser.add_argument("--port", type=int, default=8000, help="bind port (default: 8000)")
    parser.add_argument(
        "--log-level",
        choices=("critical", "error", "warning", "info", "debug", "trace"),
        default="info",
    )
    parser.add_argument(
        "--allow-remote",
        action="store_true",
        help="acknowledge that a non-loopback server has no authentication",
    )
    parser.add_argument(
        "--allowed-host",
        action="append",
        default=[],
        dest="allowed_hosts",
        metavar="NAME",
        help=(
            "a host name the editor answers to besides loopback; repeatable, "
            "and required with --allow-remote"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Console entry point for ``rheplicant-gui``."""
    arguments = _parser().parse_args(argv)
    if not 0 <= arguments.port <= 65535:
        raise SystemExit("--port must be between 0 and 65535.")
    allowed_hosts = tuple(arguments.allowed_hosts)
    try:
        _assert_bind(
            arguments.host,
            allow_remote=arguments.allow_remote,
            allowed_hosts=allowed_hosts,
        )
    except RuntimeError as error:
        # Refuse before doing any other work -- this is not a second copy of
        # the rule, it is the CLI calling the one home for it early. `serve`
        # checks again on its own, so the rule holds for programmatic callers
        # too, but main() does not depend on that to give CLI users a prompt
        # SystemExit instead of an unhandled RuntimeError.
        raise SystemExit(str(error)) from error
    serve(
        host=arguments.host,
        port=arguments.port,
        log_level=arguments.log_level,
        allow_remote=arguments.allow_remote,
        allowed_hosts=allowed_hosts,
    )
    return 0


__all__ = ["create_editor_app", "frontend_directory", "main", "serve"]


if __name__ == "__main__":  # pragma: no cover - console-script path
    raise SystemExit(main())
