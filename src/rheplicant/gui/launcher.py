"""Local launcher for the packaged YAML configuration editor.

The editor is deliberately a single-process, same-origin application.  It has
no authentication or tenant boundary, so non-loopback binding requires an
explicit acknowledgement at the command line.

Binding to loopback does not by itself keep a browser page out.  A page on a
domain the attacker controls can re-point that domain at 127.0.0.1 (DNS
rebinding) and then reach the editor as a same-origin client, under its own
host name.  The served app therefore answers only to loopback host names and
to the names given with ``--allowed-host``, and refuses a state-changing
request whose ``Origin`` is not the request's own host and port.
"""

from __future__ import annotations

import argparse
import ipaddress
import re
import socket
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from urllib.parse import urlsplit

#: Host names the served app always answers to, in their canonical spelling.
LOOPBACK_HOST_NAMES = ("127.0.0.1", "localhost", "[::1]")

#: Methods no route of the editor uses to change state.  Any other method is
#: refused when it carries an ``Origin`` other than the request's own.
_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
#: The port a URL or ``Host`` header without one means, by scheme.
_DEFAULT_PORTS = {"http": 80, "ws": 80, "https": 443, "wss": 443}
_HOST_NAME = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?")
_PORT_SUFFIX = re.compile(r"(?::[0-9]{1,5})?")


class _MissingInstall(RuntimeError):
    """The GUI extra or the packaged frontend is not installed.

    ``main`` reports this as a one-line exit rather than a traceback: the
    message already says what to install, and nothing in the stack helps.
    """


def frontend_directory() -> Path:
    """Return the installed, immutable production frontend directory."""
    root = Path(__file__).resolve().with_name("static")
    if not (root / "index.html").is_file():
        raise _MissingInstall(
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


def _host_header(value: str) -> tuple[str, int | None] | None:
    """Split a ``Host`` header value into its canonical host and its port.

    The port is None when the header gives none. None as a whole means the
    value is malformed.
    """
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
    canonical = _canonical_host(name)
    if canonical is None:
        return None
    return canonical, int(port[1:]) if port else None


def _origin(value: str) -> tuple[str, int] | None:
    """Return the canonical host and effective port an ``Origin`` names.

    None for ``null``, a scheme without a default port and no explicit one,
    or anything that does not parse as ``scheme://host[:port]``.
    """
    try:
        parts = urlsplit(value)
        port = parts.port
    except ValueError:
        return None
    host = None if parts.hostname is None else _canonical_host(parts.hostname)
    if port is None:
        port = _DEFAULT_PORTS.get(parts.scheme)
    if host is None or port is None:
        return None
    return host, port


def _allowed_host_names(
    names: Iterable[str], *, label: str = "allowed_hosts entry"
) -> frozenset[str]:
    """Return loopback plus ``names`` in canonical spelling, or refuse one.

    ``label`` names the refused value the way its caller spelled it: the
    keyword argument from the API, the flag from the command line.
    """
    if isinstance(names, str):
        # A lone string would iterate as one-letter host names, all valid.
        raise TypeError("allowed_hosts is a collection of host names, not one string.")
    allowed = set(LOOPBACK_HOST_NAMES)
    for name in names:
        canonical = _canonical_host(name)
        if canonical is None:
            raise ValueError(
                f"{label} {name!r} is not a host name or IP address; "
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
    command-line clients cannot be rebound.  A present ``Origin`` must be
    the request's own: the same host and port as ``Host``, with a missing
    port read as the scheme's default.  A loopback ``Origin`` on another port
    is another web server on this machine, not the editor.
    """

    def __init__(self, app, *, allowed: frozenset[str]) -> None:
        self.app = app
        self.allowed = allowed

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] not in {"http", "websocket"}:
            await self.app(scope, receive, send)
            return
        hosts = _header_values(scope, b"host")
        target = _host_header(hosts[0]) if len(hosts) == 1 else None
        if target is None or target[0] not in self.allowed:
            await _refuse(scope, send, 400, "Invalid Host header.")
            return
        host, port = target
        if port is None:
            port = _DEFAULT_PORTS.get(scope.get("scheme", "http"))
        if _may_change_state(scope) and any(
            _origin(origin) != (host, port)
            for origin in _header_values(scope, b"origin")
        ):
            await _refuse(scope, send, 403, "Cross-origin request refused.")
            return
        await self.app(scope, receive, send)


def _may_change_state(scope) -> bool:
    """Whether the Origin rule applies to this scope.

    A websocket scope has no method, and browsers open cross-origin
    websockets without asking the server first, so a handshake is treated
    like a write rather than like a GET.
    """
    return scope["type"] == "websocket" or scope["method"] not in _SAFE_METHODS


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
            raise _MissingInstall(
                "The GUI dependencies are not installed. Install `rheplicant[gui]`."
            ) from error
        raise
    app = create_app(frontend_directory())
    app.add_middleware(_HostGuard, allowed=allowed)
    return app


def _is_loopback(host: str) -> bool:
    candidate = host.strip().removeprefix("[").removesuffix("]")
    if candidate.casefold() == "localhost":
        return _resolves_to_loopback_only(candidate)
    try:
        return ipaddress.ip_address(candidate).is_loopback
    except ValueError:
        return False


def _resolves_to_loopback_only(name: str) -> bool:
    """Whether every address ``name`` resolves to is a loopback address.

    ``localhost`` is a name, and the bind goes to whatever the resolver
    returns for it; a hosts file can point it at an external interface.
    """
    try:
        found = socket.getaddrinfo(name, None)
    except (OSError, UnicodeError):
        return False
    addresses = {entry[4][0] for entry in found}
    try:
        return bool(addresses) and all(
            ipaddress.ip_address(address).is_loopback for address in addresses
        )
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
        _allowed_host_names(allowed_hosts, label="--allowed-host")
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
        raise _MissingInstall(
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
            "and required with --allow-remote. Also accepted without "
            "--allow-remote on a loopback bind, for a reverse proxy on this "
            "machine. A page served from a listed name can reach the editor "
            "through DNS rebinding, so list only names whose DNS you control"
        ),
    )
    return parser


#: What ``rheplicant`` exits for a usage or configuration refusal, and now
#: what this command exits for one too.
#:
#: ``raise SystemExit("message")`` prints the message and exits **1**, which
#: is what a bad ``--host`` or ``--port`` did here. ``docs/config-cli.md``
#: says 1 is "unexpected package or internal failure; a traceback is
#: printed" and 2 is the refusal -- so this command reported a rejected
#: invocation as an internal fault, and the page did not show the
#: disagreement because it documented only ``rheplicant``.
REFUSAL_EXIT = 2


def _refuse_invocation(message: str, cause: BaseException | None = None):
    """Print a refusal on stderr and exit :data:`REFUSAL_EXIT`.

    Named for the invocation rather than just ``_refuse``, because this module
    already has a ``_refuse`` -- the ASGI one that sends a 400 to a request
    with a bad ``Host`` header. Shadowing it turned 79 host-guard tests red
    at once, which is the cheap version of that mistake.
    """
    print(message, file=sys.stderr)
    raise SystemExit(REFUSAL_EXIT) from cause


def main(argv: Sequence[str] | None = None) -> int:
    """Console entry point for ``rheplicant-gui``."""
    arguments = _parser().parse_args(argv)
    if not 0 <= arguments.port <= 65535:
        _refuse_invocation("--port must be between 0 and 65535.")
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
        _refuse_invocation(str(error), error)
    try:
        serve(
            host=arguments.host,
            port=arguments.port,
            log_level=arguments.log_level,
            allow_remote=arguments.allow_remote,
            allowed_hosts=allowed_hosts,
        )
    except _MissingInstall as error:
        # NOT a refusal: the invocation was fine and the environment is not.
        # It keeps exit 1 and the message-carrying SystemExit deliberately.
        raise SystemExit(str(error)) from error
    return 0


__all__ = ["REFUSAL_EXIT", "create_editor_app", "frontend_directory", "main", "serve"]


if __name__ == "__main__":  # pragma: no cover - console-script path
    raise SystemExit(main())
