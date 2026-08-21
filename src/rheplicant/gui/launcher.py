"""Local launcher for the packaged YAML configuration editor.

The editor is deliberately a single-process, same-origin application.  It has
no authentication or tenant boundary, so non-loopback binding requires an
explicit acknowledgement at the command line.
"""

from __future__ import annotations

import argparse
import ipaddress
from collections.abc import Sequence
from pathlib import Path


def frontend_directory() -> Path:
    """Return the installed, immutable production frontend directory."""
    root = Path(__file__).resolve().with_name("static")
    if not (root / "index.html").is_file():
        raise RuntimeError(
            "The packaged GUI frontend is missing. Reinstall rheplicant from a "
            "wheel or rebuild the frontend assets."
        )
    return root


def create_editor_app():
    """Create the same-origin API and bundled frontend application."""
    try:
        from rheplicant.gui.api import create_app
    except ModuleNotFoundError as error:
        if error.name in {"fastapi", "pydantic", "starlette"}:
            raise RuntimeError(
                "The GUI dependencies are not installed. Install `rheplicant[gui]`."
            ) from error
        raise
    return create_app(frontend_directory())


def _is_loopback(host: str) -> bool:
    candidate = host.strip().removeprefix("[").removesuffix("]")
    if candidate.casefold() == "localhost":
        return True
    try:
        return ipaddress.ip_address(candidate).is_loopback
    except ValueError:
        return False


def serve(*, host: str, port: int, log_level: str) -> None:
    """Run the editor until interrupted."""
    try:
        import uvicorn
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "The GUI server is not installed. Install `rheplicant[gui]`."
        ) from error
    uvicorn.run(create_editor_app(), host=host, port=port, log_level=log_level)


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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Console entry point for ``rheplicant-gui``."""
    arguments = _parser().parse_args(argv)
    if not 0 <= arguments.port <= 65535:
        raise SystemExit("--port must be between 0 and 65535.")
    if not arguments.allow_remote and not _is_loopback(arguments.host):
        raise SystemExit(
            "Refusing a non-loopback bind without --allow-remote: the editor "
            "has no authentication or multi-user isolation."
        )
    serve(host=arguments.host, port=arguments.port, log_level=arguments.log_level)
    return 0


__all__ = ["create_editor_app", "frontend_directory", "main", "serve"]


if __name__ == "__main__":  # pragma: no cover - console-script path
    raise SystemExit(main())
