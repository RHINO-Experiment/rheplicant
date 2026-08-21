"""JAX-free bootstrap seams shared by the configuration pipeline."""

from __future__ import annotations

from collections.abc import Sequence

__all__ = ["main", "run_embedded_config"]


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch without importing the CLI or scientific package eagerly."""
    from .cli import main as cli_main

    return cli_main(argv)


def run_embedded_config(**arguments: object) -> int:
    """Enter the shared dispatcher with generation-time source snapshots."""
    from .entry import run_embedded_config as embedded_main

    return embedded_main(**arguments)  # type: ignore[arg-type]
