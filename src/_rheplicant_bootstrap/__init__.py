"""JAX-free bootstrap seams shared by the configuration pipeline."""

__all__ = ["main"]


def main() -> int:
    """Dispatch to the command-line entry point without importing it eagerly."""
    from rheplicant.cli import main as cli_main

    return cli_main()
