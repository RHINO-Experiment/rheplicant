"""Compatibility entry point for the selected FastAPI editor stack."""

from __future__ import annotations

from rheplicant.gui.api import create_app

app = create_app()

__all__ = ["create_app"]
