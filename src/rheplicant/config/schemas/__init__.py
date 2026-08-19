"""Packaged JSON Schemas for Plan 4A audit envelopes."""

from __future__ import annotations

import json
from importlib import resources

from _rheplicant_bootstrap.errors import ConfigError

SCHEMA_NAMES = ("provenance-v1", "diagnostics-v1")


def load_schema(name: str) -> dict:
    """Load one known packaged Draft-2020-12 schema."""
    if name not in SCHEMA_NAMES:
        raise ConfigError(f"unknown audit schema {name!r}.")
    payload = resources.files(__package__).joinpath(f"{name}.schema.json").read_bytes()
    return json.loads(payload)


__all__ = ["SCHEMA_NAMES", "load_schema"]
