"""Neutral exceptions shared across bootstrap and public rheplicant modules."""

from __future__ import annotations


class DirtError(Exception):
    """Base class for all public rheplicant errors."""


class ConfigError(DirtError, ValueError):
    """A configuration document was refused."""

    def __init__(self, *args: object, report: object | None = None) -> None:
        super().__init__(*args)
        self.report = report


DirtError.__module__ = "rheplicant.core.errors"
ConfigError.__module__ = "rheplicant.config.errors"
