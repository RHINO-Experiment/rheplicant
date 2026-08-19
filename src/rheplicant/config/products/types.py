"""Frozen records for detached scientific product bytes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ProductFile:
    relative_path: str
    payload: bytes
    selector: str
    run: str | None
    kind: str | None
    format: str
    metadata: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class ProductOmission:
    selector: str
    run: str | None
    kind: str | None
    reason: str


@dataclass(frozen=True, slots=True)
class ProductBundle:
    files: tuple[ProductFile, ...]
    manifest: bytes


__all__ = ["ProductBundle", "ProductFile", "ProductOmission"]
