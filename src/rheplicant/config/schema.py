"""A minimal, JSON-serializable projection of the config-document grammar.

This is not a full JSON-Schema builder (that is a larger task); it exposes the
top-level section names (with required flags), the exit kinds, the
operator/transform catalogs, and the value vocabulary — enough for a UI form to
render the document skeleton and for tool descriptions to enumerate the exits
and operators. Field-level detail per section is a later refinement.
"""

from __future__ import annotations

from typing import Any


def _list(value: Any) -> list[str]:
    return list(value)


def json_schema() -> dict[str, Any]:
    """Return the config-document grammar as a JSON-serializable dict."""
    # Local imports: keep the module import-light and avoid import-order cycles.
    from rheplicant.config import (
        ACCEPTED_UNITS,
        DERIVATIONS,
        FILE_FORMATS,
        RESOURCE_KINDS,
        SHAPE_SYMBOLS,
        VALUE_FORMS,
        VALUE_MODIFIERS,
    )
    from rheplicant.config.preflight import _NOT_YET, _REQUIRED, _RESERVED, _SECTIONS

    exits: list[str] = []
    try:
        from rheplicant.config.sections import runs as _runs
        exits = list(_runs._KINDS)
    except Exception:  # pragma: no cover - defensive; the module is always present
        exits = []

    transforms: list[str] = []
    try:
        from rheplicant.config import dimensions as _dims
        transforms = sorted(_dims._FORMULA_REGISTRY)
    except Exception:  # pragma: no cover - defensive
        transforms = []

    required = set(_REQUIRED)
    # Derived, not re-spelled: `_NOT_YET` and `_RESERVED` are preflight's own
    # refusal tables (`_structural`, in this same package), so a section that
    # moves from deferred to accepted -- or grows a fifth refused name --
    # updates this schema automatically instead of drifting from it.
    deferred = set(_NOT_YET)
    reserved = set(_RESERVED)

    def _status(name: str) -> str:
        if name in reserved:
            return "reserved"
        if name in deferred:
            return "deferred"
        return "accepted"

    return {
        "schemaVersion": "1",
        "sections": [
            {"name": name, "required": name in required, "status": _status(name)}
            for name in _SECTIONS
        ],
        "exits": exits,
        "operators": sorted(
            set(_list(VALUE_FORMS)) | set(_list(VALUE_MODIFIERS)) | set(_list(DERIVATIONS)),
        ),
        "transforms": transforms,
        "catalogs": {
            "acceptedUnits": _list(ACCEPTED_UNITS),
            "resourceKinds": _list(RESOURCE_KINDS),
            "fileFormats": _list(FILE_FORMATS),
            "shapeSymbols": _list(SHAPE_SYMBOLS),
        },
    }


__all__ = ["json_schema"]
