"""A minimal, JSON-serializable projection of the config-document grammar.

This is not a full JSON-Schema builder (that is a larger task); it exposes the
top-level section names (with required flags), the exit kinds, the
operator/transform catalogs, and the value vocabulary — enough for a UI form to
render the document skeleton and for tool descriptions to enumerate the exits
and operators. Field-level detail per section is a later refinement.
"""

from __future__ import annotations

from typing import Any

from _rheplicant_bootstrap.process import SCHEMA_VERSION


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
    from rheplicant.config import dimensions as _dims
    from rheplicant.config.preflight import (
        _NOT_YET,
        _REQUIRED,
        _RESERVED,
        _SECTIONS,
        deferred_clause,
        reserved_clause,
    )
    from rheplicant.config.sections import runs as _runs

    # No fallback to an empty list. Both tables are module constants of this
    # package, so failing to read one is a defect, and this dict is served
    # verbatim to rheplicant-agent, where an empty `exits` reads as "this
    # layer runs nothing" rather than as a fault.
    exits = list(_runs._KINDS)
    transforms = sorted(_dims._FORMULA_REGISTRY)

    required = set(_REQUIRED)
    # Derived, not re-spelled: `_NOT_YET` and `_RESERVED` are preflight's own
    # refusal tables (`_structural`, in this same package), so a section that
    # moves from deferred to accepted -- or grows a fifth refused name --
    # updates this schema automatically instead of drifting from it.
    deferred = set(_NOT_YET)
    reserved = set(_RESERVED)

    def _status(name: str) -> tuple[str, str | None]:
        """The section's classification AND the sentence that makes it actionable.

        ``_NOT_YET`` is a mapping, not a set: its value is WHO READS THE
        SECTION INSTEAD, and ``_structural`` spends that value on the person
        running the CLI.  Projecting membership alone handed this schema's
        readers ``"deferred"`` and nothing else -- the most misreadable word
        available for ``outputs:``, which is fully supported and merely read
        elsewhere.  ``_NOT_YET``'s own docstring records that this exact
        misreading was fixed once already on the Python side; keeping only
        the classification recreated the loss for every consumer downstream.

        The clause comes from preflight's own helpers rather than being
        rebuilt here.  For ``deferred`` that makes the CLI's refusal and this
        ``reason`` literally one template.  For ``reserved`` it does not:
        ``_structural`` keeps its campaign literal, which three guards pin as
        a literal, so the clause is spelled twice on purpose -- and the
        agreement is enforced by a BEHAVIOURAL test rather than by sharing,
        ``test_schema.py::TestReasonTravelsWithTheStatus``, which drives the
        real loader and compares what it raises.  Either way the two readers
        of one fact cannot diverge unnoticed; only the mechanism differs.

        ``accepted`` has nothing to explain and says so with ``None`` rather
        than an empty string, which a consumer would have to test for
        separately.
        """
        if name in reserved:
            return "reserved", reserved_clause(name)
        if name in deferred:
            return "deferred", deferred_clause(name)
        return "accepted", None

    def _section(name: str) -> dict[str, Any]:
        status, reason = _status(name)
        return {
            "name": name,
            "required": name in required,
            "status": status,
            "reason": reason,
        }

    return {
        # Derived from the bootstrap's SCHEMA_VERSION rather than written
        # again. A STRING here is this contract's own choice -- a consumer
        # branches on it -- and the document carries the int.
        "schemaVersion": str(SCHEMA_VERSION),
        "sections": [_section(name) for name in _SECTIONS],
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
