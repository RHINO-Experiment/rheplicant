"""rheplicant.config: turn a declarative document into a running twin.

This layer sits ABOVE ``rheplicant.core`` and ``rheplicant.radio``: it imports
both and neither imports it. That direction is what keeps ``core``
extractable, and ``tests/core/test_layering.py`` enforces it mechanically.

Plan 1A ships the value grammar only -- the eight forms of a value node, the
eight modifiers, and the rule that decides whether a resolved value reaches a
field as a Python scalar or as a traced array.
"""

from rheplicant.config.context import ResolutionContext
from rheplicant.config.delivery import FieldSpec, deliver, field_specs
from rheplicant.config.derive import DERIVATIONS
from rheplicant.config.errors import ConfigError
from rheplicant.config.files import FILE_FORMATS
from rheplicant.config.symbols import SHAPE_SYMBOLS, ShapeScope, resolve_extent
from rheplicant.config.units import (
    ACCEPTED_UNITS,
    Unit,
    canonical_unit,
    convert_to_canonical,
)
from rheplicant.config.values import (
    VALUE_FORMS,
    VALUE_MODIFIERS,
    ResolvedValue,
    resolve_value,
)

__all__ = [
    "ACCEPTED_UNITS",
    "DERIVATIONS",
    "FILE_FORMATS",
    "SHAPE_SYMBOLS",
    "VALUE_FORMS",
    "VALUE_MODIFIERS",
    "ConfigError",
    "FieldSpec",
    "ResolutionContext",
    "ResolvedValue",
    "ShapeScope",
    "Unit",
    "canonical_unit",
    "convert_to_canonical",
    "deliver",
    "field_specs",
    "resolve_extent",
    "resolve_value",
]
