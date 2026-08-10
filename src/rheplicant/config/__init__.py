"""rheplicant.config: turn a declarative document into a running twin.

This layer sits ABOVE ``rheplicant.core`` and ``rheplicant.radio``: it imports
both and neither imports it. That direction is what keeps ``core``
extractable, and ``tests/core/test_layering.py`` enforces it mechanically.

Plans 1A and 1B shipped the grammar underneath: the eighteen value forms with
their nine modifiers, the delivery rule that decides scalar-vs-traced off the
destination field, the path grammar, and ``resources.<kind>.<name>`` with
``ref``-is-identity construction.
"""

from rheplicant.config.context import ResolutionContext
from rheplicant.config.delivery import FieldSpec, deliver, field_specs
from rheplicant.config.derive import DERIVATIONS
from rheplicant.config.errors import ConfigError
from rheplicant.config.files import FILE_FORMATS
from rheplicant.config.paths import (
    ResolvedPath,
    compile_path,
    parse_path,
    resolve_path_on,
)
from rheplicant.config.resources import RESOURCE_KINDS, BuiltResources, build_resources
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
    "RESOURCE_KINDS",
    "SHAPE_SYMBOLS",
    "VALUE_FORMS",
    "VALUE_MODIFIERS",
    "BuiltResources",
    "ConfigError",
    "FieldSpec",
    "ResolutionContext",
    "ResolvedPath",
    "ResolvedValue",
    "ShapeScope",
    "Unit",
    "build_resources",
    "canonical_unit",
    "compile_path",
    "convert_to_canonical",
    "deliver",
    "field_specs",
    "parse_path",
    "resolve_extent",
    "resolve_path_on",
    "resolve_value",
]
