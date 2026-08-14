"""rheplicant.config: turn a declarative document into a running twin.

This layer sits ABOVE ``rheplicant.core`` and ``rheplicant.radio``: it imports
both and neither imports it. That direction is what keeps ``core``
extractable, and ``tests/core/test_layering.py`` enforces it mechanically.

Plans 1A and 1B shipped the grammar underneath: the eighteen value forms with
their nine modifiers, the delivery rule that decides scalar-vs-traced off the
destination field, the path grammar, and ``resources.<kind>.<name>`` with
``ref``-is-identity construction. Plan 2A adds the section loaders and the
document layer: ``load_document`` turns a parsed document's
``runtime:``/``observation:``/``model:``/``inference:`` into a
``ConfiguredRun`` and ``run_forward`` evaluates it. Plan 2B adds the fitting
exits: ``run_document`` executes the ``runs:`` a document declares, each
against the ``InferenceBuild`` its ``inference:`` section produced, and hands
back one ``RunResult`` per run. Plan 2C adds nine more kinds -- the conjugate
family, the cheap diagnostics, ``mmodes`` and ``predict`` -- plus ``reuse:``,
which lets one run read an earlier one's product. They add no name here on
purpose: a run kind is something a document *says*, reached through the
``run_document`` above, and the table that dispatches them
(``sections.exit_support.EXECUTORS``) is wiring rather than surface.

Plan 2D adds the last two exits -- ``nuts`` (numpyro's NUTS over the whole
parameter space) and ``npe`` (the amortized neural posterior, configured by
``inference.npe:``) -- and gives ``RunResult`` a ``variant``, so a ``predict``
reusing an earlier run can refuse to mix two builds. It adds no name here
either, and for the same reason: a caller receives a product through
``run_document``, never constructs one, so the product types stay module-local
and free to grow the ``report:``/``timings:`` fields Plan 4 will want.
"""

from rheplicant.config.context import ResolutionContext
from rheplicant.config.delivery import FieldSpec, deliver, field_specs
from rheplicant.config.derive import DERIVATIONS
from rheplicant.config.document import ConfiguredRun, load_document, run_forward
from rheplicant.config.errors import ConfigError
from rheplicant.config.files import FILE_FORMATS
from rheplicant.config.layering import apply_variant, recursive_update
from rheplicant.config.paths import (
    ResolvedPath,
    compile_path,
    parse_path,
    resolve_path_on,
)
from rheplicant.config.resources import RESOURCE_KINDS, BuiltResources, build_resources
from rheplicant.config.sections import ingest as _ingest  # noqa: F401  (registers rhino_hdf5)
from rheplicant.config.sections.inference import InferenceBuild
from rheplicant.config.sections.runs import RunResult, run_document
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
    "ConfiguredRun",
    "FieldSpec",
    "InferenceBuild",
    "ResolutionContext",
    "ResolvedPath",
    "ResolvedValue",
    "RunResult",
    "ShapeScope",
    "Unit",
    "apply_variant",
    "build_resources",
    "canonical_unit",
    "compile_path",
    "convert_to_canonical",
    "deliver",
    "field_specs",
    "load_document",
    "parse_path",
    "recursive_update",
    "resolve_extent",
    "resolve_path_on",
    "resolve_value",
    "run_document",
    "run_forward",
]
