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

Plan 3A adds the pre-flight pass: ``preflight`` takes a variant-applied
document and returns a ``Report`` of ``Finding``s -- every check decidable
from the document's text plus static class and graph introspection, run
BEFORE ``load_document`` reads a beam. It COLLECTS rather than raising, so a
document wrong four ways is refused once; ``Report.raise_if_refused`` turns
the first refusal back into the ``ConfigError`` this layer has always
raised. ``ConfigWarning`` is the non-fatal sibling ``sections/inference.py``'s
``_MODES`` has parsed since 2B with nothing to consume it. Four names, and
they are the four a CALLER touches: the registry (``CHECKS``, ``register``)
stays module-local for the reason ``EXECUTORS`` does -- a check id is
something the schema says, not something a caller registers.

Plan 3B adds two more passes -- the **axes** pass, run one line above
``build_resources`` over the resolved time and frequency grids, and the
**built** pass, run when the twin, the state and the resources all exist and
``load_document`` is ready to return -- and **adds no name here at all**.
That is a decision and not an oversight, and it is mechanical rather than
stylistic: ``inflight.axes`` takes an ``Axes``, whose ``runtime`` and
``observation`` fields come from ``build_runtime`` and ``build_observation``,
and **neither builder is exported**, so an exported ``axes()`` would be an
entry point no caller could construct an argument for. ``inflight.built`` is
further out again -- its payload carries the twin, the state and the built
resources. Both passes are things ``load_document`` runs *for* a caller, and
``preflight`` remains the one a front-end calls itself, because it is the one
that answers before anything is built. ``AXIS_CHECKS`` and ``BUILT_CHECKS``
follow ``CHECKS`` for the reason above. Pinned, either way, by
``test_config_surface.py::TestPlan3BsWiringAndItsSurface``.

Plan 3C adds the fourth pass -- **post-flight**, run after ``build_inference``
and before ``load_document`` returns, holding the checks that have to RUN the
thing they are deciding about (``check_linearity``'s ``len(scales) + 1``
forward passes per linear claim, ``identifiability``'s ``jacfwd`` plus a dense
SVD, ``prior_sensitivity``'s two Newton solves on top of that) -- and adds
**two** names here: ``gates`` and ``Gate``.

``priced``, the post-flight entry point, is NOT one of them, and the reason is
3B's precedent rather than taste: it is a pass ``load_document`` runs on the
caller's behalf, its payload carries the twin, the state, the built resources
and the resolved gates, and by the time a caller could construct one the
answer is already on ``ConfiguredRun.report``. ``preflight`` stays the one
pass a front-end calls itself because it is the one that answers *before*
anything is built.

``gates`` is the opposite shape and that is why it lands. It is a pure
function of the raw ``inference.checks:`` mapping -- text the caller already
holds -- and it answers, for free and before any money is spent, the question
``preflight`` cannot: not "is this document legal" but "which of the three
checks will actually run, in what mode, and which are simply off". That is
schema §10's "Validate" panel and it is what Plan 4 will ask when it wants the
effective mode of a check rather than the declaration ``InferenceBuild.checks``
keeps. ``Gate`` follows ``gates`` exactly as ``Finding`` follows ``Report``: a
caller CALLS ``gates`` and READS the ``Gate``s it hands back.

It assumes the section has already passed the pre-flight grammar
(``A1.checks``/``A37``): on a section that has not, a non-numeric ``rtol:``
raises ``ValueError`` and an unknown ``mode:`` word becomes a ``Gate`` in that
state, which ``runs()`` reports as standing down. Call ``preflight`` first, or
read ``gate.state`` against ``gating.STATES`` before trusting it.

``MODES``, ``STATES``, ``DEFAULT_MODE``, ``CHECK_ID``, ``verdict`` and
``check_gates`` stay module-local for the reason ``CHECKS`` does -- the words a
document may write and the defaults they fall back to are things the SCHEMA
says (§4.7.8, §11.15), not things a caller chooses -- and a caller who wants
the defaults asks ``gates(None)``, which applies them.

Binding ``preflight`` here SHADOWS the ``rheplicant.config.preflight``
subpackage attribute: after this module runs, ``config.preflight`` is the
function, and the module is reached through ``sys.modules`` or
``importlib.import_module``. That is pinned by
``test_exporting_preflight_shadows_the_subpackage_and_that_is_pinned`` rather
than left as a surprise, because a reader of ``config.preflight.model`` gets
an ``AttributeError`` and no explanation.
"""

from rheplicant.config.context import ResolutionContext
from rheplicant.config.delivery import FieldSpec, deliver, field_specs
from rheplicant.config.derive import DERIVATIONS
from rheplicant.config.document import ConfiguredRun, load_document, run_forward
from rheplicant.config.errors import ConfigError
from rheplicant.config.files import FILE_FORMATS
from rheplicant.config.findings import ConfigWarning, Finding, Report
from rheplicant.config.gating import Gate, gates
from rheplicant.config.layering import apply_variant, recursive_update
from rheplicant.config.paths import (
    ResolvedPath,
    compile_path,
    parse_path,
    resolve_path_on,
)
from rheplicant.config.preflight import preflight
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
    "ConfigWarning",
    "ConfiguredRun",
    "FieldSpec",
    "Finding",
    "Gate",
    "InferenceBuild",
    "Report",
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
    "gates",
    "load_document",
    "parse_path",
    "preflight",
    "recursive_update",
    "resolve_extent",
    "resolve_path_on",
    "resolve_value",
    "run_document",
    "run_forward",
]
