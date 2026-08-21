"""Build the GUI widget census from config registries and parser vocabularies."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

from _rheplicant_bootstrap.process import _RUNTIME_KEYS
from rheplicant.config.delivery import field_specs, mode_of
from rheplicant.config.dimensions import (
    DimensionSpec,
    registered_dimension_rows,
    signature_token,
)
from rheplicant.config.kinds.bases import BASIS_KINDS
from rheplicant.config.kinds.beams import (
    _FORMAT_KEYS,
    BEAM_FORMATS,
    BEAM_NORMALIZATIONS,
)
from rheplicant.config.kinds.projectors import _ENGINE_KEYS, ENGINES, OPTIMIZATIONS
from rheplicant.config.kinds.s_params import (
    _CABLE_KEYS,
    _TERMINATION_KEYS,
    _TOUCHSTONE_KEYS,
    COMPONENTS,
    S_PARAM_KINDS,
    TERMINATIONS,
)
from rheplicant.config.kinds.sky_models import _ALLOWED_KEYS, SKY_KINDS
from rheplicant.config.resources import RESOURCE_KINDS
from rheplicant.config.sections.benchmark import _BENCHMARK_KEYS
from rheplicant.config.sections.benchmark import _METRICS as BENCHMARK_METRICS
from rheplicant.config.sections.comparison import _COMPARE_KEYS
from rheplicant.config.sections.comparison import _METRICS as COMPARE_METRICS
from rheplicant.config.sections.compose import many_shape_problem
from rheplicant.config.sections.conjugate import (
    _CONDITION_KEYS,
    _GCR_KEYS,
    _GLS_KEYS,
    _WIENER_KEYS,
)
from rheplicant.config.sections.diagnostics import (
    _GRADIENT_KEYS,
    _IDENTIFIABILITY_KEYS,
    _MMODES_KEYS,
    _PREDICT_KEYS,
    _SCORE_KEYS,
)
from rheplicant.config.sections.exit_support import (
    DEFERRED_CHECKS,
    EXECUTORS,
    PARSERS,
    PRE_EXECUTORS,
)
from rheplicant.config.sections.exits import (
    _ADAM_DEFAULTS,
    _ESTIMATE_DEFAULTS,
    _ESTIMATE_KEYS,
    _OPTIMIZE_KEYS,
    _SAMPLE_DEFAULTS,
    _SAMPLE_KEYS,
)
from rheplicant.config.sections.inference import _INFERENCE_KEYS, _TRAINABLE_KEYS, MODES
from rheplicant.config.sections.ingest import _FROM_FILE_KEYS
from rheplicant.config.sections.model import operator_table
from rheplicant.config.sections.noise import _KIND_KEYS as NOISE_KIND_KEYS
from rheplicant.config.sections.npe import (
    _BANK_KEYS,
    _CREATE_DEFAULTS,
    _CREATE_KEYS,
    _NPE_KEYS,
    _TRAIN_DEFAULTS,
    _TRAIN_KEYS,
)
from rheplicant.config.sections.npe import (
    _SAMPLE_KEYS as NPE_SAMPLE_KEYS,
)
from rheplicant.config.sections.nuts import _INITS, _NUTS_DEFAULTS, _NUTS_KEYS
from rheplicant.config.sections.observation import (
    _AUX_KEYS,
    _ENVIRONMENT_KEYS,
    _FREQ_KEYS,
    _OBSERVATION_KEYS,
    _SITE_KEYS,
    _TIME_KEYS,
)
from rheplicant.config.sections.observed import _FORM_KEYS, _REALISE_KINDS, _SIM_KEYS
from rheplicant.config.sections.parameters import _FAN_MODES, _LATENT_KEYS
from rheplicant.config.sections.pointing import _KEYS as POINTING_KEYS
from rheplicant.config.sections.pointing import _LST_KEYS
from rheplicant.config.sections.pointing import _MODES as POINTING_MODES
from rheplicant.config.sections.runs import _KINDS as RUN_KINDS
from rheplicant.config.sections.runs import _RUN_KEYS
from rheplicant.config.sections.switching import _KEYS as SWITCHING_KEYS
from rheplicant.config.sections.transforms import _BINDING_KEYS
from rheplicant.config.sections.twin import _TWIN_KEYS
from rheplicant.config.units import UNIT_SPELLINGS
from rheplicant.config.values import _SHORTHAND as SHORTHAND
from rheplicant.config.values import VALUE_FORMS, VALUE_MODIFIERS
from rheplicant.gui.forms import (
    FormCatalog,
    FormRule,
    SourceRef,
    WidgetMetadata,
)
from rheplicant.radio.filters.base import _MODES as FILTER_MODES
from rheplicant.radio.graph import RADIO_GRAPH
from rheplicant.radio.instrument.calibration import LINESHAPES

_NO_DEFAULT = object()
_RUN_HANDLER_REGISTRIES = (PARSERS, PRE_EXECUTORS, EXECUTORS, DEFERRED_CHECKS)

#: Class-path selector -> the members a ``static_str`` field accepts.
#:
#: Keyed by the same selector the dimension registry uses, so the two are
#: comparable by construction: ``tests/gui/test_forms.py`` asserts this table
#: is EXACTLY the live ``model_field`` rows whose disposition is structural
#: and whose reason is ``"enum"``, which makes a new enum field impossible to
#: ship with an empty select.
#:
#: The members are the operators' own constants rather than copies. Both are
#: what the operator validates against -- ``filters/base.py`` and
#: ``instrument/calibration.py`` raise on anything else -- so a select built
#: from them cannot offer a member the constructor would refuse.
_ENUMS: dict[str, tuple[str, ...]] = {
    "rheplicant.radio.filters.fourier.FourierBandFilter.mode": FILTER_MODES,
    "rheplicant.radio.filters.sidereal.SiderealFilter.mode": FILTER_MODES,
    "rheplicant.radio.filters.skyspace.SkySpaceFilter.mode": FILTER_MODES,
    "rheplicant.radio.instrument.calibration.CWCalibrationOperator.lineshape": LINESHAPES,
}


def _rule(path: str, operator: str, expected: object = None) -> FormRule:
    return FormRule(path=path, operator=operator, expected=expected)  # type: ignore[arg-type]


def _all(*rules: FormRule) -> FormRule:
    return FormRule(path=None, operator="all", rules=tuple(rules))


def _any(*rules: FormRule) -> FormRule:
    return FormRule(path=None, operator="any", rules=tuple(rules))


def _label(path: str) -> str:
    tail = path.rsplit(".", 1)[-1].replace("[]", "").replace("*", "entry")
    return tail.replace("_", " ")


def _section_for(path: str) -> str:
    if path.startswith("resources.sky_models") or path.startswith("resources.projectors"):
        return "sky"
    if path.startswith("resources.beams"):
        return "beam"
    if path.startswith("resources"):
        return "resources"
    if path.startswith("model"):
        # A list node's instance prefix spells the node ``filters[]``, which is
        # not a node id: without the trim every field of every list node falls
        # through to the Instrument default, and ``filters`` is `processing`.
        node = path.split(".")[1].removesuffix("[]") if "." in path else ""
        spec = RADIO_GRAPH.nodes.get(node)
        return "backend" if spec is not None and spec.segment == "processing" else "instrument"
    if (
        path.startswith("schema_version")
        or path.startswith("defaults")
        or path.startswith("plugins")
    ):
        return "runtime"
    root = path.split(".", 1)[0].removesuffix("[]")
    # The catch-all hatch-argument destination is a value-form concern rather
    # than a thirteenth view; Resources owns the generic value-node palette.
    return "resources" if root == "python" else root


#: A non-empty mapping, so :func:`_is_fan` keeps answering correctly if the
#: FAN branch ever grows the non-emptiness rule the list branch already has.
_FAN_PROBE = {"probe": {}}


def _is_fan(node_id: str) -> bool:
    """Does this ``many`` node hold a label-keyed mapping rather than a list?

    Asked of :func:`many_shape_problem`, which is where the shape of a ``many``
    node is decided, rather than answered here: a second spelling of the FAN
    node would be a second thing to keep in step, and nothing would go red
    when the two disagreed.
    """
    return many_shape_problem(node_id, _FAN_PROBE, many=True) is None


def _instance_prefix(node_id: str) -> str:
    """Where ONE instance of a node's settings lives in a document.

    A single-slot node has its fields at ``model.<node>.<field>``. A ``many``
    node has one set per list entry (``model.<node>[].<field>``) or one per
    FAN label (``model.<node>.*.<field>``). Spelling a ``many`` node flat
    reads a list or a label mapping with ``Mapping.get`` on the field name,
    which never finds anything, so the field is permanently absent and --
    being required -- permanently *must decide*.
    """
    if not RADIO_GRAPH.nodes[node_id].many:
        return f"model.{node_id}"
    return f"model.{node_id}.*" if _is_fan(node_id) else f"model.{node_id}[]"


def _default_value(value: object) -> object:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _default_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(_default_value(item) for item in value)
    return repr(value)


def _units(dimension: str | None, unit_policy: str | None) -> tuple[str, ...]:
    """The spellings a unit select may offer for one widget.

    Empty for a compound like ``adc_count/K``: a quotient is not an atom, so
    the alphabet has no second spelling for it, and a one-element tuple would
    read as a choice where there is none. Empty too when the policy is
    ``forbidden`` -- there a unit is a refusal, so offering one would be
    inviting an error.
    """
    if unit_policy == "forbidden":
        return ()
    return UNIT_SPELLINGS.get(dimension or "", ())


def _dimension(spec: DimensionSpec) -> str:
    if spec.signature is not None:
        return signature_token(spec.signature)
    if spec.disposition == "contextual":
        return f"contextual:{spec.resolver}"
    return spec.disposition


class _Builder:
    def __init__(self) -> None:
        self.widgets: dict[str, WidgetMetadata] = {}

    def add(
        self,
        path: str,
        *,
        section: str | None = None,
        widget: str = "value",
        choices: tuple[str, ...] = (),
        required: bool = False,
        default: object = _NO_DEFAULT,
        required_when: FormRule | None = None,
        visible_when: FormRule | None = None,
        dimension: str | None = None,
        unit_policy: str | None = None,
        delivery: str | None = None,
        disabled: bool = False,
        reason: str | None = None,
        source: SourceRef | None = None,
    ) -> None:
        found = WidgetMetadata(
            path=path,
            section=_section_for(path) if section is None else section,
            label=_label(path),
            widget=widget,
            choices=choices,
            required=required,
            has_default=default is not _NO_DEFAULT,
            default=None if default is _NO_DEFAULT else _default_value(default),
            required_when=required_when,
            visible_when=visible_when,
            dimension=dimension,
            unit_policy=unit_policy,
            units=_units(dimension, unit_policy),
            delivery=delivery,
            disabled=disabled,
            reason=reason,
            sources=() if source is None else (source,),
        )
        previous = self.widgets.get(path)
        if previous is None:
            self.widgets[path] = found
            return
        sources = tuple(dict.fromkeys((*previous.sources, *found.sources)))
        # Explicit declarations replace generic dimension-derived presentation,
        # while every source reference remains attached to the one visible path.
        preferred = previous if previous.widget != "value" or found.widget == "value" else found
        self.widgets[path] = dataclasses.replace(preferred, sources=sources)

    def source(self, path: str, domain: str, selector: str, spec: DimensionSpec) -> None:
        source = SourceRef(domain, selector)
        if path in self.widgets:
            previous = self.widgets[path]
            dimension = previous.dimension or _dimension(spec)
            unit_policy = previous.unit_policy or spec.unit_policy
            self.widgets[path] = dataclasses.replace(
                previous,
                dimension=dimension,
                unit_policy=unit_policy,
                units=_units(dimension, unit_policy),
                sources=tuple(dict.fromkeys((*previous.sources, source))),
            )
        else:
            self.add(
                path,
                dimension=_dimension(spec),
                unit_policy=spec.unit_policy,
                source=source,
            )


def _basic_widgets(builder: _Builder) -> None:
    builder.add("schema_version", widget="integer", required=True)
    builder.add("defaults", widget="list", default=())
    builder.add("plugins", widget="list", default=())
    runtime_defaults = {"jax_enable_x64": False, "platform": "auto", "seed": None, "seeds": {}}
    for key in sorted(_RUNTIME_KEYS):
        choices = ("auto", "cpu", "gpu", "tpu") if key == "platform" else ()
        widget = (
            "toggle"
            if key == "jax_enable_x64"
            else "mapping"
            if key == "seeds"
            else "select"
            if choices
            else "integer"
        )
        builder.add(f"runtime.{key}", widget=widget, choices=choices, default=runtime_defaults[key])

    synthetic = _rule("observation.from_file", "absent")
    ingested = _rule("observation.from_file", "present")
    for key in sorted(_OBSERVATION_KEYS):
        builder.add(f"observation.{key}", widget="group")
    for key in sorted(_FREQ_KEYS):
        builder.add(f"observation.freq.{key}", required=True, visible_when=synthetic)
    for key in sorted(_TIME_KEYS):
        builder.add(f"observation.time.{key}", required=key == "grid", visible_when=synthetic)
    for key in sorted(_SITE_KEYS):
        builder.add(f"observation.site.{key}")
    for key in sorted(_ENVIRONMENT_KEYS):
        builder.add(
            f"observation.environment.{key}", widget="mapping" if key == "extra" else "value"
        )
    for key in sorted(_AUX_KEYS):
        builder.add(f"observation.aux.{key}")
    for key in sorted(_FROM_FILE_KEYS):
        required = key in {"format", "path", "freq_unit"}
        choices = (
            ("rhino_hdf5",) if key == "format" else ("Hz", "MHz") if key == "freq_unit" else ()
        )
        widget = (
            "file"
            if key == "path"
            else "select"
            if choices
            else "mapping"
            if key == "thermistor_columns"
            else "value"
        )
        builder.add(
            f"observation.from_file.{key}",
            widget=widget,
            choices=choices,
            required=required,
            visible_when=ingested,
        )
    builder.add(
        "observation.pointing.mode", widget="select", choices=POINTING_MODES, default="none"
    )
    for mode, keys in POINTING_KEYS.items():
        for key in sorted(keys - {"mode"}):
            builder.add(
                f"observation.pointing.{key}",
                visible_when=_rule("observation.pointing.mode", "equals", mode),
            )
    for mode, keys in _LST_KEYS.items():
        for key in sorted(keys - {"mode"}):
            builder.add(
                f"observation.pointing.lst.{key}",
                visible_when=_rule("observation.pointing.lst.mode", "equals", mode),
            )
    builder.add(
        "observation.pointing.lst.mode",
        widget="select",
        choices=tuple(_LST_KEYS),
        visible_when=_rule("observation.pointing.mode", "in", ("drift", "tracked")),
    )
    builder.add(
        "observation.switching.mode", widget="select", choices=tuple(SWITCHING_KEYS), default="none"
    )
    for mode, keys in SWITCHING_KEYS.items():
        for key in sorted(keys - {"mode"}):
            builder.add(
                f"observation.switching.{key}",
                visible_when=_rule("observation.switching.mode", "equals", mode),
            )


def _resource_widgets(builder: _Builder) -> None:
    for kind in RESOURCE_KINDS:
        builder.add(f"resources.{kind}.*.extends", widget="resource")
    builder.add("resources.arrays.*", section="resources", widget="value")
    for axis in ("time", "freq"):
        builder.add(
            f"resources.bases.*.{axis}.kind",
            widget="select",
            choices=tuple(BASIS_KINDS),
            required=True,
        )
        builder.add(f"resources.bases.*.{axis}.n_basis", widget="integer", required=True)

    builder.add("resources.beams.*.format", widget="select", choices=BEAM_FORMATS, required=True)
    beam_keys = set().union(*_FORMAT_KEYS.values()) - {"format"}
    beam_required = {
        "nside": tuple(BEAM_FORMATS),
        "normalize": tuple(BEAM_FORMATS),
        "directory": ("cst",),
        "phi0_deg": ("cst",),
        "phi_sense": ("cst",),
        "path": ("uvbeam", "healpix", "npy", "npz"),
        "order": ("healpix",),
        "freq": ("healpix",),
        "maps": ("inline",),
        "python": ("python",),
    }
    for key in sorted(beam_keys - {"horizon"}):
        formats = tuple(fmt for fmt, keys in _FORMAT_KEYS.items() if key in keys)
        choices = (
            BEAM_NORMALIZATIONS
            if key == "normalize"
            else ("ccw", "cw")
            if key == "phi_sense"
            else ("ring", "nested")
            if key == "order"
            else ()
        )
        required_when = None
        if key == "fwhm_deg":
            required_when = _rule("resources.beams.*.sigma_deg", "absent")
        builder.add(
            f"resources.beams.*.{key}",
            widget="select" if choices else "file" if key in {"path", "directory"} else "value",
            choices=tuple(choices),
            required=key in beam_required,
            required_when=required_when,
            visible_when=_rule("resources.beams.*.format", "in", formats),
        )
    builder.add(
        "resources.beams.*.horizon.mode",
        widget="select",
        choices=("none", "truncate_map", "projector_mask"),
        default="none",
    )
    for key, default in (("el_deg", 90.0), ("apod_deg", 0.0)):
        builder.add(
            f"resources.beams.*.horizon.{key}",
            default=default,
            visible_when=_rule("resources.beams.*.horizon.mode", "equals", "truncate_map"),
        )

    builder.add("resources.projectors.*.engine", widget="select", choices=ENGINES, required=True)
    projector_keys = set().union(*_ENGINE_KEYS.values()) - {"engine"}
    projector_required = {
        "matrix": ("matrix",),
        "provenance": ("matrix",),
        "lmax": ("driftscan", "general_pointing"),
        "lat_deg": ("driftscan", "general_pointing"),
        "az_deg": ("driftscan",),
        "el_deg": ("driftscan",),
        "normalize_beam": ("driftscan", "general_pointing"),
        "nside": ("general_pointing",),
        "beam": ("driftscan",),
    }
    projector_defaults = {
        "acknowledge_float32_sky": False,
        "optimizations": (),
        "beam_iterations": 3,
        "horizon_mask": False,
        "apod_deg": 0.0,
        "mask_iterations": 3,
        "uniform_sampling": False,
        "freq_chunk": None,
    }
    for key in sorted(projector_keys):
        engines = tuple(engine for engine, keys in _ENGINE_KEYS.items() if key in keys)
        required_when = None
        if key == "beam" and "general_pointing" in engines:
            required_when = _rule("resources.projectors.*.beam_alms", "absent")
        builder.add(
            f"resources.projectors.*.{key}",
            widget="list"
            if key == "optimizations"
            else "toggle"
            if key
            in {"normalize_beam", "acknowledge_float32_sky", "horizon_mask", "uniform_sampling"}
            else "value",
            choices=OPTIMIZATIONS if key == "optimizations" else (),
            required=key in projector_required,
            required_when=required_when,
            default=projector_defaults.get(key, _NO_DEFAULT),
            visible_when=_rule("resources.projectors.*.engine", "in", engines),
        )

    builder.add("resources.s_params.*.kind", widget="select", choices=S_PARAM_KINDS, required=True)
    s_param_tables = {
        "touchstone": _TOUCHSTONE_KEYS,
        "termination": _TERMINATION_KEYS,
        "cable": _CABLE_KEYS,
    }
    s_required = {"file", "termination", "behind", "length"}
    for key in sorted(set().union(*s_param_tables.values()) - {"kind"}):
        kinds = tuple(kind for kind, keys in s_param_tables.items() if key in keys)
        choices = COMPONENTS if key == "component" else TERMINATIONS if key == "termination" else ()
        builder.add(
            f"resources.s_params.*.{key}",
            widget="select" if choices else "value",
            choices=tuple(choices),
            required=key in s_required,
            required_when=_rule("resources.s_params.*.termination", "equals", "resistive")
            if key == "impedance"
            else None,
            visible_when=_rule("resources.s_params.*.kind", "in", kinds),
        )

    builder.add("resources.sky_models.*.kind", widget="select", choices=SKY_KINDS, required=True)
    sky_required = {
        "amplitude",
        "n_pix",
        "spectral_index",
        "ref_freq",
        "maps",
        "freq",
        "nside",
        "python",
    }
    for key in sorted(set().union(*_ALLOWED_KEYS.values()) - {"kind"}):
        kinds = tuple(kind for kind, keys in _ALLOWED_KEYS.items() if key in keys)
        builder.add(
            f"resources.sky_models.*.{key}",
            widget="select" if key == "order" else "value",
            choices=("ring",) if key == "order" else (),
            required=key in sky_required,
            visible_when=_rule("resources.sky_models.*.kind", "in", kinds),
        )


def _model_widgets(builder: _Builder) -> None:
    table = operator_table()
    fields: dict[str, tuple[str, list[tuple[type, Any, str]]]] = {}
    for node_id in RADIO_GRAPH._topo:
        node = RADIO_GRAPH.nodes[node_id]
        classes = table.get(node_id, ())
        # The node widget names the whole slot -- the list, or the label
        # mapping -- so it stays flat while its settings move per instance.
        builder.add(
            f"model.{node_id}",
            widget="node",
            disabled=node.reserved and not classes,
            reason=node.doc,
        )
        instance = _instance_prefix(node_id)
        if classes:
            builder.add(
                f"{instance}.type",
                widget="select",
                choices=tuple(cls.__name__ for cls in classes),
                required=len(classes) > 1,
                default=classes[0].__name__ if len(classes) == 1 else _NO_DEFAULT,
                visible_when=_rule(instance, "present"),
            )
        for cls in classes:
            prefix = f"{cls.__module__}.{cls.__qualname__}"
            for name, spec in field_specs(cls).items():
                fields.setdefault(f"{instance}.{name}", (instance, []))[1].append(
                    (cls, spec, f"{prefix}.{name}")
                )
        builder.add(
            f"{instance}.eqx_leaves",
            widget="file",
            visible_when=_rule(instance, "present"),
        )
    dimension_rows = {
        selector.selector: spec
        for selector, spec in registered_dimension_rows()
        if selector.domain == "model_field"
    }
    # The instance prefix is carried through rather than re-read off the path:
    # ``path.split(".")[1]`` would give ``"filters[]"``, which is not a node.
    for path, (instance, rows) in fields.items():
        classes = tuple(row[0].__name__ for row in rows)
        specs = [row[1] for row in rows]
        first = specs[0]
        field = next(
            item for item in dataclasses.fields(rows[0][0]) if item.name == path.rsplit(".", 1)[-1]
        )
        default = _NO_DEFAULT
        if field.default is not dataclasses.MISSING:
            default = field.default
        elif field.default_factory is not dataclasses.MISSING:
            default = field.default_factory()
        dimension_spec = dimension_rows[rows[0][2]]
        # One widget can be shared by several classes, so the members are the
        # union in declaration order; the suite pins that today's three
        # filters agree rather than leaving a disagreement to be discovered.
        choices = tuple(
            dict.fromkeys(
                member for _cls, _spec, selector in rows for member in _ENUMS.get(selector, ())
            )
        )
        builder.add(
            path,
            choices=choices,
            widget="select" if choices else "value",
            required=all(spec.required for spec in specs),
            default=default,
            visible_when=_all(
                _rule(instance, "present"),
                _rule(f"{instance}.type", "absent_or_in", classes),
            ),
            dimension=_dimension(dimension_spec),
            unit_policy=dimension_spec.unit_policy,
            delivery=mode_of(first),
        )
        for _cls, _spec, selector in rows:
            builder.source(path, "model_field", selector, dimension_rows[selector])


def _inference_widgets(builder: _Builder) -> None:
    for key in sorted(_INFERENCE_KEYS):
        builder.add(f"inference.{key}", widget="group")
    for key in sorted(_TWIN_KEYS):
        builder.add(f"inference.twin.{key}", widget="list" if key == "without" else "mapping")
    observed_keys = set(_SIM_KEYS) | set(_FORM_KEYS)
    for prefix in ("inference.observed", "inference.observed.*"):
        builder.add(
            f"{prefix}.from", widget="select", choices=("simulation", "file"), default="simulation"
        )
        for key in sorted(observed_keys - {"from"}):
            visibility = _rule(
                f"{prefix}.from", "equals", "file" if key == "file" else "simulation"
            )
            builder.add(f"{prefix}.{key}", visible_when=visibility)
        builder.add(
            f"{prefix}.realise.kind", widget="select", choices=_REALISE_KINDS, default="none"
        )
        builder.add(
            f"{prefix}.realise.sigma",
            required=True,
            visible_when=_rule(f"{prefix}.realise.kind", "equals", "homoscedastic"),
        )
        builder.add(
            f"{prefix}.realise.seed",
            required=True,
            visible_when=_rule(f"{prefix}.realise.kind", "in", ("homoscedastic", "radiometer")),
        )
    latent_defaults = {"prior": None, "linear": False, "scope": "global", "transform": "identity"}
    for key in sorted(_LATENT_KEYS):
        choices = (
            _FAN_MODES
            if key == "fan"
            else ("global", "per_epoch", "linked")
            if key == "scope"
            else ()
        )
        builder.add(
            f"inference.parameters.*.{key}",
            widget="select" if choices else "value",
            choices=tuple(choices),
            required=key == "init",
            required_when=_rule("inference.parameters.*.into", "length_gt", 1)
            if key == "fan"
            else None,
            default=latent_defaults.get(key, _NO_DEFAULT),
        )
    for key in sorted(_BINDING_KEYS):
        builder.add(
            f"inference.bindings[].{key}",
            required=key in {"latents", "into"},
            required_when=_rule("inference.bindings[].into", "length_gt", 1)
            if key == "fan"
            else None,
        )
    builder.add(
        "inference.noise.kind", widget="select", choices=tuple(NOISE_KIND_KEYS), default="none"
    )
    noise_defaults = {"source": "prediction_at_init"}
    for key in sorted(set().union(*NOISE_KIND_KEYS.values()) - {"kind"}):
        kinds = tuple(kind for kind, keys in NOISE_KIND_KEYS.items() if key in keys)
        builder.add(
            f"inference.noise.{key}",
            required=key == "sigma",
            required_when=_rule("inference.noise.kind", "equals", "radiometer")
            if key == "include_logdet"
            else None,
            default=noise_defaults.get(key, _NO_DEFAULT),
            visible_when=_rule("inference.noise.kind", "in", kinds),
        )
    for key in ("mode", "report", "reason"):
        builder.add(
            f"inference.checks.*.{key}",
            widget="select" if key == "mode" else "toggle" if key == "report" else "text",
            choices=MODES if key == "mode" else (),
            required=key == "reason",
            visible_when=_rule("inference.checks.*.mode", "equals", "skip")
            if key == "reason"
            else None,
        )
    for key in sorted(_TRAINABLE_KEYS):
        builder.add(f"inference.trainable.{key}")
    npe_tables = {
        "bank": _BANK_KEYS,
        "create": _CREATE_KEYS,
        "train": _TRAIN_KEYS,
        "sample": NPE_SAMPLE_KEYS,
    }
    npe_defaults = {**_CREATE_DEFAULTS, **_TRAIN_DEFAULTS}
    npe_visible = _rule("inference.npe", "present")
    for subsection, keys in npe_tables.items():
        for key in sorted(keys):
            builder.add(
                f"inference.npe.{subsection}.{key}",
                required=key in {"n_simulations", "seed", "n_draws"},
                default=npe_defaults.get(key, _NO_DEFAULT),
                visible_when=npe_visible,
            )
    for key in sorted(_NPE_KEYS - set(npe_tables)):
        builder.add(f"inference.npe.{key}", visible_when=npe_visible)


def _run_widgets(builder: _Builder) -> None:
    for key in sorted(_RUN_KEYS):
        choices = tuple(RUN_KINDS) if key == "kind" else ("ok", "refuse") if key == "expect" else ()
        builder.add(
            f"runs[].{key}",
            widget="select" if choices else "text",
            choices=choices,
            required=key == "kind",
            required_when=_rule("runs", "length_gt", 1) if key == "name" else None,
            default="primary"
            if key == "on"
            else "ok"
            if key == "expect"
            else None
            if key in {"variant", "reuse"}
            else _NO_DEFAULT,
        )
    tables = {
        "forward": frozenset(),
        "fisher": frozenset({"space", "jitter"}),
        "optimize": _OPTIMIZE_KEYS,
        "plan.estimate": _ESTIMATE_KEYS,
        "plan.sample": _SAMPLE_KEYS,
        "conjugate.wiener": _WIENER_KEYS,
        "conjugate.gcr": _GCR_KEYS,
        "conjugate.gls": _GLS_KEYS,
        "condition": _CONDITION_KEYS,
        "identifiability": _IDENTIFIABILITY_KEYS,
        "score_directions": _SCORE_KEYS,
        "gradient": _GRADIENT_KEYS,
        "mmodes": _MMODES_KEYS,
        "predict": _PREDICT_KEYS,
        "nuts": _NUTS_KEYS,
        "npe": frozenset(),
        "compare": _COMPARE_KEYS,
        "benchmark": _BENCHMARK_KEYS,
    }
    defaults = {**_ADAM_DEFAULTS, **_ESTIMATE_DEFAULTS, **_SAMPLE_DEFAULTS, **_NUTS_DEFAULTS}
    required = {
        "optimizer",
        "learning_rate",
        "n_steps",
        "seed",
        "num_warmup",
        "num_samples",
        "of",
        "metric",
        "tolerance",
        "variants",
        "names",
        "iterations",
        "projector",
        "sky",
    }
    all_keys = set().union(*tables.values())
    for key in sorted(all_keys):
        kinds = tuple(kind for kind, keys in tables.items() if key in keys)
        choices: tuple[str, ...] = ()
        if key == "optimizer":
            choices = ("gradient", "adam")
        elif key == "chain_method":
            choices = ("parallel", "sequential", "vectorized")
        elif key == "init":
            choices = _INITS
        elif key == "metric":
            choices = COMPARE_METRICS
        elif key == "metrics":
            choices = BENCHMARK_METRICS
        visibility: FormRule = _rule("runs[].kind", "in", kinds)
        if key in _ADAM_DEFAULTS:
            visibility = _all(visibility, _rule("runs[].optimizer", "equals", "adam"))
        builder.add(
            f"runs[].{key}",
            widget="select" if choices else "value",
            choices=tuple(choices),
            required=key in required,
            default=defaults.get(key, _NO_DEFAULT),
            visible_when=visibility,
        )


def build_catalog() -> FormCatalog:
    """Assemble a fresh catalog without making module import stateful."""
    from rheplicant.gui.form_catalog_finalize import build_catalog as assemble

    return assemble()


def catalog_drift(catalog: FormCatalog) -> tuple[str, ...]:
    """Compare one assembled catalog with every guarded live registry."""
    from rheplicant.gui.form_catalog_finalize import catalog_drift as inspect_drift

    return inspect_drift(catalog)


#: ``operator_table``, the value vocabulary and the shorthand pattern are
#: re-exported rather than merely imported. ``gui/node_forms.py`` needs all
#: four and may not reach the config layer itself -- the boundary test at
#: ``tests/config/test_config_surface.py`` keeps that list at five files, and
#: this module is one of them, so gatewaying is the role it already has.
#: ``form_catalog_finalize`` takes the same route for the same reason.
__all__ = [
    "SHORTHAND",
    "_instance_prefix",
    "_is_fan",
    "VALUE_FORMS",
    "VALUE_MODIFIERS",
    "build_catalog",
    "catalog_drift",
    "operator_table",
]
