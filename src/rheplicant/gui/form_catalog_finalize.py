"""Finalize the form catalog and guard its live-registry closure."""

from __future__ import annotations

import re
from collections import Counter

from _rheplicant_bootstrap.output.manager import (
    _OUTPUT_KEYS,
    _PLAN4B_TOP,
    _PLAN4B_WRITE,
    _PRODUCT_FORMATS,
    _REPORT_COLUMNS,
    _REPORT_FORMATS,
    _REPORT_RELATIVE,
    _STDOUT,
    _WRITE_KEYS,
)
from rheplicant.gui.form_catalog import (
    _NO_DEFAULT,
    _RUN_HANDLER_REGISTRIES,
    RESOURCE_KINDS,
    RUN_KINDS,
    _basic_widgets,
    _Builder,
    _inference_widgets,
    _model_widgets,
    _resource_widgets,
    _rule,
    _run_widgets,
    registered_dimension_rows,
)
from rheplicant.gui.forms import FormCatalog, SectionMetadata, SourceRef
from rheplicant.radio.graph import RADIO_GRAPH

_EXPECTED_DIMENSIONS = {"model_field": 67, "resource_field": 49, "config_path": 56}
_EXPECTED_RESOURCE_KINDS = (
    "arrays",
    "bases",
    "beams",
    "projectors",
    "s_params",
    "sky_models",
)
_EXPECTED_GRAPH_NODES = (
    "global_signal",
    "foregrounds",
    "point_sources",
    "uniform_sky",
    "astro_sum",
    "ionosphere",
    "atmosphere_field",
    "ground_field",
    "rfi_field",
    "field_sum",
    "beam",
    "observed_astro_sky",
    "ground_pickup",
    "t_sys_extra",
    "atmosphere",
    "astro_ant_sum",
    "beam_spill",
    "t_ant_sum",
    "antenna_loss",
    "cal_loads",
    "receiver_input",
    "noise_wave",
    "cw_tone",
    "bandpass",
    "gain",
    "noise",
    "emi",
    "adc",
    "snapshot",
    "flagging",
    "averaging",
    "apply_cal",
    "filters",
)
_EXPECTED_OUTPUT_PRODUCTS = (
    "arrays",
    "aux",
    "taps",
    "assembly",
    "estimates",
    "parameters",
    "draws",
    "losses",
    "gradients",
    "covariance",
    "prediction_bands",
    "posterior_predictives",
    "identifiability",
    "scores",
    "recovery",
    "training_history",
    "timings",
    "refusals",
    "signal_paths",
    "compare",
    "benchmark",
    "chains",
    "run_diagnostics",
)


def _output_widgets(builder: _Builder) -> None:
    output_defaults = {"clobber": False, "stdout": "summary"}
    for key in _OUTPUT_KEYS:
        builder.add(
            f"outputs.{key}",
            widget=("select" if key == "stdout" else "toggle" if key == "clobber" else "group"),
            choices=_STDOUT if key == "stdout" else (),
            default=output_defaults.get(key, _NO_DEFAULT),
        )
    for key in _WRITE_KEYS:
        builder.add(f"outputs.write.{key}", widget="toggle", default=True)
    for name in _PLAN4B_WRITE:
        builder.add(
            f"outputs.write.{name}",
            widget="product",
            choices=tuple(_PRODUCT_FORMATS[name]),
            default=False,
        )
    report_visible = _rule("outputs.report", "present")
    for key in ("rows", "columns", "reference", "relative", "format"):
        choices = (
            _REPORT_COLUMNS
            if key == "columns"
            else _REPORT_RELATIVE
            if key == "relative"
            else _REPORT_FORMATS
            if key == "format"
            else ()
        )
        default = (
            _REPORT_COLUMNS if key == "columns" else "text" if key == "format" else _NO_DEFAULT
        )
        builder.add(
            f"outputs.report.{key}",
            widget="select" if choices else "list",
            choices=tuple(choices),
            required=key == "rows",
            default=default,
            required_when=_rule("outputs.report.relative", "present")
            if key == "reference"
            else None,
            visible_when=report_visible,
        )
    for key in _PLAN4B_TOP:
        builder.add(f"outputs.{key}", widget="group")


def _campaign_widgets(builder: _Builder) -> None:
    reason = "Reserved for capability 4 (streaming evidence)."
    for path in (
        "campaign.epoch_id",
        "campaign.inputs",
        "campaign.represents",
        "campaign.archive",
        "campaign.floors",
        "campaign.compress.method",
        "campaign.compress.n_basis",
        "campaign.compress.bank",
        "campaign.compress.select",
        "campaign.compress.seed_scores",
        "campaign.compress.nuisances",
    ):
        builder.add(path, widget="text", disabled=True, reason=reason)


_RESOURCE_SELECTOR = re.compile(
    r"rheplicant\.config\.kinds\.(?P<kind>[a-z_]+)\.build_[a-z_]+\.(?P<tail>.+)\Z"
)


def _resource_path(selector: str) -> str:
    match = _RESOURCE_SELECTOR.fullmatch(selector)
    if match is None:
        raise AssertionError(f"unmapped resource dimension selector {selector!r}")
    kind = match.group("kind")
    tail = match.group("tail").split(".")
    if kind == "arrays" and tail == ["value"]:
        return "resources.arrays.*"
    if kind not in {"arrays", "bases"}:
        tail = tail[1:]
    return ".".join(("resources", kind, "*", *tail))


def _attach_dimensions(builder: _Builder) -> None:
    for selector, spec in registered_dimension_rows():
        if selector.domain == "model_field":
            continue
        path = (
            selector.selector
            if selector.domain == "config_path"
            else _resource_path(selector.selector)
        )
        builder.source(path, selector.domain, selector.selector, spec)


def build_catalog() -> FormCatalog:
    builder = _Builder()
    _basic_widgets(builder)
    _resource_widgets(builder)
    _model_widgets(builder)
    builder.add("variants.*", widget="mapping")
    _inference_widgets(builder)
    _run_widgets(builder)
    _output_widgets(builder)
    _campaign_widgets(builder)
    _attach_dimensions(builder)
    sections = (
        SectionMetadata("runtime", "Runtime", "runtime"),
        SectionMetadata("observation", "Observation", "observation"),
        SectionMetadata("resources", "Resources", "resources"),
        SectionMetadata("sky", "Sky", "resources"),
        SectionMetadata("beam", "Beam", "resources"),
        SectionMetadata("instrument", "Instrument", "model"),
        SectionMetadata("backend", "Backend", "model"),
        SectionMetadata("variants", "Variants", "variants"),
        SectionMetadata("inference", "Inference", "inference"),
        SectionMetadata("runs", "Runs", "runs"),
        SectionMetadata("outputs", "Outputs", "outputs"),
        SectionMetadata(
            "campaign",
            "Campaign",
            "campaign",
            disabled=True,
            reason="Reserved for capability 4 (streaming evidence).",
        ),
    )
    return FormCatalog(
        sections=sections,
        widgets=tuple(builder.widgets.values()),
        resource_kinds=tuple(RESOURCE_KINDS),
        run_kinds=tuple(RUN_KINDS),
        graph_nodes=RADIO_GRAPH._topo,
    )


def catalog_drift(catalog: FormCatalog) -> tuple[str, ...]:
    problems: list[str] = []
    counts = Counter(selector.domain for selector, _spec in registered_dimension_rows())
    if dict(counts) != _EXPECTED_DIMENSIONS:
        problems.append(f"dimension rows are {dict(counts)}, expected {_EXPECTED_DIMENSIONS}")
    if catalog.resource_kinds != _EXPECTED_RESOURCE_KINDS:
        problems.append(
            f"resource kinds are {catalog.resource_kinds}, expected {_EXPECTED_RESOURCE_KINDS}"
        )
    registries = tuple(set(registry) for registry in _RUN_HANDLER_REGISTRIES)
    if any(registry != set(RUN_KINDS) for registry in registries):
        problems.append("run handler registries do not equal the 18 reviewed run kinds")
    if catalog.graph_nodes != _EXPECTED_GRAPH_NODES:
        problems.append("graph nodes differ from the reviewed 33-node form contract")
    if tuple(_PLAN4B_WRITE) != _EXPECTED_OUTPUT_PRODUCTS:
        problems.append("output product selectors differ from the reviewed 23-product contract")
    live_sources = {
        SourceRef(selector.domain, selector.selector)
        for selector, _spec in registered_dimension_rows()
    }
    catalog_sources = {source for widget in catalog.widgets for source in widget.sources}
    missing = sorted(live_sources - catalog_sources, key=lambda row: (row.domain, row.selector))
    if missing:
        problems.append(f"dimension destinations missing from widgets: {missing}")
    paths = [widget.path for widget in catalog.widgets]
    if len(paths) != len(set(paths)):
        problems.append("widget paths are not unique")
    return tuple(problems)


__all__ = ["build_catalog", "catalog_drift"]
