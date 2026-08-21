from __future__ import annotations

import dataclasses

import pytest
import yaml

from rheplicant.config.dimensions import registered_dimension_rows
from rheplicant.config.resources import RESOURCE_KINDS
from rheplicant.config.sections import exits as _exits  # noqa: F401
from rheplicant.config.sections.exit_support import (
    DEFERRED_CHECKS,
    EXECUTORS,
    PARSERS,
    PRE_EXECUTORS,
)
from rheplicant.gui.forms import (
    CatalogDrift,
    WidgetMetadata,
    assert_catalog_closed,
    project_forms,
    widget_catalog,
)
from rheplicant.radio.graph import RADIO_GRAPH


def _widget(path: str) -> WidgetMetadata:
    return next(widget for widget in widget_catalog().widgets if widget.path == path)


def _projected(yaml_text: str, path: str):
    projected = project_forms(yaml.safe_load(yaml_text))
    return next(
        widget
        for section in projected.sections
        for widget in section.widgets
        if widget.path == path
    )


def test_npe_required_fields_appear_only_when_npe_is_present():
    absent = project_forms({"schema_version": 1})
    assert not any(path.startswith("inference.npe.") for path in absent.missing_required)

    present = project_forms({"schema_version": 1, "inference": {"npe": {"bank": {}}}})
    assert "inference.npe.bank.n_simulations" in present.missing_required


def test_catalog_is_frozen_closed_and_covers_every_planned_view():
    found = widget_catalog()
    assert dataclasses.is_dataclass(found)
    with pytest.raises(dataclasses.FrozenInstanceError):
        found.widgets = ()  # type: ignore[misc]
    assert tuple(section.section_id for section in found.sections) == (
        "runtime",
        "observation",
        "resources",
        "sky",
        "beam",
        "instrument",
        "backend",
        "variants",
        "inference",
        "runs",
        "outputs",
        "campaign",
    )
    assert len({widget.path for widget in found.widgets}) == len(found.widgets)
    assert next(section for section in found.sections if section.section_id == "campaign").disabled


def test_catalog_is_projected_from_every_live_registry_and_dimension_row():
    found = widget_catalog()
    assert found.resource_kinds == tuple(RESOURCE_KINDS)
    assert set(found.run_kinds) == set(EXECUTORS)
    assert set(PARSERS) == set(PRE_EXECUTORS) == set(EXECUTORS) == set(DEFERRED_CHECKS)
    assert found.graph_nodes == RADIO_GRAPH._topo

    live_dimensions = {
        (selector.domain, selector.selector) for selector, _spec in registered_dimension_rows()
    }
    projected_dimensions = {
        (source.domain, source.selector) for widget in found.widgets for source in widget.sources
    }
    assert live_dimensions <= projected_dimensions
    assert_catalog_closed(found)


def test_model_widgets_derive_required_default_dimension_and_delivery_from_classes():
    static = _widget("model.foregrounds[].ref_freq")
    assert static.required is True
    assert static.has_default is False
    assert static.delivery == "static_float"
    assert static.dimension == "Hz"
    assert static.visible_when is not None
    assert static.visible_when.rules[1].expected == ("ForegroundOperator",)

    traced = _widget("model.global_signal.depth")
    assert traced.delivery == "traced"
    assert traced.dimension == "K"
    assert traced.required is True

    defaulted = _widget("model.noise_wave.switch_key")
    assert defaulted.required is False
    assert defaulted.has_default is True
    assert defaulted.default == "receiver_input"


@pytest.mark.parametrize(
    "path",
    [
        "observation.from_file.freq_unit",
        "resources.beams.*.normalize",
        "resources.beams.*.phi0_deg",
        "resources.beams.*.phi_sense",
        "resources.projectors.*.normalize_beam",
        "resources.projectors.*.provenance",
        "model.cw_tone.line_width",
        "inference.noise.include_logdet",
        "runs[].optimizer",
        "runs[].learning_rate",
        "runs[].n_steps",
        "runs[].num_warmup",
        "runs[].num_samples",
        "outputs.report.rows",
    ],
)
def test_required_with_no_default_is_never_silently_filled(path):
    found = _widget(path)
    assert found.required or found.required_when is not None
    assert found.has_default is False


def test_observation_and_resource_discriminators_swap_whole_subforms():
    text = """\
runtime: {}
observation:
  from_file:
    format: rhino_hdf5
    path: night.h5
resources:
  beams:
    horn:
      format: cst
      nside: 8
model: {}
runs: [{kind: forward}]
"""
    assert _projected(text, "observation.from_file.freq_unit").visible
    assert _projected(text, "observation.from_file.freq_unit").must_decide
    assert not _projected(text, "observation.freq.grid").visible
    assert _projected(text, "resources.beams.horn.phi0_deg").visible
    assert _projected(text, "resources.beams.horn.phi0_deg").must_decide
    assert not _projected(text, "resources.beams.horn.fwhm_deg").visible

    gaussian = text.replace("format: cst", "format: gaussian")
    assert not _projected(gaussian, "resources.beams.horn.phi0_deg").visible
    assert _projected(gaussian, "resources.beams.horn.fwhm_deg").visible


def test_sky_beam_and_projector_forms_follow_live_discriminators():
    text = """\
runtime: {}
observation: {freq: {grid: [1.0]}, time: {grid: [0.0]}}
resources:
  sky_models:
    sky: {kind: power_law}
  projectors:
    scan: {engine: general_pointing}
model: {}
runs: [{kind: forward}]
"""
    assert _projected(text, "resources.sky_models.sky.spectral_index").visible
    assert not _projected(text, "resources.sky_models.sky.maps").visible
    assert _projected(text, "resources.projectors.scan.normalize_beam").must_decide
    assert _projected(text, "resources.projectors.scan.nside").visible
    assert not _projected(text, "resources.projectors.scan.matrix").visible


def test_model_class_inference_noise_and_check_reason_are_conditional():
    text = """\
runtime: {}
observation: {freq: {grid: [1.0]}, time: {grid: [0.0]}}
model:
  noise: {type: RadiometerNoiseOperator}
inference:
  noise: {kind: radiometer}
  checks:
    linearity: {mode: skip}
runs: [{kind: forward}]
"""
    assert _projected(text, "model.noise.channel_width").visible
    assert _projected(text, "model.noise.channel_width").must_decide
    assert not _projected(text, "model.noise.sigma").visible
    assert _projected(text, "inference.noise.include_logdet").visible
    assert _projected(text, "inference.noise.include_logdet").must_decide
    assert _projected(text, "inference.checks.linearity.reason").visible
    assert _projected(text, "inference.checks.linearity.reason").must_decide

    independent = text.replace("kind: radiometer", "kind: homoscedastic").replace(
        "mode: skip", "mode: report"
    )
    assert not _projected(independent, "inference.noise.include_logdet").visible
    assert _projected(independent, "inference.noise.sigma").visible
    assert not _projected(independent, "inference.checks.linearity.reason").visible


def test_run_kind_and_nested_optimizer_visibility_are_per_concrete_run():
    text = """\
runtime: {}
observation: {freq: {grid: [1.0]}, time: {grid: [0.0]}}
model: {}
runs:
  - {name: fit, kind: optimize, optimizer: adam}
  - {name: sample, kind: nuts}
"""
    assert _projected(text, "runs[0].learning_rate").must_decide
    assert _projected(text, "runs[0].beta1").visible
    assert not _projected(text, "runs[0].num_warmup").visible
    assert _projected(text, "runs[1].num_warmup").must_decide
    assert not _projected(text, "runs[1].beta1").visible


def test_outputs_and_campaign_have_their_planned_states():
    text = """\
runtime: {}
observation: {freq: {grid: [1.0]}, time: {grid: [0.0]}}
model: {}
runs: [{kind: forward}]
outputs:
  report: {}
"""
    assert _projected(text, "outputs.report.rows").must_decide
    signal_paths = _projected(text, "outputs.write.signal_paths")
    assert signal_paths.choices == ("svg", "html", "mermaid")
    campaign = next(
        section
        for section in project_forms(yaml.safe_load(text)).sections
        if section.section_id == "campaign"
    )
    assert campaign.disabled is True
    assert all(widget.disabled for widget in campaign.widgets)


def test_registry_drift_is_a_loud_catalog_failure(monkeypatch):
    monkeypatch.setitem(RESOURCE_KINDS._table, "new_resource", lambda *_: None)
    with pytest.raises(CatalogDrift, match="resource kinds"):
        widget_catalog()


def test_projection_detaches_immutable_parser_containers_for_api_serialization():
    projected = project_forms(
        {
            "schema_version": 1,
            "model": {"gain": {"gain": {"nested": (1, 2)}}},
        }
    )
    settings = next(
        widget
        for section in projected.sections
        for widget in section.widgets
        if widget.path == "model.gain"
    )
    assert settings.value == {"gain": {"nested": [1, 2]}}
    assert dataclasses.asdict(projected)["sections"]


class TestUnitSpellingsReachTheWidget:
    """A quantity control offers the spellings its own dimension accepts.

    Derived inside the builder from ``dimension`` and ``unit_policy`` rather
    than passed at each call site: there are five hundred widgets and one of
    them forgetting would be a control that silently offers nothing.
    """

    def test_a_temperature_and_a_frequency_offer_their_own_spellings(self):
        assert _widget("model.global_signal.depth").units == ("K", "celsius")
        assert _widget("model.global_signal.centre").units == ("Hz", "kHz", "MHz", "GHz")
        assert _widget("model.global_signal.width").units == ("Hz", "kHz", "MHz", "GHz")

    def test_a_single_spelling_dimension_still_reports_its_one_spelling(self):
        assert _widget("model.gain.gain").units == ("dimensionless",)

    def test_a_compound_dimension_offers_nothing(self):
        """``adc_count/K`` is a quotient, not an atom, so the alphabet has no
        second spelling to offer. An invented one-element tuple would read as
        a choice where there is none."""
        widget = _widget("model.adc.scale")
        assert widget.dimension == "adc_count/K"
        assert widget.units == ()

    def test_a_structural_field_offers_nothing(self):
        """``unit_policy == "forbidden"`` means a unit is a refusal, so a
        control that offered one would be inviting an error."""
        forbidden = [
            widget
            for widget in widget_catalog().widgets
            if widget.unit_policy == "forbidden"
        ]
        assert forbidden
        assert all(widget.units == () for widget in forbidden)

    def test_units_are_non_empty_exactly_when_the_dimension_is_an_atom(self):
        from rheplicant.config.units import UNIT_SPELLINGS

        for widget in widget_catalog().widgets:
            expected = (
                ()
                if widget.unit_policy == "forbidden"
                else UNIT_SPELLINGS.get(widget.dimension or "", ())
            )
            assert widget.units == expected, widget.path

    def test_the_projection_carries_them_to_the_client(self):
        projected = _projected(
            "schema_version: 1\nmodel:\n  global_signal:\n"
            "    depth: {value: 0.5, unit: K}\nruns: []\n",
            "model.global_signal.depth",
        )
        assert projected.units == ("K", "celsius")


class TestEnumMembersReachTheWidget:
    """A ``static_str`` field with a closed member list is a select.

    The members are read off the live constants the operators validate
    against, never re-spelled here: a second list is a second thing to keep in
    step, and the one that drifts is the one nothing runs.
    """

    def test_the_enum_table_is_exactly_the_live_enum_dimension_rows(self):
        from rheplicant.config.dimensions import registered_dimension_rows
        from rheplicant.gui.form_catalog import _ENUMS

        live = {
            selector.selector
            for selector, spec in registered_dimension_rows()
            if selector.domain == "model_field"
            and spec.disposition == "structural"
            and spec.reason == "enum"
        }
        assert set(_ENUMS) == live, "an enum field without members ships an empty select"

    def test_the_members_are_the_operators_own_constants(self):
        from rheplicant.gui.form_catalog import _ENUMS
        from rheplicant.radio.filters.base import _MODES
        from rheplicant.radio.instrument.calibration import LINESHAPES

        assert set(_ENUMS.values()) == {_MODES, LINESHAPES}

    def test_the_three_filters_agree_on_one_member_list(self):
        """They share one widget -- ``model.filters[].mode`` -- so a
        disagreement would have to be resolved by the census rather than
        discovered by a user."""
        assert _widget("model.filters[].mode").choices == ("extract", "remove")

    def test_a_single_class_enum_field_carries_its_members(self):
        assert _widget("model.cw_tone.lineshape").choices == ("sinc2", "gaussian")

    def test_a_structural_field_that_is_not_an_enum_offers_no_choices(self):
        """``switch_key`` is a ``Coordinates.extra`` key and ``projector`` a
        resource reference: both are structural, neither is closed."""
        assert _widget("model.noise_wave.switch_key").choices == ()
        assert _widget("model.filters[].projector").choices == ()
