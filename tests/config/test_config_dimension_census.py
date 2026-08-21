"""The authoritative dimension catalog remains complete and reachable."""

import dataclasses

import rheplicant.config.dimensions as dimension_module
from rheplicant.config.derive import DERIVATIONS
from rheplicant.config.dimension_catalog import (
    CONFIG_CONTEXTUAL,
    CONFIG_DIMENSIONS,
    CONFIG_SPECIAL,
    FORMULA_REGISTRATIONS,
    MODEL_DIMENSIONS,
    MODEL_FORMULA_BINDINGS,
    MODEL_SPECIAL,
    RESOURCE_DIMENSIONS,
    RESOURCE_OUTPUTS,
    RESOURCE_SPECIAL,
)
from rheplicant.config.dimensions import dimension_spec_for
from rheplicant.config.resources import RESOURCE_KINDS
from rheplicant.config.sections.model import operator_table
from rheplicant.config.values import VALUE_FORMS

_CONFIG_SELECTORS = frozenset(
    {
        "observation.freq.grid",
        "observation.time.grid",
        "observation.time.epoch",
        "observation.time.integration_time",
        "observation.time.channel_width",
        "observation.site.lat_deg",
        "observation.site.lon_deg",
        "observation.site.alt_m",
        "observation.environment.temperature",
        "observation.environment.humidity",
        "observation.pointing.az_deg",
        "observation.pointing.el_deg",
        "observation.pointing.selfrot_deg",
        "observation.pointing.table",
        "observation.pointing.selfrot",
        "observation.pointing.lst.lst0_deg",
        "observation.pointing.lst.from_file",
        "observation.switching.index",
        "observation.switching.dwell",
        "inference.noise.channel_width",
        "inference.noise.integration_time",
        "observation.from_file.settle_seconds",
        "observation.from_file.thermistor_columns.*",
        "inference.parameters.*.init",
        "inference.parameters.*.ref",
        "inference.parameters.*.prior.normal.loc",
        "inference.parameters.*.prior.normal.scale",
        "inference.parameters.*.prior.uniform.low",
        "inference.parameters.*.prior.uniform.high",
        "inference.parameters.*.prior.log_normal.loc",
        "inference.parameters.*.prior.log_normal.scale",
        "inference.truth.*",
        "runs[].at.*",
        "inference.observed.at.*",
        "inference.observed.*.at.*",
        "inference.observed.realise.sigma",
        "inference.observed.*.realise.sigma",
        "inference.observed.file",
        "inference.observed.*.file",
        "inference.noise.sigma",
        "inference.noise.floor",
        "observation.data",
        "observation.from_file",
        "model.*.eqx_leaves",
        "inference.parameters.*.prior.python",
        "observation.environment.extra.*",
        "observation.extra.*",
        "observation.aux.flags",
        "python.args.*",
        "resources.bases.*.time",
        "resources.bases.*.freq",
        "resources.beams.*.maps",
        "resources.beams.*.sky_fraction",
        "resources.s_params.*",
        "resources.sky_models.*.maps",
        "resources.sky_models.*.freq",
    }
)

_RESOURCE_SELECTORS = frozenset(
    {
        "rheplicant.config.kinds.bases.build_basis.time.n_basis",
        "rheplicant.config.kinds.bases.build_basis.freq.n_basis",
        "rheplicant.config.kinds.beams.build_beam.nside",
        "rheplicant.config.kinds.beams.build_beam.cst.phi0_deg",
        "rheplicant.config.kinds.beams.build_beam.npy.maps",
        "rheplicant.config.kinds.beams.build_beam.npz.maps",
        "rheplicant.config.kinds.beams.build_beam.inline.maps",
        "rheplicant.config.kinds.beams.build_beam.healpix.freq",
        "rheplicant.config.kinds.beams.build_beam.gaussian.fwhm_deg",
        "rheplicant.config.kinds.beams.build_beam.gaussian.sigma_deg",
        "rheplicant.config.kinds.beams.build_beam.horizon.el_deg",
        "rheplicant.config.kinds.beams.build_beam.horizon.apod_deg",
        "rheplicant.config.kinds.projectors.build_projector.matrix.matrix",
        "rheplicant.config.kinds.projectors.build_projector.general_pointing.beam_alms",
        "rheplicant.config.kinds.projectors.build_projector.general_pointing.lmax",
        "rheplicant.config.kinds.projectors.build_projector.general_pointing.nside",
        "rheplicant.config.kinds.projectors.build_projector.general_pointing.lat_deg",
        "rheplicant.config.kinds.projectors.build_projector.general_pointing.beam_iterations",
        "rheplicant.config.kinds.projectors.build_projector.driftscan.lmax",
        "rheplicant.config.kinds.projectors.build_projector.driftscan.lat_deg",
        "rheplicant.config.kinds.projectors.build_projector.driftscan.az_deg",
        "rheplicant.config.kinds.projectors.build_projector.driftscan.el_deg",
        "rheplicant.config.kinds.projectors.build_projector.driftscan.selfrot_deg",
        "rheplicant.config.kinds.projectors.build_projector.driftscan.apod_deg",
        "rheplicant.config.kinds.projectors.build_projector.driftscan.lst_ref_deg",
        "rheplicant.config.kinds.projectors.build_projector.driftscan.beam_iterations",
        "rheplicant.config.kinds.projectors.build_projector.driftscan.mask_iterations",
        "rheplicant.config.kinds.projectors.build_projector.driftscan.freq_chunk",
        "rheplicant.config.kinds.s_params.build_s_param.termination.z0",
        "rheplicant.config.kinds.s_params.build_s_param.termination.impedance",
        "rheplicant.config.kinds.s_params.build_s_param.termination.n",
        "rheplicant.config.kinds.s_params.build_s_param.cable.length",
        "rheplicant.config.kinds.s_params.build_s_param.cable.velocity_factor",
        "rheplicant.config.kinds.s_params.build_s_param.cable.loss",
        "rheplicant.config.kinds.sky_models.build_sky_model.uniform.amplitude",
        "rheplicant.config.kinds.sky_models.build_sky_model.uniform.n_pix",
        "rheplicant.config.kinds.sky_models.build_sky_model.power_law.amplitude",
        "rheplicant.config.kinds.sky_models.build_sky_model.power_law.spectral_index",
        "rheplicant.config.kinds.sky_models.build_sky_model.power_law.ref_freq",
        "rheplicant.config.kinds.sky_models.build_sky_model.power_law.n_pix",
        "rheplicant.config.kinds.sky_models.build_sky_model.maps.maps",
        "rheplicant.config.kinds.sky_models.build_sky_model.maps.freq",
        "rheplicant.config.kinds.sky_models.build_sky_model.maps.nside",
        "rheplicant.config.kinds.sky_models.build_sky_model.gdsm.nside",
        "rheplicant.config.kinds.arrays.build_array.value",
        "rheplicant.config.kinds.beams.build_beam.python.args.*",
        "rheplicant.config.kinds.sky_models.build_sky_model.python.args.*",
        "rheplicant.config.kinds.projectors.build_projector.matrix.provenance",
        "rheplicant.config.kinds.s_params.build_s_param.touchstone.file",
    }
)

_FORMULAS = frozenset(
    {
        "channel_spacing", "sample_cadence", "basis_matrix", "unit_mean_free",
        "horizon_fraction", "interpolate_onto", "normal", "uniform", "stack",
        "ref", "file", "python", "basis_fit", "modifier_affine",
        "transform_affine", "matmul", "basis_expand", "exp_log",
        "beam_analysis", "part_same", "part_angle", "normalize", "adc", "gain",
        "apply_cal", "cw_centre", "cw_level", "foreground", "global_signal",
        "uniform_sky", "ionosphere", "ground_pickup", "beam_spill",
        "antenna_loss", "additive_noise_emi", "noise_wave", "sky_projection",
        "basis_temperature", "filters_averaging", "sky_space",
        "radiometer_fraction",
    }
)


def _qualified(cls: type, field: str | None = None) -> str:
    name = f"{cls.__module__}.{cls.__qualname__}"
    return name if field is None else f"{name}.{field}"


def test_model_catalog_covers_all_28_classes_and_66_fields():
    classes = {cls for choices in operator_table().values() for cls in choices}
    fields = {
        _qualified(cls, field.name)
        for cls in classes
        for field in dataclasses.fields(cls)
        if field.init
    }
    assert len(classes) == 28
    assert len(fields) == 66
    assert len(MODEL_DIMENSIONS) == 56
    assert len(MODEL_SPECIAL) == 10
    assert fields == {row[0] for row in MODEL_DIMENSIONS} | set(MODEL_SPECIAL)
    assert set(MODEL_FORMULA_BINDINGS) == {_qualified(cls) for cls in classes}


def test_every_catalog_row_is_registered_and_reachable():
    rows = dimension_module.registered_dimension_rows()
    actual = {(row.domain, row.selector) for row, _ in rows}
    model = {("model_field", _qualified(cls, field.name))
             for choices in operator_table().values() for cls in choices
             for field in dataclasses.fields(cls) if field.init}
    expected = model | {
        *(('resource_field', selector) for selector in _RESOURCE_SELECTORS),
        *(('config_path', selector) for selector in _CONFIG_SELECTORS),
    }
    assert actual == expected
    for domain, selector in expected:
        assert dimension_spec_for(domain, selector) is not None


def test_six_resource_parsers_and_outputs_are_independently_censused():
    assert set(RESOURCE_KINDS) == {
        "arrays", "bases", "beams", "projectors", "s_params", "sky_models"
    }
    assert len(_RESOURCE_SELECTORS) == 49
    assert {selector for selector, _ in RESOURCE_DIMENSIONS} | set(RESOURCE_SPECIAL) == \
        _RESOURCE_SELECTORS
    assert set(RESOURCE_OUTPUTS) == {
        "resources.bases.*.time", "resources.bases.*.freq",
        "resources.beams.*.maps", "resources.beams.*.sky_fraction",
        "resources.s_params.*", "resources.sky_models.*.maps",
        "resources.sky_models.*.freq",
    }


def test_config_patterns_are_independent_of_the_catalog_constants():
    catalogued = (
        {selector for selector, _ in CONFIG_DIMENSIONS}
        | set(CONFIG_CONTEXTUAL)
        | set(CONFIG_SPECIAL)
        | set(RESOURCE_OUTPUTS)
    )
    assert catalogued == _CONFIG_SELECTORS


def test_nested_value_forms_and_six_derivations_are_independently_censused():
    assert set(VALUE_FORMS) == {
        "value", "zeros", "ones", "full", "list", "linspace", "arange",
        "modulo", "from_grid", "basis_fit", "normal", "uniform", "file",
        "ref", "from", "stack", "from_switch_order", "python",
    }
    assert set(DERIVATIONS) == {
        "channel_spacing", "sample_cadence", "basis_matrix", "unit_mean_free",
        "horizon_fraction", "interpolate_onto",
    }
    assert {
        "basis_fit", "normal", "uniform", "file", "ref", "stack", "python",
        *set(DERIVATIONS),
    } <= _FORMULAS


def test_named_formulas_and_operator_bindings_are_complete():
    by_name = {formula.name: formula for formula in FORMULA_REGISTRATIONS}
    assert set(by_name) == _FORMULAS
    assert set(dimension_module._FORMULA_REGISTRY) == _FORMULAS
    for producer, binding in MODEL_FORMULA_BINDINGS.items():
        assert binding.formulas.count(binding.output_formula) == 1, producer
        assert set(binding.formulas) <= set(by_name), producer
        assert producer in by_name[binding.output_formula].producers
        assert all(by_name[name].producers for name in binding.formulas)
    producer_roles = [
        (producer, operand.role)
        for formula in by_name.values()
        for producer in formula.producers
        for operand in formula.operands
    ]
    assert len(producer_roles) == len(set(producer_roles))
