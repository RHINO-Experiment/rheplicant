"""The authoritative, explicit A9 dimension catalog."""

from __future__ import annotations

import dataclasses

from rheplicant.config.dimensions import (
    DimensionSpec,
    FormulaOperand,
    FormulaRegistration,
    FormulaRule,
    OperatorFormulaBinding,
    register_dimension,
    register_formula_checked,
    signature,
)

MODEL_DIMENSIONS = (
    ("rheplicant.radio.instrument.adc.ADCOperator.scale", "adc_count/K"),
    ("rheplicant.radio.instrument.adc.ADCOperator.n_bits", "bits"),
    ("rheplicant.radio.instrument.antenna_loss.AntennaLossOperator.efficiency", "dimensionless"),
    ("rheplicant.radio.instrument.antenna_loss.AntennaLossOperator.t_physical", "K"),
    ("rheplicant.radio.instrument.calibration.ApplyCalibrationOperator.gain", "dimensionless"),
    ("rheplicant.radio.environment.atmosphere.AtmosphericEmissionOperator.t_atm", "K"),
    ("rheplicant.radio.backend.averaging.BackendOperator.n_chunk", "samples"),
    ("rheplicant.radio.t_sys.BasisTemperatureOperator.coeff", "K"),
    ("rheplicant.radio.t_sys.BasisTemperatureOperator.time_basis", "dimensionless"),
    ("rheplicant.radio.t_sys.BasisTemperatureOperator.freq_basis", "dimensionless"),
    ("rheplicant.radio.instrument.beam_spill.BeamSpillOperator.sky_fraction", "dimensionless"),
    ("rheplicant.radio.instrument.beam_spill.BeamSpillOperator.t_ground", "K"),
    ("rheplicant.radio.instrument.calibration.CWCalibrationOperator.amplitude", "K"),
    ("rheplicant.radio.instrument.calibration.CWCalibrationOperator.tone_freq", "Hz"),
    ("rheplicant.radio.instrument.calibration.CWCalibrationOperator.line_width", "Hz"),
    ("rheplicant.radio.instrument.calibration.CWCalibrationOperator.drift_rate", "Hz/s"),
    (
        "rheplicant.radio.instrument.calibration.CWCalibrationOperator.amplitude_drift_rate",
        "dimensionless/s",
    ),
    (
        "rheplicant.radio.instrument.calibration.CWCalibrationOperator.protect_floor",
        "dimensionless",
    ),
    ("rheplicant.radio.instrument.calibration.CalLoadOperator.t_load", "K"),
    ("rheplicant.radio.instrument.emi.EMIOperator.amplitude", "K"),
    ("rheplicant.radio.instrument.emi.EMIOperator.period", "channels"),
    ("rheplicant.radio.backend.flagging.FlaggingOperator.threshold", "adc_count"),
    ("rheplicant.radio.sky.foregrounds.ForegroundOperator.amplitude", "K"),
    ("rheplicant.radio.sky.foregrounds.ForegroundOperator.spectral_index", "dimensionless"),
    ("rheplicant.radio.sky.foregrounds.ForegroundOperator.ref_freq", "Hz"),
    ("rheplicant.radio.filters.fourier.FourierBandFilter.axis", "count"),
    ("rheplicant.radio.filters.fourier.FourierBandFilter.low", "cycles/samples"),
    ("rheplicant.radio.filters.fourier.FourierBandFilter.high", "cycles/samples"),
    ("rheplicant.radio.instrument.gain.GainOperator.gain", "dimensionless"),
    ("rheplicant.radio.sky.global_signal.GlobalSignalOperator.depth", "K"),
    ("rheplicant.radio.sky.global_signal.GlobalSignalOperator.centre", "Hz"),
    ("rheplicant.radio.sky.global_signal.GlobalSignalOperator.width", "Hz"),
    ("rheplicant.radio.environment.ground.GroundPickupOperator.coupling", "dimensionless"),
    ("rheplicant.radio.environment.ground.GroundPickupOperator.t_ground", "K"),
    ("rheplicant.radio.environment.ionosphere.IonosphereOperator.delta", "dimensionless"),
    ("rheplicant.radio.environment.ionosphere.IonosphereOperator.ref_freq", "Hz"),
    ("rheplicant.radio.instrument.noise.NoiseOperator.sigma", "K"),
    ("rheplicant.radio.instrument.noise_wave.NoiseWaveOperator.t_unc", "K"),
    ("rheplicant.radio.instrument.noise_wave.NoiseWaveOperator.t_cos", "K"),
    ("rheplicant.radio.instrument.noise_wave.NoiseWaveOperator.t_sin", "K"),
    ("rheplicant.radio.instrument.noise_wave.NoiseWaveOperator.t_rx", "K"),
    ("rheplicant.radio.instrument.noise_wave.NoiseWaveOperator.gamma_src_re", "dimensionless"),
    ("rheplicant.radio.instrument.noise_wave.NoiseWaveOperator.gamma_src_im", "dimensionless"),
    ("rheplicant.radio.instrument.noise_wave.NoiseWaveOperator.gamma_rec_re", "dimensionless"),
    ("rheplicant.radio.instrument.noise_wave.NoiseWaveOperator.gamma_rec_im", "dimensionless"),
    ("rheplicant.radio.sky.point_sources.PointSourceOperator.level", "K"),
    ("rheplicant.radio.environment.rfi.RFIOperator.amplitude", "K"),
    ("rheplicant.radio.environment.rfi.RFIOperator.occupancy", "dimensionless"),
    ("rheplicant.radio.instrument.noise.RadiometerNoiseOperator.channel_width", "Hz"),
    ("rheplicant.radio.instrument.noise.RadiometerNoiseOperator.integration_time", "s"),
    ("rheplicant.radio.instrument.receiver.ReceiverOperator.bandpass", "dimensionless"),
    ("rheplicant.radio.filters.sidereal.SiderealFilter.n_days", "count"),
    ("rheplicant.radio.sky.uniform.SkyOperator.amplitude", "K"),
    ("rheplicant.radio.filters.skyspace.SkySpaceFilter.regularization", "dimensionless"),
    ("rheplicant.radio.filters.skyspace.SkySpaceFilter.cg_tol", "dimensionless"),
    ("rheplicant.radio.filters.skyspace.SkySpaceFilter.cg_maxiter", "count"),
)

MODEL_REQUIRED_UNIT = frozenset(
    {
        "rheplicant.radio.instrument.adc.ADCOperator.scale",
        "rheplicant.radio.instrument.gain.GainOperator.gain",
        "rheplicant.radio.instrument.calibration.ApplyCalibrationOperator.gain",
        "rheplicant.radio.backend.flagging.FlaggingOperator.threshold",
        "rheplicant.radio.filters.skyspace.SkySpaceFilter.regularization",
    }
)

MODEL_SPECIAL = {
    "rheplicant.radio.instrument.calibration.CWCalibrationOperator.lineshape": (
        "structural",
        "enum",
    ),
    "rheplicant.radio.filters.fourier.FourierBandFilter.mode": ("structural", "enum"),
    "rheplicant.radio.backend.flagging.MomentRFIFlaggingOperator.config": (
        "open",
        "opaque third-party keyword mapping",
    ),
    "rheplicant.radio.backend.flagging.MomentRFIFlaggingOperator.kernel_shapes": (
        "structural",
        "heterogeneous sample-by-channel shape tuples",
    ),
    "rheplicant.radio.instrument.noise_wave.NoiseWaveOperator.switch_key": (
        "structural",
        "Coordinates.extra key",
    ),
    "rheplicant.radio.filters.sidereal.SiderealFilter.mode": ("structural", "enum"),
    "rheplicant.radio.sky.source.SkySourceOperator.sky_model": (
        "structural",
        "resource object reference",
    ),
    "rheplicant.radio.sky.source.SkySourceOperator.projector": (
        "structural",
        "resource object reference",
    ),
    "rheplicant.radio.filters.skyspace.SkySpaceFilter.projector": (
        "structural",
        "resource object reference",
    ),
    "rheplicant.radio.filters.skyspace.SkySpaceFilter.mode": ("structural", "enum"),
}

RESOURCE_DIMENSIONS = (
    ("rheplicant.config.kinds.bases.build_basis.time.n_basis", "count"),
    ("rheplicant.config.kinds.bases.build_basis.freq.n_basis", "count"),
    ("rheplicant.config.kinds.beams.build_beam.nside", "count"),
    ("rheplicant.config.kinds.beams.build_beam.cst.phi0_deg", "deg"),
    ("rheplicant.config.kinds.beams.build_beam.npy.maps", "dimensionless"),
    ("rheplicant.config.kinds.beams.build_beam.npz.maps", "dimensionless"),
    ("rheplicant.config.kinds.beams.build_beam.inline.maps", "dimensionless"),
    ("rheplicant.config.kinds.beams.build_beam.healpix.freq", "Hz"),
    ("rheplicant.config.kinds.beams.build_beam.gaussian.fwhm_deg", "deg"),
    ("rheplicant.config.kinds.beams.build_beam.gaussian.sigma_deg", "deg"),
    ("rheplicant.config.kinds.beams.build_beam.horizon.el_deg", "deg"),
    ("rheplicant.config.kinds.beams.build_beam.horizon.apod_deg", "deg"),
    ("rheplicant.config.kinds.projectors.build_projector.matrix.matrix", "dimensionless"),
    (
        "rheplicant.config.kinds.projectors.build_projector.general_pointing.beam_alms",
        "dimensionless",
    ),
    ("rheplicant.config.kinds.projectors.build_projector.general_pointing.lmax", "count"),
    ("rheplicant.config.kinds.projectors.build_projector.general_pointing.nside", "count"),
    ("rheplicant.config.kinds.projectors.build_projector.general_pointing.lat_deg", "deg"),
    (
        "rheplicant.config.kinds.projectors.build_projector.general_pointing.beam_iterations",
        "count",
    ),
    ("rheplicant.config.kinds.projectors.build_projector.driftscan.lmax", "count"),
    ("rheplicant.config.kinds.projectors.build_projector.driftscan.lat_deg", "deg"),
    ("rheplicant.config.kinds.projectors.build_projector.driftscan.az_deg", "deg"),
    ("rheplicant.config.kinds.projectors.build_projector.driftscan.el_deg", "deg"),
    ("rheplicant.config.kinds.projectors.build_projector.driftscan.selfrot_deg", "deg"),
    ("rheplicant.config.kinds.projectors.build_projector.driftscan.apod_deg", "deg"),
    ("rheplicant.config.kinds.projectors.build_projector.driftscan.lst_ref_deg", "deg"),
    ("rheplicant.config.kinds.projectors.build_projector.driftscan.beam_iterations", "count"),
    ("rheplicant.config.kinds.projectors.build_projector.driftscan.mask_iterations", "count"),
    ("rheplicant.config.kinds.projectors.build_projector.driftscan.freq_chunk", "channels"),
    ("rheplicant.config.kinds.s_params.build_s_param.termination.z0", "ohm"),
    ("rheplicant.config.kinds.s_params.build_s_param.termination.impedance", "ohm"),
    ("rheplicant.config.kinds.s_params.build_s_param.termination.n", "channels"),
    ("rheplicant.config.kinds.s_params.build_s_param.cable.length", "m"),
    ("rheplicant.config.kinds.s_params.build_s_param.cable.velocity_factor", "dimensionless"),
    ("rheplicant.config.kinds.s_params.build_s_param.cable.loss", "dimensionless"),
    ("rheplicant.config.kinds.sky_models.build_sky_model.uniform.amplitude", "K"),
    ("rheplicant.config.kinds.sky_models.build_sky_model.uniform.n_pix", "count"),
    ("rheplicant.config.kinds.sky_models.build_sky_model.power_law.amplitude", "K"),
    (
        "rheplicant.config.kinds.sky_models.build_sky_model.power_law.spectral_index",
        "dimensionless",
    ),
    ("rheplicant.config.kinds.sky_models.build_sky_model.power_law.ref_freq", "Hz"),
    ("rheplicant.config.kinds.sky_models.build_sky_model.power_law.n_pix", "count"),
    ("rheplicant.config.kinds.sky_models.build_sky_model.maps.maps", "K"),
    ("rheplicant.config.kinds.sky_models.build_sky_model.maps.freq", "Hz"),
    ("rheplicant.config.kinds.sky_models.build_sky_model.maps.nside", "count"),
    ("rheplicant.config.kinds.sky_models.build_sky_model.gdsm.nside", "count"),
)

RESOURCE_SPECIAL = {
    "rheplicant.config.kinds.arrays.build_array.value": (
        "open",
        "dimension comes from explicit unit and is retained across ref",
    ),
    "rheplicant.config.kinds.beams.build_beam.python.args.*": ("open", "executable hatch argument"),
    "rheplicant.config.kinds.sky_models.build_sky_model.python.args.*": (
        "open",
        "executable hatch argument",
    ),
    "rheplicant.config.kinds.projectors.build_projector.matrix.provenance": (
        "structural",
        "opaque provenance",
    ),
    "rheplicant.config.kinds.s_params.build_s_param.touchstone.file": (
        "structural",
        "Touchstone object consumed before interpolation",
    ),
}

RESOURCE_OUTPUTS = {
    "resources.bases.*.time": "dimensionless",
    "resources.bases.*.freq": "dimensionless",
    "resources.beams.*.maps": "dimensionless",
    "resources.beams.*.sky_fraction": "dimensionless",
    "resources.s_params.*": "dimensionless",
    "resources.sky_models.*.maps": "K",
    "resources.sky_models.*.freq": "Hz",
}

CONFIG_DIMENSIONS = (
    ("observation.freq.grid", "Hz"),
    ("observation.time.grid", "s"),
    ("observation.time.epoch", "unix_s"),
    ("observation.time.integration_time", "s"),
    ("observation.time.channel_width", "Hz"),
    ("observation.site.lat_deg", "deg"),
    ("observation.site.lon_deg", "deg"),
    ("observation.site.alt_m", "m"),
    ("observation.environment.temperature", "K"),
    ("observation.environment.humidity", "dimensionless"),
    ("observation.pointing.az_deg", "deg"),
    ("observation.pointing.el_deg", "deg"),
    ("observation.pointing.selfrot_deg", "deg"),
    ("observation.pointing.table", "deg"),
    ("observation.pointing.selfrot", "deg"),
    ("observation.pointing.lst.lst0_deg", "deg"),
    ("observation.pointing.lst.from_file", "deg"),
    ("observation.switching.index", "count"),
    ("observation.switching.dwell", "samples"),
    ("inference.noise.channel_width", "Hz"),
    ("inference.noise.integration_time", "s"),
    ("observation.from_file.settle_seconds", "s"),
    ("observation.from_file.thermistor_columns.*", "count"),
)
CONFIG_REQUIRED_UNIT = frozenset({"observation.environment.humidity"})
CONFIG_CONTEXTUAL = {
    "inference.parameters.*.init": "latent",
    "inference.parameters.*.ref": "latent",
    "inference.parameters.*.prior.normal.loc": "latent",
    "inference.parameters.*.prior.normal.scale": "latent",
    "inference.parameters.*.prior.uniform.low": "latent",
    "inference.parameters.*.prior.uniform.high": "latent",
    "inference.parameters.*.prior.log_normal.loc": "latent",
    "inference.parameters.*.prior.log_normal.scale": "latent",
    "inference.truth.*": "latent",
    "runs[].at.*": "latent",
    "inference.observed.at.*": "latent",
    "inference.observed.*.at.*": "latent",
    "inference.observed.realise.sigma": "prediction",
    "inference.observed.*.realise.sigma": "prediction",
    "inference.observed.file": "prediction",
    "inference.observed.*.file": "prediction",
    "inference.noise.sigma": "prediction",
    "inference.noise.floor": "prediction",
    "observation.data": "model_input",
}
CONFIG_SPECIAL = {
    "observation.from_file": ("structural", "RhinoObservation object returned by the reader"),
    "model.*.eqx_leaves": ("structural", "operator restore onto a declared template"),
    "inference.parameters.*.prior.python": (
        "structural",
        "distribution object returned by the Python hatch",
    ),
    "observation.environment.extra.*": ("open", "user namespace"),
    "observation.extra.*": ("open", "user namespace"),
    "observation.aux.flags": ("structural", "boolean mask where true means bad"),
    "python.args.*": ("open", "executable hatch argument"),
}


def _formula_spec(token: str) -> DimensionSpec:
    if token == "D":
        return DimensionSpec("contextual", resolver="outer", unit_policy="inherited")
    if token == "resource":
        return DimensionSpec("contextual", resolver="resource", unit_policy="inherited")
    if token in ("open", "structural"):
        return DimensionSpec(token, unit_policy="optional" if token == "open" else "forbidden")
    return DimensionSpec("fixed", signature(token), unit_policy="inherited")


def _formula(
    name: str, rule: FormulaRule, result: str, *operands: tuple[str, str, int]
) -> FormulaRegistration:
    return FormulaRegistration(
        name,
        rule,
        _formula_spec(result),
        tuple(
            FormulaOperand(role, _formula_spec(spec), exponent) for role, spec, exponent in operands
        ),
    )


FORMULA_REGISTRATIONS = (
    _formula("channel_spacing", "fixed", "Hz", ("times", "dimensionless", 1)),
    _formula("sample_cadence", "fixed", "s", ("times", "dimensionless", 1)),
    _formula("basis_matrix", "fixed", "dimensionless"),
    _formula("unit_mean_free", "fixed", "dimensionless", ("bandpass", "dimensionless", 1)),
    _formula("horizon_fraction", "fixed", "dimensionless", ("projector", "structural", 1)),
    _formula("interpolate_onto", "fixed", "dimensionless", ("of", "structural", 1)),
    _formula("normal", "same", "D", ("loc", "D", 1), ("scale", "D", 1)),
    _formula("uniform", "same", "D", ("low", "D", 1), ("high", "D", 1)),
    _formula("stack", "same", "D", ("entry[]", "D", 1)),
    _formula("ref", "same", "D", ("source", "resource", 1)),
    _formula("file", "same", "D", ("snapshot", "structural", 1)),
    _formula("python", "same", "D", ("args.*", "open", 1)),
    _formula("basis_fit", "same", "D", ("field", "D", 1), ("basis", "dimensionless", 1)),
    _formula(
        "modifier_affine",
        "affine",
        "D",
        ("value", "D", 1),
        ("scale", "dimensionless", 1),
        ("offset", "D", 1),
    ),
    _formula(
        "transform_affine",
        "affine",
        "D",
        ("value", "D", 1),
        ("scale", "dimensionless", 1),
        ("offset", "D", 1),
    ),
    _formula("matmul", "same", "D", ("design", "dimensionless", 1), ("coefficient", "D", 1)),
    _formula("basis_expand", "same", "D", ("basis", "dimensionless", 1), ("coefficient", "D", 1)),
    _formula("exp_log", "fixed", "dimensionless", ("value", "dimensionless", 1)),
    _formula("beam_analysis", "fixed", "dimensionless", ("maps", "dimensionless", 1)),
    _formula("part_same", "same", "D", ("value", "D", 1)),
    _formula("part_angle", "fixed", "deg", ("value", "open", 1)),
    _formula("normalize", "fixed", "dimensionless", ("value", "open", 1)),
    _formula("adc", "product", "adc_count", ("scale", "adc_count/K", 1), ("input", "K", 1)),
    _formula("gain", "same", "D", ("input", "D", 1), ("gain", "dimensionless", 1)),
    _formula("apply_cal", "same", "D", ("input", "D", 1), ("gain", "dimensionless", 1)),
    _formula(
        "cw_centre",
        "fixed",
        "Hz",
        ("tone_freq", "Hz", 1),
        ("drift_rate", "Hz/s", 1),
        ("time", "s", 1),
    ),
    _formula(
        "cw_level",
        "fixed",
        "K",
        ("amplitude", "K", 1),
        ("amplitude_drift_rate", "dimensionless/s", 1),
        ("time", "s", 1),
    ),
    _formula(
        "foreground",
        "fixed",
        "K",
        ("amplitude", "K", 1),
        ("spectral_index", "dimensionless", 1),
        ("frequency", "Hz", 1),
    ),
    _formula(
        "global_signal", "fixed", "K", ("depth", "K", 1), ("centre", "Hz", 1), ("width", "Hz", 1)
    ),
    _formula("uniform_sky", "fixed", "K", ("amplitude", "K", 1)),
    _formula("ionosphere", "same", "K", ("input", "K", 1), ("delta", "dimensionless", 1)),
    _formula(
        "ground_pickup",
        "fixed",
        "K",
        ("input", "K", 1),
        ("temperature", "K", 1),
        ("coupling", "dimensionless", 1),
    ),
    _formula(
        "beam_spill",
        "fixed",
        "K",
        ("sky", "K", 1),
        ("ground", "K", 1),
        ("fraction", "dimensionless", 1),
    ),
    _formula(
        "antenna_loss",
        "fixed",
        "K",
        ("input", "K", 1),
        ("physical", "K", 1),
        ("efficiency", "dimensionless", 1),
    ),
    _formula("additive_noise_emi", "same", "K", ("input", "K", 1), ("addition", "K", 1)),
    _formula("noise_wave", "fixed", "K"),
    _formula("sky_projection", "fixed", "K", ("sky", "K", 1), ("beam", "dimensionless", 1)),
    _formula(
        "basis_temperature", "fixed", "K", ("basis", "dimensionless", 1), ("coefficient", "K", 1)
    ),
    _formula("filters_averaging", "same", "D", ("input", "D", 1)),
    _formula("sky_space", "same", "D", ("input", "D", 1)),
    _formula(
        "radiometer_fraction",
        "radiometer",
        "dimensionless",
        ("channel_width", "Hz", 1),
        ("integration_time", "s", 1),
    ),
)


def _operator_binding(output_formula: str, *validation_formulas: str) -> OperatorFormulaBinding:
    return OperatorFormulaBinding((output_formula, *validation_formulas), output_formula)


MODEL_FORMULA_BINDINGS = {
    "rheplicant.radio.instrument.adc.ADCOperator": _operator_binding("adc"),
    "rheplicant.radio.instrument.antenna_loss.AntennaLossOperator": _operator_binding(
        "antenna_loss"
    ),
    "rheplicant.radio.instrument.calibration.ApplyCalibrationOperator": _operator_binding(
        "apply_cal"
    ),
    "rheplicant.radio.environment.atmosphere.AtmosphericEmissionOperator": _operator_binding(
        "additive_noise_emi"
    ),
    "rheplicant.radio.backend.averaging.BackendOperator": _operator_binding("filters_averaging"),
    "rheplicant.radio.t_sys.BasisTemperatureOperator": _operator_binding("basis_temperature"),
    "rheplicant.radio.instrument.beam_spill.BeamSpillOperator": _operator_binding("beam_spill"),
    "rheplicant.radio.instrument.calibration.CWCalibrationOperator": _operator_binding(
        "cw_level", "cw_centre"
    ),
    "rheplicant.radio.instrument.calibration.CalLoadOperator": _operator_binding(
        "additive_noise_emi"
    ),
    "rheplicant.radio.instrument.emi.EMIOperator": _operator_binding("additive_noise_emi"),
    "rheplicant.radio.backend.flagging.FlaggingOperator": _operator_binding("filters_averaging"),
    "rheplicant.radio.backend.flagging.MomentRFIFlaggingOperator": _operator_binding(
        "filters_averaging"
    ),
    "rheplicant.radio.sky.foregrounds.ForegroundOperator": _operator_binding("foreground"),
    "rheplicant.radio.filters.fourier.FourierBandFilter": _operator_binding("filters_averaging"),
    "rheplicant.radio.instrument.gain.GainOperator": _operator_binding("gain"),
    "rheplicant.radio.sky.global_signal.GlobalSignalOperator": _operator_binding("global_signal"),
    "rheplicant.radio.environment.ground.GroundPickupOperator": _operator_binding("ground_pickup"),
    "rheplicant.radio.environment.ionosphere.IonosphereOperator": _operator_binding("ionosphere"),
    "rheplicant.radio.instrument.noise.NoiseOperator": _operator_binding("additive_noise_emi"),
    "rheplicant.radio.instrument.noise.RadiometerNoiseOperator": _operator_binding(
        "additive_noise_emi", "radiometer_fraction"
    ),
    "rheplicant.radio.instrument.noise_wave.NoiseWaveOperator": _operator_binding("noise_wave"),
    "rheplicant.radio.sky.point_sources.PointSourceOperator": _operator_binding(
        "additive_noise_emi"
    ),
    "rheplicant.radio.instrument.receiver.ReceiverOperator": _operator_binding("filters_averaging"),
    "rheplicant.radio.filters.sidereal.SiderealFilter": _operator_binding("filters_averaging"),
    "rheplicant.radio.sky.source.SkySourceOperator": _operator_binding("sky_projection"),
    "rheplicant.radio.filters.skyspace.SkySpaceFilter": _operator_binding("sky_space"),
    "rheplicant.radio.sky.uniform.SkyOperator": _operator_binding("uniform_sky"),
    "rheplicant.radio.environment.rfi.RFIOperator": _operator_binding("additive_noise_emi"),
}

FORMULA_REGISTRATIONS = tuple(
    dataclasses.replace(
        registration,
        producers=tuple(
            (
                "rheplicant.config.inflight.grids._static_number"
                if registration.name == "cw_centre"
                and class_name
                == "rheplicant.radio.instrument.calibration.CWCalibrationOperator"
                else class_name
            )
            for class_name, binding in MODEL_FORMULA_BINDINGS.items()
            if registration.name in binding.formulas
        ),
    )
    for registration in FORMULA_REGISTRATIONS
)


for selector, token in MODEL_DIMENSIONS:
    register_dimension(
        selector,
        domain="model_field",
        dimension=token,
        unit_policy="required" if selector in MODEL_REQUIRED_UNIT else "optional",
    )
for selector, (disposition, reason) in MODEL_SPECIAL.items():
    register_dimension(selector, domain="model_field", disposition=disposition, reason=reason)
for selector, token in RESOURCE_DIMENSIONS:
    register_dimension(selector, domain="resource_field", dimension=token)
for selector, (disposition, reason) in RESOURCE_SPECIAL.items():
    register_dimension(selector, domain="resource_field", disposition=disposition, reason=reason)
for selector, token in CONFIG_DIMENSIONS:
    register_dimension(
        selector,
        domain="config_path",
        dimension=token,
        unit_policy="required" if selector in CONFIG_REQUIRED_UNIT else "optional",
    )
for selector, resolver in CONFIG_CONTEXTUAL.items():
    register_dimension(
        selector,
        domain="config_path",
        disposition="contextual",
        resolver=resolver,
        unit_policy="inherited",
    )
for selector, (disposition, reason) in CONFIG_SPECIAL.items():
    register_dimension(selector, domain="config_path", disposition=disposition, reason=reason)
for selector, token in RESOURCE_OUTPUTS.items():
    register_dimension(selector, domain="config_path", dimension=token, unit_policy="inherited")
for registration in FORMULA_REGISTRATIONS:
    register_formula_checked(registration)
