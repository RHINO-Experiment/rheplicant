"""Destination-aware resolution validates A9 before value-form side effects."""

import jax.numpy as jnp
import pytest

from _rheplicant_bootstrap.types import DestinationDescriptor
from rheplicant.config.context import ResolutionContext
from rheplicant.config.dimensions import DimensionEnvironment, signature
from rheplicant.config.errors import ConfigError
from rheplicant.config.values import resolve_value


def test_required_destination_refuses_a_bare_scalar():
    destination = DestinationDescriptor(
        "model.adc.scale", "model_field", "model.adc.scale"
    )
    try:
        resolve_value(2.0, ResolutionContext(), destination=destination)
    except ConfigError as exc:
        assert "dimensions" in str(exc)
    else:
        raise AssertionError("A9-required destination accepted a bare scalar")


def _kelvin_destination() -> DestinationDescriptor:
    return DestinationDescriptor(
        "model.noise.sigma",
        "model_field",
        "rheplicant.radio.instrument.noise.NoiseOperator.sigma",
    )


def test_normal_operands_inherit_the_outer_unit():
    context = ResolutionContext(seed=1, seeds={"draw": 2})
    resolved = resolve_value(
        {
            "normal": {
                "shape": [],
                "seed": {"from": "runtime.seeds.draw"},
                "loc": 2,
                "scale": 1,
            },
            "unit": "K",
        },
        context,
        destination=_kelvin_destination(),
    )
    assert resolved.unit.canonical == "K"


def test_nested_explicit_unit_must_match_outer_authority():
    context = ResolutionContext(seed=1, seeds={"draw": 2})
    with pytest.raises(ConfigError, match="normal.loc"):
        resolve_value(
            {
                "normal": {
                    "shape": [],
                    "seed": {"from": "runtime.seeds.draw"},
                    "loc": {"value": 2, "unit": "Hz"},
                    "scale": 1,
                },
                "unit": "K",
            },
            context,
            destination=_kelvin_destination(),
        )


def test_stack_operands_inherit_the_outer_unit():
    resolved = resolve_value(
        {"stack": [1, 2], "unit": "K"},
        ResolutionContext(),
        destination=_kelvin_destination(),
    )
    assert resolved.unit.canonical == "K"
    assert jnp.array_equal(resolved.value, jnp.asarray([1.0, 2.0]))


def test_full_retains_the_outer_destination_unit():
    resolved = resolve_value(
        {"full": {"shape": [2], "value": 3.0}, "unit": "K"},
        ResolutionContext(),
        destination=_kelvin_destination(),
    )
    assert resolved.unit.canonical == "K"
    assert jnp.array_equal(resolved.value, jnp.asarray([3.0, 3.0]))


def test_uniform_operands_inherit_outer_unit_and_refuse_a_mismatch():
    context = ResolutionContext(seed=1, seeds={"draw": 2})
    resolved = resolve_value(
        {
            "uniform": {
                "shape": [],
                "seed": {"from": "runtime.seeds.draw"},
                "low": 1,
                "high": 2,
            },
            "unit": "K",
        },
        context,
        destination=_kelvin_destination(),
    )
    assert resolved.unit.canonical == "K"
    with pytest.raises(ConfigError, match="uniform.high"):
        resolve_value(
            {
                "uniform": {
                    "shape": [],
                    "seed": {"from": "runtime.seeds.draw"},
                    "low": 1,
                    "high": {"value": 2, "unit": "Hz"},
                },
                "unit": "K",
            },
            context,
            destination=_kelvin_destination(),
        )


def test_fixed_derivation_keeps_its_destination_dimension():
    destination = DestinationDescriptor(
        "observation.freq.grid", "config_path", "observation.freq.grid"
    )
    context = ResolutionContext(freq=jnp.asarray([1.0, 3.0, 5.0]))
    resolved = resolve_value(
        {"from": "channel_spacing"}, context, destination=destination
    )
    assert resolved.unit.canonical == "Hz"


def test_nested_run_target_has_concrete_path_and_patterned_selector():
    from rheplicant.config.values import make_resolution_target

    destination = DestinationDescriptor(
        "runs[3].at.g", "config_path", "runs[].at.*"
    )
    context = ResolutionContext(
        dimensions=DimensionEnvironment(latent_dimensions={"g": signature("K")})
    )
    target = make_resolution_target(2.0, destination, context.dimensions)
    assert target.destination.document_path == "runs[3].at.g"
    assert target.destination.selector == "runs[].at.*"
    assert target.expected == signature("K")


def test_required_unit_refuses_before_a_file_path_is_opened(tmp_path):
    missing = tmp_path / "must-not-open.npy"
    required = DestinationDescriptor(
        "model.adc.scale",
        "model_field",
        "rheplicant.radio.instrument.adc.ADCOperator.scale",
    )
    with pytest.raises(ConfigError, match="requires unit"):
        resolve_value(
            {"file": {"path": str(missing), "format": "npy"}},
            ResolutionContext(),
            destination=required,
        )


def test_reference_dimension_is_checked_against_its_destination():
    source = jnp.asarray([1.0])
    context = ResolutionContext(
        resources={"resources.sky_models.sky.maps": source},
        dimensions=DimensionEnvironment(
            resource_dimensions={"resources.sky_models.sky.maps": signature("K")}
        ),
    )
    destination = DestinationDescriptor(
        "observation.freq.grid", "config_path", "observation.freq.grid"
    )
    with pytest.raises(ConfigError, match="requires"):
        resolve_value(
            {"ref": "resources.sky_models.sky.maps"},
            context,
            destination=destination,
        )


def test_two_matching_references_preserve_identity_and_dimension_metadata():
    source = jnp.asarray([1.0])
    dimensions = DimensionEnvironment(
        resource_dimensions={"resources.sky_models.sky.maps": signature("K")}
    )
    context = ResolutionContext(
        resources={"resources.sky_models.sky.maps": source},
        dimensions=dimensions,
    )
    first = resolve_value(
        {"ref": "resources.sky_models.sky.maps"},
        context,
        destination=_kelvin_destination(),
    )
    second = resolve_value(
        {"ref": "resources.sky_models.sky.maps"},
        context,
        destination=_kelvin_destination(),
    )
    assert first.value is source and second.value is source
    assert first.unit.canonical == second.unit.canonical == "K"
