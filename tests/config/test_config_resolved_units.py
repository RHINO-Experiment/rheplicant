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
