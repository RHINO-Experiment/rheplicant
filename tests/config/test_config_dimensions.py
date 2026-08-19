"""Dimension algebra, selectors, formulas, and compatibility records."""

import dataclasses
from types import SimpleNamespace

import pytest

from _rheplicant_bootstrap.types import DestinationDescriptor
from rheplicant.config.context import ResolutionContext
from rheplicant.config.dimensions import (
    _DIMENSION_REGISTRY,
    _FORMULA_REGISTRY,
    DimensionEnvironment,
    DimensionSignature,
    DimensionSpec,
    FormulaOperand,
    bind_resource_dimension,
    dimension_for,
    divide,
    evaluate_formula,
    multiply,
    power,
    register_dimension,
    register_dimension_formula,
    signature,
)
from rheplicant.config.errors import ConfigError
from rheplicant.config.units import _ATOMS, ACCEPTED_UNITS, Unit


@pytest.fixture
def clean_registry():
    dimensions = dict(_DIMENSION_REGISTRY)
    formulas = dict(_FORMULA_REGISTRY)
    _DIMENSION_REGISTRY.clear()
    _FORMULA_REGISTRY.clear()
    try:
        yield
    finally:
        _DIMENSION_REGISTRY.clear()
        _DIMENSION_REGISTRY.update(dimensions)
        _FORMULA_REGISTRY.clear()
        _FORMULA_REGISTRY.update(formulas)


def test_dimensionless_is_the_physical_identity():
    identity = signature("dimensionless")
    kelvin = signature("K")
    assert multiply(identity, kelvin) == kelvin
    assert multiply(kelvin, identity) == kelvin


def test_discrete_meaning_is_orthogonal_to_physics():
    assert signature("count").physical == ()
    assert signature("count").quantity == (("count", 1),)
    assert signature("samples").physical == ()
    assert signature("samples") != signature("count")
    assert divide(signature("cycles"), signature("samples")) == signature("cycles/samples")


def test_adc_identity_is_checkable():
    assert multiply(signature("adc_count/K"), signature("K")) == signature("adc_count")


def test_signatures_sort_cancel_and_raise_to_integer_powers():
    assert signature("s*K") == DimensionSignature((("temperature", 1), ("time", 1)))
    assert divide(signature("K*s"), signature("s")) == signature("K")
    assert power(signature("Hz/s"), -2) == DimensionSignature((("frequency", -2), ("time", 2)))
    with pytest.raises(ConfigError, match="integer exponent"):
        power(signature("K"), 0.5)


def test_every_accepted_unit_spelling_has_a_signature_and_unit_stays_six_fields():
    assert len(ACCEPTED_UNITS) == 14
    assert len(_ATOMS) == 20
    for spelling in _ATOMS:
        assert isinstance(signature(spelling), DimensionSignature)
    assert Unit._fields == (
        "canonical",
        "factor",
        "offset",
        "numerator",
        "denominator",
        "dimension",
    )


@pytest.mark.parametrize("token", ["celsius/s", "K/celsius", "celsius*K"])
def test_affine_atoms_are_only_bare_units(token):
    with pytest.raises(ConfigError, match="affine"):
        signature(token)


def test_selector_lookup_requires_exactly_one_match(clean_registry):
    register_dimension("model.*.scale", domain="config_path", dimension="dimensionless")
    register_dimension("model.adc.scale", domain="config_path", dimension="adc_count/K")
    with pytest.raises(ConfigError, match="ambiguous dimension selectors"):
        dimension_for(DestinationDescriptor("model.adc.scale", "config_path", "model.adc.scale"))


@pytest.mark.parametrize("selector", ["", ".x", "x.", "x..y", "x[", "x[*]", "x.**"])
def test_selector_grammar_is_closed(selector, clean_registry):
    with pytest.raises(ConfigError) as caught:
        register_dimension(selector, domain="config_path", dimension="dimensionless")
    assert str(caught.value) == (
        f"dimensions: invalid config_path selector {selector!r}; use dotted "
        "identifiers with '*' for one mapping segment and '[]' for one list index."
    )


def test_star_and_list_selectors_each_match_exactly_one_segment(clean_registry):
    register_dimension("model.*.scale", domain="config_path", dimension="K")
    register_dimension("runs[].at.*", domain="config_path", dimension="count")
    assert dimension_for(
        DestinationDescriptor("model.adc.scale", "config_path", "model.adc.scale")
    ) == signature("K")
    assert dimension_for(
        DestinationDescriptor("runs[2].at.g", "config_path", "runs[].at.g")
    ) == signature("count")
    with pytest.raises(ConfigError, match="no dimension selector"):
        dimension_for(
            DestinationDescriptor("model.deep.adc.scale", "config_path", "model.deep.adc.scale")
        )


def test_duplicate_dimension_registration_is_refused(clean_registry):
    register_dimension("x", domain="config_path", dimension="K")
    with pytest.raises(ConfigError, match="registered twice"):
        register_dimension("x", domain="config_path", dimension="K")


def test_formula_registration_is_finite_named_and_role_checked(clean_registry):
    fixed = DimensionSpec("fixed", signature("K"), unit_policy="inherited")
    operand = FormulaOperand("input", fixed, -2)
    register_dimension_formula("square_inverse", rule="product", result=fixed, operands=(operand,))
    with pytest.raises(ConfigError, match="registered twice"):
        register_dimension_formula(
            "square_inverse", rule="product", result=fixed, operands=(operand,)
        )
    with pytest.raises(ConfigError, match="non-empty unique roles"):
        register_dimension_formula(
            "bad_roles",
            rule="fixed",
            result=fixed,
            operands=(FormulaOperand("", fixed),),
        )
    with pytest.raises(ConfigError, match="non-zero integer exponent"):
        register_dimension_formula(
            "bad_exponent",
            rule="fixed",
            result=fixed,
            operands=(FormulaOperand("x", fixed, 0),),
        )


def test_closed_formula_evaluator_checks_adc(clean_registry):
    def fixed(token):
        return DimensionSpec("fixed", signature(token), unit_policy="inherited")

    register_dimension_formula(
        "adc",
        rule="product",
        result=fixed("adc_count"),
        operands=(
            FormulaOperand("scale", fixed("adc_count/K")),
            FormulaOperand("input", fixed("K")),
        ),
    )
    assert evaluate_formula(
        "adc", {"scale": signature("adc_count/K"), "input": signature("K")}
    ) == signature("adc_count")
    with pytest.raises(ConfigError, match="role 'input'"):
        evaluate_formula("adc", {"scale": signature("adc_count/K"), "input": signature("Hz")})


def test_resource_dimensions_append_once_and_context_prefix_is_unchanged():
    environment = DimensionEnvironment()
    bind_resource_dimension(environment, "resources.arrays.x", signature("K"))
    assert environment.resource_dimensions == {"resources.arrays.x": signature("K")}
    with pytest.raises(ConfigError, match="bound more than once"):
        bind_resource_dimension(environment, "resources.arrays.x", signature("K"))
    fields = tuple(field.name for field in dataclasses.fields(ResolutionContext))
    assert fields[-3:] == ("dimensions", "layer", "trace")


def test_array_source_signature_survives_a_reference():
    from rheplicant.config.kinds.arrays import build_array
    from rheplicant.config.values import resolve_value

    context = ResolutionContext()
    value = build_array(
        "resources.arrays.temperature",
        {"value": 280.0, "unit": "K"},
        context,
    )
    context = context.with_resource("resources.arrays.temperature", value)
    resolved = resolve_value({"ref": "resources.arrays.temperature"}, context)
    assert resolved.unit is not None
    assert resolved.unit.canonical == "K"


def test_fixed_resource_outputs_bind_after_success():
    context = ResolutionContext()
    built = SimpleNamespace(time=object(), freq=object())
    context = context.with_resource("resources.bases.design", built)
    assert context.dimensions.resource_dimensions == {
        "resources.bases.design.time": signature("dimensionless"),
        "resources.bases.design.freq": signature("dimensionless"),
    }
