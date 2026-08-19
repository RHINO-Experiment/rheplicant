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
    dimension_environment_for,
    dimension_for,
    divide,
    evaluate_formula,
    multiply,
    power,
    register_dimension,
    register_dimension_formula,
    signature,
    signature_label,
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


def test_dimensionless_signature_has_the_canonical_human_label():
    assert signature_label(signature("dimensionless")) == "dimensionless"


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


def test_direct_signature_construction_is_canonical_and_validated():
    assert DimensionSignature(
        (("time", 1), ("frequency", 0), ("time", -1), ("angle", 2)),
        (("count", 2), ("count", -1), ("samples", 0)),
    ) == DimensionSignature((("angle", 2),), (("count", 1),))
    with pytest.raises(ConfigError, match="integer exponent"):
        DimensionSignature((("temperature", True),))
    with pytest.raises(ConfigError, match="integer exponent"):
        DimensionSignature((("temperature", 1.5),))


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


def test_same_formulas_cover_outer_resource_and_sequence_roles():
    kelvin = signature("K")
    assert evaluate_formula("normal", {"loc": kelvin, "scale": kelvin}, result=kelvin) == kelvin
    assert evaluate_formula("uniform", {"low": kelvin, "high": kelvin}, result=kelvin) == kelvin
    assert evaluate_formula("stack", {"entry[]": (kelvin, kelvin)}, result=kelvin) == kelvin
    assert evaluate_formula("ref", {"source": kelvin}, result=kelvin) == kelvin
    assert evaluate_formula("file", {"snapshot": None}, result=kelvin) == kelvin
    with pytest.raises(ConfigError, match="shared dimension"):
        evaluate_formula("ref", {"source": signature("Hz")}, result=kelvin)


def test_affine_matrix_basis_and_exp_log_formulas_are_discriminating():
    kelvin = signature("K")
    unitless = signature("dimensionless")
    assert evaluate_formula(
        "modifier_affine",
        {"value": kelvin, "scale": unitless, "offset": kelvin},
        result=kelvin,
    ) == kelvin
    assert evaluate_formula(
        "matmul", {"design": unitless, "coefficient": kelvin}, result=kelvin
    ) == kelvin
    assert evaluate_formula(
        "basis_expand", {"basis": unitless, "coefficient": kelvin}, result=kelvin
    ) == kelvin
    assert evaluate_formula("exp_log", {"value": unitless}) == unitless
    with pytest.raises(ConfigError, match="role 'scale'"):
        evaluate_formula(
            "transform_affine",
            {"value": kelvin, "scale": signature("Hz"), "offset": kelvin},
            result=kelvin,
        )


def test_fixed_cw_and_radiometer_formulas_assert_every_operand_and_result():
    assert evaluate_formula(
        "cw_centre",
        {
            "tone_freq": signature("Hz"),
            "drift_rate": signature("Hz/s"),
            "time": signature("s"),
        },
    ) == signature("Hz")
    assert evaluate_formula(
        "radiometer_fraction",
        {"channel_width": signature("Hz"), "integration_time": signature("s")},
    ) == signature("dimensionless")
    with pytest.raises(ConfigError, match="expected"):
        evaluate_formula(
            "cw_centre",
            {
                "tone_freq": signature("Hz"),
                "drift_rate": signature("Hz"),
                "time": signature("s"),
            },
        )
    with pytest.raises(ConfigError, match="result"):
        evaluate_formula(
            "radiometer_fraction",
            {"channel_width": signature("Hz"), "integration_time": signature("s")},
            result=signature("K"),
        )


def test_rule_specific_registration_and_producer_role_conflicts_are_refused(clean_registry):
    fixed_k = DimensionSpec("fixed", signature("K"), unit_policy="inherited")
    fixed_hz = DimensionSpec("fixed", signature("Hz"), unit_policy="inherited")
    fixed_s = DimensionSpec("fixed", signature("s"), unit_policy="inherited")
    outer = DimensionSpec("contextual", resolver="outer", unit_policy="inherited")
    with pytest.raises(ConfigError, match="radiometer.*dimensionless"):
        register_dimension_formula(
            "bad_radiometer_result",
            rule="radiometer",
            result=fixed_k,
            operands=(
                FormulaOperand("channel_width", fixed_hz),
                FormulaOperand("integration_time", fixed_s),
            ),
        )
    with pytest.raises(ConfigError, match="fixed rule.*fixed result"):
        register_dimension_formula("bad_fixed_result", rule="fixed", result=outer, operands=())
    with pytest.raises(ConfigError, match="affine"):
        register_dimension_formula(
            "bad_affine_shape",
            rule="affine",
            result=outer,
            operands=(FormulaOperand("value", outer),),
        )
    register_dimension_formula(
        "producer_one",
        rule="fixed",
        result=fixed_k,
        operands=(FormulaOperand("coefficient", fixed_k),),
        producers=("example.Plugin",),
    )
    with pytest.raises(ConfigError, match="producer.*role"):
        register_dimension_formula(
            "producer_two",
            rule="fixed",
            result=fixed_k,
            operands=(FormulaOperand("coefficient", fixed_k),),
            producers=("example.Plugin",),
        )


def test_ordinary_product_formulas_refuse_discrete_quantity_multiplication(clean_registry):
    outer = DimensionSpec("contextual", resolver="outer", unit_policy="inherited")
    open_spec = DimensionSpec("open")
    register_dimension_formula(
        "illegal_quantity_product",
        rule="product",
        result=outer,
        operands=(FormulaOperand("left", open_spec), FormulaOperand("right", open_spec)),
    )
    with pytest.raises(ConfigError, match="quantity"):
        evaluate_formula(
            "illegal_quantity_product",
            {"left": signature("count"), "right": signature("samples")},
        )


def test_environment_uses_bindings_and_plugin_formula_outputs(monkeypatch, clean_registry):
    import rheplicant.config.dimensions as dimensions

    @dataclasses.dataclass
    class PluginOperator:
        graph_node = "adc"

    qualified = f"{PluginOperator.__module__}.{PluginOperator.__qualname__}"
    register_dimension_formula(
        "plugin_hz_output",
        rule="fixed",
        result=DimensionSpec("fixed", signature("Hz"), unit_policy="inherited"),
        operands=(),
        producers=(qualified,),
    )
    monkeypatch.setattr(dimensions, "operator_table", lambda: {"adc": (PluginOperator,)})
    environment = dimensions.dimension_environment_for(
        {
            "model": {"adc": {"type": "PluginOperator"}},
            "inference": {
                "parameters": {
                    "frequency": {
                        "init": 1.0,
                        "into": "adc.missing",
                        "unit": "Hz",
                    }
                }
            },
        }
    )
    assert environment.prediction_dimension == signature("Hz")
    assert environment.latent_dimensions == {"frequency": signature("Hz")}


def test_pipeline_environment_follows_stage_order_through_live_formulas():
    environment = dimension_environment_for(
        {
            "model": {
                "kind": "pipeline",
                "stages": [
                    {
                        "name": "sky",
                        "type": "SkyOperator",
                        "amplitude": {"value": 10.0, "unit": "K"},
                    },
                    {
                        "name": "adc",
                        "type": "ADCOperator",
                        "scale": {"value": 1.0, "unit": "adc_count/K"},
                        "n_bits": {"value": 12, "unit": "bits"},
                    },
                ],
            }
        }
    )

    assert environment.model_input_dimension == signature("K")
    assert environment.prediction_dimension == signature("adc_count")


def test_environment_infers_a_latent_from_its_live_model_binding():
    environment = __import__(
        "rheplicant.config.dimensions", fromlist=["dimension_environment_for"]
    ).dimension_environment_for(
        {
            "model": {
                "global_signal": {
                    "depth": {"value": 0.1, "unit": "K"},
                    "centre": {"value": 75.0, "unit": "MHz"},
                    "width": {"value": 10.0, "unit": "MHz"},
                }
            },
            "inference": {
                "parameters": {
                    "depth": {"init": 0.1, "into": "global_signal.depth"}
                }
            },
        }
    )
    assert environment.latent_dimensions["depth"] == signature("K")


def test_environment_applies_parameter_and_longhand_binding_transforms():
    environment = __import__(
        "rheplicant.config.dimensions", fromlist=["dimension_environment_for"]
    ).dimension_environment_for(
        {
            "model": {
                "global_signal": {
                    "depth": {"value": 0.1, "unit": "K"},
                    "centre": {"value": 75.0, "unit": "MHz"},
                    "width": {"value": 10.0, "unit": "MHz"},
                }
            },
            "inference": {
                "parameters": {
                    "log_depth": {"init": 0.0, "into": "global_signal.depth", "transform": "exp"},
                    "depth": {"init": 0.1},
                },
                "bindings": [
                    {"latents": ["depth"], "into": "global_signal.depth"}
                ],
            },
        }
    )
    assert environment.latent_dimensions == {
        "log_depth": signature("dimensionless"),
        "depth": signature("K"),
    }


def test_resource_dimensions_append_once_and_context_prefix_is_unchanged():
    environment = DimensionEnvironment()
    bind_resource_dimension(environment, "resources.arrays.x", signature("K"))
    assert environment.resource_dimensions == {"resources.arrays.x": signature("K")}
    with pytest.raises(ConfigError, match="bound more than once"):
        bind_resource_dimension(environment, "resources.arrays.x", signature("K"))
    fields = tuple(field.name for field in dataclasses.fields(ResolutionContext))
    assert fields[-5:] == (
        "dimensions",
        "layer",
        "trace",
        "origin_lookup",
        "capture",
    )


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


def test_real_resource_dag_binds_fixed_outputs_before_a_dependent_ref():
    import jax.numpy as jnp

    from rheplicant.config.resources import build_resources

    context = ResolutionContext(freq=jnp.arange(8.0), time=jnp.arange(6.0))
    built = build_resources(
        {
            "arrays": {"copy": {"ref": "resources.bases.design.time"}},
            "bases": {
                "design": {
                    "time": {"kind": "legendre", "n_basis": 2},
                    "freq": {"kind": "legendre", "n_basis": 3},
                }
            },
        },
        context,
    )
    assert built.order == ("resources.bases.design", "resources.arrays.copy")
    assert context.dimensions.resource_dimensions[
        "resources.bases.design.time"
    ] == signature("dimensionless")
    assert context.dimensions.resource_dimensions[
        "resources.arrays.copy"
    ] == signature("dimensionless")


def test_ref_distinguishes_an_absent_fixed_binding_from_bound_open_unknown():
    import dataclasses

    from rheplicant.config.document import load_document
    from rheplicant.config.values import resolve_value
    from tests.config.preflight_helpers import preflight_document

    production = load_document(preflight_document(resources=None)).context
    fixed = dataclasses.replace(
        production,
        resources={"resources.bases.b": SimpleNamespace(time=1.0)},
    )
    with pytest.raises(ConfigError, match="was not bound"):
        resolve_value({"ref": "resources.bases.b.time"}, fixed)

    open_unknown = dataclasses.replace(
        production, resources={"resources.arrays.a": 1.0}
    )
    bind_resource_dimension(
        open_unknown.dimensions, "resources.arrays.a", None
    )
    assert resolve_value({"ref": "resources.arrays.a"}, open_unknown).value == 1.0
