"""A9 checks declarations before construction or I/O."""

import dataclasses

from rheplicant.config import dimensions as dimension_module
from rheplicant.config.dimensions import (
    DimensionSpec,
    register_dimension,
    register_dimension_formula,
    signature,
)
from rheplicant.config.preflight import CHECKS, preflight
from tests.config.preflight_helpers import BASE_MODEL, preflight_document


@dataclasses.dataclass
class _PluginOperator:
    graph_node = "plugin"
    level: float = 1.0


def _a9(report):
    return tuple(finding for finding in report.findings if finding.check == "A9")


def test_adc_scale_refuses_kelvin_and_the_literal_remedy_passes():
    document = preflight_document(
        model={
            **BASE_MODEL,
            "adc": {
                "scale": {"value": 1.0, "unit": "K"},
                "n_bits": {"value": 12, "unit": "bits"},
            },
        }
    )
    document["inference"]["noise"]["sigma"]["unit"] = "adc_count"
    (finding,) = _a9(preflight(document))
    assert finding.where == "model.adc.scale"
    assert finding.message == (
        "model.adc.scale: unit 'K' has dimension temperature, but this "
        "destination requires adc_count/K. Use {value: 1.0, unit: "
        "adc_count/K} (check A9)."
    )
    document["model"]["adc"]["scale"]["unit"] = "adc_count/K"
    assert not _a9(preflight(document))


def test_required_unit_on_source_unstated_model_field_is_enforced():
    document = preflight_document(model={**BASE_MODEL, "gain": {"gain": 1.0}})
    assert any(finding.where == "model.gain.gain" for finding in _a9(preflight(document)))


def test_a9_is_registered_before_construction():
    assert "A9" in CHECKS


def test_only_the_two_plugin_registration_functions_are_public():
    import rheplicant.config as config

    assert config.register_dimension is register_dimension
    assert config.register_dimension_formula is register_dimension_formula


def test_plugin_field_and_formula_must_both_be_registered(monkeypatch):
    from rheplicant.config.preflight import dimensions as check

    qualified = f"{_PluginOperator.__module__}.{_PluginOperator.__qualname__}"
    saved_dimensions = dict(dimension_module._DIMENSION_REGISTRY)
    saved_formulas = dict(dimension_module._FORMULA_REGISTRY)
    monkeypatch.setattr(check, "operator_table", lambda: {"plugin": (_PluginOperator,)})
    document = {"model": {"plugin": {"level": {"value": 1.0, "unit": "K"}}}}
    try:
        before = tuple(check._dimensions(document))
        assert {finding.where for finding in before} == {
            "model.plugin",
            "model.plugin.level",
        }
        register_dimension(f"{qualified}.level", domain="model_field", dimension="K")
        register_dimension_formula(
            "plugin_output",
            rule="fixed",
            result=DimensionSpec("fixed", signature("K")),
            operands=(),
            producers=(qualified,),
        )
        assert not tuple(check._dimensions(document))
    finally:
        dimension_module._DIMENSION_REGISTRY.clear()
        dimension_module._DIMENSION_REGISTRY.update(saved_dimensions)
        dimension_module._FORMULA_REGISTRY.clear()
        dimension_module._FORMULA_REGISTRY.update(saved_formulas)


def test_sky_map_sibling_unit_is_validated_instead_of_ignored():
    from rheplicant.config.preflight import dimensions as check

    document = {
        "resources": {
            "sky_models": {
                "sky": {
                    "kind": "maps",
                    "maps": {"list": [[1.0]]},
                    "freq": {"list": [70.0], "unit": "MHz"},
                    "nside": 1,
                    "unit": "Hz",
                }
            }
        }
    }
    findings = tuple(check._dimensions(document))
    assert any(finding.where == "resources.sky_models.sky.maps" for finding in findings)
