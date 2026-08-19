"""A9 checks declarations before construction or I/O."""

import dataclasses
from collections import Counter

import equinox as eqx

from rheplicant.config import dimensions as dimension_module
from rheplicant.config.dimensions import (
    DimensionSpec,
    register_dimension,
    register_dimension_formula,
    signature,
)
from rheplicant.config.orchestration import prepare_document
from rheplicant.config.preflight import CHECKS, preflight
from rheplicant.core.operator import AbstractOperator
from tests.config.preflight_helpers import BASE_MODEL, preflight_document


@dataclasses.dataclass
class _PluginOperator:
    graph_node = "plugin"
    level: float = 1.0


class _PythonPluginOperator(AbstractOperator):
    graph_node = "gain"
    level: float = eqx.field(static=True, default=1.0)

    def __call__(self, state):
        return state


def _build_plugin_resource(name, spec, context):
    del name, context
    return spec["level"]


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


def test_plugin_defaulted_field_is_checked_even_when_omitted(monkeypatch):
    from rheplicant.config.preflight import dimensions as check

    qualified = f"{_PluginOperator.__module__}.{_PluginOperator.__qualname__}"
    saved_dimensions = dict(dimension_module._DIMENSION_REGISTRY)
    saved_formulas = dict(dimension_module._FORMULA_REGISTRY)
    monkeypatch.setattr(check, "operator_table", lambda: {"plugin": (_PluginOperator,)})
    try:
        register_dimension_formula(
            "plugin_output",
            rule="fixed",
            result=DimensionSpec("fixed", signature("K")),
            operands=(),
            producers=(qualified,),
        )
        findings = tuple(check._dimensions({"model": {"plugin": {}}}))
        assert [finding.where for finding in findings] == ["model.plugin.level"]
    finally:
        dimension_module._DIMENSION_REGISTRY.clear()
        dimension_module._DIMENSION_REGISTRY.update(saved_dimensions)
        dimension_module._FORMULA_REGISTRY.clear()
        dimension_module._FORMULA_REGISTRY.update(saved_formulas)


def test_compound_plugin_signature_is_compared_without_string_round_trip(monkeypatch):
    from rheplicant.config.preflight import dimensions as check

    qualified = f"{_PluginOperator.__module__}.{_PluginOperator.__qualname__}"
    saved_dimensions = dict(dimension_module._DIMENSION_REGISTRY)
    saved_formulas = dict(dimension_module._FORMULA_REGISTRY)
    monkeypatch.setattr(check, "operator_table", lambda: {"plugin": (_PluginOperator,)})
    try:
        register_dimension(
            f"{qualified}.level", domain="model_field", dimension="K/s"
        )
        register_dimension_formula(
            "compound_plugin_output",
            rule="fixed",
            result=DimensionSpec("fixed", signature("K/s")),
            operands=(),
            producers=(qualified,),
        )
        document = {
            "model": {"plugin": {"level": {"value": 1.0, "unit": "K/s"}}}
        }
        assert not tuple(check._dimensions(document))

        document["model"]["plugin"]["level"]["unit"] = "K"
        (finding,) = tuple(check._dimensions(document))
        assert finding.where == "model.plugin.level"
        assert "destination requires" in finding.message
    finally:
        dimension_module._DIMENSION_REGISTRY.clear()
        dimension_module._DIMENSION_REGISTRY.update(saved_dimensions)
        dimension_module._FORMULA_REGISTRY.clear()
        dimension_module._FORMULA_REGISTRY.update(saved_formulas)


def test_python_plugin_omitted_default_field_runs_completeness():
    from rheplicant.config.preflight import dimensions as check

    qualified = (
        f"{_PythonPluginOperator.__module__}.{_PythonPluginOperator.__qualname__}"
    )
    saved_formulas = dict(dimension_module._FORMULA_REGISTRY)
    try:
        register_dimension_formula(
            "python_plugin_output",
            rule="fixed",
            result=DimensionSpec("fixed", signature("K")),
            operands=(),
            producers=(qualified,),
        )
        document = {
            "model": {
                "gain": {
                    "python": (
                        f"{_PythonPluginOperator.__module__}:"
                        f"{_PythonPluginOperator.__qualname__}"
                    )
                }
            }
        }
        findings = tuple(check._dimensions(document))
        assert [finding.where for finding in findings] == ["model.gain.level"]
    finally:
        dimension_module._FORMULA_REGISTRY.clear()
        dimension_module._FORMULA_REGISTRY.update(saved_formulas)


def test_a9_resolves_config_destinations_from_the_live_registry():
    from rheplicant.config.preflight import dimensions as check

    saved = dict(dimension_module._DIMENSION_REGISTRY)
    try:
        register_dimension(
            "observation.site.*", domain="config_path", dimension="Hz"
        )
        document = {
            "observation": {
                "site": {"lat_deg": {"value": 51.0, "unit": "deg"}}
            }
        }
        findings = tuple(check._dimensions(document))
        assert any(
            finding.where == "observation.site.lat_deg"
            and "ambiguous dimension selectors" in finding.message
            for finding in findings
        )
    finally:
        dimension_module._DIMENSION_REGISTRY.clear()
        dimension_module._DIMENSION_REGISTRY.update(saved)


def test_multilayer_preflight_evaluates_each_selector_match_once(monkeypatch):
    document = preflight_document()
    document["variants"] = {
        "higher_gain": {
            "model": {"gain": {"gain": {"value": 2.0, "unit": "dimensionless"}}}
        },
        "lower_gain": {
            "model": {"gain": {"gain": {"value": 0.5, "unit": "dimensionless"}}}
        },
    }
    calls = Counter()
    selector_matches = dimension_module._selector_matches

    def counted(pattern, actual):
        calls[(pattern, actual)] += 1
        return selector_matches(pattern, actual)

    monkeypatch.setattr(dimension_module, "_selector_matches", counted)
    assert not _a9(preflight(document))
    assert calls
    assert max(calls.values()) == 1

    saved = dict(dimension_module._DIMENSION_REGISTRY)
    try:
        register_dimension("observation.freq.*", domain="config_path", dimension="Hz")
        assert any(
            finding.where.endswith("observation.freq.grid")
            and "ambiguous dimension selectors" in finding.message
            for finding in _a9(preflight(document))
        )
    finally:
        dimension_module._DIMENSION_REGISTRY.clear()
        dimension_module._DIMENSION_REGISTRY.update(saved)


def test_a9_resolves_plugin_resource_fields_from_the_live_registry():
    from rheplicant.config import resources
    from rheplicant.config.preflight import dimensions as check

    qualified = (
        f"{_build_plugin_resource.__module__}."
        f"{_build_plugin_resource.__qualname__}.level"
    )
    saved_dimensions = dict(dimension_module._DIMENSION_REGISTRY)
    saved_kinds = dict(resources._KINDS)
    try:
        resources.register_kind("plugin_dimension")(_build_plugin_resource)
        register_dimension(qualified, domain="resource_field", dimension="Hz")
        findings = tuple(
            check._dimensions(
                {
                    "resources": {
                        "plugin_dimension": {
                            "p": {"level": {"value": 1.0, "unit": "K"}}
                        }
                    }
                }
            )
        )
        assert any(
            finding.where == "resources.plugin_dimension.p.level"
            for finding in findings
        )
    finally:
        resources._KINDS.clear()
        resources._KINDS.update(saved_kinds)
        dimension_module._DIMENSION_REGISTRY.clear()
        dimension_module._DIMENSION_REGISTRY.update(saved_dimensions)


def test_latent_declaration_unit_is_the_authority_over_prior_operands():
    document = preflight_document()
    document["inference"]["parameters"]["g"] = {
        "init": 1.0,
        "unit": "K",
        "prior": {
            "normal": {
                "loc": {"value": 1.0, "unit": "Hz"},
                "scale": {"value": 1.0, "unit": "K"},
            }
        },
        "into": "gain.gain",
    }
    findings = _a9(preflight(document))
    assert any(
        finding.where == "inference.parameters.g.prior.normal.loc"
        for finding in findings
    )


def test_latent_declaration_prior_and_binding_must_agree():
    document = preflight_document()
    document["inference"]["parameters"]["g"] = {
        "init": {"value": 1.0, "unit": "K"},
        "unit": "K",
        "prior": {
            "normal": {
                "loc": {"value": 1.0, "unit": "K"},
                "scale": {"value": 1.0, "unit": "K"},
            }
        },
        "into": "gain.gain",
    }
    findings = _a9(preflight(document))
    assert any(
        finding.where == "inference.parameters.g"
        and "conflicting dimension evidence" in finding.message
        for finding in findings
    )

    document["inference"]["parameters"]["g"]["into"] = "global_signal.depth"
    assert not _a9(preflight(document))


def test_prior_only_contextual_nodes_infer_the_latent_dimension():
    environment = dimension_module.dimension_environment_for(
        {
            "inference": {
                "parameters": {
                    "x": {
                        "init": 1.0,
                        "prior": {
                            "normal": {
                                "loc": {"value": 1.0, "unit": "K"},
                                "scale": {"value": 2.0, "unit": "K"},
                            }
                        },
                    }
                }
            }
        }
    )
    assert environment.latent_dimensions == {"x": signature("K")}


def test_malformed_variant_unit_reaches_attributed_preflight_instead_of_inference():
    document = preflight_document()
    document["variants"] = {
        "bad": {
            "inference": {
                "parameters": {
                    "g": {
                        "init": {"value": 1.0, "unit": "not_a_unit"},
                    }
                }
            }
        }
    }
    try:
        prepare_document(document, scope="all_layers")
    except Exception as error:
        assert error.report is not None
        assert any(
            finding.check == "A9"
            and finding.where == "variants.bad.inference.parameters.g.init"
            and "Unknown unit 'not_a_unit'" in finding.message
            for finding in error.report.findings
        )
    else:  # pragma: no cover - the malformed unit must refuse
        raise AssertionError("malformed variant unit unexpectedly passed")


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
