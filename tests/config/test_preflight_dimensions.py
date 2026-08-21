"""A9 checks declarations before construction or I/O."""

import dataclasses
import sys
from collections import Counter

import equinox as eqx
import pytest

from rheplicant.config import dimensions as dimension_module
from rheplicant.config.dimensions import (
    DimensionSpec,
    FormulaOperand,
    register_dimension,
    register_dimension_formula,
    signature,
)
from rheplicant.config.orchestration import prepare_document
from rheplicant.config.preflight import CHECKS, preflight
from rheplicant.core.operator import AbstractOperator
from tests.config.preflight_helpers import BASE_MODEL, BASE_OBSERVATION, preflight_document


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


def test_many_filter_field_runs_required_unit_validation_at_its_actual_path():
    document = preflight_document(
        model={
            **BASE_MODEL,
            "filters": [
                {
                    "type": "SkySpaceFilter",
                    "projector": {"ref": "resources.projectors.p"},
                    "regularization": 1e-3,
                }
            ],
        }
    )

    findings = _a9(preflight(document))

    assert any(
        finding.where == "model.filters[0].regularization"
        and "model.filters[0].regularization" in finding.message
        and "requires an explicit unit declaring dimensionless" in finding.message
        for finding in findings
    )
    document["model"]["filters"][0]["regularization"] = {
        "value": 1e-3,
        "unit": "dimensionless",
    }
    assert not _a9(preflight(document))


def test_many_fan_field_runs_required_unit_validation_at_its_actual_path():
    document = preflight_document(
        observation={
            **BASE_OBSERVATION,
            "switching": {"mode": "cycle", "order": ["antenna", "ambient"]},
        },
        model={
            **BASE_MODEL,
            "cal_loads": {
                "ambient": {"t_load": {"value": 300.0, "unit": "Hz"}}
            },
        },
    )

    findings = _a9(preflight(document))

    assert any(
        finding.where == "model.cal_loads.ambient.t_load"
        and "model.cal_loads.ambient.t_load" in finding.message
        for finding in findings
    )
    document["model"]["cal_loads"]["ambient"]["t_load"] = {
        "value": 300.0,
        "unit": "K",
    }
    assert not _a9(preflight(document))


@pytest.mark.parametrize(
    "scale",
    [1.0, {"value": 1.0, "unit": "K"}],
    ids=["missing-unit", "wrong-unit"],
)
def test_compose_stage_field_runs_required_unit_validation_at_its_actual_path(scale):
    document = preflight_document(
        model={
            **BASE_MODEL,
            "adc": {
                "compose": "cascade",
                "stages": [
                    {
                        "name": "first",
                        "type": "ADCOperator",
                        "scale": scale,
                        "n_bits": {"value": 12, "unit": "bits"},
                    },
                    {
                        "name": "second",
                        "type": "ADCOperator",
                        "scale": {"value": 1.0, "unit": "adc_count/K"},
                        "n_bits": {"value": 12, "unit": "bits"},
                    },
                ],
            },
        }
    )
    document["inference"]["noise"]["sigma"]["unit"] = "adc_count"

    findings = _a9(preflight(document))

    assert any(
        finding.where == "model.adc.stages[0].scale"
        and "model.adc.stages[0].scale" in finding.message
        for finding in findings
    )
    document["model"]["adc"]["stages"][0]["scale"] = {
        "value": 1.0,
        "unit": "adc_count/K",
    }
    assert not _a9(preflight(document))


def test_pipeline_stage_field_runs_required_unit_validation_at_its_actual_path():
    document = preflight_document(inference=None)
    document["model"] = {
        "kind": "pipeline",
        "stages": [
            {
                "name": "sky",
                "type": "SkyOperator",
                "amplitude": {"value": 10.0, "unit": "K"},
            },
            {"name": "gain", "type": "GainOperator", "gain": 1.1},
        ],
    }

    findings = _a9(preflight(document))

    assert any(
        finding.where == "model.stages[1].gain"
        and "model.stages[1].gain" in finding.message
        for finding in findings
    )
    document["model"]["stages"][1]["gain"] = {
        "value": 1.1,
        "unit": "dimensionless",
    }
    assert not _a9(preflight(document))


def test_twin_replacement_field_runs_required_unit_validation_at_its_actual_path():
    document = preflight_document()
    document["inference"]["twin"]["replace"] = {"gain": {"gain": 1.0}}

    findings = _a9(preflight(document))

    assert any(
        finding.where == "inference.twin.replace.gain.gain"
        and "inference.twin.replace.gain.gain" in finding.message
        for finding in findings
    )
    document["inference"]["twin"]["replace"]["gain"]["gain"] = {
        "value": 1.0,
        "unit": "dimensionless",
    }
    assert not _a9(preflight(document))


def test_nested_operator_formula_is_resolved_from_the_live_registry():
    from rheplicant.config.preflight import dimensions as check

    saved = dict(dimension_module._FORMULA_REGISTRY)
    dimension_module._FORMULA_REGISTRY.pop("sky_space")
    document = {
        "model": {
            "filters": [
                {
                    "type": "SkySpaceFilter",
                    "projector": {"ref": "resources.projectors.p"},
                    "regularization": {
                        "value": 1e-3,
                        "unit": "dimensionless",
                    },
                }
            ]
        }
    }
    try:
        findings = tuple(check._dimensions(document))
        assert any(
            finding.where == "model.filters[0]"
            and "has no registered dimension formula" in finding.message
            for finding in findings
        )
    finally:
        dimension_module._FORMULA_REGISTRY.clear()
        dimension_module._FORMULA_REGISTRY.update(saved)


def test_non_output_validation_formula_is_also_resolved_live():
    from rheplicant.config.preflight import dimensions as check

    saved = dict(dimension_module._FORMULA_REGISTRY)
    dimension_module._FORMULA_REGISTRY.pop("radiometer_fraction")
    document = {
        "model": {
            "noise": {
                "type": "RadiometerNoiseOperator",
                "channel_width": {"value": 1.0, "unit": "Hz"},
                "integration_time": {"value": 1.0, "unit": "s"},
            }
        }
    }
    try:
        findings = tuple(check._dimensions(document))
        assert any(
            finding.where == "model.noise"
            and "has no registered dimension formula" in finding.message
            for finding in findings
        )
    finally:
        dimension_module._FORMULA_REGISTRY.clear()
        dimension_module._FORMULA_REGISTRY.update(saved)


@pytest.mark.parametrize("route", ["graph", "many", "pipeline", "twin.replace"])
def test_non_operator_python_target_stands_down_without_crashing_a9(route):
    from rheplicant.config.context import ResolutionContext
    from rheplicant.config.errors import ConfigError
    from rheplicant.config.sections.model import build_node_operator

    target = {"python": "builtins:dict"}
    if route == "graph":
        document = preflight_document(
            model={**BASE_MODEL, "gain": target}
        )
    elif route == "many":
        document = preflight_document(
            model={**BASE_MODEL, "filters": [target]}
        )
    elif route == "pipeline":
        document = preflight_document(inference=None)
        document["model"] = {
            "kind": "pipeline",
            "stages": [{"name": "invalid", **target}],
        }
    else:
        document = preflight_document()
        document["inference"]["twin"]["replace"] = {"gain": target}

    report = preflight(document)

    assert not _a9(report)
    with pytest.raises(ConfigError, match="not an AbstractOperator subclass"):
        build_node_operator("gain", target, ResolutionContext())


def test_a9_never_imports_an_unloaded_python_target(monkeypatch):
    from rheplicant.config import hatch
    from rheplicant.config.preflight import dimensions as check

    module_name = "rheplicant_task11_unloaded_plugin"
    assert module_name not in sys.modules
    live_import = hatch.importlib.import_module

    def guarded_import(name):
        if name == module_name:
            raise AssertionError("A9 evaluated an unloaded python: target")
        return live_import(name)

    monkeypatch.setattr(hatch.importlib, "import_module", guarded_import)
    document = {
        "model": {
            "gain": {"python": f"{module_name}:PluginOperator"},
        }
    }

    assert not tuple(check._dimensions(document))
    assert module_name not in sys.modules


@pytest.mark.parametrize(
    ("model", "owner_pattern"),
    [
        pytest.param(
            {
                "kind": "pipeline",
                "stages": [
                    {
                        "name": "sky",
                        "type": "SkyOperator",
                        "amplitude": {"value": 10.0, "unit": "K"},
                    }
                ],
            },
            "repairs a graph assembly",
            id="pipeline",
        ),
        pytest.param(
            {"kind": "not-a-model-kind"},
            "kind: is 'graph' .* or 'pipeline'",
            id="unknown-kind",
        ),
        pytest.param([], "model: is a mapping", id="non-mapping"),
    ],
)
def test_twin_replacement_stands_down_until_a_graph_model_builds(
    model, owner_pattern
):
    from rheplicant.config.context import ResolutionContext
    from rheplicant.config.errors import ConfigError
    from rheplicant.config.sections.compose import build_model
    from rheplicant.config.sections.twin import build_fit_twin

    twin_spec = {"replace": {"gain": {"gain": 1.0}}}
    document = preflight_document(inference={"twin": twin_spec})
    document["model"] = model

    report = preflight(document)

    assert not any(
        finding.where.startswith("inference.twin.replace")
        for finding in _a9(report)
    )
    with pytest.raises(ConfigError, match=owner_pattern):
        twin = build_model(model, ResolutionContext(), switch_order=())
        build_fit_twin(twin_spec, twin, ResolutionContext())


def test_a9_python_selection_uses_the_actual_loaded_module_attribute(monkeypatch):
    from types import ModuleType

    from rheplicant.config.preflight import dimensions as check

    module_name = "rheplicant_task11_shadowed_plugin"
    loaded_module = ModuleType(module_name)
    loaded_module._PythonPluginOperator = dict
    monkeypatch.setitem(sys.modules, module_name, loaded_module)
    monkeypatch.setattr(_PythonPluginOperator, "__module__", module_name)
    monkeypatch.setattr(
        check,
        "operator_table",
        lambda: {"gain": (_PythonPluginOperator,)},
    )
    document = {
        "model": {
            "gain": {
                "python": f"{module_name}:_PythonPluginOperator",
                "level": {"value": 1.0, "unit": "dimensionless"},
            }
        }
    }

    assert not tuple(check._dimensions(document))


@pytest.mark.parametrize("shape", ["pipeline", "compose"])
def test_stage_name_grammar_is_owned_by_the_model_builder(shape):
    from rheplicant.config.context import ResolutionContext
    from rheplicant.config.errors import ConfigError
    from rheplicant.config.sections.compose import build_model

    if shape == "pipeline":
        model = {
            "kind": "pipeline",
            "stages": [{"type": "GainOperator", "gain": 1.0}],
        }
        stage_prefix = "model.stages[0]"
    else:
        model = {
            "adc": {
                "compose": "cascade",
                "stages": [
                    {
                        "type": "ADCOperator",
                        "scale": 1.0,
                        "n_bits": {"value": 12, "unit": "bits"},
                    },
                    {
                        "name": "second",
                        "type": "ADCOperator",
                        "scale": {"value": 1.0, "unit": "adc_count/K"},
                        "n_bits": {"value": 12, "unit": "bits"},
                    },
                ],
            }
        }
        stage_prefix = "model.adc.stages[0]"
    document = preflight_document(inference=None)
    document["model"] = model

    report = preflight(document)

    assert not any(
        finding.where.startswith(stage_prefix) for finding in _a9(report)
    )
    with pytest.raises(ConfigError, match="mapping with a name"):
        build_model(model, ResolutionContext(), switch_order=())


def test_wrong_many_shape_is_owned_by_a6_before_dimension_fields():
    from rheplicant.config.context import ResolutionContext
    from rheplicant.config.errors import ConfigError
    from rheplicant.config.sections.compose import build_model

    model = {
        "filters": {
            "not_a_fan": {
                "type": "SkySpaceFilter",
                "projector": {"ref": "resources.projectors.p"},
                "regularization": 1e-3,
            }
        }
    }
    document = preflight_document(inference=None)
    document["model"] = model

    report = preflight(document)

    assert "A6" in report.checks()
    assert not any(
        finding.where.startswith("model.filters.not_a_fan")
        for finding in _a9(report)
    )
    with pytest.raises(ConfigError, match="non-empty list"):
        build_model(model, ResolutionContext(), switch_order=())


def test_pipeline_prediction_drives_contextual_noise_dimension():
    model = {
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
    document = preflight_document(
        inference={"noise": {"sigma": {"value": 0.1, "unit": "K"}}},
    )
    document["model"] = model

    findings = _a9(preflight(document))

    assert any(
        finding.where == "inference.noise.sigma"
        and "destination requires adc_count" in finding.message
        for finding in findings
    )
    document["inference"]["noise"]["sigma"]["unit"] = "adc_count"
    assert not _a9(preflight(document))


@pytest.mark.parametrize("sigma_unit", ["K", "adc_count"])
def test_invalid_compose_stands_down_model_context_until_its_builder_refuses(
    sigma_unit,
):
    from rheplicant.config.context import ResolutionContext
    from rheplicant.config.errors import ConfigError
    from rheplicant.config.sections.compose import build_model

    model = {
        **BASE_MODEL,
        "adc": {
            "compose": "cascade",
            "stages": [
                {
                    "type": "ADCOperator",
                    "scale": 1.0,
                    "n_bits": {"value": 12, "unit": "bits"},
                },
                {
                    "name": "second",
                    "type": "ADCOperator",
                    "scale": {"value": 1.0, "unit": "adc_count/K"},
                    "n_bits": {"value": 12, "unit": "bits"},
                },
            ],
        },
    }
    document = preflight_document(model=model)
    document["inference"]["noise"]["sigma"]["unit"] = sigma_unit
    document["inference"]["parameters"]["declared"] = {"unit": "Hz"}

    environment = dimension_module.dimension_environment_for(document)
    report = preflight(document)

    assert environment.model_input_dimension is None
    assert environment.prediction_dimension is None
    assert environment.latent_dimensions["declared"] == signature("Hz")
    assert not any(
        finding.where == "inference.noise.sigma" for finding in _a9(report)
    )
    with pytest.raises(ConfigError, match="every stage is a mapping with a name"):
        build_model(model, ResolutionContext(), switch_order=())


def test_model_a9_reads_the_live_operator_table_once_per_invocation(monkeypatch):
    from rheplicant.config.preflight import dimensions as check

    calls = 0
    live = check.operator_table

    def counted():
        nonlocal calls
        calls += 1
        return live()

    monkeypatch.setattr(check, "operator_table", counted)
    document = preflight_document()

    assert not tuple(check._dimensions(document))
    assert calls == 1
    assert not tuple(check._dimensions(document))
    assert calls == 2


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


def test_plugin_with_multiple_disjoint_formulas_refuses_ambiguous_output():
    from rheplicant.config.preflight import dimensions as check

    qualified = (
        f"{_PythonPluginOperator.__module__}.{_PythonPluginOperator.__qualname__}"
    )
    saved_dimensions = dict(dimension_module._DIMENSION_REGISTRY)
    saved_formulas = dict(dimension_module._FORMULA_REGISTRY)
    fixed_k = DimensionSpec("fixed", signature("K"), unit_policy="inherited")
    fixed_count = DimensionSpec(
        "fixed", signature("adc_count"), unit_policy="inherited"
    )
    try:
        register_dimension(
            f"{qualified}.level", domain="model_field", dimension="K"
        )
        register_dimension_formula(
            "ambiguous_plugin_signal",
            rule="fixed",
            result=fixed_k,
            operands=(FormulaOperand("signal", fixed_k),),
            producers=(qualified,),
        )
        register_dimension_formula(
            "ambiguous_plugin_auxiliary",
            rule="fixed",
            result=fixed_count,
            operands=(FormulaOperand("auxiliary", fixed_count),),
            producers=(qualified,),
        )
        document = {
            "model": {
                "gain": {
                    "python": (
                        f"{_PythonPluginOperator.__module__}:"
                        f"{_PythonPluginOperator.__qualname__}"
                    ),
                    "level": {"value": 1.0, "unit": "K"},
                }
            }
        }

        environment = dimension_module.dimension_environment_for(document)
        findings = tuple(check._dimensions(document))

        assert environment.prediction_dimension is None
        assert environment.model_input_dimension is None
        assert len(findings) == 1
        assert findings[0].where == "model.gain"
        assert "ambiguous output formula" in findings[0].message
        assert qualified in findings[0].message
        assert "producer must uniquely identify its output formula" in findings[0].message
    finally:
        dimension_module._DIMENSION_REGISTRY.clear()
        dimension_module._DIMENSION_REGISTRY.update(saved_dimensions)
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


def test_conflicting_hyphenated_latent_uses_a_legal_finding_path():
    document = preflight_document()
    document["inference"]["parameters"] = {
        "d-1": {
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
    }

    findings = _a9(preflight(document))

    assert any(
        finding.where == "inference.parameters"
        and "inference.parameters.d-1" in finding.message
        and "conflicting dimension evidence" in finding.message
        for finding in findings
    )


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
