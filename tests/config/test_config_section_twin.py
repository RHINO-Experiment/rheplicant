"""inference.twin: without/replace -> the repaired fit twin."""

import pytest

from rheplicant.config import ConfigError
from rheplicant.config.sections.twin import build_fit_twin
from rheplicant.core.errors import AssemblyError
from tests.config.inference_helpers import NOISY_MODEL, context, twin


class TestWithout:
    def test_none_returns_the_twin_itself(self):
        model = twin()
        repaired, replaced = build_fit_twin(None, model, context())
        assert repaired is model
        assert replaced == ()

    def test_without_drops_the_noise_node(self):
        repaired, _ = build_fit_twin({"without": ["noise"]},
                                     twin(NOISY_MODEL), context())
        assert "noise" not in repaired.lit
        assert "gain" in repaired.lit

    def test_without_an_absent_node_is_the_assemblys_own_refusal(self):
        with pytest.raises(AssemblyError, match="no operator sits at"):
            build_fit_twin({"without": ["rfi_field"]}, twin(), context())

    def test_without_takes_a_list_of_node_ids(self):
        with pytest.raises(ConfigError, match="list of node ids"):
            build_fit_twin({"without": "noise"}, twin(NOISY_MODEL), context())

    def test_without_refuses_non_string_entries(self):
        with pytest.raises(ConfigError, match="list of node ids"):
            build_fit_twin({"without": [123]}, twin(NOISY_MODEL), context())


class TestReplace:
    def test_replace_swaps_the_gain_operator(self):
        section = {"replace": {"gain": {"gain": {"value": 1.0,
                                                 "unit": "dimensionless"}}}}
        repaired, replaced = build_fit_twin(section, twin(), context())
        assert float(repaired["gain"].gain) == pytest.approx(1.0)
        assert replaced == ("gain",)

    def test_replace_spec_goes_through_the_model_sections_builder(self):
        section = {"replace": {"gain": {"gain": {"value": 1.0,
                                                 "unit": "dimensionless"},
                                        "typo": 1}}}
        with pytest.raises(ConfigError):
            build_fit_twin(section, twin(), context())

    def test_replace_must_be_a_mapping(self):
        with pytest.raises(ConfigError, match="node id"):
            build_fit_twin({"replace": ["gain"]}, twin(), context())

    def test_without_applies_before_replace(self):
        section = {"without": ["noise"],
                   "replace": {"gain": {"gain": {"value": 2.0,
                                                 "unit": "dimensionless"}}}}
        repaired, _ = build_fit_twin(section, twin(NOISY_MODEL), context())
        assert "noise" not in repaired.lit
        assert float(repaired["gain"].gain) == pytest.approx(2.0)

    def test_replacing_a_node_without_just_dropped_is_refused(self):
        # Pins the order: without runs first, so the replace lands on an
        # assembly that no longer has the node. Replace-first would let
        # this contradictory document pass silently.
        section = {"without": ["noise"],
                   "replace": {"noise": {"type": "NoiseOperator",
                                         "sigma": {"value": 0.1,
                                                   "unit": "K"}}}}
        with pytest.raises(KeyError, match="No node named"):
            build_fit_twin(section, twin(NOISY_MODEL), context())

    def test_an_empty_replace_key_means_no_replacements(self):
        # YAML ``replace:`` with nothing under it parses to None.
        model = twin()
        repaired, replaced = build_fit_twin({"replace": None}, model,
                                            context())
        assert repaired is model
        assert replaced == ()


class TestGrammar:
    def test_unknown_keys_are_swept(self):
        with pytest.raises(ConfigError, match="drop"):
            build_fit_twin({"drop": ["noise"]}, twin(), context())

    def test_a_non_mapping_section_is_refused(self):
        with pytest.raises(ConfigError, match="without"):
            build_fit_twin(["noise"], twin(), context())

    def test_a_pipeline_twin_is_refused_by_name(self):
        pipeline = twin({"kind": "pipeline",
                         "stages": [{"name": "gain", "type": "GainOperator",
                                     "gain": {"value": 1.1,
                                              "unit": "dimensionless"}}]})
        with pytest.raises(ConfigError, match="pipeline"):
            build_fit_twin({"without": ["gain"]}, pipeline, context())
