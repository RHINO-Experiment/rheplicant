"""The config layer's single refusal type, and the layer boundary it sits on."""

import pytest

from rheplicant.config import ConfigError
from rheplicant.core.errors import DirtError


class TestConfigErrorJoinsTheFamily:
    def test_it_is_catchable_as_a_rheplicant_error(self):
        with pytest.raises(DirtError):
            raise ConfigError("x")

    def test_it_is_catchable_as_a_value_error(self):
        """Generic handlers keep working -- errors.py's module docstring states
        this as the rule every concrete class in the family obeys."""
        with pytest.raises(ValueError):
            raise ConfigError("x")

    def test_it_is_not_a_data_ingestion_error(self):
        """DataIngestionError is scoped to 'a data file could not be read'. A
        config refusal is about what a document MEANT, not about reading it,
        and the two are confined to different layers."""
        from rheplicant.core.errors import DataIngestionError

        assert not issubclass(ConfigError, DataIngestionError)

    def test_it_preserves_all_exception_arguments_and_attaches_a_report(self):
        marker = object()
        error = ConfigError("first", "second", report=marker)
        assert error.args == ("first", "second")
        assert str(error) == "('first', 'second')"
        assert error.report is marker


class TestTheConfigLayerOwnsItsOwnVocabulary:
    def test_config_error_is_exported_from_the_config_package(self):
        import rheplicant.config as config

        assert "ConfigError" in config.__all__

    def test_config_error_is_not_exported_from_core_or_the_top_level(self):
        """tests/core/test_errors.py derives its class set from
        rheplicant/core/errors.py by __module__, so a ConfigError declared
        there would need an OWNERS entry. Declaring it in the config layer
        keeps that guard about core, which is what it is for."""
        import rheplicant
        import rheplicant.core

        assert "ConfigError" not in rheplicant.core.__all__
        assert "ConfigError" not in rheplicant.__all__


class TestLiveNames:
    def test_it_reflects_a_registry_that_grew_after_it_was_built(self):
        """The whole point. core/graph.py:350's registry lists its known keys
        in the refusal; a module-level tuple snapshot of a registry that four
        plans fill would list whatever was registered when it was imported."""
        from rheplicant.config.registry import LiveNames

        table = {"a": 1}
        names = LiveNames(table)
        assert sorted(names) == ["a"]
        table["b"] = 2
        assert sorted(names) == ["a", "b"]
        assert len(names) == 2

    def test_it_renders_as_a_sorted_list_in_a_message(self):
        from rheplicant.config.registry import LiveNames

        assert f"{LiveNames({'b': 0, 'a': 0})}" == "['a', 'b']"

    def test_membership_works(self):
        from rheplicant.config.registry import LiveNames

        assert "a" in LiveNames({"a": 0})
        assert "z" not in LiveNames({"a": 0})
