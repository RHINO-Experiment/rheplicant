"""The authoritative dimension catalog remains complete and reachable."""

import dataclasses

from rheplicant.config.dimension_catalog import (
    CONFIG_CONTEXTUAL,
    CONFIG_DIMENSIONS,
    CONFIG_SPECIAL,
    FORMULA_REGISTRATIONS,
    MODEL_DIMENSIONS,
    MODEL_FORMULA_BINDINGS,
    MODEL_SPECIAL,
    RESOURCE_DIMENSIONS,
    RESOURCE_OUTPUTS,
    RESOURCE_SPECIAL,
)
from rheplicant.config.dimensions import dimension_spec_for
from rheplicant.config.sections.model import operator_table


def _qualified(cls: type, field: str | None = None) -> str:
    name = f"{cls.__module__}.{cls.__qualname__}"
    return name if field is None else f"{name}.{field}"


def test_model_catalog_covers_all_28_classes_and_66_fields():
    classes = {cls for choices in operator_table().values() for cls in choices}
    fields = {
        _qualified(cls, field.name)
        for cls in classes
        for field in dataclasses.fields(cls)
        if field.init
    }
    assert len(classes) == 28
    assert len(fields) == 66
    assert len(MODEL_DIMENSIONS) == 56
    assert len(MODEL_SPECIAL) == 10
    assert fields == {row[0] for row in MODEL_DIMENSIONS} | set(MODEL_SPECIAL)
    assert set(MODEL_FORMULA_BINDINGS) == {_qualified(cls) for cls in classes}


def test_every_catalog_row_is_registered_and_reachable():
    rows = (
        [("model_field", selector) for selector, _ in MODEL_DIMENSIONS]
        + [("model_field", selector) for selector in MODEL_SPECIAL]
        + [("resource_field", selector) for selector, _ in RESOURCE_DIMENSIONS]
        + [("resource_field", selector) for selector in RESOURCE_SPECIAL]
        + [("config_path", selector) for selector, _ in CONFIG_DIMENSIONS]
        + [("config_path", selector) for selector in CONFIG_CONTEXTUAL]
        + [("config_path", selector) for selector in CONFIG_SPECIAL]
        + [("config_path", selector) for selector in RESOURCE_OUTPUTS]
    )
    assert len(rows) == len(set(rows))
    for domain, selector in rows:
        assert dimension_spec_for(domain, selector) is not None


def test_named_formulas_and_operator_bindings_are_complete():
    by_name = {formula.name: formula for formula in FORMULA_REGISTRATIONS}
    assert len(by_name) == len(FORMULA_REGISTRATIONS) == 41
    for producer, binding in MODEL_FORMULA_BINDINGS.items():
        assert binding.formulas.count(binding.output_formula) == 1, producer
        assert set(binding.formulas) <= set(by_name), producer
        assert all(producer in by_name[name].producers for name in binding.formulas)
