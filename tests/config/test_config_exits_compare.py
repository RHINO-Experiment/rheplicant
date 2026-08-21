from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from rheplicant.config import ConfigError
from rheplicant.config.sections.comparison import CompareProduct, compare_products
from rheplicant.config.sections.exit_support import (
    DEFERRED_CHECKS,
    EXECUTORS,
    PARSERS,
    PRE_EXECUTORS,
)
from rheplicant.config.sections.runs import RunResult, run_document
from tests.config.test_config_document import synthetic_document


def document(*runs):
    doc = synthetic_document()
    doc["runs"] = list(runs)
    return doc


@pytest.mark.parametrize(
    ("node", "path"),
    (
        ({"of": ["a"], "metric": "rms", "tolerance": 0}, "of"),
        ({"of": ["a", "a"], "metric": "rms", "tolerance": 0}, "of"),
        ({"of": ["a", 2], "metric": "rms", "tolerance": 0}, "of"),
        ({"of": ["a", "b"], "metric": "mean", "tolerance": 0}, "metric"),
        ({"of": ["a", "b"], "metric": "rms", "tolerance": -1}, "tolerance"),
        ({"of": ["a", "b"], "metric": "rms", "tolerance": float("inf")}, "tolerance"),
        ({"of": ["a", "b"], "metric": "rms", "tolerance": True}, "tolerance"),
        ({"of": ["a", "b"], "metric": "rms", "tolerance": 0, "extra": 1}, "extra"),
    ),
)
def test_compare_grammar_is_closed_and_pathful(node, path):
    with pytest.raises(ConfigError, match=path):
        run_document(document({"kind": "compare", **node}))


def test_compare_parse_failure_precedes_all_execution(monkeypatch):
    called = []
    original = EXECUTORS["forward"]

    def spy(parsed, configured, previous):
        called.append(parsed.name)
        return original(parsed, configured, previous)

    monkeypatch.setitem(EXECUTORS, "forward", spy)
    with pytest.raises(ConfigError, match="metric"):
        run_document(
            document(
                {"name": "a", "kind": "forward"},
                {"name": "b", "kind": "forward"},
                {
                    "name": "comparison",
                    "kind": "compare",
                    "of": ["a", "b"],
                    "metric": "unknown",
                    "tolerance": 0,
                },
            )
        )
    assert called == []


def test_compare_reads_two_prior_successful_products_in_order():
    results = run_document(
        document(
            {"name": "left", "kind": "forward"},
            {"name": "right", "kind": "forward"},
            {
                "name": "same",
                "kind": "compare",
                "of": ["left", "right"],
                "metric": "max_abs",
                "tolerance": 0,
            },
        )
    )
    assert results["same"].product == CompareProduct(
        left="left",
        right="right",
        metric="max_abs",
        value=0.0,
        tolerance=0.0,
        passed=True,
        components=128,
    )


@pytest.mark.parametrize("missing", ("left", "right"))
def test_compare_refuses_missing_or_failed_prior_products(missing):
    options = SimpleNamespace(
        name="c",
        kind="compare",
        options={"of": ("left", "right"), "metric": "rms", "tolerance": 0.0},
    )
    previous = {
        "left": RunResult("left", "forward", np.array([1.0]), None),
        "right": RunResult("right", "forward", np.array([1.0]), None),
    }
    previous.pop(missing)
    with pytest.raises(ConfigError, match=missing):
        PRE_EXECUTORS["compare"](options, None, previous)
    previous[missing] = RunResult(missing, "forward", None, ValueError("refused"))
    with pytest.raises(ConfigError, match="refused"):
        PRE_EXECUTORS["compare"](options, None, previous)


@pytest.mark.parametrize(
    ("metric", "expected"),
    (
        ("max_abs", 2.0),
        ("rms", np.sqrt(5.0 / 3.0)),
        ("max_rel_diff", 0.5),
    ),
)
def test_compare_metrics_reduce_every_component(metric, expected):
    found = compare_products(
        {"x": np.array([0.0, 2.0]), "z": (np.array([3.0]),)},
        {"x": np.array([0.0, 4.0]), "z": (np.array([2.0]),)},
        metric=metric,
        tolerance=expected,
        left_name="a",
        right_name="b",
    )
    assert found.value == pytest.approx(expected)
    assert found.passed is True
    assert found.components == 3


def test_compare_marks_one_zero_against_nonzero_as_infinite_data():
    found = compare_products(
        np.array([0.0]),
        np.array([1.0]),
        metric="max_rel_diff",
        tolerance=100,
        left_name="zero",
        right_name="one",
    )
    assert found.value == "infinity"
    assert found.passed is False


def test_unsigned_integer_difference_does_not_underflow():
    found = compare_products(
        np.array([1], dtype=np.uint64),
        np.array([3], dtype=np.uint64),
        metric="max_abs",
        tolerance=2,
        left_name="a",
        right_name="b",
    )
    assert found.value == 2.0
    assert found.passed is True


@pytest.mark.parametrize(
    ("left", "right", "word"),
    (
        ({"a": np.array([1])}, {"b": np.array([1])}, "structure"),
        ({"a": np.array([1])}, {"a": np.array([[1]])}, "shape"),
        ({"a": np.array([1], dtype=np.int64)}, {"a": np.array([1.0])}, "dtype"),
        ({"a": np.array([1])}, [np.array([1])], "structure"),
    ),
)
def test_compare_refuses_structure_shape_and_dtype_class_mismatch(left, right, word):
    with pytest.raises(ConfigError, match=word):
        compare_products(
            left,
            right,
            metric="rms",
            tolerance=0,
            left_name="a",
            right_name="b",
        )


def test_compare_is_present_in_all_four_live_registries():
    assert all(
        "compare" in registry
        for registry in (PARSERS, PRE_EXECUTORS, EXECUTORS, DEFERRED_CHECKS)
    )
    assert DEFERRED_CHECKS["compare"] == (
        "compare.left_available",
        "compare.right_available",
        "compare.products_compatible",
    )
