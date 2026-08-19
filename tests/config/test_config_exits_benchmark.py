from __future__ import annotations

from types import SimpleNamespace

import pytest

import rheplicant.config.sections.benchmark as benchmark_module
from rheplicant.config import ConfigError
from rheplicant.config.sections.benchmark import (
    BenchmarkProduct,
    benchmark_callables,
)
from rheplicant.config.sections.exit_support import (
    DEFERRED_CHECKS,
    EXECUTORS,
    PARSERS,
    PRE_EXECUTORS,
)
from rheplicant.config.sections.runs import run_document
from tests.config.test_config_document import synthetic_document


class BlockingResult:
    def __init__(self, events):
        self.events = events

    def block_until_ready(self):
        self.events.append("block")
        return self


@pytest.mark.parametrize(
    ("options", "path"),
    (
        ({}, "variants"),
        ({"variants": []}, "variants"),
        ({"variants": ["base", "base"]}, "variants"),
        ({"variants": [1]}, "variants"),
        ({"variants": ["base"], "repeats": 0}, "repeats"),
        ({"variants": ["base"], "repeats": True}, "repeats"),
        ({"variants": ["base"], "warmup": -1}, "warmup"),
        ({"variants": ["base"], "metrics": []}, "metrics"),
        ({"variants": ["base"], "metrics": ["device_memory"]}, "metrics"),
        ({"variants": ["base"], "extra": 1}, "extra"),
    ),
)
def test_benchmark_grammar_is_closed_and_counts_are_whole(options, path):
    document = synthetic_document()
    document["runs"] = [{"kind": "benchmark", **options}]
    with pytest.raises(ConfigError, match=path):
        run_document(document)


def test_benchmark_defaults_and_real_variant_execution():
    document = synthetic_document()
    document["runs"] = [
        {
            "kind": "benchmark",
            "variants": ["base", "unity_gain"],
            "repeats": 1,
            "warmup": 0,
        }
    ]
    product = run_document(document)["benchmark"].product
    assert isinstance(product, BenchmarkProduct)
    assert [variant.name for variant in product.variants] == ["base", "unity_gain"]
    for variant in product.variants:
        assert tuple(variant.metrics) == ("wall_time",)
        metric = variant.metrics["wall_time"]
        assert len(metric.samples) == 1
        assert metric.unit == "ns"


def test_benchmark_refuses_an_unknown_prepared_variant():
    document = synthetic_document()
    document["runs"] = [{"kind": "benchmark", "variants": ["missing"]}]
    with pytest.raises(ConfigError, match="missing"):
        run_document(document)


def test_warmups_block_but_do_not_enter_raw_samples():
    events = []

    def call():
        events.append("call")
        return BlockingResult(events)

    ticks = iter((10, 13, 20, 25))
    product = benchmark_callables(
        {"base": call},
        repeats=2,
        warmup=1,
        metrics=("wall_time",),
        clock_ns=lambda: next(ticks),
    )
    metric = product.variants[0].metrics["wall_time"]
    assert metric.samples == (3, 5)
    assert (metric.minimum, metric.median, metric.mean) == (3, 4.0, 4.0)
    assert events == ["call", "block", "call", "block", "call", "block"]


def test_peak_memory_is_labelled_python_traced_and_keeps_raw_repeats(monkeypatch):
    peaks = iter(((10, 30), (12, 50)))
    monkeypatch.setattr(benchmark_module.tracemalloc, "start", lambda: None)
    monkeypatch.setattr(benchmark_module.tracemalloc, "stop", lambda: None)
    monkeypatch.setattr(
        benchmark_module.tracemalloc,
        "get_traced_memory",
        lambda: next(peaks),
    )
    product = benchmark_callables(
        {"base": lambda: BlockingResult([])},
        repeats=2,
        warmup=0,
        metrics=("peak_memory",),
    )
    metric = product.variants[0].metrics["peak_memory"]
    assert metric.samples == (30, 50)
    assert metric.unit == "python_traced_bytes"
    assert "device" not in metric.unit


def test_pre_execute_receives_all_prepared_layers_not_one_configured_run():
    layers = (
        SimpleNamespace(layer=SimpleNamespace(kind="base", name=None)),
        SimpleNamespace(layer=SimpleNamespace(kind="variant", name="v")),
    )
    context = benchmark_module.BenchmarkContext(configured=object(), layers=layers)
    parsed = SimpleNamespace(name="b", options={"variants": ("base", "v")})
    PRE_EXECUTORS["benchmark"](parsed, context, {})


def test_benchmark_is_present_in_all_four_live_registries():
    assert all(
        "benchmark" in registry
        for registry in (PARSERS, PRE_EXECUTORS, EXECUTORS, DEFERRED_CHECKS)
    )
    assert DEFERRED_CHECKS["benchmark"] == ("benchmark.variants_available",)
