from __future__ import annotations

import io
import json
from types import SimpleNamespace

import numpy as np
import pytest

from _rheplicant_bootstrap.errors import ConfigError
from _rheplicant_bootstrap.output import ProductRequest, ReportRequest
from rheplicant.config.products import build_product_bundle, validate_product_bundle
from rheplicant.config.products.extractors import RUN_KIND_SELECTORS
from rheplicant.config.sections.benchmark import (
    BenchmarkMetric,
    BenchmarkProduct,
    BenchmarkVariant,
)
from rheplicant.config.sections.comparison import CompareProduct
from rheplicant.config.sections.runs import RunResult


class FakeAssembly:
    graph_name = "receiver"
    lit = ("antenna",)
    skipped = ()
    instances = ()
    materialized = ()
    aliased = ()
    placements = ((('antenna',), "antenna"),)

    def to_svg(self, title=None, theme="light"):
        return f'<svg data-theme="{theme}"><title>{title}</title></svg>'

    def to_html(self, title=None, theme="light"):
        return f"<!doctype html><p>{title}:{theme}</p>"

    def to_mermaid(self, theme="light"):
        return f"%% {theme}\ngraph TD\n"


def configured():
    inference = SimpleNamespace(
        observed=SimpleNamespace(entries={"primary": np.array([0.0, 1.0])}),
        truth={},
        truth_omitted={},
    )
    return SimpleNamespace(inference=inference, twin=FakeAssembly())


def record(rows):
    identity = SimpleNamespace(kind="base", name=None)
    layer_ref = SimpleNamespace(kind="base", name=None, identity=identity)
    prepared_layer = SimpleNamespace(layer=layer_ref, configured=configured())
    for row in rows:
        row.parsed.layer = layer_ref
    return SimpleNamespace(
        prepared=SimpleNamespace(layers=(prepared_layer,)),
        runs=tuple(rows),
        status="ok",
    )


def row(name, kind, product, *, wall=10, captured=False, error=None, reuse=None):
    return SimpleNamespace(
        parsed=SimpleNamespace(name=name, kind=kind, layer=None, reuse=reuse),
        result=RunResult(name, kind, product, error),
        status="ok",
        wall_time_ns=wall,
        error=None,
        captured_expected_refusal=captured,
    )


def test_bundle_filters_runs_encodes_names_and_records_omissions():
    execution = record(
        [
            row("raw/forward", "forward", SimpleNamespace(data=np.array([1, 2]), aux={})),
            row(
                "fit",
                "optimize",
                {"params": {"gain": np.array(2.0)}, "losses": np.array([2.0, 1.0])},
            ),
        ]
    )
    bundle = build_product_bundle(
        execution,
        requests=(
            ProductRequest("parameters", "npz", (), ()),
            ProductRequest("timings", "json", ("raw/forward",), ()),
        ),
        report=None,
        component_limit=255,
    )
    assert [file.relative_path for file in bundle.files] == [
        "runs/n-666974/parameters.npz",
        "runs/n-7261772f666f7277617264/timings.json",
    ]
    manifest = json.loads(bundle.manifest)
    assert manifest["omissions"] == [
        {
            "kind": "forward",
            "reason": "outputs.write.parameters: is not compatible with kind: forward.",
            "run": "raw/forward",
            "selector": "parameters",
        }
    ]
    validate_product_bundle(bundle, component_limit=255)


def test_explicit_incompatible_or_unknown_run_is_a_refusal():
    execution = record(
        [row("forward", "forward", SimpleNamespace(data=np.array([1]), aux={}))]
    )
    for request, word in (
        (ProductRequest("parameters", "npz", ("forward",), ()), "not compatible"),
        (ProductRequest("arrays", "npz", ("missing",), ()), "missing"),
    ):
        with pytest.raises(ConfigError, match=word):
            build_product_bundle(
                execution,
                requests=(request,),
                report=None,
                component_limit=255,
            )


def test_compare_benchmark_refusals_layers_and_report_materialize_together():
    benchmark = BenchmarkProduct(
        (
            BenchmarkVariant(
                "base",
                {
                    "wall_time": BenchmarkMetric((10, 20), 10, 15.0, 15.0, "ns")
                },
            ),
        )
    )
    execution = record(
        [
            row("comparison", "compare", CompareProduct("a", "b", "rms", 0.5, 1.0, True, 3)),
            row("speed", "benchmark", benchmark, wall=30),
            row("expected", "forward", None, captured=True, error=ValueError("intended")),
        ]
    )
    bundle = build_product_bundle(
        execution,
        requests=(
            ProductRequest("compare", "json", (), ()),
            ProductRequest("benchmark", "json", (), ()),
            ProductRequest("refusals", "txt", (), ()),
            ProductRequest("assembly", "json", (), ()),
            ProductRequest("signal_paths", "svg", (), (("themes", ("dark",)),)),
        ),
        report=ReportRequest(("speed",), ("seconds",), None, (), ("json",)),
        component_limit=255,
    )
    paths = [file.relative_path for file in bundle.files]
    assert paths == [
        "runs/n-636f6d70617269736f6e/compare.json",
        "runs/n-7370656564/benchmark.json",
        "runs/n-6578706563746564/refusals.txt",
        "layers/base/assembly.json",
        "layers/base/signal-path-dark.svg",
        "report.json",
    ]
    assert b"intended" in bundle.files[2].payload
    assert json.loads(bundle.files[0].payload)["passed"] is True
    assert json.loads(bundle.files[1].payload)["variants"][0]["name"] == "base"
    assert json.loads(bundle.files[-1].payload)["rows"][0]["statistics"]["seconds"] == 3e-08
    manifest = json.loads(bundle.manifest)
    assert [request["selector"] for request in manifest["requests"]][-1] == "report"


def test_netcdf_chains_are_deterministic_when_the_optional_writer_is_present():
    pytest.importorskip("scipy")
    execution = record(
        [
            row(
                "posterior",
                "plan.sample",
                SimpleNamespace(samples={"g": np.arange(6.0).reshape(3, 2)}),
            )
        ]
    )
    request = ProductRequest("chains", "netcdf", (), ())
    first = build_product_bundle(
        execution, requests=(request,), report=None, component_limit=255
    )
    second = build_product_bundle(
        execution, requests=(request,), report=None, component_limit=255
    )
    assert first.files[0].payload == second.files[0].payload
    assert first.files[0].payload.startswith(b"CDF")
    assert not first.files[0].payload.startswith(b"PK")
    assert io.BytesIO(first.files[0].payload).getvalue() == first.files[0].payload


def test_taps_are_written_as_separate_named_snapshots():
    execution = record(
        [
            row(
                "forward",
                "forward",
                SimpleNamespace(
                    data=np.array([1.0]),
                    aux={
                        "snapshot/early": np.array([2.0]),
                        "snapshot/late": np.array([3.0]),
                    },
                ),
            )
        ]
    )
    bundle = build_product_bundle(
        execution,
        requests=(
            ProductRequest(
                "taps",
                "npz",
                (),
                (("keys", ("early", "late")),),
            ),
        ),
        report=None,
        component_limit=255,
    )
    assert [file.relative_path for file in bundle.files] == [
        "runs/n-666f7277617264/taps/n-6561726c79.npz",
        "runs/n-666f7277617264/taps/n-6c617465.npz",
    ]


def test_fisher_prediction_is_a_band_but_not_a_posterior_predictive():
    covariance = row("cov", "fisher", {"fisher": object(), "covariance": object()})
    prediction = row(
        "width",
        "predict",
        np.array([0.1, 0.2]),
        reuse="cov",
    )
    execution = record([covariance, prediction])
    bundle = build_product_bundle(
        execution,
        requests=(ProductRequest("prediction_bands", "npz", ("width",), ()),),
        report=None,
        component_limit=255,
    )
    with np.load(io.BytesIO(bundle.files[0].payload), allow_pickle=False) as values:
        np.testing.assert_array_equal(values["std"], [0.1, 0.2])
    with pytest.raises(ConfigError, match="posterior predictive"):
        build_product_bundle(
            execution,
            requests=(
                ProductRequest("posterior_predictives", "npz", ("width",), ()),
            ),
            report=None,
            component_limit=255,
        )


def test_all_eighteen_kinds_have_a_product_registry_row():
    assert tuple(RUN_KIND_SELECTORS)[-2:] == ("compare", "benchmark")
    assert len(RUN_KIND_SELECTORS) == 18
