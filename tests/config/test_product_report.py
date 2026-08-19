from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from _rheplicant_bootstrap.errors import ConfigError
from _rheplicant_bootstrap.output import ReportRequest
from rheplicant.config.products.report import materialize_report
from rheplicant.config.sections.runs import RunResult


def execution(*rows):
    return SimpleNamespace(runs=rows, status="ok")


def sample_row(name, samples, wall=1_000_000_000):
    product = SimpleNamespace(samples={"gain": np.asarray(samples)})
    return SimpleNamespace(
        parsed=SimpleNamespace(name=name, kind="plan.sample"),
        result=RunResult(name, "plan.sample", product, None),
        wall_time_ns=wall,
        captured_expected_refusal=False,
    )


def test_report_json_and_text_share_ordered_statistics_and_relatives():
    record = execution(
        sample_row("reference", [1.0, 3.0], wall=2_000_000_000),
        sample_row("fit", [2.0, 6.0], wall=3_000_000_000),
    )
    request = ReportRequest(
        rows=("fit", "reference"),
        columns=("mean", "std", "seconds"),
        reference="reference",
        relative=("mean_sigma", "width_ratio"),
        formats=("json", "text"),
    )
    files = materialize_report(record, request, component_limit=255)
    assert [file.relative_path for file in files] == ["report.json", "report.txt"]
    value = json.loads(files[0].payload)
    assert [row["name"] for row in value["rows"]] == ["fit", "reference"]
    fit = value["rows"][0]
    assert fit["statistics"] == {
        "mean": {"mapping/n-6761696e": 4.0},
        "seconds": 3.0,
        "std": {"mapping/n-6761696e": 2.0},
    }
    assert fit["relative"]["width_ratio"] == {"mapping/n-6761696e": 2.0}
    assert fit["relative"]["mean_sigma"]["mapping/n-6761696e"] == pytest.approx(
        2.0 / np.sqrt(5.0)
    )
    text = files[1].payload.decode()
    assert text.splitlines()[0].startswith("run\tkind\tmean\tstd\tseconds")
    assert text.splitlines()[1].startswith("fit\tplan.sample\t")


def test_report_refuses_missing_rows_statistics_and_zero_reference_width():
    record = execution(sample_row("fit", [1.0, 1.0]))
    with pytest.raises(ConfigError, match="missing"):
        materialize_report(
            record,
            ReportRequest(("missing",), ("seconds",), None, (), ("json",)),
            component_limit=255,
        )
    estimate = SimpleNamespace(
        parsed=SimpleNamespace(name="point", kind="plan.estimate"),
        result=RunResult(
            "point",
            "plan.estimate",
            SimpleNamespace(values={"gain": np.array(2.0)}),
            None,
        ),
        wall_time_ns=1,
        captured_expected_refusal=False,
    )
    with pytest.raises(ConfigError, match="std"):
        materialize_report(
            execution(estimate),
            ReportRequest(("point",), ("std",), None, (), ("json",)),
            component_limit=255,
        )
    with pytest.raises(ConfigError, match="zero"):
        materialize_report(
            record,
            ReportRequest(
                ("fit",),
                ("std",),
                "fit",
                ("width_ratio",),
                ("json",),
            ),
            component_limit=255,
        )


def test_report_does_not_execute_or_call_the_product():
    class Product:
        samples = {"gain": np.array([1.0, 2.0])}

        def __call__(self):
            raise AssertionError("report reran science")

    row = sample_row("fit", [1.0, 2.0])
    row.result = RunResult("fit", "plan.sample", Product(), None)
    materialize_report(
        execution(row),
        ReportRequest(("fit",), ("mean",), None, (), ("json",)),
        component_limit=255,
    )
