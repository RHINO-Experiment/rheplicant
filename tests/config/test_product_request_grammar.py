from __future__ import annotations

from dataclasses import fields

import pytest

from _rheplicant_bootstrap.errors import ConfigError
from _rheplicant_bootstrap.output import (
    OutputRequest,
    ParsedOutputSection,
    ProductRequest,
    ReportRequest,
    parse_output_grammar,
)

DEFAULT_FORMATS = {
    "arrays": "npz",
    "aux": "npz",
    "taps": "npz",
    "assembly": "json",
    "estimates": "npz",
    "parameters": "npz",
    "draws": "npz",
    "losses": "npz",
    "gradients": "npz",
    "covariance": "npz",
    "prediction_bands": "npz",
    "posterior_predictives": "npz",
    "identifiability": "json",
    "scores": "npz",
    "recovery": "json",
    "training_history": "npz",
    "timings": "json",
    "refusals": "txt",
    "signal_paths": "svg",
    "compare": "json",
    "benchmark": "json",
    "chains": "npz",
}


def test_all_product_defaults_are_typed_and_keep_document_order():
    parsed = parse_output_grammar({"write": dict.fromkeys(DEFAULT_FORMATS, True)})
    assert parsed.products == tuple(
        ProductRequest(name, format_, (), ())
        for name, format_ in DEFAULT_FORMATS.items()
    )


def test_product_mapping_detaches_common_and_specialized_options():
    parsed = parse_output_grammar(
        {
            "write": {
                "arrays": {"format": "npz", "runs": ["forward one", "fit"]},
                "aux": {"keys": ["dust", "noise"]},
                "taps": {"keys": ["early", "late"]},
                "signal_paths": {
                    "format": "mermaid",
                    "themes": ["dark", "light"],
                },
                "chains": {"format": "netcdf", "runs": ["nuts"]},
            }
        }
    )
    assert parsed.products == (
        ProductRequest("arrays", "npz", ("forward one", "fit"), ()),
        ProductRequest("aux", "npz", (), (("keys", ("dust", "noise")),)),
        ProductRequest("taps", "npz", (), (("keys", ("early", "late")),)),
        ProductRequest(
            "signal_paths",
            "mermaid",
            (),
            (("themes", ("dark", "light")),),
        ),
        ProductRequest("chains", "netcdf", ("nuts",), ()),
    )


@pytest.mark.parametrize("name", tuple(DEFAULT_FORMATS))
def test_false_is_never_a_successful_product_noop(name):
    with pytest.raises(ConfigError) as caught:
        parse_output_grammar({"write": {name: False}})
    assert str(caught.value) == f"outputs.write.{name}: must be true or a mapping."


@pytest.mark.parametrize(
    ("name", "format_"),
    (
        ("arrays", "json"),
        ("assembly", "npz"),
        ("refusals", "json"),
        ("signal_paths", "pdf"),
        ("chains", "zarr"),
    ),
)
def test_product_formats_are_closed_per_selector(name, format_):
    with pytest.raises(ConfigError) as caught:
        parse_output_grammar({"write": {name: {"format": format_}}})
    assert str(caught.value).startswith(f"outputs.write.{name}.format:")


@pytest.mark.parametrize(
    ("node", "path"),
    (
        ({"arrays": {"mystery": True}}, "outputs.write.arrays.mystery"),
        ({"arrays": {"runs": []}}, "outputs.write.arrays.runs"),
        ({"arrays": {"runs": ["a", "a"]}}, "outputs.write.arrays.runs"),
        ({"arrays": {"runs": ["a", 1]}}, "outputs.write.arrays.runs[1]"),
        ({"aux": {"keys": []}}, "outputs.write.aux.keys"),
        ({"taps": {"keys": [""]}}, "outputs.write.taps.keys[0]"),
        ({"signal_paths": {"themes": ["blue"]}}, "outputs.write.signal_paths.themes[0]"),
        ({"assembly": {"keys": ["x"]}}, "outputs.write.assembly.keys"),
    ),
)
def test_product_mapping_errors_name_the_full_path(node, path):
    with pytest.raises(ConfigError) as caught:
        parse_output_grammar({"write": node})
    assert str(caught.value).startswith(path)


def test_report_defaults_and_explicit_formats_are_typed():
    default = parse_output_grammar({"report": {"rows": ["fit", "truth"]}})
    assert default.report == ReportRequest(
        rows=("fit", "truth"),
        columns=("mean", "std", "seconds"),
        reference=None,
        relative=(),
        formats=("text",),
    )

    explicit = parse_output_grammar(
        {
            "report": {
                "rows": ["fit", "truth"],
                "columns": ["mean", "std"],
                "reference": "truth",
                "relative": ["mean_sigma", "width_ratio"],
                "format": ["json", "text"],
            }
        }
    )
    assert explicit.report == ReportRequest(
        ("fit", "truth"),
        ("mean", "std"),
        "truth",
        ("mean_sigma", "width_ratio"),
        ("json", "text"),
    )


@pytest.mark.parametrize(
    ("node", "path"),
    (
        (True, "outputs.report"),
        ({}, "outputs.report.rows"),
        ({"rows": []}, "outputs.report.rows"),
        ({"rows": ["fit", "fit"]}, "outputs.report.rows"),
        ({"rows": ["fit"], "mystery": True}, "outputs.report.mystery"),
        ({"rows": ["fit"], "columns": ["median"]}, "outputs.report.columns[0]"),
        ({"rows": ["fit"], "reference": "other"}, "outputs.report.reference"),
        ({"rows": ["fit"], "relative": ["mean_sigma"]}, "outputs.report.reference"),
        ({"rows": ["fit"], "format": []}, "outputs.report.format"),
        ({"rows": ["fit"], "format": "csv"}, "outputs.report.format"),
    ),
)
def test_report_errors_name_the_full_path(node, path):
    with pytest.raises(ConfigError) as caught:
        parse_output_grammar({"report": node})
    assert str(caught.value).startswith(path)


def test_output_records_append_defaulted_fields_for_positional_compatibility():
    parsed = ParsedOutputSection(None, False, "summary", True, True, "json")
    request = OutputRequest("run", None, False, False, "summary", True, True, "json")
    assert parsed.products == request.products == ()
    assert parsed.report is request.report is None
    assert [field.name for field in fields(ParsedOutputSection)][-2:] == [
        "products",
        "report",
    ]
    assert [field.name for field in fields(OutputRequest)][-2:] == ["products", "report"]
