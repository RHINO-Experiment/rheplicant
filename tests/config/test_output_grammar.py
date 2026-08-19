from __future__ import annotations

import os

import pytest

from _rheplicant_bootstrap.errors import ConfigError
from _rheplicant_bootstrap.output import (
    parse_output_grammar,
    parse_output_request,
    resolve_output_request,
)
from _rheplicant_bootstrap.process import parse_raw_process_mapping
from _rheplicant_bootstrap.types import SourceInput

VALID_OUTPUTS = (
    ({}, (False, "summary", True, True, "json")),
    (
        {"clobber": True, "stdout": "verbose"},
        (True, "verbose", True, True, "json"),
    ),
    (
        {
            "write": {
                "config": True,
                "provenance": True,
                "diagnostics": {"format": "json"},
            }
        },
        (False, "summary", True, True, "json"),
    ),
)
PLAN4B_PATHS = (
    "outputs.write.arrays",
    "outputs.write.aux",
    "outputs.write.taps",
    "outputs.write.assembly",
    "outputs.write.estimates",
    "outputs.write.parameters",
    "outputs.write.draws",
    "outputs.write.losses",
    "outputs.write.gradients",
    "outputs.write.covariance",
    "outputs.write.prediction_bands",
    "outputs.write.posterior_predictives",
    "outputs.write.identifiability",
    "outputs.write.scores",
    "outputs.write.recovery",
    "outputs.write.training_history",
    "outputs.write.timings",
    "outputs.write.refusals",
    "outputs.write.signal_paths",
    "outputs.write.compare",
    "outputs.write.benchmark",
    "outputs.write.chains",
)


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "config.yaml"
    return SourceInput(
        b"schema_version: 1\n",
        str(path),
        str(path),
        str(path),
        str(tmp_path),
        "cli",
    )


def nested_mapping(path, value):
    result = value
    for name in reversed(path):
        result = {name: result}
    return result


@pytest.mark.parametrize(("node", "expected"), VALID_OUTPUTS)
def test_output_defaults_and_accepted_forms(node, expected, source):
    request = parse_output_request({"outputs": node}, source=source, command="run")
    assert (
        request.clobber,
        request.stdout,
        request.write_config,
        request.write_provenance,
        request.write_diagnostics,
    ) == expected


@pytest.mark.parametrize("path", PLAN4B_PATHS)
def test_every_plan4b_output_is_a_typed_request(path, source):
    document = nested_mapping(path.split("."), True)
    request = parse_output_request(document, source=source, command="run")
    assert [product.name for product in request.products] == [path.rsplit(".", 1)[1]]


@pytest.mark.parametrize("name", ("memory_archive", "posterior_net", "campaign"))
def test_plan4b_capabilities_remain_refused(name):
    with pytest.raises(ConfigError, match=f"outputs.{name}.*Plan 4B"):
        parse_output_grammar({name: {}})


@pytest.mark.parametrize("field", ("config", "provenance", "diagnostics"))
def test_mandatory_writes_refuse_false(field):
    with pytest.raises(ConfigError, match=f"outputs.write.{field}.*mandatory"):
        parse_output_grammar({"write": {field: False}})


@pytest.mark.parametrize("value", (None, "", 1, [], True))
def test_output_dir_is_nonempty_text(value):
    with pytest.raises(ConfigError, match="outputs.dir"):
        parse_output_grammar({"dir": value})


@pytest.mark.parametrize("value", (None, 1, "true"))
def test_clobber_is_an_exact_bool(value):
    with pytest.raises(ConfigError, match="outputs.clobber"):
        parse_output_grammar({"clobber": value})


@pytest.mark.parametrize("value", ("quiet", True, None))
def test_stdout_has_a_closed_vocabulary(value):
    with pytest.raises(ConfigError, match="outputs.stdout"):
        parse_output_grammar({"stdout": value})


def test_unknown_keys_report_the_full_path():
    with pytest.raises(ConfigError) as caught:
        parse_output_grammar({"write": {"mystery": True}})
    assert str(caught.value) == "outputs.write.mystery: unknown key"


def test_path_default_is_lexical_and_validate_does_not_derive(source):
    parsed = parse_output_grammar({})
    run = resolve_output_request(parsed, source=source, command="run")
    validate = resolve_output_request(parsed, source=source, command="validate")
    assert run.target_path == os.path.splitext(source.source_path)[0] + ".results"
    assert validate.target_path is None
    assert validate.explicit_dir is False


def test_stdin_run_requires_an_explicit_directory(source):
    stdin = SourceInput(b"{}", "<stdin>", None, "<stdin>", source.base_dir, "cli")
    with pytest.raises(ConfigError, match="stdin requires an explicit"):
        resolve_output_request(parse_output_grammar({}), source=stdin, command="run")
    request = resolve_output_request(
        parse_output_grammar({"dir": "result"}),
        source=stdin,
        command="run",
    )
    assert request.target_path == os.path.join(source.base_dir, "result")


def test_output_path_expands_without_resolving_symlinks(source, monkeypatch):
    monkeypatch.setenv("OUTPUT_LEAF", "linked/result")
    request = parse_output_request(
        {"outputs": {"dir": "$OUTPUT_LEAF"}},
        source=source,
        command="run",
    )
    assert request.target_path == os.path.join(source.base_dir, "linked", "result")


@pytest.mark.parametrize("path", ("/", "."))
def test_root_and_base_directory_are_refused(path, source):
    with pytest.raises(ConfigError, match="outputs.dir"):
        parse_output_request(
            {"outputs": {"dir": path}},
            source=source,
            command="run",
        )


def test_process_entry_passes_only_the_raw_outputs_identity():
    sentinel = {"identity": object()}
    document = {"schema_version": 1, "outputs": sentinel}
    observed = []

    def parser(raw):
        observed.append(raw)
        return ()

    result = parse_raw_process_mapping(document, parse_outputs=parser)
    assert result.outputs == ()
    assert observed == [sentinel]
    assert observed[0] is not document


def test_output_package_imports_no_main_or_science_modules():
    import subprocess
    import sys

    code = (
        "import sys; before=set(sys.modules); import _rheplicant_bootstrap.output; "
        "print(sorted(n for n in set(sys.modules)-before "
        "if n.split('.')[0] in {'rheplicant','jax','numpy'}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        text=True,
        capture_output=True,
    )
    assert result.stdout.strip() == "[]"


def test_document_must_be_a_mapping(source):
    with pytest.raises(ConfigError, match="root must be a mapping"):
        parse_output_request([], source=source, command="run")  # type: ignore[arg-type]
