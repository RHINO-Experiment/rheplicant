from __future__ import annotations

import json
import os
from pathlib import Path

import yaml

from tests.config.test_config_document import synthetic_document


def document(*, output: Path | None = None):
    value = synthetic_document()
    value.pop("variants")
    if output is not None:
        value["outputs"] = {"dir": str(output)}
    return value


def write_document(path: Path, value=None) -> bytes:
    payload = yaml.safe_dump(document() if value is None else value, sort_keys=False).encode()
    path.write_bytes(payload)
    return payload


def test_validate_prints_the_exact_summary_and_creates_no_output(tmp_path, capsys):
    from _rheplicant_bootstrap.cli import main

    config = tmp_path / "config.yaml"
    write_document(config)
    assert main(["validate", str(config)]) == 0
    streams = capsys.readouterr()
    assert streams.out == "configuration valid: base + 0 variants\n"
    assert not (tmp_path / "config.results").exists()


def test_run_publishes_the_mandatory_audit_tree(tmp_path, capsys):
    from _rheplicant_bootstrap.cli import main

    target = tmp_path / "result"
    config = tmp_path / "config.yaml"
    input_bytes = write_document(config, document(output=target))
    assert main(["run", str(config)]) == 0
    assert (target / "config.input.yaml").read_bytes() == input_bytes
    assert (target / "config.resolved.yaml").is_file()
    provenance = json.loads((target / "provenance.json").read_bytes())
    diagnostics = json.loads((target / "diagnostics.json").read_bytes())
    assert provenance["status"] == diagnostics["status"] == "ok"
    assert "configuration run complete:" in capsys.readouterr().out


def test_stdout_none_suppresses_success_only(tmp_path, capsys):
    from _rheplicant_bootstrap.cli import main

    config = tmp_path / "config.yaml"
    value = document()
    value["outputs"] = {"stdout": "none"}
    write_document(config, value)
    assert main(["validate", str(config)]) == 0
    assert capsys.readouterr().out == ""


def test_validate_never_calls_the_executor_or_acquires_an_output_lease(
    tmp_path, monkeypatch
):
    import _rheplicant_bootstrap.entry as entry
    import rheplicant.config.orchestration as orchestration
    from _rheplicant_bootstrap.cli import main

    config = tmp_path / "config.yaml"
    write_document(config)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("validate crossed a mutation/execution boundary")

    monkeypatch.setattr(entry, "acquire_output_lease", forbidden)
    monkeypatch.setattr(orchestration, "execute_prepared", forbidden)
    assert main(["validate", str(config)]) == 0


def test_validate_an_explicit_missing_destination_is_read_only(tmp_path):
    from _rheplicant_bootstrap.cli import main

    target = tmp_path / "missing" / "result"
    config = tmp_path / "config.yaml"
    write_document(config, document(output=target))
    assert main(["validate", str(config)]) == 0
    assert not target.parent.exists()


def test_validate_warns_once_for_a_trusted_writing_plugin(
    tmp_path, monkeypatch, capsys
):
    from _rheplicant_bootstrap.cli import main

    sentinel = tmp_path / "plugin-ran"
    plugin = tmp_path / "trusted_writing_plugin.py"
    plugin.write_text(
        "import os\nfrom pathlib import Path\n"
        "Path(os.environ['RHEPLICANT_TEST_SENTINEL']).write_text('ran')\n"
    )
    monkeypatch.syspath_prepend(tmp_path)
    monkeypatch.setenv("RHEPLICANT_TEST_SENTINEL", os.fspath(sentinel))
    config = tmp_path / "config.yaml"
    value = document()
    value["plugins"] = ["trusted_writing_plugin"]
    write_document(config, value)

    assert main(["validate", str(config)]) == 0
    assert sentinel.read_text() == "ran"
    assert capsys.readouterr().err == (
        "warning: trusted plugin/python code may perform unobserved filesystem I/O\n"
    )
