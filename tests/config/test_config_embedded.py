from __future__ import annotations

import base64
import json

from _rheplicant_bootstrap.entry import run_embedded_config
from tests.config.test_config_cli import document, write_document


def test_embedded_entry_uses_the_embedded_launch_mode(tmp_path):
    source_path = tmp_path / "config.yaml"
    target = tmp_path / "embedded"
    payload = write_document(source_path, document(output=target))
    assert run_embedded_config(
        input_bytes_b64=base64.b64encode(payload).decode("ascii"),
        source_path=str(source_path),
        source_realpath=str(source_path),
        source_name=str(source_path),
        base_dir=str(tmp_path),
        presets=(),
    ) == 0
    assert (target / "config.input.yaml").read_bytes() == payload
    assert b'"launch_mode": "embedded"' in (target / "provenance.json").read_bytes()


def test_embedded_entry_publishes_to_the_invocation_directory(tmp_path):
    # The document carries no `outputs` section at all, so `config.input.yaml`
    # and its digest describe exactly the bytes the author wrote; where the
    # tree landed is an invocation fact, recorded as one.
    source_path = tmp_path / "config.yaml"
    target = tmp_path / "results" / "20260822T134501Z-3f9ac2b1-k7m2xq"
    payload = write_document(source_path, document())
    assert b"outputs" not in payload
    assert run_embedded_config(
        input_bytes_b64=base64.b64encode(payload).decode("ascii"),
        source_path=str(source_path),
        source_realpath=str(source_path),
        source_name=str(source_path),
        base_dir=str(tmp_path),
        presets=(),
        outputs_dir=str(target),
    ) == 0
    assert (target / "config.input.yaml").read_bytes() == payload
    assert not (tmp_path / "config.results").exists()
    provenance = json.loads((target / "provenance.json").read_bytes())
    assert provenance["bootstrap"]["invocation_outputs_dir"] == str(target)


def test_embedded_entry_records_no_override_when_none_is_given(tmp_path):
    source_path = tmp_path / "config.yaml"
    target = tmp_path / "embedded"
    payload = write_document(source_path, document(output=target))
    assert run_embedded_config(
        input_bytes_b64=base64.b64encode(payload).decode("ascii"),
        source_path=str(source_path),
        source_realpath=str(source_path),
        source_name=str(source_path),
        base_dir=str(tmp_path),
        presets=(),
    ) == 0
    provenance = json.loads((target / "provenance.json").read_bytes())
    assert provenance["bootstrap"]["invocation_outputs_dir"] is None


def test_embedded_entry_refuses_an_override_against_an_authored_directory(tmp_path, capsys):
    source_path = tmp_path / "config.yaml"
    authored = tmp_path / "authored"
    payload = write_document(source_path, document(output=authored))
    assert run_embedded_config(
        input_bytes_b64=base64.b64encode(payload).decode("ascii"),
        source_path=str(source_path),
        source_realpath=str(source_path),
        source_name=str(source_path),
        base_dir=str(tmp_path),
        presets=(),
        outputs_dir=str(tmp_path / "override"),
    ) == 2
    assert "the document already sets outputs.dir" in capsys.readouterr().err
    assert not authored.exists()
    assert not (tmp_path / "override").exists()


def test_embedded_entry_keeps_the_products_the_invocation_asks_for(tmp_path):
    # The P2 shape: a document about the science only -- no `outputs` section at
    # all -- with both invocation parameters deciding where the tree goes and
    # what is kept in it.
    source_path = tmp_path / "config.yaml"
    target = tmp_path / "results" / "20260822T134501Z-3f9ac2b1-k7m2xq"
    payload = write_document(source_path, document())
    assert b"outputs" not in payload
    assert run_embedded_config(
        input_bytes_b64=base64.b64encode(payload).decode("ascii"),
        source_path=str(source_path),
        source_realpath=str(source_path),
        source_name=str(source_path),
        base_dir=str(tmp_path),
        presets=(),
        outputs_dir=str(target),
        outputs_write=("arrays",),
    ) == 0
    manifest = json.loads((target / "products.json").read_bytes())
    assert [row["selector"] for row in manifest["files"]] == ["arrays"]
    provenance = json.loads((target / "provenance.json").read_bytes())
    assert provenance["bootstrap"]["invocation_outputs_write"] == ["arrays"]


def test_embedded_entry_refuses_an_invocation_write_against_an_asking_document(tmp_path, capsys):
    source_path = tmp_path / "config.yaml"
    target = tmp_path / "out"
    value = document()
    value["outputs"] = {"write": {"arrays": True}}
    payload = write_document(source_path, value)
    assert run_embedded_config(
        input_bytes_b64=base64.b64encode(payload).decode("ascii"),
        source_path=str(source_path),
        source_realpath=str(source_path),
        source_name=str(source_path),
        base_dir=str(tmp_path),
        presets=(),
        outputs_dir=str(target),
        outputs_write=("draws",),
    ) == 2
    assert "the document already requests products" in capsys.readouterr().err
