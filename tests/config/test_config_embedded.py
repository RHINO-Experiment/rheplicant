from __future__ import annotations

import base64

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
