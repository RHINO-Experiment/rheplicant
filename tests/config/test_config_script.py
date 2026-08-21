from __future__ import annotations

import base64
import stat
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

from _rheplicant_bootstrap.errors import ConfigError
from _rheplicant_bootstrap.script import publish_script, render_script
from _rheplicant_bootstrap.source import read_source
from tests.config.test_config_cli import document, write_document


def test_script_contains_only_base64_source_and_shared_entry(tmp_path):
    config = tmp_path / "config.yaml"
    payload = write_document(config)
    source = read_source(str(config), base_dir=None, stdin=None)
    rendered = render_script(source, ())
    text = rendered.decode("utf-8")
    assert "_rheplicant_bootstrap.run_embedded_config" in text
    assert base64.b64encode(payload).decode("ascii") in text
    assert "build_model(" not in text
    assert "ParsedOptions" not in text


def test_script_publication_is_mode_600_and_no_clobber(tmp_path):
    output = tmp_path / "run.py"
    publish_script(b"print('ok')\n", output)
    assert output.read_bytes() == b"print('ok')\n"
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    with pytest.raises(ConfigError, match="already exists"):
        publish_script(b"different\n", output)
    assert output.read_bytes() == b"print('ok')\n"


def test_rendered_script_runs_through_the_shared_embedded_entry(tmp_path):
    config = tmp_path / "config.yaml"
    target = tmp_path / "embedded"
    payload = write_document(config, document(output=target))
    source = read_source(str(config), base_dir=None, stdin=None)
    script = tmp_path / "run.py"
    publish_script(render_script(source, ()), script)

    completed = subprocess.run(
        [sys.executable, str(script)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert (target / "config.input.yaml").read_bytes() == payload
    assert b'"launch_mode": "embedded"' in (target / "provenance.json").read_bytes()


def test_two_script_publishers_have_one_winner_and_never_clobber(tmp_path):
    output = tmp_path / "race.py"
    payloads = (b"print('first')\n", b"print('second')\n")

    def publish(payload):
        try:
            publish_script(payload, output)
        except ConfigError:
            return "refused"
        return "ok"

    with ThreadPoolExecutor(max_workers=2) as pool:
        statuses = tuple(pool.map(publish, payloads))
    assert sorted(statuses) == ["ok", "refused"]
    assert output.read_bytes() in payloads
