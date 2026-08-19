from __future__ import annotations

import json
import stat
import subprocess
import sys
from pathlib import Path

import _rheplicant_bootstrap.entry as entry
import rheplicant.config.orchestration as orchestration
from _rheplicant_bootstrap.script import publish_script, render_script
from _rheplicant_bootstrap.source import read_source
from rheplicant.config.errors import ConfigError
from tests.config.test_config_cli import document, write_document


def product_document(output: Path, *, clobber: bool = False):
    value = document(output=output)
    value["outputs"].update(
        {
            "clobber": clobber,
            "write": {"arrays": True, "assembly": True},
            "report": {
                "rows": ["forward"],
                "columns": ["seconds"],
                "format": ["json", "text"],
            },
        }
    )
    return value


def _scientific_bytes(target: Path) -> dict[str, bytes]:
    manifest = json.loads((target / "products.json").read_bytes())
    return {
        row["relative_path"]: (target / row["relative_path"]).read_bytes()
        for row in manifest["files"]
        if row["selector"] != "report"
    } | {"products.json": (target / "products.json").read_bytes()}


def test_cli_materializes_before_staging_and_publishes_the_complete_product_tree(
    tmp_path, monkeypatch
):
    from _rheplicant_bootstrap.cli import main

    target = tmp_path / "result"
    config = tmp_path / "config.yaml"
    write_document(config, product_document(target))
    original = entry.stage_bundle

    def observing_stage(authorization, candidate, platform, *, publication):
        assert not target.exists()
        assert "products.json" in candidate.files
        assert "runs/n-666f7277617264/arrays.npz" in candidate.files
        return original(
            authorization,
            candidate,
            platform,
            publication=publication,
        )

    monkeypatch.setattr(entry, "stage_bundle", observing_stage)
    assert main(["run", str(config)]) == 0
    manifest = json.loads((target / "products.json").read_bytes())
    paths = {row["relative_path"] for row in manifest["files"]}
    assert paths == {
        "runs/n-666f7277617264/arrays.npz",
        "layers/base/assembly.json",
        "report.json",
        "report.txt",
    }
    for relative in (*paths, "products.json"):
        assert stat.S_IMODE((target / relative).stat().st_mode) == 0o600
    assert stat.S_IMODE((target / "runs").stat().st_mode) == 0o700
    assert stat.S_IMODE((target / "layers/base").stat().st_mode) == 0o700


def test_validate_with_product_requests_never_materializes_or_creates_output(
    tmp_path, monkeypatch
):
    from _rheplicant_bootstrap.cli import main

    target = tmp_path / "result"
    config = tmp_path / "config.yaml"
    write_document(config, product_document(target))

    def forbidden(*_args, **_kwargs):
        raise AssertionError("validate materialized scientific products")

    monkeypatch.setattr(orchestration, "build_product_bundle", forbidden)
    assert main(["validate", str(config)]) == 0
    assert not target.exists()


def test_product_materialization_refusal_never_publishes_the_success_target(
    tmp_path, monkeypatch, capsys
):
    from _rheplicant_bootstrap.cli import main

    target = tmp_path / "result"
    config = tmp_path / "config.yaml"
    write_document(config, product_document(target))

    def refuse(*_args, **_kwargs):
        raise ConfigError("deliberate product refusal")

    monkeypatch.setattr(orchestration, "build_product_bundle", refuse)
    assert main(["run", str(config)]) == 2
    assert not target.exists()
    failures = tuple(tmp_path.glob("result.refused-*"))
    assert len(failures) == 1
    assert not (failures[0] / "products.json").exists()
    assert "deliberate product refusal" in capsys.readouterr().err


def test_direct_and_generated_launches_emit_identical_scientific_bytes(tmp_path):
    from _rheplicant_bootstrap.cli import main

    direct_target = tmp_path / "direct"
    generated_target = tmp_path / "generated"
    direct_config = tmp_path / "direct.yaml"
    generated_config = tmp_path / "generated.yaml"
    write_document(direct_config, product_document(direct_target))
    write_document(
        generated_config,
        product_document(generated_target),
    )
    assert main(["run", str(direct_config)]) == 0
    script = tmp_path / "generated.py"
    source = read_source(str(generated_config), base_dir=None, stdin=None)
    publish_script(render_script(source, ()), script)
    completed = subprocess.run(
        [sys.executable, str(script)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    direct = _scientific_bytes(direct_target)
    generated = _scientific_bytes(generated_target)
    assert direct.keys() == generated.keys()
    for relative in direct:
        if relative != "products.json":
            assert direct[relative] == generated[relative]


def test_product_tree_survives_owned_clobber(tmp_path):
    from _rheplicant_bootstrap.cli import main

    target = tmp_path / "result"
    config = tmp_path / "config.yaml"
    write_document(config, product_document(target))
    assert main(["run", str(config)]) == 0
    first = _scientific_bytes(target)
    write_document(config, product_document(target, clobber=True))
    assert main(["run", str(config)]) == 0
    second = _scientific_bytes(target)
    assert first.keys() == second.keys()
    for relative in first:
        if relative != "products.json":
            assert first[relative] == second[relative]
