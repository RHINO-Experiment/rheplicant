from __future__ import annotations

from tests.config.test_config_cli import document, write_document


def test_usage_and_stdin_base_dir_have_status_two(tmp_path, capsys, monkeypatch):
    from _rheplicant_bootstrap.cli import main

    assert main([]) == 2
    assert main(["unknown", "x.yaml"]) == 2
    assert main(["run", "-"]) == 2
    assert "--base-dir" in capsys.readouterr().err


def test_a34_refusal_publishes_a_sibling_without_mutating_target(tmp_path, capsys):
    from _rheplicant_bootstrap.cli import main

    target = tmp_path / "result"
    target.mkdir(mode=0o700)
    sentinel = target / "sentinel"
    sentinel.write_bytes(b"old")
    config = tmp_path / "config.yaml"
    write_document(config, document(output=target))
    assert main(["run", str(config)]) == 2
    assert sentinel.read_bytes() == b"old"
    siblings = tuple(tmp_path.glob("result.refused-*"))
    assert len(siblings) == 1
    assert (siblings[0] / "provenance.json").is_file()
    assert f"refused audit: {siblings[0]}\n" in capsys.readouterr().err


def test_malformed_yaml_has_no_output_or_lock(tmp_path):
    from _rheplicant_bootstrap.cli import main

    config = tmp_path / "bad.yaml"
    config.write_bytes(b"[not yaml")
    assert main(["run", str(config)]) == 2
    assert tuple(tmp_path.iterdir()) == (config,)


def test_validate_bad_existing_target_is_read_only_and_has_no_failure_bundle(
    tmp_path,
):
    from _rheplicant_bootstrap.cli import main

    target = tmp_path / "result"
    target.mkdir(mode=0o700)
    sentinel = target / "sentinel"
    sentinel.write_bytes(b"foreign")
    config = tmp_path / "config.yaml"
    value = document(output=target)
    value["outputs"]["clobber"] = True
    write_document(config, value)

    assert main(["validate", str(config)]) == 2
    assert sentinel.read_bytes() == b"foreign"
    assert not tuple(tmp_path.glob("result.refused-*"))
