from __future__ import annotations

import hashlib
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


def _published(target: Path) -> dict[str, bytes]:
    """Every published file by relative path, as the verifier wants it."""
    return {
        str(path.relative_to(target)): path.read_bytes()
        for path in sorted(target.rglob("*"))
        if path.is_file()
    }


def test_the_published_tree_carries_one_digest_over_all_of_it(tmp_path):
    """M3: before this, every artefact was hashed and nothing hashed the set.

    ``provenance.json`` records a sha256 per content artefact but is itself a
    metadata row, which ``audit/provenance.py`` forbids from carrying a digest
    at all. So a publisher could rewrite a file AND the provenance row naming
    it and produce a tree that agrees with itself everywhere -- because no
    file's digest depended on any other file's bytes.
    """
    from _rheplicant_bootstrap.audit.integrity import (
        INTEGRITY_NAME,
        root_digest,
        verify_tree,
    )
    from _rheplicant_bootstrap.cli import main

    target = tmp_path / "result"
    config = tmp_path / "config.yaml"
    write_document(config, document(output=target))
    assert main(["run", str(config)]) == 0

    published = _published(target)
    assert INTEGRITY_NAME in published, sorted(published)
    assert verify_tree(published) == ()

    manifest = json.loads(published[INTEGRITY_NAME])
    listed = {row["relative_path"] for row in manifest["files"]}
    # It covers the metadata files, which is the half that was impossible
    # before: they can only be hashed by something written after them.
    assert {"config.input.yaml", "provenance.json", "diagnostics.json"} <= listed
    # And it covers everything else in the tree except itself.
    assert listed == set(published) - {INTEGRITY_NAME}

    covered = tuple((name, published[name]) for name in sorted(listed))
    assert manifest["root_sha256"] == root_digest(covered)


def test_a_file_rewritten_together_with_its_provenance_row_is_still_caught(tmp_path):
    """The tamper this exists for, done properly rather than clumsily.

    Rewriting a recorded file alone was always detectable -- its provenance row
    disagreed. The attack that worked was rewriting the file AND the sha256 in
    the row that names it: the tree then agreed with itself completely. This
    test performs exactly that forgery and requires it to be caught.
    """
    from _rheplicant_bootstrap.audit.integrity import verify_tree
    from _rheplicant_bootstrap.cli import main

    target = tmp_path / "result"
    config = tmp_path / "config.yaml"
    write_document(config, document(output=target))
    assert main(["run", str(config)]) == 0
    assert verify_tree(_published(target)) == ()

    forged = b"# not what was run\n"
    (target / "config.input.yaml").write_bytes(forged)
    provenance = json.loads((target / "provenance.json").read_bytes())
    row = provenance["artefacts"]["input"]
    row["sha256"] = hashlib.sha256(forged).hexdigest()
    row["bytes"] = len(forged)
    (target / "provenance.json").write_bytes(
        json.dumps(provenance, sort_keys=True, ensure_ascii=False).encode()
    )

    problems = verify_tree(_published(target))
    assert problems, "the self-consistent forgery went undetected"
    assert any("config.input.yaml" in problem for problem in problems), problems


def test_editing_the_manifest_to_match_the_forgery_does_not_help_either(tmp_path):
    """The next move up, and why the root digest is a separate value.

    A forger who also edits the manifest's row must edit ``root_sha256`` to
    match; one that edits the row and leaves the root is caught by the manifest
    disagreeing with itself, before any file is read.
    """
    from _rheplicant_bootstrap.audit.integrity import INTEGRITY_NAME, verify_tree
    from _rheplicant_bootstrap.cli import main

    target = tmp_path / "result"
    config = tmp_path / "config.yaml"
    write_document(config, document(output=target))
    assert main(["run", str(config)]) == 0

    forged = b"# not what was run\n"
    (target / "config.input.yaml").write_bytes(forged)
    manifest = json.loads((target / INTEGRITY_NAME).read_bytes())
    for row in manifest["files"]:
        if row["relative_path"] == "config.input.yaml":
            row["sha256"] = hashlib.sha256(forged).hexdigest()
            row["bytes"] = len(forged)
    (target / INTEGRITY_NAME).write_bytes(
        json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode()
    )

    problems = verify_tree(_published(target))
    assert any("root_sha256" in problem for problem in problems), problems


def test_a_file_added_after_publication_is_reported(tmp_path):
    """Absence from the manifest is a finding, not a gap in coverage."""
    from _rheplicant_bootstrap.audit.integrity import verify_tree
    from _rheplicant_bootstrap.cli import main

    target = tmp_path / "result"
    config = tmp_path / "config.yaml"
    write_document(config, document(output=target))
    assert main(["run", str(config)]) == 0

    (target / "extra.txt").write_bytes(b"added later\n")
    problems = verify_tree(_published(target))
    assert any("extra.txt" in problem for problem in problems), problems


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
