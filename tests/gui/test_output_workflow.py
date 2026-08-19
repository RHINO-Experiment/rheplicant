from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml

from rheplicant.config import ConfigError
from rheplicant.gui.outputs import (
    classify_output_state,
    project_output_workflow,
    read_audit_artifact,
    set_output_product,
    set_output_report,
)

DOCUMENT = """\
schema_version: 1
defaults: [rhino_v1]
runtime: {jax_enable_x64: true}
model: {}
runs:
  - {name: "fit / one", kind: optimize, n_steps: 2}
  - {name: compare, kind: compare, variants: [base, alternate]}
variants:
  alternate: {model: {}}
outputs:
  dir: /tmp/rheplicant-gui-task-7
  clobber: false
  write:
    arrays: {runs: ["fit / one"]}
    signal_paths: {format: svg, themes: [light, dark]}
  report:
    rows: ["fit / one"]
    columns: [mean, seconds]
    format: [text, json]
"""


def _product(projection, name):
    return next(row for row in projection.products if row.name == name)


def test_projection_keeps_requested_bytes_and_exposes_preset_merged_run_view():
    found = project_output_workflow(DOCUMENT)

    assert found.requested_yaml == DOCUMENT
    resolved = yaml.safe_load(found.resolved_yaml)
    assert resolved["observation"]["site"]["lat_deg"]["value"] == 53.2367
    assert found.resolution_note.startswith("Preset-merged preview")
    assert len(found.products) == 22
    assert _product(found, "arrays").enabled is True
    assert _product(found, "arrays").runs == ("fit / one",)
    assert _product(found, "chains").formats == ("npz", "netcdf")
    assert found.report is not None
    assert found.report.formats == ("text", "json")


def test_projection_previews_injective_product_report_and_audit_paths():
    found = project_output_workflow(DOCUMENT)
    encoded_run = "n-" + b"fit / one".hex()
    encoded_variant = "n-" + b"alternate".hex()

    assert _product(found, "arrays").expected_paths == (
        f"runs/{encoded_run}/arrays.npz",
    )
    assert _product(found, "signal_paths").expected_paths == (
        "layers/base/signal-path-light.svg",
        "layers/base/signal-path-dark.svg",
        f"layers/{encoded_variant}/signal-path-light.svg",
        f"layers/{encoded_variant}/signal-path-dark.svg",
    )
    assert found.report.expected_paths == ("report.txt", "report.json")
    assert found.audit_paths[:4] == (
        "config.input.yaml",
        "config.resolved.yaml",
        "provenance.json",
        "diagnostics.json",
    )
    assert "products.json" in found.audit_paths


def test_tap_preview_encodes_both_the_run_and_key_components():
    with_taps = set_output_product(
        DOCUMENT,
        "taps",
        enabled=True,
        runs=("fit / one",),
        keys=("snapshot / raw",),
    )

    found = project_output_workflow(with_taps)

    assert _product(found, "taps").expected_paths == (
        "runs/n-666974202f206f6e65/taps/n-736e617073686f74202f20726177.npz",
    )


@pytest.mark.parametrize(
    ("arguments", "state"),
    [
        ({}, "ready_new"),
        ({"target_exists": True}, "blocked_existing"),
        ({"target_exists": True, "clobber": True}, "blocked_foreign"),
        (
            {"target_exists": True, "clobber": True, "marker_id": "owned"},
            "replace_owned",
        ),
        (
            {
                "requires_recovery": True,
                "recovery_reason": "multiple transaction update temporaries",
            },
            "ambiguous_recovery",
        ),
        ({"access_reliable": False}, "blocked_unsafe"),
        ({"ancestry_reliable": False}, "blocked_unsafe"),
    ],
)
def test_every_clobber_and_recovery_state_is_explicit(arguments, state):
    found = classify_output_state(**arguments)
    assert found.state == state
    if state.startswith("blocked") or state == "ambiguous_recovery":
        assert found.message


def test_product_and_report_transitions_are_exact_revision_content_edits():
    added = set_output_product(
        DOCUMENT,
        "chains",
        enabled=True,
        format="netcdf",
        runs=("fit / one",),
    )
    parsed = yaml.safe_load(added)
    assert parsed["outputs"]["write"]["chains"] == {
        "format": "netcdf",
        "runs": ["fit / one"],
    }
    assert (
        set_output_product(
            added,
            "chains",
            enabled=True,
            format="netcdf",
            runs=("fit / one",),
        )
        == added
    )

    changed_report = set_output_report(
        added,
        enabled=True,
        rows=("fit / one", "compare"),
        columns=("seconds",),
        reference=None,
        relative=(),
        formats=("json",),
    )
    assert yaml.safe_load(changed_report)["outputs"]["report"] == {
        "rows": ["fit / one", "compare"],
        "columns": ["seconds"],
        "format": "json",
    }

    removed = set_output_product(changed_report, "chains", enabled=False)
    assert "chains" not in yaml.safe_load(removed)["outputs"]["write"]
    without_report = set_output_report(removed, enabled=False)
    assert "report" not in yaml.safe_load(without_report)["outputs"]


def test_output_transitions_refuse_unknown_selectors_and_invalid_options():
    with pytest.raises(ConfigError, match="unknown output product"):
        set_output_product(DOCUMENT, "predicted", enabled=True)
    with pytest.raises(ConfigError, match="outputs.write.chains.format"):
        set_output_product(DOCUMENT, "chains", enabled=True, format="json")
    with pytest.raises(ConfigError, match="outputs.report.reference"):
        set_output_report(
            DOCUMENT,
            enabled=True,
            rows=("fit / one",),
            columns=("mean",),
            reference="missing",
            relative=("mean_sigma",),
            formats=("json",),
        )


def test_invalid_output_grammar_remains_visible_as_an_editable_unavailable_projection():
    invalid = DOCUMENT.replace("clobber: false", "clobber: sometimes")

    found = project_output_workflow(invalid)

    assert found.requested_yaml == invalid
    assert found.state == "unavailable"
    assert "outputs.clobber" in found.state_message
    assert len(found.products) == 22


def _audit_target(tmp_path: Path) -> tuple[Path, str]:
    target = tmp_path / "result"
    target.mkdir(mode=0o700)
    marker_id = "12345678-1234-4123-8123-123456789abc"
    marker = {"format_version": 1, "run_directory_id": marker_id}
    marker_path = target / ".rheplicant-results.json"
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    marker_path.chmod(0o600)
    resolved = target / "config.resolved.yaml"
    resolved.write_text("schema_version: 1\n", encoding="utf-8")
    resolved.chmod(0o600)
    return target, marker_id


def test_audit_links_recheck_marker_identity_and_never_follow_unknown_paths(tmp_path):
    target, marker_id = _audit_target(tmp_path)
    identity = target.stat()

    artifact = read_audit_artifact(
        str(target),
        marker_id,
        "config.resolved.yaml",
        target_device=identity.st_dev,
        target_inode=identity.st_ino,
    )
    assert artifact.payload == b"schema_version: 1\n"
    assert artifact.media_type == "application/yaml"

    with pytest.raises(ConfigError, match="no longer names the completed job"):
        read_audit_artifact(
            str(target), "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa", "config.resolved.yaml"
        )
    with pytest.raises(ConfigError, match="not an allowed audit artefact"):
        read_audit_artifact(str(target), marker_id, "../secret")

    os.symlink(tmp_path / "elsewhere", target / "provenance.json")
    with pytest.raises(ConfigError, match="regular file"):
        read_audit_artifact(str(target), marker_id, "provenance.json")

    moved = tmp_path / "old-result"
    target.rename(moved)
    replacement, replacement_marker = _audit_target(tmp_path)
    assert replacement_marker == marker_id
    with pytest.raises(ConfigError, match="no longer names the completed job"):
        read_audit_artifact(
            str(replacement),
            marker_id,
            "config.resolved.yaml",
            target_device=identity.st_dev,
            target_inode=identity.st_ino,
        )

    malformed = json.loads((replacement / ".rheplicant-results.json").read_text())
    malformed["run_directory_id"] = "not-a-plan-4-marker"
    (replacement / ".rheplicant-results.json").write_text(json.dumps(malformed))
    with pytest.raises(ConfigError, match="ownership marker is malformed"):
        read_audit_artifact(str(replacement), "not-a-plan-4-marker", "config.resolved.yaml")
