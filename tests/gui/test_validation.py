from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
import yaml

import rheplicant.config as public_config
from rheplicant.gui.document import snapshot
from rheplicant.gui.validation import validate_document
from tests.config.preflight_helpers import preflight_document


def _badge(found, section_id):
    return next(badge for badge in found.section_badges if badge.section_id == section_id)


def test_every_valid_projection_calls_public_preflight_and_keeps_all_findings(monkeypatch):
    document = preflight_document(
        model={"ghost": {}},
        variants={"bad": {"model": {"phantom": {}}}},
    )
    calls = []
    real = public_config.preflight

    def counted(candidate):
        calls.append(candidate)
        return real(candidate)

    monkeypatch.setattr(public_config, "preflight", counted)
    found = snapshot(yaml.safe_dump(document, sort_keys=False)).validation

    assert len(calls) == 1
    assert tuple((row.check, row.severity) for row in found.findings) == (
        ("A2", "refuse"),
        ("A2", "refuse"),
    )
    assert found.findings[0].where == "model"
    assert found.findings[0].attribution == "base"
    assert found.findings[1].where == "variants.bad.model"
    assert found.findings[1].attribution == "variant:bad"
    assert found.run_blocked is True
    assert _badge(found, "variants").refuse == 1


def test_preflight_severities_and_messages_cross_the_gui_boundary_verbatim(monkeypatch):
    from rheplicant.config.findings import Report, refuse, report, warn

    expected = Report(
        (
            refuse("A2", "model", "first refusal"),
            warn("A7", "runs[0]", "one warning"),
            report("C20", "inference", "one report"),
        )
    )
    monkeypatch.setattr(public_config, "preflight", lambda _document: expected)
    found = snapshot(
        yaml.safe_dump(preflight_document(variants={}), sort_keys=False)
    ).validation

    assert tuple(row.message for row in found.findings) == (
        "first refusal",
        "one warning",
        "one report",
    )
    assert tuple(row.severity for row in found.findings) == (
        "refuse",
        "warn",
        "report",
    )
    assert _badge(found, "instrument").refuse == 1
    assert _badge(found, "runs").warn == 1
    assert _badge(found, "inference").report == 1


def test_selected_package_preset_has_a_leafwise_effective_diff():
    document = {
        "schema_version": 1,
        "defaults": ["rhino_v1"],
        "runtime": {"jax_enable_x64": False},
        "model": {"gain": {"type": "GainOperator", "gain": 2.0}},
        "runs": [],
    }
    found = snapshot(yaml.safe_dump(document, sort_keys=False)).validation
    by_path = {change.path: change for change in found.preset_changes}

    assert found.selected_presets == ("rhino_v1",)
    assert by_path["runtime.jax_enable_x64"].kind == "changed"
    assert by_path["runtime.jax_enable_x64"].preset_value is True
    assert by_path["runtime.jax_enable_x64"].document_value is False
    assert by_path["model.gain"].kind == "added"
    assert by_path["model.antenna_loss"].kind == "removed"
    assert _badge(found, "runtime").preset_changes == 1


def test_preset_inheritance_counts_toward_effective_completeness():
    document = {
        "schema_version": 1,
        "defaults": ["rhino_v1"],
        "model": {},
        "runs": [],
    }
    found = snapshot(yaml.safe_dump(document, sort_keys=False))

    assert "observation.freq.grid" in found.forms.missing_required
    assert _badge(found.validation, "observation").incomplete == 1


def test_findings_and_preset_changes_are_badged_to_specific_projected_sections(monkeypatch):
    from rheplicant.config.findings import Report, refuse

    document = preflight_document(variants={})
    text = yaml.safe_dump(document, sort_keys=False)
    forms = snapshot(text).forms
    report = Report(
        (
            refuse("A1", "resources.beams.horn", "beam"),
            refuse("A1", "resources.sky_models.gsm", "sky"),
            refuse("A1", "model.flagging", "backend"),
            refuse("A1", "model.foregrounds", "instrument"),
        )
    )
    monkeypatch.setattr(public_config, "preflight", lambda _document: report)
    found = validate_document(text, document, forms)

    assert _badge(found, "beam").refuse == 1
    assert _badge(found, "sky").refuse == 1
    assert _badge(found, "backend").refuse == 1
    assert _badge(found, "instrument").refuse == 1


def test_validation_projection_is_frozen_and_counts_form_completeness():
    text = yaml.safe_dump(preflight_document(variants={}), sort_keys=False)
    projected = snapshot(text)
    found = validate_document(text, yaml.safe_load(text), projected.forms)

    assert len(found.section_badges) == 12
    assert sum(badge.incomplete for badge in found.section_badges) == len(
        projected.forms.missing_required
    )
    with pytest.raises(FrozenInstanceError):
        found.run_blocked = False  # type: ignore[misc]
