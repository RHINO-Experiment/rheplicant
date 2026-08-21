"""Live, JAX-free config validation projected for the YAML editor.

This module is the GUI's explicit validation gateway into the public config
layer.  It prepares package presets through the same bounded Plan 4 pipeline
as the command, calls :func:`rheplicant.config.preflight`, and converts the
complete report into frozen browser-facing records.  It never builds a
resource, operator, state, or inference object.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import rheplicant.config as public_config
from _rheplicant_bootstrap.errors import ConfigError
from _rheplicant_bootstrap.layering import layer_presets
from _rheplicant_bootstrap.output.manager import parse_output_grammar
from _rheplicant_bootstrap.prepare import PreparedConfig, prepare_config
from _rheplicant_bootstrap.presets import read_installed_preset
from _rheplicant_bootstrap.types import SourceInput
from rheplicant.config.findings import Finding, Report
from rheplicant.gui.forms import FormProjection, project_forms
from rheplicant.radio.graph import RADIO_GRAPH

_PROCESS_SECTIONS = frozenset(("defaults", "plugins", "outputs"))
_PRESET_SECTIONS = ("runtime", "observation", "resources", "model", "inference")
_DIRECT_SECTIONS = frozenset(
    ("runtime", "observation", "variants", "inference", "runs", "outputs", "campaign")
)
_MISSING = object()


@dataclass(frozen=True, slots=True)
class LedgerFinding:
    """One complete public pre-flight finding plus its owning layer."""

    check: str
    severity: Literal["refuse", "warn", "report"]
    where: str
    message: str
    attribution: str


@dataclass(frozen=True, slots=True)
class PresetChange:
    """One effective scientific path that differs from selected presets."""

    path: str
    kind: Literal["added", "changed", "removed"]
    preset_value: object
    document_value: object


@dataclass(frozen=True, slots=True)
class SectionBadge:
    """Completeness, finding, and preset-diff counts for one form section."""

    section_id: str
    incomplete: int
    refuse: int
    warn: int
    report: int
    preset_changes: int


@dataclass(frozen=True, slots=True)
class ValidationProjection:
    """The complete free validation view of one authoritative YAML snapshot."""

    findings: tuple[LedgerFinding, ...]
    section_badges: tuple[SectionBadge, ...]
    selected_presets: tuple[str, ...]
    preset_changes: tuple[PresetChange, ...]
    run_blocked: bool


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _same(left: object, right: object) -> bool:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return (
            len(left) == len(right)
            and all(key in right and _same(value, right[key]) for key, value in left.items())
        )
    if (
        not isinstance(left, str | bytes)
        and not isinstance(right, str | bytes)
        and isinstance(left, Sequence)
        and isinstance(right, Sequence)
    ):
        return len(left) == len(right) and all(
            _same(a, b) for a, b in zip(left, right, strict=True)
        )
    return type(left) is type(right) and left == right


def _walk_diff(path: str, preset: object, document: object) -> tuple[PresetChange, ...]:
    if preset is _MISSING and document is _MISSING:
        return ()
    if preset is _MISSING:
        return (PresetChange(path, "added", None, _plain(document)),)
    if document is _MISSING:
        return (PresetChange(path, "removed", _plain(preset), None),)
    if isinstance(preset, Mapping) and isinstance(document, Mapping):
        keys = (*preset, *(key for key in document if key not in preset))
        return tuple(
            change
            for key in keys
            for change in _walk_diff(
                f"{path}.{key}" if path else str(key),
                preset.get(key, _MISSING),
                document.get(key, _MISSING),
            )
        )
    if _same(preset, document):
        return ()
    return (PresetChange(path, "changed", _plain(preset), _plain(document)),)


def _source(yaml_text: str) -> SourceInput:
    path = "/rheplicant-gui/config.yaml"
    return SourceInput(
        yaml_text.encode("utf-8", "strict"),
        path,
        path,
        path,
        "/rheplicant-gui",
        "embedded",
    )


def _prepare(yaml_text: str) -> PreparedConfig:
    return prepare_config(
        _source(yaml_text),
        preset_provider=read_installed_preset,
        parse_outputs=parse_output_grammar,
    )


def _scientific(document: Mapping[str, object]) -> dict[str, object]:
    return {key: value for key, value in document.items() if key not in _PROCESS_SECTIONS}


def _error_where(message: str) -> str:
    head = message.split(":", 1)[0]
    if head and all(character.isalnum() or character in "_.[]-'" for character in head):
        return head
    return "document"


def _as_finding(error: ConfigError) -> Finding:
    message = str(error)
    return Finding("", "refuse", _error_where(message), message)


def _report(document: Mapping[str, object], preparation_error: ConfigError | None) -> Report:
    findings: list[Finding] = []
    try:
        findings.extend(public_config.preflight(document).findings)
    except ConfigError as error:
        report = error.report
        if isinstance(report, Report):
            findings.extend(report.findings)
        else:
            findings.append(_as_finding(error))
    if preparation_error is not None:
        candidate = _as_finding(preparation_error)
        if all(row.message != candidate.message for row in findings):
            findings.insert(0, candidate)
    return Report(tuple(findings))


def _attribution(where: str) -> str:
    pieces = where.split(".", 2)
    return f"variant:{pieces[1]}" if len(pieces) > 1 and pieces[0] == "variants" else "base"


def _ledger(report: Report) -> tuple[LedgerFinding, ...]:
    return tuple(
        LedgerFinding(
            row.check,
            row.severity,  # type: ignore[arg-type]
            row.where,
            row.message,
            _attribution(row.where),
        )
        for row in report.findings
    )


def _preset_diff(
    prepared: PreparedConfig | None,
) -> tuple[tuple[str, ...], tuple[PresetChange, ...]]:
    if prepared is None:
        return (), ()
    selected = tuple(prepared.source.bootstrap_manifest.presets)
    if not selected:
        return (), ()
    snapshots = {row.request.name: row.snapshot for row in selected}
    baseline, _ = layer_presets(
        {},
        tuple(row.request for row in selected),
        preset_provider=snapshots.__getitem__,
    )
    effective = prepared.source.layered_document
    changes = tuple(
        change
        for section in _PRESET_SECTIONS
        for change in _walk_diff(
            section,
            baseline.document.get(section, _MISSING),
            effective.get(section, _MISSING),
        )
    )
    return tuple(row.request.name for row in selected), changes


def _section_for(path: str) -> str | None:
    if path.startswith("variants."):
        return "variants"
    root, _, tail = path.partition(".")
    root = root.split("[", 1)[0]
    if root in _DIRECT_SECTIONS:
        return root
    if root == "resources":
        kind = tail.split(".", 1)[0]
        if kind == "beams":
            return "beam"
        if kind in ("projectors", "sky_models"):
            return "sky"
        return "resources"
    if root == "model":
        node_id = tail.split(".", 1)[0]
        node = RADIO_GRAPH.nodes.get(node_id)
        return "backend" if node is not None and node.segment == "processing" else "instrument"
    return None


def _badges(
    forms: FormProjection,
    findings: tuple[LedgerFinding, ...],
    changes: tuple[PresetChange, ...],
) -> tuple[SectionBadge, ...]:
    return tuple(
        SectionBadge(
            section.section_id,
            sum(widget.must_decide for widget in section.widgets),
            sum(
                row.severity == "refuse"
                and _section_for(row.where) == section.section_id
                for row in findings
            ),
            sum(
                row.severity == "warn" and _section_for(row.where) == section.section_id
                for row in findings
            ),
            sum(
                row.severity == "report"
                and _section_for(row.where) == section.section_id
                for row in findings
            ),
            sum(_section_for(row.path) == section.section_id for row in changes),
        )
        for section in forms.sections
    )


def validate_document(
    yaml_text: str,
    document: Mapping[str, object],
    forms: FormProjection,
) -> ValidationProjection:
    """Run public text pre-flight and preset diff for one parsed YAML mapping."""
    prepared: PreparedConfig | None
    preparation_error: ConfigError | None = None
    try:
        prepared = _prepare(yaml_text)
    except ConfigError as error:
        prepared = None
        preparation_error = error
        target = _scientific(document)
    else:
        target = _scientific(prepared.source.layered_document)
        forms = project_forms(prepared.source.layered_document)
    findings = _ledger(_report(target, preparation_error))
    selected, changes = _preset_diff(prepared)
    return ValidationProjection(
        findings=findings,
        section_badges=_badges(forms, findings, changes),
        selected_presets=selected,
        preset_changes=changes,
        run_blocked=any(row.severity == "refuse" for row in findings),
    )


__all__ = [
    "LedgerFinding",
    "PresetChange",
    "SectionBadge",
    "ValidationProjection",
    "validate_document",
]
