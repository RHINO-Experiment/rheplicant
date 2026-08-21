"""Framework-free output workflow projection and YAML transformations.

The requested document remains authoritative.  This module projects Plan 4's
closed output grammar, read-only path preflight, predictable product paths and
the preset-merged document shown by the GUI's "what will run" tab.  It never
creates, recovers, clobbers or publishes an output directory.
"""

from __future__ import annotations

import json
import os
import stat
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from _rheplicant_bootstrap.audit.names import encode_name
from _rheplicant_bootstrap.errors import ConfigError
from _rheplicant_bootstrap.output.manager import (
    _MARKER_ID,
    _PLAN4B_WRITE,
    _PRODUCT_DEFAULT_FORMATS,
    _PRODUCT_FORMATS,
    _REPORT_COLUMNS,
    inspect_output_path,
    parse_output_grammar,
    resolve_output_request,
)
from _rheplicant_bootstrap.output.platform import platform_adapter
from _rheplicant_bootstrap.output.types import OutputPathInspection, ParsedOutputSection
from _rheplicant_bootstrap.prepare import prepare_config
from _rheplicant_bootstrap.presets import PresetSnapshot, read_installed_preset
from _rheplicant_bootstrap.types import SourceInput
from rheplicant.gui.document import _dump, _load, _plain, _same_value

OutputStateName = Literal[
    "ready_new",
    "blocked_existing",
    "blocked_foreign",
    "replace_owned",
    "ambiguous_recovery",
    "blocked_unsafe",
    "unavailable",
]

_AUDIT_MEDIA_TYPES = {
    "config.input.yaml": "application/yaml",
    "config.resolved.yaml": "application/yaml",
    "provenance.json": "application/json",
    "diagnostics.json": "application/json",
    "products.json": "application/json",
    "report.json": "application/json",
    "report.txt": "text/plain; charset=utf-8",
}
_AUDIT_READ_LIMIT = 64 * 1024 * 1024
_EXTENSIONS = {"npz": "npz", "json": "json", "txt": "txt", "netcdf": "nc"}


@dataclass(frozen=True, slots=True)
class OutputState:
    """One explicit read-only publication or recovery state."""

    state: OutputStateName
    message: str


@dataclass(frozen=True, slots=True)
class OutputProductProjection:
    """One live product selector and its predictable candidate paths."""

    name: str
    enabled: bool
    format: str
    formats: tuple[str, ...]
    runs: tuple[str, ...]
    keys: tuple[str, ...]
    themes: tuple[str, ...]
    expected_paths: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class OutputReportProjection:
    """The report table selector projected from the closed grammar."""

    enabled: bool
    rows: tuple[str, ...]
    columns: tuple[str, ...]
    reference: str | None
    relative: tuple[str, ...]
    formats: tuple[str, ...]
    expected_paths: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class OutputWorkflowProjection:
    """Serializable output UX state derived from one exact YAML document."""

    requested_yaml: str
    resolved_yaml: str
    resolution_note: str
    target_path: str | None
    state: OutputStateName
    state_message: str
    clobber: bool
    declared_runs: tuple[str, ...]
    products: tuple[OutputProductProjection, ...]
    report: OutputReportProjection
    audit_paths: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AuditArtifact:
    """A bounded identity-checked audit artefact ready for an HTTP response."""

    payload: bytes
    media_type: str


def classify_output_state(
    *,
    target_exists: bool = False,
    marker_id: str | None = None,
    clobber: bool = False,
    requires_recovery: bool = False,
    recovery_reason: str | None = None,
    access_reliable: bool = True,
    ancestry_reliable: bool = True,
) -> OutputState:
    """Name every state transition without performing recovery or publication."""
    if requires_recovery:
        detail = recovery_reason or "transaction recovery is required"
        return OutputState(
            "ambiguous_recovery",
            f"Output recovery is ambiguous ({detail}); every named path will be preserved.",
        )
    if not access_reliable or not ancestry_reliable:
        return OutputState(
            "blocked_unsafe",
            "Output ancestry or access protections could not be proved safely.",
        )
    if not target_exists:
        return OutputState("ready_new", "The target is absent and ready for a new run.")
    if not clobber:
        return OutputState(
            "blocked_existing",
            "The target exists and outputs.clobber is false.",
        )
    if marker_id is None:
        return OutputState(
            "blocked_foreign",
            "The existing target has no trusted RHEPLICANT ownership marker.",
        )
    return OutputState(
        "replace_owned",
        "The owned result target will be recoverably replaced because clobber is true.",
    )


def _source(yaml_text: str, base_dir: str) -> SourceInput:
    try:
        normalized = os.path.abspath(base_dir)
    except Exception:
        raise ConfigError("GUI output base directory cannot be normalized.") from None
    path = os.path.join(normalized, "session.yaml")
    return SourceInput(
        yaml_text.encode("utf-8", "strict"),
        path,
        path,
        path,
        normalized,
        "embedded",
    )


def _declared(document: Mapping[str, object], key: str) -> tuple[str, ...]:
    value = document.get(key, ())
    if key == "runs":
        if isinstance(value, Mapping):
            value = (value,)
        if isinstance(value, str | bytes) or not isinstance(value, Sequence):
            return ()
        return tuple(
            name
            for row in value
            if isinstance(row, Mapping)
            for name in (row.get("name"),)
            if isinstance(name, str) and name
        )
    if not isinstance(value, Mapping):
        return ()
    return tuple(name for name in value if isinstance(name, str) and name)


def _layer_roots(variants: Sequence[str]) -> tuple[str, ...]:
    return ("layers/base", *(f"layers/{encode_name(name)}" for name in variants))


def _expected_product_paths(
    name: str,
    format_: str,
    runs: Sequence[str],
    options: Mapping[str, object],
    *,
    declared_runs: Sequence[str],
    variants: Sequence[str],
) -> tuple[str, ...]:
    if name == "assembly":
        return tuple(f"{root}/assembly.json" for root in _layer_roots(variants))
    if name == "signal_paths":
        themes = tuple(options.get("themes", ("light",)))
        extension = {"svg": "svg", "html": "html", "mermaid": "mmd"}[format_]
        return tuple(
            f"{root}/signal-path-{theme}.{extension}"
            for root in _layer_roots(variants)
            for theme in themes
        )
    chosen_runs = tuple(runs) or tuple(declared_runs)
    if name == "taps":
        keys = tuple(options.get("keys", ()))
        return tuple(
            f"runs/{encode_name(run)}/taps/{encode_name(key)}.npz"
            for run in chosen_runs
            for key in keys
        )
    extension = _EXTENSIONS[format_]
    return tuple(
        f"runs/{encode_name(run)}/{name}.{extension}" for run in chosen_runs
    )


def _product_rows(
    parsed: ParsedOutputSection,
    *,
    declared_runs: Sequence[str],
    variants: Sequence[str],
) -> tuple[OutputProductProjection, ...]:
    selected = {row.name: row for row in parsed.products}
    rows: list[OutputProductProjection] = []
    for name in _PLAN4B_WRITE:
        request = selected.get(name)
        format_ = _PRODUCT_DEFAULT_FORMATS[name] if request is None else request.format
        runs = () if request is None else request.runs
        options = {} if request is None else dict(request.options)
        rows.append(
            OutputProductProjection(
                name=name,
                enabled=request is not None,
                format=format_,
                formats=tuple(_PRODUCT_FORMATS[name]),
                runs=tuple(runs),
                keys=tuple(options.get("keys", ())),
                themes=tuple(options.get("themes", ())),
                expected_paths=()
                if request is None
                else _expected_product_paths(
                    name,
                    format_,
                    runs,
                    options,
                    declared_runs=declared_runs,
                    variants=variants,
                ),
            )
        )
    return tuple(rows)


def _report_row(parsed: ParsedOutputSection) -> OutputReportProjection:
    request = parsed.report
    if request is None:
        return OutputReportProjection(False, (), _REPORT_COLUMNS, None, (), ("text",), ())
    paths = tuple("report.txt" if value == "text" else "report.json" for value in request.formats)
    return OutputReportProjection(
        True,
        request.rows,
        request.columns,
        request.reference,
        request.relative,
        request.formats,
        paths,
    )


def _inspection_state(inspection: OutputPathInspection) -> OutputState:
    return classify_output_state(
        target_exists=inspection.target.exists,
        marker_id=inspection.target.marker_id,
        clobber=inspection.request.clobber,
        requires_recovery=inspection.recovery.requires_recovery,
        recovery_reason=inspection.recovery.reason,
        access_reliable=inspection.access.reliable,
        ancestry_reliable=all(row.reliable for row in inspection.ancestry),
    )


def _resolved_preview(
    yaml_text: str,
    source: SourceInput,
    *,
    preset_provider: Callable[[str], PresetSnapshot],
) -> tuple[str, str]:
    try:
        prepared = prepare_config(
            source,
            preset_provider=preset_provider,
            parse_outputs=parse_output_grammar,
        )
    except ConfigError as error:
        return (
            yaml_text,
            "Resolved preview is unavailable until process-entry validation passes: "
            f"{error}",
        )
    plain = _plain(prepared.source.layered_document)
    if not isinstance(plain, dict):
        raise ConfigError("preset-merged GUI document is not a mapping.")
    return (
        _dump(plain),
        "Preset-merged preview; the final resolved audit file adds runtime defaults, "
        "absolute paths, hashes, shared objects and origins after a successful Run.",
    )


def project_output_workflow(
    yaml_text: str,
    *,
    base_dir: str = "/rheplicant-gui",
    preset_provider: Callable[[str], PresetSnapshot] = read_installed_preset,
) -> OutputWorkflowProjection:
    """Project requested/resolved tabs and read-only Plan 4 output state."""
    document = _load(yaml_text)
    raw_outputs = document.get("outputs", {})
    source = _source(yaml_text, base_dir)
    declared_runs = _declared(document, "runs")
    variants = _declared(document, "variants")
    try:
        parsed = parse_output_grammar(raw_outputs)
    except ConfigError as error:
        parsed = parse_output_grammar({})
        products = _product_rows(
            parsed,
            declared_runs=declared_runs,
            variants=variants,
        )
        return OutputWorkflowProjection(
            requested_yaml=yaml_text,
            resolved_yaml=yaml_text,
            resolution_note=f"Output resolution is unavailable: {error}",
            target_path=None,
            state="unavailable",
            state_message=str(error),
            clobber=False,
            declared_runs=declared_runs,
            products=products,
            report=_report_row(parsed),
            audit_paths=(
                "config.input.yaml",
                "config.resolved.yaml",
                "provenance.json",
                "diagnostics.json",
            ),
        )
    try:
        request = resolve_output_request(parsed, source=source, command="run")
    except ConfigError as error:
        request = None
        target_state = OutputState("unavailable", str(error))
    resolved_yaml, resolution_note = _resolved_preview(
        yaml_text,
        source,
        preset_provider=preset_provider,
    )
    if request is None:
        pass
    elif request.target_path is None:
        target_state = OutputState("unavailable", "This document resolves no run target.")
    else:
        try:
            target_state = _inspection_state(
                inspect_output_path(request, platform_adapter())
            )
        except ConfigError as error:
            target_state = OutputState("blocked_unsafe", str(error))
    products = _product_rows(
        parsed,
        declared_runs=declared_runs,
        variants=variants,
    )
    report = _report_row(parsed)
    audit_paths: list[str] = [
        "config.input.yaml",
        "config.resolved.yaml",
        "provenance.json",
        "diagnostics.json",
    ]
    audit_paths.extend(
        f"variants/{encode_name(name)}/config.resolved.yaml" for name in variants
    )
    if any(row.enabled for row in products) or report.enabled:
        audit_paths.append("products.json")
    audit_paths.extend(report.expected_paths)
    return OutputWorkflowProjection(
        requested_yaml=yaml_text,
        resolved_yaml=resolved_yaml,
        resolution_note=resolution_note,
        target_path=None if request is None else request.target_path,
        state=target_state.state,
        state_message=target_state.message,
        clobber=parsed.clobber,
        declared_runs=declared_runs,
        products=products,
        report=report,
        audit_paths=tuple(audit_paths),
    )


def _render_output_edit(
    yaml_text: str,
    frozen: Mapping[str, object],
    plain: dict[str, object],
) -> str:
    outputs = plain.get("outputs", {})
    parse_output_grammar(outputs)
    if _same_value(_plain(frozen), plain):
        return yaml_text
    rendered = _dump(plain)
    _load(rendered)
    return rendered


def _plain_outputs(plain: dict[str, object]) -> dict[str, object]:
    outputs = plain.setdefault("outputs", {})
    if not isinstance(outputs, dict):
        raise ConfigError("outputs: must be a mapping.")
    return outputs


def set_output_product(
    yaml_text: str,
    name: str,
    *,
    enabled: bool,
    format: str | None = None,
    runs: Sequence[str] = (),
    keys: Sequence[str] = (),
    themes: Sequence[str] = (),
) -> str:
    """Enable, configure or remove one closed scientific product request."""
    if name not in _PLAN4B_WRITE:
        raise ConfigError(f"unknown output product {name!r}.")
    if type(enabled) is not bool:
        raise ConfigError("output product enabled must be true or false.")
    frozen = _load(yaml_text)
    plain = _plain(frozen)
    if not isinstance(plain, dict):
        raise ConfigError("GUI document root must be a mapping.")
    outputs = _plain_outputs(plain)
    write = outputs.setdefault("write", {})
    if not isinstance(write, dict):
        raise ConfigError("outputs.write: must be a mapping.")
    if not enabled:
        write.pop(name, None)
        return _render_output_edit(yaml_text, frozen, plain)
    selected_format = _PRODUCT_DEFAULT_FORMATS[name] if format is None else format
    request: dict[str, object] = {}
    if selected_format != _PRODUCT_DEFAULT_FORMATS[name]:
        request["format"] = selected_format
    if runs:
        request["runs"] = list(runs)
    if keys:
        if name not in ("aux", "taps"):
            raise ConfigError(f"outputs.write.{name}.keys: is not supported.")
        request["keys"] = list(keys)
    if themes:
        if name != "signal_paths":
            raise ConfigError(f"outputs.write.{name}.themes: is not supported.")
        request["themes"] = list(themes)
    write[name] = request or True
    return _render_output_edit(yaml_text, frozen, plain)


def set_output_report(
    yaml_text: str,
    *,
    enabled: bool,
    rows: Sequence[str] = (),
    columns: Sequence[str] = _REPORT_COLUMNS,
    reference: str | None = None,
    relative: Sequence[str] = (),
    formats: Sequence[str] = ("text",),
) -> str:
    """Configure or remove the deterministic report table request."""
    if type(enabled) is not bool:
        raise ConfigError("output report enabled must be true or false.")
    frozen = _load(yaml_text)
    plain = _plain(frozen)
    if not isinstance(plain, dict):
        raise ConfigError("GUI document root must be a mapping.")
    outputs = _plain_outputs(plain)
    if not enabled:
        outputs.pop("report", None)
        return _render_output_edit(yaml_text, frozen, plain)
    request: dict[str, object] = {
        "rows": list(rows),
        "columns": list(columns),
    }
    if reference is not None:
        request["reference"] = reference
    if relative:
        request["relative"] = list(relative)
    request["format"] = formats[0] if len(formats) == 1 else list(formats)
    outputs["report"] = request
    return _render_output_edit(yaml_text, frozen, plain)


def _read_fd(fd: int, *, maximum: int, where: str) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = os.read(fd, min(1024 * 1024, maximum + 1 - total))
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)
        total += len(chunk)
        if total > maximum:
            raise ConfigError(f"{where} exceeds the GUI download limit.")


def _read_regular_at(directory_fd: int, name: str, *, maximum: int) -> bytes:
    try:
        row = os.lstat(name, dir_fd=directory_fd)
    except OSError as error:
        raise ConfigError(f"audit artefact {name!r} is unavailable: {error}.") from None
    if not stat.S_ISREG(row.st_mode):
        raise ConfigError(f"audit artefact {name!r} is not a regular file.")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(name, flags, dir_fd=directory_fd)
    except OSError as error:
        raise ConfigError(f"audit artefact {name!r} cannot be opened safely: {error}.") from None
    try:
        opened = os.fstat(fd)
        if (opened.st_dev, opened.st_ino) != (row.st_dev, row.st_ino):
            raise ConfigError(f"audit artefact {name!r} changed while it was opened.")
        return _read_fd(fd, maximum=maximum, where=f"audit artefact {name!r}")
    finally:
        os.close(fd)


def _marker_id(directory_fd: int) -> str:
    payload = _read_regular_at(
        directory_fd,
        ".rheplicant-results.json",
        maximum=4096,
    )
    try:
        value = json.loads(payload)
    except Exception:
        raise ConfigError("audit result ownership marker is malformed.") from None
    if (
        type(value) is not dict
        or tuple(sorted(value)) != ("format_version", "run_directory_id")
        or value.get("format_version") != 1
        or type(value.get("run_directory_id")) is not str
        or _MARKER_ID.fullmatch(value["run_directory_id"]) is None
    ):
        raise ConfigError("audit result ownership marker is malformed.")
    return value["run_directory_id"]


def _open_target(target_path: str) -> int:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        return os.open(target_path, flags)
    except OSError as error:
        raise ConfigError(f"audit result directory cannot be opened safely: {error}.") from None


def read_audit_artifact(
    target_path: str,
    marker_id: str,
    relative_path: str,
    *,
    target_device: int | None = None,
    target_inode: int | None = None,
) -> AuditArtifact:
    """Read one flat audit file only while the completed job still owns target."""
    media_type = _AUDIT_MEDIA_TYPES.get(relative_path)
    if media_type is None:
        raise ConfigError(f"{relative_path!r} is not an allowed audit artefact.")
    directory_fd = _open_target(target_path)
    try:
        identity = os.fstat(directory_fd)
        if (
            target_device is not None
            and target_inode is not None
            and (identity.st_dev, identity.st_ino) != (target_device, target_inode)
        ):
            raise ConfigError(
                "The output target no longer names the completed job; refresh before opening it."
            )
        if _marker_id(directory_fd) != marker_id:
            raise ConfigError(
                "The output target no longer names the completed job; refresh before opening it."
            )
        payload = _read_regular_at(
            directory_fd,
            relative_path,
            maximum=_AUDIT_READ_LIMIT,
        )
    finally:
        os.close(directory_fd)
    return AuditArtifact(payload, media_type)


def output_summary_at_path(target: str | None) -> dict[str, object]:
    """Capture an identity-bound summary for one completed audit directory."""
    summary: dict[str, object] = {
        "target_path": target,
        "marker_id": None,
        "target_device": None,
        "target_inode": None,
        "audit_files": [],
    }
    if target is None:
        return summary
    try:
        directory_fd = _open_target(target)
    except ConfigError:
        return summary
    try:
        identity = os.fstat(directory_fd)
        try:
            marker = _marker_id(directory_fd)
        except ConfigError:
            return summary
        files: list[str] = []
        for name in _AUDIT_MEDIA_TYPES:
            try:
                row = os.lstat(name, dir_fd=directory_fd)
            except OSError:
                continue
            if stat.S_ISREG(row.st_mode):
                files.append(name)
        summary["marker_id"] = marker
        summary["target_device"] = identity.st_dev
        summary["target_inode"] = identity.st_ino
        summary["audit_files"] = files
        return summary
    finally:
        os.close(directory_fd)


def completed_output_summary(
    yaml_text: str,
    *,
    base_dir: str = "/rheplicant-gui",
) -> dict[str, object]:
    """Capture identity-bound flat audit links after a successful Plan 4 run."""
    document = _load(yaml_text)
    parsed = parse_output_grammar(document.get("outputs", {}))
    request = resolve_output_request(
        parsed,
        source=_source(yaml_text, base_dir),
        command="run",
    )
    return output_summary_at_path(request.target_path)


__all__ = [
    "AuditArtifact",
    "OutputProductProjection",
    "OutputReportProjection",
    "OutputState",
    "OutputWorkflowProjection",
    "classify_output_state",
    "completed_output_summary",
    "output_summary_at_path",
    "project_output_workflow",
    "read_audit_artifact",
    "set_output_product",
    "set_output_report",
]
