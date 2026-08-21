"""Closed output grammar, descriptor preflight, and A34 lease management."""

from __future__ import annotations

import fcntl
import json
import os
import re
import stat
import threading
from collections.abc import Mapping
from typing import cast

from _rheplicant_bootstrap.errors import ConfigError
from _rheplicant_bootstrap.frozen import static_isinstance
from _rheplicant_bootstrap.types import SourceInput

from .paths import (
    decode_journal_temp,
    internal_names,
    journal_name,
    lock_name,
    require_component_budget,
    target_digest,
)
from .platform import OutputPlatform
from .types import (
    AncestorEntryInspection,
    OutputLease,
    OutputMarker,
    OutputPathInspection,
    OutputRequest,
    ParsedOutputSection,
    ProductRequest,
    PublicationLease,
    RecoveryInspection,
    ReportRequest,
    TargetIdentity,
    VerifiedOutputLease,
)

_OUTPUT_KEYS = ("dir", "clobber", "stdout", "write", "report")
_WRITE_KEYS = ("config", "provenance", "diagnostics")
_PLAN4B_WRITE = (
    "arrays",
    "aux",
    "taps",
    "assembly",
    "estimates",
    "parameters",
    "draws",
    "losses",
    "gradients",
    "covariance",
    "prediction_bands",
    "posterior_predictives",
    "identifiability",
    "scores",
    "recovery",
    "training_history",
    "timings",
    "refusals",
    "signal_paths",
    "compare",
    "benchmark",
    "chains",
)
_PLAN4B_TOP = ("memory_archive", "posterior_net", "campaign")
_PRODUCT_DEFAULT_FORMATS = {
    "arrays": "npz",
    "aux": "npz",
    "taps": "npz",
    "assembly": "json",
    "estimates": "npz",
    "parameters": "npz",
    "draws": "npz",
    "losses": "npz",
    "gradients": "npz",
    "covariance": "npz",
    "prediction_bands": "npz",
    "posterior_predictives": "npz",
    "identifiability": "json",
    "scores": "npz",
    "recovery": "json",
    "training_history": "npz",
    "timings": "json",
    "refusals": "txt",
    "signal_paths": "svg",
    "compare": "json",
    "benchmark": "json",
    "chains": "npz",
}
_PRODUCT_FORMATS = {
    **{name: (format_,) for name, format_ in _PRODUCT_DEFAULT_FORMATS.items()},
    "signal_paths": ("svg", "html", "mermaid"),
    "chains": ("npz", "netcdf"),
}
_REPORT_COLUMNS = ("mean", "std", "seconds")
_REPORT_RELATIVE = ("mean_sigma", "width_ratio")
_REPORT_FORMATS = ("text", "json")
_STDOUT = ("none", "summary", "verbose")
_COMMANDS = ("validate", "run", "script")
_MARKER_NAME = ".rheplicant-results.json"
_MARKER_ID = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}"
)
_OPEN_DIRECTORY = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
)
_STATE_LOCK = threading.Lock()
_CLOSED_LEASES: set[int] = set()
_LEASE_PLATFORMS: dict[int, int] = {}
_INSPECTION_PLATFORMS: dict[int, int] = {}


def _failure_budget_names(target_name: str) -> tuple[str, str]:
    stamp = "99999999T999999.999999Z"
    process = os.getpid()
    return (
        f"{target_name}.refused-{stamp}-{process}",
        f"{target_name}.error-{stamp}-{process}",
    )


def _mapping(value: object, *, where: str) -> Mapping[object, object]:
    if not static_isinstance(value, Mapping):
        raise ConfigError(f"{where}: must be a mapping.")
    return cast(Mapping[object, object], value)


def _closed_mapping(
    value: object,
    *,
    where: str,
    allowed: tuple[str, ...],
) -> dict[str, object]:
    raw = _mapping(value, where=where)
    result: dict[str, object] = {}
    try:
        pairs = tuple(raw.items())
    except Exception:
        raise ConfigError(f"{where}: mapping traversal failed.") from None
    for key, item in pairs:
        if not static_isinstance(key, str):
            raise ConfigError(f"{where}: keys must be strings.")
        name = str.__str__(key)
        if name in result:
            raise ConfigError(f"{where}: keys collide after canonicalization.")
        if name not in allowed:
            raise ConfigError(f"{where}.{name}: unknown key")
        result[name] = item
    return result


def _required_true(value: object, *, where: str) -> bool:
    if not static_isinstance(value, bool):
        raise ConfigError(f"{where}: must be true.")
    if value is not True:
        raise ConfigError(f"{where}: is mandatory in Config Plan 4A and cannot be false")
    return True


def _unique_texts(
    value: object,
    *,
    where: str,
    allowed: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    if type(value) not in (list, tuple) or not value:
        raise ConfigError(f"{where}: must be a non-empty list of unique strings.")
    result: list[str] = []
    for index, item in enumerate(value):
        if not static_isinstance(item, str) or not str.__str__(item):
            raise ConfigError(f"{where}[{index}]: must be a non-empty string.")
        text = str.__str__(item)
        if allowed is not None and text not in allowed:
            raise ConfigError(f"{where}[{index}]: must be one of {list(allowed)}.")
        if text in result:
            raise ConfigError(f"{where}: entries must be unique.")
        result.append(text)
    return tuple(result)


def _product_request(name: str, value: object) -> ProductRequest:
    where = f"outputs.write.{name}"
    if static_isinstance(value, bool):
        if value is not True:
            raise ConfigError(f"{where}: must be true or a mapping.")
        raw: dict[str, object] = {}
    else:
        allowed = ("format", "runs")
        if name in ("aux", "taps"):
            allowed += ("keys",)
        if name == "signal_paths":
            allowed += ("themes",)
        try:
            raw = _closed_mapping(value, where=where, allowed=allowed)
        except ConfigError as error:
            if str(error) == f"{where}: must be a mapping.":
                raise ConfigError(f"{where}: must be true or a mapping.") from None
            raise

    raw_format = raw.get("format", _PRODUCT_DEFAULT_FORMATS[name])
    if (
        not static_isinstance(raw_format, str)
        or str.__str__(raw_format) not in _PRODUCT_FORMATS[name]
    ):
        raise ConfigError(
            f"{where}.format: must be one of {list(_PRODUCT_FORMATS[name])}."
        )
    format_ = str.__str__(raw_format)
    runs = ()
    if "runs" in raw:
        runs = _unique_texts(raw["runs"], where=f"{where}.runs")
    options: list[tuple[str, object]] = []
    if "keys" in raw:
        options.append(
            ("keys", _unique_texts(raw["keys"], where=f"{where}.keys"))
        )
    if "themes" in raw:
        options.append(
            (
                "themes",
                _unique_texts(
                    raw["themes"],
                    where=f"{where}.themes",
                    allowed=("light", "dark"),
                ),
            )
        )
    return ProductRequest(name, format_, runs, tuple(options))


def _report_request(value: object) -> ReportRequest:
    where = "outputs.report"
    raw = _closed_mapping(
        value,
        where=where,
        allowed=("rows", "columns", "reference", "relative", "format"),
    )
    if "rows" not in raw:
        raise ConfigError("outputs.report.rows: is required.")
    rows = _unique_texts(raw["rows"], where="outputs.report.rows")
    columns = _unique_texts(
        raw.get("columns", list(_REPORT_COLUMNS)),
        where="outputs.report.columns",
        allowed=_REPORT_COLUMNS,
    )
    reference = raw.get("reference")
    if reference is not None:
        if not static_isinstance(reference, str) or not str.__str__(reference):
            raise ConfigError("outputs.report.reference: must be a non-empty string.")
        reference = str.__str__(reference)
        if reference not in rows:
            raise ConfigError("outputs.report.reference: must name one of outputs.report.rows.")
    relative: tuple[str, ...] = ()
    if "relative" in raw:
        relative = _unique_texts(
            raw["relative"],
            where="outputs.report.relative",
            allowed=_REPORT_RELATIVE,
        )
        if reference is None:
            raise ConfigError("outputs.report.reference: is required for relative columns.")
    raw_formats = raw.get("format", "text")
    if static_isinstance(raw_formats, str):
        formats = _unique_texts(
            [str.__str__(raw_formats)],
            where="outputs.report.format",
            allowed=_REPORT_FORMATS,
        )
    else:
        formats = _unique_texts(
            raw_formats,
            where="outputs.report.format",
            allowed=_REPORT_FORMATS,
        )
    return ReportRequest(rows, columns, cast(str | None, reference), relative, formats)


def parse_output_grammar(raw_outputs: object) -> ParsedOutputSection:
    """Parse and detach only the raw top-level ``outputs`` value."""
    top_allowed = (*_OUTPUT_KEYS, *_PLAN4B_TOP)
    top = _closed_mapping(raw_outputs, where="outputs", allowed=top_allowed)
    for name in _PLAN4B_TOP:
        if name in top:
            raise ConfigError(f"outputs.{name}: capability is reserved for Config Plan 4B")

    directory = top.get("dir")
    if "dir" in top:
        if not static_isinstance(directory, str) or not str.__str__(directory):
            raise ConfigError("outputs.dir: must be a non-empty string.")
        directory = str.__str__(directory)
        if "\0" in directory:
            raise ConfigError("outputs.dir: contains NUL.")

    clobber = top.get("clobber", False)
    if not static_isinstance(clobber, bool):
        raise ConfigError("outputs.clobber: must be a bool.")
    clobber = bool(clobber)

    stdout = top.get("stdout", "summary")
    if not static_isinstance(stdout, str) or str.__str__(stdout) not in _STDOUT:
        raise ConfigError(f"outputs.stdout: must be one of {list(_STDOUT)}.")
    stdout = str.__str__(stdout)

    write = _closed_mapping(
        top.get("write", {}),
        where="outputs.write",
        allowed=(*_WRITE_KEYS, *_PLAN4B_WRITE),
    )
    products = tuple(
        _product_request(name, value)
        for name, value in write.items()
        if name in _PLAN4B_WRITE
    )

    write_config = _required_true(write.get("config", True), where="outputs.write.config")
    write_provenance = _required_true(
        write.get("provenance", True),
        where="outputs.write.provenance",
    )
    diagnostics = write.get("diagnostics", True)
    if static_isinstance(diagnostics, bool):
        _required_true(diagnostics, where="outputs.write.diagnostics")
    else:
        diagnostic_mapping = _closed_mapping(
            diagnostics,
            where="outputs.write.diagnostics",
            allowed=("format",),
        )
        if tuple(diagnostic_mapping) != ("format",) or diagnostic_mapping["format"] != "json":
            raise ConfigError("outputs.write.diagnostics.format: must be 'json'.")
    return ParsedOutputSection(
        directory=cast(str | None, directory),
        clobber=clobber,
        stdout=cast(str, stdout),
        write_config=write_config,
        write_provenance=write_provenance,
        write_diagnostics="json",
        products=products,
        report=_report_request(top["report"]) if "report" in top else None,
    )


def resolve_output_request(
    parsed: ParsedOutputSection,
    *,
    source: SourceInput,
    command: str,
) -> OutputRequest:
    """Resolve lexical output path facts without following a symlink."""
    if type(parsed) is not ParsedOutputSection or type(source) is not SourceInput:
        raise ConfigError("output resolution requires exact parsed and source records.")
    if command not in _COMMANDS:
        raise ConfigError(f"unknown output command {command!r}.")
    explicit = parsed.directory is not None
    raw_target = parsed.directory
    if raw_target is None:
        if command == "validate":
            target = None
        elif source.source_path == "<stdin>":
            if command == "run":
                raise ConfigError("outputs.dir: run from stdin requires an explicit directory.")
            target = None
        else:
            stem, _suffix = os.path.splitext(source.source_path)
            target = stem + ".results"
    else:
        try:
            expanded = os.path.expandvars(os.path.expanduser(raw_target))
        except Exception:
            raise ConfigError("outputs.dir: cannot expand path.") from None
        if "\0" in expanded:
            raise ConfigError("outputs.dir: contains NUL.")
        if not expanded or os.path.basename(expanded) == "":
            raise ConfigError("outputs.dir: must end in a non-empty component.")
        target = expanded if os.path.isabs(expanded) else os.path.join(source.base_dir, expanded)
    if target is not None:
        try:
            target = os.path.abspath(target)
        except Exception:
            raise ConfigError("outputs.dir: cannot normalize path.") from None
        if target == os.path.abspath(os.sep):
            raise ConfigError("outputs.dir: filesystem root is not an output target.")
        if target == os.path.abspath(source.base_dir):
            raise ConfigError("outputs.dir: cannot equal the configuration base directory.")
    return OutputRequest(
        command=cast(str, command),
        target_path=target,
        explicit_dir=explicit,
        clobber=parsed.clobber,
        stdout=parsed.stdout,
        write_config=parsed.write_config,
        write_provenance=parsed.write_provenance,
        write_diagnostics=parsed.write_diagnostics,
        products=parsed.products,
        report=parsed.report,
    )


def parse_output_request(
    document: Mapping[str, object],
    *,
    source: SourceInput,
    command: str,
) -> OutputRequest:
    if not static_isinstance(document, Mapping):
        raise ConfigError("document: configuration root must be a mapping.")
    try:
        raw = document.get("outputs", {})
    except Exception:
        raise ConfigError("document: cannot read outputs section.") from None
    return resolve_output_request(
        parse_output_grammar(raw),
        source=source,
        command=command,
    )


def _same_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return left.st_dev == right.st_dev and left.st_ino == right.st_ino


def _require_entry(row: AncestorEntryInspection) -> None:
    if not row.reliable:
        raise ConfigError(row.reason or "cannot verify ancestor access control.")
    if not row.rename_protected:
        raise ConfigError(row.reason or "ancestor entry is not rename-protected.")


def _open_child(parent_fd: int, name: str, before: os.stat_result) -> int:
    try:
        child_fd = os.open(name, _OPEN_DIRECTORY, dir_fd=parent_fd)
        after = os.fstat(child_fd)
    except OSError:
        raise ConfigError(f"output ancestor {name!r} cannot be opened safely.") from None
    if not _same_identity(before, after):
        os.close(child_fd)
        raise ConfigError(f"output ancestor {name!r} changed during descriptor walk.")
    return child_fd


def _component_limit(fd: int) -> int:
    try:
        value = os.fpathconf(fd, "PC_NAME_MAX")
    except (OSError, ValueError):
        raise ConfigError("cannot obtain output filesystem NAME_MAX from directory fd.") from None
    if type(value) is not int or value <= 0:
        raise ConfigError("output filesystem reported an invalid NAME_MAX.")
    return value


def _target_identity(parent_fd: int, target_name: str) -> TargetIdentity:
    try:
        row = os.lstat(target_name, dir_fd=parent_fd)
    except FileNotFoundError:
        return TargetIdentity(False, None, None, None)
    except OSError:
        raise ConfigError("cannot inspect output target.") from None
    if stat.S_ISLNK(row.st_mode):
        raise ConfigError("output target is a symlink.")
    if not stat.S_ISDIR(row.st_mode):
        raise ConfigError("output target exists and is not a directory.")
    return TargetIdentity(True, row.st_dev, row.st_ino, None)


def _recovery_inspection(parent_fd: int, absolute_target: str) -> RecoveryInspection:
    canonical = journal_name(absolute_target)
    digest = target_digest(absolute_target)
    prefix = f".rheplicant-jtmp-{digest}-"
    try:
        names = tuple(sorted(os.listdir(parent_fd)))
    except OSError:
        raise ConfigError("cannot inspect output transaction state.") from None
    canonical_present = canonical in names
    candidates = tuple(name for name in names if name.startswith(prefix))
    decoded = tuple(name for name in candidates if decode_journal_temp(absolute_target, name))
    reason = None
    if len(decoded) != len(candidates):
        reason = "illegal transaction update temporary"
    elif len(decoded) > 1:
        reason = "multiple transaction update temporaries"
    elif decoded and not canonical_present:
        reason = "transaction update exists without canonical journal"
    return RecoveryInspection(
        canonical_present,
        candidates,
        canonical_present or bool(candidates),
        reason,
    )


def inspect_output_path(
    request: OutputRequest,
    platform: OutputPlatform,
) -> OutputPathInspection:
    """Return facts from a read-only descriptor walk; perform no mutation."""
    if type(request) is not OutputRequest or request.target_path is None:
        raise ConfigError("output inspection requires an exact request with a target.")
    absolute = request.target_path
    if not os.path.isabs(absolute):
        raise ConfigError("output inspection requires an absolute target.")
    parent_path = os.path.dirname(absolute)
    target_name = os.path.basename(absolute)
    components = tuple(component for component in parent_path.split(os.sep) if component)
    current_fd = os.open(os.sep, _OPEN_DIRECTORY)
    current_path = os.sep
    ancestry: list[AncestorEntryInspection] = []
    missing: tuple[str, ...] = ()
    try:
        for index, component in enumerate(components):
            try:
                before = os.lstat(component, dir_fd=current_fd)
            except FileNotFoundError:
                missing = components[index:]
                break
            except OSError:
                raise ConfigError(f"cannot inspect output ancestor {component!r}.") from None
            if stat.S_ISLNK(before.st_mode):
                raise ConfigError(f"output path contains intermediate symlink {component!r}.")
            if not stat.S_ISDIR(before.st_mode):
                raise ConfigError(f"output ancestor {component!r} is not a directory.")
            entry = platform.inspect_ancestor_entry(
                current_fd,
                current_path,
                component,
                before,
            )
            _require_entry(entry)
            child_fd = _open_child(current_fd, component, before)
            os.close(current_fd)
            current_fd = child_fd
            current_path = os.path.join(current_path, component)
            ancestry.append(entry)
        nearest = current_path
        access = platform.inspect_access(current_fd, current_path)
        limit = _component_limit(current_fd)
        require_component_budget(
            (
                *missing,
                target_name,
                *internal_names(absolute),
                *_failure_budget_names(target_name),
            ),
            limit,
        )
        if missing:
            target = TargetIdentity(False, None, None, None)
            recovery = RecoveryInspection(False, (), False, None)
        else:
            target = _target_identity(current_fd, target_name)
            recovery = _recovery_inspection(current_fd, absolute)
        inspection = OutputPathInspection(
            request,
            absolute,
            nearest,
            missing,
            parent_path,
            target_name,
            target,
            access,
            tuple(ancestry),
            recovery,
            limit,
        )
    finally:
        os.close(current_fd)
    with _STATE_LOCK:
        _INSPECTION_PLATFORMS[id(inspection)] = id(platform)
    return inspection


def _validate_lock(fd: int, parent_fd: int, name: str) -> None:
    try:
        descriptor = os.fstat(fd)
        lexical = os.lstat(name, dir_fd=parent_fd)
    except OSError:
        raise ConfigError("cannot verify persistent output lock.") from None
    if (
        not _same_identity(descriptor, lexical)
        or not stat.S_ISREG(descriptor.st_mode)
        or descriptor.st_uid != os.geteuid()
        or stat.S_IMODE(descriptor.st_mode) != 0o600
        or descriptor.st_nlink != 1
    ):
        raise ConfigError("persistent output lock has insecure identity, owner, mode, or links.")


def _acquire_lock(parent_fd: int, name: str) -> int:
    base_flags = os.O_RDWR | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    created = False
    try:
        try:
            lock_fd = os.open(
                name,
                base_flags | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=parent_fd,
            )
            created = True
        except FileExistsError:
            lock_fd = os.open(name, base_flags, dir_fd=parent_fd)
        if created:
            os.fchmod(lock_fd, 0o600)
            os.fsync(lock_fd)
            os.fsync(parent_fd)
        _validate_lock(lock_fd, parent_fd, name)
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        _validate_lock(lock_fd, parent_fd, name)
        return lock_fd
    except ConfigError:
        if "lock_fd" in locals():
            os.close(lock_fd)
        raise
    except OSError:
        if "lock_fd" in locals():
            os.close(lock_fd)
        raise ConfigError("cannot acquire secure persistent output lock.") from None


def acquire_output_lease(
    inspection: OutputPathInspection,
    platform: OutputPlatform,
) -> OutputLease:
    """For run only, create permitted parents and acquire the persistent lock."""
    if type(inspection) is not OutputPathInspection or inspection.request.command != "run":
        raise ConfigError("only run output inspections can acquire a lease.")
    with _STATE_LOCK:
        expected_platform = _INSPECTION_PLATFORMS.get(id(inspection))
    if expected_platform != id(platform):
        raise ConfigError("output inspection and lease require the same platform adapter.")

    absolute = inspection.absolute_target
    parent_components = tuple(
        component for component in inspection.parent_path.split(os.sep) if component
    )
    existing_count = len(parent_components) - len(inspection.missing_components)
    current_fd = os.open(os.sep, _OPEN_DIRECTORY)
    current_path = os.sep
    ancestry: list[AncestorEntryInspection] = []
    lock_fd = -1
    try:
        for index, component in enumerate(parent_components[:existing_count]):
            before = os.lstat(component, dir_fd=current_fd)
            if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
                raise ConfigError("output ancestry changed after inspection.")
            entry = platform.inspect_ancestor_entry(current_fd, current_path, component, before)
            _require_entry(entry)
            if index >= len(inspection.ancestry) or entry != inspection.ancestry[index]:
                raise ConfigError("output ancestry changed after inspection.")
            child_fd = _open_child(current_fd, component, before)
            os.close(current_fd)
            current_fd = child_fd
            current_path = os.path.join(current_path, component)
            ancestry.append(entry)

        limit = _component_limit(current_fd)
        if limit != inspection.component_limit:
            raise ConfigError("output filesystem NAME_MAX changed after inspection.")
        require_component_budget(
            (
                *inspection.missing_components,
                inspection.target_name,
                *internal_names(absolute),
                *_failure_budget_names(inspection.target_name),
            ),
            limit,
        )
        for component in inspection.missing_components:
            try:
                os.mkdir(component, 0o700, dir_fd=current_fd)
            except FileExistsError:
                raise ConfigError("output parent appeared after inspection.") from None
            before = os.lstat(component, dir_fd=current_fd)
            child_fd = _open_child(current_fd, component, before)
            os.fchmod(child_fd, 0o700)
            os.fsync(child_fd)
            os.fsync(current_fd)
            entry = platform.inspect_ancestor_entry(current_fd, current_path, component, before)
            _require_entry(entry)
            os.close(current_fd)
            current_fd = child_fd
            current_path = os.path.join(current_path, component)
            ancestry.append(entry)
            if _component_limit(current_fd) != limit:
                raise ConfigError("created output parent changed the leased NAME_MAX.")

        if current_path != inspection.parent_path:
            raise ConfigError("output parent walk did not reach the requested directory.")
        chosen_lock_name = lock_name(absolute)
        chosen_journal_name = journal_name(absolute)
        lock_fd = _acquire_lock(current_fd, chosen_lock_name)
        lease = OutputLease(
            inspection.request,
            current_fd,
            current_path,
            inspection.target_name,
            lock_fd,
            chosen_lock_name,
            chosen_journal_name,
            tuple(ancestry),
            limit,
        )
        current_fd = -1
        lock_fd = -1
    except ConfigError:
        raise
    except OSError:
        raise ConfigError("cannot acquire output lease safely.") from None
    finally:
        if lock_fd >= 0:
            os.close(lock_fd)
        if current_fd >= 0:
            os.close(current_fd)
    with _STATE_LOCK:
        _LEASE_PLATFORMS[id(lease)] = id(platform)
        _CLOSED_LEASES.discard(id(lease))
    return lease


def require_open_output_lease(
    lease: OutputLease,
    platform: OutputPlatform | None = None,
) -> None:
    if type(lease) is not OutputLease:
        raise ConfigError("output operation requires an exact OutputLease.")
    with _STATE_LOCK:
        closed = id(lease) in _CLOSED_LEASES
        expected_platform = _LEASE_PLATFORMS.get(id(lease))
    if closed or expected_platform is None:
        raise ConfigError("output lease is closed.")
    if platform is not None and expected_platform != id(platform):
        raise ConfigError("output operation requires the lease platform adapter.")
    try:
        os.fstat(lease.parent_fd)
        os.fstat(lease.lock_fd)
    except OSError:
        raise ConfigError("output lease descriptors are closed.") from None


def close_output_lease(lease: OutputLease) -> None:
    if type(lease) is not OutputLease:
        raise ConfigError("close requires an exact OutputLease.")
    with _STATE_LOCK:
        identity = id(lease)
        if identity in _CLOSED_LEASES:
            return
        _CLOSED_LEASES.add(identity)
        _LEASE_PLATFORMS.pop(identity, None)
    try:
        fcntl.flock(lease.lock_fd, fcntl.LOCK_UN)
    except OSError:
        pass
    for fd in (lease.lock_fd, lease.parent_fd):
        try:
            os.close(fd)
        except OSError:
            pass


def revalidate_output_ancestry(lease: OutputLease, platform: OutputPlatform) -> None:
    """Rewalk root-to-parent and compare it with the held parent descriptor."""
    require_open_output_lease(lease, platform)
    components = tuple(component for component in lease.parent_path.split(os.sep) if component)
    current_fd = os.open(os.sep, _OPEN_DIRECTORY)
    current_path = os.sep
    observed: list[AncestorEntryInspection] = []
    try:
        for component in components:
            before = os.lstat(component, dir_fd=current_fd)
            if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
                raise ConfigError("leased output ancestry was replaced.")
            entry = platform.inspect_ancestor_entry(current_fd, current_path, component, before)
            _require_entry(entry)
            child_fd = _open_child(current_fd, component, before)
            os.close(current_fd)
            current_fd = child_fd
            current_path = os.path.join(current_path, component)
            observed.append(entry)
        if tuple(observed) != lease.ancestry or not _same_identity(
            os.fstat(current_fd), os.fstat(lease.parent_fd)
        ):
            raise ConfigError("leased output ancestry was replaced.")
    except OSError:
        raise ConfigError("cannot revalidate leased output ancestry.") from None
    finally:
        os.close(current_fd)


def verify_publication_under_lease(
    lease: OutputLease,
    platform: OutputPlatform,
) -> PublicationLease:
    """After recovery, re-prove ancestry/access/no-replace and NAME_MAX."""
    revalidate_output_ancestry(lease, platform)
    access = platform.inspect_access(lease.parent_fd, lease.parent_path)
    if not access.reliable:
        raise ConfigError(access.reason or "cannot verify access control.")
    if access.owner_uid != access.effective_uid:
        raise ConfigError("output parent has the wrong effective uid owner.")
    if access.mode & 0o022:
        raise ConfigError("output parent is group or other writable.")
    if not access.access_acl_is_trivial or not access.default_acl_is_trivial:
        raise ConfigError(access.reason or "output parent has non-trivial access control.")
    limit = _component_limit(lease.parent_fd)
    if limit != lease.component_limit:
        raise ConfigError("leased output NAME_MAX changed after recovery.")
    platform.verify_rename_noreplace_available(lease.parent_fd)
    return PublicationLease(lease, limit)


def _read_owned_marker(parent_fd: int, target_name: str) -> tuple[os.stat_result, OutputMarker]:
    target_fd = -1
    marker_fd = -1
    try:
        target_before = os.lstat(target_name, dir_fd=parent_fd)
        if target_before.st_uid != os.geteuid() or stat.S_IMODE(target_before.st_mode) != 0o700:
            raise ConfigError("output target ownership or mode cannot authorize clobber.")
        target_fd = _open_child(parent_fd, target_name, target_before)
        marker_stat = os.lstat(_MARKER_NAME, dir_fd=target_fd)
        if stat.S_ISLNK(marker_stat.st_mode):
            raise ConfigError("ownership marker is a symlink.")
        if not stat.S_ISREG(marker_stat.st_mode):
            raise ConfigError("ownership marker is not a regular file.")
        if (
            marker_stat.st_uid != os.geteuid()
            or stat.S_IMODE(marker_stat.st_mode) != 0o600
            or marker_stat.st_nlink != 1
        ):
            raise ConfigError("ownership marker has insecure owner, mode, or links.")
        if marker_stat.st_size > 4096:
            raise ConfigError("ownership marker must be at most 4096 bytes.")
        marker_fd = os.open(
            _MARKER_NAME,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
            dir_fd=target_fd,
        )
        chunks = []
        remaining = 4097
        while remaining:
            chunk = os.read(marker_fd, remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
        after = os.fstat(marker_fd)
        if (
            not _same_identity(marker_stat, after)
            or (marker_stat.st_size, marker_stat.st_mtime_ns, marker_stat.st_ctime_ns)
            != (after.st_size, after.st_mtime_ns, after.st_ctime_ns)
            or len(data) != marker_stat.st_size
            or len(data) > 4096
        ):
            raise ConfigError("ownership marker changed while reading.")
        try:
            decoded = json.loads(data)
        except (UnicodeError, json.JSONDecodeError):
            raise ConfigError("ownership marker is malformed.") from None
        if type(decoded) is not dict or tuple(sorted(decoded)) != (
            "format_version",
            "run_directory_id",
        ):
            raise ConfigError("ownership marker has foreign fields.")
        marker = OutputMarker(decoded["format_version"], decoded["run_directory_id"])
        if (
            type(marker.format_version) is not int
            or marker.format_version != 1
            or type(marker.run_directory_id) is not str
            or _MARKER_ID.fullmatch(marker.run_directory_id) is None
        ):
            raise ConfigError("ownership marker has an unsupported format or id.")
        expected = (
            json.dumps(
                {
                    "format_version": marker.format_version,
                    "run_directory_id": marker.run_directory_id,
                },
                sort_keys=True,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode()
        if data != expected:
            raise ConfigError("ownership marker is not canonical JSON.")
        target_after = os.fstat(target_fd)
        if not _same_identity(target_before, target_after):
            raise ConfigError("output target changed while reading its ownership marker.")
        return target_after, marker
    except FileNotFoundError:
        raise ConfigError("output target lacks an ownership marker.") from None
    finally:
        if marker_fd >= 0:
            os.close(marker_fd)
        if target_fd >= 0:
            os.close(target_fd)


def verify_a34_under_lease(
    publication: PublicationLease,
    platform: OutputPlatform,
) -> VerifiedOutputLease:
    """Check target/marker/clobber and return the authorized target view."""
    if type(publication) is not PublicationLease:
        raise ConfigError("A34 verification requires an exact PublicationLease.")
    lease = publication.lease
    require_open_output_lease(lease, platform)
    if publication.component_limit != lease.component_limit:
        raise ConfigError("publication lease has the wrong NAME_MAX.")
    revalidate_output_ancestry(lease, platform)
    try:
        target = os.lstat(lease.target_name, dir_fd=lease.parent_fd)
    except FileNotFoundError:
        return VerifiedOutputLease(publication, TargetIdentity(False, None, None, None))
    if stat.S_ISLNK(target.st_mode):
        raise ConfigError("output target is a symlink.")
    if not stat.S_ISDIR(target.st_mode):
        raise ConfigError("output target exists and is not a directory.")
    if not lease.request.clobber:
        raise ConfigError("output target exists and outputs.clobber is false.")
    observed, marker = _read_owned_marker(lease.parent_fd, lease.target_name)
    final = os.lstat(lease.target_name, dir_fd=lease.parent_fd)
    if not _same_identity(observed, final):
        raise ConfigError("output target identity changed during A34 verification.")
    return VerifiedOutputLease(
        publication,
        TargetIdentity(True, final.st_dev, final.st_ino, marker.run_directory_id),
    )


__all__ = [
    "acquire_output_lease",
    "close_output_lease",
    "inspect_output_path",
    "parse_output_grammar",
    "parse_output_request",
    "require_open_output_lease",
    "resolve_output_request",
    "revalidate_output_ancestry",
    "verify_a34_under_lease",
    "verify_publication_under_lease",
]
