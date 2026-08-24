"""Recoverable descriptor-relative publication of immutable audit bundles."""

from __future__ import annotations

import contextlib
import contextvars
import dataclasses
import datetime as dt
import hashlib
import json
import os
import re
import stat
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, cast

from _rheplicant_bootstrap.audit.bundle import AuditBundle, validate_serialized_bundle
from _rheplicant_bootstrap.audit.integrity import INTEGRITY_NAME, integrity_bytes
from _rheplicant_bootstrap.audit.types import ArtefactMaterialization
from _rheplicant_bootstrap.errors import ConfigError

from .manager import (
    require_open_output_lease,
    revalidate_output_ancestry,
)
from .paths import (
    TRANSACTION_PHASES,
    backup_name,
    decode_journal_temp,
    journal_temp_name,
    require_component_budget,
    staging_name,
)
from .platform import OutputPlatform
from .types import (
    OutputLease,
    OutputMarker,
    PublicationLease,
    RecoveryOutcome,
    TargetIdentity,
    TransactionHandle,
    TransactionInterruptionState,
    TransactionJournal,
    VerifiedOutputLease,
)

TRANSACTION_BOUNDARIES = (
    "mkdir",
    "open_create",
    "write",
    "fchmod",
    "file_fsync",
    "directory_fsync",
    "rename_noreplace",
    "rename_owned",
    "unlink",
    "rmdir",
)
_PUBLICATIONS = ("success", "refused", "error")
_TRANSACTION_ID = re.compile(r"[0-9a-f]{32}")
_MARKER_ID = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}"
)
_OPEN_DIRECTORY = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
)
_OPEN_READ = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
_OPEN_WRITE = os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
_MARKER_NAME = ".rheplicant-results.json"


@dataclass(frozen=True, slots=True)
class PersistenceEvent:
    ordinal: int
    label: str


class TransactionInterrupted(RuntimeError):
    def __init__(self, state: TransactionInterruptionState) -> None:
        super().__init__(
            f"output transaction {state.operation} was interrupted by "
            f"{state.exception_type}: {state.exception_message}"
        )
        self.state = state


_OBSERVER: contextvars.ContextVar[Callable[[PersistenceEvent], None] | None] = (
    contextvars.ContextVar("rheplicant_output_persistence_observer", default=None)
)
_ORDINAL: contextvars.ContextVar[int] = contextvars.ContextVar(
    "rheplicant_output_persistence_ordinal", default=0
)


@contextlib.contextmanager
def capture_persistence_events() -> Iterator[list[PersistenceEvent]]:
    events: list[PersistenceEvent] = []
    observer_token = _OBSERVER.set(events.append)
    ordinal_token = _ORDINAL.set(0)
    try:
        yield events
    finally:
        _ORDINAL.reset(ordinal_token)
        _OBSERVER.reset(observer_token)


@contextlib.contextmanager
def observe_persistence(
    observer: Callable[[PersistenceEvent], None],
) -> Iterator[None]:
    if not callable(observer):
        raise ConfigError("persistence observer must be callable.")
    observer_token = _OBSERVER.set(observer)
    ordinal_token = _ORDINAL.set(0)
    try:
        yield
    finally:
        _ORDINAL.reset(ordinal_token)
        _OBSERVER.reset(observer_token)


def _event(label: str) -> None:
    if label not in TRANSACTION_BOUNDARIES:
        raise AssertionError(f"unknown persistence label {label!r}")
    observer = _OBSERVER.get()
    if observer is None:
        return
    ordinal = _ORDINAL.get()
    _ORDINAL.set(ordinal + 1)
    observer(PersistenceEvent(ordinal, label))


def _canonical_json(value: object) -> bytes:
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError, OverflowError):
        raise ConfigError("transaction metadata is not canonical JSON.") from None
    return (payload + "\n").encode()


def _identity(row: TargetIdentity) -> Mapping[str, object]:
    return {
        "exists": row.exists,
        "device": row.device,
        "inode": row.inode,
        "marker_id": row.marker_id,
    }


def _journal_tree(row: TransactionJournal) -> Mapping[str, object]:
    return {
        "format_version": row.format_version,
        "transaction_id": row.transaction_id,
        "publication": row.publication,
        "phase": row.phase,
        "guarded_target_name": row.guarded_target_name,
        "publish_name": row.publish_name,
        "journal_name": row.journal_name,
        "original": _identity(row.original),
        "staging_name": row.staging_name,
        "staging_identity": (
            None if row.staging_identity is None else _identity(row.staging_identity)
        ),
        "backup_name": row.backup_name,
        "backup_identity": (
            None if row.backup_identity is None else _identity(row.backup_identity)
        ),
        "published_identity": (
            None if row.published_identity is None else _identity(row.published_identity)
        ),
        "new_marker_id": row.new_marker_id,
    }


def _journal_bytes(row: TransactionJournal) -> bytes:
    return _canonical_json(_journal_tree(row))


def _fsync_file(fd: int) -> None:
    os.fsync(fd)
    _event("file_fsync")


def _fsync_directory(fd: int) -> None:
    os.fsync(fd)
    _event("directory_fsync")


def _fchmod(fd: int, mode: int) -> None:
    os.fchmod(fd, mode)
    _event("fchmod")


def _write_all(
    fd: int,
    payload: bytes,
    *,
    on_complete: Callable[[], None] | None = None,
) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(fd, payload[offset:])
        if written <= 0:
            raise OSError("short output write made no progress")
        offset += written
        if offset == len(payload) and on_complete is not None:
            on_complete()
            on_complete = None
        _event("write")


def _open_create(parent_fd: int, name: str, mode: int = 0o600) -> int:
    fd = os.open(
        name,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
        mode,
        dir_fd=parent_fd,
    )
    _event("open_create")
    return fd


def _write_new_file(
    parent_fd: int,
    name: str,
    payload: bytes,
    *,
    on_complete: Callable[[], None] | None = None,
) -> os.stat_result:
    fd = _open_create(parent_fd, name)
    try:
        _write_all(fd, payload, on_complete=on_complete)
        _fchmod(fd, 0o600)
        _fsync_file(fd)
        row = os.fstat(fd)
    finally:
        os.close(fd)
    _fsync_directory(parent_fd)
    return row


def _replace_file(parent_fd: int, name: str, payload: bytes) -> None:
    before = os.lstat(name, dir_fd=parent_fd)
    fd = os.open(name, _OPEN_WRITE | os.O_TRUNC, dir_fd=parent_fd)
    try:
        if not _same_identity(before, os.fstat(fd)) or not stat.S_ISREG(before.st_mode):
            raise ConfigError(f"owned staged metadata {name!r} changed before rewrite.")
        _write_all(fd, payload)
        _fchmod(fd, 0o600)
        _fsync_file(fd)
    finally:
        os.close(fd)
    _fsync_directory(parent_fd)


def _read_exact_file(parent_fd: int, name: str, *, maximum: int | None = None) -> bytes:
    before = os.lstat(name, dir_fd=parent_fd)
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ConfigError(f"transaction file {name!r} is not a regular owned entry.")
    if maximum is not None and before.st_size > maximum:
        raise ConfigError(f"transaction file {name!r} exceeds {maximum} bytes.")
    fd = os.open(name, _OPEN_READ, dir_fd=parent_fd)
    try:
        chunks = []
        remaining = before.st_size + 1
        if maximum is not None:
            remaining = min(remaining, maximum + 1)
        while remaining:
            chunk = os.read(fd, remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    if (
        not _same_identity(before, after)
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_ctime_ns != after.st_ctime_ns
        or len(payload) != before.st_size
    ):
        raise ConfigError(f"transaction file {name!r} changed while reading.")
    return payload


def _same_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return left.st_dev == right.st_dev and left.st_ino == right.st_ino


def _node_identity(parent_fd: int, name: str, marker_id: str | None = None) -> TargetIdentity:
    try:
        row = os.lstat(name, dir_fd=parent_fd)
    except FileNotFoundError:
        return TargetIdentity(False, None, None, None)
    if stat.S_ISLNK(row.st_mode) or not stat.S_ISDIR(row.st_mode):
        raise ConfigError(f"transaction node {name!r} is not a directory.")
    return TargetIdentity(True, row.st_dev, row.st_ino, marker_id)


def _identity_matches(observed: TargetIdentity, expected: TargetIdentity | None) -> bool:
    if expected is None:
        return not observed.exists
    return (
        observed.exists == expected.exists
        and observed.device == expected.device
        and observed.inode == expected.inode
        and (
            expected.marker_id is None
            or observed.marker_id is None
            or observed.marker_id == expected.marker_id
        )
    )


def _marker_bytes(marker: OutputMarker) -> bytes:
    return _canonical_json(
        {
            "format_version": marker.format_version,
            "run_directory_id": marker.run_directory_id,
        }
    )


def _mkdir(parent_fd: int, name: str) -> None:
    os.mkdir(name, 0o700, dir_fd=parent_fd)
    _event("mkdir")


def _open_directory(parent_fd: int, name: str) -> int:
    return os.open(name, _OPEN_DIRECTORY, dir_fd=parent_fd)


def _create_directory(parent_fd: int, name: str) -> int:
    _mkdir(parent_fd, name)
    fd = _open_directory(parent_fd, name)
    _fchmod(fd, 0o700)
    _fsync_directory(fd)
    _fsync_directory(parent_fd)
    return fd


def _ensure_relative_file(
    root_fd: int,
    relative: str,
    payload: bytes,
    *,
    on_complete: Callable[[], None] | None = None,
) -> os.stat_result:
    if type(relative) is not str or relative.startswith("/") or "\0" in relative:
        raise ConfigError("bundle contains an invalid relative path.")
    parts = tuple(relative.split("/"))
    if not parts or any(part in ("", ".", "..") for part in parts):
        raise ConfigError("bundle contains an invalid relative path.")
    current_fd = os.dup(root_fd)
    try:
        for component in parts[:-1]:
            try:
                child_fd = _open_directory(current_fd, component)
            except FileNotFoundError:
                child_fd = _create_directory(current_fd, component)
            os.close(current_fd)
            current_fd = child_fd
        return _write_new_file(
            current_fd,
            parts[-1],
            payload,
            on_complete=on_complete,
        )
    finally:
        os.close(current_fd)


def _read_relative_file(root_fd: int, relative: str) -> bytes:
    parts = tuple(relative.split("/"))
    if not parts or any(part in ("", ".", "..") for part in parts):
        raise ConfigError("bundle contains an invalid relative path.")
    current_fd = os.dup(root_fd)
    try:
        for component in parts[:-1]:
            child_fd = _open_directory(current_fd, component)
            os.close(current_fd)
            current_fd = child_fd
        return _read_exact_file(current_fd, parts[-1])
    finally:
        os.close(current_fd)


def _bundle_rows(bundle: AuditBundle) -> tuple[tuple[str, bytes], ...]:
    validate_serialized_bundle(bundle)
    if type(bundle) is not AuditBundle or not isinstance(bundle.files, Mapping):
        raise ConfigError("transaction staging requires an exact AuditBundle.")
    rows = tuple(bundle.files.items())
    prefix = ["config.input.yaml"]
    prefix.extend(row.relative_path for row in bundle.resolved)
    names = [name for name, _payload in rows]
    # integrity.json joins the two metadata files at the tail because it shares
    # their timing, not their kind: all three can only be written once every
    # other file is final, and all three are replaced together below. It is
    # optional so that a bundle assembled by an older caller still validates.
    tail = ["provenance.json", "diagnostics.json"]
    if names[-3:-2] == [INTEGRITY_NAME]:
        tail = [INTEGRITY_NAME, *tail]
    if (
        names[: len(prefix)] != prefix
        or names[-len(tail) :] != tail
    ):
        raise ConfigError("audit bundle file order or paths are inconsistent.")
    if any(type(name) is not str or type(payload) is not bytes for name, payload in rows):
        raise ConfigError("audit bundle files must be exact text/bytes pairs.")
    if dict(rows)["config.input.yaml"] is not bundle.input:
        raise ConfigError("audit bundle input file is not its dataclass payload.")
    if dict(rows)["provenance.json"] is not bundle.provenance:
        raise ConfigError("audit bundle provenance file is not its dataclass payload.")
    if dict(rows)["diagnostics.json"] is not bundle.diagnostics:
        raise ConfigError("audit bundle diagnostics file is not its dataclass payload.")
    return rows


def _rename_noreplace(
    lease: OutputLease,
    platform: OutputPlatform,
    old_name: str,
    new_name: str,
    *,
    on_complete: Callable[[], None] | None = None,
) -> None:
    revalidate_output_ancestry(lease, platform)
    platform.rename_noreplace(lease.parent_fd, old_name, lease.parent_fd, new_name)
    if on_complete is not None:
        on_complete()
    _event("rename_noreplace")


def _rename_owned(
    lease: OutputLease,
    platform: OutputPlatform,
    old_name: str,
    new_name: str,
) -> None:
    revalidate_output_ancestry(lease, platform)
    os.rename(old_name, new_name, src_dir_fd=lease.parent_fd, dst_dir_fd=lease.parent_fd)
    _event("rename_owned")


def _unlink(parent_fd: int, name: str) -> None:
    os.unlink(name, dir_fd=parent_fd)
    _event("unlink")


def _rmdir(parent_fd: int, name: str) -> None:
    os.rmdir(name, dir_fd=parent_fd)
    _event("rmdir")


def _read_journal(parent_fd: int, name: str) -> TransactionJournal:
    payload = _read_exact_file(parent_fd, name, maximum=4096)
    try:
        value = json.loads(payload)
    except (UnicodeError, json.JSONDecodeError):
        raise ConfigError(f"transaction journal {name!r} is malformed.") from None
    if type(value) is not dict or tuple(sorted(value)) != (
        "backup_identity",
        "backup_name",
        "format_version",
        "guarded_target_name",
        "journal_name",
        "new_marker_id",
        "original",
        "phase",
        "publication",
        "publish_name",
        "published_identity",
        "staging_identity",
        "staging_name",
        "transaction_id",
    ):
        raise ConfigError(f"transaction journal {name!r} has foreign fields.")

    def identity(item: object, *, optional: bool) -> TargetIdentity | None:
        if item is None and optional:
            return None
        if type(item) is not dict or tuple(sorted(item)) != (
            "device",
            "exists",
            "inode",
            "marker_id",
        ):
            raise ConfigError(f"transaction journal {name!r} has an invalid identity.")
        row = TargetIdentity(item["exists"], item["device"], item["inode"], item["marker_id"])
        if type(row.exists) is not bool:
            raise ConfigError(f"transaction journal {name!r} has an invalid identity.")
        if row.exists:
            if type(row.device) is not int or type(row.inode) is not int:
                raise ConfigError(f"transaction journal {name!r} has an invalid identity.")
        elif (row.device, row.inode, row.marker_id) != (None, None, None):
            raise ConfigError(f"transaction journal {name!r} has an invalid absent identity.")
        return row

    row = TransactionJournal(
        value["format_version"],
        value["transaction_id"],
        value["publication"],
        value["phase"],
        value["guarded_target_name"],
        value["publish_name"],
        value["journal_name"],
        cast(TargetIdentity, identity(value["original"], optional=False)),
        value["staging_name"],
        identity(value["staging_identity"], optional=True),
        value["backup_name"],
        identity(value["backup_identity"], optional=True),
        identity(value["published_identity"], optional=True),
        value["new_marker_id"],
    )
    if (
        row.format_version != 1
        or _TRANSACTION_ID.fullmatch(row.transaction_id or "") is None
        or row.publication not in _PUBLICATIONS
        or row.phase not in TRANSACTION_PHASES
        or type(row.guarded_target_name) is not str
        or type(row.publish_name) is not str
        or type(row.journal_name) is not str
        or type(row.staging_name) is not str
        or (row.backup_name is not None and type(row.backup_name) is not str)
        or _MARKER_ID.fullmatch(row.new_marker_id or "") is None
        or payload != _journal_bytes(row)
    ):
        raise ConfigError(f"transaction journal {name!r} is not canonical or supported.")
    return row


def _publish_initial_journal(
    lease: OutputLease,
    platform: OutputPlatform,
    journal: TransactionJournal,
    *,
    on_published: Callable[[], None],
) -> None:
    temporary = journal_temp_name(
        lease.request.target_path,
        journal.transaction_id,
        journal.phase,
    )
    _write_new_file(lease.parent_fd, temporary, _journal_bytes(journal))
    _rename_noreplace(
        lease,
        platform,
        temporary,
        lease.journal_name,
        on_complete=on_published,
    )
    _fsync_directory(lease.parent_fd)


def _advance_journal(
    lease: OutputLease,
    platform: OutputPlatform,
    journal: TransactionJournal,
) -> None:
    before = os.lstat(lease.journal_name, dir_fd=lease.parent_fd)
    current = _read_journal(lease.parent_fd, lease.journal_name)
    if current.transaction_id != journal.transaction_id:
        raise ConfigError("canonical transaction journal changed ownership.")
    after_read = os.lstat(lease.journal_name, dir_fd=lease.parent_fd)
    if not _same_identity(before, after_read):
        raise ConfigError("canonical transaction journal changed while reading.")
    temporary = journal_temp_name(
        lease.request.target_path,
        journal.transaction_id,
        journal.phase,
    )
    _write_new_file(lease.parent_fd, temporary, _journal_bytes(journal))
    before_rename = os.lstat(lease.journal_name, dir_fd=lease.parent_fd)
    if not _same_identity(before, before_rename):
        raise ConfigError("canonical transaction journal changed before phase update.")
    _rename_owned(lease, platform, temporary, lease.journal_name)
    _fsync_directory(lease.parent_fd)


def _timestamped_failure_name(lease: OutputLease, publication: str) -> str:
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    return f"{lease.target_name}.{publication}-{stamp}-{os.getpid()}"


def _interrupted(
    operation: str,
    transaction_id: str | None,
    handle: TransactionHandle | None,
    materializations: Sequence[ArtefactMaterialization],
    error: Exception,
) -> TransactionInterrupted:
    return TransactionInterrupted(
        TransactionInterruptionState(
            transaction_id,
            cast(str, operation),
            handle,
            tuple(materializations),
            f"{type(error).__module__}.{type(error).__qualname__}",
            str(error),
        )
    )


def _materialization_for_file(
    bundle: AuditBundle,
    relative: str,
    payload: bytes,
) -> ArtefactMaterialization | None:
    if relative == "config.input.yaml":
        return ArtefactMaterialization(
            "input", None, relative, len(payload), hashlib.sha256(payload).hexdigest()
        )
    if relative == "provenance.json":
        return ArtefactMaterialization("provenance", None, relative, None, None)
    if relative == "diagnostics.json":
        return ArtefactMaterialization("diagnostics", None, relative, None, None)
    for index, row in enumerate(bundle.resolved):
        if relative == row.relative_path:
            slot = "resolved_base" if row.layer.kind == "base" else "resolved_variant"
            variant_index = None if slot == "resolved_base" else index - 1
            return ArtefactMaterialization(
                slot,
                variant_index,
                relative,
                len(payload),
                hashlib.sha256(payload).hexdigest(),
            )
    return None


def stage_bundle(
    authorization: VerifiedOutputLease | PublicationLease,
    candidate: AuditBundle,
    platform: OutputPlatform,
    *,
    publication: Literal["success", "refused", "error"],
) -> tuple[TransactionHandle, tuple[ArtefactMaterialization, ...]]:
    """Exclusively stage candidate bytes under a preparing journal."""
    if publication == "success":
        if type(authorization) is not VerifiedOutputLease:
            raise ConfigError("success staging requires a VerifiedOutputLease.")
        verified = authorization
        publication_lease = verified.publication
    elif publication in ("refused", "error"):
        if type(authorization) is not PublicationLease:
            raise ConfigError("failure staging requires a PublicationLease.")
        verified = None
        publication_lease = authorization
    else:
        raise ConfigError("unknown transaction publication kind.")
    lease = publication_lease.lease
    require_open_output_lease(lease, platform)
    if publication_lease.component_limit != lease.component_limit:
        raise ConfigError("publication lease has the wrong NAME_MAX.")
    revalidate_output_ancestry(lease, platform)
    rows = _bundle_rows(candidate)
    original = (
        verified.original
        if verified is not None
        else TargetIdentity(False, None, None, None)
    )
    try:
        os.lstat(lease.journal_name, dir_fd=lease.parent_fd)
    except FileNotFoundError:
        pass
    else:
        raise ConfigError("canonical transaction journal already exists; recover first.")
    for _attempt in range(128):
        transaction_id = uuid.uuid4().hex
        staged = staging_name(lease.request.target_path, transaction_id)
        backup = (
            backup_name(lease.request.target_path, transaction_id)
            if verified is not None and verified.original.exists
            else None
        )
        publish_name = (
            lease.target_name
            if publication == "success"
            else _timestamped_failure_name(lease, publication)
        )
        temporary_names = tuple(
            journal_temp_name(lease.request.target_path, transaction_id, phase)
            for phase in TRANSACTION_PHASES
        )
        exact_names = [staged, publish_name, lease.journal_name, *temporary_names]
        if backup is not None:
            exact_names.append(backup)
        require_component_budget(tuple(exact_names), publication_lease.component_limit)
        collision_names = [staged, *temporary_names]
        if publication != "success" or not original.exists:
            collision_names.append(publish_name)
        if backup is not None:
            collision_names.append(backup)
        for name in collision_names:
            try:
                os.lstat(name, dir_fd=lease.parent_fd)
            except FileNotFoundError:
                continue
            break
        else:
            break
    else:
        raise ConfigError("could not select a collision-free transaction id.")
    marker = OutputMarker(1, str(uuid.uuid4()))
    journal = TransactionJournal(
        1,
        transaction_id,
        publication,
        "preparing",
        lease.target_name,
        publish_name,
        lease.journal_name,
        original,
        staged,
        None,
        backup,
        None,
        None,
        marker.run_directory_id,
    )
    handle = TransactionHandle(
        lease,
        publication_lease,
        verified,
        publication,
        transaction_id,
        lease.target_name,
        publish_name,
        lease.journal_name,
        staged,
        backup,
        marker,
    )
    materialized: list[ArtefactMaterialization] = [
        ArtefactMaterialization("lock", None, lease.lock_name, None, None)
    ]
    staging_fd = -1
    try:
        _publish_initial_journal(
            lease,
            platform,
            journal,
            on_published=lambda: materialized.append(
                ArtefactMaterialization(
                    "journal", None, lease.journal_name, None, None
                )
            ),
        )
        _mkdir(lease.parent_fd, staged)
        staging_fd = _open_directory(lease.parent_fd, staged)
        _fchmod(staging_fd, 0o700)
        _fsync_directory(staging_fd)
        _fsync_directory(lease.parent_fd)
        staging_identity = _node_identity(lease.parent_fd, staged)
        journal = dataclasses.replace(journal, staging_identity=staging_identity)
        _advance_journal(lease, platform, journal)
        for relative, payload in rows:
            row = _materialization_for_file(candidate, relative, payload)
            on_complete = None
            if row is not None:
                def record_materialization(row=row) -> None:
                    materialized.append(row)

                on_complete = record_materialization
            _ensure_relative_file(
                staging_fd,
                relative,
                payload,
                on_complete=on_complete,
            )
        _ensure_relative_file(
            staging_fd,
            _MARKER_NAME,
            _marker_bytes(marker),
            on_complete=lambda: materialized.append(
                ArtefactMaterialization(
                    "marker", None, _MARKER_NAME, None, None
                )
            ),
        )
        _fsync_directory(staging_fd)
        _fsync_directory(lease.parent_fd)
        return handle, tuple(materialized)
    except TransactionInterrupted:
        raise
    except Exception as error:
        raise _interrupted(
            "stage_bundle", transaction_id, handle, materialized, error
        ) from error
    finally:
        if staging_fd >= 0:
            os.close(staging_fd)


def _require_handle(handle: TransactionHandle, platform: OutputPlatform) -> TransactionJournal:
    if type(handle) is not TransactionHandle:
        raise ConfigError("transaction operation requires an exact handle.")
    if handle.lease is not handle.publication_lease.lease:
        raise ConfigError("transaction handle lease views disagree.")
    if handle.verified is not None and handle.verified.publication is not handle.publication_lease:
        raise ConfigError("transaction handle verified view is unrelated.")
    require_open_output_lease(handle.lease, platform)
    revalidate_output_ancestry(handle.lease, platform)
    journal = _read_journal(handle.lease.parent_fd, handle.journal_name)
    if (
        journal.transaction_id != handle.transaction_id
        or journal.publication != handle.publication
        or journal.staging_name != handle.staging_name
        or journal.publish_name != handle.publish_name
        or journal.backup_name != handle.backup_name
    ):
        raise ConfigError("transaction handle disagrees with canonical journal.")
    return journal


def replace_staged_metadata(
    handle: TransactionHandle,
    final: AuditBundle,
    platform: OutputPlatform,
) -> None:
    """Replace only owned staged metadata and advance to prepared."""
    journal = _require_handle(handle, platform)
    if journal.phase != "preparing" or journal.staging_identity is None:
        raise ConfigError("staged metadata replacement requires preparing state.")
    rows = _bundle_rows(final)
    staging_fd = -1
    try:
        observed = _node_identity(handle.lease.parent_fd, handle.staging_name)
        if not _identity_matches(observed, journal.staging_identity):
            raise ConfigError("owned staging identity changed before metadata replacement.")
        staging_fd = _open_directory(handle.lease.parent_fd, handle.staging_name)
        replaceable = ("provenance.json", "diagnostics.json", INTEGRITY_NAME)
        for relative, payload in rows:
            if relative in replaceable:
                continue
            if _read_relative_file(staging_fd, relative) != payload:
                raise ConfigError(
                    f"final bundle changes already-staged content {relative!r}."
                )
        _replace_file(staging_fd, "provenance.json", final.provenance)
        _replace_file(staging_fd, "diagnostics.json", final.diagnostics)
        if INTEGRITY_NAME in dict(rows):
            # Computed LAST, over the rows as they now stand -- which is the
            # whole point. An integrity file written at staging time would
            # describe the provenance and diagnostics this call just replaced.
            #
            # The ownership marker is read back from the staging directory
            # rather than taken from the bundle, because it is not a bundle
            # file: the transaction writes it. Covering it anyway is what lets
            # a verifier treat ANY unlisted path as an addition, instead of
            # carrying a list of paths it agrees not to look at -- which is the
            # shape that quietly grows until it covers the thing you needed.
            covered = [
                (relative, payload)
                for relative, payload in rows
                if relative != INTEGRITY_NAME
            ]
            covered.append(
                (_MARKER_NAME, _read_exact_file(staging_fd, _MARKER_NAME))
            )
            _replace_file(
                staging_fd, INTEGRITY_NAME, integrity_bytes(tuple(covered))
            )
        if _read_exact_file(staging_fd, "provenance.json") != final.provenance:
            raise ConfigError("staged provenance verification failed.")
        if _read_exact_file(staging_fd, "diagnostics.json") != final.diagnostics:
            raise ConfigError("staged diagnostics verification failed.")
        json.loads(final.provenance)
        json.loads(final.diagnostics)
        _fsync_directory(staging_fd)
        _advance_journal(
            handle.lease,
            platform,
            dataclasses.replace(journal, phase="prepared"),
        )
    except TransactionInterrupted:
        raise
    except Exception as error:
        raise _interrupted(
            "replace_staged_metadata", handle.transaction_id, handle, (), error
        ) from error
    finally:
        if staging_fd >= 0:
            os.close(staging_fd)


def _verify_target_identity(
    lease: OutputLease,
    name: str,
    expected: TargetIdentity,
) -> None:
    observed = _node_identity(lease.parent_fd, name, expected.marker_id)
    if not _identity_matches(observed, expected):
        raise ConfigError(f"transaction target {name!r} changed identity.")


def _read_marker_id(parent_fd: int, name: str) -> str:
    directory = os.lstat(name, dir_fd=parent_fd)
    if (
        not stat.S_ISDIR(directory.st_mode)
        or directory.st_uid != os.geteuid()
        or stat.S_IMODE(directory.st_mode) != 0o700
    ):
        raise ConfigError(f"transaction node {name!r} has insecure ownership or mode.")
    directory_fd = _open_directory(parent_fd, name)
    try:
        marker_stat = os.lstat(_MARKER_NAME, dir_fd=directory_fd)
        if (
            not stat.S_ISREG(marker_stat.st_mode)
            or marker_stat.st_uid != os.geteuid()
            or stat.S_IMODE(marker_stat.st_mode) != 0o600
            or marker_stat.st_nlink != 1
        ):
            raise ConfigError(f"transaction node {name!r} has an insecure marker.")
        payload = _read_exact_file(directory_fd, _MARKER_NAME, maximum=4096)
    finally:
        os.close(directory_fd)
    try:
        value = json.loads(payload)
    except (UnicodeError, json.JSONDecodeError):
        raise ConfigError(f"transaction node {name!r} has a malformed marker.") from None
    if (
        type(value) is not dict
        or tuple(sorted(value)) != ("format_version", "run_directory_id")
        or value["format_version"] != 1
        or _MARKER_ID.fullmatch(value["run_directory_id"] or "") is None
        or payload != _canonical_json(value)
    ):
        raise ConfigError(f"transaction node {name!r} has a foreign marker.")
    return value["run_directory_id"]


def _published_identity(lease: OutputLease, name: str) -> TargetIdentity:
    marker_id = _read_marker_id(lease.parent_fd, name)
    return _node_identity(lease.parent_fd, name, marker_id)


def _safe_remove_directory(
    parent_fd: int,
    name: str,
    expected: TargetIdentity,
) -> None:
    observed = _node_identity(parent_fd, name)
    if not _identity_matches(observed, expected):
        raise ConfigError(f"refusing to remove changed transaction directory {name!r}.")
    root_fd = _open_directory(parent_fd, name)
    try:
        opened = os.fstat(root_fd)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or opened.st_dev != expected.device
            or opened.st_ino != expected.inode
            or opened.st_uid != os.geteuid()
        ):
            raise ConfigError(
                f"refusing to remove changed transaction directory {name!r}."
            )
        _remove_directory_contents(root_fd)
        _fsync_directory(root_fd)
    finally:
        os.close(root_fd)
    final = _node_identity(parent_fd, name)
    if not _identity_matches(final, expected):
        raise ConfigError(f"refusing to remove changed transaction directory {name!r}.")
    _rmdir(parent_fd, name)
    _fsync_directory(parent_fd)


def _remove_directory_contents(directory_fd: int) -> None:
    for name in sorted(os.listdir(directory_fd)):
        row = os.lstat(name, dir_fd=directory_fd)
        if row.st_uid != os.geteuid() or stat.S_ISLNK(row.st_mode):
            raise ConfigError(f"refusing to remove unproved transaction entry {name!r}.")
        if stat.S_ISREG(row.st_mode):
            file_fd = os.open(name, _OPEN_READ, dir_fd=directory_fd)
            try:
                observed = os.fstat(file_fd)
                if (
                    not _same_identity(observed, row)
                    or observed.st_nlink != 1
                    or observed.st_uid != os.geteuid()
                ):
                    raise ConfigError(
                        f"transaction entry {name!r} changed before cleanup."
                    )
                final = os.lstat(name, dir_fd=directory_fd)
                if not _same_identity(final, observed):
                    raise ConfigError(
                        f"transaction entry {name!r} changed before cleanup."
                    )
                _unlink(directory_fd, name)
                _fsync_directory(directory_fd)
            finally:
                os.close(file_fd)
        elif stat.S_ISDIR(row.st_mode):
            child_fd = _open_directory(directory_fd, name)
            try:
                if not _same_identity(row, os.fstat(child_fd)):
                    raise ConfigError(f"transaction directory {name!r} changed before cleanup.")
                _remove_directory_contents(child_fd)
                _fsync_directory(child_fd)
            finally:
                os.close(child_fd)
            final = os.lstat(name, dir_fd=directory_fd)
            if not _same_identity(row, final):
                raise ConfigError(f"transaction directory {name!r} changed before cleanup.")
            _rmdir(directory_fd, name)
            _fsync_directory(directory_fd)
        else:
            raise ConfigError(f"refusing to remove special transaction entry {name!r}.")


def _finish_journal(lease: OutputLease, transaction_id: str) -> None:
    before = os.lstat(lease.journal_name, dir_fd=lease.parent_fd)
    current = _read_journal(lease.parent_fd, lease.journal_name)
    if current.transaction_id != transaction_id:
        raise ConfigError("canonical transaction journal changed ownership.")
    after = os.lstat(lease.journal_name, dir_fd=lease.parent_fd)
    if not _same_identity(before, after):
        raise ConfigError("canonical transaction journal changed before cleanup.")
    _unlink(lease.parent_fd, lease.journal_name)
    _fsync_directory(lease.parent_fd)


def publish_success(handle: TransactionHandle, platform: OutputPlatform) -> str:
    """Atomically publish a prepared success transaction."""
    journal = _require_handle(handle, platform)
    if handle.publication != "success" or handle.verified is None:
        raise ConfigError("success publication requires a verified success handle.")
    if handle.publish_name != handle.guarded_target_name:
        raise ConfigError("success publication must target the guarded leaf.")
    if journal.phase != "prepared" or journal.staging_identity is None:
        raise ConfigError("success publication requires prepared staging.")
    lease = handle.lease
    try:
        _verify_target_identity(lease, handle.staging_name, journal.staging_identity)
        if handle.verified.original.exists:
            _verify_target_identity(lease, handle.guarded_target_name, handle.verified.original)
            if handle.backup_name is None:
                raise ConfigError("clobber transaction lacks a backup name.")
            _rename_noreplace(
                lease,
                platform,
                handle.guarded_target_name,
                handle.backup_name,
            )
            _fsync_directory(lease.parent_fd)
            backup_identity = _node_identity(lease.parent_fd, handle.backup_name)
            journal = dataclasses.replace(
                journal,
                phase="backup_moved",
                backup_identity=backup_identity,
            )
            _advance_journal(lease, platform, journal)
        _rename_noreplace(lease, platform, handle.staging_name, handle.publish_name)
        _fsync_directory(lease.parent_fd)
        published = _published_identity(lease, handle.publish_name)
        journal = dataclasses.replace(
            journal,
            phase="staging_published",
            published_identity=published,
        )
        _advance_journal(lease, platform, journal)
        published_fd = _open_directory(lease.parent_fd, handle.publish_name)
        try:
            _fsync_directory(published_fd)
        finally:
            os.close(published_fd)
        _fsync_directory(lease.parent_fd)
        journal = dataclasses.replace(journal, phase="target_durable")
        _advance_journal(lease, platform, journal)
        if handle.backup_name is not None:
            if journal.backup_identity is None:
                raise ConfigError("published transaction lost its backup identity.")
            _safe_remove_directory(
                lease.parent_fd,
                handle.backup_name,
                journal.backup_identity,
            )
            journal = dataclasses.replace(
                journal,
                phase="backup_removed",
                backup_identity=None,
            )
            _advance_journal(lease, platform, journal)
        _finish_journal(lease, journal.transaction_id)
        return os.path.join(lease.parent_path, handle.publish_name)
    except TransactionInterrupted:
        raise
    except Exception as error:
        raise _interrupted(
            "publish_success", handle.transaction_id, handle, (), error
        ) from error


def publish_failure(handle: TransactionHandle, platform: OutputPlatform) -> str:
    """Publish a prepared failure sibling without touching the guarded target."""
    journal = _require_handle(handle, platform)
    if (
        handle.publication not in ("refused", "error")
        or handle.verified is not None
        or handle.publish_name == handle.guarded_target_name
        or handle.backup_name is not None
    ):
        raise ConfigError("failure publication requires a non-target failure handle.")
    if journal.phase != "prepared" or journal.staging_identity is None:
        raise ConfigError("failure publication requires prepared staging.")
    lease = handle.lease
    try:
        _verify_target_identity(lease, handle.staging_name, journal.staging_identity)
        _rename_noreplace(lease, platform, handle.staging_name, handle.publish_name)
        _fsync_directory(lease.parent_fd)
        published = _published_identity(lease, handle.publish_name)
        journal = dataclasses.replace(
            journal,
            phase="staging_published",
            published_identity=published,
        )
        _advance_journal(lease, platform, journal)
        published_fd = _open_directory(lease.parent_fd, handle.publish_name)
        try:
            _fsync_directory(published_fd)
        finally:
            os.close(published_fd)
        _fsync_directory(lease.parent_fd)
        _advance_journal(
            lease,
            platform,
            dataclasses.replace(journal, phase="target_durable"),
        )
        _finish_journal(lease, journal.transaction_id)
        return os.path.join(lease.parent_path, handle.publish_name)
    except TransactionInterrupted:
        raise
    except Exception as error:
        raise _interrupted(
            "publish_failure", handle.transaction_id, handle, (), error
        ) from error


def discard_staging(
    handle: TransactionHandle,
    platform: OutputPlatform,
) -> tuple[str, ...]:
    """Remove only a proved owned staging tree and its journal."""
    journal = _require_handle(handle, platform)
    try:
        if journal.staging_identity is not None:
            _safe_remove_directory(
                handle.lease.parent_fd,
                handle.staging_name,
                journal.staging_identity,
            )
        _finish_journal(handle.lease, journal.transaction_id)
        return ()
    except TransactionInterrupted:
        raise
    except Exception as error:
        raise _interrupted(
            "discard_staging", handle.transaction_id, handle, (), error
        ) from error


def _ambiguous(lease: OutputLease, journal: TransactionJournal, detail: str) -> ConfigError:
    names = (
        journal.guarded_target_name,
        journal.staging_name,
        journal.backup_name or "<no-backup>",
        journal.publish_name,
        journal.journal_name,
    )
    return ConfigError(
        f"ambiguous output recovery ({detail}); no path was removed; preserved: "
        + ", ".join(names)
    )


def _journal_for_recovery(
    lease: OutputLease,
    platform: OutputPlatform,
) -> tuple[TransactionJournal | None, str | None]:
    names = tuple(sorted(os.listdir(lease.parent_fd)))
    canonical_present = lease.journal_name in names
    digest = hashlib.sha256(os.fsencode(lease.request.target_path)).hexdigest()
    prefix = f".rheplicant-jtmp-{digest}-"
    candidates = tuple(name for name in names if name.startswith(prefix))
    if len(candidates) > 1:
        raise ConfigError(
            "ambiguous output recovery: multiple transaction update temporaries; "
            "no path was removed; preserved: "
            + ", ".join((lease.target_name, lease.journal_name, *candidates))
        )
    temp_name = candidates[0] if candidates else None
    decoded_temp = (
        None
        if temp_name is None
        else decode_journal_temp(lease.request.target_path, temp_name)
    )
    if temp_name is not None and decoded_temp is None:
        raise ConfigError(
            "ambiguous output recovery: illegal transaction update temporary; "
            "no path was removed; preserved: "
            + ", ".join((lease.target_name, lease.journal_name, temp_name))
        )
    try:
        temporary = None if temp_name is None else _read_journal(lease.parent_fd, temp_name)
    except ConfigError as error:
        raise ConfigError(
            f"ambiguous output recovery ({error}); no path was removed; preserved: "
            + ", ".join((lease.target_name, lease.journal_name, cast(str, temp_name)))
        ) from None
    if temporary is not None and decoded_temp != (
        temporary.transaction_id,
        temporary.phase,
    ):
        raise _ambiguous(lease, temporary, "update name and journal bytes disagree")
    if not canonical_present:
        if temporary is None:
            return None, None
        if temporary.phase != "preparing":
            raise _ambiguous(lease, temporary, "orphan update is not preparing")
        _rename_noreplace(lease, platform, temp_name, lease.journal_name)
        _fsync_directory(lease.parent_fd)
        return temporary, None
    try:
        canonical = _read_journal(lease.parent_fd, lease.journal_name)
    except ConfigError as error:
        preserved = (lease.target_name, lease.journal_name)
        if temp_name is not None:
            preserved = (*preserved, temp_name)
        raise ConfigError(
            f"ambiguous output recovery ({error}); no path was removed; preserved: "
            + ", ".join(preserved)
        ) from None
    if temporary is None:
        return canonical, None
    if temporary.transaction_id != canonical.transaction_id:
        raise _ambiguous(lease, canonical, "foreign transaction update")
    current_index = TRANSACTION_PHASES.index(canonical.phase)
    temp_index = TRANSACTION_PHASES.index(temporary.phase)
    if temp_index not in (current_index, current_index + 1):
        raise _ambiguous(lease, canonical, "illegal journal phase jump")
    enriched = canonical
    if temp_index == current_index:
        normalized = dataclasses.replace(
            temporary,
            staging_identity=canonical.staging_identity,
            backup_identity=canonical.backup_identity,
            published_identity=canonical.published_identity,
        )
        if normalized != canonical:
            raise _ambiguous(lease, canonical, "same-phase update changes non-identity facts")
        enriched = dataclasses.replace(
            canonical,
            staging_identity=(
                temporary.staging_identity or canonical.staging_identity
            ),
            backup_identity=temporary.backup_identity or canonical.backup_identity,
            published_identity=(
                temporary.published_identity or canonical.published_identity
            ),
        )
    return enriched, temp_name


def _remove_recovery_temp(lease: OutputLease, name: str | None) -> None:
    if name is None:
        return
    _unlink(lease.parent_fd, name)
    _fsync_directory(lease.parent_fd)


def _recover_success(
    lease: OutputLease,
    platform: OutputPlatform,
    journal: TransactionJournal,
) -> RecoveryOutcome:
    target = _node_identity(lease.parent_fd, journal.guarded_target_name)
    staging = _node_identity(lease.parent_fd, journal.staging_name)
    backup = (
        TargetIdentity(False, None, None, None)
        if journal.backup_name is None
        else _node_identity(lease.parent_fd, journal.backup_name)
    )

    def is_new_publication() -> bool:
        if not target.exists:
            return False
        try:
            return (
                _read_marker_id(lease.parent_fd, journal.publish_name)
                == journal.new_marker_id
            )
        except ConfigError:
            return False

    def proved_backup() -> TargetIdentity | None:
        expected = journal.backup_identity
        if expected is None and journal.original.exists:
            expected = journal.original
        if backup.exists and expected is not None and _identity_matches(backup, expected):
            return backup
        return None
    if journal.phase == "preparing":
        if not _identity_matches(target, journal.original) or backup.exists:
            raise _ambiguous(lease, journal, "preparing identities conflict")
        if staging.exists:
            if journal.staging_identity is None or not _identity_matches(
                staging, journal.staging_identity
            ):
                raise _ambiguous(lease, journal, "preparing staging is unproved")
            _safe_remove_directory(lease.parent_fd, journal.staging_name, staging)
        _finish_journal(lease, journal.transaction_id)
        return RecoveryOutcome("cleaned_preparing", target.exists, ())

    if journal.phase in ("prepared", "backup_moved"):
        if is_new_publication() and not staging.exists:
            proved = proved_backup()
            if backup.exists and proved is None:
                raise _ambiguous(lease, journal, "published backup is unproved")
            if proved is not None:
                _safe_remove_directory(
                    lease.parent_fd,
                    cast(str, journal.backup_name),
                    proved,
                )
            _finish_journal(lease, journal.transaction_id)
            return RecoveryOutcome(
                "cleaned_backup" if proved is not None else "kept_published",
                True,
                (),
            )
        if not staging.exists or journal.staging_identity is None or not _identity_matches(
            staging, journal.staging_identity
        ):
            raise _ambiguous(lease, journal, "prepared staging is missing or changed")
        if _identity_matches(target, journal.original) and not backup.exists:
            _safe_remove_directory(lease.parent_fd, journal.staging_name, staging)
            _finish_journal(lease, journal.transaction_id)
            return RecoveryOutcome("cleaned_preparing", target.exists, ())
        if (
            not target.exists
            and journal.backup_name is not None
            and proved_backup() is not None
        ):
            _rename_noreplace(
                lease,
                platform,
                journal.backup_name,
                journal.guarded_target_name,
            )
            _fsync_directory(lease.parent_fd)
            _finish_journal(lease, journal.transaction_id)
            return RecoveryOutcome(
                "restored_backup",
                True,
                (journal.staging_name,),
            )
        raise _ambiguous(lease, journal, "prepared identities conflict")

    if journal.phase in ("staging_published", "target_durable", "backup_removed"):
        if staging.exists:
            raise _ambiguous(lease, journal, "published transaction still has staging")
        if journal.phase == "backup_removed" and backup.exists:
            raise _ambiguous(lease, journal, "removed backup unexpectedly reappeared")
        try:
            marker_id = _read_marker_id(lease.parent_fd, journal.publish_name)
        except ConfigError:
            raise _ambiguous(lease, journal, "published target marker is unproved") from None
        published = _node_identity(lease.parent_fd, journal.publish_name, marker_id)
        if marker_id != journal.new_marker_id or (
            journal.published_identity is not None
            and not _identity_matches(published, journal.published_identity)
        ):
            raise _ambiguous(lease, journal, "published target identity conflicts")
        if backup.exists:
            proved = proved_backup()
            if proved is None:
                raise _ambiguous(lease, journal, "published backup is unproved")
            _safe_remove_directory(
                lease.parent_fd,
                cast(str, journal.backup_name),
                proved,
            )
        _finish_journal(lease, journal.transaction_id)
        return RecoveryOutcome(
            "cleaned_backup" if backup.exists else "kept_published",
            journal.guarded_target_name == journal.publish_name,
            (),
        )
    raise _ambiguous(lease, journal, "unsupported success phase")


def _recover_failure(
    lease: OutputLease,
    journal: TransactionJournal,
) -> RecoveryOutcome:
    guarded = _node_identity(lease.parent_fd, journal.guarded_target_name)
    staging = _node_identity(lease.parent_fd, journal.staging_name)
    published = _node_identity(lease.parent_fd, journal.publish_name)
    if journal.backup_name is not None or journal.publish_name == journal.guarded_target_name:
        raise _ambiguous(lease, journal, "failure transaction can mutate guarded target")
    if journal.phase in ("preparing", "prepared"):
        if published.exists and not staging.exists:
            marker_id = _read_marker_id(lease.parent_fd, journal.publish_name)
            if marker_id != journal.new_marker_id:
                raise _ambiguous(lease, journal, "failure sibling marker conflicts")
            _finish_journal(lease, journal.transaction_id)
            return RecoveryOutcome(
                "completed_failure_publication",
                guarded.exists,
                (),
            )
        if published.exists:
            raise _ambiguous(lease, journal, "unpublished failure sibling already exists")
        if staging.exists:
            if journal.staging_identity is None or not _identity_matches(
                staging, journal.staging_identity
            ):
                raise _ambiguous(lease, journal, "failure staging is unproved")
            _safe_remove_directory(lease.parent_fd, journal.staging_name, staging)
        _finish_journal(lease, journal.transaction_id)
        return RecoveryOutcome("cleaned_preparing", guarded.exists, ())
    if journal.phase in ("staging_published", "target_durable"):
        if staging.exists:
            raise _ambiguous(lease, journal, "published failure still has staging")
        marker_id = _read_marker_id(lease.parent_fd, journal.publish_name)
        current_published = _node_identity(
            lease.parent_fd, journal.publish_name, marker_id
        )
        if marker_id != journal.new_marker_id or (
            journal.published_identity is not None
            and not _identity_matches(current_published, journal.published_identity)
        ):
            raise _ambiguous(lease, journal, "failure sibling marker conflicts")
        _finish_journal(lease, journal.transaction_id)
        return RecoveryOutcome("completed_failure_publication", guarded.exists, ())
    raise _ambiguous(lease, journal, "unsupported failure phase")


def recover_transaction(
    lease: OutputLease,
    platform: OutputPlatform,
) -> RecoveryOutcome:
    """Recover one proved canonical transaction without guessing ownership."""
    require_open_output_lease(lease, platform)
    revalidate_output_ancestry(lease, platform)
    journal, temp_name = _journal_for_recovery(lease, platform)
    if journal is None:
        return RecoveryOutcome(
            "none",
            _node_identity(lease.parent_fd, lease.target_name).exists,
            (),
        )
    if (
        journal.journal_name != lease.journal_name
        or journal.guarded_target_name != lease.target_name
    ):
        raise _ambiguous(lease, journal, "journal is not derived from this target")
    expected_staging = staging_name(lease.request.target_path, journal.transaction_id)
    expected_backup = (
        backup_name(lease.request.target_path, journal.transaction_id)
        if journal.backup_name is not None
        else None
    )
    if journal.staging_name != expected_staging or journal.backup_name != expected_backup:
        raise _ambiguous(lease, journal, "journal sibling codec is invalid")
    if journal.publication == "success":
        outcome = _recover_success(lease, platform, journal)
    else:
        outcome = _recover_failure(lease, journal)
    _remove_recovery_temp(lease, temp_name)
    return outcome


__all__ = [
    "PersistenceEvent",
    "TRANSACTION_BOUNDARIES",
    "TransactionInterrupted",
    "capture_persistence_events",
    "discard_staging",
    "observe_persistence",
    "publish_failure",
    "publish_success",
    "recover_transaction",
    "replace_staged_metadata",
    "stage_bundle",
]
