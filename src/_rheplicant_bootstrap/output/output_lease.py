"""The A34 lease: the lifecycle that owns an output directory.

Acquiring, revalidating, verifying a publication under, and closing. This is
the half with STATE -- one process-wide lock, and a marker file whose identity
is what ownership means -- which is why it is the piece the ruling singled out
as having its own lifecycle.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import stat
import threading

from _rheplicant_bootstrap.errors import ConfigError

from .output_primitives import (
    _OPEN_DIRECTORY,
    _component_limit,
    _open_child,
    _require_entry,
    _same_identity,
)
from .paths import (
    internal_names,
    journal_name,
    lock_name,
    require_component_budget,
)
from .platform import OutputPlatform
from .types import (
    AncestorEntryInspection,
    OutputBinding,
    OutputLease,
    OutputMarker,
    OutputPathInspection,
    PublicationLease,
    TargetIdentity,
    VerifiedOutputLease,
)

_MARKER_NAME = ".rheplicant-results.json"

_MARKER_ID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}")

#: Serialises the one mutable field an output record owns -- a lease's
#: ``binding.closed`` -- so that two threads closing the same lease unlock and
#: close its descriptors exactly once. It used to guard three module-level
#: registries keyed on ``id()``; see :class:`~.types.OutputBinding` for why
#: those are gone and what replaced them.
_STATE_LOCK = threading.Lock()


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
    # No lock: an inspection's binding is written once, before the record is
    # returned, and never again. An inspection built by hand carries a default
    # binding whose platform is None, which is refused here exactly as an
    # unregistered one was.
    if inspection.binding.platform is not platform:
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
            OutputBinding(platform),
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
    # The discard of a stale closed-flag that used to stand here existed only
    # because a recycled id could arrive already marked closed. A binding is
    # born open and belongs to this lease alone, so there is nothing to clear.
    return lease


def require_open_output_lease(
    lease: OutputLease,
    platform: OutputPlatform | None = None,
) -> None:
    if type(lease) is not OutputLease:
        raise ConfigError("output operation requires an exact OutputLease.")
    binding = lease.binding
    with _STATE_LOCK:
        closed, bound = binding.closed, binding.platform
    # Closing drops the adapter as well as setting the flag, so a lease built
    # by hand -- whose binding never had one -- reads as closed here, which is
    # what an unregistered lease has always reported.
    if closed or bound is None:
        raise ConfigError("output lease is closed.")
    if platform is not None and bound is not platform:
        raise ConfigError("output operation requires the lease platform adapter.")
    try:
        os.fstat(lease.parent_fd)
        os.fstat(lease.lock_fd)
    except OSError:
        raise ConfigError("output lease descriptors are closed.") from None


def close_output_lease(lease: OutputLease) -> None:
    if type(lease) is not OutputLease:
        raise ConfigError("close requires an exact OutputLease.")
    binding = lease.binding
    with _STATE_LOCK:
        if binding.closed:
            return
        binding.closed = True
        binding.platform = None
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
    if access.access_acl_grants_others or not access.default_acl_is_trivial:
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
