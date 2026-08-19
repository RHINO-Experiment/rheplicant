from __future__ import annotations

import json
import stat

import pytest

import _rheplicant_bootstrap.output.manager as manager
from _rheplicant_bootstrap.errors import ConfigError
from _rheplicant_bootstrap.output import (
    acquire_output_lease,
    close_output_lease,
    inspect_output_path,
    verify_a34_under_lease,
    verify_publication_under_lease,
)
from tests.config.test_output_paths import SafePlatform, request_for


def canonical_marker(identifier="12345678-1234-4234-9234-123456789abc"):
    return (
        json.dumps(
            {"format_version": 1, "run_directory_id": identifier},
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode()


def make_owned_result(tmp_path, marker=None):
    target = tmp_path / "result"
    target.mkdir()
    target.chmod(0o700)
    marker_path = target / ".rheplicant-results.json"
    marker_path.write_bytes(
        canonical_marker() if marker is None else marker
    )
    marker_path.chmod(0o600)
    return target


def run_request(path, *, clobber=False):
    return request_for(path, command="run", clobber=clobber)


def test_run_creates_only_missing_parents_and_persistent_lock(tmp_path):
    target = tmp_path / "private" / "result"
    platform = SafePlatform()
    inspection = inspect_output_path(run_request(target), platform)
    lease = acquire_output_lease(inspection, platform)
    try:
        assert (tmp_path / "private").is_dir()
        assert stat.S_IMODE((tmp_path / "private").stat().st_mode) == 0o700
        assert (tmp_path / "private" / lease.lock_name).is_file()
        publication = verify_publication_under_lease(lease, platform)
        verified = verify_a34_under_lease(publication, platform)
        assert verified.original.exists is False
        assert publication.lease is lease
    finally:
        close_output_lease(lease)
    assert (tmp_path / "private" / lease.lock_name).exists()


def test_validate_existing_and_missing_paths_never_create_lock_or_journal(tmp_path):
    platform = SafePlatform()
    targets = (tmp_path / "missing" / "result", tmp_path / "existing")
    targets[1].mkdir()
    before = tuple(sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*")))
    for target in targets:
        inspection = inspect_output_path(request_for(target), platform)
        assert inspection.request.command == "validate"
    after = tuple(sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*")))
    assert after == before
    assert not any("lock" in path.name or "journal" in path.name for path in tmp_path.rglob("*"))


def test_foreign_or_oversize_marker_never_authorizes_clobber(tmp_path):
    platform = SafePlatform()
    target = make_owned_result(tmp_path, marker=b"{}")
    inspection = inspect_output_path(run_request(target, clobber=True), platform)
    lease = acquire_output_lease(inspection, platform)
    try:
        publication = verify_publication_under_lease(lease, platform)
        with pytest.raises(ConfigError, match="ownership marker"):
            verify_a34_under_lease(publication, platform)
    finally:
        close_output_lease(lease)
    (target / ".rheplicant-results.json").write_bytes(b"x" * 4097)
    inspection = inspect_output_path(run_request(target, clobber=True), platform)
    lease = acquire_output_lease(inspection, platform)
    try:
        publication = verify_publication_under_lease(lease, platform)
        with pytest.raises(ConfigError, match="at most 4096 bytes"):
            verify_a34_under_lease(publication, platform)
    finally:
        close_output_lease(lease)


def test_owned_marker_and_post_recovery_identity_are_authorized(tmp_path):
    target = make_owned_result(tmp_path)
    platform = SafePlatform()
    lease = acquire_output_lease(
        inspect_output_path(run_request(target, clobber=True), platform), platform
    )
    try:
        publication = verify_publication_under_lease(lease, platform)
        verified = verify_a34_under_lease(publication, platform)
        assert verified.publication is publication
        assert verified.original.exists is True
        assert verified.original.marker_id == "12345678-1234-4234-9234-123456789abc"
    finally:
        close_output_lease(lease)


def test_existing_target_without_clobber_is_refused_after_recovery(tmp_path):
    target = make_owned_result(tmp_path)
    platform = SafePlatform()
    lease = acquire_output_lease(inspect_output_path(run_request(target), platform), platform)
    try:
        publication = verify_publication_under_lease(lease, platform)
        with pytest.raises(ConfigError, match="clobber is false"):
            verify_a34_under_lease(publication, platform)
    finally:
        close_output_lease(lease)


def test_insecure_parent_and_unavailable_noreplace_fail_before_a34(tmp_path):
    target = make_owned_result(tmp_path)
    platform = SafePlatform()
    lease = acquire_output_lease(
        inspect_output_path(run_request(target, clobber=True), platform), platform
    )
    try:
        tmp_path.chmod(0o770)
        with pytest.raises(ConfigError, match="group or other writable"):
            verify_publication_under_lease(lease, platform)
        tmp_path.chmod(0o700)
        platform.no_replace_available = False
        with pytest.raises(ConfigError, match="atomic no-replace"):
            verify_publication_under_lease(lease, platform)
    finally:
        tmp_path.chmod(0o700)
        close_output_lease(lease)


def test_adapter_identity_is_bound_to_inspection_and_lease(tmp_path):
    target = tmp_path / "result"
    first = SafePlatform()
    second = SafePlatform()
    inspection = inspect_output_path(run_request(target), first)
    with pytest.raises(ConfigError, match="same platform adapter"):
        acquire_output_lease(inspection, second)
    lease = acquire_output_lease(inspection, first)
    try:
        with pytest.raises(ConfigError, match="lease platform adapter"):
            verify_publication_under_lease(lease, second)
    finally:
        close_output_lease(lease)


def test_name_max_is_sourced_and_rechecked_from_the_leased_parent_fd(
    tmp_path, monkeypatch
):
    target = tmp_path / "result"
    platform = SafePlatform()
    expected = (tmp_path.stat().st_dev, tmp_path.stat().st_ino)
    calls = []
    original = manager.os.fpathconf

    def fpathconf(fd, name):
        row = manager.os.fstat(fd)
        calls.append((name, row.st_dev, row.st_ino))
        return original(fd, name)

    monkeypatch.setattr(manager.os, "fpathconf", fpathconf)
    inspection = inspect_output_path(run_request(target), platform)
    lease = acquire_output_lease(inspection, platform)
    try:
        publication = verify_publication_under_lease(lease, platform)
        assert publication.component_limit == lease.component_limit
    finally:
        close_output_lease(lease)
    assert len(calls) == 3
    assert all(name == "PC_NAME_MAX" for name, _device, _inode in calls)
    assert all((device, inode) == expected for _name, device, inode in calls)


def test_nonowning_views_close_no_fd_and_owner_closes_each_once(tmp_path, monkeypatch):
    target = tmp_path / "result"
    platform = SafePlatform()
    lease = acquire_output_lease(inspect_output_path(run_request(target), platform), platform)
    publication = verify_publication_under_lease(lease, platform)
    verified = verify_a34_under_lease(publication, platform)
    assert verified.publication.lease is lease
    observed = []
    original = manager.os.close

    def close(fd):
        observed.append(fd)
        original(fd)

    monkeypatch.setattr(manager.os, "close", close)
    close_output_lease(lease)
    close_output_lease(lease)
    assert observed.count(lease.parent_fd) == 1
    assert observed.count(lease.lock_fd) == 1
    with pytest.raises(ConfigError, match="closed"):
        manager.require_open_output_lease(lease)


def test_lock_inode_is_regular_private_owned_and_single_link(tmp_path):
    target = tmp_path / "result"
    platform = SafePlatform()
    inspection = inspect_output_path(run_request(target), platform)
    parent = tmp_path
    lock_path = parent / manager.lock_name(str(target))
    lock_path.write_text("attacker")
    lock_path.chmod(0o644)
    chmod_calls = []
    original_chmod = manager.os.fchmod

    def fchmod(fd, mode):
        chmod_calls.append((fd, mode))
        original_chmod(fd, mode)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(manager.os, "fchmod", fchmod)
        with pytest.raises(ConfigError, match="persistent output lock"):
            acquire_output_lease(inspection, platform)
    assert chmod_calls == []


def test_lock_replacement_while_waiting_for_flock_is_refused(tmp_path, monkeypatch):
    target = tmp_path / "result"
    platform = SafePlatform()
    inspection = inspect_output_path(run_request(target), platform)
    lock_path = tmp_path / manager.lock_name(str(target))
    original_flock = manager.fcntl.flock
    attacked = False

    def flock(fd, operation):
        nonlocal attacked
        result = original_flock(fd, operation)
        if operation == manager.fcntl.LOCK_EX and not attacked:
            attacked = True
            lock_path.rename(tmp_path / "old-lock")
            lock_path.write_bytes(b"")
            lock_path.chmod(0o600)
        return result

    monkeypatch.setattr(manager.fcntl, "flock", flock)
    with pytest.raises(ConfigError, match="persistent output lock"):
        acquire_output_lease(inspection, platform)
    assert attacked is True


def test_ancestor_replacement_after_acquisition_is_refused(tmp_path):
    original_parent = tmp_path / "private"
    target = original_parent / "result"
    platform = SafePlatform()
    lease = acquire_output_lease(
        inspect_output_path(run_request(target), platform), platform
    )
    try:
        original_parent.rename(tmp_path / "held-private")
        original_parent.mkdir(mode=0o700)
        with pytest.raises(ConfigError, match="ancestry was replaced"):
            verify_publication_under_lease(lease, platform)
    finally:
        close_output_lease(lease)


def test_target_swap_after_marker_read_is_refused(tmp_path, monkeypatch):
    target = make_owned_result(tmp_path)
    platform = SafePlatform()
    lease = acquire_output_lease(
        inspect_output_path(run_request(target, clobber=True), platform), platform
    )
    publication = verify_publication_under_lease(lease, platform)
    original_read = manager._read_owned_marker

    def read_then_swap(parent_fd, target_name):
        result = original_read(parent_fd, target_name)
        target.rename(tmp_path / "held-result")
        replacement = tmp_path / "result"
        replacement.mkdir(mode=0o700)
        return result

    monkeypatch.setattr(manager, "_read_owned_marker", read_then_swap)
    try:
        with pytest.raises(ConfigError, match="identity changed"):
            verify_a34_under_lease(publication, platform)
    finally:
        close_output_lease(lease)


def test_lease_dataclass_has_no_original_target_field():
    from dataclasses import fields

    from _rheplicant_bootstrap.output.types import OutputLease

    assert "original" not in {field.name for field in fields(OutputLease)}
