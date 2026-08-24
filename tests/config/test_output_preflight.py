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


class TestARecordOwnsItsOwnBinding:
    """The adapter binding and the closed flag belong to one record each.

    They used to live in three module-level registries keyed on ``id()``.
    ``id()`` is unique only among objects alive at the SAME moment, so those
    keys outlived their objects -- ``_INSPECTION_PLATFORMS`` and
    ``_CLOSED_LEASES`` were never pruned at all -- and CPython hands a freed
    address straight to the next record of that shape.

    Measured before the change, on this fixture: build an inspection through
    ``inspect_output_path``, drop the only reference, construct another
    directly on the next statement. The new one landed at the old one's
    address (reproduced on demand; about 2 in 50 when other allocations
    intervene), inherited the entry, and ``acquire_output_lease`` handed it a
    real lease -- passing a guard it had never been registered for.

    That reproduction is deliberately NOT an assertion here: it depends on the
    allocator, so a test built on it would demonstrate the hazard only
    sometimes. What is asserted instead is the property that makes the address
    irrelevant -- a binding is reachable only through the record that owns it.
    """

    def test_an_inspection_built_by_hand_carries_no_adapter(self, tmp_path):
        """Whatever address it lands on, and that is the point.

        Under the old registry this passed or failed according to where CPython
        put the object.
        """
        target = tmp_path / "result"
        platform = SafePlatform()
        original = manager.inspect_output_path(run_request(target), platform)
        forged = manager.OutputPathInspection(
            original.request,
            original.absolute_target,
            original.nearest_existing_ancestor,
            original.missing_components,
            original.parent_path,
            original.target_name,
            original.target,
            original.access,
            original.ancestry,
            original.recovery,
            original.component_limit,
        )

        assert forged.binding.platform is None
        with pytest.raises(ConfigError, match="same platform adapter"):
            acquire_output_lease(forged, platform)

    def test_a_lease_built_by_hand_reads_as_closed(self, tmp_path):
        """The descriptors in a forged lease are never fstat'ed.

        ``parent_fd=1`` and ``lock_fd=2`` are open in every process, so a lease
        that got past the binding check would go on to operate on stdout and
        stderr. The refusal has to come before that, from the binding.
        """
        target = tmp_path / "result"
        platform = SafePlatform()
        real = acquire_output_lease(
            manager.inspect_output_path(run_request(target), platform), platform
        )
        try:
            forged = manager.OutputLease(
                real.request, 1, "/", "result", 2, ".lock", ".journal",
                real.ancestry, real.component_limit,
            )
            with pytest.raises(ConfigError, match="lease is closed"):
                manager.require_open_output_lease(forged, platform)
        finally:
            close_output_lease(real)

    def test_two_adapters_that_compare_equal_are_still_two_adapters(self, tmp_path):
        """The check is identity, and an equality test would not say so.

        ``SafePlatform`` inherits identity equality, so ``is not`` and ``!=``
        agree on it and neither the old id() comparison nor a new one would be
        caught choosing wrongly. An adapter that declares itself equal to
        everything separates them: it is still not the object that walked the
        ancestry under those descriptors.
        """
        class AgreeablePlatform(SafePlatform):
            def __eq__(self, other):
                return True

            __hash__ = None

        target = tmp_path / "result"
        first = AgreeablePlatform()
        second = AgreeablePlatform()
        inspection = manager.inspect_output_path(run_request(target), first)

        assert first == second
        with pytest.raises(ConfigError, match="same platform adapter"):
            acquire_output_lease(inspection, second)

        lease = acquire_output_lease(inspection, first)
        try:
            with pytest.raises(ConfigError, match="lease platform adapter"):
                manager.require_open_output_lease(lease, second)
        finally:
            close_output_lease(lease)

    def test_no_module_level_container_grows_with_records(self, tmp_path):
        """The general form: reintroducing any id-keyed registry fails here.

        Both dictionaries this finds today are constant format tables. A guard
        on their names would have to be updated to stay true; a guard on "did
        anything module-level grow" does not.
        """
        platform = SafePlatform()

        def census():
            return {
                name: len(value)
                for name, value in vars(manager).items()
                if isinstance(value, (dict, set, list)) and not name.startswith("__")
            }

        before = census()
        for index in range(8):
            target = tmp_path / f"result{index}"
            lease = acquire_output_lease(
                manager.inspect_output_path(run_request(target), platform), platform
            )
            close_output_lease(lease)
            del lease

        assert census() == before, (
            "Something module-level in manager.py grew once per record. That is "
            "the shape of an id()-keyed registry, whose keys outlive the objects "
            "they name."
        )

    def test_each_record_gets_a_binding_of_its_own(self, tmp_path):
        """The invariant the whole change rests on, asserted directly.

        A default shared between records would pass every test above -- forged
        records would still carry no adapter -- while making one close mark
        them all closed. Caught by mutating the default factory to hand out a
        single module-level instance; the two halves below are the object
        check and the behaviour it buys.
        """
        platform = SafePlatform()
        first = acquire_output_lease(
            manager.inspect_output_path(run_request(tmp_path / "one"), platform), platform
        )
        second = acquire_output_lease(
            manager.inspect_output_path(run_request(tmp_path / "two"), platform), platform
        )
        try:
            assert first.binding is not second.binding
            close_output_lease(first)
            manager.require_open_output_lease(second, platform)
        finally:
            close_output_lease(second)

        template = (second.request, 1, "/", "x", 2, ".l", ".j", second.ancestry, 255)
        assert manager.OutputLease(*template).binding is not manager.OutputLease(*template).binding

    def test_a_lease_taken_after_a_close_is_open(self, tmp_path):
        """Regression for a workaround that was deleted with its cause.

        ``_open_lease`` used to discard the new lease's id from the closed set,
        because a recycled id could arrive already marked closed. A binding is
        born open and belongs to one lease, so the discard has nothing to do --
        but the behaviour it protected still has to hold.
        """
        target = tmp_path / "result"
        platform = SafePlatform()
        first = acquire_output_lease(
            manager.inspect_output_path(run_request(target), platform), platform
        )
        close_output_lease(first)

        second = acquire_output_lease(
            manager.inspect_output_path(run_request(target), platform), platform
        )
        try:
            manager.require_open_output_lease(second, platform)
        finally:
            close_output_lease(second)

    def test_closing_twice_closes_the_descriptors_once(self, tmp_path):
        """An fd number is recycled exactly as an id() is.

        A second close that got past the flag would call ``os.close`` on numbers
        the kernel has already handed to something else. The early return is
        what stands between this and closing another component's file.
        """
        target = tmp_path / "result"
        platform = SafePlatform()
        lease = acquire_output_lease(
            manager.inspect_output_path(run_request(target), platform), platform
        )
        closed = []
        original = manager.os.close

        def record(fd):
            closed.append(fd)
            return original(fd)

        manager.os.close = record
        try:
            close_output_lease(lease)
            close_output_lease(lease)
        finally:
            manager.os.close = original

        assert sorted(closed) == sorted({lease.lock_fd, lease.parent_fd})
        assert lease.binding.closed is True
        assert lease.binding.platform is None
