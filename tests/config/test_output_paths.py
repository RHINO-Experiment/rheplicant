from __future__ import annotations

import os
import stat

import pytest

import _rheplicant_bootstrap.output.manager as manager
from _rheplicant_bootstrap.errors import ConfigError
from _rheplicant_bootstrap.output import inspect_output_path
from _rheplicant_bootstrap.output.paths import (
    TRANSACTION_PHASES,
    backup_name,
    decode_journal_temp,
    failure_name,
    internal_names,
    journal_name,
    journal_temp_name,
    lock_name,
    require_component_budget,
    staging_name,
)
from _rheplicant_bootstrap.output.types import (
    AccessInspection,
    AncestorEntryInspection,
    OutputRequest,
)


class SafePlatform:
    no_replace_available = True

    def inspect_access(self, directory_fd, parent_path):
        row = os.fstat(directory_fd)
        return AccessInspection(
            parent_path,
            os.geteuid(),
            row.st_uid,
            stat.S_IMODE(row.st_mode),
            True,
            True,
            False,
            True,
            None,
        )

    def inspect_ancestor_entry(
        self, containing_fd, containing_path, child_name, child_stat
    ):
        row = os.fstat(containing_fd)
        sticky = bool(row.st_mode & stat.S_ISVTX)
        writable = bool(stat.S_IMODE(row.st_mode) & 0o022)
        protected = not writable or (sticky and child_stat.st_uid == os.geteuid())
        return AncestorEntryInspection(
            containing_path,
            child_name,
            row.st_dev,
            row.st_ino,
            child_stat.st_dev,
            child_stat.st_ino,
            child_stat.st_uid,
            sticky,
            protected,
            True,
            None if protected else "ancestor entry can be renamed by a non-owner",
        )

    def verify_rename_noreplace_available(self, _directory_fd):
        if not self.no_replace_available:
            raise ConfigError("atomic no-replace rename is unavailable")

    def rename_noreplace(self, old_dir_fd, old_name, new_dir_fd, new_name):
        try:
            os.lstat(new_name, dir_fd=new_dir_fd)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(new_name)
        os.rename(old_name, new_name, src_dir_fd=old_dir_fd, dst_dir_fd=new_dir_fd)


def request_for(path, *, command="validate", clobber=False):
    return OutputRequest(command, str(path), True, clobber, "summary", True, True, "json")


def test_walk_rejects_intermediate_and_leaf_symlinks(tmp_path):
    platform = SafePlatform()
    real = tmp_path / "real"
    real.mkdir()
    (tmp_path / "middle").symlink_to(real, target_is_directory=True)
    with pytest.raises(ConfigError, match="intermediate symlink"):
        inspect_output_path(request_for(tmp_path / "middle" / "result"), platform)
    (tmp_path / "leaf").symlink_to(real, target_is_directory=True)
    with pytest.raises(ConfigError, match="output target is a symlink"):
        inspect_output_path(request_for(tmp_path / "leaf"), platform)


def test_intermediate_open_refuses_symlink_swap_to_the_same_inode(tmp_path, monkeypatch):
    platform = SafePlatform()
    middle = tmp_path / "middle"
    held = tmp_path / "held"
    middle.mkdir()
    original_open = manager.os.open
    attacked = False

    def open_with_swap(path, flags, *args, **kwargs):
        nonlocal attacked
        if path == "middle" and kwargs.get("dir_fd") is not None and not attacked:
            attacked = True
            middle.rename(held)
            middle.symlink_to(held.name, target_is_directory=True)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(manager.os, "open", open_with_swap)
    with pytest.raises(ConfigError, match="cannot be opened safely"):
        inspect_output_path(request_for(middle / "result"), platform)
    assert attacked is True


def test_existing_non_directory_is_refused(tmp_path):
    leaf = tmp_path / "result"
    leaf.write_text("not a directory")
    with pytest.raises(ConfigError, match="not a directory"):
        inspect_output_path(request_for(leaf), SafePlatform())


def test_missing_suffix_is_reported_without_creation(tmp_path):
    target = tmp_path / "one" / "two" / "result"
    inspection = inspect_output_path(request_for(target), SafePlatform())
    assert inspection.nearest_existing_ancestor == str(tmp_path)
    assert inspection.missing_components == ("one", "two")
    assert inspection.target.exists is False
    assert not (tmp_path / "one").exists()
    assert not any(name.startswith(".rheplicant-") for name in os.listdir(tmp_path))


def test_internal_name_codecs_are_disjoint_and_round_trip(tmp_path):
    target = str(tmp_path / "result")
    transaction_id = "a" * 32
    assert lock_name(target) != journal_name(target)
    assert staging_name(target, transaction_id) != backup_name(target, transaction_id)
    for phase in TRANSACTION_PHASES:
        name = journal_temp_name(target, transaction_id, phase)
        assert decode_journal_temp(target, name) == (transaction_id, phase)
        assert name != journal_name(target)


def test_every_budgeted_name_passes_at_max_and_fails_one_over(tmp_path):
    target = str(tmp_path / "result")
    names = (*internal_names(target), "result")
    maximum = max(len(os.fsencode(name)) for name in names)
    require_component_budget(names, maximum)
    longest = max(names, key=lambda name: len(os.fsencode(name)))
    with pytest.raises(ConfigError, match="NAME_MAX"):
        require_component_budget((longest,), maximum - 1)


def test_internal_names_is_an_exact_bound_on_what_a_transaction_publishes():
    """Every name the runtime can write is budgeted, and budgeted exactly.

    ``manager.py`` pre-flights the NAME_MAX budget from ``internal_names``
    alone, so if that tuple under-bounds any name a transaction actually
    creates, the check passes and the write fails later on a filesystem the
    caller was told was fine.

    It used to carry a second, hand-written list beside it --
    ``_failure_budget_names`` -- spelling out a timestamp-and-pid failure
    sibling scheme. A7.6 replaced that scheme with a transaction-id one and the
    hand-written copy stayed, describing a format nothing produced any more.
    Measured at the time: the stale copy's longest name was 40 characters past
    the leaf against the real scheme's 41, so it could only have bound on a
    system with a nine-digit pid, and Linux caps ``pid_max`` at 2**22 -- seven
    digits. It was dead weight that could never fire, which is the worst kind:
    it read as protection.

    This is what replaces it. The bound is asserted EXACT rather than
    sufficient, because a transaction id is always exactly 32 hex characters,
    so ``internal_names``' ``"f" * 32`` is not a pessimistic stand-in -- it is
    the same length as every real id, and a name that is longer here is a name
    the budget genuinely does not cover.

    **What this cannot see, measured rather than assumed.** ``internal_names``
    calls the same constructors this test calls, so lengthening ``failure_name``
    itself moves both sides together and the assertion stays green -- confirmed
    by mutation. That is not a hole: it is derivation working, since a change
    to a constructor propagates into the budget automatically. The failure this
    guards is the budget DIVERGING from the real construction, and both shapes
    of that are killed -- ``"f" * 32`` shortened to ``"f" * 16`` fails on
    length, and dropping a name from the tuple fails on count.
    """
    target = "/work/result"
    real = "0123456789abcdef" * 2
    assert len(real) == 32, "a transaction id is fixed-length hex"

    budget = {len(os.fsencode(name)) for name in internal_names(target)}
    published = [
        lock_name(target),
        journal_name(target),
        staging_name(target, real),
        backup_name(target, real),
        failure_name(target, "refused", real),
        failure_name(target, "error", real),
        *(journal_temp_name(target, real, phase) for phase in TRANSACTION_PHASES),
    ]
    assert len(published) == len(internal_names(target)), (
        "internal_names and the names a transaction publishes have drifted "
        "apart in COUNT; a name that exists in one and not the other is "
        "either unbudgeted or budgeted and never written."
    )
    for name in published:
        assert len(os.fsencode(name)) in budget, (
            f"{name!r} is {len(os.fsencode(name))} bytes and internal_names "
            f"budgets only {sorted(budget)}; the pre-flight NAME_MAX check "
            "would pass and the write would fail."
        )


def test_recovery_census_is_read_only(tmp_path):
    target = tmp_path / "result"
    target.mkdir()
    absolute = str(target)
    (tmp_path / journal_name(absolute)).write_text("{}")
    temp = journal_temp_name(absolute, "b" * 32, "prepared")
    (tmp_path / temp).write_text("{}")
    inspection = inspect_output_path(request_for(target), SafePlatform())
    assert inspection.recovery.canonical_present is True
    assert inspection.recovery.update_temp_names == (temp,)
    assert inspection.recovery.requires_recovery is True


def test_partial_and_multiple_update_temps_get_fixed_reasons(tmp_path):
    target = tmp_path / "result"
    target.mkdir()
    prefix = f".rheplicant-jtmp-{journal_name(str(target)).split('-')[2].split('.')[0]}-"
    (tmp_path / f"{prefix}partial.tmp").write_text("x")
    partial = inspect_output_path(request_for(target), SafePlatform())
    assert partial.recovery.reason == "illegal transaction update temporary"
    for entry in tuple(tmp_path.iterdir()):
        if entry.name.startswith(".rheplicant-jtmp-"):
            entry.unlink()
    for phase in ("prepared", "target_durable"):
        (tmp_path / journal_temp_name(str(target), "c" * 32, phase)).write_text("x")
    multiple = inspect_output_path(request_for(target), SafePlatform())
    assert multiple.recovery.reason == "multiple transaction update temporaries"
