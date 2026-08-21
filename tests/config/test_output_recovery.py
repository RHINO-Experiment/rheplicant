from __future__ import annotations

import shutil

import pytest

import _rheplicant_bootstrap.output.transaction as transaction
from _rheplicant_bootstrap.errors import ConfigError
from _rheplicant_bootstrap.output import close_output_lease
from _rheplicant_bootstrap.output.paths import journal_temp_name
from _rheplicant_bootstrap.output.transaction import (
    TransactionInterrupted,
    observe_persistence,
    publish_success,
    recover_transaction,
    replace_staged_metadata,
    stage_bundle,
)
from tests.config.test_output_transaction import bundle, lease_for, staging_path


def test_recovery_removes_proved_preparing_staging_and_journal(tmp_path):
    _target, platform, lease, _publication, verified = lease_for(tmp_path)
    handle, _ = stage_bundle(verified, bundle(), platform, publication="success")
    try:
        outcome = recover_transaction(lease, platform)
        assert outcome.action == "cleaned_preparing"
        assert not staging_path(lease, handle).exists()
        assert not (tmp_path / lease.journal_name).exists()
    finally:
        close_output_lease(lease)


def test_recovery_discards_prepared_staging_before_publication(tmp_path):
    target, platform, lease, _publication, verified = lease_for(tmp_path)
    candidate = bundle()
    handle, _ = stage_bundle(verified, candidate, platform, publication="success")
    replace_staged_metadata(handle, candidate, platform)
    try:
        outcome = recover_transaction(lease, platform)
        assert outcome.action == "cleaned_preparing"
        assert not target.exists()
        assert not staging_path(lease, handle).exists()
    finally:
        close_output_lease(lease)


def test_crash_after_backup_move_restores_old_target_and_preserves_staging(tmp_path):
    target, platform, lease, _publication, verified = lease_for(tmp_path, existing=True)
    candidate = bundle()
    handle, _ = stage_bundle(verified, candidate, platform, publication="success")
    replace_staged_metadata(handle, candidate, platform)

    def fail_after_first_rename(event):
        if event.label == "rename_noreplace":
            raise OSError("crash after rename")

    try:
        with observe_persistence(fail_after_first_rename):
            with pytest.raises(TransactionInterrupted):
                publish_success(handle, platform)
        assert not target.exists()
        outcome = recover_transaction(lease, platform)
        assert outcome.action == "restored_backup"
        assert target.is_dir()
        assert staging_path(lease, handle).is_dir()
        assert handle.staging_name in outcome.preserved_names
    finally:
        close_output_lease(lease)


def test_crash_after_fresh_publish_keeps_complete_new_target(tmp_path):
    target, platform, lease, _publication, verified = lease_for(tmp_path)
    candidate = bundle()
    handle, _ = stage_bundle(verified, candidate, platform, publication="success")
    replace_staged_metadata(handle, candidate, platform)

    def fail_after_rename(event):
        if event.label == "rename_noreplace":
            raise OSError("crash after rename")

    try:
        with observe_persistence(fail_after_rename):
            with pytest.raises(TransactionInterrupted):
                publish_success(handle, platform)
        assert target.is_dir()
        outcome = recover_transaction(lease, platform)
        assert outcome.action == "kept_published"
        assert (target / "provenance.json").read_bytes() == candidate.provenance
    finally:
        close_output_lease(lease)


@pytest.mark.parametrize("status", ("refused", "error"))
def test_crash_after_failure_publish_completes_only_the_failure_sibling(
    tmp_path, status
):
    target, platform, lease, publication, _verified = lease_for(tmp_path)
    candidate = bundle(status)
    handle, _ = stage_bundle(
        publication,
        candidate,
        platform,
        publication=status,
    )
    replace_staged_metadata(handle, candidate, platform)

    def fail_after_rename(event):
        if event.label == "rename_noreplace":
            raise OSError("crash after failure rename")

    try:
        with observe_persistence(fail_after_rename):
            with pytest.raises(TransactionInterrupted):
                from _rheplicant_bootstrap.output.transaction import publish_failure

                publish_failure(handle, platform)
        outcome = recover_transaction(lease, platform)
        assert outcome.action == "completed_failure_publication"
        assert not target.exists()
        failure = tmp_path / handle.publish_name
        assert (failure / "provenance.json").read_bytes() == candidate.provenance
    finally:
        close_output_lease(lease)


def test_complete_uncommitted_phase_temp_is_cleaned_not_promoted(tmp_path):
    _target, platform, lease, _publication, verified = lease_for(tmp_path)
    candidate = bundle()
    handle, _ = stage_bundle(verified, candidate, platform, publication="success")
    file_fsyncs = 0

    def fail_after_journal_temp_fsync(event):
        nonlocal file_fsyncs
        if event.label == "file_fsync":
            file_fsyncs += 1
            if file_fsyncs == 3:
                raise OSError("crash after phase temp fsync")

    try:
        with observe_persistence(fail_after_journal_temp_fsync):
            with pytest.raises(TransactionInterrupted):
                replace_staged_metadata(handle, candidate, platform)
        temp = journal_temp_name(lease.request.target_path, handle.transaction_id, "prepared")
        assert (tmp_path / temp).exists()
        outcome = recover_transaction(lease, platform)
        assert outcome.action == "cleaned_preparing"
        assert not (tmp_path / temp).exists()
    finally:
        close_output_lease(lease)


def test_two_or_partial_update_temps_are_preserved_and_refused(tmp_path):
    _target, platform, lease, _publication, _verified = lease_for(tmp_path)
    digest = lease.journal_name.removeprefix(".rheplicant-journal-").removesuffix(".json")
    names = (
        f".rheplicant-jtmp-{digest}-partial.tmp",
        f".rheplicant-jtmp-{digest}-other.tmp",
    )
    for name in names:
        (tmp_path / name).write_text("partial")
    try:
        with pytest.raises(ConfigError, match="no path was removed"):
            recover_transaction(lease, platform)
        assert all((tmp_path / name).exists() for name in names)
    finally:
        close_output_lease(lease)


def test_recovery_never_deletes_unrecognized_backup(tmp_path):
    _target, platform, lease, _publication, verified = lease_for(tmp_path)
    handle, _ = stage_bundle(verified, bundle(), platform, publication="success")
    foreign = tmp_path / ".rheplicant-backup-foreign"
    foreign.mkdir()
    try:
        recover_transaction(lease, platform)
        assert foreign.is_dir()
        assert not staging_path(lease, handle).exists()
    finally:
        close_output_lease(lease)


def test_failure_recovery_refuses_a_same_marker_replacement_inode(
    tmp_path, monkeypatch
):
    _target, platform, lease, publication, _verified = lease_for(tmp_path)
    candidate = bundle("error")
    handle, _ = stage_bundle(
        publication, candidate, platform, publication="error"
    )
    replace_staged_metadata(handle, candidate, platform)
    original_finish = transaction._finish_journal

    def interrupt_finish(*_args):
        raise OSError("interrupt after durable publication")

    monkeypatch.setattr(transaction, "_finish_journal", interrupt_finish)
    with pytest.raises(TransactionInterrupted):
        transaction.publish_failure(handle, platform)
    monkeypatch.setattr(transaction, "_finish_journal", original_finish)
    published = tmp_path / handle.publish_name
    old = tmp_path / f"{handle.publish_name}.old"
    published.rename(old)
    shutil.copytree(old, published)
    try:
        with pytest.raises(ConfigError, match="no path was removed"):
            recover_transaction(lease, platform)
        assert published.is_dir()
        assert old.is_dir()
        assert (tmp_path / lease.journal_name).is_file()
    finally:
        close_output_lease(lease)
