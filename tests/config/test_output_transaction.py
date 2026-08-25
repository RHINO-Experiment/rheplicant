from __future__ import annotations

import dataclasses
import json
import os
import stat
from pathlib import Path

import pytest

import _rheplicant_bootstrap.output.transaction as transaction
from _rheplicant_bootstrap.audit.bundle import (
    candidate_serialization_snapshot,
    merge_bundle_files,
    serialize_bundle,
)
from _rheplicant_bootstrap.audit.types import ArtefactRecord, ArtefactTable
from _rheplicant_bootstrap.errors import ConfigError
from _rheplicant_bootstrap.output import (
    acquire_output_lease,
    close_output_lease,
    inspect_output_path,
    verify_a34_under_lease,
    verify_publication_under_lease,
)
from _rheplicant_bootstrap.output.paths import (
    TRANSACTION_PHASES,
    failure_name,
    journal_temp_name,
    staging_name,
)
from _rheplicant_bootstrap.output.transaction import (
    TRANSACTION_BOUNDARIES,
    TransactionInterrupted,
    capture_persistence_events,
    discard_staging,
    publish_failure,
    publish_success,
    replace_staged_metadata,
    stage_bundle,
)
from tests.config.test_audit_envelopes import INPUT, INPUT_SHA, snapshot
from tests.config.test_output_paths import SafePlatform
from tests.config.test_output_preflight import canonical_marker, run_request


def bundle(status="ok", *, artefacts=None):
    _trace, initial = snapshot(status)
    if artefacts is not None:
        initial = dataclasses.replace(initial, artefacts=artefacts)
    return serialize_bundle(
        candidate_serialization_snapshot(initial),
        status=status,
        input_bytes=INPUT,
        resolved=(),
    )


def bundle_with_products(status="ok", *, artefacts=None):
    return merge_bundle_files(
        bundle(status, artefacts=artefacts),
        {
            "runs/n-666f7277617264/arrays.npz": b"science",
            "products.json": b"manifest",
        },
    )


def lease_for(tmp_path, *, existing=False):
    target = tmp_path / "result"
    if existing:
        target.mkdir(mode=0o700)
        marker = target / ".rheplicant-results.json"
        marker.write_bytes(canonical_marker())
        marker.chmod(0o600)
    platform = SafePlatform()
    lease = acquire_output_lease(
        inspect_output_path(run_request(target, clobber=existing), platform),
        platform,
    )
    publication = verify_publication_under_lease(lease, platform)
    verified = verify_a34_under_lease(publication, platform)
    return target, platform, lease, publication, verified


def staging_path(lease, handle):
    return Path(lease.parent_path) / handle.staging_name


def test_prejournal_candidate_collision_selects_a_fresh_transaction_id(
    tmp_path, monkeypatch
):
    _target, platform, lease, _publication, verified = lease_for(tmp_path)
    first = "0" * 32
    second = "1" * 32
    ids = iter(
        (
            transaction.uuid.UUID(hex=first),
            transaction.uuid.UUID(hex=second),
            transaction.uuid.UUID("22222222-2222-4222-8222-222222222222"),
        )
    )
    monkeypatch.setattr(transaction.uuid, "uuid4", lambda: next(ids))
    collision = tmp_path / journal_temp_name(
        lease.request.target_path, first, "prepared"
    )
    collision.write_bytes(b"foreign")
    try:
        handle, _materialized = stage_bundle(
            verified, bundle(), platform, publication="success"
        )
        assert handle.transaction_id == second
        assert collision.read_bytes() == b"foreign"
        assert staging_path(lease, handle).name == staging_name(
            lease.request.target_path, second
        )
        discard_staging(handle, platform)
    finally:
        close_output_lease(lease)


def test_two_colliding_failure_first_choices_still_publish_distinct_siblings(
    tmp_path, monkeypatch
):
    """A7.6: a collision on the failure sibling's name must not repeat itself.

    ``publish_name`` for a refused/error sibling used to be built from a
    wall-clock timestamp and the process id -- neither of which the retry
    loop's freshly drawn ``transaction_id`` has any influence over.  Two
    failures whose first-choice name collided therefore collided on every
    one of the retry loop's 128 attempts too: the candidate name never
    changed between attempts, so ``stage_bundle`` raised "could not select a
    collision-free transaction id" instead of ever trying a name the
    collision could not repeat.

    Reproduced by pinning the SECOND failure's first-drawn transaction id to
    the value the FIRST failure already published under -- exactly what a
    second failure whose first choice repeats the first one's puts a real
    caller through -- and checking both still publish, in two distinct
    directories, without exhausting the retry loop.
    """
    _target, platform, lease, publication, _verified = lease_for(tmp_path)
    first_choice = "0" * 32
    second_choice = "1" * 32
    marker_id = transaction.uuid.UUID("22222222-2222-4222-8222-222222222222")
    ids = iter(
        (
            transaction.uuid.UUID(hex=first_choice),
            marker_id,
            transaction.uuid.UUID(hex=first_choice),  # collides with the first
            transaction.uuid.UUID(hex=second_choice),
            marker_id,
        )
    )
    monkeypatch.setattr(transaction.uuid, "uuid4", lambda: next(ids))

    try:
        first_candidate = bundle("error")
        first_handle, _ = stage_bundle(
            publication, first_candidate, platform, publication="error"
        )
        replace_staged_metadata(first_handle, first_candidate, platform)
        first_path = Path(publish_failure(first_handle, platform))

        second_candidate = bundle("error")
        second_handle, _ = stage_bundle(
            publication, second_candidate, platform, publication="error"
        )
        replace_staged_metadata(second_handle, second_candidate, platform)
        second_path = Path(publish_failure(second_handle, platform))
    finally:
        close_output_lease(lease)

    assert first_handle.transaction_id == first_choice
    assert second_handle.transaction_id == second_choice
    assert first_path != second_path
    assert first_path.is_dir()
    assert second_path.is_dir()


def test_staging_requires_a_typed_view_and_the_lease_platform(tmp_path):
    _target, platform, lease, _publication, verified = lease_for(tmp_path)
    try:
        with pytest.raises(ConfigError, match="VerifiedOutputLease"):
            stage_bundle(lease, bundle(), platform, publication="success")
        with pytest.raises(ConfigError, match="platform adapter"):
            stage_bundle(
                verified,
                bundle(),
                SafePlatform(),
                publication="success",
            )
        assert not (tmp_path / lease.journal_name).exists()
    finally:
        close_output_lease(lease)


def test_exact_success_budget_passes_and_one_byte_short_failure_budget_is_pure(
    tmp_path
):
    target = tmp_path / ("r" * 120)
    platform = SafePlatform()
    lease = acquire_output_lease(
        inspect_output_path(run_request(target), platform), platform
    )
    publication = verify_publication_under_lease(lease, platform)
    verified = verify_a34_under_lease(publication, platform)
    identifier = "f" * 32
    success_names = (
        lease.target_name,
        lease.journal_name,
        staging_name(lease.request.target_path, identifier),
        *(
            journal_temp_name(lease.request.target_path, identifier, phase)
            for phase in TRANSACTION_PHASES
        ),
    )
    success_limit = max(len(os.fsencode(name)) for name in success_names)
    object.__setattr__(lease, "component_limit", success_limit)
    object.__setattr__(publication, "component_limit", success_limit)
    try:
        candidate = bundle()
        handle, _ = stage_bundle(
            verified, candidate, platform, publication="success"
        )
        replace_staged_metadata(handle, candidate, platform)
        publish_success(handle, platform)
        worst_case_failure_name = failure_name(
            lease.request.target_path, "error", identifier
        )
        failure_limit = len(os.fsencode(worst_case_failure_name)) - 1
        assert failure_limit >= success_limit
        object.__setattr__(lease, "component_limit", failure_limit)
        object.__setattr__(publication, "component_limit", failure_limit)
        with pytest.raises(ConfigError, match="exceeds leased NAME_MAX"):
            stage_bundle(
                publication,
                bundle("error"),
                platform,
                publication="error",
            )
        assert not (tmp_path / lease.journal_name).exists()
    finally:
        close_output_lease(lease)


@pytest.mark.parametrize("mask", (0o000, 0o022, 0o077))
def test_stage_bundle_has_exact_modes_bytes_and_events(tmp_path, mask):
    _target, platform, lease, _publication, verified = lease_for(tmp_path)
    candidate = bundle()
    old_mask = os.umask(mask)
    try:
        with capture_persistence_events() as events:
            handle, materialized = stage_bundle(
                verified,
                candidate,
                platform,
                publication="success",
            )
            replace_staged_metadata(handle, candidate, platform)
        staged = staging_path(lease, handle)
        assert stat.S_IMODE(staged.stat().st_mode) == 0o700
        for relative, expected in candidate.files.items():
            path = staged / relative
            assert stat.S_IMODE(path.stat().st_mode) == 0o600
            assert path.read_bytes() == expected
        marker = staged / ".rheplicant-results.json"
        assert stat.S_IMODE(marker.stat().st_mode) == 0o600
        assert tuple(event.ordinal for event in events) == tuple(range(len(events)))
        assert {event.label for event in events} <= set(TRANSACTION_BOUNDARIES)
        metadata = {row.slot: row for row in materialized if row.slot in {
            "lock", "journal", "marker", "provenance", "diagnostics"
        }}
        assert set(metadata) == {"lock", "journal", "marker", "provenance", "diagnostics"}
        assert all((row.bytes, row.sha256) == (None, None) for row in metadata.values())
        assert any(row.slot == "input" and row.sha256 == INPUT_SHA for row in materialized)
        discard_staging(handle, platform)
    finally:
        os.umask(old_mask)
        close_output_lease(lease)


def test_products_share_staging_modes_and_do_not_extend_the_fixed_audit_table(
    tmp_path,
):
    target, platform, lease, _publication, verified = lease_for(tmp_path)
    candidate = bundle_with_products()
    try:
        handle, materialized = stage_bundle(
            verified,
            candidate,
            platform,
            publication="success",
        )
        replace_staged_metadata(handle, candidate, platform)
        publish_success(handle, platform)
        assert (target / "runs/n-666f7277617264/arrays.npz").read_bytes() == b"science"
        assert (target / "products.json").read_bytes() == b"manifest"
        assert stat.S_IMODE((target / "runs").stat().st_mode) == 0o700
        assert stat.S_IMODE(
            (target / "runs/n-666f7277617264/arrays.npz").stat().st_mode
        ) == 0o600
        assert {row.slot for row in materialized} == {
            "lock",
            "journal",
            "input",
            "provenance",
            "diagnostics",
            "marker",
        }
    finally:
        close_output_lease(lease)


def test_two_materialization_replaces_only_metadata_and_retains_written_history(tmp_path):
    _target, platform, lease, _publication, verified = lease_for(tmp_path)
    candidate = bundle()
    handle, _events = stage_bundle(
        verified, candidate, platform, publication="success"
    )

    def metadata(path):
        return ArtefactRecord(path, True, None, None, "metadata_envelope")

    final_table = ArtefactTable(
        marker=metadata(".rheplicant-results.json"),
        lock=metadata(lease.lock_name),
        journal=metadata(lease.journal_name),
        input=ArtefactRecord("config.input.yaml", True, len(INPUT), INPUT_SHA, None),
        resolved_base=ArtefactRecord(
            "config.resolved.yaml", False, None, None, "layer_not_complete"
        ),
        resolved_variants=(),
        provenance=metadata("provenance.json"),
        diagnostics=metadata("diagnostics.json"),
    )
    final = bundle(artefacts=final_table)
    try:
        assert candidate.provenance != final.provenance
        replace_staged_metadata(handle, final, platform)
        staged = staging_path(lease, handle)
        assert (staged / "provenance.json").read_bytes() == final.provenance
        assert (staged / "diagnostics.json").read_bytes() == final.diagnostics
        decoded = json.loads(final.provenance)
        for name in ("lock", "journal", "marker", "provenance", "diagnostics"):
            assert decoded["artefacts"][name]["written"] is True
            assert decoded["artefacts"][name]["reason"] == "metadata_envelope"
            assert decoded["artefacts"][name]["bytes"] is None
            assert decoded["artefacts"][name]["sha256"] is None
        discard_staging(handle, platform)
    finally:
        close_output_lease(lease)


def test_fresh_success_publishes_complete_tree_and_cleans_journal(tmp_path):
    target, platform, lease, _publication, verified = lease_for(tmp_path)
    candidate = bundle()
    try:
        handle, _ = stage_bundle(verified, candidate, platform, publication="success")
        replace_staged_metadata(handle, candidate, platform)
        path = publish_success(handle, platform)
        assert path == str(target)
        assert target.is_dir()
        assert (target / "config.input.yaml").read_bytes() == INPUT
        assert (target / "provenance.json").read_bytes() == candidate.provenance
        assert (target / "diagnostics.json").read_bytes() == candidate.diagnostics
        assert (target / ".rheplicant-results.json").is_file()
        assert not (tmp_path / lease.journal_name).exists()
        assert not any(path.name.startswith(".rheplicant-stage-") for path in tmp_path.iterdir())
    finally:
        close_output_lease(lease)


def test_clobber_replaces_owned_target_and_removes_proved_backup(tmp_path):
    target, platform, lease, _publication, verified = lease_for(tmp_path, existing=True)
    old_marker = verified.original.marker_id
    candidate = bundle()
    try:
        handle, _ = stage_bundle(verified, candidate, platform, publication="success")
        replace_staged_metadata(handle, candidate, platform)
        publish_success(handle, platform)
        new_marker = json.loads((target / ".rheplicant-results.json").read_bytes())
        assert new_marker["run_directory_id"] != old_marker
        assert not any(path.name.startswith(".rheplicant-backup-") for path in tmp_path.iterdir())
    finally:
        close_output_lease(lease)


@pytest.mark.parametrize("status", ("refused", "error"))
def test_failure_publication_never_mutates_main_target(tmp_path, status):
    target, platform, lease, publication, _verified = lease_for(tmp_path)
    candidate = bundle(status)
    try:
        handle, _ = stage_bundle(
            publication,
            candidate,
            platform,
            publication=status,
        )
        replace_staged_metadata(handle, candidate, platform)
        path = Path(publish_failure(handle, platform))
        assert path.name.startswith(f"result.{status}-")
        assert path.is_dir()
        assert not target.exists()
        assert (path / "config.input.yaml").read_bytes() == INPUT
        assert not (tmp_path / lease.journal_name).exists()
    finally:
        close_output_lease(lease)


def test_prepublication_failure_preserves_old_target(tmp_path, monkeypatch):
    target, platform, lease, _publication, verified = lease_for(tmp_path, existing=True)
    candidate = bundle()
    handle, _ = stage_bundle(verified, candidate, platform, publication="success")
    replace_staged_metadata(handle, candidate, platform)
    original = platform.rename_noreplace

    def refuse(*_args):
        raise OSError("before publish")

    monkeypatch.setattr(platform, "rename_noreplace", refuse)
    try:
        with pytest.raises(TransactionInterrupted):
            publish_success(handle, platform)
        marker = json.loads((target / ".rheplicant-results.json").read_bytes())
        assert marker["run_directory_id"] == verified.original.marker_id
    finally:
        monkeypatch.setattr(platform, "rename_noreplace", original)
        discard_staging(handle, platform)
        close_output_lease(lease)


def test_cleanup_checks_the_opened_root_before_removing_contents(
    tmp_path, monkeypatch
):
    _target, platform, lease, _publication, verified = lease_for(tmp_path)
    handle, _ = stage_bundle(verified, bundle(), platform, publication="success")
    original_open = transaction._open_directory
    swapped = False

    def replace_before_open(parent_fd, name):
        nonlocal swapped
        if name == handle.staging_name and not swapped:
            swapped = True
            os.rename(
                name,
                f"{name}.preserved",
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )
            os.mkdir(name, 0o700, dir_fd=parent_fd)
            replacement_fd = os.open(name, os.O_RDONLY, dir_fd=parent_fd)
            try:
                foreign_fd = os.open(
                    "foreign.txt",
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                    dir_fd=replacement_fd,
                )
                os.close(foreign_fd)
            finally:
                os.close(replacement_fd)
        return original_open(parent_fd, name)

    monkeypatch.setattr(transaction, "_open_directory", replace_before_open)
    try:
        with pytest.raises(TransactionInterrupted):
            discard_staging(handle, platform)
        assert (tmp_path / handle.staging_name / "foreign.txt").is_file()
        assert (tmp_path / f"{handle.staging_name}.preserved").is_dir()
        assert (tmp_path / lease.journal_name).is_file()
    finally:
        close_output_lease(lease)


def test_publication_revalidates_ancestry_immediately_before_rename(
    tmp_path, monkeypatch
):
    parent = tmp_path / "private"
    parent.mkdir(mode=0o700)
    target, platform, lease, _publication, verified = lease_for(parent)
    candidate = bundle()
    handle, _ = stage_bundle(verified, candidate, platform, publication="success")
    replace_staged_metadata(handle, candidate, platform)
    original_verify = transaction._verify_target_identity
    attacked = False

    def verify_then_replace_ancestor(active_lease, name, expected):
        nonlocal attacked
        original_verify(active_lease, name, expected)
        if not attacked:
            attacked = True
            parent.rename(tmp_path / "held-private")
            parent.mkdir(mode=0o700)

    monkeypatch.setattr(
        transaction, "_verify_target_identity", verify_then_replace_ancestor
    )
    try:
        with pytest.raises(TransactionInterrupted, match="ancestry was replaced"):
            publish_success(handle, platform)
        assert attacked is True
        assert not target.exists()
        assert not (tmp_path / "held-private" / "result").exists()
    finally:
        close_output_lease(lease)


class TestTheRewriteVerifiesBeforeItTruncates:
    """A7.8. ``_replace_file`` opened with ``O_TRUNC`` and checked the inode
    afterwards, so a name swapped between its ``lstat`` and its ``open`` was
    emptied and *then* refused. The refusal was correct and arrived after the
    damage, which makes it a report rather than a guard.

    The order is now open, verify, ``ftruncate``.

    The race lives INSIDE the function -- between two syscalls it makes
    itself -- so it cannot be lost from outside: a swap arranged by a test is
    simply what ``_replace_file`` stats. What can be tested is the property
    the reordering buys, which is the one that matters: **on the refusal
    path, the bytes are still there.** ``_same_identity`` is forced to
    disagree, which is what a real swap would produce at exactly that point.
    """

    def test_a_refused_rewrite_leaves_the_file_untouched(self, tmp_path, monkeypatch):
        import os

        from _rheplicant_bootstrap.errors import ConfigError
        from _rheplicant_bootstrap.output import transaction

        original = b"the bytes that must survive a refusal"
        target = tmp_path / "metadata.json"
        target.write_bytes(original)

        monkeypatch.setattr(transaction, "_same_identity", lambda left, right: False)

        parent_fd = os.open(tmp_path, os.O_RDONLY)
        try:
            with pytest.raises(ConfigError, match="changed before rewrite"):
                transaction._replace_file(parent_fd, "metadata.json", b"new payload")
        finally:
            os.close(parent_fd)

        assert target.read_bytes() == original

    def test_an_accepted_rewrite_still_replaces_the_whole_file(self, tmp_path):
        """ANTI-VACUITY, and the half a reordering can break: ``ftruncate``
        has to actually run on the accepted path, or a payload shorter than
        what it replaces would leave the tail of the old file behind."""
        import os

        from _rheplicant_bootstrap.output.transaction import _replace_file

        target = tmp_path / "metadata.json"
        target.write_bytes(b"x" * 200)
        parent_fd = os.open(tmp_path, os.O_RDONLY)
        try:
            _replace_file(parent_fd, "metadata.json", b"short")
        finally:
            os.close(parent_fd)
        assert target.read_bytes() == b"short"
