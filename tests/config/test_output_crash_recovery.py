from __future__ import annotations

import subprocess
import sys
from collections import Counter

import pytest

import _rheplicant_bootstrap.output.transaction as transaction
from _rheplicant_bootstrap.errors import ConfigError
from _rheplicant_bootstrap.output import (
    acquire_output_lease,
    close_output_lease,
    inspect_output_path,
)
from _rheplicant_bootstrap.output.transaction import (
    TransactionInterrupted,
    capture_persistence_events,
    observe_persistence,
    publish_success,
    recover_transaction,
    replace_staged_metadata,
    stage_bundle,
)
from tests.config.test_output_paths import SafePlatform
from tests.config.test_output_preflight import run_request
from tests.config.test_output_transaction import bundle, lease_for


def _complete_fresh(target, platform, verified, candidate):
    handle, _ = stage_bundle(verified, candidate, platform, publication="success")
    replace_staged_metadata(handle, candidate, platform)
    publish_success(handle, platform)
    assert target.is_dir()


def test_every_fresh_persistence_ordinal_is_observable_and_restart_safe(
    tmp_path, monkeypatch
):
    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    target, platform, lease, _publication, verified = lease_for(baseline_dir)
    candidate = bundle()
    fsync_calls = 0
    original_fsync = transaction.os.fsync

    def fsync(fd):
        nonlocal fsync_calls
        fsync_calls += 1
        original_fsync(fd)

    monkeypatch.setattr(transaction.os, "fsync", fsync)
    try:
        with capture_persistence_events() as events:
            _complete_fresh(target, platform, verified, candidate)
        ordinals = tuple(event.ordinal for event in events)
        assert ordinals == tuple(range(len(events)))
        assert Counter(event.label for event in events) == Counter(
            {
                "mkdir": 1,
                "open_create": 9,
                "write": 11,
                "fchmod": 12,
                "file_fsync": 11,
                "directory_fsync": 25,
                "rename_noreplace": 2,
                "rename_owned": 4,
                "unlink": 1,
            }
        )
        assert fsync_calls == 36
    finally:
        close_output_lease(lease)

    # Every completed persistence syscall is an independently killable boundary.
    # Ambiguity is allowed only when all paths are kept.
    for ordinal in range(len(events)):
        case = tmp_path / f"case-{ordinal}"
        case.mkdir()
        target, platform, lease, _publication, verified = lease_for(case)
        seen = 0

        def fail(event, selected_ordinal=ordinal):
            nonlocal seen
            seen += 1
            if event.ordinal == selected_ordinal:
                raise OSError(f"injected after ordinal {selected_ordinal}")

        try:
            with observe_persistence(fail):
                with pytest.raises(TransactionInterrupted):
                    _complete_fresh(target, platform, verified, candidate)
            try:
                recover_transaction(lease, platform)
            except ConfigError as error:
                assert "no path was removed" in str(error)
            assert seen > ordinal
            assert not target.exists() or (
                (target / "provenance.json").read_bytes() == candidate.provenance
                and (target / "diagnostics.json").read_bytes() == candidate.diagnostics
            )
        finally:
            close_output_lease(lease)


def test_interruption_reports_every_completed_materialization(tmp_path):
    _target, platform, lease, _publication, verified = lease_for(tmp_path)
    writes = 0

    def fail_after_input_write(event):
        nonlocal writes
        if event.label == "write":
            writes += 1
            if writes == 3:
                raise OSError("after complete input write")

    try:
        with observe_persistence(fail_after_input_write):
            with pytest.raises(TransactionInterrupted) as caught:
                stage_bundle(verified, bundle(), platform, publication="success")
        slots = [row.slot for row in caught.value.state.unreported_materializations]
        assert slots == ["lock", "journal", "input"]
    finally:
        close_output_lease(lease)


@pytest.mark.parametrize(
    ("scenario", "event_count"),
    (("fresh", 76), ("clobber", 97), ("refused", 76), ("error", 76)),
)
def test_sigkill_after_every_persistence_ordinal_is_restart_safe(
    tmp_path, scenario, event_count
):
    script = """
import os
import sys
from pathlib import Path
from _rheplicant_bootstrap.output.transaction import (
    observe_persistence, publish_success, replace_staged_metadata, stage_bundle,
)
from tests.config.test_output_transaction import bundle, lease_for

case = Path(sys.argv[1])
ordinal = int(sys.argv[2])
scenario = sys.argv[3]
target, platform, lease, publication, verified = lease_for(
    case, existing=scenario == "clobber"
)
status = scenario if scenario in ("refused", "error") else "ok"
candidate = bundle(status)
def kill(event):
    if event.ordinal == ordinal:
        os._exit(97)
with observe_persistence(kill):
    kind = "success" if scenario in ("fresh", "clobber") else scenario
    authorization = verified if kind == "success" else publication
    handle, _ = stage_bundle(authorization, candidate, platform, publication=kind)
    replace_staged_metadata(handle, candidate, platform)
    if kind == "success":
        publish_success(handle, platform)
    else:
        from _rheplicant_bootstrap.output.transaction import publish_failure
        publish_failure(handle, platform)
os._exit(0)
"""
    candidate = bundle(scenario if scenario in ("refused", "error") else "ok")
    for ordinal in range(event_count):
        case = tmp_path / f"{scenario}-{ordinal}"
        case.mkdir()
        result = subprocess.run(
            [sys.executable, "-c", script, str(case), str(ordinal), scenario],
            check=False,
        )
        assert result.returncode == 97
        target = case / "result"
        platform = SafePlatform()
        lease = acquire_output_lease(
            inspect_output_path(
                run_request(target, clobber=target.exists()),
                platform,
            ),
            platform,
        )
        try:
            try:
                recover_transaction(lease, platform)
            except ConfigError as error:
                assert "no path was removed" in str(error)
            if target.exists() and (target / "provenance.json").exists():
                assert (target / "provenance.json").read_bytes() == candidate.provenance
                assert (target / "diagnostics.json").read_bytes() == candidate.diagnostics
            elif target.exists():
                assert scenario == "clobber"
                assert (target / ".rheplicant-results.json").is_file()
            failures = tuple(case.glob(f"result.{scenario}-*"))
            assert len(failures) <= 1
            if failures:
                assert (failures[0] / "provenance.json").read_bytes() == candidate.provenance
                assert (failures[0] / "diagnostics.json").read_bytes() == candidate.diagnostics
        finally:
            close_output_lease(lease)


def test_two_writers_share_one_persistent_lock_and_one_wins(tmp_path):
    script = """
import sys
from pathlib import Path
from _rheplicant_bootstrap.errors import ConfigError
from _rheplicant_bootstrap.output import (
    acquire_output_lease, close_output_lease, inspect_output_path,
    verify_a34_under_lease, verify_publication_under_lease,
)
from _rheplicant_bootstrap.output.transaction import (
    publish_success, recover_transaction, replace_staged_metadata, stage_bundle,
)
from tests.config.test_output_paths import SafePlatform
from tests.config.test_output_preflight import run_request
from tests.config.test_output_transaction import bundle

target = Path(sys.argv[1])
platform = SafePlatform()
inspection = inspect_output_path(run_request(target), platform)
lease = acquire_output_lease(inspection, platform)
try:
    recover_transaction(lease, platform)
    publication = verify_publication_under_lease(lease, platform)
    try:
        verified = verify_a34_under_lease(publication, platform)
    except ConfigError:
        sys.exit(2)
    candidate = bundle()
    handle, _ = stage_bundle(verified, candidate, platform, publication="success")
    replace_staged_metadata(handle, candidate, platform)
    publish_success(handle, platform)
finally:
    close_output_lease(lease)
"""
    target = tmp_path / "result"
    writers = [
        subprocess.Popen([sys.executable, "-c", script, str(target)])
        for _index in range(2)
    ]
    codes = sorted(writer.wait(timeout=30) for writer in writers)
    assert codes == [0, 2]
    candidate = bundle()
    assert (target / "provenance.json").read_bytes() == candidate.provenance
    assert (target / "diagnostics.json").read_bytes() == candidate.diagnostics
    assert not any(path.name.startswith(".rheplicant-journal-") for path in tmp_path.iterdir())
