from __future__ import annotations

import os
import sys

import pytest

from _rheplicant_bootstrap.errors import ConfigError
from _rheplicant_bootstrap.output import (
    acquire_output_lease,
    close_output_lease,
    inspect_output_path,
    platform_adapter,
    verify_publication_under_lease,
)
from _rheplicant_bootstrap.output.platform_darwin import DarwinOutputPlatform
from _rheplicant_bootstrap.output.platform_linux import LinuxOutputPlatform
from _rheplicant_bootstrap.output.types import AccessInspection
from tests.config.test_output_paths import SafePlatform, request_for


def test_platform_factory_has_a_closed_host_set(monkeypatch):
    adapter = platform_adapter()
    if sys.platform == "darwin":
        assert type(adapter) is DarwinOutputPlatform
    else:
        assert type(adapter) is LinuxOutputPlatform
    monkeypatch.setattr(sys, "platform", "plan9")
    with pytest.raises(ConfigError, match="unsupported"):
        platform_adapter()


def test_unreliable_acl_and_noreplace_are_refusals(tmp_path):
    class Unreliable(SafePlatform):
        def inspect_access(self, directory_fd, parent_path):
            row = super().inspect_access(directory_fd, parent_path)
            return AccessInspection(
                row.parent_path,
                row.effective_uid,
                row.owner_uid,
                row.mode,
                False,
                False,
                True,
                False,
                "cannot verify access control",
            )

    target = tmp_path / "result"
    platform = Unreliable()
    lease = acquire_output_lease(
        inspect_output_path(request_for(target, command="run"), platform), platform
    )
    try:
        with pytest.raises(ConfigError, match="cannot verify access control"):
            verify_publication_under_lease(lease, platform)
    finally:
        close_output_lease(lease)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux adapter integration")
def test_linux_adapter_reads_access_metadata_from_fd(tmp_path):
    adapter = LinuxOutputPlatform()
    fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        inspection = adapter.inspect_access(fd, "diagnostic-only")
    finally:
        os.close(fd)
    assert inspection.owner_uid == os.geteuid()
    assert inspection.parent_path == "diagnostic-only"


@pytest.mark.skipif(sys.platform != "darwin", reason="Darwin adapter integration")
def test_darwin_adapter_reads_access_metadata_from_fd(tmp_path):
    adapter = DarwinOutputPlatform()
    fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        inspection = adapter.inspect_access(fd, "diagnostic-only")
    finally:
        os.close(fd)
    assert inspection.owner_uid == os.geteuid()
    assert inspection.parent_path == "diagnostic-only"
    assert inspection.reliable is True
    assert inspection.access_acl_is_trivial is True


# --- extended ACLs: what they GRANT, not whether they exist -----------------
#
# The distinction these tests pin cost a real blocker. macOS puts
# `group:everyone deny delete` on every home directory, and reading "this
# inode has an ACL" as "this inode is unverifiable" refused every project
# under `~` while accepting `/tmp`.

def _add_ace(path, ace: str) -> bool:
    """Add one ACE, or report that this host cannot."""
    import subprocess
    try:
        subprocess.run(["chmod", "+a", ace, str(path)], check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError):
        return False
    return True


def _inspect(path):
    adapter = DarwinOutputPlatform()
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        return adapter.inspect_access(fd, str(path))
    finally:
        os.close(fd)


darwin_only = pytest.mark.skipif(sys.platform != "darwin", reason="extended ACLs are Darwin's")


@darwin_only
def test_a_directory_with_no_acl_is_trivial_and_grants_nothing(tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()
    row = _inspect(plain)
    assert (row.access_acl_is_trivial, row.access_acl_grants_others) == (True, False)
    assert row.reliable is True


@darwin_only
def test_a_deny_only_acl_is_not_trivial_and_still_grants_nothing(tmp_path):
    # The macOS home-directory case. A deny ACE cannot hand anyone a right, so
    # it cannot make an entry renameable by a non-owner -- it only ADDS
    # protection. Refusing it refused every project under `~`.
    denied = tmp_path / "denied"
    denied.mkdir()
    if not _add_ace(denied, "everyone deny delete"):
        pytest.skip("this host will not set an ACE")
    row = _inspect(denied)
    assert row.access_acl_is_trivial is False
    assert row.access_acl_grants_others is False
    assert row.reliable is True
    assert row.reason == "deny-only access ACL"


@darwin_only
def test_an_allow_ace_still_counts_as_granting(tmp_path):
    # The guard this fix must NOT weaken: an ALLOW ACE can hand a right to
    # someone the mode bits never did.
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    if not _add_ace(allowed, "everyone allow write,delete"):
        pytest.skip("this host will not set an ACE")
    row = _inspect(allowed)
    assert row.access_acl_grants_others is True
    assert row.reliable is True


@darwin_only
def test_one_allow_among_denies_is_enough_to_refuse(tmp_path):
    mixed = tmp_path / "mixed"
    mixed.mkdir()
    if not (_add_ace(mixed, "everyone deny delete") and _add_ace(mixed, "everyone allow write")):
        pytest.skip("this host will not set an ACE")
    assert _inspect(mixed).access_acl_grants_others is True


@darwin_only
def test_an_unreadable_acl_fails_closed(tmp_path, monkeypatch):
    # A host whose ACL symbols are missing must report BOTH "grants" and
    # "unreliable", so a caller reading either one alone still refuses.
    plain = tmp_path / "plain"
    plain.mkdir()
    adapter = DarwinOutputPlatform()
    monkeypatch.setattr(adapter, "_acl_get_tag_type", None)
    fd = os.open(plain, os.O_RDONLY | os.O_DIRECTORY)
    try:
        row = adapter.inspect_access(fd, str(plain))
    finally:
        os.close(fd)
    assert row.access_acl_grants_others is True
    assert row.reliable is False


@darwin_only
def test_a_deny_only_ancestor_no_longer_blocks_publication(tmp_path):
    # The end of the chain: `inspect_ancestor_entry` is what refused, and this
    # is the shape of `/Users/<someone>`.
    parent = tmp_path / "home"
    parent.mkdir()
    child = parent / "project"
    child.mkdir()
    if not _add_ace(parent, "everyone deny delete"):
        pytest.skip("this host will not set an ACE")
    adapter = DarwinOutputPlatform()
    fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        row = adapter.inspect_ancestor_entry(fd, str(parent), "project", os.stat(child))
    finally:
        os.close(fd)
    assert row.rename_protected is True
    assert row.reliable is True
