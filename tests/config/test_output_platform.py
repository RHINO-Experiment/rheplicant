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
