"""Platform contract for ACL decisions and atomic no-replace rename."""

from __future__ import annotations

import os
import sys
from typing import Protocol

from _rheplicant_bootstrap.errors import ConfigError

from .types import AccessInspection, AncestorEntryInspection


class OutputPlatform(Protocol):
    def inspect_access(self, directory_fd: int, parent_path: str) -> AccessInspection:
        raise NotImplementedError

    def inspect_ancestor_entry(
        self,
        containing_fd: int,
        containing_path: str,
        child_name: str,
        child_stat: os.stat_result,
    ) -> AncestorEntryInspection:
        """Prove this lexical entry cannot be renamed by a non-owner."""
        raise NotImplementedError

    def verify_rename_noreplace_available(self, directory_fd: int) -> None:
        """Fail closed without creating, linking, or renaming a probe node."""
        raise NotImplementedError

    def rename_noreplace(
        self,
        old_dir_fd: int,
        old_name: str,
        new_dir_fd: int,
        new_name: str,
    ) -> None:
        raise NotImplementedError


def platform_adapter() -> OutputPlatform:
    if sys.platform == "darwin":
        from .platform_darwin import DarwinOutputPlatform

        return DarwinOutputPlatform()
    if sys.platform.startswith("linux"):
        from .platform_linux import LinuxOutputPlatform

        return LinuxOutputPlatform()
    raise ConfigError(f"output publication is unsupported on {sys.platform!r}.")


__all__ = ["OutputPlatform", "platform_adapter"]
