"""Filesystem primitives the rest of this package shares.

Opening a child directory, comparing two identities, requiring a directory
entry, bounding a path's component count. Small, and here rather than beside
any one caller because both the descriptor preflight and the A34 lease need
them -- putting them with either made those two import each other.
"""

from __future__ import annotations

import os
import stat

from _rheplicant_bootstrap.errors import ConfigError

from .types import (
    AncestorEntryInspection,
    TargetIdentity,
)

_OPEN_DIRECTORY = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
)


def _same_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return left.st_dev == right.st_dev and left.st_ino == right.st_ino


def _require_entry(row: AncestorEntryInspection) -> None:
    if not row.reliable:
        raise ConfigError(row.reason or "cannot verify ancestor access control.")
    if not row.rename_protected:
        raise ConfigError(row.reason or "ancestor entry is not rename-protected.")


def _open_child(parent_fd: int, name: str, before: os.stat_result) -> int:
    try:
        child_fd = os.open(name, _OPEN_DIRECTORY, dir_fd=parent_fd)
        after = os.fstat(child_fd)
    except OSError:
        raise ConfigError(f"output ancestor {name!r} cannot be opened safely.") from None
    if not _same_identity(before, after):
        os.close(child_fd)
        raise ConfigError(f"output ancestor {name!r} changed during descriptor walk.")
    return child_fd


def _component_limit(fd: int) -> int:
    try:
        value = os.fpathconf(fd, "PC_NAME_MAX")
    except (OSError, ValueError):
        raise ConfigError("cannot obtain output filesystem NAME_MAX from directory fd.") from None
    if type(value) is not int or value <= 0:
        raise ConfigError("output filesystem reported an invalid NAME_MAX.")
    return value


def _target_identity(parent_fd: int, target_name: str) -> TargetIdentity:
    try:
        row = os.lstat(target_name, dir_fd=parent_fd)
    except FileNotFoundError:
        return TargetIdentity(False, None, None, None)
    except OSError:
        raise ConfigError("cannot inspect output target.") from None
    if stat.S_ISLNK(row.st_mode):
        raise ConfigError("output target is a symlink.")
    if not stat.S_ISDIR(row.st_mode):
        raise ConfigError("output target exists and is not a directory.")
    return TargetIdentity(True, row.st_dev, row.st_ino, None)
