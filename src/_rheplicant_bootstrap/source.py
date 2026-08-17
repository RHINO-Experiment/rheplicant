"""Exact-byte, bounded, TOCTOU-checked source records for the CLI boundary."""

from __future__ import annotations

import os
import stat
from typing import BinaryIO

from _rheplicant_bootstrap.errors import ConfigError
from _rheplicant_bootstrap.types import SourceInput
from _rheplicant_bootstrap.yaml import YamlLimits


def _same_file_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return left.st_dev == right.st_dev and left.st_ino == right.st_ino


def _same_snapshot(left: os.stat_result, right: os.stat_result) -> bool:
    return _same_file_identity(left, right) and (
        left.st_mode,
        left.st_size,
        left.st_mtime_ns,
        left.st_ctime_ns,
    ) == (
        right.st_mode,
        right.st_size,
        right.st_mtime_ns,
        right.st_ctime_ns,
    )


def _canonical_dir(path: str) -> str:
    return os.path.realpath(os.path.abspath(path))


def _input_limit() -> int:
    return YamlLimits().input_bytes


def _byte_limit_error(source_name: str, observed: int, limit: int) -> ConfigError:
    return ConfigError(f"{source_name}: YAML byte count {observed} exceeds limit {limit}.")


def _read_path_once(source_path: str) -> tuple[bytes, str]:
    """Read one regular file descriptor once without trusting its advertised size."""
    limit = _input_limit()
    fd = -1
    try:
        before_link = os.lstat(source_path)
        flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_CLOEXEC", 0)
        fd = os.open(source_path, flags)
        before_target = os.fstat(fd)
        if not stat.S_ISREG(before_target.st_mode):
            raise ConfigError(f"{source_path}: source must be a regular file.")
        if before_target.st_size > limit:
            raise _byte_limit_error(source_path, before_target.st_size, limit)
        stream = os.fdopen(fd, "rb")
        fd = -1
        with stream:
            data = stream.read(limit + 1)
            after_target = os.fstat(stream.fileno())
        after_link = os.lstat(source_path)
        source_realpath = os.path.realpath(source_path)
        final_target = os.stat(source_realpath)
    except ConfigError:
        raise
    except (OSError, ValueError) as exc:
        raise ConfigError(f"{source_path}: cannot read source: {exc}") from exc
    finally:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass

    if not _same_snapshot(before_link, after_link):
        raise ConfigError(f"{source_path}: source link changed while reading.")
    if not _same_snapshot(before_target, after_target) or not _same_snapshot(
        after_target, final_target
    ):
        raise ConfigError(f"{source_path}: source target changed while reading.")
    if not isinstance(data, bytes):
        raise ConfigError(f"{source_path}: binary source did not produce bytes.")
    if len(data) > limit:
        raise _byte_limit_error(source_path, len(data), limit)
    return data, source_realpath


def _stdin_base_dir(base_dir: str | None) -> str:
    try:
        return _canonical_dir(base_dir if base_dir is not None else os.getcwd())
    except (OSError, ValueError) as exc:
        raise ConfigError(f"<stdin>: invalid base_dir: {exc}") from exc


def _read_stdin_once(stdin: BinaryIO | None, *, limit: int) -> bytes:
    chosen_stdin = stdin
    if chosen_stdin is None:
        import sys

        chosen_stdin = sys.stdin.buffer
    try:
        data = chosen_stdin.read(limit + 1)
    except (OSError, ValueError, TypeError) as exc:
        raise ConfigError(f"<stdin>: cannot read source: {exc}") from exc
    if not isinstance(data, bytes):
        raise ConfigError("<stdin>: binary source did not produce bytes.")
    if len(data) > limit:
        raise _byte_limit_error("<stdin>", len(data), limit)
    return data


def read_cli_source_once(
    path_or_dash: str,
    *,
    base_dir: str | None,
    stdin: BinaryIO | None,
) -> SourceInput:
    """Read one CLI source once, preserving its lexical and target identities."""
    if path_or_dash == "-":
        limit = _input_limit()
        chosen_base_dir = _stdin_base_dir(base_dir)
        data = _read_stdin_once(stdin, limit=limit)
        return SourceInput(
            input_bytes=data,
            source_path="<stdin>",
            source_realpath=None,
            source_name="<stdin>",
            base_dir=chosen_base_dir,
            launch_mode="cli",
        )

    try:
        source_path = os.path.abspath(path_or_dash)
        lexical_parent = _canonical_dir(os.path.dirname(source_path))
        if base_dir is not None and _canonical_dir(base_dir) != lexical_parent:
            raise ConfigError(
                f"{source_path}: base_dir {base_dir!r} contradicts lexical parent "
                f"{lexical_parent!r}."
            )
    except ConfigError:
        raise
    except (OSError, ValueError) as exc:
        raise ConfigError(f"{path_or_dash!r}: invalid source or base_dir: {exc}") from exc
    data, source_realpath = _read_path_once(source_path)
    return SourceInput(
        input_bytes=data,
        source_path=source_path,
        source_realpath=source_realpath,
        source_name=source_path,
        base_dir=lexical_parent,
        launch_mode="cli",
    )


def read_source(
    path_or_dash: str,
    *,
    base_dir: str | None,
    stdin: BinaryIO | None,
) -> SourceInput:
    """Public Task 2 source-entry constructor."""
    return read_cli_source_once(path_or_dash, base_dir=base_dir, stdin=stdin)


__all__ = ["read_cli_source_once", "read_source"]
