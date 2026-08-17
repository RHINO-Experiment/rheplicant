"""Exact-byte, TOCTOU-checked source records for the CLI boundary."""

from __future__ import annotations

import os
import stat
from typing import BinaryIO

from _rheplicant_bootstrap.errors import ConfigError
from _rheplicant_bootstrap.types import SourceInput


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


def _read_path_once(source_path: str) -> tuple[bytes, str]:
    try:
        before_link = os.lstat(source_path)
        with open(source_path, "rb") as stream:
            before_target = os.fstat(stream.fileno())
            if not stat.S_ISREG(before_target.st_mode):
                raise ConfigError(f"{source_path}: source must be a regular file.")
            data = stream.read()
            after_target = os.fstat(stream.fileno())
        after_link = os.lstat(source_path)
        source_realpath = os.path.realpath(source_path)
        final_target = os.stat(source_realpath)
    except ConfigError:
        raise
    except OSError as exc:
        raise ConfigError(f"{source_path}: cannot read source: {exc}") from exc

    if not _same_snapshot(before_link, after_link):
        raise ConfigError(f"{source_path}: source link changed while reading.")
    if not _same_snapshot(before_target, after_target) or not _same_snapshot(
        after_target, final_target
    ):
        raise ConfigError(f"{source_path}: source target changed while reading.")
    if not isinstance(data, bytes):
        raise ConfigError(f"{source_path}: binary source did not produce bytes.")
    return data, source_realpath


def read_cli_source_once(
    path_or_dash: str,
    *,
    base_dir: str | None,
    stdin: BinaryIO | None,
) -> SourceInput:
    """Read one CLI source once, preserving its lexical and target identities."""
    if path_or_dash == "-":
        chosen_stdin = stdin
        if chosen_stdin is None:
            import sys

            chosen_stdin = sys.stdin.buffer
        try:
            data = chosen_stdin.read()
        except OSError as exc:
            raise ConfigError(f"<stdin>: cannot read source: {exc}") from exc
        if not isinstance(data, bytes):
            raise ConfigError("<stdin>: binary source did not produce bytes.")
        return SourceInput(
            input_bytes=data,
            source_path="<stdin>",
            source_realpath=None,
            source_name="<stdin>",
            base_dir=_canonical_dir(base_dir or os.getcwd()),
            launch_mode="cli",
        )

    source_path = os.path.abspath(path_or_dash)
    lexical_parent = _canonical_dir(os.path.dirname(source_path))
    if base_dir is not None and _canonical_dir(base_dir) != lexical_parent:
        raise ConfigError(
            f"{source_path}: base_dir {base_dir!r} contradicts lexical parent {lexical_parent!r}."
        )
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
