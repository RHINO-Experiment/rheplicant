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


def _lexical_lstat(path: str) -> os.stat_result:
    """Return lexical path metadata without following its final symlink."""
    return os.lstat(path)


def _input_limit() -> int:
    return YamlLimits().input_bytes


def _byte_limit_error(source_name: str, observed: int, limit: int) -> ConfigError:
    return ConfigError(f"{source_name}: YAML byte count {observed} exceeds limit {limit}.")


def _read_bounded_forward(stream: BinaryIO, *, source_name: str, limit: int) -> bytes:
    """Accumulate legal short reads without traversing the source a second time."""
    maximum = limit + 1
    data = bytearray()
    try:
        while len(data) < maximum:
            chunk = stream.read(maximum - len(data))
            if not isinstance(chunk, bytes):
                raise ConfigError(f"{source_name}: binary source did not produce bytes.")
            if not chunk:
                break
            data.extend(chunk)
    except ConfigError:
        raise
    except (OSError, ValueError, TypeError) as exc:
        raise ConfigError(f"{source_name}: cannot read source: {exc}") from exc
    if len(data) > limit:
        raise _byte_limit_error(source_name, len(data), limit)
    return bytes(data)


def _read_stable_regular_file(
    path: str | os.PathLike[str],
    *,
    maximum: int,
    source_name: str | None,
) -> tuple[bytes, str]:
    """Return exact bytes and resolved identity from one stable descriptor."""
    fd = -1
    display_name = source_name if source_name is not None else "<source>"
    try:
        source_path = os.fspath(path)
        if not isinstance(source_path, str):
            raise TypeError("source path must be text")
        if source_name is None:
            display_name = source_path
        if maximum < 0:
            raise ValueError("maximum must be non-negative")
        before_link = _lexical_lstat(source_path)
        before_realpath = os.path.realpath(source_path)
        after_initial_resolution_link = _lexical_lstat(source_path)
        flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_CLOEXEC", 0)
        if not stat.S_ISLNK(before_link.st_mode):
            flags |= getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(source_path, flags)
        before_target = os.fstat(fd)
        if not stat.S_ISREG(before_target.st_mode):
            raise ConfigError(f"{display_name}: source must be a regular file.")
        if before_target.st_size > maximum:
            raise _byte_limit_error(display_name, before_target.st_size, maximum)
        stream = os.fdopen(fd, "rb")
        fd = -1
        with stream:
            data = _read_bounded_forward(
                stream, source_name=display_name, limit=maximum
            )
            after_target = os.fstat(stream.fileno())
        after_read_link = _lexical_lstat(source_path)
        source_realpath = os.path.realpath(source_path)
        final_target = os.stat(source_realpath)
        final_link = _lexical_lstat(source_path)
    except ConfigError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise ConfigError(f"{display_name}: cannot read source: {exc}") from exc
    finally:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass

    if not (
        _same_snapshot(before_link, after_initial_resolution_link)
        and _same_snapshot(after_initial_resolution_link, after_read_link)
        and _same_snapshot(after_read_link, final_link)
    ):
        raise ConfigError(f"{display_name}: source link changed while reading.")
    if before_realpath != source_realpath:
        raise ConfigError(f"{display_name}: source link changed while reading.")
    if not _same_snapshot(before_target, after_target) or not _same_snapshot(
        after_target, final_target
    ):
        raise ConfigError(f"{display_name}: source target changed while reading.")
    return data, source_realpath


def read_stable_regular_bytes(
    path: str | os.PathLike[str],
    *,
    maximum: int,
    source_name: str | None = None,
) -> bytes:
    """Read one unchanged regular file in one bounded forward consumption."""
    data, _ = _read_stable_regular_file(
        path, maximum=maximum, source_name=source_name
    )
    return data


def _read_path_once(source_path: str) -> tuple[bytes, str]:
    """Read a CLI path with the shared stable-file primitive."""
    return _read_stable_regular_file(
        source_path, maximum=_input_limit(), source_name=source_path
    )


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
    return _read_bounded_forward(chosen_stdin, source_name="<stdin>", limit=limit)


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


__all__ = ["read_cli_source_once", "read_source", "read_stable_regular_bytes"]
