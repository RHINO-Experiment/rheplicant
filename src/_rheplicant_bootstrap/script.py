"""Deterministic embedded-script rendering and atomic no-clobber publication."""

from __future__ import annotations

import base64
import os
import stat
import uuid
from collections.abc import Sequence

from _rheplicant_bootstrap.errors import ConfigError
from _rheplicant_bootstrap.prepare import SelectedPreset
from _rheplicant_bootstrap.presets import PresetSnapshot
from _rheplicant_bootstrap.types import SourceInput

from .output.platform import OutputPlatform, platform_adapter

_OPEN_DIRECTORY = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
)


def _literal(value: object) -> str:
    if value is None or type(value) is str:
        return repr(value)
    if type(value) is int:
        return str(value)
    raise ConfigError("generated script contains an unsupported literal.")


def _snapshot(value: object) -> PresetSnapshot:
    if type(value) is SelectedPreset:
        return value.snapshot
    if type(value) is PresetSnapshot:
        return value
    raise ConfigError("rendered presets must be selected or snapshot records.")


def render_script(
    source: SourceInput,
    presets: Sequence[SelectedPreset | PresetSnapshot],
) -> bytes:
    """Render exact source/preset bytes into one deterministic Python program."""
    if type(source) is not SourceInput or type(presets) not in (tuple, list):
        raise ConfigError("script rendering requires exact source and preset records.")
    rows = tuple(_snapshot(row) for row in presets)
    if len({row.name for row in rows}) != len(rows):
        raise ConfigError("rendered preset snapshots contain duplicate names.")
    input_b64 = base64.b64encode(source.input_bytes).decode("ascii")
    rendered_presets = []
    for row in rows:
        rendered_presets.append(
            "        {"
            f"'expanded_nodes': {row.expanded_nodes}, "
            "'input_bytes': base64.b64decode("
            f"{_literal(base64.b64encode(row.input_bytes).decode('ascii'))}), "
            f"'name': {_literal(row.name)}, "
            f"'resource': {_literal(row.resource)}, "
            f"'sha256': {_literal(row.sha256)}"
            "},"
        )
    preset_block = "\n".join(rendered_presets)
    text = (
        "import _rheplicant_bootstrap\n"
        "import base64\n\n"
        "raise SystemExit(\n"
        "    _rheplicant_bootstrap.run_embedded_config(\n"
        f"        input_bytes=base64.b64decode({_literal(input_b64)}),\n"
        f"        source_path={_literal(source.source_path)},\n"
        f"        source_realpath={_literal(source.source_realpath)},\n"
        f"        source_name={_literal(source.source_name)},\n"
        f"        base_dir={_literal(source.base_dir)},\n"
        "        presets=(\n"
        f"{preset_block}\n"
        "        ),\n"
        "    )\n"
        ")\n"
    )
    return text.encode("utf-8")


def _walk_parent(path: str, platform: OutputPlatform) -> tuple[int, str, str]:
    try:
        absolute = os.path.abspath(os.path.expandvars(os.path.expanduser(path)))
    except Exception:
        raise ConfigError("script output path cannot be normalized.") from None
    leaf = os.path.basename(absolute)
    parent = os.path.dirname(absolute)
    if not leaf or leaf in (".", "..") or "\0" in leaf:
        raise ConfigError("script output must end in a valid file name.")
    components = tuple(component for component in parent.split(os.sep) if component)
    current_fd = os.open(os.sep, _OPEN_DIRECTORY)
    current_path = os.sep
    try:
        for component in components:
            before = os.lstat(component, dir_fd=current_fd)
            if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
                raise ConfigError("script output parent contains a symlink or non-directory.")
            entry = platform.inspect_ancestor_entry(
                current_fd, current_path, component, before
            )
            if not entry.reliable or not entry.rename_protected:
                raise ConfigError(
                    entry.reason or "script output ancestor is not rename-protected."
                )
            child_fd = os.open(component, _OPEN_DIRECTORY, dir_fd=current_fd)
            after = os.fstat(child_fd)
            if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
                os.close(child_fd)
                raise ConfigError("script output ancestor changed during traversal.")
            os.close(current_fd)
            current_fd = child_fd
            current_path = os.path.join(current_path, component)
        access = platform.inspect_access(current_fd, parent)
        if (
            not access.reliable
            or access.owner_uid != access.effective_uid
            or access.mode & 0o022
            or not access.access_acl_is_trivial
            or not access.default_acl_is_trivial
        ):
            raise ConfigError(
                access.reason or "script output parent has insecure access control."
            )
        return current_fd, leaf, absolute
    except BaseException:
        os.close(current_fd)
        raise


def publish_script(
    payload: bytes,
    output: str | os.PathLike[str],
    *,
    platform: OutputPlatform | None = None,
) -> str:
    """Fsync and atomically link a mode-0600 script to an absent leaf."""
    if type(payload) is not bytes:
        raise ConfigError("generated script payload must be exact bytes.")
    chosen = platform_adapter() if platform is None else platform
    parent_fd, leaf, absolute = _walk_parent(os.fspath(output), chosen)
    temporary = f".{leaf}.rheplicant-{uuid.uuid4().hex}.tmp"
    temporary_created = False
    try:
        limit = os.fpathconf(parent_fd, "PC_NAME_MAX")
        if type(limit) is not int or max(
            len(os.fsencode(leaf)), len(os.fsencode(temporary))
        ) > limit:
            raise ConfigError("script output component exceeds filesystem NAME_MAX.")
        fd = os.open(
            temporary,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=parent_fd,
        )
        temporary_created = True
        try:
            offset = 0
            while offset < len(payload):
                written = os.write(fd, payload[offset:])
                if written <= 0:
                    raise OSError("script write made no progress")
                offset += written
            os.fchmod(fd, 0o600)
            os.fsync(fd)
        finally:
            os.close(fd)
        try:
            os.link(
                temporary,
                leaf,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileExistsError:
            raise ConfigError(f"script output {absolute!r} already exists.") from None
        os.fsync(parent_fd)
        os.unlink(temporary, dir_fd=parent_fd)
        temporary_created = False
        os.fsync(parent_fd)
        return absolute
    except ConfigError:
        raise
    except OSError as error:
        raise ConfigError(f"cannot publish generated script {absolute!r}: {error}.") from None
    finally:
        if temporary_created:
            try:
                os.unlink(temporary, dir_fd=parent_fd)
                os.fsync(parent_fd)
            except OSError:
                pass
        os.close(parent_fd)


__all__ = ["publish_script", "render_script"]
