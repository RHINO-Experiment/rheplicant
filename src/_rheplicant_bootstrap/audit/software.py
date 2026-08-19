"""Acquire closed software facts without importing scientific dependencies."""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from collections.abc import Mapping
from importlib import metadata
from pathlib import Path
from typing import cast

from _rheplicant_bootstrap.errors import ConfigError
from _rheplicant_bootstrap.frozen import freeze_evidence, static_isinstance
from _rheplicant_bootstrap.types import UNAVAILABLE_REASONS, JsonValue

SOFTWARE_KEYS = ("rheplicant", "dependencies", "python", "host")
PROJECT_KEYS = (
    "version",
    "version_reason",
    "source_root",
    "source_root_reason",
    "git_commit",
    "git_commit_reason",
    "dirty",
    "dirty_reason",
    "tracked_diff_sha256",
    "tracked_diff_sha256_reason",
)
VERSION_FACT_KEYS = ("version", "reason")
PYTHON_KEYS = ("version", "implementation", "executable")
HOST_KEYS = ("system", "release", "machine")
DEPENDENCIES = ("jax", "jaxlib", "equinox", "numpy", "numpyro")
_TIMEOUT_SECONDS = 5.0


def _version(distribution: str) -> tuple[str | None, str | None]:
    try:
        return metadata.version(distribution), None
    except metadata.PackageNotFoundError:
        return None, "not_installed"
    except Exception:
        return None, "unreadable"


def _source_root() -> tuple[Path | None, str | None]:
    try:
        distribution = metadata.distribution("rheplicant")
    except metadata.PackageNotFoundError:
        return None, "no_origin"
    except Exception:
        return None, "unreadable"
    try:
        root = Path(distribution.locate_file("")).resolve(strict=True)
        if not root.is_dir():
            return None, "not_regular_file"
        return root, None
    except OSError:
        return None, "unreadable"


def _git(root: Path, *arguments: str) -> tuple[bytes | None, str | None]:
    try:
        completed = subprocess.run(
            ("git", *arguments),
            cwd=root,
            shell=False,
            check=False,
            capture_output=True,
            timeout=_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return None, "timeout"
    except OSError:
        return None, "command_failed"
    if completed.returncode != 0:
        return None, "command_failed"
    return completed.stdout, None


def _project_facts() -> Mapping[str, JsonValue]:
    version, version_reason = _version("rheplicant")
    root, root_reason = _source_root()
    commit = None
    commit_reason = "not_a_git_checkout"
    dirty = None
    dirty_reason = "not_a_git_checkout"
    diff_hash = None
    diff_reason = "not_a_git_checkout"
    if root is not None:
        top, top_reason = _git(root, "rev-parse", "--show-toplevel")
        if top is not None:
            try:
                git_root = Path(top.decode("utf-8", "strict").strip()).resolve(strict=True)
            except (UnicodeError, OSError):
                git_root = None
            if git_root is not None:
                raw_commit, commit_reason = _git(git_root, "rev-parse", "HEAD")
                if raw_commit is not None:
                    try:
                        commit = raw_commit.decode("ascii", "strict").strip()
                    except UnicodeError:
                        commit = None
                        commit_reason = "command_failed"
                    else:
                        commit_reason = None
                status, dirty_reason = _git(git_root, "status", "--porcelain")
                if status is not None:
                    dirty = bool(status)
                    dirty_reason = None
                diff, diff_reason = _git(
                    git_root,
                    "diff",
                    "--no-ext-diff",
                    "--binary",
                )
                if diff is not None:
                    diff_hash = hashlib.sha256(diff).hexdigest()
                    diff_reason = None
        elif top_reason == "timeout":
            commit_reason = dirty_reason = diff_reason = "timeout"
    return {
        "version": version,
        "version_reason": version_reason,
        "source_root": None if root is None else str(root),
        "source_root_reason": root_reason,
        "git_commit": commit,
        "git_commit_reason": commit_reason,
        "dirty": dirty,
        "dirty_reason": dirty_reason,
        "tracked_diff_sha256": diff_hash,
        "tracked_diff_sha256_reason": diff_reason,
    }


def _exact(row: object, keys: tuple[str, ...], where: str) -> Mapping[str, object]:
    if not static_isinstance(row, Mapping) or tuple(row) != keys:
        raise ConfigError(f"{where} has the wrong fields.")
    return cast(Mapping[str, object], row)


def _complement(value: object, reason: object, where: str) -> None:
    if reason is not None and reason not in UNAVAILABLE_REASONS:
        raise ConfigError(f"{where} has an unknown unavailable reason.")
    if (value is None) == (reason is None):
        raise ConfigError(f"{where} value and reason must be exact complements.")


def validate_software(row: object) -> Mapping[str, JsonValue]:
    software = _exact(row, SOFTWARE_KEYS, "software")
    project = _exact(software["rheplicant"], PROJECT_KEYS, "software.rheplicant")
    for field in ("version", "source_root", "git_commit", "dirty", "tracked_diff_sha256"):
        _complement(project[field], project[f"{field}_reason"], f"software.rheplicant.{field}")
    dependencies = software["dependencies"]
    if not static_isinstance(dependencies, Mapping) or tuple(dependencies) != DEPENDENCIES:
        raise ConfigError("software.dependencies has the wrong fields.")
    for name in DEPENDENCIES:
        fact = _exact(dependencies[name], VERSION_FACT_KEYS, f"software.dependencies.{name}")
        _complement(fact["version"], fact["reason"], f"software.dependencies.{name}")
    _exact(software["python"], PYTHON_KEYS, "software.python")
    _exact(software["host"], HOST_KEYS, "software.host")
    frozen = freeze_evidence(software, where="software")
    return cast(Mapping[str, JsonValue], frozen)


def collect_software() -> Mapping[str, JsonValue]:
    """Acquire one detached software row without importing any dependency."""
    dependencies = {}
    for name in DEPENDENCIES:
        version, reason = _version(name)
        dependencies[name] = {"version": version, "reason": reason}
    return validate_software(
        {
            "rheplicant": _project_facts(),
            "dependencies": dependencies,
            "python": {
                "version": platform.python_version(),
                "implementation": platform.python_implementation(),
                "executable": sys.executable,
            },
            "host": {
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
            },
        }
    )


__all__ = [
    "DEPENDENCIES",
    "HOST_KEYS",
    "PROJECT_KEYS",
    "PYTHON_KEYS",
    "SOFTWARE_KEYS",
    "VERSION_FACT_KEYS",
    "collect_software",
    "validate_software",
]
