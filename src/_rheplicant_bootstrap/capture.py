"""Immutable snapshots for every declarative file and directory input."""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal, TypeAlias, TypeVar

from .errors import ConfigError
from .types import DestinationDescriptor, LayerIdentity

T = TypeVar("T")
TREE_MAGIC = b"rheplicant-captured-tree-v1\x00"


@dataclass(frozen=True, slots=True)
class CaptureIdentity:
    device: int
    inode: int
    mode: int
    size: int
    mtime_ns: int
    ctime_ns: int


@dataclass(frozen=True, slots=True)
class ManifestEntry:
    relative_path: str
    source_path: Path
    link_identity: CaptureIdentity
    target_identity: CaptureIdentity


ManifestEnumerator: TypeAlias = Callable[[Path], Sequence[ManifestEntry]]


@dataclass(frozen=True, slots=True)
class CapturedMember:
    relative_path: str
    path: str
    realpath: str
    sha256: str


@dataclass(frozen=True, slots=True)
class CapturedInput:
    document_path: str
    path: str
    realpath: str
    format: str
    kind: Literal["file", "directory"]
    sha256: str
    members: Sequence[CapturedMember]


@dataclass(frozen=True, slots=True)
class CapturedSource:
    record: CapturedInput
    snapshot_path: Path


_CAPTURE_ROUTES: dict[str, str] = {}


def register_capture_route(name: str, *, owner: str) -> None:
    if not isinstance(name, str) or not name:
        raise ConfigError("capture route names are non-empty strings")
    previous = _CAPTURE_ROUTES.get(name)
    if previous is not None and previous != owner:
        raise ConfigError(
            f"capture route {name!r} is registered by both {previous} and {owner}"
        )
    _CAPTURE_ROUTES[name] = owner


def capture_routes() -> Mapping[str, str]:
    return dict(_CAPTURE_ROUTES)


def _identity(value: os.stat_result) -> CaptureIdentity:
    return CaptureIdentity(
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def manifest_entry(root: Path, source: Path, relative_path: str) -> ManifestEntry:
    source = source.absolute()
    return ManifestEntry(
        relative_path,
        source,
        _identity(source.lstat()),
        _identity(source.stat()),
    )


def tree_digest(members: Sequence[CapturedMember]) -> str:
    digest = hashlib.sha256(TREE_MAGIC)
    for member in sorted(members, key=lambda row: row.relative_path):
        name = member.relative_path.encode("utf-8")
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(bytes.fromhex(member.sha256))
    return digest.hexdigest()


def captured_input_json(row: CapturedInput) -> Mapping[str, object]:
    return {
        "document_path": row.document_path,
        "path": row.path,
        "realpath": row.realpath,
        "format": row.format,
        "kind": row.kind,
        "sha256": row.sha256,
        "members": tuple(
            {
                "relative_path": member.relative_path,
                "path": member.path,
                "realpath": member.realpath,
                "sha256": member.sha256,
            }
            for member in row.members
        ),
    }


class CaptureService:
    """Copy a declared source once, then let readers see only the copy."""

    def __init__(
        self,
        root: Path,
        *,
        on_verified: Callable[[LayerIdentity, CapturedInput], None] | None = None,
    ) -> None:
        self._root = Path(root)
        self._on_verified = on_verified
        self._lock = threading.RLock()
        self._counter = 0
        self._captures: set[Path] = set()
        self._closed = False
        self._root.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(self._root, 0o700)

    def __enter__(self) -> CaptureService:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _slot(self, *, directory: bool) -> Path:
        with self._lock:
            if self._closed:
                raise ConfigError("capture service is closed")
            self._counter += 1
            path = self._root / f"capture-{self._counter:08d}"
            if directory:
                path.mkdir(mode=0o700)
                os.chmod(path, 0o700)
            else:
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                os.fchmod(fd, 0o600)
                os.close(fd)
            self._captures.add(path)
            return path

    @staticmethod
    def _stream(source: Path, snapshot: Path, expected: CaptureIdentity) -> str:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        source_fd = os.open(source, flags)
        try:
            if _identity(os.fstat(source_fd)) != expected:
                raise ConfigError(f"declarative input changed before capture: {source}")
            output_fd = os.open(snapshot, os.O_WRONLY | os.O_TRUNC)
            digest = hashlib.sha256()
            try:
                os.fchmod(output_fd, 0o600)
                while True:
                    chunk = os.read(source_fd, 1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
                    view = memoryview(chunk)
                    while view:
                        written = os.write(output_fd, view)
                        if written <= 0:
                            raise ConfigError("capture snapshot write made no progress")
                        view = view[written:]
                os.fsync(output_fd)
            finally:
                os.close(output_fd)
            if _identity(os.fstat(source_fd)) != expected:
                raise ConfigError(f"declarative input changed during capture: {source}")
            return digest.hexdigest()
        finally:
            os.close(source_fd)

    def capture_file(
        self,
        source: Path,
        *,
        destination: DestinationDescriptor,
        format: str,
        declared_sha256: str | None = None,
    ) -> CapturedSource:
        source = Path(source).absolute()
        try:
            before_link = _identity(source.lstat())
            before_target = _identity(source.stat())
        except OSError as exc:
            raise ConfigError(f"cannot inspect declarative input {source}: {exc}") from exc
        if not stat.S_ISREG(before_target.mode):
            raise ConfigError(f"declarative input is not a regular file: {source}")
        snapshot = self._slot(directory=False)
        try:
            digest = self._stream(source, snapshot, before_target)
            if (
                _identity(source.lstat()) != before_link
                or _identity(source.stat()) != before_target
            ):
                raise ConfigError(f"declarative input changed during capture: {source}")
            if declared_sha256 is not None and declared_sha256 != digest:
                raise ConfigError(
                    f"{source} hashes to {digest}, and this reference declares "
                    f"{declared_sha256}. The file has changed since the declaration "
                    "was written, or the declaration came from a different copy. A "
                    "run against different bytes than the ones recorded is not the "
                    "run the artefact describes."
                )
            record = CapturedInput(
                destination.document_path,
                str(source),
                str(source.resolve()),
                format,
                "file",
                digest,
                (),
            )
            return CapturedSource(record, snapshot)
        except BaseException:
            self._remove(snapshot)
            raise

    @staticmethod
    def _validate_manifest(
        root: Path, entries: Sequence[ManifestEntry]
    ) -> tuple[ManifestEntry, ...]:
        normalized: list[ManifestEntry] = []
        seen: set[str] = set()
        root = root.absolute()
        for entry in entries:
            rel = PurePosixPath(entry.relative_path)
            if (
                not entry.relative_path
                or entry.relative_path == "."
                or rel.is_absolute()
                or any(part in ("", ".", "..") for part in rel.parts)
            ):
                raise ConfigError(
                    f"capture manifest has unsafe member {entry.relative_path!r}"
                )
            name = rel.as_posix()
            if name in seen:
                raise ConfigError(f"capture manifest repeats member {name!r}")
            seen.add(name)
            expected = root.joinpath(*rel.parts).absolute()
            if entry.source_path.absolute() != expected:
                raise ConfigError(
                    f"capture manifest member {name!r} does not name root/member"
                )
            if not stat.S_ISREG(entry.target_identity.mode):
                raise ConfigError(f"capture manifest member is not regular: {name!r}")
            normalized.append(entry)
        return tuple(sorted(normalized, key=lambda row: row.relative_path))

    def capture_directory(
        self,
        source: Path,
        *,
        destination: DestinationDescriptor,
        format: str,
        enumerate_manifest: ManifestEnumerator,
        declared_sha256: str | None = None,
    ) -> CapturedSource:
        source = Path(source).absolute()
        first = self._validate_manifest(source, tuple(enumerate_manifest(source)))
        snapshot = self._slot(directory=True)
        members: list[CapturedMember] = []
        try:
            for entry in first:
                target = snapshot.joinpath(*PurePosixPath(entry.relative_path).parts)
                target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                os.chmod(target.parent, 0o700)
                fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                os.fchmod(fd, 0o600)
                os.close(fd)
                if (
                    _identity(entry.source_path.lstat()) != entry.link_identity
                    or _identity(entry.source_path.stat()) != entry.target_identity
                ):
                    raise ConfigError(
                        f"capture manifest member changed: {entry.relative_path}"
                    )
                digest = self._stream(entry.source_path, target, entry.target_identity)
                if (
                    _identity(entry.source_path.lstat()) != entry.link_identity
                    or _identity(entry.source_path.stat()) != entry.target_identity
                ):
                    raise ConfigError(
                        f"capture manifest member changed: {entry.relative_path}"
                    )
                members.append(
                    CapturedMember(
                        entry.relative_path,
                        str(entry.source_path),
                        str(entry.source_path.resolve()),
                        digest,
                    )
                )
            second = self._validate_manifest(source, tuple(enumerate_manifest(source)))
            if second != first:
                raise ConfigError(f"declarative input tree changed during capture: {source}")
            digest = tree_digest(members)
            if declared_sha256 is not None and declared_sha256 != digest:
                raise ConfigError(
                    f"{source} tree hashes to {digest}, and this reference declares "
                    f"{declared_sha256}."
                )
            record = CapturedInput(
                destination.document_path,
                str(source),
                str(source.resolve()),
                format,
                "directory",
                digest,
                tuple(members),
            )
            return CapturedSource(record, snapshot)
        except BaseException:
            self._remove(snapshot)
            raise

    @staticmethod
    def _rehash(captured: CapturedSource) -> None:
        if captured.record.kind == "file":
            digest = hashlib.sha256(captured.snapshot_path.read_bytes()).hexdigest()
            if digest != captured.record.sha256:
                raise ConfigError("captured input snapshot changed while being read")
            return
        actual_paths = tuple(
            sorted(
                path.relative_to(captured.snapshot_path).as_posix()
                for path in captured.snapshot_path.rglob("*")
                if path.is_file()
            )
        )
        expected_paths = tuple(
            sorted(member.relative_path for member in captured.record.members)
        )
        if actual_paths != expected_paths:
            raise ConfigError("captured input tree members changed while being read")
        members = tuple(
            CapturedMember(
                member.relative_path,
                member.path,
                member.realpath,
                hashlib.sha256(
                    captured.snapshot_path.joinpath(
                        *PurePosixPath(member.relative_path).parts
                    ).read_bytes()
                ).hexdigest(),
            )
            for member in captured.record.members
        )
        if tree_digest(members) != captured.record.sha256:
            raise ConfigError("captured input tree changed while being read")

    def _consume_verified(
        self,
        captured: CapturedSource,
        *,
        layer: LayerIdentity,
        reader: Callable[[Path], T],
    ) -> T:
        result: T
        error: BaseException | None = None
        try:
            try:
                result = reader(captured.snapshot_path)
            except BaseException as exc:
                error = exc
            self._rehash(captured)
            if self._on_verified is not None:
                self._on_verified(layer, captured.record)
            if error is not None:
                raise error
            return result
        finally:
            self._remove(captured.snapshot_path)

    def consume_file(
        self,
        source: Path,
        *,
        layer: LayerIdentity,
        destination: DestinationDescriptor,
        format: str,
        reader: Callable[[Path], T],
        declared_sha256: str | None = None,
    ) -> T:
        captured = self.capture_file(
            source,
            destination=destination,
            format=format,
            declared_sha256=declared_sha256,
        )
        return self._consume_verified(captured, layer=layer, reader=reader)

    def consume_directory(
        self,
        source: Path,
        *,
        layer: LayerIdentity,
        destination: DestinationDescriptor,
        format: str,
        enumerate_manifest: ManifestEnumerator,
        reader: Callable[[Path], T],
        declared_sha256: str | None = None,
    ) -> T:
        captured = self.capture_directory(
            source,
            destination=destination,
            format=format,
            enumerate_manifest=enumerate_manifest,
            declared_sha256=declared_sha256,
        )
        return self._consume_verified(captured, layer=layer, reader=reader)

    def _remove(self, path: Path) -> None:
        with self._lock:
            self._captures.discard(path)
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            captures = tuple(self._captures)
            self._captures.clear()
        for path in captures:
            self._remove(path)
        if self._root.exists():
            shutil.rmtree(self._root)
