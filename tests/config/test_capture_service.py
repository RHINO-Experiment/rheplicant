from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path

import pytest

from _rheplicant_bootstrap.capture import (
    CaptureService,
    manifest_entry,
    tree_digest,
)
from _rheplicant_bootstrap.types import DestinationDescriptor, LayerIdentity
from rheplicant.config.errors import ConfigError

DESTINATION = DestinationDescriptor(
    "observation.data", "config_path", "observation.data"
)
LAYER = LayerIdentity("base", None)


def test_reader_consumes_the_bytes_whose_digest_was_recorded(tmp_path):
    source = tmp_path / "data.txt"
    source.write_bytes(b"1\n")
    seen = {}
    verified = []
    service = CaptureService(
        tmp_path / "captures",
        on_verified=lambda layer, row: verified.append((layer, row)),
    )

    def reader(path):
        seen["path"] = Path(path)
        source.write_bytes(b"9\n")
        return Path(path).read_bytes()

    result = service.consume_file(
        source,
        layer=LAYER,
        destination=DESTINATION,
        format="txt",
        reader=reader,
    )
    record = verified[0][1]
    assert result == b"1\n"
    assert record.sha256 == hashlib.sha256(b"1\n").hexdigest()
    assert seen["path"] != source
    assert not seen["path"].exists()
    service.close()


def test_reader_exception_still_records_verified_input_and_cleans(tmp_path):
    source = tmp_path / "data.txt"
    source.write_text("data")
    verified = []
    service = CaptureService(
        tmp_path / "captures", on_verified=lambda *row: verified.append(row)
    )
    seen = {}

    def reader(path):
        seen["path"] = path
        raise LookupError("reader failed")

    with pytest.raises(LookupError, match="reader failed"):
        service.consume_file(
            source,
            layer=LAYER,
            destination=DESTINATION,
            format="txt",
            reader=reader,
        )
    assert len(verified) == 1
    assert not seen["path"].exists()
    service.close()


def test_snapshot_mutation_is_refused_and_not_recorded(tmp_path):
    source = tmp_path / "data.txt"
    source.write_text("data")
    verified = []
    service = CaptureService(
        tmp_path / "captures", on_verified=lambda *row: verified.append(row)
    )

    def reader(path):
        path.write_text("changed")

    with pytest.raises(ConfigError, match="snapshot changed"):
        service.consume_file(
            source,
            layer=LAYER,
            destination=DESTINATION,
            format="txt",
            reader=reader,
        )
    assert verified == []
    service.close()


def test_symlink_records_lexical_and_real_paths(tmp_path):
    target = tmp_path / "target.dat"
    target.write_bytes(b"x")
    link = tmp_path / "link.dat"
    link.symlink_to(target)
    rows = []
    with CaptureService(
        tmp_path / "captures", on_verified=lambda _layer, row: rows.append(row)
    ) as service:
        assert service.consume_file(
            link,
            layer=LAYER,
            destination=DESTINATION,
            format="txt",
            reader=Path.read_bytes,
        ) == b"x"
    assert rows[0].path == str(link.absolute())
    assert rows[0].realpath == str(target.resolve())


def test_capture_modes_ignore_umask(tmp_path):
    source = tmp_path / "data.txt"
    source.write_bytes(b"x")
    previous = os.umask(0)
    try:
        service = CaptureService(tmp_path / "captures")
        captured = service.capture_file(
            source, destination=DESTINATION, format="txt"
        )
    finally:
        os.umask(previous)
    assert stat.S_IMODE((tmp_path / "captures").stat().st_mode) == 0o700
    assert stat.S_IMODE(captured.snapshot_path.stat().st_mode) == 0o600
    service.close()


def test_directory_tree_is_sorted_hashed_and_reverified(tmp_path):
    root = tmp_path / "tree"
    root.mkdir()
    (root / "b.txt").write_bytes(b"b")
    (root / "a.txt").write_bytes(b"a")

    def enumerate_(source):
        return tuple(
            manifest_entry(source, path, path.relative_to(source).as_posix())
            for path in reversed(sorted(source.glob("*.txt")))
        )

    rows = []
    with CaptureService(
        tmp_path / "captures", on_verified=lambda _layer, row: rows.append(row)
    ) as service:
        names = service.consume_directory(
            root,
            layer=LAYER,
            destination=DESTINATION,
            format="cst",
            enumerate_manifest=enumerate_,
            reader=lambda snapshot: sorted(
                path.name for path in snapshot.glob("*.txt")
            ),
        )
    assert names == ["a.txt", "b.txt"]
    assert [member.relative_path for member in rows[0].members] == [
        "a.txt",
        "b.txt",
    ]
    assert rows[0].sha256 == tree_digest(rows[0].members)


@pytest.mark.parametrize("relative", ["", ".", "../x", "/absolute"])
def test_directory_manifest_rejects_unsafe_names(tmp_path, relative):
    root = tmp_path / "tree"
    root.mkdir()
    source = root / "x"
    source.write_bytes(b"x")
    entry = manifest_entry(root, source, relative)
    with CaptureService(tmp_path / "captures") as service:
        with pytest.raises(ConfigError, match="unsafe member"):
            service.capture_directory(
                root,
                destination=DESTINATION,
                format="cst",
                enumerate_manifest=lambda _root: (entry,),
            )
