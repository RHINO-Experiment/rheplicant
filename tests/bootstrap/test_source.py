"""Regression tests for the exact-byte CLI source entry seam."""

from __future__ import annotations

import io
import os

import pytest

from _rheplicant_bootstrap.errors import ConfigError
from _rheplicant_bootstrap.source import read_source


def test_source_keeps_the_exact_bytes(tmp_path):
    """Catches decoding, resolving the lexical path, or replacing its bytes."""
    raw = b"\xef\xbb\xbfschema_version: 1 # comment\n"
    path = tmp_path / "linked.yaml"
    target = tmp_path / "real.yaml"
    target.write_bytes(raw)
    path.symlink_to(target.name)

    source = read_source(str(path), base_dir=None, stdin=None)

    assert source.input_bytes == raw
    assert source.source_path == str(path.absolute())
    assert source.source_realpath == str(target.resolve())
    assert source.source_name == str(path.absolute())
    assert source.base_dir == str(tmp_path.resolve())
    assert source.launch_mode == "cli"


def test_stdin_has_one_closed_identity():
    """Catches filesystem identities leaking into standard-input records."""
    source = read_source("-", base_dir=".", stdin=io.BytesIO(b"{}"))

    assert (source.source_path, source.source_name, source.source_realpath) == (
        "<stdin>",
        "<stdin>",
        None,
    )
    assert os.path.isabs(source.base_dir)
    assert source.launch_mode == "cli"


def test_path_source_reads_the_open_file_once(tmp_path, monkeypatch):
    """Catches a second read that could observe different source bytes."""
    path = tmp_path / "one-read.yaml"
    path.write_bytes(b"a: 1\n")
    original_open = open
    reads: list[int] = []

    class ReadOnceFile:
        def __init__(self, wrapped):
            self._wrapped = wrapped

        def read(self, *args, **kwargs):
            reads.append(1)
            return self._wrapped.read(*args, **kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self._wrapped.__exit__(*args)

        def __getattr__(self, name):
            return getattr(self._wrapped, name)

    def tracked_open(*args, **kwargs):
        return ReadOnceFile(original_open(*args, **kwargs))

    monkeypatch.setattr("builtins.open", tracked_open)

    assert read_source(str(path), base_dir=None, stdin=None).input_bytes == b"a: 1\n"
    assert reads == [1]


def test_path_base_dir_must_match_the_lexical_parent(tmp_path):
    """Catches silently accepting a contradictory relative-file base directory."""
    path = tmp_path / "config.yaml"
    path.write_bytes(b"{}")

    with pytest.raises(ConfigError, match="base_dir"):
        read_source(str(path), base_dir=str(tmp_path.parent), stdin=None)
