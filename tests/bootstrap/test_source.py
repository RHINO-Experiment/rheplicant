"""Regression tests for the exact-byte CLI source entry seam."""

from __future__ import annotations

import io
import os
import subprocess
import sys
from pathlib import Path

import pytest

from _rheplicant_bootstrap import source as source_module
from _rheplicant_bootstrap.errors import ConfigError
from _rheplicant_bootstrap.source import read_source
from _rheplicant_bootstrap.yaml import YamlLimits


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


def test_path_source_reads_forward_until_eof_with_bounded_requests(tmp_path, monkeypatch):
    """Catches assuming a legal path-file short read is end-of-file."""
    path = tmp_path / "one-read.yaml"
    path.write_bytes(b"a: 1\n")
    original_fdopen = os.fdopen
    _small_source_limit(monkeypatch)
    reads: list[int] = []

    class ChunkedFile:
        def __init__(self, wrapped):
            self._wrapped = wrapped

        def read(self, size: int) -> bytes:
            reads.append(size)
            return self._wrapped.read(min(size, 2))

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self._wrapped.__exit__(*args)

        def __getattr__(self, name):
            return getattr(self._wrapped, name)

    def tracked_fdopen(*args, **kwargs):
        return ChunkedFile(original_fdopen(*args, **kwargs))

    monkeypatch.setattr(source_module.os, "fdopen", tracked_fdopen)

    assert read_source(str(path), base_dir=None, stdin=None).input_bytes == b"a: 1\n"
    assert reads == [9, 7, 5, 4]


def test_path_base_dir_must_match_the_lexical_parent(tmp_path):
    """Catches silently accepting a contradictory relative-file base directory."""
    path = tmp_path / "config.yaml"
    path.write_bytes(b"{}")

    with pytest.raises(ConfigError, match="base_dir"):
        read_source(str(path), base_dir=str(tmp_path.parent), stdin=None)


def test_cross_directory_symlink_uses_lexical_parent_for_base_dir(tmp_path):
    """Catches resolving a symlink before selecting its configuration base directory."""
    lexical_dir = tmp_path / "links"
    target_dir = tmp_path / "targets"
    lexical_dir.mkdir()
    target_dir.mkdir()
    target = target_dir / "real.yaml"
    target.write_bytes(b"{}")
    linked = lexical_dir / "config.yaml"
    linked.symlink_to(target)

    source = read_source(str(linked), base_dir=str(lexical_dir), stdin=None)

    assert source.base_dir == str(lexical_dir.resolve())
    assert source.source_realpath == str(target.resolve())


@pytest.mark.parametrize("linked", [False, True])
def test_fifo_sources_are_refused_without_blocking(tmp_path, linked):
    """Catches opening a FIFO in blocking mode or reading it before fstat refusal."""
    if not hasattr(os, "mkfifo"):
        pytest.skip("platform does not provide FIFOs")
    fifo = tmp_path / "source.fifo"
    os.mkfifo(fifo)
    source_path = fifo
    if linked:
        source_path = tmp_path / "source-link.yaml"
        source_path.symlink_to(fifo.name)

    source_root = Path(__file__).parents[2] / "src"
    environment = dict(os.environ)
    previous_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(source_root)
        if not previous_pythonpath
        else f"{source_root}{os.pathsep}{previous_pythonpath}"
    )
    script = (
        "from _rheplicant_bootstrap.errors import ConfigError\n"
        "from _rheplicant_bootstrap.source import read_source\n"
        "import sys\n"
        "try:\n"
        "    read_source(sys.argv[1], base_dir=None, stdin=None)\n"
        "except ConfigError:\n"
        "    raise SystemExit(0)\n"
        "raise SystemExit(1)\n"
    )

    completed = subprocess.run(
        [sys.executable, "-c", script, str(source_path)],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
        timeout=5,
    )

    assert completed.returncode == 0, completed.stderr


class _CountingStdin:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.read_sizes: list[int] = []
        self.offset = 0

    def read(self, size: int) -> bytes:
        self.read_sizes.append(size)
        chunk = self.payload[self.offset : self.offset + size]
        self.offset += len(chunk)
        return chunk


class _ChunkedStdin(_CountingStdin):
    def read(self, size: int) -> bytes:
        self.read_sizes.append(size)
        chunk = self.payload[self.offset : self.offset + min(size, 2)]
        self.offset += len(chunk)
        return chunk


def _small_source_limit(monkeypatch) -> None:
    monkeypatch.setattr(source_module, "YamlLimits", lambda: YamlLimits(input_bytes=8))


def test_stdin_reads_forward_until_eof_with_the_input_limit_plus_one(monkeypatch):
    """Catches assuming a legal standard-input short read means EOF."""
    _small_source_limit(monkeypatch)
    stdin = _ChunkedStdin(b"12345")

    source = read_source("-", base_dir=".", stdin=stdin)

    assert source.input_bytes == b"12345"
    assert stdin.read_sizes == [9, 7, 5, 4]


def test_chunked_stdin_overflow_stops_at_the_bounded_extra_byte(monkeypatch):
    """Catches a short-read loop that reads past the one-byte overflow boundary."""
    _small_source_limit(monkeypatch)
    stdin = _ChunkedStdin(b"1234567890")

    with pytest.raises(ConfigError, match="byte count 9 exceeds limit 8"):
        read_source("-", base_dir=".", stdin=stdin)

    assert stdin.read_sizes == [9, 7, 5, 3, 1]


@pytest.mark.parametrize("from_stdin", [False, True])
def test_source_overflow_is_refused_before_retaining_the_extra_byte(
    tmp_path, monkeypatch, from_stdin
):
    """Catches relying on fstat alone or retaining a payload above the configured bound."""
    _small_source_limit(monkeypatch)
    if from_stdin:
        stdin = _CountingStdin(b"123456789")
        with pytest.raises(ConfigError, match="byte count 9 exceeds limit 8"):
            read_source("-", base_dir=".", stdin=stdin)
        assert stdin.read_sizes == [9]
    else:
        path = tmp_path / "over-limit.yaml"
        path.write_bytes(b"123456789")
        original_fstat = os.fstat
        original_stat = os.stat

        def advertised_at_limit_fstat(fd, *args, **kwargs):
            return _resized_stat(original_fstat(fd, *args, **kwargs), 8)

        def advertised_at_limit_stat(target, *args, **kwargs):
            result = original_stat(target, *args, **kwargs)
            return _resized_stat(result, 8) if os.fspath(target) == str(path) else result

        monkeypatch.setattr(source_module.os, "fstat", advertised_at_limit_fstat)
        monkeypatch.setattr(source_module.os, "stat", advertised_at_limit_stat)
        with pytest.raises(ConfigError, match="byte count 9 exceeds limit 8"):
            read_source(str(path), base_dir=None, stdin=None)


@pytest.mark.parametrize(
    ("path_or_dash", "base_dir", "stdin"),
    [
        ("bad\x00name.yaml", None, None),
        ("-", "bad\x00base", io.BytesIO(b"{}")),
        ("-", ".", io.BytesIO(b"{}")),
    ],
)
def test_source_boundary_normalizes_expected_stream_and_path_errors(
    path_or_dash, base_dir, stdin
):
    """Catches raw path/base-dir/closed-stream exceptions leaking past ConfigError."""
    if stdin is not None and base_dir == ".":
        stdin.close()
    with pytest.raises(ConfigError):
        read_source(path_or_dash, base_dir=base_dir, stdin=stdin)


def test_source_normalizes_a_failing_stdin():
    """Catches injected standard-input OSErrors escaping the neutral boundary."""
    class FailingStdin:
        def read(self, size: int) -> bytes:
            raise OSError("read failed")

    with pytest.raises(ConfigError, match="cannot read source"):
        read_source("-", base_dir=".", stdin=FailingStdin())


def _changed_stat(result: os.stat_result) -> os.stat_result:
    fields = list(result)
    fields[1] += 1
    return os.stat_result(fields)


def _resized_stat(result: os.stat_result, size: int) -> os.stat_result:
    fields = list(result)
    fields[6] = size
    return os.stat_result(fields)


def test_changed_lexical_lstat_is_refused(tmp_path, monkeypatch):
    """Catches removing the before/after lexical lstat comparison."""
    path = tmp_path / "config.yaml"
    path.write_bytes(b"{}")
    original_lstat = os.lstat
    calls = 0

    def changed_second_lstat(target, *args, **kwargs):
        nonlocal calls
        result = original_lstat(target, *args, **kwargs)
        if os.fspath(target) == str(path):
            calls += 1
            if calls == 2:
                return _changed_stat(result)
        return result

    monkeypatch.setattr(
        source_module,
        "_lexical_lstat",
        changed_second_lstat,
        raising=False,
    )

    with pytest.raises(ConfigError, match="link changed"):
        source_module._read_path_once(str(path))


def test_late_symlink_swap_to_a_same_inode_hardlink_is_refused(tmp_path, monkeypatch):
    """Catches resolving provenance after the old final lexical-link check."""
    first_target = tmp_path / "first.yaml"
    second_target = tmp_path / "second.yaml"
    first_target.write_bytes(b"answer: 42\n")
    os.link(first_target, second_target)
    linked = tmp_path / "config.yaml"
    linked.symlink_to(first_target.name)
    original_lstat = os.lstat
    calls = 0

    def swap_after_second_lexical_lstat(target, *args, **kwargs):
        nonlocal calls
        result = original_lstat(target, *args, **kwargs)
        if os.fspath(target) == str(linked):
            calls += 1
            if calls == 2:
                linked.unlink()
                linked.symlink_to(second_target.name)
        return result

    monkeypatch.setattr(
        source_module,
        "_lexical_lstat",
        swap_after_second_lexical_lstat,
        raising=False,
    )

    with pytest.raises(ConfigError, match="link changed"):
        source_module._read_path_once(str(linked))


def test_changed_open_target_fstat_is_refused(tmp_path, monkeypatch):
    """Catches removing the before/after open-target fstat comparison."""
    path = tmp_path / "config.yaml"
    path.write_bytes(b"{}")
    original_fstat = os.fstat
    original_stat = os.stat
    calls = 0

    def changed_second_fstat(fd, *args, **kwargs):
        nonlocal calls
        calls += 1
        result = original_fstat(fd, *args, **kwargs)
        return _changed_stat(result) if calls == 2 else result

    def matching_final_stat(target, *args, **kwargs):
        result = original_stat(target, *args, **kwargs)
        return _changed_stat(result) if os.fspath(target) == str(path) else result

    monkeypatch.setattr(source_module.os, "fstat", changed_second_fstat)
    monkeypatch.setattr(source_module.os, "stat", matching_final_stat)

    with pytest.raises(ConfigError, match="target changed"):
        source_module._read_path_once(str(path))


def test_changed_final_realpath_stat_is_refused(tmp_path, monkeypatch):
    """Catches removing the final realpath-stat target comparison."""
    path = tmp_path / "config.yaml"
    path.write_bytes(b"{}")
    original_stat = os.stat

    def changed_final_stat(target, *args, **kwargs):
        result = original_stat(target, *args, **kwargs)
        return _changed_stat(result) if os.fspath(target) == str(path) else result

    monkeypatch.setattr(source_module.os, "stat", changed_final_stat)

    with pytest.raises(ConfigError, match="target changed"):
        source_module._read_path_once(str(path))
