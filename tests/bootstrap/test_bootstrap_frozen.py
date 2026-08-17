"""Regression tests for immutable bootstrap records."""

from __future__ import annotations

import pytest

from _rheplicant_bootstrap import frozen as frozen_module
from _rheplicant_bootstrap.errors import ConfigError
from _rheplicant_bootstrap.frozen import freeze, thaw


def _freeze_evidence(value, *, where):
    helper = getattr(frozen_module, "freeze_evidence", None)
    assert helper is not None, "strict evidence freezing is not implemented"
    return helper(value, where=where)


def test_freeze_is_recursive_and_thaw_is_detached():
    """Catches a shallow copy or a thaw that returns frozen children."""
    original = {"a": [{"b": 1}]}

    frozen = freeze(original)
    original["a"][0]["b"] = 2

    assert frozen["a"][0]["b"] == 1
    mutable = thaw(frozen)
    mutable["a"][0]["b"] = 3
    assert frozen["a"][0]["b"] == 1
    with pytest.raises(TypeError):
        frozen["a"] = ()


def test_freeze_evidence_canonicalizes_buffers_and_detaches_every_container():
    """Catches mutable byte buffers or shallow containers entering audit evidence."""
    first = bytearray(b"first")
    second_buffer = bytearray(b"second")
    original = {"items": [first, memoryview(second_buffer)]}

    frozen = _freeze_evidence(original, where="snapshot.document")
    first[:] = b"xxxxx"
    second_buffer[:] = b"yyyyyy"
    original["items"].append(b"late")

    assert frozen == {"items": (b"first", b"second")}
    assert isinstance(frozen["items"], tuple)
    with pytest.raises(TypeError):
        frozen["late"] = True


def test_freeze_evidence_refuses_an_unsupported_leaf_with_its_location():
    """Catches sharing a mutable arbitrary object through immutable evidence records."""
    with pytest.raises(ConfigError, match="merge.document"):
        _freeze_evidence({"unsafe": object()}, where="merge.document")


def test_freeze_evidence_refuses_cycles_with_its_location():
    """Catches recursive input hanging or overflowing the evidence freezer."""
    recursive = []
    recursive.append(recursive)
    with pytest.raises(ConfigError, match="snapshot.document"):
        _freeze_evidence(recursive, where="snapshot.document")
