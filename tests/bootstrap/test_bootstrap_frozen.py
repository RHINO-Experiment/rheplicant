"""Regression tests for immutable bootstrap records."""

from __future__ import annotations

import pytest

from _rheplicant_bootstrap.frozen import freeze, thaw


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
