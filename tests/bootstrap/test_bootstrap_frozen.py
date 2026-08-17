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


class _StatefulStr(str):
    def __new__(cls, value):
        instance = super().__new__(cls, value)
        instance.state = []
        return instance


class _StatefulInt(int):
    def __new__(cls, value):
        instance = super().__new__(cls, value)
        instance.state = []
        return instance


class _StatefulBytes(bytes):
    def __new__(cls, value):
        instance = super().__new__(cls, value)
        instance.state = []
        return instance


class _MutableHashStr(str):
    def __new__(cls, value):
        instance = super().__new__(cls, value)
        instance.salt = 0
        return instance

    def __hash__(self):
        return str.__hash__(self) ^ self.salt


def test_freeze_evidence_canonicalizes_scalar_subclasses_and_mapping_keys():
    """Catches audit evidence retaining stateful scalar-subclass identities."""
    key = _MutableHashStr("key")
    text = _StatefulStr("text")
    number = _StatefulInt(2)
    payload = _StatefulBytes(b"bytes")

    frozen = _freeze_evidence(
        {key: [text, number, payload]}, where="snapshot.document"
    )
    frozen_key = next(iter(frozen))
    frozen_values = frozen[frozen_key]

    assert type(frozen_key) is str
    assert tuple(type(value) for value in frozen_values) == (str, int, bytes)
    key.salt = 1
    text.state.append("late")
    number.state.append("late")
    payload.state.append("late")
    assert frozen["key"] == ("text", 2, b"bytes")


class _CountedMapping(dict):
    item_visits = 0

    def items(self):
        type(self).item_visits += 1
        return super().items()


def test_freeze_evidence_memoizes_completed_containers_in_a_shared_dag():
    """Catches repeatedly walking aliases instead of doing unique-node work."""
    shared = _CountedMapping({"leaf": [1]})
    original = {"first": shared, "many": [shared] * 10_000}
    _CountedMapping.item_visits = 0

    frozen = _freeze_evidence(original, where="snapshot.document")

    assert _CountedMapping.item_visits == 1
    assert frozen["first"] is frozen["many"][0]
    assert all(item is frozen["first"] for item in frozen["many"])


def test_freeze_evidence_refuses_depth_101_with_a_controlled_error():
    value = []
    for _ in range(100):
        value = [value]

    with pytest.raises(ConfigError, match=r"snapshot\.document.*depth.*101.*list"):
        _freeze_evidence(value, where="snapshot.document")


def test_freeze_evidence_refuses_unique_node_250001_with_a_controlled_error():
    value = [[] for _ in range(250_000)]

    with pytest.raises(
        ConfigError, match=r"snapshot\.document.*250001.*list"
    ):
        _freeze_evidence(value, where="snapshot.document")


def test_freeze_evidence_never_evaluates_an_unsupported_leaf_repr():
    class HostileRepresentation:
        def __repr__(self):
            raise AssertionError("repr must not run")

    with pytest.raises(
        ConfigError, match=r"snapshot\.document.*HostileRepresentation"
    ):
        _freeze_evidence(
            {"unsafe": HostileRepresentation()}, where="snapshot.document"
        )


def test_freeze_evidence_normalizes_a_nested_recursion_error():
    class RecursingList(list):
        def __iter__(self):
            raise RecursionError("injected recursion")

    with pytest.raises(ConfigError, match=r"snapshot\.document.*RecursingList"):
        _freeze_evidence(RecursingList([1]), where="snapshot.document")
