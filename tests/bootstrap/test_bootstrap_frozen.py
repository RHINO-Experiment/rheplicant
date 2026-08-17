"""Regression tests for immutable bootstrap records."""

from __future__ import annotations

import struct
from collections.abc import Mapping

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


class _ItemsMapping(Mapping):
    """Mapping fixture whose source keys need not be hashable or comparable."""

    def __init__(self, pairs):
        self._pairs = tuple(pairs)

    def __getitem__(self, key):
        for given, value in self._pairs:
            if given is key:
                return value
        raise KeyError(key)

    def __iter__(self):
        return (key for key, _ in self._pairs)

    def __len__(self):
        return len(self._pairs)

    def items(self):
        return iter(self._pairs)


class _HostileStr(str):
    def __str__(self):
        raise AssertionError("__str__ must not run")

    def split(self, *args, **kwargs):
        raise AssertionError("split must not run")

    def __eq__(self, other):
        raise AssertionError("equality must not run")

    def __hash__(self):
        raise AssertionError("hash must not run")

    def __repr__(self):
        raise AssertionError("repr must not run")


class _HostileInt(int):
    def __int__(self):
        raise AssertionError("__int__ must not run")

    def __index__(self):
        raise AssertionError("__index__ must not run")

    def __eq__(self, other):
        raise AssertionError("equality must not run")

    def __hash__(self):
        raise AssertionError("hash must not run")

    def __repr__(self):
        raise AssertionError("repr must not run")


class _HostileFloat(float):
    def __float__(self):
        raise AssertionError("__float__ must not run")

    def __eq__(self, other):
        raise AssertionError("equality must not run")

    def __hash__(self):
        raise AssertionError("hash must not run")

    def __repr__(self):
        raise AssertionError("repr must not run")


class _HostileBytes(bytes):
    def __bytes__(self):
        raise AssertionError("__bytes__ must not run")

    def __eq__(self, other):
        raise AssertionError("equality must not run")

    def __hash__(self):
        raise AssertionError("hash must not run")

    def __repr__(self):
        raise AssertionError("repr must not run")


class _HostileBytearray(bytearray):
    def __bytes__(self):
        raise AssertionError("__bytes__ must not run")

    def __buffer__(self, flags):
        raise AssertionError("__buffer__ must not run")

    def __repr__(self):
        raise AssertionError("repr must not run")


def test_freeze_evidence_scalar_canonicalization_uses_only_base_operations():
    """Catches calling overrideable conversion, hash, equality, split, or repr hooks."""
    source = _ItemsMapping(
        [
            (
                _HostileStr("key"),
                [
                    _HostileStr("text"),
                    _HostileInt(2),
                    _HostileFloat(3.5),
                    _HostileBytes(b"bytes"),
                    _HostileBytearray(b"buffer"),
                ],
            )
        ]
    )

    frozen = _freeze_evidence(source, where="snapshot.document")

    assert tuple(frozen) == ("key",)
    values = frozen["key"]
    assert tuple(type(value) for value in values) == (
        str,
        int,
        float,
        bytes,
        bytes,
    )
    assert values == ("text", 2, 3.5, b"bytes", b"buffer")


def test_freeze_evidence_preserves_float_bits_during_base_canonicalization():
    nan = struct.unpack("!d", bytes.fromhex("7ff8000000000042"))[0]
    negative_zero = _HostileFloat(-0.0)

    frozen = _freeze_evidence(
        {"nan": _HostileFloat(nan), "negative_zero": negative_zero},
        where="snapshot.document",
    )

    assert struct.pack("!d", frozen["nan"]) == struct.pack("!d", nan)
    assert struct.pack("!d", frozen["negative_zero"]) == bytes.fromhex(
        "8000000000000000"
    )


@pytest.mark.parametrize("key", [None, True, 1, 1.0, b"key"])
def test_freeze_evidence_requires_exact_string_mapping_keys(key):
    source = _ItemsMapping([(key, "value")])

    with pytest.raises(
        ConfigError,
        match=rf"snapshot\.document.*mapping key type {type(key).__name__}",
    ):
        _freeze_evidence(source, where="snapshot.document")


def test_freeze_evidence_normalizes_a_released_memoryview_without_repr():
    view = memoryview(b"payload")
    view.release()

    with pytest.raises(ConfigError, match=r"snapshot\.document.*memoryview"):
        _freeze_evidence(view, where="snapshot.document")


def test_thaw_preserves_aliases_while_detaching_from_frozen_evidence():
    shared = {"leaf": [1]}
    frozen = _freeze_evidence(
        {"first": shared, "second": shared}, where="snapshot.document"
    )

    mutable = thaw(frozen)

    assert mutable["first"] is mutable["second"]
    assert mutable["first"] is not shared
    assert mutable["first"]["leaf"] is mutable["second"]["leaf"]


def test_thaw_does_not_treat_the_interned_empty_tuple_as_an_alias():
    mutable = thaw({"first": (), "second": ()})

    assert mutable["first"] == []
    assert mutable["second"] == []
    assert mutable["first"] is not mutable["second"]


class _HostileProtocolError(Exception):
    def __str__(self):
        raise AssertionError("exception text must not run")

    def __repr__(self):
        raise AssertionError("exception repr must not run")


class _FailingItemsMapping(_ItemsMapping):
    def items(self):
        raise _HostileProtocolError()


class _FailingSequence(list):
    def __iter__(self):
        raise _HostileProtocolError()


@pytest.mark.parametrize("value", [_FailingItemsMapping([]), _FailingSequence()])
def test_freeze_evidence_normalizes_ordinary_container_protocol_failures(value):
    with pytest.raises(ConfigError, match=rf"snapshot\.document.*{type(value).__name__}"):
        _freeze_evidence(value, where="snapshot.document")


def test_freeze_evidence_does_not_catch_process_control_baseexceptions():
    class StopNow(BaseException):
        pass

    class StoppingMapping(_ItemsMapping):
        def items(self):
            raise StopNow

    with pytest.raises(StopNow):
        _freeze_evidence(StoppingMapping([]), where="snapshot.document")


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


def test_freeze_evidence_checks_cached_subtree_height_at_every_alias():
    """Catches a shallow memo hit bypassing the depth limit on a deep alias."""
    shared = [[]]
    deep = shared
    for _ in range(98):
        deep = [deep]
    value = {"shallow": shared, "deep": deep}

    with pytest.raises(ConfigError, match=r"snapshot\.document.*depth 101.*list"):
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
