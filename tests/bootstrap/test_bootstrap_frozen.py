"""Regression tests for immutable bootstrap records."""

from __future__ import annotations

import ast
import struct
import subprocess
import sys
import textwrap
import traceback
from array import array
from collections import deque
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType

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


@pytest.mark.parametrize("failure", ["config", "ordinary", "base"])
def test_freeze_evidence_type_checks_do_not_run_metaclass_equality(failure):
    class HostileConfigError(ConfigError):
        def __str__(self):
            raise AssertionError("marker text must not run")

        def __repr__(self):
            raise AssertionError("marker repr must not run")

    class HostileError(Exception):
        pass

    class StopNow(BaseException):
        pass

    marker = {
        "config": HostileConfigError("private equality marker"),
        "ordinary": HostileError("private equality marker"),
        "base": StopNow("stop"),
    }[failure]

    class HostileMeta(type):
        equality_calls = 0

        def __eq__(cls, other):
            cls.equality_calls += 1
            raise marker

        __hash__ = type.__hash__

    class HostileLeaf(metaclass=HostileMeta):
        pass

    with pytest.raises(ConfigError) as caught:
        _freeze_evidence(
            {"unsafe": HostileLeaf()}, where="snapshot.document"
        )

    assert str(caught.value) == (
        "snapshot.document: unsupported evidence leaf type HostileLeaf."
    )
    assert HostileMeta.equality_calls == 0


@pytest.mark.parametrize(
    "value",
    (array("i", (1, 2)), deque((1, 2))),
    ids=("array", "deque"),
)
def test_freeze_evidence_preserves_stdlib_virtual_sequence_compatibility(
    value,
):
    assert _freeze_evidence(value, where="snapshot.document") == (1, 2)


@pytest.mark.parametrize(
    "value",
    ({1, 2}, frozenset((1, 2)), iter((1, 2))),
    ids=("set", "frozenset", "generator"),
)
def test_freeze_evidence_rejects_unordered_or_streaming_iterables(value):
    with pytest.raises(ConfigError, match="unsupported evidence leaf type"):
        _freeze_evidence(value, where="snapshot.document")


def test_freeze_evidence_preserves_registered_abc_compatibility_without_class_hooks():
    class VirtualSequence:
        class_calls = 0

        @property
        def __class__(self):
            type(self).class_calls += 1
            raise RuntimeError("sequence class descriptor secret")

        def __init__(self):
            self.values = (1, 2)

        def __getitem__(self, index):
            return self.values[index]

        def __len__(self):
            return len(self.values)

    class VirtualMapping:
        class_calls = 0

        @property
        def __class__(self):
            type(self).class_calls += 1
            raise RuntimeError("mapping class descriptor secret")

        def __init__(self):
            self.values = {"answer": 42}

        def items(self):
            return self.values.items()

    Sequence.register(VirtualSequence)
    Mapping.register(VirtualMapping)

    assert _freeze_evidence(
        VirtualSequence(), where="snapshot.sequence"
    ) == (1, 2)
    assert _freeze_evidence(
        VirtualMapping(), where="snapshot.mapping"
    ) == {"answer": 42}
    assert VirtualSequence.class_calls == 0
    assert VirtualMapping.class_calls == 0


def test_static_abc_lookup_never_runs_an_unrelated_registered_metaclass_hook():
    class PoisonMeta(type):
        armed = False
        subclass_calls = 0

        def __subclasscheck__(cls, subclass):
            if PoisonMeta.armed:
                PoisonMeta.subclass_calls += 1
                raise KeyboardInterrupt("registered poison")
            return False

    class Poison(metaclass=PoisonMeta):
        pass

    class Plain:
        pass

    Sequence.register(Poison)
    PoisonMeta.armed = True
    try:
        assert frozen_module.static_isinstance(Plain(), Sequence) is False
    finally:
        PoisonMeta.armed = False
    assert PoisonMeta.subclass_calls == 0


def test_static_protocol_lookup_preserves_registered_base_subclasses():
    class RegisteredBase:
        def __init__(self, values):
            self.values = tuple(values)

        def __getitem__(self, index):
            return self.values[index]

        def __len__(self):
            return len(self.values)

    class RegisteredChild(RegisteredBase):
        pass

    Sequence.register(RegisteredBase)

    assert _freeze_evidence(
        RegisteredChild((1, 2)), where="snapshot.registered_base"
    ) == (1, 2)


def test_static_protocol_lookup_accepts_unregistered_mapping_and_sequence_ducks():
    class DuckSequence:
        def __init__(self):
            self.values = (1, 2)

        def __getitem__(self, index):
            return self.values[index]

        def __len__(self):
            return len(self.values)

    class DuckMapping:
        def items(self):
            return {"answer": 42}.items()

    assert _freeze_evidence(
        DuckSequence(), where="snapshot.duck_sequence"
    ) == (1, 2)
    assert _freeze_evidence(
        DuckMapping(), where="snapshot.duck_mapping"
    ) == {"answer": 42}


@pytest.mark.parametrize("protocol", (Mapping, Sequence))
def test_static_protocol_lookup_rejects_registry_only_pseudo_protocols(protocol):
    class RegistryOnly:
        pass

    protocol.register(RegistryOnly)

    with pytest.raises(ConfigError, match="unsupported evidence leaf type"):
        _freeze_evidence(RegistryOnly(), where="snapshot.registry_only")


@pytest.mark.parametrize("protocol", ("mapping", "sequence"))
def test_static_protocol_lookup_rejects_noncallable_members(protocol):
    if protocol == "mapping":
        class NonCallable:
            items = None
    else:
        class NonCallable:
            __iter__ = None

            def __getitem__(self, index):
                return index

            def __len__(self):
                return 1

    with pytest.raises(ConfigError, match="unsupported evidence leaf type"):
        _freeze_evidence(NonCallable(), where="snapshot.noncallable")


@pytest.mark.parametrize("protocol", ("mapping", "sequence"))
def test_static_protocol_lookup_wraps_ordinary_traversal_failures(protocol):
    if protocol == "mapping":
        class Failing:
            def items(self):
                raise ValueError("private mapping failure")
    else:
        class Failing:
            def __iter__(self):
                raise ValueError("private sequence failure")

            def __getitem__(self, index):
                raise AssertionError("getitem fallback must not be used")

            def __len__(self):
                return 1

    with pytest.raises(ConfigError, match="evidence protocol failed") as caught:
        _freeze_evidence(Failing(), where="snapshot.failing_protocol")
    assert "private" not in str(caught.value)


@pytest.mark.parametrize("protocol", ("mapping", "sequence"))
def test_static_protocol_lookup_never_binds_hostile_member_descriptors(protocol):
    class Descriptor:
        calls = 0

        def __get__(self, instance, owner):
            type(self).calls += 1
            raise KeyboardInterrupt("descriptor secret")

    if protocol == "mapping":
        class Hostile:
            items = Descriptor()
    else:
        class Hostile:
            __iter__ = Descriptor()

            def __getitem__(self, index):
                return index

            def __len__(self):
                return 1

    with pytest.raises(ConfigError, match="unsupported evidence leaf type"):
        _freeze_evidence(Hostile(), where="snapshot.hostile_descriptor")
    assert Descriptor.calls == 0


def test_static_protocol_lookup_never_unwraps_hostile_classmethod_descriptors():
    class Descriptor:
        calls = 0

        def __get__(self, instance, owner):
            type(self).calls += 1
            raise KeyboardInterrupt("nested descriptor secret")

    class Hostile:
        items = classmethod(Descriptor())

    with pytest.raises(ConfigError, match="unsupported evidence leaf type"):
        _freeze_evidence(Hostile(), where="snapshot.hostile_classmethod")
    assert Descriptor.calls == 0


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


@pytest.mark.parametrize(
    "source",
    (
        "\ud800",
        {"\ud800": "value"},
        {"key": "\ud800"},
    ),
)
def test_freeze_evidence_rejects_non_utf8_string_scalars_and_keys(source):
    with pytest.raises(ConfigError, match="UTF-8"):
        _freeze_evidence(source, where="snapshot.document")


def test_freeze_evidence_preserves_valid_unicode_text():
    assert _freeze_evidence(
        {"café": "U0001f40d"}, where="snapshot.document"
    ) == {"café": "U0001f40d"}


def test_freeze_evidence_validates_each_shared_string_identity_once(monkeypatch):
    shared_key = "".join(("shared", "-key"))
    shared_value = "x" * 4096
    source = tuple(
        _ItemsMapping(((shared_key, shared_value),)) for _ in range(128)
    )
    helper = getattr(frozen_module, "_require_utf8_text", None)
    assert helper is not None, "shared UTF-8 validation helper is missing"
    calls = {id(shared_key): 0, id(shared_value): 0}

    def count(value, *, where):
        if value is shared_key or value is shared_value:
            calls[id(value)] += 1
        return helper(value, where=where)

    monkeypatch.setattr(frozen_module, "_require_utf8_text", count)
    frozen = _freeze_evidence(source, where="snapshot.document")

    assert calls == {id(shared_key): 1, id(shared_value): 1}
    assert all(item[shared_key] is shared_value for item in frozen)


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


@pytest.mark.parametrize("route", ("entry", "mapping_key", "leaf", "budget", "depth"))
def test_freeze_evidence_diagnostics_never_call_metaclass_name_descriptors(
    route, monkeypatch
):
    descriptor_calls = 0

    class HostileMeta(type):
        @property
        def __name__(cls):
            nonlocal descriptor_calls
            descriptor_calls += 1
            return "ForgedName"

    class Hostile(metaclass=HostileMeta):
        pass

    hostile = Hostile()
    if route == "entry":
        source = hostile
    elif route == "mapping_key":
        source = _ItemsMapping(((hostile, 1),))
    elif route == "leaf":
        source = {"leaf": hostile}
    elif route == "budget":
        monkeypatch.setattr(frozen_module, "_EVIDENCE_NODE_LIMIT", 1)
        source = [hostile]
    else:
        monkeypatch.setattr(frozen_module, "_EVIDENCE_DEPTH_LIMIT", 1)
        source = [hostile]

    with pytest.raises(ConfigError) as caught:
        _freeze_evidence(source, where="snapshot.document")
    if "Hostile" not in str(caught.value):
        pytest.fail(f"static type name missing from {caught.value.args!r}")
    if descriptor_calls != 0:
        pytest.fail(f"metaclass __name__ descriptor ran {descriptor_calls} times")


@pytest.mark.parametrize("length", (256, 257))
def test_static_type_names_have_an_exact_bounded_diagnostic_limit(length):
    huge_type = type("H" * length, (), {})
    with pytest.raises(ConfigError) as caught:
        _freeze_evidence(huge_type(), where="snapshot.document")
    message = str(caught.value)
    if length == 256:
        assert "H" * length in message
    else:
        assert "unknown" in message
        assert len(message) < 200


def test_static_type_name_rejects_non_utf8_class_metadata(monkeypatch):
    class NonUtf8Name:
        def __get__(self, instance, owner):
            return "\ud800"

    monkeypatch.setitem(
        frozen_module._TYPE_TEXT_DESCRIPTORS,
        "__name__",
        NonUtf8Name(),
    )
    with pytest.raises(ConfigError) as caught:
        _freeze_evidence(object(), where="snapshot.document")
    assert "unknown" in str(caught.value)


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


def _frozen_concat_type():
    implementation = getattr(frozen_module, "_FrozenConcat", None)
    assert implementation is not None, "persistent frozen sequences are missing"
    return implementation


def test_frozen_concat_matches_tuple_sequence_semantics_without_recursion():
    concat_type = _frozen_concat_type()
    rows = concat_type((0, 1), ())
    assert rows.extend(()) is rows
    expected = [0, 1]
    for index in range(2_000):
        rows = rows.extend((index + 2,))
        expected.append(index + 2)

    assert len(rows) == len(expected)
    assert tuple(rows) == tuple(expected)
    assert rows[0] == expected[0]
    assert rows[-1] == expected[-1]
    assert rows[2:7] == tuple(expected[2:7])
    assert rows[::-1] == tuple(expected[::-1])
    with pytest.raises(AttributeError):
        rows._suffix = ()
    with pytest.raises(TypeError):
        rows[0] = 1
    with pytest.raises(TypeError):
        hash(rows)


def test_frozen_concat_indices_are_exact_and_do_not_run_hooks():
    concat_type = _frozen_concat_type()
    rows = concat_type((1,), (2,))

    class HostileIndex:
        def __index__(self):
            raise AssertionError("index hook must not run")

    with pytest.raises(TypeError, match="exact int or slice"):
        rows[HostileIndex()]
    with pytest.raises(TypeError, match="exact int or slice"):
        rows[True]
    with pytest.raises(TypeError, match="exact int or null"):
        rows[slice(HostileIndex(), None)]
    with pytest.raises(TypeError, match="exact int or null"):
        rows[slice(True, None)]


def test_frozen_concat_requires_exact_private_chunks_and_parents():
    concat_type = _frozen_concat_type()

    class TupleSubclass(tuple):
        pass

    class ConcatSubclass(concat_type):
        pass

    subclass = ConcatSubclass((), ())
    with pytest.raises(ConfigError, match="parent"):
        concat_type([], ())
    with pytest.raises(ConfigError, match="parent"):
        concat_type(TupleSubclass(), ())
    with pytest.raises(ConfigError, match="parent"):
        concat_type(subclass, ())
    with pytest.raises(ConfigError, match="suffix"):
        concat_type((), TupleSubclass())


def test_thaw_materializes_frozen_concat_as_an_alias_preserving_list():
    concat_type = _frozen_concat_type()
    shared = MappingProxyType({"values": (1,)})
    rows = concat_type((shared,), (shared,))

    mutable = thaw(MappingProxyType({"rows": rows, "again": rows}))

    assert type(mutable) is dict
    assert type(mutable["rows"]) is list
    assert mutable["rows"] is mutable["again"]
    assert mutable["rows"][0] is mutable["rows"][1]
    assert type(mutable["rows"][0]["values"]) is list


def test_public_evidence_freezing_materializes_frozen_concat_to_exact_tuple():
    concat_type = _frozen_concat_type()
    rows = concat_type((MappingProxyType({"a": (1,)}),), (2,))

    frozen = _freeze_evidence(rows, where="public sequence")

    assert type(frozen) is tuple
    assert type(frozen[0]) is MappingProxyType
    assert type(frozen[0]["a"]) is tuple
    assert frozen == ({"a": (1,)}, 2)


def test_frozen_concat_validation_survives_optimized_python():
    code = r'''
from _rheplicant_bootstrap.frozen import _FrozenConcat

def outcome(call):
    try:
        call()
    except Exception as exc:
        print(type(exc).__name__)
    else:
        print("accepted")

class TupleSubclass(tuple):
    pass

class ConcatSubclass(_FrozenConcat):
    pass

outcome(lambda: _FrozenConcat([], ()))
outcome(lambda: _FrozenConcat(TupleSubclass(), ()))
outcome(lambda: _FrozenConcat(ConcatSubclass((), ()), ()))
outcome(lambda: _FrozenConcat((), TupleSubclass()))
'''
    done = subprocess.run(
        [sys.executable, "-O", "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    assert done.stdout.splitlines() == [
        "ConfigError",
        "ConfigError",
        "ConfigError",
        "ConfigError",
    ]


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


def test_freeze_evidence_does_not_retain_a_foreign_recursion_cause():
    render_calls = 0

    class HostileRecursionError(RecursionError):
        def __str__(self):
            nonlocal render_calls
            render_calls += 1
            raise AssertionError("foreign recursion text must not run")

    def consume():
        raise HostileRecursionError("private recursion detail")

    with pytest.raises(ConfigError) as caught:
        frozen_module._freeze_evidence_roots(
            [True],
            where="snapshot.document",
            consume=consume,
        )

    assert caught.value.__cause__ is None
    rendered = "".join(traceback.format_exception(caught.value))
    assert "private recursion detail" not in rendered
    assert render_calls == 0


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


class _EphemeralChild(Sequence):
    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value

    def __len__(self):
        return 1

    def __getitem__(self, index):
        if index == 0:
            return self.value
        raise IndexError


class _GenerativeSequence(Sequence):
    def __init__(self, count):
        self.count = count
        self.emitted_ids = []
        self.built_sources = []

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        raise AssertionError("the declared iterator must be used")

    def __iter__(self):
        for index in range(self.count):
            child = _EphemeralChild(index)
            self.built_sources.append(child)
            self.emitted_ids.append(id(child))
            yield child


class _GenerativeItems:
    def __init__(self, count):
        self.count = count
        self.emitted_ids = []
        self.built_sources = []

    def __iter__(self):
        for index in range(self.count):
            child = _EphemeralChild(index)
            self.built_sources.append(child)
            self.emitted_ids.append(id(child))
            yield f"key_{index}", child


class _GenerativeMapping(Mapping):
    def __init__(self, count):
        self.generated = _GenerativeItems(count)

    def __len__(self):
        return self.generated.count

    def __iter__(self):
        return iter(())

    def __getitem__(self, key):
        raise KeyError(key)

    def items(self):
        return self.generated


@pytest.mark.parametrize("factory", [_GenerativeSequence, _GenerativeMapping])
def test_freeze_evidence_strongly_retains_generative_child_identities(
    factory, monkeypatch
):
    """Catches bare-id memo hits after a protocol releases an emitted child."""
    real_id = id
    def colliding_id(value):
        if isinstance(value, _EphemeralChild):
            return 7
        return real_id(value)

    monkeypatch.setattr(frozen_module, "id", colliding_id, raising=False)
    source = factory(12)

    frozen = _freeze_evidence(source, where="snapshot.document")

    values = frozen if isinstance(frozen, tuple) else tuple(frozen.values())
    assert values == tuple((index,) for index in range(12))
    emitted_ids = (
        source.emitted_ids
        if isinstance(source, _GenerativeSequence)
        else source.generated.emitted_ids
    )
    assert len(set(emitted_ids)) == 12


def test_freeze_evidence_strongly_retains_seen_scalar_subclasses(monkeypatch):
    """Catches allocator reuse undercounting the unique-node evidence budget."""
    class EphemeralInt(int):
        pass

    class Scalars(Sequence):
        def __init__(self):
            self.sources = []

        def __len__(self):
            return 4

        def __getitem__(self, index):
            if index >= 4:
                raise IndexError
            value = EphemeralInt(index)
            self.sources.append(value)
            return value

    real_id = id

    def colliding_id(value):
        if isinstance(value, EphemeralInt):
            return 11
        return real_id(value)

    monkeypatch.setattr(frozen_module, "id", colliding_id, raising=False)
    # The sequence plus four distinct scalar sources is five unique nodes,
    # even though the injected identity function collides for every scalar.
    monkeypatch.setattr(frozen_module, "_EVIDENCE_NODE_LIMIT", 4)
    source = Scalars()
    with pytest.raises(ConfigError, match=r"unique node count 5 exceeds limit 4"):
        _freeze_evidence(source, where="snapshot.document")
    assert len({real_id(value) for value in source.sources}) == 4


def test_freeze_evidence_scalar_cache_is_strong_under_identity_collisions(
    monkeypatch,
):
    class CollidingText(str):
        pass

    first = CollidingText("first")
    second = CollidingText("second")
    real_id = id

    def colliding_id(value):
        if value is first or value is second:
            return 17
        return real_id(value)

    monkeypatch.setattr(frozen_module, "id", colliding_id, raising=False)
    frozen = _freeze_evidence(
        (first, second, first, second),
        where="snapshot.document",
    )
    assert frozen == ("first", "second", "first", "second")
    assert frozen[0] is frozen[2]
    assert frozen[1] is frozen[3]


class _EphemeralMapping(Mapping):
    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value

    def __len__(self):
        return 1

    def __iter__(self):
        return iter((f"leaf_{self.value}",))

    def __getitem__(self, key):
        if key == f"leaf_{self.value}":
            return self.value
        raise KeyError(key)


class _GenerativeThawMapping(_GenerativeMapping):
    def items(self):
        self.thaw_sources = []
        for index in range(self.generated.count):
            child = _EphemeralMapping(index)
            self.thaw_sources.append(child)
            yield f"key_{index}", child


def test_thaw_strongly_retains_generative_mapping_identities(monkeypatch):
    real_id = id
    def colliding_id(value):
        if isinstance(value, _EphemeralMapping):
            return 13
        return real_id(value)

    monkeypatch.setattr(frozen_module, "id", colliding_id, raising=False)
    source = _GenerativeThawMapping(12)

    mutable = thaw(source)

    assert mutable == {
        f"key_{index}": {f"leaf_{index}": index} for index in range(12)
    }
    assert len({real_id(value) for value in source.thaw_sources}) == 12


class _ForgedCallbackConfigError(ConfigError):
    def __str__(self):
        raise AssertionError("callback exception text must not run")

    def __repr__(self):
        raise AssertionError("callback exception repr must not run")


class _FailingIterator:
    def __init__(self, marker, *, fail_iter=False, fail_next=False, bad_item=False):
        self.marker = marker
        self.fail_iter = fail_iter
        self.fail_next = fail_next
        self.bad_item = bad_item
        self.done = False

    def __iter__(self):
        if self.fail_iter:
            raise self.marker
        return self

    def __next__(self):
        if self.fail_next:
            raise self.marker
        if self.done:
            raise StopIteration
        self.done = True
        if self.bad_item:
            marker = self.marker

            class BrokenItem:
                def __iter__(self):
                    raise marker

            return BrokenItem()
        return "key", 1


class _ProtocolMapping(_ItemsMapping):
    def __init__(self, marker, seam):
        super().__init__([])
        self.marker = marker
        self.seam = seam

    def items(self):
        if self.seam == "items":
            raise self.marker
        return _FailingIterator(
            self.marker,
            fail_iter=self.seam == "iter",
            fail_next=self.seam == "next",
            bad_item=self.seam == "unpack",
        )


class _ProtocolSequence(Sequence):
    def __init__(self, marker, seam):
        self.marker = marker
        self.seam = seam

    def __len__(self):
        return 1

    def __getitem__(self, index):
        raise AssertionError("the declared iterator must be used")

    def __iter__(self):
        if self.seam == "iter":
            raise self.marker
        return _FailingIterator(self.marker, fail_next=True)


@pytest.mark.parametrize("seam", ["items", "iter", "next", "unpack"])
def test_freeze_evidence_replaces_callback_configerror_at_each_mapping_seam(seam):
    marker = _ForgedCallbackConfigError("private marker")

    with pytest.raises(ConfigError) as caught:
        _freeze_evidence(
            _ProtocolMapping(marker, seam), where="snapshot.document"
        )

    assert caught.value is not marker
    assert str(caught.value).startswith("snapshot.document: evidence protocol failed")


@pytest.mark.parametrize("seam", ["iter", "next"])
def test_freeze_evidence_replaces_callback_configerror_at_each_sequence_seam(seam):
    marker = _ForgedCallbackConfigError("private marker")

    with pytest.raises(ConfigError) as caught:
        _freeze_evidence(
            _ProtocolSequence(marker, seam), where="snapshot.document"
        )

    assert caught.value is not marker
    assert str(caught.value).startswith("snapshot.document: evidence protocol failed")


def test_sequence_node_budget_stops_protocol_consumption_without_materializing(
    monkeypatch,
):
    class Values(Sequence):
        def __init__(self):
            self.next_calls = 0

        def __len__(self):
            return 100

        def __getitem__(self, index):
            raise AssertionError("the declared iterator must be used")

        def __iter__(self):
            owner = self

            class Iterator:
                def __iter__(self):
                    return self

                def __next__(self):
                    owner.next_calls += 1
                    if owner.next_calls > 3:
                        raise AssertionError("sequence was consumed past refusal")
                    return owner.next_calls

            return Iterator()

    monkeypatch.setattr(frozen_module, "_EVIDENCE_NODE_LIMIT", 3)
    source = Values()

    with pytest.raises(ConfigError, match="unique node count 4 exceeds limit 3"):
        _freeze_evidence(source, where="snapshot.document")

    assert source.next_calls == 3


def test_mapping_node_budget_stops_protocol_consumption_without_materializing(
    monkeypatch,
):
    class Values(_ItemsMapping):
        def __init__(self):
            super().__init__([])
            self.next_calls = 0

        def items(self):
            owner = self

            class Iterator:
                def __iter__(self):
                    return self

                def __next__(self):
                    owner.next_calls += 1
                    if owner.next_calls > 2:
                        raise AssertionError("mapping was consumed past refusal")
                    index = owner.next_calls
                    return f"key_{index}", index

            return Iterator()

    monkeypatch.setattr(frozen_module, "_EVIDENCE_NODE_LIMIT", 3)
    source = Values()

    with pytest.raises(ConfigError, match="unique node count 4 exceeds limit 3"):
        _freeze_evidence(source, where="snapshot.document")

    assert source.next_calls == 2


class _RepeatedSequence(Sequence):
    def __init__(self, count, value=1):
        self.count = count
        self.value = value
        self.next_calls = 0

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        raise AssertionError("the declared iterator must be used")

    def __iter__(self):
        for _ in range(self.count):
            self.next_calls += 1
            yield self.value


class _RepeatedMapping(_ItemsMapping):
    def __init__(self, count):
        super().__init__([])
        self.count = count
        self.next_calls = 0

    def items(self):
        for index in range(self.count):
            self.next_calls += 1
            yield f"key_{index}", 1


@pytest.mark.parametrize("factory", [_RepeatedSequence, _RepeatedMapping])
def test_protocol_emission_budget_counts_repeated_scalars_incrementally(
    factory, monkeypatch
):
    monkeypatch.setattr(
        frozen_module, "_EVIDENCE_EDGE_LIMIT", 3, raising=False
    )
    source = factory(4)

    expected = (
        "snapshot.document: evidence protocol emission count 4 exceeds "
        f"limit 3 at type {type(source).__name__}."
    )
    with pytest.raises(ConfigError) as caught:
        _freeze_evidence(source, where="snapshot.document")

    assert str(caught.value) == expected
    assert source.next_calls == 4


def test_protocol_emission_budget_has_the_normative_default():
    assert frozen_module._EVIDENCE_EDGE_LIMIT == 250_000


def test_protocol_emission_budget_accepts_its_exact_bound_independently(
    monkeypatch,
):
    class EmptyMapping(_ItemsMapping):
        visits = 0

        def items(self):
            type(self).visits += 1
            return iter(())

    shared = EmptyMapping([])
    source = _RepeatedSequence(3, shared)
    monkeypatch.setattr(frozen_module, "_EVIDENCE_NODE_LIMIT", 2)
    monkeypatch.setattr(
        frozen_module, "_EVIDENCE_EDGE_LIMIT", 3, raising=False
    )

    frozen = _freeze_evidence(source, where="snapshot.document")

    assert EmptyMapping.visits == 1
    assert frozen[0] is frozen[1] is frozen[2]


@pytest.mark.parametrize("kind", ["sequence", "mapping"])
def test_infinite_repeated_emissions_are_bounded(kind):
    script = textwrap.dedent(
        """
        import sys
        import time
        from collections.abc import Mapping, Sequence

        from _rheplicant_bootstrap import frozen as frozen_module
        from _rheplicant_bootstrap.errors import ConfigError

        frozen_module._EVIDENCE_EDGE_LIMIT = 3
        frozen_module._EVIDENCE_NODE_LIMIT = 20

        class InfiniteSequence(Sequence):
            def __len__(self):
                return 1

            def __getitem__(self, index):
                raise AssertionError("iterator required")

            def __iter__(self):
                while True:
                    time.sleep(0.001)
                    yield 1

        class InfiniteMapping(Mapping):
            def __len__(self):
                return 1

            def __iter__(self):
                return iter(())

            def __getitem__(self, key):
                raise KeyError(key)

            def items(self):
                index = 0
                while True:
                    yield f"key_{index}", 1
                    index += 1

        source = InfiniteSequence() if sys.argv[1] == "sequence" else InfiniteMapping()
        try:
            frozen_module.freeze_evidence(source, where="snapshot.document")
        except ConfigError as error:
            expected = (
                "snapshot.document: evidence protocol emission count 4 "
                f"exceeds limit 3 at type {type(source).__name__}."
            )
            if str(error) != expected:
                raise SystemExit(2)
            raise SystemExit(0)
        raise SystemExit(3)
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", script, kind],
        check=False,
        capture_output=True,
        text=True,
        timeout=2,
    )

    assert completed.returncode == 0, completed.stderr


def test_mapping_edge_budget_precedes_unpacking_the_limit_plus_one_pair(
    monkeypatch,
):
    class BrokenPair:
        unpack_calls = 0

        def __iter__(self):
            type(self).unpack_calls += 1
            raise AssertionError("limit-plus-one pair must not be unpacked")

    class Values(_ItemsMapping):
        def __init__(self):
            super().__init__([])

        def items(self):
            yield "first", 1
            yield BrokenPair()

    monkeypatch.setattr(
        frozen_module, "_EVIDENCE_EDGE_LIMIT", 1, raising=False
    )

    with pytest.raises(ConfigError) as caught:
        _freeze_evidence(Values(), where="snapshot.document")

    assert str(caught.value) == (
        "snapshot.document: evidence protocol emission count 2 exceeds "
        "limit 1 at type Values."
    )
    assert BrokenPair.unpack_calls == 0


def test_mapping_next_failure_precedes_any_unemitted_edge_charge(monkeypatch):
    marker = _ForgedCallbackConfigError("private next marker")

    class Values(_ItemsMapping):
        def __init__(self):
            super().__init__([])

        def items(self):
            yield "first", 1
            raise marker

    monkeypatch.setattr(
        frozen_module, "_EVIDENCE_EDGE_LIMIT", 1, raising=False
    )

    with pytest.raises(ConfigError) as caught:
        _freeze_evidence(Values(), where="snapshot.document")

    assert id(caught.value) != id(marker)
    assert str(caught.value) == (
        "snapshot.document: evidence protocol failed at type Values."
    )


def test_cleanup_mutations_do_not_live_inside_assert_statements():
    root = Path(__file__).parents[2]
    violations = []
    for relative in (
        "src/_rheplicant_bootstrap/frozen.py",
        "src/_rheplicant_bootstrap/layering.py",
    ):
        tree = ast.parse((root / relative).read_text())
        for assertion in (
            node for node in ast.walk(tree) if isinstance(node, ast.Assert)
        ):
            if any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "pop"
                for node in ast.walk(assertion.test)
            ):
                violations.append((relative, assertion.lineno))

    assert violations == []

    expected_assignments = {
        "freeze_evidence": 2,
        "_detach_origin_tree": 1,
        "_merge_extends_compat": 1,
    }
    actual_assignments = {}
    for relative in (
        "src/_rheplicant_bootstrap/frozen.py",
        "src/_rheplicant_bootstrap/layering.py",
    ):
        tree = ast.parse((root / relative).read_text())
        for function in (
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name in expected_assignments
        ):
            actual_assignments[function.name] = sum(
                1
                for node in ast.walk(function)
                if isinstance(node, ast.Try)
                for statement in node.finalbody
                if isinstance(statement, ast.Assign)
                and isinstance(statement.value, ast.Call)
                and isinstance(statement.value.func, ast.Attribute)
                and statement.value.func.attr == "pop"
            )

    assert actual_assignments == expected_assignments
