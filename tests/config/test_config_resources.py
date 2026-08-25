"""The resources DAG: build order, extends:, identity, and the kind registry."""

import subprocess
import sys
import textwrap
import weakref
from collections import UserDict, defaultdict, deque, namedtuple
from collections.abc import Mapping, MutableMapping
from functools import partial
from types import CellType, FunctionType, MappingProxyType, MethodType

import jax.numpy as jnp
import pytest

from rheplicant.config import ConfigError
from rheplicant.config.context import ResolutionContext
from rheplicant.config.resources import (
    RESOURCE_KINDS,
    _referenced_names,
    build_resources,
    check_unknown_keys,
    merge_extends,
)

_Pair = namedtuple("_Pair", ("left", "right"))


class _FancyTuple(tuple):
    pass


class _FancyStr(str):
    pass


class _HostileCallbackConfigError(ConfigError):
    def __str__(self):
        raise AssertionError("marker text must not run")

    def __repr__(self):
        raise AssertionError("marker repr must not run")


@pytest.fixture
def context():
    return ResolutionContext(
        freq=jnp.linspace(60e6, 85e6, 4), time=jnp.arange(8.0), dtype="float32"
    )


class TestBuildOrder:
    def test_an_entry_may_reference_one_declared_earlier(self, context):
        built = build_resources(
            {"arrays": {"base": {"list": [1.0, 2.0]},
                        "scaled": {"ref": "resources.arrays.base", "scale": 10.0}}},
            context,
        )
        scaled = built.resources["resources.arrays.scaled"]
        assert [float(v) for v in scaled] == pytest.approx([10.0, 20.0])

    def test_order_in_the_document_does_not_matter(self, context):
        """A mapping has an order and a reader should not have to know it."""
        built = build_resources(
            {"arrays": {"scaled": {"ref": "resources.arrays.base", "scale": 10.0},
                        "base": {"list": [1.0, 2.0]}}},
            context,
        )
        scaled = built.resources["resources.arrays.scaled"]
        assert [float(v) for v in scaled] == pytest.approx([10.0, 20.0])
        # The document declares 'scaled' first, but 'base' must still be
        # BUILT first -- the dependency graph decides order, not the
        # document's own key order.
        assert built.order == ("resources.arrays.base", "resources.arrays.scaled")

    def test_a_cycle_is_refused_and_the_loop_is_named(self, context):
        with pytest.raises(ConfigError) as excinfo:
            build_resources(
                {"arrays": {"a": {"ref": "resources.arrays.b"},
                            "b": {"ref": "resources.arrays.a"}}},
                context,
            )
        message = str(excinfo.value)
        assert "resources.arrays.a" in message
        assert "resources.arrays.b" in message

    def test_a_reference_to_an_undeclared_entry_is_refused(self, context):
        with pytest.raises(ConfigError) as excinfo:
            build_resources({"arrays": {"a": {"ref": "resources.arrays.absent"}}}, context)
        assert "absent" in str(excinfo.value)


class TestEachEntryIsBuiltOnce:
    def test_two_references_get_the_same_object(self, context):
        """radio/instrument/beam_spill.py:89 from_projector is 'the one call
        that cannot get the weight and the sky average out of step', and
        check B9 asks whether two projectors nominally sharing a beam actually
        share the array. Building each reference afresh passes every shape
        check and silently decouples them."""
        built = build_resources(
            {"arrays": {"shared": {"list": [1.0]},
                        "left": {"ref": "resources.arrays.shared"},
                        "right": {"ref": "resources.arrays.shared"}}},
            context,
        )
        assert built.resources["resources.arrays.left"] is built.resources["resources.arrays.right"]

    def test_the_shared_objects_map_records_it(self, context):
        """schema 2.1.6: config.resolved.yaml emits a shared_objects: map, so
        identity is visible in the artefact rather than only in the spec."""
        built = build_resources(
            {"arrays": {"shared": {"list": [1.0]},
                        "left": {"ref": "resources.arrays.shared"},
                        "right": {"ref": "resources.arrays.shared"}}},
            context,
        )
        groups = [set(group) for group in built.shared_objects]
        assert {"resources.arrays.shared", "resources.arrays.left",
                "resources.arrays.right"} in groups


class TestExtends:
    def test_public_merge_is_the_neutral_function_object(self):
        from _rheplicant_bootstrap.layering import merge_extends as neutral_merge

        assert merge_extends is neutral_merge

    def test_mappings_merge(self):
        """The child's z0 supplies only `value`, so `unit` must survive from
        the parent -- a child asserting on a key it also supplied would pass
        even if the merge silently replaced the whole `z0` mapping instead of
        merging it key by key."""
        merged = merge_extends({"kind": "termination", "z0": {"value": 75.0}},
                               {"kind": "termination", "termination": "open",
                                "z0": {"value": 50.0, "unit": "ohm"}})
        assert merged["termination"] == "open"
        assert merged["z0"]["value"] == 75.0
        assert merged["z0"]["unit"] == "ohm"

    def test_lists_replace_rather_than_merge(self):
        """schema §5 rule 4, and the reason one config holds a whole
        comparison: a split across files can silently disagree in exactly the
        keys the comparison is about."""
        merged = merge_extends({"optimizations": ["cache_beam_rotation"]},
                               {"optimizations": ["read_horizon_fraction", "x"]})
        assert merged["optimizations"] == ["cache_beam_rotation"]

    def test_append_extends_a_list(self):
        merged = merge_extends({"optimizations": {"append": ["cache_beam_rotation"]}},
                               {"optimizations": ["a"]})
        assert merged["optimizations"] == ["a", "cache_beam_rotation"]

    def test_append_preserves_an_inherited_self_cycle(self):
        inherited = []
        inherited.append(inherited)

        merged = merge_extends(
            {"items": {"append": ["tail"]}},
            {"items": inherited, "alias": inherited},
        )
        items = merged["items"]

        assert items is not inherited
        assert items is merged["alias"]
        assert items[0] is items
        assert items[1] == "tail"
        assert inherited == [inherited]

    def test_merge_does_not_alias_the_inputs(self):
        """{append: [...]} used to splice the child's list in BY REFERENCE, so
        mutating the merge result reached back into the caller's own child
        document. Mutate the merged structures and check both inputs are
        untouched."""
        child = {"optimizations": {"append": [{"note": "x"}]}}
        parent = {"optimizations": [{"note": "a"}]}
        merged = merge_extends(child, parent)

        merged["optimizations"][0]["note"] = "mutated"
        merged["optimizations"].append({"note": "extra"})

        assert parent["optimizations"] == [{"note": "a"}]
        assert child["optimizations"]["append"] == [{"note": "x"}]

    def test_nested_mapping_deepcopy_returning_self_is_not_mutated(self):
        class ReturningSelfMapping(MutableMapping):
            def __init__(self, values):
                self._data = dict(values)

            def __getitem__(self, key):
                return self._data[key]

            def __setitem__(self, key, value):
                self._data[key] = value

            def __delitem__(self, key):
                del self._data[key]

            def __iter__(self):
                return iter(self._data)

            def __len__(self):
                return len(self._data)

            def __deepcopy__(self, memo):
                memo[id(self)] = self
                return self

        shared = ReturningSelfMapping({"old": 1})

        with pytest.raises(ConfigError) as caught:
            merge_extends({"node": {"new": 2}}, {"node": shared})

        assert str(caught.value) == (
            "merge_extends: compatibility traversal or deepcopy failed."
        )
        assert dict(shared) == {"old": 1}

    def test_root_dict_deepcopy_returning_self_is_not_mutated(self):
        class ReturningSelfDict(dict):
            def __deepcopy__(self, memo):
                memo[id(self)] = self
                return self

        shared = ReturningSelfDict({"old": 1})

        with pytest.raises(ConfigError) as caught:
            merge_extends({"new": 2}, shared)

        assert str(caught.value) == (
            "merge_extends: compatibility traversal or deepcopy failed."
        )
        assert shared == {"old": 1}

    def test_appended_list_deepcopy_returning_self_is_not_mutated(self):
        class ReturningSelfList(list):
            def __deepcopy__(self, memo):
                memo[id(self)] = self
                return self

        shared = ReturningSelfList(["old"])

        with pytest.raises(ConfigError) as caught:
            merge_extends(
                {"items": {"append": ["new"]}}, {"items": shared}
            )

        assert str(caught.value) == (
            "merge_extends: compatibility traversal or deepcopy failed."
        )
        assert shared == ["old"]

    def test_mutated_custom_mapping_cannot_share_caller_owned_backing(self):
        class SharedBackingMapping(MutableMapping):
            def __init__(self, values):
                self._data = dict(values)

            def __getitem__(self, key):
                return self._data[key]

            def __setitem__(self, key, value):
                self._data[key] = value

            def __delitem__(self, key):
                del self._data[key]

            def __iter__(self):
                return iter(self._data)

            def __len__(self):
                return len(self._data)

            def __deepcopy__(self, memo):
                copied = type(self)({})
                memo[id(self)] = copied
                copied._data = self._data
                return copied

        shared = SharedBackingMapping({"old": 1})

        with pytest.raises(ConfigError) as caught:
            merge_extends({"node": {"new": 2}}, {"node": shared})

        assert str(caught.value) == (
            "merge_extends: compatibility traversal or deepcopy failed."
        )
        assert dict(shared) == {"old": 1}

    def test_mutated_custom_mapping_cannot_share_instance_dict_backing(self):
        class InstanceDictMapping(MutableMapping):
            def __getitem__(self, key):
                return object.__getattribute__(self, "__dict__")[key]

            def __setitem__(self, key, value):
                object.__getattribute__(self, "__dict__")[key] = value

            def __delitem__(self, key):
                del object.__getattribute__(self, "__dict__")[key]

            def __iter__(self):
                return iter(object.__getattribute__(self, "__dict__"))

            def __len__(self):
                return len(object.__getattribute__(self, "__dict__"))

            def __deepcopy__(self, memo):
                copied = object.__new__(type(self))
                memo[id(self)] = copied
                object.__setattr__(
                    copied,
                    "__dict__",
                    object.__getattribute__(self, "__dict__"),
                )
                return copied

        source = InstanceDictMapping()
        source["old"] = 1

        with pytest.raises(ConfigError) as caught:
            merge_extends({"node": {"new": 2}}, {"node": source})

        assert str(caught.value) == (
            "merge_extends: compatibility traversal or deepcopy failed."
        )
        assert dict(source) == {"old": 1}

    def test_mutated_custom_mapping_cannot_share_slotted_state(self):
        class SlotStore:
            __slots__ = ("old", "new")

            def __init__(self):
                self.old = 1

        class SlotMapping(MutableMapping):
            __slots__ = ("store",)

            def __init__(self, store):
                self.store = store

            def __getitem__(self, key):
                return getattr(self.store, key)

            def __setitem__(self, key, value):
                setattr(self.store, key, value)

            def __delitem__(self, key):
                delattr(self.store, key)

            def __iter__(self):
                yield "old"
                try:
                    object.__getattribute__(self.store, "new")
                except AttributeError:
                    return
                yield "new"

            def __len__(self):
                return sum(1 for _ in self)

            def __deepcopy__(self, memo):
                copied = type(self)(self.store)
                memo[id(self)] = copied
                return copied

        store = SlotStore()
        source = SlotMapping(store)

        with pytest.raises(ConfigError) as caught:
            merge_extends({"node": {"new": 2}}, {"node": source})

        assert str(caught.value) == (
            "merge_extends: compatibility traversal or deepcopy failed."
        )
        assert dict(source) == {"old": 1}

    @pytest.mark.parametrize(
        "backing_kind", ["bytearray", "memoryview", "new_memoryview"]
    )
    def test_mutated_custom_mapping_cannot_share_mutable_buffer(
        self, backing_kind
    ):
        class BufferMapping(MutableMapping):
            def __init__(self, backing):
                self.backing = backing

            def __getitem__(self, key):
                offset = 0 if key == "old" else 2
                if not self.backing[offset]:
                    raise KeyError(key)
                return self.backing[offset + 1]

            def __setitem__(self, key, value):
                offset = 0 if key == "old" else 2
                self.backing[offset] = 1
                self.backing[offset + 1] = value

            def __delitem__(self, key):
                offset = 0 if key == "old" else 2
                if not self.backing[offset]:
                    raise KeyError(key)
                self.backing[offset] = 0

            def __iter__(self):
                if self.backing[0]:
                    yield "old"
                if self.backing[2]:
                    yield "new"

            def __len__(self):
                return bool(self.backing[0]) + bool(self.backing[2])

            def __deepcopy__(self, memo):
                if backing_kind == "new_memoryview":
                    copied_backing = memoryview(self.backing.obj)
                else:
                    copied_backing = self.backing
                copied = type(self)(copied_backing)
                memo[id(self)] = copied
                return copied

        raw = bytearray((1, 1, 0, 0))
        backing = raw if backing_kind == "bytearray" else memoryview(raw)
        source = BufferMapping(backing)

        with pytest.raises(ConfigError) as caught:
            merge_extends({"node": {"new": 2}}, {"node": source})

        assert str(caught.value) == (
            "merge_extends: compatibility traversal or deepcopy failed."
        )
        assert raw == bytearray((1, 1, 0, 0))

    def test_function_leaf_retains_standard_deepcopy_semantics(self):
        class Box(MutableMapping):
            def __init__(self, values):
                self._data = dict(values)

            def __getitem__(self, key):
                return self._data[key]

            def __setitem__(self, key, value):
                self._data[key] = value

            def __delitem__(self, key):
                del self._data[key]

            def __iter__(self):
                return iter(self._data)

            def __len__(self):
                return len(self._data)

        def callback():
            return None

        callback.label = "benign"
        source = Box({"old": 1, "callback": callback})

        merged = merge_extends({"node": {"new": 2}}, {"node": source})

        assert type(merged["node"]) is Box
        assert dict(merged["node"]) == {
            "old": 1,
            "callback": callback,
            "new": 2,
        }
        assert merged["node"]["callback"] is callback
        assert dict(source) == {"old": 1, "callback": callback}

    def test_empty_function_state_backing_is_protected_before_first_write(
        self,
    ):
        def callback():
            return None

        class EmptyFunctionStateMapping(MutableMapping):
            def __init__(self, anchor, virtual_old=True):
                self.anchor = anchor
                self.virtual_old = virtual_old

            @property
            def backing(self):
                return self.anchor.__dict__

            def __getitem__(self, key):
                if key == "old" and self.virtual_old:
                    return 1
                return self.backing[key]

            def __setitem__(self, key, value):
                self.backing[key] = value

            def __delitem__(self, key):
                if key == "old" and self.virtual_old:
                    self.virtual_old = False
                    return
                del self.backing[key]

            def __iter__(self):
                if self.virtual_old:
                    yield "old"
                yield from self.backing

            def __len__(self):
                return self.virtual_old + len(self.backing)

            def __deepcopy__(self, memo):
                copied = type(self)(self.anchor, self.virtual_old)
                memo[id(self)] = copied
                return copied

        source = EmptyFunctionStateMapping(callback)

        with pytest.raises(ConfigError) as caught:
            merge_extends({"node": {"new": 2}}, {"node": source})

        assert str(caught.value) == (
            "merge_extends: compatibility traversal or deepcopy failed."
        )
        assert callback.__dict__ == {}
        assert dict(source) == {"old": 1}

    def test_range_leaf_retains_standard_deepcopy_semantics(self):
        class Box(MutableMapping):
            def __init__(self, values):
                self._data = dict(values)

            def __getitem__(self, key):
                return self._data[key]

            def __setitem__(self, key, value):
                self._data[key] = value

            def __delitem__(self, key):
                del self._data[key]

            def __iter__(self):
                return iter(self._data)

            def __len__(self):
                return len(self._data)

        window = range(5)
        source = Box({"old": 1, "window": window})

        merged = merge_extends({"node": {"new": 2}}, {"node": source})

        assert type(merged["node"]) is Box
        assert merged["node"]["window"] is window
        assert dict(source) == {"old": 1, "window": window}

    @pytest.mark.parametrize("leaf_kind", ["property", "weakref"])
    def test_referential_atomic_leaf_retains_standard_deepcopy_semantics(
        self, leaf_kind
    ):
        class Box(MutableMapping):
            def __init__(self, values):
                self._data = dict(values)

            def __getitem__(self, key):
                return self._data[key]

            def __setitem__(self, key, value):
                self._data[key] = value

            def __delitem__(self, key):
                del self._data[key]

            def __iter__(self):
                return iter(self._data)

            def __len__(self):
                return len(self._data)

        if leaf_kind == "property":
            leaf = property(lambda _: None)
        else:
            class SafeReferent:
                pass

            leaf = weakref.ref(SafeReferent)
        source = Box({"old": 1, "leaf": leaf})

        merged = merge_extends({"node": {"new": 2}}, {"node": source})

        assert merged["node"]["leaf"] is leaf
        assert dict(source) == {"old": 1, "leaf": leaf}

    def test_bound_method_leaf_retains_standard_deepcopy_semantics(self):
        class Receiver:
            def __init__(self):
                self.state = []

            def callback(self):
                return None

        class Box(MutableMapping):
            def __init__(self, values):
                self._data = dict(values)

            def __getitem__(self, key):
                return self._data[key]

            def __setitem__(self, key, value):
                self._data[key] = value

            def __delitem__(self, key):
                del self._data[key]

            def __iter__(self):
                return iter(self._data)

            def __len__(self):
                return len(self._data)

        receiver = Receiver()
        callback = receiver.callback
        source = Box({"old": 1, "callback": callback})

        merged = merge_extends({"node": {"new": 2}}, {"node": source})
        copied_callback = merged["node"]["callback"]

        assert type(copied_callback) is MethodType
        assert copied_callback is not callback
        assert copied_callback.__self__ is not receiver
        assert copied_callback.__self__.state is not receiver.state
        assert dict(source) == {"old": 1, "callback": callback}

    @pytest.mark.parametrize("rebind", [False, True])
    def test_mutated_custom_mapping_cannot_share_bound_method_state(
        self, rebind
    ):
        class Store:
            def __init__(self):
                self.data = {"old": 1}

            def operate(self, operation, key=None, value=None):
                if operation == "get":
                    return self.data[key]
                if operation == "set":
                    self.data[key] = value
                    return None
                if operation == "delete":
                    del self.data[key]
                    return None
                if operation == "keys":
                    return tuple(self.data)
                if operation == "length":
                    return len(self.data)
                raise AssertionError("unknown operation")

        class MethodMapping(MutableMapping):
            def __init__(self, callback):
                self.callback = callback

            def __getitem__(self, key):
                return self.callback("get", key)

            def __setitem__(self, key, value):
                self.callback("set", key, value)

            def __delitem__(self, key):
                self.callback("delete", key)

            def __iter__(self):
                return iter(self.callback("keys"))

            def __len__(self):
                return self.callback("length")

            def __deepcopy__(self, memo):
                if rebind:
                    callback = MethodType(
                        self.callback.__func__, self.callback.__self__
                    )
                else:
                    callback = self.callback
                copied = type(self)(callback)
                memo[id(self)] = copied
                return copied

        store = Store()
        source = MethodMapping(store.operate)

        with pytest.raises(ConfigError) as caught:
            merge_extends({"node": {"new": 2}}, {"node": source})

        assert str(caught.value) == (
            "merge_extends: compatibility traversal or deepcopy failed."
        )
        assert store.data == {"old": 1}

    def test_mutated_custom_mapping_cannot_share_deque_backing(self):
        class DequeMapping(MutableMapping):
            def __init__(self, backing):
                self.backing = backing

            def __getitem__(self, key):
                return dict(self.backing)[key]

            def __setitem__(self, key, value):
                self.__delitem__(key, missing_ok=True)
                self.backing.append((key, value))

            def __delitem__(self, key, missing_ok=False):
                retained = deque(
                    pair for pair in self.backing if pair[0] != key
                )
                if len(retained) == len(self.backing) and not missing_ok:
                    raise KeyError(key)
                self.backing.clear()
                self.backing.extend(retained)

            def __iter__(self):
                return (key for key, _ in self.backing)

            def __len__(self):
                return len(self.backing)

            def __deepcopy__(self, memo):
                copied = type(self)(self.backing)
                memo[id(self)] = copied
                return copied

        backing = deque((("old", 1),))
        source = DequeMapping(backing)

        with pytest.raises(ConfigError) as caught:
            merge_extends({"node": {"new": 2}}, {"node": source})

        assert str(caught.value) == (
            "merge_extends: compatibility traversal or deepcopy failed."
        )
        assert backing == deque((("old", 1),))

    def test_mutated_custom_mapping_cannot_share_builtin_method_state(self):
        class BuiltinMethodMapping(MutableMapping):
            def __init__(self, anchor):
                self.anchor = anchor

            @property
            def backing(self):
                return self.anchor.__self__[0]

            def __getitem__(self, key):
                return self.backing[key]

            def __setitem__(self, key, value):
                self.backing[key] = value

            def __delitem__(self, key):
                del self.backing[key]

            def __iter__(self):
                return iter(self.backing)

            def __len__(self):
                return len(self.backing)

            def __deepcopy__(self, memo):
                copied = type(self)(self.anchor)
                memo[id(self)] = copied
                return copied

        backing = [{"old": 1}]
        source = BuiltinMethodMapping(backing.append)

        with pytest.raises(ConfigError) as caught:
            merge_extends({"node": {"new": 2}}, {"node": source})

        assert str(caught.value) == (
            "merge_extends: compatibility traversal or deepcopy failed."
        )
        assert backing == [{"old": 1}]

    @pytest.mark.parametrize(
        "metadata_name", ["__doc__", "__module__", "__type_params__"]
    )
    def test_mutated_custom_mapping_cannot_share_function_metadata_state(
        self, metadata_name
    ):
        backing = {"old": 1}

        def anchor():
            return None

        metadata = (backing,) if metadata_name == "__type_params__" else backing
        setattr(anchor, metadata_name, metadata)

        def clone_function(callback):
            cloned = FunctionType(
                callback.__code__,
                callback.__globals__,
                callback.__name__,
                callback.__defaults__,
                callback.__closure__,
            )
            setattr(cloned, metadata_name, getattr(callback, metadata_name))
            return cloned

        class FunctionMetadataMapping(MutableMapping):
            def __init__(self, callback):
                self.callback = callback

            @property
            def backing(self):
                state = getattr(self.callback, metadata_name)
                return state[0] if metadata_name == "__type_params__" else state

            def __getitem__(self, key):
                return self.backing[key]

            def __setitem__(self, key, value):
                self.backing[key] = value

            def __delitem__(self, key):
                del self.backing[key]

            def __iter__(self):
                return iter(self.backing)

            def __len__(self):
                return len(self.backing)

            def __deepcopy__(self, memo):
                copied = type(self)(clone_function(self.callback))
                memo[id(self)] = copied
                return copied

        source = FunctionMetadataMapping(anchor)

        with pytest.raises(ConfigError) as caught:
            merge_extends({"node": {"new": 2}}, {"node": source})

        assert str(caught.value) == (
            "merge_extends: compatibility traversal or deepcopy failed."
        )
        assert backing == {"old": 1}

    def test_function_annotate_is_tracked_without_reading_annotations(
        self, monkeypatch
    ):
        from _rheplicant_bootstrap import layering as neutral_layering

        backing = {"old": 1}
        annotation_reads = []

        def annotate(_format):
            return backing

        def anchor():
            return None

        original_reader = (
            neutral_layering._compatibility_builtin_descriptor_value
        )

        def read_descriptor(value, owner, name):
            if value is anchor and owner is FunctionType:
                if name == "__annotate__":
                    return annotate
                if name == "__annotations__":
                    annotation_reads.append(name)
                    return {}
            return original_reader(value, owner, name)

        monkeypatch.setattr(
            neutral_layering,
            "_compatibility_builtin_descriptor_value",
            read_descriptor,
        )

        class AnnotateMapping(MutableMapping):
            def __init__(self, callback):
                self.callback = callback

            def __getitem__(self, key):
                return backing[key]

            def __setitem__(self, key, value):
                backing[key] = value

            def __delitem__(self, key):
                del backing[key]

            def __iter__(self):
                return iter(backing)

            def __len__(self):
                return len(backing)

            def __deepcopy__(self, memo):
                copied = type(self)(self.callback)
                memo[id(self)] = copied
                return copied

        source = AnnotateMapping(anchor)

        with pytest.raises(ConfigError) as caught:
            merge_extends({"node": {"new": 2}}, {"node": source})

        assert str(caught.value) == (
            "merge_extends: compatibility traversal or deepcopy failed."
        )
        assert annotation_reads == []
        assert backing == {"old": 1}

    def test_hidden_function_with_annotate_refuses_cached_annotations(
        self, monkeypatch
    ):
        from _rheplicant_bootstrap import layering as neutral_layering

        backing = {"old": 1}
        annotation_reads = []

        def annotate(_format):
            return {}

        def anchor():
            return None

        function_pairs = [(anchor, annotate)]

        original_reader = (
            neutral_layering._compatibility_builtin_descriptor_value
        )

        def read_descriptor(value, owner, name):
            if owner is FunctionType:
                for function, annotation_callback in function_pairs:
                    if value is function:
                        if name == "__annotate__":
                            return annotation_callback
                        if name == "__annotations__":
                            annotation_reads.append(name)
                            return {"state": backing}
            return original_reader(value, owner, name)

        monkeypatch.setattr(
            neutral_layering,
            "_compatibility_builtin_descriptor_value",
            read_descriptor,
        )

        class CachedAnnotationMapping(MutableMapping):
            def __init__(self, callback):
                self.callback = callback

            def __getitem__(self, key):
                return backing[key]

            def __setitem__(self, key, value):
                backing[key] = value

            def __delitem__(self, key):
                del backing[key]

            def __iter__(self):
                return iter(backing)

            def __len__(self):
                return len(backing)

            def __deepcopy__(self, memo):
                annotation_callback = FunctionType(
                    annotate.__code__,
                    annotate.__globals__,
                    annotate.__name__,
                    annotate.__defaults__,
                    annotate.__closure__,
                )
                callback = FunctionType(
                    self.callback.__code__,
                    self.callback.__globals__,
                    self.callback.__name__,
                    self.callback.__defaults__,
                    self.callback.__closure__,
                )
                function_pairs.append((callback, annotation_callback))
                copied = type(self)(callback)
                memo[id(self)] = copied
                return copied

        source = CachedAnnotationMapping(anchor)

        caught = None
        try:
            merge_extends({"node": {"new": 2}}, {"node": source})
        except ConfigError as error:
            caught = error

        assert backing == {"old": 1}
        assert caught is not None
        assert str(caught) == (
            "merge_extends: compatibility traversal or deepcopy failed."
        )
        assert annotation_reads == []

    @pytest.mark.skipif(
        "__annotate__" not in vars(FunctionType),
        reason="lazy function annotations require Python 3.14 or newer",
    )
    def test_public_lazy_annotation_function_does_not_run_callback(self):
        events = []
        namespace = {"events": events}
        exec(
            "def callback(value: events.append('called') or int):\n"
            "    return value\n",
            namespace,
        )
        callback = namespace["callback"]
        assert events == []

        merged = merge_extends({"callback": callback}, {})

        assert merged["callback"] is callback
        assert events == []

    @pytest.mark.skipif(
        "__annotate__" not in vars(FunctionType),
        reason="lazy function annotations require Python 3.14 or newer",
    )
    def test_hidden_function_annotate_state_is_refused_without_calling(self):
        backing = {"old": 1}
        events = []

        def annotate(format_code):
            events.append(format_code)
            return {"state": backing}

        def anchor():
            return None

        anchor.__annotate__ = annotate

        class AnnotateMapping(MutableMapping):
            def __init__(self, callback):
                self.callback = callback

            def __getitem__(self, key):
                return backing[key]

            def __setitem__(self, key, value):
                backing[key] = value

            def __delitem__(self, key):
                del backing[key]

            def __iter__(self):
                return iter(backing)

            def __len__(self):
                return len(backing)

            def __deepcopy__(self, memo):
                copied = type(self)(self.callback)
                memo[id(self)] = copied
                return copied

        source = AnnotateMapping(anchor)

        with pytest.raises(ConfigError) as caught:
            merge_extends({"node": {"new": 2}}, {"node": source})

        assert str(caught.value) == (
            "merge_extends: compatibility traversal or deepcopy failed."
        )
        assert events == []
        assert backing == {"old": 1}

    @pytest.mark.skipif(
        "__annotate__" not in vars(FunctionType),
        reason="lazy function annotations require Python 3.14 or newer",
    )
    def test_hidden_cached_lazy_annotations_are_refused(self):
        backing = {"old": 1}
        namespace = {"backing": backing}
        exec(
            "def anchor(value: backing):\n"
            "    return value\n",
            namespace,
        )
        anchor = namespace["anchor"]
        cached = anchor.__annotations__
        assert cached["value"] is backing
        assert anchor.__annotate__ is not None

        class CachedAnnotationMapping(MutableMapping):
            def __init__(self, callback):
                self.callback = callback

            @property
            def values(self):
                return self.callback.__annotations__["value"]

            def __getitem__(self, key):
                return self.values[key]

            def __setitem__(self, key, value):
                self.values[key] = value

            def __delitem__(self, key):
                del self.values[key]

            def __iter__(self):
                return iter(self.values)

            def __len__(self):
                return len(self.values)

            def __deepcopy__(self, memo):
                annotation_callback = self.callback.__annotate__
                cloned_annotation_callback = FunctionType(
                    annotation_callback.__code__,
                    annotation_callback.__globals__,
                    annotation_callback.__name__,
                    annotation_callback.__defaults__,
                    annotation_callback.__closure__,
                )
                callback = FunctionType(
                    self.callback.__code__,
                    self.callback.__globals__,
                    self.callback.__name__,
                    self.callback.__defaults__,
                    self.callback.__closure__,
                )
                callback.__annotations__ = self.callback.__annotations__
                callback.__annotate__ = cloned_annotation_callback
                copied = type(self)(callback)
                memo[id(self)] = copied
                return copied

        source = CachedAnnotationMapping(anchor)

        caught = None
        try:
            merge_extends({"node": {"new": 2}}, {"node": source})
        except ConfigError as error:
            caught = error

        assert backing == {"old": 1}
        assert caught is not None
        assert str(caught) == (
            "merge_extends: compatibility traversal or deepcopy failed."
        )

    @pytest.mark.parametrize("wrapper_kind", ["partial", "method"])
    def test_function_metadata_is_followed_through_wrappers(
        self, wrapper_kind
    ):
        backing = {"old": 1}

        def callback():
            return None

        callback.__doc__ = backing
        wrapper = (
            partial(callback)
            if wrapper_kind == "partial"
            else MethodType(callback, object())
        )

        def wrapped_function(value):
            return value.func if wrapper_kind == "partial" else value.__func__

        def clone_function(value):
            cloned = FunctionType(
                value.__code__,
                value.__globals__,
                value.__name__,
                value.__defaults__,
                value.__closure__,
            )
            cloned.__doc__ = value.__doc__
            return cloned

        class WrappedFunctionMapping(MutableMapping):
            def __init__(self, value):
                self.wrapper = value

            @property
            def backing(self):
                return wrapped_function(self.wrapper).__doc__

            def __getitem__(self, key):
                return self.backing[key]

            def __setitem__(self, key, value):
                self.backing[key] = value

            def __delitem__(self, key):
                del self.backing[key]

            def __iter__(self):
                return iter(self.backing)

            def __len__(self):
                return len(self.backing)

            def __deepcopy__(self, memo):
                function = clone_function(wrapped_function(self.wrapper))
                wrapper = (
                    partial(function)
                    if wrapper_kind == "partial"
                    else MethodType(function, object())
                )
                copied = type(self)(wrapper)
                memo[id(self)] = copied
                return copied

        source = WrappedFunctionMapping(wrapper)

        with pytest.raises(ConfigError) as caught:
            merge_extends({"node": {"new": 2}}, {"node": source})

        assert str(caught.value) == (
            "merge_extends: compatibility traversal or deepcopy failed."
        )
        assert backing == {"old": 1}

    def test_mutated_custom_mapping_cannot_share_function_closure_state(self):
        backing = {"old": 1}

        def anchor(operation, key=None, value=None):
            if operation == "get":
                return backing[key]
            if operation == "set":
                backing[key] = value
                return None
            if operation == "delete":
                del backing[key]
                return None
            if operation == "keys":
                return tuple(backing)
            if operation == "length":
                return len(backing)
            raise AssertionError("unknown operation")

        class ClosureMapping(MutableMapping):
            def __init__(self, callback):
                self.callback = callback

            def __getitem__(self, key):
                return self.callback("get", key)

            def __setitem__(self, key, value):
                self.callback("set", key, value)

            def __delitem__(self, key):
                self.callback("delete", key)

            def __iter__(self):
                return iter(self.callback("keys"))

            def __len__(self):
                return self.callback("length")

            def __deepcopy__(self, memo):
                copied = type(self)(self.callback)
                memo[id(self)] = copied
                return copied

        source = ClosureMapping(anchor)

        with pytest.raises(ConfigError) as caught:
            merge_extends({"node": {"new": 2}}, {"node": source})

        assert str(caught.value) == (
            "merge_extends: compatibility traversal or deepcopy failed."
        )
        assert backing == {"old": 1}

    def test_mutated_custom_mapping_cannot_share_closure_cell_state(self):
        def make_anchor():
            old_present = True
            old_value = 1
            new_present = False
            new_value = 0

            def anchor(operation, key=None, value=None):
                nonlocal old_present, old_value, new_present, new_value
                is_old = key == "old"
                if operation == "get":
                    present = old_present if is_old else new_present
                    if not present:
                        raise KeyError(key)
                    return old_value if is_old else new_value
                if operation == "set":
                    if is_old:
                        old_present = True
                        old_value = value
                    else:
                        new_present = True
                        new_value = value
                    return None
                if operation == "delete":
                    if is_old:
                        if not old_present:
                            raise KeyError(key)
                        old_present = False
                    else:
                        if not new_present:
                            raise KeyError(key)
                        new_present = False
                    return None
                if operation == "keys":
                    return (
                        *(("old",) if old_present else ()),
                        *(("new",) if new_present else ()),
                    )
                if operation == "length":
                    return old_present + new_present
                if operation == "snapshot":
                    return {
                        **({"old": old_value} if old_present else {}),
                        **({"new": new_value} if new_present else {}),
                    }
                raise AssertionError("unknown operation")

            return anchor

        class CellMapping(MutableMapping):
            def __init__(self, callback):
                self.callback = callback

            def __getitem__(self, key):
                return self.callback("get", key)

            def __setitem__(self, key, value):
                self.callback("set", key, value)

            def __delitem__(self, key):
                self.callback("delete", key)

            def __iter__(self):
                return iter(self.callback("keys"))

            def __len__(self):
                return self.callback("length")

            def __deepcopy__(self, memo):
                copied = type(self)(self.callback)
                memo[id(self)] = copied
                return copied

        anchor = make_anchor()
        source = CellMapping(anchor)

        with pytest.raises(ConfigError) as caught:
            merge_extends({"node": {"new": 2}}, {"node": source})

        assert str(caught.value) == (
            "merge_extends: compatibility traversal or deepcopy failed."
        )
        assert anchor("snapshot") == {"old": 1}

    def test_mutated_custom_mapping_cannot_share_cell_contents(self):
        def make_cell(value):
            def capture():
                return value

            cell = capture.__closure__[0]
            assert type(cell) is CellType
            return cell

        class CellBackedMapping(MutableMapping):
            def __init__(self, cell):
                self.cell = cell

            @property
            def backing(self):
                return self.cell.cell_contents

            def __getitem__(self, key):
                return self.backing[key]

            def __setitem__(self, key, value):
                self.backing[key] = value

            def __delitem__(self, key):
                del self.backing[key]

            def __iter__(self):
                return iter(self.backing)

            def __len__(self):
                return len(self.backing)

            def __deepcopy__(self, memo):
                copied = type(self)(make_cell(self.cell.cell_contents))
                memo[id(self)] = copied
                return copied

        backing = {"old": 1}
        source = CellBackedMapping(make_cell(backing))

        with pytest.raises(ConfigError) as caught:
            merge_extends({"node": {"new": 2}}, {"node": source})

        assert str(caught.value) == (
            "merge_extends: compatibility traversal or deepcopy failed."
        )
        assert backing == {"old": 1}

    @pytest.mark.parametrize(
        "capture_kind", ["default", "kwdefault", "annotation", "attribute"]
    )
    def test_mutated_custom_mapping_cannot_share_function_referents(
        self, capture_kind
    ):
        backing = {"old": 1}
        if capture_kind == "default":
            def anchor(value=backing):
                return value
        elif capture_kind == "kwdefault":
            def anchor(*, value=backing):
                return value
        else:
            def anchor():
                return None

            if capture_kind == "annotation":
                anchor.__annotations__["state"] = backing
            else:
                anchor.state = backing

        class FunctionMapping(MutableMapping):
            def __init__(self, callback):
                self.callback = callback

            @property
            def backing(self):
                if capture_kind == "annotation":
                    return self.callback.__annotations__["state"]
                if capture_kind == "attribute":
                    return self.callback.state
                return self.callback()

            def __getitem__(self, key):
                return self.backing[key]

            def __setitem__(self, key, value):
                self.backing[key] = value

            def __delitem__(self, key):
                del self.backing[key]

            def __iter__(self):
                return iter(self.backing)

            def __len__(self):
                return len(self.backing)

            def __deepcopy__(self, memo):
                copied = type(self)(self.callback)
                memo[id(self)] = copied
                return copied

        source = FunctionMapping(anchor)

        with pytest.raises(ConfigError) as caught:
            merge_extends({"node": {"new": 2}}, {"node": source})

        assert str(caught.value) == (
            "merge_extends: compatibility traversal or deepcopy failed."
        )
        assert backing == {"old": 1}

    @pytest.mark.parametrize(
        "state_kind", ["attributes", "kwdefaults", "annotations"]
    )
    def test_mutated_custom_mapping_cannot_share_function_state_mapping(
        self, state_kind
    ):
        if state_kind == "attributes":
            def anchor():
                return None

            anchor.__dict__["old"] = 1
            backing = anchor.__dict__
        elif state_kind == "kwdefaults":
            def anchor(*, old=1):
                return old

            backing = anchor.__kwdefaults__
        else:
            def anchor(old: 1):
                return old

            backing = anchor.__annotations__

        class FunctionStateMapping(MutableMapping):
            def __init__(self, callback):
                self.callback = callback

            @property
            def values(self):
                if state_kind == "attributes":
                    return self.callback.__dict__
                if state_kind == "kwdefaults":
                    return self.callback.__kwdefaults__
                return self.callback.__annotations__

            def __getitem__(self, key):
                return self.values[key]

            def __setitem__(self, key, value):
                self.values[key] = value

            def __delitem__(self, key):
                del self.values[key]

            def __iter__(self):
                return iter(self.values)

            def __len__(self):
                return len(self.values)

            def __deepcopy__(self, memo):
                copied = type(self)(self.callback)
                memo[id(self)] = copied
                return copied

        source = FunctionStateMapping(anchor)

        with pytest.raises(ConfigError) as caught:
            merge_extends({"node": {"new": 2}}, {"node": source})

        assert str(caught.value) == (
            "merge_extends: compatibility traversal or deepcopy failed."
        )
        assert backing == {"old": 1}

    @pytest.mark.parametrize("anchor_kind", ["property", "weakref"])
    def test_mutated_custom_mapping_cannot_share_referential_atomic_state(
        self, anchor_kind
    ):
        backing = {"old": 1}
        if anchor_kind == "property":
            def getter(_):
                return backing

            anchor = property(getter)

            def values():
                return anchor.fget(None)
        else:
            class Store:
                def __init__(self, data):
                    self.data = data

            store = Store(backing)
            anchor = weakref.ref(store)

            def values():
                return anchor().data

        class ReferentialMapping(MutableMapping):
            def __init__(self, reference):
                self.reference = reference

            def __getitem__(self, key):
                return values()[key]

            def __setitem__(self, key, value):
                values()[key] = value

            def __delitem__(self, key):
                del values()[key]

            def __iter__(self):
                return iter(values())

            def __len__(self):
                return len(values())

            def __deepcopy__(self, memo):
                copied = type(self)(self.reference)
                memo[id(self)] = copied
                return copied

        source = ReferentialMapping(anchor)

        with pytest.raises(ConfigError) as caught:
            merge_extends({"node": {"new": 2}}, {"node": source})

        assert str(caught.value) == (
            "merge_extends: compatibility traversal or deepcopy failed."
        )
        assert backing == {"old": 1}

    def test_mutated_custom_mapping_cannot_share_property_doc_state(self):
        backing = {"old": 1}
        anchor = property(doc=backing)

        class PropertyDocMapping(MutableMapping):
            def __init__(self, descriptor):
                self.descriptor = descriptor

            @property
            def values(self):
                return self.descriptor.__doc__

            def __getitem__(self, key):
                return self.values[key]

            def __setitem__(self, key, value):
                self.values[key] = value

            def __delitem__(self, key):
                del self.values[key]

            def __iter__(self):
                return iter(self.values)

            def __len__(self):
                return len(self.values)

            def __deepcopy__(self, memo):
                copied = type(self)(self.descriptor)
                memo[id(self)] = copied
                return copied

        source = PropertyDocMapping(anchor)

        with pytest.raises(ConfigError) as caught:
            merge_extends({"node": {"new": 2}}, {"node": source})

        assert str(caught.value) == (
            "merge_extends: compatibility traversal or deepcopy failed."
        )
        assert backing == {"old": 1}

    def test_property_state_scan_avoids_computed_descriptor_callbacks(self):
        class Getter:
            calls = 0
            __slots__ = ()

            def __call__(self, instance):
                return None

            def __getattribute__(self, name):
                if name == "__isabstractmethod__":
                    type(self).calls += 1
                    raise RuntimeError("private marker")
                return object.__getattribute__(self, name)

        class DescriptorMapping(MutableMapping):
            def __init__(self, data, descriptor):
                self.data = data
                self.descriptor = descriptor

            def __getitem__(self, key):
                return self.data[key]

            def __setitem__(self, key, value):
                self.data[key] = value

            def __delitem__(self, key):
                del self.data[key]

            def __iter__(self):
                return iter(self.data)

            def __len__(self):
                return len(self.data)

            def __deepcopy__(self, memo):
                copied = type(self)(dict(self.data), property(Getter()))
                memo[id(self)] = copied
                return copied

        source = DescriptorMapping({"old": 1}, property(Getter()))

        merged = merge_extends({"node": {"new": 2}}, {"node": source})

        assert dict(merged["node"]) == {"old": 1, "new": 2}
        assert dict(source) == {"old": 1}
        assert Getter.calls == 0

    def test_property_name_is_in_the_safe_metadata_allowlist(
        self, monkeypatch
    ):
        from _rheplicant_bootstrap import layering as neutral_layering

        backing = {"old": 1}
        descriptor = property()
        original_reader = (
            neutral_layering._compatibility_builtin_descriptor_value
        )
        name_reads = []

        def read_descriptor(value, owner, name):
            if value is descriptor and owner is property and name == "__name__":
                name_reads.append(name)
                return backing
            return original_reader(value, owner, name)

        monkeypatch.setattr(
            neutral_layering,
            "_compatibility_builtin_descriptor_value",
            read_descriptor,
        )

        class PropertyNameMapping(MutableMapping):
            def __init__(self, data, metadata):
                self.data = data
                self.metadata = metadata

            def __getitem__(self, key):
                return self.data[key]

            def __setitem__(self, key, value):
                self.data[key] = value

            def __delitem__(self, key):
                del self.data[key]

            def __iter__(self):
                return iter(self.data)

            def __len__(self):
                return len(self.data)

            def __deepcopy__(self, memo):
                copied = type(self)(dict(self.data), self.metadata)
                memo[id(self)] = copied
                return copied

        source = PropertyNameMapping({"old": 1}, descriptor)

        with pytest.raises(ConfigError) as caught:
            merge_extends({"node": {"new": 2}}, {"node": source})

        assert str(caught.value) == (
            "merge_extends: compatibility traversal or deepcopy failed."
        )
        assert name_reads
        assert backing == {"old": 1}

    @pytest.mark.skipif(
        "__name__" not in vars(property),
        reason="property.__name__ requires Python 3.13 or newer",
    )
    def test_mutated_custom_mapping_cannot_share_property_name_state(self):
        backing = {"old": 1}
        descriptor = property()
        descriptor.__name__ = backing

        class PropertyNameMapping(MutableMapping):
            def __init__(self, metadata):
                self.metadata = metadata

            @property
            def values(self):
                return self.metadata.__name__

            def __getitem__(self, key):
                return self.values[key]

            def __setitem__(self, key, value):
                self.values[key] = value

            def __delitem__(self, key):
                del self.values[key]

            def __iter__(self):
                return iter(self.values)

            def __len__(self):
                return len(self.values)

            def __deepcopy__(self, memo):
                copied = type(self)(self.metadata)
                memo[id(self)] = copied
                return copied

        source = PropertyNameMapping(descriptor)

        with pytest.raises(ConfigError) as caught:
            merge_extends({"node": {"new": 2}}, {"node": source})

        assert str(caught.value) == (
            "merge_extends: compatibility traversal or deepcopy failed."
        )
        assert backing == {"old": 1}

    def test_mutated_custom_mapping_cannot_share_builtin_metadata_state(
        self,
    ):
        backing = {"old": 1}
        anchor = "".join
        anchor.__module__ = backing

        class BuiltinMetadataMapping(MutableMapping):
            def __init__(self, callback):
                self.callback = callback

            @property
            def values(self):
                return self.callback.__module__

            def __getitem__(self, key):
                return self.values[key]

            def __setitem__(self, key, value):
                self.values[key] = value

            def __delitem__(self, key):
                del self.values[key]

            def __iter__(self):
                return iter(self.values)

            def __len__(self):
                return len(self.values)

            def __deepcopy__(self, memo):
                copied = type(self)(self.callback)
                memo[id(self)] = copied
                return copied

        source = BuiltinMetadataMapping(anchor)

        with pytest.raises(ConfigError) as caught:
            merge_extends({"node": {"new": 2}}, {"node": source})

        assert str(caught.value) == (
            "merge_extends: compatibility traversal or deepcopy failed."
        )
        assert backing == {"old": 1}

    def test_mutated_custom_mapping_cannot_share_weakref_subclass_state(
        self,
    ):
        class Store:
            def __init__(self):
                self.data = {"old": 1}

        class Reference(weakref.ref):
            def __deepcopy__(self, memo):
                copied = type(self)(self())
                memo[id(self)] = copied
                return copied

        store = Store()
        anchor = Reference(store)

        class ReferenceMapping(MutableMapping):
            def __init__(self, reference):
                self.reference = reference

            @property
            def values(self):
                return self.reference().data

            def __getitem__(self, key):
                return self.values[key]

            def __setitem__(self, key, value):
                self.values[key] = value

            def __delitem__(self, key):
                del self.values[key]

            def __iter__(self):
                return iter(self.values)

            def __len__(self):
                return len(self.values)

            def __deepcopy__(self, memo):
                copied = type(self)(self.reference.__deepcopy__(memo))
                memo[id(self)] = copied
                return copied

        source = ReferenceMapping(anchor)

        with pytest.raises(ConfigError) as caught:
            merge_extends({"node": {"new": 2}}, {"node": source})

        assert str(caught.value) == (
            "merge_extends: compatibility traversal or deepcopy failed."
        )
        assert store.data == {"old": 1}

    @pytest.mark.parametrize("callable_proxy", [False, True])
    def test_mutated_custom_mapping_refuses_hidden_weakref_proxy(
        self, callable_proxy
    ):
        class Store:
            def __init__(self):
                self.data = {"old": 1}

        class CallableStore(Store):
            def __call__(self):
                return None

        store = CallableStore() if callable_proxy else Store()

        class ProxyMapping(MutableMapping):
            def __init__(self, anchor):
                self.anchor = anchor

            def __getitem__(self, key):
                return self.anchor.data[key]

            def __setitem__(self, key, value):
                self.anchor.data[key] = value

            def __delitem__(self, key):
                del self.anchor.data[key]

            def __iter__(self):
                return iter(self.anchor.data)

            def __len__(self):
                return len(self.anchor.data)

            def __deepcopy__(self, memo):
                copied = type(self)(weakref.proxy(store, lambda _: None))
                memo[id(self)] = copied
                return copied

        source = ProxyMapping(weakref.proxy(store))

        with pytest.raises(ConfigError) as caught:
            merge_extends({"node": {"new": 2}}, {"node": source})

        assert str(caught.value) == (
            "merge_extends: compatibility traversal or deepcopy failed."
        )
        assert store.data == {"old": 1}

    @pytest.mark.parametrize("callable_proxy", [False, True])
    def test_public_weakref_proxy_retains_standard_deepcopy_semantics(
        self, callable_proxy
    ):
        class Referent:
            def __init__(self):
                self.state = []

        class CallableReferent(Referent):
            def __call__(self):
                return None

        class Box(MutableMapping):
            def __init__(self, values):
                self._data = dict(values)

            def __getitem__(self, key):
                return self._data[key]

            def __setitem__(self, key, value):
                self._data[key] = value

            def __delitem__(self, key):
                del self._data[key]

            def __iter__(self):
                return iter(self._data)

            def __len__(self):
                return len(self._data)

        referent = CallableReferent() if callable_proxy else Referent()
        proxy = weakref.proxy(referent)
        source = Box({"old": 1, "proxy": proxy})

        merged = merge_extends({"node": {"new": 2}}, {"node": source})
        copied_referent = merged["node"]["proxy"]

        assert type(copied_referent) is type(referent)
        assert copied_referent is not referent
        assert copied_referent.state is not referent.state
        assert dict(source) == {"old": 1, "proxy": proxy}

    def test_arbitrary_leaf_does_not_bind_custom_dict_descriptor(self):
        class Leaf:
            calls = 0

            @property
            def __dict__(self):
                type(self).calls += 1
                raise RuntimeError("private dict marker")

            def __deepcopy__(self, memo):
                copied = object.__new__(type(self))
                memo[id(self)] = copied
                return copied

        class Box(MutableMapping):
            def __init__(self, values):
                self._data = dict(values)

            def __getitem__(self, key):
                return self._data[key]

            def __setitem__(self, key, value):
                self._data[key] = value

            def __delitem__(self, key):
                del self._data[key]

            def __iter__(self):
                return iter(self._data)

            def __len__(self):
                return len(self._data)

        leaf = Leaf()
        source = Box({"old": 1, "leaf": leaf})

        merged = merge_extends({"node": {"new": 2}}, {"node": source})

        assert type(merged["node"]["leaf"]) is Leaf
        assert merged["node"]["leaf"] is not leaf
        assert Leaf.calls == 0
        assert dict(source) == {"old": 1, "leaf": leaf}

    @pytest.mark.parametrize("reuse_holder", [True, False])
    def test_mutated_custom_mapping_refuses_caller_owned_opaque_state(
        self, reuse_holder
    ):
        class Holder:
            def __init__(self, values):
                self.data = dict(values)

        class HolderMapping(MutableMapping):
            def __init__(self, holder):
                self.holder = holder

            def __getitem__(self, key):
                return self.holder.data[key]

            def __setitem__(self, key, value):
                self.holder.data[key] = value

            def __delitem__(self, key):
                del self.holder.data[key]

            def __iter__(self):
                return iter(self.holder.data)

            def __len__(self):
                return len(self.holder.data)

            def __deepcopy__(self, memo):
                if reuse_holder:
                    copied_holder = self.holder
                else:
                    copied_holder = Holder({})
                    copied_holder.data = self.holder.data
                copied = type(self)(copied_holder)
                memo[id(self)] = copied
                return copied

        holder = Holder({"old": 1})
        source = HolderMapping(holder)

        with pytest.raises(ConfigError) as caught:
            merge_extends({"node": {"new": 2}}, {"node": source})

        assert str(caught.value) == (
            "merge_extends: compatibility traversal or deepcopy failed."
        )
        assert holder.data == {"old": 1}

    def test_unique_custom_mapping_with_arbitrary_leaf_remains_supported(self):
        class Leaf:
            def __init__(self):
                self.state = {1: ["private state"]}

        class Box(MutableMapping):
            def __init__(self, values):
                self._data = dict(values)

            def __getitem__(self, key):
                return self._data[key]

            def __setitem__(self, key, value):
                self._data[key] = value

            def __delitem__(self, key):
                del self._data[key]

            def __iter__(self):
                return iter(self._data)

            def __len__(self):
                return len(self._data)

        leaf = Leaf()
        box = Box({"old": 1, "leaf": leaf})

        merged = merge_extends({"node": {"new": 2}}, {"node": box})
        copied = merged["node"]

        assert type(copied) is Box
        assert dict(copied) == {"old": 1, "leaf": copied["leaf"], "new": 2}
        assert type(copied["leaf"]) is Leaf
        assert copied["leaf"] is not leaf
        assert copied["leaf"].state is not leaf.state
        assert copied["leaf"].state == {1: ["private state"]}
        assert copied["leaf"].state[1] is not leaf.state[1]
        assert dict(box) == {"old": 1, "leaf": leaf}

    def test_public_leaf_deepcopy_returning_self_remains_supported(self):
        class Leaf:
            def __deepcopy__(self, memo):
                memo[id(self)] = self
                return self

        class Box(MutableMapping):
            def __init__(self, values):
                self._data = dict(values)

            def __getitem__(self, key):
                return self._data[key]

            def __setitem__(self, key, value):
                self._data[key] = value

            def __delitem__(self, key):
                del self._data[key]

            def __iter__(self):
                return iter(self._data)

            def __len__(self):
                return len(self._data)

        leaf = Leaf()
        source = Box({"old": 1, "leaf": leaf})

        merged = merge_extends({"node": {"new": 2}}, {"node": source})

        assert type(merged["node"]) is Box
        assert merged["node"]["leaf"] is leaf
        assert dict(source) == {"old": 1, "leaf": leaf}

    def test_instance_dict_public_leaf_returning_self_remains_supported(self):
        class Leaf:
            def __deepcopy__(self, memo):
                memo[id(self)] = self
                return self

        class InstanceDictMapping(MutableMapping):
            def __getitem__(self, key):
                return object.__getattribute__(self, "__dict__")[key]

            def __setitem__(self, key, value):
                object.__getattribute__(self, "__dict__")[key] = value

            def __delitem__(self, key):
                del object.__getattribute__(self, "__dict__")[key]

            def __iter__(self):
                return iter(object.__getattribute__(self, "__dict__"))

            def __len__(self):
                return len(object.__getattribute__(self, "__dict__"))

        leaf = Leaf()
        source = InstanceDictMapping()
        source["old"] = 1
        source["leaf"] = leaf

        merged = merge_extends({"node": {"new": 2}}, {"node": source})

        assert type(merged["node"]) is InstanceDictMapping
        assert merged["node"]["leaf"] is leaf
        assert dict(merged["node"]) == {
            "old": 1,
            "leaf": leaf,
            "new": 2,
        }
        assert dict(source) == {"old": 1, "leaf": leaf}

    @pytest.mark.parametrize("failure", ["config", "ordinary", "base"])
    def test_unique_custom_mapping_does_not_read_instance_class(
        self, failure
    ):
        if failure == "config":
            marker = _HostileCallbackConfigError("private class marker")
        elif failure == "ordinary":
            marker = RuntimeError("private class marker")
        else:
            class StopNow(BaseException):
                pass

            marker = StopNow("stop")

        class HostileClassMapping(MutableMapping):
            class_lookups = 0

            def __init__(self, values):
                self._data = dict(values)

            def __getattribute__(self, name):
                if name == "__class__":
                    type(self).class_lookups += 1
                    raise marker
                return object.__getattribute__(self, name)

            def __getitem__(self, key):
                return self._data[key]

            def __setitem__(self, key, value):
                self._data[key] = value

            def __delitem__(self, key):
                del self._data[key]

            def __iter__(self):
                return iter(self._data)

            def __len__(self):
                return len(self._data)

        source = HostileClassMapping({"old": 1})

        merged = merge_extends(
            {"node": {"new": 2}}, {"node": source}
        )

        assert dict(merged["node"]) == {"old": 1, "new": 2}
        assert dict(source) == {"old": 1}
        assert HostileClassMapping.class_lookups == 0

    def test_pure_replacement_may_honor_deepcopy_returning_self(self):
        class ReturningSelfList(list):
            def __deepcopy__(self, memo):
                memo[id(self)] = self
                return self

        shared = ReturningSelfList(["replacement"])

        merged = merge_extends({"items": shared}, {})

        assert merged["items"] is shared
        assert shared == ["replacement"]

    def test_frozen_mapping_replacements_keep_tuple_type_and_detach_leaves(self):
        """Catches the public merge converting tuples to lists or sharing leaves."""
        class MutableLeaf:
            def __init__(self, values):
                self.values = values

        parent_leaf = MutableLeaf(["parent"])
        child_leaf = MutableLeaf(["child"])
        parent = MappingProxyType({"parent_tuple": (parent_leaf,)})
        child = MappingProxyType({"child_tuple": (child_leaf,)})

        merged = merge_extends(child, parent)
        merged["parent_tuple"][0].values.append("changed")
        merged["child_tuple"][0].values.append("changed")

        assert isinstance(merged["parent_tuple"], tuple)
        assert isinstance(merged["child_tuple"], tuple)
        assert parent_leaf.values == ["parent"]
        assert child_leaf.values == ["child"]

    def test_deepcopy_keeps_an_immutable_tuple_object_identical(self):
        """Catches rebuilding tuples that standard deepcopy deliberately reuses."""
        immutable = (1, "two", b"three")

        merged = merge_extends({"value": immutable}, {})

        assert merged["value"] is immutable

    @pytest.mark.parametrize(
        "value",
        [
            _Pair("left", ["right"]),
            _FancyTuple(("left", ["right"])),
        ],
    )
    def test_deepcopy_preserves_namedtuple_and_tuple_subclasses(self, value):
        merged = merge_extends({"value": value}, {})
        copied = merged["value"]

        assert type(copied) is type(value)
        assert copied is not value
        copied[-1].append("changed")
        assert value[-1] == ["right"]

    def test_deepcopy_preserves_a_tuple_list_cycle(self):
        loop = []
        cycle = (loop,)
        loop.append(cycle)

        copied = merge_extends({"cycle": cycle}, {})["cycle"]

        assert copied is not cycle
        assert copied[0] is not loop
        assert copied[0][0] is copied

    def test_deepcopy_preserves_topology_through_nested_mapping_views(self):
        shared = ["value"]
        nested = (MappingProxyType({"shared": shared}), shared)

        copied = merge_extends({"nested": nested}, {})["nested"]

        assert isinstance(copied[0], dict)
        assert copied[0]["shared"] is copied[1]
        assert copied[1] is not shared

    def test_overlapping_cyclic_mapping_merge_never_leaks_recursion_error(self):
        parent = {}
        parent["loop"] = parent
        child = {}
        child["loop"] = child

        try:
            merged = merge_extends(child, parent)
        except ConfigError:
            return

        assert merged["loop"] is merged

    def test_one_deepcopy_context_preserves_aliases_across_both_roots(self):
        shared = {"leaf": [1]}
        parent = {"kept": shared, "items": [shared]}
        child = {
            "replacement": shared,
            "items": {"append": [shared]},
        }

        merged = merge_extends(child, parent)

        assert merged["kept"] is merged["replacement"]
        assert merged["kept"] is merged["items"][0]
        assert merged["kept"] is merged["items"][1]
        assert merged["kept"] is not shared

    def test_nested_defaultdict_preserves_ordinary_deepcopy_semantics(self):
        nested = defaultdict(list, {"value": [1]})

        copied = merge_extends({"nested": nested}, {})["nested"]

        assert isinstance(copied, defaultdict)
        assert copied.default_factory is list
        assert copied is not nested

    def test_recursively_patched_defaultdict_preserves_its_concrete_type(self):
        nested = defaultdict(list, {"old": 1})

        copied = merge_extends(
            {"nested": {"new": 2}}, {"nested": nested}
        )["nested"]

        assert isinstance(copied, defaultdict)
        assert copied.default_factory is list
        assert copied == {"old": 1, "new": 2}
        assert copied is not nested

    def test_recursively_patched_userdict_preserves_deepcopy_type_and_state(self):
        nested = UserDict({"old": 1})
        nested.extra = {"state": []}

        merged = merge_extends(
            {"left": {"new": 2}}, {"left": nested, "right": nested}
        )
        copied = merged["left"]
        untouched = merged["right"]

        assert type(copied) is UserDict
        assert type(untouched) is UserDict
        assert copied == {"old": 1, "new": 2}
        assert untouched == {"old": 1}
        assert copied is not untouched
        assert copied.data is not untouched.data
        assert copied.extra == {"state": []}
        assert copied.extra is untouched.extra
        assert copied.extra is not nested.extra
        assert copied is not nested
        assert nested == {"old": 1}

    def test_recursive_split_accepts_a_reconnected_dict_subclass_backref(self):
        class ReconnectingDict(dict):
            def __copy__(self):
                copied = type(self)(dict.items(self))
                copied.ref = copied
                return copied

        shared = ReconnectingDict({"old": 1})
        shared.ref = shared

        merged = merge_extends(
            {"left": {"new": 2}}, {"left": shared, "right": shared}
        )

        assert merged["left"] == {"old": 1, "new": 2}
        assert merged["right"] == {"old": 1}
        assert merged["left"].ref is merged["left"]
        assert merged["right"].ref is merged["right"]
        assert shared.ref is shared

    @pytest.mark.parametrize("kind", ["userdict", "dict_subclass"])
    def test_recursive_split_refuses_a_hidden_self_backref(self, kind):
        if kind == "userdict":
            shared = UserDict({"old": 1})
        else:
            class CopyingDict(dict):
                def __copy__(self):
                    copied = type(self)(dict.items(self))
                    copied.__dict__.update(self.__dict__)
                    return copied

            shared = CopyingDict({"old": 1})
        shared.ref = shared

        with pytest.raises(ConfigError) as caught:
            merge_extends(
                {"left": {"new": 2}}, {"left": shared, "right": shared}
            )

        assert str(caught.value) == (
            "merge_extends: compatibility traversal or deepcopy failed."
        )
        assert shared.ref is shared
        assert "new" not in shared

    @pytest.mark.parametrize("failure", ["config", "ordinary", "base"])
    def test_hidden_split_state_traversal_normalizes_only_exceptions(
        self, failure
    ):
        if failure == "config":
            marker = _HostileCallbackConfigError("private state marker")
        elif failure == "ordinary":
            class HostileError(Exception):
                def __str__(self):
                    raise AssertionError("marker text must not run")

                def __repr__(self):
                    raise AssertionError("marker repr must not run")

            marker = HostileError("private state marker")
        else:
            class StopNow(BaseException):
                pass

            marker = StopNow("stop")

        class FailingState(Mapping):
            def __len__(self):
                return 1

            def __iter__(self):
                raise marker

            def __getitem__(self, key):
                raise marker

            def items(self):
                raise marker

            def __deepcopy__(self, memo):
                copied = type(self)()
                memo[id(self)] = copied
                return copied

        shared = UserDict({"old": 1})
        shared.extra = FailingState()

        if failure == "base":
            with pytest.raises(BaseException) as caught:
                merge_extends(
                    {"left": {"new": 2}},
                    {"left": shared, "right": shared},
                )
            assert caught.value is marker
            return

        with pytest.raises(ConfigError) as caught:
            merge_extends(
                {"left": {"new": 2}},
                {"left": shared, "right": shared},
            )

        assert caught.value is not marker
        assert str(caught.value) == (
            "merge_extends: compatibility traversal or deepcopy failed."
        )

    def test_recursive_patch_never_mutates_a_custom_mapping_alias_backing(self):
        class Box(MutableMapping):
            def __init__(self, values):
                self._data = dict(values)
                self.state = {"shared": []}

            def __getitem__(self, key):
                return self._data[key]

            def __setitem__(self, key, value):
                self._data[key] = value

            def __delitem__(self, key):
                del self._data[key]

            def __iter__(self):
                return iter(self._data)

            def __len__(self):
                return len(self._data)

        shared = Box({"old": 1})
        with pytest.raises(ConfigError) as caught:
            merge_extends(
                {"left": {"new": 2}}, {"left": shared, "right": shared}
            )

        assert str(caught.value) == (
            "merge_extends: compatibility traversal or deepcopy failed."
        )
        assert dict(shared) == {"old": 1}

    def test_recursive_copy_protocol_lookup_failure_is_replaced_statically(self):
        marker = _HostileCallbackConfigError("private copy lookup marker")

        class HostileMeta(type(MutableMapping)):
            def __getattribute__(cls, name):
                if name == "__copy__":
                    raise marker
                return super().__getattribute__(name)

        class HostileMapping(MutableMapping, metaclass=HostileMeta):
            def __init__(self, values):
                self._data = dict(values)

            def __getitem__(self, key):
                return self._data[key]

            def __setitem__(self, key, value):
                self._data[key] = value

            def __delitem__(self, key):
                del self._data[key]

            def __iter__(self):
                return iter(self._data)

            def __len__(self):
                return len(self._data)

        shared = HostileMapping({"old": 1})

        with pytest.raises(ConfigError) as caught:
            merge_extends(
                {"left": {"new": 2}}, {"left": shared, "right": shared}
            )

        assert id(caught.value) != id(marker)
        assert str(caught.value) == (
            "merge_extends: compatibility traversal or deepcopy failed."
        )
        assert dict(shared) == {"old": 1}

    def test_recursive_split_refuses_a_custom_copy_with_shared_backing(self):
        class CustomBox(MutableMapping):
            mutations = 0

            def __init__(self, values):
                self._data = dict(values)

            def __getitem__(self, key):
                return self._data[key]

            def __setitem__(self, key, value):
                type(self).mutations += 1
                self._data[key] = value

            def __delitem__(self, key):
                del self._data[key]

            def __iter__(self):
                return iter(self._data)

            def __len__(self):
                return len(self._data)

            def __copy__(self):
                copied = type(self)({})
                copied._data = self._data
                return copied

        shared = CustomBox({"old": 1})

        with pytest.raises(ConfigError) as caught:
            merge_extends(
                {"left": {"new": 2}}, {"left": shared, "right": shared}
            )

        assert str(caught.value) == (
            "merge_extends: compatibility traversal or deepcopy failed."
        )
        assert dict(shared) == {"old": 1}
        assert CustomBox.mutations == 0

    def test_recursive_split_refuses_external_dict_subclass_storage(self):
        class ExternalStoreDict(dict):
            def __init__(self, values):
                dict.__init__(self)
                self.store = dict(values)

            def __getitem__(self, key):
                return self.store[key]

            def __iter__(self):
                return iter(self.store)

            def __len__(self):
                return len(self.store)

            def items(self):
                return self.store.items()

            def __copy__(self):
                copied = type(self)({})
                copied.store = self.store
                return copied

        shared = ExternalStoreDict({"old": 1})

        with pytest.raises(ConfigError) as caught:
            merge_extends(
                {"left": {"new": 2}}, {"left": shared, "right": shared}
            )

        assert str(caught.value) == (
            "merge_extends: compatibility traversal or deepcopy failed."
        )
        assert dict(shared.items()) == {"old": 1}

    @pytest.mark.parametrize("location", ["nested", "root"])
    def test_unique_external_dict_subclass_storage_is_refused(
        self, location
    ):
        class ExternalStoreDict(dict):
            def __init__(self, values):
                dict.__init__(self)
                self.store = dict(values)

            def __getitem__(self, key):
                return self.store[key]

            def __iter__(self):
                return iter(self.store)

            def __len__(self):
                return len(self.store)

            def items(self):
                return self.store.items()

        shared = ExternalStoreDict({"old": 1})

        with pytest.raises(ConfigError) as caught:
            if location == "nested":
                merge_extends({"node": {"new": 2}}, {"node": shared})
            else:
                merge_extends({"new": 2}, shared)

        assert str(caught.value) == (
            "merge_extends: compatibility traversal or deepcopy failed."
        )
        assert dict(shared.items()) == {"old": 1}

    def test_recursive_split_refuses_a_dict_getattribute_override(self):
        class AttributeDict(dict):
            lookup_calls = 0

            def __getattribute__(self, name):
                if name == "__getattribute__":
                    type(self).lookup_calls += 1
                return object.__getattribute__(self, name)

            def __copy__(self):
                return type(self)(dict.items(self))

        shared = AttributeDict({"old": 1})

        with pytest.raises(ConfigError) as caught:
            merge_extends(
                {"left": {"new": 2}}, {"left": shared, "right": shared}
            )

        assert str(caught.value) == (
            "merge_extends: compatibility traversal or deepcopy failed."
        )
        assert shared == {"old": 1}
        assert AttributeDict.lookup_calls == 0

    def test_recursive_split_refuses_hidden_reference_to_split_descendant(self):
        class AttrDict(dict):
            def __copy__(self):
                copied = type(self)(dict.items(self))
                copied.__dict__.update(self.__dict__)
                return copied

        nested = {"old": 1}
        shared = AttrDict({"nested": nested})
        shared.ref = nested

        with pytest.raises(ConfigError) as caught:
            merge_extends(
                {"left": {"nested": {"new": 2}}},
                {"left": shared, "right": shared},
            )

        assert str(caught.value) == (
            "merge_extends: compatibility traversal or deepcopy failed."
        )
        assert shared.ref is nested
        assert nested == {"old": 1}

    def test_recursive_patch_never_breaks_a_cycle_through_an_object_attribute(self):
        class Holder:
            pass

        shared = {"old": 1}
        holder = Holder()
        holder.ref = shared
        shared["holder"] = holder

        with pytest.raises(ConfigError) as caught:
            merge_extends(
                {"left": {"new": 2}}, {"left": shared, "right": shared}
            )

        assert str(caught.value) == (
            "merge_extends: compatibility traversal or deepcopy failed."
        )
        assert holder.ref is shared
        assert "new" not in shared

    def test_recursive_patch_mutates_a_unique_mapping_with_an_arbitrary_leaf(self):
        class Leaf:
            def __init__(self):
                self.state = []

        leaf = Leaf()
        parent = {"node": {"old": 1, "leaf": leaf}}

        merged = merge_extends({"node": {"new": 2}}, parent)
        node = merged["node"]

        assert node["old"] == 1
        assert node["new"] == 2
        assert type(node["leaf"]) is Leaf
        assert node["leaf"] is not leaf
        assert node["leaf"].state == []
        assert node["leaf"].state is not leaf.state
        assert parent == {"node": {"old": 1, "leaf": leaf}}

    def test_recursive_patch_preserves_a_unique_object_attribute_backref(self):
        class Holder:
            pass

        node = {"old": 1}
        holder = Holder()
        holder.ref = node
        node["holder"] = holder
        parent = {"node": node}

        merged = merge_extends({"node": {"new": 2}}, parent)
        copied = merged["node"]

        assert copied["holder"].ref is copied
        assert copied["new"] == 2
        assert holder.ref is node
        assert "new" not in node

    def test_recursive_patch_splits_nested_descendants_of_a_shared_parent(self):
        descendant = {"old": 1}
        shared = {"nested": descendant}
        parent = {"left": shared, "right": shared}

        merged = merge_extends(
            {"left": {"nested": {"new": 2}}}, parent
        )

        assert merged["left"] is not merged["right"]
        assert merged["left"]["nested"] == {"old": 1, "new": 2}
        assert merged["right"]["nested"] == {"old": 1}
        assert merged["left"]["nested"] is not merged["right"]["nested"]
        assert parent == {"left": shared, "right": shared}
        assert descendant == {"old": 1}

    def test_recursive_patch_keeps_a_nested_list_alias_during_append(self):
        items = [1]
        shared = {"items": items}
        parent = {"left": shared, "right": shared}

        merged = merge_extends(
            {"left": {"items": {"append": [2]}}}, parent
        )

        assert merged["left"]["items"] == [1, 2]
        assert merged["right"]["items"] == [1, 2]
        assert merged["left"]["items"] is merged["right"]["items"]
        assert parent == {"left": shared, "right": shared}
        assert items == [1]

    def test_recursive_patch_preserves_a_stateful_str_subclass_leaf(self):
        class StatefulStr(str):
            def __new__(cls, value):
                instance = str.__new__(cls, value)
                instance.state = []
                return instance

        leaf = StatefulStr("leaf")
        parent = {"node": {"old": 1, "leaf": leaf}}

        merged = merge_extends({"node": {"new": 2}}, parent)
        copied = merged["node"]["leaf"]

        assert type(copied) is StatefulStr
        assert copied == "leaf"
        assert copied is not leaf
        assert copied.state == []
        assert copied.state is not leaf.state

    def test_deepcopy_protocol_lookup_failure_is_replaced_statically(self):
        marker = _HostileCallbackConfigError("private deepcopy lookup marker")

        class HostileMeta(type(MutableMapping)):
            def __getattribute__(cls, name):
                if name == "__deepcopy__":
                    raise marker
                return super().__getattribute__(name)

        class HostileMapping(MutableMapping, metaclass=HostileMeta):
            def __init__(self, values):
                self._data = dict(values)

            def __getitem__(self, key):
                return self._data[key]

            def __setitem__(self, key, value):
                self._data[key] = value

            def __delitem__(self, key):
                del self._data[key]

            def __iter__(self):
                return iter(self._data)

            def __len__(self):
                return len(self._data)

        with pytest.raises(ConfigError) as caught:
            merge_extends({"node": HostileMapping({"old": 1})}, {})

        assert caught.value is not marker
        assert str(caught.value) == (
            "merge_extends: compatibility traversal or deepcopy failed."
        )

    @pytest.mark.parametrize("seam", ["discover", "reachability"])
    @pytest.mark.parametrize("failure", ["config", "ordinary", "base"])
    def test_compatibility_type_checks_do_not_run_metaclass_equality(
        self, seam, failure
    ):
        if failure == "config":
            marker = _HostileCallbackConfigError("private equality marker")
        elif failure == "ordinary":
            class HostileError(Exception):
                def __str__(self):
                    raise AssertionError("marker text must not run")

                def __repr__(self):
                    raise AssertionError("marker repr must not run")

            marker = HostileError("private equality marker")
        else:
            class StopNow(BaseException):
                pass

            marker = StopNow("stop")

        class HostileMeta(type):
            equality_calls = 0

            def __eq__(cls, other):
                cls.equality_calls += 1
                raise marker

            __hash__ = type.__hash__

        class HostileLeaf(metaclass=HostileMeta):
            pass

        leaf = HostileLeaf()
        if seam == "discover":
            merged = merge_extends({"value": leaf}, {})

            assert type(merged["value"]) is HostileLeaf
        else:
            from _rheplicant_bootstrap.layering import (
                _compatibility_reaches_mapping,
            )

            with pytest.raises(ConfigError) as caught:
                _compatibility_reaches_mapping([leaf], {})

            assert str(caught.value) == (
                "merge_extends: compatibility traversal or deepcopy failed."
            )
        assert HostileMeta.equality_calls == 0

    @pytest.mark.parametrize("hook", ["__eq__", "__hash__"])
    @pytest.mark.parametrize("failure", ["config", "ordinary", "base"])
    def test_virtual_fallback_does_not_bind_metaclass_descriptors(
        self, hook, failure
    ):
        from _rheplicant_bootstrap.layering import (
            _compatibility_has_mro_base,
        )

        if failure == "config":
            marker = _HostileCallbackConfigError("private descriptor marker")
        elif failure == "ordinary":
            marker = RuntimeError("private descriptor marker")
        else:
            class StopNow(BaseException):
                pass

            marker = StopNow("stop")

        class HostileDescriptor:
            calls = 0

            def __get__(self, instance, owner):
                self.calls += 1
                raise marker

        descriptor = HostileDescriptor()
        namespace = {hook: descriptor}
        if hook == "__eq__":
            namespace["__hash__"] = type.__hash__
        HostileMeta = type("HostileMeta", (type,), namespace)
        HostileLeaf = HostileMeta("HostileLeaf", (), {})

        assert not _compatibility_has_mro_base(
            HostileLeaf(), (dict, Mapping)
        )
        assert descriptor.calls == 0

    @pytest.mark.parametrize(
        "seam", ["merge_root", "recursive_root", "key", "append", "inherited"]
    )
    @pytest.mark.parametrize("failure", ["config", "ordinary", "base"])
    def test_compatibility_diagnostics_do_not_run_metaclass_name_lookup(
        self, seam, failure
    ):
        from _rheplicant_bootstrap.layering import recursive_update

        if failure == "config":
            marker = _HostileCallbackConfigError("private name marker")
        elif failure == "ordinary":
            marker = RuntimeError("private name marker")
        else:
            class StopNow(BaseException):
                pass

            marker = StopNow("stop")

        class HostileMeta(type):
            name_lookups = 0

            def __getattribute__(cls, name):
                if name == "__name__":
                    lookups = type.__getattribute__(cls, "name_lookups")
                    type.__setattr__(cls, "name_lookups", lookups + 1)
                    raise marker
                return type.__getattribute__(cls, name)

        class HostileValue(metaclass=HostileMeta):
            pass

        class KeyMapping(Mapping):
            def __getitem__(self, key):
                return 1

            def __iter__(self):
                return iter((value,))

            def __len__(self):
                return 1

            def items(self):
                return iter(((value, 1),))

        value = HostileValue()
        with pytest.raises(ConfigError) as caught:
            if seam == "merge_root":
                merge_extends(value, {})
            elif seam == "recursive_root":
                recursive_update({}, value)
            elif seam == "key":
                merge_extends(KeyMapping(), {})
            elif seam == "append":
                merge_extends({"items": {"append": value}}, {"items": []})
            else:
                merge_extends({"items": {"append": []}}, {"items": value})

        messages = {
            "merge_root": "merge_extends: child is a mapping; got HostileValue.",
            "recursive_root": (
                "recursive_update: patch is a mapping; got HostileValue."
            ),
            "key": "merge_extends: keys are strings; got HostileValue.",
            "append": "'items': append is a sequence; got HostileValue.",
            "inherited": (
                "'items' is extended with {append: ...} but the inherited "
                "value is HostileValue, not a list."
            ),
        }
        assert str(caught.value) == messages[seam]
        assert HostileMeta.name_lookups == 0

    @pytest.mark.parametrize("failure", ["config", "ordinary", "base"])
    def test_compatibility_type_name_normalizes_only_exceptions(
        self, failure
    ):
        from _rheplicant_bootstrap.layering import _compatibility_type_name

        if failure == "config":
            marker = _HostileCallbackConfigError("private name marker")
        elif failure == "ordinary":
            marker = RuntimeError("private name marker")
        else:
            class StopNow(BaseException):
                pass

            marker = StopNow("stop")

        class HostileDescriptor:
            def __get__(self, instance, owner):
                raise marker

        class HostileMeta(type):
            __name__ = HostileDescriptor()

        class HostileValue(metaclass=HostileMeta):
            pass

        if failure == "base":
            with pytest.raises(BaseException) as caught:
                _compatibility_type_name(HostileValue())
            assert caught.value is marker
            return

        with pytest.raises(ConfigError) as caught:
            _compatibility_type_name(HostileValue())

        assert caught.value is not marker
        assert str(caught.value) == (
            "merge_extends: compatibility traversal or deepcopy failed."
        )

    def test_recursive_patch_never_breaks_a_plain_dict_subclass_backref(self):
        class AttrDict(dict):
            pass

        shared = AttrDict({"old": 1})
        shared.ref = shared

        try:
            merged = merge_extends(
                {"left": {"new": 2}}, {"left": shared, "right": shared}
            )
        except ConfigError as caught:
            assert str(caught) == (
                "merge_extends: compatibility traversal or deepcopy failed."
            )
            assert shared.ref is shared
            assert "new" not in shared
            return

        left = merged["left"]
        right = merged["right"]
        assert left is not right
        assert left.ref is left
        assert right.ref is right
        assert shared.ref is shared
        assert "new" not in shared

    def test_recursive_patch_splits_a_plain_dict_subclass(self):
        class Plain(dict):
            pass

        shared = Plain({"old": 1})
        parent = {"left": shared, "right": shared}

        merged = merge_extends({"left": {"new": 2}}, parent)

        assert type(merged["left"]) is Plain
        assert type(merged["right"]) is Plain
        assert merged["left"] == {"old": 1, "new": 2}
        assert merged["right"] == {"old": 1}
        assert merged["left"] is not merged["right"]
        assert parent == {"left": shared, "right": shared}
        assert shared == {"old": 1}

    def test_virtual_mapping_with_a_benign_metaclass_remains_supported(self):
        class BenignMeta(type):
            pass

        class VirtualMapping(metaclass=BenignMeta):
            def __init__(self, values):
                self._data = dict(values)

            def __getitem__(self, key):
                return self._data[key]

            def __iter__(self):
                return iter(self._data)

            def __len__(self):
                return len(self._data)

            def items(self):
                return self._data.items()

        Mapping.register(VirtualMapping)
        child = VirtualMapping({"new": 2})

        merged = merge_extends(child, {"old": 1})

        assert merged == {"old": 1, "new": 2}
        assert child._data == {"new": 2}

    def test_recursive_patch_splits_one_occurrence_of_a_shared_parent(self):
        nested = {"leaf": []}
        shared = {"old": 1, "nested": nested}
        parent = {"left": shared, "right": shared}

        merged = merge_extends({"left": {"added": 2}}, parent)

        assert merged["left"] == {
            "old": 1,
            "nested": {"leaf": []},
            "added": 2,
        }
        assert merged["right"] == {"old": 1, "nested": {"leaf": []}}
        assert merged["left"] is not merged["right"]
        assert merged["left"]["nested"] is merged["right"]["nested"]
        assert merged["right"] is not shared

    @pytest.mark.parametrize("mutual", [False, True])
    def test_recursive_patch_never_silently_breaks_parent_cycles(self, mutual):
        node = {}
        if mutual:
            sibling = {"back": node}
            node["next"] = sibling
        else:
            node["self"] = node

        try:
            merged_node = merge_extends(
                {"node": {"added": 2}}, {"node": node}
            )["node"]
        except ConfigError:
            return

        if mutual:
            assert merged_node["next"]["back"] is merged_node
        else:
            assert merged_node["self"] is merged_node

    def test_recursive_patch_bypasses_detached_dict_subclass_hooks(self):
        marker = _HostileCallbackConfigError("private mutation marker")

        class HostileDict(dict):
            def __init__(self, *args, **kwargs):
                self.armed = False
                dict.__init__(self, *args, **kwargs)

            def __deepcopy__(self, memo):
                copied = type(self)()
                memo[id(self)] = copied
                dict.update(copied, dict.items(self))
                copied.armed = True
                return copied

            def __copy__(self):
                copied = type(self)()
                dict.update(copied, dict.items(self))
                copied.armed = True
                return copied

            def get(self, *args, **kwargs):
                if self.armed:
                    raise marker
                return dict.get(self, *args, **kwargs)

            def pop(self, *args, **kwargs):
                if self.armed:
                    raise marker
                return dict.pop(self, *args, **kwargs)

            def __setitem__(self, key, value):
                if self.armed:
                    raise marker
                return dict.__setitem__(self, key, value)

        copied = merge_extends(
            {"nested": {"new": 2}},
            {"nested": HostileDict({"old": 1})},
        )["nested"]

        assert type(copied) is HostileDict
        assert dict(copied) == {"old": 1, "new": 2}

    def test_nested_custom_mapping_uses_its_deepcopy_protocol(self):
        class CustomMapping(dict):
            calls = 0

            def __deepcopy__(self, memo):
                type(self).calls += 1
                copied = type(self)({"copied": True})
                memo[id(self)] = copied
                return copied

        nested = CustomMapping({"source": True})

        copied = merge_extends({"nested": nested}, {})["nested"]

        assert type(copied) is CustomMapping
        assert copied == {"copied": True}
        assert CustomMapping.calls == 1

    def test_callback_configerror_from_deepcopy_is_replaced_statically(self):
        marker = _HostileCallbackConfigError("private deepcopy marker")

        class FailingLeaf:
            def __deepcopy__(self, memo):
                raise marker

        with pytest.raises(ConfigError) as caught:
            merge_extends({"value": FailingLeaf()}, {})

        assert caught.value is not marker
        assert "merge_extends" in str(caught.value)

    @pytest.mark.parametrize("seam", ["items", "iter", "next", "unpack"])
    def test_callback_configerror_from_mapping_traversal_is_replaced_statically(
        self, seam
    ):
        marker = _HostileCallbackConfigError("private mapping marker")

        class FailingIterator:
            def __init__(self):
                self.done = False

            def __iter__(self):
                if seam == "iter":
                    raise marker
                return self

            def __next__(self):
                if seam == "next":
                    raise marker
                if self.done:
                    raise StopIteration
                self.done = True
                if seam == "unpack":
                    class BrokenItem:
                        def __iter__(self):
                            raise marker

                    return BrokenItem()
                return "value", 1

        class FailingMapping(Mapping):
            def __len__(self):
                return 1

            def __iter__(self):
                return iter(("value",))

            def __getitem__(self, key):
                return 1

            def items(self):
                if seam == "items":
                    raise marker
                return FailingIterator()

        with pytest.raises(ConfigError) as caught:
            merge_extends(FailingMapping(), {})

        assert caught.value is not marker
        assert "merge_extends" in str(caught.value)

    def test_nested_append_mapping_callback_configerror_is_replaced_statically(self):
        marker = _HostileCallbackConfigError("private append marker")

        class FailingAppend(Mapping):
            def __len__(self):
                return 1

            def __iter__(self):
                raise marker

            def __getitem__(self, key):
                raise marker

            def __contains__(self, key):
                raise marker

        with pytest.raises(ConfigError) as caught:
            merge_extends({"items": FailingAppend()}, {"items": []})

        assert caught.value is not marker
        assert "merge_extends" in str(caught.value)

    def test_compatibility_protocols_do_not_catch_baseexceptions(self):
        class StopNow(BaseException):
            pass

        class StoppingLeaf:
            def __deepcopy__(self, memo):
                raise StopNow

        class StoppingMapping(Mapping):
            def __len__(self):
                return 1

            def __iter__(self):
                raise StopNow

            def __getitem__(self, key):
                raise StopNow

            def items(self):
                raise StopNow

        with pytest.raises(StopNow):
            merge_extends({"value": StoppingLeaf()}, {})
        with pytest.raises(StopNow):
            merge_extends(StoppingMapping(), {})

    def test_compatibility_keys_are_canonicalized_before_use(self):
        class HostileKey(str):
            armed = False

            def startswith(self, *args, **kwargs):
                if self.armed:
                    raise AssertionError("startswith must not run")
                return str.startswith(self, *args, **kwargs)

            def __hash__(self):
                if self.armed:
                    raise AssertionError("hash must not run")
                return str.__hash__(self)

            def __eq__(self, other):
                if self.armed:
                    raise AssertionError("equality must not run")
                return str.__eq__(self, other)

            def __repr__(self):
                raise AssertionError("repr must not run")

        key = HostileKey("value")
        child = {key: 2}
        key.armed = True

        merged = merge_extends(child, {"kept": 1})

        assert merged == {"kept": 1, "value": 2}
        assert type(next(key for key in merged if key == "value")) is str

    def test_key_canonicalization_does_not_change_a_cross_role_value_alias(self):
        class StatefulStr(str):
            def __new__(cls, value):
                instance = str.__new__(cls, value)
                instance.state = []
                return instance

        shared = StatefulStr("shared")

        merged = merge_extends({shared: shared, "alias": shared}, {})
        copied_key = next(iter(merged))
        copied_value = merged["shared"]
        copied_alias = merged["alias"]

        assert type(copied_key) is str
        assert type(copied_value) is StatefulStr
        assert copied_value == "shared"
        assert copied_value is copied_alias
        assert copied_value is not shared
        assert copied_value.state == []
        assert copied_value.state is not shared.state

    @pytest.mark.parametrize("kind", ["list", "tuple", "set", "frozenset"])
    def test_compatibility_reachability_terminates_on_container_cycles(self, kind):
        script = textwrap.dedent(
            """
            import sys

            from _rheplicant_bootstrap.layering import (
                _compatibility_reaches_mapping,
            )

            class HashableList(list):
                __hash__ = object.__hash__

            kind = sys.argv[1]
            if kind == "list":
                cycle = []
                cycle.append(cycle)
            elif kind == "tuple":
                bridge = []
                cycle = (bridge,)
                bridge.append(cycle)
            elif kind == "set":
                bridge = HashableList()
                cycle = {bridge}
                bridge.append(cycle)
            else:
                bridge = HashableList()
                cycle = frozenset({bridge})
                bridge.append(cycle)

            print(_compatibility_reaches_mapping([cycle], {}))
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
        assert completed.stdout == "False\n"

    @pytest.mark.parametrize("kind", ["list", "tuple", "set", "frozenset"])
    def test_compatibility_reachability_identity_cache_handles_collisions(
        self, kind, monkeypatch
    ):
        from _rheplicant_bootstrap import layering as neutral_layering

        class HashableMapping(dict):
            __hash__ = object.__hash__

        target = HashableMapping()
        if kind == "list":
            empty, reaching = [], [target]
        elif kind == "tuple":
            empty, reaching = (), (target,)
        elif kind == "set":
            empty, reaching = set(), {target}
        else:
            empty, reaching = frozenset(), frozenset({target})
        real_id = id

        def colliding_id(value):
            if value is empty or value is reaching:
                return 7
            return real_id(value)

        monkeypatch.setattr(
            neutral_layering, "id", colliding_id, raising=False
        )

        assert neutral_layering._compatibility_reaches_mapping(
            [reaching, empty], target
        )

    @pytest.mark.parametrize("failure", ["config", "ordinary", "base"])
    def test_appended_traversal_normalizes_only_ordinary_exceptions(
        self, failure
    ):
        if failure == "config":
            marker = _HostileCallbackConfigError("private append marker")
        elif failure == "ordinary":
            class HostileError(Exception):
                def __str__(self):
                    raise AssertionError("marker text must not run")

                def __repr__(self):
                    raise AssertionError("marker repr must not run")

            marker = HostileError("private append marker")
        else:
            class StopNow(BaseException):
                pass

            marker = StopNow("stop")

        class ArmedList(list):
            def __init__(self, values=(), *, armed=False):
                list.__init__(self, values)
                self.armed = armed

            def __iter__(self):
                if self.armed:
                    raise marker
                return list.__iter__(self)

            def __deepcopy__(self, memo):
                copied = type(self)(list.__iter__(self), armed=True)
                memo[id(self)] = copied
                return copied

        inherited = ["old"]
        appended = ArmedList(["new"])

        if failure == "base":
            with pytest.raises(BaseException) as caught:
                merge_extends(
                    {"items": {"append": appended}}, {"items": inherited}
                )
            assert caught.value is marker
            return

        with pytest.raises(ConfigError) as caught:
            merge_extends(
                {"items": {"append": appended}}, {"items": inherited}
            )

        assert id(caught.value) != id(marker)
        assert str(caught.value) == (
            "merge_extends: compatibility traversal or deepcopy failed."
        )

    def test_append_does_not_reiterate_the_detached_inherited_list(self):
        marker = AssertionError("detached inherited list was iterated")

        class ArmedInherited(list):
            def __init__(self, values=(), *, armed=False):
                list.__init__(self, values)
                self.armed = armed

            def __iter__(self):
                if self.armed:
                    raise marker
                return list.__iter__(self)

            def __deepcopy__(self, memo):
                copied = type(self)(list.__iter__(self), armed=True)
                memo[id(self)] = copied
                return copied

        merged = merge_extends(
            {"items": {"append": ["new"]}},
            {"items": ArmedInherited(["old"])},
        )

        items = merged["items"]
        assert list.__len__(items) == 2
        assert list.__getitem__(items, 0) == "old"
        assert list.__getitem__(items, 1) == "new"

    @pytest.mark.parametrize("kind", ["list", "tuple", "set", "frozenset"])
    def test_public_merge_terminates_when_parent_contains_a_container_cycle(
        self, kind
    ):
        script = textwrap.dedent(
            """
            import sys

            from _rheplicant_bootstrap.errors import ConfigError
            from _rheplicant_bootstrap.layering import merge_extends

            class HashableList(list):
                __hash__ = object.__hash__

            kind = sys.argv[1]
            if kind == "list":
                cycle = []
                cycle.append(cycle)
            elif kind == "tuple":
                bridge = []
                cycle = (bridge,)
                bridge.append(cycle)
            elif kind == "set":
                bridge = HashableList()
                cycle = {bridge}
                bridge.append(cycle)
            else:
                bridge = HashableList()
                cycle = frozenset({bridge})
                bridge.append(cycle)

            try:
                result = merge_extends(
                    {"node": {"added": 2}},
                    {"node": {"cycle": cycle}},
                )
            except ConfigError:
                raise SystemExit(0)
            if result["node"]["added"] != 2:
                raise SystemExit(2)
            raise SystemExit(0)
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

    def test_optimized_python_preserves_cleanup_and_shared_graph_behavior(self):
        script = textwrap.dedent(
            """
            from _rheplicant_bootstrap.frozen import freeze_evidence
            from _rheplicant_bootstrap.layering import OriginNode, merge_extends
            from _rheplicant_bootstrap.types import Origin

            shared_mapping = {"leaf": [1]}
            shared_sequence = [shared_mapping]
            frozen = freeze_evidence(
                {
                    "mapping": [shared_mapping, shared_mapping],
                    "sequence": [shared_sequence, shared_sequence],
                },
                where="optimized",
            )
            if frozen["mapping"][0] is not frozen["mapping"][1]:
                raise SystemExit(2)
            if frozen["sequence"][0] is not frozen["sequence"][1]:
                raise SystemExit(3)

            leaf = OriginNode(Origin("user"), {})
            origins = OriginNode(None, {"left": leaf, "right": leaf})
            if origins.children["left"] is not origins.children["right"]:
                raise SystemExit(4)

            patch = {"new": 2}
            merged = merge_extends(
                {"left": patch, "right": patch},
                {"left": {"old": 1}, "right": {"old": 1}},
            )
            if merged != {
                "left": {"old": 1, "new": 2},
                "right": {"old": 1, "new": 2},
            }:
                raise SystemExit(5)
            print("shared-graphs-ok")
            """
        )
        completed = []
        for optimization in ((), ("-O",)):
            completed.append(
                subprocess.run(
                    [sys.executable, *optimization, "-c", script],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
            )

        assert [run.returncode for run in completed] == [0, 0], [
            run.stderr for run in completed
        ]
        assert [run.stdout for run in completed] == [
            "shared-graphs-ok\n",
            "shared-graphs-ok\n",
        ]

    def test_compatibility_refuses_keys_that_collide_after_canonicalization(self):
        class ItemsMapping(Mapping):
            def __len__(self):
                return 2

            def __iter__(self):
                return iter(("value", "value"))

            def __getitem__(self, key):
                raise KeyError(key)

            def items(self):
                return iter((("value", 1), (_FancyStr("value"), 2)))

        with pytest.raises(ConfigError, match="collide"):
            merge_extends(ItemsMapping(), {})

    def test_append_with_a_sibling_key_is_refused(self):
        """{append: [...], other: ...} used to fall through to the
        wholesale-replace branch (merged.get(key) is a list, not a dict, so
        the dict-merge branch does not match either), silently discarding the
        parent's list and storing the whole child dict in its place. Any
        sibling key alongside `append` must be refused, not routed around."""
        with pytest.raises(ConfigError) as excinfo:
            merge_extends({"optimizations": {"append": ["x"], "mode": "extra"}},
                          {"optimizations": ["a"]})
        message = str(excinfo.value)
        assert "optimizations" in message
        assert "mode" in message

    def test_appending_onto_a_non_list_is_refused(self):
        with pytest.raises(ConfigError) as excinfo:
            merge_extends({"z0": {"append": [1.0]}}, {"z0": {"value": 50.0}})
        assert "z0" in str(excinfo.value)

    def test_a_tilde_key_deletes(self):
        merged = merge_extends({"~apod_deg": None}, {"apod_deg": 1.0, "lmax": 8})
        assert "apod_deg" not in merged
        assert merged["lmax"] == 8

    def test_a_tilde_key_must_carry_null(self):
        with pytest.raises(ConfigError, match="deletion value must be null"):
            merge_extends({"~apod_deg": 1.0}, {"apod_deg": 1.0})

    def test_extending_a_sibling_of_another_kind_is_refused(self, context):
        with pytest.raises(ConfigError) as excinfo:
            build_resources(
                {"arrays": {"a": {"list": [1.0]}},
                 "bases": {"b": {"extends": "a", "time": {"kind": "legendre", "n_basis": 2}}}},
                context,
            )
        message = str(excinfo.value)
        assert "bases" in message
        assert "arrays" in message
        # With an empty kind registry, the "unknown kind(s)" refusal also
        # names both 'arrays' and 'bases' (they are both unknown), which
        # would satisfy the two asserts above for the wrong reason. Pin
        # wording that only the real cross-kind extends: refusal uses.
        assert "SAME kind" in message


class TestTheKindRegistry:
    def test_an_unknown_kind_is_refused_and_the_registered_ones_are_listed(self, context):
        with pytest.raises(ConfigError) as excinfo:
            build_resources({"antennas": {"a": {}}}, context)
        message = str(excinfo.value)
        assert "antennas" in message
        for kind in RESOURCE_KINDS:
            assert kind in message, kind
        # RESOURCE_KINDS now holds 'arrays' (Task 3), so the loop above is no
        # longer vacuous -- it already checks that the rendered list names
        # every registered kind. Pin it explicitly too, so a rendering bug
        # that happened to keep 'arrays' out of `message` while the loop
        # above (reading the same `message`) still passed would be caught.
        assert "'arrays'" in message

    def test_all_six_kinds_are_registered(self):
        assert set(RESOURCE_KINDS) == {
            "arrays", "beams", "sky_models", "projectors", "s_params", "bases"
        }


class TestTheDagWithAProbeKind:
    """A throwaway registered kind, so extends: and cycle behaviour can be
    exercised today without waiting on Tasks 3-8 for a real kind builder."""

    @pytest.fixture
    def probe_kind(self):
        from rheplicant.config.resources import _KINDS, register_kind

        @register_kind("probe")
        def _build(name, spec, context):
            return dict(spec)

        yield "probe"
        _KINDS.pop("probe")

    def test_a_child_declared_before_its_grandparent_still_sees_it(self, context, probe_kind):
        """CRITICAL bug this catches: a single pass over the document in
        declaration order merges `a` over `b`'s spec as written, and `a` is
        declared first here, before `b` has merged over `c`. A single-pass
        implementation loses `c`'s `base` key from the built result and
        leaks a stray `extends: 'c'` into the spec `probe`'s builder sees."""
        built = build_resources(
            {
                "probe": {
                    "a": {"extends": "b", "own": 1},
                    "b": {"extends": "c", "mid": 2},
                    "c": {"base": 3},
                }
            },
            context,
        )
        result = built.resources["resources.probe.a"]
        assert result["base"] == 3
        assert result["mid"] == 2
        assert result["own"] == 1
        assert "extends" not in result

    def test_an_extends_cycle_is_refused_and_the_loop_is_named(self, context, probe_kind):
        """CRITICAL bug this catches: a single pass over `specs.items()` only
        ever follows `extends:` one hop per entry, so `a extends b, b extends
        a` never revisits `a` and the cycle is silently accepted."""
        with pytest.raises(ConfigError) as excinfo:
            build_resources(
                {"probe": {"a": {"extends": "b"}, "b": {"extends": "a"}}},
                context,
            )
        message = str(excinfo.value)
        assert "resources.probe.a" in message
        assert "resources.probe.b" in message

    def test_a_direct_self_reference_is_refused_as_a_cycle(self, context, probe_kind):
        """The `if name != dotted` filter on dependencies used to hide a
        direct `{ref: resources.probe.a}` self-reference from the build
        graph entirely -- it was silently dropped as its own dependency
        instead of being caught by the loop-detection `_build` already has."""
        with pytest.raises(ConfigError) as excinfo:
            build_resources({"probe": {"a": {"ref": "resources.probe.a"}}}, context)
        message = str(excinfo.value)
        assert message.count("resources.probe.a") >= 2


class TestUnknownKeyHints:
    def test_a_hint_is_appended_for_the_key_it_names(self):
        with pytest.raises(ConfigError) as excinfo:
            check_unknown_keys(
                "resources.x.y", {"z0": 1.0}, frozenset({"kind"}),
                label="kind: probe", hints={"z0": "z0 lives elsewhere."},
            )
        message = str(excinfo.value)
        assert message.endswith("z0 lives elsewhere.")
        assert "does not take ['z0']" in message

    def test_a_note_is_appended_unconditionally(self):
        with pytest.raises(ConfigError, match=r"it takes \['kind'\]\. Always\."):
            check_unknown_keys(
                "resources.x.y", {"w": 1}, frozenset({"kind"}),
                label="kind: probe", note="Always.",
            )

    def test_no_unknown_keys_means_no_raise_even_with_hints(self):
        check_unknown_keys(
            "resources.x.y", {"kind": "probe"}, frozenset({"kind"}),
            label="kind: probe", note="Always.", hints={"z0": "..."},
        )


class TestAKindPrefixIsADependency:
    """``from_switch_order: {resource: resources.<kind>}`` reads every entry of
    that kind, and the build graph has to know it.

    A8.5. ``_referenced_names`` recorded a dependency only for a three-part
    ``resources.<kind>.<name>``; a two-part kind PREFIX contributed nothing,
    so every entry the stack would read was invisible to the graph that
    decides build order.

    The consequence was not subtle and not benign. Measured before the fix,
    on one document whose only variable was the order of a mapping's keys:

        stacked declared LAST   -> BUILT
        stacked declared FIRST  -> REFUSED, "resources.arrays has no entry
                                   for ['hot', 'cold']"

    naming two entries the document plainly declares.
    ``TestBuildOrder.test_order_in_the_document_does_not_matter`` states the
    invariant that broke -- "a mapping has an order and a reader should not
    have to know it" -- and the refusal blamed the document for it.

    **A trap for anyone re-measuring this.** Build each document with a FRESH
    ``ResolutionContext``. The environment object is shared by every
    dependency snapshot (see ``_build``), so two ``build_resources`` calls on
    one context make the second complain that a resource "was bound more than
    once" -- which is the probe re-binding, not the code under test. That
    artefact cost a diagnosis here.
    """

    @staticmethod
    def _context():
        return ResolutionContext(
            freq=jnp.linspace(60e6, 85e6, 4),
            time=jnp.arange(8.0),
            dtype="float32",
            switch_order=("hot", "cold"),
        )

    ENTRIES = {"hot": {"list": [1.0, 2.0]}, "cold": {"list": [3.0, 4.0]}}
    STACK = {"stacked": {"from_switch_order": {"resource": "resources.arrays"}}}

    @pytest.mark.parametrize(
        ("label", "section"),
        [
            ("stacked last", {"arrays": {**ENTRIES, **STACK}}),
            ("stacked first", {"arrays": {**STACK, **ENTRIES}}),
        ],
    )
    def test_the_document_builds_whichever_order_it_declares(self, label, section):
        built = build_resources(section, self._context())
        assert built.resources["resources.arrays.stacked"].shape == (2, 2), label
        # The stack is built AFTER both entries it reads, in both documents.
        order = list(built.order)
        assert order[-1] == "resources.arrays.stacked", (label, order)
        assert set(order[:-1]) == {"resources.arrays.hot", "resources.arrays.cold"}

    def test_the_scan_expands_a_kind_prefix_to_that_kinds_entries(self):
        declared = frozenset(
            {
                "resources.s_params.hot",
                "resources.s_params.cold",
                "resources.arrays.elsewhere",
            }
        )
        node = {"from_switch_order": {"resource": "resources.s_params", "part": "re"}}
        assert _referenced_names(node, declared) == {
            "resources.s_params.hot",
            "resources.s_params.cold",
        }

    def test_a_sub_value_of_an_entry_is_not_a_fourth_entry(self):
        """``resources.s_params.hot.calibration`` is an attribute OF ``hot``,
        not a sibling of it -- the same distinction ``from_switch_order``'s own
        ``"." not in ...`` filter makes when it builds ``available``. An
        expansion that took every ``startswith`` match would claim a
        dependency on a name no entry answers to."""
        declared = frozenset(
            {"resources.s_params.hot", "resources.s_params.hot.calibration"}
        )
        node = {"ref": "resources.s_params"}
        assert _referenced_names(node, declared) == {"resources.s_params.hot"}

    def test_a_trailing_dot_does_not_manufacture_a_resource_name(self):
        """``resources.s_params.`` is a spelling ``from_switch_order`` accepts
        -- it calls ``rstrip('.')`` on it. Before the fix ``split('.')`` gave
        three parts and the scan recorded the phantom ``resources.s_params.``,
        which matches no entry and which the missing-reference check below
        refuses as a resource the document does not declare. One layer
        accepted a spelling the other rejected.
        """
        declared = frozenset({"resources.s_params.hot", "resources.s_params.cold"})
        node = {"from_switch_order": {"resource": "resources.s_params."}}
        found = _referenced_names(node, declared)
        assert found == {"resources.s_params.hot", "resources.s_params.cold"}
        assert not any(name.endswith(".") for name in found)

    def test_without_a_declared_set_a_kind_prefix_still_contributes_nothing(self):
        """ANTI-VACUITY, and a compatibility statement in one assertion.

        ``declared`` defaults to empty, and with no set there is nothing to
        expand to -- which is exactly the behaviour this function had before.
        So the tests above are known to be exercising the NEW path rather than
        passing on something that was always true, and a caller that does not
        pass the set is not silently given a different answer.
        """
        node = {"from_switch_order": {"resource": "resources.s_params"}}
        assert _referenced_names(node) == set()
        assert _referenced_names(node, frozenset()) == set()

    def test_an_entry_that_stacks_its_own_kind_does_not_report_a_loop(self):
        """The expansion names the stacking entry itself, and a self-edge would
        be reported as "these entries reference each other in a loop" -- true
        of the expansion, false of the document. The entry cannot READ itself
        either way: the stack sees only what is already built.
        """
        built = build_resources(
            {"arrays": {**self.STACK, **self.ENTRIES}}, self._context()
        )
        assert "resources.arrays.stacked" in built.resources
