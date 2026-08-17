"""The resources DAG: build order, extends:, identity, and the kind registry."""

from collections import defaultdict, namedtuple
from collections.abc import Mapping
from types import MappingProxyType

import jax.numpy as jnp
import pytest

from rheplicant.config import ConfigError
from rheplicant.config.context import ResolutionContext
from rheplicant.config.resources import (
    RESOURCE_KINDS,
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
