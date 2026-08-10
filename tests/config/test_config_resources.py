"""The resources DAG: build order, extends:, identity, and the kind registry."""

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
