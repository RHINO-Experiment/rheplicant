"""Document layering: variants are one-level deep-merge patches over the base."""

from collections.abc import Mapping
from types import MappingProxyType

import pytest

from rheplicant.config import ConfigError
from rheplicant.config.layering import apply_variant, recursive_update


def test_public_layering_is_the_bootstrap_implementation():
    """Catches public wrappers drifting from the one neutral merge engine."""
    from _rheplicant_bootstrap import layering as neutral
    from rheplicant.config import layering as public
    from rheplicant.config import resources

    assert public.recursive_update is neutral.recursive_update
    assert public.apply_variant is neutral.apply_variant
    assert resources.merge_extends is neutral.merge_extends

BASE = {
    "schema_version": 1,
    "observation": {
        "switching": {"order": ["antenna", "ambient", "hot"], "dwell": 1},
    },
    "model": {
        "cal_loads": {"ambient": {"t_load": 300.0}, "hot": {"t_load": 400.0}},
    },
    "variants": {
        "seven": {
            "observation": {"switching": {"order": ["antenna", "ambient", "hot", "ns"]}},
            "model": {"cal_loads": {"ns": {"t_load": 1200.0}}},
        },
        "no_hot": {"model": {"cal_loads": {"~hot": None}}},
        "chains": {"variants": {"another": {}}},
        "rewrites_version": {"schema_version": 2},
        "not_a_mapping": [1, 2],
        "deletes_version": {"~schema_version": None},
        "deletes_variants": {"~variants": None},
    },
}


class TestRecursiveUpdate:
    def test_mappings_merge_and_lists_replace(self):
        merged = recursive_update(BASE, BASE["variants"]["seven"])
        order = merged["observation"]["switching"]["order"]
        assert order == ["antenna", "ambient", "hot", "ns"]      # list REPLACED
        assert merged["observation"]["switching"]["dwell"] == 1  # sibling SURVIVES
        loads = merged["model"]["cal_loads"]
        assert set(loads) == {"ambient", "hot", "ns"}            # mapping MERGED

    def test_tilde_deletes(self):
        merged = recursive_update(BASE, {"model": {"cal_loads": {"~hot": None}}})
        assert set(merged["model"]["cal_loads"]) == {"ambient"}

    def test_append_extends(self):
        base = {"runs": [1, 2]}
        merged = recursive_update(base, {"runs": {"append": [3]}})
        assert merged["runs"] == [1, 2, 3]

    def test_neither_input_is_mutated(self):
        before = repr(BASE)
        merged = recursive_update(BASE, BASE["variants"]["seven"])
        assert repr(BASE) == before
        assert merged is not BASE

    def test_frozen_mappings_keep_tuples_and_deepcopy_arbitrary_leaves(self):
        """Catches routing the public compatibility merge through evidence thawing."""
        class MutableLeaf:
            def __init__(self, values):
                self.values = values

        base_leaf = MutableLeaf(["base"])
        patch_leaf = MutableLeaf(["patch"])
        base = MappingProxyType({"kept": (base_leaf,)})
        patch = MappingProxyType({"added": (patch_leaf,)})

        merged = recursive_update(base, patch)
        merged["kept"][0].values.append("changed")
        merged["added"][0].values.append("changed")

        assert isinstance(merged["kept"], tuple)
        assert isinstance(merged["added"], tuple)
        assert base_leaf.values == ["base"]
        assert patch_leaf.values == ["patch"]

    def test_non_mapping_inputs_are_refused(self):
        with pytest.raises(ConfigError, match="mapping"):
            recursive_update([1], {"a": 1})
        with pytest.raises(ConfigError, match="mapping"):
            recursive_update({"a": 1}, [1])


class TestApplyVariant:
    def test_a_variant_applies_by_name(self):
        merged = apply_variant(BASE, "seven")
        assert "ns" in merged["model"]["cal_loads"]
        assert merged["schema_version"] == 1

    def test_an_unknown_name_is_refused_listing_the_declared_ones(self):
        with pytest.raises(ConfigError) as excinfo:
            apply_variant(BASE, "missing")
        message = str(excinfo.value)
        assert "missing" in message
        assert "seven" in message and "no_hot" in message

    def test_a_variant_may_not_declare_variants(self):
        with pytest.raises(ConfigError, match="one level deep"):
            apply_variant(BASE, "chains")

    def test_a_variant_may_not_rewrite_the_schema_version(self):
        with pytest.raises(ConfigError, match="schema_version"):
            apply_variant(BASE, "rewrites_version")

    def test_a_non_mapping_patch_is_refused(self):
        with pytest.raises(ConfigError, match="mapping"):
            apply_variant(BASE, "not_a_mapping")

    def test_a_document_with_no_variants_refuses_every_name(self):
        with pytest.raises(ConfigError, match="declares no variants"):
            apply_variant({"schema_version": 1}, "any")

    def test_the_tilde_shapes_of_the_guards_are_also_refused(self):
        """A guard that matches only the rewrite shape lets ~key delete what
        it protects (the 1B guard-matcher lesson)."""
        with pytest.raises(ConfigError, match="schema_version"):
            apply_variant(BASE, "deletes_version")
        with pytest.raises(ConfigError, match="one level deep"):
            apply_variant(BASE, "deletes_variants")

    @pytest.mark.parametrize(
        "seam",
        [
            "document_get",
            "variants_bool",
            "variants_contains",
            "variants_lookup",
            "variants_iter",
            "patch_contains",
        ],
    )
    def test_mapping_callback_configerrors_are_replaced_statically(self, seam):
        class CallbackMarker(ConfigError):
            def __str__(self):
                raise AssertionError("marker text must not run")

            def __repr__(self):
                raise AssertionError("marker repr must not run")

        marker = CallbackMarker("private variant marker")

        class CallbackMapping(Mapping):
            def __init__(self, role):
                self.role = role

            def __len__(self):
                if self.role == "variants_bool":
                    raise marker
                return 1

            def __iter__(self):
                if self.role == "variants_iter":
                    raise marker
                return iter(
                    ("added",)
                    if self.role in {"patch", "patch_contains"}
                    else ("one",)
                )

            def items(self):
                if self.role in {"patch", "patch_contains"}:
                    return iter((("added", 2),))
                return iter((("one", self["one"]),))

            def __contains__(self, key):
                if self.role in {"variants_contains", "patch_contains"}:
                    raise marker
                if self.role == "variants_iter":
                    return False
                return key == "one"

            def __getitem__(self, key):
                if self.role == "variants_lookup":
                    raise marker
                if self.role in {"patch", "patch_contains"}:
                    if key == "added":
                        return 2
                    raise KeyError(key)
                if self.role == "variants_patch":
                    return CallbackMapping("patch_contains")
                if self.role.startswith("variants"):
                    return CallbackMapping("patch")
                raise KeyError(key)

        class Document(dict):
            def get(self, key, default=None):
                if seam == "document_get":
                    raise marker
                return super().get(key, default)

        document = Document(
            kept=1,
            variants=CallbackMapping(
                "variants_patch" if seam == "patch_contains" else seam
            )
            if seam.startswith("variants") or seam == "patch_contains"
            else {"one": {"added": 2}}
        )

        try:
            merged = apply_variant(document, "one")
        except ConfigError as caught:
            assert caught is not marker
            assert "variant" in str(caught)
        else:
            assert set(merged) == {"kept", "variants", "added"}
            assert merged["kept"] == 1
            assert merged["added"] == 2
            assert dict(merged["variants"].items()) == {
                "one": {"added": 2}
            }

    def test_variant_name_is_canonicalized_before_mapping_operations(self):
        class HostileName(str):
            def __hash__(self):
                raise AssertionError("hash must not run")

            def __eq__(self, other):
                raise AssertionError("equality must not run")

            def __repr__(self):
                raise AssertionError("repr must not run")

        merged = apply_variant(BASE, HostileName("seven"))

        assert "ns" in merged["model"]["cal_loads"]

    def test_variant_rejects_a_nonstring_name_without_using_repr(self):
        class HostileName:
            def __repr__(self):
                raise AssertionError("repr must not run")

        with pytest.raises(ConfigError, match="HostileName"):
            apply_variant(BASE, HostileName())

    def test_variant_mapping_protocols_do_not_catch_baseexceptions(self):
        class StopNow(BaseException):
            pass

        class StoppingDocument(dict):
            def items(self):
                raise StopNow

        with pytest.raises(StopNow):
            apply_variant(StoppingDocument(BASE), "seven")

    def test_required_variant_mapping_failure_is_normalized_statically(self):
        class Marker(ConfigError):
            def __str__(self):
                raise AssertionError("marker text must not run")

            def __repr__(self):
                raise AssertionError("marker repr must not run")

        marker = Marker("private variant marker")

        class FailingDocument(Mapping):
            def __len__(self):
                return 1

            def __iter__(self):
                raise marker

            def __getitem__(self, key):
                raise marker

            def items(self):
                raise marker

        with pytest.raises(ConfigError) as caught:
            apply_variant(FailingDocument(), "seven")

        assert caught.value is not marker
        assert str(caught.value) == "apply_variant: mapping traversal failed."
