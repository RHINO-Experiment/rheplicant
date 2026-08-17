"""Document layering: variants are one-level deep-merge patches over the base."""

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
