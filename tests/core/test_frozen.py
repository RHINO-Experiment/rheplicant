"""Tests for FrozenMapping: the immutable, hashable container behind State.meta."""

import jax.numpy as jnp
import pytest

from replicant.core.frozen import FrozenMapping


class TestConstruction:
    def test_empty(self):
        fm = FrozenMapping()
        assert len(fm) == 0

    def test_from_dict(self):
        fm = FrozenMapping({"telescope": "RHINO", "n_dish": 1})
        assert fm["telescope"] == "RHINO"
        assert fm["n_dish"] == 1

    def test_from_kwargs(self):
        fm = FrozenMapping(obs_id="demo-001")
        assert fm["obs_id"] == "demo-001"

    def test_from_pairs(self):
        fm = FrozenMapping([("a", 1), ("b", 2)])
        assert dict(fm) == {"a": 1, "b": 2}

    def test_idempotent(self):
        """FrozenMapping(FrozenMapping(...)) must be cheap and exact (converter re-runs)."""
        fm = FrozenMapping({"a": 1})
        assert FrozenMapping(fm) == fm

    def test_rejects_non_str_keys(self):
        with pytest.raises(TypeError, match="keys must be strings"):
            FrozenMapping({1: "a"})

    @pytest.mark.parametrize("bad", [[1, 2], {"nested": "dict"}, jnp.zeros(3)])
    def test_rejects_unhashable_values(self, bad):
        with pytest.raises(TypeError, match="hashable"):
            FrozenMapping({"k": bad})


class TestMappingSemantics:
    def test_iter_and_contains(self):
        fm = FrozenMapping({"a": 1, "b": 2})
        assert set(fm) == {"a", "b"}
        assert "a" in fm and "c" not in fm

    def test_get_with_default(self):
        fm = FrozenMapping({"a": 1})
        assert fm.get("missing", 42) == 42

    def test_missing_key_raises(self):
        with pytest.raises(KeyError):
            FrozenMapping()["missing"]


class TestHashEquality:
    def test_equal_content_equal_hash(self):
        a = FrozenMapping({"x": 1, "y": "s"})
        b = FrozenMapping({"y": "s", "x": 1})  # insertion order must not matter
        assert a == b
        assert hash(a) == hash(b)

    def test_unequal(self):
        assert FrozenMapping({"x": 1}) != FrozenMapping({"x": 2})
        assert FrozenMapping({"x": 1}) != {"x": 1}  # plain dict is not a FrozenMapping

    def test_usable_as_dict_key(self):
        d = {FrozenMapping({"a": 1}): "cached"}
        assert d[FrozenMapping({"a": 1})] == "cached"


class TestImmutability:
    def test_no_setitem(self):
        fm = FrozenMapping({"a": 1})
        with pytest.raises(TypeError):
            fm["a"] = 2  # type: ignore[index]

    def test_no_attribute_assignment(self):
        fm = FrozenMapping({"a": 1})
        with pytest.raises(AttributeError):
            fm.new_attr = 1  # type: ignore[attr-defined]


class TestFunctionalUpdates:
    def test_set_returns_new(self):
        fm = FrozenMapping({"a": 1})
        fm2 = fm.set(b=2)
        assert fm2 == FrozenMapping({"a": 1, "b": 2})
        assert fm == FrozenMapping({"a": 1})  # original untouched

    def test_remove_returns_new(self):
        fm = FrozenMapping({"a": 1, "b": 2})
        assert fm.remove("b") == FrozenMapping({"a": 1})
        assert "b" in fm

    def test_or_merge(self):
        fm = FrozenMapping({"a": 1}) | {"b": 2}
        assert isinstance(fm, FrozenMapping)
        assert dict(fm) == {"a": 1, "b": 2}

    def test_repr_shows_items(self):
        assert "a" in repr(FrozenMapping({"a": 1}))
