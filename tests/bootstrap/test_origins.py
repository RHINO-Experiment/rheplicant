"""Origin-preserving merge behavior and deletion evidence."""

from __future__ import annotations

import pytest

from _rheplicant_bootstrap.errors import ConfigError
from _rheplicant_bootstrap.frozen import thaw
from _rheplicant_bootstrap.layering import (
    DeletionRecord,
    MergeResult,
    OriginNode,
    initial_merge,
    merge_with_origins,
    origins_at,
)
from _rheplicant_bootstrap.types import Origin


def test_append_replace_delete_and_origins_are_distinct():
    parent = initial_merge(
        {"kept": 1, "replaced": [1], "extended": [1], "gone": 2},
        origin=Origin("preset", "rhino_v1"),
    )
    result = merge_with_origins(
        parent,
        {"replaced": [2], "extended": {"append": [2]}, "~gone": None},
        origin=Origin("user"),
    )
    assert thaw(result.document) == {"kept": 1, "replaced": [2], "extended": [1, 2]}
    assert origins_at(result.origins, ("kept",)) == Origin("preset", "rhino_v1")
    assert origins_at(result.origins, ("replaced", 0)) == Origin("user")
    assert origins_at(result.origins, ("extended", 0)) == Origin("preset", "rhino_v1")
    assert origins_at(result.origins, ("extended", 1)) == Origin("user")
    assert result.deletions == (DeletionRecord(("gone",), Origin("user")),)
    assert isinstance(result.deletions, tuple)
    assert isinstance(result.deletions[0].path, tuple)


def test_nested_mapping_merge_keeps_container_and_untouched_child_origins():
    parent = initial_merge(
        {"section": {"kept": 1}}, origin=Origin("preset", "one")
    )
    result = merge_with_origins(
        parent, {"section": {"added": 2}}, origin=Origin("user")
    )
    assert origins_at(result.origins, ("section",)) == Origin("preset", "one")
    assert origins_at(result.origins, ("section", "kept")) == Origin("preset", "one")
    assert origins_at(result.origins, ("section", "added")) == Origin("user")


def test_non_null_deletion_is_refused_without_mutating_the_parent():
    parent = initial_merge({"kept": 1}, origin=Origin("user"))
    with pytest.raises(ConfigError, match="deletion value must be null"):
        merge_with_origins(parent, {"~kept": 1}, origin=Origin("user"))
    assert thaw(parent.document) == {"kept": 1}


@pytest.mark.parametrize(
    "patch",
    [
        {"items": {"append": "not-a-list"}},
        {"items": {"append": [2], "other": True}},
    ],
)
def test_malformed_append_is_refused(patch):
    parent = initial_merge({"items": [1]}, origin=Origin("preset", "one"))
    with pytest.raises(ConfigError, match="append"):
        merge_with_origins(parent, patch, origin=Origin("user"))


def test_origins_at_refuses_a_path_missing_from_the_origin_tree():
    result = initial_merge({"kept": 1}, origin=Origin("user"))
    with pytest.raises(ConfigError, match="origin path"):
        origins_at(result.origins, ("missing",))


def test_direct_origin_records_canonicalize_all_mappings_and_sequences():
    """Catches direct construction leaving mutable audit-record containers."""
    leaf = OriginNode(Origin("user"), {})
    children = {"value": leaf}
    root = OriginNode(None, children)
    path = ["value"]
    deletion = DeletionRecord(path, Origin("user"))
    deletions = [deletion]
    document = {"value": [bytearray(b"one")]}

    result = MergeResult(document, root, deletions)
    children.clear()
    path.append("late")
    deletions.clear()
    document["value"][0][:] = b"two"
    document["value"].append(b"late")

    assert tuple(root.children) == ("value",)
    assert deletion.path == ("value",)
    assert result.deletions == (deletion,)
    assert result.document["value"] == (b"one",)
    with pytest.raises(TypeError):
        root.children["late"] = leaf


def test_origin_merge_canonicalizes_buffers_and_refuses_unsupported_leaves():
    """Catches mutable arbitrary leaves leaking through origin-bearing evidence."""
    buffer = bytearray(b"one")
    result = initial_merge({"value": buffer}, origin=Origin("user"))
    buffer[:] = b"two"
    assert result.document["value"] == b"one"

    with pytest.raises(ConfigError, match="document"):
        initial_merge({"unsafe": object()}, origin=Origin("user"))
