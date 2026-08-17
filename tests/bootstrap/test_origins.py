"""Origin-preserving merge behavior and deletion evidence."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

import pytest

from _rheplicant_bootstrap import layering as layering_module
from _rheplicant_bootstrap.errors import ConfigError
from _rheplicant_bootstrap.frozen import thaw
from _rheplicant_bootstrap.layering import (
    DeletionRecord,
    MergeResult,
    OriginNode,
    initial_merge,
    merge_extends,
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
    item = OriginNode(Origin("user"), {})
    leaf = OriginNode(Origin("user"), {0: item})
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


def test_merge_result_recanonicalizes_forged_deletion_and_origin_evidence():
    origin = Origin("user")
    forged_deletion = object.__new__(DeletionRecord)
    mutable_path = [_StatefulStr("value")]
    object.__setattr__(forged_deletion, "path", mutable_path)
    object.__setattr__(forged_deletion, "origin", origin)

    forged_origin = object.__new__(Origin)
    object.__setattr__(forged_origin, "kind", _StatefulStr("user"))
    object.__setattr__(forged_origin, "name", None)
    forged_leaf = object.__new__(OriginNode)
    object.__setattr__(forged_leaf, "origin", forged_origin)
    object.__setattr__(forged_leaf, "children", {})
    root = OriginNode(None, {"value": forged_leaf})

    result = MergeResult({"value": 1}, root, [forged_deletion])
    mutable_path.append("late")

    assert result.deletions[0].path == ("value",)
    assert type(result.deletions[0].path[0]) is str
    leaf_origin = result.origins.children["value"].origin
    assert leaf_origin is not None
    assert type(leaf_origin.kind) is str
    assert type(result.origins.children) is MappingProxyType


def test_origin_merge_canonicalizes_buffers_and_refuses_unsupported_leaves():
    """Catches mutable arbitrary leaves leaking through origin-bearing evidence."""
    buffer = bytearray(b"one")
    result = initial_merge({"value": buffer}, origin=Origin("user"))
    buffer[:] = b"two"
    assert result.document["value"] == b"one"

    with pytest.raises(ConfigError, match="document"):
        initial_merge({"unsafe": object()}, origin=Origin("user"))


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


class _ItemsMapping(Mapping):
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


def test_origin_records_canonicalize_origins_and_segments_to_exact_builtins():
    kind = _StatefulStr("preset")
    name = _StatefulStr("one")
    segment = _StatefulStr("value")
    index = _StatefulInt(0)
    item = OriginNode(Origin(kind, name), {})
    sequence = OriginNode(Origin(kind, name), {index: item})
    root = OriginNode(None, {segment: sequence})
    deletion = DeletionRecord([segment, index], Origin(kind, name))
    result = MergeResult({"value": [1]}, root, [deletion])

    root_segment = next(iter(result.origins.children))
    sequence_segment = next(iter(result.origins.children[root_segment].children))
    assert type(root_segment) is str
    assert type(sequence_segment) is int
    assert type(item.origin.kind) is str
    assert type(item.origin.name) is str
    assert tuple(type(part) for part in deletion.path) == (str, int)
    assert type(deletion.origin.kind) is str
    assert type(deletion.origin.name) is str


def test_origin_record_scalar_canonicalization_uses_only_base_operations():
    """Catches invoking hostile conversion, comparison, hash, split, or repr hooks."""
    origin = Origin(_HostileStr("preset"), _HostileStr("one"))
    item = OriginNode(origin, {})
    sequence = OriginNode(
        origin,
        _ItemsMapping([(_HostileInt(0), item)]),
    )
    root = OriginNode(
        None,
        _ItemsMapping([(_HostileStr("value"), sequence)]),
    )
    deletion = DeletionRecord(
        [_HostileStr("value"), _HostileInt(0)], origin
    )
    result = MergeResult({"value": [1]}, root, [deletion])

    root_key = next(iter(result.origins.children))
    index = next(iter(result.origins.children[root_key].children))
    assert type(root_key) is str
    assert type(index) is int
    assert type(item.origin.kind) is str
    assert type(item.origin.name) is str
    assert tuple(type(part) for part in deletion.path) == (str, int)


@pytest.mark.parametrize(
    ("factory", "needle"),
    [
        (lambda: OriginNode(object(), {}), "origin"),
        (lambda: DeletionRecord((), Origin("user")), "non-empty"),
        (lambda: DeletionRecord(("value",), object()), "origin"),
    ],
)
def test_origin_records_validate_origins_and_nonempty_deletion_paths(factory, needle):
    with pytest.raises(ConfigError, match=needle):
        factory()


def test_merge_entry_points_validate_origins_even_for_empty_documents():
    with pytest.raises(ConfigError, match="origin"):
        initial_merge({}, origin=object())

    parent = initial_merge({}, origin=Origin("user"))
    with pytest.raises(ConfigError, match="origin"):
        merge_with_origins(parent, {}, origin=object())


def test_merge_result_requires_an_exact_parallel_origin_tree():
    user = Origin("user")
    valid_leaf = OriginNode(user, {})
    valid_root = OriginNode(None, {"value": valid_leaf})
    result = MergeResult({"value": 1}, valid_root, ())
    assert result.origins is valid_root

    invalid_roots = (
        OriginNode(user, {"value": valid_leaf}),
        OriginNode(None, {"value": OriginNode(None, {})}),
        OriginNode(None, {}),
        OriginNode(None, {"value": OriginNode(user, {"extra": valid_leaf})}),
    )
    for invalid_root in invalid_roots:
        with pytest.raises(ConfigError, match="origin tree"):
            MergeResult({"value": 1}, invalid_root, ())


def test_merge_result_requires_sequence_indices_to_match_exactly():
    user = Origin("user")
    wrong_indices = OriginNode(
        None,
        {"items": OriginNode(user, {1: OriginNode(user, {})})},
    )

    with pytest.raises(ConfigError, match="origin tree"):
        MergeResult({"items": [1]}, wrong_indices, ())


@pytest.mark.parametrize("alias", [True, 1.0])
def test_merge_result_compares_origin_segments_by_exact_type_and_value(alias):
    """Catches bool/float keys comparing equal to the required integer index."""
    user = Origin("user")
    first = OriginNode(user, {})
    second = OriginNode(user, {})
    forged_sequence = object.__new__(OriginNode)
    object.__setattr__(forged_sequence, "origin", user)
    object.__setattr__(
        forged_sequence,
        "children",
        MappingProxyType({0: first, alias: second}),
    )
    root = OriginNode(None, {"items": forged_sequence})

    with pytest.raises(ConfigError, match="origin tree"):
        MergeResult({"items": [1, 2]}, root, ())


def test_origins_at_rejects_boolean_sequence_indices():
    result = initial_merge({"items": ["zero", "one"]}, origin=Origin("user"))

    with pytest.raises(ConfigError, match=r"origin path segment.*bool"):
        origins_at(result.origins, ("items", True))


def test_origins_at_normalizes_child_mapping_protocol_failures():
    class HostileLookupError(Exception):
        def __str__(self):
            raise AssertionError("exception text must not run")

        def __repr__(self):
            raise AssertionError("exception repr must not run")

    class FailingChildren(_ItemsMapping):
        def __getitem__(self, key):
            raise HostileLookupError

    forged_root = object.__new__(OriginNode)
    object.__setattr__(forged_root, "origin", None)
    object.__setattr__(forged_root, "children", FailingChildren([]))

    with pytest.raises(ConfigError, match="origin child lookup failed"):
        origins_at(forged_root, ("value",))


def test_distinct_empty_sequences_remain_distinct_per_edge_after_thaw_and_origins():
    result = initial_merge({"first": [], "second": []}, origin=Origin("user"))
    mutable = thaw(result.document)

    assert mutable["first"] is not mutable["second"]
    assert result.origins.children["first"] is not result.origins.children["second"]


def test_origin_alias_memo_is_shared_within_but_not_across_initial_merges():
    shared = {"leaf": 1}

    first = initial_merge(
        {"left": shared, "right": shared}, origin=Origin("user")
    )
    second = initial_merge(
        {"left": shared, "right": shared}, origin=Origin("user")
    )

    assert first.origins.children["left"] is first.origins.children["right"]
    assert second.origins.children["left"] is second.origins.children["right"]
    assert first.origins.children["left"] is not second.origins.children["left"]


def test_one_branch_merge_splits_that_occurrence_but_keeps_nested_aliases():
    nested = {"value": 1}
    shared = {"nested": nested}
    parent = initial_merge(
        {"left": shared, "right": shared}, origin=Origin("user")
    )

    result = merge_with_origins(
        parent,
        {"left": {"added": 2}},
        origin=Origin("preset", "one"),
    )

    assert result.document["left"] is not result.document["right"]
    assert result.document["left"]["nested"] is result.document["right"]["nested"]
    assert result.origins.children["left"] is not result.origins.children["right"]
    assert (
        result.origins.children["left"].children["nested"]
        is result.origins.children["right"].children["nested"]
    )


def test_identical_recursive_merge_state_reuses_one_result_fragment():
    shared_base = {"kept": {"leaf": 1}}
    shared_patch = {"added": {"leaf": 2}}
    parent = initial_merge(
        {"left": shared_base, "right": shared_base}, origin=Origin("user")
    )

    result = merge_with_origins(
        parent,
        {"left": shared_patch, "right": shared_patch},
        origin=Origin("preset", "one"),
    )

    assert result.document["left"] is result.document["right"]
    assert result.origins.children["left"] is result.origins.children["right"]


def test_append_preserves_inherited_patch_and_operation_alias_topology():
    inherited_item = {"base": 1}
    inherited = [inherited_item, inherited_item]
    appended_item = {"new": 2}
    appended = [appended_item, appended_item]
    append_patch = {"append": appended}
    parent = initial_merge(
        {"first": inherited, "second": inherited}, origin=Origin("user")
    )

    result = merge_with_origins(
        parent,
        {"first": append_patch, "second": append_patch},
        origin=Origin("preset", "one"),
    )

    first = result.document["first"]
    first_origins = result.origins.children["first"]
    assert first is result.document["second"]
    assert first_origins is result.origins.children["second"]
    assert first[0] is first[1]
    assert first[2] is first[3]
    assert first_origins.children[0] is first_origins.children[1]
    assert first_origins.children[2] is first_origins.children[3]


def test_cached_merge_replays_deletion_evidence_for_every_occurrence():
    shared_base = {"gone": 1, "kept": 2}
    shared_patch = {"~gone": None}
    parent = initial_merge(
        {"left": shared_base, "right": shared_base}, origin=Origin("user")
    )

    result = merge_with_origins(
        parent,
        {"left": shared_patch, "right": shared_patch},
        origin=Origin("preset", "one"),
    )

    assert result.document["left"] is result.document["right"]
    assert result.deletions == (
        DeletionRecord(("left", "gone"), Origin("preset", "one")),
        DeletionRecord(("right", "gone"), Origin("preset", "one")),
    )


def test_parallel_validation_refuses_one_document_alias_with_divergent_origins():
    shared = {"leaf": 1}
    user = Origin("user")
    preset = Origin("preset", "one")
    user_node = OriginNode(user, {"leaf": OriginNode(user, {})})
    preset_node = OriginNode(preset, {"leaf": OriginNode(preset, {})})
    root = OriginNode(None, {"left": user_node, "right": preset_node})

    with pytest.raises(ConfigError, match="origin tree"):
        MergeResult({"left": shared, "right": shared}, root, ())


class _CountingOriginNode(OriginNode):
    allocations = 0

    def __post_init__(self):
        type(self).allocations += 1
        super().__post_init__()


class _CountingChildren(Mapping):
    iterations = 0
    lookups = 0

    def __init__(self, values):
        self._values = dict(values)

    def __getitem__(self, key):
        type(self).lookups += 1
        return self._values[key]

    def __iter__(self):
        type(self).iterations += 1
        return iter(self._values)

    def __len__(self):
        return len(self._values)


def _instrument_origin_children(root: OriginNode) -> int:
    pending = [root]
    visited: set[int] = set()
    while pending:
        node = pending.pop()
        if id(node) in visited:
            continue
        visited.add(id(node))
        original = dict(node.children)
        object.__setattr__(node, "children", _CountingChildren(original))
        pending.extend(original.values())
    return len(visited)


def test_shared_binary_dag_remains_linear_through_merge_and_validation(monkeypatch):
    """Census-proves freeze/thaw/origin/validation work stays per unique node."""
    levels = 12
    shared: object = {"leaf": 1}
    for _ in range(levels):
        shared = {"left": shared, "right": shared}
    document = {"tree": shared}
    monkeypatch.setattr(layering_module, "OriginNode", _CountingOriginNode)
    _CountingOriginNode.allocations = 0

    result = initial_merge(document, origin=Origin("user"))
    initial_allocations = _CountingOriginNode.allocations
    _CountingOriginNode.allocations = 0
    merged = merge_with_origins(
        result, {"marker": 2}, origin=Origin("preset", "one")
    )
    merge_allocations = _CountingOriginNode.allocations
    unique_origin_nodes = _instrument_origin_children(result.origins)
    _CountingChildren.iterations = 0
    _CountingChildren.lookups = 0
    layering_module._validate_parallel_origin_tree(result.document, result.origins)
    validation_iterations = _CountingChildren.iterations
    validation_lookups = _CountingChildren.lookups
    document_cursor = merged.document["tree"]
    origin_cursor = merged.origins.children["tree"]
    aliases_preserved = True
    for _ in range(levels):
        aliases_preserved = aliases_preserved and (
            document_cursor["left"] is document_cursor["right"]
            and origin_cursor.children["left"] is origin_cursor.children["right"]
        )
        document_cursor = document_cursor["left"]
        origin_cursor = origin_cursor.children["left"]

    expected_unique = levels + 3
    assert (
        initial_allocations,
        unique_origin_nodes,
        validation_iterations,
        validation_lookups,
        merge_allocations,
        aliases_preserved,
    ) == (
        expected_unique,
        expected_unique,
        expected_unique,
        levels * 2 + 2,
        2,
        True,
    )


class _HostileInput:
    def __repr__(self):
        raise AssertionError("repr must not run")


def test_merge_root_preconditions_name_only_the_rejected_type():
    parent = initial_merge({}, origin=Origin("user"))
    calls = (
        lambda hostile: initial_merge(hostile, origin=Origin("user")),
        lambda hostile: merge_with_origins(
            hostile, {}, origin=Origin("user")
        ),
        lambda hostile: merge_with_origins(
            parent, hostile, origin=Origin("user")
        ),
        lambda hostile: merge_extends(hostile, {}),
        lambda hostile: merge_extends({}, hostile),
    )

    for call in calls:
        with pytest.raises(ConfigError, match="HostileInput"):
            call(_HostileInput())


def test_empty_deletion_key_is_refused_identically_by_both_merge_paths():
    message = "layering deletion key must name a value after '~'."
    parent = initial_merge({"kept": 1}, origin=Origin("user"))

    with pytest.raises(ConfigError) as origin_error:
        merge_with_origins(parent, {"~": None}, origin=Origin("user"))
    with pytest.raises(ConfigError) as compatibility_error:
        merge_extends({"~": None}, {"kept": 1})

    assert str(origin_error.value) == message
    assert str(compatibility_error.value) == message
