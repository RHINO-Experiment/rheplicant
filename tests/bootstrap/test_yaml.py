"""Regression tests for bounded, single-document safe YAML parsing."""

from __future__ import annotations

import re

import pytest

from _rheplicant_bootstrap.errors import ConfigError
from _rheplicant_bootstrap.yaml import (
    BoundedSafeLoader,
    LoadedYaml,
    YamlLimits,
    compose_one_bounded,
    construct_plain_bounded,
    safe_load_document,
)


@pytest.mark.parametrize(
    ("payload", "needle"),
    [
        (b"a: 1\na: 2\n", "duplicate key 'a'"),
        (b"---\na: 1\n---\nb: 2\n", "exactly one YAML document"),
        (b"x: !!python/object/apply:os.system ['id']\n", "unsafe YAML tag"),
        (b"a: &a [*a]\n", "recursive alias"),
        (b"\xff", "UTF-8"),
    ],
)
def test_unsafe_or_ambiguous_yaml_is_refused(payload, needle):
    """Catches each dangerous loader branch accepting its document."""
    with pytest.raises(ConfigError, match=re.escape(needle)):
        safe_load_document(payload, source_name="case.yaml")


def test_duplicate_key_error_identifies_both_key_marks():
    """Catches duplicate detection that loses either the first or second mark."""
    with pytest.raises(ConfigError) as caught:
        safe_load_document(b"a: 1\na: 2\n", source_name="marks.yaml")

    message = str(caught.value)
    assert "duplicate key 'a'" in message
    assert "line 1, column 1" in message
    assert "line 2, column 1" in message


def test_empty_yaml_returns_a_zero_count_loaded_record():
    """Catches treating an empty document as a later-stage mapping refusal."""
    assert safe_load_document(b"", source_name="empty.yaml") == LoadedYaml(
        value=None,
        composed_nodes=0,
        expanded_nodes=0,
        alias_events=0,
    )


@pytest.mark.parametrize(
    ("payload", "limits", "limit_name", "observed", "bound"),
    [
        (b"123456789", YamlLimits(input_bytes=8), "byte count", 9, 8),
        (b"[a, b, c, d, e, f]", YamlLimits(composed_nodes=6), "composed nodes", 7, 6),
        (b"[[[x]]]", YamlLimits(depth=3), "depth", 4, 3),
        (b"abcde", YamlLimits(scalar_bytes=4), "scalar bytes", 5, 4),
        (
            b"a: &a x\nb: *a\nc: *a\nd: *a\n",
            YamlLimits(alias_events=2),
            "alias events",
            3,
            2,
        ),
        (
            b"[&a [1, 2], *a, 1, 2]",
            YamlLimits(expanded_nodes=8),
            "expanded nodes",
            9,
            8,
        ),
    ],
)
def test_yaml_limits_name_the_source_observed_count_and_bound(
    payload, limits, limit_name, observed, bound
):
    """Catches each resource limit checking too late or reporting opaquely."""
    with pytest.raises(ConfigError) as caught:
        safe_load_document(payload, source_name="limits.yaml", limits=limits)

    message = str(caught.value)
    assert "limits.yaml" in message
    assert limit_name in message
    assert str(observed) in message
    assert str(bound) in message


@pytest.mark.parametrize(
    ("payload", "limits", "expected"),
    [
        (b"1234567\n", YamlLimits(input_bytes=8), 1),
        (b"[a, b, c, d, e]", YamlLimits(composed_nodes=6), 6),
        (b"[[x]]", YamlLimits(depth=3), 3),
        (b"abcd", YamlLimits(scalar_bytes=4), "abcd"),
        (
            b"a: &a x\nb: *a\nc: *a\n",
            YamlLimits(alias_events=2),
            2,
        ),
        (b"[&a [1, 2], *a, 1]", YamlLimits(expanded_nodes=8), 8),
    ],
)
def test_yaml_accepts_values_exactly_at_each_limit(payload, limits, expected):
    """Catches off-by-one refusal at a documented inclusive resource bound."""
    loaded = safe_load_document(payload, source_name="boundary.yaml", limits=limits)

    if limits.composed_nodes == 6:
        assert loaded.composed_nodes == expected
    elif limits.depth == 3:
        assert loaded.value == (("x",),)
    elif limits.scalar_bytes == 4:
        assert loaded.value == expected
    elif limits.alias_events == 2:
        assert loaded.alias_events == expected
    elif limits.expanded_nodes == 8:
        assert loaded.expanded_nodes == expected
    else:
        assert loaded.value == 1234567


def test_alias_expansion_is_detached_between_uses():
    """Catches construction reusing an aliased list before it is frozen."""
    limits = YamlLimits()
    composed = compose_one_bounded(
        "[&a [1], *a]", source_name="aliases.yaml", limits=limits
    )
    value, _ = construct_plain_bounded(composed.root, source_name="aliases.yaml", limits=limits)

    assert value == [[1], [1]]
    assert value[0] is not value[1]


@pytest.mark.parametrize(
    "payload",
    [
        b'"\\uD800"',
        b"!!bool nope",
        b'!!int ""',
        b'!!float ""',
    ],
)
def test_malformed_core_scalars_are_normalized_to_config_error(payload):
    """Catches raw PyYAML scalar-constructor and surrogate exceptions leaking out."""
    with pytest.raises(ConfigError):
        safe_load_document(payload, source_name="malformed.yaml")


def test_nul_yaml_reader_error_is_normalized_to_config_error():
    """Catches a ReaderError raised before composition enters the neutral wrapper."""
    with pytest.raises(ConfigError):
        safe_load_document(b"\x00", source_name="nul.yaml")


@pytest.mark.parametrize(
    "payload",
    [
        b"!!python/name:os.system ''",
        b"!!python/tuple [1]",
        b"!!python/object {answer: 42}",
    ],
)
def test_unsafe_scalar_sequence_and_mapping_tags_are_refused(payload):
    """Catches a non-scalar unsafe tag slipping past the plain-value constructor."""
    with pytest.raises(ConfigError, match="unsafe YAML tag"):
        safe_load_document(payload, source_name="tags.yaml")


@pytest.mark.parametrize("payload", [b"11: first\n0xB: second\n", b"true: first\n1: second\n"])
def test_semantically_equal_yaml_keys_are_duplicate_keys(payload):
    """Catches duplicate detection based on lexical spelling rather than constructed keys."""
    with pytest.raises(ConfigError, match="duplicate key"):
        safe_load_document(payload, source_name="semantic-duplicates.yaml")


@pytest.mark.parametrize("payload", [b"? [a]\n: value\n", b"? {a: b}\n: value\n"])
def test_unhashable_yaml_keys_are_normalized_to_config_error(payload):
    """Catches raw TypeError from a sequence or mapping used as a mapping key."""
    with pytest.raises(ConfigError, match="not scalar"):
        safe_load_document(payload, source_name="unhashable-keys.yaml")


def test_loader_is_disposed_after_success_and_refusal(monkeypatch):
    """Catches a loader retained after either successful or refused composition."""
    original_dispose = BoundedSafeLoader.dispose
    disposed: list[BoundedSafeLoader] = []

    def tracked_dispose(loader):
        disposed.append(loader)
        original_dispose(loader)

    monkeypatch.setattr(BoundedSafeLoader, "dispose", tracked_dispose)

    safe_load_document(b"answer: 42", source_name="success.yaml")
    with pytest.raises(ConfigError):
        safe_load_document(b"---\na: 1\n---\nb: 2\n", source_name="refusal.yaml")

    assert len(disposed) == 2
