"""Neutral preset selection and document layering behavior."""

from __future__ import annotations

import pytest

from _rheplicant_bootstrap.errors import ConfigError
from _rheplicant_bootstrap.frozen import freeze, thaw
from _rheplicant_bootstrap.layering import layer_presets, parse_default
from _rheplicant_bootstrap.presets import PresetRequest, PresetSnapshot


def _snapshot(name: str, document: dict) -> PresetSnapshot:
    return PresetSnapshot(
        name=name,
        resource=f"rheplicant/config/presets/{name}.yaml",
        input_bytes=b"fixture",
        sha256="0" * 64,
        document=freeze(document),
        expanded_nodes=1,
    )


def _provider(documents: dict[str, dict]):
    return lambda name: _snapshot(name, documents[name])


def test_omitted_only_and_explicit_empty_only_are_distinct():
    assert parse_default("rhino_v1").only is None
    with pytest.raises(ConfigError, match="only: must select at least one path"):
        parse_default({"from": "rhino_v1", "only": []})


def test_parse_default_constructs_sequence_fields_as_tuples():
    request = parse_default(
        {"from": "rhino_v1", "only": ["runtime", "observation.site"]}
    )
    assert request.name == "rhino_v1"
    assert request.only == ("runtime", "observation.site")
    assert isinstance(request.only, tuple)


def test_direct_request_construction_canonicalizes_and_detaches_only():
    """Catches custom callers bypassing parse_default's tuple invariant."""
    only = ["runtime"]
    request = PresetRequest(name="one", only=only)
    only.append("model")

    assert request.only == ("runtime",)
    with pytest.raises(ConfigError, match="only: must select at least one path"):
        PresetRequest(name="one", only=[])


class _CountedSegment(str):
    comparisons = 0

    def __eq__(self, other):
        type(self).comparisons += 1
        return super().__eq__(other)

    __hash__ = str.__hash__


class _CountedSelector(str):
    def __new__(cls, segment: _CountedSegment):
        instance = super().__new__(cls, segment)
        instance.segment = segment
        return instance

    def split(self, sep=None, maxsplit=-1):
        if sep == ".":
            return [self.segment]
        return super().split(sep, maxsplit)


def test_only_overlap_work_is_bounded_by_total_selector_segments():
    """Catches restoring a pairwise-quadratic overlap scan."""
    count = 2_000
    segments = [_CountedSegment(f"section_{index}") for index in range(count)]
    selectors = [_CountedSelector(segment) for segment in segments]
    document = {segment: index for index, segment in enumerate(segments)}
    request = parse_default({"from": "one", "only": selectors})
    _CountedSegment.comparisons = 0

    result, _ = layer_presets(
        {}, (request,), preset_provider=_provider({"one": document})
    )

    assert len(result.document) == count
    assert _CountedSegment.comparisons <= count * 8


@pytest.mark.parametrize(
    ("only", "needle"),
    [
        (["absent"], "does not exist"),
        (["observation", "observation.site"], "overlap"),
        (["runtime", "runtime"], "duplicate"),
        (["model.sky"], "select model as a whole"),
    ],
)
def test_invalid_only_selections_are_refused(only, needle):
    request = parse_default({"from": "one", "only": only})
    provider = _provider(
        {"one": {"runtime": {}, "observation": {"site": {}}, "model": {"sky": {}}}}
    )
    with pytest.raises(ConfigError, match=needle):
        layer_presets({}, (request,), preset_provider=provider)


def test_only_selects_exact_subtrees_and_preserves_request_snapshot_order():
    request = parse_default(
        {"from": "one", "only": ["runtime", "observation.site"]}
    )
    result, selected = layer_presets(
        {"schema_version": 1},
        (request,),
        preset_provider=_provider(
            {
                "one": {
                    "runtime": {"jax_enable_x64": True},
                    "observation": {"site": {"lat": 1}, "pointing": {"mode": "drift"}},
                }
            }
        ),
    )

    assert thaw(result.document) == {
        "runtime": {"jax_enable_x64": True},
        "observation": {"site": {"lat": 1}},
        "schema_version": 1,
    }
    assert selected[0][0] is request
    assert selected[0][1].name == "one"
    assert isinstance(selected, tuple)


def test_a_preset_name_may_appear_only_once():
    requests = (parse_default("one"), parse_default({"from": "one", "only": ["runtime"]}))
    with pytest.raises(ConfigError, match="more than once"):
        layer_presets({}, requests, preset_provider=_provider({"one": {"runtime": {}}}))


def test_two_preset_models_replace_instead_of_union():
    result, _ = layer_presets(
        {"schema_version": 1},
        (parse_default("first"), parse_default("second")),
        preset_provider=_provider(
            {
                "first": {"model": {"first_node": {"x": 1}}},
                "second": {"model": {"second_node": {"y": 2}}},
            }
        ),
    )
    assert thaw(result.document)["model"] == {"second_node": {"y": 2}}


def test_explicit_null_preset_model_replaces_an_earlier_candidate():
    """Catches conflating a present null model with an absent model key."""
    result, _ = layer_presets(
        {},
        (parse_default("first"), parse_default("second")),
        preset_provider=_provider(
            {
                "first": {"model": {"first_node": {"x": 1}}},
                "second": {"model": None},
            }
        ),
    )
    assert "model" in result.document
    assert result.document["model"] is None


def test_custom_provider_snapshot_cannot_change_layered_evidence_after_construction():
    """Catches trusting a provider's mutable document after snapshot construction."""
    provider_document = {"runtime": {"values": [bytearray(b"one")]}}
    snapshot = _snapshot("one", provider_document)
    provider_document["runtime"]["values"][0][:] = b"two"
    provider_document["runtime"]["values"].append(b"late")

    result, selected = layer_presets(
        {}, (parse_default("one"),), preset_provider=lambda _: snapshot
    )

    assert result.document["runtime"]["values"] == (b"one",)
    assert selected[0][1].document["runtime"]["values"] == (b"one",)


def test_user_model_without_inherit_replaces_the_preset_candidate():
    result, _ = layer_presets(
        {"model": {"user_node": {"x": 2}}},
        (parse_default("one"),),
        preset_provider=_provider({"one": {"model": {"preset_node": {"x": 1}}}}),
    )
    assert thaw(result.document)["model"] == {"user_node": {"x": 2}}


def test_model_inherit_copies_named_candidate_nodes_and_is_consumed():
    result, _ = layer_presets(
        {
            "model": {
                "inherit": ["kept"],
                "kept": {"user": 2},
                "new": {"z": 3},
            }
        },
        (parse_default("one"),),
        preset_provider=_provider(
            {"one": {"model": {"kept": {"preset": 1}, "dropped": {"x": 0}}}}
        ),
    )
    assert thaw(result.document)["model"] == {
        "kept": {"preset": 1, "user": 2},
        "new": {"z": 3},
    }
    assert "inherit" not in result.document["model"]


@pytest.mark.parametrize(
    ("inherit", "needle"),
    [(["absent"], "absent"), (["kept", "kept"], "repeated")],
)
def test_model_inherit_refuses_absent_and_repeated_candidate_nodes(inherit, needle):
    with pytest.raises(ConfigError, match=needle):
        layer_presets(
            {"model": {"inherit": inherit}},
            (parse_default("one"),),
            preset_provider=_provider({"one": {"model": {"kept": {"x": 1}}}}),
        )
