"""Neutral preset selection and document layering behavior."""

from __future__ import annotations

import sys
from collections.abc import Mapping

import pytest

from _rheplicant_bootstrap import layering as layering_module
from _rheplicant_bootstrap.errors import ConfigError
from _rheplicant_bootstrap.frozen import thaw
from _rheplicant_bootstrap.layering import layer_presets, parse_default
from _rheplicant_bootstrap.presets import PresetRequest, PresetSnapshot

_FIXTURE_BYTES = b"fixture"
_FIXTURE_SHA256 = "f16d05ec6b29248d2c61adb1e9263f78e4f7bace1b955014a2d17872cfe4064d"


def _snapshot(name: str, document: dict) -> PresetSnapshot:
    return PresetSnapshot(
        name=name,
        resource=f"rheplicant/config/presets/{name}.yaml",
        input_bytes=_FIXTURE_BYTES,
        sha256=_FIXTURE_SHA256,
        document=document,
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


def test_parse_default_does_not_invoke_scalar_subclass_hooks():
    class HostileText(str):
        def split(self, *args, **kwargs):
            raise AssertionError("split must not run")

        def __eq__(self, other):
            raise AssertionError("equality must not run")

        def __hash__(self):
            raise AssertionError("hash must not run")

        def __repr__(self):
            raise AssertionError("repr must not run")

    request = parse_default(
        {"from": HostileText("one"), "only": [HostileText("runtime")]}
    )

    assert request == PresetRequest("one", ("runtime",))


class _ArmedMappingKey(str):
    armed = False

    def __hash__(self):
        if self.armed:
            raise AssertionError("source-key hash must not run")
        return str.__hash__(self)

    def __eq__(self, other):
        if self.armed:
            raise AssertionError("source-key equality must not run")
        return str.__eq__(self, other)

    def __repr__(self):
        raise AssertionError("repr must not run")


def test_parse_default_canonicalizes_mapping_keys_before_set_or_lookup():
    source_key = _ArmedMappingKey("from")
    raw = {source_key: "one", "only": ["runtime"]}
    source_key.armed = True

    request = parse_default(raw)

    assert request == PresetRequest("one", ("runtime",))


def test_parse_default_rejects_a_nonstring_selector_without_using_repr():
    class HostileSelector:
        def __repr__(self):
            raise AssertionError("repr must not run")

    with pytest.raises(ConfigError, match=r"only:.*HostileSelector"):
        parse_default({"from": "one", "only": [HostileSelector()]})


class _ForgedCallbackConfigError(ConfigError):
    def __str__(self):
        raise AssertionError("callback exception text must not run")

    def __repr__(self):
        raise AssertionError("callback exception repr must not run")


class _FailingItemsIterator:
    def __init__(self, marker, seam):
        self.marker = marker
        self.seam = seam
        self.done = False

    def __iter__(self):
        if self.seam == "iter":
            raise self.marker
        return self

    def __next__(self):
        if self.seam == "next":
            raise self.marker
        if self.done:
            raise StopIteration
        self.done = True
        if self.seam == "unpack":
            marker = self.marker

            class BrokenItem:
                def __iter__(self):
                    raise marker

            return BrokenItem()
        return "from", "one"


class _ProtocolMapping(Mapping):
    def __init__(self, marker, seam):
        self.marker = marker
        self.seam = seam

    def __len__(self):
        return 1

    def __iter__(self):
        return iter(("from",))

    def __getitem__(self, key):
        if key == "from":
            return "one"
        raise KeyError(key)

    def items(self):
        if self.seam == "items":
            raise self.marker
        return _FailingItemsIterator(self.marker, self.seam)


@pytest.mark.parametrize("seam", ["items", "iter", "next", "unpack"])
def test_parse_default_replaces_callback_configerror_at_each_mapping_seam(seam):
    marker = _ForgedCallbackConfigError("private marker")

    with pytest.raises(ConfigError) as caught:
        parse_default(_ProtocolMapping(marker, seam))

    assert caught.value is not marker
    assert str(caught.value) == "defaults: preset entry mapping traversal failed."


def test_only_overlap_work_is_bounded_by_total_selector_segments():
    """Catches restoring a pairwise-quadratic overlap scan."""
    count = 2_000
    selectors = [f"resources.section_{index}.leaf" for index in range(count)]
    document = {
        "resources": {
            f"section_{index}": {"leaf": index} for index in range(count)
        }
    }
    request = parse_default({"from": "one", "only": selectors})
    target_code = layering_module._select_only.__code__
    line_events = 0

    def count_lines(frame, event, arg):
        nonlocal line_events
        if frame.f_code is target_code and event == "line":
            line_events += 1
        return count_lines

    # Restore the PREVIOUS tracer, not None. coverage installs its CTracer
    # through sys.settrace, so handing back None uninstalls the measurement
    # from this thread permanently and pytest-cov never notices -- measured:
    # 1982 consecutive tests then recorded nothing under a serial run, and
    # plugins.py went from 88 % to 16 % on tests/bootstrap alone.
    previous = sys.gettrace()
    sys.settrace(count_lines)
    try:
        result, _ = layer_presets(
            {}, (request,), preset_provider=_provider({"one": document})
        )
    finally:
        sys.settrace(previous)

    assert len(result.document["resources"]) == count
    total_segments = count * 3
    assert line_events <= total_segments * 40


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


def test_user_document_is_strictly_frozen_before_the_provider_runs():
    user_document = {"runtime": {"value": "before"}}
    snapshot = _snapshot("one", {})

    def provider(_):
        user_document["runtime"]["value"] = "after"
        return snapshot

    result, _ = layer_presets(
        user_document, (parse_default("one"),), preset_provider=provider
    )

    assert result.document["runtime"]["value"] == "before"


def test_invalid_user_evidence_is_refused_before_the_provider_runs():
    provider_calls: list[str] = []

    def provider(name):
        provider_calls.append(name)
        return _snapshot(name, {})

    with pytest.raises(ConfigError, match="layer_presets document"):
        layer_presets(
            {"unsafe": object()},
            (parse_default("one"),),
            preset_provider=provider,
        )

    assert provider_calls == []


def test_preset_provider_must_return_a_snapshot_without_using_repr():
    class NotSnapshot:
        def __repr__(self):
            raise AssertionError("repr must not run")

    with pytest.raises(ConfigError, match="NotSnapshot.*PresetSnapshot"):
        layer_presets(
            {},
            (parse_default("one"),),
            preset_provider=lambda _: NotSnapshot(),
        )


def test_preset_provider_snapshot_name_must_match_the_request():
    with pytest.raises(ConfigError, match="other.*one"):
        layer_presets(
            {},
            (parse_default("one"),),
            preset_provider=lambda _: _snapshot("other", {}),
        )


class _HostileProviderError(Exception):
    def __str__(self):
        raise AssertionError("exception text must not run")

    def __repr__(self):
        raise AssertionError("exception repr must not run")


def test_preset_provider_ordinary_failure_is_static_and_configerror_is_preserved():
    def failing_provider(_):
        raise _HostileProviderError

    with pytest.raises(ConfigError, match=r"provider.*one"):
        layer_presets({}, (parse_default("one"),), preset_provider=failing_provider)

    marker = ConfigError("provider marker")

    def refusing_provider(_):
        raise marker

    with pytest.raises(ConfigError) as caught:
        layer_presets({}, (parse_default("one"),), preset_provider=refusing_provider)
    assert caught.value is marker


def test_preset_provider_does_not_catch_process_control_baseexceptions():
    class StopNow(BaseException):
        pass

    def stopping_provider(_):
        raise StopNow

    with pytest.raises(StopNow):
        layer_presets({}, (parse_default("one"),), preset_provider=stopping_provider)


def test_request_sequence_protocol_failure_is_normalized_statically():
    class HostileRequestsError(Exception):
        def __str__(self):
            raise AssertionError("exception text must not run")

        def __repr__(self):
            raise AssertionError("exception repr must not run")

    class FailingRequests:
        def __iter__(self):
            raise HostileRequestsError

    with pytest.raises(ConfigError, match="request sequence traversal failed"):
        layer_presets({}, FailingRequests(), preset_provider=lambda _: None)


def test_preset_provider_revalidates_an_exact_but_forged_snapshot():
    forged = object.__new__(PresetSnapshot)
    object.__setattr__(forged, "name", "one")
    object.__setattr__(
        forged, "resource", "rheplicant/config/presets/one.yaml"
    )
    object.__setattr__(forged, "input_bytes", _FIXTURE_BYTES)
    object.__setattr__(forged, "sha256", "0" * 64)
    object.__setattr__(forged, "document", {})
    object.__setattr__(forged, "expanded_nodes", 0)

    with pytest.raises(ConfigError, match=r"sha256.*does not match"):
        layer_presets(
            {}, (parse_default("one"),), preset_provider=lambda _: forged
        )


@pytest.mark.parametrize(
    "section",
    [
        "schema_version",
        "defaults",
        "plugins",
        "variants",
        "runs",
        "outputs",
        "campaign",
        "unexpected_section",
    ],
)
def test_preset_provider_cannot_introduce_a_forbidden_section(section):
    forged = object.__new__(PresetSnapshot)
    object.__setattr__(forged, "name", "one")
    object.__setattr__(
        forged, "resource", "rheplicant/config/presets/one.yaml"
    )
    object.__setattr__(forged, "input_bytes", _FIXTURE_BYTES)
    object.__setattr__(forged, "sha256", _FIXTURE_SHA256)
    object.__setattr__(forged, "document", {section: {}})
    object.__setattr__(forged, "expanded_nodes", 1)

    with pytest.raises(ConfigError, match=section):
        layer_presets(
            {}, (parse_default("one"),), preset_provider=lambda _: forged
        )


def test_provider_snapshot_validation_replaces_callback_configerror():
    marker = _ForgedCallbackConfigError("private snapshot marker")
    forged = object.__new__(PresetSnapshot)
    object.__setattr__(forged, "name", "one")
    object.__setattr__(
        forged, "resource", "rheplicant/config/presets/one.yaml"
    )
    object.__setattr__(forged, "input_bytes", _FIXTURE_BYTES)
    object.__setattr__(forged, "sha256", _FIXTURE_SHA256)
    object.__setattr__(forged, "document", _ProtocolMapping(marker, "items"))
    object.__setattr__(forged, "expanded_nodes", 1)

    with pytest.raises(ConfigError) as caught:
        layer_presets(
            {}, (parse_default("one"),), preset_provider=lambda _: forged
        )

    assert caught.value is not marker
    assert "preset:one" in str(caught.value)


def test_preset_provider_requires_the_exact_snapshot_class():
    class SnapshotSubclass(PresetSnapshot):
        pass

    snapshot = SnapshotSubclass(
        name="one",
        resource="rheplicant/config/presets/one.yaml",
        input_bytes=_FIXTURE_BYTES,
        sha256=_FIXTURE_SHA256,
        document={},
        expanded_nodes=0,
    )

    with pytest.raises(ConfigError, match=r"SnapshotSubclass.*PresetSnapshot"):
        layer_presets(
            {}, (parse_default("one"),), preset_provider=lambda _: snapshot
        )


def test_only_selection_preserves_shared_preset_subtree_topology():
    shared = {"leaf": 1}
    result, _ = layer_presets(
        {},
        (parse_default({"from": "one", "only": ["runtime"]}),),
        preset_provider=_provider(
            {"one": {"runtime": {"first": shared, "second": shared}}}
        ),
    )

    runtime = result.document["runtime"]
    runtime_origins = result.origins.children["runtime"]
    assert runtime["first"] is runtime["second"]
    assert (
        runtime_origins.children["first"]
        is runtime_origins.children["second"]
    )


def test_user_model_without_inherit_replaces_the_preset_candidate():
    result, _ = layer_presets(
        {"model": {"user_node": {"x": 2}}},
        (parse_default("one"),),
        preset_provider=_provider({"one": {"model": {"preset_node": {"x": 1}}}}),
    )
    assert thaw(result.document)["model"] == {"user_node": {"x": 2}}


def test_explicit_user_null_model_replaces_the_preset_candidate():
    result, _ = layer_presets(
        {"model": None},
        (parse_default("one"),),
        preset_provider=_provider(
            {"one": {"model": {"preset_node": {"x": 1}}}}
        ),
    )

    assert "model" in result.document
    assert result.document["model"] is None
    assert result.origins.children["model"].origin is not None
    assert result.origins.children["model"].origin.kind == "user"


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
