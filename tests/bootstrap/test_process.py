"""JAX-free process-entry grammar for Config Plan 4A."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence

import pytest

from _rheplicant_bootstrap.errors import ConfigError
from _rheplicant_bootstrap.presets import PresetRequest
from _rheplicant_bootstrap.process import (
    EffectiveProcessEntry,
    RawProcessEntry,
    RuntimeSpec,
    parse_effective_process_mapping,
    parse_raw_process_mapping,
    parse_runtime,
)
from _rheplicant_bootstrap.variants import LayerRef


def fake_parse_outputs(raw_outputs: object) -> object:
    """Task 17's strict seam: it receives the section value, not a document."""
    if not isinstance(raw_outputs, Mapping):
        raise ConfigError("outputs: test grammar requires a mapping or omission.")
    return tuple(raw_outputs.items())


@pytest.mark.parametrize("bad", [True, 1.0, "1", None, 2])
def test_schema_version_is_the_integer_one(bad):
    with pytest.raises(ConfigError, match="schema_version: 1"):
        parse_raw_process_mapping(
            {"schema_version": bad}, parse_outputs=fake_parse_outputs
        )


@pytest.mark.parametrize(
    "defaults",
    [
        42,
        ["../x"],
        [{"from": "rhino_v1", "extra": 1}],
        [{"from": "rhino_v1", "only": "runtime"}],
    ],
)
def test_defaults_has_only_the_three_approved_forms(defaults):
    with pytest.raises(ConfigError) as caught:
        parse_raw_process_mapping(
            {"schema_version": 1, "defaults": defaults},
            parse_outputs=fake_parse_outputs,
        )
    assert str(caught.value) == (
        "defaults: must be a list whose entries are preset names or mappings "
        "with exactly 'from' and optional 'only'."
    )


def test_non_string_top_level_keys_are_refused_in_source_order():
    document = {"schema_version": 1, 7: None, "z": None}
    with pytest.raises(ConfigError) as caught:
        parse_raw_process_mapping(document, parse_outputs=fake_parse_outputs)
    assert str(caught.value) == (
        "document: top-level key 2 must be a string; got int 7."
    )


def test_raw_runtime_may_be_absent_but_a_present_value_is_a_mapping():
    raw = parse_raw_process_mapping(
        {"schema_version": 1}, parse_outputs=fake_parse_outputs
    )
    assert raw.raw_runtime is None

    with pytest.raises(ConfigError) as caught:
        parse_raw_process_mapping(
            {"schema_version": 1, "runtime": []},
            parse_outputs=fake_parse_outputs,
        )
    assert str(caught.value) == "runtime: must be a mapping when present."


@pytest.mark.parametrize("section", ["defaults", "plugins", "runtime"])
def test_explicit_null_process_sections_are_not_treated_as_omitted(section):
    with pytest.raises(ConfigError):
        parse_raw_process_mapping(
            {"schema_version": 1, section: None},
            parse_outputs=fake_parse_outputs,
        )


@pytest.mark.parametrize(
    "name", ["", ".plug", "plug.", "two..dots", "not-a-module", "for"]
)
def test_plugin_names_are_nonempty_dotted_identifiers(name):
    with pytest.raises(ConfigError) as caught:
        parse_raw_process_mapping(
            {"schema_version": 1, "plugins": [name]},
            parse_outputs=fake_parse_outputs,
        )
    assert str(caught.value) == (
        f"plugins[0]: {name!r} is not a dot-separated Python module name."
    )


def test_duplicate_plugins_and_presets_are_refused_in_declaration_order():
    with pytest.raises(ConfigError) as caught:
        parse_raw_process_mapping(
            {"schema_version": 1, "plugins": ["one", "one"]},
            parse_outputs=fake_parse_outputs,
        )
    assert str(caught.value) == "plugins[1]: duplicate module name 'one'."

    with pytest.raises(ConfigError) as caught:
        parse_raw_process_mapping(
            {"schema_version": 1, "defaults": ["one", {"from": "one"}]},
            parse_outputs=fake_parse_outputs,
        )
    assert str(caught.value) == "defaults[1]: duplicate package preset 'one'."


def test_outputs_parser_receives_only_the_raw_outputs_value():
    seen: list[object] = []

    def parse_outputs(value: object) -> object:
        seen.append(value)
        return "parsed-output"

    outputs = {"dir": "results"}
    raw = parse_raw_process_mapping(
        {"schema_version": 1, "runtime": {}, "outputs": outputs},
        parse_outputs=parse_outputs,
    )

    assert seen == [outputs]
    assert raw.outputs == "parsed-output"


def test_parser_results_and_direct_process_records_freeze_nested_outputs():
    parsed_output = {"nested": ["before"]}
    raw = parse_raw_process_mapping(
        {"schema_version": 1},
        parse_outputs=lambda value: parsed_output,
    )
    direct = EffectiveProcessEntry(
        runtime=RuntimeSpec(False, "auto", None, {}),
        plugins=[],
        outputs={"nested": ["before"]},
    )
    parsed_output["nested"].append("after")

    assert raw.outputs == {"nested": ("before",)}
    assert direct.outputs == {"nested": ("before",)}
    with pytest.raises(TypeError):
        raw.outputs["new"] = 1


@pytest.mark.parametrize(
    ("value", "expected_type"),
    [
        (type("Text", (str,), {})("x"), str),
        (type("Integer", (int,), {})(1), int),
        (type("Floating", (float,), {})(1.5), float),
        (type("Blob", (bytes,), {})(b"x"), bytes),
    ],
)
def test_process_output_scalar_subclasses_are_canonicalized(
    value, expected_type
):
    runtime = RuntimeSpec(False, "auto", None, {})
    raw = RawProcessEntry(1, [], [], None, value)
    effective = EffectiveProcessEntry(runtime, [], value)

    assert type(raw.outputs) is expected_type
    assert type(effective.outputs) is expected_type


def test_direct_raw_process_record_freezes_all_container_fields():
    runtime = {"seeds": {"main": 1}}
    outputs = {"nested": [1]}
    record = RawProcessEntry(1, [], ["plug"], runtime, outputs)
    runtime["seeds"]["main"] = 2
    outputs["nested"].append(2)

    assert record.defaults == ()
    assert record.plugins == ("plug",)
    assert record.raw_runtime == {"seeds": {"main": 1}}
    assert record.outputs == {"nested": (1,)}


def test_raw_schema_version_is_canonicalized_without_subclass_comparison():
    class HostileOne(int):
        def __eq__(self, other):
            raise RuntimeError("hostile equality")

        def __ne__(self, other):
            raise RuntimeError("hostile inequality")

    record = RawProcessEntry(HostileOne(1), [], [], None, {})
    parsed = parse_raw_process_mapping(
        {"schema_version": HostileOne(1)},
        parse_outputs=fake_parse_outputs,
    )

    assert type(record.schema_version) is int
    assert record.schema_version == 1
    assert type(parsed.schema_version) is int
    with pytest.raises(ConfigError, match="schema_version"):
        RawProcessEntry(1.0, [], [], None, {})


def test_direct_process_records_reject_nested_record_subclasses():
    class RequestSubclass(PresetRequest):
        pass

    class RuntimeSubclass(RuntimeSpec):
        pass

    class RawSubclass(RawProcessEntry):
        pass

    with pytest.raises(ConfigError, match="defaults"):
        RawProcessEntry(
            1,
            [RequestSubclass("one", None)],
            [],
            None,
            {},
        )
    with pytest.raises(ConfigError, match="effective runtime"):
        EffectiveProcessEntry(
            RuntimeSubclass(False, "auto", None, {}),
            [],
            {},
        )
    with pytest.raises(ConfigError, match="raw process entry"):
        parse_effective_process_mapping(
            {"schema_version": 1, "runtime": {}},
            (),
            raw=RawSubclass(1, [], [], None, {}),
            parse_outputs=fake_parse_outputs,
        )


def test_omitted_outputs_is_passed_as_an_empty_mapping():
    seen: list[object] = []

    def parse_outputs(value: object) -> object:
        seen.append(value)
        return "parsed-output"

    parse_raw_process_mapping(
        {"schema_version": 1}, parse_outputs=parse_outputs
    )

    assert seen == [{}]


def test_runtime_defaults_are_complete_and_recursively_detached():
    named = {"simulation": 7}
    parsed = parse_runtime({"seeds": named})
    named["simulation"] = 9

    assert parsed == RuntimeSpec(
        jax_enable_x64=False,
        platform="auto",
        seed=None,
        seeds={"simulation": 7},
    )
    assert type(parsed.seeds).__name__ == "mappingproxy"


class _GetMustNotRun(Mapping[str, object]):
    def __init__(self, values: Mapping[str, object]):
        self._values = dict(values)

    def __getitem__(self, key: str) -> object:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def get(self, key, default=None):
        raise RuntimeError("hostile get")


class _TripleItems(Mapping[str, object]):
    def __getitem__(self, key):
        if key == "schema_version":
            return 1
        raise KeyError(key)

    def __iter__(self):
        return iter(("schema_version",))

    def __len__(self):
        return 1

    def items(self):
        return [("schema_version", 1, "hidden")]


class _AbortItems(BaseException):
    pass


class _BaseExceptionItems(_TripleItems):
    def items(self):
        raise _AbortItems("stop")


class _AbortLayers(BaseException):
    pass


class _BrokenLayers(Sequence):
    def __init__(self, failure):
        self.failure = failure

    def __getitem__(self, index):
        raise self.failure

    def __len__(self):
        return 1


def test_runtime_freezes_before_reading_and_never_calls_source_get():
    parsed = parse_runtime(_GetMustNotRun({"platform": "cpu"}))
    assert parsed.platform == "cpu"


def test_mapping_item_emissions_are_strict_pairs_and_baseexception_propagates():
    with pytest.raises(ConfigError, match="top-level mapping traversal"):
        parse_raw_process_mapping(
            _TripleItems(), parse_outputs=fake_parse_outputs
        )
    with pytest.raises(_AbortItems, match="stop"):
        parse_raw_process_mapping(
            _BaseExceptionItems(), parse_outputs=fake_parse_outputs
        )


def test_runtime_direct_construction_freezes_seeds():
    seeds = {"main": 1}
    parsed = RuntimeSpec(False, "auto", None, seeds)
    seeds["main"] = 2
    assert parsed.seeds == {"main": 1}


@pytest.mark.parametrize(
    ("section", "message"),
    [
        (None, "runtime: is a mapping; got NoneType."),
        ({"jax_enable_x64": 1}, "runtime.jax_enable_x64 is a bool; got int."),
        (
            {"platform": "cuda"},
            "runtime.platform is one of ['auto', 'cpu', 'gpu', 'tpu']; "
            "got 'cuda'.",
        ),
        ({"seed": True}, "runtime.seed is an int or null; got bool."),
        ({"seeds": []}, "runtime.seeds is a mapping of name -> int; got list."),
        ({"unknown": 1}, "runtime: unknown keys ['unknown']."),
    ],
)
def test_runtime_grammar_is_closed_without_importing_jax(section, message):
    with pytest.raises(ConfigError) as caught:
        parse_runtime(section)
    assert str(caught.value) == message


def test_effective_runtime_is_required():
    raw = parse_raw_process_mapping(
        {"schema_version": 1}, parse_outputs=fake_parse_outputs
    )
    with pytest.raises(ConfigError) as caught:
        parse_effective_process_mapping(
            {"schema_version": 1},
            (),
            raw=raw,
            parse_outputs=fake_parse_outputs,
        )
    assert str(caught.value) == (
        "runtime: is required after package presets are layered."
    )


def test_effective_layer_sequence_protocol_is_controlled():
    raw = parse_raw_process_mapping(
        {"schema_version": 1}, parse_outputs=fake_parse_outputs
    )
    with pytest.raises(ConfigError, match="layer sequence traversal"):
        parse_effective_process_mapping(
            {"schema_version": 1, "runtime": {}},
            _BrokenLayers(RuntimeError("boom")),
            raw=raw,
            parse_outputs=fake_parse_outputs,
        )
    with pytest.raises(_AbortLayers, match="stop"):
        parse_effective_process_mapping(
            {"schema_version": 1, "runtime": {}},
            _BrokenLayers(_AbortLayers("stop")),
            raw=raw,
            parse_outputs=fake_parse_outputs,
        )
    with pytest.raises(ConfigError, match="LayerRef"):
        parse_effective_process_mapping(
            {"schema_version": 1, "runtime": {}},
            (object(),),
            raw=raw,
            parse_outputs=fake_parse_outputs,
        )


def test_effective_plugins_are_revalidated_and_must_match_raw_entry():
    raw = parse_raw_process_mapping(
        {"schema_version": 1, "plugins": ["base.plugin"]},
        parse_outputs=fake_parse_outputs,
    )
    with pytest.raises(ConfigError, match="plugins"):
        parse_effective_process_mapping(
            {
                "schema_version": 1,
                "plugins": ["changed.plugin"],
                "runtime": {},
            },
            (),
            raw=raw,
            parse_outputs=fake_parse_outputs,
        )


@pytest.mark.parametrize(
    "document",
    [{}, {"runtime": {"platform": "cuda"}}],
)
def test_variant_runtime_refusals_keep_the_complete_raw_variant_name(document):
    raw = parse_raw_process_mapping(
        {"schema_version": 1}, parse_outputs=fake_parse_outputs
    )
    layer = LayerRef(
        kind="variant",
        name="slash/雪",
        prefix="variants.slash/雪",
        document=document,
        declared_runs=(),
    )

    with pytest.raises(ConfigError) as caught:
        parse_effective_process_mapping(
            {"schema_version": 1, "runtime": {}},
            (layer,),
            raw=raw,
            parse_outputs=fake_parse_outputs,
        )

    assert "variants.slash/雪.runtime" in str(caught.value)
