"""JAX-free process-entry grammar for Config Plan 4A."""

from __future__ import annotations

import os
import subprocess
import sys
from collections import deque
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

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
    validate_variant_process_sections,
)
from _rheplicant_bootstrap.types import Origin
from _rheplicant_bootstrap.variants import LayerRef


class _InstanceClassTrap:
    def __init__(self, *, forged_type: type | None) -> None:
        self.forged_type = forged_type
        self.calls = 0

    @property
    def __class__(self):
        self.calls += 1
        if self.forged_type is None:
            raise RuntimeError("descriptor secret")
        return self.forged_type


class _DescriptorMapping(Mapping):
    def __init__(self, values) -> None:
        self.values = dict(values)
        self.class_calls = 0

    @property
    def __class__(self):
        self.class_calls += 1
        raise RuntimeError("mapping descriptor secret")

    def __getitem__(self, key):
        return self.values[key]

    def __iter__(self):
        return iter(self.values)

    def __len__(self):
        return len(self.values)


class _DescriptorSequence(Sequence):
    def __init__(self, values) -> None:
        self.values = tuple(values)
        self.class_calls = 0

    @property
    def __class__(self):
        self.class_calls += 1
        raise RuntimeError("sequence descriptor secret")

    def __getitem__(self, index):
        return self.values[index]

    def __len__(self):
        return len(self.values)


def fake_parse_outputs(raw_outputs: object) -> object:
    """Task 17's strict seam: it receives the section value, not a document."""
    if not isinstance(raw_outputs, Mapping):
        raise ConfigError("outputs: test grammar requires a mapping or omission.")
    return tuple(raw_outputs.items())


@pytest.mark.parametrize("forged_type", (None, str))
def test_parse_runtime_never_reads_instance_class_descriptors(forged_type):
    value = _InstanceClassTrap(forged_type=forged_type)
    with pytest.raises(ConfigError) as caught:
        parse_runtime({"platform": value})
    assert value.calls == 0
    assert "descriptor secret" not in str(caught.value)


@pytest.mark.parametrize("forged_type", (None, bool))
def test_runtime_spec_bool_check_uses_the_actual_static_type(forged_type):
    value = _InstanceClassTrap(forged_type=forged_type)
    with pytest.raises(ConfigError) as caught:
        RuntimeSpec(value, "auto", None, {})
    assert value.calls == 0
    assert "descriptor secret" not in str(caught.value)


def test_static_mapping_and_sequence_controls_keep_nominal_protocols():
    runtime = _DescriptorMapping({"platform": "auto"})
    assert parse_runtime(runtime).platform == "auto"
    assert runtime.class_calls == 0

    plugins = _DescriptorSequence(("example.plugin",))
    parsed = parse_raw_process_mapping(
        {"schema_version": 1, "plugins": plugins},
        parse_outputs=fake_parse_outputs,
    )
    assert parsed.plugins == ("example.plugin",)
    assert plugins.class_calls == 0


def test_raw_plugins_preserve_stdlib_virtual_sequence_compatibility():
    parsed = parse_raw_process_mapping(
        {
            "schema_version": 1,
            "plugins": deque(("example.plugin",)),
        },
        parse_outputs=fake_parse_outputs,
    )
    assert parsed.plugins == ("example.plugin",)


def test_raw_plugins_preserve_getitem_only_ordered_sequence_compatibility():
    class GetitemOnly:
        def __init__(self):
            self.values = ("example.plugin",)

        def __getitem__(self, index):
            return self.values[index]

        def __len__(self):
            return len(self.values)

    parsed = parse_raw_process_mapping(
        {"schema_version": 1, "plugins": GetitemOnly()},
        parse_outputs=fake_parse_outputs,
    )
    assert parsed.plugins == ("example.plugin",)


@pytest.mark.parametrize(
    "plugins",
    (
        {"example.alpha", "example.beta"},
        frozenset(("example.alpha", "example.beta")),
        iter(("example.alpha", "example.beta")),
    ),
    ids=("set", "frozenset", "generator"),
)
def test_raw_plugins_reject_unordered_or_streaming_iterables(plugins):
    with pytest.raises(
        ConfigError, match="plugins: must be a list of Python module names"
    ):
        parse_raw_process_mapping(
            {"schema_version": 1, "plugins": plugins},
            parse_outputs=fake_parse_outputs,
        )


def test_unordered_plugin_refusal_is_independent_of_python_hash_seed():
    source = str(Path(__file__).parents[2] / "src")
    script = """
from _rheplicant_bootstrap.errors import ConfigError
from _rheplicant_bootstrap.process import parse_raw_process_mapping

try:
    parse_raw_process_mapping(
        {
            "schema_version": 1,
            "plugins": {"example.alpha", "example.beta"},
        },
        parse_outputs=lambda raw: (),
    )
except ConfigError as error:
    print(error)
else:
    raise SystemExit("unordered plugins were accepted")
"""
    observed = set()
    for seed in ("1", "2", "3"):
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": source},
        )
        assert completed.returncode == 0, completed.stderr
        observed.add(completed.stdout.strip())
    assert observed == {"plugins: must be a list of Python module names."}


def test_iterable_only_plugins_are_rejected_without_iteration():
    class IterableOnly:
        calls = 0

        def __iter__(self):
            type(self).calls += 1
            raise AssertionError("unordered iterable was consumed")

    with pytest.raises(
        ConfigError, match="plugins: must be a list of Python module names"
    ):
        parse_raw_process_mapping(
            {"schema_version": 1, "plugins": IterableOnly()},
            parse_outputs=fake_parse_outputs,
        )
    assert IterableOnly.calls == 0


@pytest.mark.parametrize("plugins", ({"example.plugin": 1},))
def test_mapping_plugins_are_not_misclassified_as_sequences(plugins):
    with pytest.raises(
        ConfigError, match="plugins: must be a list of Python module names"
    ):
        parse_raw_process_mapping(
            {"schema_version": 1, "plugins": plugins},
            parse_outputs=fake_parse_outputs,
        )


def test_items_only_mapping_plugins_are_not_misclassified_as_sequences():
    class ItemsOnly:
        def items(self):
            return (("example.plugin", 1),)

        def __getitem__(self, index):
            return "example.plugin"

        def __len__(self):
            return 1

    with pytest.raises(
        ConfigError, match="plugins: must be a list of Python module names"
    ):
        parse_raw_process_mapping(
            {"schema_version": 1, "plugins": ItemsOnly()},
            parse_outputs=fake_parse_outputs,
        )


def test_plugin_sequence_budget_precedes_processing_the_limit_plus_one_item(
    monkeypatch,
):
    from _rheplicant_bootstrap import process as process_module

    class HostileItem:
        class_calls = 0

        @property
        def __class__(self):
            type(self).class_calls += 1
            raise AssertionError("limit-plus-one item was inspected")

    hostile = HostileItem()

    class InfiniteOrdered:
        next_calls = 0

        def __len__(self):
            return 0

        def __getitem__(self, index):
            return "unused"

        def __iter__(self):
            while True:
                type(self).next_calls += 1
                yield "example.plugin" if self.next_calls == 1 else hostile

    monkeypatch.setattr(process_module, "_PROCESS_SEQUENCE_ENTRY_LIMIT", 1)
    with pytest.raises(ConfigError, match="entry count exceeds limit 1"):
        parse_raw_process_mapping(
            {"schema_version": 1, "plugins": InfiniteOrdered()},
            parse_outputs=fake_parse_outputs,
        )
    assert InfiniteOrdered.next_calls == 2
    assert HostileItem.class_calls == 0


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


class _ProcessItemsOnly:
    def __init__(self, pairs):
        self.pairs = tuple(pairs)

    def items(self):
        return iter(self.pairs)


class _ProcessBrokenPair:
    def __init__(self):
        self.unpack_calls = 0

    def __iter__(self):
        self.unpack_calls += 1
        raise AssertionError("over-budget process pair was unpacked")


@pytest.mark.parametrize("route", ("root", "variants", "patch"))
def test_process_mapping_budgets_precede_limit_plus_one_unpack(
    route,
    monkeypatch,
):
    from _rheplicant_bootstrap import process as process_module

    broken = _ProcessBrokenPair()
    if route == "root":
        document = _ProcessItemsOnly((("variants", {}), broken))
    elif route == "variants":
        variants = _ProcessItemsOnly((("first", {}), broken))
        document = _ProcessItemsOnly((("variants", variants),))
    else:
        patch = _ProcessItemsOnly((("runtime", {}), broken))
        variants = _ProcessItemsOnly((("first", patch),))
        document = _ProcessItemsOnly((("variants", variants),))
    monkeypatch.setattr(
        process_module,
        "_PROCESS_MAPPING_ENTRY_LIMIT",
        1,
        raising=False,
    )

    with pytest.raises(ConfigError, match="entry count exceeds limit 1"):
        validate_variant_process_sections(document)
    assert broken.unpack_calls == 0


def test_public_variant_validator_consumes_only_the_items_protocol():
    document = _ProcessItemsOnly(
        (
            (
                "variants",
                _ProcessItemsOnly(
                    (("safe", _ProcessItemsOnly(())),)
                ),
            ),
        )
    )
    validate_variant_process_sections(document)


@pytest.mark.parametrize(
    ("failure", "is_base"),
    ((ValueError("private"), False), (KeyboardInterrupt(), True)),
)
def test_public_variant_validator_has_a_static_items_failure_boundary(
    failure,
    is_base,
):
    class FailingItems:
        def items(self):
            raise failure

    if is_base:
        with pytest.raises(KeyboardInterrupt):
            validate_variant_process_sections(FailingItems())
    else:
        with pytest.raises(ConfigError, match="mapping traversal") as caught:
            validate_variant_process_sections(FailingItems())
        assert "private" not in str(caught.value)


@pytest.mark.parametrize("callback_field", ("raw_runtime", "outputs"))
def test_raw_process_record_restores_its_complete_entry_snapshot(callback_field):
    record = object.__new__(RawProcessEntry)

    class MutatingItems:
        def items(self):
            object.__setattr__(record, "schema_version", 2)
            object.__setattr__(record, "defaults", ())
            object.__setattr__(record, "plugins", ("mutated.plugin",))
            object.__setattr__(record, "raw_runtime", None)
            object.__setattr__(record, "outputs", None)
            return (("stable", True),)

    raw_runtime = MutatingItems() if callback_field == "raw_runtime" else {}
    outputs = MutatingItems() if callback_field == "outputs" else {}
    RawProcessEntry.__init__(
        record,
        1,
        (PresetRequest("base", None),),
        ("base.plugin",),
        raw_runtime,
        outputs,
    )

    assert record.schema_version == 1
    assert tuple(item.name for item in record.defaults) == ("base",)
    assert record.plugins == ("base.plugin",)
    assert record.raw_runtime == (
        {"stable": True} if callback_field == "raw_runtime" else {}
    )
    assert record.outputs == (
        {"stable": True} if callback_field == "outputs" else {}
    )


def test_runtime_spec_restores_all_fields_after_seed_protocol_callbacks():
    spec = object.__new__(RuntimeSpec)

    class MutatingSeeds:
        def items(self):
            object.__setattr__(spec, "jax_enable_x64", "forged")
            object.__setattr__(spec, "platform", "cpu")
            object.__setattr__(spec, "seed", 99)
            object.__setattr__(spec, "seeds", {})
            return (("main", 2),)

    RuntimeSpec.__init__(spec, False, "auto", 1, MutatingSeeds())
    assert spec.jax_enable_x64 is False
    assert spec.platform == "auto"
    assert spec.seed == 1
    assert spec.seeds == {"main": 2}


def test_effective_process_record_restores_fields_after_output_callbacks():
    entry = object.__new__(EffectiveProcessEntry)
    runtime = RuntimeSpec(False, "auto", None, {})

    class MutatingOutput:
        def items(self):
            object.__setattr__(entry, "runtime", object())
            object.__setattr__(entry, "plugins", ("mutated.plugin",))
            object.__setattr__(entry, "outputs", None)
            return (("stable", True),)

    EffectiveProcessEntry.__init__(
        entry,
        runtime,
        ("base.plugin",),
        MutatingOutput(),
    )
    assert type(entry.runtime) is RuntimeSpec
    assert entry.runtime is not runtime
    assert entry.runtime.platform == "auto"
    assert entry.plugins == ("base.plugin",)
    assert entry.outputs == {"stable": True}


def test_runtime_direct_construction_freezes_seeds():
    seeds = {"main": 1}
    parsed = RuntimeSpec(False, "auto", None, seeds)
    seeds["main"] = 2
    assert parsed.seeds == {"main": 1}


@pytest.mark.parametrize(
    ("section", "message"),
    [
        (None, "runtime: is a mapping; got NoneType (None)."),
        (
            {"jax_enable_x64": 1},
            "runtime.jax_enable_x64 is a bool; got int (1).",
        ),
        (
            {"jax_enable_x64": {}},
            "runtime.jax_enable_x64 is a bool; got dict ({}).",
        ),
        (
            {"platform": "cuda"},
            "runtime.platform is one of ['auto', 'cpu', 'gpu', 'tpu']; "
            "got 'cuda'.",
        ),
        (
            {"platform": 1},
            "runtime.platform is one of ['auto', 'cpu', 'gpu', 'tpu']; got 1.",
        ),
        (
            {"platform": None},
            "runtime.platform is one of ['auto', 'cpu', 'gpu', 'tpu']; got None.",
        ),
        (
            {"platform": []},
            "runtime.platform is one of ['auto', 'cpu', 'gpu', 'tpu']; got [].",
        ),
        (
            {"platform": {}},
            "runtime.platform is one of ['auto', 'cpu', 'gpu', 'tpu']; got {}.",
        ),
        (
            {"seed": True},
            "runtime.seed is an int or null; got bool (True). null is legal "
            "and recorded -- it means State.key = None, and a run that "
            "realises randomness will say so when it asks for the key.",
        ),
        (
            {"seeds": []},
            "runtime.seeds is a mapping of name -> int; got list ([]).",
        ),
        (
            {"unknown": 1},
            "runtime: the runtime section does not take ['unknown']; it takes "
            "['jax_enable_x64', 'platform', 'seed', 'seeds'].",
        ),
        (
            {"x64_required_by": ["projector"]},
            "runtime: the runtime section does not take "
            "['x64_required_by']; it takes ['jax_enable_x64', 'platform', "
            "'seed', 'seeds']. x64_required_by is emitted by the loader into "
            "the resolved record, never written by hand.",
        ),
        (
            {"seeds": {"sample": 1.5}},
            "runtime.seeds.sample is an int; got float (1.5).",
        ),
        (
            {"seeds": {1: 1}},
            "runtime.seeds keys are strings; got 1.",
        ),
    ],
)
def test_runtime_grammar_is_closed_without_importing_jax(section, message):
    with pytest.raises(ConfigError) as caught:
        parse_runtime(section)
    assert str(caught.value) == message


def test_runtime_grammar_never_formats_hostile_custom_values():
    class Hostile:
        def __repr__(self):
            raise AssertionError("repr must not run")

        def __str__(self):
            raise AssertionError("str must not run")

    with pytest.raises(ConfigError) as caught:
        parse_runtime(Hostile())
    assert str(caught.value) == "runtime: is a mapping; got Hostile."

    with pytest.raises(ConfigError) as caught:
        parse_runtime({"jax_enable_x64": Hostile()})
    assert str(caught.value) == (
        "runtime.jax_enable_x64 is a bool; got Hostile."
    )


def test_runtime_type_diagnostics_never_call_a_metaclass_name_descriptor():
    descriptor_calls = 0

    class HostileMeta(type):
        @property
        def __name__(cls):
            nonlocal descriptor_calls
            descriptor_calls += 1
            return "ForgedName"

    class Hostile(metaclass=HostileMeta):
        pass

    hostile = Hostile()
    cases = (
        (hostile, "runtime: is a mapping; got Hostile."),
        (
            {"jax_enable_x64": hostile},
            "runtime.jax_enable_x64 is a bool; got Hostile.",
        ),
        (
            {"seed": hostile},
            "runtime.seed is an int or null; got Hostile. null is legal and "
            "recorded -- it means State.key = None, and a run that realises "
            "randomness will say so when it asks for the key.",
        ),
        (
            {"seeds": hostile},
            "runtime.seeds is a mapping of name -> int; got Hostile.",
        ),
    )

    for section, expected in cases:
        with pytest.raises(ConfigError) as caught:
            parse_runtime(section)
        if str(caught.value) != expected:
            pytest.fail(f"unexpected static diagnostic: {caught.value.args!r}")
    if descriptor_calls != 0:
        pytest.fail(f"metaclass __name__ descriptor ran {descriptor_calls} times")


def test_process_and_origin_type_diagnostics_use_only_static_class_metadata():
    descriptor_calls = 0

    class HostileMeta(type):
        @property
        def __name__(cls):
            nonlocal descriptor_calls
            descriptor_calls += 1
            return "ForgedName"

    class Hostile(metaclass=HostileMeta):
        pass

    class Pairs(Mapping):
        def __init__(self, pairs):
            self._pairs = pairs

        def __getitem__(self, key):
            raise KeyError(key)

        def __iter__(self):
            return iter(())

        def __len__(self):
            return len(self._pairs)

        def items(self):
            return iter(self._pairs)

    hostile = Hostile()

    def hostile_parser(_raw):
        class HostileError(RuntimeError, metaclass=HostileMeta):
            pass

        raise HostileError("foreign")

    routes = (
        lambda: parse_raw_process_mapping(
            hostile, parse_outputs=fake_parse_outputs
        ),
        lambda: parse_raw_process_mapping(
            Pairs(((hostile, 1),)), parse_outputs=fake_parse_outputs
        ),
        lambda: parse_raw_process_mapping(
            {"schema_version": 1, "plugins": [hostile]},
            parse_outputs=fake_parse_outputs,
        ),
        lambda: parse_raw_process_mapping(
            {"schema_version": 1}, parse_outputs=hostile_parser
        ),
        lambda: parse_runtime(Pairs(((hostile, 1),))),
        lambda: validate_variant_process_sections({"variants": hostile}),
        lambda: validate_variant_process_sections(
            {"variants": Pairs(((hostile, {}),))}
        ),
    )
    for route in routes:
        with pytest.raises(ConfigError) as caught:
            route()
        if "Hostile" not in str(caught.value):
            pytest.fail(f"static type name missing from {caught.value.args!r}")

    for route in (lambda: Origin(hostile), lambda: Origin("user", hostile)):
        with pytest.raises(ValueError) as caught:
            route()
        if "Hostile" not in str(caught.value):
            pytest.fail(f"static Origin type name missing from {caught.value.args!r}")
    if descriptor_calls != 0:
        pytest.fail(f"metaclass __name__ descriptor ran {descriptor_calls} times")


@pytest.mark.parametrize("location", ["root", "seeds"])
def test_runtime_error_rendering_is_bounded_for_cyclic_sequences(location):
    recursive = []
    recursive.append(recursive)
    section = recursive if location == "root" else {"seeds": recursive}

    with pytest.raises(ConfigError) as caught:
        parse_runtime(section)
    assert "list" in str(caught.value)
    assert len(str(caught.value)) < 200


def test_runtime_error_rendering_is_bounded_for_deep_and_wide_sequences():
    deep = []
    for _ in range(200):
        deep = [deep]
    wide = list(range(10_000))

    for section in ({"platform": deep}, {"seeds": wide}):
        with pytest.raises(ConfigError) as caught:
            parse_runtime(section)
        assert len(str(caught.value)) < 200


@pytest.mark.parametrize(
    "value", [10**5000, "x" * 10_000], ids=("integer", "string")
)
def test_runtime_error_rendering_is_bounded_for_oversized_scalars(value):
    with pytest.raises(ConfigError) as caught:
        parse_runtime({"jax_enable_x64": value})
    assert "jax_enable_x64" in str(caught.value)
    assert len(str(caught.value)) < 200


def test_runtime_renderer_rejects_oversized_text_before_repr(monkeypatch):
    from _rheplicant_bootstrap import process as process_module

    def forbidden_repr(_value):
        pytest.fail("oversized runtime text reached repr")

    monkeypatch.setattr(process_module, "repr", forbidden_repr, raising=False)
    with pytest.raises(ConfigError) as caught:
        parse_runtime({"jax_enable_x64": "x" * 121})
    assert str(caught.value) == (
        "runtime.jax_enable_x64 is a bool; got str."
    )


class _CollidingSeedNames(Mapping[str, object]):
    def __getitem__(self, key):
        raise KeyError(key)

    def __iter__(self):
        return iter(())

    def __len__(self):
        return 2

    def items(self):
        class Text(str):
            pass

        return [("same", 1), (Text("same"), 2)]


def test_runtime_seed_names_cannot_collide_after_canonicalization():
    with pytest.raises(ConfigError, match="collide"):
        parse_runtime({"seeds": _CollidingSeedNames()})


class _TooManySeeds(Mapping[str, object]):
    def __getitem__(self, key):
        raise KeyError(key)

    def __iter__(self):
        return iter(())

    def __len__(self):
        return 250_001

    def items(self):
        return ((f"seed_{index}", index) for index in range(250_001))


def test_runtime_seed_mapping_has_a_bounded_emission_budget():
    with pytest.raises(ConfigError, match="entry count"):
        parse_runtime({"seeds": _TooManySeeds()})


class _TooManyRuntimeEntries(Mapping[str, object]):
    def __getitem__(self, key):
        raise KeyError(key)

    def __iter__(self):
        return iter(())

    def __len__(self):
        return 250_001

    def items(self):
        return ((f"unknown_{index}", index) for index in range(250_001))


def test_runtime_top_level_mapping_has_a_bounded_emission_budget():
    with pytest.raises(ConfigError, match="entry count"):
        parse_runtime(_TooManyRuntimeEntries())


class _RuntimeBrokenPair:
    def __init__(self, error=None):
        self.calls = 0
        self.error = error

    def __iter__(self):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return iter(("hidden", 1))


class _RuntimePairMapping(Mapping[str, object]):
    def __init__(self, pairs):
        self.pairs = tuple(pairs)

    def __getitem__(self, key):
        raise KeyError(key)

    def __iter__(self):
        return iter(())

    def __len__(self):
        return len(self.pairs)

    def items(self):
        return iter(self.pairs)


@pytest.mark.parametrize("route", ("runtime", "seeds"))
def test_runtime_budgets_limit_plus_one_before_pair_unpack(route, monkeypatch):
    from _rheplicant_bootstrap import process as process_module

    broken = _RuntimeBrokenPair()
    first = ("platform", "auto") if route == "runtime" else ("first", 1)
    mapping = _RuntimePairMapping((first, broken))
    section = mapping if route == "runtime" else {"seeds": mapping}
    monkeypatch.setattr(process_module, "_RUNTIME_MAPPING_ENTRY_LIMIT", 1)

    with pytest.raises(ConfigError, match="entry count exceeds limit 1"):
        parse_runtime(section)
    assert broken.calls == 0


@pytest.mark.parametrize("route", ("runtime", "seeds"))
def test_runtime_bounded_pair_unpack_propagates_baseexception(
    route,
    monkeypatch,
):
    from _rheplicant_bootstrap import process as process_module

    broken = _RuntimeBrokenPair(KeyboardInterrupt())
    mapping = _RuntimePairMapping((broken,))
    section = mapping if route == "runtime" else {"seeds": mapping}
    monkeypatch.setattr(process_module, "_RUNTIME_MAPPING_ENTRY_LIMIT", 1)

    with pytest.raises(KeyboardInterrupt):
        parse_runtime(section)
    assert broken.calls == 1


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


def test_effective_layer_kind_is_canonicalized_before_comparison():
    comparison_calls = 0

    class HostileKind(str):
        def __ne__(self, other):
            nonlocal comparison_calls
            comparison_calls += 1
            raise SystemExit("kind comparison hook must not run")

    raw = parse_raw_process_mapping(
        {"schema_version": 1}, parse_outputs=fake_parse_outputs
    )
    layer = LayerRef(
        kind="variant",
        name="safe",
        prefix="variants.safe",
        document={"runtime": {}},
        declared_runs=(),
    )
    object.__setattr__(layer, "kind", HostileKind("variant"))

    parsed = parse_effective_process_mapping(
        {"schema_version": 1, "runtime": {}},
        (layer,),
        raw=raw,
        parse_outputs=fake_parse_outputs,
    )

    assert parsed.runtime.platform == "auto"
    assert comparison_calls == 0


def test_effective_layer_name_is_canonicalized_before_diagnostics():
    format_calls = 0

    class HostileName(str):
        def __format__(self, format_spec):
            nonlocal format_calls
            format_calls += 1
            raise AssertionError("name formatting hook must not run")

    raw = parse_raw_process_mapping(
        {"schema_version": 1}, parse_outputs=fake_parse_outputs
    )
    layer = LayerRef(
        kind="variant",
        name="safe",
        prefix="variants.safe",
        document={},
        declared_runs=(),
    )
    object.__setattr__(layer, "name", HostileName("safe"))

    with pytest.raises(ConfigError) as caught:
        parse_effective_process_mapping(
            {"schema_version": 1, "runtime": {}},
            (layer,),
            raw=raw,
            parse_outputs=fake_parse_outputs,
        )

    assert "variants.safe.runtime" in str(caught.value)
    assert format_calls == 0


def test_effective_layer_identities_are_snapshotted_before_document_callbacks():
    comparison_calls = 0

    class HostileKind(str):
        def __ne__(self, other):
            nonlocal comparison_calls
            comparison_calls += 1
            raise AssertionError("late kind comparison hook must not run")

    second = LayerRef(
        kind="variant",
        name="second",
        prefix="variants.second",
        document={"runtime": {}},
        declared_runs=(),
    )

    class MutatingDocument:
        def items(self):
            object.__setattr__(second, "kind", HostileKind("variant"))
            return (("runtime", {}),)

    first = LayerRef(
        kind="variant",
        name="first",
        prefix="variants.first",
        document={},
        declared_runs=(),
    )
    object.__setattr__(first, "document", MutatingDocument())
    raw = parse_raw_process_mapping(
        {"schema_version": 1}, parse_outputs=fake_parse_outputs
    )

    parsed = parse_effective_process_mapping(
        {"schema_version": 1, "runtime": {}},
        (first, second),
        raw=raw,
        parse_outputs=fake_parse_outputs,
    )

    assert parsed.runtime.platform == "auto"
    assert comparison_calls == 0


def test_effective_variant_layer_reads_the_proved_items_protocol_not_get():
    class ItemsOnly:
        def items(self):
            return (("runtime", {}),)

    raw = parse_raw_process_mapping(
        {"schema_version": 1}, parse_outputs=fake_parse_outputs
    )
    layer = LayerRef(
        kind="variant",
        name="duck",
        prefix="variants.duck",
        document={},
        declared_runs=(),
    )
    object.__setattr__(layer, "document", ItemsOnly())

    parsed = parse_effective_process_mapping(
        {"schema_version": 1, "runtime": {}},
        (layer,),
        raw=raw,
        parse_outputs=fake_parse_outputs,
    )
    assert parsed.runtime.platform == "auto"


@pytest.mark.parametrize(
    ("failure", "is_base"),
    ((ValueError("private"), False), (KeyboardInterrupt(), True)),
)
def test_effective_variant_layer_items_failure_has_a_controlled_boundary(
    failure,
    is_base,
):
    class FailingItems:
        def items(self):
            raise failure

    raw = parse_raw_process_mapping(
        {"schema_version": 1}, parse_outputs=fake_parse_outputs
    )
    layer = LayerRef(
        kind="variant",
        name="broken",
        prefix="variants.broken",
        document={},
        declared_runs=(),
    )
    object.__setattr__(layer, "document", FailingItems())

    if not is_base:
        with pytest.raises(ConfigError, match="layer mapping traversal"):
            parse_effective_process_mapping(
                {"schema_version": 1, "runtime": {}},
                (layer,),
                raw=raw,
                parse_outputs=fake_parse_outputs,
            )
    else:
        with pytest.raises(KeyboardInterrupt):
            parse_effective_process_mapping(
                {"schema_version": 1, "runtime": {}},
                (layer,),
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


def test_effective_parser_bounds_a_forged_raw_plugin_sequence(monkeypatch):
    from _rheplicant_bootstrap import process as process_module

    class InfiniteOrdered:
        next_calls = 0

        def __len__(self):
            return 0

        def __getitem__(self, index):
            return "unused"

        def __iter__(self):
            while True:
                type(self).next_calls += 1
                yield "base.plugin"

    raw = parse_raw_process_mapping(
        {"schema_version": 1}, parse_outputs=fake_parse_outputs
    )
    object.__setattr__(raw, "plugins", InfiniteOrdered())
    monkeypatch.setattr(process_module, "_PROCESS_SEQUENCE_ENTRY_LIMIT", 1)

    with pytest.raises(ConfigError, match="entry count exceeds limit 1"):
        parse_effective_process_mapping(
            {"schema_version": 1, "runtime": {}},
            (),
            raw=raw,
            parse_outputs=fake_parse_outputs,
        )
    assert InfiniteOrdered.next_calls == 2


def test_effective_parser_snapshots_raw_plugins_before_document_protocols():
    raw = parse_raw_process_mapping(
        {"schema_version": 1, "plugins": ["base.plugin"]},
        parse_outputs=fake_parse_outputs,
    )

    class MutatingDocument:
        def items(self):
            object.__setattr__(raw, "plugins", ("after.plugin",))
            return (
                ("schema_version", 1),
                ("plugins", ["base.plugin"]),
                ("runtime", {}),
            )

    parsed = parse_effective_process_mapping(
        MutatingDocument(),
        (),
        raw=raw,
        parse_outputs=fake_parse_outputs,
    )
    assert raw.plugins == ("after.plugin",)
    assert parsed.plugins == ("base.plugin",)


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
