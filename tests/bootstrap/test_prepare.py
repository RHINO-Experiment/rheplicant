"""The sole raw -> preset -> effective Config Plan 4A preparation pipeline."""

from __future__ import annotations

import hashlib
import subprocess
import sys
import tracemalloc
from collections import Counter
from collections.abc import Mapping, Sequence
from time import perf_counter
from types import MappingProxyType

import pytest

from _rheplicant_bootstrap import frozen as frozen_module
from _rheplicant_bootstrap import layering as layering_module
from _rheplicant_bootstrap import prepare as prepare_module
from _rheplicant_bootstrap.errors import ConfigError
from _rheplicant_bootstrap.frozen import thaw
from _rheplicant_bootstrap.layering import OriginNode, initial_merge
from _rheplicant_bootstrap.prepare import (
    BootstrapManifest,
    ConfigSource,
    PreparedConfig,
    SelectedPreset,
    prepare_config,
)
from _rheplicant_bootstrap.presets import PresetRequest, PresetSnapshot
from _rheplicant_bootstrap.types import Origin, SourceInput
from _rheplicant_bootstrap.variants import LayerEnumeration


def fake_parse_outputs(raw_outputs: object) -> object:
    if not isinstance(raw_outputs, Mapping):
        raise ConfigError("outputs: test grammar requires a mapping or omission.")
    return tuple(raw_outputs.items())


def source(raw: bytes) -> SourceInput:
    return SourceInput(
        input_bytes=raw,
        source_path="/tmp/config.yaml",
        source_realpath="/tmp/config.yaml",
        source_name="/tmp/config.yaml",
        base_dir="/tmp",
        launch_mode="cli",
    )


def _physical_evidence_slots(*roots: object) -> int:
    """Count unique built-in/private container reference slots."""
    overlay_type = getattr(layering_module, "_OverlayMapping", ())
    concat_type = getattr(frozen_module, "_FrozenConcat", ())
    pending = list(roots)
    seen: set[int] = set()
    slots = 0
    while pending:
        value = pending.pop()
        identity = id(value)
        if identity in seen:
            continue
        seen.add(identity)
        if overlay_type and type(value) is overlay_type:
            slots += (
                len(value._delta) + len(value._hidden) + len(value._end_keys)
            )
            pending.append(value._base)
            pending.extend(value._delta.values())
        elif type(value) is MappingProxyType:
            slots += len(value)
            pending.extend(value.values())
        elif concat_type and type(value) is concat_type:
            slots += len(value._suffix)
            pending.append(value._parent)
            pending.extend(value._suffix)
        elif type(value) is tuple:
            slots += tuple.__len__(value)
            pending.extend(value)
        elif type(value) is OriginNode:
            pending.append(value.children)
    return slots


def snapshot(name: str, *, nodes: int = 1, raw: bytes = b"x") -> PresetSnapshot:
    return PresetSnapshot(
        name=name,
        resource=f"rheplicant/config/presets/{name}.yaml",
        input_bytes=raw,
        sha256=hashlib.sha256(raw).hexdigest(),
        document={"runtime": {"jax_enable_x64": True}},
        expanded_nodes=nodes,
    )


def no_presets(name: str) -> PresetSnapshot:
    raise AssertionError(f"unexpected preset request {name!r}")


def jax_modules() -> frozenset[str]:
    import sys

    return frozenset(
        name for name in sys.modules if name == "jax" or name.startswith("jax.")
    )


def test_prepare_records_only_the_four_successful_boundaries_in_order():
    boundaries: list[tuple[str, object]] = []
    prepared = prepare_config(
        source(b"schema_version: 1\nruntime: {}\nruns: []\n"),
        preset_provider=no_presets,
        parse_outputs=fake_parse_outputs,
        boundary_completed=lambda stage, layer: boundaries.append((stage, layer)),
    )

    assert boundaries == [
        ("source", None),
        ("raw_process_entry", None),
        ("preset_layering", None),
        ("effective_process_entry", None),
    ]
    assert prepared.source.input_bytes == (
        b"schema_version: 1\nruntime: {}\nruns: []\n"
    )
    assert prepared.source.bootstrap_manifest.input_sha256 == hashlib.sha256(
        prepared.source.input_bytes
    ).hexdigest()
    assert len(prepared.layers) == 1


def test_prepare_data_is_frozen_and_builder_copies_are_detached():
    prepared = prepare_config(
        source(
            b"schema_version: 1\nruntime:\n  seeds: {a: 1}\nruns: []\n"
        ),
        preset_provider=no_presets,
        parse_outputs=fake_parse_outputs,
    )
    layer = prepared.layers[0]
    copy = layer.mutable_document()
    copy["runtime"]["seeds"]["a"] = 2

    assert thaw(layer.document)["runtime"]["seeds"] == {"a": 1}
    with pytest.raises(TypeError):
        layer.document["runtime"] = {}


def test_variant_process_sections_and_runtime_disagreements_are_early():
    raw = b"""schema_version: 1
runtime: {platform: cpu}
runs: []
outputs: {dir: here}
variants:
  x: {outputs: {dir: elsewhere}}
  y: {runtime: {platform: gpu}}
"""
    before = jax_modules()
    with pytest.raises(ConfigError) as caught:
        prepare_config(
            source(raw),
            preset_provider=no_presets,
            parse_outputs=fake_parse_outputs,
        )
    assert str(caught.value) == (
        "variants.x.outputs: variants may not change process-entry section "
        "'outputs'; edit the base outputs section."
    )
    assert jax_modules() == before


@pytest.mark.parametrize(
    ("section", "expected_boundaries"),
    [
        ("runtime", [("source", None)]),
        (
            "variants",
            [
                ("source", None),
                ("raw_process_entry", None),
                ("preset_layering", None),
            ],
        ),
    ],
)
def test_explicit_null_is_refused_at_its_first_validation_boundary(
    section, expected_boundaries
):
    boundaries: list[tuple[str, object]] = []
    raw = f"schema_version: 1\n{section}: null\nruns: []\n".encode()

    with pytest.raises(ConfigError):
        prepare_config(
            source(raw),
            preset_provider=no_presets,
            parse_outputs=fake_parse_outputs,
            boundary_completed=lambda stage, layer: boundaries.append(
                (stage, layer)
            ),
        )

    assert boundaries == expected_boundaries


def test_all_runtime_disagreements_are_named_before_jax_import():
    raw = b"""schema_version: 1
runtime: {jax_enable_x64: true, platform: cpu}
runs: []
variants:
  x: {runtime: {platform: gpu}}
  y: {runtime: {jax_enable_x64: false}}
"""
    before = jax_modules()
    with pytest.raises(ConfigError) as caught:
        prepare_config(
            source(raw),
            preset_provider=no_presets,
            parse_outputs=fake_parse_outputs,
        )
    assert str(caught.value) == (
        "runtime: process-global settings disagree: runtime.platform='cpu', "
        "variants.x.runtime.platform='gpu'; runtime.jax_enable_x64=true, "
        "variants.y.runtime.jax_enable_x64=false."
    )
    assert jax_modules() == before


def test_aggregate_budget_refuses_the_crossing_snapshot_before_layering_gets_it(
    monkeypatch,
):
    raw = b"schema_version: 1\ndefaults: [one, two]\nruns: []\n"
    snapshots = {
        "one": snapshot("one", nodes=125_000, raw=b"one"),
        "two": snapshot("two", nodes=125_000, raw=b"two"),
    }
    provider_calls: Counter[str] = Counter()
    returned_to_layering: list[str] = []

    def provider(name: str) -> PresetSnapshot:
        provider_calls[name] += 1
        return snapshots[name]

    real_layer_presets = prepare_module.layer_presets

    def observing_layer_presets(document, requests, *, preset_provider):
        def observe(name):
            chosen = preset_provider(name)
            returned_to_layering.append(name)
            return chosen

        return real_layer_presets(
            document, requests, preset_provider=observe
        )

    monkeypatch.setattr(prepare_module, "layer_presets", observing_layer_presets)

    with pytest.raises(ConfigError, match="aggregate YAML expanded nodes"):
        prepare_config(
            source(raw),
            preset_provider=provider,
            parse_outputs=fake_parse_outputs,
        )

    assert provider_calls == Counter(one=1, two=1)
    assert returned_to_layering == ["one"]


def test_aggregate_byte_budget_refuses_crossing_snapshot_before_merge(
    monkeypatch,
):
    raw = b"schema_version: 1\ndefaults: [one, two]\nruns: []\n"
    half_limit = b"x" * (8 * 1024 * 1024)
    snapshots = {
        "one": snapshot("one", raw=half_limit),
        "two": snapshot("two", raw=half_limit),
    }
    provider_calls: Counter[str] = Counter()
    returned_to_layering: list[str] = []

    def provider(name: str) -> PresetSnapshot:
        provider_calls[name] += 1
        return snapshots[name]

    real_layer_presets = prepare_module.layer_presets

    def observing_layer_presets(document, requests, *, preset_provider):
        def observe(name):
            chosen = preset_provider(name)
            returned_to_layering.append(name)
            return chosen

        return real_layer_presets(document, requests, preset_provider=observe)

    monkeypatch.setattr(prepare_module, "layer_presets", observing_layer_presets)

    with pytest.raises(ConfigError, match="aggregate YAML byte count"):
        prepare_config(
            source(raw),
            preset_provider=provider,
            parse_outputs=fake_parse_outputs,
        )

    assert provider_calls == Counter(one=1, two=1)
    assert returned_to_layering == ["one"]


def test_source_yaml_and_every_selected_preset_are_consumed_once(monkeypatch):
    raw = b"schema_version: 1\ndefaults: [one]\nruns: []\n"
    yaml_calls = 0
    provider_calls = 0
    real_safe_load = prepare_module.safe_load_document

    def counted_safe_load(*args, **kwargs):
        nonlocal yaml_calls
        yaml_calls += 1
        return real_safe_load(*args, **kwargs)

    def provider(name: str) -> PresetSnapshot:
        nonlocal provider_calls
        provider_calls += 1
        return snapshot(name)

    monkeypatch.setattr(prepare_module, "safe_load_document", counted_safe_load)
    prepared = prepare_config(
        source(raw),
        preset_provider=provider,
        parse_outputs=fake_parse_outputs,
    )

    assert yaml_calls == 1
    assert provider_calls == 1
    assert prepared.source.bootstrap_manifest.presets[0].snapshot.input_bytes == b"x"


def test_budget_rejects_a_snapshot_subclass_before_reading_its_fields():
    class HostileSnapshot(PresetSnapshot):
        def __getattribute__(self, name):
            if name in {"input_bytes", "expanded_nodes"}:
                state = object.__getattribute__(self, "__dict__")
                state["budget_field_reads"] = state.get(
                    "budget_field_reads", 0
                ) + 1
            return super().__getattribute__(name)

    given = HostileSnapshot(
        name="one",
        resource="rheplicant/config/presets/one.yaml",
        input_bytes=b"x",
        sha256=hashlib.sha256(b"x").hexdigest(),
        document={"runtime": {"jax_enable_x64": True}},
        expanded_nodes=1,
    )
    object.__setattr__(given, "budget_field_reads", 0)

    with pytest.raises(ConfigError) as caught:
        prepare_config(
            source(b"schema_version: 1\ndefaults: [one]\nruns: []\n"),
            preset_provider=lambda name: given,
            parse_outputs=fake_parse_outputs,
        )

    assert given.budget_field_reads == 0
    assert str(caught.value) == (
        "defaults: preset provider for 'one' must return PresetSnapshot."
    )


def test_prepare_rejects_a_source_subclass_before_reading_its_fields():
    class HostileSource(SourceInput):
        def __getattribute__(self, name):
            if name in {"input_bytes", "source_name"}:
                state = object.__getattribute__(self, "__dict__")
                state["source_field_reads"] = state.get(
                    "source_field_reads", 0
                ) + 1
            return super().__getattribute__(name)

    given = HostileSource(
        input_bytes=b"schema_version: 1\nruntime: {}\nruns: []\n",
        source_path="/tmp/config.yaml",
        source_realpath="/tmp/config.yaml",
        source_name="/tmp/config.yaml",
        base_dir="/tmp",
        launch_mode="cli",
    )
    object.__setattr__(given, "source_field_reads", 0)

    with pytest.raises(ConfigError, match="source must be a SourceInput record"):
        prepare_config(
            given,
            preset_provider=no_presets,
            parse_outputs=fake_parse_outputs,
        )

    assert given.source_field_reads == 0


@pytest.mark.parametrize(
    "patch",
    [
        {"source_path": "config.yaml", "source_name": "config.yaml"},
        {"source_realpath": None},
        {"launch_mode": "worker"},
        {"input_bytes": object()},
        {"source_name": object()},
    ],
)
def test_invalid_source_facts_are_refused_before_any_pipeline_effect(patch):
    events: list[object] = []
    fields = dict(
        input_bytes=(
            b"schema_version: 1\ndefaults: [one]\n"
            b"runtime: {}\nruns: []\n"
        ),
        source_path="/tmp/config.yaml",
        source_realpath="/tmp/config.yaml",
        source_name="/tmp/config.yaml",
        base_dir="/tmp",
        launch_mode="cli",
    )

    def parse_outputs(value):
        events.append("parse_outputs")
        return ()

    def provider(name):
        events.append(("provider", name))
        return snapshot(name)

    with pytest.raises(ConfigError):
        prepare_config(
            SourceInput(**{**fields, **patch}),
            preset_provider=provider,
            parse_outputs=parse_outputs,
            boundary_completed=lambda stage, layer: events.append(
                (stage, layer)
            ),
        )

    assert events == []


def test_source_scalar_subclasses_and_mutable_bytes_are_copied_without_hooks():
    class HostileText(str):
        def __bool__(self):
            raise RuntimeError("hostile truth")

        def __eq__(self, other):
            raise RuntimeError("hostile equality")

    class HostileBytes(bytearray):
        def __bytes__(self):
            raise RuntimeError("hostile bytes")

        def __getitem__(self, key):
            raise RuntimeError("hostile item")

    prepared = prepare_config(
        SourceInput(
            input_bytes=HostileBytes(
                b"schema_version: 1\nruntime: {}\nruns: []\n"
            ),
            source_path=HostileText("/tmp/config.yaml"),
            source_realpath=HostileText("/tmp/config.yaml"),
            source_name=HostileText("/tmp/config.yaml"),
            base_dir=HostileText("/tmp"),
            launch_mode=HostileText("cli"),
        ),
        preset_provider=no_presets,
        parse_outputs=fake_parse_outputs,
    )

    assert type(prepared.source.input_bytes) is bytes
    assert type(prepared.source.source_path) is str
    assert type(prepared.source.bootstrap_manifest.launch_mode) is str


def test_preset_sources_are_published_by_bytes_not_only_by_name():
    """M5: a historical tree said only "rhino_v1" while the name was free to
    resolve to something else later.

    The digest could not go into ``provenance.json`` without a format bump --
    ``presets`` is ``{"items": {"type": "string"}}`` in the closed
    ``provenance-v1`` schema, and adding a key to the ``bootstrap`` object is
    refused by its ``additionalProperties: false``. Publishing the BYTES needs
    neither, is strictly more recoverable than a digest, and inherits a digest
    anyway because ``integrity.json`` covers every published file.
    """
    from types import SimpleNamespace

    from _rheplicant_bootstrap.entry import PRESETS_DIRECTORY, presets_bundle_files

    raw = b"runtime:\n  jax_enable_x64: true\n"
    one = SelectedPreset(PresetRequest("rhino_v1", None), snapshot("rhino_v1", raw=raw))
    two = SelectedPreset(PresetRequest("extra", None), snapshot("extra", raw=b"other\n"))
    prepared = SimpleNamespace(
        source=SimpleNamespace(
            bootstrap_manifest=SimpleNamespace(presets=(one, two))
        )
    )

    files = presets_bundle_files(prepared)
    assert set(files) == {
        f"{PRESETS_DIRECTORY}/rhino_v1.yaml",
        f"{PRESETS_DIRECTORY}/extra.yaml",
    }
    # The bytes, verbatim -- not a re-serialisation of the parsed document,
    # which would silently normalise whatever the installed file actually said.
    assert files[f"{PRESETS_DIRECTORY}/rhino_v1.yaml"] == raw

    # And the digest that arrives for free is the one the snapshot validated.
    assert (
        hashlib.sha256(files[f"{PRESETS_DIRECTORY}/rhino_v1.yaml"]).hexdigest()
        == one.snapshot.sha256
    )


def test_no_presets_publishes_no_preset_directory():
    """An empty mapping, so a run without presets gains no empty directory."""
    from types import SimpleNamespace

    from _rheplicant_bootstrap.entry import presets_bundle_files

    prepared = SimpleNamespace(
        source=SimpleNamespace(bootstrap_manifest=SimpleNamespace(presets=()))
    )
    assert presets_bundle_files(prepared) == {}


def test_manifest_direct_construction_validates_scalars_and_nested_records():
    valid = dict(
        protocol_version=1,
        launch_mode="cli",
        input_sha256="0" * 64,
        presets=(),
        source_name="<stdin>",
        source_path="<stdin>",
        source_realpath=None,
        base_dir="/tmp",
    )
    with pytest.raises(ConfigError):
        BootstrapManifest(**{**valid, "protocol_version": True})
    with pytest.raises(ConfigError):
        BootstrapManifest(**{**valid, "launch_mode": "worker"})
    with pytest.raises(ConfigError):
        BootstrapManifest(**{**valid, "input_sha256": "not-a-digest"})
    with pytest.raises(ConfigError):
        BootstrapManifest(**{**valid, "presets": [object()]})

    class SelectedSubclass(SelectedPreset):
        pass

    selected = SelectedSubclass(PresetRequest("one", None), snapshot("one"))
    with pytest.raises(ConfigError, match="SelectedPreset"):
        BootstrapManifest(**{**valid, "presets": [selected]})

    exact = SelectedPreset(PresetRequest("one", None), snapshot("one"))
    with pytest.raises(ConfigError, match="duplicate preset"):
        BootstrapManifest(**{**valid, "presets": [exact, exact]})

    file_source = {
        **valid,
        "source_name": "/tmp/config.yaml",
        "source_path": "/tmp/config.yaml",
        "source_realpath": "/tmp/config.yaml",
    }
    with pytest.raises(ConfigError, match="absolute normalized path"):
        BootstrapManifest(
            **{
                **file_source,
                "source_name": "config.yaml",
                "source_path": "config.yaml",
            }
        )
    with pytest.raises(ConfigError, match="source_realpath"):
        BootstrapManifest(**{**file_source, "source_realpath": None})
    with pytest.raises(ConfigError, match="stdin source_realpath"):
        BootstrapManifest(
            **{**valid, "source_realpath": "/tmp/config.yaml"}
        )
    with pytest.raises(ConfigError, match="base_dir"):
        BootstrapManifest(**{**valid, "base_dir": "relative"})


def test_selected_preset_direct_construction_requires_matching_records():
    snap = snapshot("one")
    with pytest.raises(ConfigError):
        SelectedPreset(object(), snap)
    with pytest.raises(ConfigError):
        SelectedPreset(PresetRequest("two", None), snap)

    class RequestSubclass(PresetRequest):
        pass

    class SnapshotSubclass(PresetSnapshot):
        pass

    with pytest.raises(ConfigError, match="PresetRequest"):
        SelectedPreset(RequestSubclass("one", None), snap)
    with pytest.raises(ConfigError, match="PresetSnapshot"):
        SelectedPreset(
            PresetRequest("one", None),
            SnapshotSubclass(
                name="one",
                resource="rheplicant/config/presets/one.yaml",
                input_bytes=b"x",
                sha256=hashlib.sha256(b"x").hexdigest(),
                document={"runtime": {"jax_enable_x64": True}},
                expanded_nodes=1,
            ),
        )


def test_config_source_direct_construction_validates_bytes_hash_and_source():
    manifest = BootstrapManifest(
        protocol_version=1,
        launch_mode="cli",
        input_sha256=hashlib.sha256(b"a").hexdigest(),
        presets=(),
        source_name="<stdin>",
        source_path="<stdin>",
        source_realpath=None,
        base_dir="/tmp",
    )
    document = {"schema_version": 1}
    merged = initial_merge(document, origin=Origin("user"))
    fields = dict(
        input_bytes=b"a",
        source_path="<stdin>",
        source_realpath=None,
        source_name="<stdin>",
        base_dir="/tmp",
        parsed_document=document,
        layered_document=document,
        origins=merged.origins,
        bootstrap_manifest=manifest,
    )
    with pytest.raises(ConfigError):
        ConfigSource(**{**fields, "input_bytes": 3})
    with pytest.raises(ConfigError):
        ConfigSource(**{**fields, "input_bytes": b"changed"})
    with pytest.raises(ConfigError):
        ConfigSource(**{**fields, "source_name": "different"})
    with pytest.raises(ConfigError, match="origin tree"):
        ConfigSource(
            **{
                **fields,
                "origins": OriginNode(origin=None, children={}),
            }
        )
    with pytest.raises(ConfigError, match="concrete origins"):
        ConfigSource(
            **{
                **fields,
                "origins": OriginNode(
                    origin=None,
                    children={
                        "schema_version": OriginNode(
                            origin=None, children={}
                        )
                    },
                ),
            }
        )

    object.__setattr__(manifest, "base_dir", "relative")
    with pytest.raises(ConfigError, match="base_dir"):
        ConfigSource(
            **{
                **fields,
                "base_dir": "relative",
                "bootstrap_manifest": manifest,
            }
        )


def test_direct_record_validation_still_runs_under_python_optimized_mode():
    code = r'''
import hashlib
from _rheplicant_bootstrap.layering import OriginNode
from _rheplicant_bootstrap.prepare import BootstrapManifest, ConfigSource
from _rheplicant_bootstrap.process import RawProcessEntry

def outcome(callback):
    try:
        callback()
    except Exception as error:
        print(type(error).__name__)
    else:
        print("accepted")

outcome(lambda: RawProcessEntry(1, (), (), [], {}))
outcome(lambda: RawProcessEntry(1.0, (), (), None, {}))
manifest = BootstrapManifest(
    1, "cli", hashlib.sha256(b"a").hexdigest(), (),
    "<stdin>", "<stdin>", None, "/tmp",
)
outcome(lambda: ConfigSource(
    b"a", "<stdin>", None, "<stdin>", "/tmp", [], {},
    OriginNode(None, {}), manifest,
))
outcome(lambda: ConfigSource(
    b"a", "<stdin>", None, "<stdin>", "/tmp", {}, {"x": 1},
    OriginNode(None, {}), manifest,
))
'''
    done = subprocess.run(
        [sys.executable, "-O", "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    assert done.stdout.splitlines() == [
        "ConfigError",
        "ConfigError",
        "ConfigError",
        "ConfigError",
    ]


def test_prepared_config_direct_construction_validates_nested_record_types():
    with pytest.raises(ConfigError):
        PreparedConfig(
            source=object(),
            process=object(),
            layers=(),
            layer_origins={},
            layer_deletions={},
        )


class _AbortPreparedEvidence(BaseException):
    pass


class _BrokenPreparedEvidence(Mapping):
    def __init__(self, failure):
        self.failure = failure

    def __getitem__(self, key):
        raise self.failure

    def __iter__(self):
        raise self.failure

    def __len__(self):
        return 1


class _BrokenPreparedDeletionSequence(Sequence):
    def __init__(self, failure):
        self.failure = failure

    def __getitem__(self, index):
        raise self.failure

    def __iter__(self):
        raise self.failure

    def __len__(self):
        return 1


def test_prepared_config_validates_parallel_origins_and_mapping_protocols():
    prepared = prepare_config(
        source(b"schema_version: 1\nruntime: {}\nruns: []\n"),
        preset_provider=no_presets,
        parse_outputs=fake_parse_outputs,
    )
    base = prepared.layers[0]
    fields = dict(
        source=prepared.source,
        process=prepared.process,
        layers=prepared.layers,
        layer_origins=prepared.layer_origins,
        layer_deletions=prepared.layer_deletions,
    )

    with pytest.raises(ConfigError, match="origin tree"):
        PreparedConfig(
            **{
                **fields,
                "layer_origins": {
                    base.identity: OriginNode(None, {})
                },
            }
        )
    with pytest.raises(ConfigError, match="mapping traversal"):
        PreparedConfig(
            **{
                **fields,
                "layer_origins": _BrokenPreparedEvidence(
                    RuntimeError("boom")
                ),
            }
        )
    with pytest.raises(_AbortPreparedEvidence, match="stop"):
        PreparedConfig(
            **{
                **fields,
                "layer_origins": _BrokenPreparedEvidence(
                    _AbortPreparedEvidence("stop")
                ),
            }
        )

    with pytest.raises(ConfigError, match="deletion sequence traversal"):
        PreparedConfig(
            **{
                **fields,
                "layer_deletions": {
                    base.identity: _BrokenPreparedDeletionSequence(
                        RuntimeError("boom")
                    )
                },
            }
        )
    with pytest.raises(_AbortPreparedEvidence, match="stop"):
        PreparedConfig(
            **{
                **fields,
                "layer_deletions": {
                    base.identity: _BrokenPreparedDeletionSequence(
                        _AbortPreparedEvidence("stop")
                    )
                },
            }
        )


def test_pipeline_preserves_the_trusted_deletion_ledger_objects(monkeypatch):
    real = prepare_module.enumerate_layers_once
    captured = None

    def observing(*args, **kwargs):
        nonlocal captured
        captured = real(*args, **kwargs)
        return captured

    monkeypatch.setattr(prepare_module, "enumerate_layers_once", observing)
    prepared = prepare_config(
        source(
            b"schema_version: 1\nruntime: {}\nruns: []\n"
            b"variants:\n  x: {}\n"
        ),
        preset_provider=no_presets,
        parse_outputs=fake_parse_outputs,
    )

    assert captured is not None
    assert all(
        type(rows) is prepare_module._DeletionLedger
        for rows in prepared.layer_deletions.values()
    )
    assert prepared.layer_deletions is captured.deletions
    assert all(
        prepared.layer_deletions[identity]
        is captured.deletions[identity]
        for identity in captured.deletions
    )

    ordinary = PreparedConfig(
        source=prepared.source,
        process=prepared.process,
        layers=prepared.layers,
        layer_origins=prepared.layer_origins,
        layer_deletions=prepared.layer_deletions,
    )
    assert all(
        type(layer.document) is MappingProxyType for layer in ordinary.layers
    )
    assert all(
        type(root.children) is MappingProxyType
        for root in ordinary.layer_origins.values()
    )
    assert all(
        type(rows) is tuple for rows in ordinary.layer_deletions.values()
    )

    forged = object.__new__(LayerEnumeration)
    object.__setattr__(forged, "layers", ordinary.layers)
    object.__setattr__(forged, "origins", ordinary.layer_origins)
    object.__setattr__(
        forged, "deletions", prepared.layer_deletions
    )
    with pytest.raises(
        ConfigError, match="trusted prepared enumeration"
    ):
        prepare_module._trusted_prepared_config(
            prepared.source, prepared.process, forged
        )


def test_prepare_before_preflight_shares_wide_root_evidence():
    """A rejected root must not first allocate one full copy per variant."""
    width = 10_000
    variant_count = 100
    slot_ceiling = 30_000
    peak_ceiling = 64 * 1024 * 1024
    unknown = "".join(
        f"unknown_{index}: null\n" for index in range(width)
    )
    variants = "variants:\n" + "".join(
        f"  v{index}: {{}}\n" for index in range(variant_count)
    )
    raw = (
        "schema_version: 1\nruntime: {}\nruns: []\n"
        f"{unknown}{variants}"
    ).encode()

    from rheplicant.config.preflight import preflight

    tracemalloc.start()
    started = perf_counter()
    try:
        prepared = prepare_config(
            source(raw),
            preset_provider=no_presets,
            parse_outputs=fake_parse_outputs,
        )
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    elapsed = perf_counter() - started
    with pytest.raises(ConfigError, match="This document declares"):
        preflight(thaw(prepared.source.layered_document))

    document_entries = sum(len(layer.document) for layer in prepared.layers)
    origin_entries = sum(
        len(root.children) for root in prepared.layer_origins.values()
    )
    physical_document_slots = _physical_evidence_slots(
        *(layer.document for layer in prepared.layers)
    )
    physical_origin_slots = _physical_evidence_slots(
        *prepared.layer_origins.values()
    )
    assert document_entries > 1_000_000 and origin_entries > 1_000_000
    assert (
        max(physical_document_slots, physical_origin_slots) <= slot_ceiling
        and peak <= peak_ceiling
    ), (
        "prepare_config materialized "
        f"{physical_document_slots} document slots and "
        f"{physical_origin_slots} origin slots before preflight "
        f"refused the root; logical=({document_entries}, {origin_entries}), "
        f"peak={peak / (1024 * 1024):.1f}MiB, "
        f"elapsed={elapsed * 1000:.1f}ms, "
        f"ceilings=({slot_ceiling}, {peak_ceiling / (1024 * 1024):.0f}MiB)"
    )


def test_prepare_shares_inherited_sequence_prefixes_across_append_variants():
    """Sibling append layers must not each retain the inherited tuple."""
    width = 10_000
    variant_count = 100
    slot_ceiling = 25_000
    runs = "runs:\n" + "  - {kind: forward}\n" * width
    variants = "variants:\n" + "".join(
        f"  v{index}:\n"
        "    runs: {append: [{kind: forward}]}\n"
        for index in range(variant_count)
    )
    raw = f"schema_version: 1\nruntime: {{}}\n{runs}{variants}".encode()

    started = perf_counter()
    prepared = prepare_config(
        source(raw),
        preset_provider=no_presets,
        parse_outputs=fake_parse_outputs,
    )
    elapsed = perf_counter() - started
    run_sequences = tuple(layer.document["runs"] for layer in prepared.layers)
    logical_slots = sum(len(rows) for rows in run_sequences)
    physical_slots = _physical_evidence_slots(*run_sequences)

    assert logical_slots > 1_000_000
    assert physical_slots <= slot_ceiling, (
        "prepare_config materialized "
        f"{physical_slots} inherited run-sequence slots across "
        f"{len(run_sequences)} layers; logical={logical_slots}, "
        f"elapsed={elapsed * 1000:.1f}ms, ceiling={slot_ceiling}"
    )


def test_prepare_shares_wide_nested_mappings_and_parallel_origins():
    """Touching one nested key per sibling must not copy the wide parent."""
    width = 10_000
    variant_count = 100
    slot_ceiling = 35_000
    model = "model:\n" + "".join(
        f"  base_{index}: null\n" for index in range(width)
    )
    variants = "variants:\n" + "".join(
        f"  v{index}:\n    model: {{added_{index}: null}}\n"
        for index in range(variant_count)
    )
    raw = (
        f"schema_version: 1\nruntime: {{}}\nruns: []\n{model}{variants}"
    ).encode()

    started = perf_counter()
    prepared = prepare_config(
        source(raw),
        preset_provider=no_presets,
        parse_outputs=fake_parse_outputs,
    )
    elapsed = perf_counter() - started
    models = tuple(layer.document["model"] for layer in prepared.layers)
    origin_models = tuple(
        root.children["model"]
        for root in prepared.layer_origins.values()
    )
    logical_document_entries = sum(len(model) for model in models)
    logical_origin_entries = sum(
        len(root.children) for root in origin_models
    )
    physical_document_slots = _physical_evidence_slots(*models)
    physical_origin_slots = _physical_evidence_slots(*origin_models)

    assert logical_document_entries > 1_000_000
    assert logical_origin_entries > 1_000_000
    assert max(
        physical_document_slots, physical_origin_slots
    ) <= slot_ceiling, (
        "prepare_config materialized "
        f"{physical_document_slots} nested document slots and "
        f"{physical_origin_slots} parallel-origin slots; "
        f"logical=({logical_document_entries}, {logical_origin_entries}), "
        f"elapsed={elapsed * 1000:.1f}ms, ceiling={slot_ceiling}"
    )


def test_importing_all_four_task_modules_in_a_clean_process_adds_no_jax():
    code = r'''
import sys
before = set(sys.modules)
import _rheplicant_bootstrap.path_syntax
import _rheplicant_bootstrap.prepare
import _rheplicant_bootstrap.process
import _rheplicant_bootstrap.variants
banned = {"jax", "jaxlib", "equinox", "numpyro", "rheplicant"}
print(sorted(name for name in set(sys.modules) - before
             if name.split(".", 1)[0] in banned))
'''
    done = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    assert done.stdout.strip() == "[]"
