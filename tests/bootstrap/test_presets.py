"""Exact bytes and safe discovery for installed package presets."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import pytest

from _rheplicant_bootstrap import presets
from _rheplicant_bootstrap.errors import ConfigError
from _rheplicant_bootstrap.frozen import thaw
from _rheplicant_bootstrap.presets import (
    PresetRequest,
    PresetSnapshot,
    read_installed_preset,
)

PRESET_BYTES = b"""# RHINO v1 example-derived starting values.
# These are copied from repository examples, not measured instrument constants.
runtime:
  jax_enable_x64: true
observation:
  freq:
    grid:
      linspace:
        start: 60.0
        stop: 85.0
      unit: MHz
  site:
    lat_deg: {value: 53.2367, unit: deg}
  pointing:
    mode: drift
    az_deg: {value: 0.0, unit: deg}
    el_deg: {value: 90.0, unit: deg}
model:
  antenna_loss:
    efficiency: 0.97
    t_physical: {value: 293.0, unit: K}
  beam_spill:
    t_ground: {value: 290.0, unit: K}
  atmosphere:
    t_atm: {value: 3.0, unit: K}
  noise_wave:
    t_rx: {value: 290.0, unit: K}
"""
PRESET_SHA256 = "863c3a7bfdccd428bb5bf69e87facf8286975b1c87e7b6c4a96f754a2376d7c4"
FIXTURE_BYTES = b"fixture"
FIXTURE_SHA256 = "f16d05ec6b29248d2c61adb1e9263f78e4f7bace1b955014a2d17872cfe4064d"
MAXIMUM_PRESET_BYTES = 16 * 1024 * 1024
MAXIMUM_EXPANDED_NODES = 250_000


def _resource_path() -> Path:
    return Path(__file__).parents[2] / "src/rheplicant/config/presets/rhino_v1.yaml"


def test_rhino_v1_bytes_are_the_normative_partial_candidate():
    """Catches normalized prose, invented required facts, or resource drift."""
    raw = _resource_path().read_bytes()
    assert raw == PRESET_BYTES
    assert hashlib.sha256(raw).hexdigest() == PRESET_SHA256

    snapshot = read_installed_preset("rhino_v1")
    document = thaw(snapshot.document)
    assert set(document) == {"runtime", "observation", "model"}
    assert "num" not in document["observation"]["freq"]["grid"]["linspace"]
    assert "endpoint" not in document["observation"]["freq"]["grid"]["linspace"]
    assert "materialise" not in document["observation"]["pointing"]
    assert "sky_fraction" not in document["model"]["beam_spill"]
    assert set(document["model"]["noise_wave"]) == {"t_rx"}
    assert snapshot.input_bytes == PRESET_BYTES
    assert snapshot.sha256 == PRESET_SHA256


class _RecordedFile:
    def __init__(self, resource: str, path: Path) -> None:
        self.resource = resource
        self.path = path

    def as_posix(self) -> str:
        return self.resource

    def locate(self) -> Path:
        return self.path


class _Distribution:
    def __init__(
        self,
        *,
        files: tuple[_RecordedFile, ...] = (),
        direct_url: dict[str, object] | None = None,
    ) -> None:
        self.files = files
        self.direct_url = direct_url

    def read_text(self, name: str) -> str | None:
        assert name == "direct_url.json"
        return None if self.direct_url is None else json.dumps(self.direct_url)


def test_recorded_distribution_resource_is_read_without_package_import(tmp_path, monkeypatch):
    """Catches bypassing RECORD in favor of importing the JAX-backed package."""
    path = tmp_path / "recorded.yaml"
    path.write_bytes(b"runtime: {}\n")
    resource = "rheplicant/config/presets/rhino_v1.yaml"
    distribution = _Distribution(files=(_RecordedFile(resource, path),))
    monkeypatch.setattr(presets.importlib.metadata, "distribution", lambda _: distribution)

    snapshot = read_installed_preset("rhino_v1")

    assert snapshot.input_bytes == b"runtime: {}\n"
    assert thaw(snapshot.document) == {"runtime": {}}


def test_explicit_editable_distribution_uses_the_unique_package_root(tmp_path, monkeypatch):
    """Catches editable installs losing package data when RECORD omits it."""
    root = tmp_path / "rheplicant"
    resource = root / "config/presets/rhino_v1.yaml"
    resource.parent.mkdir(parents=True)
    resource.write_bytes(b"model: {}\n")
    distribution = _Distribution(direct_url={"dir_info": {"editable": True}})
    monkeypatch.setattr(presets.importlib.metadata, "distribution", lambda _: distribution)
    monkeypatch.setattr(
        presets.importlib.util,
        "find_spec",
        lambda _: SimpleNamespace(submodule_search_locations=(str(root),)),
    )

    snapshot = read_installed_preset("rhino_v1")

    assert snapshot.input_bytes == b"model: {}\n"


@pytest.mark.parametrize("direct_url", [None, {}, {"dir_info": {"editable": False}}])
def test_missing_record_is_not_treated_as_editable_without_explicit_metadata(
    direct_url, monkeypatch
):
    """Catches a source-looking install silently enabling the fallback."""
    distribution = _Distribution(direct_url=direct_url)
    monkeypatch.setattr(presets.importlib.metadata, "distribution", lambda _: distribution)
    monkeypatch.setattr(
        presets.importlib.util,
        "find_spec",
        lambda _: pytest.fail("find_spec must not run for a non-editable distribution"),
    )

    with pytest.raises(ConfigError, match="installed distribution does not contain"):
        read_installed_preset("rhino_v1")


def test_malformed_editable_metadata_is_normalized_to_config_error(monkeypatch):
    """Catches malformed distribution JSON escaping the neutral boundary."""
    distribution = _Distribution(direct_url={"dir_info": None})
    monkeypatch.setattr(presets.importlib.metadata, "distribution", lambda _: distribution)

    with pytest.raises(ConfigError, match="cannot discover package preset"):
        read_installed_preset("rhino_v1")


def test_distribution_protocol_failure_is_normalized_without_rendering(monkeypatch):
    class HostileMetadataError(Exception):
        def __str__(self):
            raise AssertionError("exception text must not run")

        def __repr__(self):
            raise AssertionError("exception repr must not run")

    def fail_distribution(_):
        raise HostileMetadataError

    monkeypatch.setattr(
        presets.importlib.metadata, "distribution", fail_distribution
    )

    with pytest.raises(ConfigError, match="cannot discover package preset"):
        read_installed_preset("rhino_v1")


def test_malformed_real_record_csv_is_normalized_to_config_error(tmp_path, monkeypatch):
    """Catches importlib.metadata's csv.Error escaping the neutral boundary."""
    dist_info = tmp_path / "rheplicant-1.0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text("Name: rheplicant\nVersion: 1.0\n")
    oversized_field = "x" * (csv.field_size_limit() + 1)
    (dist_info / "RECORD").write_text(oversized_field)
    distribution = importlib.metadata.PathDistribution(dist_info)
    monkeypatch.setattr(presets.importlib.metadata, "distribution", lambda _: distribution)

    with pytest.raises(ConfigError, match="cannot discover package preset"):
        read_installed_preset("rhino_v1")


def test_editable_package_root_must_be_unique(monkeypatch):
    distribution = _Distribution(direct_url={"dir_info": {"editable": True}})
    monkeypatch.setattr(presets.importlib.metadata, "distribution", lambda _: distribution)
    monkeypatch.setattr(
        presets.importlib.util,
        "find_spec",
        lambda _: SimpleNamespace(submodule_search_locations=("first", "second")),
    )

    with pytest.raises(ConfigError, match="package root is not unique"):
        read_installed_preset("rhino_v1")


@pytest.mark.parametrize("name", ["../rhino_v1", "rhino.v1", "", "1rhino"])
def test_invalid_preset_names_are_refused_before_discovery(name, monkeypatch):
    monkeypatch.setattr(
        presets.importlib.metadata,
        "distribution",
        lambda _: pytest.fail("invalid names must not reach distribution discovery"),
    )
    with pytest.raises(ConfigError, match="invalid package preset name"):
        read_installed_preset(name)


def test_unregistered_preset_name_is_refused():
    with pytest.raises(ConfigError, match="unknown package preset 'another'"):
        read_installed_preset("another")


def test_preset_document_rejects_process_entry_sections(tmp_path, monkeypatch):
    path = tmp_path / "bad.yaml"
    path.write_bytes(b"runtime: {}\nplugins: [anything]\n")
    resource = "rheplicant/config/presets/rhino_v1.yaml"
    distribution = _Distribution(files=(_RecordedFile(resource, path),))
    monkeypatch.setattr(presets.importlib.metadata, "distribution", lambda _: distribution)

    with pytest.raises(ConfigError, match="plugins"):
        read_installed_preset("rhino_v1")


def test_direct_snapshot_construction_detaches_and_canonicalizes_evidence():
    """Catches custom providers returning time-varying shallow snapshot records."""
    raw = bytearray(FIXTURE_BYTES)
    document = {"runtime": {"values": [bytearray(b"one")]}}

    snapshot = PresetSnapshot(
        name="one",
        resource="rheplicant/config/presets/one.yaml",
        input_bytes=raw,
        sha256=FIXTURE_SHA256,
        document=document,
        expanded_nodes=1,
    )
    raw[:] = b"changed"
    document["runtime"]["values"][0][:] = b"two"
    document["runtime"]["values"].append(b"late")

    assert snapshot.input_bytes == b"fixture"
    assert snapshot.document["runtime"]["values"] == (b"one",)
    with pytest.raises(TypeError):
        snapshot.document["late"] = True


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


class _StatefulBytes(bytes):
    def __new__(cls, value):
        instance = super().__new__(cls, value)
        instance.state = []
        return instance


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


class _HostileBytes(bytes):
    def __bytes__(self):
        raise AssertionError("__bytes__ must not run")

    def __eq__(self, other):
        raise AssertionError("equality must not run")

    def __hash__(self):
        raise AssertionError("hash must not run")

    def __repr__(self):
        raise AssertionError("repr must not run")


class _HostileBytearray(bytearray):
    def __bytes__(self):
        raise AssertionError("__bytes__ must not run")

    def __buffer__(self, flags):
        raise AssertionError("__buffer__ must not run")

    def __repr__(self):
        raise AssertionError("repr must not run")


def test_preset_records_canonicalize_every_scalar_field_to_exact_builtins():
    name = _StatefulStr("one")
    selector = _StatefulStr("runtime")
    resource = _StatefulStr("rheplicant/config/presets/one.yaml")
    raw = _StatefulBytes(b"fixture")
    sha256 = _StatefulStr(FIXTURE_SHA256)
    expanded_nodes = _StatefulInt(2)

    request = PresetRequest(name=name, only=[selector])
    snapshot = PresetSnapshot(
        name=name,
        resource=resource,
        input_bytes=raw,
        sha256=sha256,
        document={"runtime": {}},
        expanded_nodes=expanded_nodes,
    )

    assert type(request.name) is str
    assert type(request.only[0]) is str
    assert type(snapshot.name) is str
    assert type(snapshot.resource) is str
    assert type(snapshot.input_bytes) is bytes
    assert type(snapshot.sha256) is str
    assert type(snapshot.expanded_nodes) is int


def test_preset_record_scalar_canonicalization_uses_only_base_operations():
    """Catches invoking hostile conversion, comparison, split, hash, or repr hooks."""
    request = PresetRequest(
        name=_HostileStr("one"), only=[_HostileStr("runtime")]
    )
    snapshot = PresetSnapshot(
        name=_HostileStr("one"),
        resource=_HostileStr("rheplicant/config/presets/one.yaml"),
        input_bytes=_HostileBytes(FIXTURE_BYTES),
        sha256=_HostileStr(FIXTURE_SHA256),
        document={"runtime": {}},
        expanded_nodes=_HostileInt(2),
    )

    assert request.name == "one"
    assert request.only == ("runtime",)
    assert type(snapshot.name) is str
    assert type(snapshot.resource) is str
    assert type(snapshot.input_bytes) is bytes
    assert type(snapshot.sha256) is str
    assert type(snapshot.expanded_nodes) is int

    buffered = PresetSnapshot(
        name="one",
        resource="rheplicant/config/presets/one.yaml",
        input_bytes=_HostileBytearray(FIXTURE_BYTES),
        sha256=FIXTURE_SHA256,
        document={},
        expanded_nodes=0,
    )
    assert type(buffered.input_bytes) is bytes


@pytest.mark.parametrize(
    ("overrides", "needle"),
    [
        ({"name": object()}, "name"),
        ({"resource": object()}, "resource"),
        ({"resource": ""}, "resource"),
        ({"sha256": object()}, "sha256"),
        ({"sha256": "not-a-digest"}, "sha256"),
        ({"expanded_nodes": True}, "expanded_nodes"),
        ({"expanded_nodes": -1}, "expanded_nodes"),
    ],
)
def test_snapshot_direct_construction_validates_every_scalar_field(overrides, needle):
    arguments = {
        "name": "one",
        "resource": "rheplicant/config/presets/one.yaml",
        "input_bytes": FIXTURE_BYTES,
        "sha256": FIXTURE_SHA256,
        "document": {"runtime": {}},
        "expanded_nodes": 1,
    }
    arguments.update(overrides)

    with pytest.raises(ConfigError, match=needle):
        PresetSnapshot(**arguments)


def test_snapshot_digest_must_equal_the_canonical_input_bytes():
    with pytest.raises(ConfigError, match="sha256.*does not match"):
        PresetSnapshot(
            name="one",
            resource="rheplicant/config/presets/one.yaml",
            input_bytes=FIXTURE_BYTES,
            sha256="0" * 64,
            document={},
            expanded_nodes=0,
        )


def test_snapshot_enforces_exact_input_and_expansion_limits():
    at_limit = b"x" * MAXIMUM_PRESET_BYTES
    snapshot = PresetSnapshot(
        name="one",
        resource="rheplicant/config/presets/one.yaml",
        input_bytes=at_limit,
        sha256=hashlib.sha256(at_limit).hexdigest(),
        document={},
        expanded_nodes=MAXIMUM_EXPANDED_NODES,
    )
    assert len(snapshot.input_bytes) == MAXIMUM_PRESET_BYTES
    assert snapshot.expanded_nodes == MAXIMUM_EXPANDED_NODES

    over_limit = at_limit + b"x"
    with pytest.raises(ConfigError, match=r"input_bytes.*16777217.*16777216"):
        PresetSnapshot(
            name="one",
            resource="rheplicant/config/presets/one.yaml",
            input_bytes=over_limit,
            sha256=hashlib.sha256(over_limit).hexdigest(),
            document={},
            expanded_nodes=0,
        )
    with pytest.raises(ConfigError, match=r"expanded_nodes.*250001.*250000"):
        PresetSnapshot(
            name="one",
            resource="rheplicant/config/presets/one.yaml",
            input_bytes=FIXTURE_BYTES,
            sha256=FIXTURE_SHA256,
            document={},
            expanded_nodes=MAXIMUM_EXPANDED_NODES + 1,
        )


def test_snapshot_normalizes_a_released_memoryview_without_using_repr():
    view = memoryview(FIXTURE_BYTES)
    view.release()

    with pytest.raises(ConfigError, match=r"preset:one.*input_bytes.*memoryview"):
        PresetSnapshot(
            name="one",
            resource="rheplicant/config/presets/one.yaml",
            input_bytes=view,
            sha256=FIXTURE_SHA256,
            document={},
            expanded_nodes=0,
        )


@pytest.mark.parametrize("name", [object(), "", "not.valid"])
def test_request_direct_construction_validates_name(name):
    with pytest.raises(ConfigError, match="name"):
        PresetRequest(name=name, only=None)


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


class _ArmedKey(str):
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


def test_validate_preset_document_canonicalizes_keys_before_dict_operations():
    key = _ArmedKey("runtime")
    loaded = {key: {}}
    key.armed = True

    validated = presets.validate_preset_document("one", loaded)

    assert validated == {"runtime": {}}
    assert type(next(iter(validated))) is str


def test_validate_preset_document_refuses_canonical_key_collisions():
    loaded = _ItemsMapping(
        [(_HostileStr("runtime"), {}), (_HostileStr("runtime"), {})]
    )

    with pytest.raises(ConfigError, match="keys collide"):
        presets.validate_preset_document("one", loaded)


def test_validate_preset_document_rejects_hostile_nonstring_key_without_repr():
    class HostileKey:
        def __repr__(self):
            raise AssertionError("repr must not run")

    with pytest.raises(ConfigError, match="HostileKey"):
        presets.validate_preset_document(
            "one", _ItemsMapping([(HostileKey(), {})])
        )


def test_clean_bootstrap_preset_read_imports_neither_rheplicant_nor_jax():
    """Catches resource discovery importing the scientific package initializer."""
    source_root = Path(__file__).parents[2] / "src"
    program = (
        "import json, sys; "
        f"sys.path.insert(0, {str(source_root)!r}); "
        "from _rheplicant_bootstrap.presets import read_installed_preset; "
        "read_installed_preset('rhino_v1'); "
        "print(json.dumps(sorted(set(sys.modules) & {'rheplicant', 'jax'})))"
    )
    environment = os.environ.copy()
    completed = subprocess.run(
        [sys.executable, "-I", "-c", program],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert json.loads(completed.stdout) == []
