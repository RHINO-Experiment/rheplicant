"""Exact bytes and safe discovery for installed package presets."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from _rheplicant_bootstrap import presets
from _rheplicant_bootstrap.errors import ConfigError
from _rheplicant_bootstrap.frozen import thaw
from _rheplicant_bootstrap.presets import read_installed_preset

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
