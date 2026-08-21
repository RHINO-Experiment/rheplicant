from __future__ import annotations

from collections.abc import Mapping

import pytest
import yaml

from rheplicant.config import ConfigError
from rheplicant.gui.form_edits import set_form_value

BASE_YAML = """\
schema_version: 1
runtime:
  jax_enable_x64: true
  platform: auto
  seed: 3
observation:
  from_file:
    format: rhino_hdf5
    path: /srv/data/original.h5
    freq_unit: Hz
  pointing:
    mode: none
inference:
  checks:
    linearity:
      mode: skip
      reason: before
model: {}
runs: []
"""

DOTTED_BEAM_YAML = """\
schema_version: 1
resources:
  beams:
    horn.dot:
      format: inline
model: {}
runs: []
"""

GENERIC_YAML = BASE_YAML.replace(
    "model: {}\n",
    "resources:\n  beams:\n    horn: {format: inline}\nmodel: {}\n",
)


def _value_at(document: str, path: str) -> object:
    value: object = yaml.safe_load(document)
    for component in path.split("."):
        assert isinstance(value, Mapping)
        value = value[component]
    return value


@pytest.mark.parametrize(
    ("path", "value", "expected_type"),
    [
        ("runtime.platform", "cpu", str),
        ("runtime.jax_enable_x64", False, bool),
        ("runtime.seed", 0, int),
        ("observation.from_file.path", "/srv/data/new.h5", str),
        ("inference.checks.linearity.reason", "", str),
    ],
)
def test_closed_widget_contracts_accept_exact_values_including_false_zero_and_empty(
    path: str,
    value: object,
    expected_type: type[object],
):
    changed = set_form_value(BASE_YAML, path, value)

    found = _value_at(changed, path)
    assert type(found) is expected_type
    assert found == value


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("runtime.seed", True),
        ("runtime.jax_enable_x64", 0),
        ("runtime.platform", 1),
        ("runtime.platform", "quantum"),
        ("observation.from_file.path", b"not-a-server-path"),
        ("inference.checks.linearity.reason", 8),
    ],
)
def test_closed_widget_contracts_reject_wrong_types_and_undeclared_choices(
    path: str,
    value: object,
):
    with pytest.raises(ConfigError, match="requires"):
        set_form_value(BASE_YAML, path, value)


def test_semantic_noops_return_the_exact_original_yaml_bytes():
    unusual = BASE_YAML.replace("  seed: 3", "  seed: 3  # keep this spelling")
    absent = unusual.replace("  seed: 3  # keep this spelling\n", "")

    assert set_form_value(unusual, "runtime.seed", 3) == unusual
    assert set_form_value(absent, "runtime.seed", None, remove=True) == absent


@pytest.mark.parametrize(
    ("document", "path", "value"),
    [
        (
            BASE_YAML.replace(
                "jax_enable_x64: true", "jax_enable_x64: false  # keep false"
            ),
            "runtime.jax_enable_x64",
            False,
        ),
        (
            BASE_YAML.replace("seed: 3", "seed: 0  # keep zero"),
            "runtime.seed",
            0,
        ),
        (
            BASE_YAML.replace("reason: before", "reason: ''  # keep empty"),
            "inference.checks.linearity.reason",
            "",
        ),
    ],
)
def test_false_zero_and_empty_string_noops_preserve_exact_yaml_bytes(
    document: str,
    path: str,
    value: object,
):
    assert set_form_value(document, path, value) == document


def test_set_updates_inserts_missing_mapping_parents_and_removes_the_exact_leaf():
    changed = set_form_value(BASE_YAML, "runtime.seed", 7)
    assert _value_at(changed, "runtime.seed") == 7

    without_runtime = """\
schema_version: 1
model: {}
runs: []
"""
    inserted = set_form_value(without_runtime, "runtime.seed", 0)
    assert _value_at(inserted, "runtime.seed") == 0

    removed = set_form_value(changed, "runtime.seed", None, remove=True)
    runtime = _value_at(removed, "runtime")
    assert isinstance(runtime, Mapping)
    assert "seed" not in runtime


def test_changed_serialization_preserves_existing_mapping_order():
    before = yaml.safe_load(BASE_YAML)
    changed = yaml.safe_load(
        set_form_value(BASE_YAML, "runtime.jax_enable_x64", False)
    )

    assert tuple(changed) == tuple(before)
    assert tuple(changed["runtime"]) == (
        "jax_enable_x64",
        "platform",
        "seed",
    )

    without_runtime = "schema_version: 1\nmodel: {}\nruns: []\n"
    inserted = yaml.safe_load(set_form_value(without_runtime, "runtime.seed", 1))
    assert tuple(inserted) == ("schema_version", "model", "runs", "runtime")


@pytest.mark.parametrize(
    "path",
    [
        "resources.beams.*.normalize",
        "runs[0].name",
    ],
)
def test_literal_wildcard_and_list_index_paths_are_refused_as_nonconcrete(path: str):
    document = BASE_YAML.replace(
        "runs: []", "runs:\n  - name: forward\n    kind: forward"
    )

    with pytest.raises(ConfigError, match="exact concrete"):
        set_form_value(document, path, "changed")


@pytest.mark.parametrize(
    "path",
    [
        "runtime.seed.extra",
        "not.in.the.catalog",
    ],
)
def test_unknown_or_inexact_paths_are_refused_with_bounded_yaml_guidance(path: str):
    with pytest.raises(ConfigError, match="Edit this value in YAML") as caught:
        set_form_value(BASE_YAML, path, "changed")

    assert len(str(caught.value)) < 240


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("observation.site.alt_m", 100.0),
        ("runtime.seeds", {}),
        ("defaults", []),
        ("resources.beams.horn.extends", "other"),
        ("model.gain", {}),
        ("model.atmosphere_field", {}),
        ("observation.from_file", {}),
        ("outputs.write.arrays", True),
        ("campaign.epoch_id", "reserved"),
    ],
)
def test_every_generic_widget_kind_and_campaign_remain_yaml_only(
    path: str,
    value: object,
):
    with pytest.raises(ConfigError, match="Edit this value in YAML") as caught:
        set_form_value(GENERIC_YAML, path, value)

    assert len(str(caught.value)) < 240


def test_hidden_primitive_widget_is_refused_with_bounded_yaml_guidance():
    """``cw_tone`` is absent from this document, so the widget exists and is
    invisible -- which is a different refusal from a path that names no widget
    at all, even though both end in the same guidance sentence. The example
    has to be a SINGLE-SLOT node: a ``many`` node's fields are declared per
    instance (``model.foregrounds[].type``), so the flat spelling this test
    used to carry stopped naming a widget and quietly moved the test onto the
    unknown-path branch, where it went on passing."""
    with pytest.raises(ConfigError, match="This projected field is unavailable") as caught:
        set_form_value(BASE_YAML, "model.cw_tone.line_width", 1.0)

    assert "Edit this value in YAML" in str(caught.value)
    assert len(str(caught.value)) < 240


def test_dotted_mapping_wildcard_key_is_refused_as_an_ambiguous_flat_path():
    with pytest.raises(ConfigError, match="Edit this value in YAML") as caught:
        set_form_value(
            DOTTED_BEAM_YAML,
            "resources.beams.horn.dot.format",
            "gaussian",
        )

    assert len(str(caught.value)) < 240


def test_absent_scalar_leaf_does_not_make_a_longer_prefix_an_exact_field():
    without_seed = BASE_YAML.replace("  seed: 3\n", "")

    with pytest.raises(ConfigError, match="not one exact projected field"):
        set_form_value(without_seed, "runtime.seed.extra", 4)


def test_existing_scalar_parent_is_refused_instead_of_replaced_by_a_mapping():
    scalar_parent = "schema_version: 1\nruntime: occupied\nmodel: {}\nruns: []\n"

    with pytest.raises(ConfigError, match="non-mapping parent"):
        set_form_value(scalar_parent, "runtime.seed", 4)


@pytest.mark.parametrize(
    "document",
    [
        "model: [secret-value-never-repeat",
        "- secret-value-never-repeat\n- another\n",
        "null\n",
    ],
)
def test_malformed_or_nonmapping_yaml_is_refused_without_echoing_input(document: str):
    with pytest.raises(ConfigError, match="safe YAML mapping") as caught:
        set_form_value(document, "runtime.seed", 1)

    assert "secret-value-never-repeat" not in str(caught.value)
    assert len(str(caught.value)) < 240
