from __future__ import annotations

import dataclasses

import pytest
import yaml

from rheplicant.config import ConfigError
from rheplicant.gui import NodeCard, replace_yaml, set_node, snapshot
from rheplicant.radio.graph import RADIO_GRAPH

BASE = """\
runtime:
  jax_enable_x64: true
model:
  gain:
    type: GainOperator
    gain: 1.0
runs:
  - name: forward
    kind: forward
"""


def test_snapshot_is_framework_free_and_uses_the_live_graph_contract():
    found = snapshot(BASE)
    assert found.yaml_text == BASE
    assert found.walk_order == RADIO_GRAPH._topo
    assert len(found.nodes) == 33
    assert sum(node.editable for node in found.nodes) == 28
    assert tuple(node.node_id for node in found.nodes) == RADIO_GRAPH._topo
    assert tuple(node.node_id for node in found.nodes if node.lit) == ("gain",)
    assert all(isinstance(node, NodeCard) for node in found.nodes)
    assert tuple(section.section_id for section in found.forms.sections) == (
        "runtime",
        "observation",
        "resources",
        "sky",
        "beam",
        "instrument",
        "backend",
        "variants",
        "inference",
        "runs",
        "outputs",
        "campaign",
    )
    assert "schema_version" in found.forms.missing_required
    assert dataclasses.is_dataclass(found)
    with pytest.raises(dataclasses.FrozenInstanceError):
        found.yaml_text = "model: {}\n"  # type: ignore[misc]


def test_snapshot_svg_carries_every_node_id_once_and_marks_the_lit_node():
    found = snapshot(BASE)
    for node_id in RADIO_GRAPH._topo:
        assert found.svg.count(f'data-node-id="{node_id}"') == 1
    gain = next(node for node in found.nodes if node.node_id == "gain")
    assert gain.lit is True
    assert gain.description == RADIO_GRAPH.nodes["gain"].doc


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("- one\n- two\n", "root must be a mapping"),
        ("model: []\n", "model: must be a mapping"),
        ("model:\n  gain: 1\n", "model.gain: must be a mapping"),
        ("model: {}\nmodel: {}\n", "duplicate key 'model'"),
        ("model: !unsafe {}\n", "YAML tag"),
    ],
)
def test_invalid_or_unsafe_yaml_is_refused_in_the_layer_voice(value, message):
    with pytest.raises(ConfigError, match=message):
        snapshot(value)


def test_noop_edits_keep_the_authoritative_yaml_byte_for_byte():
    assert replace_yaml(BASE).yaml_text == BASE
    assert set_node(
        BASE,
        "gain",
        enabled=True,
        settings={"type": "GainOperator", "gain": 1.0},
    ).yaml_text == BASE
    assert set_node(
        BASE,
        "gain",
        enabled=True,
        settings={"gain": 1.0, "type": "GainOperator"},
    ).yaml_text == BASE


def test_bool_and_number_settings_are_not_mistaken_for_noop_edits():
    document = BASE.replace("gain: 1.0", "gain: true")
    found = set_node(
        document,
        "gain",
        enabled=True,
        settings={"type": "GainOperator", "gain": 1.0},
    )
    assert found.yaml_text != document
    assert yaml.safe_load(found.yaml_text)["model"]["gain"]["gain"] == 1.0
    assert type(yaml.safe_load(found.yaml_text)["model"]["gain"]["gain"]) is float


def test_enable_edit_and_disable_are_document_transformations():
    enabled = set_node(
        "runtime:\n  jax_enable_x64: true\nmodel: {}\n",
        "gain",
        enabled=True,
        settings={"type": "GainOperator", "gain": 1.25},
    )
    assert yaml.safe_load(enabled.yaml_text)["model"]["gain"] == {
        "type": "GainOperator",
        "gain": 1.25,
    }
    assert next(n for n in enabled.nodes if n.node_id == "gain").lit

    edited = set_node(
        enabled.yaml_text,
        "gain",
        enabled=True,
        settings={"type": "GainOperator", "gain": 0.75},
    )
    assert yaml.safe_load(edited.yaml_text)["model"]["gain"]["gain"] == 0.75

    disabled = set_node(edited.yaml_text, "gain", enabled=False)
    assert yaml.safe_load(disabled.yaml_text) == {
        "runtime": {"jax_enable_x64": True},
        "model": {},
    }
    assert not next(n for n in disabled.nodes if n.node_id == "gain").lit


def test_disabling_one_node_preserves_every_sibling_and_section():
    document = BASE.replace(
        "  gain:\n    type: GainOperator\n    gain: 1.0\n",
        "  bandpass:\n    type: ReceiverOperator\n  gain:\n"
        "    type: GainOperator\n    gain: 1.0\n",
    )
    found = set_node(document, "gain", enabled=False)
    parsed = yaml.safe_load(found.yaml_text)
    assert parsed["model"] == {"bandpass": {"type": "ReceiverOperator"}}
    assert parsed["runs"] == [{"name": "forward", "kind": "forward"}]


def test_editing_a_node_does_not_coerce_unrelated_mapping_keys():
    found = set_node(
        BASE + "metadata:\n  1: one\n",
        "gain",
        enabled=True,
        settings={"type": "GainOperator", "gain": 1.25},
    )
    assert yaml.safe_load(found.yaml_text)["metadata"] == {1: "one"}


@pytest.mark.parametrize("node_id", ["missing", "astro_sum", "receiver_input"])
def test_only_real_operator_slots_can_be_changed(node_id):
    with pytest.raises(ConfigError, match="operator slot"):
        set_node(BASE, node_id, enabled=True, settings={"type": "Anything"})


def test_settings_are_a_mapping_and_cannot_be_sent_for_disable():
    with pytest.raises(ConfigError, match="settings must be a mapping"):
        set_node(BASE, "gain", enabled=True, settings=[1])  # type: ignore[arg-type]
    with pytest.raises(ConfigError, match="disabled node cannot carry settings"):
        set_node(BASE, "gain", enabled=False, settings={"gain": 2.0})


def test_a_fresh_model_mapping_is_added_without_reordering_existing_sections():
    found = set_node(
        "runtime:\n  jax_enable_x64: true\nruns: []\n",
        "gain",
        enabled=True,
        settings={"type": "GainOperator", "gain": 1.0},
    )
    assert tuple(yaml.safe_load(found.yaml_text)) == ("runtime", "runs", "model")


def test_plain_selection_state_is_not_smuggled_into_yaml():
    found = snapshot(BASE)
    assert "selected" not in found.yaml_text
    assert "hover" not in found.yaml_text
