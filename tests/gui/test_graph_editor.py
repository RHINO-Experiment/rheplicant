from __future__ import annotations

from collections import OrderedDict

import pytest
import yaml

from rheplicant.config import ConfigError
from rheplicant.gui import (
    compose_node,
    move_node_instance,
    place_node,
    set_many_node,
    set_node,
    set_snapshot_before,
    snapshot,
)

GRAPH_DOCUMENT = """\
schema_version: 1
model:
  foregrounds:
    - {amplitude: 1000.0}
    - {amplitude: 500.0}
  cal_loads:
    ambient: {t_load: 300.0}
    hot: {t_load: 400.0}
  gain:
    compose: cascade
    stages:
      - {name: lna, type: GainOperator, gain: 1.0}
      - {name: post, type: GainOperator, gain: 2.0}
  filters:
    - {type: FourierBandFilter, low: 0.02, high: 0.10}
    - {type: SkySpaceFilter, n_modes: 3}
variants:
  low_gain:
    model:
      gain:
        compose: cascade
        stages:
          - {name: lna, type: GainOperator, gain: 0.5}
          - {name: post, type: GainOperator, gain: 1.0}
      ~foregrounds: null
runs: []
"""


def _node(found, node_id):
    return next(node for node in found.nodes if node.node_id == node_id)


def test_snapshot_projects_many_compose_counts_and_explanations_without_building():
    found = snapshot(GRAPH_DOCUMENT)

    assert _node(found, "foregrounds").configuration == "sum"
    assert _node(found, "foregrounds").count == 2
    assert tuple(item.label for item in _node(found, "foregrounds").instances) == (
        "foregrounds 1",
        "foregrounds 2",
    )
    assert _node(found, "cal_loads").configuration == "fan"
    assert tuple(item.label for item in _node(found, "cal_loads").instances) == (
        "ambient",
        "hot",
    )
    assert _node(found, "filters").configuration == "chain"
    assert _node(found, "gain").configuration == "compose"
    assert _node(found, "gain").stage_names == ("lna", "post")
    assert _node(found, "atmosphere_field").configuration == "reserved"
    assert "python:" in _node(found, "atmosphere_field").explanation
    assert _node(found, "astro_sum").configuration == "junction"
    assert "adds" in _node(found, "astro_sum").explanation
    assert _node(found, "receiver_input").configuration == "selector"
    assert "switching" in _node(found, "receiver_input").explanation

    assert "foregrounds (x2)" in found.svg
    assert "cal loads (x2)" in found.svg
    assert "filters (x2)" in found.svg
    assert found.base_diagram.counts.lit == 4
    assert found.base_diagram.counts.instances == 7
    assert found.base_diagram.counts.reserved == 3


def test_backend_is_a_processing_only_projection_of_the_same_live_graph():
    found = snapshot(GRAPH_DOCUMENT)

    assert found.backend_diagram.walk_order == (
        "snapshot",
        "flagging",
        "averaging",
        "apply_cal",
        "filters",
    )
    assert 'data-node-id="filters"' in found.backend_diagram.svg
    assert 'data-node-id="gain"' not in found.backend_diagram.svg
    assert found.backend_diagram.counts.lit == 1


def test_variant_diagrams_are_resolved_against_base_and_name_exact_changed_nodes():
    found = snapshot(GRAPH_DOCUMENT)

    assert len(found.variant_diagrams) == 1
    variant = found.variant_diagrams[0]
    assert variant.name == "low_gain"
    assert variant.changed_nodes == ("foregrounds", "gain")
    assert "foregrounds" in tuple(node.node_id for node in found.nodes if node.lit)
    assert "foregrounds" not in tuple(node.node_id for node in variant.nodes if node.lit)
    assert variant.svg != found.base_diagram.svg


def test_configure_and_disable_variant_nodes_touch_only_the_variant_route():
    assert (
        set_node(
            GRAPH_DOCUMENT,
            "cal_loads",
            enabled=True,
            settings={"ambient": {"t_load": 300.0}, "hot": {"t_load": 400.0}},
            variant="low_gain",
        ).yaml_text
        == GRAPH_DOCUMENT
    )
    configured = set_node(
        GRAPH_DOCUMENT,
        "bandpass",
        enabled=True,
        settings={"type": "ReceiverOperator", "bandpass": [1.0, 1.0]},
        variant="low_gain",
    )
    parsed = yaml.safe_load(configured.yaml_text)
    assert parsed["model"] == yaml.safe_load(GRAPH_DOCUMENT)["model"]
    assert tuple(parsed["variants"]["low_gain"]["model"]) == (
        "gain",
        "~foregrounds",
        "bandpass",
    )
    assert parsed["runs"] == []

    disabled = set_node(
        configured.yaml_text,
        "gain",
        enabled=False,
        variant="low_gain",
    )
    patch = yaml.safe_load(disabled.yaml_text)["variants"]["low_gain"]["model"]
    assert "gain" not in patch
    assert patch["~gain"] is None
    assert "gain" not in tuple(
        node.node_id for node in disabled.variant_diagrams[0].nodes if node.lit
    )


def test_many_nodes_keep_list_or_fan_shape_and_only_lists_can_move():
    filters = [
        {"type": "FourierBandFilter", "name": "first"},
        {"type": "SkySpaceFilter", "name": "second"},
        {"type": "DelayFilter", "name": "third"},
    ]
    configured = set_many_node(GRAPH_DOCUMENT, "filters", filters)
    moved = move_node_instance(configured.yaml_text, "filters", 2, 0)
    assert [item["name"] for item in yaml.safe_load(moved.yaml_text)["model"]["filters"]] == [
        "third",
        "first",
        "second",
    ]
    assert tuple(yaml.safe_load(moved.yaml_text)["model"]) == (
        "foregrounds",
        "cal_loads",
        "gain",
        "filters",
    )

    fan = OrderedDict(
        (name, {"t_load": value})
        for name, value in (
            ("ambient", 295.0),
            ("hot", 410.0),
            ("noise_source", 900.0),
        )
    )
    configured_fan = set_many_node(moved.yaml_text, "cal_loads", fan)
    assert tuple(yaml.safe_load(configured_fan.yaml_text)["model"]["cal_loads"]) == tuple(fan)
    with pytest.raises(ConfigError, match="FAN order is locked"):
        move_node_instance(configured_fan.yaml_text, "cal_loads", 2, 0)


def test_compose_preserves_stage_order_and_sibling_routes():
    stages = [
        {"name": "first", "type": "GainOperator", "gain": 0.75},
        {"name": "second", "type": "GainOperator", "gain": 1.25},
    ]
    found = compose_node(GRAPH_DOCUMENT, "gain", "cascade", stages)
    parsed = yaml.safe_load(found.yaml_text)

    assert [stage["name"] for stage in parsed["model"]["gain"]["stages"]] == [
        "first",
        "second",
    ]
    assert parsed["model"]["foregrounds"] == yaml.safe_load(GRAPH_DOCUMENT)["model"]["foregrounds"]
    assert parsed["variants"] == yaml.safe_load(GRAPH_DOCUMENT)["variants"]


def test_region_placement_auto_syncs_the_key_to_at_last_without_reordering_siblings():
    document = """\
model:
  global_signal: {depth: 0.5}
  cw_tone: {python: pkg:Tone, amplitude: 1.0}
  gain: {gain: 1.0}
runs: []
"""
    found = place_node(
        document,
        "cw_tone",
        ["noise", "emi"],
        {"python": "pkg:Tone", "amplitude": 1.0},
    )
    model = yaml.safe_load(found.yaml_text)["model"]
    assert tuple(model) == ("global_signal", "emi", "gain")
    assert model["emi"]["at"] == ["noise", "emi"]
    assert "cw_tone" not in model

    occupied = document.replace(
        "  gain: {gain: 1.0}\n",
        "  emi: {type: EMIOperator}\n  gain: {gain: 1.0}\n",
    )
    with pytest.raises(ConfigError, match="already configured"):
        place_node(
            occupied,
            "cw_tone",
            ["noise", "emi"],
            {"python": "pkg:Tone"},
        )


def test_snapshot_button_updates_model_and_aux_product_in_one_document_transform():
    document = """\
model:
  flagging: {type: FlaggingOperator, threshold: 4.0}
outputs:
  stdout: summary
  write:
    arrays: true
    aux:
      format: npz
      keys: [flags]
runs: []
"""
    found = set_snapshot_before(document, "flagging", "raw")
    parsed = yaml.safe_load(found.yaml_text)
    assert parsed["model"]["flagging"]["snapshot_before"] == "raw"
    assert parsed["outputs"]["write"]["aux"] == {
        "format": "npz",
        "keys": ["flags", "snapshot/raw"],
    }
    assert parsed["outputs"]["write"]["arrays"] is True
    assert parsed["outputs"]["stdout"] == "summary"
    assert parsed["runs"] == []
    assert set_snapshot_before(found.yaml_text, "flagging", "raw").yaml_text == found.yaml_text

    with pytest.raises(ConfigError, match="processing-segment"):
        set_snapshot_before(document, "gain", "raw")


def test_disabling_an_absent_node_is_an_exact_noop_and_reserved_slots_require_python():
    document = "runtime:\n  jax_enable_x64: true\nruns: []\n"
    assert set_node(document, "gain", enabled=False).yaml_text == document
    with pytest.raises(ConfigError, match="reserved.*python"):
        set_node(
            document,
            "atmosphere_field",
            enabled=True,
            settings={"type": "AtmosphereOperator"},
        )
