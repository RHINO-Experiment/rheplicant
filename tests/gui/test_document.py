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


VARIANT_YAML = """\
schema_version: 1
model:
  gain:
    gain: {value: 1.1, unit: dimensionless}
  cw_tone:
    python: {target: example_package.example_module.Tone}
  bandpass:
    bandpass: {ref: resources.arrays.flat}
    nonsense_key: 1
variants:
  low_gain:
    model:
      gain:
        gain: {value: 0.9, unit: dimensionless}
runs:
  - name: forward
    kind: forward
"""


def _card(diagram, node_id: str) -> NodeCard:
    return next(node for node in diagram.nodes if node.node_id == node_id)


class TestTheTypedNodeFormRidesOnTheNodeCard:
    """It lives on the card rather than beside it, because the card is what a
    variant re-resolves. ``session.document.forms`` projects the BASE document
    only, so a typed form driven off that would show base numbers under a
    variant's name -- silently, and only for the fields a variant changed."""

    def test_a_lit_single_slot_node_carries_its_typed_fields(self):
        card = _card(snapshot(VARIANT_YAML).base_diagram, "gain")

        assert card.typed_form is True
        assert card.typed_form_reason is None
        assert card.selected_type == "GainOperator"
        assert card.type_choices == ("GainOperator",)
        assert [field.name for field in card.fields] == ["gain"]
        assert card.fields[0].number == 1.1
        assert card.fields[0].unit == "dimensionless"
        assert card.fields[0].typed is True

    def test_a_variant_card_shows_the_variants_own_numbers(self):
        found = snapshot(VARIANT_YAML)
        base = _card(found.base_diagram, "gain")
        variant = _card(found.variant_diagrams[0], "gain")

        assert found.variant_diagrams[0].name == "low_gain"
        assert base.fields[0].number == 1.1
        assert variant.fields[0].number == 0.9

    def test_a_python_node_shows_the_security_gates_reason(self):
        card = _card(snapshot(VARIANT_YAML).base_diagram, "cw_tone")

        assert card.typed_form is False
        assert card.typed_form_reason == (
            "python: target; its class is not resolved in the browser."
        )
        assert card.fields == ()

    def test_an_unknown_written_key_is_reported_without_disabling_the_form(self):
        card = _card(snapshot(VARIANT_YAML).base_diagram, "bandpass")

        assert card.typed_form is True
        assert card.extra_keys == ("nonsense_key",)

    def test_every_node_card_answers_the_question_one_way_or_the_other(self):
        """A card with neither a typed form nor a reason would render an empty
        panel and no explanation for it."""
        for card in snapshot(VARIANT_YAML).base_diagram.nodes:
            assert (card.typed_form_reason is None) == card.typed_form

    def test_the_card_survives_the_dataclass_serialisation_the_api_uses(self):
        found = dataclasses.asdict(snapshot(VARIANT_YAML).base_diagram)
        card = next(node for node in found["nodes"] if node["node_id"] == "gain")

        assert card["fields"][0]["units"] == ("dimensionless",)
        assert card["fields"][0]["written"] == {"value": 1.1, "unit": "dimensionless"}


class TestTheCardCarriesWhatATypeChangeCosts:
    def test_a_two_class_node_names_what_each_choice_would_remove(self):
        text = VARIANT_YAML.replace(
            "  bandpass:\n", "  noise:\n    type: NoiseOperator\n    sigma: 0.05\n  bandpass:\n"
        )
        card = _card(snapshot(text).base_diagram, "noise")

        assert card.selected_type == "NoiseOperator"
        assert card.removed_by_type == {
            "NoiseOperator": (),
            "RadiometerNoiseOperator": ("sigma",),
        }

    def test_a_refused_node_still_answers_with_empty_costs(self):
        """The client reads this map for every card. A card that answered
        with nothing at all would be a lookup on undefined."""
        for card in snapshot(VARIANT_YAML).base_diagram.nodes:
            assert set(card.removed_by_type) == set(card.type_choices)


MANY_YAML = """\
schema_version: 1
observation:
  switching: {mode: cycle, order: [antenna, hot, cold]}
model:
  filters:
    - {type: SiderealFilter, n_days: 3, mode: extract}
    - {type: FourierBandFilter, axis: time, low: 0 Hz, high: 1 Hz}
  cal_loads:
    hot: {t_load: {value: 350.0, unit: K}}
    cold: {}
runs:
  - name: forward
    kind: forward
"""


class TestEveryInstanceCarriesItsOwnFields:
    def test_a_chain_entry_answers_for_itself(self):
        card = _card(snapshot(MANY_YAML).base_diagram, "filters")

        assert card.typed_form is False, "the node has no single field set"
        assert [instance.instance_id for instance in card.instances] == [
            "filters_1", "filters_2"
        ]
        first, second = card.instances
        assert first.selected_type == "SiderealFilter"
        assert second.selected_type == "FourierBandFilter"
        assert {f.name for f in first.fields} == {"n_days", "mode"}
        assert next(f for f in second.fields if f.name == "low").number == 0.0

    def test_a_fan_label_answers_for_itself(self):
        card = _card(snapshot(MANY_YAML).base_diagram, "cal_loads")

        hot, cold = card.instances
        assert (hot.label, cold.label) == ("hot", "cold")
        assert next(f for f in hot.fields if f.name == "t_load").number == 350.0
        assert next(f for f in cold.fields if f.name == "t_load").present is False

    def test_a_single_slot_node_still_has_no_instances(self):
        card = _card(snapshot(MANY_YAML).base_diagram, "gain")

        assert card.instances == ()
        assert card.typed_form is True

    def test_a_chain_entry_keeps_the_field_every_filter_owns(self):
        card = _card(snapshot(MANY_YAML).base_diagram, "filters")

        assert card.instances[0].removed_by_type["FourierBandFilter"] == ("n_days",)


COMPOSED_YAML = """\
schema_version: 1
model:
  gain:
    compose: cascade
    stages:
      - {name: coarse, type: GainOperator, gain: 1.1}
      - {name: fine, type: GainOperator, gain: 1.01}
  beam_spill:
    from: projector
    projector: {ref: resources.projectors.drift}
    t_ground: {value: 300.0, unit: K}
runs:
  - name: forward
    kind: forward
"""


class TestStagesAndRoutesReachTheCard:
    def test_each_stage_is_named_and_addressed(self):
        card = _card(snapshot(COMPOSED_YAML).base_diagram, "gain")

        assert card.typed_form is False
        assert [stage.label for stage in card.stages] == ["coarse", "fine"]
        assert [stage.slot for stage in card.stages] == [("stages", "0"), ("stages", "1")]
        assert next(f for f in card.stages[0].fields if f.name == "gain").number == 1.1
        assert next(f for f in card.stages[1].fields if f.name == "gain").number == 1.01

    def test_a_from_route_offers_its_own_keys_beside_the_reason(self):
        card = _card(snapshot(COMPOSED_YAML).base_diagram, "beam_spill")

        assert card.typed_form is False
        assert card.typed_form_reason.startswith("from: projector")
        assert [entry.name for entry in card.from_fields] == ["projector", "t_ground"]
        assert next(e for e in card.from_fields if e.name == "t_ground").number == 300.0

    def test_a_plain_node_has_neither(self):
        card = _card(snapshot(COMPOSED_YAML).base_diagram, "noise")

        assert card.stages == ()
        assert card.from_fields == ()

    def test_an_instance_knows_where_it_lives(self):
        card = _card(snapshot(MANY_YAML).base_diagram, "filters")

        assert [instance.slot for instance in card.instances] == [("0",), ("1",)]

    def test_a_fan_instance_is_addressed_by_label(self):
        card = _card(snapshot(MANY_YAML).base_diagram, "cal_loads")

        assert [instance.slot for instance in card.instances] == [("hot",), ("cold",)]


RESOURCE_YAML = """\
schema_version: 1
resources:
  projectors:
    drift: {lmax: 8, optimizations: []}
    second: {lmax: 8, optimizations: []}
  beams:
    horn: {nside: 8}
model:
  observed_astro_sky:
    projector: {ref: resources.projectors.drift}
variants:
  other:
    resources:
      projectors:
        third: {lmax: 8, optimizations: []}
runs:
  - name: forward
    kind: forward
"""


class TestAPickerOffersWhatTheDocumentDeclares:
    def test_it_lists_the_declared_names_of_that_kind_only(self):
        card = _card(snapshot(RESOURCE_YAML).base_diagram, "observed_astro_sky")
        projector = next(f for f in card.fields if f.name == "projector")

        assert projector.control == "resource"
        assert projector.choices == (
            "resources.projectors.drift",
            "resources.projectors.second",
        )
        assert projector.written == {"ref": "resources.projectors.drift"}

    def test_a_variant_offers_the_resources_of_its_own_layer(self):
        """A variant may declare resources the base does not. Offering the
        base's list under a variant's name would propose a reference that
        layer cannot resolve."""
        found = snapshot(RESOURCE_YAML)
        variant = next(d for d in found.variant_diagrams if d.name == "other")
        card = next(n for n in variant.nodes if n.node_id == "observed_astro_sky")
        projector = next(f for f in card.fields if f.name == "projector")

        assert "resources.projectors.third" in projector.choices

    def test_every_field_carries_the_sentence_its_operator_writes(self):
        card = _card(snapshot(RESOURCE_YAML).base_diagram, "observed_astro_sky")

        assert all(entry.help for entry in card.fields)
