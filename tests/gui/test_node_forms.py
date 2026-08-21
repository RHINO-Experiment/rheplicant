"""One lit single-slot node's settings, projected as labelled controls.

The node inspector has one control today: a raw JSON textarea. It stays --
every editable node still gets it, always -- but for the twenty nodes that
hold exactly one operator instance the same settings are also offered as
typed fields, with the unit spellings their own dimension accepts.

The nine gates below decide which nodes those are, and they are ordered: the
first one that fails supplies the reason the inspector shows instead. Gate 5
is a security gate rather than a convenience one -- resolving a ``python:``
target means importing a module the document names, and drawing a form is not
a reason to do that.
"""

from __future__ import annotations

import pytest

from rheplicant.gui.forms import widget_catalog
from rheplicant.gui.node_forms import (
    NodeFieldSet,
    classify_value,
    project_node_fields,
)
from rheplicant.radio.graph import RADIO_GRAPH

CATALOG = widget_catalog()

#: Measured from ``RADIO_GRAPH`` and ``operator_table()``, not asserted from
#: the plan: 18 single-slot nodes hold exactly one operator class, 2 hold two,
#: 1 holds none, 3 are reserved, 5 are junctions or selectors and 4 are many.
ONE_CLASS = (
    "global_signal", "point_sources", "uniform_sky", "ionosphere", "rfi_field",
    "observed_astro_sky", "ground_pickup", "atmosphere", "beam_spill",
    "antenna_loss", "noise_wave", "cw_tone", "bandpass", "gain", "emi", "adc",
    "averaging", "apply_cal",
)
TWO_CLASSES = ("noise", "flagging")
NO_OPERATOR = ("snapshot",)
RESERVED = ("atmosphere_field", "ground_field", "beam")
COMPOSITION = ("astro_sum", "field_sum", "astro_ant_sum", "t_ant_sum", "receiver_input")
MANY = ("foregrounds", "t_sys_extra", "cal_loads", "filters")

#: A module path that does not exist and is never imported. The point of gate
#: 5 is that nothing here is resolved, so the value is arbitrary -- but a
#: fixture naming a real dangerous callable would read as an instruction to
#: someone skimming, and would trip a security scanner besides.
PYTHON_TARGET = "example_package.example_module.ExampleOperator"

COMPOSITION_REASON = "Automatic junction or selector: not an operator slot."
RESERVED_REASON = (
    "Reserved graph slot with no shipped operator; configure it through python:."
)
MANY_REASON = "Many node: each instance carries its own fields."
SHAPE_REASON = "Node settings are not a mapping; edit them as YAML."
COMPOSE_REASON = "Composed node: the stages own the fields."
PYTHON_REASON = "python: target; its class is not resolved in the browser."
AT_REASON = "Region placement: at: requires python:."
NO_OPERATOR_REASON = "No shipped operator registers at this node; python: is the route."


def _project(node_id: str, settings: object) -> NodeFieldSet:
    return project_node_fields(node_id, settings, CATALOG)


def _field(found: NodeFieldSet, name: str):
    return next(field for field in found.fields if field.name == name)


class TestTheCensusThisReleaseCovers:
    def test_the_six_categories_partition_all_thirty_three_nodes(self):
        """A node in no category would silently get no test at all, and a node
        in two would be tested against whichever expectation came first."""
        named = (*ONE_CLASS, *TWO_CLASSES, *NO_OPERATOR, *RESERVED, *COMPOSITION, *MANY)

        assert len(named) == len(set(named)) == 33
        assert set(named) == set(RADIO_GRAPH._topo)


class TestTheGatesInOrder:
    @pytest.mark.parametrize("node_id", COMPOSITION)
    def test_gate_1_a_junction_or_selector_is_not_an_operator_slot(self, node_id):
        found = _project(node_id, {})

        assert found.typed_form is False
        assert found.typed_form_reason == COMPOSITION_REASON

    @pytest.mark.parametrize("node_id", RESERVED)
    def test_gate_2_a_reserved_slot_says_so_before_anything_else(self, node_id):
        """Ordered ahead of gate 8 deliberately: a reserved node also has no
        operator, and "reserved" is the more useful of the two true sentences
        because it names the reason there is no operator."""
        found = _project(node_id, {"python": {"target": PYTHON_TARGET}})

        assert found.typed_form is False
        assert found.typed_form_reason == RESERVED_REASON

    @pytest.mark.parametrize("node_id", MANY)
    def test_gate_3_a_many_node_is_refused_whatever_shape_it_arrives_in(self, node_id):
        """The FAN is why this asks the GRAPH rather than the settings.
        ``cal_loads`` is a label-keyed mapping, so a shape test alone would
        pass it through and then read its LABELS as field names -- offering a
        ``t_load`` control that writes one level too high."""
        for settings in ({}, [{"t_load": 1.0}], {"hot": {"t_load": 1.0}}, None):
            found = _project(node_id, settings)

            assert found.typed_form is False
            assert found.typed_form_reason == MANY_REASON

    def test_gate_3_a_single_node_whose_settings_are_not_a_mapping(self):
        """Unreachable through ``_node_cards`` -- ``document._model`` refuses
        it first -- but this is a public function and a list here would
        otherwise reach ``.get`` on a list."""
        found = _project("gain", [1, 2])

        assert found.typed_form is False
        assert found.typed_form_reason == SHAPE_REASON

    def test_gate_4_a_composed_node_lets_its_stages_own_the_fields(self):
        found = _project("gain", {"compose": [{"type": "GainOperator"}]})

        assert found.typed_form is False
        assert found.typed_form_reason == COMPOSE_REASON

    def test_gate_5_a_python_target_is_never_resolved_to_draw_a_form(self):
        """THE SECURITY GATE. Resolving ``python:`` means importing a module
        the document names. Drawing a form is not a reason to run a user's
        code, so the inspector says why instead."""
        found = _project("gain", {"python": {"target": PYTHON_TARGET}})

        assert found.typed_form is False
        assert found.typed_form_reason == PYTHON_REASON

    def test_gate_5_beats_a_written_type(self):
        """A document carrying both is refused elsewhere; here the safe answer
        has to win regardless of the order the keys were written in."""
        found = _project(
            "gain", {"type": "GainOperator", "python": {"target": PYTHON_TARGET}}
        )

        assert found.typed_form is False
        assert found.typed_form_reason == PYTHON_REASON

    def test_gate_6_a_region_placement_needs_the_python_route(self):
        found = _project("gain", {"at": "receiver_input"})

        assert found.typed_form is False
        assert found.typed_form_reason == AT_REASON

    def test_gate_7_a_from_route_names_the_route_it_refuses(self):
        """The route is in the reason because ``from:`` has several, and
        "not a field set" without saying which one is being talked about
        sends the reader back to the YAML to find out."""
        found = _project("gain", {"from": "preset"})

        assert found.typed_form is False
        assert found.typed_form_reason == "from: preset is a constructor route, not a field set."

    @pytest.mark.parametrize("node_id", NO_OPERATOR)
    def test_gate_8_a_node_no_operator_registers_at(self, node_id):
        found = _project(node_id, {})

        assert found.typed_form is False
        assert found.typed_form_reason == NO_OPERATOR_REASON

    @pytest.mark.parametrize("node_id", TWO_CLASSES)
    def test_gate_9_two_classes_and_no_type_shows_the_select_alone(self, node_id):
        """Not a refusal: the form IS typed, it just cannot know which fields
        to offer until the class is chosen. Taking ``classes[0]`` instead
        would show one class's fields for a document that has chosen none --
        and the config layer refuses exactly that."""
        found = _project(node_id, {})

        assert found.typed_form is True
        assert found.typed_form_reason is None
        assert found.selected_type is None
        assert found.fields == ()
        assert len(found.type_choices) == 2

    @pytest.mark.parametrize("node_id", TWO_CLASSES)
    def test_gate_9_an_unknown_written_type_shows_the_select_alone(self, node_id):
        found = _project(node_id, {"type": "NoSuchOperator"})

        assert found.typed_form is True
        assert found.selected_type is None
        assert found.fields == ()

    @pytest.mark.parametrize("node_id", ONE_CLASS)
    def test_gate_9_one_class_needs_no_written_type(self, node_id):
        found = _project(node_id, {})

        assert found.typed_form is True
        assert found.typed_form_reason is None
        assert found.type_choices == (found.selected_type,)
        assert found.fields, f"{node_id} has a class but projected no fields"


class TestTheValueFormClassifier:
    """Only three written shapes are typed: a bare scalar, the
    ``<number> <unit>`` shorthand, and the ``{value, unit}`` envelope. Every
    other value node -- ``{file: ...}``, ``{linspace: ...}``, anything
    carrying a modifier -- keeps the textarea, because a control that cannot
    represent a shape must not be allowed to overwrite it.
    """

    @pytest.mark.parametrize(
        ("written", "form", "number", "unit"),
        [
            (0, "bare", 0, None),
            (0.5, "bare", 0.5, None),
            (-2.5, "bare", -2.5, None),
            (7, "bare", 7, None),
            (False, "bare", None, None),
            (True, "bare", None, None),
            ("0.1 K", "shorthand", 0.1, "K"),
            ("75 MHz", "shorthand", 75.0, "MHz"),
            ("1e3 Hz", "shorthand", 1000.0, "Hz"),
            ("extract", "bare", None, None),
            ("0.1", "bare", None, None),
            ({"value": 0}, "quantity", 0, None),
            ({"value": 0, "unit": "K"}, "quantity", 0, "K"),
            ({"value": 0.5, "unit": "MHz"}, "quantity", 0.5, "MHz"),
            ({"value": 1, "scale": 2}, "value", None, None),
            ({"value": 1, "unit": "K", "as": "traced"}, "value", None, None),
            ({"file": "beam.h5"}, "file", None, None),
            ({"ref": "resources.arrays.flat"}, "ref", None, None),
            ({"linspace": {"start": 0, "stop": 1, "num": 4}}, "linspace", None, None),
            ({"python": {"target": PYTHON_TARGET}}, "python", None, None),
            ({"value": 1, "ref": "x"}, "unknown", None, None),
            ({}, "unknown", None, None),
            ({"nonsense": 1}, "unknown", None, None),
            ([], "unknown", None, None),
            ([1, 2], "unknown", None, None),
            (None, "null", None, None),
        ],
    )
    def test_the_whole_table(self, written, form, number, unit):
        reading = classify_value(written)

        assert reading.form == form
        assert reading.number == number
        assert type(reading.number) is type(number)
        assert reading.unit == unit

    def test_zero_is_a_written_number_and_not_an_absence(self):
        """``if not value`` is the bug this row exists to refuse: ``0`` is a
        legal, meaningful setting for every field on this graph, and reading
        it as absent would silently offer to fill in a default over it."""
        assert classify_value(0).number == 0
        assert classify_value(0.0).number == 0.0
        assert classify_value({"value": 0, "unit": "K"}).number == 0

    def test_a_bool_carries_no_number_because_bool_is_an_int(self):
        """``isinstance(False, int)`` is True, so a bool reaches a numeric
        branch unless it is stopped first -- and ``false`` arriving as ``0``
        reads as a legal value everywhere downstream.
        ``config/values.py`` keeps a branch for the same reason."""
        assert classify_value(False).number is None
        assert classify_value(True).number is None

    def test_only_the_exact_envelope_keys_are_a_quantity(self):
        """One key beyond ``{value, unit}`` and the control can no longer
        round-trip what is written, so it must not claim to."""
        assert classify_value({"value": 1, "unit": "K"}).form == "quantity"
        for extra in ("scale", "offset", "dtype", "as", "axis", "part"):
            assert classify_value({"value": 1, "unit": "K", extra: 2}).form == "value"


class TestGlobalSignalEndToEnd:
    """The acceptance case: three required traced quantities, two dimensions."""

    SETTINGS = {
        "depth": {"value": 0.5, "unit": "K"},
        "centre": {"value": 75.0, "unit": "MHz"},
        "width": "5 MHz",
    }

    def test_it_projects_three_typed_quantity_fields(self):
        found = _project("global_signal", self.SETTINGS)

        assert found.typed_form is True
        assert found.selected_type == "GlobalSignalOperator"
        assert set(field.name for field in found.fields) == {"depth", "centre", "width"}
        assert all(field.control == "quantity" for field in found.fields)
        assert all(field.required for field in found.fields)
        assert all(field.delivery == "traced" for field in found.fields)
        assert all(field.typed for field in found.fields)
        assert found.extra_keys == ()

    def test_each_field_offers_the_spellings_its_own_dimension_accepts(self):
        found = _project("global_signal", self.SETTINGS)

        assert _field(found, "depth").units == ("K", "celsius")
        assert _field(found, "centre").units == ("Hz", "kHz", "MHz", "GHz")
        assert _field(found, "width").units == ("Hz", "kHz", "MHz", "GHz")

    def test_the_shorthand_and_the_envelope_read_the_same_way(self):
        found = _project("global_signal", self.SETTINGS)

        assert (_field(found, "centre").number, _field(found, "centre").unit) == (75.0, "MHz")
        assert (_field(found, "width").number, _field(found, "width").unit) == (5.0, "MHz")
        assert _field(found, "centre").form == "quantity"
        assert _field(found, "width").form == "shorthand"

    def test_an_unwritten_field_is_an_empty_control_rather_than_a_missing_one(self):
        found = _project("global_signal", {"depth": {"value": 0.5, "unit": "K"}})

        centre = _field(found, "centre")
        assert centre.present is False
        assert centre.form == "absent"
        assert centre.number is None
        assert centre.typed is True

    def test_a_field_written_as_zero_reads_as_written(self):
        """The classifier knowing ``0`` is a number is not enough: the
        PROJECTION has to agree, and ``present = ... and bool(value)`` is the
        one-token change that makes a written zero read as an empty control
        offering to fill in a default over it. Caught by mutation V2, which
        survived a suite that only tested the classifier in isolation."""
        for written, number in (
            (0, 0),
            (0.0, 0.0),
            ({"value": 0, "unit": "dimensionless"}, 0),
            ("0 dimensionless", 0.0),
        ):
            found = _project("gain", {"gain": written})
            field = _field(found, "gain")

            assert field.present is True, written
            assert field.number == number, written
            assert type(field.number) is type(number), written
            assert field.typed is True, written
            assert field.form != "absent", written

    def test_an_empty_string_and_a_false_are_also_written_values(self):
        """Same trap, other falsy shapes: both are values a document can
        legitimately carry, and neither is an absence."""
        assert _field(_project("noise_wave", {"switch_key": ""}), "switch_key").present is True
        assert _field(_project("noise_wave", {"switch_key": False}), "switch_key").present is True

    def test_a_shape_the_controls_cannot_represent_keeps_its_textarea(self):
        found = _project(
            "global_signal",
            {**self.SETTINGS, "depth": {"file": "depth.h5", "column": "d"}},
        )

        assert found.typed_form is True
        assert _field(found, "depth").typed is False
        assert _field(found, "depth").form == "file"


class TestTheSelectedClassDecidesTheFields:
    def test_noise_without_a_type_offers_the_choices_and_no_fields(self):
        found = _project("noise", {})

        assert set(found.type_choices) == {"NoiseOperator", "RadiometerNoiseOperator"}
        assert found.selected_type is None
        assert found.fields == ()

    def test_radiometer_noise_offers_its_own_two_fields_and_not_sigma(self):
        """``sigma`` belongs to ``NoiseOperator``. Both classes register at
        this node, so both fields are in the catalog under the same node --
        offering the wrong one is the failure mode this test exists for."""
        found = _project(
            "noise",
            {"type": "RadiometerNoiseOperator", "channel_width": "1 MHz"},
        )

        assert found.selected_type == "RadiometerNoiseOperator"
        assert {field.name for field in found.fields} == {
            "channel_width",
            "integration_time",
        }

    def test_plain_noise_offers_sigma(self):
        found = _project("noise", {"type": "NoiseOperator", "sigma": {"value": 0.05, "unit": "K"}})

        assert {field.name for field in found.fields} == {"sigma"}
        assert _field(found, "sigma").number == 0.05


class TestControlDerivation:
    """``toggle`` is deliberately unreachable: the live census of all 66 model
    fields is 38 traced, 15 static_float, 6 static_int, 5 static_str, 1
    static_mapping, 1 static_tuple and ZERO static_bool. It is declared so the
    first boolean field gets a checkbox rather than a text box."""

    @pytest.mark.parametrize(
        ("node_id", "field_name", "control"),
        [
            ("global_signal", "depth", "quantity"),
            ("cw_tone", "lineshape", "select"),
            ("noise_wave", "switch_key", "text"),
            ("adc", "n_bits", "integer"),
        ],
    )
    def test_the_control_follows_delivery_and_disposition(self, node_id, field_name, control):
        found = _project(node_id, {})

        assert _field(found, field_name).control == control

    def test_no_model_field_projects_a_toggle_today(self):
        settings_by_node = [(node_id, {}) for node_id in ONE_CLASS]
        settings_by_node += [
            (node_id, {"type": name})
            for node_id in TWO_CLASSES
            for name in _project(node_id, {}).type_choices
        ]
        for node_id, settings in settings_by_node:
            found = _project(node_id, settings)
            assert found.fields
            assert all(field.control != "toggle" for field in found.fields)

    def test_an_enum_field_carries_its_members_and_a_free_string_does_not(self):
        assert _field(_project("cw_tone", {}), "lineshape").choices == ("sinc2", "gaussian")
        assert _field(_project("noise_wave", {}), "switch_key").choices == ()


class TestExtraKeysNeverDisableTheTypedForm:
    def test_snapshot_before_and_eqx_leaves_land_in_extra_keys(self):
        """Both are real, supported keys that this release has no control for.
        Refusing the whole form over them would take the typed fields away
        from every node that uses either."""
        found = _project(
            "gain",
            {
                "gain": {"value": 1.1, "unit": "dimensionless"},
                "snapshot_before": "pre_gain",
                "eqx_leaves": "leaves.eqx",
            },
        )

        assert found.typed_form is True
        assert found.typed_form_reason is None
        assert set(found.extra_keys) == {"snapshot_before", "eqx_leaves"}
        assert _field(found, "gain").number == 1.1

    def test_type_is_not_an_extra_key(self):
        found = _project("noise", {"type": "NoiseOperator"})

        assert found.extra_keys == ()

    def test_an_unknown_key_is_reported_rather_than_dropped(self):
        found = _project("gain", {"nonsense": 1})

        assert found.extra_keys == ("nonsense",)


class TestWhatAChangeOfTypeWouldCost:
    """Changing ``type:`` is confirm-and-clear, and the confirmation has to
    name what it clears. The server computes that: it owns the catalog, so it
    is the only side that knows which written keys belong to which class."""

    def test_it_names_the_keys_the_other_class_has_no_place_for(self):
        found = _project(
            "noise", {"type": "NoiseOperator", "sigma": {"value": 0.05, "unit": "K"}}
        )

        assert found.removed_by_type == {
            "NoiseOperator": (),
            "RadiometerNoiseOperator": ("sigma",),
        }

    def test_an_unwritten_field_costs_nothing_to_abandon(self):
        found = _project("noise", {"type": "NoiseOperator"})

        assert found.removed_by_type == {"NoiseOperator": (), "RadiometerNoiseOperator": ()}

    def test_keys_no_class_owns_are_never_offered_up(self):
        """``snapshot_before`` and ``eqx_leaves`` belong to the node, not to
        its operator class, and an unknown key belongs to nobody. A type
        change has no business removing any of them."""
        found = _project(
            "flagging",
            {
                "type": "FlaggingOperator",
                "threshold": 5.0,
                "snapshot_before": "pre_flag",
                "eqx_leaves": "leaves.eqx",
                "nonsense": 1,
            },
        )

        assert found.removed_by_type["MomentRFIFlaggingOperator"] == ("threshold",)
        assert set(found.extra_keys) == {"snapshot_before", "eqx_leaves", "nonsense"}

    def test_a_document_that_has_chosen_no_class_can_lose_nothing(self):
        found = _project("noise", {"sigma": {"value": 0.05, "unit": "K"}})

        assert found.selected_type is None
        assert found.removed_by_type == {"NoiseOperator": (), "RadiometerNoiseOperator": ()}

    @pytest.mark.parametrize("node_id", ONE_CLASS)
    def test_a_single_class_node_has_one_entry_and_nothing_to_lose(self, node_id):
        found = _project(node_id, {})

        assert list(found.removed_by_type) == list(found.type_choices)
        assert all(cost == () for cost in found.removed_by_type.values())

    def test_a_field_both_classes_own_survives_the_change(self):
        """Asserted on the rule itself, because no node this release can reach
        exercises it: both two-class nodes share NO fields, so today the
        subtraction below is a no-op on every live node. The overlap does
        exist in the catalog -- all three filters own ``mode`` -- and becomes
        reachable the moment many-node cards arrive."""
        from rheplicant.gui.node_forms import _removed_by_type

        assert _removed_by_type(
            written=("mode", "axis", "low", "snapshot_before"),
            current=frozenset({"mode", "axis", "low", "high"}),
            candidate=frozenset({"mode", "n_days"}),
        ) == ("axis", "low")

    def test_todays_two_class_nodes_really_do_share_nothing(self):
        """The census the test above is excused by. If this goes red, an
        end-to-end test of the shared-field rule is now possible and should
        be written."""
        from rheplicant.config.delivery import field_specs
        from rheplicant.gui.form_catalog import operator_table

        for node_id in TWO_CLASSES:
            first, second = operator_table()[node_id]
            assert not set(field_specs(first)) & set(field_specs(second)), node_id


FILTERS = [
    {"type": "SiderealFilter", "n_days": 3, "mode": "extract"},
    {"type": "FourierBandFilter", "axis": "time", "low": "0 Hz", "high": "1 Hz"},
]
CAL_LOADS = {"hot": {"t_load": {"value": 350.0, "unit": "K"}}, "cold": {}}


def _instances(node_id: str, settings: object):
    from rheplicant.gui.node_forms import project_node_instances

    return project_node_instances(node_id, settings, CATALOG)


class TestOneFieldSetPerInstance:
    """A ``many`` node has no single field set -- that is what gate 3 says --
    but each of its instances is one operator's settings and has one."""

    def test_the_node_itself_points_at_its_instances(self):
        found = _project("filters", FILTERS)

        assert found.typed_form is False
        assert found.typed_form_reason == MANY_REASON
        assert found.fields == ()

    def test_a_chain_projects_one_set_per_entry_in_order(self):
        first, second = _instances("filters", FILTERS)

        assert (first.instance_key, second.instance_key) == ("0", "1")
        assert first.selected_type == "SiderealFilter"
        assert second.selected_type == "FourierBandFilter"
        assert {f.name for f in first.fields} == {"n_days", "mode"}
        assert {f.name for f in second.fields} == {
            "axis", "low", "high", "mode",
        }

    def test_each_entry_reads_its_own_values(self):
        first, second = _instances("filters", FILTERS)

        assert _field(first, "n_days").number == 3
        assert _field(first, "mode").written == "extract"
        assert _field(second, "low").number == 0.0
        assert _field(second, "low").unit == "Hz"

    def test_a_fan_projects_one_set_per_label_in_document_order(self):
        hot, cold = _instances("cal_loads", CAL_LOADS)

        assert (hot.instance_key, cold.instance_key) == ("hot", "cold")
        assert _field(hot, "t_load").number == 350.0
        assert _field(cold, "t_load").present is False
        assert _field(cold, "t_load").typed is True

    def test_a_shape_that_is_neither_projects_nothing(self):
        assert _instances("filters", {"not": "a list"}) == ()
        assert _instances("cal_loads", [{"t_load": 1.0}]) == ()
        assert _instances("filters", None) == ()

    def test_a_single_slot_node_has_no_instances(self):
        assert _instances("gain", {"gain": 1.0}) == ()

    def test_an_instance_answers_the_gates_on_its_own(self):
        """Each entry is one operator's settings, so gates 4 to 9 are asked of
        it rather than of the node. Gate 5 in particular: a ``python:`` entry
        in a chain is still a module the document names."""
        gated = _instances(
            "filters",
            [
                {"python": {"target": PYTHON_TARGET}},
                {"type": "SiderealFilter", "n_days": 1},
            ],
        )

        assert gated[0].typed_form is False
        assert gated[0].typed_form_reason == PYTHON_REASON
        assert gated[1].typed_form is True

    def test_a_field_every_class_owns_survives_a_type_change_here(self):
        """The end-to-end case P1 could only assert on the rule: all three
        filters own ``mode``, so changing one entry's class keeps it and
        drops only what the new class has no field for."""
        first, _second = _instances("filters", FILTERS)

        assert first.removed_by_type["FourierBandFilter"] == ("n_days",)
        assert "mode" not in first.removed_by_type["FourierBandFilter"]
        assert first.removed_by_type["SkySpaceFilter"] == ("n_days",)
