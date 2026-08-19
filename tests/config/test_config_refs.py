"""Forms 5 and 7: dimension-aware references and destination-preserving stacks."""

import dataclasses
import types

import jax.numpy as jnp
import pytest

from rheplicant.config import ConfigError
from rheplicant.config.context import ResolutionContext
from rheplicant.config.values import resolve_value


@pytest.fixture
def context():
    gamma = jnp.asarray([0.1 + 0.2j, 0.3 + 0.4j])
    beam = types.SimpleNamespace(maps=jnp.ones((2, 12)), sky_fraction=jnp.asarray([0.5, 0.5]))
    return ResolutionContext(
        freq=jnp.linspace(60e6, 85e6, 2),
        time=jnp.arange(4.0),
        dtype="float32",
        switch_order=("antenna", "ambient", "hot"),
        resources={
            "resources.arrays.gamma": gamma,
            "resources.beams.horn": beam,
            "resources.s_params.ambient": jnp.asarray([0.01 + 0.0j, 0.02 + 0.0j]),
            "resources.s_params.hot": jnp.asarray([0.03 + 0.0j, 0.04 + 0.0j]),
            "resources.s_params.antenna": jnp.asarray([0.05 + 0.0j, 0.06 + 0.0j]),
        },
    )


class TestRefIsIdentityNotACopy:
    def test_it_returns_the_same_object(self, context):
        """radio/instrument/beam_spill.py:89 from_projector is 'the one call
        that cannot get the weight and the sky average out of step'. Two
        projectors that nominally share a beam must share the ARRAY; a loader
        that reconstructs each reference passes every shape check and silently
        decouples them."""
        got = resolve_value({"ref": "resources.arrays.gamma"}, context)
        assert got.value is context.resources["resources.arrays.gamma"]

    def test_two_references_to_one_name_are_the_same_object(self, context):
        first = resolve_value({"ref": "resources.arrays.gamma"}, context).value
        second = resolve_value({"ref": "resources.arrays.gamma"}, context).value
        assert first is second

    def test_a_named_sub_value_is_reachable(self, context):
        """radio/beams.py:150 horizon_truncated_beam returns
        (truncated_maps, sky_fraction) -- one call, two products -- and
        sky_fraction is exactly what BeamSpillOperator(sky_fraction=) wants.
        v0 consumed the maps and dropped the fraction, leaving the user with
        from: projector, which on a truncated beam returns about 1.0 and
        silently deletes the (1 - f_sky) * T_ground term."""
        got = resolve_value({"ref": "resources.beams.horn.sky_fraction"}, context)
        assert got.value is context.resources["resources.beams.horn"].sky_fraction

    def test_a_ref_to_a_non_array_object_survives_the_modifier_pass(self, context):
        """Catches an implementation that puts every form's result through
        jnp.asarray on the way out: a beam, a projector and an operator are all
        legal ref targets and none of them is an array, so a coercion here
        raises on a document that is entirely correct."""
        got = resolve_value({"ref": "resources.beams.horn"}, context)
        assert got.value is context.resources["resources.beams.horn"]
        assert got.source == "ref"

    def test_a_modifier_on_a_ref_does_not_mutate_the_referenced_object(self, context):
        original = context.resources["resources.arrays.gamma"]
        got = resolve_value({"ref": "resources.arrays.gamma", "part": "re"}, context)
        assert jnp.iscomplexobj(original), "the shared object must be untouched"
        assert not jnp.iscomplexobj(got.value)
        # Catches a form that writes its modified result back into the context:
        # a jnp array cannot be mutated in place, but the mapping entry can be
        # rebound, and then the first document line to name gamma decides what
        # every later one gets. Asserted against the context, not against
        # `original`, because rebinding leaves `original` complex either way.
        assert context.resources["resources.arrays.gamma"] is original
        assert resolve_value({"ref": "resources.arrays.gamma"}, context).value is original

    def test_a_modifier_on_a_sub_value_ref_leaves_the_attribute_alone(self, context):
        """The attribute route is the one where in-place really is reachable --
        setattr on a beam works where .at[].set() on an array does not -- so a
        scaled sky_fraction stored back onto the beam would be read by every
        BeamSpillOperator built from it afterwards."""
        beam = context.resources["resources.beams.horn"]
        before = beam.sky_fraction
        got = resolve_value({"ref": "resources.beams.horn.sky_fraction", "scale": 2.0}, context)
        assert float(got.value[0]) == pytest.approx(1.0)
        assert beam.sky_fraction is before

    def test_a_unit_on_a_ref_converts_and_the_bare_ref_declares_none(self, context):
        """Catches the unit branch dropped (the ref returned raw with a unit
        written, which is a factor of 1e6 on a frequency) and catches it
        applied unconditionally, which would make a bare ref a conversion
        against a unit that is None."""
        converted = resolve_value(
            {"ref": "resources.beams.horn.sky_fraction", "unit": "MHz"}, context
        )
        assert float(converted.value[0]) == pytest.approx(0.5e6)
        assert converted.unit.canonical == "Hz"
        assert resolve_value({"ref": "resources.arrays.gamma"}, context).unit is None


class TestRefRefusals:
    def test_an_unknown_name_lists_the_names_that_exist(self, context):
        with pytest.raises(ConfigError) as excinfo:
            resolve_value({"ref": "resources.arrays.gama"}, context)
        message = str(excinfo.value)
        assert "gama" in message
        assert "resources.arrays.gamma" in message

    def test_an_unknown_sub_value_names_what_the_object_offers(self, context):
        with pytest.raises(ConfigError) as excinfo:
            resolve_value({"ref": "resources.beams.horn.fraction"}, context)
        message = str(excinfo.value)
        assert "fraction" in message
        assert "sky_fraction" in message

    def test_a_ref_that_names_nothing_under_resources_is_refused(self, context):
        """Catches the resources. prefix guard removed. Without it the name
        falls through to 'no resource named ...', which lists the resources
        and so still mentions 'resources.' -- but a document reaching for an
        axis does not want a longer list of resources, it wants to be told the
        axis has its own form. The remedy is the part that has to survive."""
        with pytest.raises(ConfigError) as excinfo:
            resolve_value({"ref": "observation.freq.grid"}, context)
        message = str(excinfo.value)
        assert "resources." in message
        assert "from_grid" in message

    def test_a_ref_that_is_not_a_string_is_refused(self, context):
        with pytest.raises(ConfigError) as excinfo:
            resolve_value({"ref": 42}, context)
        assert "42" in str(excinfo.value)


class TestStack:
    def test_it_adds_one_axis(self, context):
        got = resolve_value(
            {"stack": [{"list": [1.0, 2.0]}, {"list": [3.0, 4.0]}], "axis": 0,
             "unit": "dimensionless"},
            context,
        )
        assert got.value.shape == (2, 2)

    def test_it_is_a_container_not_a_computation(self, context):
        got = resolve_value({"stack": [{"value": 1.0}, {"value": 2.0}]}, context)
        assert [float(v) for v in got.value] == pytest.approx([1.0, 2.0])

    def test_an_integer_axis_stacks_and_a_string_axis_is_the_modifier(self, context):
        deep = resolve_value(
            {"stack": [{"list": [1.0, 2.0]}, {"list": [3.0, 4.0]}], "axis": 1}, context
        )
        assert deep.value.shape == (2, 2)
        flagged = resolve_value(
            {"stack": [{"value": 1.0}, {"value": 2.0}], "axis": "freq"}, context
        )
        assert flagged.value.shape == (2,)
        assert flagged.modifiers["axis"] == "freq"

    def test_the_integer_axis_decides_where_the_new_axis_goes(self, context):
        """Catches `axis:` read but ignored, and catches it hardcoded to 0. Two
        equal-length entries make BOTH axes give shape (2, 2), so a shape
        assertion alone leaves the mutation alive; the interleaving is what
        differs, and a stacked-on-the-wrong-axis gamma_src is exactly the
        shape-legal transposition this form exists to prevent."""
        rows = resolve_value(
            {"stack": [{"list": [1.0, 2.0]}, {"list": [3.0, 4.0]}], "axis": 0}, context
        )
        columns = resolve_value(
            {"stack": [{"list": [1.0, 2.0]}, {"list": [3.0, 4.0]}], "axis": 1}, context
        )
        assert [float(v) for v in rows.value[0]] == pytest.approx([1.0, 2.0])
        assert [float(v) for v in columns.value[0]] == pytest.approx([1.0, 3.0])

    def test_an_empty_or_non_list_stack_is_refused(self, context):
        for bad in ([], {"list": [1.0]}, None):
            with pytest.raises(ConfigError) as excinfo:
                resolve_value({"stack": bad}, context)
            # Not merely "stack" -- the unregistered-form refusal names the
            # form too, so this test would pass with no resolver at all.
            assert "non-empty list of value nodes" in str(excinfo.value)

    def test_a_ref_inside_a_stack_reaches_the_named_object(self, context):
        """Catches a stack that resolves its entries as literals rather than as
        value nodes -- the entries are the same grammar as anything else, which
        is what makes {stack: [{ref: ...}, {ref: ...}]} the long-hand of
        from_switch_order."""
        got = resolve_value(
            {"stack": [{"ref": "resources.s_params.antenna"},
                       {"ref": "resources.s_params.hot"}], "part": "re"},
            context,
        )
        assert got.value.shape == (2, 2)
        assert float(got.value[0, 0]) == pytest.approx(0.05)


class TestFromSwitchOrder:
    def test_it_matches_by_name_not_by_position(self, context):
        """noise_wave.gamma_src's row order is fixed by switching.order, and a
        transposition there is shape-legal and costs tens of kelvin
        (radio/instrument/noise_wave.py:83-86). Matching by name is what makes
        check A15 a structural consequence rather than a separate guard."""
        got = resolve_value(
            {"from_switch_order": {"resource": "resources.s_params", "part": "re"}}, context
        )
        assert got.value.shape == (3, 2)
        assert float(got.value[0, 0]) == pytest.approx(0.05)  # antenna, first
        assert float(got.value[1, 0]) == pytest.approx(0.01)  # ambient, second
        assert float(got.value[2, 0]) == pytest.approx(0.03)  # hot, third

    def test_reordering_the_resources_mapping_does_not_reorder_the_rows(self, context):
        """The declaration order of a YAML mapping is not the switching order,
        and a loader that stacks resources.items() in the order it walks them
        agrees with the fixture often enough to look right. Catches matching by
        position, and catches sorting the labels (which here would put ambient
        first)."""
        reversed_resources = dict(reversed(list(context.resources.items())))
        shuffled = dataclasses.replace(context, resources=reversed_resources)
        got = resolve_value(
            {"from_switch_order": {"resource": "resources.s_params", "part": "re"}}, shuffled
        )
        assert [float(row[0]) for row in got.value] == pytest.approx([0.05, 0.01, 0.03])

    def test_part_im_takes_the_other_half(self, context):
        """Catches re and im swapped inside the form. The fixture's s_params
        are real-valued complex numbers, so the imaginary rows are all zero and
        the real rows are not -- which is the direction that matters:
        NoiseWaveOperator takes gamma_src_re and gamma_src_im as separate
        fields and a zeroed sine component is a well-formed model."""
        real = resolve_value(
            {"from_switch_order": {"resource": "resources.s_params", "part": "re"}}, context
        )
        imaginary = resolve_value(
            {"from_switch_order": {"resource": "resources.s_params", "part": "im"}}, context
        )
        assert float(real.value[0, 0]) == pytest.approx(0.05)
        assert float(imaginary.value[0, 0]) == pytest.approx(0.0)
        assert not bool(jnp.any(imaginary.value))

    def test_a_label_with_no_entry_is_refused_and_both_sides_are_listed(self, context):
        thinned = dataclasses.replace(
            context,
            resources={k: v for k, v in context.resources.items() if k != "resources.s_params.hot"},
        )
        with pytest.raises(ConfigError) as excinfo:
            resolve_value(
                {"from_switch_order": {"resource": "resources.s_params", "part": "re"}}, thinned
            )
        message = str(excinfo.value)
        assert "hot" in message  # the label with nothing behind it
        assert "antenna" in message  # the order it came from

    def test_an_extra_entry_the_order_does_not_name_is_refused(self, context):
        widened = context.with_resource("resources.s_params.spare", jnp.asarray([0.0 + 0.0j]))
        with pytest.raises(ConfigError) as excinfo:
            resolve_value(
                {"from_switch_order": {"resource": "resources.s_params", "part": "re"}}, widened
            )
        assert "spare" in str(excinfo.value)

    def test_a_run_with_no_switching_order_is_refused(self, context):
        """Catches the empty-order guard dropped. It survives a bare
        'switching' assertion, because with entries present the extra-entry
        check fires instead and its message quotes switching.order -- so the
        assertion is on the half only this guard says. And with NOTHING under
        the prefix nothing fires at all: jnp.stack([]) then raises a
        ValueError naming neither the document nor switching.mode."""
        orderless = dataclasses.replace(context, switch_order=())
        with pytest.raises(ConfigError) as excinfo:
            resolve_value(
                {"from_switch_order": {"resource": "resources.s_params", "part": "re"}}, orderless
            )
        assert "switching.mode" in str(excinfo.value)
        with pytest.raises(ConfigError):
            resolve_value({"from_switch_order": {"resource": "resources.nothing"}}, orderless)

    def test_a_malformed_spec_and_an_unknown_part_are_refused(self, context):
        """Catches the spec shape check and the part alphabet each dropped. An
        unknown part must not fall through as 'leave it complex': gamma_src_re
        is a real field and a complex array reaching it is a separate, later,
        much less legible failure."""
        with pytest.raises(ConfigError) as excinfo:
            resolve_value({"from_switch_order": "resources.s_params"}, context)
        assert "resource" in str(excinfo.value)
        with pytest.raises(ConfigError) as excinfo:
            resolve_value(
                {"from_switch_order": {"resource": "resources.s_params", "part": "real"}}, context
            )
        assert "real" in str(excinfo.value)
        with pytest.raises(ConfigError) as excinfo:
            resolve_value(
                {"from_switch_order": {"resource": "resources.s_params", "prt": "re"}}, context
            )
        assert "prt" in str(excinfo.value)

    def test_a_deeper_name_under_the_prefix_is_not_a_switch_entry(self, context):
        """resources.s_params.hot.extra is an attribute of one entry, not a
        fourth entry. Catches a prefix match written as startswith alone, which
        would count it as an extra label and refuse a document that is right."""
        nested = context.with_resource(
            "resources.s_params.hot.calibration", jnp.asarray([0.0 + 0.0j])
        )
        got = resolve_value(
            {"from_switch_order": {"resource": "resources.s_params", "part": "re"}}, nested
        )
        assert got.value.shape == (3, 2)
