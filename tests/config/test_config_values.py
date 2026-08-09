"""The form dispatcher, and form 1: a scalar with its unit."""

import jax.numpy as jnp
import pytest

from rheplicant.config import ConfigError
from rheplicant.config import values as values_module
from rheplicant.config.context import ResolutionContext
from rheplicant.config.values import VALUE_FORMS, ResolvedValue, register_form, resolve_value


@pytest.fixture
def context():
    return ResolutionContext(
        freq=jnp.linspace(60e6, 85e6, 8),
        time=jnp.arange(64.0),
        dtype="float32",
    )


class TestFormOneScalar:
    def test_a_mapping_with_value_and_unit(self, context):
        resolved = resolve_value({"value": 290.0, "unit": "K"}, context)
        assert resolved.value == pytest.approx(290.0)
        assert resolved.unit.canonical == "K"
        assert resolved.source == "scalar"

    def test_the_unit_is_converted_on_read(self, context):
        assert resolve_value({"value": 60.0, "unit": "MHz"}, context).value == pytest.approx(6e7)

    def test_the_shorthand_parses_to_the_same_thing(self, context):
        shorthand = resolve_value("290 K", context)
        longhand = resolve_value({"value": 290.0, "unit": "K"}, context)
        assert shorthand.value == pytest.approx(longhand.value)
        assert shorthand.unit.canonical == longhand.unit.canonical

    def test_the_shorthand_carries_a_compound_unit(self, context):
        assert resolve_value("2.5 adc_count/K", context).unit.canonical == "adc_count/K"

    def test_a_bare_number_is_dimensionless(self, context):
        """Legal only where the key is marked dimensionless or count -- which
        is the caller's business, not this function's. What this layer refuses
        to do is invent a unit for it."""
        resolved = resolve_value(0.97, context)
        assert resolved.value == pytest.approx(0.97)
        assert resolved.unit is None

    def test_a_bare_integer_stays_an_integer(self, context):
        """n_bits: 12 must not become 12.0 on the way to a static int field."""
        assert type(resolve_value(12, context).value) is int

    def test_a_boolean_stays_a_boolean(self, context):
        """MEASURED: deleting the bool branch changes nothing -- bool IS an int
        in Python and the int branch returns the same object by the same
        expression, so that mutant is equivalent. What this catches is the
        branch that NORMALISES: int(node) or float(node), after which a bool
        field receives 1 and reads as true whatever the document said."""
        assert type(resolve_value(True, context).value) is bool
        assert resolve_value(False, context).value is False

    def test_the_modifiers_are_carried_on_the_result(self, context):
        """`axis:` is recorded here and applied nowhere, so a resolver that
        drops it is invisible until an array form goes looking. Both returns
        of _resolve_scalar are checked -- they are two statements and only one
        of them runs per call.

        The value was `1` until modifiers.py gave `axis:` its closed table
        (`time`, `freq`, `none`) and resolve_value started validating against
        it. Nothing about what this test asserts has changed -- `1` was only
        ever a stand-in for "some key the scalar branch must not drop" -- but
        it is now a value the grammar refuses, so it names a real axis."""
        assert resolve_value({"value": 1.0, "unit": "K", "axis": "freq"}, context).modifiers == {
            "unit": "K",
            "axis": "freq",
        }
        assert resolve_value({"value": 1.0, "axis": "freq"}, context).modifiers == {"axis": "freq"}

    def test_a_string_that_is_not_a_number_and_a_unit_is_refused(self, context):
        with pytest.raises(ConfigError) as excinfo:
            resolve_value("about 290 K", context)
        message = str(excinfo.value)
        assert "about 290 K" in message
        assert "value:" in message  # the longhand remedy


class TestTheDispatcher:
    def test_two_forms_in_one_mapping_are_refused_and_both_are_named(self, context):
        with pytest.raises(ConfigError) as excinfo:
            resolve_value({"value": 1.0, "zeros": [4]}, context)
        message = str(excinfo.value)
        assert "value" in message
        assert "zeros" in message

    def test_a_mapping_with_no_form_is_refused_and_the_forms_are_listed(self, context):
        with pytest.raises(ConfigError) as excinfo:
            resolve_value({"unit": "K"}, context)
        message = str(excinfo.value)
        for form in VALUE_FORMS:
            assert form in message, form

    def test_an_unknown_key_beside_a_form_is_refused(self, context):
        """Check A1 at the value-node level. A typo'd modifier is the failure
        this exists for: `scaling: 20` beside a linspace is silently ignored
        by any loader that only reads the keys it knows."""
        with pytest.raises(ConfigError) as excinfo:
            resolve_value({"value": 1.0, "unit": "K", "scaling": 20.0}, context)
        message = str(excinfo.value)
        assert "scaling" in message
        assert "scale" in message  # the real key, named because it is IN the modifier table

    def test_the_refusal_names_the_near_modifier_and_not_only_the_table(self, context):
        """MEASURED: replace the whole `near` comprehension with repr(key) and
        the test above still passes, because 'scale' is in the table listing
        that follows. This one splits the message at the table and asks for
        the suggestion on its own."""
        with pytest.raises(ConfigError) as excinfo:
            resolve_value({"value": 1.0, "scaling": 20.0}, context)
        suggestion, _, table = str(excinfo.value).partition("The forms are")
        assert "did you mean" in suggestion
        assert "scale" in suggestion
        assert "scale" in table  # and the full table is still listed after it

    def test_a_bare_list_is_refused_rather_than_falling_through(self, context):
        """A YAML sequence and a stack share a Python type, so a bare list has
        to be refused by name. Catches the non-mapping guard being dropped:
        the key loop then runs over the list's ELEMENTS and dies with a
        TypeError from key[:3], which is not a refusal anyone can act on."""
        with pytest.raises(ConfigError) as excinfo:
            resolve_value([1.0, 2.0], context)
        message = str(excinfo.value)
        assert "list" in message
        assert "list:" in message  # the remedy, not merely the type name

    def test_a_registered_form_is_dispatched_to_its_resolver(self, monkeypatch, context):
        """The extension point Tasks 6..12 hang off. Catches register_form
        returning the function without storing it, and catches the dispatcher
        passing the resolver something other than (node, context, modifiers)."""
        monkeypatch.setattr(values_module, "_RESOLVERS", dict(values_module._RESOLVERS))
        seen = {}

        @register_form("zeros")
        def _resolver(node, ctx, modifiers):
            seen.update(node=node, ctx=ctx, modifiers=modifiers)
            return ResolvedValue(0.0, None, "zeros", modifiers)

        resolved = resolve_value({"zeros": [4], "unit": "K"}, context)
        assert resolved.source == "zeros"
        assert seen["node"] == {"zeros": [4], "unit": "K"}
        assert seen["ctx"] is context
        assert seen["modifiers"] == {"unit": "K"}


class TestTheContextIsAValue:
    def test_adding_a_resource_returns_a_new_context(self, context):
        widened = context.with_resource("resources.arrays.g", object())
        assert "resources.arrays.g" in widened.resources
        assert "resources.arrays.g" not in context.resources

    def test_the_shape_scope_comes_off_the_axes(self, context):
        assert context.shape_scope.n_freq == 8
        assert context.shape_scope.n_time == 64

    def test_n_source_falls_back_to_one_when_nothing_is_switching(self, context):
        """MEASURED: drop the `or 1` and every other test here still passes,
        while the common case -- a run with no switching -- gets n_source = 0
        and any shape declared against it becomes a zero-length axis: finite,
        correctly-shaped and empty. The second assert is the other half, so
        the fallback cannot be a constant 1 instead."""
        assert context.shape_scope.n_source == 1
        switching = ResolutionContext(switch_order=("antenna", "load", "noise"))
        assert switching.shape_scope.n_source == 3

    def test_only_beams_and_projectors_are_offered_as_shape_candidates(self, context):
        """symbols.py quotes these names when n_pix has no nside to resolve
        against. Catches the prefix filter being dropped, which would offer
        every array as somewhere an nside might be declared -- a refusal that
        sends the reader to entries which cannot answer it."""
        widened = (
            context.with_resource("resources.beams.horn", object())
            .with_resource("resources.arrays.gain", object())
            .with_resource("resources.projectors.p", object())
        )
        assert widened.shape_scope.candidates == (
            "resources.beams.horn",
            "resources.projectors.p",
        )
