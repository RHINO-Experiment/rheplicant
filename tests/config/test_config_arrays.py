"""Form 2: the eight array constructors, and the shape symbols they take."""

import jax.numpy as jnp
import pytest

from rheplicant.config import ConfigError
from rheplicant.config.context import ResolutionContext
from rheplicant.config.delivery import ARRAY_FORMS
from rheplicant.config.values import resolve_value


@pytest.fixture
def context():
    return ResolutionContext(
        freq=jnp.linspace(60e6, 85e6, 8), time=jnp.arange(64.0), dtype="float32"
    )


class TestTheConstructors:
    def test_zeros_takes_a_shape_of_symbols(self, context):
        got = resolve_value({"zeros": [3, "n_freq"], "unit": "K"}, context)
        assert got.value.shape == (3, 8)
        assert float(got.value.sum()) == 0.0

    def test_ones(self, context):
        got = resolve_value({"ones": ["n_freq"], "unit": "dimensionless"}, context)
        assert got.value.shape == (8,)
        # The second assert, not the shape, is what tells `ones` from `zeros`:
        # they agree on every shape, and a unit gain silently arriving as zero
        # gain is a forward pass that runs and returns nothing but the noise.
        assert [float(v) for v in got.value] == [1.0] * 8

    def test_full(self, context):
        got = resolve_value(
            {"full": {"shape": ["n_source", "n_freq"], "value": 0.05}, "unit": "dimensionless"},
            context,
        )
        assert got.value.shape == (1, 8)
        assert float(got.value[0, 0]) == pytest.approx(0.05)

    def test_list_takes_a_one_dimensional_literal(self, context):
        got = resolve_value({"list": [300.0, 400.0, 1200.0], "unit": "K"}, context)
        assert got.value.shape == (3,)
        # Shape alone cannot see a constructor that keeps the shape and loses
        # the numbers -- and `list` is the one form whose numbers ARE the
        # document, so nothing else in the run can put them back.
        assert [float(v) for v in got.value] == [300.0, 400.0, 1200.0]

    def test_list_takes_an_n_dimensional_literal(self, context):
        got = resolve_value({"list": [[1.0, 2.0], [3.0, 4.0]], "unit": "dimensionless"}, context)
        assert got.value.shape == (2, 2)
        assert [[float(v) for v in row] for row in got.value] == [[1.0, 2.0], [3.0, 4.0]]

    def test_list_takes_the_runs_dtype_rather_than_the_literals_own(self, context):
        """A YAML `[1, 2, 3]` is a list of Python ints, and jnp.asarray without
        an explicit dtype makes it int32. delivery.py measured what that costs:
        AntennaLossOperator(efficiency=1) stores int32, an integer array is not
        an *inexact* array, so eqx.partition(op, eqx.is_inexact_array) returns
        [] and the field is silently untrainable. A YAML 1 and a YAML 1.0 must
        not differ in what can be inferred, and this is where that is decided."""
        got = resolve_value({"list": [1, 2, 3], "unit": "K"}, context)
        assert got.value.dtype == jnp.float32

    def test_linspace(self, context):
        got = resolve_value(
            {"linspace": {"start": 60.0, "stop": 85.0, "num": 8, "endpoint": True}, "unit": "MHz"},
            context,
        )
        assert got.value.shape == (8,)
        assert float(got.value[0]) == pytest.approx(6e7)
        assert float(got.value[-1]) == pytest.approx(8.5e7)

    def test_arange(self, context):
        got = resolve_value(
            {"arange": {"start": 0.0, "step": 2.0, "num": 64}, "unit": "s"}, context
        )
        assert got.value.shape == (64,)
        assert float(got.value[1]) == pytest.approx(2.0)

    def test_modulo(self, context):
        got = resolve_value({"modulo": {"num": "n_time", "period": 7}, "unit": "count"}, context)
        assert got.value.shape == (64,)
        assert int(got.value[7]) == 0

    def test_from_grid_returns_the_runs_own_axis(self, context):
        got = resolve_value({"from_grid": "freq"}, context)
        assert got.value.shape == (8,)
        assert float(got.value[0]) == pytest.approx(60e6)

    def test_from_grid_names_the_two_axes_on_a_typo(self, context):
        with pytest.raises(ConfigError) as excinfo:
            resolve_value({"from_grid": "frequency"}, context)
        message = str(excinfo.value)
        assert "frequency" in message
        assert "freq" in message
        assert "time" in message


class TestLinspaceEndpoint:
    def test_endpoint_is_required(self, context):
        """schema 2.1.3: 'endpoint: remains REQUIRED -- the sidereal-turn
        contract is decided by exactly that key'. And measured elsewhere:
        linspace(60, 85, 8) with the endpoint included has a channel spacing of
        25/7 MHz, while a config declaring 25/8 MHz passes a band/n_freq
        cross-check and fails the real one."""
        with pytest.raises(ConfigError) as excinfo:
            resolve_value(
                {"linspace": {"start": 60.0, "stop": 85.0, "num": 8}, "unit": "MHz"}, context
            )
        message = str(excinfo.value)
        assert "endpoint" in message
        assert "sidereal" in message  # why it cannot be defaulted

    def test_including_and_excluding_the_endpoint_differ(self, context):
        included = resolve_value(
            {"linspace": {"start": 60.0, "stop": 85.0, "num": 8, "endpoint": True}, "unit": "MHz"},
            context,
        ).value
        excluded = resolve_value(
            {"linspace": {"start": 60.0, "stop": 85.0, "num": 8, "endpoint": False}, "unit": "MHz"},
            context,
        ).value
        assert float(included[-1]) != float(excluded[-1])


class TestTheUnitAppliesToTheWholeArray:
    def test_a_linspace_in_mhz_converts_every_element(self, context):
        got = resolve_value(
            {"linspace": {"start": 60.0, "stop": 85.0, "num": 3, "endpoint": True}, "unit": "MHz"},
            context,
        )
        assert [float(v) for v in got.value] == pytest.approx([6.0e7, 7.25e7, 8.5e7])

    def test_the_result_carries_the_canonical_unit_not_the_declared_one(self, context):
        """`ResolvedValue.unit` is what `check_field_name_unit` reads to
        cross-check a `_deg` or `_m` field name against the value, and what
        `config.resolved.yaml` records. Returning None where the parsed Unit
        belongs leaves every numeric assertion in this file green -- the
        conversion still happened -- and switches that cross-check off for
        every array in the document. Canonical, not declared: the numbers have
        already been multiplied by 1e6 by the time this is read, so reporting
        'MHz' here would describe the input rather than the value."""
        got = resolve_value(
            {"linspace": {"start": 60.0, "stop": 85.0, "num": 3, "endpoint": True}, "unit": "MHz"},
            context,
        )
        assert got.unit.canonical == "Hz"
        assert got.unit.dimension == "frequency"


class TestAHardCodedExtentIsReported:
    def test_a_literal_matching_n_freq_is_reported_not_refused(self, context):
        got = resolve_value({"zeros": [8], "unit": "K"}, context)
        assert got.value.shape == (8,)
        assert got.modifiers.get("_shadowed") == {0: "n_freq"}


class TestTheArithmeticIsWrittenDown:
    def test_arange_starts_where_it_says(self, context):
        """Catches two arithmetic slips at once: `step * arange(num)`, which
        drops `start`, and `(start + arange(num)) * step`, which adds it
        before the multiply instead of after. TestTheConstructors.test_arange
        starts at 0.0, where all three expressions agree elementwise -- both
        mutants pass the whole of the rest of this file (measured)."""
        got = resolve_value(
            {"arange": {"start": 100.0, "step": 2.0, "num": 4}, "unit": "s"}, context
        )
        assert [float(v) for v in got.value] == pytest.approx([100.0, 102.0, 104.0, 106.0])


class TestACountPositionTakesAShapeSymbol:
    """`num` and `period` are shape positions, so ``resolve_extent`` reads
    them and a grid length is written once. Each test below catches that one
    position being read with ``int()`` instead -- which passes every
    literal-count test in this file and refuses the symbol the table exists
    for, so the document goes back to hand-copying the extent.
    """

    def test_linspace_takes_a_symbol_for_num(self, context):
        got = resolve_value(
            {
                "linspace": {"start": 60.0, "stop": 85.0, "num": "n_freq", "endpoint": True},
                "unit": "MHz",
            },
            context,
        )
        assert got.value.shape == (8,)

    def test_arange_takes_a_symbol_for_num(self, context):
        got = resolve_value(
            {"arange": {"start": 0.0, "step": 1.0, "num": "n_time"}, "unit": "s"}, context
        )
        assert got.value.shape == (64,)

    def test_modulo_takes_a_symbol_for_period(self, context):
        got = resolve_value(
            {"modulo": {"num": "n_time", "period": "n_freq"}, "unit": "count"}, context
        )
        assert [int(v) for v in got.value[:10]] == [0, 1, 2, 3, 4, 5, 6, 7, 0, 1]


class TestEachFormsKeysAreExactlyItsOwn:
    """``_require_keys``. Deleting it outright leaves every constructor test
    in this file green -- measured -- because they all write complete, correct
    specs. These are the tests that hold it up.
    """

    def test_full_refuses_a_spec_missing_its_value(self, context):
        """Without the guard this is a bare KeyError out of ``spec['value']``:
        not a ConfigError, so nothing above can attach the document location,
        and the reader is shown a Python traceback instead of the key."""
        with pytest.raises(ConfigError) as excinfo:
            resolve_value({"full": {"shape": ["n_freq"]}, "unit": "K"}, context)
        assert "value" in str(excinfo.value)

    def test_full_refuses_a_spec_missing_its_shape(self, context):
        """The other half of the same guard, and a different failure without
        it: ``spec['shape']`` raises KeyError before ``_shape`` is ever
        reached, so the shape-position message never gets a chance to run."""
        with pytest.raises(ConfigError) as excinfo:
            resolve_value({"full": {"value": 0.05}, "unit": "K"}, context)
        assert "shape" in str(excinfo.value)

    def test_a_mistyped_key_inside_a_form_is_refused_not_ignored(self, context):
        """The ``extra`` half. This is the silent kind: with the check gone,
        {arange: {..., stpe: 10}} builds the array from the three keys it did
        recognise and returns it, so the run differs from the document by
        exactly the key that was mistyped -- finite, correctly shaped, and
        nothing downstream can tell. Same argument the dispatcher makes for
        an unknown key beside a form."""
        with pytest.raises(ConfigError) as excinfo:
            resolve_value(
                {"arange": {"start": 0.0, "step": 2.0, "num": 8, "stpe": 10}, "unit": "s"}, context
            )
        assert "stpe" in str(excinfo.value)

    def test_a_scalar_where_a_shape_belongs_is_refused(self, context):
        """{zeros: 8} rather than {zeros: [8]}. The tempting repair is to wrap
        a bare integer for the writer, which silently turns a scalar into an
        8-vector -- correctly shaped for nothing in particular, and broadcast
        against everything downstream without complaint."""
        with pytest.raises(ConfigError) as excinfo:
            resolve_value({"zeros": 8, "unit": "K"}, context)
        assert "shape" in str(excinfo.value)

    def test_the_scalar_advice_is_this_modules_and_not_the_shared_helpers(self, context):
        """The shape check and the check A41 report now live in
        symbols.resolve_shape, shared with the draw forms. Everything about
        that pass is common to both callers except this clause: a scalar zeros
        is {value: 0.0} and a scalar normal is an empty shape. Pinned so a
        later collapse into one shared string fails here instead of telling a
        zeros writer to write a draw."""
        with pytest.raises(ConfigError) as excinfo:
            resolve_value({"zeros": 8, "unit": "K"}, context)
        message = str(excinfo.value)
        assert "{value: 0.0}" in message
        assert "shape: []" not in message

    def test_modulo_refuses_a_period_of_zero(self, context):
        """Measured: jnp.arange(8) % 0 does not raise, it returns
        [0 0 0 0 0 0 0 0]. So without this guard a period that came out of a
        symbol as 0 gives a switching pattern that is silently constant --
        every sample assigned to source 0 -- rather than an error."""
        with pytest.raises(ConfigError) as excinfo:
            resolve_value({"modulo": {"num": "n_time", "period": 0}, "unit": "count"}, context)
        assert "period" in str(excinfo.value)


class TestWhatTheNodeReportsAboutItself:
    def test_every_constructor_reports_its_own_name_as_the_source(self, context):
        """``delivery.deliver`` gates check A40 -- an array form landing on a
        static field, which measurably fails three different ways -- on
        ``source in ARRAY_FORMS`` and on nothing else. A constructor that
        reports the wrong name, or 'scalar', therefore disables that check for
        itself while every number it produces stays correct. Nothing else in
        this file reads ``.source``, so without this the name is free."""
        nodes = {
            "zeros": {"zeros": ["n_freq"]},
            "ones": {"ones": ["n_freq"]},
            "full": {"full": {"shape": ["n_freq"], "value": 1.0}},
            "list": {"list": [1.0, 2.0]},
            "linspace": {"linspace": {"start": 0.0, "stop": 1.0, "num": 4, "endpoint": True}},
            "arange": {"arange": {"start": 0.0, "step": 1.0, "num": 4}},
            "modulo": {"modulo": {"num": "n_time", "period": 7}},
            "from_grid": {"from_grid": "freq"},
        }
        for form, node in nodes.items():
            assert resolve_value(node, context).source == form
            assert form in ARRAY_FORMS

    def test_the_declared_modifiers_survive_resolution(self, context):
        """``_finish`` rebuilds the modifier dict so it can add ``_shadowed``.
        Building it from {} rather than {**modifiers} drops every key the
        document actually wrote -- including ``axis:``, which is recorded
        precisely so a later stage can read it, and ``as:``, which delivery
        cross-checks against the destination field. Both would go silently."""
        got = resolve_value({"zeros": ["n_freq"], "unit": "K", "axis": "freq"}, context)
        assert got.modifiers["unit"] == "K"
        assert got.modifiers["axis"] == "freq"

    def test_the_modifiers_survive_a_node_that_declares_no_unit(self, context):
        """The other branch of ``_finish``, and it has to be reached on its
        own: a form already in canonical units -- from_grid reads the run's
        own Hz grid -- returns before the conversion, and that early return
        rebuilds the modifier dict too. Rebuilding only the converting branch
        correctly passes every ``unit:``-carrying test in this file
        (measured), and drops ``axis:`` for exactly the forms that never
        declare a unit."""
        got = resolve_value({"from_grid": "freq", "axis": "freq"}, context)
        assert got.unit is None
        assert got.modifiers["axis"] == "freq"
