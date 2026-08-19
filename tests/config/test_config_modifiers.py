"""The eight modifiers: what each one declares, and what it refuses."""

import math

import jax.numpy as jnp
import pytest

from rheplicant.config import ConfigError
from rheplicant.config.context import ResolutionContext
from rheplicant.config.modifiers import DTYPES, NOISE_AXES, NORMALIZATIONS, PARTS
from rheplicant.config.values import resolve_value


@pytest.fixture
def context():
    return ResolutionContext(
        freq=jnp.linspace(60e6, 85e6, 8), time=jnp.arange(64.0), dtype="float32"
    )


class TestTheTablesAreClosed:
    """Every table below drives both the validation and the parametrisation
    that exercises it, so a table walked by a ``for`` loop can be emptied or
    extended without any test noticing. These pin the contents literally.
    """

    def test_the_four_tables_are_what_the_schema_says(self):
        assert set(PARTS) == {"re", "im", "abs", "angle"}
        assert set(NORMALIZATIONS) == {"none", "mean1", "pixel_sum", "max1"}
        assert set(DTYPES) == {"float32", "float64", "complex64", "complex128"}
        assert set(NOISE_AXES) == {"time", "freq", "none"}


class TestScaleAndOffset:
    def test_an_affine_relabel_survives_as_the_physical_statement(self, context):
        """schema 2.1.10: 250 + 20*linspace(-1,1,8) stays readable as a
        linspace with a scale and an offset. 30*cos(linspace(0,3,8)) does not,
        and goes to the python: hatch."""
        got = resolve_value(
            {
                "linspace": {"start": -1.0, "stop": 1.0, "num": 3, "endpoint": True},
                "scale": 20.0,
                "offset": 250.0,
                "unit": "K",
            },
            context,
        )
        assert [float(v) for v in got.value] == pytest.approx([230.0, 250.0, 270.0])

    def test_scale_alone_and_offset_alone_both_work(self, context):
        """Also the defaults, and they are not interchangeable: swap them --
        scale defaulting to 0.0, offset to 1.0 -- and the first line gives 7
        and the second 3, both finite and both plausible."""
        assert float(resolve_value({"value": 2.0, "scale": 3.0}, context).value) == pytest.approx(
            6.0
        )
        assert float(resolve_value({"value": 2.0, "offset": 3.0}, context).value) == pytest.approx(
            5.0
        )

    def test_the_order_is_scale_then_offset_and_the_message_says_so(self, context):
        """scale*v + offset, not scale*(v + offset). One level, no nesting, so
        there is no precedence to reason about -- but there is an order, and a
        reader who assumes the other one is off by scale*offset."""
        got = resolve_value({"value": 2.0, "scale": 10.0, "offset": 1.0}, context)
        assert float(got.value) == pytest.approx(21.0)

    def test_an_integer_scale_on_an_integer_form_still_leaves_it_floating(self, context):
        """The ``float()`` around both keys, and it takes both written as
        integers to see it: `modulo` produces int32, and with either coercion
        left in place the other operand promotes the result anyway. Drop the
        pair and `{scale: 2, offset: 1}` over a switching pattern stays int32
        -- which delivery.py measured the cost of: an integer array is not an
        *inexact* array, so eqx.partition(op, eqx.is_inexact_array) returns []
        and the field is silently untrainable. Every other value in this file
        is already floating before a scale reaches it, so nothing else here
        can see the coercions at all (measured)."""
        got = resolve_value(
            {
                "modulo": {"num": "n_time", "period": 7},
                "unit": "count",
                "scale": 2,
                "offset": 1,
            },
            context,
        )
        assert got.value.dtype == jnp.float32
        assert [float(v) for v in got.value[:4]] == pytest.approx([1.0, 3.0, 5.0, 7.0])

    def test_the_unit_has_already_been_applied_when_the_offset_lands(self, context):
        """`unit:` is not in apply_modifiers' order at all -- the form applies
        it -- so scale: and offset: act on the CANONICAL number. The scale
        alone cannot show this (2*60 MHz and 2*6e7 Hz are the same quantity);
        the offset can, because an offset written against the declared unit
        would be multiplied by 1e6 as well and give 1.21e8 rather than
        1.2e8 + 1."""
        got = resolve_value({"value": 60.0, "unit": "MHz", "scale": 2.0, "offset": 1.0}, context)
        assert float(got.value) == pytest.approx(1.2e8 + 1.0)


class TestPart:
    def test_angle_emits_canonical_degree_metadata(self, context):
        got = resolve_value(
            {"value": 1.0 + 1.0j, "unit": "dimensionless", "part": "angle"},
            context,
        )
        assert got.modifiers["unit"] == "deg"

    def test_each_part_of_a_complex_value(self, context):
        """MEASURED, and a departure from the plan's node: {list: [1+2j]} does
        not reach this layer at all -- arrays._list casts the literal to the
        run's dtype first and jnp.asarray([1+2j], dtype='float32') raises
        TypeError. A complex value gets here from ref:, python: or file: (the
        touchstone reader's S-parameters are the reason part: exists), and a
        scalar node carrying one stands in for those.
        """
        node = {"value": 1.0 + 2.0j}
        assert float(resolve_value({**node, "part": "re"}, context).value) == pytest.approx(1.0)
        assert float(resolve_value({**node, "part": "im"}, context).value) == pytest.approx(2.0)
        assert float(resolve_value({**node, "part": "abs"}, context).value) == pytest.approx(
            5.0**0.5
        )
        assert float(resolve_value({**node, "part": "angle"}, context).value) == pytest.approx(
            math.atan2(2.0, 1.0)
        )

    def test_the_part_is_taken_before_the_scale_and_the_offset(self, context):
        """part -> scale/offset, not the reverse. On the real part the two
        orders agree -- scale and offset are real, so Re(s*v + o) = s*Re(v) + o
        -- and that is exactly why the imaginary part is the one to ask:
        10*Im(1+2j) + 1 = 21, while Im(10*(1+2j) + 1) = 20. A reflection
        coefficient read off a file and rescaled is this node."""
        got = resolve_value(
            {"value": 1.0 + 2.0j, "part": "im", "scale": 10.0, "offset": 1.0}, context
        )
        assert float(got.value) == pytest.approx(21.0)

    def test_an_unknown_part_is_refused_and_the_set_is_listed(self, context):
        with pytest.raises(ConfigError) as excinfo:
            resolve_value({"value": 1.0, "part": "real"}, context)
        message = str(excinfo.value)
        assert "real" in message
        for part in PARTS:
            assert part in message, part


class TestPartImOnAValueThatIsNotComplex:
    """`jnp.imag` of a real array is exactly 0 everywhere -- right, and never
    what a document meant. It is the only one of the four whose answer cannot
    depend on the value, so it is the only one refused here.
    """

    def test_it_is_refused_rather_than_returning_a_column_of_zeros(self, context):
        with pytest.raises(ConfigError) as excinfo:
            resolve_value({"list": [1.0, 2.0], "unit": "dimensionless", "part": "im"}, context)
        assert "float32" in str(excinfo.value)  # the dtype it actually found

    def test_the_message_says_why_nothing_downstream_could_have_caught_it(self, context):
        """A zero `gamma_src_im` is a well-formed noise-wave model that has no
        sine component, and no check below here separates "no sine component"
        from "the sine component was deleted when the config was read"."""
        with pytest.raises(ConfigError) as excinfo:
            resolve_value({"value": 1.0, "part": "im"}, context)
        message = str(excinfo.value)
        assert "gamma_src_im" in message  # the field it would silently empty
        assert "extends:" in message  # cause two: the wrong resource
        assert "{value: 0.0}" in message  # how to SAY the imaginary part is zero

    @pytest.mark.parametrize(
        ("part", "expected"),
        [("re", [-3.0, 4.0]), ("abs", [3.0, 4.0]), ("angle", [math.pi, 0.0])],
    )
    def test_the_other_three_parts_stay_legal_on_a_real_value(self, part, expected, context):
        """The point of the refusal is that it is narrow. `re` is a no-op an
        extends: merge can legitimately produce, `abs` is the absolute value
        and `angle` is 0 or pi -- all three answer something the value decides.
        Without this, widening the guard to "part: on a real value" later would
        pass every test that only checks the refusal."""
        got = resolve_value({"list": [-3.0, 4.0], "unit": "dimensionless", "part": part}, context)
        assert [float(v) for v in got.value] == pytest.approx(expected)

    def test_im_on_a_genuinely_complex_value_is_untouched(self, context):
        """The complement of the refusal: the case it exists to protect."""
        got = resolve_value({"value": 1.0 + 2.0j, "part": "im"}, context)
        assert float(got.value) == pytest.approx(2.0)

    def test_an_integer_value_is_refused_too_and_not_only_a_float_one(self, context):
        """`modulo` is integer, and `jnp.issubdtype(int32, complexfloating)` is
        False just as it is for float32 -- but a guard written as
        `dtype == float32` or `not jnp.isrealobj(...)` would treat the two
        differently for no reason the document could see."""
        with pytest.raises(ConfigError) as excinfo:
            resolve_value(
                {"modulo": {"num": "n_time", "period": 7}, "unit": "count", "part": "im"}, context
            )
        assert "int32" in str(excinfo.value)


class TestPartImAfterAWideningDtype:
    """`dtype: complex64` over a real value makes the TYPE complex and leaves
    every imaginary part zero. Judging `part: im` on the post-cast dtype would
    let exactly that document through -- guaranteed zeros, no symptom -- so the
    question is asked of what the value was on entry, before any declared cast.
    """

    def test_a_widening_dtype_does_not_make_part_im_meaningful(self, context):
        with pytest.raises(ConfigError) as excinfo:
            resolve_value(
                {
                    "list": [1.0, 2.0],
                    "unit": "dimensionless",
                    "dtype": "complex64",
                    "part": "im",
                },
                context,
            )
        assert "complex64" in str(excinfo.value)  # the dtype the cast produced

    def test_the_message_says_the_zero_is_by_construction(self, context):
        """The clause this path needs and the plain one does not: the value is
        complex, so 'not complex' would read as simply wrong. What makes it a
        refusal is that the zero was put there by the cast rather than measured
        -- and the resistive-termination remedy answers this path unchanged."""
        with pytest.raises(ConfigError) as excinfo:
            resolve_value({"value": 1.0, "dtype": "complex64", "part": "im"}, context)
        message = str(excinfo.value)
        assert "by construction" in message
        assert "by measurement" in message
        assert "{value: 0.0}" in message  # the remedy still reaches this path

    @pytest.mark.parametrize(
        ("part", "expected"),
        [("re", [-3.0, 4.0]), ("abs", [3.0, 4.0]), ("angle", [math.pi, 0.0])],
    )
    def test_the_other_three_parts_stay_legal_through_a_widening_dtype(
        self, part, expected, context
    ):
        """The narrowness complement for this path. Threading "was it complex
        on entry" into `_part` puts a second condition within reach of a guard
        that creeps, and `re`/`abs`/`angle` are as legal after a widening cast
        as before one."""
        got = resolve_value(
            {"list": [-3.0, 4.0], "unit": "dimensionless", "dtype": "complex64", "part": part},
            context,
        )
        assert [float(v) for v in got.value] == pytest.approx(expected)

    def test_im_survives_a_cast_that_did_not_widen_anything(self, context):
        """The case the guard must not catch: a genuinely complex value that
        also declares a complex dtype. Reading the flag off the post-cast value
        would pass this; reading it off the pre-cast value must too."""
        got = resolve_value({"value": 1.0 + 2.0j, "dtype": "complex64", "part": "im"}, context)
        assert float(got.value) == pytest.approx(2.0)


class TestColumn:
    def test_it_turns_n_into_n_by_one(self, context):
        """CalLoadOperator.t_load reads (n, 1) as per-sample and (n,) as
        per-frequency (radio/instrument/calibration.py:716-722). On a square
        grid the two are indistinguishable by shape, which is why the key
        exists rather than being inferred. n_freq is 8 here and not 1, so
        array[None, :] -- the other one-line way to add an axis -- is a
        different shape and this fails."""
        got = resolve_value({"zeros": ["n_freq"], "unit": "K", "column": True}, context)
        assert got.value.shape == (8, 1)

    def test_it_is_refused_on_anything_that_is_not_one_dimensional(self, context):
        with pytest.raises(ConfigError) as excinfo:
            resolve_value({"zeros": [2, 3], "unit": "K", "column": True}, context)
        assert "(2, 3)" in str(excinfo.value)

    def test_a_false_column_is_not_the_same_as_an_absent_one(self, context):
        """`column: false` is a written statement that the value is
        per-frequency. It must leave the shape alone -- a guard keyed on
        `"column" in modifiers` rather than on the value would promote it."""
        got = resolve_value({"zeros": ["n_freq"], "unit": "K", "column": False}, context)
        assert got.value.shape == (8,)


class TestNormalize:
    @pytest.mark.parametrize("kind", ["mean1", "pixel_sum", "max1"])
    def test_normalization_emits_dimensionless_metadata(self, kind, context):
        got = resolve_value(
            {"list": [1.0, 2.0], "unit": "K", "normalize": kind}, context
        )
        assert got.modifiers["unit"] == "dimensionless"

    @pytest.mark.parametrize(
        ("kind", "expected"),
        [
            ("none", [1.0, 2.0, 3.0, 4.0]),
            ("mean1", [0.4, 0.8, 1.2, 1.6]),
            ("pixel_sum", [0.1, 0.2, 0.3, 0.4]),
            ("max1", [0.25, 0.5, 0.75, 1.0]),
        ],
    )
    def test_every_declared_convention_is_applied(self, kind, expected, context):
        """The plan asserted only ``.shape`` here, which all four share: with
        that assertion alone `mean1` can be implemented as `pixel_sum`, `max1`
        as `mean1`, and `none` as any of them, and every case still passes.
        The four divisors of [1,2,3,4] are 1, 2.5, 10 and 4, so the values are
        what tell the conventions apart -- which is the whole point of the key,
        since the numbers cannot say which convention produced them."""
        got = resolve_value(
            {"list": [1.0, 2.0, 3.0, 4.0], "unit": "dimensionless", "normalize": kind}, context
        )
        assert got.value.shape == (4,)
        assert [float(v) for v in got.value] == pytest.approx(expected)

    def test_pixel_sum_makes_the_sum_one(self, context):
        got = resolve_value(
            {"list": [1.0, 3.0], "unit": "dimensionless", "normalize": "pixel_sum"}, context
        )
        assert float(got.value.sum()) == pytest.approx(1.0)

    def test_the_reduction_is_over_the_whole_array_and_not_the_last_axis(self, context):
        """``_normalize`` calls jnp.sum/mean/max with no ``axis=``, so a 2-D
        value is normalised globally rather than row by row. Every other case
        in this file is 1-D, where the two agree exactly -- swap in
        jnp.sum(array, axis=-1, keepdims=True) and the whole file still passes
        (measured). The 2-D case is the one the key exists for: an
        (n_freq, n_pix) beam declared pixel_sum sums to 1 over the whole cube,
        not to 1 per channel, and the leading element is 0.125 either way you
        read the document but 0.5 under the other reduction."""
        got = resolve_value(
            {"list": [[1.0, 1.0], [1.0, 5.0]], "unit": "dimensionless", "normalize": "pixel_sum"},
            context,
        )
        assert float(got.value.sum()) == pytest.approx(1.0)
        assert [float(v) for v in got.value.ravel()] == pytest.approx([0.125, 0.125, 0.125, 0.625])

    def test_it_is_applied_after_the_scale_and_the_offset(self, context):
        """normalize -> after scale/offset. Nothing in the un-offset case can
        see the order, because normalising is idempotent up to a scale: with
        `scale:` alone both orders give the same array. An `offset:` breaks
        that, and breaks it in the direction that matters -- normalise first
        and the offset is added afterwards, so the beam a document declares
        unit-sum has a sum of 5."""
        got = resolve_value(
            {
                "list": [1.0, 3.0],
                "unit": "dimensionless",
                "offset": 2.0,
                "normalize": "pixel_sum",
            },
            context,
        )
        assert float(got.value.sum()) == pytest.approx(1.0)
        assert [float(v) for v in got.value] == pytest.approx([3.0 / 8.0, 5.0 / 8.0])

    def test_an_unknown_convention_is_refused_with_the_measurement(self, context):
        """The output's unit is decided by the PAIR (beam normalisation,
        normalize_beam): 32838 K against 200 K on a uniform 200 K sky."""
        with pytest.raises(ConfigError) as excinfo:
            resolve_value({"value": 1.0, "normalize": "unit"}, context)
        message = str(excinfo.value)
        assert "unit" in message
        assert "32838" in message  # the measurement that makes this a declaration


class TestNormalizeRefusesADivisorThatCannotDivide:
    """Three of the four conventions divide, and each can be handed a divisor
    of zero or of nan. Measured before this guard existed:
    ``{list: [1.0, -1.0], normalize: pixel_sum}`` returned ``[inf, -inf]`` and
    ``{zeros: [n_freq], normalize: mean1}`` returned ``[nan, nan, nan]`` --
    two documents that read as perfectly ordinary, neither refused.
    """

    @pytest.mark.parametrize(
        ("kind", "node"),
        [
            ("mean1", {"list": [1.0, -1.0]}),
            ("pixel_sum", {"list": [1.0, -1.0]}),
            ("max1", {"zeros": ["n_freq"]}),
        ],
    )
    def test_a_zero_divisor_is_refused_by_every_convention_that_divides(self, kind, node, context):
        with pytest.raises(ConfigError) as excinfo:
            resolve_value({**node, "unit": "dimensionless", "normalize": kind}, context)
        assert kind in str(excinfo.value)

    @pytest.mark.parametrize("kind", ["mean1", "pixel_sum", "max1"])
    def test_a_non_finite_divisor_is_refused_too(self, kind, context):
        """Not only zero -- an inf anywhere in the value makes all three
        reductions inf, and dividing by inf gives a quiet array of zeros
        rather than anything that looks wrong. A guard written as `== 0`
        alone passes this straight through."""
        with pytest.raises(ConfigError) as excinfo:
            resolve_value(
                {"list": [float("inf"), 1.0], "unit": "dimensionless", "normalize": kind}, context
            )
        assert kind in str(excinfo.value)

    @pytest.mark.parametrize("kind", ["mean1", "pixel_sum"])
    def test_a_nan_divisor_is_refused_where_a_comparison_guard_would_not(self, kind, context):
        """The case the explicit ``isfinite`` branch exists for: inf + -inf is
        nan, and nan answers False to `> 0`, to `< 0` and to `== 0` alike, so
        every ordering-based way of writing this guard admits precisely the
        worst divisor of the three."""
        with pytest.raises(ConfigError) as excinfo:
            resolve_value(
                {
                    "list": [float("inf"), float("-inf")],
                    "unit": "dimensionless",
                    "normalize": kind,
                },
                context,
            )
        assert "nan" in str(excinfo.value)

    def test_the_refusal_quotes_the_divisor_and_names_the_way_out(self, context):
        with pytest.raises(ConfigError) as excinfo:
            resolve_value({"zeros": ["n_freq"], "unit": "K", "normalize": "mean1"}, context)
        message = str(excinfo.value)
        assert "mean1" in message
        assert "0.0" in message  # the divisor it actually computed, quoted
        assert "normalize: none" in message  # what to write if that is the intent
        for kind in NORMALIZATIONS:
            assert kind in message, kind

    def test_the_same_value_under_normalize_none_is_still_legal(self, context):
        """The guard belongs to the division, not to the key. A zero-sum value
        that declares `normalize: none` divides by nothing and says so, which
        is exactly the remedy the refusal offers -- a guard hung on the
        presence of `normalize:` would refuse its own advice."""
        got = resolve_value({"zeros": ["n_freq"], "unit": "K", "normalize": "none"}, context)
        assert [float(v) for v in got.value] == pytest.approx([0.0] * 8)


class TestARealDtypeOverAComplexValue:
    """Widening is lossless and stays legal; narrowing deletes the phase.
    Measured before this guard: ``{value: 1+2j, dtype: float32}`` returned
    ``1.0`` -- no exception, no warning, no record that anything was lost.
    """

    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    def test_it_is_refused_rather_than_dropping_the_imaginary_part(self, dtype, context):
        with pytest.raises(ConfigError) as excinfo:
            resolve_value({"value": 1.0 + 2.0j, "dtype": dtype}, context)
        assert dtype in str(excinfo.value)

    def test_the_refusal_names_part_as_the_deliberate_route(self, context):
        """`part: re` states "the real part is what I want" where a reader can
        see it. That is the distinction this schema draws everywhere else --
        between a computation the numbers imply and a declaration only the
        document can make -- and the refusal has to hand the reader the second
        one rather than merely blocking the first."""
        with pytest.raises(ConfigError) as excinfo:
            resolve_value({"value": 1.0 + 2.0j, "dtype": "float32"}, context)
        message = str(excinfo.value)
        for part in ("re", "abs", "angle"):
            assert f"part: {part}" in message, part

    def test_writing_part_alongside_the_dtype_does_not_get_past_it(self, context):
        """The order is dtype -> part, so `{dtype: float32, part: re}` really
        does delete the phase before `part` ever sees the value. The remedy is
        to REPLACE the dtype: key, not to add to it, and the refusal has to
        hold here or it would be advice the grammar itself contradicts."""
        with pytest.raises(ConfigError):
            resolve_value({"value": 1.0 + 2.0j, "dtype": "float32", "part": "re"}, context)

    def test_widening_a_real_value_to_complex_is_not_refused(self, context):
        """The other direction loses nothing and has a real use: a field that
        must be complex, given a real starting value. A guard written on "the
        dtypes disagree" rather than on the direction would refuse it."""
        got = resolve_value(
            {"list": [1.0, 2.0], "unit": "dimensionless", "dtype": "complex64"}, context
        )
        assert jnp.issubdtype(got.value.dtype, jnp.complexfloating)
        assert [complex(v) for v in got.value] == [1.0 + 0j, 2.0 + 0j]

    def test_a_complex_dtype_over_a_complex_value_keeps_the_phase(self, context):
        got = resolve_value({"value": 1.0 + 2.0j, "dtype": "complex64"}, context)
        assert complex(got.value) == pytest.approx(1.0 + 2.0j)


class TestDtype:
    def test_it_overrides_the_run_dtype(self, context):
        got = resolve_value({"zeros": ["n_freq"], "unit": "K", "dtype": "complex64"}, context)
        assert jnp.issubdtype(got.value.dtype, jnp.complexfloating)

    def test_it_is_applied_before_the_part(self, context):
        """dtype -> part, not the reverse. Both orders give the same numbers
        here -- |3| and |4| either way -- and differ in the result's DTYPE:
        take the part last and the array is left complex64 with a zero
        imaginary component, which delivery._as_traced then preserves as
        complex, so a field expecting a real magnitude receives a complex
        array that is finite, correctly shaped and equal elementwise."""
        got = resolve_value(
            {"list": [3.0, 4.0], "unit": "dimensionless", "dtype": "complex64", "part": "abs"},
            context,
        )
        assert not jnp.issubdtype(got.value.dtype, jnp.complexfloating)
        assert [float(v) for v in got.value] == pytest.approx([3.0, 4.0])

    def test_an_unknown_dtype_is_refused_and_the_four_are_listed(self, context):
        with pytest.raises(ConfigError) as excinfo:
            resolve_value({"zeros": [2], "dtype": "float16"}, context)
        message = str(excinfo.value)
        for name in ("float32", "float64", "complex64", "complex128"):
            assert name in message


class TestAxisIsRecordedNotApplied:
    def test_a_declared_axis_is_carried_through(self, context):
        got = resolve_value({"zeros": ["n_freq"], "unit": "K", "axis": "freq"}, context)
        assert got.modifiers["axis"] == "freq"

    def test_a_declared_axis_changes_nothing_about_the_value(self, context):
        """`axis:` is read by inference/noise.py and applied by nobody here.
        The shape is the assertion: an implementation that took the key as a
        transpose, a reshape or a sum would still record it."""
        got = resolve_value({"ones": ["n_time"], "unit": "K", "axis": "time"}, context)
        assert got.value.shape == (64,)
        assert [float(v) for v in got.value[:3]] == pytest.approx([1.0, 1.0, 1.0])

    def test_an_unknown_axis_is_refused(self, context):
        with pytest.raises(ConfigError) as excinfo:
            resolve_value({"zeros": [8], "axis": "frequency"}, context)
        message = str(excinfo.value)
        for axis in NOISE_AXES:
            assert axis in message, axis

    def test_the_measurement_is_quoted(self, context):
        """check_noise_std_axis exists because a per-time sigma applied
        per-frequency on a square grid gives error bars of 0.00004..0.00354
        against a flat 0.00010 -- every shape correct."""
        with pytest.raises(ConfigError) as excinfo:
            resolve_value({"zeros": [8], "axis": "frequency"}, context)
        assert "0.00010" in str(excinfo.value)


class TestEveryFormGoesThroughTheModifiers:
    """``resolve_value`` has two places a resolved value comes from -- the
    scalar branch and the registered resolver -- and a wiring that reaches only
    one leaves every modifier silently unapplied for exactly the forms on the
    other side, with the value still finite and correctly shaped.
    """

    def test_the_scalar_branch(self, context):
        assert float(resolve_value({"value": 2.0, "scale": 3.0}, context).value) == pytest.approx(
            6.0
        )

    def test_the_registered_resolver_branch(self, context):
        got = resolve_value({"zeros": ["n_freq"], "unit": "K", "offset": 5.0}, context)
        assert [float(v) for v in got.value] == pytest.approx([5.0] * 8)

    def test_the_shadowing_report_is_not_mistaken_for_a_modifier(self, context):
        """``arrays._finish`` writes ``_shadowed`` into the same dict the
        modifiers live in -- it is check A41's report, not a key the document
        wrote. ``apply_modifiers`` reads that dict, so a version that validated
        every key it found against the modifier table would refuse a perfectly
        legal {zeros: [8]} for the sole reason that the literal 8 happens to
        equal n_freq."""
        got = resolve_value({"zeros": [8], "unit": "K", "offset": 1.0}, context)
        assert got.modifiers["_shadowed"] == {0: "n_freq"}
        assert [float(v) for v in got.value] == pytest.approx([1.0] * 8)

    def test_unit_and_as_are_left_alone_rather_than_refused(self, context):
        """Two of the eight are applied elsewhere: ``unit:`` by the form and
        ``as:`` by delivery. They arrive in the same dict, so this pins that
        apply_modifiers passes over them instead of treating them as
        unrecognised."""
        got = resolve_value({"value": 290.0, "unit": "K", "as": "traced"}, context)
        assert float(got.value) == pytest.approx(290.0)
        assert got.modifiers["as"] == "traced"
