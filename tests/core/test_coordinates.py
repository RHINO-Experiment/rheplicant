"""Tests for the Coordinates and Environment containers."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rheplicant.core.coordinates import (
    MAX_TIME_RESOLUTION_IN_SAMPLES,
    Coordinates,
)
from rheplicant.core.environment import Environment
from rheplicant.core.errors import StateValidationError


class TestCoordinatesConstruction:
    def test_all_optional(self):
        c = Coordinates()
        assert c.time is None and c.freq is None and c.pointing is None
        assert c.extra == {}

    def test_basic(self):
        c = Coordinates(time=jnp.arange(4.0), freq=jnp.arange(3.0))
        assert c.time.shape == (4,)
        assert c.freq.shape == (3,)

    def test_converter_accepts_numpy_and_lists(self):
        c = Coordinates(time=np.arange(4.0), freq=[1.0, 2.0])
        assert isinstance(c.time, jax.Array)
        assert isinstance(c.freq, jax.Array)

    def test_pointing_shape(self):
        c = Coordinates(pointing=jnp.zeros((5, 2)))
        assert c.pointing.shape == (5, 2)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"time": jnp.zeros((2, 2))},  # time must be 1D
            {"freq": jnp.zeros((2, 2))},  # freq must be 1D
            {"pointing": jnp.zeros(5)},  # pointing must be 2D
        ],
    )
    def test_structural_validation(self, kwargs):
        with pytest.raises(StateValidationError):
            Coordinates(**kwargs)


class TestCoordinatesFunctional:
    def test_replace_returns_new(self):
        c = Coordinates(time=jnp.arange(4.0))
        c2 = c.replace(freq=jnp.arange(3.0))
        assert c2.freq is not None and c.freq is None
        assert jnp.array_equal(c2.time, c.time)

    def test_replace_reruns_validation(self):
        c = Coordinates()
        with pytest.raises(StateValidationError):
            c.replace(time=jnp.zeros((2, 2)))

    def test_extra_is_traced(self):
        """Arrays in `extra` must appear as pytree leaves (differentiable/vmappable)."""
        c = Coordinates(time=jnp.arange(4.0), extra={"az": jnp.zeros(4)})
        leaves = jax.tree.leaves(c)
        assert len(leaves) == 2  # time + extra["az"]

    def test_is_pytree_roundtrip(self):
        c = Coordinates(time=jnp.arange(4.0), extra={"az": jnp.zeros(4)})
        leaves, treedef = jax.tree.flatten(c)
        c2 = jax.tree.unflatten(treedef, leaves)
        assert jnp.array_equal(c2.time, c.time)


class TestATimeAxisTheStoredDtypeCannotCarry:
    """The one VALUE check in this container, and where its edges are.

    ``time`` is the only coordinate any operator does arithmetic on, and it is
    stored through ``jnp.asarray`` — float32 unless x64 is on. A unix-second
    axis (~1.75e9) therefore lands on a 128 s grid before anything downstream
    runs. Measured on 8 samples 100 s apart, before this check existed::

        stored [1750000000, 1750000128, 1750000256, 1750000256, 1750000384,
                1750000512, 1750000640, 1750000640]     6 distinct, of 8
        BackendOperator(n_chunk=2) chunk times
               [1750000128, 1750000256, 1750000384, 1750000640]
        float64 truth
               [1750000050, 1750000250, 1750000450, 1750000650]
        error  [78, 6, -66, -10] s, against a 100 s cadence

    Every axis below is deliberately IRREGULAR where it can be, so a check that
    silently used a mean, a median or the first gap would give a different
    answer from one that uses the smallest.
    """

    UNIX = 1_750_000_000.0 + 100.0 * np.arange(8)

    def test_a_unix_second_axis_is_refused_and_names_both_remedies(self):
        with pytest.raises(StateValidationError) as excinfo:
            Coordinates(time=self.UNIX)
        message = str(excinfo.value)
        assert "representable" in message
        assert "read_rhino_observation" in message  # where such an axis comes from
        assert "start of the run" in message  # remedy 1
        assert "JAX_ENABLE_X64" in message  # remedy 2
        assert "float32" in message  # the dtype that could not carry it
        assert "128" in message  # its representable spacing, measured

    def test_the_same_run_measured_from_its_own_start_is_stored_exactly(self):
        """The remedy, run. Gaps 100/50/250/300 so that a mean (175) and a
        smallest (50) differ, and every stored value differs from every other."""
        elapsed = np.array([0.0, 100.0, 150.0, 400.0, 700.0])
        coords = Coordinates(time=elapsed)
        np.testing.assert_array_equal(np.asarray(coords.time), elapsed)

    def test_the_threshold_is_pinned_from_both_sides(self):
        """At an anchor of 1e7 s the float32 grid is exactly 1 s, so the ratio
        against the cadence is 1/cadence and the threshold (1e-2) falls between
        two cadences two percent apart. Both axes are stored EXACTLY — only the
        ratio moves across the cut, so nothing else can explain the difference.

        This pair is also what separates ``np.spacing(times.max())`` from
        ``np.spacing(float(times.max()))``: the second promotes to a Python
        float and answers for float64 (1.9e-9 s here), which passes both sides
        and is blind to the very thing the check is for. The unix axis above is
        refused either way, because its samples have already collided; only a
        sub-collision loss like this one tells the two apart.
        """
        with pytest.raises(StateValidationError, match="representable"):
            Coordinates(time=1.0e7 + 99.0 * np.arange(4))

        accepted = 1.0e7 + 101.0 * np.arange(4)
        np.testing.assert_array_equal(np.asarray(Coordinates(time=accepted).time), accepted)

    @pytest.mark.parametrize("cadence", [0.05, 1.0, 60.0])
    @pytest.mark.parametrize(("n_samples", "carried"), [(50_000, True), (300_000, False)])
    def test_a_relative_float32_axis_runs_out_at_order_1e5_samples(
        self, n_samples, carried, cadence
    ):
        """The price of fixing this by making the axis RELATIVE rather than
        float64, pinned from both sides at three cadences four orders apart.

        For a uniform axis measured from its own start the peak is
        ``n * cadence``, so the ratio is ``spacing(n*cadence)/cadence``; and
        because ``spacing(x)`` is within a factor of two of ``x * 2**-23`` the
        cadence very nearly cancels, leaving the sample COUNT as what the cut
        constrains. The limit is of order 1e5, in [8.4e4, 1.7e5], not one exact
        number — where inside that band it falls depends on where ``n*cadence``
        sits in its binade, which is why this brackets it rather than pinning
        it. A four-hour run at 1 s (1.4e4 samples) is an order of magnitude
        clear; the same run at 0.05 s (2.9e5) is refused and needs x64.
        """
        times = cadence * np.arange(n_samples, dtype=np.float64)
        if carried:
            assert Coordinates(time=times).time.shape == (n_samples,)
        else:
            with pytest.raises(StateValidationError, match="representable"):
                Coordinates(time=times)

    def test_the_exact_cut_at_one_second_cadence_is_two_to_the_seventeen(self):
        """One cadence pinned exactly, so the band above is anchored to a
        measured number rather than only to an argument. At 1 s the peak of an
        n-sample axis is n-1, and the float32 binade boundary at 2**17 puts the
        cut between 131072 samples (spacing 0.0078 s, ratio 0.0078) and 131073
        (spacing 0.0156 s, ratio 0.0156)."""
        assert Coordinates(time=np.arange(131072, dtype=np.float64)).time.shape == (131072,)
        with pytest.raises(StateValidationError, match="representable"):
            Coordinates(time=np.arange(131073, dtype=np.float64))

    def test_the_ratio_itself_is_the_documented_constant(self):
        """Not a free-floating 1e-2: the cut is stated once and imported."""
        assert MAX_TIME_RESOLUTION_IN_SAMPLES == 1e-2

    @pytest.mark.parametrize(
        ("label", "times"),
        [
            ("all", [np.nan, np.nan, np.nan]),
            ("interior", [0.0, 100.0, np.nan, 300.0]),
            ("infinite", [0.0, 100.0, np.inf, 300.0]),
            ("single", [np.nan]),
        ],
    )
    def test_a_non_finite_axis_is_named_rather_than_defeating_the_comparison(
        self, label, times
    ):
        """The trap this check would otherwise walk into.

        NaN compares False against everything, so ``nan > 0`` is False and a
        NaN gap is silently dropped from "the smallest distinct gap". An
        ALL-NaN axis then has no gap left to check at all and would sail
        straight through a guard that only ever compares. Naming non-finiteness
        first is what makes the comparison below safe to write.
        """
        with pytest.raises(StateValidationError, match="non-finite"):
            Coordinates(time=np.array(times))

    def test_a_traced_axis_is_stepped_over_rather_than_crashed_on(self):
        """Under jit / vmap / grad there are no values to compare, and forcing
        them would turn a precision guard into a tracer leak. The axis handed
        in here is the refused one, so a check that did not step aside would
        fail loudly rather than quietly."""

        def build(t):
            return Coordinates(time=t).time.sum()

        traced = jnp.asarray(self.UNIX)
        assert np.isfinite(float(jax.jit(build)(traced)))
        assert np.isfinite(float(jax.vmap(build)(traced.reshape(2, 4)).sum()))
        assert np.isfinite(float(jax.grad(build)(traced).sum()))

    def test_a_single_sample_axis_has_no_sampling_to_resolve(self):
        """One sample has no interval, so there is nothing to compare against
        — and nothing a coarse grid can have taken away from it."""
        one = np.asarray(Coordinates(time=np.array([1.75e9])).time)
        assert one.shape == (1,) and one[0] == pytest.approx(1.75e9, rel=1e-6)

    def test_an_empty_axis_is_accepted(self):
        assert Coordinates(time=np.array([])).time.shape == (0,)

    def test_repeated_timestamps_are_measured_by_the_smallest_DISTINCT_gap(self):
        """Where this check deliberately stops short of the CW operator's.

        Two samples sharing a timestamp is degenerate, but it is not evidence
        of lost precision — a container cannot tell a genuine repeat from a
        collision, and refusing every repeat would make this class reject data
        it has no business judging. ``CWCalibrationOperator`` does refuse a zero
        gap, because it subtracts times and a collapsed pair silently stops its
        tone drifting; that belongs to the arithmetic, not to the store.

        Nothing is lost on the motivating defect by stopping here: rounding
        makes every surviving gap a multiple of the representable spacing, so a
        uniformly-quantised axis that has collided still shows a smallest
        distinct gap of one or two grid steps and is refused above.
        """
        repeated = np.array([0.0, 0.0, 100.0, 300.0])  # gaps 0, 100, 200
        np.testing.assert_array_equal(np.asarray(Coordinates(time=repeated).time), repeated)

    def test_a_constant_axis_is_accepted_for_the_same_reason(self):
        assert np.asarray(Coordinates(time=np.full(4, 7.0)).time).tolist() == [7.0] * 4

    def test_a_descending_axis_is_measured_by_the_MAGNITUDE_of_its_gaps(self):
        """A reversed axis has negative gaps, and a negative cadence makes the
        comparison wave anything through. Gaps 300/100/200 descending, so the
        magnitude ordering is not the array ordering either."""
        down = np.array([600.0, 300.0, 200.0, 0.0])
        np.testing.assert_array_equal(np.asarray(Coordinates(time=down).time), down)

    def test_an_axis_anchored_on_a_future_epoch_is_refused_too(self):
        """``np.spacing`` of a NEGATIVE number is negative, and a negative
        resolution compares below any positive threshold. It is the magnitude
        the dtype has to carry, whichever side of the epoch it is on."""
        with pytest.raises(StateValidationError, match="representable"):
            Coordinates(time=-1_750_000_000.0 - 100.0 * np.arange(8))

    def test_an_mjd_axis_is_judged_on_its_cadence_not_on_its_unit(self):
        """MJD is not per se legitimate or illegitimate — the question is only
        whether the store resolves the sampling. At MJD 60000 the float32 grid
        is 3.9e-3 d (337 s): daily samples are carried fine, and a 100 s cadence
        (1.16e-3 d) is not, which is exactly as broken as float32 unix seconds
        and refused for the same measured reason.
        """
        daily = 60000.0 + np.array([0.0, 1.0, 3.0, 7.0])
        np.testing.assert_array_equal(np.asarray(Coordinates(time=daily).time), daily)

        with pytest.raises(StateValidationError, match="representable"):
            Coordinates(time=60000.0 + (100.0 / 86400.0) * np.arange(8))

    def test_an_integer_axis_is_left_alone(self):
        """``np.spacing`` on an integer promotes to float64 and answers 5.7e-14
        — the same dtype blindness this check exists to avoid, arriving from the
        other side. An integer axis represents integers exactly; whatever was
        truncated was truncated before this container ever saw it, and the
        promoted answer would be a statement about a dtype the axis is not in.
        """
        stamps = np.array([1_750_000_000, 1_750_000_100, 1_750_000_300], dtype=np.int64)
        assert np.asarray(Coordinates(time=stamps).time).dtype.kind == "i"

    def test_freq_is_not_subject_to_this_check(self):
        """Scope, stated: ``time`` is the axis operators do arithmetic on
        (``BackendOperator`` averages it, ``CWCalibrationOperator`` drifts
        against it). A frequency axis is indexed, not differenced, so it is not
        checked here — a 1.75 GHz band with 100 Hz channels is as unresolvable
        in float32 as the time axis above, and would need its own argument
        before this container started refusing it."""
        c = Coordinates(freq=1.75e9 + 100.0 * np.arange(8))
        assert len(set(np.asarray(c.freq).tolist())) < 8  # the loss is real, and allowed

    def test_replace_re_runs_the_check(self):
        """The check has to survive the functional-update path, which is how
        every operator writes a new time axis."""
        good = Coordinates(time=np.arange(4.0) * 100.0)
        with pytest.raises(StateValidationError, match="representable"):
            good.replace(time=self.UNIX)


class TestEnvironment:
    def test_all_optional(self):
        e = Environment()
        assert e.temperature is None and e.humidity is None and e.extra == {}

    def test_basic_and_converter(self):
        e = Environment(temperature=np.array([280.0, 281.0]), humidity=[0.4, 0.5])
        assert isinstance(e.temperature, jax.Array)
        assert isinstance(e.humidity, jax.Array)

    def test_replace_returns_new(self):
        e = Environment(temperature=jnp.array(280.0))
        e2 = e.replace(humidity=jnp.array(0.5))
        assert e.humidity is None and e2.humidity is not None

    def test_extra_is_traced(self):
        e = Environment(extra={"wind_speed": jnp.zeros(3)})
        assert len(jax.tree.leaves(e)) == 1
