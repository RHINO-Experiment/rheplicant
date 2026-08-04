"""The CW calibration tone: where it may sit, what it may be, what it protects.

Three separate silent failures are pinned here, all of them measured on this
code before it was changed:

* the tone assembled cleanly *downstream* of the gain and its gain response
  dropped to exactly 1.0 — a calibrator tracking nothing;
* its amplitude was a differentiable leaf, so the one thing that makes it
  useful (being KNOWN) was not enforced anywhere;
* its frequency was never checked against the observing band, and ``argmin``
  always returns some channel.
"""

from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from rheplicant import Coordinates, Environment, State
from rheplicant.core.errors import DirtError, PipelineError, StateValidationError
from rheplicant.core.graph import AssemblyError, At
from rheplicant.core.operator import AbstractOperator
from rheplicant.core.pipeline import Pipeline
from rheplicant.radio import (
    PROTECTED_KEY,
    CWCalibrationOperator,
    GainOperator,
    ReceiverOperator,
    assemble,
)

# Deliberately non-square and asymmetric: a symmetric fixture has hidden a
# transposed index in this package three times.
N_TIME, N_FREQ = 6, 5
FREQ = jnp.linspace(60e6, 85e6, N_FREQ)
TONE_CHANNEL = 2
TONE_FREQ = float(FREQ[TONE_CHANNEL])
TONE_KELVIN = 5000.0
# Neither flat nor symmetric, so a wrong channel or a dropped factor shows up
# as a different number rather than the same one.
BANDPASS = jnp.array([0.80, 0.95, 1.10, 1.05, 0.90])
GAIN = jnp.array([1.5, 1.55, 1.6, 1.65, 1.7, 1.75])


@pytest.fixture
def state():
    return State(
        data=jnp.full((N_TIME, N_FREQ), 10.0),
        coords=Coordinates(time=jnp.arange(N_TIME, dtype=float), freq=FREQ),
        env=Environment(temperature=jnp.array(280.0)),
        meta={"telescope": "RHINO", "obs_id": "cw-000"},
    )


# One channel spacing, which for the default 'sinc2' lineshape is the response
# of a critically sampled unwindowed FFT. Every tone in this file sits exactly
# on a channel centre, where sinc() has a zero in every OTHER channel — so the
# numbers below are the placeholder's numbers, unchanged, and that is the point:
# giving the line a shape is not a regression for an on-centre tone.
CHANNEL = float(FREQ[1] - FREQ[0])


def _tone(**kwargs):
    settings = {
        "amplitude": TONE_KELVIN,
        "tone_freq": TONE_FREQ,
        "line_width": CHANNEL,
    }
    return CWCalibrationOperator(**{**settings, **kwargs})


class TestTheOrderingConstraintIsEnforced:
    """What the refusal is FOR — measured on both sides of it.

    There are two refusals now, one per composition route: ``assemble`` by
    reachability on the graph, ``Pipeline`` by the order of its own stage names.
    They protect the same measurement, so it is taken once, below the container
    that would otherwise refuse to produce it.
    """

    def test_downstream_of_the_gain_the_tone_response_is_exactly_one(self, state):
        """The silent failure, and the refusal that now stands in front of it.

        This is the whole case for the mechanism: nothing about the wrong
        composition errors, nothing is NaN, the shapes are right, and the
        calibrator is worthless.

        Both composition routes now refuse to *build* it — ``assemble`` by
        reachability on the graph, ``Pipeline`` sequence-locally — so the
        measurement is taken by applying the three operators directly, which is
        precisely what the refused ``Pipeline`` would have done to the state.
        The bypass is deliberate and it is the point: this number is the
        evidence the check exists for, and a suite that could no longer produce
        it would leave the refusal looking like an unmotivated cost to the next
        person who finds it inconvenient. The second half of the test is the
        refusal itself, so the bug and its fix are pinned in one place.
        """
        receiver = ReceiverOperator(bandpass=BANDPASS)
        gain = GainOperator(gain=GAIN)
        tone = _tone()

        # -- the physics: what Pipeline(receiver, gain, tone) would have run ---
        wrong = tone(gain(receiver(state)))
        bare = Pipeline(receiver, gain)
        response = (wrong.data - bare(state).data)[:, TONE_CHANNEL]
        assert jnp.allclose(response, TONE_KELVIN)  # gain response 1.0, exactly
        # ... which is not what the tone does where it belongs: b * g, and the
        # two disagree because BANDPASS and GAIN are neither 1.0 nor each other.
        assert not jnp.allclose(response, TONE_KELVIN * BANDPASS[TONE_CHANNEL] * GAIN)

        # -- the fix: that composition can no longer be built ------------------
        with pytest.raises(PipelineError) as excinfo:
            Pipeline(receiver, gain, tone)
        assert "must_precede" in str(excinfo.value)

    def test_upstream_of_the_gain_the_tone_response_is_b_times_g(self, state):
        """The correct placement, and the reason the constraint is stated."""
        right = Pipeline(
            _tone(), ReceiverOperator(bandpass=BANDPASS), GainOperator(gain=GAIN)
        )
        bare = Pipeline(ReceiverOperator(bandpass=BANDPASS), GainOperator(gain=GAIN))
        response = (right(state).data - bare(state).data)[:, TONE_CHANNEL]
        expected = TONE_KELVIN * BANDPASS[TONE_CHANNEL] * GAIN
        assert jnp.allclose(response, expected)
        # ... and it is the product, not one of the two factors
        assert not jnp.allclose(response, TONE_KELVIN)

    def test_assemble_refuses_the_tone_downstream_of_the_gain(self):
        with pytest.raises(AssemblyError) as excinfo:
            assemble(
                ReceiverOperator(bandpass=BANDPASS),
                GainOperator(gain=GAIN),
                At("noise", _tone()),
            )
        message = str(excinfo.value)
        assert "CWCalibrationOperator" in message
        assert "'bandpass'" in message and "'gain'" in message
        assert "'noise'" in message
        assert "response is exactly 1.0" in message

    def test_both_routes_refuse_it_and_agree_on_why(self):
        """The two refusals differ by LAYER and must not differ on the physics.

        ``assemble`` speaks about a graph and raises ``AssemblyError``;
        ``Pipeline`` speaks about a sequence it can see the names of and raises
        ``PipelineError``. That difference is deliberate — each names the thing
        that was misconfigured, and the graph one can say more, because it knows
        the tone was put on ``noise`` and that ``bandpass`` is unreachable from
        there, where the sequence one only knows ``gain`` runs first. It costs a
        caller nothing: both derive from ``DirtError`` and from ``ValueError``,
        so one ``except`` clause catches either.

        What may not drift is what they say goes *wrong*, and both quote the
        operator's own ``must_precede_because`` back.
        """
        with pytest.raises(AssemblyError) as by_graph:
            assemble(
                ReceiverOperator(bandpass=BANDPASS),
                GainOperator(gain=GAIN),
                At("noise", _tone()),
            )
        with pytest.raises(PipelineError) as by_sequence:
            Pipeline(
                ReceiverOperator(bandpass=BANDPASS), GainOperator(gain=GAIN), _tone()
            )
        for excinfo in (by_graph, by_sequence):
            message = str(excinfo.value)
            assert "CWCalibrationOperator" in message  # the operator
            assert "['bandpass', 'gain']" in message  # the constraint it declared
            assert "response is exactly 1.0" in message  # what goes silently wrong
            assert isinstance(excinfo.value, DirtError)
            assert isinstance(excinfo.value, ValueError)
        assert type(by_graph.value) is not type(by_sequence.value)

    def test_assemble_accepts_the_tone_at_its_home_node(self, state):
        twin = assemble(
            _tone(), ReceiverOperator(bandpass=BANDPASS), GainOperator(gain=GAIN)
        )
        assert twin.lit == ("cw_tone", "bandpass", "gain")

    def test_the_constraint_is_declared_where_the_graph_can_read_it(self):
        assert CWCalibrationOperator.must_precede == ("bandpass", "gain")
        assert CWCalibrationOperator.must_precede_because


class TestTheAmplitudeIsKnownAndStatic:
    def test_the_amplitude_is_not_a_differentiable_leaf(self):
        """The premise of the whole calibration route is that this is KNOWN.

        A free amplitude is absorbed exactly by the gain it is meant to track,
        so inferring it removes the only thing the tone contributes.

        The unfiltered ``tree_leaves`` is the load-bearing one. ``eqx.filter(
        op, eqx.is_inexact_array)`` filters on ARRAYS, and the converter has
        already turned the amplitude into a Python float — so that assertion
        passes with ``static=True`` removed, and this claim would go unpinned.
        """
        assert jax.tree_util.tree_leaves(_tone()) == []
        assert jax.tree_util.tree_leaves(eqx.filter(_tone(), eqx.is_inexact_array)) == []
        assert isinstance(_tone().amplitude, float)

    def test_a_jax_scalar_is_accepted_and_stored_as_a_number(self):
        """Existing call sites pass ``jnp.array(50.0)``; it must keep working
        and must NOT end up as an unhashable array in the treedef."""
        op = _tone(amplitude=jnp.array(50.0))
        assert op.amplitude == 50.0 and isinstance(op.amplitude, float)
        hash(jax.tree_util.tree_structure(op))  # static fields must stay hashable

    def test_a_non_scalar_amplitude_is_refused(self):
        with pytest.raises(StateValidationError, match="amplitude must be a scalar"):
            _tone(amplitude=jnp.ones(N_FREQ))

    def test_a_one_element_vector_is_still_not_a_scalar(self):
        """``float()`` would happily accept it, and it would then broadcast."""
        with pytest.raises(StateValidationError, match=r"shape \(1,\)"):
            _tone(amplitude=jnp.ones(1))

    def test_a_traced_amplitude_is_refused(self):
        """Differentiating through the construction is the way this is reached."""

        def build(a):
            return _tone(amplitude=a).amplitude

        with pytest.raises(StateValidationError, match="KNOWN, static number"):
            jax.jit(build)(jnp.array(1.0))

    def test_a_non_numeric_amplitude_is_refused(self):
        with pytest.raises(StateValidationError, match="KNOWN, static number"):
            _tone(amplitude="loud")


class TestTheToneFrequencyIsInsideTheBand:
    def test_a_tone_above_the_band_is_refused(self, state):
        with pytest.raises(StateValidationError, match="outside the observed band"):
            _tone(tone_freq=200e6)(state)

    def test_a_tone_below_the_band_is_refused(self, state):
        with pytest.raises(StateValidationError, match="outside the observed band"):
            _tone(tone_freq=1e6)(state)

    def test_a_tone_just_outside_the_band_is_refused_on_either_side(self, state):
        """The refusing side AT the boundary, which 200 MHz does not pin.

        A tone 115 MHz outside a 25 MHz band survives any amount of accidental
        widening of the accepted interval; a hundredth of a channel (62.5 kHz)
        does not, and that is what makes this a boundary test rather than a
        smoke test.
        """
        for outside in (float(FREQ[-1]) + 0.01 * CHANNEL,
                        float(FREQ[0]) - 0.01 * CHANNEL):
            with pytest.raises(StateValidationError, match="outside the observed band"):
                _tone(tone_freq=outside)(state)

    def test_the_band_edges_are_inside_it(self, state):
        """Closed interval: a tone at the first or last channel is legitimate."""
        for edge, channel in ((float(FREQ[0]), 0), (float(FREQ[-1]), N_FREQ - 1)):
            out = _tone(tone_freq=edge)(state)
            assert out.data[0, channel] == 10.0 + TONE_KELVIN

    def test_missing_coords_are_refused(self):
        bare = State(data=jnp.ones((N_TIME, N_FREQ)), meta={"obs_id": "x"})
        with pytest.raises(StateValidationError, match="requires state.coords.freq"):
            _tone()(bare)

    def test_coords_without_a_frequency_axis_are_refused(self):
        """The other half of the same guard. Coordinates carrying only a time
        axis is the ordinary shape of a pipeline that has not reached the
        spectrometer yet, and it is not None."""
        timed = State(
            data=jnp.ones((N_TIME, N_FREQ)),
            coords=Coordinates(time=jnp.arange(N_TIME, dtype=float)),
            meta={"obs_id": "x"},
        )
        with pytest.raises(StateValidationError, match="requires state.coords.freq"):
            _tone()(timed)

    def test_the_check_survives_jit_over_a_closed_over_band(self, state):
        """Layer 1, the pattern that matters: coords are a closure constant, so
        the values are still readable inside the trace and this raises at trace
        time rather than modelling a spike in the wrong channel."""
        bad = _tone(tone_freq=200e6)
        with pytest.raises(StateValidationError, match="outside the observed band"):
            jax.jit(lambda d: bad(state.with_data(d)).data)(state.data)

    def test_a_traced_band_skips_the_check_rather_than_crashing(self, state):
        """Layer 2, and the limit, stated rather than papered over.

        With coords passed as a traced ARGUMENT the values genuinely cannot be
        read, so no exception is possible and none is attempted. A valid tone
        must still compile and give the same answer as the eager run.
        """
        run = eqx.filter_jit(lambda op, s: op(s))
        good = _tone()
        assert jnp.allclose(run(good, state).data, good(state).data)
        # and the out-of-band tone is NOT caught here — this is the limit. It
        # runs, and what it produces is worse than a spike in the wrong channel:
        # 200 MHz is so far outside a 60-85 MHz band that the sinc2 wings are
        # nearly flat across it, so the full 5000 K is smeared over the WHOLE
        # band, in no channel the tone is anywhere near.
        injected = run(_tone(tone_freq=200e6), state).data - state.data
        assert jnp.isclose(injected[0].sum(), TONE_KELVIN, rtol=1e-3)
        assert float(injected[0].min()) > 0.1 * float(injected[0].max())


class TestTheToneProtectsItsOwnChannel:
    def test_the_injected_channel_is_marked_protected(self, state):
        out = _tone()(state)
        protected = out.aux[PROTECTED_KEY]
        assert protected.shape == (N_FREQ,) and protected.dtype == jnp.bool_
        assert bool(protected[TONE_CHANNEL])
        assert int(protected.sum()) == 1

    def test_it_declares_what_it_provides(self):
        assert f"aux.{PROTECTED_KEY}" in CWCalibrationOperator.provides

    def test_two_tones_protect_both_channels(self, state):
        """Composing, not clobbering: whichever ran second would otherwise
        unprotect the first."""
        both = Pipeline(_tone(), _tone(tone_freq=float(FREQ[0])))(state)
        protected = both.aux[PROTECTED_KEY]
        assert [bool(x) for x in protected] == [True, False, True, False, False]

    def test_protection_rides_the_trunk_from_cw_tone_to_the_flagger(self, state):
        """The stages between ``cw_tone`` and ``flagging`` must not drop aux."""

        class Trunk(AbstractOperator):
            requires: ClassVar[tuple[str, ...]] = ("data",)
            provides: ClassVar[tuple[str, ...]] = ("data",)

            def __call__(self, s):
                return s.with_data(s.data * 2.0)

        out = Pipeline(
            _tone(), ReceiverOperator(bandpass=BANDPASS), GainOperator(gain=GAIN), Trunk()
        )(state)
        assert bool(out.aux[PROTECTED_KEY][TONE_CHANNEL])
