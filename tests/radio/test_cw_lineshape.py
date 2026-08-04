"""The CW tone as a real line: spectral shape, width, drift, and what it wets.

The placeholder added one number to one channel. That is not a thing a
spectrometer can produce: a monochromatic injection is observed through the
channel response, so it has a *shape* and it wets several channels; and over a
run both its centre frequency and its level drift.

Three failure modes are pinned here, each of which is finite, correctly shaped
and silent if it goes wrong:

* the injected total depending on the lineshape or on where the tone falls
  between channels — the tone's level is a known instrument setting, so the
  total is the one thing that must not move;
* the protection mask covering one index while the line wets five, which hands
  the flagger four bright channels and destroys the calibrator it exists to
  keep;
* a drifting tone protected by a static channel mask, which protects the wrong
  channels for every sample except the first.

Fixtures are deliberately non-square (4 x 11), the tone channel is not the
middle one, and the drift moves the line by a whole number of channels, so a
transposed axis or a stuck index shows up as a different number.

The time AXIS the drift is measured against — its precision, its anchor, and
the waterfall mask it leaves behind — is pinned separately, in
``tests/radio/test_cw_time_axis.py``.
"""

import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rheplicant import Coordinates, State
from rheplicant.core.errors import StateValidationError
from rheplicant.core.pipeline import Pipeline
from rheplicant.radio import (
    PROTECTED_KEY,
    CWCalibrationOperator,
    GainOperator,
    ReceiverOperator,
    protect,
)

# 4 x 11: non-square, and 11 channels is enough room for a line with wings.
N_TIME, N_FREQ = 4, 11
CHANNEL = 1e6                                   # channel spacing [Hz]
FREQ = 60e6 + CHANNEL * jnp.arange(N_FREQ, dtype=float)
TONE_CHANNEL = 4                                # 64 MHz — NOT the middle (5)
TONE_FREQ = float(FREQ[TONE_CHANNEL])
TONE_KELVIN = 5000.0
TIME = 100.0 * jnp.arange(N_TIME, dtype=float)  # 0, 100, 200, 300 s
# 1e4 Hz/s over 300 s moves the centre exactly 3 channels: 4 -> 5 -> 6 -> 7.
DRIFT_PER_CHANNEL = CHANNEL / 100.0


@pytest.fixture
def state():
    return State(
        data=jnp.full((N_TIME, N_FREQ), 10.0),
        coords=Coordinates(time=TIME, freq=FREQ),
        meta={"telescope": "RHINO", "obs_id": "cw-line-000"},
    )


def _tone(**kwargs):
    settings = {
        "amplitude": TONE_KELVIN,
        "tone_freq": TONE_FREQ,
        "line_width": CHANNEL,
    }
    return CWCalibrationOperator(**{**settings, **kwargs})


def _injected(op, state):
    """What the operator added, per sample and channel."""
    return np.asarray(op(state).data - state.data)


# ------------------------------------------------------- the line has a shape


class TestTheInjectedTotalIsTheKnownLevel:
    """``amplitude`` is the tone's total, not whatever lands in one channel.

    The tone's level is the one thing the operator KNOWS, so it must not
    depend on the spectrometer's channelisation or on where the line happens
    to fall between two channels. Normalising the lineshape over the sampled
    channels is what makes that true.
    """

    @pytest.mark.parametrize("lineshape", ["sinc2", "gaussian"])
    @pytest.mark.parametrize("width", [1.0 * CHANNEL, 2.3 * CHANNEL])
    @pytest.mark.parametrize("offset", [0.0, 0.5 * CHANNEL, 0.31 * CHANNEL])
    def test_the_total_is_the_amplitude_whatever_the_shape(
        self, state, lineshape, width, offset
    ):
        op = _tone(lineshape=lineshape, line_width=width, tone_freq=TONE_FREQ + offset)
        total = _injected(op, state).sum(axis=-1)
        assert np.allclose(total, TONE_KELVIN, rtol=1e-4), total

    def test_the_total_does_not_move_when_the_line_widens(self, state):
        narrow = _injected(_tone(lineshape="gaussian", line_width=0.5 * CHANNEL), state)
        # 2.4 channels: nearly five times the narrow one, and just inside the
        # width ceiling this band imposes (0.25 x 10 MHz = 2.5 channels).
        wide = _injected(_tone(lineshape="gaussian", line_width=2.4 * CHANNEL), state)
        assert np.allclose(narrow.sum(), wide.sum(), rtol=1e-4)
        # ... and the widening is real: the peak channel drops a long way.
        assert wide[0, TONE_CHANNEL] < 0.5 * narrow[0, TONE_CHANNEL]


class TestTheSincSquaredResponse:
    """A critically-sampled unwindowed FFT, which is the assumption stated."""

    def test_a_tone_on_a_channel_centre_lands_entirely_in_that_channel(self, state):
        """The placeholder's answer, recovered — and now as a consequence.

        ``sinc(k)`` vanishes at every nonzero integer, so a tone exactly on a
        channel centre with ``line_width`` one channel really does put all of
        its power in that channel. This is why the change is not a regression
        for on-centre tones.
        """
        injected = _injected(_tone(), state)
        assert np.allclose(injected[:, TONE_CHANNEL], TONE_KELVIN, rtol=1e-6)
        others = np.delete(injected[0], TONE_CHANNEL)
        assert np.abs(others).max() < 1e-6 * TONE_KELVIN

    def test_a_tone_halfway_between_channels_scallops(self, state):
        """The classic half-bin loss: the peak channel keeps ~0.42, not 1.0.

        This is the number that makes a tone-tracked gain biased if it is
        ignored — the same physical tone reads 3.8 dB lower simply for sitting
        between two channels.
        """
        injected = _injected(_tone(tone_freq=TONE_FREQ + 0.5 * CHANNEL), state)
        left, right = injected[0, TONE_CHANNEL], injected[0, TONE_CHANNEL + 1]
        assert np.isclose(left, right, rtol=1e-5)          # symmetric about the line
        assert 0.40 < left / TONE_KELVIN < 0.43
        assert np.isclose(injected[0].sum(), TONE_KELVIN, rtol=1e-4)


class TestTheGaussianResponse:
    def test_the_wings_fall_off_as_the_gaussian_says(self, state):
        """Ratios, not absolutes, so the normalisation cannot fake this."""
        sigma = 1.7 * CHANNEL
        injected = _injected(_tone(lineshape="gaussian", line_width=sigma), state)[0]
        peak = injected[TONE_CHANNEL]
        for step in (1, 2, 3):
            expected = np.exp(-0.5 * (step * CHANNEL / sigma) ** 2)
            assert np.isclose(injected[TONE_CHANNEL + step] / peak, expected, rtol=1e-4)
            assert np.isclose(injected[TONE_CHANNEL - step] / peak, expected, rtol=1e-4)

    def test_a_narrow_gaussian_at_the_width_floor_is_still_exact(self, state):
        injected = _injected(_tone(lineshape="gaussian", line_width=0.25 * CHANNEL), state)
        assert np.isfinite(injected).all()
        assert np.isclose(injected[0].sum(), TONE_KELVIN, rtol=1e-4)
        # 0.25 channels still leaks exp(-8) into each neighbour, so the peak
        # keeps 0.999 of the total rather than all of it.
        assert np.isclose(injected[0, TONE_CHANNEL], TONE_KELVIN, rtol=2e-3)

    def test_a_gaussian_the_band_check_could_not_see_is_finite_not_nan(self, state):
        """Where the shifted-exponent evaluation earns its keep.

        The band check has one documented hole: traced coords, where the tone's
        offset from every channel cannot be read. A gaussian 115 channels away
        from the nearest one gives ``exp(-6612)``, which is zero in float32 for
        EVERY channel — and ``0/0`` in the normalisation. Written as
        ``exp(-z^2/2)`` directly, the hole in a validity check turns into NaN
        across the whole run; with the peak exponent subtracted the largest
        weight is exactly 1, so the answer is a wrong-but-finite delta on the
        nearest channel and the failure stays where it started.
        """
        run = eqx.filter_jit(lambda o, s: o(s))
        out = run(_tone(lineshape="gaussian", tone_freq=200e6), state)
        injected = np.asarray(out.data - state.data)
        assert np.isfinite(injected).all()
        assert np.isclose(injected[0].sum(), TONE_KELVIN, rtol=1e-4)
        assert injected[0].argmax() == N_FREQ - 1      # the nearest channel


# ------------------------------------------------------------------- refusals


class TestTheLineshapeIsOneThisOperatorKnows:
    def test_an_unknown_lineshape_is_refused_and_both_are_named(self):
        with pytest.raises(StateValidationError) as excinfo:
            _tone(lineshape="lorentzian")
        message = str(excinfo.value)
        assert "'sinc2'" in message and "'gaussian'" in message

    @pytest.mark.parametrize("shape", ["sinc2", "gaussian"])
    def test_both_declared_shapes_are_accepted(self, shape):
        assert _tone(lineshape=shape).lineshape == shape


class TestTheWidthIsAWidthThisGridCouldHaveProduced:
    """Boundary validation: the floor is checked from BOTH sides."""

    def test_a_non_positive_width_is_refused(self):
        for width in (0.0, -CHANNEL):
            with pytest.raises(StateValidationError, match="line_width must be > 0"):
                _tone(line_width=width)

    def test_a_sinc2_narrower_than_one_channel_is_refused(self, state):
        with pytest.raises(StateValidationError, match="narrower than the channel"):
            _tone(line_width=0.99 * CHANNEL)(state)

    def test_a_sinc2_of_exactly_one_channel_is_accepted(self, state):
        """The other side of the same boundary — and it is the canonical value
        for a critically-sampled unwindowed FFT, so it must not be rejected."""
        assert np.isfinite(_injected(_tone(line_width=CHANNEL), state)).all()

    def test_a_gaussian_narrower_than_a_quarter_channel_is_refused(self, state):
        with pytest.raises(StateValidationError, match="narrower than the channel"):
            _tone(lineshape="gaussian", line_width=0.249 * CHANNEL)(state)

    def test_a_gaussian_of_exactly_a_quarter_channel_is_accepted(self, state):
        assert np.isfinite(
            _injected(_tone(lineshape="gaussian", line_width=0.25 * CHANNEL), state)
        ).all()

    def test_the_floor_differs_by_lineshape(self, state):
        """A width legal for one shape and illegal for the other — a single
        floor for both would either reject the canonical FFT value or admit a
        sinc2 whose sampled channels sit on its nulls."""
        half = 0.5 * CHANNEL
        assert np.isfinite(
            _injected(_tone(lineshape="gaussian", line_width=half), state)
        ).all()
        with pytest.raises(StateValidationError, match="narrower than the channel"):
            _tone(lineshape="sinc2", line_width=half)(state)

    def test_a_line_wider_than_a_quarter_of_the_band_is_refused(self, state):
        """The ceiling, which the floor's own docstring implies and nothing
        enforced: past it the injection is a pedestal across the band, every
        channel clears ``protect_floor``, and the flagger is off for the run."""
        with pytest.raises(StateValidationError, match="wider than a LINE") as excinfo:
            _tone(line_width=2.51 * CHANNEL)(state)
        message = str(excinfo.value)
        assert "PEDESTAL" in message
        # ... and it names the number the caller probably wanted instead
        assert "one channel here is 1e+06 Hz" in message

    def test_a_line_of_exactly_a_quarter_of_the_band_is_accepted(self, state):
        """The other side of the same boundary: 0.25 x 10 MHz = 2.5 channels."""
        assert np.isfinite(_injected(_tone(line_width=2.5 * CHANNEL), state)).all()

    def test_the_band_the_typo_produces_is_the_one_that_is_refused(self, state):
        """``line_width=25e6`` where ``25e6 / (N_FREQ - 1)`` was meant — one
        keystroke, and the whole band becomes a single uniform pedestal."""
        with pytest.raises(StateValidationError, match="wider than a LINE"):
            _tone(line_width=float(FREQ[-1] - FREQ[0]))(state)

    def test_a_coarse_grid_does_not_make_a_one_channel_line_too_wide(self):
        """Four channels: a quarter of the band is three quarters of ONE
        channel, so a pure band-fraction ceiling would refuse the width the
        floor calls canonical. The ceiling never falls below two channels."""
        coarse_freq = 60e6 + 8e6 * jnp.arange(4, dtype=float)   # span 24 MHz
        coarse = State(
            data=jnp.full((N_TIME, 4), 10.0),
            coords=Coordinates(time=TIME, freq=coarse_freq),
            meta={"obs_id": "coarse"},
        )
        out = _tone(tone_freq=float(coarse_freq[1]), line_width=8e6)(coarse)
        injected = np.asarray(out.data - coarse.data)
        assert np.isclose(injected[0].sum(), TONE_KELVIN, rtol=1e-4)

    def test_a_descending_frequency_grid_is_still_measured_by_its_spacing(self):
        """``median(diff)`` is negative on a descending grid, and a negative
        floor passes every width. Only ``median(|diff|)`` refuses this."""
        down = State(
            data=jnp.full((N_TIME, N_FREQ), 10.0),
            coords=Coordinates(time=TIME, freq=FREQ[::-1]),
            meta={"obs_id": "descending"},
        )
        with pytest.raises(StateValidationError, match="narrower than the channel"):
            _tone(line_width=0.5 * CHANNEL)(down)

    def test_a_non_uniform_grid_is_measured_by_the_median_spacing(self):
        """One narrow channel at the bottom, ten ordinary ones. The FIRST
        spacing is 0.1 MHz and the MEAN is 0.9 MHz; the median is 1.0 MHz, and
        the two widths below separate all three."""
        edges = jnp.array([60.0, 60.1] + [60.0 + k for k in range(1, 10)]) * 1e6
        ragged = State(
            data=jnp.full((N_TIME, N_FREQ), 10.0),
            coords=Coordinates(time=TIME, freq=edges),
            meta={"obs_id": "ragged"},
        )
        for width in (0.5 * CHANNEL, 0.95 * CHANNEL):
            with pytest.raises(StateValidationError, match="narrower than the channel"):
                _tone(tone_freq=float(edges[4]), line_width=width)(ragged)

    def test_a_single_channel_grid_cannot_establish_a_spacing(self):
        """One channel has no spacing to compare against, so the check steps
        aside rather than inventing one — and steps aside QUIETLY.

        ``np.median`` of an empty diff is NaN and warns as it goes, and a NaN
        floor makes every ``<`` comparison False, so dropping the guard leaves
        a check that silently passes everything while emitting a numpy warning
        from inside an operator. Warnings-as-errors is what pins that.
        """
        one = State(
            data=jnp.full((N_TIME, 1), 10.0),
            coords=Coordinates(time=TIME, freq=jnp.array([TONE_FREQ])),
            meta={"obs_id": "one-channel"},
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = _tone(line_width=1e-3 * CHANNEL)(one)
        assert np.allclose(np.asarray(out.data), 10.0 + TONE_KELVIN, rtol=1e-5)

    def test_a_width_that_is_not_a_static_number_is_refused(self):
        with pytest.raises(StateValidationError, match="KNOWN, static number"):
            _tone(line_width="wide")

    def test_an_array_width_is_refused(self):
        with pytest.raises(StateValidationError, match="line_width must be a scalar"):
            _tone(line_width=jnp.ones(N_FREQ))


class TestTheProtectionFloorIsAFraction:
    @pytest.mark.parametrize("floor", [0.0, -0.1, 1.5])
    def test_a_floor_outside_zero_to_one_is_refused(self, floor):
        with pytest.raises(StateValidationError, match="protect_floor"):
            _tone(protect_floor=floor)

    def test_a_floor_of_exactly_one_protects_only_the_peak(self, state):
        """The other side of the boundary: 1.0 is legal and means "the peak
        channel only"."""
        mask = _tone(
            lineshape="gaussian", line_width=2.0 * CHANNEL, protect_floor=1.0
        )(state).aux[PROTECTED_KEY]
        assert int(np.asarray(mask).sum()) == 1


# ----------------------------------------------------------------- the drift


class TestTheCentreDrifts:
    def test_the_line_moves_a_channel_at_a_time(self, state):
        """The measurement: peak channel 4 -> 5 -> 6 -> 7 across the run."""
        injected = _injected(_tone(drift_rate=DRIFT_PER_CHANNEL), state)
        assert list(injected.argmax(axis=-1)) == [4, 5, 6, 7]
        # and it really is the whole tone moving, not a smear
        assert np.allclose(injected.max(axis=-1), TONE_KELVIN, rtol=1e-5)

    @pytest.mark.parametrize("anchor", [0.0, 43200.0, 1.0e6])
    def test_the_drift_is_measured_from_the_first_sample_not_from_zero(
        self, state, anchor
    ):
        """The anchor, pinned — every other fixture in this file starts its
        time axis at zero, where ``t`` and ``t - t[0]`` are the same array.

        Both places that compute the anchor are covered by this one test, and
        they must agree: with the injection anchored and the band guard not,
        the guard validates a RELATIVE centre while the operator places an
        ABSOLUTE one, and passes a tone it then puts hundreds of MHz out of
        band. Noon (43200 s) and 1e6 s are both real ways to tag a run, and
        both are far enough from zero that a dropped anchor is a refusal, not
        a small error.
        """
        shifted = state.replace(coords=state.coords.replace(time=TIME + anchor))
        injected = _injected(_tone(drift_rate=DRIFT_PER_CHANNEL), shifted)
        assert list(injected.argmax(axis=-1)) == [4, 5, 6, 7]
        assert np.allclose(injected.max(axis=-1), TONE_KELVIN, rtol=1e-5)

    def test_without_drift_the_line_stays_put(self, state):
        injected = _injected(_tone(), state)
        assert list(injected.argmax(axis=-1)) == [TONE_CHANNEL] * N_TIME

    def test_a_tone_that_drifts_out_of_the_band_during_the_run_is_refused(self, state):
        """In band at the first sample, gone by the last — which a check on
        the first sample alone would never notice."""
        with pytest.raises(StateValidationError) as excinfo:
            _tone(drift_rate=100.0 * DRIFT_PER_CHANNEL)(state)
        message = str(excinfo.value)
        assert "outside the observed band" in message
        # and it says WHY the centre left, not just that it is out
        assert "drifting at" in message

    def test_a_tone_that_drifts_downwards_out_of_the_band_is_refused(self, state):
        with pytest.raises(StateValidationError, match="outside the observed band"):
            _tone(drift_rate=-100.0 * DRIFT_PER_CHANNEL)(state)

    def test_a_drift_that_stays_in_band_is_accepted(self, state):
        """Boundary: 6 channels of drift from channel 4 lands on channel 10,
        the last one, and is legal."""
        out = _tone(drift_rate=6.0 * DRIFT_PER_CHANNEL / 3.0)(state)
        assert np.asarray(out.data - state.data).argmax(axis=-1)[-1] == N_FREQ - 1

    def test_a_drifting_tone_without_times_is_refused(self):
        no_time = State(
            data=jnp.full((N_TIME, N_FREQ), 10.0),
            coords=Coordinates(freq=FREQ),
            meta={"obs_id": "no-time"},
        )
        with pytest.raises(StateValidationError, match="coords.time"):
            _tone(drift_rate=DRIFT_PER_CHANNEL)(no_time)

    def test_traced_times_skip_the_run_check_rather_than_crashing(self, state):
        """The SECOND escape hatch, which needs its own reachable case.

        A band that is a closure constant while the sample times are a traced
        argument is the ordinary shape of a scan over time chunks — and it is
        the only way to reach this escape, because a fully traced state trips
        the frequency one first and returns before the times are touched. Here
        the band IS readable, so the width check still runs; the run's EXTENT
        is not, so the drift check cannot. Without the escape a legal jitted
        pipeline dies on ``TracerArrayConversionError`` raised from inside a
        validity check.
        """
        def over_times(t):
            return _tone(drift_rate=DRIFT_PER_CHANNEL)(
                state.replace(coords=state.coords.replace(time=t))
            ).data

        assert np.allclose(
            np.asarray(jax.jit(over_times)(TIME)),
            np.asarray(over_times(TIME)),
            rtol=1e-5,
        )
        # and the drift out of band is NOT caught here — this is the limit
        def wild(t):
            return _tone(drift_rate=100.0 * DRIFT_PER_CHANNEL)(
                state.replace(coords=state.coords.replace(time=t))
            ).data

        assert np.isfinite(np.asarray(jax.jit(wild)(TIME))).all()
        # ... while the same tone with concrete times is still refused
        with pytest.raises(StateValidationError, match="outside the observed band"):
            wild(TIME)

    def test_a_static_tone_needs_no_times(self):
        """The requirement is the drift's, not the operator's — a pipeline with
        no time axis must keep working."""
        no_time = State(
            data=jnp.full((N_TIME, N_FREQ), 10.0),
            coords=Coordinates(freq=FREQ),
            meta={"obs_id": "no-time"},
        )
        out = _tone()(no_time)
        assert np.isclose(float(np.asarray(out.data)[0, TONE_CHANNEL]), 10.0 + TONE_KELVIN)


class TestTheLevelDrifts:
    RATE = -1e-3  # -0.1 %/s: 1.0 -> 0.7 over the 300 s run

    def test_the_total_follows_the_declared_drift(self, state):
        injected = _injected(_tone(amplitude_drift_rate=self.RATE), state)
        expected = TONE_KELVIN * (1.0 + self.RATE * np.asarray(TIME))
        assert np.allclose(injected.sum(axis=-1), expected, rtol=1e-4)
        # asymmetric on purpose: last sample is 0.7 of the first, not 1.0
        assert np.isclose(injected.sum(axis=-1)[-1] / injected.sum(axis=-1)[0], 0.7,
                          rtol=1e-4)

    def test_a_level_that_passes_through_zero_is_refused(self, state):
        """A tone that turns into a notch part-way through the run is finite,
        correctly shaped and nonsense."""
        with pytest.raises(StateValidationError, match="stops being a tone"):
            _tone(amplitude_drift_rate=-4e-3)(state)

    def test_a_level_that_reaches_exactly_zero_is_refused(self):
        """The closed side of that boundary, and the case ``<=`` was written
        for: the tone does not go negative, it VANISHES at the last sample —
        and the mask is still written, so the flagger is told to keep a channel
        with no tone left in it. Times and rate are powers of two so the last
        level is exactly 0.0 rather than a rounding away from it."""
        powers = State(
            data=jnp.full((N_TIME, N_FREQ), 10.0),
            coords=Coordinates(time=jnp.array([0.0, 128.0, 256.0, 512.0]), freq=FREQ),
            meta={"obs_id": "exact-zero"},
        )
        assert 1.0 + (-1.0 / 512.0) * 512.0 == 0.0     # the arithmetic, pinned
        with pytest.raises(StateValidationError, match="stops being a tone"):
            _tone(amplitude_drift_rate=-1.0 / 512.0)(powers)

    def test_a_level_drift_alone_leaves_the_mask_one_dimensional(self, state):
        """Nothing moves in frequency, so the contaminated channels do not
        change — a waterfall mask here would cost n_time x n_freq for nothing."""
        mask = _tone(amplitude_drift_rate=self.RATE)(state).aux[PROTECTED_KEY]
        assert mask.shape == (N_FREQ,)

    def test_a_time_varying_tone_refuses_data_the_times_do_not_match(self):
        """``(1, n_freq)`` data would broadcast against the per-sample level and
        silently return a run of the wrong length."""
        odd = State(
            data=jnp.full((1, N_FREQ), 10.0),
            coords=Coordinates(time=TIME, freq=FREQ),
            meta={"obs_id": "odd"},
        )
        with pytest.raises(StateValidationError, match="time samples"):
            _tone(drift_rate=DRIFT_PER_CHANNEL)(odd)


# -------------------------------------------------------------- what it wets


class TestTheProtectionCoversWhatTheToneWets:
    WIDE = {"lineshape": "gaussian", "line_width": 1.0 * CHANNEL}

    def test_a_line_with_width_protects_more_than_one_channel(self, state):
        mask = np.asarray(_tone(**self.WIDE)(state).aux[PROTECTED_KEY])
        assert mask.sum() > 1
        assert bool(mask[TONE_CHANNEL])

    def test_the_protected_set_is_exactly_the_channels_above_the_floor(self, state):
        op = _tone(**self.WIDE)
        injected = _injected(op, state)[0]
        mask = np.asarray(op(state).aux[PROTECTED_KEY])
        expected = injected >= op.protect_floor * injected.max()
        assert list(mask) == list(expected)
        # a sigma of one channel with a 1% floor reaches +/-3 channels
        assert list(np.nonzero(mask)[0]) == [1, 2, 3, 4, 5, 6, 7]

    def test_lowering_the_floor_protects_more(self, state):
        counts = [
            int(np.asarray(_tone(**self.WIDE, protect_floor=f)(state).aux[PROTECTED_KEY]).sum())
            for f in (0.5, 1e-2, 1e-4)
        ]
        assert counts[0] < counts[1] < counts[2]

    def test_the_protected_set_is_contiguous_around_the_line(self, state):
        indices = np.nonzero(np.asarray(_tone(**self.WIDE)(state).aux[PROTECTED_KEY]))[0]
        assert list(indices) == list(range(indices[0], indices[-1] + 1))

    def test_a_line_between_two_channels_protects_both(self, state):
        """The tie case, and the reason the rule is ``>=`` a level rather than
        "the best N channels": ranking would keep one of two exactly equal
        channels and hand the other to the flagger."""
        mask = np.asarray(
            _tone(tone_freq=TONE_FREQ + 0.5 * CHANNEL, protect_floor=1.0)(state)
            .aux[PROTECTED_KEY]
        )
        assert list(np.nonzero(mask)[0]) == [TONE_CHANNEL, TONE_CHANNEL + 1]

    def test_how_much_of_the_tone_the_protected_set_actually_holds(self, state):
        """Stated as a measurement, because it is the number that matters:
        under-protect and this drops, and the calibrator goes with it."""
        op = _tone(**self.WIDE)
        injected = _injected(op, state)[0]
        mask = np.asarray(op(state).aux[PROTECTED_KEY])
        assert injected[mask].sum() / injected.sum() > 0.999


class TestADriftingToneProtectsADriftingSetOfChannels:
    def test_the_mask_is_a_waterfall_and_it_moves(self, state):
        mask = np.asarray(
            _tone(drift_rate=DRIFT_PER_CHANNEL)(state).aux[PROTECTED_KEY]
        )
        assert mask.shape == (N_TIME, N_FREQ)
        assert [list(np.nonzero(row)[0]) for row in mask] == [[4], [5], [6], [7]]

    def test_each_sample_is_measured_against_its_own_peak(self, state):
        """Half a channel per sample: the line alternates between sitting on a
        channel and sitting between two, so the samples have genuinely
        different peak levels (1.0 and 0.42 of the total). A protection floor
        taken against the run's global peak instead of each sample's own gives
        counts [1, 0, 1, 0] — measured: on the two samples where the line sits
        between channels it protects NOTHING at all, and hands both wet
        channels to the flagger."""
        mask = np.asarray(
            _tone(drift_rate=0.5 * DRIFT_PER_CHANNEL, protect_floor=0.9)(state)
            .aux[PROTECTED_KEY]
        )
        assert [int(row.sum()) for row in mask] == [1, 2, 1, 2]
        assert [list(np.nonzero(row)[0]) for row in mask] == [[4], [4, 5], [5], [5, 6]]

    def test_a_static_mask_would_have_protected_the_wrong_channels(self, state):
        """What the 1-D mask costs a drifting tone, measured: three of the four
        samples would have had their tone channel handed to the flagger."""
        moving = np.asarray(_tone(drift_rate=DRIFT_PER_CHANNEL)(state).aux[PROTECTED_KEY])
        frozen = np.broadcast_to(moving[0], moving.shape)
        assert (moving & frozen).sum() == 1
        assert moving.sum() == N_TIME

    def test_a_drifting_mask_composes_with_a_static_one(self, state):
        """A second, non-drifting tone must not lose its channel, and must not
        flatten the first one's waterfall back to a channel mask."""
        both = Pipeline(
            _tone(drift_rate=DRIFT_PER_CHANNEL),
            _tone(tone_freq=float(FREQ[0])),
        )(state)
        mask = np.asarray(both.aux[PROTECTED_KEY])
        assert mask.shape == (N_TIME, N_FREQ)
        assert [list(np.nonzero(row)[0]) for row in mask] == [
            [0, 4], [0, 5], [0, 6], [0, 7]
        ]


# ------------------------------------------------------ the aux contract now


class TestProtectRefusesAMaskItCannotCompose:
    """``protect`` writes; ``unflag_protected`` reads. A mask that cannot
    compose has to be refused at the WRITE, where the operator that built it is
    still on the stack — a flagger three stages later cannot say who wrote it,
    and a pipeline with no flagger never notices at all."""

    def test_a_scalar_mask_is_refused_at_the_write(self):
        with pytest.raises(StateValidationError, match="ndim=0"):
            protect({}, jnp.array(True))

    def test_a_three_dimensional_mask_is_refused_at_the_write(self):
        with pytest.raises(StateValidationError, match="ndim=3"):
            protect({}, jnp.ones((1, N_TIME, N_FREQ), dtype=bool))

    def test_a_mask_from_a_different_band_will_not_compose(self):
        first = jnp.zeros(N_FREQ, dtype=bool).at[TONE_CHANNEL].set(True)
        with pytest.raises(StateValidationError, match="channels"):
            protect(protect({}, first), jnp.ones(N_FREQ + 1, dtype=bool))

    def test_two_waterfalls_of_different_lengths_will_not_compose(self):
        already = protect({}, jnp.zeros((N_TIME, N_FREQ), dtype=bool))
        with pytest.raises(StateValidationError, match="time samples"):
            protect(already, jnp.ones((N_TIME + 1, N_FREQ), dtype=bool))

    def test_a_channel_mask_and_a_waterfall_compose_to_a_waterfall(self):
        channel = jnp.zeros(N_FREQ, dtype=bool).at[0].set(True)
        water = jnp.zeros((N_TIME, N_FREQ), dtype=bool).at[2, TONE_CHANNEL].set(True)
        combined = np.asarray(protect(protect({}, channel), water)[PROTECTED_KEY])
        assert combined.shape == (N_TIME, N_FREQ)
        assert [list(np.nonzero(row)[0]) for row in combined] == [
            [0], [0], [0, TONE_CHANNEL], [0]
        ]


class TestWhatAWideToneActuallyMeasures:
    """The consequence for calibration, which is the point of the operator.

    A delta on one channel probes ``b(nu_cw) * g(t)`` — one bandpass value.
    A line with width probes ``sum_k w_k b(nu_k) * g(t)``: a LINESHAPE-WEIGHTED
    AVERAGE of the bandpass over the line's wings. On a curved bandpass those
    are different numbers, and reading the second as the first biases the
    bandpass estimate at the tone's channel by the curvature times the line's
    second moment. Pinned here so nobody re-derives it from the placeholder.
    """

    # Curved, and asymmetric about the tone channel, so a weighted average
    # cannot coincide with the centre value by symmetry.
    BANDPASS = jnp.array(
        [0.60, 0.68, 0.78, 0.85, 0.90, 0.98, 1.10, 1.26, 1.30, 1.24, 1.10]
    )
    GAIN = jnp.array([1.5, 1.6, 1.7, 1.8])

    def _response(self, state, op):
        chain = [ReceiverOperator(bandpass=self.BANDPASS), GainOperator(gain=self.GAIN)]
        with_tone = Pipeline(op, *chain)(state).data
        without = Pipeline(*chain)(state).data
        return np.asarray(with_tone - without)

    def test_a_delta_tone_probes_the_bandpass_at_its_own_channel(self, state):
        """The placeholder's claim, still exactly true for an on-centre sinc2."""
        response = self._response(state, _tone())[:, TONE_CHANNEL]
        expected = TONE_KELVIN * float(self.BANDPASS[TONE_CHANNEL]) * np.asarray(self.GAIN)
        assert np.allclose(response, expected, rtol=1e-5)

    def test_a_wide_tone_probes_a_weighted_average_of_the_bandpass(self, state):
        op = _tone(lineshape="gaussian", line_width=1.5 * CHANNEL)
        weights = _injected(op, state)[0] / TONE_KELVIN
        total = self._response(state, op).sum(axis=-1)
        expected = TONE_KELVIN * float(weights @ np.asarray(self.BANDPASS)) * np.asarray(
            self.GAIN
        )
        assert np.allclose(total, expected, rtol=1e-4)
        # ... and it is NOT the bandpass at the tone's own channel: on this
        # curve the weighted average sits 2.37% high (0.9213 against 0.9000),
        # in float32 and float64 alike, and that is the bias.
        centre = TONE_KELVIN * float(self.BANDPASS[TONE_CHANNEL]) * np.asarray(self.GAIN)
        assert not np.allclose(total, centre, rtol=1e-2)
        assert 1.020 < float(total[0] / centre[0]) < 1.030


# --------------------------------------------------------------- the honesty


class TestWideningTheLineCostsTheToneItsLeverage:
    """The claim the docstring makes, measured rather than asserted.

    A known tone buys nothing against a free-per-cell ``T_ant``; it earns its
    keep only against a frequency-SMOOTH one, and then only because a narrow
    line is not in the span of the smooth basis. Widen the line and it moves
    INTO that span, so the very realism added here reduces what the tone buys.
    That is a real cost of modelling the line honestly, and it is stated so
    nobody reads "now with a lineshape" as "now more informative".
    """

    @staticmethod
    def _residual_outside_a_smooth_basis(width, degree=4):
        """Fraction of the tone's profile a degree-``degree`` polynomial in
        frequency cannot represent."""
        op = _tone(lineshape="gaussian", line_width=width)
        nu = np.asarray(FREQ)
        profile = np.asarray(
            op(
                State(
                    data=jnp.zeros((1, N_FREQ)),
                    coords=Coordinates(freq=FREQ),
                    meta={"obs_id": "profile"},
                )
            ).data
        )[0]
        x = (nu - nu.mean()) / (nu.max() - nu.mean())
        basis = np.vander(x, degree + 1)
        fit = basis @ np.linalg.lstsq(basis, profile, rcond=None)[0]
        return float(np.linalg.norm(profile - fit) / np.linalg.norm(profile))

    def test_a_wider_line_is_more_absorbable_by_a_smooth_basis(self):
        """Measured, in channel widths: 0.84, 0.75, 0.43, 0.21, 0.092, 0.038.

        The sweep stops at 2.5 channels because that is the widest line this
        band admits at all (``MAX_WIDTH_IN_BAND_FRACTION`` x 10 MHz) — past it
        the injection is a pedestal, not a line, and the operator refuses it.
        The trend is already an order of magnitude inside the legal range.
        """
        residuals = [
            self._residual_outside_a_smooth_basis(w * CHANNEL)
            for w in (0.25, 0.5, 1.0, 1.5, 2.0, 2.5)
        ]
        assert residuals == sorted(residuals, reverse=True), residuals
        # a quarter-channel line is almost entirely outside the smooth basis;
        # the widest legal line is mostly inside it — 22x less to gain.
        assert residuals[0] > 0.8
        assert residuals[-1] < 0.05
        assert residuals[0] / residuals[-1] > 20


class TestItStillCompilesAndStillCarriesNoParameters:
    def test_the_new_settings_are_all_static(self):
        """``tree_leaves(op)`` UNFILTERED, which is what "static" actually means.

        ``eqx.filter(op, eqx.is_inexact_array)`` filters on ARRAYS, and every
        one of these fields has already been coerced to a Python float or str
        by its converter — so that assertion holds whether or not ``static=True``
        is there, and it held for five separate mutations that each removed one.
        The distinction is real and this is where it shows: with ``static=True``
        the leaves are ``[]`` and ``jax.grad`` returns ``[]``; without it the
        field is a leaf and ``jax.grad`` raises ``ConcretizationTypeError``.
        """
        op = _tone(lineshape="gaussian", drift_rate=1.0, amplitude_drift_rate=1e-4)
        assert jax.tree_util.tree_leaves(op) == []
        assert jax.tree_util.tree_leaves(eqx.filter(op, eqx.is_inexact_array)) == []
        hash(jax.tree_util.tree_structure(op))

    def test_a_drifting_tone_runs_under_jit(self, state):
        op = _tone(drift_rate=DRIFT_PER_CHANNEL, amplitude_drift_rate=-1e-4)
        run = eqx.filter_jit(lambda o, s: o(s))
        assert np.allclose(np.asarray(run(op, state).data), np.asarray(op(state).data),
                           rtol=1e-5)
