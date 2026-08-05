"""The axis a drift is measured against, and what a drifting mask leaves behind.

Giving the tone a drift made ``coords.time`` load-bearing for the first time in
this package: every other ``requires=("coords.time", ...)`` uses it for a shape
or hands it to limTOD, and this operator is the only place that does ARITHMETIC
on its values. Two consequences got their own failures, and neither of them is
about the lineshape:

* ``coords.time`` is stored through ``jnp.asarray`` — float32 unless x64 is on —
  so a unix-second axis is quantised before ``t - t[0]`` ever runs, and the
  anchor cannot recover what the store threw away. The injection and the band
  guard read the SAME corrupted elapsed values, so the guard cannot see it;
* the protection mask became time-indexed, and a stage that reshapes the run
  (``BackendOperator``) updates ``data`` and ``coords.time`` together and leaves
  ``aux`` alone, so the mask is left describing an axis that no longer exists.

The time axes here deliberately do NOT all start at zero, which is the fixture
symmetry that hid the anchor in the first place.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from rheplicant import Coordinates, State
from rheplicant.core.errors import StateValidationError
from rheplicant.core.pipeline import Pipeline
from rheplicant.radio import (
    PROTECTED_KEY,
    BackendOperator,
    CWCalibrationOperator,
    unflag_protected,
)

# Same 4 x 11 grid as tests/radio/test_cw_lineshape.py: non-square, and the
# tone channel (4) is not the middle one (5).
N_TIME, N_FREQ = 4, 11
CHANNEL = 1e6
FREQ = 60e6 + CHANNEL * jnp.arange(N_FREQ, dtype=float)
TONE_CHANNEL = 4
TONE_FREQ = float(FREQ[TONE_CHANNEL])
TONE_KELVIN = 5000.0
TIME = 100.0 * jnp.arange(N_TIME, dtype=float)      # 0, 100, 200, 300 s
DRIFT_PER_CHANNEL = CHANNEL / 100.0                 # one channel per sample


@pytest.fixture
def state():
    return State(
        data=jnp.full((N_TIME, N_FREQ), 10.0),
        coords=Coordinates(time=TIME, freq=FREQ),
        meta={"telescope": "RHINO", "obs_id": "cw-time-000"},
    )


def _tone(**kwargs):
    settings = {
        "amplitude": TONE_KELVIN,
        "tone_freq": TONE_FREQ,
        "line_width": CHANNEL,
    }
    return CWCalibrationOperator(**{**settings, **kwargs})


def _injected(op, state):
    return np.asarray(op(state).data - state.data)


class TestTheTimeAxisMustBeAbleToExpressItsOwnCadence:
    """The drift is only as good as the axis it is measured against.

    Measured before any guard existed, on this grid with ``drift_rate`` 1e4
    Hz/s — one channel per 100 s sample:

        coords.time            elapsed              peak ch     protected/11
        [0,100,200,300]        [0,100,200,300]      [4,5,6,7]   1, 1, 1, 1
        1.75e9 + the same      [0,128,256,256]      [4,5,7,7]   1, 6, 8, 8

    Two of the four samples collapse onto the same time, the tone lands in the
    wrong channel, and the mask blows out from one channel to eight of eleven —
    this operator's own named silent failure, protecting so much that genuine
    RFI survives. Nothing raised and nothing was NaN; under ``JAX_ENABLE_X64=1``
    the same run gives ``[4,5,6,7]``, so the cause is precision, not logic.

    **A position stated here has since been overturned, deliberately.** This
    class used to assert that a *static* tone on a unix-second axis is fine and
    must be accepted, on the grounds that a tone which never subtracts anything
    is not harmed by a coarse axis. That is true of this operator and false of
    the axis: the same 8-sample axis handed to ``BackendOperator`` produces
    chunk timestamps wrong by up to 78 s of a 100 s cadence, silently, and
    ``Coordinates`` cannot know which consumer comes next. The refusal
    therefore moved to ``Coordinates.__check_init__``, where the rounding
    actually happens, and the axis no longer reaches any operator — see
    ``tests/core/test_coordinates.py::TestATimeAxisTheStoredDtypeCannotCarry``.
    What survives here is what is genuinely this operator's own: the check runs
    only when the tone DRIFTS, and it counts a zero gap that the container
    deliberately allows.
    """

    def _at(self, state, anchor, cadence=100.0):
        times = anchor + cadence * jnp.arange(N_TIME, dtype=float)
        return state.replace(coords=state.coords.replace(time=times))

    def test_a_unix_second_axis_never_reaches_the_operator_any_more(self, state):
        """Refused at the store, with the remedies this operator used to name."""
        with pytest.raises(StateValidationError) as excinfo:
            _tone(drift_rate=DRIFT_PER_CHANNEL)(self._at(state, 1.75e9))
        message = str(excinfo.value)
        assert "read_rhino_observation" in message      # where the axis comes from
        assert "JAX_ENABLE_X64" in message              # one of the two remedies
        assert "start of the run" in message            # the other

    def test_the_same_run_measured_from_its_own_start_is_accepted(self, state):
        """The remedy, run: subtract the epoch BEFORE the axis is stored."""
        injected = _injected(_tone(drift_rate=DRIFT_PER_CHANNEL), self._at(state, 0.0))
        assert list(injected.argmax(axis=-1)) == [4, 5, 6, 7]

    def test_a_collided_axis_is_refused_only_when_the_tone_actually_drifts(self, state):
        """Both branches of the clause this operator still owns alone.

        ``[0, 0, 100, 250]`` is stored exactly in float32 and has a repeated
        timestamp, which ``Coordinates`` accepts on purpose — it cannot tell a
        genuine repeat from a collision. This operator can act on it, because it
        SUBTRACTS times: across a zero gap the tone does not move, so a drift
        measured there is silently wrong. Gaps 0/100/150, so a check reading the
        mean (83) or the last gap rather than the minimum would pass this axis.
        """
        collided = state.replace(
            coords=state.coords.replace(time=jnp.array([0.0, 0.0, 100.0, 250.0]))
        )
        with pytest.raises(StateValidationError, match="representable"):
            _tone(drift_rate=DRIFT_PER_CHANNEL)(collided)

        # The other branch: a tone that does not move never subtracts anything,
        # so the same axis costs it nothing and must be accepted.
        out = _tone()(collided)
        assert np.isclose(
            float(np.asarray(out.data)[0, TONE_CHANNEL]), 10.0 + TONE_KELVIN, rtol=1e-5
        )

    def test_the_threshold_is_pinned_from_both_sides(self):
        """The operator's own copy of the ratio, evaluated directly.

        Routed around ``Coordinates`` on purpose: the container now refuses the
        99 s axis first, so going through a State would pin the container's
        threshold and say nothing about this one. Both methods must agree at the
        boundary, and they are only comparable when each is called on its own.

        At an anchor of 1e7 s the float32 grid is exactly 1 s, so the ratio
        against the cadence is 1/cadence and the threshold (1e-2) sits between
        two cadences two percent apart. Both axes are stored exactly and are
        otherwise identical — only the ratio moves across the cut.

        This is also the pair that separates ``np.spacing(times.max())`` from
        ``np.spacing(float(times.max()))``: the latter answers for float64 no
        matter what the array holds, and would pass BOTH sides.
        """
        refuse = CWCalibrationOperator._refuse_a_time_axis_it_cannot_resolve
        stored = lambda cadence: np.asarray(  # noqa: E731 - one expression, read inline
            jnp.asarray(1.0e7 + cadence * np.arange(N_TIME, dtype=float))
        )

        assert refuse(stored(101.0)) is None
        with pytest.raises(StateValidationError, match="representable"):
            refuse(stored(99.0))

    def test_a_single_sample_run_has_no_cadence_to_compare_against(self):
        """One sample has no interval, so the check steps aside rather than
        taking the minimum of an empty array — and its elapsed time is exactly
        zero, so there is nothing for the precision to spoil."""
        one = State(
            data=jnp.full((1, N_FREQ), 10.0),
            coords=Coordinates(time=jnp.array([1.75e9]), freq=FREQ),
            meta={"obs_id": "one-sample"},
        )
        injected = _injected(_tone(drift_rate=DRIFT_PER_CHANNEL), one)
        assert list(injected.argmax(axis=-1)) == [TONE_CHANNEL]

    def test_a_descending_time_axis_is_measured_by_the_size_of_its_gaps(self):
        """A reversed axis has negative gaps, and a negative cadence makes the
        comparison pass anything. It is the MAGNITUDE of the interval that the
        precision has to resolve, so this axis is fine and must be accepted."""
        down = State(
            data=jnp.full((N_TIME, N_FREQ), 10.0),
            coords=Coordinates(time=TIME[::-1], freq=FREQ),
            meta={"obs_id": "reversed"},
        )
        injected = _injected(_tone(drift_rate=DRIFT_PER_CHANNEL), down)
        assert list(injected.argmax(axis=-1)) == [4, 3, 2, 1]

    def test_an_axis_anchored_on_a_future_epoch_is_refused_too(self, state):
        """``np.spacing`` of a NEGATIVE number is negative, and a negative
        resolution compares below any positive threshold. The magnitude is what
        the dtype has to carry, whichever side of the epoch it is on."""
        with pytest.raises(StateValidationError, match="representable"):
            _tone(drift_rate=DRIFT_PER_CHANNEL)(self._at(state, -1.75e9))


class TestADriftingMaskThroughAShapeChangingStage:
    """What a TIME-INDEXED mask costs that a channel mask never did.

    ``aux[PROTECTED_KEY]`` became a waterfall when the tone learned to drift,
    and the stages that reshape a run — ``BackendOperator`` averaging into
    chunks, say — used to update ``data`` and ``coords.time`` together and leave
    ``aux`` alone, so row ``i`` of the mask named the channels the tone wet at
    sample ``i`` of an axis that no longer existed. Two failures were measured
    on this fixture at ``n_chunk=2``, in this order: first a raw ``TypeError``
    out of ``&`` ("incompatible shapes for broadcasting: (2, 11), (4, 11)"),
    then — once ``unflag_protected`` grew its own guard — a branded refusal
    naming the staleness. The mask now comes across the average instead, and
    the guard below stays as the backstop for a mask that arrives from
    somewhere this operator never saw.
    """

    def test_averaging_after_a_drifting_tone_brings_the_mask_with_it(self, state):
        """One channel per 100 s sample, chunks of two, so the reduced mask
        protects a PAIR of channels per chunk — the union of what the tone wet
        while the chunk was integrating, which is what the chunk mean contains.
        """
        out = Pipeline(
            _tone(drift_rate=DRIFT_PER_CHANNEL), BackendOperator(n_chunk=2)
        )(state)
        assert out.data.shape == (N_TIME // 2, N_FREQ)
        mask = np.asarray(out.aux[PROTECTED_KEY])
        assert mask.shape == (N_TIME // 2, N_FREQ)
        assert [np.flatnonzero(row).tolist() for row in mask] == [[4, 5], [6, 7]]

        kept = np.asarray(unflag_protected(jnp.ones(out.data.shape, dtype=bool), out.aux))
        assert [np.flatnonzero(~row).tolist() for row in kept] == [[4, 5], [6, 7]]

    def test_a_mask_from_an_axis_nobody_reduced_is_still_refused(self, state):
        """The backstop, unchanged: a waterfall mask that reached the flags
        without passing through the stage that reshaped them."""
        stale = jnp.ones((N_TIME, N_FREQ), dtype=bool)
        with pytest.raises(StateValidationError, match="stale"):
            unflag_protected(
                jnp.zeros((N_TIME // 2, N_FREQ), dtype=bool), {PROTECTED_KEY: stale}
            )

    def test_a_static_tone_survives_the_same_stage(self, state):
        """The contrast, and the reason this is not a blanket refusal: a
        ``(n_freq,)`` channel mask is not bound to any time axis, so the same
        pipeline with a tone that does not drift is still fine."""
        out = Pipeline(_tone(), BackendOperator(n_chunk=2))(state)
        kept = np.asarray(unflag_protected(jnp.ones(out.data.shape, dtype=bool), out.aux))
        assert not kept[:, TONE_CHANNEL].any()
        assert kept[:, TONE_CHANNEL + 2].all()

    def test_a_single_row_waterfall_is_refused_rather_than_broadcast(self):
        """The dangerous shape, and the worse of the two: one row broadcasts
        over every sample, so a mask left behind by a single-chunk average
        unflagged the WHOLE run — measured at 0 of 44 flags left, silently."""
        with pytest.raises(StateValidationError, match="over 1 time samples"):
            unflag_protected(
                jnp.ones((N_TIME, N_FREQ), dtype=bool),
                {PROTECTED_KEY: jnp.ones((1, N_FREQ), dtype=bool)},
            )
