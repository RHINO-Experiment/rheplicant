"""What ``BackendOperator`` does to ``state.aux`` when it reshapes the run.

Averaging changes the length of the time axis. ``data`` and ``coords.time``
were always updated together; ``aux`` was not, and the three cases measured on
the ``(6, 4)`` fixture below with ``n_chunk=3`` failed with three very different
amounts of noise:

===========================  ====================================================
``aux`` entry                before
===========================  ====================================================
``aux["flags"]``             carried at ``(6, 4)``; ``FlaggedNoise.std`` refused
                             two stages later — "flags shape (6, 4) does not
                             match the prediction shape (2, 4)"
``aux["protected"]``, 2-D    carried at ``(6, 4)``; the next ``FlaggingOperator``
                             refused it, naming the staleness
``aux["switch"]``, 1-D       carried at ``(6,)``, **no error anywhere**, and each
                             output chunk spanned both switch positions
===========================  ====================================================

The third is the defect: the first two are loud only because something
downstream knows what shape those keys are supposed to have. A key the package
has never heard of has no such consumer.

Fixture shape discipline: ``n_time=6``, ``n_freq=4``, ``n_chunk=3``,
``n_out=2`` — four distinct numbers, so a reduction along the wrong axis, a
transposed reshape, or a chunking off by one all produce a shape that differs
from the right one rather than one that happens to match.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rheplicant import Coordinates, State
from rheplicant.core.errors import StateValidationError
from rheplicant.core.operator import SnapshotOperator
from rheplicant.core.pipeline import Pipeline
from rheplicant.inference.noise import FlaggedNoise, HomoscedasticNoise
from rheplicant.radio import (
    PROTECTED_KEY,
    BackendOperator,
    FlaggingOperator,
    unflag_protected,
)
from rheplicant.radio.backend.averaging import FLAGS_KEY, SNAPSHOT_PREFIX
from rheplicant.radio.protection import reduce_protection

N_TIME, N_FREQ, N_CHUNK = 6, 4, 3
N_OUT = N_TIME // N_CHUNK
assert len({N_TIME, N_FREQ, N_CHUNK, N_OUT}) == 4, "the fixture must stay asymmetric"


def make_state(aux=None, n_time=N_TIME):
    return State(
        data=jnp.arange(n_time * N_FREQ, dtype=float).reshape(n_time, N_FREQ),
        coords=Coordinates(
            time=jnp.arange(n_time, dtype=float),
            freq=jnp.linspace(60e6, 85e6, N_FREQ),
        ),
        aux={} if aux is None else aux,
        meta={"telescope": "RHINO", "obs_id": "backend-aux"},
    )


@pytest.fixture
def state():
    return make_state()


def averaged(aux, n_chunk=N_CHUNK):
    return BackendOperator(n_chunk=n_chunk)(make_state(aux))


# A single flagged sample in the SECOND chunk and the THIRD channel: a reduction
# that reversed the chunk order, transposed the reshape, or ran along the
# frequency axis all give a different answer from this one.
FLAGGED_TIME, FLAGGED_FREQ = 4, 2


def one_flag(n_time=N_TIME):
    return jnp.zeros((n_time, N_FREQ), dtype=bool).at[FLAGGED_TIME, FLAGGED_FREQ].set(True)


class TestTheDeclarations:
    """``provides`` is a contract, and this operator now writes two aux paths."""

    def test_it_declares_the_aux_paths_it_writes(self):
        assert f"aux.{FLAGS_KEY}" in BackendOperator.provides
        assert f"aux.{PROTECTED_KEY}" in BackendOperator.provides

    def test_it_declares_reading_them_too(self):
        """It reduces what is there rather than creating it — 'reads if present',
        the same sense ``GroundPickupOperator`` uses for ``env.temperature``."""
        assert f"aux.{FLAGS_KEY}" in BackendOperator.requires
        assert f"aux.{PROTECTED_KEY}" in BackendOperator.requires

    def test_the_axis_declarations_did_not_go_away(self):
        for path in ("data", "coords.time"):
            assert path in BackendOperator.requires and path in BackendOperator.provides


class TestFlagsReachTheChunkAxis:
    def test_the_shape_follows_the_data(self, state):
        out = BackendOperator(n_chunk=N_CHUNK)(state.replace(aux={FLAGS_KEY: one_flag()}))
        assert out.data.shape == (N_OUT, N_FREQ)
        assert out.aux[FLAGS_KEY].shape == (N_OUT, N_FREQ)

    def test_a_chunk_with_one_flagged_sample_is_flagged(self):
        """``any``, and the reason: this placeholder averages every sample in the
        chunk, flagged ones included, so one bad sample contaminates the mean."""
        flags = np.asarray(averaged({FLAGS_KEY: one_flag()}).aux[FLAGS_KEY])
        assert flags[:, FLAGGED_FREQ].tolist() == [False, True]
        assert not flags[:, FLAGGED_FREQ - 1].any()

    def test_all_would_have_given_the_other_answer(self):
        """The contrast pinned, so a change from ``any`` to ``all`` is deliberate:
        ``all`` calls the contaminated chunk clean and carries the RFI forward."""
        reduced = one_flag().reshape(N_OUT, N_CHUNK, N_FREQ)
        assert np.asarray(reduced.all(axis=1))[:, FLAGGED_FREQ].tolist() == [False, False]
        assert np.asarray(reduced.any(axis=1))[:, FLAGGED_FREQ].tolist() == [False, True]

    def test_the_result_is_boolean(self):
        """Every consumer of ``aux['flags']`` takes it for a boolean mask; an
        accumulated integer count reduces to a mask rather than to a number."""
        counts = jnp.zeros((N_TIME, N_FREQ), dtype=jnp.int32).at[FLAGGED_TIME, 2].set(2)
        out = averaged({FLAGS_KEY: counts}).aux[FLAGS_KEY]
        assert out.dtype == jnp.bool_
        assert np.asarray(out)[:, 2].tolist() == [False, True]

    def test_a_nan_sample_flags_its_chunk(self):
        """``nan > x`` is False, so a comparison-based reduction would let the
        contaminated chunk through as clean. This one reads truthiness."""
        flags = jnp.zeros((N_TIME, N_FREQ)).at[FLAGGED_TIME, FLAGGED_FREQ].set(jnp.nan)
        out = np.asarray(averaged({FLAGS_KEY: flags}).aux[FLAGS_KEY])
        assert out[:, FLAGGED_FREQ].tolist() == [False, True]

    def test_flagged_noise_now_accepts_what_averaging_produced(self, state):
        """The downstream that used to refuse: 'flags shape (6, 4) does not
        match the prediction shape (2, 4)'."""
        out = BackendOperator(n_chunk=N_CHUNK)(state.replace(aux={FLAGS_KEY: one_flag()}))
        sigma = FlaggedNoise(
            HomoscedasticNoise(sigma=jnp.array(1.0)), out.aux[FLAGS_KEY]
        ).std(out.data)
        assert sigma.shape == (N_OUT, N_FREQ)
        assert bool(jnp.isinf(sigma[1, FLAGGED_FREQ]))
        assert int(jnp.isinf(sigma).sum()) == 1

    def test_flags_on_some_other_axis_are_left_for_their_own_consumer(self):
        """A ``flags`` array that was not on the time axis to begin with was
        already broken before this stage; it is carried, and ``FlaggedNoise``
        refuses it as it always did. Not silently fixed, not doubly refused."""
        bad = jnp.zeros((N_FREQ, N_TIME), dtype=bool)
        out = averaged({FLAGS_KEY: bad})
        assert out.aux[FLAGS_KEY].shape == (N_FREQ, N_TIME)
        with pytest.raises(StateValidationError, match="does not match the prediction"):
            FlaggedNoise(HomoscedasticNoise(sigma=jnp.array(1.0)), out.aux[FLAGS_KEY]).std(
                out.data
            )


class TestProtectionReachesTheChunkAxis:
    def test_a_waterfall_mask_is_reduced(self):
        mask = jnp.zeros((N_TIME, N_FREQ), dtype=bool).at[FLAGGED_TIME, FLAGGED_FREQ].set(True)
        out = np.asarray(averaged({PROTECTED_KEY: mask}).aux[PROTECTED_KEY])
        assert out.shape == (N_OUT, N_FREQ)
        assert out[:, FLAGGED_FREQ].tolist() == [False, True]

    def test_a_channel_mask_is_carried_unchanged(self):
        """The contrast, and the reason this is not a blanket reduction: a
        ``(n_freq,)`` mask names channels, so no change to the time axis
        stales it. ``n_freq != n_time`` here is what makes that checkable."""
        mask = jnp.zeros(N_FREQ, dtype=bool).at[FLAGGED_FREQ].set(True)
        out = averaged({PROTECTED_KEY: mask})
        assert out.aux[PROTECTED_KEY].shape == (N_FREQ,)
        assert jnp.array_equal(out.aux[PROTECTED_KEY], mask)

    def test_the_flagger_downstream_now_accepts_the_mask(self, state):
        """Before: 'aux['protected'] is a waterfall mask over 6 time samples but
        the flags cover 2 ... has left this one stale'."""
        mask = jnp.zeros((N_TIME, N_FREQ), dtype=bool).at[:, FLAGGED_FREQ].set(True)
        out = Pipeline(
            BackendOperator(n_chunk=N_CHUNK), FlaggingOperator(threshold=-1.0)
        )(state.replace(aux={PROTECTED_KEY: mask}))
        flags = np.asarray(out.aux[FLAGS_KEY])
        assert flags.shape == (N_OUT, N_FREQ)
        assert not flags[:, FLAGGED_FREQ].any()  # protected, all chunks
        assert flags[:, FLAGGED_FREQ - 1].all()  # everything else over threshold

    def test_a_partly_on_calibrator_protects_every_chunk_it_touched(self):
        """``any`` again, and for the same reason: the chunk mean carries the
        tone's power from whichever sample it was on for."""
        mask = jnp.zeros((N_TIME, N_FREQ), dtype=bool).at[2, FLAGGED_FREQ].set(True)
        out = np.asarray(averaged({PROTECTED_KEY: mask}).aux[PROTECTED_KEY])
        assert out[:, FLAGGED_FREQ].tolist() == [True, False]


class TestReduceProtection:
    """The re-derivation ``unflag_protected`` tells the caller to perform."""

    def test_a_waterfall_reduces_with_any(self):
        mask = jnp.zeros((N_TIME, N_FREQ), dtype=bool).at[1, 3].set(True)
        out = np.asarray(reduce_protection(mask, N_CHUNK))
        assert out.shape == (N_OUT, N_FREQ)
        assert out[:, 3].tolist() == [True, False]

    def test_a_channel_mask_comes_back_as_it_went_in(self):
        mask = jnp.zeros(N_FREQ, dtype=bool).at[1].set(True)
        assert jnp.array_equal(reduce_protection(mask, N_CHUNK), mask)

    @pytest.mark.parametrize("mask", [jnp.array(True), jnp.ones((1, N_TIME, N_FREQ), bool)])
    def test_a_mask_that_is_neither_shape_is_refused(self, mask):
        with pytest.raises(StateValidationError, match="channel mask or a"):
            reduce_protection(mask, N_CHUNK)

    def test_an_indivisible_time_axis_is_refused(self):
        """A partial chunk has no honest reduction, and padding one would
        protect samples that were never observed."""
        with pytest.raises(StateValidationError, match="not divisible"):
            reduce_protection(jnp.ones((N_TIME, N_FREQ), dtype=bool), 4)

    def test_the_reduction_is_over_time_and_not_over_frequency(self):
        """``(6, 4) -> (2, 4)``; a reduction over the other axis would give
        ``(6, 2)``, which only the asymmetric fixture can tell apart."""
        assert reduce_protection(jnp.ones((N_TIME, N_FREQ), dtype=bool), N_CHUNK).shape == (
            N_OUT,
            N_FREQ,
        )


class TestAnUnknownPerTimeArrayIsRefusedByName:
    """The case with nothing downstream that could know."""

    SWITCH = jnp.array([0, 1, 0, 1, 0, 1])

    def test_the_switch_index_is_refused(self):
        with pytest.raises(StateValidationError) as excinfo:
            averaged({"switch": self.SWITCH})
        message = str(excinfo.value)
        assert "aux['switch']" in message
        assert "(6,)" in message and "leading axis 6" in message
        assert "no reduction" in message

    def test_the_message_says_how_to_get_past_it(self):
        with pytest.raises(StateValidationError, match="pop it from aux"):
            averaged({"switch": self.SWITCH})

    def test_the_message_says_why_averaging_an_index_is_not_the_answer(self):
        """The register's alternative was a per-chunk constancy check. That is a
        VALUE check — it cannot run under jit, and a NaN would walk through any
        comparison built on it — so the refusal is on shape, and applies whether
        or not the chunk happens to be constant."""
        with pytest.raises(StateValidationError, match="no right value even in principle"):
            averaged({"switch": self.SWITCH})

    def test_a_constant_chunk_is_refused_just_the_same(self):
        """``[7, 7, 7, 9, 9, 9]`` reduces cleanly by eye, and is still refused:
        the alternative rule would have to read the values to know that, and
        would then hand back ``[7.0, 9.0]`` — a float where an index was."""
        with pytest.raises(StateValidationError, match="aux\\['switch'\\]"):
            averaged({"switch": jnp.array([7, 7, 7, 9, 9, 9])})

    def test_a_two_dimensional_unknown_is_refused(self):
        with pytest.raises(StateValidationError, match=r"aux\['weights'\].*\(6, 4\)"):
            averaged({"weights": jnp.ones((N_TIME, N_FREQ))})

    def test_an_all_nan_per_time_array_is_refused(self):
        """The guard reads ``shape``, never a value, so NaN cannot walk past it
        the way it walks past ``nan > x``."""
        with pytest.raises(StateValidationError, match=r"aux\['drift'\]"):
            averaged({"drift": jnp.full(N_TIME, jnp.nan)})

    def test_one_per_time_member_of_a_pytree_aux_entry_is_enough(self):
        """An aux entry holding a tuple of arrays is walked, so a per-time
        member cannot hide behind a per-frequency one."""
        with pytest.raises(StateValidationError, match=r"aux\['pair'\]"):
            averaged({"pair": (jnp.zeros(N_FREQ), jnp.zeros(N_TIME))})

    def test_the_refusal_survives_jit(self):
        """Shape is known at trace time, so this is not a guard that quietly
        stops running the moment the operator is compiled."""
        with pytest.raises(StateValidationError, match=r"aux\['switch'\]"):
            jax.jit(BackendOperator(n_chunk=N_CHUNK))(make_state({"switch": self.SWITCH}))


class TestWhatIsNotRefused:
    """The other branch of every guard above — a refusal that fired on
    everything would pass most of this file and be useless."""

    def test_an_array_on_a_different_axis_is_carried(self):
        bandpass = jnp.arange(N_FREQ, dtype=float)
        out = averaged({"bandpass": bandpass})
        assert jnp.array_equal(out.aux["bandpass"], bandpass)

    def test_a_scalar_is_carried(self):
        out = averaged({"tsys": jnp.array(35.0), "n_dishes": 4})
        assert float(out.aux["tsys"]) == 35.0 and out.aux["n_dishes"] == 4

    def test_an_empty_aux_stays_empty(self, state):
        assert BackendOperator(n_chunk=N_CHUNK)(state).aux == {}

    def test_a_snapshot_is_carried_at_the_pre_average_length(self, state):
        """Deliberate, and the one aux entry whose staleness is the point: a
        snapshot is a record of the axis that existed BEFORE the destructive
        step, which is the whole documented reason to take one."""
        out = Pipeline(SnapshotOperator(name="raw"), BackendOperator(n_chunk=N_CHUNK))(state)
        assert out.data.shape == (N_OUT, N_FREQ)
        assert out.aux[f"{SNAPSHOT_PREFIX}raw"].shape == (N_TIME, N_FREQ)
        assert jnp.array_equal(out.aux[f"{SNAPSHOT_PREFIX}raw"], state.data)

    def test_n_chunk_one_leaves_every_per_time_array_alone(self):
        """The boundary of the guard: ``n_chunk=1`` does not change the length
        of the time axis, so nothing in aux went stale and refusing an unknown
        per-time array would be a false positive on a run this operator did not
        reshape."""
        switch = jnp.array([0, 1, 0, 1, 0, 1])
        out = averaged({"switch": switch, FLAGS_KEY: one_flag()}, n_chunk=1)
        assert out.data.shape == (N_TIME, N_FREQ)
        assert jnp.array_equal(out.aux["switch"], switch)
        assert jnp.array_equal(out.aux[FLAGS_KEY], one_flag())

    def test_averaging_the_whole_run_gives_a_legitimately_single_row_mask(self):
        """The other boundary, ``n_chunk == n_time``. A single-row waterfall is
        the shape ``unflag_protected`` calls the dangerous one, because a stale
        one broadcasts over every sample and protects the whole run. Here it is
        not stale — the run really is one chunk — and the flags it meets are one
        row too, so the pair is accepted and protects exactly that row."""
        mask = jnp.zeros((N_TIME, N_FREQ), dtype=bool).at[3, FLAGGED_FREQ].set(True)
        out = averaged({PROTECTED_KEY: mask, FLAGS_KEY: one_flag()}, n_chunk=N_TIME)
        assert out.data.shape == (1, N_FREQ)
        assert out.aux[PROTECTED_KEY].shape == (1, N_FREQ)
        assert out.aux[FLAGS_KEY].shape == (1, N_FREQ)
        kept = np.asarray(unflag_protected(jnp.ones((1, N_FREQ), dtype=bool), out.aux))
        assert np.flatnonzero(~kept[0]).tolist() == [FLAGGED_FREQ]

    def test_an_indivisible_chunk_is_still_the_first_thing_reported(self):
        """Order pinned: a run this operator cannot chunk at all is a more
        fundamental complaint than one aux key it cannot reduce, and a caller
        who fixed the aux key first would just meet the other error."""
        with pytest.raises(StateValidationError, match="not divisible by n_chunk=4"):
            averaged({"switch": jnp.array([0, 1, 0, 1, 0, 1])}, n_chunk=4)

    def test_the_accepted_path_survives_jit(self):
        out = jax.jit(BackendOperator(n_chunk=N_CHUNK))(make_state({FLAGS_KEY: one_flag()}))
        assert out.aux[FLAGS_KEY].shape == (N_OUT, N_FREQ)
        assert np.asarray(out.aux[FLAGS_KEY])[:, FLAGGED_FREQ].tolist() == [False, True]

    def test_the_input_state_is_untouched(self):
        """Immutability, on the field this change started writing to."""
        aux = {FLAGS_KEY: one_flag(), PROTECTED_KEY: jnp.ones((N_TIME, N_FREQ), dtype=bool)}
        state = make_state(aux)
        BackendOperator(n_chunk=N_CHUNK)(state)
        assert state.aux[FLAGS_KEY].shape == (N_TIME, N_FREQ)
        assert state.aux[PROTECTED_KEY].shape == (N_TIME, N_FREQ)
        assert aux[FLAGS_KEY].shape == (N_TIME, N_FREQ)


class TestWhenTheLeadingAxisOnlyCoincides:
    """A deliberately SQUARE grid — the one shape every other test here avoids.

    The guard is on shape, so it cannot distinguish an array that is genuinely
    per-time from one whose leading axis merely happens to equal ``n_time``.
    That trade is made twice, in opposite directions, and both are pinned here.
    """

    SQUARE = 4
    CHUNK = 2

    def _state(self, aux):
        return State(
            data=jnp.ones((self.SQUARE, self.SQUARE)),
            coords=Coordinates(
                time=jnp.arange(self.SQUARE, dtype=float),
                freq=jnp.linspace(60e6, 85e6, self.SQUARE),
            ),
            aux=aux,
            meta={"obs_id": "square"},
        )

    def test_a_channel_mask_is_still_a_channel_mask(self):
        """``protected`` is a KNOWN key, so its own convention settles the
        ambiguity — 1-D means channels, exactly as ``unflag_protected`` reads
        it. Refusing here would refuse a legitimate square-grid run."""
        mask = jnp.zeros(self.SQUARE, dtype=bool).at[1].set(True)
        out = BackendOperator(n_chunk=self.CHUNK)(self._state({PROTECTED_KEY: mask}))
        assert out.data.shape == (self.SQUARE // self.CHUNK, self.SQUARE)
        assert jnp.array_equal(out.aux[PROTECTED_KEY], mask)

    def test_an_unknown_array_of_that_length_is_refused_anyway(self):
        """The other side of the trade, and the honest limit: for a key with no
        convention there is nothing to consult, so a per-FREQUENCY array on a
        square grid is refused too. A false refusal that names the key and says
        what to do beats a wrong-length array nothing can detect."""
        with pytest.raises(StateValidationError, match=r"aux\['bandpass'\]"):
            BackendOperator(n_chunk=self.CHUNK)(
                self._state({"bandpass": jnp.arange(self.SQUARE, dtype=float)})
            )


class TestBothMasksThroughOnePipeline:
    """Flags and protection reduced together, and still meaning what they did."""

    def test_a_protected_channel_survives_the_average_and_the_flagger(self, state):
        protected = jnp.zeros((N_TIME, N_FREQ), dtype=bool).at[:, FLAGGED_FREQ].set(True)
        out = Pipeline(
            BackendOperator(n_chunk=N_CHUNK), FlaggingOperator(threshold=10.0)
        )(state.replace(aux={PROTECTED_KEY: protected, FLAGS_KEY: one_flag()}))
        assert out.aux[PROTECTED_KEY].shape == (N_OUT, N_FREQ)
        assert out.aux[FLAGS_KEY].shape == (N_OUT, N_FREQ)
        assert not np.asarray(out.aux[FLAGS_KEY])[:, FLAGGED_FREQ].any()

    def test_the_reduced_mask_still_reads_as_a_mask(self, state):
        """``unflag_protected`` accepts the reduced waterfall on the reduced
        flags — the pair that used to fail with 'has left this one stale'."""
        protected = jnp.zeros((N_TIME, N_FREQ), dtype=bool).at[0, FLAGGED_FREQ].set(True)
        out = BackendOperator(n_chunk=N_CHUNK)(state.replace(aux={PROTECTED_KEY: protected}))
        kept = np.asarray(unflag_protected(jnp.ones(out.data.shape, dtype=bool), out.aux))
        assert kept[:, FLAGGED_FREQ].tolist() == [False, True]
