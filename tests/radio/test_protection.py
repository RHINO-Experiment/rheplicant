"""Protected channels, and both flaggers honouring them.

The headline measurement is at the bottom: with no protection, MomentRFI flags
the CW tone's channel at fraction 1.0 — correctly, by its own lights, since a
narrow persistent spike is what RFI looks like — and flagging sits downstream
of ``cw_tone`` on the same trunk. The pipeline that is meant to use the
calibrator destroys it on the first observation.
"""

import importlib.util

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
    FlaggingOperator,
    MomentRFIFlaggingOperator,
    protect,
    unflag_protected,
)

needs_momentrfi = pytest.mark.skipif(
    importlib.util.find_spec("MomentRFI") is None, reason="MomentRFI not installed"
)

# Non-square, and the tone channel is not the middle one.
N_TIME, N_FREQ = 6, 5
FREQ = jnp.linspace(60e6, 85e6, N_FREQ)
TONE_CHANNEL, RFI_CHANNEL = 2, 4
TONE_KELVIN = 5000.0


@pytest.fixture
def state():
    data = jnp.full((N_TIME, N_FREQ), 10.0)
    # A genuine RFI burst in a DIFFERENT channel and only some samples, so a
    # flagger that simply stopped flagging would be caught.
    data = data.at[1:3, RFI_CHANNEL].set(9e3)
    return State(
        data=data,
        coords=Coordinates(time=jnp.arange(N_TIME, dtype=float), freq=FREQ),
        meta={"telescope": "RHINO", "obs_id": "protect-000"},
    )


def _tone():
    return CWCalibrationOperator(
        amplitude=TONE_KELVIN, tone_freq=float(FREQ[TONE_CHANNEL])
    )


class TestTheAuxContract:
    def test_protect_writes_the_mask_when_there_is_none(self):
        mask = jnp.array([False, True, False, False, False])
        assert jnp.array_equal(protect({}, mask)[PROTECTED_KEY], mask)

    def test_protect_composes_with_an_existing_mask(self):
        first = jnp.array([True, False, False, False, False])
        second = jnp.array([False, False, True, False, False])
        combined = protect(protect({}, first), second)[PROTECTED_KEY]
        assert [bool(x) for x in combined] == [True, False, True, False, False]

    def test_protect_leaves_the_rest_of_aux_alone(self):
        aux = {"flags": jnp.zeros((N_TIME, N_FREQ), dtype=bool)}
        out = protect(aux, jnp.zeros(N_FREQ, dtype=bool))
        assert "flags" in out and out is not aux

    def test_no_protection_declared_leaves_flags_untouched(self):
        flags = jnp.zeros((N_TIME, N_FREQ), dtype=bool).at[0, 0].set(True)
        assert unflag_protected(flags, {}) is flags

    def test_a_channel_mask_clears_that_channel_in_every_sample(self):
        flags = jnp.ones((N_TIME, N_FREQ), dtype=bool)
        mask = jnp.zeros(N_FREQ, dtype=bool).at[TONE_CHANNEL].set(True)
        out = unflag_protected(flags, {PROTECTED_KEY: mask})
        assert not bool(out[:, TONE_CHANNEL].any())
        assert bool(out[:, RFI_CHANNEL].all())

    def test_a_waterfall_mask_clears_only_the_samples_it_names(self):
        """A switched calibrator protects its channel only while it is on."""
        flags = jnp.ones((N_TIME, N_FREQ), dtype=bool)
        mask = jnp.zeros((N_TIME, N_FREQ), dtype=bool).at[1:3, TONE_CHANNEL].set(True)
        out = unflag_protected(flags, {PROTECTED_KEY: mask})
        assert [bool(x) for x in out[:, TONE_CHANNEL]] == [
            True, False, False, True, True, True
        ]

    def test_an_integer_mask_is_read_as_a_mask(self):
        """Nonzero means protected, and the flags stay BOOLEAN.

        Both halves are needed. Without the cast, ``flags & ~mask`` on a 0/1
        int mask gives numerically the right answer by accident — through
        bitwise-not on integers — and returns int32 flags, which every
        downstream consumer of ``aux["flags"]`` takes for a boolean mask. And
        an accumulated count of 2 comes back UNPROTECTED, because ``~2`` is
        ``-3`` and ``1 & -3`` is 1.
        """
        flags = jnp.ones((N_TIME, N_FREQ), dtype=bool)
        for value in (1, 2):
            mask = jnp.zeros(N_FREQ, dtype=jnp.int32).at[TONE_CHANNEL].set(value)
            out = unflag_protected(flags, {PROTECTED_KEY: mask})
            assert out.dtype == jnp.bool_, value
            assert not bool(out[:, TONE_CHANNEL].any()), value
            assert bool(out[:, RFI_CHANNEL].all()), value

    def test_a_scalar_mask_is_refused(self):
        with pytest.raises(StateValidationError, match="ndim=0"):
            unflag_protected(jnp.ones((N_TIME, N_FREQ), dtype=bool),
                             {PROTECTED_KEY: jnp.array(True)})

    def test_a_three_dimensional_mask_is_refused(self):
        with pytest.raises(StateValidationError, match="ndim=3"):
            unflag_protected(jnp.ones((N_TIME, N_FREQ), dtype=bool),
                             {PROTECTED_KEY: jnp.ones((1, N_TIME, N_FREQ), dtype=bool)})

    def test_a_mask_from_a_different_band_is_refused(self):
        """It would broadcast only by accident, and protect whichever channels
        happened to line up."""
        with pytest.raises(StateValidationError, match="channels but the"):
            unflag_protected(jnp.ones((N_TIME, N_FREQ), dtype=bool),
                             {PROTECTED_KEY: jnp.ones(N_FREQ + 1, dtype=bool)})

    def test_a_transposed_waterfall_mask_is_refused(self):
        """The non-square fixture is what makes this catchable at all."""
        with pytest.raises(StateValidationError, match="channels but the"):
            unflag_protected(jnp.ones((N_TIME, N_FREQ), dtype=bool),
                             {PROTECTED_KEY: jnp.ones((N_FREQ, N_TIME), dtype=bool)})


class TestTheThresholdFlagger:
    def test_without_protection_the_tone_channel_is_flagged_everywhere(self, state):
        """What the flagger does on its own — correct, and fatal."""
        toned = state.with_data(state.data.at[:, TONE_CHANNEL].add(TONE_KELVIN))
        flags = FlaggingOperator(threshold=100.0)(toned).aux["flags"]
        assert float(flags[:, TONE_CHANNEL].mean()) == 1.0

    def test_with_protection_the_tone_survives_and_the_rfi_does_not(self, state):
        out = Pipeline(_tone(), FlaggingOperator(threshold=100.0))(state)
        flags = out.aux["flags"]
        assert float(flags[:, TONE_CHANNEL].mean()) == 0.0
        assert [bool(x) for x in flags[:, RFI_CHANNEL]] == [
            False, True, True, False, False, False
        ]

    def test_the_flagger_is_unchanged_where_nothing_declared_protection(self, state):
        flags = FlaggingOperator(threshold=100.0)(state).aux["flags"]
        assert int(flags.sum()) == 2  # the two RFI samples, nothing else


# --------------------------------------------------------------- MomentRFI --

MRFI_TIME, MRFI_FREQ = 48, 24
MRFI_FREQ_AXIS = jnp.linspace(60e6, 90e6, MRFI_FREQ)
MRFI_TONE_CHANNEL = 7


def _waterfall() -> jnp.ndarray:
    spectrum = 300.0 * (MRFI_FREQ_AXIS / 70e6) ** -2.5
    drift = 1.0 + 0.02 * jnp.arange(MRFI_TIME, dtype=float)[:, None] / MRFI_TIME
    noise = 0.01 * jax.random.normal(
        jax.random.key(11), (MRFI_TIME, MRFI_FREQ)
    ) * spectrum[None, :]
    return spectrum[None, :] * drift + noise


@needs_momentrfi
class TestMomentRFI:
    """The measured headline: fraction 1.0 -> 0.0 on the tone's channel."""

    FITTER = {"sigma_threshold": 4.0}
    KERNELS = ((3, 3), (1, 5))

    def _flagger(self):
        return MomentRFIFlaggingOperator(config=self.FITTER, kernel_shapes=self.KERNELS)

    def _state(self):
        return State(
            data=_waterfall(),
            coords=Coordinates(
                time=jnp.arange(MRFI_TIME, dtype=float), freq=MRFI_FREQ_AXIS
            ),
            meta={"telescope": "RHINO", "obs_id": "momentrfi-protect"},
        )

    def test_the_tone_channel_goes_from_fully_flagged_to_fully_kept(self):
        tone = CWCalibrationOperator(
            amplitude=5000.0, tone_freq=float(MRFI_FREQ_AXIS[MRFI_TONE_CHANNEL])
        )
        toned = tone(self._state())

        # Same data, same flagger: only the protection differs.
        unprotected = toned.replace(aux={})
        without = self._flagger()(unprotected).aux["flags"]
        with_protection = self._flagger()(toned).aux["flags"]

        assert float(np.asarray(without)[:, MRFI_TONE_CHANNEL].mean()) == 1.0
        assert float(np.asarray(with_protection)[:, MRFI_TONE_CHANNEL].mean()) == 0.0

    def test_protection_does_not_stop_the_flagger_flagging_real_rfi(self):
        """The mechanism must not be a global off switch."""
        rfi_channel = 17
        state = self._state()
        state = state.with_data(state.data.at[10:14, rfi_channel].multiply(6.0))
        tone = CWCalibrationOperator(
            amplitude=5000.0, tone_freq=float(MRFI_FREQ_AXIS[MRFI_TONE_CHANNEL])
        )
        flags = np.asarray(self._flagger()(tone(state)).aux["flags"])
        assert flags[10:14, rfi_channel].all()
        assert not flags[:, MRFI_TONE_CHANNEL].any()
