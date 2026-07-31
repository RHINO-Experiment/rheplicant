"""The MomentRFI flagging bridge, the masked likelihood, and the route between.

Flagging only matters because of what it changes downstream, so the last class
here does not check a mask — it checks that routing a real flagger's output
through :class:`~rheplicant.inference.noise.FlaggedNoise` removes a bias that
is measurably present without it.
"""

import importlib.util

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rheplicant import State
from rheplicant.core.errors import StateValidationError
from rheplicant.inference import (
    FlaggedNoise,
    GaussianLikelihood,
    MaskedGaussianLikelihood,
    NoiseModelLikelihood,
    RadiometerNoise,
)
from rheplicant.radio import MomentRFIFlaggingOperator

needs_momentrfi = pytest.mark.skipif(
    importlib.util.find_spec("MomentRFI") is None, reason="MomentRFI not installed"
)

N_TIME, N_FREQ = 64, 32
FREQ = jnp.linspace(60e6, 90e6, N_FREQ)
CHANNEL_WIDTH = float(FREQ[1] - FREQ[0])
INTEGRATION_TIME = 1.0
AMP_TRUE, SPECTRAL_INDEX = 300.0, 2.5
RFI_CHANNELS = (7, 21)


def _spectrum(amp):
    return amp * (FREQ / 70e6) ** -SPECTRAL_INDEX


def _waterfall(key, rfi_boost=0.0, rfi_channels=RFI_CHANNELS):
    """A smooth power-law sky under radiometer noise, optionally contaminated."""
    noise = RadiometerNoise(CHANNEL_WIDTH, INTEGRATION_TIME)
    clean = jnp.broadcast_to(_spectrum(AMP_TRUE), (N_TIME, N_FREQ))
    data = clean * (1.0 + noise.fractional * jax.random.normal(key, clean.shape))
    for channel in rfi_channels:
        data = data.at[:, channel].multiply(1.0 + rfi_boost)
    return data, noise


class TestMaskedGaussianLikelihood:
    def test_matches_unmasked_when_flags_none(self):
        pred, obs = jnp.zeros(4), jnp.array([1.0, -1.0, 0.5, 2.0])
        masked = MaskedGaussianLikelihood(noise_std=jnp.array(1.5))
        plain = GaussianLikelihood(noise_std=jnp.array(1.5))
        assert jnp.allclose(masked(pred, obs), plain(pred, obs))

    def test_flagged_samples_do_not_contribute(self):
        pred = jnp.zeros(4)
        obs = jnp.array([1.0, -1.0, 0.5, 1e6])  # huge RFI-like outlier at [3]
        flags = jnp.array([False, False, False, True])
        lik = MaskedGaussianLikelihood(noise_std=jnp.array(1.0), flags=flags)
        clean = GaussianLikelihood(noise_std=jnp.array(1.0))
        assert jnp.allclose(lik(pred, obs), clean(pred[:3], obs[:3]))

    def test_gradient_blocked_on_flagged(self):
        obs = jnp.array([1.0, 2.0])
        flags = jnp.array([False, True])
        lik = MaskedGaussianLikelihood(noise_std=jnp.array(1.0), flags=flags)
        g = jax.grad(lambda p: lik(p, obs))(jnp.zeros(2))
        assert g[0] != 0.0 and g[1] == 0.0


@needs_momentrfi
class TestMomentRFIFlaggingOperator:
    @pytest.fixture
    def spiky_state(self):
        key = jax.random.key(0)
        base = 100.0 + jax.random.normal(key, (64, 32))
        spiked = base.at[10, 5].set(1e5).at[40, 20].set(1e5)
        return State(data=spiked)

    def test_flags_obvious_spikes(self, spiky_state):
        out = MomentRFIFlaggingOperator()(spiky_state)
        flags = out.aux["flags"]
        assert flags.dtype == jnp.bool_ and flags.shape == spiky_state.data.shape
        assert bool(flags[10, 5]) and bool(flags[40, 20])
        assert flags.mean() < 0.2  # not everything flagged

    def test_prior_flags_are_preserved(self, spiky_state):
        prior = jnp.zeros(spiky_state.data.shape, dtype=bool).at[0, 0].set(True)
        out = MomentRFIFlaggingOperator()(spiky_state.replace(aux={"flags": prior}))
        assert bool(out.aux["flags"][0, 0])

    def test_rejects_non_2d(self):
        with pytest.raises(StateValidationError, match="2D"):
            MomentRFIFlaggingOperator()(State(data=jnp.ones(8)))

    def test_it_is_jittable(self, spiky_state):
        """``jax.pure_callback`` is the permanent integration, not a stopgap:
        a boolean decision has no gradient, but it must still survive jit."""
        eager = MomentRFIFlaggingOperator()(spiky_state).aux["flags"]
        jitted = jax.jit(
            lambda d: MomentRFIFlaggingOperator()(State(data=d)).aux["flags"]
        )(spiky_state.data)
        assert jnp.array_equal(eager, jitted)

    def test_config_reaches_the_fitter(self, spiky_state):
        loose = MomentRFIFlaggingOperator(config={"sigma_threshold": 8.0})
        tight = MomentRFIFlaggingOperator(config={"sigma_threshold": 3.0})
        assert (
            loose(spiky_state).aux["flags"].sum()
            < tight(spiky_state).aux["flags"].sum()
        )


@needs_momentrfi
class TestBroadRounds:
    """``kernel_shapes`` buys the matched filter's sqrt(K), not a tighter cut.

    A spatially continuous emitter adds ~linearly under a box kernel while
    thermal noise adds in quadrature, so averaging K pixels lifts it by
    sqrt(K). Below the per-pixel threshold that is the difference between
    seeing nothing at all and seeing all of it.
    """

    PATCH = (slice(20, 28), slice(10, 18))

    @pytest.fixture
    def faint_blob(self):
        data, noise = _waterfall(jax.random.key(1), rfi_boost=0.0)
        # 3 sigma per pixel: under the fitter's default 4 sigma cut.
        contaminated = data.at[self.PATCH].multiply(1.0 + 3.0 * noise.fractional)
        truth = np.zeros((N_TIME, N_FREQ), bool)
        truth[self.PATCH] = True
        return contaminated, truth

    def test_round_zero_alone_cannot_see_it(self, faint_blob):
        contaminated, truth = faint_blob
        flags = np.asarray(MomentRFIFlaggingOperator()(State(data=contaminated)).aux["flags"])
        assert flags[truth].mean() < 0.1

    def test_a_box_kernel_recovers_it(self, faint_blob):
        contaminated, truth = faint_blob
        flags = np.asarray(
            MomentRFIFlaggingOperator(kernel_shapes=((3, 3),))(
                State(data=contaminated)
            ).aux["flags"]
        )
        assert flags[truth].mean() > 0.9
        assert flags[~truth].mean() < 0.02  # and does not flag the sky to do it


@needs_momentrfi
class TestFlagsReachTheNoiseCovariance:
    """The point of flagging, stated as the bias it removes.

    A persistent narrow-band emitter on 2 of 32 channels pulls a
    maximum-likelihood amplitude high. The flags are not applied to the data —
    they are wrapped around the noise model, which is where a sample that was
    not observed belongs.
    """

    @pytest.fixture
    def contaminated(self):
        return _waterfall(jax.random.key(0), rfi_boost=0.6)

    @staticmethod
    def _ml_amplitude(data, noise):
        """Argmax over a grid — no optimizer, so nothing to tune or diagnose."""
        like = NoiseModelLikelihood(noise)
        amps = jnp.linspace(0.9 * AMP_TRUE, 1.3 * AMP_TRUE, 2001)
        logp = jax.vmap(
            lambda a: like(jnp.broadcast_to(_spectrum(a), data.shape), data)
        )(amps)
        return amps[jnp.argmax(logp)]

    def test_unflagged_the_rfi_biases_the_fit(self, contaminated):
        data, noise = contaminated
        assert self._ml_amplitude(data, noise) > 1.02 * AMP_TRUE

    def test_flagged_it_recovers_the_truth(self, contaminated):
        data, noise = contaminated
        flags = MomentRFIFlaggingOperator()(State(data=data)).aux["flags"]
        recovered = self._ml_amplitude(data, FlaggedNoise(noise, flags))
        assert jnp.allclose(recovered, AMP_TRUE, rtol=2e-3)

    def test_it_matches_what_perfect_knowledge_would_have_given(self, contaminated):
        """The flagger is not merely *better* — on this contamination it is as
        good as knowing exactly which channels were hit."""
        data, noise = contaminated
        found = MomentRFIFlaggingOperator()(State(data=data)).aux["flags"]
        oracle = jnp.zeros(data.shape, bool).at[:, list(RFI_CHANNELS)].set(True)
        assert jnp.allclose(
            self._ml_amplitude(data, FlaggedNoise(noise, found)),
            self._ml_amplitude(data, FlaggedNoise(noise, oracle)),
            rtol=1e-6,
        )
