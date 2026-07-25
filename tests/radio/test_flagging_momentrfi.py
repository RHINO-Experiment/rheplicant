"""Tests for the MomentRFI flagging bridge and the masked likelihood."""

import importlib.util

import jax
import jax.numpy as jnp
import pytest

from rheplicant import State
from rheplicant.core.errors import StateValidationError
from rheplicant.inference import GaussianLikelihood, MaskedGaussianLikelihood
from rheplicant.radio import MomentRFIFlaggingOperator


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


@pytest.mark.skipif(
    importlib.util.find_spec("MomentRFI") is None, reason="MomentRFI not installed"
)
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
