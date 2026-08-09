"""NoiseModel.realise: draw one noisy observation under this model.

The generator side of the object whose sigma the likelihood, the weights, the
Fisher matrix and the NumPyro scale all already read. Having both on one
object is what lets a caller guarantee the scatter it generated and the
scatter it assumes are the same number.
"""

import jax
import jax.numpy as jnp
import pytest

from rheplicant.inference import FlaggedNoise, HomoscedasticNoise, RadiometerNoise

N_TIME, N_FREQ = 64, 8


@pytest.fixture
def prediction():
    return jnp.full((N_TIME, N_FREQ), 100.0)


def test_homoscedastic_is_additive(prediction):
    noise = HomoscedasticNoise(sigma=jnp.array(2.0))
    drawn = noise.realise(prediction, key=jax.random.key(0))
    residual = drawn - prediction
    assert drawn.shape == prediction.shape
    assert abs(float(residual.std()) - 2.0) < 0.2


def test_radiometer_is_multiplicative(prediction):
    """d -> d(1 + f w), the form the radiometer equation actually says."""
    noise = RadiometerNoise(channel_width=1e6, integration_time=1.0)
    drawn = noise.realise(prediction, key=jax.random.key(0))
    fractional = (drawn - prediction) / prediction
    assert abs(float(fractional.std()) - noise.fractional) < 0.1 * noise.fractional


def test_radiometer_realisation_scatter_matches_its_own_std(prediction):
    """The point of the seam: what it draws and what it assumes agree."""
    noise = RadiometerNoise(channel_width=1e6, integration_time=1.0)
    drawn = noise.realise(prediction, key=jax.random.key(1))
    expected = float(noise.std(prediction).mean())
    assert abs(float((drawn - prediction).std()) - expected) < 0.1 * expected


def test_one_key_reproduces_the_draw(prediction):
    noise = HomoscedasticNoise(sigma=jnp.array(2.0))
    a = noise.realise(prediction, key=jax.random.key(7))
    b = noise.realise(prediction, key=jax.random.key(7))
    assert jnp.array_equal(a, b)


def test_different_keys_give_different_draws(prediction):
    noise = HomoscedasticNoise(sigma=jnp.array(2.0))
    a = noise.realise(prediction, key=jax.random.key(7))
    b = noise.realise(prediction, key=jax.random.key(8))
    assert not jnp.array_equal(a, b)


def test_flagged_delegates_to_the_wrapped_model(prediction):
    """Flags describe what was OBSERVED, not what the sky did. A flagged
    sample still has a true value; it is the measurement that is missing."""
    flags = jnp.zeros((N_TIME, N_FREQ), dtype=bool).at[0].set(True)
    inner = HomoscedasticNoise(sigma=jnp.array(2.0))
    wrapped = FlaggedNoise(inner, flags=flags)
    key = jax.random.key(3)
    assert jnp.array_equal(wrapped.realise(prediction, key=key),
                           inner.realise(prediction, key=key))


def test_realise_survives_jit(prediction):
    import equinox as eqx

    noise = RadiometerNoise(channel_width=1e6, integration_time=1.0)
    out = eqx.filter_jit(lambda n, p, k: n.realise(p, key=k))(
        noise, prediction, jax.random.key(0)
    )
    assert out.shape == prediction.shape
