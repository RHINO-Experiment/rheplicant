import jax.numpy as jnp


def test_session_runs_in_float64():
    assert jnp.zeros(1).dtype == jnp.float64
