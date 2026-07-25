"""Shared fixtures: a small (n_time=8, n_freq=4) observation context."""

import jax
import jax.numpy as jnp
import pytest

from replicant import Coordinates, Environment, State

N_TIME = 8
N_FREQ = 4


@pytest.fixture
def coords():
    return Coordinates(
        time=jnp.linspace(0.0, 7.0, N_TIME),
        freq=jnp.linspace(60e6, 85e6, N_FREQ),
    )


@pytest.fixture
def template_state(coords):
    """A seeded, data-less state ready to be pushed through a forward pipeline."""
    return State(
        coords=coords,
        env=Environment(temperature=jnp.array(280.0)),
        key=jax.random.key(0),
        meta={"telescope": "RHINO", "obs_id": "test-000"},
    )
