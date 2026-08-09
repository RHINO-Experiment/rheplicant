"""MapSky: fixed brightness maps, and the grid they were built on.

The class every example and doc page used to declare by hand. The addition
over those copies is ``freq``: a map evaluated on a grid it was not built for
returns a smooth, plausible, wrong temperature, and the shape guard is the
only part of that failure a static check can see.
"""

import jax
import jax.numpy as jnp
import pytest

from rheplicant.core.errors import StateValidationError
from rheplicant.radio import MapSky

N_FREQ, N_PIX = 4, 12


@pytest.fixture
def grid():
    return jnp.linspace(60e6, 85e6, N_FREQ)


@pytest.fixture
def maps():
    return jnp.arange(N_FREQ * N_PIX, dtype=jnp.float32).reshape(N_FREQ, N_PIX)


def test_call_returns_the_maps_unchanged(grid, maps):
    sky = MapSky(maps=maps, freq=grid)
    assert jnp.array_equal(sky(grid), maps)


def test_call_ignores_its_freq_argument_when_the_shape_agrees(grid, maps):
    """Documented behaviour: the argument is not consulted, only shape-checked."""
    sky = MapSky(maps=maps, freq=grid)
    other = grid + 1e6          # same length, different values
    assert jnp.array_equal(sky(other), maps)


def test_call_refuses_a_grid_of_a_different_length(grid, maps):
    sky = MapSky(maps=maps, freq=grid)
    with pytest.raises(StateValidationError, match="built on a grid of 4"):
        sky(jnp.linspace(60e6, 85e6, N_FREQ + 1))


def test_refuses_maps_that_are_not_two_dimensional(grid):
    with pytest.raises(StateValidationError, match="maps must be"):
        MapSky(maps=jnp.zeros(N_PIX), freq=grid)


def test_refuses_a_freq_axis_that_is_not_one_dimensional(maps):
    with pytest.raises(StateValidationError, match="freq must be"):
        MapSky(maps=maps, freq=jnp.zeros((N_FREQ, 1)))


def test_refuses_a_maps_freq_length_mismatch(maps):
    with pytest.raises(StateValidationError, match="disagree"):
        MapSky(maps=maps, freq=jnp.linspace(60e6, 85e6, N_FREQ + 1))


def test_maps_are_a_differentiable_leaf(grid, maps):
    """A map can be inferred, not merely assumed — so it must carry gradient."""
    sky = MapSky(maps=maps, freq=grid)
    grad = jax.grad(lambda s: s(grid).sum())(sky)
    assert jnp.array_equal(grad.maps, jnp.ones_like(maps))


def test_survives_jit(grid, maps):
    import equinox as eqx

    sky = MapSky(maps=maps, freq=grid)
    out = eqx.filter_jit(lambda s, f: s(f))(sky, grid)
    assert jnp.array_equal(out, maps)
