"""Tests for the Coordinates and Environment containers."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rheplicant.core.coordinates import Coordinates
from rheplicant.core.environment import Environment
from rheplicant.core.errors import StateValidationError


class TestCoordinatesConstruction:
    def test_all_optional(self):
        c = Coordinates()
        assert c.time is None and c.freq is None and c.pointing is None
        assert c.extra == {}

    def test_basic(self):
        c = Coordinates(time=jnp.arange(4.0), freq=jnp.arange(3.0))
        assert c.time.shape == (4,)
        assert c.freq.shape == (3,)

    def test_converter_accepts_numpy_and_lists(self):
        c = Coordinates(time=np.arange(4.0), freq=[1.0, 2.0])
        assert isinstance(c.time, jax.Array)
        assert isinstance(c.freq, jax.Array)

    def test_pointing_shape(self):
        c = Coordinates(pointing=jnp.zeros((5, 2)))
        assert c.pointing.shape == (5, 2)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"time": jnp.zeros((2, 2))},  # time must be 1D
            {"freq": jnp.zeros((2, 2))},  # freq must be 1D
            {"pointing": jnp.zeros(5)},  # pointing must be 2D
        ],
    )
    def test_structural_validation(self, kwargs):
        with pytest.raises(StateValidationError):
            Coordinates(**kwargs)


class TestCoordinatesFunctional:
    def test_replace_returns_new(self):
        c = Coordinates(time=jnp.arange(4.0))
        c2 = c.replace(freq=jnp.arange(3.0))
        assert c2.freq is not None and c.freq is None
        assert jnp.array_equal(c2.time, c.time)

    def test_replace_reruns_validation(self):
        c = Coordinates()
        with pytest.raises(StateValidationError):
            c.replace(time=jnp.zeros((2, 2)))

    def test_extra_is_traced(self):
        """Arrays in `extra` must appear as pytree leaves (differentiable/vmappable)."""
        c = Coordinates(time=jnp.arange(4.0), extra={"az": jnp.zeros(4)})
        leaves = jax.tree.leaves(c)
        assert len(leaves) == 2  # time + extra["az"]

    def test_is_pytree_roundtrip(self):
        c = Coordinates(time=jnp.arange(4.0), extra={"az": jnp.zeros(4)})
        leaves, treedef = jax.tree.flatten(c)
        c2 = jax.tree.unflatten(treedef, leaves)
        assert jnp.array_equal(c2.time, c.time)


class TestEnvironment:
    def test_all_optional(self):
        e = Environment()
        assert e.temperature is None and e.humidity is None and e.extra == {}

    def test_basic_and_converter(self):
        e = Environment(temperature=np.array([280.0, 281.0]), humidity=[0.4, 0.5])
        assert isinstance(e.temperature, jax.Array)
        assert isinstance(e.humidity, jax.Array)

    def test_replace_returns_new(self):
        e = Environment(temperature=jnp.array(280.0))
        e2 = e.replace(humidity=jnp.array(0.5))
        assert e.humidity is None and e2.humidity is not None

    def test_extra_is_traced(self):
        e = Environment(extra={"wind_speed": jnp.zeros(3)})
        assert len(jax.tree.leaves(e)) == 1
