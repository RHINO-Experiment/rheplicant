"""Tests for State: the immutable pytree container at the heart of rheplicant."""

import jax
import jax.numpy as jnp
import pytest

from rheplicant.core.coordinates import Coordinates
from rheplicant.core.environment import Environment
from rheplicant.core.errors import MissingKeyError, StateValidationError
from rheplicant.core.frozen import FrozenMapping
from rheplicant.core.state import State


@pytest.fixture
def state():
    return State(
        data=jnp.ones((4, 3)),
        coords=Coordinates(time=jnp.arange(4.0), freq=jnp.arange(3.0)),
        env=Environment(temperature=jnp.array(280.0)),
        aux={"weights": jnp.ones((4, 3))},
        key=jax.random.key(0),
        meta={"telescope": "RHINO", "obs_id": "demo-001"},
    )


class TestConstruction:
    def test_empty_state(self):
        s = State()
        assert s.data is None and s.coords is None and s.env is None
        assert s.aux == {} and s.key is None
        assert isinstance(s.meta, FrozenMapping) and len(s.meta) == 0

    def test_full_state(self, state):
        assert state.data.shape == (4, 3)
        assert state.meta["telescope"] == "RHINO"

    def test_meta_converter(self):
        """Plain dicts are converted to FrozenMapping at the boundary."""
        s = State(meta={"a": 1})
        assert isinstance(s.meta, FrozenMapping)

    def test_rejects_raw_uint32_key(self):
        with pytest.raises(StateValidationError, match="typed PRNG key"):
            State(key=jax.random.PRNGKey(0))  # old-style raw key

    def test_rejects_non_key_array(self):
        with pytest.raises(StateValidationError, match="typed PRNG key"):
            State(key=jnp.zeros(2))

    def test_rejects_non_str_aux_keys(self):
        with pytest.raises(StateValidationError, match="aux keys"):
            State(aux={1: jnp.zeros(2)})

    def test_rejects_wrong_container_types(self):
        with pytest.raises(StateValidationError, match="coords"):
            State(coords={"time": jnp.arange(4.0)})
        with pytest.raises(StateValidationError, match="env"):
            State(env={"temperature": 280.0})


class TestImmutability:
    def test_attribute_assignment_raises(self, state):
        with pytest.raises(AttributeError):
            state.data = jnp.zeros(3)

    def test_replace_returns_new_leaves_original(self, state):
        s2 = state.replace(data=jnp.zeros((4, 3)))
        assert jnp.all(s2.data == 0)
        assert jnp.all(state.data == 1)  # original untouched

    def test_replace_reruns_converters(self, state):
        s2 = state.replace(meta={"new": "meta"})
        assert isinstance(s2.meta, FrozenMapping)

    def test_replace_reruns_validation(self, state):
        with pytest.raises(StateValidationError):
            state.replace(key=jnp.zeros(2))

    def test_with_data(self, state):
        s2 = state.with_data(jnp.zeros((4, 3)))
        assert jnp.all(s2.data == 0)
        assert s2.meta == state.meta and s2.coords is state.coords


class TestPRNGProtocol:
    def test_next_key_advances(self, state):
        subkey, s2 = state.next_key()
        assert not jnp.array_equal(
            jax.random.key_data(s2.key), jax.random.key_data(state.key)
        )
        assert not jnp.array_equal(
            jax.random.key_data(subkey), jax.random.key_data(state.key)
        )

    def test_next_key_deterministic(self):
        k1, _ = State(key=jax.random.key(42)).next_key()
        k2, _ = State(key=jax.random.key(42)).next_key()
        assert jnp.array_equal(jax.random.key_data(k1), jax.random.key_data(k2))

    def test_successive_subkeys_differ(self, state):
        k1, s2 = state.next_key()
        k2, _ = s2.next_key()
        assert not jnp.array_equal(jax.random.key_data(k1), jax.random.key_data(k2))

    def test_missing_key_raises(self):
        with pytest.raises(MissingKeyError, match="jax.random.key"):
            State().next_key()


class TestPytreeBehavior:
    def test_flatten_roundtrip_preserves_meta(self, state):
        leaves, treedef = jax.tree.flatten(state)
        s2 = jax.tree.unflatten(treedef, leaves)
        assert s2.meta == state.meta
        assert jnp.array_equal(s2.data, state.data)

    def test_meta_is_static_not_a_leaf(self, state):
        """meta must live in the treedef, not among traced leaves."""
        leaves = jax.tree.leaves(state)
        assert not any(isinstance(leaf, FrozenMapping) for leaf in leaves)

    def test_jit_caches_on_same_meta(self):
        traces = []

        @jax.jit
        def f(s):
            traces.append(1)
            return s.data

        f(State(data=jnp.ones(3), meta={"a": 1}))
        f(State(data=jnp.zeros(3), meta={"a": 1}))  # same structure: cached
        assert len(traces) == 1

        f(State(data=jnp.ones(3), meta={"a": 2}))  # meta changed: retrace
        assert len(traces) == 2

    def test_grad_flows_through_state(self):
        def loss(s: State) -> jax.Array:
            return jnp.sum(s.data**2)

        s = State(data=jnp.arange(3.0))
        g = jax.grad(loss)(s)
        assert jnp.array_equal(g.data, 2 * jnp.arange(3.0))
