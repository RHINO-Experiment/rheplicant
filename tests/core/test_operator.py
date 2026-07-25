"""Tests for AbstractOperator and LambdaOperator."""

from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from replicant.core.operator import AbstractOperator, LambdaOperator
from replicant.core.state import State


class Scaler(AbstractOperator):
    """Minimal concrete operator used throughout these tests."""

    requires: ClassVar[tuple[str, ...]] = ("data",)
    provides: ClassVar[tuple[str, ...]] = ("data",)

    factor: jax.Array

    def __call__(self, state: State) -> State:
        return state.with_data(state.data * self.factor)


class TestAbstractOperator:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            AbstractOperator()

    def test_concrete_subclass_works(self):
        op = Scaler(factor=jnp.array(2.0))
        out = op(State(data=jnp.arange(3.0)))
        assert jnp.array_equal(out.data, jnp.array([0.0, 2.0, 4.0]))

    def test_default_contract_is_empty(self):
        assert LambdaOperator(fn=lambda s: s).requires == ()
        assert LambdaOperator(fn=lambda s: s).provides == ()

    def test_declared_contract(self):
        assert Scaler(factor=jnp.array(1.0)).requires == ("data",)
        assert Scaler(factor=jnp.array(1.0)).provides == ("data",)

    def test_params_are_filterable_leaves(self):
        """Differentiable params must be exactly the inexact-array fields."""
        op = Scaler(factor=jnp.array(2.0))
        params = eqx.filter(op, eqx.is_inexact_array)
        leaves = jax.tree.leaves(params)
        assert len(leaves) == 1
        assert jnp.array_equal(leaves[0], jnp.array(2.0))

    def test_grad_wrt_params(self):
        op = Scaler(factor=jnp.array(2.0))
        s = State(data=jnp.array([3.0]))

        def loss(op):
            return jnp.sum(op(s).data)

        g = eqx.filter_grad(loss)(op)
        assert jnp.allclose(g.factor, 3.0)  # d(3*f)/df = 3


class TestLambdaOperator:
    def test_wraps_state_function(self):
        op = LambdaOperator(fn=lambda s: s.with_data(s.data + 1))
        out = op(State(data=jnp.zeros(2)))
        assert jnp.array_equal(out.data, jnp.ones(2))

    def test_on_data_lifts_array_function(self):
        op = LambdaOperator.on_data(jnp.exp)
        out = op(State(data=jnp.zeros(3)))
        assert jnp.array_equal(out.data, jnp.ones(3))

    def test_is_an_operator(self):
        assert isinstance(LambdaOperator(fn=lambda s: s), AbstractOperator)

    def test_jit_compatible(self):
        op = LambdaOperator.on_data(lambda d: d * 2)
        out = eqx.filter_jit(op)(State(data=jnp.arange(3.0)))
        assert jnp.array_equal(out.data, jnp.array([0.0, 2.0, 4.0]))
