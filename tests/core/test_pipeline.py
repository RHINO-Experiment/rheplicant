"""Tests for the Pipeline composite operator."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from replicant.core.errors import PipelineError
from replicant.core.operator import AbstractOperator, LambdaOperator
from replicant.core.pipeline import Pipeline
from replicant.core.state import State


class AddOne(AbstractOperator):
    def __call__(self, state: State) -> State:
        return state.with_data(state.data + 1)


class Double(AbstractOperator):
    def __call__(self, state: State) -> State:
        return state.with_data(state.data * 2)


@pytest.fixture
def state():
    return State(data=jnp.array([1.0, 2.0]))


class TestConstruction:
    def test_empty_rejected(self):
        with pytest.raises(PipelineError, match="at least one"):
            Pipeline()

    def test_non_operator_rejected(self):
        with pytest.raises(PipelineError, match="AbstractOperator"):
            Pipeline(lambda s: s)  # bare function: must be wrapped in LambdaOperator

    def test_auto_names_strip_operator_suffix(self):
        pipe = Pipeline(AddOne(), Double())
        assert pipe.names == ("addone", "double")

    def test_auto_names_deduplicate(self):
        pipe = Pipeline(AddOne(), AddOne(), AddOne())
        assert pipe.names == ("addone", "addone_2", "addone_3")

    def test_explicit_names(self):
        pipe = Pipeline(AddOne(), Double(), names=("inc", "dbl"))
        assert pipe.names == ("inc", "dbl")

    def test_wrong_name_count_rejected(self):
        with pytest.raises(PipelineError, match="names"):
            Pipeline(AddOne(), Double(), names=("only-one",))

    def test_duplicate_names_rejected(self):
        with pytest.raises(PipelineError, match="unique"):
            Pipeline(AddOne(), Double(), names=("x", "x"))


class TestExecution:
    def test_application_order(self, state):
        """(x+1)*2 != x*2+1 — order must be first-to-last."""
        out = Pipeline(AddOne(), Double())(state)
        assert jnp.array_equal(out.data, jnp.array([4.0, 6.0]))
        out = Pipeline(Double(), AddOne())(state)
        assert jnp.array_equal(out.data, jnp.array([3.0, 5.0]))

    def test_nested_equals_flat(self, state):
        nested = Pipeline(Pipeline(AddOne(), Double()), AddOne())
        flat = Pipeline(AddOne(), Double(), AddOne())
        assert jnp.array_equal(nested(state).data, flat(state).data)

    def test_run_with_intermediates(self, state):
        pipe = Pipeline(AddOne(), Double())
        final, intermediates = pipe.run_with_intermediates(state)
        assert len(intermediates) == 2
        assert jnp.array_equal(intermediates[0].data, jnp.array([2.0, 3.0]))
        assert jnp.array_equal(intermediates[1].data, final.data)


class TestAccess:
    def test_getitem_by_index_and_name(self):
        pipe = Pipeline(AddOne(), Double(), names=("inc", "dbl"))
        assert isinstance(pipe[0], AddOne)
        assert isinstance(pipe["dbl"], Double)

    def test_unknown_name_raises(self):
        pipe = Pipeline(AddOne(), names=("inc",))
        with pytest.raises(KeyError, match="inc"):
            pipe["missing"]

    def test_len_and_iter(self):
        pipe = Pipeline(AddOne(), Double())
        assert len(pipe) == 2
        assert [type(s).__name__ for s in pipe] == ["AddOne", "Double"]


class TestFunctionalUpdates:
    def test_replace_stage_by_name(self, state):
        pipe = Pipeline(AddOne(), Double(), names=("inc", "dbl"))
        pipe2 = pipe.replace_stage("inc", Double())
        assert jnp.array_equal(pipe2(state).data, jnp.array([4.0, 8.0]))  # (x*2)*2
        assert isinstance(pipe["inc"], AddOne)  # original untouched
        assert pipe2.names == pipe.names

    def test_replace_stage_by_index(self):
        pipe = Pipeline(AddOne(), Double())
        pipe2 = pipe.replace_stage(1, AddOne())
        assert isinstance(pipe2[1], AddOne)

    def test_replace_stage_rejects_non_operator(self):
        pipe = Pipeline(AddOne())
        with pytest.raises(PipelineError, match="AbstractOperator"):
            pipe.replace_stage(0, lambda s: s)


class TestJaxTransforms:
    def test_pipeline_is_an_operator(self):
        assert isinstance(Pipeline(AddOne()), AbstractOperator)

    def test_jit(self, state):
        pipe = Pipeline(AddOne(), Double())
        out = eqx.filter_jit(pipe)(state)
        assert jnp.array_equal(out.data, jnp.array([4.0, 6.0]))

    def test_vmap_over_data_batch(self):
        pipe = Pipeline(AddOne(), LambdaOperator.on_data(lambda d: d * 2))
        batch = jnp.arange(6.0).reshape(3, 2)
        out = jax.vmap(lambda d: pipe(State(data=d)).data)(batch)
        assert out.shape == (3, 2)
        assert jnp.array_equal(out, (batch + 1) * 2)

    def test_grad_through_pipeline_params(self, state):
        class Gain(AbstractOperator):
            g: jax.Array

            def __call__(self, s):
                return s.with_data(s.data * self.g)

        pipe = Pipeline(AddOne(), Gain(g=jnp.array(3.0)))

        def loss(pipe):
            return jnp.sum(pipe(state).data)

        grads = eqx.filter_grad(loss)(pipe)
        # d/dg sum((x+1)*g) = sum(x+1) = 2+3 = 5
        assert jnp.allclose(grads.stages[1].g, 5.0)
