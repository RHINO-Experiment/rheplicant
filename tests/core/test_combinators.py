"""Tests for SumOperator: parallel additive composition."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from rheplicant.core.combinators import SumOperator
from rheplicant.core.errors import PipelineError
from rheplicant.core.operator import AbstractOperator
from rheplicant.core.pipeline import Pipeline
from rheplicant.core.state import State


class Constant(AbstractOperator):
    """Source-type test operator: creates data, ignoring existing data."""

    value: jax.Array

    def __call__(self, state: State) -> State:
        return state.with_data(self.value * jnp.ones(3))


class NoisySource(AbstractOperator):
    """Source-type operator that consumes randomness."""

    def __call__(self, state: State) -> State:
        subkey, state = state.next_key()
        return state.with_data(jax.random.normal(subkey, (3,)))


class TestConstruction:
    def test_empty_rejected(self):
        with pytest.raises(PipelineError, match="at least one"):
            SumOperator()

    def test_non_operator_rejected(self):
        with pytest.raises(PipelineError, match="AbstractOperator"):
            SumOperator(lambda s: s)

    def test_names(self):
        op = SumOperator(Constant(value=jnp.array(1.0)), Constant(value=jnp.array(2.0)))
        assert op.names == ("constant", "constant_2")
        named = SumOperator(Constant(value=jnp.array(1.0)), names=("fg",))
        assert named.names == ("fg",)

    def test_access(self):
        op = SumOperator(
            Constant(value=jnp.array(1.0)),
            Constant(value=jnp.array(2.0)),
            names=("a", "b"),
        )
        assert len(op) == 2
        assert op["b"].value == 2.0
        assert op[0].value == 1.0


class TestExecution:
    def test_sums_branch_contributions(self):
        op = SumOperator(Constant(value=jnp.array(1.0)), Constant(value=jnp.array(2.0)))
        out = op(State())
        assert jnp.array_equal(out.data, jnp.full(3, 3.0))

    def test_replaces_existing_data(self):
        """SumOperator is a source combinator: input data is not accumulated."""
        op = SumOperator(Constant(value=jnp.array(1.0)))
        out = op(State(data=jnp.full(3, 100.0)))
        assert jnp.array_equal(out.data, jnp.ones(3))

    def test_preserves_context(self):
        op = SumOperator(Constant(value=jnp.array(1.0)))
        out = op(State(meta={"telescope": "demo"}))
        assert out.meta["telescope"] == "demo"

    def test_branches_can_be_pipelines(self):
        """Composite pattern: a branch may itself be a Pipeline (e.g. sky -> ionosphere)."""
        double = Pipeline(
            Constant(value=jnp.array(2.0)),
            # multiply existing data by 3 via a nested operator
            _Scale(factor=jnp.array(3.0)),
        )
        op = SumOperator(double, Constant(value=jnp.array(1.0)))
        out = op(State())
        assert jnp.array_equal(out.data, jnp.full(3, 7.0))  # 2*3 + 1


class _Scale(AbstractOperator):
    factor: jax.Array

    def __call__(self, state: State) -> State:
        return state.with_data(state.data * self.factor)


class _PyTreeSource(AbstractOperator):
    """Source producing a dict-of-streams payload (documented State.data form)."""

    value: jax.Array

    def __call__(self, state: State) -> State:
        return state.with_data({"tod": self.value * jnp.ones(3), "tone": self.value * jnp.ones(2)})


class _ShapedSource(AbstractOperator):
    n: int = eqx.field(static=True)

    def __call__(self, state: State) -> State:
        return state.with_data(jnp.ones((self.n, 2)))


class TestDataPytreeSemantics:
    """Regression tests: accumulation must be leafwise, loud on mismatch."""

    def test_dict_payloads_sum_leafwise(self):
        op = SumOperator(_PyTreeSource(value=jnp.array(1.0)), _PyTreeSource(value=jnp.array(2.0)))
        out = op(State())
        assert jnp.array_equal(out.data["tod"], jnp.full(3, 3.0))
        assert jnp.array_equal(out.data["tone"], jnp.full(2, 3.0))

    def test_tuple_payloads_sum_not_concatenate(self):
        class TupleSource(AbstractOperator):
            v: jax.Array

            def __call__(self, state):
                return state.with_data((self.v * jnp.ones(2), self.v * jnp.ones(4)))

        out = SumOperator(TupleSource(v=jnp.array(1.0)), TupleSource(v=jnp.array(2.0)))(State())
        assert isinstance(out.data, tuple) and len(out.data) == 2  # NOT a 4-tuple
        assert jnp.array_equal(out.data[0], jnp.full(2, 3.0))

    def test_shape_mismatch_raises_not_broadcasts(self):
        op = SumOperator(_ShapedSource(n=8), _ShapedSource(n=1), names=("full", "avg"))
        with pytest.raises(PipelineError, match="shape"):
            op(State())

    def test_structure_mismatch_raises(self):
        op = SumOperator(
            Constant(value=jnp.array(1.0)),
            _PyTreeSource(value=jnp.array(1.0)),
            names=("array", "dict"),
        )
        with pytest.raises(PipelineError, match="structure"):
            op(State())

    @pytest.mark.parametrize("none_first", [True, False])
    def test_none_data_branch_raises_regardless_of_order(self, none_first):
        passthrough = _Identity()
        source = Constant(value=jnp.array(1.0))
        branches = (passthrough, source) if none_first else (source, passthrough)
        op = SumOperator(*branches, names=("a", "b"))
        with pytest.raises(PipelineError, match="produced no data"):
            op(State())  # state.data is None; passthrough contributes None

    def test_branches_never_see_caller_data(self):
        """D6 enforced: branch input data is stripped to None."""
        from rheplicant.core.operator import LambdaOperator

        probe = LambdaOperator(
            fn=lambda s: s.with_data(jnp.ones(3) if s.data is None else jnp.full(3, 999.0))
        )
        out = SumOperator(probe)(State(data=jnp.full(3, 5.0)))
        assert jnp.array_equal(out.data, jnp.ones(3))


class TestSelectOperator:
    from rheplicant.core.combinators import SelectOperator

    def _state(self, switch):
        from rheplicant import Coordinates

        return State(
            coords=Coordinates(time=jnp.arange(float(len(switch))),
                               extra={"switch_state": jnp.asarray(switch)}),
        )

    def _branch(self, value):
        class TimeSource(AbstractOperator):
            v: jax.Array

            def __call__(self, state):
                n = state.coords.time.shape[0]
                return state.with_data(self.v * jnp.ones((n, 2)))

        return TimeSource(v=jnp.array(value))

    def test_selects_per_time_sample(self):
        from rheplicant.core.combinators import SelectOperator

        op = SelectOperator(self._branch(1.0), self._branch(10.0), names=("ant", "load"))
        out = op(self._state([0, 1, 0, 1]))
        assert jnp.array_equal(out.data[:, 0], jnp.array([1.0, 10.0, 1.0, 10.0]))

    def test_out_of_range_selects_nothing(self):
        from rheplicant.core.combinators import SelectOperator

        op = SelectOperator(self._branch(1.0), self._branch(10.0))
        out = op(self._state([0, 7, 1]))
        assert jnp.array_equal(out.data[:, 0], jnp.array([1.0, 0.0, 10.0]))

    def test_missing_switch_state_raises(self):
        from rheplicant import Coordinates
        from rheplicant.core.combinators import SelectOperator
        from rheplicant.core.errors import StateValidationError

        op = SelectOperator(self._branch(1.0), self._branch(2.0))
        with pytest.raises(StateValidationError, match="switch_state"):
            op(State(coords=Coordinates(time=jnp.arange(3.0))))

    def test_non_integer_switch_rejected(self):
        from rheplicant.core.combinators import SelectOperator
        from rheplicant.core.errors import StateValidationError

        op = SelectOperator(self._branch(1.0), self._branch(2.0))
        with pytest.raises(StateValidationError, match="integer"):
            op(self._state([0.5, 1.0]))

    def test_branches_get_independent_keys_and_no_data(self):
        from rheplicant.core.combinators import SelectOperator

        class Probe(AbstractOperator):
            def __call__(self, state):
                assert state.data is None
                n = state.coords.time.shape[0]
                return state.with_data(jax.random.normal(state.next_key()[0], (n, 1)))

        op = SelectOperator(Probe(), Probe())
        s = self._state([0, 1]).replace(key=jax.random.key(0), data=jnp.ones((2, 1)))
        out = op(s)
        assert out.data.shape == (2, 1)
        assert not jnp.array_equal(
            jax.random.key_data(out.key), jax.random.key_data(s.key)
        )

    def test_jit_and_grad(self):
        from rheplicant.core.combinators import SelectOperator

        op = SelectOperator(self._branch(2.0), self._branch(5.0))
        s = self._state([0, 1, 1])
        out = eqx.filter_jit(op)(s)
        assert jnp.array_equal(out.data[:, 0], jnp.array([2.0, 5.0, 5.0]))

        def loss(op):
            return jnp.sum(op(s).data)

        g = eqx.filter_grad(loss)(op)
        assert jnp.allclose(g.branches[0].v, 2.0)  # one sample x two freq
        assert jnp.allclose(g.branches[1].v, 4.0)  # two samples x two freq

    def test_switch_length_mismatch_raises(self):
        """Regression: a wrong-length switch must not silently broadcast."""
        from rheplicant import Coordinates
        from rheplicant.core.combinators import SelectOperator
        from rheplicant.core.errors import StateValidationError

        op = SelectOperator(self._branch(1.0), self._branch(10.0))
        state = State(
            coords=Coordinates(time=jnp.arange(4.0),
                               extra={"switch_state": jnp.array([1])}),
        )
        with pytest.raises(StateValidationError, match="leading time axis"):
            op(state)

    def test_scalar_leaf_rejected(self):
        """Regression: a scalar branch leaf must not be broadcast into a time axis."""
        from rheplicant.core.combinators import SelectOperator
        from rheplicant.core.errors import StateValidationError

        class ScalarSource(AbstractOperator):
            v: jax.Array

            def __call__(self, state):
                return state.with_data(self.v)

        op = SelectOperator(ScalarSource(v=jnp.array(1.0)), ScalarSource(v=jnp.array(2.0)))
        with pytest.raises(StateValidationError, match="leading time axis"):
            op(self._state([0, 1]))

    def test_structure_mismatch_names_select_operator(self):
        """Regression: _tree_add errors name the combinator that raised them."""
        from rheplicant.core.combinators import SelectOperator

        class DictSource(AbstractOperator):
            def __call__(self, state):
                n = state.coords.time.shape[0]
                return state.with_data({"a": jnp.ones((n, 1))})

        op = SelectOperator(self._branch(1.0), DictSource())
        with pytest.raises(PipelineError, match="SelectOperator branch"):
            op(self._state([0, 1]))

    @pytest.mark.parametrize("none_first", [True, False])
    def test_a_branch_that_produces_no_data_is_refused(self, none_first):
        """The SelectOperator half of a refusal SumOperator already had tested.

        Both combinators strip ``data`` to None before calling a branch, so a
        transform-shaped branch contributes None and there is nothing to select
        from. SumOperator's copy of this guard was covered and this one was
        not, and they are separate lines carrying separate messages.

        Parametrized over position because the guard sits inside the branch
        loop while the accumulator is still None on the first iteration: a
        version that tested the accumulator instead of the contribution would
        pass with the passthrough second and fail with it first.
        """
        from rheplicant.core.combinators import SelectOperator

        passthrough = _Identity()
        source = self._branch(2.0)
        branches = (passthrough, source) if none_first else (source, passthrough)
        names = ("dead", "live") if none_first else ("live", "dead")
        op = SelectOperator(*branches, names=names)
        with pytest.raises(PipelineError) as excinfo:
            op(self._state([0, 1, 0]))
        # Names the combinator and the offending branch: SumOperator's copy of
        # the sentence differs only in the first word, and a message that named
        # the wrong branch would still match on "produced no data".
        assert "SelectOperator branch 'dead' produced no data" in str(excinfo.value)


class TestReplaceBranch:
    def test_swaps_and_preserves_names(self):
        op = SumOperator(
            Constant(value=jnp.array(1.0)), Constant(value=jnp.array(2.0)),
            names=("a", "b"),
        )
        op2 = op.replace_branch("b", Constant(value=jnp.array(10.0)))
        assert jnp.array_equal(op2(State()).data, jnp.full(3, 11.0))
        assert op2.names == op.names
        assert op["b"].value == 2.0  # original untouched

    def test_rejects_non_operator(self):
        op = SumOperator(Constant(value=jnp.array(1.0)))
        with pytest.raises(PipelineError, match="AbstractOperator"):
            op.replace_branch(0, lambda s: s)


class _Identity(AbstractOperator):
    def __call__(self, state: State) -> State:
        return state


class TestPRNGSemantics:
    def test_branches_get_independent_keys(self):
        """Two stochastic branches must NOT reuse the same key."""
        op = SumOperator(NoisySource(), NoisySource())
        single = SumOperator(NoisySource())
        out2 = op(State(key=jax.random.key(0)))
        out1 = single(State(key=jax.random.key(0)))
        # If keys were reused, the two-branch sum would be exactly 2x one branch.
        assert not jnp.allclose(out2.data, 2.0 * out1.data)

    def test_output_key_advances(self):
        op = SumOperator(Constant(value=jnp.array(1.0)))
        s = State(key=jax.random.key(0))
        out = op(s)
        assert not jnp.array_equal(
            jax.random.key_data(out.key), jax.random.key_data(s.key)
        )

    def test_reproducible(self):
        op = SumOperator(NoisySource(), NoisySource())
        a = op(State(key=jax.random.key(7)))
        b = op(State(key=jax.random.key(7)))
        assert jnp.array_equal(a.data, b.data)

    def test_keyless_state_works_for_deterministic_branches(self):
        op = SumOperator(Constant(value=jnp.array(1.0)))
        out = op(State())
        assert out.key is None


class TestJaxTransforms:
    def test_jit_and_grad(self):
        op = SumOperator(Constant(value=jnp.array(1.0)), Constant(value=jnp.array(2.0)))
        out = eqx.filter_jit(op)(State())
        assert jnp.array_equal(out.data, jnp.full(3, 3.0))

        def loss(op):
            return jnp.sum(op(State()).data)

        grads = eqx.filter_grad(loss)(op)
        assert jnp.allclose(grads.branches[0].value, 3.0)  # d(sum(v*ones(3)))/dv
        assert jnp.allclose(grads.branches[1].value, 3.0)

    def test_is_an_operator_and_nests_in_pipeline(self):
        sky = SumOperator(Constant(value=jnp.array(1.0)), Constant(value=jnp.array(2.0)))
        full = Pipeline(sky, _Scale(factor=jnp.array(10.0)))
        out = full(State())
        assert jnp.array_equal(out.data, jnp.full(3, 30.0))
