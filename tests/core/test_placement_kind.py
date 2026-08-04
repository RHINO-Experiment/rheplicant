"""A placed operator must agree with its node about who creates the data.

``Assembly.has_source`` was read off the node kinds of the template — "is there
a live ``source`` node in this fold" — and ``At`` will put any operator at any
node. So the one fact the ``__call__`` guard is built on was a fact about the
*graph*, and the guard was wrong in both directions whenever the operator
disagreed with its node:

* a SOURCE operator at a TRANSFORM node assembled with ``has_source=False``,
  so the guard whose whole sentence is "caller-supplied ``state.data`` would be
  discarded" passed — in exactly the case where the operator discards it, along
  with whatever the upstream branch computed;
* a TRANSFORM operator at a SOURCE node assembled with ``has_source=True``, so
  the guard demanded ``data=None`` — the one input that makes it die with
  ``TypeError: unsupported operand type(s) for *: 'NoneType' and ...``.

The fix screens the disagreement at placement, which fixes both at once: once
the operator's declared home and its actual node agree about creating data, the
node kind IS the operator kind and the existing derivation is sound.

The screen can only read what an operator declares, so an operator with no
``graph_node`` is not screened. That hole is pinned below rather than left for
someone to discover, because it is the whole of what this check does not cover.
"""

from typing import ClassVar

import jax
import jax.numpy as jnp
import pytest

from rheplicant.core.graph import AssemblyError, At, NodeSpec, SignalGraph, assemble
from rheplicant.core.operator import AbstractOperator
from rheplicant.core.state import State

S, T, J = "source", "transform", "junction"

# Two source slots and two transform slots, so "wrong kind" can be told apart
# from "wrong node": moving an operator between two nodes of the SAME kind is
# allowed and is tested, moving it across kinds is not.
#
#   s1 --\
#         +--(j)--> t1 --> t2
#   s2 --/
GRAPH = SignalGraph(
    "placement-kind",
    {
        "s1": NodeSpec(S, "first source slot"),
        "s2": NodeSpec(S, "second source slot"),
        "j": NodeSpec(J, "the sources meet"),
        "t1": NodeSpec(T, "first transform slot"),
        "t2": NodeSpec(T, "second transform slot / sink"),
    },
    [("s1", "j"), ("s2", "j"), ("j", "t1"), ("t1", "t2")],
)

# Distinct primes throughout, so any misplacement changes the number.
S1_VALUE, S2_VALUE, T1_FACTOR, T2_FACTOR = 2.0, 5.0, 7.0, 11.0
CALLER_DATA = 13.0


class _Src(AbstractOperator):
    graph_node: ClassVar[str | None] = None
    value: jax.Array

    def __call__(self, state):
        return state.with_data(self.value * jnp.ones(3))


class SrcS1(_Src):
    graph_node: ClassVar[str] = "s1"


class SrcS2(_Src):
    graph_node: ClassVar[str] = "s2"


class SrcNowhere(_Src):
    """Declares a home that this template does not have."""

    graph_node: ClassVar[str] = "not_a_node_here"


class _Mul(AbstractOperator):
    graph_node: ClassVar[str | None] = None
    factor: jax.Array

    def __call__(self, state):
        return state.with_data(state.data * self.factor)


class MulT1(_Mul):
    graph_node: ClassVar[str] = "t1"


class MulT2(_Mul):
    graph_node: ClassVar[str] = "t2"


class TestTheDisagreementIsRefused:
    def test_a_source_operator_on_a_transform_node(self):
        """Direction 1 — the placement that used to discard the caller's data."""
        with pytest.raises(AssemblyError) as excinfo:
            assemble(GRAPH, At("t1", SrcS1(value=jnp.array(S1_VALUE))))
        message = str(excinfo.value)
        assert "SrcS1" in message  # the operator
        assert "'s1'" in message and "'t1'" in message  # both nodes
        assert "source" in message and "transform" in message  # both kinds

    def test_a_transform_operator_on_a_source_node(self):
        """Direction 2 — the placement whose guard refused the only input that ran."""
        with pytest.raises(AssemblyError) as excinfo:
            assemble(GRAPH, At("s1", MulT1(factor=jnp.array(T1_FACTOR))))
        message = str(excinfo.value)
        assert "MulT1" in message
        assert "'t1'" in message and "'s1'" in message
        assert "source" in message and "transform" in message

    def test_the_refusal_says_what_would_have_gone_wrong(self):
        """The message has to carry the argument, not just the mismatch."""
        with pytest.raises(AssemblyError) as excinfo:
            assemble(GRAPH, At("t1", SrcS1(value=jnp.array(S1_VALUE))))
        message = str(excinfo.value)
        assert "has_source" in message
        assert "discard" in message


class TestKindAgreementIsAllThatIsAsked:
    """The screen must not narrow ``At`` into "place it where it says"."""

    def test_a_source_at_a_different_SOURCE_node_still_assembles(self):
        asm = assemble(GRAPH, At("s2", SrcS1(value=jnp.array(S1_VALUE))))
        assert asm.has_source is True
        assert asm.lit == ("s2",)
        assert jnp.allclose(asm(State()).data, S1_VALUE)

    def test_a_transform_at_a_different_TRANSFORM_node_still_assembles(self):
        asm = assemble(GRAPH, At("t2", MulT1(factor=jnp.array(T1_FACTOR))))
        assert asm.has_source is False
        assert asm.lit == ("t2",)
        out = asm(State(data=jnp.full(3, CALLER_DATA)))
        assert jnp.allclose(out.data, CALLER_DATA * T1_FACTOR)

    def test_the_ordinary_assembly_is_untouched(self):
        asm = assemble(
            GRAPH,
            SrcS1(value=jnp.array(S1_VALUE)),
            SrcS2(value=jnp.array(S2_VALUE)),
            MulT1(factor=jnp.array(T1_FACTOR)),
            MulT2(factor=jnp.array(T2_FACTOR)),
        )
        assert asm.has_source is True
        expected = (S1_VALUE + S2_VALUE) * T1_FACTOR * T2_FACTOR
        assert jnp.allclose(asm(State()).data, expected)


class TestRegions:
    """A region is entered at its FIRST node, so that is what it must match."""

    def test_a_region_covering_the_operators_own_home_is_untouched(self):
        asm = assemble(GRAPH, At(("s1", "j", "t1"), SrcS1(value=jnp.array(S1_VALUE))))
        assert asm.has_source is True
        assert jnp.allclose(asm(State()).data, S1_VALUE)

    def test_a_transform_entering_a_region_at_a_source_node_is_refused(self):
        with pytest.raises(AssemblyError) as excinfo:
            assemble(GRAPH, At(("s1", "j", "t1"), MulT1(factor=jnp.array(T1_FACTOR))))
        assert "'s1'" in str(excinfo.value)

    def test_a_source_entering_a_region_at_a_transform_node_is_refused(self):
        with pytest.raises(AssemblyError) as excinfo:
            assemble(
                GRAPH,
                SrcS1(value=jnp.array(S1_VALUE)),
                At(("t1", "t2"), SrcS2(value=jnp.array(S2_VALUE))),
            )
        assert "'t1'" in str(excinfo.value)

    def test_a_transform_rooted_region_by_a_transform_is_untouched(self):
        asm = assemble(
            GRAPH,
            SrcS1(value=jnp.array(S1_VALUE)),
            At(("t1", "t2"), MulT2(factor=jnp.array(T2_FACTOR))),
        )
        assert jnp.allclose(asm(State()).data, S1_VALUE * T2_FACTOR)


class TestTheGuardIsNowTrueInBothDirections:
    def test_a_sourced_assembly_refuses_caller_data(self):
        asm = assemble(GRAPH, SrcS1(value=jnp.array(S1_VALUE)))
        with pytest.raises(AssemblyError, match="discarded"):
            asm(State(data=jnp.full(3, CALLER_DATA)))

    def test_a_transform_chain_refuses_a_dataless_state(self):
        asm = assemble(GRAPH, MulT1(factor=jnp.array(T1_FACTOR)))
        assert asm.has_source is False
        with pytest.raises(AssemblyError, match="transform chain"):
            asm(State(data=None))

    def test_a_transform_chain_runs_on_caller_data(self):
        """Direction 2's crash, as the working call it should always have been."""
        asm = assemble(GRAPH, MulT1(factor=jnp.array(T1_FACTOR)))
        out = asm(State(data=jnp.full(3, CALLER_DATA)))
        assert jnp.allclose(out.data, CALLER_DATA * T1_FACTOR)


class TestWhatTheScreenCannotSee:
    """The hole, pinned. Both cases are "there is nothing to compare against"."""

    def test_an_operator_declaring_no_home_is_not_screened(self):
        """Every ``graph_node = None`` operator — surrogates, lambdas, one-offs.

        The assembly below is still the wrong one: a source at a transform
        node, reporting ``has_source=False`` and discarding the caller's data
        without a word. Nothing in the operator says it creates data, so
        nothing here can say it does not belong. If a declaration for this
        ever exists, delete this test and screen the case.
        """
        undeclared_source = _Src(value=jnp.array(S1_VALUE))
        asm = assemble(GRAPH, At("t1", undeclared_source))
        assert asm.has_source is False
        out = asm(State(data=jnp.full(3, CALLER_DATA)))
        assert jnp.allclose(out.data, S1_VALUE)  # the caller's 13.0 is gone

    def test_a_home_this_template_does_not_have_is_not_screened(self):
        """The declaration is about some other graph, so it says nothing here."""
        asm = assemble(GRAPH, At("t1", SrcNowhere(value=jnp.array(S1_VALUE))))
        assert asm.has_source is False
