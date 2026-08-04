"""``must_precede``: the ordering constraint the graph checks instead of prose.

The failure this closes is silent. ``calibration.py`` stated its ordering
constraint in a module docstring, ``At("noise", cw)`` assembled cleanly, and
the tone's gain response dropped to exactly 1.0 — a calibrator that monitors
nothing, in a model that runs, differentiates and looks entirely healthy.

The mechanism is generic on purpose: any operator whose physics depends on
*where* it sits declares that in the graph's own nouns, and no line of
``assemble()`` knows what a calibration tone is.
"""

from typing import ClassVar

import jax
import jax.numpy as jnp
import pytest

from rheplicant.core.graph import (
    AssemblyError,
    At,
    NodeSpec,
    SignalGraph,
    assemble,
)
from rheplicant.core.operator import AbstractOperator
from rheplicant.core.state import State

S, T, J = "source", "transform", "junction"

# A DIAMOND, not a chain. The two branches are deliberately of different
# lengths and carry different factors, so an assembly that folded them the
# wrong way round would produce a different number rather than the same one.
#
#   a -> amp ------\
#                   +--> j -> out
#   b -------------/
#
# Toposort (stable Kahn, declaration order) is  a, b, amp, j, out.  So `b`
# sorts BEFORE `amp` while no signal from `b` ever reaches `amp`: this graph
# is the one that tells a reachability check apart from a toposort-index one.
DIAMOND = SignalGraph(
    "diamond",
    {
        "a": NodeSpec(S, "branch A source"),
        "b": NodeSpec(S, "branch B source"),
        "amp": NodeSpec(T, "amplifier, on branch A only"),
        "j": NodeSpec(J, "the two branches meet"),
        "out": NodeSpec(T, "sink"),
    },
    [("a", "amp"), ("amp", "j"), ("b", "j"), ("j", "out")],
)


class Src(AbstractOperator):
    graph_node: ClassVar[str | None] = None
    value: jax.Array

    def __call__(self, state):
        return state.with_data(self.value * jnp.ones(3))


class SrcA(Src):
    graph_node: ClassVar[str] = "a"


class SrcB(Src):
    graph_node: ClassVar[str] = "b"


class Mul(AbstractOperator):
    graph_node: ClassVar[str | None] = None
    factor: jax.Array

    def __call__(self, state):
        return state.with_data(state.data * self.factor)


class Amp(Mul):
    graph_node: ClassVar[str] = "amp"


class Out(Mul):
    graph_node: ClassVar[str] = "out"


class NeedsTheAmp(Src):
    """A source that is worthless unless the amplifier sees it."""

    graph_node: ClassVar[str] = "a"
    must_precede: ClassVar[tuple[str, ...]] = ("amp",)
    must_precede_because: ClassVar[str] = "Unamplified it is below the noise."


class NeedsTheAmpButSaysNothing(Src):
    """The same constraint with no rationale — the field is optional."""

    graph_node: ClassVar[str] = "a"
    must_precede: ClassVar[tuple[str, ...]] = ("amp",)


class NeedsANodeThatIsNotThere(Src):
    graph_node: ClassVar[str] = "a"
    must_precede: ClassVar[tuple[str, ...]] = ("amp", "preamp")


@pytest.fixture
def state():
    return State(data=None, meta={"obs_id": "ordering-000"})


class TestTheConstraintIsChecked:
    def test_a_satisfied_constraint_assembles_and_still_computes(self, state):
        """The mechanism must not change what a correct assembly produces."""
        twin = assemble(
            DIAMOND,
            NeedsTheAmp(value=jnp.array(2.0)),
            Amp(factor=jnp.array(10.0)),
            Out(factor=jnp.array(1.0)),
        )
        assert jnp.allclose(twin(state).data, 20.0)
        assert "a" in twin.lit

    def test_a_violated_constraint_is_refused(self, state):
        """Placed on the branch that never reaches the amplifier."""
        with pytest.raises(AssemblyError) as excinfo:
            assemble(
                DIAMOND,
                SrcA(value=jnp.array(1.0)),
                At("b", NeedsTheAmp(value=jnp.array(2.0))),
                Amp(factor=jnp.array(10.0)),
                Out(factor=jnp.array(1.0)),
            )
        message = str(excinfo.value)
        assert "NeedsTheAmp" in message  # the operator
        assert "['amp']" in message  # the constraint it declared
        assert "'b'" in message  # where it actually landed
        assert "below the noise" in message  # what goes silently wrong

    def test_reachability_not_toposort_index_is_what_is_checked(self, state):
        """The distinguishing case, and the reason for the diamond fixture.

        ``b`` sorts BEFORE ``amp`` in this template's toposort, so a check
        written against the topological index would accept this placement. No
        signal from ``b`` reaches ``amp``, which is the thing the physics
        actually asks for, so the reachability check refuses it.
        """
        assert DIAMOND._topo.index("b") < DIAMOND._topo.index("amp")
        with pytest.raises(AssemblyError, match="not reachable"):
            assemble(
                DIAMOND,
                SrcA(value=jnp.array(1.0)),
                At("b", NeedsTheAmp(value=jnp.array(2.0))),
                Amp(factor=jnp.array(10.0)),
                Out(factor=jnp.array(1.0)),
            )

    def test_an_operator_declaring_nothing_is_unaffected(self, state):
        """Every other operator in the package is in this case."""
        twin = assemble(
            DIAMOND,
            SrcA(value=jnp.array(1.0)),
            At("b", SrcB(value=jnp.array(2.0))),
            Amp(factor=jnp.array(10.0)),
            Out(factor=jnp.array(1.0)),
        )
        assert jnp.allclose(twin(state).data, 12.0)  # 1*10 (branch A) + 2 (branch B)


class TestWhatTheConstraintDoesNotClaim:
    def test_an_absent_stage_is_not_a_violation(self, state):
        """No amplifier in this assembly, so there is no amplifier to miss.

        Refusing here would reject a source-only assembly for the sake of a
        stage it never asked for — and the ``amp`` node contracts to identity,
        so nothing is silently skipped either.
        """
        twin = assemble(
            DIAMOND,
            At("b", NeedsTheAmp(value=jnp.array(2.0))),
            Out(factor=jnp.array(3.0)),
        )
        assert jnp.allclose(twin(state).data, 6.0)

    def test_a_region_covering_the_stage_satisfies_it(self, state):
        """One operator implementing both stages orders them internally.

        The graph has nothing to say about an ordering that happens inside a
        single operator, so it says nothing rather than guessing.
        """
        twin = assemble(
            DIAMOND,
            At(("a", "amp"), NeedsTheAmp(value=jnp.array(2.0))),
            Out(factor=jnp.array(1.0)),
        )
        assert jnp.allclose(twin(state).data, 2.0)

    def test_a_rationale_is_optional(self, state):
        """The refusal still names operator, constraint and placement."""
        with pytest.raises(AssemblyError) as excinfo:
            assemble(
                DIAMOND,
                SrcA(value=jnp.array(1.0)),
                At("b", NeedsTheAmpButSaysNothing(value=jnp.array(2.0))),
                Amp(factor=jnp.array(10.0)),
                Out(factor=jnp.array(1.0)),
            )
        message = str(excinfo.value)
        assert "NeedsTheAmpButSaysNothing" in message
        assert "['amp']" in message and "'b'" in message
        assert "  " not in message  # no gap where the missing rationale was


class TestTheDeclarationItselfIsChecked:
    def test_a_constraint_naming_an_unknown_node_is_refused(self, state):
        """Otherwise a typo turns the constraint back into unenforced prose."""
        with pytest.raises(AssemblyError, match="not a node of graph"):
            assemble(
                DIAMOND,
                NeedsANodeThatIsNotThere(value=jnp.array(2.0)),
                Amp(factor=jnp.array(10.0)),
                Out(factor=jnp.array(1.0)),
            )

    def test_the_unknown_node_is_refused_even_when_nothing_else_is_wrong(self, state):
        """``a -> amp`` is a perfectly good placement for the OTHER constraint;
        the refusal is about the declaration, not about this assembly."""
        with pytest.raises(AssemblyError, match="preamp"):
            assemble(
                DIAMOND,
                NeedsANodeThatIsNotThere(value=jnp.array(2.0)),
                Amp(factor=jnp.array(10.0)),
                Out(factor=jnp.array(1.0)),
            )


class TestTheDefaultsOnAbstractOperator:
    def test_declaring_nothing_is_the_default(self):
        assert AbstractOperator.must_precede == ()
        assert AbstractOperator.must_precede_because == ""

    def test_a_subclass_inherits_its_base_declaration(self):
        """Same MRO resolution ``graph_node`` gets, for the same reason."""

        class Narrower(NeedsTheAmp):
            pass

        assert Narrower.must_precede == ("amp",)
        assert "below the noise" in Narrower.must_precede_because
