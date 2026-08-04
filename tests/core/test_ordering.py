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

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from rheplicant.core import pipeline as pipeline_module
from rheplicant.core.errors import PipelineError
from rheplicant.core.graph import (
    AssemblyError,
    At,
    NodeSpec,
    SignalGraph,
    assemble,
)
from rheplicant.core.operator import AbstractOperator
from rheplicant.core.pipeline import Pipeline
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


class Tone(AbstractOperator):
    """A level ADDED to the signal — the shape of the motivating failure.

    Multiplying stages commute with the constraint being wrong; adding one does
    not. Injected after the amplifier this tone comes out at its own level, so
    the two orders give two different numbers (30.0 and 12.0) rather than the
    same one by a symmetry of the fixture.
    """

    graph_node: ClassVar[str | None] = None
    must_precede: ClassVar[tuple[str, ...]] = ("amp",)
    must_precede_because: ClassVar[str] = (
        "A tone injected downstream of the amplifier has a response of "
        "exactly 1.0, so it monitors nothing."
    )
    level: jax.Array

    def __call__(self, state):
        return state.with_data(state.data + self.level)


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


class TestTheHandBuiltRoute:
    """The same constraint, on the route that does not go through a graph.

    ``assemble`` and a hand-built ``Pipeline`` compile to the same composition,
    and only the first one checked. So the refusal was one line of call site
    away from silence::

        assemble(..., At('noise', tone))   AssemblyError: 'bandpass' is not reachable
        Pipeline(sky, band, gain, tone)    no error; the tone's response is 1.0

    A Pipeline has no graph, so it cannot ask what is reachable from where; it
    has ``names``, so it can ask what comes after what *in this sequence*. That
    is a weaker check and it is the strongest one the vocabulary supports.
    """

    def _stages(self):
        return SrcA(value=jnp.array(1.0)), Tone(level=jnp.array(2.0)), Amp(
            factor=jnp.array(10.0)
        )

    def test_the_correct_order_builds_and_computes(self, state):
        src, tone, amp = self._stages()
        pipeline = Pipeline(src, tone, amp, names=("a", "tone", "amp"))
        assert jnp.allclose(pipeline(state).data, 30.0)  # (1 + 2) * 10

    def test_the_wrong_order_is_refused(self, state):
        """The pipeline that ran happily and reported a tone response of 1.0."""
        src, tone, amp = self._stages()
        with pytest.raises(PipelineError) as excinfo:
            Pipeline(src, amp, tone, names=("a", "amp", "tone"))
        message = str(excinfo.value)
        assert "Tone" in message  # the operator
        assert "['amp']" in message  # the constraint it declared
        assert "'tone'" in message and "'amp'" in message  # both stage names
        assert "response of exactly 1.0" in message  # what goes silently wrong

    def test_the_refused_pipeline_is_the_one_that_computed_the_wrong_number(self):
        """The refusal has to be worth something: name the number it prevents.

        Built through the stages directly, the rejected order gives 12.0 where
        the physics asks for 30.0 — the tone added downstream of the gain it is
        supposed to monitor, its response exactly 1.0 instead of 10.0.
        """
        src, tone, amp = self._stages()
        by_hand = tone(amp(src(State(data=None))))
        assert jnp.allclose(by_hand.data, 12.0)

    def test_auto_derived_names_are_checked_too(self, state):
        """The names are the vocabulary, however they were arrived at."""
        src, tone, amp = self._stages()
        # `Amp` auto-names to "amp", which is exactly the node id the tone
        # declares — so the constraint binds without anyone passing `names=`.
        assert Pipeline(src, tone, amp).names == ("srca", "tone", "amp")
        with pytest.raises(PipelineError, match="\\['amp'\\]"):
            Pipeline(src, amp, tone)

    def test_replace_stage_re_checks(self, state):
        """The functional-update path rebuilds, so it must refuse the same."""
        src, tone, amp = self._stages()
        innocent = Pipeline(
            src, amp, Mul(factor=jnp.array(3.0)), names=("a", "amp", "tone")
        )
        assert jnp.allclose(innocent(state).data, 30.0)  # 1 * 10 * 3
        with pytest.raises(PipelineError, match="\\['amp'\\]"):
            innocent.replace_stage("tone", tone)

    def test_a_stage_that_is_not_present_is_not_a_violation(self, state):
        """The same rule ``assemble`` applies to a node that was never lit.

        A stricter reading would refuse every partial pipeline — every twin
        built without its amplifier — for the sake of a stage it never asked
        for.
        """
        src, tone, _ = self._stages()
        pipeline = Pipeline(src, tone, names=("a", "tone"))
        assert jnp.allclose(pipeline(state).data, 3.0)

    def test_a_constraint_naming_itself_is_not_a_violation(self, state):
        """Parity with ``_check_ordering``, which skips ``target in path``."""

        class SelfNaming(Tone):
            must_precede: ClassVar[tuple[str, ...]] = ("tone",)

        pipeline = Pipeline(
            SrcA(value=jnp.array(1.0)), SelfNaming(level=jnp.array(2.0)),
            names=("a", "tone"),
        )
        assert jnp.allclose(pipeline(state).data, 3.0)


class TestWhatTheSequenceCheckCannotSee:
    """Stated as tests so the weaker check is not mistaken for the stronger."""

    def test_a_target_inside_a_nested_composite_is_invisible(self, state):
        """``names`` is one level deep, and so is the check.

        The inner Pipeline is one stage called ``inner`` to the outer one, so
        ``amp`` is not a name the outer sequence has. ``assemble`` sees this
        case because the graph names nodes, not stages.
        """
        pipeline = Pipeline(
            SrcA(value=jnp.array(1.0)),
            Pipeline(Amp(factor=jnp.array(10.0)), names=("amp",)),
            Tone(level=jnp.array(2.0)),
            names=("a", "inner", "tone"),
        )
        assert jnp.allclose(pipeline(state).data, 12.0)  # the wrong number, unrefused

    def test_a_constraint_naming_a_stage_nothing_has_is_silent(self, state):
        """``assemble`` refuses an unknown NODE; a Pipeline has no node list.

        A typo in ``must_precede`` is indistinguishable from a stage this
        pipeline legitimately does not have, so the sequence check cannot
        refuse it without refusing every partial pipeline as well.
        """
        pipeline = Pipeline(
            NeedsANodeThatIsNotThere(value=jnp.array(2.0)),
            Amp(factor=jnp.array(10.0)),
            names=("a", "amp"),
        )
        assert jnp.allclose(pipeline(state).data, 20.0)

    def test_construction_is_the_only_time_it_runs(self, state):
        """It must cost nothing per call, and raise nothing mid-trace.

        Equinox rebuilds a Module through ``tree_unflatten``, which does not go
        through ``__init__`` — so a jit trace, a gradient and an
        ``eqx.tree_at`` edit of an already-built Pipeline never re-run the
        check. That is what makes putting it in ``__init__`` safe rather than a
        per-call tax on the forward model.
        """
        calls = []
        real = pipeline_module.check_stage_ordering

        def counting(*args, **kwargs):
            calls.append(1)
            return real(*args, **kwargs)

        src, tone, amp = SrcA(value=jnp.array(1.0)), Tone(level=jnp.array(2.0)), Amp(
            factor=jnp.array(10.0)
        )
        built = Pipeline(src, tone, amp, names=("a", "tone", "amp"))
        pipeline_module.check_stage_ordering = counting
        try:
            eqx.filter_jit(built)(state)
            eqx.filter_grad(lambda p: jnp.sum(p(state).data))(built)
            eqx.tree_at(lambda p: p.stages[2].factor, built, jnp.array(3.0))
            leaves, treedef = jax.tree_util.tree_flatten(built)
            jax.tree_util.tree_unflatten(treedef, leaves)
        finally:
            pipeline_module.check_stage_ordering = real
        assert calls == []


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
