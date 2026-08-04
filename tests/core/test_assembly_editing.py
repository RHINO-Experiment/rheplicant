"""Editing an assembly: `without` removes a stage, `replace_node` refuses junk.

Two entry points used to accept an object they could not use and fail later
with a bare Python error — `replace_node(node, None)` returned a live Assembly
whose `lit` and whose rendering both still claimed the stage, and `assemble`
had no operator type check at all, so a non-operator reached
`op.must_precede` and died on `AttributeError`. Both refuse by name here, and
`without` is the supported answer `replace_node(..., None)` was reaching for.

Every fixture is numerically distinguishable: no two nodes carry the same
factor, so dropping the wrong one is a different number rather than the same
one.
"""

from typing import ClassVar

import jax
import jax.numpy as jnp
import pytest

from rheplicant.core.errors import PipelineError
from rheplicant.core.graph import (
    Assembly,
    AssemblyError,
    At,
    NodeSpec,
    SignalGraph,
    assemble,
    register_graph,
)
from rheplicant.core.operator import AbstractOperator
from rheplicant.core.state import State

S, T, J = "source", "transform", "junction"


class Src(AbstractOperator):
    graph_node: ClassVar[str | None] = None
    value: jax.Array

    def __call__(self, state):
        return state.with_data(self.value * jnp.ones(3))


class SrcA(Src):
    graph_node: ClassVar[str] = "a"


class SrcB(Src):
    graph_node: ClassVar[str] = "b"


class SrcC(Src):
    graph_node: ClassVar[str] = "c"


class Mul(AbstractOperator):
    graph_node: ClassVar[str | None] = None
    factor: jax.Array

    def __call__(self, state):
        return state.with_data(state.data * self.factor)


class MulT1(Mul):
    graph_node: ClassVar[str] = "t1"


class MulT2(Mul):
    graph_node: ClassVar[str] = "t2"


@pytest.fixture(scope="module")
def graph():
    """a, b -> (j1) -> t1 -> (j2) <- c ; j2 -> t2 -> t3."""
    g = SignalGraph(
        "editing-graph",
        {
            "a": NodeSpec(S), "b": NodeSpec(S), "j1": NodeSpec(J), "t1": NodeSpec(T),
            "c": NodeSpec(S), "j2": NodeSpec(J), "t2": NodeSpec(T), "t3": NodeSpec(T),
        },
        [
            ("a", "j1"), ("b", "j1"), ("j1", "t1"),
            ("t1", "j2"), ("c", "j2"), ("j2", "t2"), ("t2", "t3"),
        ],
    )
    register_graph(g)
    return g


@pytest.fixture(scope="module")
def many_graph():
    """a (MANY sources), b -> (j1) -> t1."""
    g = SignalGraph(
        "editing-many-graph",
        {"a": NodeSpec(S, many=True), "b": NodeSpec(S), "j1": NodeSpec(J), "t1": NodeSpec(T)},
        [("a", "j1"), ("b", "j1"), ("j1", "t1")],
    )
    register_graph(g)
    return g


@pytest.fixture(scope="module")
def chain_graph():
    """s -> t: the smallest assembly whose has_source can flip."""
    g = SignalGraph(
        "editing-chain-graph", {"s": NodeSpec(S), "t": NodeSpec(T)}, [("s", "t")]
    )
    register_graph(g)
    return g


@pytest.fixture
def blank():
    return State(data=None)


@pytest.fixture
def full(graph):
    """((2 + 3) * 5 + 7) * 11 = 352 — and every single-drop is a distinct number."""
    return assemble(
        graph,
        SrcA(value=jnp.array(2.0)),
        SrcB(value=jnp.array(3.0)),
        SrcC(value=jnp.array(7.0)),
        MulT1(factor=jnp.array(5.0)),
        MulT2(factor=jnp.array(11.0)),
    )


def value(assembly, state):
    return float(assembly(state).data[0])


class TestWithoutValues:
    def test_the_undropped_assembly_is_the_baseline(self, full, blank):
        assert value(full, blank) == 352.0

    @pytest.mark.parametrize(
        ("node_id", "expected"),
        [("a", 242.0), ("b", 187.0), ("c", 275.0), ("t1", 132.0), ("t2", 32.0)],
    )
    def test_dropping_each_node_gives_its_own_number(self, full, blank, node_id, expected):
        """242 != 187: dropping `a` and dropping `b` cannot be confused."""
        assert value(full.without(node_id), blank) == expected

    def test_the_original_is_untouched(self, full, blank):
        full.without("a")
        assert value(full, blank) == 352.0

    def test_the_placement_record_is_in_template_order_not_call_order(self, graph):
        """`assemble` promises argument order is irrelevant; the record is part of it."""
        ops = [
            MulT2(factor=jnp.array(11.0)),
            SrcB(value=jnp.array(3.0)),
            SrcA(value=jnp.array(2.0)),
        ]
        forwards, backwards = assemble(graph, *ops), assemble(graph, *reversed(ops))
        assert forwards.placements == ((("a",), "a"), (("b",), "b"), (("t2",), "t2"))
        assert forwards.placements == backwards.placements


class TestWithoutMetadata:
    def test_lit_drops_the_node(self, full):
        assert full.lit == ("a", "b", "t1", "c", "t2")  # template declaration order
        assert full.without("t1").lit == ("a", "b", "c", "t2")

    def test_the_rendering_no_longer_shows_the_node_as_lit(self, full):
        """`replace_node(node, None)` left the picture claiming the stage was there."""
        assert "class t1 lit;" in full.to_mermaid()
        assert "class t1 lit;" not in full.without("t1").to_mermaid()

    def test_has_source_flips_when_the_last_source_goes(self, chain_graph, blank):
        chain = assemble(
            chain_graph, At("s", SrcA(value=jnp.array(2.0))), At("t", MulT1(factor=jnp.array(5.0)))
        )
        assert chain.has_source is True and value(chain, blank) == 10.0
        stripped = chain.without("s")
        assert stripped.has_source is False
        # ...and it now says so: it needs caller data, and refuses data=None.
        with pytest.raises(AssemblyError, match="pure transform chain"):
            stripped(blank)
        assert float(stripped(State(data=jnp.ones(3) * 4.0)).data[0]) == 20.0

    def test_a_pass_through_junction_is_reported_as_skipped(self, full):
        """Dropping `c` leaves j2 with one upstream: traversed, not materialized."""
        dropped = full.without("c")
        assert "j2" in dropped.skipped
        assert "j2" not in dropped.materialized
        assert "j2" in full.materialized


class TestWithoutRegionsAndManyNodes:
    def test_a_kept_region_is_re_placed_over_its_whole_path(self, graph, blank):
        region = assemble(
            graph,
            SrcA(value=jnp.array(2.0)),
            SrcB(value=jnp.array(3.0)),
            At(("t2", "t3"), MulT2(factor=jnp.array(13.0))),
        )
        assert region.lit == ("a", "b", "t2", "t3")
        dropped = region.without("a")
        assert dropped.lit == ("b", "t2", "t3")
        assert value(dropped, blank) == 39.0

    def test_dropping_any_covered_node_drops_the_whole_region(self, graph, blank):
        region = assemble(
            graph,
            SrcA(value=jnp.array(2.0)),
            At(("t2", "t3"), MulT2(factor=jnp.array(13.0))),
        )
        assert region.without("t3").lit == ("a",)
        assert value(region.without("t3"), blank) == 2.0

    def test_every_instance_of_a_many_node_survives_a_drop_elsewhere(
        self, many_graph, blank
    ):
        multi = assemble(
            many_graph,
            At("a", SrcA(value=jnp.array(2.0))),
            At("a", SrcA(value=jnp.array(3.0))),
            At("b", SrcB(value=jnp.array(11.0))),
            MulT1(factor=jnp.array(7.0)),
        )
        assert value(multi, blank) == 112.0  # (2 + 3 + 11) * 7
        assert value(multi.without("b"), blank) == 35.0  # (2 + 3) * 7 — both kept
        assert value(multi.without("a"), blank) == 77.0  # 11 * 7 — both gone

    def test_a_many_node_is_dropped_whole(self, many_graph):
        multi = assemble(
            many_graph,
            At("a", SrcA(value=jnp.array(2.0))),
            At("a", SrcA(value=jnp.array(3.0))),
            At("b", SrcB(value=jnp.array(11.0))),
            MulT1(factor=jnp.array(7.0)),
        )
        assert dict(multi.instances) == {"a": ("a_1", "a_2")}
        assert multi.without("a").instances == ()


class TestWithoutRefusals:
    def test_a_node_carrying_nothing_is_refused_by_name(self, full):
        with pytest.raises(AssemblyError, match="no operator sits at 't3'"):
            full.without("t3")

    def test_an_unknown_node_id_is_refused_the_same_way(self, full):
        with pytest.raises(AssemblyError, match="no operator sits at 'nonesuch'"):
            full.without("nonesuch")

    def test_dropping_the_only_operator_is_refused(self, chain_graph):
        lone = assemble(chain_graph, At("s", SrcA(value=jnp.array(2.0))))
        with pytest.raises(AssemblyError, match="would leave nothing to assemble"):
            lone.without("s")

    def test_an_assembly_built_by_hand_has_no_recipe(self, blank):
        by_hand = Assembly(
            operator=SrcA(value=jnp.array(2.0)),
            graph_name="editing-chain-graph",
            lit=("s",),
            skipped=(),
            has_source=True,
        )
        with pytest.raises(AssemblyError, match="carries no placement record"):
            by_hand.without("s")

    def test_a_drop_that_breaks_the_graph_is_refused_in_assembles_own_words(
        self, graph
    ):
        """`t1` is a transform branch into j2; with `a` and `b` gone it has no source."""
        both = assemble(
            graph,
            SrcA(value=jnp.array(2.0)),
            SrcC(value=jnp.array(7.0)),
            MulT1(factor=jnp.array(5.0)),
        )
        with pytest.raises(AssemblyError, match="no live source upstream"):
            both.without("a")


class TestWithoutAndOrdering:
    def test_dropping_a_must_precede_target_is_not_a_violation(self, graph, blank):
        """`without` re-runs _check_ordering, and an absent node contracts to identity.

        Measured on the shipped template too: CWCalibrationOperator declares
        must_precede=('bandpass', 'gain'), and dropping either leaves the tone
        placed and the assembly runnable.
        """

        class Tone(Mul):
            graph_node: ClassVar[str] = "t1"
            must_precede: ClassVar[tuple[str, ...]] = ("t2",)

        both = assemble(
            graph,
            SrcA(value=jnp.array(2.0)),
            Tone(factor=jnp.array(5.0)),
            MulT2(factor=jnp.array(11.0)),
        )
        assert value(both, blank) == 110.0
        dropped = both.without("t2")
        assert dropped.lit == ("a", "t1")
        assert value(dropped, blank) == 10.0

    def test_a_violated_placement_is_still_refused_after_the_drop(self, graph):
        """The re-run is a real check, not a formality: the refusal survives it."""

        class Tone(Mul):
            graph_node: ClassVar[str] = "t2"
            must_precede: ClassVar[tuple[str, ...]] = ("t1",)

        with pytest.raises(AssemblyError, match="is not reachable"):
            assemble(
                graph,
                SrcA(value=jnp.array(2.0)),
                SrcB(value=jnp.array(3.0)),
                MulT1(factor=jnp.array(5.0)),
                Tone(factor=jnp.array(11.0)),
            )


class TestReplaceNodeRefusals:
    def test_none_is_refused_and_points_at_without(self, full):
        with pytest.raises(AssemblyError, match=r"assembly.without\('t1'\)"):
            full.replace_node("t1", None)

    def test_a_non_operator_is_refused_by_the_shared_screen(self, full):
        with pytest.raises(PipelineError, match="replace_node operator 0 is float"):
            full.replace_node("t1", 3.0)

    def test_a_real_operator_still_replaces(self, full, blank):
        swapped = full.replace_node("t1", MulT1(factor=jnp.array(2.0)))
        assert value(swapped, blank) == 187.0  # ((2 + 3) * 2 + 7) * 11


class TestAssembleTypeCheck:
    def test_a_bare_module_is_refused_by_name_not_by_attribute_error(self, graph):
        class Bare:
            pass

        with pytest.raises(PipelineError, match="assemble operator 0 is Bare"):
            assemble(graph, At("t1", Bare()))

    def test_the_refusal_survives_an_unwrapped_operator_too(self, graph):
        with pytest.raises(PipelineError, match="assemble operator 1 is NoneType"):
            assemble(graph, SrcA(value=jnp.array(2.0)), None)
