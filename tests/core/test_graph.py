"""Tests for SignalGraph templates and graph-guided assembly."""

from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from replicant.core.combinators import SumOperator
from replicant.core.graph import (
    Assembly,
    AssemblyError,
    At,
    NodeSpec,
    SignalGraph,
    assemble,
    register_graph,
)
from replicant.core.operator import AbstractOperator
from replicant.core.pipeline import Pipeline
from replicant.core.state import State

S, T, J = "source", "transform", "junction"


class Src(AbstractOperator):
    """Test source: value * ones(3)."""

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


@pytest.fixture
def graph():
    """a, b -> (J1) -> t1 -> (J2) <- c ; J2 -> t2 -> sinkT.

    Exercises: multi-source junction, mid-trunk junction, trunk transforms.
    """
    return SignalGraph(
        "test-graph",
        {
            "a": NodeSpec(S),
            "b": NodeSpec(S),
            "j1": NodeSpec(J),
            "t1": NodeSpec(T),
            "c": NodeSpec(S),
            "j2": NodeSpec(J),
            "t2": NodeSpec(T),
            "t3": NodeSpec(T),
        },
        [
            ("a", "j1"), ("b", "j1"), ("j1", "t1"),
            ("t1", "j2"), ("c", "j2"), ("j2", "t2"), ("t2", "t3"),
        ],
    )


class TestTemplateValidation:
    def test_cycle_rejected(self):
        with pytest.raises(AssemblyError, match="cycle"):
            SignalGraph("bad", {"x": NodeSpec(T), "y": NodeSpec(T)}, [("x", "y"), ("y", "x")])

    def test_two_sinks_rejected(self):
        with pytest.raises(AssemblyError, match="sink"):
            SignalGraph(
                "bad", {"a": NodeSpec(S), "x": NodeSpec(T), "y": NodeSpec(T)},
                [("a", "x"), ("a", "y")],
            )

    def test_junction_degree_enforced(self):
        with pytest.raises(AssemblyError, match="in-degree"):
            SignalGraph(
                "bad", {"a": NodeSpec(S), "j": NodeSpec(J)}, [("a", "j")]
            )

    def test_source_indegree_enforced(self):
        with pytest.raises(AssemblyError, match="in-degree"):
            SignalGraph(
                "bad", {"b": NodeSpec(S), "a": NodeSpec(S)}, [("b", "a")]
            )

    def test_unknown_edge_node(self):
        with pytest.raises(AssemblyError, match="unknown"):
            SignalGraph("bad", {"a": NodeSpec(S)}, [("a", "ghost")])


class TestResolution:
    def test_unknown_class_needs_at(self, graph):
        with pytest.raises(AssemblyError, match="At"):
            assemble(graph, Src(value=jnp.array(1.0)))  # no graph_node

    def test_at_overrides(self, graph):
        out = assemble(graph, At("a", Src(value=jnp.array(2.0))))(State())
        assert jnp.array_equal(out.data, jnp.full(3, 2.0))

    def test_at_unknown_node(self, graph):
        with pytest.raises(AssemblyError, match="not a node"):
            assemble(graph, At("ghost", SrcA(value=jnp.array(1.0))))

    def test_junction_is_not_a_slot(self, graph):
        with pytest.raises(AssemblyError, match="junction"):
            assemble(graph, At("j1", SrcA(value=jnp.array(1.0))))

    def test_duplicate_on_single_instance_node(self, graph):
        with pytest.raises(AssemblyError, match="single instance"):
            assemble(graph, SrcA(value=jnp.array(1.0)), SrcA(value=jnp.array(2.0)))

    def test_subclass_inherits_slot(self, graph):
        class MySrcA(SrcA):
            pass

        out = assemble(graph, MySrcA(value=jnp.array(3.0)))(State())
        assert jnp.array_equal(out.data, jnp.full(3, 3.0))

    def test_empty_rejected(self, graph):
        with pytest.raises(AssemblyError, match="at least one"):
            assemble(graph)


class TestFolding:
    def test_single_source(self, graph):
        asm = assemble(graph, SrcA(value=jnp.array(1.0)))
        assert isinstance(asm, Assembly)
        assert jnp.array_equal(asm(State()).data, jnp.ones(3))
        assert asm.lit == ("a",)
        assert asm.skipped == ()  # nothing lit downstream -> no identity span

    def test_two_sources_materialize_junction(self, graph):
        asm = assemble(graph, SrcA(value=jnp.array(1.0)), SrcB(value=jnp.array(2.0)))
        assert isinstance(asm.operator, SumOperator)
        assert asm.operator.names == ("a", "b")
        assert jnp.array_equal(asm(State()).data, jnp.full(3, 3.0))

    def test_skip_transform_between_lit_nodes(self, graph):
        asm = assemble(graph, SrcA(value=jnp.array(1.0)), MulT2(factor=jnp.array(10.0)))
        assert jnp.array_equal(asm(State()).data, jnp.full(3, 10.0))
        assert "t1" in asm.skipped  # traversed as identity between lit nodes

    def test_mid_trunk_junction_upstream_becomes_branch(self, graph):
        """Trunk flowing into a junction folds as branch 0 of the Sum."""
        asm = assemble(
            graph,
            SrcA(value=jnp.array(1.0)),
            MulT1(factor=jnp.array(2.0)),
            SrcC(value=jnp.array(5.0)),
        )
        assert isinstance(asm.operator, SumOperator)
        assert jnp.array_equal(asm(State()).data, jnp.full(3, 7.0))  # 1*2 + 5

    def test_pure_transform_chain_on_caller_data(self, graph):
        asm = assemble(graph, MulT1(factor=jnp.array(2.0)), MulT2(factor=jnp.array(3.0)))
        assert not asm.has_source
        out = asm(State(data=jnp.ones(3)))
        assert jnp.array_equal(out.data, jnp.full(3, 6.0))

    def test_transform_rooted_sum_branch_rejected(self, graph):
        with pytest.raises(AssemblyError, match="no live source"):
            assemble(graph, MulT1(factor=jnp.array(2.0)), SrcC(value=jnp.array(1.0)))

    def test_full_graph(self, graph):
        asm = assemble(
            graph,
            SrcA(value=jnp.array(1.0)), SrcB(value=jnp.array(2.0)),
            MulT1(factor=jnp.array(10.0)), SrcC(value=jnp.array(4.0)),
            MulT2(factor=jnp.array(0.5)),
        )
        # ((1+2)*10 + 4) * 0.5 = 17
        assert jnp.array_equal(asm(State()).data, jnp.full(3, 17.0))


class TestCallerDataGuards:
    def test_sourced_assembly_rejects_caller_data(self, graph):
        asm = assemble(graph, SrcA(value=jnp.array(1.0)))
        with pytest.raises(AssemblyError, match="discarded"):
            asm(State(data=jnp.ones(3)))

    def test_transform_chain_requires_caller_data(self, graph):
        asm = assemble(graph, MulT1(factor=jnp.array(2.0)))
        with pytest.raises(AssemblyError, match="needs caller-supplied"):
            asm(State())


class TestDeterminism:
    def test_argument_order_is_irrelevant(self, graph):
        ops = [SrcA(value=jnp.array(1.0)), SrcB(value=jnp.array(2.0)),
               MulT2(factor=jnp.array(3.0))]
        asm1 = assemble(graph, *ops)
        asm2 = assemble(graph, *reversed(ops))
        assert eqx.tree_equal(asm1, asm2)
        s = State(key=jax.random.key(0))
        assert jnp.array_equal(asm1(s).data, asm2(s).data)

    def test_branch_order_is_graph_declaration_order(self, graph):
        asm = assemble(graph, SrcB(value=jnp.array(2.0)), SrcA(value=jnp.array(1.0)))
        assert asm.operator.names == ("a", "b")  # not call order


class TestAssemblyErgonomics:
    def test_node_id_access(self, graph):
        asm = assemble(
            graph, SrcA(value=jnp.array(1.0)), SrcC(value=jnp.array(2.0)),
            MulT2(factor=jnp.array(3.0)),
        )
        assert asm["a"].value == 1.0
        assert asm["t2"].factor == 3.0
        with pytest.raises(KeyError, match="ghost"):
            asm["ghost"]

    def test_replace_node(self, graph):
        asm = assemble(graph, SrcA(value=jnp.array(1.0)), MulT2(factor=jnp.array(3.0)))
        asm2 = asm.replace_node("a", SrcA(value=jnp.array(10.0)))
        assert jnp.array_equal(asm2(State()).data, jnp.full(3, 30.0))
        assert asm["a"].value == 1.0  # original untouched

    def test_jit_and_grad(self, graph):
        asm = assemble(graph, SrcA(value=jnp.array(2.0)), MulT2(factor=jnp.array(3.0)))
        out = eqx.filter_jit(asm)(State())
        assert jnp.array_equal(out.data, jnp.full(3, 6.0))

        def loss(a):
            return jnp.sum(a(State()).data)

        grads = eqx.filter_grad(loss)(asm)
        assert jnp.allclose(grads["a"].value, 9.0)  # d(3*v*3)/dv

    def test_nests_in_pipeline(self, graph):
        asm = assemble(graph, SrcA(value=jnp.array(1.0)))
        from replicant.core.operator import LambdaOperator

        outer = Pipeline(asm, LambdaOperator.on_data(lambda d: d + 1))
        assert jnp.array_equal(outer(State()).data, jnp.full(3, 2.0))

    def test_repr_lists_skipped(self, graph):
        asm = assemble(graph, SrcA(value=jnp.array(1.0)), MulT2(factor=jnp.array(2.0)))
        assert "t1" in repr(asm)

    def test_mermaid_render(self, graph):
        register_graph(graph)
        asm = assemble(graph, SrcA(value=jnp.array(1.0)), MulT2(factor=jnp.array(2.0)))
        mm = asm.to_mermaid()
        assert "class a lit" in mm
        assert "class t1 wire" in mm
        assert "class b dim" in mm

    def test_traversed_junctions_render_as_wire(self, graph):
        """Pass-through junctions are part of the lit path, not dead nodes."""
        register_graph(graph)
        asm = assemble(graph, SrcA(value=jnp.array(1.0)), MulT2(factor=jnp.array(2.0)))
        mm = asm.to_mermaid()
        assert "class j1 wire" in mm
        assert "class j2 wire" in mm

    def test_single_operator_assembly_node_access(self, graph):
        """Regression: a one-node assembly must expose its node id."""
        asm = assemble(graph, SrcA(value=jnp.array(1.0)))
        assert asm["a"].value == 1.0
        asm2 = asm.replace_node("a", SrcA(value=jnp.array(7.0)))
        assert jnp.array_equal(asm2(State()).data, jnp.full(3, 7.0))

    def test_node_lookup_prefers_outer_fold_level(self, graph):
        """Regression: user-internal stage names must not shadow graph nodes."""
        inner = Pipeline(
            Mul(factor=jnp.array(1.0)), Mul(factor=jnp.array(2.0)),
            names=("prep", "t2"),  # deliberately collides with graph node t2
        )
        asm = assemble(
            graph,
            SrcA(value=jnp.array(1.0)),
            At("t1", inner),
            MulT2(factor=jnp.array(10.0)),
        )
        assert isinstance(asm["t2"], MulT2)  # the graph node, not the inner stage

    def test_duplicate_edges_rejected(self):
        with pytest.raises(AssemblyError, match="duplicate edges"):
            SignalGraph(
                "dup",
                {"a": NodeSpec(S), "j": NodeSpec(J), "t": NodeSpec(T)},
                [("a", "j"), ("a", "j"), ("j", "t")],
            )


class TestRegionCoverage:
    """One operator covering a contiguous region of the template."""

    def test_region_replaces_its_segment(self, graph):
        """At(('a','j1','t1'), op): op implements source+sum+transform at once."""
        region_op = Src(value=jnp.array(6.0))  # pretends to be the whole chain
        asm = assemble(graph, At(("a", "j1", "t1"), region_op), SrcC(value=jnp.array(4.0)))
        # region output joins j2 as a sourced branch alongside c
        assert jnp.array_equal(asm(State()).data, jnp.full(3, 10.0))
        assert set(("a", "j1", "t1")) <= set(asm.lit)

    def test_region_addressed_by_last_node(self, graph):
        region_op = Src(value=jnp.array(6.0))
        asm = assemble(graph, At(("a", "j1", "t1"), region_op))
        assert asm["t1"] is region_op or asm["t1"].value == 6.0

    def test_live_branch_into_region_interior_rejected(self, graph):
        """b feeds j1, which is interior to the region -> atomicity error."""
        with pytest.raises(AssemblyError, match="covered by the region"):
            assemble(
                graph,
                At(("a", "j1", "t1"), Src(value=jnp.array(1.0))),
                SrcB(value=jnp.array(2.0)),
            )

    def test_non_contiguous_region_rejected(self, graph):
        with pytest.raises(AssemblyError, match="contiguous"):
            assemble(graph, At(("a", "t1"), Src(value=jnp.array(1.0))))

    def test_region_overlap_with_instance_rejected(self, graph):
        with pytest.raises(AssemblyError, match="disjoint"):
            assemble(
                graph,
                At(("a", "j1", "t1"), Src(value=jnp.array(1.0))),
                MulT1(factor=jnp.array(2.0)),
            )

    def test_region_endpoint_on_junction_rejected(self, graph):
        with pytest.raises(AssemblyError, match="interior"):
            assemble(graph, At(("a", "j1"), Src(value=jnp.array(1.0))))

    def test_forking_interior_makes_region_unclosed(self):
        """Regression: a region may not hide a signal that forks out of it."""
        forked = SignalGraph(
            "forked",
            {"a": NodeSpec(S), "x": NodeSpec(T), "y": NodeSpec(T),
             "z": NodeSpec(T), "j": NodeSpec(J)},
            [("a", "x"), ("x", "y"), ("x", "z"), ("y", "j"), ("z", "j")],
        )
        with pytest.raises(AssemblyError, match="not closed"):
            assemble(forked, At(("a", "x", "y"), Src(value=jnp.array(1.0))))

    def test_unsourced_region_error_names_region_root(self):
        """Regression: the culprit in the error is the region's FIRST node."""
        g = SignalGraph(
            "chain",
            {"p": NodeSpec(T), "q": NodeSpec(T), "c": NodeSpec(S),
             "j": NodeSpec(J), "t": NodeSpec(T)},
            [("p", "q"), ("q", "j"), ("c", "j"), ("j", "t")],
        )
        with pytest.raises(AssemblyError, match="'p'"):
            assemble(
                g,
                At(("p", "q"), Mul(factor=jnp.array(2.0))),
                At("c", Src(value=jnp.array(1.0))),
            )

    def test_transform_rooted_region_consumes_upstream(self, graph):
        """A region starting on a transform chains onto the incoming signal."""
        region_op = Mul(factor=jnp.array(5.0))  # covers t2..t3 as one unit
        asm = assemble(
            graph,
            SrcA(value=jnp.array(2.0)),
            At(("t2", "t3"), region_op),
        )
        assert jnp.array_equal(asm(State()).data, jnp.full(3, 10.0))
        assert asm["t3"].factor == 5.0


class TestManyNodes:
    @pytest.fixture
    def many_graph(self):
        return SignalGraph(
            "many-graph",
            {"a": NodeSpec(S, many=True), "b": NodeSpec(S), "j": NodeSpec(J),
             "t": NodeSpec(T, many=True)},
            [("a", "j"), ("b", "j"), ("j", "t")],
        )

    def test_multi_instance_source(self, many_graph):
        asm = assemble(
            many_graph,
            At("a", Src(value=jnp.array(1.0))),
            At("a", Src(value=jnp.array(2.0))),
            At("b", Src(value=jnp.array(4.0))),
        )
        assert jnp.array_equal(asm(State()).data, jnp.full(3, 7.0))

    def test_multi_instance_transform_chains_in_call_order(self, many_graph):
        asm = assemble(
            many_graph,
            At("a", Src(value=jnp.array(2.0))),
            At("t", Mul(factor=jnp.array(3.0))),
            At("t", Mul(factor=jnp.array(5.0))),
        )
        assert jnp.array_equal(asm(State()).data, jnp.full(3, 30.0))
