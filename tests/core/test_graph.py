"""Tests for SignalGraph templates and graph-guided assembly."""

from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from rheplicant.core.combinators import SumOperator
from rheplicant.core.graph import (
    AmbiguousNodeError,
    Assembly,
    AssemblyError,
    At,
    NodeSpec,
    SignalGraph,
    assemble,
    register_graph,
)
from rheplicant.core.operator import AbstractOperator
from rheplicant.core.pipeline import Pipeline
from rheplicant.core.state import State

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
        from rheplicant.core.operator import LambdaOperator

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


class TestCompositionSymbols:
    """A sum and a switch are operations ON operators, and must not look like one.

    The three composition structures are drawn to one convention: a cascade is
    an arrow, a sum is a symbol the wire runs through, a switch is a *different*
    symbol the wire runs through. Nothing enforced it, and the package drifted
    into drawing both operations as identical circles labelled ``+`` and ``sw``
    — two operations rendered as peers of the operator boxes around them, and
    told apart only by reading two characters.
    """

    @pytest.fixture
    def both(self):
        """src -> (junction) <- src ; junction -> (selector) <- src ; -> sink."""
        return SignalGraph(
            "sum-and-switch",
            {
                "p": NodeSpec(S), "q": NodeSpec(S), "sum": NodeSpec(J),
                "load": NodeSpec(S), "switch": NodeSpec("selector"),
                "out": NodeSpec(T),
            },
            [("p", "sum"), ("q", "sum"), ("sum", "switch"), ("load", "switch"),
             ("switch", "out")],
        )

    def test_mermaid_gives_them_different_shapes(self, both):
        """Different SHAPES, not one shape with two labels."""
        lines = {
            line.split("(")[0].split("{")[0].split("[")[0].strip(): line.strip()
            for line in both.to_mermaid().splitlines()
            if line.startswith("  ") and "-->" not in line and "class" not in line
        }
        sum_shape = lines["sum"].removeprefix("sum")
        switch_shape = lines["switch"].removeprefix("switch")
        box_shape = lines["out"].removeprefix("out")
        assert sum_shape != switch_shape
        # A label alone would satisfy "different"; strip the labels and the
        # shapes must still differ, which is the property a reader relies on.
        assert {sum_shape[0], switch_shape[0]} == {"(", "{"}
        assert box_shape.startswith("[") and box_shape != sum_shape

    def test_svg_draws_only_operators_as_boxes(self, both):
        """One <rect> per operator slot, and none for the two operations."""
        svg = both.to_svg()
        slots = [n for n, s in both.nodes.items() if s.kind in ("source", "transform")]
        assert svg.count("<rect") == len(slots)
        # The sum is a circle, the switch is not — otherwise they would again be
        # one shape distinguished by its contents.
        assert svg.count("<circle") == 1 + 2  # the sum, plus the switch's terminals

    def test_svg_exposes_stable_node_identity_and_slot_accessibility(self, both):
        """A GUI consumes the renderer's ids; it must not recover them from labels."""
        svg = both.to_svg()
        for node_id, spec in both.nodes.items():
            assert svg.count(f'data-node-id="{node_id}"') == 1
            assert f'data-node-kind="{spec.kind}"' in svg
        assert svg.count('role="button"') == 4
        assert svg.count('tabindex="0"') == 4
        assert 'data-node-id="sum" aria-disabled="true"' in svg
        assert 'data-node-id="switch" aria-disabled="true"' in svg
        assert "source node p" in svg
        assert "junction node sum" in svg

    def test_svg_symbols_are_smaller_than_a_box_and_the_wire_reaches_them(self, both):
        """The wire must arrive AT the symbol, not stop where a box would end."""
        from rheplicant.core.render import _NODE_H, _SUM_R, _SW_R, _half_height

        assert _SUM_R < _NODE_H / 2 and _SW_R < _NODE_H / 2
        assert _half_height("junction") == _SUM_R
        assert _half_height("selector") == _SW_R
        assert _half_height("transform") == _half_height("source") == _NODE_H / 2

    def test_theme_selects_a_palette_and_an_unknown_one_is_refused(self, both):
        assert both.to_svg(theme="dark") != both.to_svg(theme="light")
        with pytest.raises(ValueError, match="Unknown theme"):
            both.to_svg(theme="solarized")


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


class TestManyNodeAddressing:
    """A ``many`` node id addresses one operator or none — never "whichever".

    With one instance the bare id IS the address (unchanged). With two, the
    bare id would have to answer for both: the fold that sums them, or an
    arbitrary one of them. Either answer is a finite, correctly-shaped, wrong
    handle — through ``replace_node`` it deletes a component outright — so the
    bare id becomes an error that names the per-instance ids instead.
    """

    @pytest.fixture
    def many_graph(self):
        return SignalGraph(
            "many-addressing",
            {"a": NodeSpec(S, many=True), "b": NodeSpec(S), "j": NodeSpec(J),
             "t": NodeSpec(T, many=True)},
            [("a", "j"), ("b", "j"), ("j", "t")],
        )

    @pytest.fixture
    def two_at_a(self, many_graph):
        """10 + 20 at the many source ``a``; nothing else live."""
        return assemble(
            many_graph,
            At("a", Src(value=jnp.array(10.0))),
            At("a", Src(value=jnp.array(20.0))),
        )

    def test_single_instance_id_still_addresses_the_operator(self, many_graph):
        """The constraint everything else must not break."""
        asm = assemble(many_graph, At("a", Src(value=jnp.array(10.0))))
        assert isinstance(asm["a"], Src)
        assert asm["a"].value == 10.0

    def test_ambiguous_source_id_raises_and_names_the_instances(self, two_at_a):
        with pytest.raises(AmbiguousNodeError) as excinfo:
            two_at_a["a"]
        message = str(excinfo.value)
        assert "a_1" in message and "a_2" in message

    def test_instance_ids_address_the_instances(self, two_at_a):
        assert two_at_a["a_1"].value == 10.0
        assert two_at_a["a_2"].value == 20.0

    def test_replace_node_on_ambiguous_id_refuses_without_deleting(self, two_at_a):
        """The measured bug: this used to return 0.0 — instance 2 deleted."""
        before = two_at_a(State()).data
        assert jnp.array_equal(before, jnp.full(3, 30.0))
        with pytest.raises(AmbiguousNodeError):
            two_at_a.replace_node("a", Src(value=jnp.array(0.0)))
        assert jnp.array_equal(two_at_a(State()).data, before)

    def test_replace_node_by_instance_id_keeps_the_sibling(self, two_at_a):
        """What the error message tells you to write actually works."""
        swapped = two_at_a.replace_node("a_1", Src(value=jnp.array(0.0)))
        assert jnp.array_equal(swapped(State()).data, jnp.full(3, 20.0))
        assert jnp.array_equal(two_at_a(State()).data, jnp.full(3, 30.0))

    def test_ambiguous_transform_id_raises(self, many_graph):
        asm = assemble(
            many_graph,
            At("a", Src(value=jnp.array(2.0))),
            At("t", Mul(factor=jnp.array(3.0))),
            At("t", Mul(factor=jnp.array(5.0))),
        )
        with pytest.raises(AmbiguousNodeError, match="t_1"):
            asm["t"]
        assert asm["t_1"].factor == 3.0
        assert asm["t_2"].factor == 5.0

    def test_multi_instance_forward_output_is_unchanged(self, many_graph):
        """Renaming instances is a naming change only: the physics is bitwise."""
        ops = [At("a", Src(value=jnp.array(1.0))), At("a", Src(value=jnp.array(2.0))),
               At("b", Src(value=jnp.array(4.0)))]
        asm = assemble(many_graph, *ops)
        hand = SumOperator(
            SumOperator(Src(value=jnp.array(1.0)), Src(value=jnp.array(2.0))),
            Src(value=jnp.array(4.0)),
        )
        state = State(key=jax.random.key(0))
        assert jnp.array_equal(asm(state).data, hand(state).data)

    def test_assembly_reports_its_multiplicity(self, two_at_a, many_graph):
        """`lit` alone said the same thing for one instance and for two."""
        one = assemble(many_graph, At("a", Src(value=jnp.array(10.0))))
        assert one.lit == two_at_a.lit  # the template node is lit either way
        assert one.instances == ()
        assert two_at_a.instances == (("a", ("a_1", "a_2")),)
        assert "x2" in repr(two_at_a) and "x2" not in repr(one)

    def test_mermaid_shows_the_multiplicity(self, two_at_a, many_graph):
        register_graph(many_graph)
        one = assemble(many_graph, At("a", Src(value=jnp.array(10.0))))
        assert "(x2)" in two_at_a.to_mermaid()
        assert "(x2)" not in one.to_mermaid()
        assert "(x2)" in two_at_a.to_svg()


class TestFoldReplacement:
    """``replace_node`` on a junction would discard every branch feeding it."""

    def test_replace_node_on_a_materialized_junction_refuses(self, graph):
        asm = assemble(graph, SrcA(value=jnp.array(10.0)), SrcB(value=jnp.array(20.0)))
        before = asm(State()).data
        assert jnp.array_equal(before, jnp.full(3, 30.0))
        with pytest.raises(AssemblyError, match="junction"):
            asm.replace_node("j1", SrcA(value=jnp.array(0.0)))
        assert jnp.array_equal(asm(State()).data, before)

    def test_reading_a_materialized_junction_still_works(self, graph):
        """Reading the fold is how you inspect branch order; only writing lies."""
        asm = assemble(graph, SrcA(value=jnp.array(10.0)), SrcB(value=jnp.array(20.0)))
        assert isinstance(asm["j1"], SumOperator)
        assert asm["j1"].names == ("a", "b")


@pytest.fixture
def many_source_graph():
    """``a`` is a ``many`` source reaching the sink by exactly ONE path."""
    return SignalGraph(
        "many-source",
        {"a": NodeSpec(S, many=True), "b": NodeSpec(S), "j": NodeSpec(J),
         "t": NodeSpec(T)},
        [("a", "j"), ("b", "j"), ("j", "t")],
    )


@pytest.fixture
def fork_rejoin_graph():
    """``x`` reaches the junction by TWO paths: ``x -> p -> j`` and ``x -> q -> j``.

    The shape that makes a node's operator appear twice in the folded tree, and
    that makes the fold mint a repeated branch label — the two things that break
    addressing.
    """
    return SignalGraph(
        "fork-rejoin",
        {
            "x": NodeSpec(S, many=True),
            "p": NodeSpec(T),
            "q": NodeSpec(T),
            "j": NodeSpec(J),
            "out": NodeSpec(T),
        },
        [("x", "p"), ("x", "q"), ("p", "j"), ("q", "j"), ("j", "out")],
    )


@pytest.fixture
def id_collision_graph():
    """``x`` (many, 2 instances) sits beside a REAL node literally named ``x_1``.

    ``x`` reaches the sink by exactly ONE path, so ``_fold_duplicates`` reports
    nothing for it and the "folded in twice" branch of ``_check_promised_ids``
    never fires here — unlike ``fork_rejoin_graph``. The only thing that can
    catch ``x_1`` (the id ``_instance_names`` mints for instance 1 of ``x``)
    resolving to the unrelated node ``x_1`` instead is the identity round-trip
    half of the guard.
    """
    return SignalGraph(
        "collide-with-a-real-node",
        {
            "x": NodeSpec(S, many=True),
            "x_1": NodeSpec(S),
            "j": NodeSpec(J),
            "t": NodeSpec(T),
        },
        [("x", "j"), ("x_1", "j"), ("j", "t")],
    )


class TestPromisedIdsAddressTheirOwnInstance:
    """Every id :class:`AmbiguousNodeError` hands out must reach that instance.

    ``_instance_names`` mints ``x_1..x_n``. ``_dedup`` independently mints
    ``x, x_2, x_3, ...`` for repeated branch labels, and the two namespaces
    overlap from ``_2`` on. ``_find_named`` is breadth-first, so the outer
    ``_dedup`` label wins: ``x_2`` reached a fold over a whole path rather than
    instance 2, and ``replace_node("x_2", ...)`` — literally what the error
    message tells the caller to write — rewrote that path instead. Following
    the instructions deleted physics, which is worse than not being told.
    """

    def test_named_ids_resolve_to_the_very_objects_placed(self, many_source_graph):
        """The contract, checked by identity: not an equal operator, THE one."""
        placed = [Src(value=jnp.array(10.0)), Src(value=jnp.array(20.0))]
        asm = assemble(many_source_graph, *(At("a", op) for op in placed))
        ((nid, names),) = asm.instances
        assert nid == "a"
        for name, op in zip(names, placed, strict=True):
            assert asm[name] is op

    def test_the_message_names_exactly_the_ids_that_work(self, many_source_graph):
        """The message is built from ``instances``; so is the guarantee above."""
        placed = [Src(value=jnp.array(10.0)), Src(value=jnp.array(20.0))]
        asm = assemble(many_source_graph, *(At("a", op) for op in placed))
        with pytest.raises(AmbiguousNodeError) as excinfo:
            asm["a"]
        ((_, names),) = asm.instances
        for name in names:
            assert name in str(excinfo.value)

    def test_ids_that_would_collide_are_refused_at_assemble(self, fork_rejoin_graph):
        """``x_2`` is both instance 2 and the fold's label for the second path.

        Measured before this guard: ``asm["x_2"]`` handed back a fold, and
        ``replace_node("x_2", Src(0))`` took the forward output 60 -> 30 with no
        error and no shape change — where dropping instance 2 is 20.
        """
        with pytest.raises(AssemblyError, match="x_2"):
            assemble(
                fork_rejoin_graph,
                At("x", Src(value=jnp.array(10.0))),
                At("x", Src(value=jnp.array(20.0))),
            )

    def test_the_refusal_says_the_node_is_folded_in_twice(self, fork_rejoin_graph):
        with pytest.raises(AssemblyError) as excinfo:
            assemble(
                fork_rejoin_graph,
                At("x", Src(value=jnp.array(10.0))),
                At("x", Src(value=jnp.array(20.0))),
            )
        message = str(excinfo.value)
        assert "'x'" in message and "2 paths" in message

    def test_assemble_refuses_when_a_minted_id_collides_with_a_real_node(self, id_collision_graph):
        """The identity round-trip half, caught with nothing else in play.

        ``x`` reaches the sink by ONE path here, so ``duplicates`` is empty
        and the "folded in twice" branch above never fires — this can only be
        caught by checking that ``x_1`` (minted for instance 1 of ``x``)
        resolves back to the very object placed there, not merely to
        *something*. Measured before this guard existed (relaxing the check
        to ``found is None``): assemble ACCEPTED and silently aliased
        instance 1 of ``x`` to the unrelated node ``x_1``'s own operator.
        """
        with pytest.raises(AssemblyError) as excinfo:
            assemble(
                id_collision_graph,
                At("x", Src(value=jnp.array(10.0))),
                At("x", Src(value=jnp.array(20.0))),
                At("x_1", Src(value=jnp.array(100.0))),
            )
        message = str(excinfo.value)
        assert "'x'" in message
        assert "'x_1'" in message
        assert "resolves to" in message

    def test_the_refused_advice_would_have_deleted_the_wrong_node(
        self, id_collision_graph, monkeypatch
    ):
        """Anchor the guard on the physics it protects, not just the raise.

        ``x`` reaching the sink by one path means ``_fold_duplicates`` reports
        no duplicates for this graph, so disabling ``_check_promised_ids``
        entirely is behaviourally identical, HERE, to relaxing only the
        identity half to ``found is None``: the "folded in twice" branch was
        never going to fire either way (see the previous test's fixture
        docstring). Bypassing the guard reproduces exactly what a caller
        would see under that relaxation, then follows the refused message's
        own advice — ``replace_node`` by the id it names — to show what
        accepting it would have broken: not instance 1 of ``x`` (value 10),
        but the unrelated node ``x_1`` (value 100).
        """
        import rheplicant.core.graph as graph_module

        monkeypatch.setattr(graph_module, "_check_promised_ids", lambda *a, **k: None)
        asm = assemble(
            id_collision_graph,
            At("x", Src(value=jnp.array(10.0))),
            At("x", Src(value=jnp.array(20.0))),
            At("x_1", Src(value=jnp.array(100.0))),
        )
        assert asm.aliased == ()  # the OTHER half is right: nothing is folded in twice
        before = asm(State()).data
        assert jnp.array_equal(before, jnp.full(3, 130.0))  # 10 + 20 + 100

        # 'x_1' does not address instance 1 of x -- it silently resolves to the
        # unrelated node x_1's own operator instead.
        resolved = asm["x_1"]
        assert resolved.value == 100.0  # node x_1's own operator ...
        assert resolved.value != 10.0  # ... not the promised instance-1 operator

        # What the (relaxed) message tells the caller to write: replace instance
        # 1 of x by its id.
        swapped = asm.replace_node("x_1", Src(value=jnp.array(0.0)))
        after = swapped(State()).data
        assert jnp.array_equal(after, jnp.full(3, 30.0))  # 10 + 20 + 0: node x_1 deleted

        # What the caller meant -- drop instance 1 of x -- is a different number,
        # and the advice silently produced neither an error nor that number.
        meant = assemble(
            id_collision_graph,
            At("x", Src(value=jnp.array(20.0))),
            At("x_1", Src(value=jnp.array(100.0))),
        )
        assert jnp.array_equal(meant(State()).data, jnp.full(3, 120.0))
        assert not jnp.array_equal(after, meant(State()).data)


class TestAliasedNodeIsNotWritable:
    """A node the fold placed twice cannot be written through by one id.

    ``eqx.tree_at`` rewrites the single position ``_find_named`` reaches. When
    a node's contribution reaches the sink by two paths the fold embeds its
    operator twice, so rewriting through the node id leaves the other copy
    live: a finite, correctly-shaped, wrong forward model. Reading is still
    honest — it returns the operator that genuinely sits there — so only the
    write refuses, exactly as for a materialized junction.
    """

    @pytest.fixture
    def single_at_x(self, fork_rejoin_graph):
        """One source at the fork-rejoin node: 10 down both paths, summed = 20."""
        return assemble(fork_rejoin_graph, At("x", Src(value=jnp.array(10.0))))

    def test_the_forward_model_is_untouched(self, single_at_x):
        """The guard is about addressing; the physics was never in question."""
        assert jnp.array_equal(single_at_x(State()).data, jnp.full(3, 20.0))

    def test_reading_the_aliased_node_still_works(self, single_at_x):
        assert isinstance(single_at_x["x"], Src)
        assert single_at_x["x"].value == 10.0

    def test_replace_node_refuses_instead_of_rewriting_one_copy(self, single_at_x):
        """Measured before this guard: 10.0, where zeroing ``x`` is 0.0."""
        before = single_at_x(State()).data
        with pytest.raises(AssemblyError, match="more than one"):
            single_at_x.replace_node("x", Src(value=jnp.array(0.0)))
        assert jnp.array_equal(single_at_x(State()).data, before)

    def test_a_node_reached_by_one_path_is_still_writable(self, graph):
        """The guard must not fire on the ordinary shape."""
        asm = assemble(graph, SrcA(value=jnp.array(10.0)), SrcB(value=jnp.array(20.0)))
        swapped = asm.replace_node("a", SrcA(value=jnp.array(0.0)))
        assert jnp.array_equal(swapped(State()).data, jnp.full(3, 20.0))

    def test_reusing_one_operator_object_at_two_nodes_is_not_aliasing(self, graph):
        """Placed twice on purpose is not folded twice by accident.

        The counts are compared against how often the caller placed the object,
        so this keeps working: ``_find_named`` reaches position ``a`` by name
        and ``tree_at`` rewrites that one, which is what was asked for.
        """
        shared = SrcA(value=jnp.array(10.0))
        asm = assemble(graph, At("a", shared), At("b", shared))
        assert jnp.array_equal(asm(State()).data, jnp.full(3, 20.0))
        swapped = asm.replace_node("a", SrcA(value=jnp.array(0.0)))
        assert jnp.array_equal(swapped(State()).data, jnp.full(3, 10.0))


class TestReprSurfacesFanOut:
    """The repr names fan-out nodes, and stays silent when there are none.

    ``aliased`` is the one condition the framework cannot refuse its way out of.
    ``replace_node`` and ``ParameterSpace.validate`` both consult it by name, so
    every framework write path is covered; a hand-rolled
    ``eqx.tree_at(lambda a: a[nid].x, ...)`` goes through neither, and still
    rewrites one copy of a node the fold embedded several times — leaving a
    finite, correctly-shaped, half-updated model. Printing the condition is how
    a user meets it without already knowing to ask.
    """

    def test_a_fan_out_node_is_named_in_the_repr(self, fork_rejoin_graph):
        """``x`` reaches the sink by two paths, so the fold embeds it twice."""
        asm = assemble(fork_rejoin_graph, At("x", Src(value=jnp.array(10.0))))
        assert asm.aliased == ("x",)
        assert "aliased-at-several-positions=['x']" in repr(asm)

    def test_the_ordinary_assembly_grows_no_empty_field(self, graph):
        """The common case is ``aliased == ()``.

        A field reporting that on every assembly would be noise in its own
        right, and would train the reader straight past the one place the
        warning can ever appear.
        """
        asm = assemble(graph, SrcA(value=jnp.array(10.0)), SrcB(value=jnp.array(20.0)))
        assert asm.aliased == ()
        assert "aliased" not in repr(asm)

    def test_the_empty_repr_is_unchanged(self, graph):
        """Pin the whole string, not a substring.

        The empty form is what ``docs/sky-to-receiver.md`` shows a reader, so
        it is a published interface: adding ``aliased`` must not have moved a
        character of it.
        """
        asm = assemble(graph, SrcA(value=jnp.array(10.0)), SrcB(value=jnp.array(20.0)))
        assert repr(asm) == (
            "Assembly(graph='test-graph', lit=['a', 'b'], skipped-as-identity=[])"
        )

    def test_the_non_empty_repr_is_the_empty_one_plus_one_field(
        self, fork_rejoin_graph
    ):
        """The new field is appended, so nothing a reader already parses moves."""
        asm = assemble(fork_rejoin_graph, At("x", Src(value=jnp.array(10.0))))
        assert repr(asm) == (
            "Assembly(graph='fork-rejoin', lit=['x'], skipped-as-identity=[], "
            "aliased-at-several-positions=['x'])"
        )
