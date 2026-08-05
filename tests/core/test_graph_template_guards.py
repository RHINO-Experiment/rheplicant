"""Template-shape refusals, and the one that turned out to be unreachable.

``SignalGraph.__init__`` validates the template's shape before anything is
assembled on it, and ``get_graph`` is the lookup ``Assembly.without`` re-runs
through. A coverage run found three of these refusals never executed:

* ``Transform node {n!r} must have in-degree <= 1`` -- one of three in-degree
  guards in ``_validate``, all raising ``AssemblyError``, so this file checks
  they stay distinguishable from each other rather than only that each fires;
* ``No registered SignalGraph named {name!r}`` -- the registry miss;
* ``Nothing to assemble: no provided node reaches the sink`` -- which is not a
  missing test. It cannot be reached, and the last section here is the
  evidence for that claim rather than a test of the line.
"""

import itertools
from typing import ClassVar

import jax
import jax.numpy as jnp
import pytest

from rheplicant.core import graph as graph_module
from rheplicant.core.graph import (
    AssemblyError,
    At,
    NodeSpec,
    SignalGraph,
    assemble,
    get_graph,
    register_graph,
)
from rheplicant.core.operator import AbstractOperator

S, T, J, SEL = "source", "transform", "junction", "selector"


class Src(AbstractOperator):
    """A source carrying a distinct value, so branches stay tellable apart."""

    graph_node: ClassVar[str | None] = None
    value: jax.Array

    def __call__(self, state):
        return state.with_data(self.value * jnp.ones(3))


class Tr(AbstractOperator):
    graph_node: ClassVar[str | None] = None
    factor: jax.Array

    def __call__(self, state):
        return state.with_data(state.data * self.factor)


# ---------------------------------------------------------------------------
# in-degree guards
# ---------------------------------------------------------------------------


def test_a_transform_fed_by_two_nodes_is_refused():
    with pytest.raises(AssemblyError, match=r"Transform node 't' must have in-degree <= 1"):
        SignalGraph(
            "two-into-one",
            {"a": NodeSpec(S), "b": NodeSpec(S), "t": NodeSpec(T)},
            [("a", "t"), ("b", "t")],
        )


def test_the_refusal_reports_the_in_degree_it_found():
    """Not just that it was too high -- three parents must read as three.

    The count is the only part of the sentence that tells a template author how
    far off the declaration is, and it is computed separately from the test the
    guard makes.
    """
    with pytest.raises(AssemblyError) as excinfo:
        SignalGraph(
            "three-into-one",
            {"a": NodeSpec(S), "b": NodeSpec(S), "c": NodeSpec(S), "t": NodeSpec(T)},
            [("a", "t"), ("b", "t"), ("c", "t")],
        )
    assert "got 3" in str(excinfo.value), str(excinfo.value)


def test_a_transform_with_one_parent_is_accepted():
    """The other branch: the guard is ``indeg > 1``, not ``indeg != 1``."""
    template = SignalGraph(
        "one-into-one", {"a": NodeSpec(S), "t": NodeSpec(T)}, [("a", "t")]
    )
    assert template.sink == "t"


def test_a_parentless_transform_is_accepted_although_the_docstring_says_otherwise():
    """Characterisation, not endorsement.

    ``SignalGraph``'s own docstring says transforms have in-degree *exactly*
    one; the guard only refuses in-degree above one, so a transform with no
    parent builds. Assembly handles it -- such a node contracts to identity
    when nothing is placed on it, and an operator placed there is fed
    ``data=None`` and refused by ``_check_slot_kinds`` -- but the template-level
    asymmetry is real and is pinned here so that tightening it later is a
    visible decision rather than a silent one.
    """
    template = SignalGraph("lonely-transform", {"t": NodeSpec(T)}, [])
    assert template.sink == "t"


def test_the_three_in_degree_refusals_differ_from_each_other():
    """One method, three guards, one exception type.

    ``pytest.raises(AssemblyError, match="in-degree")`` is satisfied by all
    three, and would still be satisfied if two of them had been merged into one
    over-broad sentence -- which is the failure a per-guard substring match
    cannot see.
    """
    messages = {}
    with pytest.raises(AssemblyError) as excinfo:
        SignalGraph(
            "source-with-parent",
            {"a": NodeSpec(S), "b": NodeSpec(S), "t": NodeSpec(T)},
            [("a", "b"), ("b", "t")],
        )
    messages["source"] = str(excinfo.value)
    with pytest.raises(AssemblyError) as excinfo:
        SignalGraph(
            "transform-with-two-parents",
            {"a": NodeSpec(S), "b": NodeSpec(S), "t": NodeSpec(T)},
            [("a", "t"), ("b", "t")],
        )
    messages["transform"] = str(excinfo.value)
    with pytest.raises(AssemblyError) as excinfo:
        SignalGraph(
            "junction-with-one-parent",
            {"a": NodeSpec(S), "j": NodeSpec(J)},
            [("a", "j")],
        )
    messages["junction"] = str(excinfo.value)

    assert len(set(messages.values())) == 3, messages
    assert "Source node" in messages["source"]
    assert "Transform node" in messages["transform"]
    assert "Junction node" in messages["junction"]


# ---------------------------------------------------------------------------
# the template registry
# ---------------------------------------------------------------------------


@pytest.fixture
def registry(monkeypatch):
    """A copy of the process-wide template registry, discarded after the test."""
    monkeypatch.setattr(graph_module, "_GRAPHS", dict(graph_module._GRAPHS))
    return graph_module


def test_an_unregistered_name_is_refused(registry):
    with pytest.raises(KeyError, match="No registered SignalGraph named"):
        get_graph("no-such-template")


def test_the_refusal_lists_the_templates_that_are_registered(registry):
    """The half of the sentence a ``match=`` on the first clause never reaches.

    ``known:`` exists so that a typo is diagnosable from the message alone. A
    guard that reported an empty list, or somebody else's registry, would
    satisfy every ``pytest.raises`` above.
    """
    register_graph(SignalGraph("registered-here", {"a": NodeSpec(S)}, []))
    with pytest.raises(KeyError) as excinfo:
        get_graph("registered-hear")
    message = str(excinfo.value)
    assert "registered-hear" in message, message
    assert "registered-here" in message, message


def test_a_registered_name_is_returned(registry):
    """The other branch, and the identity: the lookup returns THAT template."""
    template = register_graph(SignalGraph("round-trip", {"a": NodeSpec(S)}, []))
    assert get_graph("round-trip") is template


# ---------------------------------------------------------------------------
# "Nothing to assemble" -- the evidence that it cannot happen
# ---------------------------------------------------------------------------


def _single_sink_templates(max_nodes: int = 5):
    """Every single-sink template on 2..max_nodes nodes, junction and selector.

    Kinds are forced by in-degree, which is what ``_validate`` requires anyway:
    in-degree 0 is a source, 1 a transform, >= 2 a junction or a selector.
    """
    for n in range(2, max_nodes + 1):
        names = [f"n{i}" for i in range(n)]
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        for size in range(n - 1, len(pairs) + 1):
            for edges in itertools.combinations(pairs, size):
                indeg, outdeg = [0] * n, [0] * n
                for a, b in edges:
                    indeg[b] += 1
                    outdeg[a] += 1
                if sum(1 for i in range(n) if outdeg[i] == 0) != 1:
                    continue
                for fan_in_kind in (J, SEL):
                    nodes = {
                        names[i]: NodeSpec(
                            S if indeg[i] == 0 else (T if indeg[i] == 1 else fan_in_kind)
                        )
                        for i in range(n)
                    }
                    yield SignalGraph(
                        "probe", nodes, [(names[a], names[b]) for a, b in edges]
                    )


def test_every_node_of_a_valid_template_reaches_the_sink():
    """The structural fact the unreachability argument rests on.

    A template is a finite DAG (``_toposort`` refuses cycles) with exactly one
    node of out-degree zero (``_validate`` refuses any other count). Following
    out-edges from any node therefore terminates, and can only terminate there.
    """
    checked = 0
    for template in _single_sink_templates():
        for node in template.nodes:
            seen, frontier = set(), [node]
            while frontier:
                for successor in template._out[frontier.pop()]:
                    if successor not in seen:
                        seen.add(successor)
                        frontier.append(successor)
            assert node == template.sink or template.sink in seen, (
                f"{node!r} does not reach {template.sink!r} in {template.edges}"
            )
        checked += 1
    assert checked > 100, f"only {checked} templates generated -- the search collapsed"


def test_an_empty_assembly_is_refused_before_the_sink_is_consulted():
    """Why "at least one node is live" is a precondition and not a hope.

    The branded refusal for "no operators" is this one, not the sink guard, and
    the sink guard is downstream of it.
    """
    template = SignalGraph("trivial", {"a": NodeSpec(S)}, [])
    with pytest.raises(AssemblyError, match="assemble\\(\\) needs at least one operator"):
        assemble(template)


def test_no_single_placement_on_any_small_template_empties_the_sink():
    """The search behind the ``pragma: no cover`` in ``assemble``.

    Every single-sink template on up to five nodes, junction and selector
    variants, every legal single placement and every contiguous two- and
    three-node region: ``assemble`` either succeeds or refuses for some other,
    branded reason. It never says "Nothing to assemble".

    This is the honest form of the claim. It does not prove unreachability --
    it pins the argument in the source comment against the code, so that an
    edit which makes the sink reachable-and-empty fails here rather than
    silently turning a live pragma into a dead one.
    """
    attempts = 0
    for template in _single_sink_templates():
        placements = []
        for node, spec in template.nodes.items():
            if spec.kind == S:
                placements.append((At(node, Src(value=jnp.asarray(2.0))),))
            elif spec.kind == T:
                placements.append((At(node, Tr(factor=jnp.asarray(3.0))),))
        edge_set = set(template.edges)
        for a, b in edge_set:
            paths = [(a, b)] + [(a, b, c) for c in template.nodes if (b, c) in edge_set]
            for path in paths:
                placements.append((At(path, Src(value=jnp.asarray(5.0))),))
                placements.append((At(path, Tr(factor=jnp.asarray(7.0))),))
        for args in placements:
            attempts += 1
            try:
                built = assemble(template, *args)
            except AssemblyError as error:
                assert "Nothing to assemble" not in str(error), (
                    f"reached the unreachable guard: {template.edges} with {args}"
                )
                continue
            assert built.operator is not None
    assert attempts > 500, f"only {attempts} placements tried -- the search collapsed"


def test_a_live_placement_reaches_the_sink_and_runs():
    """The other branch, spelled out once on a concrete template.

    Two sources into a junction, then a transform to the sink. The values are
    asymmetric (2 and 5, scaled by 3) so that a fold which dropped a branch,
    doubled one, or applied the transform to the wrong side is a different
    number rather than the same one.
    """
    from rheplicant.core.state import State

    template = SignalGraph(
        "sum-then-scale",
        {"a": NodeSpec(S), "b": NodeSpec(S), "j": NodeSpec(J), "t": NodeSpec(T)},
        [("a", "j"), ("b", "j"), ("j", "t")],
    )
    built = assemble(
        template,
        At("a", Src(value=jnp.asarray(2.0))),
        At("b", Src(value=jnp.asarray(5.0))),
        At("t", Tr(factor=jnp.asarray(3.0))),
    )
    out = built(State())
    assert jnp.allclose(out.data, jnp.full(3, (2.0 + 5.0) * 3.0))
