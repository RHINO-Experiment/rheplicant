"""SignalGraph: declarative signal-path templates and graph-guided assembly.

The composition of a physical forward model is implicit in its signal path.
A :class:`SignalGraph` records that path once — sources, transforms, and sum
junctions — and :func:`assemble` compiles a *set* of operator instances into
the ordinary ``Pipeline`` / ``SumOperator`` nesting induced by the provided
nodes:

* absent **source** nodes are pruned;
* absent **transform** nodes contract to identity (the signal passes through);
* a **junction** with one live incoming branch passes through, with two or
  more it materializes as a ``SumOperator`` — branch order is the graph's
  edge declaration order, never the call-site order, so the same provided
  set always folds to the same tree (same names, same PRNG stream, same jit
  cache entry).

The result is an :class:`Assembly` — itself an operator — wrapping the folded
composite plus static metadata (which nodes are lit, which were skipped) for
rendering the lit/dim signal-path view. Nothing new exists at runtime: an
Assembly is inspectable, differentiable, and replaceable exactly like the
hand-built composite it compiles to.

Operators declare their home node via the ``graph_node`` ClassVar (resolved
through the MRO, so subclasses inherit it); :class:`At` overrides placement
per instance.
"""

import dataclasses
from collections.abc import Iterable, Sequence
from typing import Literal

import equinox as eqx

from rheplicant.core.combinators import SelectOperator, SumOperator
from rheplicant.core.errors import DirtError
from rheplicant.core.operator import AbstractOperator
from rheplicant.core.pipeline import Pipeline
from rheplicant.core.state import State


class AssemblyError(DirtError, ValueError):
    """A provided operator set cannot be assembled on the signal graph."""


@dataclasses.dataclass(frozen=True)
class NodeSpec:
    """One node of a signal-path template.

    Attributes:
        kind: ``"source"`` (creates data; in-degree 0), ``"transform"``
            (data -> data; in-degree 1), ``"junction"`` (sum point), or
            ``"selector"`` (switched point: one branch selected per time
            sample via ``coords.extra[<node_id>]``). Junctions and selectors
            have in-degree >= 2 and are never operator slots.
        doc: one-line description shown in renderings.
        many: sources only — allow multiple instances (folded as sibling
            Sum branches). For the sink-side ``filters``-style transform
            chain use ``many`` on a transform: instances chain in call order.
        segment: grouping label for rendering (e.g. "forward", "processing").
        reserved: node exists in the physics but has no shipped operator yet
            (an equivalent-entry placeholder leaf).
    """

    kind: Literal["source", "transform", "junction", "selector"]
    doc: str = ""
    many: bool = False
    segment: str = "forward"
    reserved: bool = False


@dataclasses.dataclass(frozen=True)
class At:
    """Place ``op`` at ``node`` regardless of its class registration.

    ``node`` may also be a tuple of node ids: the operator then *covers* that
    contiguous region of the template (it implements all of those stages at
    once). Regions are atomic — no other live branch may feed their interior —
    and are addressed by their LAST covered node id in the assembly.
    """

    node: str | tuple[str, ...]
    op: AbstractOperator


class SignalGraph:
    """An immutable signal-path template (DAG with a single sink).

    Args:
        name: template identifier (used by Assembly metadata / renderers).
        nodes: ordered ``{node_id: NodeSpec}`` mapping (order fixes ``lit``
            ordering and toposort tie-breaking).
        edges: ``(src, dst)`` pairs following signal flow. Edge declaration
            order is part of the contract: it fixes junction branch order.

    Validated at construction: DAG-ness; every node reaches a unique sink;
    junctions have in-degree >= 2; sources have in-degree 0; transforms have
    in-degree exactly 1.
    """

    def __init__(
        self,
        name: str,
        nodes: dict[str, NodeSpec],
        edges: Sequence[tuple[str, str]],
    ):
        self.name = name
        self.nodes = dict(nodes)
        self.edges = tuple(edges)
        if len(set(self.edges)) != len(self.edges):
            dupes = sorted({e for e in self.edges if self.edges.count(e) > 1})
            raise AssemblyError(f"SignalGraph {name!r} declares duplicate edges: {dupes}.")
        self._in: dict[str, tuple[str, ...]] = {n: () for n in self.nodes}
        self._out: dict[str, tuple[str, ...]] = {n: () for n in self.nodes}
        for a, b in self.edges:
            if a not in self.nodes or b not in self.nodes:
                raise AssemblyError(f"Edge ({a!r}, {b!r}) references an unknown node.")
            self._in[b] = self._in[b] + (a,)
            self._out[a] = self._out[a] + (b,)
        self._topo = self._toposort()
        self._validate()

    # -- template validation -------------------------------------------------

    def _toposort(self) -> tuple[str, ...]:
        indeg = {n: len(self._in[n]) for n in self.nodes}
        # stable Kahn: repeatedly take the first declaration-order node with indeg 0
        order, remaining = [], dict(indeg)
        while remaining:
            ready = [n for n in self.nodes if n in remaining and remaining[n] == 0]
            if not ready:
                raise AssemblyError(f"SignalGraph {self.name!r} contains a cycle.")
            n = ready[0]
            del remaining[n]
            order.append(n)
            for m in self._out[n]:
                remaining[m] -= 1
        return tuple(order)

    def _validate(self):
        sinks = [n for n in self.nodes if not self._out[n]]
        if len(sinks) != 1:
            raise AssemblyError(
                f"SignalGraph {self.name!r} must have exactly one sink, found {sinks}."
            )
        self.sink = sinks[0]
        for n, spec in self.nodes.items():
            indeg = len(self._in[n])
            if spec.kind == "source" and indeg != 0:
                raise AssemblyError(f"Source node {n!r} must have in-degree 0, got {indeg}.")
            if spec.kind == "transform" and indeg > 1:
                raise AssemblyError(
                    f"Transform node {n!r} must have in-degree <= 1, got {indeg}."
                )
            if spec.kind in ("junction", "selector") and indeg < 2:
                raise AssemblyError(
                    f"{spec.kind.capitalize()} node {n!r} must have in-degree >= 2, "
                    f"got {indeg}."
                )

    # -- rendering -----------------------------------------------------------

    def to_mermaid(self, lit: Iterable[str] = (), skipped: Iterable[str] = ()) -> str:
        """Render the template as a mermaid flowchart with lit/dim styling.

        ``lit`` nodes are highlighted, ``skipped`` (traversed-as-identity)
        nodes are half-lit, everything else is dimmed — the signal-path view
        of what an assembly simulates.
        """
        lit, skipped = set(lit), set(skipped)
        lines = ["flowchart TD"]
        for n, spec in self.nodes.items():
            label = n.replace("_", " ")
            if spec.kind == "junction":
                shape = '(("+"))'
            elif spec.kind == "selector":
                shape = '(("sw"))'
            else:
                shape = f'["{label}"]'
            lines.append(f"  {n}{shape}")
        for a, b in self.edges:
            lines.append(f"  {a} --> {b}")
        lines.append("  classDef lit fill:#FAC775,stroke:#854F0B,color:#412402;")
        lines.append("  classDef wire fill:#F1EFE8,stroke:#854F0B,color:#444441;")
        lines.append("  classDef dim fill:#F1EFE8,stroke:#B4B2A9,color:#B4B2A9;")
        for n in self.nodes:
            cls = "lit" if n in lit else ("wire" if n in skipped else "dim")
            lines.append(f"  class {n} {cls};")
        return "\n".join(lines)

    def to_html(
        self,
        lit: Iterable[str] = (),
        skipped: Iterable[str] = (),
        title: str | None = None,
    ) -> str:
        """Standalone HTML page of the template with lit/dim signal-path styling."""
        from rheplicant.core.render import signal_path_html

        return signal_path_html(self, lit=lit, skipped=skipped, title=title)

    def to_svg(
        self,
        lit: Iterable[str] = (),
        skipped: Iterable[str] = (),
        title: str | None = None,
    ) -> str:
        """Self-contained ``<svg>`` of the template, for embedding (docs, notebooks)."""
        from rheplicant.core.render import signal_path_svg

        return signal_path_svg(self, lit=lit, skipped=skipped, title=title)

    def __repr__(self) -> str:
        return f"SignalGraph({self.name!r}, {len(self.nodes)} nodes, {len(self.edges)} edges)"


_GRAPHS: dict[str, SignalGraph] = {}


def register_graph(graph: SignalGraph) -> SignalGraph:
    """Register a template so Assembly.to_mermaid can find it by name."""
    _GRAPHS[graph.name] = graph
    return graph


def get_graph(name: str) -> SignalGraph:
    if name not in _GRAPHS:
        raise KeyError(f"No registered SignalGraph named {name!r}; known: {list(_GRAPHS)}")
    return _GRAPHS[name]


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------


class Assembly(AbstractOperator):
    """A graph-assembled operator: the folded composite + lit-node metadata.

    Call it like any operator. Access the operator placed at a node with
    ``assembly[node_id]`` (independent of fold nesting), swap one with
    :meth:`replace_node`, render the lit/dim signal path with
    :meth:`to_mermaid`.

    If the assembly contains live sources it *generates* its data — calling
    it on a state that already carries data raises, because that data would
    be silently discarded (pass ``data=None``). Source-free assemblies are
    transform chains operating on caller data.
    """

    operator: AbstractOperator
    graph_name: str = eqx.field(static=True)
    lit: tuple[str, ...] = eqx.field(static=True)
    skipped: tuple[str, ...] = eqx.field(static=True)
    has_source: bool = eqx.field(static=True)
    root_label: str = eqx.field(static=True, default="")

    def __call__(self, state: State) -> State:
        if self.has_source and state.data is not None:
            raise AssemblyError(
                "This assembly contains source operators and generates its own data; "
                "caller-supplied state.data would be discarded. Pass a state with "
                "data=None (or drop the sources to build a transform chain)."
            )
        if not self.has_source and state.data is None:
            raise AssemblyError(
                "This assembly is a pure transform chain (no source operators); "
                "it needs caller-supplied state.data to act on."
            )
        return self.operator(state)

    def __getitem__(self, node_id: str) -> AbstractOperator:
        if node_id and node_id == self.root_label:
            return self.operator
        found = _find_named(self.operator, node_id)
        if found is None:
            raise KeyError(f"No node named {node_id!r} in this assembly; lit: {self.lit}")
        return found

    def replace_node(self, node_id: str, operator: AbstractOperator) -> "Assembly":
        """Return a new Assembly with the operator at ``node_id`` swapped."""
        target = self[node_id]

        def where(a: "Assembly") -> AbstractOperator:
            return a[node_id]

        del target  # existence check only
        return eqx.tree_at(where, self, operator)

    def to_mermaid(self) -> str:
        """Lit/dim mermaid rendering via the registered template."""
        return get_graph(self.graph_name).to_mermaid(lit=self.lit, skipped=self.skipped)

    def to_html(self, title: str | None = None) -> str:
        """Standalone HTML page: the full graph with this assembly's nodes lit."""
        return get_graph(self.graph_name).to_html(
            lit=self.lit, skipped=self.skipped, title=title
        )

    def to_svg(self, title: str | None = None) -> str:
        """Self-contained ``<svg>`` with this assembly's nodes lit, for embedding."""
        return get_graph(self.graph_name).to_svg(
            lit=self.lit, skipped=self.skipped, title=title
        )

    def __repr__(self) -> str:
        return (
            f"Assembly(graph={self.graph_name!r}, lit={list(self.lit)}, "
            f"skipped-as-identity={list(self.skipped)})"
        )


def _find_named(op: AbstractOperator, name: str) -> AbstractOperator | None:
    # Breadth-first, so graph-node labels (outermost fold levels) win over
    # identically-named stages inside user-provided nested composites.
    queue: list[AbstractOperator] = [op]
    while queue:
        next_level: list[AbstractOperator] = []
        for current in queue:
            if isinstance(current, (Pipeline, SumOperator, SelectOperator)):
                parts = current.stages if isinstance(current, Pipeline) else current.branches
                for part_name, part in zip(current.names, parts, strict=True):
                    if part_name == name:
                        return part
                    next_level.append(part)
        queue = next_level
    return None


@dataclasses.dataclass
class _Branch:
    """A folding intermediate: an ordered chain of (name, op) plus provenance."""

    stages: list[tuple[str, AbstractOperator]]
    sourced: bool
    origin: str = ""  # root node for diagnostics (defaults to the first stage)

    def to_operator(self) -> AbstractOperator:
        if len(self.stages) == 1:
            return self.stages[0][1]
        names = _dedup([n for n, _ in self.stages])
        return Pipeline(*[op for _, op in self.stages], names=names)

    @property
    def label(self) -> str:
        return self.stages[0][0]


def _dedup(names: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    out = []
    for n in names:
        counts[n] = counts.get(n, 0) + 1
        out.append(n if counts[n] == 1 else f"{n}_{counts[n]}")
    return out


def _validate_region(graph: SignalGraph, path: tuple[str, ...], op: AbstractOperator):
    """A region claim must be a contiguous template path with non-junction ends."""
    edge_set = set(graph.edges)
    for a, b in zip(path, path[1:], strict=False):
        if (a, b) not in edge_set:
            raise AssemblyError(
                f"{type(op).__name__} claims region {path}, but ({a!r}, {b!r}) is "
                "not a template edge — a region must cover a contiguous path."
            )
    for end in (path[0], path[-1]):
        if graph.nodes[end].kind in ("junction", "selector"):
            raise AssemblyError(
                f"Region {path} of {type(op).__name__} starts/ends on the "
                f"{graph.nodes[end].kind} node {end!r}; junctions/selectors may only "
                "be interior to a region."
            )
    covered = set(path)
    for n in path[:-1]:
        for successor in graph._out[n]:
            if successor not in covered:
                raise AssemblyError(
                    f"Region {path} of {type(op).__name__} is not closed: interior "
                    f"node {n!r} has an edge to {successor!r} outside the region, "
                    "whose consumers would lose the intermediate signal. Cover the "
                    "fork too, or use component operators."
                )


def _resolve(
    graph: SignalGraph, operators: Sequence[AbstractOperator | At]
) -> tuple[dict[str, list[AbstractOperator]], list[tuple[tuple[str, ...], AbstractOperator]]]:
    placement: dict[str, list[AbstractOperator]] = {}
    regions: list[tuple[tuple[str, ...], AbstractOperator]] = []
    for item in operators:
        if isinstance(item, At):
            node, op = item.node, item.op
        else:
            op = item
            node = None
            for klass in type(op).__mro__:
                node = getattr(klass, "graph_node", None)
                if node is not None:
                    break
            if node is None:
                raise AssemblyError(
                    f"{type(op).__name__} declares no graph_node and no At(...) wrapper "
                    f"was given; wrap it as At(node_id, op). Known nodes: {list(graph.nodes)}"
                )
        nodes = (node,) if isinstance(node, str) else tuple(node)
        for n in nodes:
            if n not in graph.nodes:
                raise AssemblyError(
                    f"{type(op).__name__}: {n!r} is not a node of graph "
                    f"{graph.name!r}; known nodes: {list(graph.nodes)}"
                )
        if len(nodes) > 1:
            _validate_region(graph, nodes, op)
            regions.append((nodes, op))
            continue
        (node,) = nodes
        spec = graph.nodes[node]
        if spec.kind in ("junction", "selector"):
            raise AssemblyError(
                f"Node {node!r} is a {spec.kind} — junctions/selectors are never "
                "operator slots; they materialize automatically as "
                "SumOperator/SelectOperator."
            )
        existing = placement.setdefault(node, [])
        if existing and not spec.many:
            raise AssemblyError(
                f"Two operators provided for node {node!r} "
                f"({type(existing[0]).__name__} and {type(op).__name__}); this node "
                "accepts a single instance. Compose them explicitly and wrap with "
                "At(...) if that is intended."
            )
        existing.append(op)

    # Regions are atomic: no node may belong to two claims of any kind.
    seen: dict[str, str] = {n: f"operator at {n!r}" for n in placement}
    for path, op in regions:
        for n in path:
            if n in seen:
                raise AssemblyError(
                    f"Node {n!r} is claimed both by the region {path} of "
                    f"{type(op).__name__} and by {seen[n]} — claims must be disjoint."
                )
        for n in path:
            seen[n] = f"the region {path} of {type(op).__name__}"
    return placement, regions


def assemble(
    graph: SignalGraph, *operators: AbstractOperator | At
) -> Assembly:
    """Compile a set of operators into the sub-pipeline they induce on ``graph``.

    See the module docstring for the contraction rules. Raises
    :class:`AssemblyError` on unknown/ambiguous placement, junction slots,
    duplicate single-instance nodes, or a transform-rooted branch feeding a
    materialized junction (a sum branch must contain a source).
    """
    if not operators:
        raise AssemblyError("assemble() needs at least one operator.")
    placement, regions = _resolve(graph, operators)
    region_of: dict[str, int] = {}
    for idx, (path, _) in enumerate(regions):
        for n in path:
            region_of[n] = idx
    region_entry: dict[int, _Branch | None] = {}

    exprs: dict[str, _Branch | None] = {}
    skipped: list[str] = []
    for nid in graph._topo:
        spec = graph.nodes[nid]
        instances = placement.get(nid, [])
        upstream = [e for e in (exprs[p] for p in graph._in[nid]) if e is not None]

        if nid in region_of:
            idx = region_of[nid]
            path, region_op = regions[idx]
            if nid == path[0]:
                region_entry[idx] = upstream[0] if upstream else None
            else:
                for parent in graph._in[nid]:
                    if parent not in path and exprs[parent] is not None:
                        raise AssemblyError(
                            f"Live branch from {parent!r} feeds node {nid!r}, which is "
                            f"covered by the region {path} of "
                            f"{type(region_op).__name__} — regions are atomic; drop "
                            "the branch or use component operators instead."
                        )
            if nid == path[-1]:
                up = region_entry[idx]
                stages = (list(up.stages) if up else []) + [(nid, region_op)]
                sourced = graph.nodes[path[0]].kind == "source" or (
                    up.sourced if up else False
                )
                exprs[nid] = _Branch(stages, sourced, origin=path[0])
            else:
                exprs[nid] = None
            continue

        if spec.kind in ("junction", "selector"):
            if len(upstream) == 0:
                exprs[nid] = None
            elif len(upstream) == 1:
                skipped.append(nid)  # traversed pass-through junction/selector
                exprs[nid] = upstream[0]
            else:
                unsourced = [b for b in upstream if not b.sourced]
                if unsourced:
                    bad = unsourced[0].origin or unsourced[0].stages[0][0]
                    raise AssemblyError(
                        f"Transform {bad!r} feeds {spec.kind} {nid!r} with no live "
                        "source upstream — a branch must generate its own "
                        "contribution. Provide a source on that branch or drop it."
                    )
                branch_names = _dedup([b.label for b in upstream])
                branch_ops = [b.to_operator() for b in upstream]
                if spec.kind == "junction":
                    combined = SumOperator(*branch_ops, names=branch_names)
                else:
                    combined = SelectOperator(
                        *branch_ops, names=branch_names, switch_key=nid
                    )
                exprs[nid] = _Branch([(nid, combined)], sourced=True)
        elif spec.kind == "source":
            if not instances:
                exprs[nid] = None
            elif len(instances) == 1:
                exprs[nid] = _Branch([(nid, instances[0])], sourced=True)
            else:
                names = _dedup([nid] * len(instances))
                exprs[nid] = _Branch(
                    [(nid, SumOperator(*instances, names=names))], sourced=True
                )
        else:  # transform
            up = upstream[0] if upstream else None
            if instances:
                stages = list(up.stages) if up else []
                stages += [(nid, op) for op in instances]
                exprs[nid] = _Branch(
                    stages, sourced=up.sourced if up else False
                )
            else:
                if up is not None:
                    skipped.append(nid)
                exprs[nid] = up

    final = exprs[graph.sink]
    if final is None:
        raise AssemblyError("Nothing to assemble: no provided node reaches the sink.")

    claimed = set(placement) | set(region_of)
    lit = tuple(n for n in graph.nodes if n in claimed)
    live_span = _live_span(graph, lit)
    return Assembly(
        operator=final.to_operator(),
        graph_name=graph.name,
        lit=lit,
        skipped=tuple(n for n in skipped if n in live_span),
        has_source=final.sourced,
        root_label=final.stages[0][0] if len(final.stages) == 1 else "",
    )


def _live_span(graph: SignalGraph, lit: tuple[str, ...]) -> set[str]:
    """Nodes lying on a path between two lit nodes (for skip reporting)."""
    reach_from_lit: set[str] = set()
    frontier = set(lit)
    while frontier:
        n = frontier.pop()
        for m in graph._out[n]:
            if m not in reach_from_lit:
                reach_from_lit.add(m)
                frontier.add(m)
    reaches_lit: set[str] = set()
    frontier = set(lit)
    while frontier:
        n = frontier.pop()
        for m in graph._in[n]:
            if m not in reaches_lit:
                reaches_lit.add(m)
                frontier.add(m)
    return reach_from_lit & reaches_lit
