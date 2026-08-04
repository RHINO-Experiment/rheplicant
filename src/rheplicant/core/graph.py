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

**Addressing.** ``assembly[node_id]`` reaches the operator at a node whatever
the fold did with it. A ``many`` node holding several instances is the one
case a single id cannot answer for, so its instances are named ``x_1``,
``x_2``, … and the bare ``x`` raises :class:`AmbiguousNodeError` listing them.
With one instance nothing changes — ``x`` is still the address, and a
:class:`~rheplicant.inference.parameters.ParameterSpace` written against it is
untouched until a sibling actually arrives.
"""

import dataclasses
from collections.abc import Iterable, Mapping, Sequence
from typing import Literal

import equinox as eqx

from rheplicant.core.combinators import SelectOperator, SumOperator
from rheplicant.core.errors import DirtError
from rheplicant.core.operator import AbstractOperator
from rheplicant.core.pipeline import Pipeline
from rheplicant.core.state import State


class AssemblyError(DirtError, ValueError):
    """A provided operator set cannot be assembled on the signal graph."""


class AmbiguousNodeError(AssemblyError):
    """A node id was used as an address, but it holds more than one operator.

    Raised by :meth:`Assembly.__getitem__` / :meth:`Assembly.replace_node` for
    a ``many=True`` node carrying several instances. Answering with one of
    them (or with the fold over all of them) would silently pick a different
    operator than the caller means — and, through ``replace_node``, silently
    delete the siblings. The message names every instance id instead.
    """


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
        many: sources only — allow multiple instances. They compose the way
            their CONSUMER composes: sibling Sum branches into a junction,
            sibling *selector* branches into a selector (one switch position
            each, in the order they were provided). For the sink-side
            ``filters``-style transform chain use ``many`` on a transform:
            instances chain in call order.
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

    def to_mermaid(
        self,
        lit: Iterable[str] = (),
        skipped: Iterable[str] = (),
        counts: Mapping[str, int] | None = None,
    ) -> str:
        """Render the template as a mermaid flowchart with lit/dim styling.

        ``lit`` nodes are highlighted, ``skipped`` (traversed-as-identity)
        nodes are half-lit, everything else is dimmed — the signal-path view
        of what an assembly simulates.

        ``counts`` maps a node id to the number of operator instances sitting
        on it. A ``many`` node is one box however many instances it carries,
        so the count is shown in the label: an unannotated box would render
        two components as one.
        """
        lit, skipped = set(lit), set(skipped)
        counts = dict(counts or {})
        lines = ["flowchart TD"]
        for n, spec in self.nodes.items():
            label = n.replace("_", " ")
            if counts.get(n, 1) > 1:
                label = f"{label} (x{counts[n]})"
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
        counts: Mapping[str, int] | None = None,
    ) -> str:
        """Standalone HTML page of the template with lit/dim signal-path styling."""
        from rheplicant.core.render import signal_path_html

        return signal_path_html(
            self, lit=lit, skipped=skipped, title=title, counts=counts
        )

    def to_svg(
        self,
        lit: Iterable[str] = (),
        skipped: Iterable[str] = (),
        title: str | None = None,
        counts: Mapping[str, int] | None = None,
    ) -> str:
        """Self-contained ``<svg>`` of the template, for embedding (docs, notebooks)."""
        from rheplicant.core.render import signal_path_svg

        return signal_path_svg(
            self, lit=lit, skipped=skipped, title=title, counts=counts
        )

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

    Attributes:
        lit: template nodes this assembly claims, in template order.
        skipped: nodes traversed as identity between lit ones.
        instances: ``(node_id, instance_ids)`` for every node carrying more
            than one operator. ``lit`` names template nodes, and a node is one
            node however many operators sit on it — this is where the
            multiplicity lives, and what makes the bare node id ambiguous.
        materialized: junction/selector nodes that actually became a
            ``SumOperator``/``SelectOperator`` (rather than passing through).
        aliased: nodes the fold embedded at more than one position, because
            their contribution reaches the sink by more than one path. Reading
            them is honest — ``self[nid]`` IS the operator sitting there — so
            only :meth:`replace_node` refuses. Note the limit: a hand-rolled
            ``eqx.tree_at(lambda a: a[nid].x, ...)``, which is also how
            ``ParameterSpace`` binds, does not go through ``replace_node`` and
            still rewrites one copy only.

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
    instances: tuple[tuple[str, tuple[str, ...]], ...] = eqx.field(
        static=True, default=()
    )
    materialized: tuple[str, ...] = eqx.field(static=True, default=())
    aliased: tuple[str, ...] = eqx.field(static=True, default=())

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
        siblings = dict(self.instances).get(node_id)
        if siblings is not None:
            raise AmbiguousNodeError(
                f"{node_id!r} holds {len(siblings)} operator instances in this "
                f"assembly, so it addresses none of them: {list(siblings)}. Use one "
                f"of those ids — e.g. assembly[{siblings[0]!r}] — or "
                f"assembly.operator to reach the fold that sums them. (With a "
                f"single instance {node_id!r} is still the address; it stopped "
                "being one the moment a second operator was placed there.)"
            )
        if node_id and node_id == self.root_label:
            return self.operator
        found = _find_named(self.operator, node_id)
        if found is None:
            raise KeyError(f"No node named {node_id!r} in this assembly; lit: {self.lit}")
        return found

    def replace_node(self, node_id: str, operator: AbstractOperator) -> "Assembly":
        """Return a new Assembly with the operator at ``node_id`` swapped.

        Raises rather than swapping when ``node_id`` names something that is
        not one operator: a ``many`` node carrying several instances
        (:class:`AmbiguousNodeError`), a junction/selector that assembly
        materialized as a combinator, or a node the fold embedded at more than
        one position. In all three ``eqx.tree_at`` would happily rewrite one
        position — dropping live branches from the forward model, or leaving
        the node's other copies in it, with no shape change and no complaint.
        """
        target = self[node_id]  # raises AmbiguousNodeError on a multi-instance node
        if node_id in self.aliased:
            raise AssemblyError(
                f"{node_id!r} is folded into this assembly at more than one place: "
                "its contribution reaches the sink by several paths, so the operator "
                "sits in several branches. Replacing it would rewrite the one branch "
                "this id reaches and silently leave the others in the forward model. "
                f"Re-assemble() with the operator you want at {node_id!r}."
            )
        if node_id in self.materialized:
            names = getattr(target, "names", ())
            raise AssemblyError(
                f"{node_id!r} is a junction/selector that this assembly materialized "
                f"as {type(target).__name__} over {list(names)}; it is not an "
                "operator slot, and replacing it would drop those branches from the "
                "forward model. Replace one of them by its own node id, or "
                "re-assemble() with the operator set you want."
            )

        def where(a: "Assembly") -> AbstractOperator:
            return a[node_id]

        del target  # existence check only
        return eqx.tree_at(where, self, operator)

    @property
    def _counts(self) -> dict[str, int]:
        """Instances per node, for renderings — one lit box may be several."""
        return {nid: len(names) for nid, names in self.instances}

    def to_mermaid(self) -> str:
        """Lit/dim mermaid rendering via the registered template."""
        return get_graph(self.graph_name).to_mermaid(
            lit=self.lit, skipped=self.skipped, counts=self._counts
        )

    def to_html(self, title: str | None = None) -> str:
        """Standalone HTML page: the full graph with this assembly's nodes lit."""
        return get_graph(self.graph_name).to_html(
            lit=self.lit, skipped=self.skipped, title=title, counts=self._counts
        )

    def to_svg(self, title: str | None = None) -> str:
        """Self-contained ``<svg>`` with this assembly's nodes lit, for embedding."""
        return get_graph(self.graph_name).to_svg(
            lit=self.lit, skipped=self.skipped, title=title, counts=self._counts
        )

    def __repr__(self) -> str:
        # `lit` names template nodes, and a node stays one node however many
        # operators sit on it — so the multiplicity is reported beside it
        # rather than folded into the list, where it would read as a node id.
        counts = self._counts
        lit = [
            f"{nid} x{counts[nid]}" if nid in counts else nid for nid in self.lit
        ]
        return (
            f"Assembly(graph={self.graph_name!r}, lit={lit}, "
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
                        return _descend_to_own_stage(part, name)
                    next_level.append(part)
        queue = next_level
    return None


def _positions(root: AbstractOperator, target: AbstractOperator) -> int:
    """How many positions of the folded tree ``target`` occupies (by identity)."""
    count = 0
    queue: list[AbstractOperator] = [root]
    while queue:
        current = queue.pop()
        if current is target:
            count += 1
        if isinstance(current, (Pipeline, SumOperator, SelectOperator)):
            queue.extend(
                current.stages if isinstance(current, Pipeline) else current.branches
            )
    return count


def _fold_duplicates(
    root: AbstractOperator,
    placement: dict[str, list[AbstractOperator]],
    regions: Sequence[tuple[tuple[str, ...], AbstractOperator]],
) -> dict[str, int]:
    """Nodes whose operator the FOLD put at more than one position, and how many.

    A node whose contribution reaches the sink by several paths is folded in
    once per path. ``_find_named`` reaches one of those positions and
    ``eqx.tree_at`` rewrites that one, so writing through the node id leaves
    the other copies live — a finite, correctly-shaped, wrong forward model.

    Placing ONE operator object at several nodes is deliberate and not this, so
    the occurrence count is compared against how often the caller placed it
    rather than against 1.
    """
    slots: list[tuple[str, AbstractOperator]] = [
        (nid, op) for nid, ops_at in placement.items() for op in ops_at
    ]
    slots += [(path[-1], op) for path, op in regions]
    placed: dict[int, int] = {}
    for _, op in slots:
        placed[id(op)] = placed.get(id(op), 0) + 1
    duplicates: dict[str, int] = {}
    for nid, op in slots:
        found = _positions(root, op)
        if found > placed[id(op)]:
            duplicates[nid] = max(duplicates.get(nid, 0), found)
    return duplicates


def _check_promised_ids(
    root: AbstractOperator,
    multi: dict[str, tuple[str, ...]],
    placement: dict[str, list[AbstractOperator]],
    duplicates: dict[str, int],
) -> None:
    """Every per-instance id the assembly will hand out must reach its instance.

    :func:`_instance_names` mints ``x_1..x_n``; :func:`_dedup` independently
    mints ``x, x_2, x_3, ...`` for repeated branch labels, and the two overlap
    from ``_2`` on. Both arise from the same graph shape — a node reaching a
    fold by several paths — so the collision is reported as what it is rather
    than as a naming accident. An id that resolves to something other than the
    operator placed there would be handed to the caller BY
    :class:`AmbiguousNodeError` and then written through, which is worse than
    saying nothing.
    """
    for nid, names in multi.items():
        if nid in duplicates:
            raise AssemblyError(
                f"Node {nid!r} carries {len(names)} operator instances, and its "
                f"contribution reaches the sink by {duplicates[nid]} paths — so the "
                f"fold embeds each instance {duplicates[nid]} times and labels the "
                f"repeated branches {nid!r}, {nid + '_2'!r}, ... . Those labels "
                f"collide with the per-instance ids {list(names)}, leaving no id that "
                f"names one instance: reading {nid + '_2'!r} would reach a whole "
                "branch and writing it would rewrite that branch instead. Give the "
                "paths their own nodes so each instance has one home; placing ONE "
                f"composed operator at {nid!r} also removes the ambiguity, though a "
                "node folded in twice stays unwritable."
            )
        for index, (name, op) in enumerate(zip(names, placement[nid], strict=True), 1):
            found = _find_named(root, name)
            if found is not op:
                raise AssemblyError(
                    f"Node {nid!r} would report {name!r} as the id of instance "
                    f"{index} ({type(op).__name__}), but that id resolves to "
                    f"{type(found).__name__ if found is not None else 'nothing'} in "
                    "the assembled operator — it addresses the wrong part of the "
                    "forward model, and replace_node/ParameterSpace would rewrite "
                    f"that part. Re-assemble with one operator at {nid!r}."
                )


def _descend_to_own_stage(part: AbstractOperator, name: str) -> AbstractOperator:
    """Resolve a name that labels a FOLD rooted at a node to the node itself.

    A branch spanning ``sky -> spill`` is labelled by its first node, so a
    sibling Sum names it ``sky`` while the Pipeline inside it also has a stage
    named ``sky``. ``assembly["sky"]`` must be the operator AT that node, not
    the fold that starts there — otherwise ``eqx.tree_at(lambda a: a["sky"].amp,
    ...)`` reaches a Pipeline and fails on an attribute the caller can see in
    the source. Descending while the match keeps re-naming itself resolves it.
    """
    while isinstance(part, Pipeline) and name in part.names:
        part = part.stages[part.names.index(name)]
    return part


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


def _feeds_only_selectors(graph: SignalGraph, node: str) -> bool:
    """Does every consumer of ``node`` switch between its inputs rather than sum?

    Decides how multiple instances at a ``many`` source fold. A node feeding
    both a selector and a junction has no single answer, so it keeps the Sum —
    the conservative reading, and no shipped graph has such a node.
    """
    consumers = graph._out[node]
    return bool(consumers) and all(
        graph.nodes[c].kind == "selector" for c in consumers
    )


def _dedup(names: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    out = []
    for n in names:
        counts[n] = counts.get(n, 0) + 1
        out.append(n if counts[n] == 1 else f"{n}_{counts[n]}")
    return out


def _instance_names(node_id: str, count: int) -> tuple[str, ...]:
    """Names addressing the ``count`` operator instances placed at ``node_id``.

    One instance keeps the bare node id: the single-instance assembly is
    exactly what it always was, down to the static names, so existing spaces
    and examples are untouched.

    Two or more get a 1-based suffix **including the first** — ``x_1``,
    ``x_2`` — rather than ``x``, ``x_2``. Suffixing only the later ones would
    leave ``x`` naming both one particular instance *and* the fold that a
    consumer labels by the node id, and the breadth-first lookup in
    :func:`_find_named` resolves that collision to the fold: ``assembly[x]``
    then hands back a ``SumOperator`` and ``replace_node(x, ...)`` overwrites
    every instance with one operator. Making the bare id name nothing turns
    that into :class:`AmbiguousNodeError`, which can say what to write.
    """
    if count == 1:
        return (node_id,)
    return tuple(f"{node_id}_{i}" for i in range(1, count + 1))


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
    # A `many` source whose consumers are all selectors contributes one branch
    # PER INSTANCE rather than one summed branch — a switch picks a source per
    # sample, it does not add them up. Kept beside `exprs` rather than widening
    # it so that every other path folds bit-for-bit as before (SumOperator's
    # per-branch PRNG splitting makes a flatter tree a different run, not just
    # a different shape).
    fan: dict[str, list[_Branch]] = {}
    skipped: list[str] = []
    # Addressing bookkeeping, both static: which node ids carry several
    # operator instances (so the bare id addresses none of them in particular),
    # and which junction/selector nodes actually materialized as a combinator
    # (so replace_node there would discard every branch feeding it).
    multi: dict[str, tuple[str, ...]] = {}
    materialized: list[str] = []

    def upstream_of(nid: str) -> list[_Branch]:
        out: list[_Branch] = []
        for parent in graph._in[nid]:
            if parent in fan:
                out.extend(fan[parent])
            elif exprs[parent] is not None:
                out.append(exprs[parent])
        return out
    for nid in graph._topo:
        spec = graph.nodes[nid]
        instances = placement.get(nid, [])
        upstream = upstream_of(nid)
        if len(instances) > 1:
            multi[nid] = _instance_names(nid, len(instances))

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
                materialized.append(nid)
                exprs[nid] = _Branch([(nid, combined)], sourced=True)
        elif spec.kind == "source":
            if not instances:
                exprs[nid] = None
            elif len(instances) == 1:
                exprs[nid] = _Branch([(nid, instances[0])], sourced=True)
            elif _feeds_only_selectors(graph, nid):
                names = _instance_names(nid, len(instances))
                fan[nid] = [
                    _Branch([(name, op)], sourced=True, origin=nid)
                    for name, op in zip(names, instances, strict=True)
                ]
                exprs[nid] = fan[nid][0]  # liveness marker; consumers read `fan`
            else:
                names = _instance_names(nid, len(instances))
                exprs[nid] = _Branch(
                    [(nid, SumOperator(*instances, names=names))], sourced=True
                )
        else:  # transform
            up = upstream[0] if upstream else None
            if instances:
                stages = list(up.stages) if up else []
                names = _instance_names(nid, len(instances))
                stages += list(zip(names, instances, strict=True))
                exprs[nid] = _Branch(
                    stages, sourced=up.sourced if up else False,
                    origin=up.origin if up else nid,
                )
            else:
                if up is not None:
                    skipped.append(nid)
                exprs[nid] = up

    final = exprs[graph.sink]
    if final is None:
        raise AssemblyError("Nothing to assemble: no provided node reaches the sink.")

    operator = final.to_operator()
    # Addressing closure: the ids this assembly is about to promise must reach
    # the operators they name, and a node the fold duplicated cannot be written
    # through at all. Both are decided on the BUILT tree, so they cannot drift
    # from what `_find_named`/`eqx.tree_at` actually do to it.
    duplicates = _fold_duplicates(operator, placement, regions)
    _check_promised_ids(operator, multi, placement, duplicates)

    claimed = set(placement) | set(region_of)
    lit = tuple(n for n in graph.nodes if n in claimed)
    live_span = _live_span(graph, lit)
    return Assembly(
        operator=operator,
        graph_name=graph.name,
        lit=lit,
        skipped=tuple(n for n in skipped if n in live_span),
        has_source=final.sourced,
        root_label=final.stages[0][0] if len(final.stages) == 1 else "",
        instances=tuple((nid, names) for nid, names in multi.items()),
        materialized=tuple(materialized),
        aliased=tuple(n for n in graph.nodes if n in duplicates),
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
