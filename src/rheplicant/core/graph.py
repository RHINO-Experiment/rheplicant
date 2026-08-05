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
per instance — freely between nodes of the same kind, and not across the
source/transform line, because a node's kind is what says whether the operator
there creates the data or acts on data reaching it. ``has_source``, the
``__call__`` guard and the "a summed branch must contain a source" rule are all
read off that kind, so an operator disagreeing with its node makes all three
wrong at once; :func:`_check_slot_kinds` refuses the disagreement instead.

**Ordering.** An operator whose physics depends on *where* it sits — a
calibration tone that only tracks a gain it passes through — declares that in
the graph's own nouns with
:attr:`~rheplicant.core.operator.AbstractOperator.must_precede`, and
:func:`assemble` refuses a placement that violates it. Because ``At`` can put
any operator at any node, an ordering constraint stated only in a docstring is
one nothing checks: the tone assembles cleanly downstream of the gain, and its
gain response silently drops to 1.0.

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
import jax

from rheplicant.core.combinators import SelectOperator, SumOperator
from rheplicant.core.errors import DirtError
from rheplicant.core.operator import AbstractOperator
from rheplicant.core.pipeline import Pipeline, validate_operators
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
            (data -> data; in-degree at most 1), ``"junction"`` (sum point), or
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

    "Regardless of its class registration" stops at the node's *kind*: an
    operator that declares a ``graph_node`` may be moved to any other node of
    the same kind, and not across the source/transform line. A source at a
    transform node discards the signal reaching that node, and a transform at a
    source node is handed ``data=None``; neither is a placement, and
    :func:`assemble` refuses both — see :func:`_check_slot_kinds`.
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
    in-degree AT MOST 1 — a parentless transform is permitted, because it is
    not a defect the template has to catch (see ``__check_init__``).
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
                # `> 1`, not `!= 1`, and deliberately. A PARENTLESS transform is
                # not a defect this container has to refuse. Measured: with
                # nothing placed on it the node contracts to identity and the
                # model runs; with an operator placed on it, assembly refuses
                # via the guard that names the real problem -- "Transform 't'
                # feeds junction 'j' with no live source upstream". Refusing it
                # here would reject legitimate templates in order to restate a
                # check that already exists, from further away and with less
                # information to phrase it well.
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
        aliased: **public, and the thing to check before writing a selector** —
            nodes the fold embedded at more than one position, because their
            contribution reaches the sink by more than one path. Reading them
            is honest — ``self[nid]`` IS the operator sitting there — so only
            writing refuses: :meth:`replace_node` and
            :meth:`~rheplicant.inference.parameters.ParameterSpace.validate`
            both consult this tuple, since either would otherwise rewrite one
            copy and leave the others live in the forward model. What neither
            can see is a hand-rolled ``eqx.tree_at(lambda a: a[nid].x, ...)``:
            that call goes through no framework code, so nothing intercepts it
            and it still rewrites one copy only. Consult ``.aliased`` yourself
            before writing one — ``repr(assembly)`` names these nodes when
            there are any, and says nothing when there are none, so the
            condition reaches a reader who did not know to ask for it.
        placements: the recipe :meth:`without` re-assembles from, as
            ``(template nodes, address)`` per placed operator, in template
            order. Addresses rather than operators on purpose — see the field's
            own comment.

    If the assembly contains live sources it *generates* its data — calling
    it on a state that already carries data raises, because that data would
    be silently discarded (pass ``data=None``). Source-free assemblies are
    transform chains operating on caller data.

    ``has_source`` is read off the template's node kinds, not off the
    operators, and that is only sound because :func:`_check_slot_kinds` refuses
    a placement whose operator disagrees with its node about creating data. An
    operator declaring no ``graph_node`` cannot be screened, so it can still be
    placed on the wrong kind of node and make this flag — and therefore the
    guard above — wrong in either direction.
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
    # The recipe, as ADDRESSES rather than operators: (template nodes, the id
    # `self[...]` reaches this operator by). Storing the operators themselves
    # would put every one of them at a second pytree position, which is exactly
    # the `aliased` failure this class refuses to create. Addresses are static
    # strings, so the fold stays the only place an operator lives, and
    # `without` recovers the set by reading them back off the built tree.
    placements: tuple[tuple[tuple[str, ...], str], ...] = eqx.field(
        static=True, default=()
    )

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

        ``operator`` must be an :class:`~rheplicant.core.operator.AbstractOperator`.
        ``None`` in particular is refused by name: it reads as "take this stage
        out", and it used to return an Assembly whose ``lit`` and whose mermaid
        rendering both still claimed the stage, which then died on the next call
        with ``TypeError: 'NoneType' object is not callable``. Removing a stage
        is :meth:`without`.
        """
        if operator is None:
            raise AssemblyError(
                f"replace_node({node_id!r}, None) does not remove the stage — it would "
                "return an Assembly whose metadata and rendering still claim "
                f"{node_id!r} is present, and which raises TypeError: 'NoneType' object "
                f"is not callable the next time it runs. Use assembly.without({node_id!r}) "
                "to drop it, which re-assembles and reports lit/skipped/has_source "
                "honestly."
            )
        validate_operators((operator,), "replace_node")
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

    def without(self, node_id: str) -> "Assembly":
        """Return a new Assembly with the operator(s) at ``node_id`` dropped.

        The supported answer to "this stage must not be here" — the sentence
        :func:`~rheplicant.inference.parameters.refuse_stochastic_stages` says
        about a noise stage in a twin you infer with, and the one
        :meth:`replace_node` used to invite with ``None`` and then answer
        wrongly.

        Not tree surgery: this re-runs :func:`assemble` over the remaining
        operators, recovered from the built tree by the addresses recorded in
        ``placements``. So the result is exactly the assembly you would
        have got by not providing that operator in the first place — same fold,
        same ``lit``/``skipped``/``has_source``/``materialized``, and every
        assembly-time refusal re-run. If dropping the stage leaves something
        that cannot be assembled — a summed branch with no source left on it,
        say — you get that refusal, in assemble's own words, rather than a model
        that quietly changed meaning. Dropping a stage another operator names in
        :attr:`~rheplicant.core.operator.AbstractOperator.must_precede` is NOT
        such a case: an absent node contracts to identity, so there is nothing
        left to pass through and nothing to violate — the same rule
        :func:`_check_ordering` applies to a node that was never lit.

        A ``many`` node is dropped whole: every instance on it goes. Use
        ``assemble()`` directly to keep some of them.

        Raises:
            AssemblyError: if ``node_id`` carries no operator in this assembly,
                if it is the only one, or if this Assembly was not built by
                :func:`assemble` (so there is no recipe to re-run).
        """
        if not self.placements:
            raise AssemblyError(
                "This Assembly carries no placement record, so without() has no recipe "
                "to re-assemble from. Only assemble() builds that record; an Assembly "
                "constructed directly cannot be edited this way."
            )
        kept = [entry for entry in self.placements if node_id not in entry[0]]
        if len(kept) == len(self.placements):
            raise AssemblyError(
                f"without({node_id!r}): no operator sits at {node_id!r} in this "
                f"assembly. Lit nodes: {list(self.lit)}."
            )
        if not kept:
            raise AssemblyError(
                f"without({node_id!r}) would leave nothing to assemble — {node_id!r} "
                "carries the only operator here. An empty assembly is not a model; "
                "drop the assembly instead."
            )
        return assemble(
            get_graph(self.graph_name),
            *(At(nodes if len(nodes) > 1 else nodes[0], self[address])
              for nodes, address in kept),
        )

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
        # `aliased` appears ONLY when it is non-empty, and the asymmetry is the
        # point. It is empty for every shipped graph and for every user graph
        # whose nodes each reach the sink by one path, so an always-present
        # `aliased=[]` would spend a field on the answer "nothing here" and
        # teach the reader to skip past the one place the warning can appear.
        # Non-empty, it names the nodes the fold embedded at several positions:
        # `replace_node` and `ParameterSpace.validate` already refuse to write
        # through them, but a hand-rolled `eqx.tree_at` goes through neither and
        # rewrites one copy only. Seeing them here is how that is noticed
        # without knowing to ask -- see the `aliased` attribute for the rest.
        fan_out = (
            f", aliased-at-several-positions={list(self.aliased)}"
            if self.aliased
            else ""
        )
        return (
            f"Assembly(graph={self.graph_name!r}, lit={lit}, "
            f"skipped-as-identity={list(self.skipped)}{fan_out})"
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


def _children(op: AbstractOperator) -> tuple[AbstractOperator, ...]:
    """The fold's composite spine, in one place: what holds operators.

    Everything that walks a fold by identity descends through exactly these,
    so a new composite type has one place to be taught rather than several to
    be forgotten in.
    """
    if isinstance(op, Pipeline):
        return tuple(op.stages)
    if isinstance(op, (SumOperator, SelectOperator)):
        return tuple(op.branches)
    return ()


def _children_through_assemblies(op: AbstractOperator) -> tuple[AbstractOperator, ...]:
    """:func:`_children`, also stepping into an Assembly's folded operator.

    Kept separate because :func:`_positions` must NOT step into one: it asks
    what *this* fold did, and a nested assembly's contents were placed by an
    earlier one.
    """
    if isinstance(op, Assembly):
        return (op.operator,)
    return _children(op)


def _spine_pairs(
    root: AbstractOperator, mirror: AbstractOperator | None = None
) -> Iterable[tuple[AbstractOperator, AbstractOperator | None]]:
    """Every position of ``root``, paired with the same position of ``mirror``.

    ``mirror`` is a ``tree_map`` copy of ``root``, so the two have the same
    spine and each pair is one position seen twice: once as the real operator
    (which a fold may have placed more than once, by identity) and once as
    whatever the copy put there. Pass ``None`` to walk ``root`` alone.
    """
    queue: list[tuple[AbstractOperator, AbstractOperator | None]] = [(root, mirror)]
    while queue:
        current, twin = queue.pop()
        yield current, twin
        children = _children_through_assemblies(current)
        if twin is None:
            queue.extend((child, None) for child in children)
        else:
            queue.extend(
                zip(children, _children_through_assemblies(twin), strict=True)
            )


class _LeafPath:
    """One tagged leaf path, wrapped so that flattening cannot expand it.

    ``tree_map_with_path`` writing the bare path would leave a *tuple* in leaf
    position, and any later ``tree_leaves`` would flatten it into its
    components. An opaque object is a leaf, so the path reads back out whole.
    """

    __slots__ = ("path",)

    def __init__(self, path: tuple):
        self.path = path


def _aliased_leaf_paths(pipeline: AbstractOperator) -> dict[tuple, str]:
    """``{leaf key path: node id}`` for every leaf an aliased node owns.

    :attr:`Assembly.aliased` names the *nodes* the fold embedded more than
    once; an ``eqx.tree_at`` selector lands on a *leaf*, so anything vetting a
    selector — :meth:`ParameterSpace.validate
    <rheplicant.inference.parameters.ParameterSpace.validate>` — needs the
    leaves those nodes own.

    Every copy is reported, not only the one :func:`_find_named` reaches: a
    selector spelled out by hand can name the second copy, and rewriting that
    one leaves the first live — the same wrong answer from the other end.
    Assemblies nested inside a larger composite are covered too.
    """
    if not any(
        isinstance(op, Assembly) and op.aliased for op, _ in _spine_pairs(pipeline)
    ):
        return {}
    tagged = jax.tree_util.tree_map_with_path(lambda path, _: _LeafPath(path), pipeline)
    owned: dict[tuple, str] = {}
    for current, twin in _spine_pairs(pipeline, tagged):
        if not isinstance(current, Assembly):
            continue
        for node_id in current.aliased:
            target = _find_named(current.operator, node_id)
            if target is None:  # pragma: no cover - assemble() checked these ids
                continue
            for below, below_twin in _spine_pairs(current.operator, twin.operator):
                if below is target:
                    owned.update(
                        (tag.path, node_id)
                        for tag in jax.tree_util.tree_leaves(below_twin)
                    )
    return owned


def _positions(root: AbstractOperator, target: AbstractOperator) -> int:
    """How many positions of the folded tree ``target`` occupies (by identity)."""
    count = 0
    queue: list[AbstractOperator] = [root]
    while queue:
        current = queue.pop()
        if current is target:
            count += 1
        queue.extend(_children(current))
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


def _descendants(graph: SignalGraph, node: str) -> set[str]:
    """Every node the signal leaving ``node`` can reach, following the edges."""
    seen: set[str] = set()
    frontier = [node]
    while frontier:
        for successor in graph._out[frontier.pop()]:
            if successor not in seen:
                seen.add(successor)
                frontier.append(successor)
    return seen


def _declared_node(op: AbstractOperator) -> str | tuple[str, ...] | None:
    """The operator's declared home node, resolved through the MRO.

    ``AbstractOperator.graph_node`` is ``None``, so a plain ``getattr`` on the
    class already answers for most operators; the walk exists for the class
    that sets ``graph_node = None`` on an intermediate base and declares the
    real node on the subclasses of it, which is how the shipped families and
    the test fixtures are written.
    """
    for klass in type(op).__mro__:
        node = getattr(klass, "graph_node", None)
        if node is not None:
            return node
    return None


def _creates_data(graph: SignalGraph, node: str) -> bool:
    """Does the template say the operator at ``node`` generates its own data?"""
    return graph.nodes[node].kind == "source"


def _check_slot_kinds(
    graph: SignalGraph,
    placement: dict[str, list[AbstractOperator]],
    regions: Sequence[tuple[tuple[str, ...], AbstractOperator]],
) -> None:
    """A placed operator must agree with its node about who creates the data.

    ``Assembly.has_source`` — and with it the ``__call__`` guard, and the fold's
    "every summed branch must contain a source" rule — is read off the template's
    node kinds. That is only sound while the operator sitting at a node does what
    the node kind says, and ``At`` will put any operator at any node. Both
    disagreements produce a model that runs:

    * a source operator at a transform node overwrites ``state.data``, so the
      caller's data and everything the upstream branch computed are discarded —
      while ``has_source`` is False and the guard that exists to say exactly that
      stays silent;
    * a transform operator at a source node leaves ``has_source`` True, so the
      guard demands ``data=None``, which is the one input that makes the
      operator die on ``NoneType``.

    Refusing the disagreement fixes both at once rather than patching each guard:
    with the kinds in agreement the node kind IS the operator kind, and the
    existing derivation is correct by construction. It is a *placement* refusal,
    so it is raised before :func:`_check_ordering`'s physics one — where an
    operator sits has to be settled before what it must precede can mean
    anything.

    Only ``graph_node`` can be consulted, so an operator that declares no home
    (or one belonging to some other template) is not screened. That is the whole
    of what this misses, it is stated in :func:`assemble`'s docstring, and it is
    pinned in ``tests/core/test_placement_kind.py``. Deriving the answer instead
    by running each operator under ``jax.eval_shape`` at assemble time does not
    work here: ``assemble`` has no coordinates to build a probe state from, and
    every shipped source raises ``StateValidationError`` without them — the
    probe classifies 0 of 10 shipped operators correctly, so it would report
    every assembly as source-free and the guard would then refuse every
    legitimate forward run.
    """
    slots: list[tuple[str, tuple[str, ...], AbstractOperator]] = [
        (node, (node,), op) for node, ops_at in placement.items() for op in ops_at
    ]
    # A region is *entered* at its first node — that is the node whose kind
    # decides whether the branch generates its own data, and the node whose
    # upstream (or absence of one) the operator will be handed — so that is the
    # one it has to match. Covering the declared home somewhere further down the
    # region does not excuse the entry: an operator that multiplies its input,
    # entered where there is no input, still meets `data=None`.
    slots += [(path[0], path, op) for path, op in regions]
    for node, path, op in slots:
        declared = _declared_node(op)
        if not isinstance(declared, str) or declared not in graph.nodes:
            continue
        if _creates_data(graph, declared) == _creates_data(graph, node):
            continue
        where = (
            f"the region {tuple(path)}, entered at {node!r}"
            if len(path) > 1
            else f"node {node!r}"
        )
        if _creates_data(graph, declared):
            consequence = (
                f"An operator that creates data, placed where data arrives, "
                f"overwrites it: the caller's data and everything the branch "
                f"upstream of {node!r} computed are discarded — while has_source "
                "stays False, so the guard whose whole sentence is 'caller-supplied "
                "state.data would be discarded' says nothing."
            )
        else:
            consequence = (
                f"An operator that consumes data, placed where data is created, is "
                f"handed the nothing that precedes {node!r} — while has_source stays "
                "True, so Assembly.__call__ demands data=None, which is the one "
                "input that makes this operator die on NoneType."
            )
        raise AssemblyError(
            f"{type(op).__name__} declares graph_node={declared!r}, a "
            f"{graph.nodes[declared].kind} node of graph {graph.name!r}, but this "
            f"assembly places it at {where}, which is a "
            f"{graph.nodes[node].kind} node. {consequence} Place it at a "
            f"{graph.nodes[declared].kind} node — At() moves an operator freely "
            "between nodes of one kind — or give the template a node that says what "
            "this operator actually does."
        )


def _check_ordering(
    graph: SignalGraph,
    placement: dict[str, list[AbstractOperator]],
    regions: Sequence[tuple[tuple[str, ...], AbstractOperator]],
) -> None:
    """Enforce every placed operator's ``must_precede`` against the template.

    The constraint is REACHABILITY, not a toposort index. A toposort is a total
    order over a DAG, so it also orders nodes on branches that never meet, and
    "cw_tone happens to sort before bandpass" would be satisfied by a placement
    whose output never reaches the bandpass at all. What the physics asks is
    that this operator's contribution flows THROUGH the named stage, and that
    is exactly a directed path.

    A constraint is checked only against nodes that are LIT. An absent node
    contracts to identity, so there is no bandpass to pass through and nothing
    to violate — refusing there would reject a sky-only assembly for the sake
    of a stage it never asked for.

    A node id the template does not have is refused rather than skipped: an
    unenforceable declaration is prose in a ClassVar, which is the condition
    this mechanism exists to end.
    """
    covered = {n for path, _ in regions for n in path}
    lit = set(placement) | covered
    slots: list[tuple[tuple[str, ...], AbstractOperator]] = [
        ((node,), op) for node, ops_at in placement.items() for op in ops_at
    ]
    slots += list(regions)
    for path, op in slots:
        required = op.must_precede
        if not required:
            continue
        downstream = _descendants(graph, path[-1])
        for target in required:
            if target not in graph.nodes:
                raise AssemblyError(
                    f"{type(op).__name__} declares must_precede={list(required)}, but "
                    f"{target!r} is not a node of graph {graph.name!r}; known nodes: "
                    f"{list(graph.nodes)}. An ordering constraint naming a node the "
                    "template does not have can never be checked, so it would sit in "
                    "the class as unenforced prose."
                )
            if target not in lit or target in path:
                continue
            if target not in downstream:
                because = (
                    f" {op.must_precede_because}" if op.must_precede_because else ""
                )
                raise AssemblyError(
                    f"{type(op).__name__} declares must_precede="
                    f"{list(required)}, but this assembly places it at "
                    f"{path[-1]!r}, from which {target!r} is not reachable on graph "
                    f"{graph.name!r} — so nothing this operator contributes ever "
                    f"passes through {target!r}, and whatever {target!r} does to the "
                    f"signal is absent from it.{because} Place it at a node upstream "
                    f"of {target!r} (its declared home is "
                    f"{getattr(type(op), 'graph_node', None)!r}), or drop {target!r} "
                    "from the assembly if the stage genuinely is not there."
                )


def _claimed_nodes(
    graph: SignalGraph, item: AbstractOperator | At
) -> tuple[str, ...]:
    """The template nodes ``item`` claims: from ``At(...)``, or its registration.

    One node for an ordinary placement, several for a region claim. Every id is
    checked against the template here, before anything downstream asks what kind
    of node it is — an unknown id has no kind to answer with, and the message
    that names the known nodes is the one a typo needs.
    """
    if isinstance(item, At):
        node, op = item.node, item.op
    else:
        op = item
        node = _declared_node(op)
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
    return nodes


def _place_at_node(
    graph: SignalGraph,
    node: str,
    op: AbstractOperator,
    placement: dict[str, list[AbstractOperator]],
) -> None:
    """Record ``op`` at a single-node slot, refusing what is not one.

    Junctions and selectors are never slots — they materialize from the branches
    that reach them — and a node that is not ``many`` holds one operator, so a
    second one is a mistake rather than a composition.
    """
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


def _check_disjoint_claims(
    placement: dict[str, list[AbstractOperator]],
    regions: Sequence[tuple[tuple[str, ...], AbstractOperator]],
) -> None:
    """Regions are atomic: no node may belong to two claims of any kind.

    Checked over the whole provided set rather than per item, because the
    conflict is between claims and either one may be read first.
    """
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


def _resolve(
    graph: SignalGraph, operators: Sequence[AbstractOperator | At]
) -> tuple[dict[str, list[AbstractOperator]], list[tuple[tuple[str, ...], AbstractOperator]]]:
    # `Pipeline` and both combinators screen their members through this; before
    # `must_precede` landed, a non-operator here reached the fold and failed
    # there. It now fails earlier and worse, on `op.must_precede` — so the
    # screen belongs at the top of the one route that skipped it.
    validate_operators(
        tuple(item.op if isinstance(item, At) else item for item in operators), "assemble"
    )
    placement: dict[str, list[AbstractOperator]] = {}
    regions: list[tuple[tuple[str, ...], AbstractOperator]] = []
    for item in operators:
        op = item.op if isinstance(item, At) else item
        nodes = _claimed_nodes(graph, item)
        if len(nodes) > 1:
            _validate_region(graph, nodes, op)
            regions.append((nodes, op))
        else:
            _place_at_node(graph, nodes[0], op, placement)
    _check_disjoint_claims(placement, regions)
    return placement, regions


# ---------------------------------------------------------------------------
# the fold: one helper per node kind, dispatched from _fold_graph's loop
#
# Each helper answers one question -- "what branch does THIS node contribute?"
# -- and returns it, so the loop below is a dispatch on node kind and nothing
# else. The accumulators a helper records into (`skipped`, `materialized`,
# `fan`, `entry`) are named in its signature rather than closed over, so what
# each one writes is readable without reading the loop.
# ---------------------------------------------------------------------------


def _upstream_of(
    graph: SignalGraph,
    nid: str,
    exprs: dict[str, "_Branch | None"],
    fan: dict[str, list[_Branch]],
) -> list[_Branch]:
    """The live branches arriving at ``nid``, in the template's in-edge order.

    A fanned ``many`` source contributes one branch per instance (see
    :func:`_fold_source`); every other live parent contributes its single
    folded branch. Dead parents contribute nothing, which is how an absent
    node prunes rather than propagating a hole.
    """
    out: list[_Branch] = []
    for parent in graph._in[nid]:
        if parent in fan:
            out.extend(fan[parent])
        elif exprs[parent] is not None:
            out.append(exprs[parent])
    return out


def _fold_region(
    graph: SignalGraph,
    nid: str,
    path: tuple[str, ...],
    region_op: AbstractOperator,
    upstream: list[_Branch],
    exprs: dict[str, "_Branch | None"],
    entry: dict[int, "_Branch | None"],
    idx: int,
) -> "_Branch | None":
    """Fold one node of a region claim: record its entry, close it at its end.

    A region is ONE operator covering a contiguous run of template nodes, so
    only the last node contributes a branch — the interior ones fold to
    ``None`` and the operator is appended once, at the end, onto whatever
    entered at ``path[0]``. ``entry[idx]`` carries that entering branch across
    the nodes in between.

    :func:`_validate_region` already checked the template side of atomicity at
    resolve time; this is the assembly-time half, which only the *provided set*
    can violate. A live branch reaching the region's interior would have its
    signal silently dropped, because the covering operator is handed the entry
    branch and nothing else.

    ``sourced`` is read off the region's FIRST node, not its last: entering on a
    source node is what makes the whole region generate its own contribution.
    """
    if nid == path[0]:
        entry[idx] = upstream[0] if upstream else None
    else:
        for parent in graph._in[nid]:
            if parent not in path and exprs[parent] is not None:
                raise AssemblyError(
                    f"Live branch from {parent!r} feeds node {nid!r}, which is "
                    f"covered by the region {path} of "
                    f"{type(region_op).__name__} — regions are atomic; drop "
                    "the branch or use component operators instead."
                )
    if nid != path[-1]:
        return None
    up = entry[idx]
    stages = (list(up.stages) if up else []) + [(nid, region_op)]
    sourced = graph.nodes[path[0]].kind == "source" or (up.sourced if up else False)
    return _Branch(stages, sourced, origin=path[0])


def _fold_junction(
    nid: str,
    spec: NodeSpec,
    upstream: list[_Branch],
    skipped: list[str],
    materialized: list[str],
) -> "_Branch | None":
    """Fold a junction/selector: prune, pass through, or materialize a combinator.

    Zero live branches and the node is not there at all; one and it is
    *traversed* (recorded in ``skipped`` — the signal passes through unchanged);
    two or more and it becomes the ``SumOperator``/``SelectOperator`` the
    template says it is, recorded in ``materialized`` so
    :meth:`Assembly.replace_node` can refuse to overwrite it and drop the
    branches feeding it.

    Branch order is ``graph._in`` order — the template's edge declaration order,
    never the call-site order — so the same provided set always folds to the
    same tree: same names, same PRNG stream, same jit cache entry.

    Every summed branch must carry its own source. A transform-rooted branch
    arriving here generates nothing, so it would contribute the identity of
    whatever reached IT — which is the upstream branch, added to the sum a
    second time.
    """
    if not upstream:
        return None
    if len(upstream) == 1:
        skipped.append(nid)  # traversed pass-through junction/selector
        return upstream[0]
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
        combined: AbstractOperator = SumOperator(*branch_ops, names=branch_names)
    else:
        combined = SelectOperator(*branch_ops, names=branch_names, switch_key=nid)
    materialized.append(nid)
    return _Branch([(nid, combined)], sourced=True)


def _fold_source(
    graph: SignalGraph,
    nid: str,
    instances: list[AbstractOperator],
    fan: dict[str, list[_Branch]],
) -> "_Branch | None":
    """Fold a source node: nothing, one operator, or several composed by consumer.

    Several instances at a ``many`` source compose the way their CONSUMER
    composes. Feeding only selectors they stay separate branches — a switch
    picks one per sample, it does not add them up — and reach their consumers
    through ``fan``; anything else sums them here. ``exprs`` still gets the
    first fanned branch as a liveness marker, because that is what tells a
    downstream node the source is live at all; the consumer reads ``fan``.
    """
    if not instances:
        return None
    if len(instances) == 1:
        return _Branch([(nid, instances[0])], sourced=True)
    names = _instance_names(nid, len(instances))
    if _feeds_only_selectors(graph, nid):
        fan[nid] = [
            _Branch([(name, op)], sourced=True, origin=nid)
            for name, op in zip(names, instances, strict=True)
        ]
        return fan[nid][0]  # liveness marker; consumers read `fan`
    return _Branch([(nid, SumOperator(*instances, names=names))], sourced=True)


def _fold_transform(
    nid: str,
    instances: list[AbstractOperator],
    upstream: list[_Branch],
    skipped: list[str],
) -> "_Branch | None":
    """Fold a transform node: chain its instances onto the incoming branch.

    With no operator the node contracts to identity and the parent branch
    passes through unchanged — recorded in ``skipped`` only when there IS a
    parent, since a node with nothing upstream was never on the signal path to
    report a skip on. Several instances (the ``filters``-style ``many``
    transform) chain in call order.

    The branch keeps the PARENT's provenance: ``sourced`` and ``origin`` are
    what :func:`_fold_junction` reads to decide whether this branch may join a
    sum, and to name it if it may not.
    """
    up = upstream[0] if upstream else None
    if not instances:
        if up is not None:
            skipped.append(nid)
        return up
    stages = list(up.stages) if up else []
    stages += list(zip(_instance_names(nid, len(instances)), instances, strict=True))
    return _Branch(
        stages,
        sourced=up.sourced if up else False,
        origin=up.origin if up else nid,
    )


@dataclasses.dataclass(frozen=True)
class _Fold:
    """What folding a graph over a provided set produced.

    Attributes:
        final: the branch standing at the sink — the whole forward model.
        skipped: nodes traversed as identity, in fold order.
        multi: ``{node id: instance ids}`` for nodes carrying several operators.
        materialized: junction/selector nodes that became a combinator.
    """

    final: "_Branch | None"
    skipped: list[str]
    multi: dict[str, tuple[str, ...]]
    materialized: list[str]


def _fold_graph(
    graph: SignalGraph,
    placement: dict[str, list[AbstractOperator]],
    regions: Sequence[tuple[tuple[str, ...], AbstractOperator]],
) -> _Fold:
    """Walk the template in topological order, folding each node to a branch.

    The loop is a dispatch on what the node IS — covered by a region, a
    junction/selector, a source, or a transform — and each arm is one named
    helper above. Nodes are visited in ``graph._topo`` order, so every parent
    has folded before its children ask for it.
    """
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

    for nid in graph._topo:
        spec = graph.nodes[nid]
        instances = placement.get(nid, [])
        upstream = _upstream_of(graph, nid, exprs, fan)
        if len(instances) > 1:
            multi[nid] = _instance_names(nid, len(instances))

        if nid in region_of:
            idx = region_of[nid]
            path, region_op = regions[idx]
            exprs[nid] = _fold_region(
                graph, nid, path, region_op, upstream, exprs, region_entry, idx
            )
        elif spec.kind in ("junction", "selector"):
            exprs[nid] = _fold_junction(nid, spec, upstream, skipped, materialized)
        elif spec.kind == "source":
            exprs[nid] = _fold_source(graph, nid, instances, fan)
        else:
            exprs[nid] = _fold_transform(nid, instances, upstream, skipped)

    return _Fold(
        final=exprs[graph.sink],
        skipped=skipped,
        multi=multi,
        materialized=materialized,
    )


def assemble(
    graph: SignalGraph, *operators: AbstractOperator | At
) -> Assembly:
    """Compile a set of operators into the sub-pipeline they induce on ``graph``.

    See the module docstring for the contraction rules. Raises
    :class:`AssemblyError` on unknown/ambiguous placement, junction slots,
    duplicate single-instance nodes, an operator placed on a node of the other
    kind (:func:`_check_slot_kinds` — a source at a transform node or the
    reverse), a transform-rooted branch feeding a materialized junction (a sum
    branch must contain a source), or a violated
    :attr:`~rheplicant.core.operator.AbstractOperator.must_precede` ordering
    constraint.

    **Limitation — the envelope is in-trees.** The package's promise is that
    any assembled graph serves both forward modelling *and* inference. That
    holds while every node reaches the sink by exactly one path, and it is
    stated here rather than assumed because assembly folds the graph to a
    **tree**: a node reached by several paths is folded in once per path, so
    one operator object ends up at several positions. The forward model is
    right either way — each path contributes as the graph says. What breaks is
    *writing* to such a node afterwards, because ``eqx.tree_at`` rewrites the
    one position a selector reaches and leaves the other copies live: a
    finite, correctly-shaped, wrong model in which the parameter is only
    partly free. Those nodes are recorded in :attr:`Assembly.aliased`, and
    both :meth:`Assembly.replace_node` and
    :meth:`~rheplicant.inference.parameters.ParameterSpace.validate` refuse to
    write through them rather than answer wrongly.

    No shipped graph is affected: every node of the radio template reaches the
    sink by exactly one path, so ``aliased`` is always empty there. This bites
    user-defined graphs only.
    """
    if not operators:
        raise AssemblyError("assemble() needs at least one operator.")
    placement, regions = _resolve(graph, operators)
    _check_slot_kinds(graph, placement, regions)
    _check_ordering(graph, placement, regions)
    fold = _fold_graph(graph, placement, regions)

    final = fold.final
    # Unreachable, and kept as an assertion rather than deleted. `assemble`
    # refuses an empty operator list above, so at least one node is live; a
    # template has exactly one sink and no cycles, so following out-edges from
    # any node terminates at that sink; and liveness only ever propagates
    # downstream -- an instance-less transform passes its parent through, a
    # junction/selector with one live parent is traversed, and a region
    # contributes a live branch at `path[-1]`, which is the only position a
    # sink can occupy in a region (a sink has no out-edge to be interior by).
    # `tests/core/test_graph_template_guards.py` searches every single-sink
    # template on up to five nodes for a placement that empties the sink.
    if final is None:  # pragma: no cover - unreachable; see the note above
        raise AssemblyError("Nothing to assemble: no provided node reaches the sink.")

    operator = final.to_operator()
    # Addressing closure: the ids this assembly is about to promise must reach
    # the operators they name, and a node the fold duplicated cannot be written
    # through at all. Both are decided on the BUILT tree, so they cannot drift
    # from what `_find_named`/`eqx.tree_at` actually do to it.
    duplicates = _fold_duplicates(operator, placement, regions)
    _check_promised_ids(operator, fold.multi, placement, duplicates)

    claimed = set(placement) | {n for path, _ in regions for n in path}
    lit = tuple(n for n in graph.nodes if n in claimed)
    live_span = _live_span(graph, lit)
    return Assembly(
        operator=operator,
        graph_name=graph.name,
        lit=lit,
        skipped=tuple(n for n in fold.skipped if n in live_span),
        has_source=final.sourced,
        root_label=final.stages[0][0] if len(final.stages) == 1 else "",
        instances=tuple((nid, names) for nid, names in fold.multi.items()),
        materialized=tuple(fold.materialized),
        aliased=tuple(n for n in graph.nodes if n in duplicates),
        placements=_placement_addresses(graph, placement, regions),
    )


def _placement_addresses(
    graph: SignalGraph,
    placement: dict[str, list[AbstractOperator]],
    regions: Sequence[tuple[tuple[str, ...], AbstractOperator]],
) -> tuple[tuple[tuple[str, ...], str], ...]:
    """``(template nodes, address)`` per placed operator — the recipe `without` re-runs.

    The address is the id ``Assembly.__getitem__`` reaches that operator by:
    the node id for a single instance, the minted instance id when several sit
    on a ``many`` node (:func:`_instance_names` decides both, so the two cannot
    drift), and the LAST covered node for a region, which is how the class
    docstring says regions are addressed.

    Sorted by TEMPLATE order, not by the order the operators were provided in.
    ``assemble`` promises that argument order is irrelevant — two assemblies of
    the same operator set compare equal — and this field is part of the
    Assembly, so a record that remembered the call would quietly break that.
    """
    order = {nid: i for i, nid in enumerate(graph.nodes)}
    entries = [
        ((nid,), address)
        for nid, ops_at in placement.items()
        for address in _instance_names(nid, len(ops_at))
    ]
    entries += [(path, path[-1]) for path, _ in regions]
    return tuple(sorted(entries, key=lambda entry: (order[entry[0][0]], entry[1])))


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
