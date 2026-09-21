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
wrong at once; :func:`~rheplicant.core.fold._check_slot_kinds` refuses the
disagreement instead.

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
``x_2``, … and the bare ``x`` raises
:class:`~rheplicant.core.errors.AmbiguousNodeError` listing them.
With one instance nothing changes — ``x`` is still the address, and a
:class:`~rheplicant.inference.parameters.ParameterSpace` written against it is
untouched until a sibling actually arrives.
"""

from collections.abc import Iterable

import equinox as eqx
import jax

from rheplicant.core.errors import AmbiguousNodeError, AssemblyError
from rheplicant.core.fold import (
    _check_ordering,
    _check_slot_kinds,
    _fold_graph,
)
from rheplicant.core.operator import AbstractOperator
from rheplicant.core.pipeline import validate_operators
from rheplicant.core.state import State

from .graph_placement import _check_disjoint_claims as _check_disjoint_claims
from .graph_placement import (
    _check_promised_ids,
    _children,
    _find_named,
    _fold_duplicates,
    _LeafPath,
    _live_span,
    _placement_addresses,
    _resolve,
)
from .graph_placement import _claimed_nodes as _claimed_nodes
from .graph_placement import _descend_to_own_stage as _descend_to_own_stage
from .graph_placement import _place_at_node as _place_at_node
from .graph_placement import _positions as _positions
from .graph_template import _GRAPHS as _GRAPHS
from .graph_template import _MERMAID_THEMES as _MERMAID_THEMES
from .graph_template import (
    At,
    SignalGraph,
    get_graph,
)

# Re-exported so this module's importers keep working. `X as X` is the
# explicit re-export form: a plain import of a name this file does not
# itself use is F401, and `ruff --fix` deletes it whatever the comment
# on the line says.
from .graph_template import NodeSpec as NodeSpec
from .graph_template import register_graph as register_graph

# The fold lives in `core/fold.py` and the arrow points ONE way: this module
# imports those six, and nothing there imports this one. `SignalGraph` and
# `NodeSpec` reach the fold only as annotations, so a `TYPE_CHECKING` import
# covers them; what used to make the split impossible was `AssemblyError`,
# which the fold RAISES and so needed at runtime -- see `fold.py`'s docstring.
# Imported under their own names rather than as `fold.x` so that the call
# sites read as they always did and a monkeypatch of this module's attribute
# (which `tests/core/test_graph.py` does to `_check_promised_ids`) keeps
# working the same way for any of them.

# `AmbiguousNodeError` and `AssemblyError` are imported above, not defined
# here: they moved to `core/errors.py` -- where every other error in the
# package lives -- so a module wanting to raise one need not import this file
# and take its whole dependency tree along. They stay importable from here
# because that is where they were defined for the package's history and
# thirteen files still reach for them at this path.
#
# Deliberately NO `__all__`. Adding one listing just those two truncated
# `automodule:: rheplicant.core.graph` to exactly those two members --
# SignalGraph, NodeSpec, At, Assembly, assemble, register_graph and get_graph
# all vanished from the API page, silently, because a member that is not
# documented raises no warning. If an `__all__` is ever wanted here it has to
# be the module's whole public surface.


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
    operators, and that is only sound because
    :func:`~rheplicant.core.fold._check_slot_kinds` refuses a placement whose
    operator disagrees with its node about creating data. An operator
    declaring no ``graph_node`` cannot be screened, so it can still be
    placed on the wrong kind of node and make this flag — and therefore the
    guard above — wrong in either direction.
    """

    operator: AbstractOperator
    graph_name: str = eqx.field(static=True)
    lit: tuple[str, ...] = eqx.field(static=True)
    skipped: tuple[str, ...] = eqx.field(static=True)
    has_source: bool = eqx.field(static=True)
    root_label: str = eqx.field(static=True, default="")
    instances: tuple[tuple[str, tuple[str, ...]], ...] = eqx.field(static=True, default=())
    materialized: tuple[str, ...] = eqx.field(static=True, default=())
    aliased: tuple[str, ...] = eqx.field(static=True, default=())
    # The recipe, as ADDRESSES rather than operators: (template nodes, the id
    # `self[...]` reaches this operator by). Storing the operators themselves
    # would put every one of them at a second pytree position, which is exactly
    # the `aliased` failure this class refuses to create. Addresses are static
    # strings, so the fold stays the only place an operator lives, and
    # `without` recovers the set by reading them back off the built tree.
    placements: tuple[tuple[tuple[str, ...], str], ...] = eqx.field(static=True, default=())

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
        (:class:`~rheplicant.core.errors.AmbiguousNodeError`), a junction/selector that assembly
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
        :func:`~rheplicant.core.fold._check_ordering` applies to a node that
        was never lit.

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
            *(At(nodes if len(nodes) > 1 else nodes[0], self[address]) for nodes, address in kept),
        )

    @property
    def _counts(self) -> dict[str, int]:
        """Instances per node, for renderings — one lit box may be several."""
        return {nid: len(names) for nid, names in self.instances}

    def to_mermaid(self, theme: str = "light") -> str:
        """Lit/dim mermaid rendering via the registered template.

        ``theme`` is ``"light"`` or ``"dark"``; see :meth:`SignalGraph.to_mermaid`.
        """
        return get_graph(self.graph_name).to_mermaid(
            lit=self.lit, skipped=self.skipped, counts=self._counts, theme=theme
        )

    def to_html(self, title: str | None = None, theme: str = "light") -> str:
        """Standalone HTML page: the full graph with this assembly's nodes lit."""
        return get_graph(self.graph_name).to_html(
            lit=self.lit,
            skipped=self.skipped,
            title=title,
            counts=self._counts,
            theme=theme,
        )

    def to_svg(self, title: str | None = None, theme: str = "light") -> str:
        """Self-contained ``<svg>`` with this assembly's nodes lit, for embedding.

        ``theme`` is ``"light"`` or ``"dark"``; see :meth:`SignalGraph.to_svg`.
        """
        return get_graph(self.graph_name).to_svg(
            lit=self.lit,
            skipped=self.skipped,
            title=title,
            counts=self._counts,
            theme=theme,
        )

    def __repr__(self) -> str:
        # `lit` names template nodes, and a node stays one node however many
        # operators sit on it — so the multiplicity is reported beside it
        # rather than folded into the list, where it would read as a node id.
        counts = self._counts
        lit = [f"{nid} x{counts[nid]}" if nid in counts else nid for nid in self.lit]
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
        fan_out = f", aliased-at-several-positions={list(self.aliased)}" if self.aliased else ""
        return (
            f"Assembly(graph={self.graph_name!r}, lit={lit}, "
            f"skipped-as-identity={list(self.skipped)}{fan_out})"
        )


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
            queue.extend(zip(children, _children_through_assemblies(twin), strict=True))


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
    if not any(isinstance(op, Assembly) and op.aliased for op, _ in _spine_pairs(pipeline)):
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
                        (tag.path, node_id) for tag in jax.tree_util.tree_leaves(below_twin)
                    )
    return owned


def assemble(graph: SignalGraph, *operators: AbstractOperator | At) -> Assembly:
    """Compile a set of operators into the sub-pipeline they induce on ``graph``.

    See the module docstring for the contraction rules. Raises
    :class:`~rheplicant.core.errors.AssemblyError` on unknown/ambiguous placement, junction slots,
    duplicate single-instance nodes, an operator placed on a node of the other
    kind (:func:`~rheplicant.core.fold._check_slot_kinds` — a source at a
    transform node or the reverse), a transform-rooted branch feeding a
    materialized junction (a sum
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
