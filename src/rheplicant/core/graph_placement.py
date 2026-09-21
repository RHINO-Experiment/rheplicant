"""Resolving a template's addresses onto a real operator tree.

Finding named nodes, walking children through nested assemblies, folding
duplicate placements and refusing overlapping claims. This is the arithmetic
of WHERE something goes; `Assembly` beside it decides what to do once the
answer is known.

Everything here needs the template types and nothing here needs `Assembly`,
which is what makes the seam hold: the three helpers that do need it stay
beside it in `graph.py`.
"""

from collections.abc import Sequence

from rheplicant.core.combinators import SelectOperator, SumOperator
from rheplicant.core.errors import AssemblyError
from rheplicant.core.fold import (
    _declared_node,
    _instance_names,
    _validate_region,
)
from rheplicant.core.operator import AbstractOperator
from rheplicant.core.pipeline import Pipeline, validate_operators

from .graph_template import (
    At,
    SignalGraph,
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


class _LeafPath:
    """One tagged leaf path, wrapped so that flattening cannot expand it.

    ``tree_map_with_path`` writing the bare path would leave a *tuple* in leaf
    position, and any later ``tree_leaves`` would flatten it into its
    components. An opaque object is a leaf, so the path reads back out whole.
    """

    __slots__ = ("path",)

    def __init__(self, path: tuple):
        self.path = path


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

    :func:`~rheplicant.core.fold._instance_names` mints ``x_1..x_n``;
    :func:`~rheplicant.core.fold._dedup` independently mints
    ``x, x_2, x_3, ...`` for repeated branch labels, and the two overlap
    from ``_2`` on. Both arise from the same graph shape — a node reaching a
    fold by several paths — so the collision is reported as what it is rather
    than as a naming accident. An id that resolves to something other than the
    operator placed there would be handed to the caller BY
    :class:`~rheplicant.core.errors.AmbiguousNodeError` and then written
    through, which is worse than saying nothing.
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


def _claimed_nodes(graph: SignalGraph, item: AbstractOperator | At) -> tuple[str, ...]:
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


def _placement_addresses(
    graph: SignalGraph,
    placement: dict[str, list[AbstractOperator]],
    regions: Sequence[tuple[tuple[str, ...], AbstractOperator]],
) -> tuple[tuple[tuple[str, ...], str], ...]:
    """``(template nodes, address)`` per placed operator — the recipe `without` re-runs.

    The address is the id ``Assembly.__getitem__`` reaches that operator by:
    the node id for a single instance, the minted instance id when several sit
    on a ``many`` node (:func:`~rheplicant.core.fold._instance_names` decides
    both, so the two cannot drift), and the LAST covered node for a region,
    which is how the class
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
