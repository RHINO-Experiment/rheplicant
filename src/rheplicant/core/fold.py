"""The fold: how a provided operator set contracts onto a SignalGraph template.

:func:`~rheplicant.core.graph.assemble` does two things. It *resolves* a
provided set into claims on template nodes -- which is about the call, and
stays in :mod:`rheplicant.core.graph` beside the classes it validates -- and
then it **folds** the template over those claims into the ordinary
``Pipeline``/``SumOperator``/``SelectOperator`` nesting they induce. This
module is the second half: one helper per node kind, the two placement checks
the fold's derivations depend on, and the small naming algebra all of them
share.

**The dependency points one way, and that is the whole reason this file can
exist.** ``graph.py`` imports from here; nothing here imports ``graph.py``.
``SignalGraph`` and ``NodeSpec`` appear only in ANNOTATIONS -- every use is
attribute access on a value (``graph._in``, ``spec.kind``), never an
``isinstance``, a default or a construction -- so ``from __future__ import
annotations`` plus a ``TYPE_CHECKING`` import is enough, and the runtime edge
disappears. ``AssemblyError`` used to make this impossible: the fold raises it,
so it was needed at runtime, and while it was defined in ``graph.py`` any
module that folded had to import ``graph.py`` back. It now lives in
:mod:`rheplicant.core.errors`, which is what unblocked the split.

Three of the helpers here -- :func:`_declared_node`, :func:`_instance_names`
and :func:`_validate_region` -- are also called by ``graph.py``'s resolve step.
They live on this side because the fold needs them too and the edge has to
point one way; putting them in ``graph.py`` would restore the cycle for the
sake of the shorter import list.

Nothing in this module is public. It is deliberately absent from
``docs/api.md``: every name starts with an underscore, so an
``automodule`` directive would document zero members while adding a page
heading that promises otherwise.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import TYPE_CHECKING

from rheplicant.core.combinators import SelectOperator, SumOperator
from rheplicant.core.errors import AssemblyError
from rheplicant.core.operator import AbstractOperator
from rheplicant.core.pipeline import Pipeline

if TYPE_CHECKING:  # annotations only -- see the module docstring
    from rheplicant.core.graph import NodeSpec, SignalGraph


# ---------------------------------------------------------------------------
# naming algebra and template queries -- the vocabulary the rest of the file
# (and graph.py's resolve step) is written in
# ---------------------------------------------------------------------------


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
    :func:`~rheplicant.core.graph._find_named` resolves that collision to the
    fold: ``assembly[x]`` then hands back a ``SumOperator`` and
    ``replace_node(x, ...)`` overwrites every instance with one operator.
    Making the bare id name nothing turns that into
    :class:`~rheplicant.core.errors.AmbiguousNodeError`, which can say what to
    write.
    """
    if count == 1:
        return (node_id,)
    return tuple(f"{node_id}_{i}" for i in range(1, count + 1))


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


# ---------------------------------------------------------------------------
# placement checks -- run by `assemble` BEFORE the fold, because both of the
# fold's derivations (`has_source`, "a summed branch carries its own source")
# read the template's node kinds and are only sound while the operator at a
# node does what its kind says
# ---------------------------------------------------------------------------


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
    of what this misses, it is stated in
    :func:`~rheplicant.core.graph.assemble`'s docstring, and it is
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


# ---------------------------------------------------------------------------
# the fold: one helper per node kind, dispatched from _fold_graph's loop
#
# Each helper answers one question -- "what branch does THIS node contribute?"
# -- and returns it, so the loop below is a dispatch on node kind and nothing
# else. The accumulators a helper records into (`skipped`, `materialized`,
# `fan`, `entry`) are named in its signature rather than closed over, so what
# each one writes is readable without reading the loop.
# ---------------------------------------------------------------------------


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


def _upstream_of(
    graph: SignalGraph,
    nid: str,
    exprs: dict[str, _Branch | None],
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
    exprs: dict[str, _Branch | None],
    entry: dict[int, _Branch | None],
    idx: int,
) -> _Branch | None:
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
) -> _Branch | None:
    """Fold a junction/selector: prune, pass through, or materialize a combinator.

    Zero live branches and the node is not there at all; one and it is
    *traversed* (recorded in ``skipped`` — the signal passes through unchanged);
    two or more and it becomes the ``SumOperator``/``SelectOperator`` the
    template says it is, recorded in ``materialized`` so
    :meth:`~rheplicant.core.graph.Assembly.replace_node` can refuse to
    overwrite it and drop the
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
) -> _Branch | None:
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
) -> _Branch | None:
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

    final: _Branch | None
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
