"""The template grammar: what a signal graph DECLARES.

A node spec, an `At` address, and the graph that holds them, plus the process
registry a named graph is looked up in. This half is a description; it places
nothing and assembles nothing, and it is the half a document author writes.
"""

import dataclasses
from collections.abc import Iterable, Mapping, Sequence
from typing import Literal

from rheplicant.core.errors import AssemblyError
from rheplicant.core.operator import AbstractOperator


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
    :func:`~rheplicant.core.graph.assemble` refuses both — see
    :func:`~rheplicant.core.fold._check_slot_kinds`.
    """

    node: str | tuple[str, ...]
    op: AbstractOperator


#: Mermaid's own three-class palette, one triplet per theme. Deliberately
#: separate from :data:`rheplicant.core.render._THEMES`, which keys on the five
#: SVG roles: mermaid has no wire colour and no per-kind fill, so the two
#: cannot share a table without one of them drifting to fit the other.
#: Each value is ``(fill, stroke, text)``.
_MERMAID_THEMES: dict[str, dict[str, tuple[str, str, str]]] = {
    "light": {
        "lit": ("#FAC775", "#854F0B", "#412402"),
        "wire": ("#F1EFE8", "#854F0B", "#444441"),
        "dim": ("#F1EFE8", "#B4B2A9", "#B4B2A9"),
    },
    "dark": {
        "lit": ("#4A3A12", "#E3B341", "#F0D896"),
        "wire": ("#1C1F24", "#8B949E", "#C9D1D9"),
        "dim": ("#1C1F24", "#3D4148", "#6E7681"),
    },
}


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
                raise AssemblyError(f"Transform node {n!r} must have in-degree <= 1, got {indeg}.")
            if spec.kind in ("junction", "selector") and indeg < 2:
                raise AssemblyError(
                    f"{spec.kind.capitalize()} node {n!r} must have in-degree >= 2, got {indeg}."
                )

    # -- rendering -----------------------------------------------------------

    def to_mermaid(
        self,
        lit: Iterable[str] = (),
        skipped: Iterable[str] = (),
        counts: Mapping[str, int] | None = None,
        theme: str = "light",
    ) -> str:
        """Render the template as a mermaid flowchart with lit/dim styling.

        ``lit`` nodes are highlighted, ``skipped`` (traversed-as-identity)
        nodes are half-lit, everything else is dimmed — the signal-path view
        of what an assembly simulates.

        ``counts`` maps a node id to the number of operator instances sitting
        on it. A ``many`` node is one box however many instances it carries,
        so the count is shown in the label: an unannotated box would render
        two components as one.

        Operators are **boxes**; the two composition operations are **symbols
        the wire runs through** and are given shapes of their own — a circled
        plus for a sum, a rhombus for a switch. Mermaid has no line art, so the
        shape carries the distinction here; ``to_svg`` draws the switch's lever.
        Both used to be circles differing only in their label, which made two
        operations *on* operators look like two more operators.

        ``theme`` is ``"light"`` or ``"dark"``, matching :meth:`to_svg` and
        :meth:`to_html`. An unknown name raises rather than falling back to a
        default, because a silently-light diagram in a dark page is exactly the
        failure the argument exists to prevent.
        """
        lit, skipped = set(lit), set(skipped)
        counts = dict(counts or {})
        lines = ["flowchart TD"]
        for n, spec in self.nodes.items():
            label = n.replace("_", " ")
            if counts.get(n, 1) > 1:
                label = f"{label} (x{counts[n]})"
            if spec.kind == "junction":
                shape = '(("+"))'  # circle + plus = the summing-junction symbol
            elif spec.kind == "selector":
                shape = '{"/"}'  # rhombus + lever = the switch symbol
            else:
                shape = f'["{label}"]'
            lines.append(f"  {n}{shape}")
        for a, b in self.edges:
            lines.append(f"  {a} --> {b}")
        palette = _MERMAID_THEMES[theme]  # KeyError names the unknown theme
        for cls in ("lit", "wire", "dim"):
            fill, stroke, text = palette[cls]
            lines.append(f"  classDef {cls} fill:{fill},stroke:{stroke},color:{text};")
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
        theme: str = "light",
    ) -> str:
        """Standalone HTML page of the template with lit/dim signal-path styling."""
        from rheplicant.core.render import signal_path_html

        return signal_path_html(
            self, lit=lit, skipped=skipped, title=title, counts=counts, theme=theme
        )

    def to_svg(
        self,
        lit: Iterable[str] = (),
        skipped: Iterable[str] = (),
        title: str | None = None,
        counts: Mapping[str, int] | None = None,
        theme: str = "light",
    ) -> str:
        """Self-contained ``<svg>`` of the template, for embedding (docs, notebooks).

        ``theme`` is ``"light"`` or ``"dark"``. An ``<img>``-embedded SVG cannot
        read the host page's theme, so a page that switches renders a pair.
        """
        from rheplicant.core.render import signal_path_svg

        return signal_path_svg(
            self, lit=lit, skipped=skipped, title=title, counts=counts, theme=theme
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
