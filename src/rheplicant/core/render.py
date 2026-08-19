"""Standalone HTML/SVG rendering of signal-path graphs with lit/dim styling.

Shows the full template with the provided nodes lit, traversed-as-identity
nodes half-lit ("wire"), and everything else dimmed — the signal-path view
of what an assembly simulates. Produced from Python so it always reflects
the actual template. Two output forms, both self-contained (no external
assets — the opacity classes are styled inside the SVG itself):

- :func:`signal_path_html` — a full page; write it to a file and open it
  in a browser: ``pathlib.Path("signal_path.html").write_text(assembly.to_html())``
- :func:`signal_path_svg` — just the ``<svg>`` element, for embedding in
  documentation or notebooks (``assembly.to_svg()``).
"""

import html as _html
from collections.abc import Iterable, Mapping

_NODE_W, _NODE_H = 150, 40
_X_GAP, _Y_GAP = 24, 42
_MARGIN = 40

#: The two composition symbols, sized against the operator box they must not be
#: mistaken for. An operator is a BOX the signal enters and leaves; a sum or a
#: switch is a SYMBOL the wire passes through, so both are drawn small, unfilled,
#: in the wire's own colour, and the edges stop at their outline rather than at a
#: box that is not there. ``+`` and ``sw`` used to be two labels in one circle,
#: which made the three compositions look like one kind of thing and both
#: operations look like peers of the operators around them.
_SUM_R = 10  # circled plus: radius
_SW_R = 15  # switch diamond: half-diagonal

#: One palette per theme. Operators take a KIND colour; the two composition
#: symbols take ``wire``, so "operator" and "operation on operators" stay apart
#: even where colour is all a reader has.
_THEMES = {
    "light": {
        "source": ("#EEEDFE", "#534AB7", "#3C3489"),
        "transform": ("#E6F1FB", "#185FA5", "#0C447C"),
        "processing": ("#F1EFE8", "#5F5E5A", "#444441"),
        "wire": "#8C8A82",
        "lit": "#BA7517",
    },
    "dark": {
        "source": ("#241E3D", "#A371F7", "#D2A8FF"),
        "transform": ("#0D2137", "#58A6FF", "#A5D6FF"),
        "processing": ("#1C1F24", "#8B949E", "#C9D1D9"),
        "wire": "#6E7681",
        "lit": "#E3B341",
    },
}

_STYLE = """
body { font-family: system-ui, sans-serif; background: #faf9f5; color: #2c2c2a;
       margin: 24px; }
h1 { font-size: 18px; font-weight: 600; }
p.legend { font-size: 13px; color: #5f5e5a; }
"""
_DARK_PAGE = "body{background:#1f1e1b;color:#d3d1c7}p.legend{color:#b4b2a9}"

# Inside the SVG so the rendering survives standalone embedding (<img>, docs).
_SVG_STYLE = ".lit{opacity:1}.wire{opacity:.55}.dim{opacity:.22}"


def _layers(graph) -> dict[str, int]:
    """Layer assignment: longest path from the roots, with root sources
    pulled down to sit just above their consumers (so e.g. calibration
    loads render next to the switch they feed instead of at the top with
    an edge crossing the whole diagram)."""
    layer: dict[str, int] = {}
    for nid in graph._topo:
        parents = graph._in[nid]
        layer[nid] = 0 if not parents else max(layer[p] for p in parents) + 1
    for nid in graph.nodes:
        if not graph._in[nid] and graph._out[nid]:
            layer[nid] = min(layer[s] for s in graph._out[nid]) - 1
    return layer


def _palette(theme: str) -> dict:
    """The colours for ``theme``, refusing a name that has none."""
    if theme not in _THEMES:
        raise ValueError(f"Unknown theme {theme!r}; known themes: {list(_THEMES)}.")
    return _THEMES[theme]


def _page_style(theme: str) -> str:
    """Chrome for the standalone page, matched to the figure it wraps.

    ``theme="dark"`` PINS a dark page, because a dark-palette figure on light
    chrome is unreadable. The default keeps following the reader's system, as it
    did before there was a choice: the light palette is pale boxes with dark
    text of their own and stays legible on either background.
    """
    if theme == "dark":
        return _STYLE + _DARK_PAGE
    return _STYLE + f"@media (prefers-color-scheme: dark) {{{_DARK_PAGE}}}"


def _half_height(kind: str) -> float:
    """How far a node reaches above and below its centre, for edge attachment.

    The wire has to STOP at an operator's box and RUN THROUGH a sum or a switch,
    so it is drawn to the symbol's own outline. Attaching every edge at the box
    half-height instead left a gap around symbols that are smaller than a box,
    and the gap is what made them read as undersized nodes.
    """
    if kind == "junction":
        return _SUM_R
    if kind == "selector":
        return _SW_R
    return _NODE_H / 2


def _sum_symbol(x: float, y: float, stroke: str, width: float) -> str:
    """A circled plus on the wire: the branches reaching this node ADD."""
    return (
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{_SUM_R}" fill="none" '
        f'stroke="{stroke}" stroke-width="{width}"/>'
        f'<path d="M{x - 5:.1f} {y:.1f}h10M{x:.1f} {y - 5:.1f}v10" fill="none" '
        f'stroke="{stroke}" stroke-width="{width}" stroke-linecap="round"/>'
    )


def _switch_symbol(x: float, y: float, stroke: str, width: float) -> str:
    """A single-pole lever in a diamond: ONE branch is connected per sample.

    Drawn the way the flow runs — the throws are at the top where the branches
    arrive, the pole at the bottom where the one selected signal leaves, and the
    lever reaches only one of them. It is the electrical symbol rather than a
    letter because what has to be legible here is *what happens*, not the node's
    name, which the ``<title>`` carries for anyone who hovers.
    """
    return (
        f'<path d="M{x:.1f} {y - _SW_R:.1f}L{x + _SW_R:.1f} {y:.1f}'
        f'L{x:.1f} {y + _SW_R:.1f}L{x - _SW_R:.1f} {y:.1f}Z" fill="none" '
        f'stroke="{stroke}" stroke-width="{width}" stroke-linejoin="round"/>'
        f'<circle cx="{x:.1f}" cy="{y + 7:.1f}" r="2" fill="{stroke}"/>'
        f'<circle cx="{x + 6.5:.1f}" cy="{y - 5.5:.1f}" r="1.8" fill="{stroke}"/>'
        f'<path d="M{x:.1f} {y + 7:.1f}L{x - 7:.1f} {y - 6:.1f}" fill="none" '
        f'stroke="{stroke}" stroke-width="1.8" stroke-linecap="round"/>'
    )


def signal_path_svg(
    graph,
    lit: Iterable[str] = (),
    skipped: Iterable[str] = (),
    title: str | None = None,
    counts: Mapping[str, int] | None = None,
    theme: str = "light",
) -> str:
    """Render ``graph`` as a self-contained ``<svg>`` element (lit/dim styling).

    The opacity classes are styled inside the SVG, so the result embeds
    anywhere a plain image does — documentation pages, notebooks, ``<img>``.

    ``counts`` maps a node id to how many operator instances an assembly put
    on it; ``many`` nodes carrying more than one say so in their label,
    because the template draws them as a single box either way.

    ``theme`` selects the palette (``"light"`` or ``"dark"``). One SVG cannot
    serve both: embedded as an ``<img>`` it cannot see the page's theme class,
    which is why the documentation commits a pair and shows one of them.
    """
    # Deferred import: graph.py calls into this module from a method body, so a
    # top-level import here would merely be redundant, not cyclic — kept local
    # to keep render.py importable standalone.
    from rheplicant.core.graph import _live_span

    pal = _palette(theme)
    lit_set = set(lit)
    counts = dict(counts or {})
    # skipped nodes are traversed-as-identity; the live span normally covers
    # them, but explicit skipped input keeps callers authoritative.
    active = lit_set | set(skipped) | _live_span(graph, tuple(lit_set))

    layer = _layers(graph)
    by_layer: dict[int, list[str]] = {}
    for nid in graph.nodes:
        by_layer.setdefault(layer[nid], []).append(nid)
    n_layers = max(by_layer) + 1 if by_layer else 0
    pitch = _NODE_W + _X_GAP

    # Barycenter sweeps: pull each node toward the mean x of its neighbours,
    # so the trunk aligns vertically and sources flank their junctions.
    xs: dict[str, float] = {}
    for nids in by_layer.values():
        for i, nid in enumerate(nids):
            xs[nid] = i * pitch
    for _ in range(4):
        for nids in by_layer.values():
            targets = {}
            for nid in nids:
                neighbours = graph._in[nid] + graph._out[nid]
                targets[nid] = (
                    sum(xs[m] for m in neighbours) / len(neighbours)
                    if neighbours else xs[nid]
                )
            order = sorted(nids, key=lambda n: targets[n])
            mean_target = sum(targets.values()) / len(targets)
            start = mean_target - (len(nids) - 1) * pitch / 2
            for i, nid in enumerate(order):
                xs[nid] = start + i * pitch

    x_min = min(xs.values())
    x_max = max(xs.values())
    width = int(x_max - x_min + _NODE_W + 2 * _MARGIN)
    height = 2 * _MARGIN + n_layers * (_NODE_H + _Y_GAP)
    centers: dict[str, tuple[float, float]] = {}
    for lyr, nids in by_layer.items():
        for nid in nids:
            centers[nid] = (
                xs[nid] - x_min + _NODE_W / 2 + _MARGIN,
                _MARGIN + lyr * (_NODE_H + _Y_GAP) + _NODE_H / 2,
            )

    parts: list[str] = []
    parts.append(f"<style>{_SVG_STYLE}</style>")
    parts.append(
        '<defs><marker id="arr" viewBox="0 0 10 10" refX="8" refY="5" '
        'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
        '<path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" '
        'stroke-width="1.5" stroke-linecap="round"/></marker></defs>'
    )
    for a, b in graph.edges:
        (xa, ya), (xb, yb) = centers[a], centers[b]
        edge_lit = a in active and b in active
        stroke = pal["lit"] if edge_lit else pal["wire"]
        cls = "lit" if edge_lit else "dim"
        stroke_w = 2 if edge_lit else 1
        y1 = ya + _half_height(graph.nodes[a].kind)
        y2 = yb - _half_height(graph.nodes[b].kind) - 4
        parts.append(
            f'<line class="{cls}" x1="{xa:.0f}" y1="{y1:.0f}" '
            f'x2="{xb:.0f}" y2="{y2:.0f}" stroke="{stroke}" '
            f'stroke-width="{stroke_w}" marker-end="url(#arr)"/>'
        )
    for nid, spec in graph.nodes.items():
        x, y = centers[nid]
        escaped_id = _html.escape(nid, quote=True)
        escaped_kind = _html.escape(spec.kind, quote=True)
        description = f": {spec.doc}" if spec.doc else ""
        title_text = _html.escape(f"{spec.kind} node {nid}{description}")
        if spec.kind in ("junction", "selector"):
            # Wire furniture: the composition symbols carry the WIRE's colour,
            # never a node colour, and light up with the wire that runs through
            # them. They are never operator slots, so `lit` never names one.
            on = nid in active
            # Class "lit"/"dim", never "wire": the half-lit tier says "an
            # operator could sit here and none does", and none ever can here.
            # At .55 the symbol would also read as fainter than the full-opacity
            # wire drawn through it, which is the one thing it must not do.
            draw = _sum_symbol if spec.kind == "junction" else _switch_symbol
            parts.append(
                f'<g class="{"lit" if on else "dim"}" data-node-id="{escaped_id}" '
                f'aria-disabled="true" data-node-kind="{escaped_kind}">'
                f"<title>{title_text}; this composition node is not an operator slot.</title>"
                f'{draw(x, y, pal["lit"] if on else pal["wire"], 1.6 if on else 1.1)}</g>'
            )
            continue
        state = "lit" if nid in lit_set else ("wire" if nid in active else "dim")
        fill, border, text = pal[
            "processing" if spec.segment == "processing" else spec.kind
        ]
        if nid in lit_set:
            border = pal["lit"]
        border_w = 2 if nid in lit_set else 0.75
        text_label = nid.replace("_", " ")
        if counts.get(nid, 1) > 1:
            text_label = f"{text_label} (x{counts[nid]})"
        label = _html.escape(text_label)
        dash = ' stroke-dasharray="5 4"' if spec.reserved else ""
        parts.append(
            f'<g class="{state}" data-node-id="{escaped_id}" role="button" tabindex="0" '
            f'data-node-kind="{escaped_kind}" aria-label="Edit {escaped_id}">'
            f'<title>{title_text}</title><rect x="{x - _NODE_W / 2:.0f}" '
            f'y="{y - _NODE_H / 2:.0f}" width="{_NODE_W}" height="{_NODE_H}" '
            f'rx="8" fill="{fill}" stroke="{border}" stroke-width="{border_w}"{dash}/>'
            f'<text x="{x:.0f}" y="{y:.0f}" text-anchor="middle" '
            f'dominant-baseline="central" font-size="12.5" fill="{text}">{label}</text></g>'
        )

    label = _html.escape(title or f"Signal path: {graph.name}")
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="{label}">'
        f'{"".join(parts)}</svg>'
    )


def signal_path_html(
    graph,
    lit: Iterable[str] = (),
    skipped: Iterable[str] = (),
    title: str | None = None,
    counts: Mapping[str, int] | None = None,
    theme: str = "light",
) -> str:
    """Render ``graph`` as a standalone HTML page with lit/dim signal-path styling."""
    page_title = _html.escape(title or f"Signal path: {graph.name}")
    counts = dict(counts or {})
    lit_line = _html.escape(
        ", ".join(
            f"{nid} (x{counts[nid]})" if counts.get(nid, 1) > 1 else nid
            for nid in sorted(set(lit))
        )
        or "none"
    )
    svg = signal_path_svg(
        graph, lit=lit, skipped=skipped, title=title, counts=counts, theme=theme
    )
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{page_title}</title><style>{_page_style(theme)}</style></head><body>"
        f"<h1>{page_title}</h1>"
        f"<p class='legend'>Boxes are operators: lit = provided ({lit_line}); "
        "half-lit = traversed as identity; dashed = reserved placeholder leaves. "
        "The two symbols on the wire are not operators — &#8853; sums the branches "
        "reaching it, the lever in the diamond selects one of them per sample.</p>"
        f"{svg}</body></html>"
    )
