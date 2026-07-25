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
from collections.abc import Iterable

_NODE_W, _NODE_H = 150, 40
_X_GAP, _Y_GAP = 24, 42
_MARGIN = 40

_KIND_FILL = {
    "source": ("#EEEDFE", "#534AB7", "#3C3489"),
    "transform": ("#E6F1FB", "#185FA5", "#0C447C"),
    "junction": ("#F1EFE8", "#5F5E5A", "#444441"),
    "selector": ("#FAEEDA", "#854F0B", "#633806"),
}
_PROCESSING_FILL = ("#F1EFE8", "#5F5E5A", "#444441")
_LIT_STROKE = "#BA7517"

_STYLE = """
body { font-family: system-ui, sans-serif; background: #faf9f5; color: #2c2c2a;
       margin: 24px; }
h1 { font-size: 18px; font-weight: 600; }
p.legend { font-size: 13px; color: #5f5e5a; }
@media (prefers-color-scheme: dark) {
  body { background: #1f1e1b; color: #d3d1c7; }
  p.legend { color: #b4b2a9; }
}
"""

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


def signal_path_svg(
    graph,
    lit: Iterable[str] = (),
    skipped: Iterable[str] = (),
    title: str | None = None,
) -> str:
    """Render ``graph`` as a self-contained ``<svg>`` element (lit/dim styling).

    The opacity classes are styled inside the SVG, so the result embeds
    anywhere a plain image does — documentation pages, notebooks, ``<img>``.
    """
    # Deferred import: graph.py calls into this module from a method body, so a
    # top-level import here would merely be redundant, not cyclic — kept local
    # to keep render.py importable standalone.
    from replicant.core.graph import _live_span

    lit_set = set(lit)
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
        stroke = _LIT_STROKE if edge_lit else "#b4b2a9"
        cls = "lit" if edge_lit else "dim"
        stroke_w = 2 if edge_lit else 1
        parts.append(
            f'<line class="{cls}" x1="{xa:.0f}" y1="{ya + _NODE_H / 2:.0f}" '
            f'x2="{xb:.0f}" y2="{yb - _NODE_H / 2 - 4:.0f}" stroke="{stroke}" '
            f'stroke-width="{stroke_w}" marker-end="url(#arr)"/>'
        )
    for nid, spec in graph.nodes.items():
        x, y = centers[nid]
        fill, stroke, text = (
            _PROCESSING_FILL if spec.segment == "processing" else _KIND_FILL[spec.kind]
        )
        state = "lit" if nid in lit_set else ("wire" if nid in active else "dim")
        border = _LIT_STROKE if nid in lit_set else stroke
        border_w = 2 if nid in lit_set else 0.75
        label = _html.escape(nid.replace("_", " "))
        if spec.kind in ("junction", "selector"):
            symbol = "+" if spec.kind == "junction" else "sw"
            parts.append(
                f'<g class="{state}"><title>{_html.escape(nid)}</title>'
                f'<circle cx="{x:.0f}" cy="{y:.0f}" r="14" '
                f'fill="{fill}" stroke="{border}" stroke-width="{border_w}"/>'
                f'<text x="{x:.0f}" y="{y:.0f}" text-anchor="middle" '
                f'dominant-baseline="central" font-size="12" fill="{text}">{symbol}</text></g>'
            )
        else:
            dash = ' stroke-dasharray="5 4"' if spec.reserved else ""
            parts.append(
                f'<g class="{state}"><rect x="{x - _NODE_W / 2:.0f}" '
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
) -> str:
    """Render ``graph`` as a standalone HTML page with lit/dim signal-path styling."""
    page_title = _html.escape(title or f"Signal path: {graph.name}")
    lit_line = _html.escape(", ".join(sorted(set(lit))) or "none")
    svg = signal_path_svg(graph, lit=lit, skipped=skipped, title=title)
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{page_title}</title><style>{_STYLE}</style></head><body>"
        f"<h1>{page_title}</h1>"
        f"<p class='legend'>lit = provided operators ({lit_line}); half-lit = "
        "traversed as identity; dashed = reserved placeholder leaves.</p>"
        f"{svg}</body></html>"
    )
