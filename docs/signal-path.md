# The canonical signal path

## Operators, and three ways to compose them

A signal path is not a special kind of object. It is **operators** — each one `State in, State out` — plus exactly **three structures** for putting them together. There is nothing else in the graph:

:::{list-table}
:header-rows: 1
:widths: 14 22 30 34

* - Structure
  - Combinator
  - Meaning
  - Node kind it comes from
* - **Cascade**
  - `Pipeline`
  - one after another; each transforms what the last produced
  - a chain of `transform` nodes
* - **Sum**
  - `SumOperator`
  - independent contributions that **add**
  - a `junction` node `(+)`
* - **Switch**
  - `SelectOperator`
  - alternatives, one **selected** per time sample
  - a `selector` node `(sw)`
:::

Nodes come in the matching four kinds: a **source** creates data (in-degree 0), a **transform** changes it (in-degree 1), a **junction** sums its inputs and a **selector** switches between them (both in-degree ≥ 2). That is the entire vocabulary. `assemble(*operators)` lights the sub-path your operators induce and folds it into exactly those three combinators, so the composition is a consequence of the physics you declared rather than something you wrote out.

Two rules follow from the table rather than being extra:

- **A junction or selector with one live input is traversed as identity.** No `SumOperator` around a single branch, no switch array for a twin with no calibration load. Partial models come free because a structure with nothing to combine is not a structure.
- **Several instances at one `many` node compose the way their consumer composes** — summed into a junction, switched at a selector. Two `CalLoadOperator`s are two switch positions, not one load worth their sum.

:::{admonition} The graph is a template, not the framework
:class: note
`rheplicant.radio.RADIO_GRAPH` is **RHINO's** structure: a single-antenna, switched-load, drift-scanning horn. It is the *default*, not the definition. The machinery underneath — `SignalGraph`, the four node kinds, the three combinators, `assemble` — knows nothing about radio astronomy, and a different instrument is a different template registered the same way:

```python
from rheplicant.core.graph import NodeSpec, SignalGraph, register_graph

MY_GRAPH = register_graph(SignalGraph(
    "my-instrument",
    {"emitter":  NodeSpec("source", "what I emit"),
     "response": NodeSpec("transform", "what my box does")},
    [("emitter", "response")],
))
```

*Planned:* a documented path for supplying a custom graph end to end — operators declaring `graph_node` against it, rendering, and the assembly rules — so that *which instrument* becomes a configuration choice rather than a fork. The pieces are already public and are what `RADIO_GRAPH` itself is built from; what is missing is the guide.
:::

---

## RHINO's template

The single-antenna path every assembly lights up. The diagram itself is generated from the live `rheplicant.radio.RADIO_GRAPH` at documentation build time, so it cannot drift from the code; the prose around it is written by hand. `(+)` nodes are sum junctions, `(sw)` the antenna/cal-load selector; see the [tour](tour.md#graph-assembly) for the assembly rules and [the operator catalog](operators.md) for what lives at each node.

```{mermaid} _generated/radio-graph.mmd
```

## Lit and dim: what an assembly simulates

`assemble(*operators)` never shows you only the piece you built — rendering always draws the *full* template, with the provided operators lit (colored, amber signal path), traversed-as-identity nodes half-lit, and everything else dimmed. Reserved placeholder leaves are dashed. Both examples below are real renders, generated at build time with `assembly.to_svg()` (`to_html()` produces the same figure as a standalone page).

A partial twin — five operators, `assemble(global_signal, foregrounds, ionosphere, beam, gain)`; every junction on the way passes through and the rest of the receiver chain is identity:

```{figure} signal-path-partial.svg
:alt: Partial twin with five lit operators

Partial twin: the beam-convolved sky through the gain.
```

A fuller twin — nine operators including RFI, ground pickup, atmospheric emission, and switched calibration loads (the `sw` selector node); note the two dashed entrances (`atmosphere_field` and `ground_field`) staying dim. Dashed means *reserved* — the node is part of the physics and no shipped operator declares it yet — which is a stronger claim than merely unlit, and one that goes stale the moment an operator lands. `t_sys_extra` was dashed until `BasisTemperatureOperator` arrived on it; the flag is now derived against the operator registry by a test, so the drawing cannot claim absent physics that is in fact present.

```{figure} signal-path-fuller.svg
:alt: Fuller twin with nine lit operators

Fuller twin: sky, RFI, ground, atmosphere, and calibration loads.
```
