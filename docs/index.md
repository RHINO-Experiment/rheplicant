# RHEPLICANT

```{image} _static/rheplicant-banner.png
:alt: rheplicant — digital twin for the RHINO experiment
:align: center
:width: 560px
```

A **REPLIC**a of an **ANT**enna — a JAX + Equinox framework for building
differentiable replicas of single-antenna radio telescopes: horns, dipoles,
and dishes alike.

A RHEPLICANT twin is one pure function from sky and instrument parameters to raw
data. Because every stage is differentiable, the same twin that *simulates*
an observation also *calibrates* it: gradients, Bayesian posteriors, Fisher
forecasts, and neural surrogates all run through the instrument model
itself.

## A twin is operators plus three structures

There are only two ingredients. **Operators** — each one `State in, State out`
— and **three ways to compose them**:

:::{list-table}
:header-rows: 1
:widths: 16 22 62

* - Structure
  - Combinator
  - Physics it expresses
* - **Cascade**
  - `Pipeline`
  - sequential effects: each stage transforms what the last produced
* - **Sum**
  - `SumOperator`
  - independent contributions that add into one signal
* - **Switch**
  - `SelectOperator`
  - alternative paths, one selected per time sample
:::

A *canonical signal path* is then just a template saying which operators exist
and which structure joins them: node kinds `source`, `transform`, `junction`
and `selector` map one-to-one onto "creates data", cascade, sum and switch. You
provide a set of operators, and [`assemble`](tour.md#graph-assembly) folds
them into exactly those three combinators — so composition is a consequence of
the physics you declared rather than something you wrote out.

The template shipped as the default,
[`RADIO_GRAPH`](signal-path.md), is **RHINO's** structure: a single-antenna,
switched-load, drift-scanning horn. It is a default, not the framework — the
machinery underneath knows nothing about radio astronomy, and another
instrument is another template registered the same way.

## The eight principles

1. **Everything is an operator acting on a state** — one contract covers
   sky models, instrument effects, processing, filters, neural networks; and
   exactly three structures compose them.
2. **The twin is a differentiable function** — `jit`/`grad`/`vmap` apply to
   the entire instrument; systematics become inferable parameters.
3. **Composition is physics, implicit in the signal path** — cascades,
   sums and switches assemble themselves from the canonical graph, which is
   a template you can replace rather than a fixed instrument.
4. **Purity everywhere** — immutable states, randomness as data, one seed
   reproduces a run.
5. **Forward models never contain inference** — one seam serves every
   inference engine, and a `ParameterSpace` re-parameterizes freely without
   ever editing the instrument description.
6. **Interfaces first, physics second** — placeholder bodies, real tested
   contracts; ports replace functions, never structure.
7. **Loud failure over silent wrongness** — trace-time validation,
   provenance-tagged matrices, assembly-time graph errors.
8. **The core is domain-agnostic** — radio astronomy is the first
   application, not the design center (a test enforces the layering).

The [README](https://github.com/RHINO-Experiment/rheplicant#readme) expands
each principle.

## Where to start

:::{list-table}
:header-rows: 1
:widths: 40 60

* - You want to…
  - Start here
* - install it
  - [Install](install.md) — limTOD comes from source, and `uv` needs `--frozen`
* - understand the whole thing in one sitting
  - [The guided tour](tour.md) — one worked example, simulated then inferred
* - simulate an instrument
  - [The canonical signal path](signal-path.md), then
    [the operator catalog](operators.md)
* - **turn a RHINO recording into a `State`**
  - [Ingestion](ingestion.md) — every other page starts from a `State` that
    already exists; this is where one comes from
* - see one instrument end to end
  - [From the sky to the receiver](sky-to-receiver.md)
* - fit or sample parameters
  - [Tutorial: an exact posterior](tutorial-gcr.md) or
    [Tutorial: a gradient posterior](tutorial-nuts.md) first, then
    [Inference](inference.md) for the rules they cite
* - keep a campaign after the recordings are archived
  - [Evidence](evidence.md) — accumulate likelihood factors, discard the data
* - run something and read its output
  - [Examples](examples.md) — thirteen scripts, with measured wall clocks
* - look something up
  - [The API reference](api.md), [contracts between stages](contracts.md),
    [design decisions](design.md)
:::

```{toctree}
:maxdepth: 2
:caption: Start here

install
tour
ingestion
```

```{toctree}
:maxdepth: 2
:caption: Concepts

signal-path
operators
contracts
```

```{toctree}
:maxdepth: 2
:caption: The instrument

sky-to-receiver
sky-engines
```

```{toctree}
:maxdepth: 2
:caption: Inference

inference
tutorial-gcr
tutorial-nuts
```

```{toctree}
:maxdepth: 2
:caption: Reference

examples
api
design
changelog
```
