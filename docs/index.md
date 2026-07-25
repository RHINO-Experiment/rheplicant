# REPLICANT

A **REPLIC**a of an **ANT**enna — a JAX + Equinox framework for building
differentiable replicas of single-antenna radio telescopes: horns, dipoles,
and dishes alike.

A REPLICANT twin is one pure function from sky and instrument parameters to raw
data. Because every stage is differentiable, the same twin that *simulates*
an observation also *calibrates* it: gradients, Bayesian posteriors, Fisher
forecasts, and neural surrogates all run through the instrument model
itself.

## The eight principles

1. **Everything is an operator acting on a state** — one contract covers
   sky models, instrument effects, processing, filters, neural networks.
2. **The twin is a differentiable function** — `jit`/`grad`/`vmap` apply to
   the entire instrument; systematics become inferable parameters.
3. **Composition is physics, implicit in the signal path** — chains,
   sums, and switches assemble themselves from the canonical graph.
4. **Purity everywhere** — immutable states, randomness as data, one seed
   reproduces a run.
5. **Forward models never contain inference** — one seam
   (`build_forward_fn`) serves every inference engine.
6. **Interfaces first, physics second** — placeholder bodies, real tested
   contracts; ports replace functions, never structure.
7. **Loud failure over silent wrongness** — trace-time validation,
   provenance-tagged matrices, assembly-time graph errors.
8. **The core is domain-agnostic** — radio astronomy is the first
   application, not the design center (a test enforces the layering).

The [README](https://github.com/zzhang0123/replicant-telescope#readme) expands
each principle; start reading the docs with the [guided tour](tour.md).

```{toctree}
:maxdepth: 2

tour
operators
signal-path
api
design
changelog
```
