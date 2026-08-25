# Inference

```{include} _migration-to-bayesmith.md
```

A digital twin is only half a research tool. The other half is running it
backwards: given data, what were the sky, the beam, the gain? RHEPLICANT does
that through one object — a **parameter space** — that every inference engine
reads: the two calibrators, the NumPyro bridge, Fisher forecasting, and the
conjugate-Gaussian solver.

The design question it answers is not "how do I fit a parameter", which JAX
already answers. It is: **what if the thing you want to infer is not a number
the model happens to store?** A beam is described by a width and a pointing
offset; the model holds a response matrix. A gain is one physical quantity;
the model holds it in three stages. A positive quantity is best explored in
its logarithm. In every case the parameters you want are a *function* of the
numbers the model has — and re-deriving them is an inference concern, so it
belongs in the inference layer, not in the instrument description.

:::{list-table}
:header-rows: 1
:widths: 34 66

* - Page
  - What it answers
* - [Parameter spaces](inference-spaces.md)
  - What is free, and how it enters the twin. `Latent`, `Bind`, the three
    binding shapes, and what `validate()` can and cannot check.
* - [Noise, linear blocks, conditioning](inference-linear.md)
  - The noise model *is* the likelihood; a block that is linear in its latents
    has a posterior in closed form; and a small residual is not a small error.
* - [Plans and engines](inference-plans.md)
  - Several engines over one space, declared as a partition rather than written
    as a loop. Also the four routes, and inference with no likelihood at all.
* - [Evidence](evidence.md)
  - Accumulating likelihood factors across a campaign so the recordings can be
    thrown away — including a nuisance that drifts from night to night, and
    which campaign diagnostics can and cannot see a shared error.
:::

Two tutorials work one problem each end to end, and are the fastest way in:
[an exact posterior for a big linear block](tutorial-gcr.md) and
[a gradient posterior, and how to tell it is wrong](tutorial-nuts.md).

```{toctree}
:maxdepth: 2
:hidden:

inference-spaces
inference-linear
inference-plans
evidence
```
