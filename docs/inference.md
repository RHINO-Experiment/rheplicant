# Inference

```{include} _bayesmith-note.md
```

A digital twin is only half a research tool. The other half is running it
backwards: given data, what were the sky, the beam, the gain? RHEPLICANT does
that through one object, a **parameter space**, which every route reads: the
calibrators, the NumPyro bridge, the Fisher forecast, the conjugate solves, a
sampling plan and the amortized posterior. The arithmetic behind most of them
is bayesmith's; this layer says what is free and how it enters the twin.

JAX already fits a parameter the model stores. A parameter space covers the
case where the quantity to infer is not a number the model stores. A beam is
described by a width and a pointing offset; the model holds a response matrix.
A gain is one physical quantity; the model holds it in three stages. A positive
quantity is best explored in its logarithm. In each case the parameters you
want are a function of the numbers the model has. Deriving one from the other
is an inference concern, so it belongs in the inference layer, not in the
instrument description.

:::{list-table}
:header-rows: 1
:widths: 34 66

* - Page
  - What it answers
* - [Parameter spaces](inference-spaces.md)
  - What is free, and how it enters the twin. `Latent`, `Bind`, the three
    binding shapes, and what `validate()` can and cannot check.
* - [Noise, linear blocks, conditioning](inference-linear.md)
  - The noise model is the likelihood; a block that is linear in its latents
    has a posterior in closed form; and a small residual is not a small error.
* - [Plans and engines](inference-plans.md)
  - Several engines over one space, declared as a partition rather than written
    as a loop. Also the four routes, and inference with no likelihood at all.
* - [Evidence](evidence.md)
  - Accumulating likelihood factors across a campaign so the recordings can be
    thrown away. It covers a nuisance that drifts from night to night, and
    which campaign diagnostics can and cannot see a shared error.
:::

Two tutorials each work one problem end to end and are the place to start:
[an exact posterior for a big linear block](tutorial-gcr.md) and
[a gradient posterior, and how to tell it is wrong](tutorial-nuts.md).

The same routes can be declared in a YAML document instead of written in
Python: see [inference in a configuration document](config-inference.md).

```{toctree}
:maxdepth: 2
:hidden:

inference-spaces
inference-linear
inference-plans
evidence
```
