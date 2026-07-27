# Inferring anything: parameter spaces

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

---

## Two words

Everything rests on separating two ideas that are easy to conflate.

`Latent`
: **A named quantity you infer.** It is what a sampler draws or an optimizer
  steps: a name, an initial value (which fixes its shape and dtype), optionally
  a prior, optionally a declaration that it enters the model linearly. A latent
  knows *nothing* about the pipeline. `log_gain` is a latent; so is a
  10⁴-element vector of sky `alm` coefficients.

`Bind`
: **A rule turning latents into pipeline leaf values.** It names the latents it
  consumes, the leaves it writes, and optionally the function between them. A
  bind knows nothing about priors.

A pipeline leaf is *what the instrument model holds*. A latent is *what you
chose to infer*. Keeping them apart is what makes re-parameterization free.

```{mermaid}
flowchart LR
    subgraph L["what you infer — Latent"]
        A["fwhm"]
        B["log_e"]
        C["log_gain"]
    end
    subgraph BD["how it enters — Bind"]
        D["fn: gaussian_beam_alms"]
        E["fn: exp"]
    end
    subgraph P["what the model holds — pipeline leaves"]
        F["projector.beam_alms<br/>(n_freq, n_alm)"]
        G["gain_lna.gain"]
        H["gain_backend.gain"]
    end
    A --> D
    B --> D
    D --> F
    C --> E
    E --> G
    E --> H
```

Note what the diagram makes obvious: the arrows are **many-to-many**. Two
latents produce one leaf; one latent produces two. That is exactly what a
scheme attaching priors directly to leaves cannot express.

---

## The three shapes

::::{grid} 1 1 3 3
:gutter: 2

:::{grid-item-card} Direct
:class-header: sd-font-weight-bold

One latent, one leaf, unchanged.
^^^
```python
Bind("sky_alms",
     into=lambda p: p["sky"]
                     .sky_model.alms)
```
+++
The case the old positional priors covered.
:::

:::{grid-item-card} Tied
:class-header: sd-font-weight-bold

One latent, several leaves, transformed.
^^^
```python
Bind("log_gain",
     into=(lambda p: p["gain_lna"].gain,
           lambda p: p["gain_bk"].gain),
     fn=jnp.exp)
```
+++
One physical quantity, three stages.
:::

:::{grid-item-card} Derived
:class-header: sd-font-weight-bold

Several latents, one leaf, through a model.
^^^
```python
Bind(("fwhm", "log_e"),
     into=lambda p: p["sky"]
                     .projector.beam_alms,
     fn=beam_alms_from)
```
+++
Two scalars driving 10⁴ numbers.
:::
::::

A `fn` returning a single array is written to **every** selector in `into` —
that is what makes tying one line. Returning a tuple addresses each selector
separately.

---

## A beam from two numbers

The whole point, in one figure. The latents are a beam width and a pointing
offset; the pipeline leaf they feed is the full response matrix. Nothing in the
instrument description changed to make this possible.

```python
from rheplicant.inference import Bind, Latent, ParameterSpace

space = ParameterSpace(
    latents=[
        Latent("fwhm",   init=0.50, prior=dist.Uniform(0.15, 0.70)),
        Latent("offset", init=0.00, prior=dist.Normal(0.0, 0.4)),
    ],
    bindings=[
        Bind(("fwhm", "offset"),
             into=lambda p: p["sky"].projector.matrix,
             fn=beam_matrix),
    ],
)
model = to_numpyro_model(twin, state, space, noise_std=0.02)
```

:::{figure} _static/inference-posterior-light.svg
:figclass: only-light
:alt: NUTS posterior over two beam latents, recovering the truth
:width: 100%

Left: the NUTS posterior over the two latents, with the truth marked. Right:
the same posterior against the Fisher forecast computed at the truth — they
agree, as they should for a model this close to linear in its parameters.
:::

:::{figure} _static/inference-posterior-dark.svg
:figclass: only-dark
:alt: NUTS posterior over two beam latents, recovering the truth
:width: 100%

Left: the NUTS posterior over the two latents, with the truth marked. Right:
the same posterior against the Fisher forecast computed at the truth — they
agree, as they should for a model this close to linear in its parameters.
:::

Sample sites are named by their **latents**, so the samples come back keyed by
the coordinates the model was declared in. A `log_gain` latent is one site
named `log_gain`, even though its value reaches a leaf called `gain`, and it
stays one site if it drives five stages.

That naming reaches the error bars too:

```python
cov = parameter_covariance(fisher_information(forward, fitted, noise_std=0.02))
cov.sigma("fwhm")             # not cov.matrix[0, 0]
cov.block("fwhm", "log_gain")  # the cross-covariance, by name
```

---

## What gets checked

Every one of these failure modes produces a **finite, correctly-shaped, wrong**
inference rather than an exception. So they are all errors, and all of them are
caught at declaration or build time — `validate()` runs on shapes alone
(`jax.eval_shape`), once per build rather than per evaluation, so there is no
reason to make it skippable.

| Checked | Without the check |
|---|---|
| Latent names are unique | NumPyro sites silently overwrite each other |
| Every `Bind` names a declared latent | `KeyError` from deep inside a trace |
| **Every latent is bound by something** | It samples happily and returns the prior |
| No leaf is written twice | One binding silently wins |
| Every selector reaches a real array leaf | Confusing failure inside `tree_at` |
| Produced shape matches its target | A broadcast that is shaped right and means nothing |
| Produced dtype *kind* matches | Complex into a real leaf is a modelling error, not a cast |
| A prior's shape matches its latent's | Sampling a different-sized thing than you bind |
| Binding preserves the pytree structure | `vmap`, `jit` and Fisher flattening all break |

---

## Linear blocks

Some parameters enter the model linearly — sky `alm` coefficients, noise-wave
amplitudes, anything whose contribution is a matrix acting on it. Those are
also the *big* ones: a sky at `lmax` 191 across 32 channels is ~10⁶ real
degrees of freedom, where gradient samplers are hopeless and a conjugate
Gaussian solve is exactly right.

Declare it, and the declaration is checked before anything exploits it:

```python
space = ParameterSpace.direct(
    "sky_delta", init=jnp.zeros_like(maps),
    into=lambda p: p["sky"].sky_model.maps,
    fn=lambda delta: mean_sky + delta,      # affine, not just linear
    linear=True,
)
check_linearity(space, twin, state)          # compares against its own linearization
block = linear_operator(space, twin, state)  # A, Aᵀ and the offset — no matrix formed
solved, residual = wiener_solve(block, observed, noise_std=0.02, prior_std=1.0)
```

:::{figure} _static/inference-linear-light.svg
:figclass: only-light
:alt: A sky map recovered in closed form from a declared-linear block
:width: 100%

Left: the posterior mean against the truth, inside a 68% band from 400 exact
GCR draws. The band is the honest part — it widens exactly where the ~20°-wide
beam stops constraining the sky, and the mean's small-scale ripple lives inside
it rather than being mistaken for signal. Right: RMS error per channel, before
and after. Both the mean and the draws are conjugate-gradient solves; the same
calls take sky alms, where a gradient sampler is not an option.
:::

:::{figure} _static/inference-linear-dark.svg
:figclass: only-dark
:alt: A sky map recovered in closed form from a declared-linear block
:width: 100%

Left: the posterior mean against the truth, inside a 68% band from 400 exact
GCR draws. The band is the honest part — it widens exactly where the ~20°-wide
beam stops constraining the sky, and the mean's small-scale ripple lives inside
it rather than being mistaken for signal. Right: RMS error per channel, before
and after. Both the mean and the draws are conjugate-gradient solves; the same
calls take sky alms, where a gradient sampler is not an option.
:::

:::{admonition} Probe at extreme scales, not reasonable ones
:class: important

`check_linearity` probes at 10⁻³, 1 and 10³ times the latent's own magnitude.
The span is the point: `x + εx²` is indistinguishable from linear near the
origin and grossly nonlinear far from it, so a suite of "reasonable" probes
signs off on exactly the blocks that fail in a sampler's tails.

A block fails only if it exceeds both a relative tolerance *and* an absolute
floor set by the arithmetic's own roundoff. Without that floor the relative
measure explodes at small probes — where the response variation is vanishing
but roundoff is not — and rejects perfectly linear blocks. That false positive
would be worse than no check, because the cure a user reaches for is to switch
the check off.
:::

`linear_operator` never forms a matrix: `A` comes from `jax.linearize` and `Aᵀ`
from `jax.vjp`, so applying a 10⁶-dimensional block costs one forward
evaluation.

### Sampling it, exactly

`wiener_solve` gives the posterior **mean**. `gcr_sample` gives a posterior
**draw** — by adding two white-noise terms to that same right-hand side:

```text
(AᵀN⁻¹A + S⁻¹) x  =  AᵀN⁻¹(d − offset)  +  AᵀN⁻¹ᐟ² ω₁  +  S⁻¹ᐟ² ω₂
```

with `ω₁`, `ω₂` standard normal on the data and on the latent. The right-hand
side then has mean `AᵀN⁻¹(d − offset)` and covariance equal to the operator
itself, so `x = M⁻¹b` carries the posterior mean **and** covariance
`M⁻¹ M M⁻¹ = M⁻¹` — exactly.

```python
sample, residual = gcr_sample(block, observed, noise_std=0.02,
                              prior_std=1.0, key=jax.random.key(0))
```

Both take `prior_mean=`, which defaults to zero — wrong for most physical
quantities, since a noise-wave temperature sits near 250 K, not near zero. An
affine binding that adds the same offset gives the identical Gaussian, but
putting it on the prior says what it means.

This is a constrained realization, not a Markov chain: every call is an
independent draw, with no burn-in and no convergence to diagnose. It costs the
same single CG solve as the mean, because the fluctuation enters the
right-hand side and never the operator — which is what makes a 10⁶-dimensional
block samplable at all.

:::{tip}
**Gibbs.** A block is only linear *given* the other latents, so pass `at=` to
rebuild it wherever they currently are, and check the linearity claim once
outside the loop:

```python
check_linearity(space, twin, state, "sky_alms")        # once
for _ in range(n_sweeps):
    block = linear_operator(space, twin, state, "sky_alms",
                            at=values, check=False)     # every sweep
    values["sky_alms"], _ = gcr_sample(block, observed, noise_std=sigma,
                                       prior_std=s, key=next(keys))
    values = update_the_nonlinear_ones(values)          # NUTS, MH, optimize...
```

Omitting `at=` is silent, not loud: the block keeps describing the model at its
declared starting point, which is right for exactly one sweep.
:::

---

## One space, every engine

::::{grid} 1 2 2 2
:gutter: 2

:::{grid-item-card} Optimize
```python
forward, start = space.forward_fn(twin, state)
fitted, losses = AdamCalibrator(
    n_steps=2000).fit(forward, start, data)
```
+++
Needed no changes: a dict is a pytree.
:::

:::{grid-item-card} Sample
```python
model = to_numpyro_model(
    twin, state, space, noise_std=0.02)
mcmc.run(key, observed=data)
```
+++
Sites named by latent.
:::

:::{grid-item-card} Forecast
```python
cov = parameter_covariance(
    fisher_information(forward, start,
                       noise_std=0.02))
cov.sigma("fwhm")
```
+++
Rows carry their names.
:::

:::{grid-item-card} Solve or sample exactly
```python
block = linear_operator(space, twin, state)
mean, _ = wiener_solve(
    block, data, noise_std=0.02, prior_std=1.0)
draw, _ = gcr_sample(
    block, data, noise_std=0.02, prior_std=1.0,
    key=jax.random.key(0))
```
+++
For blocks too big for a gradient sampler.
:::
::::

:::{note}
[`build_forward_fn`](api.md) is not superseded by this and stays. The two
answer different questions: `build_forward_fn` partitions a whole subtree into
trainables, which is what a neural surrogate's MLP weights want;
`forward_fn` carries parameters that were chosen, transformed, or shared, which
is what physical fitting wants.
:::

---

## When the blocks are not enough

`ParameterSpace.raw` takes a bind function outright:

```python
space = ParameterSpace.raw(
    latents=[Latent("x", init=x0, prior=dist.Normal(0.0, 1.0))],
    bind=lambda pipeline, values: ...,   # anything, as long as the treedef survives
)
```

The structural checks still apply — a bind function that quietly changes the
pipeline's pytree structure is still caught — but the per-selector checks
cannot run, because there are no selectors to inspect. Reach for it when the
declarative form genuinely cannot express what you need, not to save a line.

---

## Run it

[`examples/inferring_anything.py`](https://github.com/RHINO-Experiment/rheplicant/blob/main/examples/inferring_anything.py)
does all three on one twin, from truth to recovery: a beam derived from two
scalars, a gain tied across two stages in log space, and a sky map declared
linear and solved by CG. The instrument description is written once at the top
and never edited.

```bash
uv run python examples/inferring_anything.py
```

The figures on this page come from
[`docs/_generate_inference_figures.py`](https://github.com/RHINO-Experiment/rheplicant/blob/main/docs/_generate_inference_figures.py),
which runs the same code rather than illustrating it.
