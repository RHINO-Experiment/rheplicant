# Tutorial: a gradient posterior, and how to tell it is wrong

Most parameters are not linear in the model. A beam width sits inside a
Gaussian, a pointing offset inside a wrapped angular difference — no conjugate
solve exists, and gradient MCMC is the tool. This tutorial infers three such
parameters on the same ring toy that
[Tutorial: GCR](tutorial-gcr.md) used for its 256-pixel sky.

**It fails first.** That is deliberate: the run below produces a posterior that
looks perfectly reasonable and is worthless, and the only reason to know is that
the diagnostics were read before the answer. Diagnosing it is most of the
tutorial.

```bash
uv run --frozen python examples/tutorial_nuts.py
```

Everything below is that script's actual output.

---

## Step 1 — why this is not a conjugate solve

```text
STEP 1  three parameters that are NOT linear in the model
  data   (256, 4) = 1024 samples
  truth  fwhm=0.15, offset=0.12, gain=1.1
```

`fwhm` sits inside `exp(-x²/fwhm²)` and `offset` inside a wrapped difference.
Neither is affine, so `check_linearity` would **refuse** the claim. Three
unknowns against 1024 samples is exactly a gradient sampler's size.

## Step 2 — priors, and two latents into one leaf

```python
space = ParameterSpace(
    latents=[
        Latent("fwhm",     init=0.30, prior=dist.Uniform(0.05, 0.60)),
        Latent("offset",   init=0.00, prior=dist.Normal(0.0, 0.40)),
        Latent("log_gain", init=0.00, prior=dist.Normal(0.0, 0.20)),
    ],
    bindings=[
        Bind(("fwhm", "offset"),
             into=lambda p: p["sky"].projector.matrix, fn=beam_matrix),
        Bind("log_gain", into=lambda p: p["gain"].gain, fn=jnp.exp),
    ],
)
```

Every latent needs a prior — a prior-free latent is a free parameter, fine for
an optimizer and meaningless in a posterior, and the bridge refuses it rather
than inventing a flat one.

`log_gain` rather than `gain`: positive by construction, and unbounded is what
NUTS explores well. The **site** is named `log_gain`, so samples come back in
the coordinates the model was declared in.

## Step 3 — the model

```python
model = to_numpyro_model(twin, state, space, noise_std=noise)
```

The noise model goes in whole. With `RadiometerNoise` the observation scale is a
function of the sampled parameters, so its log-determinant is in the potential
automatically.

## Step 4 — the obvious run, and its diagnostics

```python
mcmc = numpyro.infer.MCMC(
    numpyro.infer.NUTS(model),
    num_warmup=1000, num_samples=2000, num_chains=4,
    chain_method="vectorized",
)
mcmc.run(key, observed=observed, extra_fields=("diverging",))
```

```text
  as written  (2.5 s)
    site             mean       std    n_eff    r_hat
    fwhm          0.48712   0.19512        2   61.182
    offset       -0.20934   1.09871        2  840.224
    log_gain      0.10970   0.00841        2   28.686
    divergences 0 / 8000      -> DO NOT USE THIS POSTERIOR
```

:::{danger}
**Nothing raised. Nothing was NaN. The means look like numbers.** This is what a
broken posterior looks like from the outside.
:::

| Diagnostic | What it is | Threshold |
|---|---|---|
| `r_hat` | between-chain vs within-chain variance | > 1.01 ⇒ the chains have not agreed |
| `n_eff` | independent-equivalent draws | here **2** out of 8000 |
| `diverging` | the integrator could not follow the geometry | any at all ⇒ investigate |

`extra_fields=("diverging",)` is not optional book-keeping. A divergence is the
sampler telling you it failed, and discarding that quietly is how a biased
posterior comes to look fine.

## Step 5 — diagnose it: two hypotheses, both testable

**Hypothesis 1 — multimodal, with chains in different modes.** Scan the
log-posterior along `offset`:

```text
    local maxima within 2000 nats of the peak: 1 (at offset=+0.120)
    -> UNIMODAL. Hypothesis 1 is wrong.
```

**Hypothesis 2 — the posterior is a needle inside its prior.** Compare widths:

```text
    prior sigma on offset   0.4000
    posterior sigma will be 0.0008  (500x narrower)
    log-posterior at the declared start: -63481 nats below the peak
```

That is the answer. NUTS's default `init_to_uniform` draws in the
**unconstrained** space, lands in the haystack, and warmup then adapts a step
size for wherever it landed.

## Step 6 — the fix, and two fixes that are not the fix

```python
from rheplicant.inference import init_to_declared

kernel = numpyro.infer.NUTS(model, init_strategy=init_to_declared(space))
```

```text
  init_to_declared(space)  (1.2 s)
    site             mean       std    n_eff    r_hat
    fwhm          0.14919   0.00637     1327    1.002
    offset        0.12199   0.00083     5451    1.000
    log_gain      0.09514   0.00031     9411    1.000
    divergences 0 / 8000      -> HEALTHY
```

`r_hat` 840 → **1.002**; `n_eff` 2 → **1327**; and it is *faster*, because a
sampler that is not lost takes fewer leapfrog steps.

:::{note}
`ParameterSpace` already declares where to start — `Latent(..., init=...)`, which
the calibrators and `check_linearity` both use. NUTS does not read it unless you
say so. `init_to_declared(space)` is that one line.

And the declared init here is not even good: it is the deliberately mis-set
starting point, 63 481 nats below the peak. It only has to be somewhere a
gradient can be followed.
:::

Measured on this same problem, neither of the two things one reaches for first
changes anything at all:

| Attempt | `r_hat` | `n_eff` |
|---|---|---|
| as written | 840 | 2 |
| tighten the priors | 1123 | 2 |
| triple the warmup | 1124 | 2 |
| **`init_to_declared(space)`** | **1.002** | **1327** |

Diagnostics tell you the posterior is wrong. They do not tell you *why*, and
guessing costs more than the two scans in step 5.

## Step 6b — now the answer

```text
  the answer, now that it is one:
    fwhm         0.14919 +/- 0.00637   truth  0.15000   ( -0.1 sigma)
    offset       0.12199 +/- 0.00083   truth  0.12000   ( +2.4 sigma)
    log_gain     0.09514 +/- 0.00031   truth  0.09531   ( -0.5 sigma)
    posterior correlations:
      fwhm      x offset    -0.02
      fwhm      x log_gain  -0.07
      offset    x log_gain  -0.01
```

Near-orthogonal — and that is a property of this **design**, not a given.
`beam_matrix` normalizes each row to sum to one, so widening the beam smooths
the sky without changing the total throughput, and `fwhm` cannot trade against
the gain.

A model where it *could* would show a correlation near 1 here, the posterior
would be a ridge, and `n_eff` would fall while `r_hat` still looked fine — which
is why the correlations are worth printing even when they turn out to be dull.

## Step 7 — an independent check

```text
STEP 7  Fisher forecast at the truth, as an independent check
  site          NUTS std  Fisher std   ratio
  fwhm           0.00637     0.00660    0.97
  offset         0.00083     0.00081    1.01
  log_gain       0.00031     0.00031    0.99
```

They agree, as they should for a model this close to linear in its parameters.
Where they *disagree*, the Fisher matrix is the one that is wrong: it is a local
quadratic, and the posterior is the actual shape.

## Step 8 — does the model explain its data?

```text
STEP 8  posterior predictive
  model spread  0.168 K   noise sigma 3.300 K
  pull (residual / total sigma): mean -0.001, std 0.999
```

A pull with mean 0 and standard deviation 1 is a model that explains its data.
Well under 1 means the error bars are too big; over 1 means the model is missing
something the data can see.

---

## Which engine

The choice is not taste, it is the shape of the problem.

| Situation | Engine | Why |
|---|---|---|
| linear in the parameter, Gaussian noise | `gcr_sample` | exact, independent draws, one CG solve each; scales to 10⁶ dof |
| nonlinear, few parameters | NUTS | no conjugate structure to exploit, and none needed |
| both at once | Gibbs | draw the linear block exactly with `at=` pinning the nonlinear ones, move those with NUTS, repeat |
| no tractable likelihood at all | [NPE](inference.md#inference-without-a-likelihood) | amortized, and validated against the two above |

This run: 3 parameters, 1.2 s, 1327 effective draws. The same sampler on the
256-pixel sky of [Tutorial: GCR](tutorial-gcr.md) would explore a space 85×
bigger with no conjugate structure to exploit — possible, and pointless when an
exact draw costs one linear solve.

## A checklist for any NUTS run

1. `extra_fields=("diverging",)`, always.
2. `num_chains >= 4` — `r_hat` needs more than one chain to mean anything.
3. `init_strategy=init_to_declared(space)`.
4. Read `r_hat`, `n_eff` and divergences **before** the means.
5. Cross-check the widths against `fisher_information` where the model is near-linear.
6. Check the posterior-predictive pull is ~N(0, 1).
