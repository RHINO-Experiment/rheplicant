# Tutorial: a gradient posterior, and how to tell it is wrong

Most parameters are not linear in the model. A beam width sits inside a Gaussian
and a pointing offset inside a wrapped angular difference. No conjugate solve
exists for them, and gradient MCMC is the tool. This tutorial infers three such
parameters on the same ring toy that [Tutorial: GCR](tutorial-gcr.md) used for
its 256-pixel sky.

The first run fails. It raises nothing and returns a posterior that looks
reasonable, and only the diagnostics show that it cannot be used. Diagnosing it
is most of the tutorial.

```bash
.venv/bin/python examples/tutorial_nuts.py
```

The `text` blocks below are that script's output.

---

## Step 1: why this is not a conjugate solve

```text
STEP 1  three parameters that are NOT linear in the model
  data   (256, 4) = 1024 samples
  truth  fwhm=0.15, offset=0.12, gain=1.1
```

`fwhm` sits inside `exp(-x²/fwhm²)` and `offset` inside a wrapped difference.
Neither is affine, so `check_linearity` would refuse the claim. Three
unknowns against 1024 samples is a size that suits a gradient sampler.

## Step 2: priors, and two latents into one leaf

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

Every latent needs a prior. A prior-free latent is a free parameter, fine for
an optimizer and meaningless in a posterior, and the bridge refuses it rather
than inventing a flat one.

`log_gain` rather than `gain`: positive by construction, and unbounded is what
NUTS explores well. The site is named `log_gain`, so samples come back in
the coordinates the model was declared in.

## Step 3: the model

```python
model = to_numpyro_model(twin, state, space, noise_std=noise)
```

The noise model goes in whole. With `RadiometerNoise` the observation scale is a
function of the sampled parameters, so its log-determinant is in the potential
automatically.

## Step 4: the default run and its diagnostics

```python
mcmc = numpyro.infer.MCMC(
    numpyro.infer.NUTS(model),
    num_warmup=1000, num_samples=2000, num_chains=4,
    chain_method="vectorized",
)
mcmc.run(key, observed=observed, extra_fields=("diverging",))
```

```text
  as written  (2.9 s)
    site             mean       std    n_eff    r_hat
    fwhm          0.48721   0.19497        2   62.660
    offset       -0.20938   1.09869        2  845.775
    log_gain      0.10970   0.00841        2   28.454
    divergences 0 / 8000      -> DO NOT USE THIS POSTERIOR
```

:::{danger}
**Nothing raised and nothing was NaN.** The means are finite numbers, and the
posterior is broken.
:::

| Diagnostic | What it is | Threshold |
|---|---|---|
| `r_hat` | between-chain vs within-chain variance | > 1.01 ⇒ the chains have not agreed |
| `n_eff` | independent-equivalent draws | here **2** out of 8000 |
| `diverging` | the integrator could not follow the geometry | any at all ⇒ investigate |

Read the divergence count on every run, from
`mcmc.get_extra_fields()["diverging"]`. NUTS records it by default, so the
`extra_fields=("diverging",)` above only makes that explicit. A divergence is
the sampler reporting that it failed, and a run whose count is not read can
return a biased posterior that looks fine.

numpyro's `summary` reports `r_hat` and `n_eff` per element. The latents here
are scalars, so `float()` on them works. For a non-scalar latent read the
largest `r_hat` and the smallest `n_eff`; `float()` on the array raises. The
split `r_hat` needs at least four draws per chain
(`num_samples // thinning >= 4`), and numpyro asserts it with no message.

## Step 5: diagnosis with two hypotheses

**Hypothesis 1: multimodal, with chains in different modes.** Scan the
log-posterior along `offset`:

```text
    local maxima within 2000 nats of the peak: 1 (at offset=+0.120)
    -> UNIMODAL. Hypothesis 1 is wrong.
```

**Hypothesis 2: the posterior is far narrower than its prior.** Compare widths:

```text
    prior sigma on offset   0.4000
    posterior sigma will be 0.0008  (500x narrower)
    log-posterior at the declared start: -63481 nats below the peak
```

Hypothesis 2 holds. NUTS's default `init_to_uniform` draws in the unconstrained
space, lands far from the peak, and warmup then adapts a step size for wherever
it landed.

## Step 6: the fix, and two attempts that do not help

```python
from rheplicant.inference import init_to_declared

kernel = numpyro.infer.NUTS(model, init_strategy=init_to_declared(space))
```

```text
  init_to_declared(space)  (1.6 s)
    site             mean       std    n_eff    r_hat
    fwhm          0.14935   0.00629     1259    1.003
    offset        0.12197   0.00081     5667    1.000
    log_gain      0.09514   0.00032     9726    1.000
    divergences 0 / 8000      -> HEALTHY
```

`r_hat` 846 → 1.003; `n_eff` 2 → 1259; and it is faster, because a
sampler that is not lost takes fewer leapfrog steps.

:::{note}
`ParameterSpace` already declares where to start: `Latent(..., init=...)`, which
the calibrators and `check_linearity` both use. NUTS does not read it unless you
say so, and `init_to_declared(space)` says so.

The declared init here is not a good one: it is the deliberately mis-set
starting point, 63 481 nats below the peak. It only has to be somewhere a
gradient can be followed.
:::

Measured on this same problem, tightening the priors and tripling the warmup
leave the run unconverged:

| Attempt | `r_hat` | `n_eff` |
|---|---|---|
| as written | 846 | 2 |
| tighten the priors | 1123 | 2 |
| triple the warmup | 1124 | 2 |
| **`init_to_declared(space)`** | **1.003** | **1259** |

Diagnostics tell you the posterior is wrong. They do not tell you why, and
guessing costs more than the two scans in step 5.

## Step 6b: the answer

```text
  the answer, now that it is one:
    fwhm         0.14935 +/- 0.00629   truth  0.15000   ( -0.1 sigma)
    offset       0.12197 +/- 0.00081   truth  0.12000   ( +2.4 sigma)
    log_gain     0.09514 +/- 0.00032   truth  0.09531   ( -0.5 sigma)
    posterior correlations:
      fwhm      x offset    -0.02
      fwhm      x log_gain  -0.08
      offset    x log_gain  +0.01
```

The parameters are near-orthogonal, which is a property of this design.
`beam_matrix` normalizes each row to sum to one, so widening the beam smooths
the sky without changing the total throughput, and `fwhm` cannot trade against
the gain.

A model where it could would show a correlation near 1 here, the posterior
would be a ridge, and `n_eff` would fall while `r_hat` still looked fine, so
the correlations are worth printing even when they are small.

## Step 7: an independent check

```text
STEP 7  Fisher forecast at the truth, as an independent check
  site          NUTS std  Fisher std   ratio
  fwhm           0.00629     0.00660    0.95
  offset         0.00081     0.00081    1.00
  log_gain       0.00032     0.00031    1.01
```

They agree, as they should for a model this close to linear in its parameters.
Where they disagree, the Fisher matrix is the one that is wrong: it is a local
quadratic, and the posterior is the actual shape.

## Step 8: posterior predictive check

```text
STEP 8  posterior predictive
  model spread  0.169 K   noise sigma 3.300 K
  pull (residual / total sigma): mean +0.003, std 0.999
```

A pull with mean 0 and standard deviation 1 is a model that explains its data.
Well under 1 means the error bars are too big; over 1 means the model is missing
something the data can see.

---

## Which engine

The structure of the problem determines the engine.

| Situation | Engine | Why |
|---|---|---|
| linear in the parameter, Gaussian noise | `gcr_sample` | exact, independent draws, one CG solve each; scales to 10⁶ dof |
| nonlinear, few parameters | NUTS | no conjugate structure to exploit, and none needed |
| both at once | Gibbs | draw the linear block exactly with `at=` pinning the nonlinear ones, move those with NUTS, repeat |
| no tractable likelihood at all | [NPE](inference-plans.md#inference-without-a-likelihood) | amortized, and validated against the two above |

This run: 3 parameters, 1.6 s, 1259 effective draws. The same sampler on the
256-pixel sky of [Tutorial: GCR](tutorial-gcr.md) would explore a space 85×
bigger with no conjugate structure to exploit, which is possible and
unnecessary when an exact draw costs one linear solve.

## A checklist for any NUTS run

1. Read the divergence count, always.
2. `num_chains >= 4`: `r_hat` needs more than one chain to mean anything.
3. `init_strategy=init_to_declared(space)`.
4. Read `r_hat`, `n_eff` and divergences before the means.
5. Cross-check the widths against `fisher_information` where the model is near-linear.
6. Check the posterior-predictive pull is ~N(0, 1).

The same run can be declared in a document as `kind: nuts`, which starts from
`init_to_declared` by default and returns `r_hat`, `n_eff` and the divergence
count on its product. See
[the two that sample a posterior](config-inference.md#the-two-that-sample-a-posterior).
