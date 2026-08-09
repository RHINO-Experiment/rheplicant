# Tutorial: an exact posterior for a big linear block

Some parameters enter the model **linearly** — sky pixels, `alm` coefficients,
noise-wave amplitudes — and those are also the *big* ones. This tutorial infers
256 sky pixels with no chain, no burn-in and nothing to diagnose, because for a
linear-Gaussian block the posterior is available in closed form and every draw
is independent.

Its companion, [Tutorial: NUTS](tutorial-nuts.md), takes the same instrument and
asks the opposite question.

```bash
.venv/bin/python examples/tutorial_gcr.py
```

Everything below is that script's actual output.

---

## Step 1 — the world

A ring of 64 sky pixels, scanned by a Gaussian beam through a known gain, in
4 frequency channels.

```python
def twin(maps, fwhm=FWHM):
    return Pipeline(
        SkySourceOperator(
            sky_model=MapSky(maps=maps),
            projector=MatrixProjector(beam_matrix(fwhm, OFFSET)),
        ),
        GainOperator(gain=jnp.asarray(GAIN)),
        names=("sky", "gain"),
    )

noise = RadiometerNoise(CHANNEL_WIDTH, INTEGRATION_TIME)
clean = twin(true_maps)(state).data
observed = clean * (1.0 + noise.fractional * jax.random.normal(key, clean.shape))
```

```text
STEP 1  the world
  data      (256, 4) = 1024 samples
  unknowns  256 sky pixels (64 x 4 channels)
  beam      FWHM 8.6 deg = 1.5 pixels
  noise     fractional 0.0100, so sigma spans 2.48 .. 4.14 K
```

The noise is **multiplicative** — `d = prediction × (1 + w)`. That is what makes
the covariance something to be *found* rather than handed in, and it is the
whole reason step 4 exists.

## Step 2 — declare the block linear, and have the claim checked

```python
space = ParameterSpace.direct(
    "sky_maps",
    init=jnp.full_like(true_maps, MEAN_SKY),
    into=lambda p: p["sky"].sky_model.maps,
    linear=True,
)
errors = check_linearity(space, start, state)
block = linear_operator(space, start, state)
```

```text
STEP 2  the linearity claim, checked before anything exploits it
  worst relative departure from affine: 0.00e+00
```

`linear=True` is a **claim**, and everything downstream exploits it. A false one
does not raise — it returns a confident, correctly-shaped, wrong posterior. So
check it once, before the loop, not never.

## Step 3 — read the conditioning before solving

```python
kappa = condition_estimate(block, noise_std=..., prior_std=PRIOR_STD)
CG_TOL = TARGET_ERROR / float(kappa)
```

```text
STEP 3  conditioning, which decides what tolerance means
  kappa = 1.89e+01
  a residual of tol bounds the ERROR by kappa*tol, so tol=1e-6 certifies
  only 1.9e-05 -- pick tol from kappa, not from habit
  -> for a relative error of 1e-04, tol = target/kappa = 5.3e-06
```

CG reports a **residual**; what you need bounded is the **error**. They differ by
κ. Choosing `tol` without knowing κ is choosing an accuracy you have not
computed — see [Conditioning](inference-linear.md#conditioning-why-a-residual-is-not-an-accuracy).

## Step 4 — the covariance is not given

σ tracks the prediction, so the weights depend on the solution and the solution
depends on the weights. `iterative_gls` resolves that by fixed point.

```python
found = iterative_gls(block, observed, noise=noise, prior_std=PRIOR_STD,
                      prior_mean=MEAN_SKY, tol=CG_TOL, maxiter=2000)
```

```text
STEP 4  iterative GLS: sigma tracks the prediction, so solve for both
  5 reweights, last relative step 2.01e-10, converged=True
  recovered sigma vs the truth's own: max relative error 1.79%
```

:::{important}
**Check `converged`.** A covariance that is not a fixed point is still a number,
and a draw conditioned on it is still a draw.
:::

## Step 5 — the mean, then exact draws

`gcr_sample` is unchanged by any of the above: it takes the σ that came back.

```python
mean, _ = wiener_solve(block, observed, noise_std=found.noise_std, ...)
draws = jax.vmap(lambda k: gcr_sample(
    block, observed, noise_std=found.noise_std, key=k, ...)[0]
)(jax.random.split(key, 200))
```

```text
STEP 5  the posterior
  Wiener mean: RMS error 4.089 K against a sky spanning 223.7 .. 377.8 K
  200 exact GCR draws, each one CG solve -- no burn-in, nothing to
  diagnose, every draw independent
```

## Step 6 — is the answer honest?

```text
STEP 6  reading it
  posterior sigma  3.17 .. 5.54 K (4.1% of the 100 K prior on average)
  truth within 1 sigma of the mean: 68% of pixels (68% is the target)
  RMS error 4.09 K against a mean posterior sigma of 4.09 K
```

The RMS error **equals** the posterior σ, and coverage is **exactly 68 %**. That
is what calibrated means, and it is the thing a point estimate can never tell
you.

## Step 7 — now break it on purpose

Widen the beam to 3.6 pixels — wider than the structure it is meant to measure —
and change nothing else.

```text
STEP 7  the same solve with a beam 2.3x wider than the structure
                            FWHM 9 deg   FWHM 20 deg
  beam / pixel                     1.5           3.6
  kappa                           18.9         257.4
  RMS error [K]                   4.09         19.92
  posterior sigma [K]             4.09         62.55
  % of prior width                 4.1          62.5
  68% coverage                      68           100
```

:::{tip}
**Nothing failed.** The wide beam cannot resolve the ring, so the prior holds
those directions up. κ says so beforehand (13× larger), the posterior width says
so afterwards (63 % of the prior, against 4 %), and coverage climbing to 100 % is
what an answer looks like when it is mostly prior.

An estimator without error bars would have returned a map that looks much the
same and told you none of this.
:::

---

## Where each step's rule comes from

| Step | Rule | Reference |
|---|---|---|
| 2 | a linearity claim gets checked before it is exploited | [Linear blocks](inference-linear.md#linear-blocks) |
| 3 | `tol` is a residual; κ·`tol` is the error | [Conditioning](inference-linear.md#conditioning-why-a-residual-is-not-an-accuracy) |
| 4 | radiometer noise ⇒ the covariance must be found | [When the covariance is not given](inference-linear.md#when-the-covariance-is-not-given) |
| 5 | one CG solve per independent draw | [Sampling it, exactly](inference-linear.md#sampling-it-exactly) |
| 7 | the prior holds up what the data cannot see | [The noise model](inference-linear.md#the-noise-model) |

## Next

- [Tutorial: NUTS](tutorial-nuts.md) — the same instrument, the nonlinear
  parameters, and what a broken sampler looks like from the outside.
- [`examples/gls_gcr.py`](https://github.com/RHINO-Experiment/rheplicant/blob/main/examples/gls_gcr.py)
  — the same machinery on RHINO's switched noise-wave calibration, where the
  point estimate is weight-independent but the error bars are not.
