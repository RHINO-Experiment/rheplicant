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
| **Every latent reaches the model** — named in a binding, or (for a raw bind) shown to move it | It samples happily and returns the prior |
| No leaf is written twice | `eqx.tree_at` lets the last write win, silently |
| A raw bind is not combined with declarative bindings | `bind()` would apply only the raw one |
| Binding preserves every leaf's shape and dtype kind | A treedef encodes neither; the value is broadcast |
| Every selector reaches a real array leaf | Confusing failure inside `tree_at` |
| Produced shape matches its target | A broadcast that is shaped right and means nothing |
| Produced dtype *kind* matches | Complex into a real leaf is a modelling error, not a cast |
| A prior's shape matches its latent's | Sampling a different-sized thing than you bind |
| Binding preserves the pytree structure | `vmap`, `jit` and Fisher flattening all break |

---

## The noise model

Every inference route needs one number per sample: how noisy is it? The
likelihood needs it, a Wiener solve and a GCR draw need it as a weight, the
Fisher matrix needs it, a NumPyro observation site needs it as a scale. Passing
it to each of them as a bare `noise_std` quietly assumes the answer is *given*
and *constant*. For a radiometer it is neither:

$$\sigma(d) = \frac{|d|}{\sqrt{\Delta\nu\,\tau}}$$

σ is a function of the very thing being inferred. So there is one object that
answers the question, and every route takes it:

:::{list-table}
:header-rows: 1
:widths: 26 40 34

* - Model
  - σ
  - `depends_on_prediction`
* - `HomoscedasticNoise(sigma)`
  - a constant — what a bare `noise_std` always meant
  - `False`
* - `RadiometerNoise(Δν, τ)`
  - `|prediction| / √(Δν·τ)`
  - `True`
* - `FlaggedNoise(base, flags)`
  - the wrapped model, `∞` where flagged
  - inherited
:::

Anywhere the package writes `noise_std=`, a noise model is accepted in its
place — the argument is polymorphic, so nothing downstream changed signature:

```python
from rheplicant.inference import RadiometerNoise, FlaggedNoise

noise = FlaggedNoise(RadiometerNoise(channel_width=61e3, integration_time=1.0),
                     flags=state.aux["flags"])

fisher_information(forward, params, noise_std=noise)
```

`FlaggedNoise` is how RFI flags reach the covariance: by **wrapping a noise
model**, not by threading a `flags=` keyword through five separate functions. An
infinite σ is a self-describing encoding of "this sample was not observed", and
every consumer turns it into a clean zero rather than a NaN.

`depends_on_prediction` is not a hint. It is the claim a solver branches on:
`False` means one solve, `True` means the covariance has to be found before it
can be used.

### The term that is not a constant

The Gaussian log-density is

$$\log p = -\tfrac{1}{2}\sum_i\left[\frac{r_i^2}{\sigma_i(\theta)^2}
  + \log 2\pi\sigma_i(\theta)^2\right]$$

When σ is constant the second term is an additive constant and dropping it
changes nothing. When σ depends on the prediction, dropping it — which is
exactly what generalized least squares does — gives a **different estimator**,
one with no penalty for shrinking the prediction to make the variance small.

For the multiplicative model both are solvable by hand. With $d_i = \theta(1+w_i)$
and $w\sim\mathcal N(0,f^2)$:

| Objective | Stationary point | Expectation |
|---|---|---|
| GLS (log-det dropped) | $\hat\theta = \sum d^2 / \sum d$ | $\theta_{\rm true}(1+f^2)$ — biased high |
| Full Gaussian | $nf^2\theta^2 + \theta\sum d - \sum d^2 = 0$ | $\theta_{\rm true}$ — unbiased |

So the term GLS discards as a normalization is precisely the one that removes
the bias. `NoiseModelLikelihood` keeps it by default; `include_logdet=False` is
the explicit, documented GLS variant rather than an oversight.

:::{warning}
The same thing happens to the Fisher matrix, and more quietly. When the
covariance carries parameter dependence, $J^\top N^{-1} J$ **is not** the Fisher
information — there is a second term,

$$F = J^\top\Sigma^{-1}J + \tfrac{1}{2}
  \operatorname{tr}\!\left(\Sigma^{-1}\partial\Sigma\,\Sigma^{-1}\partial\Sigma\right),$$

which for a diagonal covariance is $2\,(\partial\log\sigma)^\top(\partial\log\sigma)$.
`fisher_information` includes it whenever the noise model reports
`depends_on_prediction`. Under `RadiometerNoise` it is a clean factor,
$F = (1+2f^2)\,J^\top N^{-1} J$ — so reporting only the first term forecasts
error bars too wide by $\sqrt{1+2f^2}$: a plausible number, and the wrong one.
:::

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

`check_linearity` probes at 10⁻³, 1 and 10³ times the latent's own magnitude,
taken from `max|init|`. The span is the point: a knee, a saturation, or a small
quadratic is indistinguishable from linear below some scale and grossly
nonlinear above it, so a suite of "reasonable" probes signs off on exactly the
blocks that fail in a sampler's tails.

One sharp edge, since the examples above walk straight into it: an **all-zero
`init` has no scale to take**, so the probes fall back to absolute. If the
latent lives at 10⁶ — sky alms in kelvin — give a representative `init` or pass
`scales=` explicitly, or the sweep never reaches the regime a sampler will.

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

### When the covariance is not given

Both solvers above take `noise_std` and neither cares where it came from. Under
`HomoscedasticNoise` it comes from you and there is nothing more to say. Under
the default `RadiometerNoise` there is: σ tracks the prediction, so the weights
depend on the solution and the solution depends on the weights. Neither is
available first.

`iterative_gls` supplies the missing half by fixed point — solve at the current
σ, recompute σ at the new prediction, repeat — and **nothing about
`gcr_sample` changes**:

```python
found = iterative_gls(block, observed, noise=RadiometerNoise(dnu, tau),
                      prior_std=PRIOR)

draw, _ = gcr_sample(block, observed, noise_std=found.noise_std,
                     prior_std=PRIOR, key=key)
```

It is the same iteratively-reweighted GLS as hydra-tod's
`hydra_tod.linear_sampler.iterative_gls` — a test checks the two agree — but
**matrix-free**: hydra-tod forms a dense `U` and `N_inv`, while here the
algorithm runs on the block's JVP and VJP, which is what makes 10⁶ degrees of
freedom possible at all.

Check `found.converged`. A covariance that is not a fixed point is still a
number, and a draw conditioned on it is still a draw.

:::{tip}
**The mean can be weight-independent while the width is not.** In
[`examples/gls_gcr.py`](https://github.com/RHINO-Experiment/rheplicant/blob/main/examples/gls_gcr.py)
three switched loads meet three per-channel noise-wave unknowns, so the reduced
system is **square** — one solution, and the weights cancel out of it. The
point estimate is unmoved by reweighting, exactly, not approximately.

The posterior covariance $(A^\top\Sigma^{-1}A + S^{-1})^{-1}$ depends on Σ all
the same, and a GCR draw is precisely a draw of that width. In that example a
frozen σ reports error bars wrong by −8 % to +8 %, in both directions, on a
point estimate that was already right. "The fit came out the same" is not
evidence the covariance did not matter.

Where the system *is* over-determined across genuinely different noise levels,
the estimate moves too: on a prediction spanning a decade, freezing σ costs a
factor of ≈2.3 in recovered RMS error.
:::

:::{warning}
`reweight_tol` cannot have a fixed default, and neither should yours. Two
independent floors bound how small a step is measurable: the arithmetic's
epsilon (`1.2e-7` in float32 — so a plausible-looking `1e-8` is exactly this
trap), and **the inner CG tolerance**, since consecutive solves differ by their
own residual whatever the outer iteration does. The latter binds in float64,
where a tight `tol=1e-10` sits five orders of magnitude above `eps`. The default
is `max(8·eps, tol)`. Ask for less than either and the run does not fail
quietly — it spends `max_reweights` steps and reports `converged=False` for a
fixed point it had reached.
:::

**What this estimator is.** Freezing σ inside each solve is what makes every
step a linear-Gaussian problem, and it is also what makes the converged answer
*generalized least squares* rather than the maximum of the full Gaussian
likelihood: the log-determinant's dependence on the solution is held fixed
rather than differentiated ([above](#the-term-that-is-not-a-constant)). GLS is
the right thing to condition a constrained realization on, because a GCR draw
*is* a draw from a linear-Gaussian posterior at a given covariance. If you want
the full likelihood's mode or posterior, that is a gradient sampler's job.

---

## Conditioning: why a residual is not an accuracy

CG only ever reports on itself: `‖M x̂ - b‖`, the **residual** — how well `x̂`
satisfies the equation it was asked to solve. What a caller actually needs
bounded is `‖x̂ - x*‖`, the **error** — how close `x̂` is to the truth. The two
are related through the condition number of the normal operator
`M = AᵀN⁻¹A + S⁻¹`:

```text
‖x̂ - x*‖ / ‖x*‖  ≤  κ(M) · ‖M x̂ - b‖ / ‖b‖
```

For a well-conditioned block (κ ≈ 1) the two coincide and a small residual is
a small error — no further thought required. But κ is large **by design**
exactly when these solvers matter most: whenever the data does not fully
identify some direction in the block — one calibration load against three
per-channel unknowns, a flagged channel, a short integration — the prior is
the only thing holding that direction down, so `λ_min(M)` is exactly
`1 / prior_std²` and κ runs past `1e6`. CG happily converges its residual on
the well-constrained directions, which dominate the aggregate norm, while the
prior-dominated directions sit at their starting value: a residual that
*looks* converged is not, and a draw built from it comes back with far too
little scatter. That is exactly what used to happen before this fix:
`gcr_sample` on a badly-conditioned block reported a posterior σ three
orders of magnitude too narrow, while its residual sat comfortably under the
old, residual-only guard.

`condition_estimate(block, noise_std=..., prior_std=...)` reports κ,
matrix-free — the same two power iterations `wiener_solve`/`gcr_sample`
already run internally to guard themselves, exposed so a caller can choose
`tol` instead of guessing it:

```python
kappa = condition_estimate(block, noise_std=0.5, prior_std=100.0)
target_error = 1e-3
solved, residual = wiener_solve(block, observed, noise_std=0.5, prior_std=100.0,
                                tol=target_error / kappa, maxiter=4000)
```

`require_convergence` (default `1e-3`) already bounds `κ · relative_residual`
rather than the residual alone, so a block the data does not identify raises
instead of returning a silently wrong answer, and names the remedy in the
error. [`examples/noise_wave_gcr.py`](https://github.com/RHINO-Experiment/rheplicant/blob/main/examples/noise_wave_gcr.py)
shows both ends: its three-load block (κ ≈ 2.69e1) passes at the library's
default `tol=1e-6`; its `--one-source` variant (κ ≈ 4.35e6 — one load
against three per-channel unknowns, so two of every three directions are
prior-dominated) raises at that same default, and needs `tol=1e-10,
maxiter=4000` to converge — which then reports per-channel σ ≈ 71–106 K
against the 100 K prior, i.e. the prior width recovered *honestly* where the
data says nothing, rather than the ≈0.03 K a residual-only guard used to let
through.

:::{admonition} The guard is not free
:class: note

Estimating κ costs `2 · POWER_ITERATIONS` operator applications — two power
iterations per end of the spectrum — on top of the CG solve itself, which
roughly **doubles** a well-conditioned solve where CG converges in a handful
of iterations. In a Gibbs loop, where the conditioning barely moves sweep to
sweep, estimate κ once outside the loop and pass `require_convergence=None`
inside, the same bargain `linear_operator`'s `check` argument offers for
`check_linearity`:

```python
kappa = condition_estimate(block, noise_std=sigma, prior_std=s)  # once
tol = target_error / kappa
for _ in range(n_sweeps):
    block = linear_operator(space, twin, state, "sky_alms",
                            at=values, check=False)
    values["sky_alms"], _ = gcr_sample(block, observed, noise_std=sigma,
                                       prior_std=s, tol=tol, maxiter=4000,
                                       require_convergence=None,
                                       key=next(keys))
```
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

## Inference without a likelihood

Every engine above evaluates a likelihood. Simulation-based inference does not:
it draws pairs $(\theta, x)$ from the prior and the simulator, fits a
conditional density $q(\theta \mid x)$ to them, and reads the posterior off $q$
at the data actually observed.

```python
thetas, bank = simulate_pairs(twin, state, space, noise=noise,
                              key=jax.random.key(0), n_simulations=32_768)
q = NeuralPosterior.create(thetas, bank, key=jax.random.key(1))
q, history = train_posterior(q, thetas, bank, key=jax.random.key(2))

draws = q.sample(observed, key=jax.random.key(3), n_samples=4000)
```

Nothing there needs the noise to be Gaussian, the model to be differentiable,
or a normalization to be tractable — only a simulator, which the twin already
is. And the cost is **amortized**: a second observation is a forward pass, not
another chain.

The density is a conditional Gaussian mixture (an MLP → weights, means,
scales). A normalizing flow is more expressive; a mixture is a few dozen lines,
is exact for a Gaussian posterior at one component, and keeps the failure modes
legible.

:::{danger}
**An approximate posterior has no internal notion of being wrong.** A
badly-fitted $q$ returns a smooth, confident, correctly-centred, incorrect
distribution and reports nothing amiss. Both failure modes are real, and they
push in opposite directions — measured on the package's own linear-Gaussian
test problem, where the exact answer is available from `gcr_sample`:

| simulations | steps | components | width / exact |
|---|---|---|---|
| 8 192 | 1 500 | 1 | 0.88 |
| 8 192 | 4 000 | 1 | 0.84 |
| 8 192 | 4 000 | 2 | **0.60** |
| 32 768 | 1 500 | 1 | **0.98** |
| 32 768 | 1 500 | 2 | 1.07 |

*Too few simulations*: draws come from the prior, so only a fraction
$\sigma_\text{post}/\sigma_\text{prior}$ of them land near any given
observation, and the width comes out wrong. *Too many steps on a small bank*:
over-fitting, which makes $q$ too **narrow** — the failure that looks like a
better answer.

`train_posterior` holds out a validation split by default and **returns the
best validation step, not the last**, because the training loss falls
monotonically straight through the point where the fit stops being a
posterior. Prefer few components. And validate on a problem you can solve
exactly before trusting one you cannot — which is what
`tests/inference/test_npe.py` does, and why NUTS was wired to the noise model
first.
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
