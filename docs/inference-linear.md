# Noise, linear blocks, and conditioning

```{include} _migration-to-bayesmith.md
```

Giving the noise is giving the likelihood, and a block that is genuinely linear
in its latents has a posterior in closed form. These belong on one page because
they are one argument: the noise model decides what `S` and `N` are, the linear
machinery consumes them, and the conditioning section says when the answer it
returns is worth the digits it prints.

- [The noise model](#the-noise-model) — and the log-det term that is not a constant
- [Linear blocks](#linear-blocks) — `check_linearity`, `linear_operator`,
  `wiener_solve`, `gcr_sample`; where `S` comes from; `noise=` vs `noise_std=`
- [Conditioning](#conditioning-why-a-residual-is-not-an-accuracy) — why a
  residual is not an accuracy

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

Wherever an exit **has a prediction to evaluate the model at**, `noise_std=`
takes one in the bare array's place and no signature changed to allow it —
`fisher_information` and `to_numpyro_model` both normalize through
`as_noise_model`:

```python
from rheplicant.inference import RadiometerNoise, FlaggedNoise

noise = FlaggedNoise(RadiometerNoise(channel_width=61e3, integration_time=1.0),
                     flags=state.aux["flags"])

fisher_information(forward, params, noise_std=noise)
```

The conjugate solves are the exception, and the exception is the interesting
part: they have no prediction — it is what they solve for — so `wiener_solve`,
`gcr_sample` and `condition_estimate` take a decided σ array and refuse a model
rather than freezing it at an arbitrary point. That is the whole of the
[`noise=` / `noise_std=` split](#noise-or-noise_std-the-keyword-is-the-type),
below.

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
check_linearity(space, twin, state, names=("sky_delta",))   # against its own linearization
block = linear_operator(space, twin, state, names=("sky_delta",))  # A, Aᵀ, offset
solved, residual = wiener_solve(block, observed, noise_std=0.02,
                                prior_std={"sky_delta": 1.0})
solved["sky_delta"]                          # the answer, under its own name
```

`names=` is the spelling to reach for, including — as here — for a block of
**one** latent. Its answer is a `{name: array}` dict, which is the shape
everything downstream reads; the singular `name="sky_delta"` is legitimate and
different, and is covered [below](#one-latent-or-a-group). Note what the plural
costs and what it buys: `prior_std` becomes one entry per member, because `S` is
block-diagonal over a group rather than a multiple of the identity — and one
number spread across a noise-wave temperature in kelvin and a gain of order one
would be a prior nobody declared. Omit it entirely and each latent's own
`Latent(prior=...)` drives the solve.

That snippet is a fragment — `maps`, `mean_sky`, `twin`, `state` and `observed`
come from your own model — so it is not run as written. The four calls in it, at
exactly this spelling, are what
[`examples/inferring_anything.py`](https://github.com/RHINO-Experiment/rheplicant/blob/main/examples/inferring_anything.py)
executes end to end; the [next section](#one-latent-or-a-group) has the
self-contained version with its real output.

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

:::{admonition} The refusal carries its numbers, so nothing has to parse it
:class: tip

`check_linearity` **returns** `{scale: relative departure}` when the block
passes. When it does not, it raises `LinearityRefused` — a
`ParameterSpaceError`, with the same message it always had, so an existing
`except ParameterSpaceError` needs no change — and the same measurement is on
the exception:

```python
from rheplicant.inference import LinearityRefused

try:
    errors = check_linearity(space, twin, state, name="amp")
except LinearityRefused as refused:
    errors = refused.errors        # {scale: departure}, every probe
    refused.failed                 # the scales that exceeded rtol, ascending
    refused.rtol                   # the tolerance actually used
```

Read the **trend**, not a worst case. "Departs at 1× and 10³× but not at
10⁻³×" is a knee or a saturation and points at the regime; "departs
everywhere" is a wrong parameterization. A maximum over the table cannot tell
those apart, which is why the whole table is what is carried.

A departure may be **non-finite**: if the prediction's own arithmetic breaks
down at a probe, that probe is counted as a failure — `nan > rtol` is `False`,
so treating it as a pass would be exactly backwards — and `nan` is what the
table holds for it. It means "the linearization could not be evaluated here",
which is not zero.
:::

`linear_operator` never forms a matrix: `A` comes from `jax.linearize` and `Aᵀ`
from `jax.vjp`, so applying a 10⁶-dimensional block costs one forward
evaluation.

### One latent, or a group

`linear_operator` takes `name=` for one latent and `names=` for several exported
as **one** block. Both are legitimate, and they are not interchangeable — which
is why passing both is refused rather than guessed.

```python
block = linear_operator(space, twin, state, names=("t_unc", "t_cos", "t_sin"))
solved, residual = wiener_solve(block, observed, noise_std=0.02)
solved            # {"t_unc": Array, "t_cos": Array, "t_sin": Array}
```

A group's `x` is a `{name: array}` dict, and so is the answer. That is the
point: the physical names survive the solve instead of the caller slicing an
anonymous stacked vector — and the dict is the shape the rest of the package
consumes. Run both spellings against one model and the difference is not in the
number, it is in what the number can be handed to:

```python
from rheplicant.core.pipeline import Pipeline
from rheplicant.radio import GainOperator, SkyOperator

state = State(coords=Coordinates(time=jnp.linspace(0.0, 60.0, 8),
                                 freq=jnp.linspace(60e6, 85e6, 4)),
              env=Environment(temperature=jnp.array(280.0)), key=jax.random.key(0),
              meta={"telescope": "my-antenna", "obs_id": "tour-001"})
twin = Pipeline(SkyOperator(amplitude=jnp.array(100.0)),
                GainOperator(gain=jnp.array(1.0)), names=("sky", "gain"))
space = ParameterSpace.direct("gain", init=1.0, into=lambda p: p["gain"].gain,
                              prior=dist.Normal(1.0, 0.3), linear=True)
forward, _ = space.forward_fn(twin, state)
observed = forward({"gain": jnp.array(1.1)})

grouped = linear_operator(space, twin, state, names=("gain",))
singular = linear_operator(space, twin, state, name="gain")
many, _ = wiener_solve(grouped, observed, noise_std=0.5)
one, _ = wiener_solve(singular, observed, noise_std=0.5)

print("many:", many, "| forward(many) ->", jnp.shape(forward(many)))
print("one: ", repr(one))
forward(one)
```

```text
many: {'gain': Array(1.099999, dtype=float32)} | forward(many) -> (8, 4)
one:  Array(1.099999, dtype=float32)
TypeError: JAX does not support string indexing; got idx='gain'
```

`names=("gain",)` is a legitimate **group of one**, and is how a partition holds
one-latent and many-latent blocks without special-casing either; the plan's own
engine always spells it that way. Reach for `names=` by default. The singular is
not deprecated and is not going away — it is the one-latent shorthand, and its
bare array is the right thing when you are about to do linear algebra with it
rather than put it back into the model.

When you do have a bare one, wrap it as `{block.name: x}` before it meets
anything else. `LinearBlock.as_dict` is that call, and it is a no-op on the
grouped form, so it is correct whichever spelling built the block:

```python
print(singular.as_dict(one), "| forward(...) ->",
      jnp.shape(forward(singular.as_dict(one))))
print("grouped.as_dict(many) == many:", grouped.as_dict(many) == many)
```

```text
{'gain': Array(1.099999, dtype=float32)} | forward(...) -> (8, 4)
grouped.as_dict(many) == many: True
```

Six consumers need that wrap, and none of their exceptions names the actual
mistake: `space.forward_fn`'s `forward` and `space.bind` raise the `TypeError`
above, `identifiability(at=)` and `linear_operator(at=)` raise `TypeError:
iteration over a 0-d array`, `conditional_potential` raises `TypeError:
'jaxlib._jax.ArrayImpl' object is not a mapping`, and `fisher_information`
raises the string-indexing one again from inside a `jacfwd` trace.
`tests/inference/test_linear_block_as_dict.py` pins all six, so if any of them
ever starts accepting the bare form, that is a test going red rather than a
paragraph going quietly stale.

**Solving a group jointly is not the same as alternating over its members.** Two
latents the data barely tells apart are resolved in one CG here, where
alternation converges at the rate of their correlation while reporting a
converged residual and a well-conditioned block at every step. The joint κ that
`condition_estimate` reports for the group is the honest one — and "group the
correlated latents into ONE `Block`" is exactly what `SamplingPlan`'s own
non-convergence message recommends.

Not everything can be grouped, and the check says so: for a group,
`check_linearity` verifies **joint** affinity, which a bilinear pair fails. A
`gain × T_ant` model is refused as a group and belongs in two blocks of one
[plan](inference-plans.md#a-plan-one-partition-two-exits).

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
                              prior_std={"sky_delta": 1.0}, key=jax.random.key(0))
sample["sky_delta"]                         # same shape the solve returned
```

Both take `prior_mean=`, which defaults to zero — wrong for most physical
quantities, since a noise-wave temperature sits near 250 K, not near zero. An
affine binding that adds the same offset gives the identical Gaussian, but
putting it on the prior says what it means.

### Where `S` comes from

Both keywords default to the latent's own declaration. `Latent(prior=...)` is
the one place this package says what a quantity is a priori, and it is what
`to_numpyro_model` reads — so it is what these solves read too, and a space
handed to NUTS and to `gcr_sample` targets one posterior rather than two:

```python
space = ParameterSpace.direct(
    "t_nw", init=jnp.zeros((3, N_FREQ)),
    into=lambda p: p["rx"].noise_wave_temps,
    prior=dist.Normal(250.0, 50.0), linear=True,
)
block = linear_operator(space, pipeline, template, names=("t_nw",))
mean, _ = wiener_solve(block, observed, noise_std=sigma)   # S is already known
```

The keywords remain, for a latent with no declared prior. What is *not*
allowed is passing one that contradicts the declaration — that raises, naming
both numbers, rather than letting one of the two silently win. A declared
prior with no conjugate Gaussian form (a Half-Normal, a Uniform, a LogNormal)
also raises here: these routines solve `(AᵀN⁻¹A + S⁻¹)x = b`, and substituting
such a prior's mean and variance would hand back a finite, confident posterior
for a model you did not declare. Sample that space with NUTS instead.

With several latents in the space, `linear_operator(..., names=("gain",))`
carries **that** latent's declaration — each block gets its own `S`, which is
what makes the Gibbs sweep below sound. And the contradiction check reads the two
values, not the context: concrete numbers are compared normally under `jit`,
`eqx.filter_jit` and inside `iterative_gls`'s reweighting loop. Only a keyword
that is *itself* a tracer is refused as undecidable, because then there is no
number yet to compare.

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
check_linearity(space, twin, state, names=("sky_alms",))   # once
for _ in range(n_sweeps):
    block = linear_operator(space, twin, state, names=("sky_alms",),
                            at=values, check=False)         # every sweep
    drawn, _ = gcr_sample(block, observed, noise_std=sigma,
                          prior_std={"sky_alms": s}, key=next(keys))
    values = {**values, **drawn}                            # the dict merges straight in
    values = update_the_nonlinear_ones(values)              # NUTS, MH, optimize...
```

That merge is the grouped spelling paying for itself: `drawn` is already keyed
by latent, so the update is one `{**values, **drawn}` and stays right when the
block later grows a second member. It is exactly what `SamplingPlan`'s conjugate
engine does internally, which is why that engine always spells the block
`names=` even for one latent.

Omitting `at=` is silent, not loud: the block keeps describing the model at its
declared starting point, which is right for exactly one sweep.

The sketch is not runnable as written — `update_the_nonlinear_ones` is yours —
but its shape is: a grouped draw merges into `values` and `at=values` rebuilds
the block from it, with no unpacking anywhere in the loop.

That loop is what [`SamplingPlan`](inference-plans.md#a-plan-one-partition-two-exits) declares, so
you do not write it — including the two things a hand-rolled version leaves out,
which are the ones that go wrong.
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

What comes back is a `GLSResult`, and it carries the fixed point's whole
provenance rather than just the answer — on a 64×4 design at
`RadiometerNoise(1e6, 1.0)`:

```text
solution   [11.999567 -4.997814  3.001249  7.997115]   # truth [12, -5, 3, 8]
noise_std  shape (64,), 0.0054 .. 0.0625               # the σ the fixed point found
residual   2.009e-07     iterations 5
delta      2.394e-07     converged  True
```

`solution` and `noise_std` are the two halves of the answer — the second is what
you hand to `gcr_sample` above. The other two numbers are separate fields
because they measure different loops: `residual` is the *inner* CG residual of
the final solve, `delta` the *outer* reweighting step `‖x_new - x‖ / ‖x_new‖`.
`converged` reports on `delta` only — a tight CG residual says nothing about
whether the covariance reached a fixed point, which is the whole distinction the
`reweight_tol` warning below turns on.

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

### `noise=` or `noise_std=`: the keyword is the type

Two spellings have now run past each other on this page — `iterative_gls(...,
noise=RadiometerNoise(...))` on one line and `gcr_sample(...,
noise_std=found.noise_std)` on the next — and they are **not** two names for one
argument.

* **`noise_std=`** names a σ that has already been decided: an array.
  It appears on `wiener_solve`, `gcr_sample`, `condition_estimate`,
  `fisher_information` and `to_numpyro_model`.
* **`noise=`** names the *rule that decides* one: a `NoiseModel`.
  It appears on `iterative_gls`, `SamplingPlan.estimate` and
  `SamplingPlan.sample`.

A previous recommendation was to rename `noise_std=` to `noise=` throughout,
keeping the old spelling as a deprecated alias. **It is rejected**, and the
reason is written here rather than left in a review nobody can find: at the
conjugate solves the two are not interchangeable, because there is nothing for a
rule to be evaluated *at*. `noise.std(prediction)` needs a prediction; a Wiener
solve's prediction is exactly what it is solving for. Renaming the keyword would
make the wrong call type-check without making it meaningful — the solve would
have to freeze σ at some arbitrary point and hand back the result as a
posterior.

So the conjugate seam refuses a model **by name**, and says which of the two
problems it is. Both calls below were run against the self-contained model in
[One latent, or a group](#one-latent-or-a-group); the refusal fires on the
argument's *type* and the exit's own name, before anything block-specific, so it
reads the same for any block:

```python
from rheplicant.inference import HomoscedasticNoise

wiener_solve(block, observed, noise_std=HomoscedasticNoise(sigma=0.5))
```

```text
ParameterSpaceError: wiener_solve takes a plain sigma array, not a
HomoscedasticNoise. The conjugate solves compute 1/sigma**2 directly; pass
`noise.std(...)`, or the sigma you built the model from.
```

```python
wiener_solve(block, observed,
             noise_std=RadiometerNoise(channel_width=61e3, integration_time=1.0))
```

```text
ParameterSpaceError: wiener_solve was given RadiometerNoise, whose sigma
depends on the prediction — but a conjugate solve has no prediction to
evaluate it at, because the prediction is what it solves for. Freeze it
yourself at the parameter tuple you mean (`noise.std(prediction)`) and pass
that array, which also makes explicit that the result is an exact draw at
THAT covariance and not from the full model's conditional. A SamplingPlan
does this per sweep.
```

The second message is longer because that case is not a packaging problem.
Freezing σ is a real statistical choice with a stated consequence — an exact
draw at *that* covariance, which is not the full model's conditional — and it
belongs to whoever knows which parameter tuple to freeze at. `iterative_gls`
is that choice made by fixed point, `SamplingPlan` is it made per sweep, and
both take `noise=` for precisely that reason.

Measured across the exits, one call per cell:

| exit | keyword | bare σ array | `HomoscedasticNoise` | `RadiometerNoise` |
|---|---|---|---|---|
| `wiener_solve` | `noise_std=` | ✅ | ❌ named refusal | ❌ named refusal |
| `gcr_sample` | `noise_std=` | ✅ | ❌ named refusal | ❌ named refusal |
| `condition_estimate` | `noise_std=` | ✅ | ❌ named refusal | ❌ named refusal |
| `fisher_information` | `noise_std=` | ✅ | ✅ | ✅ |
| `to_numpyro_model` | `noise_std=` | ✅ | ✅ | ✅ |
| `iterative_gls` | `noise=` | ❌ named refusal | ✅ | ✅ |
| `SamplingPlan.estimate` | `noise=` | ✅ | ✅ | ✅ |
| `SamplingPlan.sample` | `noise=` | ✅ | ✅ | ✅ |

Three edges that table makes visible, and none of them softens the split:

* `fisher_information` and `to_numpyro_model` write `noise_std=` and take a
  model anyway. They can — both *have* a prediction, so `as_noise_model`
  normalizes and `noise.std(prediction)` is answerable. The keyword says what
  the argument is; where both readings are usable, both are accepted.
* `condition_estimate` refuses a model with the same sentence its two siblings
  give. It used to refuse with `TypeError: Value 'HomoscedasticNoise(...)' with
  dtype object is not a valid JAX array type`, because it never ran the shared
  `_check_solve_arguments` — so neither the seam refusal nor the 1-D axis check
  reached it. Both run now, and the second matters more than the message did:
  this is the function a caller is told to consult to pick `tol` for those
  solves, and a κ computed under a different reading of the same 1-D σ answers
  a different question than the solve it was computed for. Measured, the two
  explicit readings give different condition numbers.
* `iterative_gls` writes `noise=` and takes **only** a model — a bare array is
  refused by name. It used to raise `AttributeError: 'ArrayImpl' object has no
  attribute 'depends_on_prediction'`, an attribute the caller never wrote, from
  a layer they were not thinking about. It is the one `noise=` exit that does
  not route through `as_noise_model`, and that is deliberate rather than an
  oversight: its whole subject is the fixed point a prediction-dependent σ
  implies, so a decided array leaves it nothing to iterate. Accepting one
  silently would answer a question nobody asked. The refusal names both ways
  out — `wiener_solve` for what a constant σ actually wants, or
  `HomoscedasticNoise(sigma)` if you want the fixed-point machinery anyway, in
  which case it returns after one step with `converged=True`.

  Note the direction: this exit refuses an **array** and wants a model, while
  the conjugate solves refuse a **model** and want an array. Both are the same
  rule seen from two sides — whether the exit has a prediction at which a
  prediction-dependent σ could be evaluated.

The rest of the argument lives in two docstrings, and was measured rather than
asserted. `_refuse_a_noise_model_at_the_conjugate_seam` carries the messages
above; `_check_solve_arguments` carries why this seam is deliberately *not*
routed through `as_noise_model`, which is that `1/σ²` and `inverse_variance`
agree on every finite σ, on `inf` (both exactly `0`), on `0` and on a negative
σ — and disagree on **NaN**, where the conjugate solves propagate it and the
caller finds out, while `inverse_variance` maps it to weight `0.0`, which
*means* "unobserved". In a conjugate solve a silently dropped sample moves the
posterior **width**, not only the point, with nothing reporting how many went.

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
    block = linear_operator(space, twin, state, names=("sky_alms",),
                            at=values, check=False)
    drawn, _ = gcr_sample(block, observed, noise_std=sigma,
                          prior_std={"sky_alms": s}, tol=tol, maxiter=4000,
                          require_convergence=None, key=next(keys))
    values = {**values, **drawn}
```
:::
