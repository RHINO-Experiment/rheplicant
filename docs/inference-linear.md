# Noise, linear blocks, and conditioning

```{include} _bayesmith-note.md
```

Giving the noise is giving the likelihood, and a block that is linear in its
latents has a posterior in closed form. The two are on one page because they
connect: the noise model decides `N` and the prior decides `S`, the linear machinery
consumes them, and the conditioning section says when the answer it returns is
as accurate as its residual suggests.

- [The noise model](#the-noise-model), and the log-det term that is not a constant
- [Linear blocks](#linear-blocks): `check_linearity`, `linear_operator`,
  `wiener_solve`, `gcr_sample`; where `S` comes from; `noise=` vs `noise_std=`
- [Conditioning](#conditioning): why a
  residual is not an accuracy

---

## The noise model

Every inference route needs one number per sample: its noise level. The
likelihood needs it, a Wiener solve and a GCR draw need it as a weight, the
Fisher matrix needs it, a NumPyro observation site needs it as a scale. Passing
it to each of them as a bare `noise_std` assumes it is given and constant. For
a radiometer it is neither:

$$\sigma(d) = \frac{|d|}{\sqrt{\Delta\nu\,\tau}}$$

σ is a function of the quantity being inferred. One object, a noise model,
supplies σ, and every route takes it:

:::{list-table}
:header-rows: 1
:widths: 26 40 34

* - Model
  - σ
  - `depends_on_prediction`
* - `HomoscedasticNoise(sigma)`
  - a constant, which is what a bare `noise_std` means
  - `False`
* - `RadiometerNoise(Δν, τ)`
  - `|prediction| / √(Δν·τ)`
  - `True`
* - `FlaggedNoise(base, flags)`
  - the wrapped model, `∞` where flagged
  - inherited
:::

Wherever an exit has a prediction to evaluate the model at, `noise_std=` takes
a noise model in the bare array's place. `fisher_information` and
`to_numpyro_model` both normalize through `as_noise_model`:

```python
from rheplicant.inference import RadiometerNoise, FlaggedNoise

noise = FlaggedNoise(RadiometerNoise(channel_width=61e3, integration_time=1.0),
                     flags=state.aux["flags"])

fisher_information(forward, params, noise_std=noise)
```

The conjugate solves are the exception. They have no prediction, because the
prediction is what they solve for, so `wiener_solve`, `gcr_sample` and
`condition_estimate` take a decided σ array and refuse a model rather than
freezing it at an arbitrary point. This is the
[`noise=` / `noise_std=` split](#noise-and-noise_std),
below.

`FlaggedNoise` is how RFI flags reach the covariance in the solvers: it wraps a
noise model. (`fisher_information` and `to_numpyro_model` also take a `flags=`
keyword.) An infinite σ encodes "this sample was not observed", and every
consumer turns it into a zero rather than a NaN.

A solver branches on `depends_on_prediction`: `False` means one solve, `True`
means the covariance has to be found before it can be used.

### The log-determinant term

The Gaussian log-density is

$$\log p = -\tfrac{1}{2}\sum_i\left[\frac{r_i^2}{\sigma_i(\theta)^2}
  + \log 2\pi\sigma_i(\theta)^2\right]$$

When σ is constant the second term is an additive constant and dropping it
changes nothing. When σ depends on the prediction, dropping it, as generalized
least squares does, gives a different estimator: one with no penalty for
shrinking the prediction to make the variance small.

For the multiplicative model both are solvable by hand. With $d_i = \theta(1+w_i)$
and $w\sim\mathcal N(0,f^2)$:

| Objective | Stationary point | Expectation |
|---|---|---|
| GLS (log-det dropped) | $\hat\theta = \sum d^2 / \sum d$ | $\theta_{\rm true}(1+f^2)$, biased high |
| Full Gaussian | $nf^2\theta^2 + \theta\sum d - \sum d^2 = 0$ | $\theta_{\rm true}$, unbiased |

The term GLS discards as a normalization is the one that removes the bias.
`NoiseModelLikelihood` keeps it by default; `include_logdet=False` selects the
GLS variant.

:::{warning}
The same thing happens to the Fisher matrix. When the covariance carries
parameter dependence, $J^\top N^{-1} J$ is not the Fisher information. There is
a second term,

$$F = J^\top\Sigma^{-1}J + \tfrac{1}{2}
  \operatorname{tr}\!\left(\Sigma^{-1}\partial\Sigma\,\Sigma^{-1}\partial\Sigma\right),$$

which for a diagonal covariance is $2\,(\partial\log\sigma)^\top(\partial\log\sigma)$.
`fisher_information` includes it whenever the noise model reports
`depends_on_prediction`. Under `RadiometerNoise` it is an overall factor,
$F = (1+2f^2)\,J^\top N^{-1} J$, so reporting only the first term forecasts
error bars too wide by $\sqrt{1+2f^2}$.
:::

---

## Linear blocks

Some parameters enter the model linearly: sky `alm` coefficients, noise-wave
amplitudes, anything whose contribution is a matrix acting on it. Those are
also the large ones: a sky at `lmax` 191 across 32 channels is ~10⁶ real
degrees of freedom, where a gradient sampler is not an option and a conjugate
Gaussian solve is the right method.

The declaration is checked before anything uses it:

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

Use `names=` by default, including, as here, for a block of one latent. Its
answer is a `{name: array}` dict, which is the shape everything downstream
reads. The singular `name="sky_delta"` is legitimate and different, and is
covered [below](#one-latent-or-a-group). With `names=`, `prior_std` takes one
entry per member, because `S` is block-diagonal over a group rather than a
multiple of the identity: one number spread across a noise-wave temperature in
kelvin and a gain of order one would be a prior nobody declared. Omit it and
each latent's own `Latent(prior=...)` drives the solve.

That snippet is a fragment (`maps`, `mean_sky`, `twin`, `state` and `observed`
come from your own model), so it is not run as written.
[`examples/inferring_anything.py`](https://github.com/RHINO-Experiment/rheplicant/blob/main/examples/inferring_anything.py)
executes the same four calls, at this spelling, end to end; the
[next section](#one-latent-or-a-group) has the self-contained version with its
output.

:::{figure} _static/inference-linear-light.svg
:figclass: only-light
:alt: A sky map recovered in closed form from a declared-linear block
:width: 100%

Left: the posterior mean against the truth, inside a 68% band from 400 exact
GCR draws. The band widens where the ~20°-wide beam stops constraining the sky,
and the mean's small-scale ripple lies inside it. Right: RMS error per channel,
before and after. Both the mean and the draws are conjugate-gradient solves;
the same calls take sky alms, where a gradient sampler is not an option.
:::

:::{figure} _static/inference-linear-dark.svg
:figclass: only-dark
:alt: A sky map recovered in closed form from a declared-linear block
:width: 100%

Left: the posterior mean against the truth, inside a 68% band from 400 exact
GCR draws. The band widens where the ~20°-wide beam stops constraining the sky,
and the mean's small-scale ripple lies inside it. Right: RMS error per channel,
before and after. Both the mean and the draws are conjugate-gradient solves;
the same calls take sky alms, where a gradient sampler is not an option.
:::

:::{admonition} Probe scales
:class: important

`check_linearity` probes at 10⁻³, 1 and 10³ times the latent's prior width,
because the prior is where a sampler will go. A knee, a saturation or a small
quadratic is indistinguishable from linear below some scale and grossly
nonlinear above it, so probes at one moderate scale pass the blocks that fail
in a sampler's tails.

A latent with no Gaussian prior falls back to `max|init|`, and to 1.0 if that
is zero, which makes the probes absolute. If such a latent lives at 10⁶ (sky
alms in kelvin), give it a representative `init` or pass `scales=`, or the
sweep never reaches the regime a sampler will.

The check runs at the declared outside values and, by default, at two more
points drawn from the outside latents' priors (`at_points=`). `noise=` adds a
second criterion in units of sigma. The tolerances and the number of points
are bayesmith's (`bayesmith.exact.linearity`).

A block fails only if it exceeds both a relative tolerance and an absolute
floor set by the arithmetic's own roundoff. Without that floor the relative
measure blows up at small probes, where the response variation is vanishing
but roundoff is not, and rejects linear blocks. The cure a user reaches for
after such a false positive is to switch the check off.
:::

:::{admonition} What `LinearityRefused` carries
:class: tip

`check_linearity` returns `{scale: relative departure}` when the block
passes. When it does not, it raises `LinearityRefused`. That is a
`ParameterSpaceError` with its message unchanged, so an existing
`except ParameterSpaceError` needs no change. The same measurement is on the
exception:

```python
from rheplicant.inference import LinearityRefused

try:
    errors = check_linearity(space, twin, state, name="amp")
except LinearityRefused as refused:
    errors = refused.errors        # {scale: departure}, every probe
    refused.failed                 # the scales that exceeded rtol, ascending
    refused.rtol                   # the tolerance actually used
```

Read the trend, not a worst case. "Departs at 1× and 10³× but not at
10⁻³×" is a knee or a saturation and points at the regime; "departs
everywhere" is a wrong parameterization. A maximum over the table cannot tell
those apart, so the exception carries the whole table.

A departure may be non-finite. If the prediction's own arithmetic breaks down
at a probe, that probe is counted as a failure and `nan` is what the table
holds for it. (`nan > rtol` is `False`, so a bare comparison would read it as
a pass.) The `nan` means "the linearization could not be evaluated here",
which is not zero.

An entry may also print as `unresolved:…`. That is
`bayesmith.exact.linearity.Unresolved`, a float: a departure that sits below
the arithmetic's roundoff floor and would otherwise have exceeded `rtol`.
Re-run in float64 to resolve it, and do not format the table with `.1e`
without checking for it.
:::

`linear_operator` never forms a matrix: `A` comes from `jax.linearize` and `Aᵀ`
from `jax.vjp`, so applying a 10⁶-dimensional block costs one forward
evaluation.

### One latent, or a group

`linear_operator` takes `name=` for one latent and `names=` for several exported
as one block. Both are legitimate and they are not interchangeable, so passing
both is refused.

```python
block = linear_operator(space, twin, state, names=("t_unc", "t_cos", "t_sin"))
solved, residual = wiener_solve(block, observed, noise_std=0.02)
solved            # {"t_unc": Array, "t_cos": Array, "t_sin": Array}
```

A group's `x` is a `{name: array}` dict, and so is the answer. The physical
names survive the solve, so the caller does not slice an anonymous stacked
vector, and the dict is the shape the rest of the package consumes. Run against
one model, the two spellings return the same number and differ in what that
number can be handed to:

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

`names=("gain",)` is a legitimate group of one, and is how a partition holds
one-latent and many-latent blocks without special-casing either; the plan's own
engine always spells it that way. Use `names=` by default. The singular is not
deprecated and will not be removed. It is the one-latent shorthand, and its
bare array is the right form when you are about to do linear algebra with it
rather than put it back into the model.

Wrap a bare array as `{block.name: x}` before passing it to anything else.
`LinearBlock.as_dict` does that, and it is a no-op on the grouped form, so it
is correct whichever spelling built the block:

```python
print(singular.as_dict(one), "| forward(...) ->",
      jnp.shape(forward(singular.as_dict(one))))
print("grouped.as_dict(many) == many:", grouped.as_dict(many) == many)
```

```text
{'gain': Array(1.099999, dtype=float32)} | forward(...) -> (8, 4)
grouped.as_dict(many) == many: True
```

Six consumers need that wrap, and none of their exceptions names the
mistake: `space.forward_fn`'s `forward` and `space.bind` raise the `TypeError`
above, `identifiability(at=)` and `linear_operator(at=)` raise `TypeError:
iteration over a 0-d array`, `conditional_potential` raises `TypeError:
'jaxlib._jax.ArrayImpl' object is not a mapping`, and `fisher_information`
raises the string-indexing one again from inside a `jacfwd` trace.
`tests/inference/test_linear_block_as_dict.py` pins all six.

Solving a group jointly is not the same as alternating over its members. Two
latents the data barely tells apart are resolved in one CG here, where
alternation converges at the rate of their correlation while reporting a
converged residual and a well-conditioned block at every step. The joint κ that
`condition_estimate` reports for the group shows the degeneracy, and
`SamplingPlan`'s own non-convergence message recommends the same remedy: "group
the correlated latents into ONE `Block`".

Not everything can be grouped. For a group, `check_linearity` verifies joint
affinity, which a bilinear pair fails. A `gain × T_ant` model is refused as a
group and belongs in two blocks of one
[plan](inference-plans.md#a-plan-one-partition-two-exits).

### Sampling a linear block

`wiener_solve` gives the posterior mean. `gcr_sample` gives a posterior draw,
by adding two white-noise terms to that same right-hand side:

```text
(AᵀN⁻¹A + S⁻¹) x  =  AᵀN⁻¹(d − offset)  +  AᵀN⁻¹ᐟ² ω₁  +  S⁻¹ᐟ² ω₂
```

with `ω₁`, `ω₂` standard normal on the data and on the latent. The right-hand
side then has mean `AᵀN⁻¹(d − offset)` and covariance equal to the operator
itself, so `x = M⁻¹b` carries the posterior mean and covariance
`M⁻¹ M M⁻¹ = M⁻¹` exactly.

```python
sample, residual = gcr_sample(block, observed, noise_std=0.02,
                              prior_std={"sky_delta": 1.0}, key=jax.random.key(0))
sample["sky_delta"]                         # same shape the solve returned
```

Both take `prior_mean=`, which defaults to zero. Zero is wrong for most
physical quantities: a noise-wave temperature sits near 250 K. An affine
binding that adds the same offset gives the identical Gaussian; putting it on
the prior states it as the prior mean.

### Where `S` comes from

Both keywords default to the latent's own declaration. `Latent(prior=...)` is
the one place this package says what a quantity is a priori. `to_numpyro_model`
reads it and so do these solves, so a space handed to NUTS and to `gcr_sample`
targets the same posterior:

```python
space = ParameterSpace.direct(
    "t_nw", init=jnp.zeros((3, N_FREQ)),
    into=lambda p: p["rx"].noise_wave_temps,
    prior=dist.Normal(250.0, 50.0), linear=True,
)
block = linear_operator(space, pipeline, template, names=("t_nw",))
mean, _ = wiener_solve(block, observed, noise_std=sigma)   # S is already known
```

The keywords remain, for a latent with no declared prior. Passing one that
contradicts the declaration raises, naming both numbers; neither of the two
silently wins. A declared prior with no conjugate Gaussian form (a Half-Normal,
a Uniform, a LogNormal) also raises here: these routines solve
`(AᵀN⁻¹A + S⁻¹)x = b`, and substituting such a prior's mean and variance would
hand back a finite posterior for a model you did not declare. Sample that space
with NUTS instead.

With several latents in the space, `linear_operator(..., names=("gain",))`
carries that latent's declaration, so each block gets its own `S`. The Gibbs
sweep below relies on this. The contradiction check reads the two values, not
the context: concrete numbers are compared normally under `jit`,
`eqx.filter_jit` and inside `iterative_gls`'s reweighting loop. Only a keyword
that is itself a tracer is refused as undecidable, because then there is no
number yet to compare.

This is a constrained realization, not a Markov chain: every call is an
independent draw, with no burn-in and no convergence to diagnose. It costs the
same single CG solve as the mean, because the fluctuation enters the
right-hand side and never the operator, so a 10⁶-dimensional block can be
sampled.

:::{tip}
**Gibbs.** A block is only linear given the other latents, so pass `at=` to
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

With the grouped spelling `drawn` is already keyed by latent, so the update is
one `{**values, **drawn}` and stays right when the block later grows a second
member. `SamplingPlan`'s conjugate engine does the same internally, and for
that reason always spells the block `names=`, even for one latent.

Omitting `at=` raises nothing: the block keeps describing the model at its
declared starting point, which is right for exactly one sweep.

The sketch is not runnable as written, because `update_the_nonlinear_ones` is
yours. Its structure carries over: a grouped draw merges into `values` and
`at=values` rebuilds the block from it, with no unpacking in the loop.

[`SamplingPlan`](inference-plans.md#a-plan-one-partition-two-exits) declares
that loop, so you do not write it. It includes the two things a hand-rolled
version leaves out, which are the ones that go wrong.
:::

---

### When the covariance is not given

Both solvers above take `noise_std` wherever it came from. Under
`HomoscedasticNoise` it comes from you. Under `RadiometerNoise` σ tracks the
prediction, so the weights depend on the solution and the solution depends on
the weights. Neither is available first.

`iterative_gls` supplies the covariance by fixed point (solve at the current
σ, recompute σ at the new prediction, repeat), and the call to `gcr_sample`
does not change:

```python
found = iterative_gls(block, observed, noise=RadiometerNoise(dnu, tau),
                      prior_std=PRIOR)

draw, _ = gcr_sample(block, observed, noise_std=found.noise_std,
                     prior_std=PRIOR, key=key)
```

The reweighting loop is `bayesmith.exact.gls.iterative_gls`. It is the same
iteratively-reweighted GLS as hydra-tod's
`hydra_tod.linear_sampler.iterative_gls`, and `tests/inference/test_gls.py`
checks it against a transcription of that function. hydra-tod forms a dense
`U` and `N_inv`; this one is matrix-free on the block's JVP and VJP, which
makes 10⁶ degrees of freedom feasible. This package converts the block and
supplies σ at each prediction.

It returns a `GLSResult`, which carries the answer and the fixed point's
provenance. On a 64×4 design at `RadiometerNoise(1e6, 1.0)`:

```text
solution   [11.999567 -4.997814  3.001249  7.997115]   # truth [12, -5, 3, 8]
noise_std  shape (64,), 0.0054 .. 0.0625               # the σ the fixed point found
residual   2.009e-07     iterations 5
delta      2.394e-07     converged  True
```

`solution` and `noise_std` are the two halves of the answer; the second is what
you hand to `gcr_sample` above. The other two numbers are separate fields
because they measure different loops: `residual` is the inner CG residual of
the final solve, `delta` the outer reweighting step `‖x_new - x‖ / ‖x_new‖`.
`converged` reports on `delta` only: a tight CG residual says nothing about
whether the covariance reached a fixed point. The `reweight_tol` warning below
depends on that distinction.

Check `found.converged`. A covariance that is not a fixed point is still a
number, and a draw conditioned on it is still a draw.

:::{tip}
The mean can be weight-independent while the width is not. In
[`examples/gls_gcr.py`](https://github.com/RHINO-Experiment/rheplicant/blob/main/examples/gls_gcr.py)
three switched loads meet three per-channel noise-wave unknowns, so the reduced
system is square: it has one solution, and the weights cancel out of it.
Reweighting leaves the point estimate exactly unchanged.

The posterior covariance $(A^\top\Sigma^{-1}A + S^{-1})^{-1}$ still depends on
Σ, and a GCR draw has that width. In that example a frozen σ reports error bars
wrong by −8 % to +8 % on a point estimate that was already right. "The fit came
out the same" is not evidence the covariance did not matter.

Where the system is over-determined across different noise levels, the
estimate moves too: on a prediction spanning a decade, freezing σ costs a
factor of ≈2.3 in recovered RMS error.
:::

:::{warning}
`reweight_tol` cannot have a fixed default, and neither should yours. Two
independent floors bound how small a step is measurable. The first is the
arithmetic's epsilon: `1.2e-7` in float32, so `1e-8` is below it. The second
is the inner CG tolerance, since consecutive solves differ by their own
residual whatever the outer iteration does. The second binds in float64, where
a tight `tol=1e-10` sits five orders of magnitude above `eps`. The default is
`max(8·eps, tol)`. Ask for less than either and the run spends `max_reweights`
steps and reports `converged=False` for a fixed point it had reached.
:::

**What this estimator is.** Freezing σ inside each solve makes every step a
linear-Gaussian problem. It also makes the converged answer generalized least
squares rather than the maximum of the full Gaussian likelihood: the
log-determinant's dependence on the solution is held fixed rather than
differentiated ([above](#the-log-determinant-term)). GLS is the right
thing to condition a constrained realization on, because a GCR draw is a draw
from a linear-Gaussian posterior at a given covariance. For the full
likelihood's mode or posterior, use a gradient sampler.

### Log space

`RadiometerNoise` generates `d = μ (1 + f w)`. Take logs:

```text
log d = log μ + log(1 + f w)
```

and `log(1 + f w) → N(0, f²)` to first order. Two things follow.

A block whose `log μ` is affine becomes conjugate. A gain bound as
`Bind("log_gain", into=..., fn=jnp.exp)` makes the prediction `exp(log_gain)·S`,
which is not affine in `log_gain`: `linear_operator` refuses it, and the only
route left is a gradient block. `log_linear_operator` exports the same block
from `log` of the prediction and returns an ordinary `LinearBlock` that
`wiener_solve` and `gcr_sample` accept unchanged:

```python
block = log_linear_operator(space, twin, state, "log_gain", at=values)
y, sigma = to_log_space(observed, noise)
draw, _ = gcr_sample(block, y, noise_std=sigma, prior_std=PRIOR, key=key)
```

σ is constant there, so there is no fixed point to find.
`Var[log(1 + f w)]` is a function of `f` alone: measured at prediction
magnitudes 1, 10³ and 10⁶ it moves only in the fifth significant figure, which
is the Monte Carlo floor of the measurement. The reweighting that
[`iterative_gls`](#when-the-covariance-is-not-given) performs has nothing to
iterate on: one solve, not a loop. The GLS-versus-full-likelihood distinction
on this page is a consequence of σ tracking the prediction, and in log space it
does not.

A summed sky is no obstacle. For `d = g (T_ant + T_nw + tone)`, `log` of the
sum is not affine in the sky coefficients, but the gain block does not need it
to be. Conditional on the sky, `log d = log g + log S` with `log S` a known
constant, and a known constant added to the prediction is what
`LinearBlock.offset` holds. So the gain is log-linear whatever the sky is made
of, and the sky block stays an ordinary linear block in the original space.
Each block takes the space its own conditional is affine in.

An additive term downstream of the gain does break it: `d = g S + C` is
log-linear in neither, because `log` does not distribute over that sum.
`check_log_linearity` refuses it. A receiver offset is the realistic case.

**The approximation, and its size.** First order is not exact:
`E[log(1 + f w)] = −f²/2`, and the variance exceeds `f²`. `to_log_space` adds
the `f²/2` back (a constant, so it is exact arithmetic rather than an estimate),
and `f` above `FIRST_ORDER_MAX_FRACTIONAL` is refused. Measured over 2×10⁷ draws:

| `f` | `Var / f² − 1` | `mean / (−f²/2)` |
|---|---|---|
| 4.0e-3 | below the measurement floor | 1.00 |
| 0.06 | 0.0088 | 1.006 |
| 0.10 | 0.0258 | 1.016 |
| 0.30 | 0.3983 | 1.185 |

The operating range is the top of that table:
`f = 1/√(Δν·τ)` is 4.05×10⁻³ for the 61 kHz × 1 s configuration this page uses,
where the mean shift is 8×10⁻⁶. The refusal at 0.06 is therefore not a limit
met by observing. It fires on a `channel_width` or `integration_time` that is
not what was intended.

:::{note}
Both exits refuse a non-positive value by name rather than letting `log`
produce a NaN. NaN fails every comparison, so a NaN departure reads as passing
`check_linearity`'s `departure > rtol` test and a NaN residual reads as a
converged solve. Flagged samples are exempt: an unobserved sample may hold
anything and is carried through at infinite σ.
:::

### `noise=` and `noise_std=`

Two spellings appear on this page: `iterative_gls(...,
noise=RadiometerNoise(...))` on one line and `gcr_sample(...,
noise_std=found.noise_std)` on the next. They are not two names for one
argument.

* **`noise_std=`** names a σ that has already been decided: an array.
  It appears on `wiener_solve`, `gcr_sample`, `condition_estimate`,
  `fisher_information` and `to_numpyro_model`.
* **`noise=`** names the rule that decides one: a `NoiseModel`.
  It appears on `iterative_gls`, `SamplingPlan.estimate` and
  `SamplingPlan.sample`.

The two keywords are not merged into a single `noise=`. At the conjugate solves
the two are not interchangeable, because there is nothing for a rule to be
evaluated at: `noise.std(prediction)` needs a prediction, and a Wiener solve's
prediction is what it is solving for. A single keyword would make the wrong
call type-check without making it meaningful: the solve would have to freeze σ
at some arbitrary point and hand back the result as a posterior.

So the conjugate seam refuses a model by name, and says which of the two
problems it is. Both calls below were run against the self-contained model in
[One latent, or a group](#one-latent-or-a-group). The refusal fires on the
argument's type and the exit's own name, before anything block-specific, so it
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
Freezing σ is a statistical choice with a stated consequence: an exact draw at
that covariance, which is not the full model's conditional. The choice belongs
to whoever knows which parameter tuple to freeze at. `iterative_gls` makes it
by fixed point and `SamplingPlan` makes it per sweep, which is why both take
`noise=`.

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

Three notes on that table:

* `fisher_information` and `to_numpyro_model` write `noise_std=` and also take
  a model. Both have a prediction, so `as_noise_model` normalizes and
  `noise.std(prediction)` is answerable. The keyword says what the argument is;
  where both readings are usable, both are accepted.
* `condition_estimate` refuses a model with the same sentence its two siblings
  give. It used to refuse with `TypeError: Value 'HomoscedasticNoise(...)' with
  dtype object is not a valid JAX array type`, because it never ran the shared
  `_check_solve_arguments`, so neither the seam refusal nor the 1-D axis check
  reached it. Both run now. The second is the more important: a κ computed
  under a different reading of the same 1-D σ describes a different system
  from the one that is solved. Measured, the two explicit readings give
  different condition numbers.
* `iterative_gls` writes `noise=` and takes only a model: a bare array is
  refused by name. It used to raise `AttributeError: 'ArrayImpl' object has no
  attribute 'depends_on_prediction'`. It is the one `noise=` exit that does not
  route through `as_noise_model`. Its subject is the fixed point a
  prediction-dependent σ implies, so a decided array leaves it nothing to
  iterate. The refusal names both ways out: `wiener_solve` for a constant σ,
  or `HomoscedasticNoise(sigma)` if you want the fixed-point machinery anyway,
  in which case it returns after one step with `converged=True`.

  This exit refuses an array and wants a model, while the conjugate solves
  refuse a model and want an array. Both follow one rule: whether the exit has
  a prediction at which a prediction-dependent σ could be evaluated.

The rest of the argument is in two docstrings, and was measured.
`_refuse_a_noise_model_at_the_conjugate_seam` carries the messages above.
`_check_solve_arguments` carries why this seam is not routed through
`as_noise_model`: `1/σ²` and `inverse_variance` agree on every finite σ, on
`inf` (both exactly `0`), on `0` and on a negative σ, and disagree on NaN. The
conjugate solves propagate a NaN and the caller finds out, while
`inverse_variance` maps it to weight `0.0`, which means "unobserved". In a
conjugate solve a silently dropped sample moves the posterior width, not only
the point, with nothing reporting how many went.

---

## Conditioning

CG reports only `‖M x̂ - b‖`, the **residual**: how well `x̂` satisfies the
equation it was asked to solve. What a caller needs bounded is `‖x̂ - x*‖`, the
**error**: how close `x̂` is to the truth. The two are related through the
condition number of the normal operator `M = AᵀN⁻¹A + S⁻¹`:

```text
‖x̂ - x*‖ / ‖x*‖  ≤  κ(M) · ‖M x̂ - b‖ / ‖b‖
```

For a well-conditioned block (κ ≈ 1) the two coincide and a small residual is
a small error. But κ is large by design when these solvers matter most.
Whenever the data does not fully identify some direction in the block (one
calibration load against three per-channel unknowns, a flagged channel, a
short integration), the prior is the only thing holding that direction down,
so `λ_min(M)` is exactly `1 / prior_std²` and κ runs past `1e6`. CG converges
its residual on the well-constrained directions, which dominate the aggregate
norm, while the prior-dominated directions sit at their starting value. The
residual then looks converged when the solve is not, and a draw built from it
comes back with far too little scatter. Measured: `gcr_sample` on a
badly-conditioned block reported a posterior σ three orders of magnitude too
narrow while its residual sat under a residual-only tolerance.

Two functions report κ, matrix-free, and they are for different jobs:

- `condition_bound(block, noise_std=..., prior_std=...)` is an upper bound on
  κ. It is the number to divide an accuracy target by, and the number the
  guard below reads. It is conservative: on a block the data identifies in
  every direction it can read orders of magnitude above the true κ.
- `condition_estimate(...)` measures κ with a second power iteration and is
  biased low. It is a diagnostic for seeing that a degeneracy is there. A
  tolerance chosen from it is too loose by that bias.

```python
kappa = condition_bound(block, noise_std=0.5, prior_std=100.0)
target_error = 1e-3
solved, residual = wiener_solve(block, observed, noise_std=0.5, prior_std=100.0,
                                tol=target_error / kappa, maxiter=4000)
```

`require_convergence=` is the guard, and it is off by default. Pass a target
(`require_convergence=1e-3`) and the solve raises unless
`condition_bound × relative residual` is below it. Without it, a block the
data does not identify returns whatever CG produced. The solve and the guard
are `bayesmith.exact.solve`; the guard's error is raised inside bayesmith and
is not one of this package's error classes.

[`examples/noise_wave_gcr.py`](https://github.com/RHINO-Experiment/rheplicant/blob/main/examples/noise_wave_gcr.py)
shows both ends. Its three-load block has κ ≈ 2.69e1. Its `--one-source`
variant has κ ≈ 4.35e6: one load against three per-channel unknowns, so two
of every three directions are prior-dominated. The script passes
`tol=1e-10, maxiter=4000` for that case and reports per-channel σ ≈ 71–106 K
against the 100 K prior, the prior width where the data says nothing. The
script records what `tol=1e-6` gave on the same block: σ ≈ 0.03 K with a
residual that looked converged. With `require_convergence=1e-3` passed, that
solve raises instead.

:::{admonition} Cost of the guard
:class: note

The bound costs `POWER_ITERATIONS` (12) operator applications on top of the CG
solve, which is a large fraction of a well-conditioned solve where CG
converges in a handful of iterations. In a Gibbs loop, where the conditioning
barely moves sweep to sweep, compute the bound once outside the loop and
leave the guard off inside. `linear_operator`'s `check` argument offers the
same trade for `check_linearity`:

```python
kappa = condition_bound(block, noise_std=sigma, prior_std=s)  # once
tol = target_error / kappa
for _ in range(n_sweeps):
    block = linear_operator(space, twin, state, names=("sky_alms",),
                            at=values, check=False)
    drawn, _ = gcr_sample(block, observed, noise_std=sigma,
                          prior_std={"sky_alms": s}, tol=tol, maxiter=4000,
                          key=next(keys))
    values = {**values, **drawn}
```
:::
