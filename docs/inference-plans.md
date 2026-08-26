# Plans and engines

```{include} _migration-to-bayesmith.md
```

One space can be stepped by several engines at once. A `SamplingPlan` declares
the partition and *derives* each block's engine from what the latents already
say about themselves, rather than restating it.

- [A plan: one partition, two exits](#a-plan-one-partition-two-exits)
- [One space, every engine](#one-space-every-engine)
- [Inference without a likelihood](#inference-without-a-likelihood)
- [Tutorials](#tutorials)
- [Run it](#run-it)

---

## A plan: one partition, two exits

The [linear machinery](inference-linear.md) answers for **one** block.
`wiener_solve` is a linear-Gaussian block's posterior mean, `gcr_sample` an
exact draw from that same conditional, and its
[Gibbs tip](inference-linear.md#sampling-it-exactly) sketches the loop you would
write by hand to put several blocks together. `SamplingPlan` is that loop,
declared rather than written:

```python
plan = SamplingPlan(
    space,
    Block("gain"),                  # conjugate — gain is Latent(..., linear=True)
    Block("t_coeff"),               # conjugate
    Block("beam_fwhm", steps=20),   # not declared linear → gradient
)

est   = plan.estimate(twin, state, observed, noise=noise)
draws = plan.sample(twin, state, observed, noise=noise, key=k, n_sweeps=200)
```

Every number quoted below is measured on the package's own fixture in
`tests/inference/test_plan.py`: a bilinear `gain × T_ant` model, a 6-element
gain against a `(3, 4)` time × frequency coefficient basis, 54 data points and
18 parameters. It is asymmetric in every dimension on purpose.

### The engine is derived, never restated

`Latent(..., linear=True)` already says which machinery a latent can take, so
`Block` does not ask again. A block whose members are all declared linear is
solved by the [conjugate routines](inference-linear.md#linear-blocks); anything
else is stepped by gradient.

There is a third engine, `log_conjugate`, and it is *not* derived — because
there is no declaration to derive it from. A block is
[conjugate in log space](inference-linear.md#the-same-model-in-log-space-where-sigma-is-constant)
when the prediction is `exp` of an affine map, which no `Latent` field states;
it is either asked for with `engine="log_conjugate"` or found by
[`auto_blocks`](#the-partition-can-be-derived-too) probing for it. Asking for it
on a latent declared `linear=True` is refused, because the two claims exclude
each other: `exp` of an affine function is affine only where it is constant.

The plan's `repr` reports what it derived:

```text
SamplingPlan(('gain'):conjugate, ('t_coeff'):conjugate)
SamplingPlan(('amp'):conjugate, ('centre'):gradient)
```

Exactly one case is genuinely ambiguous — a block mixing declared-linear and
non-linear members — and it is refused rather than guessed:

```text
Block('amp', 'centre') mixes declared-linear latents ['amp'] with non-linear
ones ['centre'], so which engine it takes cannot be derived. A conjugate solve
needs the whole block affine; a gradient step does not exploit the linear
members' structure at all, which for a high-dimensional linear block is the
difference between tractable and hopeless. Split them into separate blocks, or
say engine='gradient' to step the whole block by gradient deliberately.
```

`engine=` exists for that override and for nothing else. `steps=` on a
conjugate block raises for the same reason: a Wiener solve has no inner steps,
so accepting the argument would silently ignore it.

### The partition is checked, and the check is the point

Every latent of the space in exactly one block. Both ways of getting that wrong
are refused by name, and the dangerous one is the omission:

```text
This plan does not cover latent(s) ['t_coeff']: every latent of the space must
be in exactly one block. An omitted latent is silently frozen at its declared
init for the whole run — the sweep converges, the joint chi-squared settles, and
nothing anywhere reports that a parameter you declared was never inferred. Add
it to a block, or drop it from the space.
```

A latent in *two* blocks is refused too: the second update each sweep would be
solving a conditional the first had just invalidated, and every diagnostic would
report the second's answer as if the first had never run.

### The partition can be derived, too

Declaring the blocks stays available and stays the honest default for a model
you know. When you would rather not, `auto_blocks` reads the partition off the
model, and `SamplingPlan.automatic` is the one-liner over it:

```python
plan = SamplingPlan.automatic(space, twin, state, noise=noise)

# the same thing, with the blocks in reach to inspect or amend
plan = SamplingPlan(space, *auto_blocks(space, twin, state, noise=noise, steps=20))
```

`noise=` is optional and only the log-space half needs it — see below.

The rule is conjugate blocks for the latents declared `linear=True` and one
gradient block for everything else — with two refinements, both of which the
probe supplies and neither of which you declare.

**One block per factor, not one per latent and not one for all.**
`Latent(..., linear=True)` is a claim about **one** latent: the prediction is
affine in it *with the others held fixed*. A conjugate block over several of
them claims something strictly stronger, that the prediction is affine in them
**jointly**, and on a multilinear model that is false while every member's own
declaration is true. So sweeping every linear latent into one block is wrong,
and one block per latent is needlessly worse — the correct partition is one
block per *factor*, holding all of that factor's latents. On
`gain × (B_ant @ t_ant + B_nw @ t_nw + tone)`:

```text
SamplingPlan(('t_ant', 't_nw'):conjugate, ('gain'):conjugate)
```

**And log-linearity is discovered, not declared.** A latent the prediction is
`exp`-affine in — a gain bound as `Bind(..., fn=jnp.exp)` — has no `linear=True`
to read, and there is deliberately no `log_linear=True` either: the same probe
that finds the grouping asks
[`check_log_linearity`](inference-linear.md#the-same-model-in-log-space-where-sigma-is-constant)
and routes it to a `log_conjugate` block.

**Discovery needs the noise, and this is half the question rather than a
detail.** A log-conjugate block is a claim about the *likelihood*: taking logs
simplifies a multiplicative noise and merely restates an additive one as a
different likelihood from the one declared, and the first-order equivalence
holds only up to `FIRST_ORDER_MAX_FRACTIONAL`. So `auto_blocks` takes `noise=`
and applies both refusals when it partitions. Without it, no `log_conjugate`
block is claimed at all and an `UncheckedLogRouteWarning` names the latents
that qualified on their prediction alone — conservative, because a gradient
block is always a sound verdict, and loud, because the alternative is a
partition promising a route `to_log_space` will refuse.

On the same model with the gain in log space, nothing is declared about the
gain at all and the partition comes back closed-form throughout:

```text
SamplingPlan(('t_ant', 't_nw'):conjugate, ('log_gain'):log_conjugate)
```

The two probes take separate `scales`, and that is not tidiness: feeding the
linear default's `1e3` entry to a log probe sends it through an exponential
that overflows, the check refuses, and a genuinely log-linear latent is filed as
gradient — a misclassification that costs a conjugate block and reports nothing.
`log_scales=` is the argument, `LOG_DEFAULT_SCALES` the default.

#### A worked example: all three engines from one model

For `d = exp(Ax) · (By + exp(Cz)) · (1 + f w)` — a matrix-exponential gain over
a summed sky under radiometer noise — declare `linear=True` on `y` alone (the
one latent the prediction is affine in) and let the probes decide the rest:

```text
SamplingPlan(('y'):conjugate, ('x'):log_conjugate, ('z'):gradient)
```

`y` is conjugate in the original space, solved each sweep with sigma frozen at
the current prediction — the GLS-flavoured choice this page documents above.
`x` is discovered: nothing was declared for it, but `log(prediction) = Ax +
log(By + exp(Cz))` is affine in it given the others, so the probe routes it to
the log-space engine. `z` fails both probes and takes NUTS. Three engines, one
sweep.

Two things this example teaches beyond the partition:

* **A correct partition is not a convergence proof.** This model is fully
  identified (`identifiability` reports nullity 0) with
  `weakest_identified = 5.4e-4` — one direction constrained thousands of times
  more weakly than the rest, because `exp(Ax)`'s shape partly trades against
  the sky's. Single-latent Gibbs alternation random-walks along that valley:
  measured, two 400-sweep chains from different starts each reported tight
  widths while sitting sixteen of those widths apart, and the joint
  chi-squared `rhat` flagged nothing — chi-squared is FLAT along a valley by
  construction, which is a blindness this page's own monitoring section
  warns about in its other form. On such a model, put the strongly coupled
  latents in ONE block by hand (`Block("y", "z", steps=...)`), and run a
  second chain from a different start before believing any of them.
* **Where this package's plans stop.** A `Latent`'s prior is a fixed
  distribution; a prior *parameterised by another latent* — a field `w1`
  whose statistics a hyperparameter sets — has no spelling in a
  `ParameterSpace`, and a plan cannot sweep what it cannot declare. That
  hierarchical variant of this same model is the second worked example in
  bayesmith's documentation (`docs/factor-partition-examples.md` in that
  repository), where the graph paradigm carries it natively and the
  partition rule it showcases — a hyperparameter is ejected from every exact
  block, because an exact block solving only against data would silently
  drop the `p(w1 | y)` factor — has no counterpart here to drift from.

**Pairs settle it.** For latents already known to be affine on their own, every
diagonal block of the group's Hessian vanishes, so joint affinity is exactly the
claim that the off-diagonal ones do too — a question about pairs. Probing the
`C(n, 2)` pairs with `check_linearity` therefore decides a property of all
`2ⁿ − n − 1` subsets, and the verdicts colour a graph whose groups are the
blocks. The cost is that quadratic count of probes, each a linearization plus
one forward per entry in `scales`; nothing here switches on a size, for the same
reason `check_identifiability` does not.

Deriving the partition changes nothing downstream. The blocks are checked
exactly as declared ones are, each conjugate block's joint linearity re-verified
at the first sweep, and — the part worth saying plainly — **a derived partition
is not a reason to believe a model**. Two coupled conjugate blocks are precisely
the configuration whose degenerate case converges quietly onto an arbitrary
point with every per-block guard green; `identifiability` is what sees that, and
both exits still run it by default.

### Two methods, not a mode flag

`key=None | key` is the right *implementation* and the wrong *interface*. A
caller's intent is "give me the best fit" or "give me draws", not "here is a
PRNG key" — and two methods make the invalid combinations unrepresentable
rather than merely validated: `key` is required on `sample` and absent from
`estimate`, so "asked for draws and forgot the key" cannot be written down.
`n_sweeps` and `warmup` belong to one, `max_iter` and `tol` to the other,
because they mean nothing to the other. Both return the same currency —
`result.diagnostics` and `result.names` — so a caller can log or assert on a run
without knowing which exit produced it.

That shared currency has a name. `PlanResult` is the protocol both exits
satisfy: `.diagnostics` is a `PlanDiagnostics`, `.names` are the latents. What
differs is what the answer *is*, which is the honest difference between the two:

| exit | returns | the answer |
|---|---|---|
| `plan.estimate` | `Estimate` | `.values` — one array per latent |
| `plan.sample` | `Draws` | `.samples` — a chain per latent, plus `.n_draw`, `.mean`, `.std` (properties, not calls) |

Annotate against `PlanResult` when a function of yours should take either, and
against `Estimate` or `Draws` when it genuinely needs the values or the chain.

Both exits on the fixture above, at `HomoscedasticNoise(1.0)`:

```text
plan.estimate   sweeps 93   converged True   chi2 1.16085e+08 -> 0.0136245
                block residuals {('gain',): 7.56e-07, ('t_coeff',): 1.03e-06}
                max |T_ant - truth| = 0.0332 K

plan.sample     n_sweeps  60   kept  30   rhat 1.434   converged False
                n_sweeps 200   kept 100   rhat 0.99    converged True
                n_sweeps 600   kept 300   rhat 1.001   converged True
```

The 93 sweeps and the `rhat = 1.434` are the same fact seen twice: these two
blocks are strongly correlated, so the alternation moves slowly, and 30 kept
draws are nowhere near stationarity. Neither exit hides it — `converged` is
`False` on the short run and `PlanDiagnostics.rhat` says by how much.

That diagnostic is computed by `split_rhat`, which is exported and worth
reaching for on any chain you hold — cut the trace in half, treat the halves as
two chains, compare the variance between them against the variance within:

```python
import jax, jax.numpy as jnp
from rheplicant.inference import split_rhat

split_rhat(jax.random.normal(jax.random.key(0), (200,)))   # 1.0043  — iid
split_rhat(jnp.arange(200.0))                              # 2.6326  — pure drift
```

Two degenerate readings are deliberate rather than defensive: halves that are
each constant at the *same* value give `1.0` (nothing to mix), and halves each
constant at *different* values give `inf` (a chain that moved once and stopped).

**A trace too short to halve is refused, not answered**, and that is the part
worth knowing before you call it. The minimum is `MIN_DRAWS = 4` — two halves of
two, and it lives in `rheplicant.inference.plan`, not on the package surface.
Below it the diagnostic is not weak but undefined:

```text
split_rhat was given 3 value(s) and a split-r_hat needs at least 4 — two halves
of two. Below that the mixing diagnostic is not weak, it is undefined: halves of
one have no variance within them to divide by, so the answer came back as nan
rather than as this refusal — and a nan passes no threshold test in either
direction, which makes an undefined diagnostic read as whichever verdict the
caller tested for. SamplingPlan.sample refuses the same count on
(n_sweeps - warmup); this is that refusal, for the trace you brought yourself.
```

The middle clause is the reason this is an exception and not a `nan`: `rhat <=
rhat_max` is `False` for a nan and so is `rhat > rhat_max`, so a threshold guard
reads an undefined diagnostic as whichever answer it happened to test for.
`SamplingPlan.sample` enforces the same minimum on `n_sweeps - warmup`, so
reaching this by that route is not possible; `split_rhat` enforces it anyway,
because it is public and does not trust its one in-package caller.

### Convergence is monitored on the joint χ², never a per-block residual

This is the module's reason for existing. A hand-rolled alternating solve over
this same bilinear model, with a free antenna temperature per `(time, frequency)`
cell, lands hundreds to thousands of kelvin from the truth while **every
per-block guard this package ships reports green**: `check_linearity` passes at
every sweep, because each conditional genuinely *is* affine; the per-block
condition number is ≈1.47; and the CG residual reads ~`1e-7`. Nothing in the
sweep is wrong. The *partition* is, and no per-block number is entitled to
notice — a residual and a condition number are both computed from the block
being solved.

How far from the truth is not a property of the degeneracy but of where the
solve started, because the answer is the initial offset carried along the null
direction and left there. Measured in
`tests/inference/test_degenerate_partition.py`:

| start | rms error | CG residual | κ |
|---|---|---|---|
| at the truth | 0.014 K | 1.3e-07 | 1.467 |
| 1 % off | 27.4 K | 1.1e-07 | 1.466 |
| 25 % off | 704 K | 1.0e-07 | 1.452 |
| 100 % off | 2962 K | 9.6e-07 | 1.432 |

Four decades of error, and the guards read alike down the column — including
across the gap between the row that is right and the row that is
catastrophically wrong. There is no threshold to place between them, which is
why the remedy is a different measurement rather than a tighter tolerance.
Iterating is not one either: five sweeps and two hundred agree to four figures,
because the solve reaches the solution manifold at once and then has nowhere
left to move.

So the monitored quantity is the **joint** χ² at the current parameter tuple,
across sweeps. When it has not settled, the refusal says so and names what the
per-block numbers were doing at the time:

```text
SamplingPlan.estimate did not converge: after 4 sweeps the JOINT chi-squared is
still falling by 728774 per sweep (chi2 = 2.31316e+06), which is above
tol=1e-08. Note what this does NOT show up in: every conjugate block's own CG
residual is 4.18e-07 or better, because a per-block residual is computed from
the block and converges at every sweep of an alternation that is going nowhere.
```

It is a *decrease* that is tested, not a change: block-coordinate descent cannot
increase the objective, so once a sweep stops reducing it there is nothing left
to reach. Testing `|χ²[k] − χ²[k−1]|` instead walks into the floor
[`iterative_gls`](inference-linear.md#when-the-covariance-is-not-given) documents for its own
`reweight_tol` — consecutive sweeps differ by roughly the inner solver's noise
whatever the outer iteration does. Measured in float32 on the motivating model,
the plateau sits at χ² = 2.7e-3 and jitters by 1.2e-3 a sweep, so a converged run
would have been refused for 300 sweeps and counting.

:::{important}
**The joint χ² catches a slow partition, not a degenerate one.** Running the
free-per-cell parameterization with `check_identifiability=False` and `tol=None`
for 40 sweeps:

```text
worst per-block residual  5.55e-07
joint chi2                8.88e-06
max |T_ant - truth|        1044.69 K
max |gain  - truth|            0.308
```

The joint χ² is *also* tiny, because a degenerate model fits the data exactly —
it just does so at an arbitrary point of the null space. χ² is the right monitor
for blocks that are identified but correlated; the rank test below is the only
thing that sees the other failure. They are two guards, not one guard twice.
:::

### The identifiability check, and its cadence

`identifiability()` looks across blocks at the joint Jacobian and refuses a
model with a null space before a sweep runs, naming the degenerate directions as
combinations of **latents** — "you have 6 blind directions" tells a user they
have a problem and nothing about which:

```text
SamplingPlan.estimate refuses this model: its joint Jacobian has nullity 6 of 60
parameters, so that many independent directions leave the prediction unchanged
and any answer along them is arbitrary. No per-block guard can see this — a
residual and a condition number are both computed from the block being solved —
so the run would otherwise converge quietly onto one arbitrary point of the null
space. The degenerate directions, as shares of each latent:
  direction 0: t_ant 0.50, gain 0.50
  direction 1: t_ant 0.50, gain 0.50
  direction 2: t_ant 0.50, gain 0.50
  direction 3: gain 0.50, t_ant 0.50
  ... and 2 more
```

Read the shares: each blind direction is half gain and half `t_ant`, which is
the bilinear degeneracy `gain × T_ant = (c·gain) × (T_ant/c)` written down. The
repair is a re-parameterization — the `(3, 4)` basis — not a tighter tolerance.

Called directly, `identifiability()` hands back an `IdentifiabilityReport`
rather than raising, which is how you ask the question before committing to a
partition. On a healthy 64×4 design:

```text
names ('coeffs',)   n_par 4   n_data 64
rank  4             nullity 0
rtol  1e-08         threshold 1.2210e-08
singular_values [1.22098371 1.01958727 0.90795122 0.80328398]
```

`rank` and `nullity` are the verdict; `singular_values`, `null_space` and
`column_norms` are what it was read off, so a borderline case can be inspected
instead of argued about. `threshold` is `rtol × σ_max`, and `rtol` defaults to
`DEFAULT_RANK_RTOL` (`1e-8`) — a *relative* cut, which is why it does not need
retuning when the design's overall scale changes.

`check_identifiability=` takes three values, and there is no size heuristic on
purpose, because the cost is a dense Jacobian and a dense SVD, `n_data × n_par`
float64 words:

| value | when the rank test runs |
|---|---|
| `"once"` (default) | at the starting values, before the first sweep |
| `"each_sweep"` | at every parameter tuple the run visits |
| `False` | never |

`"each_sweep"` is cheap for a small model and strictly more informative: a
nonlinear model's identifiability is a property of *where you are*, so a check
only at the start misses a degeneracy that opens up near the parameters you
actually reach. Both exits check by default, and the point estimate is the
**more** dangerous one to skip — a chain at least has `r_hat` to scream with,
while a point estimate has no diagnostic at all and CG converges quietly onto an
arbitrary point of the null space.

### Two limitations that are real

:::{warning}
**`identifiability()` refuses a complex latent, so a complex sky-`alm` block must
pass `check_identifiability=False`.**

```text
Latent(s) ['alm'] are complex. The prediction is real, so the map from complex
coefficients to data is R-linear but not C-linear and its rank over C is not the
number you want — a block with n complex coefficients has 2n real degrees of
freedom, and they can be identified separately. Declare the real and imaginary
parts as separate latents, or ask about a different block with names=.
```

Two independent reasons point the same way for the 10⁶-coefficient sky block
that `linear.py` exists for: the rank test cannot *analyse* it, and it could not
*afford* to — a dense `n_data × n_par` SVD is exactly what a matrix-free solver
was built to avoid. So that plan runs with the guard off, and the run has no
cross-block check at all. Declaring the real and imaginary parts as separate
real latents buys the check back on a small block; on a big one, nothing does.
:::

:::{danger}
**A block taking a finite number of NUTS steps is Metropolis-within-Gibbs, not
an exact conditional draw.** A conjugate block's GCR draw *is* exact, so a plan
of conjugate blocks is an exact Gibbs sampler with nothing to tune. The moment
one block takes `steps` gradient steps, the scheme is still valid and still
targets the right stationary distribution — and is no longer exact, because the
inner step count now sets the mixing. `steps=` reads as a performance knob and
is a statistical assumption.

Measured on the plan's other fixture (`amp` conjugate, `centre` gradient, 120
sweeps, truth `centre = 0.350`, `init = 0.100`, `key=jax.random.key(1)` — the
one table on this page that does not use `key(0)`, because a stuck chain is a
sampling accident and `key(0)` happens to escape at `steps=2`; that it depends
on the seed is the point, not a caveat):

| `steps` | `rhat` | `converged` | posterior `centre` |
|---|---|---|---|
| 2 | 1.070 | **False** | 0.1000 ± 0.0000 |
| 10 | 0.983 | True | 0.3500 ± 0.0026 |
| 50 | 0.991 | True | 0.3502 ± 0.0026 |

At `steps=2` the chain **never left its initial value** — a posterior reported as
a point mass 0.25 away from the truth, at zero width. Note how nearly it passed:
`r_hat` came in at 1.070 against a 1.05 threshold, and it is `r_hat` of the joint
χ², which still moves because the *other* block is moving. Read
`diagnostics.rhat`, and look at the draws of the latent you care about.
:::

:::{note}
**"Group the correlated latents into ONE Block" is not always available.** The
non-convergence message above names that remedy, and for the bilinear model it
is measured on, `check_linearity` refuses it — correctly:

```text
Latents ['gain', 't_coeff'] are each declared linear=True, but the prediction is
not affine in them JOINTLY [...]. Each conditional of a bilinear model is affine
on its own, which is why this is not caught one latent at a time — and why these
two cannot share one linear block. Split them into separate blocks and alternate,
or re-parameterize so the joint map really is affine. identifiability(space,
pipeline, state) will tell you what the split costs before you choose it.
```

Grouping works when the joint map really is affine — several noise-wave
amplitudes, a sky block and an offset. When the coupling between the blocks is
what makes them correlated, the remedy is the other one the message names: more
sweeps, and `identifiability(space, pipeline, state, names=...)` to see what the
split is costing before you choose it.
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
block = linear_operator(
    space, twin, state, names=("sky_delta",))
mean, _ = wiener_solve(
    block, data, noise_std=0.02)
draw, _ = gcr_sample(
    block, data, noise_std=0.02,
    key=jax.random.key(0))
```
+++
Answers keyed by latent; too big for a gradient sampler.
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

Every engine so far evaluates a likelihood. Simulation-based inference does not:
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

## Tutorials

These pages say what each piece is. The tutorials walk one problem through end
to end, in the order you would actually do it, with the scripts' real output:

:::{list-table}
:header-rows: 1
:widths: 30 70

* - Tutorial
  - What it covers
* - [An exact posterior for a big linear block](tutorial-gcr.md)
  - 256 sky pixels, no chain. Checking the linearity claim, reading κ before
    choosing `tol`, `iterative_gls` for the covariance, `gcr_sample` for the
    draws — then widening the beam until the prior takes over, and watching
    every diagnostic say so.
* - [A gradient posterior, and how to tell it is wrong](tutorial-nuts.md)
  - Three nonlinear beam parameters. **It fails first**: `r_hat = 840`,
    `n_eff = 2`, nothing raised. Diagnosing that — two hypotheses, one right —
    is most of the page, and the fix is one line.
:::

---

## Run it

[`examples/inferring_anything.py`](https://github.com/RHINO-Experiment/rheplicant/blob/main/examples/inferring_anything.py)
does all three on one twin, from truth to recovery: a beam derived from two
scalars, a gain tied across two stages in log space, and a sky map declared
linear and solved by CG. The instrument description is written once at the top
and never edited.

```bash
.venv/bin/python examples/inferring_anything.py
```

The figures on this page come from
[`docs/_generate_inference_figures.py`](https://github.com/RHINO-Experiment/rheplicant/blob/main/docs/_generate_inference_figures.py),
which runs the same code rather than illustrating it.
