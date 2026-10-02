# Plans and engines

```{include} _bayesmith-note.md
```

One space can be stepped by several engines at once. A `SamplingPlan` declares
the partition and derives each block's engine from the latents' own
declarations.

- [A plan: one partition, two exits](#a-plan-one-partition-two-exits)
- [One space, every engine](#one-space-every-engine)
- [Inference without a likelihood](#inference-without-a-likelihood)
- [Tutorials](#tutorials)
- [Run it](#run-it)

---

## A plan: one partition, two exits

The [linear machinery](inference-linear.md) answers for one block.
`wiener_solve` is a linear-Gaussian block's posterior mean, `gcr_sample` an
exact draw from that same conditional, and its
[Gibbs tip](inference-linear.md#sampling-a-linear-block) sketches the loop you would
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

Unless another test file is named, the numbers below are measured on the
fixture in `tests/inference/test_plan.py`: a bilinear `gain × T_ant` model, a 6-element
gain against a `(3, 4)` time × frequency coefficient basis, 54 data points and
18 parameters. It is asymmetric in every dimension on purpose.

### How a block's engine is derived

`Latent(..., linear=True)` already says which machinery a latent can take, so
`Block` does not ask again. A block whose members are all declared linear is
solved by the [conjugate routines](inference-linear.md#linear-blocks); anything
else is stepped by gradient. In `estimate` a gradient block takes `steps` Adam
steps (this package's own descent, with a per-latent step of
`learning_rate × max|init|`) and then a Newton polish from
`bayesmith.optimize.certify`. In `sample` it takes `steps` NUTS steps.

A third engine, `log_conjugate`, is not derived, because there is no declaration
to derive it from. A block is [conjugate in log
space](inference-linear.md#log-space)
when the prediction is `exp` of an affine map, which no `Latent` field states;
it is either asked for with `engine="log_conjugate"` or found by
[`auto_blocks`](#deriving-the-partition) probing for it. Asking for it
on a latent declared `linear=True` is refused, because the two claims exclude
each other: `exp` of an affine function is affine only where it is constant.

The plan's `repr` reports what it derived:

```text
SamplingPlan(('gain'):conjugate, ('t_coeff'):conjugate)
SamplingPlan(('amp'):conjugate, ('centre'):gradient)
```

One case is ambiguous, a block mixing declared-linear and non-linear members,
and it is refused:

```text
Block('amp', 'centre') mixes declared-linear latents ['amp'] with non-linear
ones ['centre'], so which engine it takes cannot be derived. A conjugate solve
needs the whole block affine; a gradient step does not exploit the linear
members' structure at all, which for a high-dimensional linear block is the
difference between tractable and hopeless. Split them into separate blocks, or
say engine='gradient' to step the whole block by gradient deliberately.
```

`engine=` exists for that override and for asking for `log_conjugate`. `steps=` on a
conjugate block raises for the same reason: a Wiener solve has no inner steps,
so accepting the argument would silently ignore it.

### The partition check

Every latent of the space must be in exactly one block. A latent in no block
and a latent in two are both refused by name. The omission is refused with:

```text
This plan does not cover latent(s) ['t_coeff']: every latent of the space must
be in exactly one block. An omitted latent is silently frozen at its declared
init for the whole run — the sweep converges, the joint chi-squared settles, and
nothing anywhere reports that a parameter you declared was never inferred. Add
it to a block, or drop it from the space.
```

A latent in two blocks is refused too: the second update each sweep would be
solving a conditional the first had just invalidated, and every diagnostic would
report the second's answer as if the first had never run.

### Deriving the partition

Declaring the blocks stays available and stays the default for a model
you know. When you would rather not, `auto_blocks` reads the partition off the
model, and `SamplingPlan.automatic` is the one-liner over it. The grouping
loop is `bayesmith.dispatch.factor.first_fit`; this package supplies the
pairwise probes it reads.

```python
plan = SamplingPlan.automatic(space, twin, state, noise=noise)

# the same thing, with the blocks in reach to inspect or amend
plan = SamplingPlan(space, *auto_blocks(space, twin, state, noise=noise, steps=20))
```

`noise=` is optional and only the log-space half needs it (see below).

The rule is conjugate blocks for the latents declared `linear=True` and one
gradient block for everything else, with two refinements that the probe
supplies and you do not declare.

**Grouping by factor.** `Latent(..., linear=True)` is a claim about one latent:
the prediction is affine in it with the others held fixed. A conjugate block
over several of them claims something strictly stronger, that the prediction is
affine in them jointly, and on a multilinear model that is false while every
member's own declaration is true. Sweeping every linear latent into one block is
therefore wrong, and one block per latent is worse than it needs to be. The
correct partition is one block per factor, holding all of that factor's latents.
On `gain × (B_ant @ t_ant + B_nw @ t_nw + tone)`:

```text
SamplingPlan(('t_ant', 't_nw'):conjugate, ('gain'):conjugate)
```

**Log-linearity.** A latent the prediction is `exp`-affine in (a gain bound as
`Bind(..., fn=jnp.exp)`) has no `linear=True` to read, and by design there is no
`log_linear=True` either. The same probe that finds the grouping asks
[`check_log_linearity`](inference-linear.md#log-space)
and routes it to a `log_conjugate` block.

**The noise in discovery.** Discovering a log-conjugate block needs the noise.
A log-conjugate block is a claim about the likelihood: taking logs
simplifies a multiplicative noise and restates an additive one as a
different likelihood from the one declared, the first-order equivalence
holds only up to `FIRST_ORDER_MAX_FRACTIONAL`, and a declared
`RadiometerNoise.floor` makes sigma constant wherever it binds, so any floor
above zero has no log route. `auto_blocks` therefore takes `noise=` and applies
all three refusals (`log_route_refusal`) when it partitions. Without it, no
`log_conjugate` block is claimed and an `UncheckedLogRouteWarning`
names the latents that qualified on their prediction alone. A gradient
block is always a sound verdict, and the warning is issued because the
alternative is a partition promising a route `to_log_space` will refuse.

On the same model with the gain in log space, nothing is declared about the
gain and every block of the partition is closed-form:

```text
SamplingPlan(('t_ant', 't_nw'):conjugate, ('log_gain'):log_conjugate)
```

The two probes take separate `scales`. Feeding the linear default's `1e3` entry
to a log probe sends it through an exponential that overflows, the check
refuses, and a log-linear latent is filed as gradient: a misclassification that
costs a conjugate block and reports nothing. `log_scales=` is the argument,
`LOG_DEFAULT_SCALES` the default.

#### A worked example: all three engines from one model

For `d = exp(Ax) · (By + exp(Cz)) · (1 + f w)`, a matrix-exponential gain over
a summed sky under radiometer noise, declare `linear=True` on `y` alone (the
one latent the prediction is affine in) and let the probes decide the rest:

```text
SamplingPlan(('y'):conjugate, ('x'):log_conjugate, ('z'):gradient)
```

`y` is conjugate in the original space, solved each sweep with sigma frozen at
the current prediction, the GLS-flavoured choice described under
[When the covariance is not given](inference-linear.md#when-the-covariance-is-not-given).
`x` is discovered: nothing was declared for it, but `log(prediction) = Ax +
log(By + exp(Cz))` is affine in it given the others, so the probe routes it to
the log-space engine. `z` fails both probes and takes NUTS. One sweep runs all
three engines.

The example also shows two limits:

* **A correct partition is not a convergence proof.** This model is fully
  identified (`identifiability` reports nullity 0) with
  `weakest_identified = 5.4e-4`: one direction is constrained thousands of
  times more weakly than the rest, because `exp(Ax)`'s shape partly trades
  against the sky's. Single-latent Gibbs alternation random-walks along that
  valley: measured, two 400-sweep chains from different starts each reported
  tight widths while sitting sixteen of those widths apart, and the joint
  chi-squared `rhat` flagged nothing, because chi-squared is flat along a
  valley by construction. This page's monitoring section warns about the same
  blindness in its other form. On such a model, put the strongly coupled
  latents in one block by hand (`Block("y", "z", steps=...)`), and run a
  second chain from a different start before believing any of them.
* **Where this package's plans stop.** A `Latent`'s prior is a fixed
  distribution; a prior parameterised by another latent (a field `w1` whose
  statistics a hyperparameter sets) has no spelling in a `ParameterSpace`, and a
  plan cannot sweep what it cannot declare. Build that model as a bayesmith
  graph directly. Its documentation works the hierarchical variant of this same
  model (`docs/factor-partition-examples.md` in that repository): a
  hyperparameter is ejected from every exact block, because an exact block
  solving only against data would drop the `p(w1 | y)` factor.

**Why pairs suffice.** For latents already known to be affine on their own, every
diagonal block of the group's Hessian vanishes, so joint affinity is exactly the
claim that the off-diagonal ones do too, a question about pairs. Probing the
`C(n, 2)` pairs with `check_linearity` therefore decides a property of all
`2ⁿ − n − 1` subsets, and the verdicts colour a graph whose groups are the
blocks. The cost is that quadratic count of probes, each a linearization plus
one forward per entry in `scales`; nothing here switches on a size, for the same
reason `check_identifiability` does not.

Deriving the partition changes nothing downstream. The blocks are checked as
declared ones are, and each conjugate block's joint linearity is re-verified at
the first sweep. A derived partition is not a reason to believe a model. Two
coupled conjugate blocks are the configuration whose degenerate case converges
quietly onto an arbitrary point with every per-block guard green;
`identifiability` detects that, and both exits still run it by default.

### Two exits: estimate and sample

Internally one sweep takes `key=None | key`. The interface is two methods and no
mode flag: `key` is required on `sample` and absent from `estimate`, so a
request for draws without a key cannot be written. `n_sweeps` and `warmup`
belong to `sample`, `max_iter` and `tol` to `estimate`, because each pair means
nothing to the other method. Both return `result.diagnostics` and
`result.names`, so a caller can log or assert on a run without knowing which
exit produced it.

`PlanResult` is the protocol both exits satisfy: `.diagnostics` is a
`PlanDiagnostics`, `.names` are the latents. The exits differ in the answer they
return:

| exit | returns | the answer |
|---|---|---|
| `plan.estimate` | `Estimate` | `.values`: one array per latent |
| `plan.sample` | `Draws` | `.samples`: a chain per latent, plus `.n_draw`, `.mean`, `.std` (properties, not calls) |

Annotate against `PlanResult` when a function of yours should take either, and
against `Estimate` or `Draws` when it needs the values or the chain.

Both exits on the fixture above, at `HomoscedasticNoise(1.0)`:

```text
plan.estimate   sweeps 94   converged True   objective 5.80427e+07 -> 140.936
                chi2 1.16085e+08 -> 0.0062980   effective_tol 7.63e-06
                distance_bound 0.079 σ   contraction 0.124   certificate 19 products
                block residuals {('gain',): 7.61e-07, ('t_coeff',): 4.02e-07}
                max |T_ant - truth| = 0.0252 K, 0.079 posterior σ from the MAP

plan.sample     n_sweeps  60   kept  30   rhat 1.434   converged False
                n_sweeps 200   kept 100   rhat 0.99    converged True
                n_sweeps 600   kept 300   rhat 1.001   converged True
```

The estimate runs in float32, so its change tolerance is the float32 floor
rather than the default `tol = 1e-8`, and it converges at sweep 94 of its 100
(see the monitoring section below). Those 94 sweeps and the `rhat = 1.434` have
one cause: these two blocks are strongly correlated, so the alternation moves
slowly, and 30 kept draws are far from stationarity. Neither exit hides it:
`converged` is `False` on the short run and `PlanDiagnostics.rhat` says by how
much.

That diagnostic is computed by `split_rhat`, which is exported and applies to
any chain you hold. It cuts the trace in half, treats the halves as
two chains, and compares the variance between them against the variance within:

```python
import jax, jax.numpy as jnp
from rheplicant.inference import split_rhat

split_rhat(jax.random.normal(jax.random.key(0), (200,)))   # 1.0043  — iid
split_rhat(jnp.arange(200.0))                              # 2.6326  — pure drift
```

Two degenerate inputs have defined results: halves that are each constant at the
same value give `1.0` (nothing to mix), and halves each constant at different
values give `inf` (a chain that moved once and stopped).

A trace too short to halve is refused. The minimum is `MIN_DRAWS = 4`, two
halves of two, exported from `rheplicant.inference`. Below it the diagnostic is
undefined:

```text
split_rhat was given 3 value(s) and a split-r_hat needs at least 4 — two halves
of two. Below that the mixing diagnostic is not weak, it is undefined: halves of
one have no variance within them to divide by, so the answer came back as nan
rather than as this refusal — and a nan passes no threshold test in either
direction, which makes an undefined diagnostic read as whichever verdict the
caller tested for. SamplingPlan.sample refuses the same count on
(n_sweeps - warmup); this is that refusal, for the trace you brought yourself.
```

This is an exception and not a `nan` because `rhat <= rhat_max` is `False` for a
nan and so is `rhat > rhat_max`, so a threshold guard reads an undefined
diagnostic as whichever answer it tested for. `SamplingPlan.sample` enforces the
same minimum on `n_sweeps - warmup`, so a plan cannot reach this refusal.
`split_rhat` enforces it as well because it is public; the plan is its one
in-package caller.

### Convergence monitoring

A hand-rolled alternating solve over this same bilinear model, with a free
antenna temperature per `(time, frequency)` cell, lands hundreds to thousands of
kelvin from the truth while every per-block guard this package ships reports
green: `check_linearity` passes at every sweep, because each conditional is
affine; the per-block condition number is ≈1.47; and the CG residual reads
~`1e-7`. The fault is in the partition and not in any sweep, and no per-block
number can show it, because a residual and a condition number are both computed
from the block being solved.

The distance from the truth is set by where the solve started and not by the
degeneracy, because the answer is the initial offset carried along the null
direction and left there. Measured in
`tests/inference/test_degenerate_partition.py`:

| start | rms error | CG residual | κ |
|---|---|---|---|
| at the truth | 0.014 K | 1.3e-07 | 1.467 |
| 1 % off | 27.4 K | 1.1e-07 | 1.466 |
| 25 % off | 704 K | 1.0e-07 | 1.452 |
| 100 % off | 2962 K | 9.6e-07 | 1.432 |

The error spans five orders of magnitude and the guards read alike down the
column,
including between the row that is right and the rows that are wrong. There is no
threshold to place between them, so the remedy is a different measurement and
not a tighter tolerance. More sweeps are not a remedy either: five sweeps and
two hundred agree to four figures, because the solve reaches the solution
manifold at once and then does not move.

The monitored quantity is therefore a joint one at the current parameter tuple,
across sweeps. For `plan.estimate` it is the joint negative log posterior `f`,
`Conditioning.neg_log_posterior` (the objective every block update descends),
and what certifies a run is the **Newton decrement** of `f` at the point it
would return. The stop rule is `bayesmith.optimize.certify`'s: the schedule,
the decrement and the tightening of `solve_tol`. The plan supplies the joint
objective and the sweep, and the constants named below are re-exports.

* **the certificate**: `λ² = gᵀH⁻¹g` over every latent, for `g` and `H` the
  gradient and Hessian of `f`. Near its minimum `f` is quadratic,
  `f − f* = d² / 2` for `d` the Mahalanobis distance under the posterior
  precision, and `λ` is that distance. A run converges when `λ` is at most
  `sqrt(2 gap_tol)` (0.1 posterior σ by default), which does not grow with `N`
  and does not depend on which block is slow. Recorded as
  `PlanDiagnostics.distance_bound`, with the Hessian-vector products it cost in
  `.certificate_iterations` and the number of times it ran in
  `.certificate_attempts`.
* **the schedule**: the decrement costs a gradient and a solve, so it is asked
  only where a stop is plausible: the last two changes of `f` within
  `t × max(|f|, 1)`, `t = max(tol, 64 ε)` for `ε` the machine epsilon of the
  objective's dtype (`OBJECTIVE_FLOOR_EPS = 64`, recorded as
  `.effective_tol`), and this sweep's decrease either below what the
  arithmetic resolves or inside `gap_tol` after extrapolation at its estimated
  contraction `ρ = D[k] / D[k−1]` (recorded as `.contraction`). Those two
  tests schedule; neither certifies. After a candidate the decrement refuses,
  the next one waits twice as long, so a run that never certifies pays
  `O(log max_iter)` decrements.

**How the decrement is computed, and why an inexact solve is safe.** The
numerics are in `bayesmith.optimize.certify`. They know nothing about models:
the module takes a callable and a pytree, and the plan supplies the joint
objective. That is why the module moved: it was written in this package and
lifted into bayesmith byte for byte in bayesmith 0.10.0, with its 50 unit tests;
the acceptance test that drives a whole estimate to its MAP stayed here, because
it needs this package's models. `H` is never formed from the model: every
product is `jax.jvp` of `jax.grad` of `f`. Up to 1024 latents the decrement
takes `n` such products, assembles the Hessian, scales it by its diagonal and
solves by eigendecomposition; above that it runs conjugate gradients on the
products alone. Either way the solve is inexact, and the verdict is never `gᵀx`
as it stands. With `r = g − Hx` the true residual (recomputed with one more
product, because a recursive one drifts) and `μ` a lower bound on the smallest
eigenvalue of `H`,

```text
λ  ≤  (R + sqrt(R² + 4 gᵀx)) / 2,      R = |r| / sqrt(μ)
```

by Cauchy–Schwarz in the `H⁻¹` inner product, inflated by `1/sqrt(1 − ε κ)`
for the rounding of `g` and `H` themselves. The run certifies on that upper
bound and records it, so stopping the solve early can only make it refuse.

**Where the lower bound comes from, and why only a proof certifies.** Below 1024
latents it is measured: the scaled Hessian's own smallest eigenvalue, which also
decides positive definiteness, so an indefinite or numerically singular Hessian
is refused. Above it the Hessian is not formed, and there are two ways for a run
to have a floor:

| floor | where it comes from | certifies |
|---|---|---|
| `dense` | the formed Hessian's smallest eigenvalue (≤ 1024 latents) | yes |
| `supplied` | the prior precision, where this plan can prove `H = JᵀN⁻¹J + P` | yes |
| `probe` | a 32-step Lanczos probe's Ritz interval | **no** |
| `none` | no usable floor at all | no |

Which one a run used is recorded as `PlanDiagnostics.floor_source`. The
proof for the second is the model's: with the prediction affine in every
latent jointly, a sigma that does not depend on it, and a Normal prior on
every latent, `H = JᵀN⁻¹J + P ⪰ P ⪰ min(1/σ_prior²)` whatever the data. Each
condition is checked. Joint affinity is checked by the same `check_linearity` a
conjugate block passes, asked of all the latents at once, because two blocks
that are each affine are not jointly affine (`gain × sky` is the standing
case). The floor is the widest prior scale, the smallest precision.

The third class is a heuristic. `θ₀ − β|s₀|` bounds the distance from `θ₀` to
some eigenvalue of `H`, not to the smallest, so where the Krylov space never
reaches the bottom of the spectrum the probe's floor can sit above it: measured
over 480 random spectra, 5 in 240 did so in float32, the worst by a factor 7.4.
A probed floor therefore never certifies here. It is kept because it can soundly
refuse: non-positive curvature, or a bound already outside the threshold. A run
left with only a probe keeps sweeping and refuses at `max_iter` with "cannot
certify this estimate at this size and precision", naming the three things that
would make a proof: a single conjugate block with Normal priors (or any
partition of a jointly affine model), fewer latents than the dense limit, or
float64 where the precision is what blocks the dense path.

Both halves of that were found by construction: a spectrum
clustered at 1 with one eigenvalue at 1e-8 and a gradient whose component
along it sat just under the iteration's residual, where a condition number
read off the iteration's own Lanczos matrix said 1.2 against a true 1e8 and
turned a true 0.067 σ into a reported 0.050; and an indefinite Hessian at a
zero gradient, which the iterative path certified at distance zero while the
dense path refused it.

What it costs, measured on this machine in float64 against one sweep of the
same plan:

| model | latents | products | one decrement | in sweeps |
|---|---|---|---|---|
| power law, 4096 channels | 2 | 3 | 0.2 ms | 0.13 |
| two collinear blocks, `N = 1e6` | 2 | 3 | 0.8 ms | 1.4 |
| one conjugate block of 256 coefficients | 256 | 257 | 28 ms | 10 |
| one conjugate block of 1024 coefficients | 1024 | 1025 | 322 ms | 16 |
| one conjugate block of 2000 coefficients | 2000 | 2 | 7.5 ms | 0.17 |

The last row shows the prior floor's effect: above 1024 latents the
Hessian is not formed, and a floor the model proves lets the iteration stop
at the second product instead of probing the spectrum.

When it has not converged, the refusal names what failed and what the per-block
numbers were doing at the time:

```text
SamplingPlan.estimate did not converge: after 4 sweeps the JOINT negative log
posterior is still changing by -3.64e+05 per sweep (objective = 1.15674e+06,
chi2 = 2.31319e+06). A verdict needs two consecutive sweep-to-sweep changes each
within 7.63e-06 of it, relative (tol=1e-08, floored at 64 machine epsilons of
the objective's dtype), then the Newton decrement within 0.1 posterior sigma.
The decrease contracts by 0.7323 per sweep, leaving a gap of about 1.36e+06
nats. Note what this does NOT show up in: every conjugate block's own CG
residual is 5.56e-07 or better, because a per-block residual is computed from
the block and converges at every sweep of an alternation that is going nowhere.
In float32 the objective's own rounding can keep it moving, and a gradient
below that rounding cannot be descended at all: run in float64
(JAX_ENABLE_X64=1).
```

A run that did reach a candidate is refused in the decrement's own terms
instead: the distance it measured, what the solve did, and the `solve_tol` the
closed-form blocks ended at.

The changes counted are between sweep outputs, never from the starting values
(the first change only seeds the contraction), so the earliest verdict is at
sweep 3 (`EARLIEST_CONVERGED_SWEEP`) whatever `min_sweeps` says below that, and
`max_iter` of 1 or 2 with a `tol` always refuses. A config document that asks
for that is refused before it runs, by pre-flight check A25.

The joint χ² is still recorded (`PlanDiagnostics.chi2`) and is no longer the
test. It was until 0.9.0, as a decrease: any sweep that did not lower χ² counted
as converged. With a prior the MAP is not the χ² minimum, so a sweep moving
towards the MAP raises χ², and the rule read that rise as convergence: measured
1 to 15 posterior σ from the exact MAP, including a plan of two conjugate blocks
and no gradient block. An exact conditional update cannot raise `f` beyond the
arithmetic's noise, so a rise beyond its resolution is never convergence.

**Why a curvature check, and not the changes alone.** A tolerance relative to
`|f|` certifies a distance of about `sqrt(2 t |f| / (1 − ρ))`, and `|f|` is
about `N / 2` for `N` data. Measured: the change test alone passing two
collinear conjugate blocks 0.6 posterior σ from the MAP at `N = 1e6` in float64
and 16.5 σ in float32. Extrapolating the decreases at their contraction removes
the `|f|` but not the second failure: `ρ` read from the decreases is the fastest
mode still moving, so a slow mode hidden under a fast one passes. Measured: two
correlated pairs, one started 30 σ off and one 1 σ off, certified 0.5 to 1.0 σ
away, and 4618 random dense precisions giving false certificates up to 1.33 σ.
The decrement is a distance and has neither failure. Two collinear templates
started 20 σ off, re-measured with it
(`tests/inference/test_estimate_reaches_map.py`,
`test_estimate_large_n_float32.py`):

| N | r | float64: sweeps, σ from the MAP | float32 |
|---|---|---|---|
| 1e4 | 0.864 | 28, 0.006 | 21, 0.050 |
| 1e4 | 0.993 | 431, 0.059 | 410, 0.078 |
| 1e5 | 0.961 | 71, 0.069 | 68, 0.088 |
| 1e6 | 0.866 | 20, 0.073 | 19, 0.097 |
| 1e6 | 0.993 | 401, 0.098 | 523, 0.019 |

`tol` keeps its meaning and default, so code that passes a `tol` still asks for
what it asked for; what changed is that the change test no longer suffices, and a
run that used to report converged may now take more sweeps or refuse.

**The resolution.** The change is summed term by term, so the constant parts of
`f` (every prior's normalizer) and the bulk of the χ² sum cancel exactly instead
of costing `ε |f|`; what remains is resolved to about `4 ε sqrt(Σ term²)`
(`RESOLUTION_EPS`). A decrease below that is treated as no decrease: the sweep
becomes a candidate and the decrement decides. Before the decrement, the
resolution was the certificate's own floor, and a float32 run whose decreases
fell below it was refused however good its answer: that floor was measured 50 to
1500 times coarser than the distance it stood for, and the float32 column above
is the same grid that used to refuse on it.

**When the inner solves are the obstacle.** A conjugate block solved to
`solve_tol` has a fixed point that is not the MAP, and on correlated blocks the
offset is not small: 0.11 posterior σ at the default `1e-6` on the bilinear
fixture at noise 0.30. The decrement measures that offset as a distance
and refuses, so when a sweep shows inexactness (the objective rising beyond
its resolution, or a candidate refused) the closed-form blocks' tolerance is
divided by 100, down to a floor of `1e-12` in float64 and two machine epsilons
in float32, and the value the run ended at is recorded as
`PlanDiagnostics.solve_tol`. On that fixture the run converges at `1e-8`,
0.002 σ from the MAP in float64 and 0.072 σ in float32; without the tightening
both exhaust 3000 sweeps and refuse. A model that certifies at the caller's
`solve_tol` is never tightened.

**What is left.** The decrement is a statement about the quadratic model of `f`
at the returned point: where the curvature changes over a posterior σ it is
local. Above 1024 latents a model that is not jointly affine (a bilinear
gain, a log-space block, anything a gradient block is there for) has no floor
to prove and so no certificate; it refuses, and `tol=None` is the way
to get the answer without a claim. A direction of negative curvature that
neither the probe nor the Krylov space meets is not seen. In float32 a
model whose posterior σ is small against its latents' magnitudes has a gradient
that is mostly rounding, and no solve recovers it; those runs refuse, naming
float64. Grouping the correlated latents into one `Block` removes the slowness
itself.

The change test's float32 floor exists because float32 cannot resolve
`tol = 1e-8`: its epsilon is 1.19e-7, and conjugate solves at `solve_tol = 1e-6`
move the objective at its plateau by tens of ulps a sweep, the inner-solver
floor [`iterative_gls`](inference-linear.md#when-the-covariance-is-not-given)
documents for its own `reweight_tol`. On the fixture above, with the trace
replayed against the float64 MAP, a floor of 4 ε never stops, 64 ε stops at
sweep 94 and 0.079 posterior σ, and 256 ε at sweep 89 and 0.13 σ. In float64 the
floor is 1.4e-14 and `tol` governs; the same fixture stops at sweep 133,
0.003 σ, its conjugate solves tightened to `solve_tol = 1e-8` on the way.

:::{important}
**A joint quantity catches a slow partition, not a degenerate one.** Running the
free-per-cell parameterization with `check_identifiability=False` and `tol=None`
for 40 sweeps:

```text
worst per-block residual  5.55e-07
joint chi2                8.88e-06
max |T_ant - truth|        1044.69 K
max |gain  - truth|            0.308
```

The joint χ² is also tiny, because a degenerate model fits the data exactly, at
an arbitrary point of the null space. The joint objective and its certificate
catch blocks that are identified but correlated; only the rank test below sees
the other failure.
:::

### The identifiability check, and its cadence

`identifiability()` looks across blocks at the joint Jacobian and refuses a
model with a null space before a sweep runs, naming the degenerate directions as
combinations of latents. A count alone ("you have 6 blind directions") says
that there is a problem and not where:

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

Each blind direction is half gain and half `t_ant`, which is the bilinear
degeneracy `gain × T_ant = (c·gain) × (T_ant/c)`. The repair is a
re-parameterization (the `(3, 4)` basis), not a tighter tolerance.

Called directly, `identifiability()` returns an `IdentifiabilityReport`
and does not raise, so the question can be asked before committing to a
partition. On a healthy 64×4 design:

```text
names ('coeffs',)   n_par 4   n_data 64
rank  4             nullity 0
rtol  1e-08         threshold 1.2210e-08
singular_values [1.22098371 1.01958727 0.90795122 0.80328398]
```

`rank` and `nullity` are the verdict; `singular_values`, `null_space` and
`column_norms` are what it was read off, so a borderline case can be
inspected. `threshold` is `rtol × σ_max`, and `rtol` defaults to
`DEFAULT_RANK_RTOL` (`1e-8`). The cut is relative, so it does not need
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
nonlinear model's identifiability depends on the parameter values, so a check
only at the start misses a degeneracy that opens up near the parameters the run
reaches. Both exits check by default. Skipping the check is more dangerous on
the point estimate: a chain still has `r_hat`, while a point estimate has no
diagnostic and CG converges quietly onto an arbitrary point of the null space.

### Two limitations

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

Two independent reasons apply to the 10⁶-coefficient sky block that `linear.py`
exists for: the rank test cannot analyse it, and it could not afford to, because
a dense `n_data × n_par` SVD is what a matrix-free solver was built to avoid.
That plan therefore runs with the guard off, and the run has no cross-block
check. Declaring the real and imaginary parts as separate real latents restores
the check on a small block; on a big one, nothing does.
:::

:::{danger}
**A block taking a finite number of NUTS steps is Metropolis-within-Gibbs, not
an exact conditional draw.** A conjugate block's GCR draw is exact, so a plan
of conjugate blocks is an exact Gibbs sampler with nothing to tune. Once
one block takes `steps` gradient steps, the scheme is still valid and still
targets the right stationary distribution, but it is no longer exact, because
the inner step count now sets the mixing. `steps=` reads as a performance knob
and is a statistical assumption.

Measured on the plan's other fixture (`amp` conjugate, `centre` gradient, 120
sweeps, truth `centre = 0.350`, `init = 0.100`, `key=jax.random.key(1)`; this
is the one table on this page that does not use `key(0)`, because a stuck chain
is a sampling accident that depends on the seed, and `key(0)` happens to escape
at `steps=2`):

| `steps` | `rhat` | `converged` | posterior `centre` |
|---|---|---|---|
| 2 | 1.070 | **False** | 0.1000 ± 0.0000 |
| 10 | 0.983 | True | 0.3500 ± 0.0026 |
| 50 | 0.991 | True | 0.3502 ± 0.0026 |

At `steps=2` the chain never left its initial value: a posterior reported as
a point mass 0.25 away from the truth, at zero width. It nearly passed:
`r_hat` came in at 1.070 against a 1.05 threshold, and it is `r_hat` of the joint
χ², which still moves because the other block is moving. Read
`diagnostics.rhat`, and look at the draws of the latent you care about.
:::

:::{note}
**"Group the correlated latents into ONE Block" is not always available.** The
non-convergence message names that remedy, and for the bilinear model it
is measured on, `check_linearity` refuses it, correctly:

```text
Latents ['gain', 't_coeff'] are each declared linear=True, but the prediction is
not affine in them JOINTLY [...]. Each conditional of a bilinear model is affine
on its own, which is why this is not caught one latent at a time — and why these
two cannot share one linear block. Split them into separate blocks and alternate,
or re-parameterize so the joint map really is affine. identifiability(space,
pipeline, state) will tell you what the split costs before you choose it.
```

Grouping works when the joint map is affine: several noise-wave amplitudes, a
sky block and an offset. When the coupling between the blocks is what makes them
correlated, the remedy is the other one the message names: more sweeps, and
`identifiability(space, pipeline, state, names=...)` to see what the split is
costing before you choose it.
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
at the observed data.

```python
thetas, bank = simulate_pairs(twin, state, space, noise=noise,
                              key=jax.random.key(0), n_simulations=32_768)
q = NeuralPosterior.create(thetas, bank, key=jax.random.key(1))
q, history = train_posterior(q, thetas, bank, key=jax.random.key(2))

draws = q.sample(observed, key=jax.random.key(3), n_samples=4000)
```

Nothing there needs the noise to be Gaussian, the model to be differentiable,
or a normalization to be tractable. It needs only a simulator, which the twin
already is. The cost is **amortized**: a second observation is a forward pass,
not another chain.

The density is a conditional Gaussian mixture (an MLP → weights, means,
scales). A normalizing flow is more expressive; a mixture is exact for a
Gaussian posterior at one component and keeps the failure modes legible. The
density and its training are `bayesmith.amortize`, and `history` is its
`TrainingHistory`. `simulate_pairs` is this package's, because the simulator
is the twin.

:::{danger}
**An approximate posterior does not report its own error.** A badly-fitted $q$
returns a smooth, confident, correctly-centred, incorrect distribution and no
warning. There are two failure modes, and they push in opposite directions.
Measured on the package's own linear-Gaussian test problem, where the exact
answer is available from `gcr_sample`:

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
over-fitting, which makes $q$ too narrow, the failure that looks like a
better answer.

`train_posterior` holds out a validation split by default and returns the best
validation step, not the last, because the training loss falls monotonically
through the point where the fit stops being a posterior. Prefer few components.
Validate on a problem you can solve exactly before trusting one you cannot.
`tests/inference/test_npe.py` does that, and it is why NUTS was wired to the
noise model first.
:::

---

## Tutorials

These pages say what each piece is. Each tutorial walks one problem through end
to end, in the order you would do it, with its script's output:

:::{list-table}
:header-rows: 1
:widths: 30 70

* - Tutorial
  - What it covers
* - [An exact posterior for a big linear block](tutorial-gcr.md)
  - 256 sky pixels, no chain. Checking the linearity claim, reading κ before
    choosing `tol`, `iterative_gls` for the covariance, `gcr_sample` for the
    draws, then widening the beam until the prior takes over and reading that
    off the diagnostics.
* - [A gradient posterior, and how to tell it is wrong](tutorial-nuts.md)
  - Three nonlinear beam parameters. The first run fails: `r_hat = 846`,
    `n_eff = 2`, nothing raised. Diagnosing that (two hypotheses, one right)
    is most of the page, and the fix is one line.
:::

---

## Run it

[`examples/three_ways_to_a_posterior.py`](https://github.com/RHINO-Experiment/rheplicant/blob/main/examples/three_ways_to_a_posterior.py)
compares the routes on one model, and
[`examples/inferring_anything.py`](https://github.com/RHINO-Experiment/rheplicant/blob/main/examples/inferring_anything.py)
exercises the three binding shapes of [parameter spaces](inference-spaces.md)
on one twin, from truth to recovery.

```bash
.venv/bin/python examples/three_ways_to_a_posterior.py
```

[`examples/gibbs_plan.py`](https://github.com/RHINO-Experiment/rheplicant/blob/main/examples/gibbs_plan.py)
builds one `SamplingPlan` over six latents, with conjugate blocks and a
gradient block. At 0.9.1 it does not run to its end;
[the examples page](examples.md) says where it stops.
