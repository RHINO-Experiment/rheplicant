# Config inference: the fit twin, the likelihood, and the exits

Plan 2B of the config layer: the same document that assembled the twin now
declares what is free, what the likelihood is, and what the fit compares
against — and `runs:`, the list of exits that use them.

```python
from rheplicant.config import run_document

results = run_document(document)   # {run name: RunResult}, declaration order
```

The [configuration CLI](config-cli.md) uses the same orchestration with a
parse-before-execute guarantee: the base and every variant complete text,
axes, built, run-parser, and post-flight validation before the first base run
executes. Kind-specific options are parsed exactly once. A check recorded as
deferred is completed at its declared built/post-flight boundary, never
rediscovered by an executor after earlier runs have already changed state.

## The fit twin

`inference.twin` repairs the model twin rather than redeclaring it:
`without:` drops stochastic stages (the supported repair for the
stochastic-stage refusal — it is `Assembly.without` by name), and `replace:`
swaps one node's operator for a declared node spec, spelled exactly like the
node it replaces. A binding into a node `replace:` just rebuilt is refused —
it would overwrite the replacement at bind time (check B8). A `kind: pipeline`
model is rebuilt, not repaired: declare the fit pipeline as its own variant.

## Latents

`inference.parameters` is a mapping of latent name → spec. `init:` is
required — it is the authority on the latent's shape and dtype. `prior:`
names one family (`normal`, `uniform`, `log_normal`, or `python:`), and a
scalar family **broadcasts to the declared init's shape**: `Latent` refuses
`prior.shape() != init.shape`, and four levels of braces for
`dist.Normal(jnp.zeros(8), 400.0)` was v0's mistake, not the user's.
`linear: true` is a claim the twin must honour. A latent reaches the twin
through `into:`/`transform:`/`fan:` on its own entry, or through an entry of
`inference.bindings:` (several latents into several targets) — the two
spellings are mutually exclusive per latent. `joint_prior:
{jeffreys: {over: [...]}}` is the one joint-prior type the package knows.

## Transforms

The registry is closed for the same reason the derivation registry is: every
entry names a callable this package already ships, so a transform is a
reference rather than arithmetic.

| Spelling | Meaning |
|---|---|
| `identity`, `exp`, `log`, `sum` | what they say |
| `split_rows` | rows of one latent to several targets (`fan: distribute`) |
| `unit_mean_bandpass` | the receiver's bandpass identifiability convention |
| `affine: {scale, offset}` | `scale * v + offset` |
| `matmul: {design}` | `design @ coefficients` |
| `log_link_basis: {kind, n_basis, axis}` | `exp(B @ c)` on the run's own grid |
| `basis_expand: {basis: {ref: ...}}` | a declared `resources.bases` expansion |
| `beam_analysis: {nside, lmax, iterations}` | beam maps → true alms, per frequency |
| `python: mod:fn` | escape hatch; must declare its own `fan:` |

A declared `fan:` that contradicts the transform's own is refused (check
A38). `beam_analysis` is the one transform that exists to fix a wrong answer
rather than to save typing: `DriftScanProjector`'s only traced field is
`beam_alms`, so a latent bound straight into it moves `d/d(alm)` — a
different quantity, in a different basis, from the `d/d(beam map)` gradient
you meant, and both are finite and correctly shaped. Declaring
`transform: {beam_analysis: {nside: ..., lmax: ...}}` puts the map-space
latent on one side and the analysis on the other. It analyses in the
true-alm (healpy) convention, which is the one a beam needs; the quadrature
convention differs by `npix/4pi` and would rescale the beam silently.

Two refusals guard it, in this order and **neither of them an upper edge on
`lmax`**: `nside:` must be 2 or more (at `nside: 1` no `lmax` works at all),
and then `lmax:` must be at least `2 * nside - 1`. Above that floor the sweep
found no ceiling — from `nside: 2` to `nside: 16`, powers of two and not, the
lower edge is exactly `2 * nside - 1` every time, and an `lmax:` far above it
is legal and is not refused.

## Noise

Two different things in this package are called "noise model";
`inference.noise` builds only the likelihood's, never the graph node
(`model.noise`). Four kinds:

- `kind: none` — no likelihood is built, so the exits that never weigh a
  residual still run: `forward`, `optimize`, `identifiability`,
  `score_directions`, `mmodes`, and `gradient` on any objective but `chi2`.
  The exits that do weigh one are refused naming this kind:
  `fisher`, `plan.estimate`, `plan.sample`, `conjugate.wiener`,
  `conjugate.gcr`, `conjugate.gls`, `condition`, `nuts`, `npe`, and
  `gradient` under `objective: chi2`. `predict` is reachable under neither
  list — every run it can reuse needs a noise model, so the refusal arrives
  from the reused run rather than from the `predict`.
- `kind: homoscedastic` — `sigma:` is a value node; a 1-D sigma must declare
  `axis: time` or `axis: freq`, because it reads equally well along either
  axis of `(n_time, n_freq)` data (check A26).
- `kind: radiometer` — `channel_width:` and `integration_time:` default to
  `{from: observation}`, plus an optional `floor:`. `include_logdet:` is
  required and has no default: it is required exactly when the sigma depends
  on the prediction and refused otherwise — for a constant sigma it changes
  nothing (check A49). `false` is the documented GLS variant, a *different*
  estimator biased high by `(1 + f^2)`, and a lost declaration would come
  back `true` with no error.
- `kind: radiometer_frozen` — this layer's construct; it exists nowhere in
  `src/` on purpose. The sigma is DECIDED into an array — the one form the
  conjugate seam accepts — from `|observed|` (`source: observed`) or one
  forward evaluation at the declared inits (`source: prediction_at_init`),
  with the radiometer's fractional factor and the `floor:` applied.

`flags: {from: observation}` wraps a built model in the flag mask declared
at `observation.aux.flags` — the one place a flag mask lives.

## Observed

`inference.observed` has three forms: a simulation, a file, or a mapping of
named observations (a run's `on:` picks one; the primary is the entry named
`primary`, or the only one there is).

`from: simulation` predicts with the FULL twin by default (`twin: fit` opts
down), injects truth through `at: {latent: value}`, and adds scatter through
`realise:` — kinds `none`, `homoscedastic`, `radiometer` and `from_model`,
the last drawing with `inference.noise`'s own model so the generator and the
likelihood cannot disagree. Every drawing `realise:` names its seed as
`seed: {from: runtime.seeds.<name>}`; a name `runtime.seeds` does not
declare is derived from the root seed by a blake2s digest, and
`runtime.seed: null` is refused — with no root there is nothing to derive
from. `file:` reads an array that must match the run's grids EXACTLY —
broadcast-compatible is the dangerous case (check C11).

## Truth

With a simulated primary observation, what `observed.at` injected is
remembered as each latent's truth. A latent not named there gets its truth
read off the twin's own leaf when one `into:` path reaches one leaf through
identity; anything else — several leaves, a non-invertible transform, a
binding — records *why* the truth is omitted rather than guessing.
`inference.truth:` overrides per latent.

## Checks

`inference.checks` gates the three checks that cost something, each with
`mode: refuse | warn | report | skip`, and `mode: skip` carries its own
`reason:` (check A37) — three unrelated skips sharing one sentence was v0's
mistake. **Since Plan 3C this section is not a record: it decides what
`load_document` runs and what it charges you for.** The grammar is checked in
the pre-flight pass, before anything is built; the checks themselves run in
[the post-flight pass](config-validation.md#the-post-flight-pass-and-what-it-costs),
after `build_inference` and immediately before `load_document` returns.

| check | what it costs | default |
|---|---|---|
| `linearity` | `check_linearity` — a fixed number of forward passes per `linear: true` latent, and it does not grow with `n_par` | `refuse` |
| `identifiability` | one `jacfwd` through the forward model plus a dense `(n_data, n_par)` SVD | `off` |
| `prior_sensitivity` | `identifiability`'s work plus two Newton solves — 3.031 s cold against a 0.715 s build | `off` |

`off` is not a mode you can write: it is what a check that nobody asked for
is in, and it is deliberately not spelled `skip`, because A37 makes a written
skip carry a reason and a check nobody asked for has no author to write one.
`identifiability` alone also takes an `rtol:`. The whole cross-product of
`mode` with `report:` — including the `auto_skip` that reports a check you
asked for and could not have — is [one table on the validation
page](config-validation.md#a-gate-what-runs-what-a-failure-costs-and-what-is-recorded),
and `gates(...)` is the free, text-only function that answers what a given
document will actually pay for.

## The npe section

`inference.npe:` configures the amortized neural posterior, and it is a
section rather than a run's keys because `kind: npe` needs **four**
independent named seeds and a run carries one. Five subsections, each named
for the package call it feeds:

| Subsection | What it configures | Keys |
|---|---|---|
| `bank:` | `simulate_pairs` — the (parameters, data) training set | `n_simulations:`, `seed:` |
| `embed:` | the per-datum embedding `create` is handed | `ravel` (the default) or `{python: mod:fn}` |
| `create:` | `NeuralPosterior.create` — the estimator's architecture | `n_components:`, `width:`, `depth:`, `min_scale:`, `seed:` |
| `train:` | `train_posterior` — the optimisation | `n_steps:`, `batch_size:`, `learning_rate:`, `validation_fraction:`, `beta1:`, `beta2:`, `eps:`, `seed:` |
| `sample:` | `NeuralPosterior.sample` — the draw | `n_draws:`, `seed:` |

Two keys are renamed on the way in and nothing else is: `seed:` becomes the
package's `key=`, and `n_draws:` becomes `n_samples=`. Everything else is the
package's own parameter name, so a knob this page does not list is a knob the
package does not take. **No default is restated**: a key the document omits is
a key the package decides, which matters most at `create.n_components:`, whose
package default is 4 and whose shipped example passes 1 because 4 over-fits.

`embed:` resolves to a callable when the document is *read*, not when the run
executes — a bad `{python: ...}` is refused before the bank is simulated, which
is the expensive half. It takes the datum and nothing else
(`jax.vmap(embed)(data)`), and `args:`/`literal:` — the value grammar's way of
spelling a CALL — have no meaning here, because this key hands over a function
rather than the result of one.

`validation_fraction: 0.0` is legal and makes the trained history's validation
array empty; the product's `validation_loss` then has length 0, which is honest
and easy to mis-plot.

## Runs

`runs:` is required: a list of exits (one mapping is one run, named by its
kind). `name:` is required when there are several, and must be unique;
`variant:` builds that layer of the document; `on:` names the observed entry
(default `primary`); `reuse:` names an earlier run whose product this one
reads, and may only look backwards — runs execute in declaration order, so
naming a later run reads exactly like naming one that does not exist;
`expect: refuse` turns a demonstration refusal into a checkable assertion —
the refusal becomes the run's product, and success becomes the failure.
Kind-specific keys travel untouched, and each executor sweeps its own.
`runs:` is read from the BASE document — a variant patching it changes what
that variant accepts, never which runs execute.

### The five that fit

- `forward` — the twin on the state; no kind-specific keys.
- `fisher` — Fisher information and parameter covariance at the inits;
  `space:` (a bool) and `jitter:`.
- `optimize` — `optimizer: gradient | adam`, `learning_rate:` and
  `n_steps:` are all required (the shipped default sits five orders of
  magnitude from what a real fit has needed); `loss:` is `mse` or
  `{python: ...}`; it moves `inference.trainable` or
  `inference.parameters`, never both.
- `plan.estimate` — a blockwise point estimate; `blocks:` is required, and a
  seed is refused (the asymmetry is the package's own; check A29).
- `plan.sample` — blockwise posterior draws; `blocks:`, a named `seed:` and
  `n_sweeps:` are required; `warm_start: {kind: plan.estimate, blocks:,
  move:}` moves only the named inits.

### The conjugate family

Three exact linear-Gaussian solves over one block of latents. All three take
`names:` — always the grouped spelling, even for a block of one — plus
`prior_std:` and `prior_mean:`. Those two are **per member**: a mapping keyed
by latent name. A scalar is broadcast for a block of one and refused for a
block of several (check A51), because latent widths differ by orders of
magnitude and a wrongly-regularised block-diagonal prior returns a finite,
correctly-shaped answer with no residual signature. `prior_std:` becomes
required as soon as **any** member of the block lacks a prior — not only when
they all do. The refusal is the package's, and it names the members that are
short: *"needs a prior_std for `['d']` — the other members of this block have
one, which does not help"*. They do not help because with no prior at all the
normal operator can be singular, and CG returns a finite, arbitrary answer
rather than failing.

- `conjugate.wiener` — the posterior **mean**, and only the mean. A mean with
  no error bar is not a posterior, so `width:` is how you ask for one, and it
  is required: `width: fisher` buys the Gaussian width around that mean, and
  `width: none` says out loud that you only wanted the point. `width: draws`
  is refused by name — draws are `kind: conjugate.gcr`, declared as their own
  run over the same `names:`, so that a seed is required where it is used and
  refused where it is not.
- `conjugate.gcr` — constrained-realisation draws. `n_draws:` and a named
  `seed:` are the two that matter: one draw is a random number, not a
  posterior, and every posterior width in this project's own scripts comes
  from a stack of hundreds. `noise_from: gls` runs `iterative_gls` first and
  draws at the covariance it converges to, instead of at the declared sigma —
  the same solve `kind: conjugate.gls` reports, run inline. That route reads
  the noise as a *rule*, so `noise_from: gls` beside
  `noise.kind: radiometer_frozen` is refused: the sigma is already fixed and
  there is no fixed point left to find — and `noise_from: gls` is check A27's
  own answer for `noise.kind: radiometer`, so a user who takes both arrives
  exactly there (check A28). It reads no
  earlier run and has nothing to do with `reuse:`.
- `conjugate.gls` — iteratively reweighted least squares, and **the route for
  radiometer noise**. `conjugate.wiener` and `conjugate.gcr` need a decided
  sigma array; `noise.kind: radiometer` is a model whose sigma depends on the
  prediction, so pairing the two is refused naming both ways out — this kind,
  or `noise.kind: radiometer_frozen` (check A27). **The inverse pairing is
  refused too.** This kind reads `inference.noise` as a *rule* it iterates, so
  `noise.kind: radiometer_frozen` hands it an array that was decided before
  any run saw it, and a decided array is not a rule: declare
  `noise.kind: radiometer` to iterate, or run `conjugate.wiener`, which is
  what a decided sigma wants (check A28). A run that did not converge
  refuses to hand its covariance on without
  `acknowledge_unconverged_covariance: true`.

### The three you run before paying for a fit

- `identifiability` — can these latents be told apart by this data at all?
  It is the only diagnostic that sees across Gibbs blocks, and it is what
  answers design questions ("how many calibration loads do I need?") without
  the fit. `names:`, `at:` and `rtol:`. It forces float64 for its own
  duration, so a float32 document still gets a supported verdict; a model
  that pins its output dtype with an explicit cast is refused with "even with
  x64", and that remedy belongs to the model.
- `condition` — the conditioning of the same block, from `names:`,
  `prior_std:` and no data at all (`on:` therefore decides nothing here, and
  `prior_mean:` is refused: κ is set by the prior's width, not its centre).
  `iterations:` caps the power iteration, and `seed:` is **optional** — the
  one exit in the family where it is, because `condition_estimate`'s own
  `key` has a default where `gcr_sample`'s has none. Read κ before choosing a
  solver `tol:`: at κ = 1e7 the default 1e-6 bounds the relative error by 10,
  which is no digits.
- `score_directions` — which direction in data space each latent moves.
  `names:` and `at:`. The result comes back in the order you asked for, not
  in sorted order, and that is deliberate.

### The three that answer a question about the data

- `gradient` — one differentiation, no optimiser: `objective:` is `chi2`,
  `sum_squares`, `mean`, `mse` or `{python: ...}`, `of:` names what to
  differentiate with respect to, and `at:` says where. `chi2` is the first
  consumer of `inference.noise.include_logdet:`.
- `mmodes` — what a drift scan actually sees: a complex `(n_freq, lmax + 1)`
  array, from `projector: {ref: ...}` and `sky: {ref: ...}` — **those two
  keys and nothing else**. There is no `beam:` (the beam is the projector's
  own traced `beam_alms`, and the expansion has no argument to give it one)
  and no `coords:` (they come off the built state). Two things it needs from
  elsewhere in the document, and **both of those refusals are the package's
  own, deliberately not paraphrased here**: `observation.pointing.lst:`,
  which is what writes the LST grid the expansion reads; and, if the document
  materialises a `pointing:`, one that does not disagree with the projector's
  fixed az/el, which is rejected at 1e-3 deg even though nothing about the
  m-modes is wrong. Both arrive as a `StateValidationError` rather than a
  config-layer refusal, so seeing one from a document is not a bug in this
  layer. What this layer *does* refuse in its own voice is a projector with
  `normalize_beam: true`, quoting the code's own `measured ~18x off` — the
  `x` is ASCII there and `×` in schema §4.7.9, and the message follows the
  source rather than the schema.
- `predict` — push a fitted posterior back out to data space, choosing its
  route from what `reuse:` names — a `fisher` run, or a `plan.sample`, `nuts`
  or `npe` run, and nothing else (any other kind is refused by name, and so is
  spelling the link `from:`). A `fisher` run's covariance goes through
  `propagate_covariance`, the delta method, and comes back as a prediction
  standard deviation shaped like the data; nothing is drawn on that route, so
  `n_draw:` is refused there rather than ignored. The other three carry
  **samples**, which are pushed through the twin one by one, `n_draw:` thinning
  them from the tail, and those predictions are **noiseless** — they are the
  model's mean, not simulated data, so do not compare their scatter with an
  observation's. `n_draw:` above what the run kept is refused, and the refusal
  says why in that run's own terms: `plan.sample` discarded its warmup before
  returning, `nuts`' `get_samples()` returns the post-warmup draws alone
  (`num_samples` × `num_chains` is the whole chain), and `npe` drew exactly the
  `inference.npe.sample.n_draws:` it was asked for and has no warmup to
  recover — so on that last one the remedy is to raise `n_draws:` and draw
  more. On a multi-chain `nuts` product an `n_draw:` at or below `num_samples`
  reads **one chain**: `get_samples()` concatenates the chains in order, so the
  tail of the flat stack is the last chain's tail — ask for more than
  `num_samples` and you get the whole of the last chain plus the tail of the
  one before it. The samples route also needs
  numpyro, which the covariance route does not. A `predict` that declares a
  different `variant:` from the run it reuses is refused by name: pushing one
  build's product through another build's model mixes two builds, and the
  answer would come back finite, correctly shaped and about 1 % wrong — the
  package's structure and name checks catch only the mismatches that move the
  parameter layout.

### The two that sample a posterior

- `nuts` — numpyro's No-U-Turn sampler over the whole parameter space, through
  `to_numpyro_model`. `num_warmup:`, `num_samples:` and a named `seed:` are all
  required — the first two because numpyro's own `MCMC` gives them no defaults,
  the seed because a draw needs one (check A29). `init:` says where the chain
  starts: it defaults to `declared`, each latent's own `init:`, rather than to
  numpyro's uniform default — which is not a tuning knob, because on the
  package's own ring toy that difference is `r_hat = 1.002` against
  `r_hat = 840`. `init: ref` is the opt-in alternative and starts at each
  latent's `ref:` instead; a latent with no `ref:` is refused by name rather
  than falling back to its `init:`. `num_chains:`, `chain_method:`,
  `thinning:` and `progress_bar:` ride on `MCMC` and `target_accept_prob:` on
  the kernel. The product carries the latents **and not** the deterministic
  prediction site — `get_samples()` returns that too, and its per-sample shape
  is the whole data grid — beside `r_hat`, `n_eff` and a divergence count.
  Unlike the conjugate family it takes the noise **model**, not a decided sigma
  array: a prediction-dependent sigma is the point on this route, because the
  likelihood's own `-log σ` becomes part of the potential automatically.
- `npe` — the amortized neural posterior: simulate a bank, train a density
  estimator on it, and draw from the estimator conditioned on the real data.
  It takes no kind-specific keys at all; everything it needs is
  [`inference.npe:`](#the-npe-section). It needs a prior on **every** latent —
  `simulate_pairs` samples from them and consults `joint_prior:` not at all,
  which is where it differs from `nuts`, and the refusal names both ways out.
  It reads `inference.noise` as a *rule* — `simulate_pairs` draws the scatter
  for every pair it makes — so `noise.kind: radiometer_frozen`, which decides
  one array before any run sees it, is refused: declare `radiometer` or
  `homoscedastic`, either of which is a rule to draw from. There is no
  amortized-posterior exit that takes a decided array, so on this kind it is
  the sigma that has to change (check A28).

Either product can be reused by `predict`, which thins with `n_draw:` from the
tail — with the multi-chain caveat the `predict` bullet above spells out. It is
worth reading twice before quoting a width from a thinned multi-chain product:
what comes back is the end of one chain, which is what was asked for and is not
what "the last 50 draws of the posterior" usually means.

### Cross-run comparison and variant benchmarks

- `compare` — consumes exactly two earlier successful run products. The two
numeric pytrees must have the same structure, mapping keys, shapes, and dtype
classes. A tolerance miss is a successful, serializable comparison whose
`passed` field is false; missing runs or incompatible products are refusals.

```yaml
runs:
  - {name: base_prediction, kind: forward}
  - {name: repeated_prediction, kind: forward}
  - name: agreement
    kind: compare
    of: [base_prediction, repeated_prediction]
    metric: max_rel_diff       # max_rel_diff | rms | max_abs
    tolerance: 1.0e-10
```

- `benchmark` — evaluates named prepared layers rather than prior runs. `repeats`
defaults to 5, `warmup` to 1, and `metrics` to `[wall_time]`. Warmups are
blocked but excluded; measured JAX results are blocked before timing stops.
Raw samples plus minimum/median/mean are retained. `peak_memory` is labelled
`python_traced_bytes`: it is tracemalloc's Python allocation peak, not device
memory.

```yaml
runs:
  - name: layer_cost
    kind: benchmark
    variants: [base, unity_gain]
    repeats: 5
    warmup: 1
    metrics: [wall_time, peak_memory]
```

Both result kinds can be published with `outputs.write.compare: true` or
`outputs.write.benchmark: true`. All other scientific output selectors,
including posterior chains, estimates, recovery records, prediction bands and
reports, are documented on the [configuration CLI page](config-cli.md#scientific-products).

## A complete document

Adapted from the suite's own recovery test: `plan.estimate` recovers the
gain that `observed.at` injected, and `plan.sample` draws around it.

```yaml
schema_version: 1

runtime:
  seed: 20260806
  seeds: {sample: 11}

observation:
  meta: {telescope: RHINO}
  freq:
    grid:
      linspace: {start: 60.0, stop: 85.0, num: 8, endpoint: true}
      unit: MHz
  time:
    grid:
      arange: {start: 0.0, step: 2.0, num: 16}
      unit: s
  environment:
    temperature: {value: 280.0, unit: K}

model:
  global_signal:
    depth: {value: 0.5, unit: K}
    centre: {value: 75.0, unit: MHz}
    width: {value: 5.0, unit: MHz}
  gain:
    gain: {value: 1.1, unit: dimensionless}

inference:
  parameters:
    g:
      init: 1.0
      linear: true
      into: gain.gain
      prior: {normal: {loc: 1.0, scale: 10.0}}
  noise:
    kind: homoscedastic
    sigma: {value: 0.05, unit: K}
  observed:
    from: simulation
    at: {g: 1.5}
    realise:
      kind: homoscedastic
      sigma: {value: 0.05, unit: K}
      seed: {from: runtime.seeds.observed_noise}

runs:
  - name: recover
    kind: plan.estimate
    blocks: [{names: [g]}]
  - name: posterior
    kind: plan.sample
    blocks: [{names: [g]}]
    seed: {from: runtime.seeds.sample}
    n_sweeps: 10
    warmup: 4
    check_identifiability: false
```

One operational note: `plan.estimate` and `plan.sample` default to
`check_identifiability: "once"`, which runs a dense-Jacobian rank test per
fit. Identifiability forces x64 for its own duration and casts the latents,
so a float32 run still gets a supported verdict;
`check_identifiability: false` skips the cost. A model that PINS its output
dtype to float32 — an `astype` inside an operator — is refused by that check
with "even with x64 enabled"; the remedy is the model's, not the config's.

## A conjugate document

The same model as above, asked three questions instead of one: *can this gain
be identified at all*, then *what is it, with an error bar*, and finally
*what does that error bar look like back in data space*. All four runs are
cheap, and the first is the one worth reading first.

```yaml
schema_version: 1

runtime:
  seed: 20260806

observation:
  meta: {telescope: RHINO}
  freq:
    grid:
      linspace: {start: 60.0, stop: 85.0, num: 8, endpoint: true}
      unit: MHz
  time:
    grid:
      arange: {start: 0.0, step: 2.0, num: 16}
      unit: s
  environment:
    temperature: {value: 280.0, unit: K}

model:
  global_signal:
    depth: {value: 0.5, unit: K}
    centre: {value: 75.0, unit: MHz}
    width: {value: 5.0, unit: MHz}
  gain:
    gain: {value: 1.1, unit: dimensionless}

inference:
  parameters:
    g:
      init: 1.0
      linear: true
      into: gain.gain
      prior: {normal: {loc: 1.0, scale: 10.0}}
  noise:
    kind: homoscedastic
    sigma: {value: 0.05, unit: K}
  observed:
    from: simulation
    at: {g: 1.5}
    realise:
      kind: homoscedastic
      sigma: {value: 0.05, unit: K}
      seed: {from: runtime.seeds.observed_noise}

runs:
  - name: identifiable
    kind: identifiability
    names: [g]
  - name: mean
    kind: conjugate.wiener
    names: [g]
    prior_std: {g: 10.0}
    prior_mean: {g: 1.0}
    width: fisher
  - name: at_init
    kind: fisher
  - name: spread
    kind: predict
    reuse: at_init
```

`identifiable` comes back with `rank: 1`, `nullity: 0` against `n_data: 128`
— one latent, fully constrained by the 16 × 8 grid. `mean` comes back at
`g = 1.52`, which is the injected 1.5 pulled a little by the prior's
`loc: 1.0` and by the realised noise, with a Fisher width of 0.016. Read the
two together: the width is meaningful *because* the rank was full, and on a
document where it is not, `conjugate.wiener` still returns a finite number
with no sign that anything is wrong.

`at_init` and `spread` are the pair that shows `reuse:` working. `at_init` is
the Fisher covariance of `g` at its declared init; `spread` pushes that
covariance through the twin by the delta method and comes back as a `(16, 8)`
array of prediction standard deviations, one per sample, the largest of them
0.0079 K. Note where `at_init` sits in the list: `reuse:` may only look
backwards, so a `predict` declared above the run it names is refused exactly
as if that run did not exist — declaration order *is* execution order.

This document is executed by
`tests/config/test_config_surface.py::TestTheWorkedDocumentOnThePage`, which
reads the YAML out of this page rather than a copy of it.

## A posterior document

The same model again, asked the two questions the exits above exist for: *what
does the full posterior look like*, sampled exactly, and *what does an
amortized estimator say about it*. Each is pushed back out to data space by a
`predict` that reuses it.

**The sizes here are deliberately small so that this page's own document runs
inside the test suite**, and they are not recommendations: 200 warmup draws
land `r_hat` at 0.9965 on this one-latent document — with `n_eff` 43.6 out of
200 and no divergences — and would not be enough on a real one, and 50 training
steps over 64 simulations is an estimator that has not converged;
`train_posterior`'s own default is 3000 steps.

```yaml
schema_version: 1

runtime:
  seed: 20260806
  seeds: {chain: 3, bank: 1, create: 2, train: 4, draws: 5}

observation:
  meta: {telescope: RHINO}
  freq:
    grid:
      linspace: {start: 60.0, stop: 85.0, num: 8, endpoint: true}
      unit: MHz
  time:
    grid:
      arange: {start: 0.0, step: 2.0, num: 16}
      unit: s
  environment:
    temperature: {value: 280.0, unit: K}

model:
  global_signal:
    depth: {value: 0.5, unit: K}
    centre: {value: 75.0, unit: MHz}
    width: {value: 5.0, unit: MHz}
  gain:
    gain: {value: 1.1, unit: dimensionless}

inference:
  parameters:
    g:
      init: 1.0
      linear: true
      into: gain.gain
      prior: {normal: {loc: 1.0, scale: 10.0}}
  noise:
    kind: homoscedastic
    sigma: {value: 0.05, unit: K}
  observed:
    from: simulation
    at: {g: 1.5}
    realise:
      kind: homoscedastic
      sigma: {value: 0.05, unit: K}
      seed: {from: runtime.seeds.observed_noise}
  npe:
    bank:
      n_simulations: 64
      seed: {from: runtime.seeds.bank}
    create:
      n_components: 1
      width: 16
      depth: 2
      seed: {from: runtime.seeds.create}
    train:
      n_steps: 50
      batch_size: 32
      seed: {from: runtime.seeds.train}
    sample:
      n_draws: 12
      seed: {from: runtime.seeds.draws}

runs:
  - name: chain
    kind: nuts
    num_warmup: 200
    num_samples: 200
    seed: {from: runtime.seeds.chain}
  - name: chain_spread
    kind: predict
    reuse: chain
    n_draw: 50
  - name: amortized
    kind: npe
  - name: npe_spread
    kind: predict
    reuse: amortized
```

`chain` comes back with 200 draws of `g` at a mean of 1.5216 — against the
1.5225 the conjugate document above reaches by an exact solve, from a route
that shares no code with it below `inference.noise`. `chain_spread` pushes the
last 50 of those through the twin and comes back `(50, 16, 8)`: one prediction
per draw, and **noiseless** — the likelihood's own scatter is not added back,
so these are model means and not simulated data.

`amortized` trains the estimator and draws 12 times from it, and `npe_spread`
pushes those to `(12, 16, 8)`. At these sizes the npe draws are wide and are
not a posterior anybody should read: measured, they have a mean of 1.14 and a
standard deviation of 8.6 around an injected truth of 1.5. What the pair
demonstrates is the route, which is why the tests over this document pin
shapes and keys for it and pin a recovered number only for the chain.

This document is executed by
`tests/config/test_config_surface.py::TestThePosteriorDocumentOnThePage`, which
reads the YAML out of this page rather than a copy of it.
