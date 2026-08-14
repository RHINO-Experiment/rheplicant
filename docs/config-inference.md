# Config inference: the fit twin, the likelihood, and the exits

Plan 2B of the config layer: the same document that assembled the twin now
declares what is free, what the likelihood is, and what the fit compares
against — and `runs:`, the list of exits that use them.

```python
from rheplicant.config import run_document

results = run_document(document)   # {run name: RunResult}, declaration order
```

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
  `conjugate.gcr`, `conjugate.gls`, `condition`, and `gradient` under
  `objective: chi2`. `predict` is reachable under neither list — both of its
  routes reuse a run that needs a noise model, so the refusal arrives from
  the reused run rather than from the `predict`.
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

`inference.checks` records `identifiability`, `linearity` and
`prior_sensitivity`, each with `mode: refuse | warn | report | skip`, and
`mode: skip` carries its own `reason:` (check A37) — three unrelated skips
sharing one sentence was v0's mistake. The section is grammar plus record in
2B; its gating is Plan 3's validate.

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
  the same solve `kind: conjugate.gls` reports, run inline. It reads no
  earlier run and has nothing to do with `reuse:`.
- `conjugate.gls` — iteratively reweighted least squares, and **the route for
  radiometer noise**. `conjugate.wiener` and `conjugate.gcr` need a decided
  sigma array; `noise.kind: radiometer` is a model whose sigma depends on the
  prediction, so pairing the two is refused naming both ways out — this kind,
  or `noise.kind: radiometer_frozen` (check A27). A run that did not converge
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
  route from what `reuse:` names — a `fisher` run or a `plan.sample` run, and
  nothing else (any other kind is refused by name, and so is spelling the
  link `from:`). A `fisher` run's covariance goes through
  `propagate_covariance`, the delta method, and comes back as a prediction
  standard deviation shaped like the data; nothing is drawn on that route, so
  `n_draw:` is refused there rather than ignored. A `plan.sample` run's
  **samples** are pushed through the twin one by one, `n_draw:` thinning them
  from the tail, and those predictions are **noiseless** — they are the
  model's mean, not simulated data, so do not compare their scatter with an
  observation's. The samples route also needs numpyro, which the covariance
  route does not. A `predict` that declares a different `variant:` from the
  run it reuses is refused by name: pushing one build's product through
  another build's model mixes two builds, and the answer would come back
  finite, correctly shaped and about 1 % wrong — the package's structure and
  name checks catch only the mismatches that move the parameter layout.

`npe` and `inference.npe:` are refused by name and arrive with
Plan 2D; `compare` and `benchmark` arrive with Plan 4. Consuming any of these
products from `outputs:` is Plan 4's too — for now a product is what
`run_document` hands back.

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
