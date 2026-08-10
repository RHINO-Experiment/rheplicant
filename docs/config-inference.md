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
| `python: mod:fn` | escape hatch; must declare its own `fan:` |

A declared `fan:` that contradicts the transform's own is refused (check
A38). `beam_analysis` arrives with Plan 2C, alongside the driftscan exits
that consume it.

## Noise

Two different things in this package are called "noise model";
`inference.noise` builds only the likelihood's, never the graph node
(`model.noise`). Four kinds:

- `kind: none` — legal only for `forward` and `optimize`.
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
(default `primary`); `expect: refuse` turns a demonstration refusal into a
checkable assertion — the refusal becomes the run's product, and success
becomes the failure. Kind-specific keys travel untouched, and each executor
sweeps its own. `runs:` is read from the BASE document — a variant patching
it changes what that variant accepts, never which runs execute.

Five kinds run in 2B:

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

The 2C kinds — `nuts`, the `conjugate.*` family, `gradient`,
`identifiability`, `score_directions`, `condition`, `mmodes`, `predict`,
`npe` — are refused by name, as are `reuse:` and `inference.npe:`;
`compare` and `benchmark` arrive with Plan 4.

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
