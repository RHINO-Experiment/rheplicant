# Config validation: the pre-flight pass, and what it decides from text

Plans 3A, 3B and 3C of the config layer: everything a document can be refused
for **before** a file is read or a beam is analysed, and — since 3B and 3C —
three later slots for the things text alone cannot settle. On a toy nside-16
beam, `build_resources` is 1.397 s of `load_document`'s 1.536 s — 90.9 % — and
on a real CST directory the share goes up. A refusal that arrives after that
has been paid for is a refusal that cost what it was meant to save.

The last of the three slots is the exception that proves the rule, and it is
Plan 3C's: some checks cannot be decided without *running* the thing, so they
cost money by construction. What 3C buys is not that they are free — it is
that you can **see the price and decline to pay it**. That is [the gate](#a-gate-what-runs-what-a-failure-costs-and-what-is-recorded).

```python
from rheplicant.config import preflight

report = preflight(document)      # a Report; nothing is built, no file is read
report.raise_if_refused()         # ConfigError, the first refusal verbatim
```

## The pre-flight pass

`preflight` reads three things and no fourth: the document's own mapping,
`RADIO_GRAPH`, and operator classes resolved by name. It never constructs an
operator, never resolves a value node, never opens a file, and never runs the
model forward. That is not a policy — it is what makes the pass free.

**What "free" is, measured rather than claimed.** The pass runs once per
document and once more per declared **variant**, because a variant *is* a
different document and is checked as one. Cold — one call in a fresh process —
the worked document below is **3.5 ms**, and a document with forty
`plan.sample` runs and twenty variants is **58 ms**, against a budget of
0.15 s. It is linear in the number of variants from there, so a document with
several dozen of them costs several dozen passes; if you write one, time it.
(These numbers moved at 2026-08-19, when the audit-evidence pipeline gained
its hardened enumeration: the budget was re-measured against that contract,
not the contract weakened to fit the old one.)
Nothing in the pass grows with the size of a *beam*, which is the comparison
that matters: `build_resources` is 1.397 s of `load_document`'s 1.536 s on a
toy nside-16 beam, and worse on a real CST directory.

It **collects**. `load_document`'s own section sweep raises on the first
problem, which makes a user with four errors pay four round trips; `preflight`
runs every registered check and hands back all of them. The exception is
structural: unknown section names, `schema_version`, the required sections and
four whole sections this layer does not read — `outputs:`, `defaults:` and
`plugins:`, which arrive with Plan 4, and `campaign:`, which is reserved with
capability 4 — are refused immediately, because every other check assumes the
document's top level is well formed.

## What a Report carries

A `Finding` is four fields: `check` (the schema §6 id, or `""`), `severity`,
`where` — **a path into your document**, the line to edit, never a path into
the package — and `message`, one sentence carrying the fix.

| method | what it gives |
|---|---|
| `report.refusals()` | the findings that stop the document running |
| `report.warnings()` | the findings that do not |
| `report.of(severity)` | the findings at one severity — `"refuse"`, `"warn"` or `"report"`, the third being a finding that is neither fatal nor advice |
| `report.checks()` | the set of ids that fired |
| `report.raise_if_refused()` | `ConfigError` with the first refusal verbatim, and a tail naming how many others there are |
| `report.emit_warnings()` | each warning through `warnings.warn(..., ConfigWarning)` |

`ConfigWarning` is a `UserWarning`, so `warnings.filterwarnings("error",
category=ConfigWarning)` turns a document this package *will* run but probably
should not into a failure of your own choosing.

## What it decides, and from what

Three sources, and the third is the interesting one.

- **The document's text.** Section and key names, the `kind:` words, the
  numbers a document writes literally. `runs[].kind` beside
  `inference.noise.kind` is a two-word table that decides checks A27 and A28,
  which until now fired inside an executor.
- **`RADIO_GRAPH`.** Node ids, node kinds, the edge list and `must_precede`.
  Static and free.
- **Operator classes, resolved by name and introspected.** `model.noise:
  {type: NoiseOperator}` names a class; that class declares `"key"` in
  `requires`, which is this package's contract for *draws its own randomness*.
  So check A30 asks the operator rather than consulting a list of node names —
  it catches any operator that declares it, including one that has not shipped
  yet. Resolving a class is an attribute lookup. **Constructing** one is not
  in scope, so a node spelled `python: mod:factory` is left to the build.

## What it cannot decide

Anything that needs a value. `observation.freq.grid` may be `{value: [...]}`,
`{file: ...}` or `{ref: ...}`, and only the first is text; a check that wants
the grid's length wants a resolved value node. The same is true of every shape
comparison, every unit identity over a derived scalar, and every rank or
Jacobian question. This pass is what stops you paying for a beam analysis to
reach a refusal that two words already ruled out.

`inference.twin` is the clearest boundary. Whether a *repaired* twin still
contains a stochastic stage is decided here from `model:` plus
`twin.without:`/`twin.replace:`, all three of which are text. Whether that
twin's Jacobian has a null direction is not.

## The three later slots, and what each one buys

Plan 3B added two more passes inside `load_document` and Plan 3C added a
third. Each is decided by the inputs it reads, not by where the schema files
it, and each has its own cost bound.

| slot | when it runs | what it may read on top of the text | bound |
|---|---|---|---|
| **the axes pass** | after `runtime:` and `observation:` are built, **before `resources:`** | `RuntimeFacts`, `ObservationBuild`, `ResolutionContext` — the resolved time and frequency grids | under 0.01 s |
| **the built pass** | when everything is built and `load_document` is ready to return | `BuiltResources`, the twin, the `State`, the `InferenceBuild` | one `jax.eval_shape` per check, no forward pass |
| **the post-flight pass** | after `build_inference`, immediately before `load_document` returns | everything the built pass reads, **plus the resolved gates** | a forward pass, a Jacobian or a Newton solve — **and it is the one you can decline** |

The axes pass is the one that still saves the beam: it sits one line above
`build_resources`, so a run whose time axis the stored dtype cannot carry, or
whose calibration tone falls outside its own observed band, is refused before
the 90.9 % is spent. It costs about a hundredth of what it saves.

**The built pass runs after the money is spent, and this page says so rather
than implying otherwise.** Schema §6's preamble — "all run before any file is
read that is not needed to decide them, and before any beam is analysed" —
is **false for the twin-shaped rows**, and `config/paths.py`'s module
docstring already states the honest version. A check that compares the fit
twin's switch positions against the declared switch order, or two projectors'
analysed beams, needs the objects to exist; there is nothing textual left to
ask. Those findings arrive before the *fit* — which is the expensive thing
they are protecting — and after the build, which they cannot be moved in
front of without a shape-only twin built from `ShapeDtypeStruct` stand-ins.
That is a mechanism of its own and no plan has taken it.

So the ordering guarantee is per slot, and only the first slot is free:

- text pass — nothing is read, nothing is built;
- axes pass — the grids exist, no file behind them has been opened for a
  beam;
- built pass — everything exists; the saving is the fit, not the build;
- post-flight pass — everything exists *and* the check runs the twin; the
  saving is the fit, and it is the only slot whose price the document can
  refuse to pay.

## The post-flight pass, and what it costs

Three checks cannot be decided by text, by a grid or by a shape trace, because
the question each one asks is *what does this model actually do*. They ship in
a fourth pass, run after `build_inference` and immediately before
`load_document` returns.

| check | what it runs | id | default |
|---|---|---|---|
| `linearity` | `check_linearity` — `len(scales) + 1` forward passes **per `linear: true` latent**, at scales `(1e-3, 1, 1e3)` | `C12` | `refuse` |
| `identifiability` | one `jacfwd` through the forward model plus a dense `(n_data, n_par)` SVD | `C13` | `off` |
| `prior_sensitivity` | `identifiability`'s work plus two Newton solves | `C19` | `off` |

`linearity` is the only one on by default; `off` is what the other two are in
until a document asks, and it is a state no document can write.

**A document that lights `model.adc` and declares `linear: true` on a latent
bound upstream of it is refused by `linearity` at the defaults above, whether
or not the converter actually saturates.** `check_linearity` probes each
`linear: true` claim at `(1e-3, 1, 1e3)` times the latent's own scale, and a
converter that clips nothing at `1x` still clips hard at `1000x`. Measured on
the most benign ADC this package can build (`model.adc: {scale: 1.0, n_bits:
12}`, achieved peak `12.116166 adc_count` against a `2048 adc_count` clip
limit — the real forward pass clips *nothing*): `linearity` still refuses,
departure `5.32e+00` at the `1000x` probe against `rtol=1.19e-03`, with the
`0.001x` and `1x` probes both exactly `0`. **This is correct, not a false
positive** — a converter is a deliberate non-linearity, and the claim really
is false at the probe's outer scale even when it is true at the run's own
operating point. The escape is `linearity`'s own `mode:` — decline the claim
in writing, `inference.checks.linearity: {mode: skip, reason: "..."}` — which
is what the refusal's own message names.

**The defaults are forced by measurement, not by taste.** Measured on a
two-latent document on a 16 × 8 grid with no beam: `load_document` cold is
0.715 s; `check_linearity` is 0.188 s cold and 0.007 s warm *per linear
latent*; `identifiability` is 0.468 s cold; `prior_sensitivity` is **3.031 s
cold** — four times the whole build it is bolted onto. The structural
statement is the load-bearing one and it does not move with the hardware:
`linearity` costs a fixed number of forward passes per linear claim and does
not grow with `n_par`, while the other two grow with both `n_par` and
`n_data`. That is why one is on and two are off, and it is what schema
§11.15 / D-C4 already decided.

Two more checks run here and are not gated, because they are not
`inference.checks` names: `C16` (ADC saturation — one forward evaluation) and
`C18` (the two sigmas — two attribute reads on objects that already exist).

**This pass saves nothing and the page says so.** Schema §6's preamble — *"all
run before any file is read that is not needed to decide them, and before any
beam is analysed"* — is false about every check registered here. What the pass
buys is that these checks run **at all**, in this layer's voice and with this
layer's `where`, instead of detonating an hour later inside a fit.

**A cost-ordering consequence, measured rather than implied.** With
`linearity` on by default, `load_document` pays a full `check_linearity`
before a run-time per-kind grammar error is ever reported. Measured warm on
the conjugate document, min–max over nine loads: **8.5–9.0 ms** with the gate
at its default against **3.6–3.7 ms** with it declined. Every *pre-flight*
check is still in front of that; only the per-kind run grammar, which
`execute_run` decides, now sits behind a priced check.

## A gate: what runs, what a failure costs, and what is recorded

`inference.checks:` is where a document sets the price it is willing to pay.

```yaml
inference:
  checks:
    linearity:         {mode: refuse, report: true}
    identifiability:   {mode: warn, rtol: 1.0e-8}
    prior_sensitivity: {mode: skip, reason: "reported separately for this campaign"}
```

**`mode` and `report:` are orthogonal axes, and asking one question of both is
how this got shipped wrong three times in prose.** The two questions are:

- **`mode` decides what a FAILURE produces** — and, for `skip`, that the check
  does not run at all;
- **`report:` decides whether the check's NUMBERS are recorded when it
  PASSES.**

### Six effective states, four of them writable

| state | written as | meaning |
|---|---|---|
| `refuse` | `mode: refuse` | run it; a failure stops the document |
| `warn` | `mode: warn` | run it; a failure is a `ConfigWarning` |
| `report` | `mode: report` | run it; a failure is recorded and nothing else |
| `skip` | `mode: skip` + `reason:` | **do not run it**, and the document says why |
| `off` | *nothing at all* | do not run it; nobody asked |
| `auto_skip` | *nothing at all* | it was asked for and is **undefined here**; the reason is generated |

**`off` is not spelled `skip`, and that is a decision.** Check A37 requires
every written `mode: skip` to carry its own `reason:` — three unrelated skips
sharing one sentence was v0's mistake. A check that is merely *off* has no
author to write one. Collapsing the two would either force a fake reason into
the record or force A37 to exempt a case it cannot tell apart. So: **a skip
needs a reason because somebody chose it; an off does not because nobody did.**
A37 reads the document's text and therefore never sees `off` or `auto_skip` —
neither is in the document, and neither can be typed into one.

**`auto_skip` is the check you asked for and could not have.** A complex or
non-floating latent has no derivative to take, so `identifiability` and
`prior_sensitivity` are undefined on it. Rather than raise from inside the
package, the gate stands the check down and **reports** — under `C14`, naming
the latent and its dtype. It reports **even when `report: false`**, and that
asymmetry is the point: `report:` governs the numbers of a check that *ran*,
and silence about a check that did not is the failure mode this state exists
to end.

The two predicates are **not** one predicate. `identifiability` and
`prior_sensitivity` refuse complex *and* non-floating latents;
`check_linearity` accepts a complex latent and refuses only a non-floating
one. A single shared predicate would lose `C12` on a legitimate complex
latent.

### The cross-product, as one table

`mode` × `did it fail` × `report:` is decided in exactly one function. The
rows below are **eighteen cells, not twelve rows**: in the `report:` column
`either`, `—` and `ignored` each stand for both `true` and `false`, so the
three of them expand. The `failed` column's `—` does **not** expand — a gate
that never ran has no failure to have had.

| state | the check ran? | it failed | `report:` | the ONE finding |
|---|---|---|---|---|
| `refuse` | yes | yes | either | **REFUSE**, the failure sentence |
| `refuse` | yes | no | true | **REPORT**, the numbers |
| `refuse` | yes | no | false | *none* |
| `warn` | yes | yes | either | **WARN**, the failure sentence |
| `warn` | yes | no | true | **REPORT**, the numbers |
| `warn` | yes | no | false | *none* |
| `report` | yes | yes | either | **REPORT**, the failure sentence |
| `report` | yes | no | true | **REPORT**, the numbers |
| `report` | yes | no | false | *none* |
| `skip` | no | — | — | *none* |
| `off` | no | — | — | *none* |
| `auto_skip` | no | — | ignored | **REPORT** under `C14`, the generated reason — **always** |

Four consequences, each of which a reader will otherwise guess wrongly:

- **Exactly one finding per gated check per document, never two.** A failure
  with `report: true` is ONE finding at the mode's severity **whose message
  carries the numbers** — not a refusal *and* a report. Two would double-count
  in `report.checks()` and make `raise_if_refused`'s "N more refusals" tail
  wrong.
- **`{mode: skip, report: true}` has no cell**, because it asks to record the
  numbers of a check that will not run. It is refused by name, in pre-flight,
  before anything is built.
- **An `auto_skip` reports even at `report: false`.**
- **A REPORT finding never reaches `warnings.warn`.** `report.emit_warnings()`
  emits `report.warnings()`, which is `of("warn")` alone.

### What a gate is, in code

`gates(document["inference"]["checks"])` is free, is a pure function of that
mapping, and applies the defaults — so it answers "what will this document
actually pay for" before a single object is built. Pass `None` and you get the
defaults themselves.

It assumes the section has already passed the pre-flight grammar
(`A1.checks`/`A37`): on a section that has not, a non-numeric `rtol:` raises
`ValueError` and an unknown `mode:` word becomes a `Gate` in that state, which
`runs()` reports as standing down. Call `preflight` first, or read
`gate.state` against `gating.STATES` before trusting it.

```python
from rheplicant.config import gates

for name, gate in gates(document["inference"]["checks"]).items():
    print(name, gate.state, "runs" if gate.runs() else "stands down")
```

A `Gate` is five fields: `name`, `state`, `record` — the document's `report:`,
always `False` for a state that does not run — `reason`, and `rtol`, which
`identifiability` alone takes. `gate.where()` is the line a *user* edits to
change the price, which is never the line that caused the failure.

**Cardinality is three, always**, whatever the document says. A caller that
had to write `gates.get("linearity")` would write `.get("linearity", <its own
default>)` instead, and then there would be two default tables and one of them
would be wrong.

### A refused document produces no record at all

This is a real limitation and not an oversight. `raise_if_refused` raises, so
**no `ConfiguredRun` is returned and there is nowhere to hang the findings**.
A user who wrote `{mode: refuse, report: true}` and was refused gets the
refusal's own sentence — which carries the numbers — and nothing structured.
A document that *loads* carries the whole thing: `run.report` holds every
finding from all four passes, in pass order.

Closing the gap needs either a `ConfigError` that carries the `Report` or a
`load_document` that returns before raising. Both are API changes and both
belong to the plan that writes `diagnostics.json`, which **still does not
exist**: schema §4.7.8 names it twice, and `outputs:` is refused wholesale
until Plan 4.

**Four unrelated things in this layer are spelled "report", and conflating
any two of them is the likeliest way to ship something that reads right:**

| spelling | what it is |
|---|---|
| `inference.checks.<n>.report: true` | a **document key**: record this check's numbers. `Gate.record` |
| `mode: report` | a **document mode**: do not gate on this check's failure |
| `report.of("report")` | a **severity**, the third one, and its constructor |
| `run.report` | the whole **`Report`** a document earned |

## A document that is wrong four ways

Every error below is decided from text, and all four come back from one call.

```yaml
schema_version: 1
runtime: {seed: 20260814}
observation:
  meta: {telescope: RHINO}
  freq: {grid: {linspace: {start: 60.0, stop: 85.0, num: 8, endpoint: true},
                unit: MHz}}
  time: {grid: {arange: {start: 0.0, step: 2.0, num: 16}, unit: s}}
  environment: {temperature: {value: 280.0, unit: K}}
model:
  global_signal: {depth: {value: 0.5, unit: K},
                  centre: {value: 75.0, unit: MHz},
                  width: {value: 5.0, unit: MHz}}
  bandpass: {bandpass: {ones: [n_freq]}}
  gain: {gain: {value: 1.1, unit: dimensionless}}
  noise: {type: NoiseOperator, sigma: {value: 0.5, unit: K}}
inference:
  parameters:
    b: {init: {ones: [n_freq]}, into: bandpass.bandpass}
    g: {init: 1.0, linear: true, into: gain.gain}
  noise:
    kind: radiometer
    include_logdet: true
    channel_width: {value: 1.0, unit: MHz}
    integration_time: {value: 2.0, unit: s}
  observed: {from: simulation, at: {g: 1.5}}
runs:
  - {kind: conjugate.wiener, width: none, names: [g]}
```

**`{value: 1.0, unit: MHz}`, not `{value: 1.0e6, unit: Hz}`.** Measured:
`yaml.safe_load("v: 1.0e6")["v"]` is the **string** `'1.0e6'` — YAML 1.1
requires a sign in the exponent, so `1.0e+6` is a float and `1.0e6` is not.
The page's document is parsed by the suite, so the exponent form would reach
the value grammar as a string and refuse for a reason the page is not about.

Four findings, **in registry order — which is the order a reader meets them,
and A27 is first**. The rule that decides that order is not the one an earlier
version of this page gave: alphabetical position in `preflight/__init__.py`'s
foot-import block decides nothing on its own. **A foot-imported module's
checks register after everything its own head imports transitively register**
— measured, `beam_spill` sorts first in that block and its check lands
two-thirds of the way down, because it head-imports `document` and `model`.
`fitting` still registers before `model` here, so A27 and A28 are bound before
A30 and A33 — and `gated`, which imports neither, registers its C18 slot after
both, so C18 is last. Write the bullets in that order and keep them in it;
`_ordered_ids_on_the_page` is what makes a silent re-sort a red test.

- **A27** — `kind: conjugate.wiener` takes a sigma already decided into an
  array, and `inference.noise.kind: radiometer` makes sigma a function of the
  prediction, which a conjugate solve has not got. Fix: `kind: conjugate.gls`,
  which iterates the covariance it implies — and drop `width:` with it, which
  is `conjugate.wiener`'s key and not that exit's — or
  `inference.noise.kind: radiometer_frozen`, which decides one sigma up front
  and keeps this exit. That second way out is **three edits and not one**: the
  frozen kind does not take `include_logdet:`, so drop it (check A49), and it
  does not default its `source:`, so write `source: observed`, which is where
  a frozen sigma comes from. See
  [the noise section](config-inference.md#noise).
- **A30** — `model.noise` draws its own randomness and `inference.twin.without:`
  does not drop it. A `conjugate.wiener` run closes the twin over one template
  state, so that draw would be the same realisation added to every prediction
  alike. `kind: forward` and `kind: mmodes` keep the node — neither closes over
  a fit twin — and every other kind cannot. Fix:
  `inference.twin: {without: [noise]}`.
- **A33** — `b` is free into `bandpass` and `g` is free into `gain`. The two
  multiply the same prediction, so only their product is constrained and the
  fit has one exactly null direction. Fix: `transform: unit_mean_bandpass` on
  `b`, whose free vector is the `(n_freq - 1,)` mean-1 coordinates rather than
  the bandpass itself — so `init:` becomes `{ones: [7]}` on this document's
  eight channels. Written as `{ones: [n_freq]}` the transform hands the fit a
  nine-channel bandpass for eight channels of data, and the bind is refused.
- **C18** — `model.noise` draws this document's data with `NoiseOperator`, and
  `inference.noise.kind: radiometer` weighs the likelihood with a different
  noise model. They are not two spellings of one sigma, so there is no number
  to compare, and the fit's error bars are wrong by whatever the two models
  differ by. Fix: `model.noise.type: RadiometerNoiseOperator`, carrying the
  same `channel_width:`/`integration_time:` that `inference.noise` already
  declares, so the operator that draws the data agrees with the one that
  weighs it.

Each has been read off the real `Report` rather than described: this document
is executed by the test suite, which asserts that these are exactly the checks
it earns, **in this order**, that each fix named above really does clear the
finding it is offered for, and that the document with all four applied
**loads** — the last of which is what caught two of these three A/A/A
remedies being incomplete.

## Reading the report in code

```python
report = preflight(document)
for finding in report.refusals():
    print(f"{finding.where}: {finding.message}")
report.emit_warnings()
report.raise_if_refused()
```

`load_document` and `run_document` call the pass for you, so a document that
reaches either has already been through it — the explicit call is for a
front-end that wants the whole list rather than the first refusal, which is
what schema §10's "Validate" button is.
