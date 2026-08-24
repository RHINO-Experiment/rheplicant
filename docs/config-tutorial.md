# Tutorial: your first configuration

Two documents, both complete and both executed by the test suite. The first
simulates; the second fits a parameter and recovers a value that was injected
into the data. Neither is a fragment — copy either one into a file and run it.

A configuration document is a declarative route from exact YAML bytes to a
validated, audited run, so there is no step here where you drop into Python to
finish the job. What each section is for, and the order they are assembled in,
is [the document's anatomy](config-anatomy.md).

## A document that simulates

Four sections are required — `runtime`, `observation`, `model`, `runs` — and
four sections is a runnable document. `schema_version` is optional but worth
writing.

```yaml
schema_version: 1

runtime:
  seed: 20260824

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

runs:
  - {name: simulate, kind: forward}
```

`observation:` fixes the two axes and nothing else has to agree with them by
hand: the 16 times and 8 frequencies here become the shape of everything
downstream. `model:` names two nodes of the canonical radio graph and gives
each its fields; every node you do not name is simply absent. `kind: forward`
runs the twin once and hands back a `State`, so `run.data` comes out
`(16, 8)` — the absorption trough at 75 MHz, times the gain.

Two habits the grammar enforces from the first line. **A number carries its
unit** — `{value: 0.5, unit: K}`, not `0.5` — and the unit is checked against
what the field is declared to be, so a depth in MHz is refused rather than
silently believed. **A grid is a form, not a list**: `linspace` and `arange`
are two of the eighteen form keys, and writing one wrong is refused naming
the families it could have been. Both are [Values and
units](config-values.md).

## A document that fits

To fit, a document needs three more things, and they all live under
`inference:`: which model field is free (`parameters`), how noisy the data is
(`noise`), and where the data comes from (`observed`). Here it comes from the
model itself with the gain set to 1.4, which is what makes the fit checkable —
the answer is known before the run starts.

```yaml
schema_version: 1

runtime:
  seed: 20260824

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
    sigma: {value: 0.01, unit: K}
  observed:
    from: simulation
    at: {g: 1.4}
    realise:
      kind: homoscedastic
      sigma: {value: 0.01, unit: K}
      seed: {from: runtime.seeds.observed_noise}

runs:
  - {name: simulate, kind: forward}
  - {name: fit, kind: optimize, optimizer: adam, learning_rate: 0.05, n_steps: 300}
```

`into: gain.gain` is the whole of the wiring: the latent `g` replaces that
field of that node, and `model.gain.gain: 1.1` becomes the value used when `g`
is not being fitted. `at: {g: 1.4}` is the injection, `realise:` adds the noise
the fit will have to see through, and the two runs execute in declaration
order. `fit` returns `{"params": {"g": ...}, "losses": [...]}` and lands on
**1.4002** against the 1.4 that went in.

Nothing about `optimize` is defaulted for you: `optimizer`, `learning_rate` and
`n_steps` are all required, because the shipped default learning rate sits five
orders of magnitude from what a real fit has needed. Omit one and the refusal
names it. Misspell one — `steps` for `n_steps` — and the refusal lists the
seven keys the kind does take. That is the general shape of being wrong here:
the vocabulary comes back with the complaint.

`optimize` is one of eighteen run kinds. The conjugate solvers, sampling plans,
NUTS, the neural posterior and the cheap diagnostics are [Inference,
likelihoods and exits](config-inference.md#runs), which also carries a longer
worked document under [*A complete
document*](config-inference.md#a-complete-document).

## Running it from the command line

`run_document` in Python executes a schedule and hands back results. The
command line does that and more: exact-byte source identity, the declared JAX
runtime, package presets, plugins, resolved YAML, and an audited output tree.

```bash
rheplicant validate observation.yaml
rheplicant run observation.yaml
```

`validate` parses, applies presets, establishes the runtime, imports plugins
and validates the base plus every variant — including every run kind's options
— without creating a directory, a lock or a result. Only `run` publishes
anything. A file named `config.yaml` defaults to `config.results/`; anything
else needs `outputs.dir`, which is one of the three sections the command line
owns and `load_document` refuses.

A package preset is an exact, hashed YAML layer rather than a replacement for
the user document. `rhino_v1` intentionally leaves instrument-specific facts
for the user:

```yaml
schema_version: 1
defaults: [rhino_v1]
runtime: {jax_enable_x64: true, platform: cpu, seed: 20260817}
outputs:
  dir: results/rhino-night-1
  clobber: false
runs:
  - {name: simulate, kind: forward}
```

The remaining observation, beam-orientation, and model facts still have to be
declared; `defaults: [rhino_v1]` alone is not a runnable observation.

What a successful run publishes, what a refusal publishes instead, and how the
scientific products are selected are all on [the command line
page](config-cli.md#audit-and-scientific-output-trees).

## Where to go next

- [The document's anatomy](config-anatomy.md) — every section, and the order
  they are built in.
- [Values and units](config-values.md) — the eighteen form keys, and why a
  number carries its unit.
- [Resources and captured inputs](config-resources.md) — reading a beam, an
  S-parameter or a sky model, byte-exactly.
- [Validation passes and findings](config-validation.md) — what is checked
  when, and what a finding tells you.
