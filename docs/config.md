# Configuration documents

A configuration document is a declarative route from exact YAML bytes to a
validated, audited run. One document is the truth: the Python API, the command
line and the browser workbench are three surfaces over the same accepted bytes,
and none of them is a second configuration language.

## What v1 covers, and what it does not

Worth knowing before you write anything, because two of the package's headline
capabilities are **deliberately** out of reach from YAML today and the refusal
messages are the only other place that says so.

| | |
|---|---|
| **Covered** | One observation, end to end: the instrument model, the resources it reads, the likelihood and noise, and every fitting exit — forward, Fisher, optimize, the conjugate solvers, sampling plans, NUTS, NPE and the diagnostics. |
| **Deferred: `campaign:`** | Streaming evidence — compressing each night to a fixed-size likelihood factor and discarding the data — has no YAML surface. The section name is reserved and refused, so a document that reaches for it is told, not ignored. |
| **Deferred: `type: NeuralOperator`** | The neural surrogate is refused with its capability named. Its `mlp:` field is an object no value node can express. |
| **Python only** | A custom graph *topology*. Configuration targets the canonical radio graph; `compose:` with `cascade`/`sum`/`many` gives bounded composition freedom inside it, but new nodes, junctions and selectors are written in Python. |

## The document, section by section

Twelve section names are recognised. Four are required, four are accepted and
optional, and four are refused today — the loader's own refusal tables are what
`json_schema()` reports as each section's `status`, so this table and the schema
cannot drift apart.

| Section | Status |
|---|---|
| `runtime`, `observation`, `model`, `runs` | **required** |
| `schema_version`, `resources`, `variants`, `inference` | accepted |
| `defaults`, `plugins`, `outputs` | refused by the mapping API; the **command line** handles them |
| `campaign` | reserved, refused — see above |

That third row is the one that surprises people: `load_document` and the CLI are
not the same surface. The CLI adds presets, plugins and the output tree on top
of the same orchestration.

## Reading order

The pages below are written to be read in this order. The first three are the
document's own grammar; the last two are the surfaces that consume it.

:::{list-table}
:header-rows: 1
:widths: 30 70

* - Page
  - What it answers
* - [Values and units](config-values.md)
  - How any single number, array or reference is written, and how a unit is
    checked rather than assumed. Eighteen form keys in eight families.
* - [Resources and captured inputs](config-resources.md)
  - Where data comes from: files, beams, S-parameters, sky models, and how a
    path becomes a byte-exact recorded input.
* - [Observation and model sections](config-sections.md)
  - The build order, and the two sections that describe the instrument.
* - [Inference, likelihoods and exits](config-inference.md)
  - The likelihood, the noise model, the parameters, and every run kind. A
    complete worked document lives here under *A complete document*.
* - [Validation passes and findings](config-validation.md)
  - What is checked, when, and what a finding tells you. Checks needing a built
    model are deferred to a named later boundary, never to an executor.
:::

## The three surfaces

- **Python** — `load_document` on a mapping. The smallest surface, and the one
  the other two are built on.
- **[Command line](config-cli.md)** — adds exact-byte source identity, runtime
  ordering, output security, resolved YAML and provenance. It parses the base
  and every declared variant, and every run kind's options, **before** the first
  executor is called.
- **[Workbench](config-gui.md)** — four browser views over the accepted bytes.
  Raw YAML and safe field drafts are submitted with an expected revision, and
  every accepted edit returns complete YAML. Quick checks are immediate
  projections; full validation, previews and declared actions are explicit jobs
  whose results stay bound to the revision and digest that produced them.

```{toctree}
:maxdepth: 2
:hidden:

config-values
config-resources
config-sections
config-inference
config-validation
config-cli
config-gui
```
