# Configuration documents

A configuration document is a declarative route from exact YAML bytes to a
validated, audited run. The Python API, the command line and the browser
workbench are three surfaces over the same accepted bytes; none of them is a
second configuration language.

## What v1 covers, and what it does not

Two of the package's capabilities are out of reach from YAML today. The
refusal messages are the only other place that says so.

| | |
|---|---|
| **Covered** | One observation, end to end: the instrument model, the resources it reads, the likelihood and noise, and every fitting exit: forward, Fisher, optimize, the conjugate solvers, sampling plans, NUTS, NPE and the diagnostics. |
| **Deferred: `campaign:`** | Streaming evidence (compressing each night to a fixed-size likelihood factor and discarding the data) has no YAML surface. The section name is reserved, and a document that uses it is refused. |
| **Deferred: `type: NeuralOperator`** | The neural surrogate is refused with its capability named. Its `mlp:` field is an object no value node can express. |
| **Python only** | A custom graph topology. Configuration targets the canonical radio graph; `compose:` with `cascade`/`sum`/`many` gives bounded composition freedom inside it, but new nodes, junctions and selectors are written in Python. |

## The document, section by section

Twelve top-level names are recognised. Four sections are required, four names
are accepted, and four are refused today. The loader's own refusal tables are
what `rheplicant.config.schema.json_schema()` reports as each section's
`status`.

| Section | Status |
|---|---|
| `runtime`, `observation`, `model`, `runs` | required |
| `schema_version`, `resources`, `variants`, `inference` | accepted |
| `defaults`, `plugins`, `outputs` | refused by the mapping API; the command line handles them |
| `campaign` | reserved, refused; see above |

`schema_version: 1` is listed as accepted because it is a key and not a
section, but a document without it is refused: write it in every document.

The third row is the difference between `load_document` and the CLI: the CLI
adds presets, plugins and the output tree on top of the same orchestration.

## Reading order

Start with the tutorial if you have not written a document before. The five
reference pages after it are the document's grammar, in the order they build
on each other. The command line and the workbench, which consume a document,
are in the next section.

:::{list-table}
:header-rows: 1
:widths: 30 70

* - Page
  - What it answers
* - [Tutorial: your first configuration](config-tutorial.md)
  - Two complete documents, one that simulates and one that fits, and the two
    commands that validate and run them. Both are executed by the test suite.
* - [The document's anatomy](config-anatomy.md)
  - Every recognised section as one skeleton, the order they are built in, and
    the two that describe the instrument in full.
* - [Values and units](config-values.md)
  - How any single number, array or reference is written, and how a unit is
    checked. Eighteen form keys in eight families.
* - [Resources and captured inputs](config-resources.md)
  - Where data comes from: files, beams, S-parameters, sky models, and how a
    path becomes a byte-exact recorded input.
* - [Inference, likelihoods and exits](config-inference.md)
  - The likelihood, the noise model, the parameters, and every run kind. A
    complete worked document lives here under *A complete document*.
* - [Validation passes and findings](config-validation.md)
  - What is checked, when, and what a finding tells you. Checks needing a built
    model are deferred to a named later boundary, never to an executor.
:::

## The three surfaces

- **Python**: `load_document` on a mapping. The smallest surface, and the one
  the other two are built on.
- **[Command line](config-cli.md)**: adds exact-byte source identity, runtime
  ordering, output security, resolved YAML and provenance. It parses the base
  and every declared variant, and every run kind's options, before the first
  executor is called.
- **[Workbench](config-gui.md)**: four browser views over the accepted bytes.
  Raw YAML and safe field drafts are submitted with an expected revision, and
  every accepted edit returns complete YAML. Quick checks are immediate
  projections; full validation, previews and declared actions are explicit jobs
  whose results stay bound to the revision and digest that produced them.

```{toctree}
:maxdepth: 2
:hidden:

config-tutorial
config-anatomy
config-values
config-resources
config-inference
config-validation
config-cli
config-gui
```
