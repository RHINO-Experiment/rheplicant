# The document's anatomy

Twelve section names are recognised, and a document is built by walking them in
an order that is not the order you write them in. This page is the shape of the
whole document and the order it is assembled in. Four sections have their
reference here: `runtime:`, `variants:`, `observation:` and `model:`.

Which sections are required, which are accepted, and which are refused is
[the overview's table](config.md#the-document-section-by-section), derived from
the loader's own registries. What a single value or unit may look like is
[Values and units](config-values.md). If you have not written a document
before, [the tutorial](config-tutorial.md) builds one from four sections up.

```python
from rheplicant.config import load_document, run_forward

run = load_document(document)      # a plain dict, schema v1
out = run_forward(run)             # = run.twin(run.state)
```

## The skeleton

Every recognised section, in the order a document reads best, with the page
that answers it. This is a shape, not a document to run: the last four are
refused by `load_document`, three of them because the command line owns them.

```yaml
schema_version: 1                 # the grammar version; 1 is the only one

runtime:                          # dtype, platform, PRNG seeds
  seed: 20260824

observation:                      # axes, site, environment, switching
  freq: {grid: {linspace: {start: 60.0, stop: 85.0, num: 8}, unit: MHz}}
  time: {grid: {arange: {start: 0.0, step: 2.0, num: 16}, unit: s}}

resources:                        # files, beams, S-parameters, sky models
  bases: {}

variants:                         # named layers over the base document
  hot: {model: {gain: {gain: {value: 1.2, unit: dimensionless}}}}

model:                            # one node key, one operator
  global_signal:
    depth: {value: 0.5, unit: K}
    centre: {value: 75.0, unit: MHz}
    width: {value: 5.0, unit: MHz}

inference:                        # latents, noise, and the observed data
  parameters: {}

runs:                             # what to execute, in declaration order
  - {name: simulate, kind: forward}

defaults: [rhino_v1]              # package presets        -- command line only
plugins: []                       # imported before validation -- ditto
outputs: {dir: results/night-1}   # where the audit tree lands -- ditto
campaign: {}                      # reserved; refused with its capability named
```

## The build order

`load_document` builds in the order the sections feed each other: variants
apply first; `runtime:` decides the dtype and the PRNG key; `observation:`
becomes the grids, the resolution context, `Coordinates`, `Environment` and
the switch cycle; `resources:` resolve against that context; `model:`
assembles the twin; and an ingested recording is finished last, because
`to_state`'s `source_order` is read off the assembled twin.

That order is why `resources:` may refer to an axis and `observation:` may not
refer to a resource, and why a variant that changes a grid changes everything
downstream of it.

Four validation passes are threaded through that order, and where each one
sits determines what it can decide: the [pre-flight
pass](config-validation.md#the-pre-flight-pass) runs first over the document's
own text; the **axes** pass runs once the grids exist and one line above
`resources:`, the expensive step; the **built** pass runs over the twin, the
state and the built resources; and the **post-flight** pass runs last, after
`build_inference`, holding the checks that have to run the model to decide
anything. Only the first two run in front of a beam; see [the three later
slots](config-validation.md#the-three-later-slots-and-what-each-one-buys).
The post-flight pass is the only one whose price a document can decline, in
[`inference.checks:`](config-inference.md#checks).

## runtime

The section is required and may be empty. It takes four keys:

| Key | Default | What it sets |
|---|---|---|
| `jax_enable_x64` | `false` | float64 instead of float32 for every traced value |
| `platform` | `auto` | the JAX platform: `auto`, `cpu`, `gpu` or `tpu` |
| `seed` | `null` | the root PRNG seed, an integer or `null` |
| `seeds` | `{}` | named integer seeds, under any names |

`jax_enable_x64` and `platform` are process-global in JAX. The command line
applies them from the document before any array exists. `load_document`
cannot, so it checks that the process already agrees with the document and
refuses when it does not; see
[the delivery rule](config-values.md#the-delivery-rule) for the float64 case.

`seed` becomes `State.key`. `seed: null` is legal and means the state has no
key; a run that draws randomness then refuses when it asks for one.

Every draw in a document names its seed, `seed: {from: runtime.seeds.<name>}`.
If `runtime.seeds` declares that name, its integer is used. If it does not,
the seed is derived from the root `seed` and the name, and with `seed: null`
the draw is refused. Either way the integer used is recorded in
`provenance.json`.

## variants

`variants:` maps a name to a patch over the base document. Each variant is
the base with its patch merged in, and it is validated as a document of its
own, so `validate` checks the base and every variant.

```yaml
variants:
  hot:
    model:
      gain: {gain: {value: 1.2, unit: dimensionless}}
```

The patch merges by
[the four rules `extends:` uses](config-resources.md#extends-and-its-four-rules):
mappings merge key by key, a list replaces the base's list, `{append: [...]}`
extends it, and `~key: null` deletes `key`.

A patch may change any section but these:

- `variants` itself. Layering is one level deep; no variant builds on another.
- `schema_version`.
- `defaults`, `plugins` and `outputs`, which belong to the command line and
  are the same for every layer.

`runs:` is read from the base document. A run names the layer it executes on
with `variant: <name>`, and `compare` and `benchmark` take a list of them; see
[Runs](config-inference.md#runs). In Python,
`load_document(document, variant="hot")` builds one variant. The command line
writes each variant's resolved document to
`variants/<encoded-name>/config.resolved.yaml` in the published tree.

## observation

The synthetic form declares the axes (`freq.grid`, `time.grid`, checked by
dimension, not spelling), the site, the environment, `pointing:` (four modes;
`materialise:` is written, never inferred) and `switching:` (one `order` list
fixes the switch indices, the `cal_loads` order and the `gamma_src` rows).
The ingested form (`from_file: {format: rhino_hdf5, freq_unit: ...}`) reads a
RHINO recording as an object; `freq_unit` is required because the file does
not record it and its two producers disagree.

LSTs arrive three ways: `lst: {mode: uniform_turn}` (the FFT grid, endpoint
excluded), `lst: {from_file: ...}`, or `lst: {mode: from_site}`, computed by
`rheplicant.radio.site.lst_grid_deg` from `site.{lat_deg,lon_deg,alt_m}` and
`time.epoch`.

## model

One node key, one operator: fields are delivered off the class (static vs
traced is the field's own `eqx.field` metadata), object fields
(`observed_astro_sky.sky_model`, `.projector`) take a declared resource by
`ref` identity, `many` nodes take a list (SUM/CHAIN) or a label-keyed mapping
in switch order (FAN), `compose:` stacks several stages at one node,
`snapshot_before:` preserves the raw data, and `model.<node>.eqx_leaves:`
reconstructs a node's arrays from a saved equinox file onto the template the
node's own declared fields build (statics stay the document's). Lighting
`beam_spill` and `ground_pickup` together requires
`acknowledge_double_count: true`.

## What is refused, and where it goes instead

`inference:` and `runs:` have [their own page](config-inference.md); `runs:`
is required, and [the eighteen kinds it runs](config-inference.md#runs) are
listed there.
`outputs:`, `defaults:`, and `plugins:` are refused by `load_document` and
handled by the [configuration command](config-cli.md); scientific products are
selected under `outputs.write`. `campaign:` is reserved with capability 4.

Most of what a document can be refused for is decided before any of this
runs; see [the pre-flight pass](config-validation.md#the-pre-flight-pass).
