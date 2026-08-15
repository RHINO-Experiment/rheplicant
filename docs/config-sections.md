# Config sections: observation, model, and the forward run

Plan 2A of the config layer: a parsed document's sections become the
package's own objects, with no YAML and no CLI (both arrive with Plan 4).

```python
from rheplicant.config import load_document, run_forward

run = load_document(document)      # a plain dict, schema v1
out = run_forward(run)             # = run.twin(run.state)
```

What the document's `inference:` and `runs:` sections say — the fit twin,
the latents, the likelihood and the exits — is
[its own page](config-inference.md).

## The build order

`load_document` builds in the order the sections feed each other: variants
apply first; `runtime:` decides the dtype and the PRNG key; `observation:`
becomes the grids, the resolution context, `Coordinates`, `Environment` and
the switch cycle; `resources:` resolve against that context; `model:`
assembles the twin; and an ingested recording is finished last, because
`to_state`'s `source_order` is read off the assembled twin.

## observation

The synthetic form declares the axes (`freq.grid`, `time.grid` — checked by
DIMENSION, not spelling), the site, the environment, `pointing:` (four modes;
`materialise:` is written, never inferred) and `switching:` (one `order` list
fixes the switch indices, the `cal_loads` order and the `gamma_src` rows).
The ingested form (`from_file: {format: rhino_hdf5, freq_unit: ...}`) reads a
RHINO recording as an object; `freq_unit` is required because the file does
not record it and its two producers disagree.

LSTs arrive three ways: `lst: {mode: uniform_turn}` (the FFT grid, endpoint
excluded), `lst: {from_file: ...}`, or `lst: {mode: from_site}` — computed by
`rheplicant.radio.site.lst_grid_deg` from `site.{lat_deg,lon_deg,alt_m}` and
`time.epoch`.

## model

One node key, one operator: fields are delivered off the class (static vs
traced is the field's own `eqx.field` metadata), object fields
(`observed_astro_sky.sky_model`, `.projector`) take a declared resource by
`ref` IDENTITY, `many` nodes take a list (SUM/CHAIN) or a label-keyed mapping
in switch order (FAN), `compose:` stacks several stages at one node,
`snapshot_before:` preserves the raw data, and `model.<node>.eqx_leaves:`
reconstructs a node's arrays from a saved equinox file onto the template the
node's own declared fields build (statics stay the document's — that is the
point). Lighting `beam_spill` and `ground_pickup` together requires
`acknowledge_double_count: true`.

## What is refused, and where it goes instead

`inference:` and `runs:` are read now — [their own page](config-inference.md)
covers them, `runs:` is required, and
[the sixteen kinds it runs](config-inference.md#runs) now include the
conjugate family, the cheap diagnostics, NUTS and the neural posterior. What
still goes elsewhere: `outputs:`, `defaults:` and `plugins:` arrive with
Plan 4's CLI, and so do `compare` and `benchmark`; `campaign:` stays reserved
with capability 4.

Most of what a document can be refused for is now decided *before* any of this
runs — see [the pre-flight pass](config-validation.md#the-pre-flight-pass).
