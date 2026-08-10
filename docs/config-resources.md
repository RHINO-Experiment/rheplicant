# Resources and paths in a config document

[Values in a config document](config-values.md) is the grammar for one
fragment. This page is the two pieces built on top of it: **resources**, a
`resources:` section that names objects the package's own constructors build
— a beam, a sky model, a projector — and **paths**, the dotted strings that
address a leaf of an assembled twin for `inference.parameters`. Both are
exercised directly, the same way the value grammar is, because the document
loader that reads a whole YAML file is not shipped yet:

```bash
.venv/bin/python -c "
from rheplicant.config import ResolutionContext, build_resources
import jax.numpy as jnp

context = ResolutionContext(freq=jnp.linspace(60e6, 85e6, 4), dtype='float32')
built = build_resources({'arrays': {'a': {'list': [1.0, 2.0]}}}, context)
print(built.resources['resources.arrays.a'])
"
```

For the signatures, see [the API reference](api.md).

## What a resource is, and why it is built once

A `resources:` section is a mapping of *kind* to *name* to *entry* —
`resources.<kind>.<name>` — and each entry is built through the package's own
constructor for that kind: `resources.beams.horn` calls the beam builder,
`resources.projectors.driftscan` calls `DriftScanProjector.from_beam_maps`,
and so on. It exists for the same reason `resources.arrays` exists in the
value grammar (see [values: a reference to a named
resource](config-values.md#5-a-reference-to-a-named-resource)): naming, not
nesting, is how this schema composes. A run needs one beam analysed once and
referenced by two projectors, not one beam analysed twice that happen to
agree.

**Built once, and each reference is the same object, not a copy.** This is a
physics requirement, not an optimisation. `BeamSpillOperator.from_projector`
is documented as "the one call that cannot get the weight and the sky average
out of step" — a spill correction computed from one beam array and a sky
average computed from a second, independently-built copy of the *same*
nominal beam can drift apart by construction, passing every shape check while
silently decoupling. `build_resources` builds the dependency graph implied by
every `{ref: resources.<kind>.<name>}` it finds, builds each entry exactly
once in that order, and hands identical objects to every reference:

```python
built = build_resources(
    {"arrays": {"shared": {"list": [1.0]},
                "left": {"ref": "resources.arrays.shared"},
                "right": {"ref": "resources.arrays.shared"}}},
    context,  # a ResolutionContext
)
built.resources["resources.arrays.left"] is built.resources["resources.arrays.right"]
# => True
```

`built.shared_objects` records every such group — grouped by `id()` — because
schema §2.1.6 promises a `shared_objects:` map in `config.resolved.yaml`:
identity should be visible in the artefact a run produces, not only in the
spec that produced it. Order in the document does not matter; a cycle
(`a` referencing `b` referencing `a`) is refused by name, because
`resources.<kind>.<name>` is a let-binding and a let-binding that refers to
itself has no value.

## The six kinds

Every registered kind is listed in `rheplicant.config.RESOURCE_KINDS`. One
worked entry each:

### `resources.arrays`

Any value node, bound to a name. There is no package constructor behind it —
it exists because the value grammar deliberately has no expression language,
and naming an inner call so an outer one can reference it is how `f(g(x), y)`
is spelled here:

```yaml
resources:
  arrays:
    inner: {list: [1.0, 2.0, 3.0, 4.0]}
    outer: {python: "jax.numpy:multiply",
            args: {x1: {ref: resources.arrays.inner}, x2: {value: 2.0}}}
```

### `resources.bases`

A `SeparableBasis` with a time design matrix and a frequency design matrix.
`n` is never written — it is always `len(observation.<axis>.grid)` — because a
basis built for another band "returns a smooth, plausible, wrong temperature"
and taking the sample count from the run's own grid makes that structurally
impossible rather than merely discouraged:

```yaml
resources:
  bases:
    t_ant:
      time: {kind: legendre, n_basis: 3}
      freq: {kind: legendre, n_basis: 2}
```

This kind also closes the value grammar's `basis_fit` form, which Plan 1A
declared and left unregistered: `{basis_fit: {basis: {ref: resources.bases.
<name>}, field: <value node>}}` returns the least-squares coefficients of a
field on a named basis, through `SeparableBasis.fit`.

### `resources.sky_models`

What the sky *is*, kept apart from the beam that sees it — `kind: maps` is
the one place in this layer that can catch a `MapSky` built for one band and
evaluated on another, because `MapSky.__call__` does not consult its `freq`
argument beyond the shape, so a mismatch is otherwise finite, plausible and
wrong:

```yaml
resources:
  sky_models:
    galaxy: {kind: gdsm, nside: 64}
```

`kind: gdsm` is the D-C8 addition: the GSM16 model through
`limTOD.sky_model.GDSM_sky_model`, evaluated per-channel on the run's own
`observation.freq.grid` and stacked into a `MapSky` — so, unlike `kind: maps`,
there is no separate grid to declare and no band-mismatch check to fail,
because the maps are generated on the run's grid rather than declared against
one. It takes `nside` only; there is no amplitude to scale, because the model
decides it. It needs `pygdsm`, which is optional and arrives through limTOD's
own extra — the missing-package refusal names `pip install "limTOD[gdsm]"`.
The other four kinds are `uniform`, `power_law`, and the `python:` hatch.

### `resources.beams`

A raw `(n_freq, n_pix)` array plus what the file cannot say. A beam is not an
object anywhere else in this package — `cst_beam_maps` returns a bare
`np.ndarray` — so this kind returns a small container holding the maps *and*
the sky fraction a horizon truncation produces, because dropping the fraction
(v0's behaviour) silently deletes the `(1 - f_sky) * T_ground` term
`BeamSpillOperator` needs:

```yaml
resources:
  beams:
    horn:
      format: cst
      directory: data/cst_horn/
      nside: 32
      normalize: pixel_sum
      phi0_deg: 90.0
      phi_sense: ccw
```

Two declarations have no default and no preset may ever supply them.
`normalize:` — the output's unit is decided by the *pair* (this key, the
projector's `normalize_beam`), and neither half is inferable from the other:
32838 K against 200 K on a uniform 200 K sky for an unnormalised beam, 100.42
K against 99.79 K for a unit-pixel-sum one. `phi0_deg` / `phi_sense` —
required for `format: cst` only, refused everywhere else — because a mirrored
beam passes every integral, every peak and every azimuthally-symmetric
diagnostic unchanged: there is no numerical symptom, so the only protection
is that the value was stated by someone who knew the horn.

`format: uvbeam` and `format: healpix` are the D-C7 addition, both building
the raw array from a file:

```yaml
resources:
  beams:
    horn_uvb:
      format: uvbeam
      path: data/horn.beamfits
      nside: 32
      normalize: pixel_sum
```

`format: uvbeam` reads a pyuvdata `UVBeam` file and samples it per channel
through limTOD's own bridge (`limTOD.uvbeam.uvbeam_to_healpix_maps`), one
frequency per call in MHz, onto the run's `observation.freq.grid`. It takes
neither `phi0_deg`/`phi_sense` (the bridge carries its own azimuth
convention) nor `frame:` (the bridge's output is `beam_local` by
construction). It needs `pyuvdata`, declared as this package's own `uvbeam`
extra — `uv pip install -e ".[uvbeam]"` — checked before the path is even
resolved, because the alternative is a run that fails only after everything
else has been built.

```yaml
resources:
  beams:
    horn_hpx:
      format: healpix
      path: data/horn.fits
      order: ring
      freq: {from_grid: freq}
      nside: 32
      normalize: none
      frame: beam_local
```

`format: healpix` requires two declarations a FITS file cannot be trusted to
carry: `order:` (`ring` or `nested` — RING-versus-NESTED, not a CST meridian,
is the fact this format's file cannot state, so it is a required declaration
rather than an inference; a `nested` declaration is reordered to RING exactly,
and a declaration that contradicts the file's own `ORDERING` header is refused
rather than trusted), and `freq:` (the grid the file's columns were built on,
column *i* belonging to `freq[i]`, checked against the run's own grid the same
way `kind: maps` checks a sky). `frame:` is still required, the same as any
other raw-array format.

### `resources.projectors`

How the sky is seen, with the beam folded in. Field names are the Python
names verbatim:

```yaml
resources:
  projectors:
    driftscan:
      engine: driftscan
      beam: {ref: resources.beams.horn}
      lat_deg: -30.7
      az_deg: 0.0
      el_deg: 90.0
      lmax: 32
      normalize_beam: true
      acknowledge_float32_sky: true  # this page's own context is float32 -- see below
```

`normalize_beam` has no default and must be written — `false` returns
`integral(B . T dOmega)`, which is not a temperature (32838 K against 200 K on
a uniform 200 K sky). `beam_frame` and `beam_ref_lst_deg` are not writable at
all: they are set only by `to_reference_frame()`, and writing them by hand
would drive the object into exactly the state its own `__check_init__` exists
to catch — the config route is the ordered `optimizations: [cache_beam_
rotation]` list plus `lst_ref_deg`.

Both real-sky engines (`driftscan`, `general_pointing`) are also gated on
precision, which is why the example above writes `acknowledge_float32_sky:
true`: the map/alm steps carry an O(10 %) error in `float32` that is
invisible by every diagnostic available — the result comes back finite,
correctly shaped and plausibly structured, exactly the shape of failure
`normalize_beam` and `phi0_deg`/`phi_sense` exist elsewhere in this layer to
guard against. That is why `acknowledge_float32_sky` is the only key in this
whole layer that exists to acknowledge a problem rather than to solve one:
`runtime.jax_enable_x64: true` is the real fix — x64 is process-global and
part of the hashed config — and writing `acknowledge_float32_sky: true` on
the entry is the explicit opt-out for a run that has decided to accept the
error instead.

This kind also lands the `horizon_fraction` derivation Plan 1A deferred:
`{from: horizon_fraction, projector: {ref: resources.projectors.<name>}}`
reads `DriftScanProjector.horizon_fraction()`, and refuses on a projector
already folded into its reference frame, because the unmasked denominator is
gone by then.

### `resources.s_params`

Reflection coefficients, from a file or from `rhino-cal`:

```yaml
resources:
  s_params:
    lna_input:
      kind: touchstone
      file: {path: data/lna.s2p}
      component: s11
```

`kind: touchstone` reads through the *public* `file:` value node — `format:
touchstone` is registered with `array=False` (below), so the result is a
`Touchstone` object, not an array. `kind: termination` and `kind: cable` call
`rhino_cal_jax.termination_gamma` / `.cable_gamma`, an optional `cal` extra
that is not on PyPI by design. `z0` is a `kind: termination` key and nowhere
else: `Touchstone.z0` is parsed and never read by any other module, while
`termination_gamma(z0=)` is read, so the key exists exactly where it is
consumed. This kind also lands the `interpolate_onto` derivation Plan 1A
deferred, for reading a component of a Touchstone sweep back onto the run's
frequency grid without going through `kind: touchstone`'s own resource entry.

## `array=False`: a reader whose result is not an array

`format: touchstone` is the first of three file formats registered with
`register_reader(..., array=False)` — the other two are `rhino_hdf5`, a RHINO
recording read as an object in [`observation:`](config-sections.md#observation),
and `eqx_leaves`, a saved equinox file read onto a `model:` node's own
template (see [config-sections.md](config-sections.md#model)). Every other
reader's result is `jnp.asarray`'d before it reaches a field; a `Touchstone`
is a dataclass of three fields (`freq_hz`, `s`, `z0`), and wrapping it in
`jnp.asarray` would either mangle it or raise. `array=False` tells
`files.py`'s `file:` form to hand the reader's return value back unwrapped,
and to refuse any modifier written on that node — `unit:`, `part:`, `scale:`
all describe what an array's numbers *are*, and a `Touchstone` is not one.

## `extends:`, and its four rules

An entry may `extends:` a sibling **of the same kind** — the keys of two kinds
describe different constructors, so a merge across kinds would produce an
entry no builder can read, and is refused by name. Extending resolves the
whole chain before anything is built, so a child declared before its own
parent in the document still sees the parent's fully-resolved spec, and an
`extends:` cycle is refused by name rather than silently merging over an
incomplete parent.

1. **Mappings merge**, key by key, recursively. Child
   `{kind: termination, z0: {value: 75.0, unit: ohm}}` over parent
   `{kind: termination, termination: open, z0: {value: 50.0, unit: ohm}}`
   gives `{kind: termination, termination: open, z0: {value: 75.0, unit:
   ohm}}` — `termination:` is inherited, `z0.value:` is overridden.

2. **Lists replace, they do not merge.** This is schema §5 rule 4, and the
   reason is a whole comparison forced into one document: split across two
   files, the halves of a list can silently disagree in exactly the keys the
   comparison is about. Child `{optimizations: [cache_beam_rotation]}` over
   parent `{optimizations: [read_horizon_fraction, x]}` gives
   `{optimizations: [cache_beam_rotation]}` — the parent's list is gone
   entirely, not merged into.

3. **`{append: [...]}` extends a list**, for the cases that do want to add
   rather than replace. Child `{optimizations: {append: [cache_beam_
   rotation]}}` over parent `{optimizations: [a]}` gives `{optimizations: [a,
   cache_beam_rotation]}`.

4. **A `~key` entry deletes `key`** from the inherited spec. Child
   `{~apod_deg: null}` over parent `{apod_deg: 1.0, lmax: 8}` gives
   `{lmax: 8}` — `apod_deg` is gone, `lmax` is inherited unchanged.

## The path grammar

`inference.parameters.Bind(into=...)` holds Python **callables**, not
strings, and `ParameterSpace._resolve_targets` *invokes* them against a copy
of the twin whose every leaf has been replaced by its own key path. A path
string in a config document — `"noise_wave.t_unc"` — has to become one of
those callables, and `compile_path` is the compiler:

```python
from rheplicant.config import compile_path

selector = compile_path("gain.gain")
selector(twin) is twin["gain"].gain  # twin is a Pipeline holding a "gain" stage
# => True
```

Resolution is also **eager**: `resolve_path_on` walks the path against a
tagged twin immediately, before the forward model is built and anything is
traced — because the alternative is the same refusal arriving from
`ParameterSpace` once a CST directory has already been read and analysed,
which is exactly the class of failure schema §6's "before any expensive work"
promise exists to prevent.

Every refusal below quotes **both spellings** — the path as the document
wrote it (`"gain.gain"`) and the path as the twin's own machinery renders it
(`".stages[0].gain"`, via `jax.tree_util.keystr`) — because the package's own
messages quote only the second, and a reader who typed the first has never
seen it.

### The six refusals

1. **The walk does not reach an array leaf.** Two ways this happens: the walk
   stops on an operator or other container with fields still below it (write
   one more step), or the walk lands on a genuine static field — a value in
   the object's treedef rather than among its traced leaves, unreachable by
   *any* path, not just this one.
2. **An ambiguous multi-operator node.** A head that names a node holding two
   or more operators raises `AmbiguousNodeError` during the walk itself, and
   this layer reproduces the package's own composite message — the instance
   ids, and "addresses none of them" — rather than inventing a second one.
3. **A path into an aliased node.** `Assembly.aliased` names a node folded
   into the graph at more than one place. Binding it would rewrite the one
   branch the path reaches and silently leave the others in the forward
   model — finite, correctly shaped, wrong, because the latent would then
   read as frozen everywhere but that one branch.
4. **Two declared paths reaching one leaf.** One binding would silently win,
   and which one is an implementation detail of document order. If two
   quantities really are the same number, declare one latent and give its
   `Bind` both targets with `fan: broadcast` instead.
5. **A `twin.replace` target colliding with a binding's.** This is schema
   check B8, and it is Plan 2's to implement — `twin.replace` and
   `inference.parameters` are both `inference:` keys, and neither exists
   until Plan 2. The comparison machinery (`refuse_duplicate_targets`) is
   already in place for it to call once the replace targets exist.
6. **A multi-node region addressed by the wrong key.** `At((a, b, c), op)`
   covers a contiguous region, and the fold labels the covering operator with
   the region's **last** node — not its first, which is where slot kinds are
   screened on entry, a different check entirely. A config key naming any
   other covered node would resolve to a bare `KeyError` rather than the
   refusal the schema promises; this check catches it by name first.

```bash
.venv/bin/python - <<'PY'
import jax.numpy as jnp
from rheplicant.config import compile_path
from rheplicant.core.pipeline import Pipeline
from rheplicant.inference import Bind, Latent, ParameterSpace
from rheplicant.radio.instrument.antenna_loss import AntennaLossOperator

twin = Pipeline(AntennaLossOperator(efficiency=jnp.asarray(0.97),
                                    t_physical=jnp.asarray(293.0)), names=["antenna_loss"])
space = ParameterSpace(
    latents=[Latent("eta", init=jnp.asarray(0.97))],
    bindings=[Bind("eta", into=compile_path("antenna_loss.efficiency"))],
)
space.validate(twin)
print("ParameterSpace.validate accepted a compiled config path")
PY
```
