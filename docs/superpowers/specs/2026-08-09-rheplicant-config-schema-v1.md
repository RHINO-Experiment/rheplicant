# rheplicant config schema v1

**Status.** v1 supersedes SCHEMA v0. Every BLOCKER and MAJOR finding from the five stress ports and the completeness critic is addressed below; each MINOR finding is either fixed or refused in one line in §7. Two capabilities stay deferred (§8), and the shape of their holes is confirmed or corrected against the source.

**Three v1 changes require package work before a loader can be written.** They are called out as `PACKAGE CHANGE` where they appear and collected in §11. They are: ship `MapSky` as a real `AbstractSkyModel`; ship a fractional-noise realisation seam for `observed.from: simulation`; give `SnapshotOperator` a reachable placement. Nothing else in v1 needs the package to change.

---

## 1. What the config is, and what it is not

### 1.1 What it is

One YAML file declares **one instrument on one observation, and one or more exits taken over it**. That is the single structural change from v0's "one file, one run", and it is forced by measurement: four of the five example scripts stress-ported are *comparisons* — two twins, or one twin through two engines — and a schema that cannot hold a comparison cannot hold `examples/gibbs_plan.py`, `examples/driftscan_mmode.py`, `examples/three_ways_to_a_posterior.py` or `examples/sky_to_noise_wave.py`. Splitting them across files does not preserve the comparison: `defaults:` merges mappings and replaces lists, so the two halves can silently disagree in exactly the keys the comparison is about.

The file therefore has three layers:

```
base document      what the instrument is, what was observed        (§4)
variants:          named deep-merge patches over the base           (§5.3)
runs:              a list of exits, each naming a variant           (§4.7)
```

`runs:` may be a single mapping — the one-run case is not penalised.

### 1.2 What it is not

* **It is not a programming language.** There is no expression grammar, no trig, no powers, no user-defined functions, no control flow, no loops. Composition is by *naming*, not by nesting: `resources.arrays.<name>` binds a value node to a name and `{ref: ...}` reads it back, which gives a DAG of named quantities and nothing more. Anything that needs arithmetic beyond an affine relabel (`scale:`/`offset:`) or a normalisation declaration goes through the one Python escape hatch (§2.3), and pays the stated cost.
* **It is not a second API.** Field names are the Python field names, verbatim, including their unit suffixes (`lat_deg`, `apod_deg`, `lst_ref_deg`). v0 stripped those suffixes in `resources.projectors` while the path grammar kept them, so one field had two spellings; v1 removes that contradiction by keeping the Python name everywhere and letting the value grammar carry a redundant, *checked* unit.
* **It is not a wrapper that hides the package's refusals.** Every runtime refusal that can be decided from text becomes a config-validation refusal (§6), with the same wording and the same quoted numbers.
* **It is not authoritative about physics the code refuses to default.** `phi0_deg`, `phi_sense`, `normalize_beam`, `line_width`, `freq_unit`, `include_logdet`, ADC-trunk units — these have no schema default and no preset may supply them (§5.4).

### 1.3 The two things a config must be able to say that v0 could not

1. **"Simulate here, fit there."** The truth that made the data and the value the fit starts from are different numbers about the same leaf. v1 separates them: `model` holds the instrument, `inference.observed.at` holds the truth, `inference.parameters.<name>.init` holds the start, and `inference.twin.replace` holds any other difference between the simulator and the fit target.
2. **"And did it work?"** With `observed.from: simulation` the truth is known by construction and already in the file. v1 derives it and writes `recovery.json` — per latent: truth, estimate, posterior sigma, absolute error, pull. Every one of the five ported scripts computed exactly this and v0 had nowhere to put it.

---

## 2. The three grammars

Defined once here. Used unchanged everywhere below.

### 2.1 THE VALUE GRAMMAR

Anywhere the schema says **value node**, exactly one of these eight *forms* is legal, optionally carrying any of the seven *modifiers*.

#### 2.1.1 Delivery: static vs traced (the v0 blocker, fixed)

v0 said "the loader ALWAYS finishes with `jnp.asarray(...)`". That is **wrong for every `eqx.field(static=True)` field**, and it is fatal, not cosmetic. Verified in this repository:

* `ADCOperator.n_bits` is `int = eqx.field(static=True)` (`src/rheplicant/radio/instrument/adc.py:34`) and `__check_init__` runs `isinstance(self.n_bits, int)` (`adc.py:37`) — an array there **raises**. Same for `EMIOperator.period` (`radio/instrument/emi.py:38`) and `BackendOperator.n_chunk` (`radio/backend/averaging.py:155`).
* `ForegroundOperator.ref_freq` (`radio/sky/foregrounds.py:41`), `IonosphereOperator.ref_freq` (`radio/environment/ionosphere.py:39`), `RFIOperator.occupancy` (`radio/environment/rfi.py:41`), `FlaggingOperator.threshold` (`radio/backend/flagging.py:71`) do **not** raise — they put a JAX array in the treedef and equinox warns `A JAX array is being set as static!`, which is precisely the jit-cache-key corruption the asarray rule claimed to prevent.
* Only six shipped classes attach `converter=jnp.asarray`: `BasisTemperatureOperator` (`radio/t_sys.py:109-111`), `BeamSpillOperator` (`radio/instrument/beam_spill.py:85-86`), `AntennaLossOperator` (`radio/instrument/antenna_loss.py:66-67`), `NoiseWaveOperator` (`radio/instrument/noise_wave.py:206-213`), plus `Coordinates`/`Environment`'s `as_array_or_none`. So a raw Python float elsewhere either raises (`GainOperator`, `ReceiverOperator`, `CalLoadOperator`, `ApplyCalibrationOperator` all read `.ndim`) or yields **zero traced leaves** under `eqx.partition(op, eqx.is_inexact_array)`.

**v1 rule.** The loader consults the target operator's own dataclass field metadata:

| target field | loader delivers |
|---|---|
| `eqx.field(static=True)` typed `int` | Python `int` |
| `eqx.field(static=True)` typed `float` | Python `float` |
| `eqx.field(static=True)` typed `str` / `bool` | Python `str` / `bool` |
| `eqx.field(static=True)` typed `tuple`/`Mapping` | Python tuple / `FrozenMapping` |
| anything else | `jnp.asarray(...)` under the run's dtype |

Every node in §4.5's table carries a `deliver` column so this is on the page, not in the loader. A value node may state `as: static_int \| static_float \| static_str \| traced` to make the expectation explicit; a mismatch with the field metadata is a config error. **A `file:`, `draw:`, `stack:` or array-spec form landing on a static field is refused by name** (check A40).

`CWCalibrationOperator` is the one class that guards itself: its static fields carry converters (`radio/instrument/calibration.py:358-369`) that refuse a traced value with a named message. The schema mirrors, not duplicates, that guard.

#### 2.1.2 Form 1 — scalar

```yaml
t_ground: {value: 290.0, unit: K}
t_ground: "290 K"                # shorthand, same grammar
efficiency: 0.97                 # bare number: key marked `dimensionless`
n_bits: 12                       # bare number: key marked `count`, delivered as int
```

#### 2.1.3 Form 2 — array spec

```yaml
{zeros: [3, n_freq], unit: K}
{ones:  [n_freq],    unit: dimensionless}
{full:  {shape: [n_source, n_freq], value: 0.05}, unit: dimensionless}
{list:  [300.0, 400.0, 1200.0], unit: K}
{list:  [[1.0, 2.0], [3.0, 4.0]], unit: dimensionless}     # N-D literal, v1
{linspace: {start: 60.0, stop: 85.0, num: 8, endpoint: true}, unit: MHz}
{arange:   {start: 0.0, step: 2.0, num: 64}, unit: s}
{modulo:   {num: n_time, period: 7}, unit: count}
{from_grid: freq}                                          # the run's own axis, values
{basis_fit: {basis: {ref: resources.bases.t_ant}, field: <value node>}}
```

**Shape symbols (new in v1).** Any integer position in a shape, and any `num:`, accepts a symbol from a **closed** table, optionally with an integer offset or an integer multiple:

| symbol | resolves to | source |
|---|---|---|
| `n_time` | `len(observation.time.grid)` | the run's own axis |
| `n_freq` | `len(observation.freq.grid)` | the run's own axis |
| `n_source` | `len(observation.switching.order)` | §4.1.5 |
| `n_pix` | `12 * nside**2` of the named beam/sky | resource |
| `n_alm` | `(lmax+1)(lmax+2)/2` of the named projector | resource |
| `n_load` | `n_source - 1` | derived |

Legal arithmetic on a symbol: `n_freq - 1`, `n_freq + 1`, `2 * n_time`. Nothing else. This is *not* an expression language — it is a symbol table with an integer offset, and it exists because `n_freq` appeared five times by hand in one 90-line stress port (`examples/radio_digital_twin.py:71-78`) with nothing tying the copies together. Check A41 warns when a bare integer in a shape equals `n_freq` or `n_time`, naming the symbol it should have been.

`endpoint:` remains **required** on `linspace` — the sidereal-turn contract is decided by exactly that key.

#### 2.1.4 Form 3 — draw (new in v1)

```yaml
{normal:  {shape: [n_freq, n_pix], loc: 100.0, scale: 20.0, seed: {from: runtime.seeds.sky_structure}}, unit: K}
{uniform: {shape: [n_time], low: 0.0, high: 1.0, seed: {from: runtime.seeds.jitter}}}
```

`loc` and `scale` are themselves value nodes. `seed:` is **required** and must name an entry of `runtime.seeds` (not a literal), so every realisation in the run is enumerated in one place and lands in `provenance.json`. A random draw is a *generator*, not a computation: it has no operands to compose and no result to feed back, so it does not open the door the escape-hatch rule closes.

Three of the five stress ports were blocked on this (`examples/driftscan_mmode.py:61`, `examples/sky_to_noise_wave.py:109`, `examples/three_ways_to_a_posterior.py:76`), and the alternative — ship a binary blob whose provenance the config cannot state — defeats §4.8's whole argument.

#### 2.1.5 Form 4 — file reference

```yaml
{file: {path: <str>, format: <fmt>, unit: <str>, ...format keys...}}
```

`path` resolves relative to the directory of the YAML that mentions it, then against `paths.roots`, then absolute. `~` and `${ENV}` expand.

| `format` | reader | extra keys |
|---|---|---|
| `npy` | `numpy.load` | — |
| `npz` | `numpy.load` | `key:` (required) |
| `txt` | `numpy.loadtxt` | `column:`, `columns:`, `skiprows:` |
| `csv` | `numpy.genfromtxt` | `columns:` (named), `delimiter:` |
| `touchstone` | `radio/touchstone.py` `read_touchstone` + `interpolate_onto` | `flipped:`, `component: s11\|s12\|s21\|s22`, `allow_extrapolation:`, `onto: freq` |
| `cst_dir` | `radio/beams.py:114` `cst_beam_maps(directory, freq_hz, *, nside, suffix=".txt", phi0_deg=0.0, phi_sense="ccw")` | `suffix:`, `nside:` req, `phi0_deg:` **req, no default**, `phi_sense:` **req, no default** |
| `healpix` | `PACKAGE CHANGE` — needed by `MapSky` | `nside:`, `order: ring\|nested`, `frame:`, `freq_key:` |
| `rhino_hdf5` | `radio/rhino.py:469` `read_rhino_observation(path, *, freq_unit, thermistor_columns=None, settle_seconds=5.0, thermistor_unit="celsius")` | `freq_unit:` **req, no default** |
| `eqx_leaves` | `eqx.tree_serialise_leaves` | template rebuilt from the spec |

Every file reference may carry `sha256:`; if absent it is computed and recorded (§4.8).

#### 2.1.6 Form 5 — reference

```yaml
{ref: resources.projectors.rhino_drift}
{ref: resources.beams.rhino_horn.sky_fraction}      # a named SUB-VALUE, v1
{ref: resources.projectors.rhino_drift.beam_alms}   # a traced leaf of a resource, v1
{ref: resources.arrays.gamma_antenna}
{ref: observation.freq.grid}
{ref: inference.observed}                            # the observed data, v1
```

`ref` resolves to the **same Python object**, not a copy. v1 widens it in three directions, each forced by a measured failure:

* **Named sub-values.** `radio/beams.py:150` `horizon_truncated_beam` returns `(truncated_maps, sky_fraction)` — one call, two products — and `sky_fraction` is exactly what `BeamSpillOperator(sky_fraction=...)` wants. v0 consumed the maps and discarded the fraction, leaving the user with `from: projector`, which on a truncated beam returns ≈1.0 and silently deletes the `(1 - f_sky) * T_ground` term.
* **Traced leaves of a constructed resource.** `examples/driftscan_mmode.py:84` hands `drift.beam_alms` to the general-pointing engine so both engines see the *same* alms; v0 would have re-analysed the beam with a different transform and destroyed the 2e-16 agreement the comparison exists to demonstrate.
* **`inference.observed`.** `examples/sky_to_noise_wave.py:286` builds `noise_std = observed / sqrt(delta_nu * t_int)` — a decided sigma frozen at the data. Without a ref target there is nothing to point at.

`config.resolved.yaml` emits a `shared_objects:` map naming which resources ended up as one object, so identity is visible in the artefact rather than only in this spec.

#### 2.1.7 Form 6 — derivation

```yaml
{from: <name>, ...arguments...}
```

A **closed** registry, one entry per package function. v1 registry:

| name | maps to | arguments |
|---|---|---|
| `horizon_fraction` | `DriftScanProjector.horizon_fraction()` | `projector: {ref}` |
| `unit_mean_free` | `radio/instrument/receiver.py:103` | `bandpass: <value node>` |
| `basis_matrix` | `core/basis.py:177 basis_matrix(kind, *, n, n_basis)` | `kind:`, `n_basis:`, axis implied |
| `interpolate_onto` | `radio/touchstone.py interpolate_onto` | `of: {ref}`, `onto: freq` |
| `thermistor_column` | `radio/rhino.py:725 cal_load_operators` | `label:` |
| `channel_spacing` | `median(abs(diff(coords.freq)))` | `times: <float> = 1.0` |
| `sample_cadence` | `median(abs(diff(coords.time)))` | `times: <float> = 1.0` |
| `site` | `observation.site.lat_deg` etc. | — |
| `pointing` | `observation.pointing.*` | — |
| `observation` | `observation.time.*`, `observation.freq.*` | — |
| `model` | the leaf a `model` key holds | `path: <path string>` |

`channel_spacing` and `sample_cadence` are new and are not conveniences. `CWCalibrationOperator` floors `line_width` at `MIN_WIDTH_IN_CHANNELS * median(|diff(freq)|)` and check A13 already forces the key to be written with no default; v0 then handed the user the arithmetic (`25e6 / (N_FREQ - 1)` = `806451.6129032258`) that the code declines to guess. Rounding it to `0.8 MHz` is refused at trace time; rounding the other way silently mis-sizes the protection mask.

`{from: model, path: adc.scale}` closes the other measured drift: `inference.noise.sigma` restating `adc.scale * noise.sigma` by hand (`examples/gibbs_plan.py:187`). It is a reference, not arithmetic — the product itself, if needed, goes through `scale:` (§2.1.10) or the escape hatch.

#### 2.1.8 Form 7 — stack

```yaml
{stack: [<value node>, <value node>, ...], axis: 0}
{from_switch_order: {resource: resources.s_params, part: re}}   # sugar, now DEFINED
```

v0's canonical example used `from_switch_order` and §2 never defined it — for `gamma_src`, the one leaf whose row transposition is "shape-legal and costs tens of kelvin" (`radio/instrument/noise_wave.py:83-86`). v1 defines it as sugar for `{stack: [...]}` over the entries named by `observation.switching.order`, **matched by name, not by position**, refusing a missing or extra label. Check A15 becomes a structural consequence.

`stack` is a container, not a computation: it has no operator and no result type other than "one more axis".

#### 2.1.9 Form 8 — the escape hatch

See §2.3.

#### 2.1.10 Modifiers (legal on every form)

| modifier | meaning |
|---|---|
| `unit:` | the value's unit; converted to canonical on read (§4.9) |
| `dtype:` | `float32\|float64\|complex64\|complex128`, default from `runtime` |
| `as:` | `traced\|static_int\|static_float\|static_str`, normally inferred (§2.1.1) |
| `axis:` | `time\|freq\|none` — mandatory for a 1-D noise sigma (`inference/noise.py` `check_noise_std_axis`) |
| `column: true` | force `(n,)` to `(n, 1)`, the only shape `CalLoadOperator.t_load` reads as per-sample |
| `scale:` / `offset:` | affine relabel: `scale * v + offset`. One level. Not composable into an expression. |
| `part:` | `re\|im\|abs\|angle` of a complex value node |
| `normalize:` | `none\|mean1\|pixel_sum\|max1` — a **declaration of convention**, not a computation |

`normalize:` earns its place with a measurement. On `examples/driftscan_mmode.py`'s own construction at nside 8 / lmax 23, a unit-pixel-sum beam with `normalize_beam: false` gives a mean of 100.42 K against a 99.79 K sky — a temperature. The schema's A12 records 32838 K vs 200 K for an *unnormalised* beam with the same flag. The output's unit is decided by the **pair** (beam normalisation, `normalize_beam`), and v0 required the second half to be written while offering no way to state the first.

`scale:`/`offset:` is the boundary of what the value grammar computes. `250 + 20*linspace(-1,1,8)` is `{linspace: {start: -1.0, stop: 1.0, num: 8, endpoint: true}, scale: 20.0, offset: 250.0, unit: K}` — the physical statement survives. `30*cos(linspace(0,3,8))` does **not**, and goes to `python:` (§2.3, §7).

### 2.2 THE PATH GRAMMAR

```
path      := head ( "." step )*
head      := node_id                      # assembly[node_id]
           | node_id "_" INT              # k-th instance of a `many` node
           | region_key                   # a multi-node `at:` region  -> see below
           | node_id "." stage_name       # a composite node's named stage  -> §4.5.4
step      := IDENT                        # getattr
           | IDENT "[" INT "]"            # getattr then sequence getitem
           | "[" INT "]"                  # sequence getitem
```

Resolution mirrors `ParameterSpace._resolve_targets`: compile the head to `assembly[head]` (`core/graph.py` `Assembly.__getitem__`, breadth-first `_find_named`), later steps to `getattr`/`getitem`; evaluate against `jax.tree_util.tree_map_with_path` over the twin; require a leaf path.

**Region addressing (new statement, v0 had this wrong).** `At` with a tuple of node ids is "addressed by their LAST covered node id in the assembly" (`core/graph.py:127-133`). v0 let a user write `my_stage: {at: [noise_wave, cw_tone, bandpass]}` and then defined the path head as `assembly[node_id]`, so `into: my_stage.field` would never resolve and the failure is a plain name error, not the "landed on static configuration" message §3.2 promised. **v1 requires the config key of a multi-node `at:` claim to equal `at[-1]`** (check A47), so the config key and the addressable name are the same string by construction.

Six refusals, all at validation time:

1. the path does not reach an array leaf → refuse with the package's own wording;
2. the head names a `many` node with more than one instance and no `_k` suffix (`AmbiguousNodeError`);
3. the head names an aliased node (`_reject_aliased_targets`) — always empty for `RADIO_GRAPH`, bites user graphs;
4. two bindings resolve to the same leaf;
5. a `twin.replace` targets a leaf a binding also targets (check B8);
6. the head is a region key that is not `at[-1]`.

**Multi-target.**

```yaml
into: [noise_wave.t_unc, noise_wave.t_cos, noise_wave.t_sin]
fan: distribute        # REQUIRED whenever `into` has more than one entry
```

The code's `Bind(fan=None)` infers broadcast-vs-distribute from a Python container type (`inference/parameters.py:277` — measured at a factor of 2.5 on two scalar leaves). A YAML file has no container type to infer from, so **the schema is stricter than the code**.

### 2.3 THE ONE ESCAPE HATCH

```yaml
python: "package.module:attribute"
args: {<name>: <value node>, ...}      # EVERY arg value is a value node (§2.1)
literal: {<name>: <any yaml>}          # forwarded verbatim, not resolved
```

**`args` values are value nodes.** v0 said "forwarded as keyword arguments" and never said whether the value grammar applied inside them — which mattered, because for three of the five stress ports `python:` was the *only* route to the sky and an unresolved `{file: ...}` would be forwarded as a literal dict. v1 states it, and adds `literal:` for the rare mapping that must pass through untouched.

**Permitted at exactly nine sites** (seven in v0, plus two the ports proved necessary):

1. `model.<node>.python` — a custom `AbstractOperator` class
2. any **value node**
3. `resources.sky_models.<name>.python` — an `AbstractSkyModel` subclass
4. `resources.beams.<name>.python` — **new**: returns `(n_freq, n_pix)` linear power. `examples/driftscan_mmode.py:57-60` and `examples/sky_to_noise_wave.py:104-107` both build a beam analytically, and v0 excluded the beams block from the hatch, so *both scripts died before step 1*.
5. any operator field declared `Callable` **and** `static=True` — **new**: `LambdaOperator.fn` (`core/operator.py:104`) is the shipped case, exported from the top-level package, and v0 accidentally forbade configuring it.
6. `inference.parameters.<name>.transform.python` — the `Bind.fn`
7. `inference.parameters.<name>.prior.python` — a distribution factory
8. `inference.bind.python` — the whole-space `ParameterSpace.raw_bind`
9. `runs[].loss.python` — a `loss_fn` for the calibrators

`plugins: [pkg.mod]` is **not** a second hatch: it only imports modules so registry names resolve. `model.<node>.extra` is **not** one either: it forwards keyword arguments to an already-selected third-party class (`MomentRFI.IterativeSurfaceFitter`, `radio/backend/flagging.py:124`), it is `eqx.field(static=True, converter=FrozenMapping)`, and it is restricted to hashable scalars.

**Stated cost.** The run is no longer reproducible from the config alone — the hash covers the *string*, not the code. The object must be importable in the run process and, inside the gradient path, jax-traceable. If it lands on a static field it is hashed by identity, so an equivalent re-created function misses the jit cache. With `raw_bind`, `ParameterSpace.validate`'s per-selector checks cannot run at all.

**What is routed here, deliberately.** Elementwise arithmetic beyond `scale`/`offset`; trig, powers, logs on arrays; nested function calls with positional operands; user-defined pipeline topologies outside `RADIO_GRAPH` and outside `model.kind: pipeline`. Composition of *named* quantities is available without the hatch via `resources.arrays.<name>` + `{ref: ...}` (§4.4.1), and that is deliberately as far as it goes.

---

## 3. Top-level section map

| section | required? | what it specifies | maps to |
|---|---|---|---|
| `schema_version` | **yes** | `1` | loader dispatch |
| `defaults` | no (`[]`) | preset layering, in order, user last | §5 `recursive_update` |
| `plugins` | no (`[]`) | dotted modules imported so registry names resolve | `importlib` |
| `runtime` | **yes** | dtype, platform, seeds — part of the hashed config | `jax.config`, `jax.random.key` |
| `observation` | **yes** | axes, site, pointing, switching, environment, meta, aux | `State(coords=Coordinates(...), env=Environment(...), meta=..., key=...)` |
| `resources` | no (`{}`) | named reusable objects: beams, projectors, sky models, bases, s-params, arrays | constructed once; `ref` is identity |
| `model` | **yes** | the instrument, node by node | `rheplicant.radio.assemble(*operators)` (or `Pipeline`, §4.5.5) |
| `variants` | no (`{}`) | named deep-merge patches over the base document | re-assembled twins |
| `inference` | no | twin repair, observed, latents, noise, checks, truth | `ParameterSpace`, `NoiseModel` |
| `runs` | **yes** | list of exits (or one mapping) | `SamplingPlan`, `wiener_solve`, `MCMC`, calibrators, … |
| `outputs` | no (`{stdout: summary}`) | what is written and where | new surface |
| `campaign` | no | **RESERVED, capability 4** — refused in v1 by name | §8.2 |

Twelve sections. Maximum *physics* nesting depth is three (`model.<node>.<field>`, `resources.<kind>.<name>`); anything deeper is the value grammar.

**Every section except `schema_version`, `runtime`, `observation`, `model` and `runs` is optional, and optionality is marked per key in §4.** v0 marked none of `site`, `pointing`, `switching`, `time.epoch`, `time.integration_time`, `time.channel_width`, `resources` or `inference.noise` optional, which pushed the user to invent a latitude, a pointing mode, a switch cycle and a noise sigma for a run that has none — four fabricated facts that then land in `config.resolved.yaml` and `provenance.json` as if they were measurements. A config that lies about what the run knew is worse than a missing key.

---

## 4. Section by section

### 4.0 `runtime`

| key | type | unit | req | default | Python target |
|---|---|---|---|---|---|
| `jax_enable_x64` | bool | — | no | `false` | `jax.config.update("jax_enable_x64", ...)`, applied before any array exists |
| `platform` | `cpu\|gpu\|tpu\|auto` | — | no | `auto` | `jax.config.update("jax_platform_name", ...)` |
| `seed` | int or `null` | — | conditional | — | `State.key = jax.random.key(seed)`; `null` means `State.key = None` |
| `seeds.<name>` | int | — | no | — | **open namespace** of user-named ints, referenced as `{from: runtime.seeds.<name>}` |
| `x64_required_by` | list[str] | — | auto | — | emitted, not written |

**`seed` is required only when the run realises randomness** — `model` lights a stochastic node, `observed.from: simulation` with a declared realisation, or a run kind in `{plan.sample, conjugate.gcr, nuts, npe}`. Otherwise `seed: null` is legal and recorded. `examples/sky_to_noise_wave.py:233,275` builds `State` with no key at all, and forcing one is not free: with `State.key` set, `SelectOperator.__call__` splits a subkey per branch on every call, so the config-built run traverses a different PRNG path than the script.

**`seeds` is an open namespace.** v0 named exactly four (`sample`, `gcr`, `npe_bank`, `surrogate_init`); `examples/three_ways_to_a_posterior.py` consumes ten distinct keys. v0's fallback — `jax.random.fold_in(key, hash(name))` — depends on Python's salted string hash and is not reproducible across processes. v1 replaces it: an unnamed seed is `fold_in(key, int.from_bytes(blake2s(name, digest_size=4).digest()))`.

**x64 is physics.** It is process-global and part of the hashed config. New check A44: any `resources.projectors` entry with `engine: driftscan` or `engine: general_pointing` **requires** `jax_enable_x64: true` or an explicit `acknowledge_float32_sky: true` on that entry, quoting `radio/sky/general_pointing.py:28-32` — "the map↔alm steps carry O(10%) errors in float32". A 10% error on the beam-weighted sky is larger than every effect `normalize_beam`, `phi0_deg` and `phi_sense` are required keys for, and it is invisible: the maps come back finite, correctly shaped and plausibly structured.

### 4.1 `observation`

Two mutually exclusive forms: **synthetic** (declare the axes) and **ingested** (declare the file).

#### 4.1.1 Synthetic form

| key | type | unit | req | default | Python target |
|---|---|---|---|---|---|
| `meta.<k>` | hashable scalar | — | no | `{}` | `State.meta` (`FrozenMapping`, STATIC — part of the jit cache key) |
| `freq.grid` | value node | Hz | **yes** | — | `Coordinates.freq`, `(n_freq,)` |
| `time.grid` | value node | s | **yes** | — | `Coordinates.time`, `(n_time,)`, **relative to run start** |
| `time.epoch` | scalar | `unix_s` | no | — | `meta["time_epoch_unix_s"]` |
| `time.integration_time` | scalar | s | conditional | — | fans out; **required iff** `inference.noise.kind: radiometer*` |
| `time.channel_width` | scalar | Hz | conditional | — | fans out; same condition |
| `site.lat_deg` | scalar | deg | conditional | — | fans to every projector's `lat_deg`; **required iff** a projector reads `{from: site}` |
| `site.lon_deg` | scalar | deg | no | — | **recorded only** — nothing in `src/` consumes it |
| `site.alt_m` | scalar | m | no | — | **recorded only** |
| `pointing` | mapping | — | no | `{mode: none}` | §4.1.4 |
| `switching` | mapping | — | no | `{mode: none}` | §4.1.5 |
| `environment.temperature` | value node | K | no | — | `Environment.temperature` (traced; read by `ground_pickup`) |
| `environment.humidity` | value node | *unstated in source* | no | — | `Environment.humidity` — config must declare the unit |
| `environment.extra.<k>` | value node | — | no | `{}` | `Environment.extra` |
| `extra.<key>` | value node | — | no | `{}` | **new**: writes `Coordinates.extra[key]` directly |
| `aux.flags` | value node | bool | no | — | `State.aux["flags"]`, `(n_time, n_freq)`, **TRUE = BAD** |
| `data` | value node | — | no | — | `State.data`. FORBIDDEN when `model` lights any source node |

`meta` values must be hashable scalars (`core/frozen.py`); a YAML list becomes a tuple. Every `meta` value is part of the jit cache key, so a per-observation `obs_id` costs one recompilation per recording.

`time.grid` must be **seconds from the start of the run**; the epoch goes in `meta`. This is not style — `core/coordinates.py:66` sets `MAX_TIME_RESOLUTION_IN_SAMPLES = 1e-2` and the module documents the float32 failure it prevents (78 s of error out of a 100 s cadence, nothing raised, every shape right).

**`observation.extra` is new** and closes a real hole. `noise_wave.switch_key` is a config key (`radio/instrument/noise_wave.py:214`) and v0 had only one producer of `coords.extra` — `switching`, which writes `receiver_input` and nothing else. Writing `switch_key: my_switch` produced a run that raised after everything upstream was built. `coords.extra` is the package's general side-channel (`lst_deg`, `selfrot_deg`, `receiver_input`), and `Coordinates.extra` is an open `dict[str, Any]` (`core/coordinates.py:194`). Check A45: every `switch_key` in `model` must name `receiver_input` or a key of `observation.extra`.

#### 4.1.2 Ingested form

```yaml
observation:
  from_file:
    format: rhino_hdf5
    path: 2026-05-11_rhino.hd5f
    freq_unit: MHz                    # REQUIRED, no default (radio/rhino.py:469)
    thermistor_columns: {ambient: 0, hot: 1}
    settle_seconds: {value: 5.0, unit: s}
    thermistor_unit: celsius
  pointing: {...}                     # to_state sets none of these
  environment: {...}
```

`to_state` (`radio/rhino.py:602`) sets `data`, `coords.time` (relative), `coords.freq`, `coords.extra["receiver_input"]`, `aux["flags"]`, `meta["time_epoch_unix_s"]` — and no env, no key, no pointing, no `lst_deg`. Those are declared alongside and merged with `State.replace` / `Coordinates.replace`, which re-validate.

`source_order` is **derived**, never written: the loader reads `assembly["receiver_input"].names` off the twin, exactly as `to_state`'s docstring instructs, and passes that. See §4.1.5 for why the user's labels are no longer claimed to equal it.

#### 4.1.3 `observation.aux` and the flags convention

`FlaggingOperator` declares `provides = ("aux.flags",)` and leaves `data` untouched (`radio/backend/flagging.py`), so the whole product of the `flagging` node is `aux["flags"]`. v0 could write `predicted` and `observed` (which are `.data`) and had no key for aux at all — a config that lit `flagging` could save nothing that node produced. Fixed in `outputs` (§4.8).

#### 4.1.4 `observation.pointing`

Discriminated on `mode`, with `none` a first-class value.

```yaml
pointing: {mode: none}          # DEFAULT. No projector in this run.
```

```yaml
pointing:
  mode: drift                   # -> DriftScanProjector's static az/el/selfrot
  az_deg: {value: 0.0, unit: deg}
  el_deg: {value: 90.0, unit: deg}
  selfrot_deg: {value: 0.0, unit: deg}
  materialise: [pointing, selfrot_deg]   # which Coordinates fields the loader writes
  lst:
    mode: uniform_turn          # -> DriftScanProjector.uniform_lst_grid(n_time, lst0)
    n_time: n_time
    lst0_deg: {value: 0.0, unit: deg}
  # or: lst: {from_file: {...}} -> coords.extra["lst_deg"], (n_time,) deg
```

```yaml
pointing:
  mode: tracked                 # -> GeneralPointingProjector
  table: {file: {path: scan.csv, format: csv, columns: [az_deg, el_deg]}}
  lst: {from_file: {...}}       # REQUIRED
  selfrot: {from_file: {...}}   # optional, defaults to zeros
```

```yaml
pointing:
  mode: baked                   # -> MatrixProjector; geometry is inside the matrix
  provenance: {built_by: ..., lat_deg: ..., lst_range_deg: ...}   # REQUIRED, unverifiable
```

**`materialise:` is new and is written, not inferred.** v0 said the loader derives `Coordinates.pointing` from az/el "so they cannot disagree" and said nothing about `selfrot_deg`. It is read on both sides and read differently: `DriftScanProjector._reject_disagreeing_pointing` checks `coords.extra["selfrot_deg"]` against its static value *only if present*, while `GeneralPointingProjector._zyz` defaults it to zeros if absent. A loader that guesses either way silently changes which observation a cross-engine comparison simulated. Now it is a key, and it lands in `config.resolved.yaml`.

`coords.extra["lst_deg"]` in **degrees** is required by both real engines and is produced by nothing in the package except `uniform_lst_grid`. `lon_deg` and `alt_m` remain inert: there is no UTC → LST bridge in `src/` (open question §11.4).

#### 4.1.5 `observation.switching`

```yaml
switching: {mode: none}         # DEFAULT
```

```yaml
switching:
  mode: cycle
  order: [antenna, ambient, hot, noise_source]   # index 0 MUST be the literal `antenna`
  cycle: round_robin                             # | from_file | none
  dwell: {value: 1, unit: samples}
  # or: index: {file: {...}}                     # explicit (n_time,) int array
```

Compiles to `coords.extra["receiver_input"]`, an integer `(n_time,)` array. Index 0 is the antenna chain, 1…n the `cal_loads` in the order `model.cal_loads` gives them.

**The label fiction is corrected.** v0's checks A14 (`order[0] == "antenna"`) and B5 (`order` must equal `assembly["receiver_input"].names`) are **mutually unsatisfiable against the real code**: the assembled names are derived from graph node ids, and on `examples/gibbs_plan.py`'s chain they are `('astro_sum', 'cal_loads_1', 'cal_loads_2', 'cal_loads_3')`. No user label reaches them. v1's rule:

* `order[0]` is the reserved literal `antenna`; `order[1:]` are the **keys of `model.cal_loads`**, in switch order.
* `switching.order` fixes, in one place: the switch indices, the order of `model.cal_loads`, the row order of `noise_wave.gamma_src`, and the `thermistor_columns` labels for an ingested run.
* B5 is restated as a **positional** check: `len(assembly["receiver_input"].names) == len(order)` and the k-th assembled name corresponds to the k-th declared label. The correspondence is recorded in `config.resolved.yaml` as `switch_map: {0: [antenna, astro_sum], 1: [ambient, cal_loads_1], ...}` so the user can read what actually happened.
* `mode: none` requires that `model` declares no `cal_loads` **and** that `noise_wave.gamma_src` has exactly one row. This is a first-class state of the code — `NoiseWaveOperator._source_index` returns all-zeros exactly when `n_source == 1` and the switch key is absent — and v0 could only express it by accident.

### 4.2 `observation` ⇄ `inference` fan-out

`observation.time.integration_time` and `channel_width` are declared once and fan to `RadiometerNoise.channel_width` / `.integration_time` (both `eqx.field(static=True)`, `inference/noise.py:119`). They are cross-checked against `median(diff(coords.time))` (`{from: sample_cadence}`) and against **both** `band/n_freq` and `median(|diff(coords.freq)|)`.

The second half of that check is new and catches a real inconsistency v0 passed silently: `examples/sky_to_noise_wave.py` declares `channel_width = 25e6/8 = 3.125 MHz` while `linspace(60, 85, 8)` with the endpoint included has a spacing of `25/7 = 3.571 MHz`. v0's cross-check was against `band/n_freq` only, so it passed. v1 warns beyond a few percent and names the endpoint convention as the usual cause.

`observation.site.lat_deg` fans to every projector's `lat_deg`, which today is duplicated per projector with nothing checking agreement.

### 4.3 `sky` — where it lives

There is no top-level `sky` section, deliberately. A sky in this package is two independent facts that are separately reusable and separately expensive:

* **what the sky is** → `resources.sky_models.<name>` (§4.4.3)
* **how it is seen** → `resources.projectors.<name>` (§4.4.4), which holds the beam

and they meet at exactly one node, `model.observed_astro_sky`, whose two fields are `sky_model` and `projector` (`radio/sky/source.py:25`). Giving the sky its own top-level section would put the beam in two places, which is the coupling `{ref: ...}` object identity exists to prevent (`radio/instrument/beam_spill.py:89` `from_projector` — "the one call that cannot get the weight and the sky average out of step").

### 4.4 `resources`

`resources.<kind>.<name>`, referenced as `{ref: resources.<kind>.<name>}`. Each named entry is constructed **once**. Any entry may carry `extends: <name>` (deep merge over a sibling of the same kind, mappings merge / lists replace) — `examples/driftscan_mmode.py` declares three projectors differing in two keys and v0 made each restate nine.

#### 4.4.1 `resources.arrays.<name>` — new, and load-bearing

```yaml
resources:
  arrays:
    open_8:      {python: "rhino_cal_jax:termination_gamma", args: {kind: open, n: n_freq}}
    gamma_ant:   {python: "rhino_cal_jax:cable_gamma",
                  args: {gamma_termination: {ref: resources.arrays.open_8},
                         freq: {from_grid: freq},
                         length: {value: 2.0, unit: m},
                         loss: 0.92}}
    sky_structure: {normal: {shape: [n_pix], loc: 0.0, scale: 1.0,
                             seed: {from: runtime.seeds.sky_structure}}}
```

Any value node may be **named**. This is the whole of v1's answer to "the schema cannot express `f(g(x), y)`": composition is by naming, not by nesting. It is a let-binding, not an expression language — there is no operator, no precedence, no evaluation order to reason about, and the resulting DAG is exactly the `ref` graph the resources section already is.

This unblocks `examples/gibbs_plan.py:112-119` and `examples/sky_to_noise_wave.py:158-165` (seven reflection coefficients built by nested `rhino_cal_jax` calls), and it gives the genuinely-shared, genuinely-expensive object of that script a home — v0's `resources` section was empty for it.

`resources.arrays` entries support the full modifier set, so `{ref: resources.arrays.gamma_ant, part: re}` is the real part.

#### 4.4.2 `resources.beams.<name>`

A beam is not an object in this package: it is a raw `(n_freq, n_pix)` array. The block declares the form plus the conventions the file cannot carry.

| key | type | unit | req | default | notes |
|---|---|---|---|---|---|
| `format` | `cst \| npy \| npz \| inline \| gaussian \| python` | — | **yes** | — | `inline`, `gaussian`, `python` are new |
| `directory` / `path` | path | — | for `cst`/`npy`/`npz` | — | `radio/beams.py:114` |
| `maps` | value node | dimensionless | for `inline` | — | **new** — a computed beam |
| `fwhm_deg` / `sigma_deg` | value node | deg | for `gaussian` | — | **new** — scalar or `(n_freq,)` |
| `nside` | int | count | **yes** | — | HEALPix RING |
| `suffix` | str | — | no | `".txt"` | `cst` only |
| `phi0_deg` | scalar | deg | **yes for `cst`** | *none* | a fact about the as-built horn |
| `phi_sense` | `ccw \| cw` | — | **yes for `cst`** | *none* | handedness |
| `frame` | `beam_local \| reference` | — | for raw arrays | `beam_local` | declared, unverifiable |
| `quantity` | `linear_power` | — | no | `linear_power` | `cst_beam_maps` returns `10**(dBi/10)`, unnormalised |
| `normalize` | `none \| pixel_sum \| solid_angle` | — | **yes** | *none* | **new** — see below |
| `horizon.mode` | `none \| truncate_map \| projector_mask` | — | no | `none` | `truncate_map` → `radio/beams.py:150` |
| `horizon.el_deg` | scalar | deg | no | `90` | `truncate_map` accepts **only 90** |
| `horizon.apod_deg` | scalar | deg | no | `0.0` | cosine apodisation |

**Sub-values exposed:** `.maps` and `.sky_fraction` (the second return of `horizon_truncated_beam`, `radio/beams.py:150`, shape `(n_freq,)`).

**`format: gaussian` and `format: inline` are not conveniences.** Two of the five ported examples build the beam analytically and are *designed* to run without the unpublished CST dataset; under v0 neither was runnable, so the first thing a new user does — run a shipped example on a machine with no beam files — failed.

**`phi0_deg` and `phi_sense` are required for `format: cst` only.** They describe how a CST export's azimuth maps onto the beam-local chart (`radio/beams.py`) and are meaningless for a raw HEALPix array or a synthesised map; v0 required them for every format, which is exactly the invent-a-value habit the requirement exists to break. For raw-array formats `frame:` is required instead — for a raw array that is the genuinely unverifiable fact.

**`normalize:` is required with no default**, for the same reason `normalize_beam` is: the output's unit is decided by the pair. A user who lifts `normalize_beam: false` from a preset built for a unit-sum beam and applies it to a raw CST beam is off by ~1.6e4 with every shape correct.

#### 4.4.3 `resources.sky_models.<name>`

Discriminated on `kind`.

| `kind` | maps to | keys |
|---|---|---|
| `uniform` | `radio/sky/model.py:39 UniformSkyModel` | `amplitude` (K), `n_pix` |
| `power_law` | `radio/sky/model.py:58 PowerLawSkyModel` | `amplitude` (scalar or `(n_pix,)`, K), `spectral_index`, `ref_freq` (Hz, **static**), `n_pix` |
| `maps` | `PACKAGE CHANGE` — `MapSky` | `maps: <value node>` **or** `{file: {format: healpix}}`, `freq: {from_grid: freq}`, `nside`, `order`, `frame`, `unit: K` |
| `python` | the escape hatch | `python:`, `args:` |

**`MapSky` must ship.** Verified: `src/rheplicant/radio/sky/model.py` contains `AbstractSkyModel` (`:27`), `UniformSkyModel` (`:39`) and `PowerLawSkyModel` (`:58`) and nothing else. `MapSky` is re-declared by hand in `examples/driftscan_mmode.py:45`, `examples/inferring_anything.py:54`, `examples/sky_to_noise_wave.py:88`, `examples/tutorial_nuts.py:54`, `docs/_generate_receiver_figures.py:109`, `docs/_generate_inference_figures.py:93` and three docs pages. It is the sky model every example uses and it is not in the package.

v1 decides the v0 open question: **ship it, with two constructors** — `maps: <value node>` as well as a file. The maps in `examples/driftscan_mmode.py:61` and `examples/sky_to_noise_wave.py:109-112` are *drawn*, not read, and will never be on disk. The alternative — force every real sky through `python:` — makes the most common configuration in the project also the least reproducible, and in one case requires importing an example module whose import re-executes an argparse block and twenty prints.

Validation must state that `MapSky.__call__` ignores its `freq` argument, so a `(n_freq, n_pix)` array declared against a different grid is the silent failure the constructor invites: check C5 requires `n_pix == 12*nside**2` and `maps.shape[0] == n_freq`.

#### 4.4.4 `resources.projectors.<name>`

Discriminated on `engine`. **Field names are the Python names, verbatim** — this is the v0 contradiction removed.

```yaml
engine: driftscan                # -> radio/sky/driftscan.py
beam: {ref: resources.beams.rhino_horn}    # -> from_beam_maps(), the CORRECT analysis
lmax: 191
nside: <int>                     # omitted on the from_beam_maps path (inferred)
lat_deg: {from: site}
az_deg: {from: pointing}
el_deg: {from: pointing}
selfrot_deg: {from: pointing}
normalize_beam: <bool>           # REQUIRED, no default
horizon_mask: false
apod_deg: {value: 0.0, unit: deg}
mask_iterations: 3               # the MASK re-analysis; requires horizon_mask: true
beam_iterations: 3               # NEW: from_beam_maps(iterations=) — the BEAM analysis
lst_ref_deg: {unit: deg}         # REQUIRED iff optimizations contains cache_beam_rotation
uniform_sampling: false
freq_chunk: null
optimizations: [read_horizon_fraction, cache_beam_rotation]   # ORDERED
acknowledge_float32_sky: false   # only route past check A44
```

```yaml
engine: general_pointing         # -> radio/sky/general_pointing.py
beam: {ref: ...}                 # a beam resource -> hp.map2alm, OR
beam_alms: {ref: resources.projectors.drift.beam_alms}   # NEW: share the drift engine's
lmax: 191
nside: 64                        # not inferable here
lat_deg: {from: site}
normalize_beam: <bool>           # REQUIRED
```

```yaml
engine: matrix                   # -> radio/sky/projection.py MatrixProjector
matrix: {file: {path: ..., format: npy}}
provenance: {...}                # REQUIRED
```

Hard rules:

* **`normalize_beam` has no default and must be written.** `false` returns `∫B·T dΩ`, not a temperature: 32838 K vs 200 K on a uniform 200 K sky.
* **`beam_frame` and `beam_ref_lst_deg` are not writable keys.** They are set only by `to_reference_frame()`, expressed as the ordered `optimizations:` list. `__check_init__` exists to catch a hand-set pair; a YAML that wrote them would drive the object into the state that guard exists to catch.
* **`beam_iterations` is new and matters.** `from_beam_maps(iterations=)` is the `map2alm_iter` count of the **beam** analysis and directly sets `beam_alms` — the projector's only traced array (every other field is `eqx.field(static=True)`, `radio/sky/driftscan.py:189-203`). v0 exposed `mask_iterations`, a different field, which a user would reasonably set believing they had tuned the beam. Worse: `iterations` is a constructor argument and **not a stored field**, so it is not recoverable from the built object — `config.resolved.yaml` cannot fill it in and `provenance.json` cannot record it after the fact. The one place it can ever be captured is the config key that did not exist.
* **`lst_ref_deg` is required when `optimizations` contains `cache_beam_rotation`** (check A48). `to_reference_frame()` raises without one, and `__check_init__` refuses `beam_frame="reference"` with `lst_ref_deg=None` — after the beam file has been read and analysed, which is exactly the class of failure §6's "before any expensive work" promise exists to prevent. No silent default: defaulting it to `lst0_deg` re-anchors the m-mode phases.
* **`optimizations` is ordered**, and `read_horizon_fraction` is dropped from it. It is not an optimisation — it *produces* `f_sky`, which `beam_spill` consumes — and it now has exactly one spelling, `{from: horizon_fraction}` (§4.4.2 sub-values are preferred where the beam was truncated). C7 becomes a plain refusal: `{from: horizon_fraction}` against a projector whose `optimizations` contain `cache_beam_rotation` is refused, quoting the code, which raises anyway.

#### 4.4.5 `resources.s_params.<name>`

Discriminated on `kind`.

```yaml
kind: touchstone
file: {path: horn.s1p, format: touchstone}
component: s11
flipped: false                # genuine port reversal — a fact about the VNA wiring
allow_extrapolation: false
onto: freq
```

```yaml
kind: termination             # NEW -> rhino_cal_jax.termination_gamma
termination: open | short | matched | resistive
impedance: {value: 45.0, unit: ohm}      # required for `resistive`
z0: {value: 50.0, unit: ohm}
n: n_freq
```

```yaml
kind: cable                   # NEW -> rhino_cal_jax.cable_gamma
behind: {ref: resources.s_params.antenna_open}
length: {value: 2.0, unit: m}
velocity_factor: 1.0
loss: 0.92
onto: freq
```

Every entry exposes `.re`, `.im`, `.complex`. `kind: termination` and `kind: cable` require the optional `cal` extra (check A35).

`z0` appears here and nowhere else: v0 correctly refused a global reference impedance because `Touchstone.z0` is parsed and never read — but `termination_gamma(z0=)` **is** read, so the key exists exactly where it is consumed.

#### 4.4.6 `resources.bases.<name>`

```yaml
time: {kind: legendre, n_basis: 3}     # n taken from the run's n_time
freq: {kind: legendre, n_basis: 2}     # n taken from the run's n_freq
```

→ `core/basis.py:177 basis_matrix(kind, n=, n_basis=)` then `core/basis.py:294 SeparableBasis(time=, freq=)`. **`n` is never written** — it comes from the grid, which is what makes a basis built for another band impossible. A `file:` route for a design matrix is **refused** (§7).

`kind: fourier` is a different span and a *claim* that the quantity is periodic on that axis; `legendre` and `polynomial` span the same functions and differ only in conditioning.

### 4.5 `model` — the instrument

`model` is discriminated on an optional `kind:`, defaulting to `graph`.

#### 4.5.1 Node spec

```yaml
model:
  <node_id>:
    type: <ClassName>          # required only where >1 class registers at the node
    from: <alt_constructor>    # optional: preset / guarded constructor
    <field>: <value node>      # the operator's own constructor fields, Python names
    at: <node_id | [n1, ..., nk]>   # relocation or contiguous region claim
    compose: cascade | sum     # NEW — several stages at one node
    stages: [{name: ..., type: ..., ...}, ...]   # with `compose`
    snapshot_before: <name>    # NEW — SnapshotOperator ahead of this node
    extra: {...}               # opaque passthrough (flagging only)
    python: "pkg.mod:Class"    # escape hatch
```

`many` nodes take a list (SUM/CHAIN) or a label-keyed mapping (FAN) — §4.5.3.

#### 4.5.2 The node table

All 32 nodes of `RADIO_GRAPH` (`radio/graph.py:97-192`, verified by enumeration). Kinds: `S` source, `T` transform, `J` junction, `X` selector, `R` reserved. `J`/`X` are **not config keys** — `assemble()` refuses them as operator slots.

| node id | kind | list? | class(es) | fields — `deliver` in brackets |
|---|---|---|---|---|
| `global_signal` | S | no | `GlobalSignalOperator` | `depth` K [traced], `centre` Hz [traced], `width` Hz [traced] — Gaussian σ, not FWHM |
| `foregrounds` | S | **yes, SUM** | `ForegroundOperator` | `amplitude` K [traced], `spectral_index` [traced], `ref_freq` Hz [**static_float**, `sky/foregrounds.py:41`] |
| `point_sources` | S | no | `PointSourceOperator` | `level` K [traced] |
| `uniform_sky` | S | no | `SkyOperator` | `amplitude` K [traced] |
| `astro_sum` | J | — | — | **refused as a key** |
| `ionosphere` | T | no | `IonosphereOperator` | `delta` [traced], `ref_freq` Hz [**static_float**, `environment/ionosphere.py:39`] |
| `atmosphere_field` | R | no | *none shipped* | `python:` or `at:` only |
| `ground_field` | R | no | *none shipped* | `python:` or `at:` only |
| `rfi_field` | S | no | `RFIOperator` | `amplitude` K [traced], `occupancy` [**static_float**, `environment/rfi.py:41`]. **STOCHASTIC** |
| `field_sum` | J | — | — | refused |
| `beam` | R | no | *none shipped* | `python:` or `at:` only; the beam belongs to the projector |
| `observed_astro_sky` | S | no | `SkySourceOperator` (`sky/source.py:25`) | `sky_model: {ref}`, `projector: {ref}` |
| `ground_pickup` | S | no | `GroundPickupOperator` | `coupling` [traced], `t_ground` K [traced] — **fallback only** when `env.temperature` is absent |
| `t_sys_extra` | S | **yes, SUM** | `BasisTemperatureOperator` (`t_sys.py:107-111`) | `from: basis` + `basis: {ref}` + `coeff` K `(n_k,n_j)` [traced] |
| `atmosphere` | S | no | `AtmosphericEmissionOperator` | `t_atm` K [traced] |
| `astro_ant_sum` | J | — | — | refused |
| `beam_spill` | T | no | `BeamSpillOperator` (`instrument/beam_spill.py:85`) | `sky_fraction` [traced] + `t_ground` K [traced], **or** `from: projector` + `projector: {ref}` |
| `t_ant_sum` | J | — | — | refused |
| `antenna_loss` | T | no | `AntennaLossOperator` (`:66`) | `efficiency` [traced], `t_physical` K [traced] |
| `cal_loads` | S | **yes, FAN** | `CalLoadOperator` (`calibration.py:648`) | `t_load` K [traced] — scalar, `(n_freq,)`, or `(n_time,1)` with `column: true`; **or** `from: thermistors` + `label:` |
| `receiver_input` | X | — | — | **refused as a key**; the cycle is `observation.switching` |
| `noise_wave` | T | no | `NoiseWaveOperator` (`:204-214`) | `t_unc`,`t_cos`,`t_sin`,`t_rx` K [traced]; `gamma_src_re/im` `(n_source,n_freq)` [traced]; `gamma_rec_re/im` `(n_freq,)` [traced]; `switch_key` [**static_str**, default `receiver_input`] |
| `cw_tone` | T | no | `CWCalibrationOperator` (`calibration.py:358-369`) | `amplitude` K [**static_float**], `tone_freq` Hz [**static_float**], `line_width` Hz [**static_float**, no default], `lineshape` [**static_str**, `sinc2`], `drift_rate` Hz/s [static, 0], `amplitude_drift_rate` 1/s [static, 0], `protect_floor` [static, 1e-2]. **All static — no inferable parameter, by design** |
| `bandpass` | T | no | `ReceiverOperator` (`receiver.py:139`) | `bandpass` [traced]; convention: **mean 1, shape only** |
| `gain` | T | no | `GainOperator` | `gain` [traced] — carries the ABSOLUTE level; unit **unstated in source**, config must declare |
| `noise` | T | no | `NoiseOperator` | `sigma` K [traced]. **STOCHASTIC, additive** |
| `emi` | T | no | `EMIOperator` (`emi.py:38`) | `amplitude` K [traced], `period` channels [**static_int**] |
| `adc` | T | no | `ADCOperator` (`adc.py:31-41`) | `scale` [traced, unit unstated], `n_bits` bits [**static_int**] |
| `flagging` | T | no | `FlaggingOperator` (`flagging.py:69`) **or** `MomentRFIFlaggingOperator` (`:121`) — `type:` **required** | `threshold` [**static_float**, unit unstated] / `extra: {...}` [static mapping] + `kernel_shapes` [static tuple] |
| `averaging` | T | no | `BackendOperator` (`averaging.py:153-155`) | `n_chunk` samples [**static_int**; `n_time % n_chunk == 0`] |
| `apply_cal` | T | no | `ApplyCalibrationOperator` (`calibration.py:765`) | `gain` [traced, unit unstated] |
| `filters` | T | **yes, CHAIN** | `FourierBandFilter` / `SiderealFilter` / `SkySpaceFilter` — `type:` **required** | see below |

`filters` entries (all guard fields static — `radio/filters/fourier.py:40-43`, `sidereal.py:38-39`, `skyspace.py:52-54`):

* `FourierBandFilter`: `axis` [static_int, 0=time, 1=freq], `low`, `high` [static_float, `0 ≤ low < high ≤ 0.5` cycles/sample], `mode` [static_str, `extract\|remove`]
* `SiderealFilter`: `n_days` [static_int, ≥2, `n_time % n_days == 0`], `mode`. Assumes n_days concatenated identical day-major LST grids — surfaced as a warning.
* `SkySpaceFilter`: `projector: {ref}`, `regularization` [traced, unit unstated], `cg_tol` [static_float, 1e-8], `cg_maxiter` [static_int, 100], `mode`

#### 4.5.3 `many` — three meanings, three shapes

```yaml
foregrounds:                       # SUM — order-free
  - {amplitude: {value: 2500.0, unit: K}, spectral_index: 2.55, ref_freq: {value: 70.0, unit: MHz}}

cal_loads:                         # FAN — order IS the switch index, 1..n
  ambient: {t_load: {value: 300.0, unit: K}}
  hot:     {t_load: {value: 400.0, unit: K}}
# keys are observation.switching.order[1:], in that order.

filters:                           # CHAIN — order IS call order
  - {type: FourierBandFilter, axis: 0, low: 0.02, high: 0.5, mode: extract}
```

#### 4.5.4 `compose:` — several stages at one node (new)

`assemble` refuses two operators at a non-`many` node and *names the supported route in the message*: "Compose them explicitly and wrap with `At(...)` if that is intended." v0 held exactly one operator per key, so that route was unwritable — and it is the repository's headline inference demonstration: `examples/inferring_anything.py:96-110` builds two `GainOperator`s so one `log_gain` latent can drive both leaves through `exp`, which is what `fan: broadcast` is *for*.

```yaml
model:
  gain:
    compose: cascade                    # | sum
    stages:
      - {name: gain_lna,     type: GainOperator, gain: {value: 1.0, unit: dimensionless}}
      - {name: gain_backend, type: GainOperator, gain: {value: 1.0, unit: dimensionless}}
```

Compiles to `At("gain", Pipeline(*stages, names=(...)))`. The path grammar's head gains `node_id.stage_name` (`Assembly.__getitem__` descends by name through `_find_named`, so `gain_lna` resolves once the Pipeline carries the names). `compose: sum` is refused at a transform node and `compose: cascade` at a source node, mirroring `_check_slot_kinds`.

#### 4.5.5 `model.kind: pipeline` — non-graph composition (new)

```yaml
model:
  kind: pipeline
  stages:
    - {name: sky,  type: SkyOperator,  amplitude: {value: 100.0, unit: K}}
    - {name: gain, type: GainOperator, gain: {value: 1.1, unit: dimensionless}}
```

Eight of the thirteen files in `examples/` build a `core.pipeline.Pipeline` directly with their own stage names; seven use `assemble()`. Under graph-only addressing the stage a script calls `sky` must be renamed `uniform_sky` (`radio/sky/uniform.py:36`), so every `p["sky"]` selector in a ported script stops resolving. The path grammar needs no change — `Pipeline.__getitem__` resolves by name.

**Stated cost, in the schema:** `kind: pipeline` gives up the graph's structural checks (junction/selector refusals, `must_precede`, region contiguity, lit/skipped reporting). `check_stage_ordering` still runs on the declared order. Presets that carry a `model` are refused against `kind: pipeline`.

#### 4.5.6 `at:` — relocation and region claims

```yaml
bandpass:
  type: NeuralOperator      # deferred, capability 3
  at: bandpass              # required: NeuralOperator declares no graph_node

bandpass:                   # the KEY must equal at[-1] — see §2.2
  python: "pkg:MyOperator"
  at: [noise_wave, cw_tone, bandpass]
```

Refusals mirrored at validation: unknown node id; a region that is not contiguous template edges; a region whose ends are a junction or selector; an interior escape edge; an overlapping claim; a source-vs-transform kind disagreement (`core/fold.py`); and the new key-equals-`at[-1]` rule.

#### 4.5.7 `snapshot_before:` — keeping the raw data (new)

`SnapshotOperator` (`core/operator.py:128`) is exported from the top-level package (`src/rheplicant/__init__.py:32`), is named by `radio/filters/base.py` as *the* prescribed mechanism ("Filters typically run on calibrated data; preserve the raw data first with `SnapshotOperator`"), is used by `examples/sky_projection_and_filters.py`, and has no `graph_node` and no config surface. The processing segment is destructive by construction — `ApplyCalibrationOperator` divides, `SkySpaceFilter(mode="remove")` subtracts — and v0 could light both and then write only the filtered output.

```yaml
model:
  apply_cal:
    gain: {file: {path: gsol.npy, format: npy, unit: adc_count_per_K}}
    snapshot_before: raw          # -> At(("apply_cal",), Pipeline(SnapshotOperator("raw"), op))
```

Writes `state.aux["snapshot/raw"]` (`core/state.py` `State.checkpoint`), saved by `outputs.write.aux`. `PACKAGE CHANGE` is optional here: the `At((node,), Pipeline(...))` form works today; giving `SnapshotOperator` its own graph node would be cleaner (§11.9).

### 4.6 `variants`

```yaml
variants:
  seven_position:
    observation:
      switching:
        order: [antenna, ambient, hot, noise_source, load_600, load_900, load_150]
    model:
      cal_loads:
        load_600: {t_load: {value: 600.0, unit: K}}
        load_900: {t_load: {value: 900.0, unit: K}}
        load_150: {t_load: {value: 150.0, unit: K}}
```

Each entry is a deep-merge patch over the base document, using **the same `recursive_update` semantics as `defaults:`** — mappings merge, lists replace, `~key: null` deletes. Validation runs once per variant. `outputs.dir` gains a per-variant subdirectory. `config.resolved.yaml` is emitted per variant.

This is layering, not scripting: there is no ordering between variants, no variant may reference another, and the merge algorithm already exists for presets. It buys the identifiability-vs-cadence sweep that is `examples/gibbs_plan.py`'s entire thesis (four positions cannot carry six latents — nullity 2 of 34, singular values 1.6e-16 and 1.0e-16 against 1.82; seven give nullity 0) and `examples/driftscan_mmode.py`'s cross-engine and speed comparisons.

### 4.7 `inference` and `runs`

```yaml
inference:
  twin:
    without: [noise, rfi_field]
    replace:
      gain: {gain: {value: 1.0, unit: dimensionless}}     # NEW
  observed: {...}          # §4.7.1
  parameters: {...}        # §4.7.2
  bindings: [...]          # §4.7.3
  joint_prior: {...}       # §4.7.4
  trainable: {...}         # §4.7.5  (un-deferred from capability 3)
  noise: {...}             # §4.7.6
  truth: {...}             # §4.7.7  (usually derived)
  checks: {...}            # §4.7.8
  npe: {...}               # §4.7.10 (un-deferred from capability 3)

runs:
  - {name: ..., kind: ..., variant: ..., ...}   # §4.7.9
```

`twin.without` is the supported repair for the stochastic-stage refusal (`refuse_stochastic_stages`, `core/operator.py`; `inference/parameters.py`), measured at 10.6σ of bias **with the error bar unchanged bit for bit**, so the schema requires it whenever the model lights `noise` or `rfi_field`.

**`twin.replace` is new.** `Assembly.replace_node` is a public, guarded method with three named refusals (`core/graph.py:445-496`), and `examples/radio_digital_twin.py:103-107` uses it. v0's `inference.twin` had exactly one key, so "simulate with the measured bandpass, fit starting from a flat one" or "simulate with a 14-bit ADC, fit assuming 12" had no expression at all. Each entry is the §4.5.1 node spec and is validated through `replace_node`'s own three refusals. Check B8: a `twin.replace` on a leaf that a `parameters.<name>.into` path also targets is refused, naming both — otherwise the replacement is overwritten by `init` and the config says two contradictory things.

#### 4.7.1 `inference.observed`

```yaml
observed:
  from: simulation
  twin: full                 # full | fit          DEFAULT: full
  at: {fg_log_amp: ..., gain: {value: 1.1, unit: dimensionless}}   # truth values
  realise:                   # what scatter goes INTO the data
    kind: from_model | homoscedastic | radiometer | none
    sigma: {value: 5.0, unit: K}          # for homoscedastic
    seed: {from: runtime.seeds.observed_noise}
```

or

```yaml
observed: {file: {path: night1.npz, format: npz, key: waterfall, unit: adc_count}}
```

or, over several observations:

```yaml
observed:
  primary: {from: simulation, at: {gain: 1.10}, realise: {...}}
  second:  {from: simulation, at: {gain: 1.05}, realise: {..., seed: {from: runtime.seeds.other}}}
```

Three fixes here, all measured:

* **`twin: full` is the stated default.** v0's key order put `twin.without` above `observed`, which reads as drop-then-simulate — handing the calibrator noiseless, RFI-free "data". `examples/radio_digital_twin.py:97-102` is explicit that the simulation twin must carry `noise` and `rfi_field` ("that is where the noise and the RFI belong, in the data") and the fit twin must not. Both readings give the same shapes, the same loss-curve shape and the same diagnostics.
* **`at:` separates truth from start.** v0 forced the truth into `model.gain.gain`, so the file asserted "this instrument's gain is 1.1" while the script's own comment said the model starts mis-calibrated at 1.0.
* **`realise:` puts the scatter where it belongs.** `examples/three_ways_to_a_posterior.py:73-76` and `examples/sky_to_noise_wave.py:278` both add noise *outside* the model, with the same sigma the likelihood uses. v0's only route was a `model.noise` node whose sigma had to be kept equal to `inference.noise.sigma` by hand — a mismatch is a silent, correctly-shaped, wrong posterior. `realise.kind: radiometer` is the fractional form `d → d(1+w)` and is a `PACKAGE CHANGE`: `RadiometerNoise` exists as a sigma *rule* (`inference/noise.py:119`) and never as a generator, while `NoiseOperator` is additive. The config layer applies it, so no graph node is needed.

New check A42: `observed.from: simulation` with `twin: fit` while `model` lights `noise` or `rfi_field` → warn, naming the stages whose realisation has been removed from the data.

#### 4.7.2 `inference.parameters`

| key | type | req | default | Python target |
|---|---|---|---|---|
| `init` | value node | **yes** | — | `Latent.init` (`parameters.py:210`) — authority on shape and dtype |
| `prior` | prior spec | no | `null` | `Latent.prior`. `null` = *free*: usable by the calibrators and `plan.estimate`; refused by `to_numpyro_model`, gradient draws, `simulate_pairs`, `fisher_information(space=)` |
| `linear` | bool | no | `false` | `Latent.linear` — a checkable claim; derives the block engine |
| `scope` | `global` | no | `global` | `Latent.scope`; `per_epoch`/`linked` **RESERVED, capability 4** |
| `support` | `[lo, hi]` | no | — | **RESERVED, capability 4** — refused in v1 (§8.2) |
| `hyper` | mapping | no | — | **RESERVED, capability 4** — refused in v1 |
| `into` | path string or list | no | — | sugar for a one-latent `Bind` |
| `transform` | transform spec | no | `identity` | the `Bind.fn` |
| `fan` | `broadcast\|distribute` | req iff `into` is a list | — | `Bind.fan` |
| `ref` | value node | no | `init` | `init_to_declared` / NUTS start |
| `unit` | str | no | inherited from the target leaf | see below |
| `latex` | str | no | — | plot label |
| `renames` | str or list | no | — | aliases so an older config's chains resolve |

**A latent inherits its target leaf's declared unit through `into:`.** `init`, `prior.loc` and `prior.scale` are all in that unit; declaring a conflicting one is a validation error. v0 required `model.gain.gain` to declare a unit and then left the three numbers that live in it untied to it.

Warning, not refusal: an all-zero `init`. `check_linearity` takes its probe scales from `max|init|` and the gradient engine's Adam step is `DEFAULT_LEARNING_RATE * max|init|`; both fall back to 1.0, so the probes silently become absolute rather than relative.

**Prior spec.**

```yaml
prior: {normal: {loc: 0.0, scale: 400.0}}          # loc broadcasts against init.shape
prior: {normal: {loc: {zeros: [n_freq]}, scale: 400.0}, unit: K}
prior: {uniform: {low: 0.05, high: 0.60}}
prior: {log_normal: {loc: ..., scale: ...}}
prior: {python: "numpyro.distributions:StudentT", args: {df: 3, loc: 0.0, scale: 1.0}}
```

A scalar `loc`/`scale` **broadcasts against the latent's declared `init` shape** (v0 required `{zeros: [8]}` — four levels of braces for `dist.Normal(jnp.zeros(8), 400.0)` — while already requiring `prior.shape() == init.shape`, which makes the expansion unambiguous).

Recorded and warned: only Gaussian families (and `Independent`/`ExpandedDistribution` wrappers) survive `wiener_solve`, `gcr_sample`, `iterative_gls`, `fisher_information(space=)` and `prior_sensitivity`.

**Transform registry** (each entry declares its own `fan` and its own **shape signature**, new in v1):

| name | maps to | shape | `fan` |
|---|---|---|---|
| `identity` | `fn=None` | `(s) -> (s)` | broadcast |
| `exp` / `log` / `sum` | `jnp.exp` / `jnp.log` / `jnp.sum` | `(s) -> (s)` / `(s) -> ()` | broadcast |
| `affine: {scale:, offset:}` | `scale*v + offset` | `(s) -> (s)` | broadcast |
| `split_rows` | `tuple(v)` | `(k, n) -> k × (n,)` | **distribute** |
| `basis_expand: {basis: {ref}}` | `SeparableBasis.expand` | `(n_k, n_j) -> (n_time, n_freq)` | broadcast |
| `unit_mean_bandpass` | `radio/instrument/receiver.py:70` | **`(n-1,) -> (n,)`** | broadcast |
| `matmul: {design: <value node>}` | `design @ c` | `(n_basis,) -> (n,)` | broadcast |
| `log_link_basis: {kind, n_basis}` | `exp(basis_matrix(...) @ c)` | `(n_basis,) -> (n,)` | broadcast |
| `beam_analysis: {nside, lmax, iterations}` | `limtod_jax.map2alm_iter` | `(n_freq, n_pix) -> (n_freq, n_alm)` | broadcast |
| `python: "mod:fn"` | the escape hatch | traced by `jax.eval_shape` | must declare `fan` |

`unit_mean_bandpass`'s signature is on the page because check A33 **requires** that transform whenever `bandpass` and `gain` are both free — so the schema mandates the one transform whose input shape differs from its output shape and v0 never checked the off-by-one. The failure surfaces as a channel-count error naming `ReceiverOperator`, which `receiver.py:70-101` says points at the wrong thing.

`beam_analysis` is new and closes a measured wrong-answer. `DriftScanProjector`'s **only** non-static field is `beam_alms` (`driftscan.py:189-203`), so the only binding v0 could write is `into: ...projector.beam_alms`, giving `d(chi²)/d(alm)` — a different quantity in a different basis from the `d/d(map)` gradient `examples/driftscan_mmode.py:150-161` computes. Both are finite, correctly shaped, and pass every structural check.

New check C17: run `jax.eval_shape` on each binding's declared transform at the declared `init`, and require the result's shape to equal the target leaf's shape under the declared `fan`.

#### 4.7.3 `inference.bindings`

```yaml
bindings:
  - latents: [t_unc, t_cos, t_sin]
    into: [noise_wave.t_unc, noise_wave.t_cos, noise_wave.t_sin]
    fan: distribute
  - latents: [fwhm_deg, offset_deg]
    into: observed_astro_sky.projector.beam_alms
    transform: {python: "my_pkg.beams:beam_matrix", fan: broadcast}
```

`latents` order is the positional argument order of `transform`. `into` on a parameter and an entry in `bindings` for the same name are mutually exclusive.

#### 4.7.4 `inference.joint_prior`

```yaml
joint_prior:
  jeffreys: {over: [t_unc, t_cos, t_sin, t_rx], rank_rtol: 1.0e-8}
```

→ `inference/priors.py JeffreysPrior`, the only joint-prior type the package knows. Three cross-field refusals: a latent in `over` may not carry its own `prior`; `joint_prior` + any `runs[].kind: plan.*` is refused; `joint_prior` + `kind: fisher` with `space: true` is refused (a Jeffreys prior is *defined* as the square root of that determinant).

Recorded surprise: under `RadiometerNoise` on a bare power law the Jeffreys prior **is** the flat prior; under `HomoscedasticNoise` the same block gives p(log A) ∝ A². The noise model chooses the prior's shape.

#### 4.7.5 `inference.trainable` — un-deferred from capability 3

```yaml
trainable:
  all: false
  nodes: [bandpass]                 # every inexact leaf under these nodes
  leaves: [gain.gain]               # §2.2 path strings
```

Compiles to `build_forward_fn(filter_spec=...)` (`inference/forward.py:24`), whose **default is `eqx.is_inexact_array` — every inexact array in the whole twin**. v0 reserved this behind capability 3 and refused it, which mis-scoped it twice: `examples/neural_surrogate.py:52-58` needs it and so would anyone training a whole `noise_wave` block, neither involving NPE; and "fit every free number" is one Python call while the `ParameterSpace` route requires enumerating every latent with an `init`, a `prior` and an `into:`. Deferring the cheap case made it more expensive than the expensive one.

`trainable` is calibrator-only (no priors, no Bayesian exits) and is mutually exclusive with `parameters`/`bindings` in the same run.

#### 4.7.6 `inference.noise`

```yaml
noise: {kind: none}                 # legal only with runs[].kind in {forward, optimize, gradient, benchmark}
```

```yaml
noise:
  kind: homoscedastic               # -> inference/noise.py:95
  sigma: <value node>
  axis: none | time | freq          # REQUIRED when sigma is 1-D
  flags: {from: observation}        # optional; wraps FlaggedNoise (:173)
```

```yaml
noise:
  kind: radiometer                  # -> inference/noise.py:119
  channel_width: {from: observation}
  integration_time: {from: observation}
  floor: {value: 0.0, unit: K}
  include_logdet: <bool>            # REQUIRED, no default — see below
  flags: {from: observation}
```

```yaml
noise:
  kind: radiometer_frozen           # NEW
  source: observed | prediction_at_init
  channel_width: {from: observation}
  integration_time: {from: observation}
  floor: {value: 0.0, unit: K}
```

**`include_logdet` is required for any prediction-dependent noise model, and has no default.** `NoiseModelLikelihood.include_logdet` is `eqx.field(static=True, default=True)` (`inference/noise.py:333`) and the module docstring is explicit: dropping the `log 2πσ²` term "gives a *different estimator*, one with no penalty for shrinking the prediction to make the variance small… GLS returns `sum d²/sum d`, biased high by `(1 + f²)`, while the full density is asymptotically unbiased." It is unrecoverable after the fact: `inference/archive.py` refuses to load an archive that mixes them, and a lost `include_logdet=False` "comes back `True`" with no error. A v1 config that could not record it would produce runs that are not safely accumulable into a later campaign. Refused (as changing nothing) when the noise model is prediction-independent.

**`kind: radiometer_frozen` is new** and is the one form the conjugate seam accepts. `examples/sky_to_noise_wave.py:281-286` states the physics: the noise is fractional, so sigma is ~2× larger on antenna samples than on the loads, and "a scalar sigma would weight them equally and throw that away". v0's `homoscedastic` discards that; its `radiometer` is prediction-dependent and refused by name at `wiener_solve` (`inference/linear.py:1031`). So v0 forced the user either to lose the weighting or to change the estimator.

**The three noise seams, and which exit accepts which:**

| exit | accepts | refuses |
|---|---|---|
| `plan.estimate`/`plan.sample`, `fisher`, `nuts` | a NoiseModel **or** a bare sigma | — |
| `conjugate.wiener` / `conjugate.gcr` | a **decided** sigma array (`noise_std=`) | a NoiseModel, by name (`linear.py:1031`) |
| `conjugate.gls` (`gls.py:102 iterative_gls`) | a **NoiseModel** (`noise=`) | a bare array, by name |

`kind: radiometer` with `kind: conjugate.wiener` is refused at validation and the message names `conjugate.gls` **and** `radiometer_frozen` as the two routes.

#### 4.7.7 `inference.truth`

```yaml
truth: {<latent name>: <value node>}       # usually omitted
```

When `observed.from: simulation`, the truth is **derived**: `observed.at.<name>` if present, else the value the `model` leaf that `into:` targets held before binding. It is written to `config.resolved.yaml` and consumed by `outputs.write.recovery` (§4.8). All five ported scripts computed truth-vs-estimate and reported a pull; v0 had no place for it in the case it calls "the common case".

#### 4.7.8 `inference.checks`

```yaml
checks:
  identifiability:   {mode: refuse, rtol: 1.0e-8, report: true}
  linearity:         {mode: refuse, report: true}
  prior_sensitivity: {mode: skip, reason: "reported separately for this campaign"}
```

Each entry is a mapping with `mode: refuse | warn | report | skip`, an optional `report: true` (put the numbers in `diagnostics.json` rather than only gating), and a **per-entry `reason:` required when `mode: skip`**. v0 had one `reason_for_skip` string covering however many checks were skipped; three unrelated skips shared one sentence.

`mode: report` is new and is what `examples/gibbs_plan.py:208-211` actually does — the standalone `identifiability()` call is a printed report, not a gate. `checks.identifiability` gates the run; `runs[].check_identifiability` (which mirrors `SamplingPlan`'s own `once | each_sweep | false`) is a separate key and is not confusable with it.

`report: true` on `linearity` records the per-scale relative errors that `check_linearity` returns (`inference/linear.py:453`), so a claim that passes at 9e-13 and one that passes at 9e-4 are distinguishable in the run record.

---

**CORRECTED BY PLAN 3C — this section stated no default, and the code returned none either.** What is above is the grammar; what follows is what the section actually does now, and every clause of it was written against a measurement rather than against this paragraph.

**1. There ARE defaults, and they live in `config/gating.py::DEFAULT_MODE`.** Before 3C, `sections/inference.py::_checks(None)` returned `{}` — so "what mode is `linearity` in on a document that does not mention it" was a question with no answer, and every caller would have supplied its own default table. The defaults are now, measured on a two-latent 16 × 8 document with no beam:

| check | id | default | why |
|---|---|---|---|
| `linearity` | C12 | `refuse` | `len(scales) + 1` forward passes **per `linear: true` latent**; 0.188 s cold, 0.007 s warm, and it does **not** grow with `n_par` |
| `identifiability` | C13 | `off` | one `jacfwd` through the forward model plus a dense `(n_data, n_par)` SVD; 0.468 s cold |
| `prior_sensitivity` | C19 | `off` | `identifiability`'s work plus two Newton solves; **3.031 s cold against a 0.715 s `load_document`** |

This is D-C4 (*"`check_linearity` always on, with `mode: skip` plus a written reason as the only escape"*) plus the cost table, and it is now enforced rather than described. `gating.gates(section)` applies them; **cardinality is three whatever the document says.**

**2. `mode` and `report:` are ORTHOGONAL, and this section asked one question of both.** That ambiguity shipped in prose three times. The decision, made once in `gating.verdict`:

* **`mode` decides what a FAILURE produces** — and, for `skip`, that the check does not run at all.
* **`report:` decides whether the check's NUMBERS are recorded when it PASSES.**

There are **six effective states**, of which four are writable. `off` (nobody asked) and `auto_skip` (asked for and undefined on this document) are not in the mode vocabulary and cannot be typed into a document. `off` is deliberately **not** spelled `skip`: A37 makes every written `mode: skip` carry its own `reason:`, and a default-off check has no author to write one, so collapsing the two would either force a fake reason into the record or force A37 to exempt a case it cannot distinguish. A37 reads the document's text and therefore never sees either.

**Exactly one finding per gated check per document.** A failure with `report: true` is ONE finding at the mode's severity whose message carries the numbers — never a refusal *and* a report; two would double-count in `Report.checks()` and make `raise_if_refused`'s "N more refusals" tail wrong.

**3. `{mode: skip, report: true}` has no cell and is now refused by name**, in pre-flight, before anything is built. Measured before 3C: accepted. It asks to record the numbers of a check that will not run.

**4. `auto_skip` reports under `C14` even at `report: false`.** A complex or non-floating latent has no derivative, so `identifiability` and `prior_sensitivity` are undefined on it; rather than let `ParameterSpaceError` out of the package the gate stands the check down and reports, naming the latent and its dtype. **The predicates differ and are not one predicate**: `identifiability` and `prior_sensitivity` refuse complex *and* non-floating; `check_linearity` accepts a complex latent and refuses only a non-floating one.

**5. `rtol:` is legal on `identifiability` alone, and that is a real gap.** `check_linearity(rtol=)` exists with a default of `1e4 * eps` (`inference/linear.py`), and `inference.checks.linearity` **cannot express it**. Recorded by 3C, not fixed: widening the key set is a schema change no shipped document needs.

**6. `diagnostics.json` still does not exist.** This section names it twice. `outputs:` is refused wholesale by the pre-flight pass's deferred-section table until Plan 4, so 3C deliberately did not create it. What 3C did instead is put the whole outcome on the object: **`ConfiguredRun.report` carries every finding from all four passes, in pass order.** A **refused** document still produces no structured record at all — `raise_if_refused` raises, no `ConfiguredRun` is returned, and there is nowhere to hang the findings. Closing that needs either a `ConfigError` carrying the `Report` or a `load_document` that returns before raising; both are API changes and both are Plan 4's.

**7. Where each of these runs.** The `checks:` **grammar** is decided in the pre-flight pass from text alone (slots `A1.checks` and `A37`) — measured before 3C, **seven of seven** of these refusals arrived *after* the beam had been read. The checks **themselves** run in a fourth pass, post-flight, after `build_inference` and immediately before `load_document` returns. See §6's corrected preamble.

**8. UNDOCUMENTED CONTRACT CHANGE, found at 3C's integration and recorded here rather than smoothed over: `check: false` at a conjugate exit no longer suffices on its own.** With `linearity` at `refuse` by default, a user who wrote `check: false` on a `conjugate.*` run to decline that exit's own linearity check must now **also** write `inference.checks.linearity: {mode: skip, reason: "…"}`, because the two are different knobs and only the second is the gate. **Neither message names the other.** Confirmed not an advice loop — both escapes work when applied literally — but six shipped fixtures had to declare the skip, which is the measurement of how easy the trap is. A message that names the sibling knob is a one-line fix nobody has taken.

**9. A benign converter is not a benign gate.** Any document that lights `adc` and declares `linear: true` on a latent bound upstream of it is refused at C12 **by default, whether or not it saturates**. Measured on the most benign ADC the package can build (`{scale: 1.0, n_bits: 12}`, peak 12.116166 `adc_count` against a 2048 limit, so the real forward pass clips **nothing**): C12 still refuses, departure `5.32e+00` at the `1000x` probe against `rtol = 1.19e-03`, with the `0.001x` and `1x` probes both exactly 0. The refusal is **correct** — `check_linearity`'s `DEFAULT_SCALES` are `(1e-3, 1.0, 1e3)` and a clip at `1000x` is a real departure from linearity — and the escape is C12's own gate. It is written down because "my ADC does not saturate, so why am I refused" is the question this will produce.

#### 4.7.9 `runs` — the exits

```yaml
runs:
  - name: <str>                # required when there is more than one entry
    kind: <see table>
    variant: <name>            # default: the base document
    on: <observed name>        # default: the single/primary observation
    reuse: <earlier run name>  # reuse a fitted estimator or covariance
    expect: ok | refuse        # default: ok
    ...kind-specific keys...
```

| `kind` | maps to | keys (all with no default unless shown) |
|---|---|---|
| `forward` | evaluate `model` on the State | — |
| `plan.estimate` | `plan.py:816 SamplingPlan.estimate(...)` | `blocks`, `max_iter=45`, `tol=1e-3`, `min_sweeps`, `check_identifiability=once`, `solve_tol=1e-6`, `solve_guard=1e-3`. **Refuses a seed.** |
| `plan.sample` | `plan.py:962 .sample(...)` | as above plus `seed` **required**, `n_sweeps`, `warmup`, `rhat_max=1.05`, `warm_start` |
| `nuts` | `numpyro_bridge.py:132 to_numpyro_model` + `numpyro.infer.MCMC` | **`num_warmup` req, `num_samples` req**, `num_chains=1`, `chain_method=sequential`, `target_accept_prob=0.8`, `thinning=1`, `progress_bar=false`, `init=declared`, `seed` req |
| `conjugate.wiener` | `linear.py:578 linear_operator(names=)` + `:1139 wiener_solve` | `names`, `prior_std`, `prior_mean`, `tol=1e-6`, `maxiter=null`, `require_convergence=1e-3`, `check=true`, `width: none\|draws\|fisher` |
| `conjugate.gcr` | `linear_operator` + `:1534 gcr_sample`, vmapped | as above plus **`n_draws` (default 1)**, `seed` req, `noise_from: declared\|gls` |
| `conjugate.gls` | `linear_operator` + `gls.py:102 iterative_gls` | `names`, `prior_std`, `prior_mean`, `tol`, `maxiter`, `reweight_tol`, `min_reweights`, `max_reweights`, `require_convergence` |
| `optimize` | `calibrate.py:118 GradientCalibrator` / `:171 AdamCalibrator` | **`optimizer: gradient\|adam` req, `learning_rate` req, `n_steps` req**, `beta1=0.9`, `beta2=0.999`, `eps=1e-8` (adam only), `loss: mse \| {python:}` |
| `fisher` | `uncertainty.py:378 fisher_information` + `:478 parameter_covariance` | `space: bool`, `jitter=0.0` |
| `gradient` | `jax.grad` of a named objective | `objective: chi2 \| sum_squares \| mean \| {python:}`, `of: <path or [paths]>`, `at: {<name>: <value node>}` |
| `identifiability` | `identifiability.py:418` | `names`, `at`, `rtol` |
| `score_directions` | `reduced_basis.py:114` | `names`, `at` |
| `condition` | `linear.py:1337 condition_estimate` | `names`, `prior_std`, `iterations`, `seed` |
| `mmodes` | `DriftScanProjector.mmodes` | `projector: {ref}`, `sky: {ref}` |
| `predict` | `uncertainty.py:561 push_forward` / `predict_from_samples` | `from: <run name>`, `n_draw` |
| `compare` | difference two runs' products | `of: [a, b]`, `metric: max_rel_diff\|rms\|max_abs`, `tolerance` |
| `benchmark` | time named variants | `variants: [...]`, `repeats=5`, `warmup=1`, `report: [wall_time, peak_memory]` |
| `npe` | `npe.py:72 simulate_pairs` + `:183 NeuralPosterior.create` + `:295 train_posterior` + `:257 sample` | §4.7.10 |

Notes on the load-bearing ones:

* **`optimize` is spelled out** exactly as `plan.sample` is. `examples/radio_digital_twin.py:112` is `GradientCalibrator(learning_rate=2e-7, n_steps=200)`; the shipped default is 1e-2 (`calibrate.py:126`) — five orders of magnitude above what that fit needs — and the two calibrators are different algorithms behind an identical `.fit` signature. `loss: mse` names the shipped `mean_squared_error` so the escape hatch is not needed for the default.
* **`conjugate.gcr` gains `n_draws`.** `gcr_sample` returns one `(x, info)` pair and its own docstring says to vmap over split keys; `examples/gibbs_plan.py:196-199` and `examples/three_ways_to_a_posterior.py:98-103` both draw 500 and 4000. One draw is a random number, not a posterior — and every posterior-sigma column in both scripts comes from the stack.
* **`conjugate.gcr` gains `noise_from: gls`**, which closes the dead end check A27 created. `GLSResult`'s first field is `noise_std` — "the converged sigma: **the covariance**, and the whole point of the exercise. Feed it to `gcr_sample` or `wiener_solve`" (`inference/gls.py:74-92`), and `examples/gls_gcr.py:146-198` does exactly that. Without it, the only radiometer-noise conjugate exit returns the right mean and no error bar. `converged`, `iterations` and `delta` land in `diagnostics.json`, and `converged: false` refuses to proceed without `acknowledge_unconverged_covariance: true`.
* **`prior_std` and `prior_mean` are per-member.** `examples/inferring_anything.py:212-217` writes `prior_std={"sky_delta": SKY_SCALE}` and states the rule — "A grouped block also takes its prior PER MEMBER, since S is block-diagonal over the group." A scalar broadcasts and warns; the mapping form is **required** when the block names more than one latent, because their widths differ by orders of magnitude and a block-diagonal S returns a finite, correctly-shaped, wrongly-regularised answer with no residual signature. `prior_std` is **required** when every latent in the block declares `prior: null` (`linear.py:1009 _require_prior_std`). An override of a declared prior is recorded in `diagnostics.json` and refused alongside `checks.prior_sensitivity: refuse`.
* **The schema always compiles to `linear_operator(names=[...])`**, even for a block of one — `names=('gain',)` is blessed as "a legitimate group of one", and six downstream functions raise on the bare `name=` form.
* **`conjugate.wiener` gains `width:`.** `wiener_solve` returns the posterior mean only; a mean with no error bar is not a posterior. `width: draws` dispatches to the gcr route; `width: fisher` to `fisher_information` + `parameter_covariance`; `width: none` refuses `outputs.write.draws` and names `conjugate.gcr`.
* **`warm_start` gains its own `blocks:` and `move:`.** `examples/gibbs_plan.py:237,295` uses `Block(*FG, steps=200)` for the estimate and `steps=25` for the sample — 200 Adam steps to find the mode, 25 NUTS steps per Gibbs sweep. `Block.steps` "reads as a performance knob and is a **statistical assumption**" (`plan.py:276-299`), so collapsing them is not cosmetic. `move:` declares which inits the warm start moves (the script moves only the gradient block's; that coincidence is not a statement v0 could make or check).
* **`expect: refuse`** is new. Two exits in `examples/gibbs_plan.py` (`:175-178`, `:240-246`) exist *only* to be refused and to print the refusal, and that is the script's thesis. With `expect: refuse`, the validation checks that would have refused are downgraded to run-and-capture, the exception type and full message go to `refusal.txt`, and the process exits non-zero if the step **succeeds**. That turns a demonstration into a checkable assertion, and in a real campaign into the regression test that says "this design is still under-determined".
* **`identifiability`, `score_directions`, `condition`, `gradient`, `mmodes` are first-class cheap exits.** Each is a diagnostic the package documents as the thing to consult *before* committing to a long fit, and v0 could reach none of them:
  - `condition_estimate`'s own docstring instructs the user to "call it once outside the loop, choose `tol` from it"; at κ=1e7 the default tol=1e-6 bounds the relative error by 10 — no digits at all.
  - `identifiability` is the only diagnostic that sees across Gibbs blocks, and it is what tells a user how many calibration loads to build — a design question that should not require paying for the fit.
  - `score_directions` (`reduced_basis.py:114`) needs only a space, a pipeline and a state, answers "which direction in data space does this parameter move", and v0 deferred it behind capability 4, which does not need it.
  - `mmodes` returns a complex `(n_freq, lmax+1)` array and is "what a drift scan actually sees". Validation refuses `kind: mmodes` against a projector with `normalize_beam: true`, quoting the code's own "measured ~18× off".

#### 4.7.10 `inference.npe` — un-deferred from capability 3

```yaml
npe:
  bank:  {n_simulations: 32768, seed: {from: runtime.seeds.npe_bank}, cache: {file: {...}}}
  embed: ravel | {python: "mod:fn"}
  create: {n_components: 1, width: 64, depth: 2, min_scale: 1.0e-3,
           seed: {from: runtime.seeds.npe_create}}
  train: {n_steps: 3000, batch_size: 256, learning_rate: 1.0e-3,
          validation_fraction: 0.1, beta1: 0.9, beta2: 0.999, eps: 1.0e-8,
          seed: {from: runtime.seeds.npe_train}}
  sample: {n_draws: 4000, seed: {from: runtime.seeds.npe_sample}}
```

**Capability 3 is split.** NPE (`inference/npe.py`: `simulate_pairs:72`, `NeuralPosterior:149`, `create:183`, `sample:257`, `train_posterior:295`) touches `radio/surrogate.py` **not at all**, needs only equinox — already a hard dependency — is exported from `rheplicant.inference`, and is exercised by a first-class example. Bundling it into "capability 3 (neural surrogates)" and refusing it removed a working, dependency-light exit from the config for a reason that does not apply to it. Only `model.<node>.type: NeuralOperator` stays deferred (§8.1).

The reserved v0 shape also lacked three keys the real signatures require: `create`'s own PRNG key (`npe.py:183`, distinct from the bank seed), `train_posterior`'s key (`:295`), and its `beta1`/`beta2`/`eps`.

`outputs.write.training_history` is required alongside: `train_posterior` returns the estimator at its **best validation step**, and `best_step` versus `n_steps` is the only in-band signal that training was long enough or too long — over-fitting an NPE "makes it over-confident, which is the failure that does not look like one".

### 4.8 `outputs`

Optional; defaults to `{stdout: summary}`. The package has exactly one writer (`inference/archive.py:214 save_memory`, capability 4) and no `[project.scripts]` entry point, so everything here is new surface.

```yaml
outputs:
  dir: results/rhino_gibbs          # optional
  clobber: false
  stdout: none | summary | verbose
  report:                           # declarative cross-run table
    rows: [exact, nuts, npe]
    columns: [mean, std, seconds]
    reference: exact
    relative: [mean_sigma, width_ratio]
    format: [text, json]
  write:
    config:             true        # ALWAYS: config.input.yaml + config.resolved.yaml
    provenance:         true        # ALWAYS: provenance.json
    predicted:          {format: npz}
    observed:           {format: npz}
    aux:                {keys: [flags, protected, "snapshot/raw"]}
    taps:               [beam_spill, antenna_loss, noise_wave]
    assembly:           {format: json}
    estimate:           {format: npz}      # plan route
    fitted:             {format: npz}      # optimizer route
    draws:              {format: npz}      # name-keyed {latent: (n_draw, *shape)}
    losses:             {format: npz}
    gradients:          {format: npz}
    covariance:         {format: npz}
    prediction_band:    {format: npz}
    posterior_predictive: {format: npz, n_draw: 200}
    identifiability:    {format: json}
    scores:             {format: npz}
    recovery:           {format: json}
    timings:            {format: json}
    training_history:   {format: npz}
    refusal:            {format: txt}
    diagnostics:        {format: json}     # a LIST of phases
    signal_path:        {format: svg | html | mermaid, themes: [light, dark]}
    memory_archive:     {...}              # RESERVED, capability 4
```

Why each of the new ones exists, in one line each:

* **`aux` / `taps`.** The whole product of `flagging` is `aux["flags"]`; `aux["protected"]` is the calibrator-protection contract; `taps` compiles to `SnapshotOperator` insertions (§4.5.7) so the six-stage loss cascade a user needs to debug a 30 K discrepancy is recoverable. v0 wrote `.data` only.
* **`assembly`.** `print(twin)` is the first line of output of three of the five ported scripts and is what tells a user which of the 32 template nodes their operator set lit (`Assembly.lit`/`.skipped`), plus `assembly["receiver_input"].names` — the switch order `noise_wave.py:83-86` explicitly tells the user to read off the twin rather than assume. An SVG is not greppable or diffable.
* **`fitted` vs `estimate`.** `estimate` is `Estimate.values` from a `SamplingPlan`; a calibrator returns a params pytree. v0 overloaded one key onto two different objects, so the one number `examples/radio_digital_twin.py` exists to produce had no documented output path.
* **`losses`.** `calibrator.fit` returns `(params, losses)` of shape `(n_steps,)`, and "3.386e+04 → 1.411e+04" *is* the demonstration.
* **`covariance`.** `run.kind: fisher`'s entire product. Written name-keyed (`FlatMatrix.sigma(name)`, `.block(a, b)`, `uncertainty.py:97-186`) — flattening it to an anonymous matrix throws away what §10 already insists on for chains, and the cross-block correlation is how `examples/inferring_anything.py:180-186` distinguishes degeneracy from mis-specification.
* **`prediction_band` / `posterior_predictive`.** `propagate_covariance` (`uncertainty.py:509`) is the delta-method band a user plots against their data; `predict_from_samples` is the only in-band check that a fitted model reproduces the observation it was fitted to.
* **`recovery`.** Truth, estimate, posterior mean, posterior sigma, absolute error, pull — per latent. See §4.7.7.
* **`diagnostics` is a list of phases** (`warm_start`, `sample`), plus `rhat_per_latent: {name: array}` computed with the already-public `split_rhat` over each component's trace. `PlanDiagnostics.rhat` is the split-r̂ of the **joint** chi-squared — one scalar — which hides the one channel of an 8-vector that did not mix.
* **`signal_path` gains `html` and `mermaid`.** `Assembly` ships three renderers (`core/graph.py:559,565,572`), `to_html(title=, theme=)` is the interactive form, and `examples/render_signal_path.py` exists solely to write it. `to_mermaid()` takes no theme (`:559`) and is offered light-only.
* **`report` and `timings`.** "engine 2 against engine 1: mean 0.02σ, width 1.00×" is `examples/three_ways_to_a_posterior.py`'s actual output and the sentence that says NUTS reproduced the conjugate answer. `timings` records compile / first-call / steady-state separately, since JAX makes the first call unrepresentative.

**Two config artefacts are always written.** `config.input.yaml` is byte-for-byte what the user wrote; `config.resolved.yaml` is the same with presets merged, defaults filled, paths absolute, file hashes recorded, `shared_objects:` listed, `switch_map:` listed, and **every key annotated with its origin** (`# from rhino_v1`). "What I asked for", "what actually ran" and "where did that value come from" are three different questions.

`provenance.json` carries `rheplicant.__version__`, `jax`/`jaxlib`/`equinox`/`numpyro`/`numpy` versions, the git SHA, `jax_enable_x64`, the resolved-config SHA-256, the SHA-256 of every input file, every named seed, the platform, and per-run wall times.

### 4.9 Units policy

Field names are exactly the Python field names, so the unit cannot live in the key name. Therefore:

* Every **dimensional** value is `{value:, unit:}` or an array/file/draw spec carrying `unit:`. The shorthand `"290 K"` parses to the same thing.
* A **bare number** is legal only where the key is marked `dimensionless` or `count`.
* **Fields whose Python name already carries a unit** (`lat_deg`, `apod_deg`, `lst_ref_deg`, `az_deg`, `el_deg`, `selfrot_deg`, `lst0_deg`, `alt_m`) still take `{value:, unit:}`; the declared unit is **checked against the suffix**, turning the redundancy into a free consistency check instead of a contradiction.
* **Canonical units** (converted on read, never stored otherwise): frequency `Hz`; time `s`; temperature `K` (`celsius` converts with +273.15); angle `deg`; length `m`; impedance `ohm`; `dimensionless`; `count`; `samples`; `bits`; `channels`; `cycles/sample`.
* **Compound units, new in v1.** `unit:` accepts a quotient or product over the atomic alphabet: `adc_count/K`, `K/s`, `Hz/s`, `1/s`, `K*s`. v0 permitted only `adc_count` or `dimensionless` on the post-ADC trunk while its own canonical example wrote `adc_count_per_K`, which §1 never defined — so a user modelling `ADCOperator.scale`, documented in the source as "counts per kelvin at unit gain", was forced to declare a wrong unit. Launder a guess into provenance and every future dimensional check is defeated. With the quotient grammar, `adc_count/K × K = adc_count` is a checkable identity and check A9 can verify derived scalars.
* **Five fields have no unit anywhere in the source** — `adc.scale`, `gain.gain`, `apply_cal.gain`, `flagging.threshold`, `filters[].regularization`. The config **must** declare one and it is recorded as a *config-level declaration*, not a measurement. §11.2 settles which token.

---

## 5. Presets and layering

```yaml
defaults: []                              # DEFAULT: no preset. Omitting the key means none.
defaults: [rhino_v1]
defaults: [{from: rhino_v1, only: [runtime, observation.site]}]      # partial adoption
```

Rules:

1. Presets ship as `rheplicant/config/presets/<name>.yaml`. `rhino_v1` would be the package's **first** preset — `RADIO_GRAPH` is a topology with zero numeric defaults, and every RHINO number today lives only in `examples/` and `docs/_generate_*.py`.
2. Merge order: each `defaults` entry in order, then the user's file **last**.
3. The merge is a deep, **namespace-preserving** `recursive_update`. Explicitly *not* hera_sim's recursive flattening, whose own docstring admits "any parameters whose names are not unique will take on the value specified last" — under which `model.bandpass.gain` and `model.gain.gain` collide silently.
4. **Mappings merge; lists replace.** `{append: [...]}` extends instead of replacing. `~key: null` deletes.
5. **`model` is node-set-replacing, not merging (changed in v1).** Three of five stress agents hit this independently. `model` is a mapping, so under rule 4 a preset that declares `model.antenna_loss` **lights that node in every inheriting config**, removable only by a per-node `~model.antenna_loss: null`. `rhino_v1` would carry `antenna_loss`, `beam_spill`, `atmosphere` and `noise_wave`; every one of the fourteen shipped examples lights a different subset (four nodes, one node, two nodes…). So `defaults: [rhino_v1]` would be unusable for thirteen of fourteen, and the failure is silent in the direction that matters: an inherited η=0.97 quietly multiplies every prediction by 0.97 and adds 8.8 K. v1: a preset's `model` is adopted wholesale or not at all, with `model: {inherit: [antenna_loss, beam_spill], ...}` to take a named subset, and the validator prints "preset rhino_v1 added 4 model nodes: …".
6. **A preset may not set anything the schema marks required-with-no-default** — `beams.phi0_deg`, `beams.phi_sense`, `beams.normalize`, `projectors.normalize_beam`, `cw_tone.line_width`, `from_file.freq_unit`, `noise.include_logdet`. Those are facts about a specific instrument or a specific file, and a preset supplying them would be guessing on the user's behalf in exactly the place the code refuses to. §11.5 offers a `provisional: true` route for the beam pair.
7. `rhino_v1` would carry `observation.site.lat_deg` 53.2367 (Jodrell Bank), drift pointing at az 0 / el 90, a 60–85 MHz band, `antenna_loss` η=0.97 / T_phys=293 K, `beam_spill` t_ground=290 K, `atmosphere` t_atm=3 K, `noise_wave` t_rx≈290 K, and `runtime.jax_enable_x64: true`. **Every one is an example value harvested from `examples/`, not a measured instrument constant**, and the preset must say so in a `provenance:` comment block.

---

## 6. Validation: every check before anything expensive

Group **A** is pure text; **B** needs the assembled twin (no data); **C** needs one `jax.eval_shape` or one small Jacobian.

**CORRECTED BY PLANS 3B AND 3C — the sentence that used to end this line was "All run before any file is read that is not needed to decide them, and before any beam is analysed", and it is false.** It is false in two different ways and both are measured:

* **3B:** it is false for the twin-shaped rows. A check that compares the fit twin's switch positions against a declared switch order, or two projectors' analysed beams, needs the objects to exist. Those findings arrive before the *fit* — which is the expensive thing they protect — and after the build.
* **3C:** it is false for every priced row. `C12`, `C13`, `C16`, `C18` and `C19` run the model. What they buy is not that they are free but that the document can **see the price and decline to pay it** (§4.7.8's corrections).

**Three groups, four passes, and the two do not line up.** The letters here classify a check by *what it needs*; the implementation classifies it by *when it can be run*, and that is one distinction finer:

| pass | what it may read | which §6 rows land here |
|---|---|---|
| **pre-flight** (text) | the document's mapping, `RADIO_GRAPH`, operator classes resolved by name | all of A, plus `C18`'s two-word family half |
| **axes** | the resolved time and frequency grids, one line **above** `build_resources` | `A13`'s grid bounds, `C1`, `C2`, `C3`, `C8`, **`C15`** |
| **built** | the twin, the state, the built resources; `jax.eval_shape` only | `A43`, `B5`, `B9`, `C9` |
| **post-flight** (priced) | everything the built pass reads, plus the resolved gates; may evaluate the twin | `C12`, `C13`, `C16`, `C18`, `C19` |

The **axes** pass is the one the letters have no name for, and it is the only later pass that still saves the beam: on a toy nside-16 beam `build_resources` is 90.9 % of `load_document`, and the axes pass sits one line above it for about a hundredth of what it saves.

### A. Pure config

| # | check | pre-empts |
|---|---|---|
| A1 | unknown key anywhere → refuse | a typo'd key becoming a 0.1% systematic |
| A2 | unknown node id in `model` | `AssemblyError` |
| A3 | a junction/selector used as an operator slot (`astro_sum`, `field_sum`, `astro_ant_sum`, `t_ant_sum`, `receiver_input`) | `core/graph.py` |
| A4 | a reserved node (`beam`, `atmosphere_field`, `ground_field`) given a `type:` | no shipped operator |
| A5 | two operators at a non-`many` node without `compose:` | `core/graph.py` `_place_at_node`, whose message names the `At(...)` route |
| A6 | a `many` node given a scalar spec, or a non-`many` node given a list | shape confusion |
| A7 | `type:` missing at `flagging` or `filters` | two/three classes share the node |
| A8 | `cw_tone` relocated at or after `bandpass`/`gain` via `at:` | `must_precede`; quote `must_precede_because` |
| A9 | every dimensional value carries a convertible unit; unit-suffixed field names agree with their declared unit; derived scalars satisfy their dimensional identity | §4.9. **UNASSIGNED after Plan 3C, and it is the ONLY row that is.** Measured at 3C's integration commit: `grep -rn "check A9" src/ tests/` returns nothing, while every other id in this section is either registered in one of the four passes or (C7, C11, C17, B4) shipped in place inside `config/sections/`. It needs a node-field → dimension table that no plan has built. Its obligation 1 is the one with a live consequence: `adc.scale: {unit: "K"}` is **accepted** today although `adc_count` and `adc_count/K` are in the unit grammar and `units.py` names D-C1 as the reason they are |
| A10 | `observation.from_file.freq_unit` present | `radio/rhino.py:469` — no default; the file does not record it and its two producers disagree |
| A11 | `beams.phi0_deg` / `phi_sense` present **for `format: cst`**, refused otherwise; `frame:` present for raw-array formats | `radio/beams.py` |
| A12 | `beams.normalize` present; `projectors.normalize_beam` present | the output's unit is the pair (32838 K vs 200 K; 100.42 K vs 99.07 K) |
| A13 | `cw_tone.line_width` present **and in range**: `MIN_WIDTH_IN_CHANNELS[lineshape] * median(\|diff(freq)\|) ≤ line_width ≤ 0.25 * band`; `freq.min() ≤ tone_freq ≤ freq.max()`; **and the same for the drifted centre** `tone_freq + drift_rate*(t_max-t_min)`; `0 < protect_floor ≤ 1` | `radio/instrument/calibration.py`. Pure arithmetic on four scalars and two grids, and v0 let all of it fire after the beam had been analysed |
| A14 | `switching.order[0] == "antenna"`; `order[1:]` equals the keys of `model.cal_loads` in order; every label appears once | switch index / `gamma_src` row order |
| A15 | `noise_wave.gamma_src` has exactly `len(order)` rows, assembled by `from_switch_order` matched **by name** | a transposition is shape-legal and costs tens of kelvin |
| A16 | every latent appears in **exactly one** block of the plan | `plan.py` — "silently frozen at its declared init … nothing anywhere reports that a parameter you declared was never inferred" |
| A17 | `steps:` on an all-linear block | `plan.py` raises rather than ignoring |
| A18 | a block mixing `linear: true` and `false` without `engine:` | `plan.py` |
| A19 | `engine: conjugate` over a non-linear member | `plan.py` |
| A20 | `joint_prior` + any `kind: plan.*` | `plan.py` |
| A21 | `joint_prior` + `kind: fisher` with `space: true` | `uncertainty.py` |
| A22 | a latent in `joint_prior.over` also carrying its own `prior` | `priors.py` |
| A23 | a prior-less latent with a run kind that requires priors (`nuts`, a gradient block of `plan.sample`, `npe`, `fisher(space=)`) | `numpyro_bridge.py`, `engines.py` |
| A24 | `n_sweeps - warmup >= 4` (`MIN_DRAWS`) | below that split-r̂ is undefined and a NaN passes no threshold in either direction |
| A25 | `1 <= min_sweeps <= max_iter`; `max_iter >= 1`; `n_sweeps >= 1`; `num_samples >= 1` | `plan.py` |
| A26 | a 1-D noise `sigma` without `axis:` | `check_noise_std_axis`; measured error bars 0.00004…0.00354 vs a flat 0.00010 |
| A27 | `kind: conjugate.wiener\|conjugate.gcr` with `noise.kind: radiometer` | `linear.py:1031`; message names **both** `conjugate.gls` and `radiometer_frozen` |
| A28 | `kind: conjugate.gls` with a decided array sigma | `gls.py` |
| A29 | seed present for `plan.sample`/`conjugate.gcr`/`nuts`/`npe`; **absent** for `plan.estimate` | the asymmetry is in the code |
| A30 | `model` lights `noise`/`rfi_field` and `twin.without` does not drop them | 10.6σ of bias with the error bar unchanged bit for bit |
| A31 | `observation.data` present while `model` lights any source node | `Assembly.__call__` |
| A32 | `beam_spill` + `ground_pickup` both lit without `acknowledge_double_count: true` | documented and deliberately unenforced in code |
| A33 | `bandpass` and `gain` both free without `bandpass.transform: unit_mean_bandpass` | one exactly null direction (n_par 11, rank 10, nullity 1) |
| A34 | `outputs.dir` exists and `clobber: false` — **only when `outputs.dir` is present** | destroying a previous run mid-fit |
| A35 | optional dependencies importable **and providing the declared features**: `rhino_cal_jax` for `noise_wave`/`s_params` kinds; `MomentRFI`; `h5py`; `numpyro`; and per-feature `hasattr` gates on `limtod_jax` mirroring `radio/beams.py` `_require_limtod_jax` | a 40-minute failure at the first import. Widened from v0: `import limtod_jax` succeeds on 1.6 while `horizon.mode: truncate_map` fails **after** the CST directory has been read and sampled |
| A36 | `jax_enable_x64: true` when `memory_archive` is requested | `archive.py` |
| A37 | every `checks.<name>.mode: skip` carries its own `reason:` | §4.7.8 |
| A38 | `fan` present whenever `into` has >1 target; consistent with the transform registry | §2.2 |
| A39 | capability-3/4 keys present → refuse, naming the deferred capability | §8 |
| A40 | a `file:`/`draw:`/`stack:`/array-spec value node landing on a **static** field | measured: `ADCOperator(n_bits=Array(12))` raises; `ForegroundOperator(ref_freq=Array(...))` corrupts the jit key |
| A41 | a literal integer in a shape equal to `n_freq` or `n_time` → warn, naming the symbol | five hand-copied grid lengths in one 90-line port |
| A42 | `observed.from: simulation` + `twin: fit` while `model` lights a stochastic node → warn | §4.7.1 |
| A43 | `cw_tone` and `flagging` both lit: compare `protect_floor * amplitude * max(w)` against `flagging.threshold` (through `adc.scale`) | `calibration.py` gives the rule and both directions are silent; docs measure the unprotected case at twelve flagged samples — "That is the calibrator, gone" |
| A44 | `engine: driftscan\|general_pointing` requires `jax_enable_x64: true` or `acknowledge_float32_sky: true` | O(10%) errors in float32, invisible |
| A45 | every `switch_key` names `receiver_input` or a key of `observation.extra` | `noise_wave.py` raises after everything upstream is built |
| A46 | every label in `switching.order[1:]` appears in `model.cal_loads` and, for `from: thermistors`, in `from_file.thermistor_columns`; two labels sharing a thermistor column → warn, naming both | legal (docs show it) and also exactly what a typo looks like |
| A47 | a multi-node `at:` region's config key equals `at[-1]` | `core/graph.py:127-133` |
| A48 | `optimizations` containing `cache_beam_rotation` requires `lst_ref_deg` | `to_reference_frame()` raises after the beam is analysed |
| A49 | `noise.include_logdet` present iff the noise model is prediction-dependent | a `(1 + f²)` bias, unrecoverable after the fact |
| A50 | `beams.horizon.mode: truncate_map` + any `beam_spill.from: projector` referencing that beam → refuse, naming the ≈1.0 fraction it would have produced; likewise `from: projector` against a `cache_beam_rotation` projector | the ground term silently vanishes / the call raises |
| A51 | `conjugate.*` over a multi-latent block requires the mapping form of `prior_std` | block-diagonal S, no residual signature |
| A52 | `pointing.mode: none` refused when `model` lights `observed_astro_sky` or any projector is referenced; `site.lat_deg` required iff a projector reads `{from: site}` | §4.1 |

### B. Needs the assembled twin

| # | check |
|---|---|
| B1 | every `into:` path resolves to an **array leaf** on `_tag_leaves_with_paths(twin)` |
| B2 | no path names an aliased node |
| B3 | no two bindings write the same leaf |
| B4 | every latent is bound — "the posterior would just return the prior" |
| B5 | `len(assembly["receiver_input"].names) == len(switching.order)`, positionally; `switch_map` recorded |
| B6 | `at:` regions contiguous, closed, non-overlapping, not across the source/transform line |
| B7 | `filters[].projector` and `observed_astro_sky.projector` are the same object when they reference the same resource name |
| B8 | a `twin.replace` target that a binding also targets → refuse, naming both |
| B9 | two projectors nominally sharing a beam actually share the array |

### C. Needs one shape trace or one small Jacobian

| # | check |
|---|---|
| C1 | **time-axis precision**: `np.spacing(max\|t\|) <= 1e-2 * min(\|diff t\|)` over non-zero gaps, under the declared dtype. `core/coordinates.py:66`. A uniform float32 axis caps at ~1e5 samples (exactly 131072 at 1 s); a 4 h run at 0.05 s is refused, with the fix named |
| C2 | non-finite `time`, `lst_deg`, `pointing`, `selfrot` — the adjoint would return a finite, correctly-shaped, identically **zero** map |
| C3 | `uniform_sampling: true` ⇒ uniform full sidereal turn, endpoint excluded, `2*lmax < n_time` |
| C4 | `DriftScanProjector`'s static az/el/selfrot agree with `Coordinates.pointing` to 1e-3 deg |
| C5 | `sky.n_pix == 12*nside**2`; `len(beam_alms) == (lmax+1)(lmax+2)/2`; sky maps on the run's frequency grid |
| C6 | `horizon.mode: truncate_map` ⇒ `el_deg == 90` |
| C7 | `{from: horizon_fraction}` refused against a `cache_beam_rotation` projector |
| C8 | `n_time % n_chunk == 0`; `n_time % n_days == 0` and `n_days >= 2`; `0 <= low < high <= 0.5` |
| C9 | a bare 1-D array into `cal_loads.t_load` or a `noise_wave` temperature where `n_time == n_freq` ⇒ demand `column: true` |
| C10 | `BasisTemperatureOperator.coeff` matches `(n_k, n_j)` of the referenced basis, built for **this** grid |
| C11 | `observed.shape == prediction.shape` **exactly** (broadcast-compatible is the dangerous case) |
| C12 | `check_linearity` on every `linear: true` claim, at the declared inits, over scales `(1e-3, 1, 1e3)`; margins recorded |
| C13 | **identifiability** at the declared inits: `rank`, `nullity`, `participation()` per null direction |
| C14 | a complex or non-floating latent with identifiability enabled → auto-skip with the reason recorded |
| C15 | switch-position count vs the declared noise-wave block: per channel the rank is `min(n_src, k) * n_freq`, `k = 4` free families (3 if `t_rx` held). **Reported, not refused** — no counting rule survives a basis parameterisation. **CORRECTED BY 3C: this row ships in the AXES pass, not in group C and not in pre-flight.** It needs neither the twin nor a Jacobian, so group C overstates it; but it is not decidable from text either, and that half is measured. The only text reader for `n_freq`, `preflight/values.py::_a41_scope`, returns `None` for a grid that is not `linspace`/`arange`/`modulo`/a list, for a symbolic `num:`, and for **every ingested run** — and it couples the two axes. A pre-flight C15 would therefore be silent by construction on a whole class of documents. The axes pass carries `context.shape_scope`, which gives `n_freq` and `n_source` unconditionally, and still runs one line above `build_resources` |
| C16 | **ADC saturation** (new): one forward evaluation; report `saturated_fraction = mean(\|data*scale\| >= 2**(n_bits-1))`. Warn above 0, refuse above 0.1%, naming `adc.scale` and `adc.n_bits` and the achieved peak in counts. ~~Refuse outright when any latent's `into:` path lies upstream of `adc`.~~ **CORRECTED BY 3C: the topology is a SEVERITY ESCALATOR, not an independent refusal, and the measurement is why.** 27 of `RADIO_GRAPH`'s 33 nodes are upstream of `adc`, `gain` among them, so the unconditional reading refuses a document measured to run correctly (`saturated_fraction = 0`, `d(sum)/d(gain) = 1252.16`). The shipped rule is: `saturated_fraction == 0` → nothing; `0 < f <= 0.001` → **warn**; `f > 0.001` → **refuse**; and `f > 0` **AND** a latent bound upstream of `adc` → **refuse whatever the fraction**, because that latent's gradient is dead. This overrules Plan 3A §7's reading of the same sentence. `>=` at the limit counts as saturated, matching `ADCOperator.__call__`'s own `jnp.clip`. **A clipped sample has exactly zero gradient**: the fit does not move, chi² is flat, `check_linearity` passes (a constant *is* affine), identifiability reports full rank on the unsaturated samples, and the loss curve is a horizontal line that reads as "converged". `adc.scale` is one of the five unit-unstated fields, so its magnitude is exactly the number a user has no calibration for |
| C17 | **transform shape** (new): `jax.eval_shape` each binding's transform at its `init`; require the result to match the target leaf under the declared `fan`. Also warn when `bandpass` uses `unit_mean_bandpass` and the `gain` init was not scaled by the discarded `mean(bandpass)`. **SHIPPED by Plan 3B, in place inside `config/sections/`** (`grep -rn "check C17" src/` → three occurrences in three files: `sections/transforms.py` (two lines, the implementation), `sections/observed.py` (one line, the implementation), and `preflight/model.py` (one line, a cross-referencing comment on A33's shape half — not an implementation, and until this correction it wrongly called the row Plan 3C's), so it registers no registry slot and no census finds it. It is **not** unassigned residue — do not re-implement it |
| C18 | **two sigmas** (D-C17): a `noise`-node operator that DRAWS and an `inference.noise` that WEIGHS must agree — the same family, and the same fractional scatter (radiometer) or the same sigma (homoscedastic). Nothing in the package keeps them equal and a drift is finite, correctly shaped, wrong and invisible to every diagnostic. **Added by 3C, in TWO slots sharing one row**: `C18.kind` decides the family from `model.noise.type` against `inference.noise.kind`, in the text pass; the bare `C18` compares the numbers on the built objects, in the post-flight pass. **It is NOT a gate** — it is not an `inference.checks` name and is absent from `gating.CHECK_ID`. **Scoped to a GENERATING twin, and to the `noise` node**: it stands down unless the primary observation is `from: simulation` **and** the twin that drew it lights `model.noise` — `twin: full` (the shipped default), or `twin: fit` whose `inference.twin.without:`/`replace:` does not remove `noise`. `rfi_field` also draws and `inference.noise` has no RFI counterpart to disagree with |
| C19 | **prior sensitivity**: `inference.checks.prior_sensitivity` gates `prior_sensitivity` (`inference/sensitivity.py`), which `inference.checks` has named since 2B and no §6 row did. **Added by 3C together with the gate that reads it.** It is one of the three `inference.checks` names and is **off by default** — `identifiability`'s `jacfwd` plus SVD plus two Newton solves, 3.031 s cold against a 0.715 s build. Its findings carry `C19`; a complex or non-floating latent turns it into a `C14` auto-skip, never a `ParameterSpaceError` out of the package. See §4.7.8's corrections for the whole cross-product |

---

## 7. Named refusals

What v1 deliberately cannot do, and what to do instead.

| the config cannot | why | do this instead |
|---|---|---|
| **elementwise arithmetic beyond `scale`/`offset`** — trig, powers, logs, products of two value nodes | this is the escape-hatch boundary. `30*cos(linspace(0,3,8))` needs an evaluator, precedence, and a namespace; that is a programming language | `resources.arrays.<name>: {python: "my_pkg:t_cos_truth", args: {n: n_freq}}` and `{ref: ...}`. State the cost: the run is no longer reproducible from the config alone |
| **nested function calls with positional operands** | same reason | name the inner call as a `resources.arrays` entry and `{ref: ...}` it into `args:` (§4.4.1) |
| **arbitrary `SignalGraph` topologies** | `Assembly.aliased` is the documented hazard for user-defined graphs and nothing intercepts a hand-rolled `eqx.tree_at` through an aliased node | `model.kind: pipeline` for a linear chain; `python:` at a node for anything else. A `graph:` section is refused in v1 (§11.13) |
| **a `file:` route for a `basis_matrix` design matrix** | "a basis built for another band would return a smooth, plausible, wrong temperature" (`radio/t_sys.py`) | `{kind, n_basis}` with `n` taken from the grid. §11.8 offers a provenance-checked alternative |
| **writing `beam_frame` / `beam_ref_lst_deg`** | `__check_init__` exists to catch a hand-set pair | `optimizations: [cache_beam_rotation]` + `lst_ref_deg` |
| **`scope: per_epoch \| linked`, `transitions`, `support`, `hyper`, `campaign`, `memory_archive`** | capability 4 | §8.2 |
| **`model.<node>.type: NeuralOperator`, `posterior_net`** | capability 3 | §8.1 |
| **chain persistence in an archive** | `save_memory` refuses a `ChainMemory` outright — "a HyperTransition's builder is a Python callable with no textual form for a manifest to record" | not offerable, ever |
| **`blocks[].learning_rate`** | `engines.py` accepts it, `plan.py` never passes it | reserved and refused with a message naming the gap. §11.9 |

### MINOR findings deliberately not fixed, one line each

* **`to_mermaid` gaining a theme argument** — not fixed. `to_mermaid()` takes no arguments (`core/graph.py:559`); v1 offers it light-only and offers `to_html`/`to_svg` for themed output, which is a package question not a schema one (§11.14).
* **`GeneralPointingProjector` missing from `radio.__all__`** — not fixed in the schema. Verified absent (`radio/__init__.py:158`, which lists `DriftScanProjector` and `MatrixProjector` only); `engine: general_pointing` resolves the class by module path, and the export is a package cleanup (§11.10).
* **`z0` for Touchstone** — not offered. `Touchstone.z0` is parsed and never read by any other module; `z0` appears only under `s_params.kind: termination`, where `termination_gamma(z0=)` reads it.
* **A hand-written reference cross-check** (`examples/sky_to_noise_wave.py:236-247` recomputing the noise-wave model from rhino-cal primitives and asserting agreement at roundoff) — explicitly out of scope, stated rather than left as an apparent gap. That is a test, not a run.
* **`observation.data_unit` as a top-level declaration** — not added; the per-field declaration plus the compound-unit grammar (§4.9) makes the K→counts conversion checkable without a second place to state it. See §11.2.

---

## 8. The deferred holes

### 8.1 Capability 3 — neural surrogates (shrunk, and corrected)

**Corrected shape.** v0 bundled four things under capability 3 and refused all of them. Three of the four do not need it:

* **NPE ships in v1** (§4.7.10). It touches `radio/surrogate.py` not at all.
* **`inference.trainable` ships in v1** (§4.7.5). It is `build_forward_fn`'s `filter_spec`, whose default is the whole twin.
* **`outputs.write.training_history` ships in v1.** It is NPE's diagnostic.

**What remains deferred** — exactly one node type and one output:

```yaml
model:
  bandpass:
    type: NeuralOperator          # radio/surrogate.py:29 — the only shipped operator with
    at: bandpass                  # NO graph_node, deliberately: "a surrogate's placement
    from: create                  # is a modelling decision". At() permits only same-kind
    seed: {from: runtime.seeds.surrogate_init}   # nodes, so `at:` is transform nodes only.
    f_min: {value: 60.0, unit: MHz}              # static (surrogate.py:46)
    f_max: {value: 85.0, unit: MHz}              # static (surrogate.py:47)
    width: 16                                    # create(width=16, depth=2), :56
    depth: 2
    weights: {file: {format: eqx_leaves, path: surrogate.eqx}}

outputs:
  write:
    posterior_net: {...}
```

**A seed plus an architecture, never weights inline.** v1 refuses both keys by name, quoting `radio/surrogate.py:29`. Neither requires the v1 structure to change: the node spec, `at:`, `from:` and the `eqx_leaves` file format all exist.

### 8.2 Capability 4 — streaming evidence (corrected: it touches three sections, not one)

v0 claimed capability 4 "adds one top-level section and widens one enum". Reading `inference/reduced_basis.py`, `inference/compress.py` and `inference/factorize.py` shows that is wrong in three places, and v1 reserves the keys **where they will actually live** so the structure does not have to be redesigned later.

```yaml
campaign:                                # RESERVED, refused in v1
  epoch_id: <str>                        # the recording's data hash
  inputs: {<product>: <content hash>}    # beam, cal solution, flag table
  represents: {<product>: [<global latent names>]}
  archive: {path: <str>, format_version: 3}
  floors: {<latent>: <value node>}       # diagnostics.systematic_floor
  compress:                              # THE SECOND ROUTE, absent from v0
    method: linear | reduced_basis
    n_basis: <int>                       # refused above the whitened bank's numerical rank
    bank: {n_draws: <int>, seed: <name>}
    select: svd | greedy
    seed_scores: true
    nuisances:
      <name>: {design: <value node>, prior_std: ..., prior_mean: ..., shape: [...]}

inference:
  parameters:
    <name>:
      scope: per_epoch | linked          # RESERVED
      support: [lo, hi]                  # RESERVED — a PER-LATENT interval, not a campaign key
      hyper: {of: [<global names>], python: "mod:builder"}   # RESERVED — the EIGHTH python site
  transitions:
    <name>: {ou: {tau_epochs: <int>, sigma: ..., width: ...}}   # RESERVED

outputs:
  write:
    memory_archive: {...}                # RESERVED
```

Three corrections to the v0 hole, each from the source:

1. **The reduced-basis route was missing entirely.** `build_reduced_basis(space, pipeline, state, *, noise, bank, n_basis, at, names, method="svd", seed_scores=True, support, extra_directions)` (`inference/reduced_basis.py:425`) and `compress_reduced_basis` (`inference/compress.py:614`) are the route for models that are not affine, and `compress_linear` (`:280`) additionally takes `templates`, `nuisance_design`, `nuisance_prior_std`, `nuisance_prior_mean`, `nuisance_shapes`. `n_basis` is refused above the whitened bank's numerical rank (`reduced_basis.py:371 numerical_rank`), so it is a checkable key, not a free knob.
2. **`support` is per-latent, not per-campaign.** It is the interval the prior-draw bank populated, and it belongs next to `prior`.
3. **`hyper` is an eighth `python:` site.** `Factorization.hyper` (`inference/factorize.py:30`) maps a per-epoch latent to `(global latent names, builder)` where `builder` is a Python callable. §2.3's list is nine-in-v1, ten-at-capability-4, and that is stated rather than discovered later.

**`score_directions` and `basis_fidelity` are not deferred.** `score_directions` ships as a v1 run kind (§4.7.9); `basis_fidelity(basis, scores)` (`reduced_basis.py:684`) then arrives at capability 4 consuming an already-expressible product, instead of bringing its own input surface. The measured case for seeding: an unseeded basis leaves the `t21_depth` score direction with residual fraction 0.562 at n_S=3, against 1.5e-16 seeded — "a complete repair rather than an improvement".

**`hashlib` appears nowhere in `src/`.** v1 computes and records SHA-256 for every input file (§4.8), so `campaign.inputs` becomes derivable later at zero user cost. That is the decided answer to v0's open question, and it matters: capability 4's whole correctness argument rests on two epochs not silently sharing an unmodelled input product — measured at 52.6σ at N=640 with every diagnostic clean.

---

## 9. Complete worked example

```yaml
# =============================================================================
# rhino_gibbs.yaml — SCHEMA v1
#
# Simulate a switched drift scan of a RHINO-like horn at FOUR and at SEVEN
# switch positions, show that four cannot carry six latents and seven can,
# then fit the four noise-wave families in closed form jointly with the
# foreground amplitude and index by NUTS.
#
# Run:  rheplicant run rhino_gibbs.yaml
# Needs: rhino-cal-jax (the `cal` extra), numpyro, limTOD >= 1.9.
# =============================================================================

schema_version: 1
defaults: [{from: rhino_v1, only: [runtime, observation.site]}]
plugins: [rhino_cal_jax]

# -----------------------------------------------------------------------------
runtime:
  jax_enable_x64: true            # required by A44: a driftscan projector is declared
  platform: auto
  seed: 20260806                  # -> State.key = jax.random.key(20260806)
  seeds:                          # open namespace; every realisation named once
    sample: 11
    gcr: 7
    observed_noise: 99
    sky_structure: 0

# -----------------------------------------------------------------------------
observation:
  meta: {telescope: RHINO, obs_id: tour-001}      # STATIC: part of the jit cache key

  freq:
    grid: {linspace: {start: 60.0, stop: 85.0, num: 8, endpoint: true}, unit: MHz}

  time:
    grid:  {arange: {start: 0.0, step: 2.0, num: 64}, unit: s}   # RELATIVE to run start
    epoch: {value: 1785312000.0, unit: unix_s}                   # -> meta[...]
    integration_time: {value: 2.0,   unit: s}       # required: the noise kind is radiometer_frozen
    channel_width:    {value: 3.125, unit: MHz}     # cross-checked against BOTH conventions

  site:
    lat_deg: {value: 53.2367, unit: deg}    # fans to every projector's lat_deg
    lon_deg: {value: -2.3085, unit: deg}    # recorded_only: nothing in src/ consumes it
    alt_m:   {value: 78.0,    unit: m}      # recorded_only

  pointing:
    mode: drift
    az_deg:      {value: 0.0,  unit: deg}
    el_deg:      {value: 90.0, unit: deg}
    selfrot_deg: {value: 0.0,  unit: deg}
    materialise: [pointing, selfrot_deg]    # written, not guessed
    lst:
      mode: uniform_turn                    # full sidereal turn, endpoint EXCLUDED
      n_time: n_time
      lst0_deg: {value: 0.0, unit: deg}

  switching:
    mode: cycle
    # THE ordered list. Fixes: the switch indices, the order of model.cal_loads,
    # the ROW order of noise_wave.gamma_src, and (for an ingested run) the
    # thermistor labels. Index 0 is the reserved literal `antenna`.
    order: [antenna, ambient, hot, noise_source]
    cycle: round_robin
    dwell: {value: 1, unit: samples}        # -> arange(n_time) % 4

  environment:
    temperature: {value: 280.0, unit: K}    # traced; read by ground_pickup if lit

# -----------------------------------------------------------------------------
resources:

  beams:
    rhino_horn:
      format: cst
      directory: ~/Dataspace/RHINO/CST_beams/HornDryGround
      suffix: ".txt"
      nside: 64
      phi0_deg:  {value: 0.0, unit: deg}    # required for cst: a fact about the horn
      phi_sense: ccw                        # required for cst: handedness
      quantity:  linear_power
      normalize: pixel_sum                  # required, no default — pairs with normalize_beam
      horizon:
        mode:     truncate_map              # 1.04x, not 8.2x
        el_deg:   {value: 90.0, unit: deg}  # only 90 is supported by this route
        apod_deg: {value: 3.0,  unit: deg}
      # exposes .maps and .sky_fraction (horizon_truncated_beam's second return)

  projectors:
    rhino_drift:
      engine: driftscan
      beam:  {ref: resources.beams.rhino_horn}
      lmax:  191
      lat_deg:     {from: site}
      az_deg:      {from: pointing}
      el_deg:      {from: pointing}
      selfrot_deg: {from: pointing}
      normalize_beam:   true                # required: false returns int(B T)dOmega
      horizon_mask:     false               # the map is already truncated
      uniform_sampling: true
      beam_iterations:  3                   # from_beam_maps(iterations=) — NOT mask_iterations
      freq_chunk: null
      optimizations: []

  s_params:
    # One entry per switch position plus the receiver. `kind: termination` and
    # `kind: cable` are analytic; `kind: touchstone` reads a VNA sweep. Composition
    # is by NAMING (`behind:`), never by nesting.
    antenna_open: {kind: termination, termination: open, n: n_freq, z0: {value: 50.0, unit: ohm}}
    antenna:      {kind: cable, behind: {ref: resources.s_params.antenna_open},
                   length: {value: 2.0, unit: m}, loss: 0.92, onto: freq}
    ambient:      {kind: termination, termination: resistive, impedance: {value: 10.0, unit: ohm}, n: n_freq}
    hot_short:    {kind: termination, termination: short, n: n_freq}
    hot:          {kind: cable, behind: {ref: resources.s_params.hot_short},
                   length: {value: 0.4, unit: m}, loss: 0.98, onto: freq}
    noise_source: {kind: touchstone, file: {path: sweeps/noise_source.s1p, format: touchstone},
                   component: s11, onto: freq}
    load_600:     {kind: termination, termination: resistive, impedance: {value: 600.0, unit: ohm}, n: n_freq}
    load_900:     {kind: termination, termination: resistive, impedance: {value: 900.0, unit: ohm}, n: n_freq}
    load_150:     {kind: termination, termination: resistive, impedance: {value: 150.0, unit: ohm}, n: n_freq}
    receiver:     {kind: termination, termination: resistive, impedance: {value: 45.0, unit: ohm}, n: n_freq}

  arrays:
    # A named value node. Anything computed once and referenced twice lives here,
    # regardless of what produced it.
    gain_t: {python: "my_pkg.instrument:slow_gain_drift",
             args: {time: {from_grid: time}, period: {value: 60.0, unit: s}, depth: 0.02}}

# -----------------------------------------------------------------------------
model:

  global_signal:
    depth:  {value: 0.5,  unit: K}
    centre: {value: 75.0, unit: MHz}
    width:  {value: 5.0,  unit: MHz}       # Gaussian sigma, not FWHM

  foregrounds:                              # `many`, SUM — order-free
    - amplitude:      {value: 2500.0, unit: K}
      spectral_index: 2.55
      ref_freq:       {value: 70.0, unit: MHz}    # deliver: static_float (foregrounds.py:41)

  beam_spill:
    # NOT `from: projector` — check A50 refuses that against a truncated beam,
    # where horizon_fraction() would return ~1.0 and delete the ground term.
    sky_fraction: {ref: resources.beams.rhino_horn.sky_fraction}
    t_ground:     {value: 290.0, unit: K}

  antenna_loss:
    efficiency: 0.97
    t_physical: {value: 293.0, unit: K}

  cal_loads:                                # `many`, FAN: keys are switching.order[1:]
    ambient:      {t_load: {value: 300.0,  unit: K}}
    hot:          {t_load: {value: 400.0,  unit: K}}
    noise_source: {t_load: {value: 1200.0, unit: K}}

  noise_wave:
    t_unc: {linspace: {start: -1.0, stop: 1.0, num: n_freq, endpoint: true},
            scale: 20.0, offset: 250.0, unit: K}       # 250 +/- 20 K tilt, as a STATEMENT
    t_cos: {zeros: [n_freq], unit: K}
    t_sin: {full: {shape: [n_freq], value: -40.0}, unit: K}
    t_rx:  {linspace: {start: -1.0, stop: 1.0, num: n_freq, endpoint: true},
            scale: 5.0, offset: 290.0, unit: K}
    # from_switch_order stacks by NAME over observation.switching.order — so a
    # gamma_src transposition (shape-legal, tens of kelvin) cannot happen.
    gamma_src_re: {from_switch_order: {resource: resources.s_params, part: re}}   # (n_source, n_freq)
    gamma_src_im: {from_switch_order: {resource: resources.s_params, part: im}}
    gamma_rec_re: {ref: resources.s_params.receiver, part: re}                    # (n_freq,)
    gamma_rec_im: {ref: resources.s_params.receiver, part: im}
    switch_key: receiver_input                                                    # deliver: static_str

  bandpass:
    # SHAPE only, mean 1. The gain carries the absolute level.
    bandpass: {file: {path: bandpass_measured.npy, format: npy, unit: dimensionless},
               normalize: mean1}

  gain:
    gain: {ref: resources.arrays.gain_t, unit: dimensionless}    # (n_time,)

  adc:
    scale:  {value: 0.25, unit: adc_count/K}   # unit UNSTATED in source; declared here
    n_bits: 12                                 # deliver: static_int (adc.py:34)

# -----------------------------------------------------------------------------
variants:
  seven_position:
    observation:
      switching:
        order: [antenna, ambient, hot, noise_source, load_600, load_900, load_150]
    model:
      cal_loads:
        load_600: {t_load: {value: 600.0, unit: K}}
        load_900: {t_load: {value: 900.0, unit: K}}
        load_150: {t_load: {value: 150.0, unit: K}}

# -----------------------------------------------------------------------------
inference:

  twin:
    without: []          # `noise` is not a model node: the scatter is in `observed.realise`

  observed:
    from: simulation
    twin: full           # the default, stated: the simulator is the FULL model
    realise:
      kind: radiometer   # fractional, d -> d(1+w) — PACKAGE CHANGE, see section 11
      seed: {from: runtime.seeds.observed_noise}

  parameters:
    t_unc: {init: {zeros: [n_freq], unit: K}, prior: {normal: {loc: 0.0, scale: 400.0}},
            linear: true, into: noise_wave.t_unc}
    t_cos: {init: {zeros: [n_freq], unit: K}, prior: {normal: {loc: 0.0, scale: 400.0}},
            linear: true, into: noise_wave.t_cos}
    t_sin: {init: {zeros: [n_freq], unit: K}, prior: {normal: {loc: 0.0, scale: 400.0}},
            linear: true, into: noise_wave.t_sin}
    t_rx:  {init: {zeros: [n_freq], unit: K}, prior: {normal: {loc: 0.0, scale: 400.0}},
            linear: true, into: noise_wave.t_rx}

    fg_log_amp:
      init:      7.6009                       # log(2000 K)
      prior:     {normal: {loc: 7.6009, scale: 0.5}}
      linear:    false
      into:      foregrounds.amplitude
      transform: exp                          # positivity by construction
      latex:     "\\log A_\\mathrm{fg}"
    fg_beta:
      init:   2.30
      prior:  {normal: {loc: 2.30, scale: 0.3}}
      linear: false
      into:   foregrounds.spectral_index
      latex:  "\\beta"

  noise:
    kind: radiometer_frozen        # the one form the conjugate seam accepts
    source: observed               # sigma = |observed| / sqrt(dnu * tau)
    channel_width:    {from: observation}
    integration_time: {from: observation}
    floor: {value: 0.0, unit: K}

  checks:
    identifiability: {mode: refuse, rtol: 1.0e-8, report: true}
    linearity:       {mode: refuse, report: true}
    prior_sensitivity: {mode: skip, reason: "reported separately for this campaign"}

# -----------------------------------------------------------------------------
runs:

  # 1. The design question, answered before any fit is paid for.
  - name: ident_four
    kind: identifiability
    rtol: 1.0e-8
  - name: ident_seven
    kind: identifiability
    variant: seven_position
    rtol: 1.0e-8

  # 2. The refusal that IS the demonstration: four positions cannot carry six latents.
  - name: four_position_refusal
    kind: plan.estimate
    expect: refuse                 # non-zero exit if it SUCCEEDS; message -> refusal.txt
    blocks:
      - {names: [t_unc, t_cos, t_sin, t_rx]}
      - {names: [fg_log_amp, fg_beta], steps: 200}
    max_iter: 45
    tol: 1.0e-3

  # 3. The conditioning number, so the solver tolerance below is chosen, not guessed.
  - name: kappa
    kind: condition
    variant: seven_position
    names: [t_unc, t_cos, t_sin, t_rx]

  # 4. The fit: warm start to the mode, then Gibbs.
  - name: gibbs
    kind: plan.sample
    variant: seven_position
    blocks:
      - {names: [t_unc, t_cos, t_sin, t_rx]}     # engine DERIVED: conjugate
      - {names: [fg_log_amp, fg_beta], steps: 25}  # engine DERIVED: gradient
    warm_start:
      kind: plan.estimate
      max_iter: 45
      tol: 1.0e-3
      blocks: [{names: [fg_log_amp, fg_beta], steps: 200}]   # 200 Adam steps, not 25
      move: [fg_log_amp, fg_beta]                            # conjugate inits stay at zero
    n_sweeps: 26
    warmup:   8                                              # 26 - 8 = 18 >= MIN_DRAWS(4)
    rhat_max: 1.05
    seed: {from: runtime.seeds.sample}
    solve_tol: 1.0e-6
    solve_guard: 1.0e-3
    check_identifiability: once

  # 5. An exact conjugate reference for the linear block, with an error bar.
  - name: exact
    kind: conjugate.gcr
    variant: seven_position
    names: [t_unc, t_cos, t_sin, t_rx]
    n_draws: 500
    prior_std: {t_unc: 400.0, t_cos: 400.0, t_sin: 400.0, t_rx: 400.0}   # PER MEMBER
    tol: 1.0e-10
    maxiter: 4000
    seed: {from: runtime.seeds.gcr}

  # 6. Did the Gibbs chain reproduce the exact answer?
  - name: gibbs_vs_exact
    kind: compare
    of: [gibbs, exact]
    metric: max_rel_diff
    tolerance: 1.0e-2

# -----------------------------------------------------------------------------
outputs:
  dir: results/rhino_gibbs
  clobber: false
  stdout: summary
  report:
    rows: [exact, gibbs]
    columns: [mean, std, seconds]
    reference: exact
    relative: [mean_sigma, width_ratio]
    format: [text, json]
  write:
    config:          true          # config.input.yaml + config.resolved.yaml
    provenance:      true          # versions, git sha, x64, seeds, input hashes
    assembly:        {format: json}    # lit / skipped / instances / receiver_input.names
    predicted:       {format: npz}
    observed:        {format: npz}
    estimate:        {format: npz}
    draws:           {format: npz}     # name-keyed {latent: (n_draw, *shape)}
    covariance:      {format: npz}
    recovery:        {format: json}    # truth, estimate, sigma, error, PULL
    identifiability: {format: json}    # rank, n_par, nullity, participation shares
    refusal:         {format: txt}
    diagnostics:     {format: json}    # a LIST of phases, with rhat_per_latent
    timings:         {format: json}
    signal_path:     {format: html, themes: [light, dark]}
```

---

## 10. What this implies for a GUI

The GUI is the schema's projection. Every top-level section becomes one panel; every key's *type* determines its widget; every `required-with-no-default` becomes a blocking field that cannot be silently filled; every check in §6 becomes an inline validation message that appears **before** the Run button is enabled. The two artefacts of §4.8 become two tabs: "what I asked for" and "what will run".

The overall frame is a **left rail of twelve sections with completeness badges** (complete / incomplete / refusing), a **centre editing pane**, and a **right dock** with three permanently-visible things: the live signal-path diagram, the validation ledger, and the YAML mirror. The YAML mirror is bidirectional and is the authority — a GUI that cannot round-trip to text is a GUI whose output cannot be reviewed, diffed, or put in a repository.

### 10.1 Global affordances

* **Required-with-no-default fields render as empty with a red "must decide" chip and a one-line quote from the source.** `normalize_beam` shows "false returns ∫B·T dΩ, not a temperature — measured 32838 K vs 200 K". `phi_sense` shows "getting it wrong mirrors 30–60% azimuthal structure into the wrong half of the sky and leaves every integral and every azimuthally-symmetric diagnostic unchanged". These are not tooltips; they are the reason the field has no default and they belong on the surface.
* **Every value node is one composite widget** with a form selector (scalar / array / draw / file / ref / from / stack / python) and a unit combo. The unit combo is *filtered by the key's dimension*, so a temperature field cannot offer `Hz`. For the five unit-unstated fields it offers the declaration tokens and is labelled "declaration, not measurement".
* **Static vs traced is shown, not hidden.** A static field gets a distinct chip ("static — part of the jit cache key") and its form selector offers only scalar and `python:`. This is the single most confusing thing in the package for a newcomer and putting it in the widget removes an entire class of error.
* **Every `{ref: ...}` field is a searchable picker over the resources that exist**, and hovering a ref highlights the target resource card. Object identity is drawn as a line in the diagram, because for a cross-engine comparison that identity *is* the experiment.
* **The Run button is disabled while any group-A check fails**, with the failing checks listed by number and message. Group-B and group-C checks run on a "Validate" button (they need the twin and one trace) and their results persist with a staleness marker.
* **Undo/redo over the config document**, and a diff view against the preset, so "what did I change from `rhino_v1`" is one click.

### 10.2 Section by section

**`runtime` — a small settings strip, always visible at the top.**
`jax_enable_x64` is a toggle that shows a live consequence line: when a driftscan/general-pointing projector is declared and the toggle is off, it turns red and quotes the O(10%) figure (check A44). `platform` is a segmented control. `seed` is an integer field with a dice button and a null checkbox — the null state is labelled "this run realises no randomness" and greys itself out when any stochastic node or realising exit is declared. `seeds` is an editable name→int table with a "used by" column populated from wherever each name is referenced, so an orphan seed is visible.

**`observation` — a wizard-shaped panel with a big discriminator at the top: Synthetic | From file.**

* *From file* is a **file drop zone**. On drop, the GUI reads the HDF5 header and shows what it found — but `freq_unit` stays an empty required radio pair (Hz | MHz) with the note "the file does not record it and its two producers disagree". Nothing autodetects it.
* *Synthetic* shows two axis builders. Each is a small sub-form (linspace / arange / file) with a **live preview strip**: first three values, last three, count, spacing, and the derived `n_freq`/`n_time` badges that the shape symbols elsewhere will resolve to. The time axis carries a **precision gauge** implementing check C1 — a bar showing `spacing / np.spacing(max|t|)` against the 100× requirement, turning red before the user commits to a 4-hour run at 0.05 s.
* `endpoint` is a checkbox that is *never* pre-checked without a visible consequence note, because the sidereal-turn contract is decided by exactly that key.
* `site` is a **map pin plus three numeric fields**, with `lon_deg` and `alt_m` visibly greyed and labelled "recorded only — nothing in the package consumes this". Greying an inert field is honest; hiding it is not, because a user will look for it.
* `pointing` is a **mode picker (none | drift | tracked | baked)** that swaps the sub-form entirely. `drift` shows three angle dials plus an LST sub-mode; `tracked` shows a file drop for the scan table with a column mapper; `baked` shows a required provenance block with a permanent "unverifiable" banner. `materialise:` is a two-checkbox row, defaulted but visible, because it decides which of two engines reads zeros.
* `switching` is the section that most needs a bespoke widget: **an ordered chip list**, `antenna` locked at position 0, drag to reorder, and each chip shows its derived switch index. Below it, a **cycle preview**: the first 24 samples of `coords.extra["receiver_input"]` as coloured blocks. Reordering the chips live-reorders the `cal_loads` cards, the `gamma_src` row list and the thermistor mapping, because those four are one fact (§4.1.5). A `mode: none` toggle sits above and collapses the whole thing.
* `environment` and `extra` are simple key/value tables. `humidity` carries the "unit unstated in source" chip.

**`sky` — not a section; two cards inside `resources`, cross-linked.** The GUI *should* present them together as a "Sky" tab that shows the sky-model card and the projector card side by side with the shared beam drawn between them, because that is how a user thinks about it — but it edits `resources.sky_models` and `resources.projectors`, and the YAML mirror shows exactly that. A tab is a view; inventing a `sky:` section would put the beam in two places.

**`beam` — a card with a format picker and, for `cst`, a directory drop zone.**
On drop, the GUI samples the export and shows a **HEALPix mollweide preview** and a peak-dBi-vs-frequency plot. `phi0_deg` and `phi_sense` remain empty and required, with the preview annotated by a rotating azimuth marker so the user can see what `phi_sense` does — a picture is the only honest way to ask that question. `normalize` is a required radio group whose selection updates a **live "output unit" indicator** at the bottom of the card: "with `normalize: pixel_sum` + `normalize_beam: true`, the forward model returns a temperature". That indicator is the pair, rendered. `horizon.mode: truncate_map` reveals `el_deg` (locked to 90) and an `apod_deg` slider, and the card then exposes `sky_fraction` as a **drag-out chip** the user drops onto `beam_spill.sky_fraction` — which is how the one-call-two-products problem becomes obvious rather than invisible.
`format: gaussian` shows a FWHM slider with the same preview; `format: inline` and `python:` show a value-node widget with a shape assertion.

**`instrument` — the centrepiece: the signal-path diagram IS the editor.**
Render `Assembly.to_html`'s graph, all 32 nodes, dim for unlit, solid for lit, dashed for reserved, hatched for junctions/selectors. Clicking a node opens its field form in a side sheet. Toggling a node on/off is literally `assemble` / `Assembly.without`. Junctions and the selector are **not clickable as operator slots** and say why on hover. `many` nodes show an instance count badge and open as a list (SUM/CHAIN, drag to reorder for `filters`) or a label-keyed table (FAN, order locked to `switching.order`). `compose:` appears as a "+ add stage" affordance on a single-instance node, and the stages get their own sub-nodes in the diagram with their names, so `gain.gain_lna` is a thing you can click and bind to. `at:` regions render as a box drawn around the covered nodes, and the region's key auto-syncs to `at[-1]` with a note.
Each node's form shows its fields with the static/traced chips, and each field shows its shape symbol resolution ("`[n_source, n_freq]` → `(4, 8)`") live.
`snapshot_before` is a small camera icon on each processing-segment node.
A persistent footer strip shows `lit: 16 · skipped-as-identity: 6 · reserved: 3` — the numbers `print(twin)` gives, which are currently only reachable by printing.

**`backend` — a sub-view of the same diagram, filtered to `segment: processing`.**
`flagging`, `averaging`, `apply_cal`, `filters`. Worth its own tab because these four are the destructive stages: the tab carries a banner — "these stages overwrite `data`; add a snapshot to keep the raw waterfall" — with a one-click "add snapshot" that sets `snapshot_before` and adds `snapshot/raw` to `outputs.write.aux`. `filters` is a drag-to-reorder chain with per-entry type pickers; `FourierBandFilter` gets a **band slider on a 0…0.5 cycles/sample axis** with the `low < high` constraint enforced by the widget geometry, which is strictly better than a validation message.

**`variants` — a tab strip with a "+ variant" button.**
Each variant is a **diff editor against the base**: only the patched keys are shown, with the base value greyed beside each. The diagram renders the base and the variant side by side with changed nodes highlighted. This is where the identifiability-vs-cadence comparison becomes a two-click operation instead of two files.

**`inference` — four stacked panels.**

* *Twin repair.* Two lists: `without` (multi-select over lit nodes, with stochastic nodes pre-flagged and a red banner when A30 fails) and `replace` (a node picker plus the same node form, with B8 shown inline if the leaf is also a latent).
* *Observed.* A discriminator (simulation | file | several). In simulation mode, the `at:` truth table appears with one row per latent, pre-filled from the model leaf and editable — which is where "simulate here, fit there" becomes visible instead of a footnote. `realise` is a small noise picker with its own seed reference.
* *Parameters.* A **table, one row per latent**, columns: name, init (value-node widget), prior (a distribution picker with a live density sparkline), linear (checkbox with a "checkable claim" chip), into (a **path picker that browses the assembled twin's leaf tree**, not a text box — the tree already exists via `_tag_leaves_with_paths`), transform (registry dropdown showing itsshape signature and its implied `fan`), fan, unit (inherited, greyed unless overridden), latex. An all-zero `init` shows an amber chip quoting the probe-scale fallback. Rows whose `into` fails to resolve show the package's own "landed on static configuration" wording inline.
* *Noise.* A kind picker (none | homoscedastic | radiometer | radiometer_frozen) that **greys the run kinds it is incompatible with** in the runs panel below, rather than waiting for check A27 — the three noise seams are the single most confusing thing in the inference layer and the GUI can make them a structural constraint instead of an error message. `axis:` appears only when sigma is 1-D and is then required. `include_logdet` appears only for prediction-dependent kinds, as a required radio pair labelled "full Gaussian density" vs "generalized least squares", with the `(1 + f²)` bias quoted under the second.
* *Checks.* Four rows, each a mode segmented control plus a `report` toggle plus a reason box that becomes required and red when `skip` is selected. The reason box is the point: switching off the only diagnostic that sees across Gibbs blocks should cost a sentence, and the sentence lands in the run log.

**`runs` — an ordered list of exit cards.**
Each card is collapsed to `name · kind · variant · on`. Expanding reveals **only the keys that kind accepts** — this is the section where v0's biggest usability failure lived, and a conditional form fixes it by construction. `plan.sample` shows blocks (a drag-to-assign widget: latents as chips, blocks as bins, with each bin's derived engine shown as a badge — conjugate or gradient — and `steps` enabled only on gradient bins); `nuts` shows chain-length fields; `optimize` shows the optimizer discriminator with `beta1/beta2/eps` revealed only for adam; `conjugate.*` shows `prior_std` as a **per-member table** when the block has more than one latent, with the scalar form offered only for a block of one. `expect: refuse` is a small toggle labelled "this step is expected to fail" that flips the card's border and adds `refusal.txt` to outputs. `warm_start` is a nested, collapsible card with its own blocks override and a `move` multi-select.
A "+ add exit" menu is grouped: **Diagnose** (identifiability, score_directions, condition, gradient, benchmark, compare), **Fit** (plan.estimate, optimize), **Sample** (plan.sample, nuts, conjugate.gcr, npe), **Forecast** (fisher, predict), **Evaluate** (forward, mmodes). Grouping them by *question asked* is how a user who has never written JAX picks the right one.

**`outputs` — a checklist with previews.**
`dir` is a folder picker with a clobber warning that shows what is already there. `write` is a two-column checklist; each checked item shows its expected filename and shape (`draws.npz — {t_unc: (18, 8), …}`), computed from the config before the run. `report` is a small table designer over the declared run names. `signal_path` shows its own live preview in both themes. `stdout` is a three-way picker whose "summary" option previews the exact lines that will be printed.

**`campaign` — present, permanently disabled, with its keys visible and a "capability 4" banner.**
Showing a deferred section greyed is better than hiding it: it tells a user the shape of what is coming and stops them inventing a different spelling for it.

### 10.3 Where the live preview belongs

Four places, and only four, because a preview that costs a forward evaluation must be asked for.

1. **Free, continuous, always on: the signal-path diagram.** It is pure structure — lit/skipped/reserved/materialised — and it updates on every keystroke. It is also the only thing that tells a user which of the 32 nodes their operator set actually lit.
2. **Free, continuous: axis and shape previews.** First/last values, count, spacing, resolved shape symbols, the time-precision gauge, the FourierBandFilter band, the prior density sparkline, the beam mollweide (computed once per beam load, cached by file hash).
3. **One trace, on demand: the "Validate" button.** Runs group B and C — `jax.eval_shape` for the transform signatures, one small Jacobian for `check_linearity` and `identifiability`, one forward evaluation for the ADC saturation fraction. Results render as a ledger with per-check status and the numbers, and go stale (visibly) on any edit that could change them. `identifiability` is the one worth spending the trace on unprompted the first time a `parameters` table is completed, because the answer changes the *design* — how many loads to build — and it is cheap relative to the fit it precedes.
4. **On demand, costed: "Preview forward".** One evaluation of the model on the state, rendering the predicted waterfall, the per-node taps if any are declared, the saturation fraction, and the uniform-sky probe (`projector.forward(full(200 K))` → mean) that checks `normalize_beam` did what the user meant. The button shows an estimated cost derived from `nside`, `lmax`, `n_freq` and the declared optimizations, because at nside 64 / lmax 191 / 32 channels the sky analysis alone is 176 ms of a 193 ms forward and 114 MB of its 114 MB peak.

Everything beyond that — the fit itself — is a job, not a preview, and the GUI should say so and hand back a run id.

### 10.4 Which sections are conditional on which keys

This is the dependency graph the GUI must encode; each edge is also a check in §6.

| this appears / becomes required | when |
|---|---|
| `observation.site.lat_deg` | any projector reads `{from: site}` |
| `observation.pointing` (non-`none`) | `model.observed_astro_sky` is lit or any projector is referenced |
| `observation.switching` (non-`none`) | `model.cal_loads` is non-empty, or `noise_wave.gamma_src` has >1 row |
| `observation.time.integration_time` / `channel_width` | `inference.noise.kind` starts with `radiometer` |
| `beams.phi0_deg` / `phi_sense` | `format: cst` (and are *refused* otherwise) |
| `beams.frame` | any raw-array format |
| `projectors.lst_ref_deg` | `optimizations` contains `cache_beam_rotation` |
| `projectors.mask_iterations` | `horizon_mask: true` (refused otherwise) |
| `runtime.jax_enable_x64` forced true | any driftscan / general-pointing projector |
| `runtime.seed` | a stochastic node, a realising `observed`, or a sampling run kind |
| `model.<node>.type` | `flagging` or `filters` |
| `cw_tone.line_width` bounds | the frequency grid and `lineshape` |
| `bandpass.transform: unit_mean_bandpass` | `bandpass` and `gain` are both free |
| `inference.twin.without` | `model` lights `noise` or `rfi_field` |
| `inference.noise.axis` | sigma is 1-D |
| `inference.noise.include_logdet` | the noise model is prediction-dependent |
| `runs[].seed` | kind is `plan.sample` / `conjugate.gcr` / `nuts` / `npe`; **refused** for `plan.estimate` |
| `runs[].n_draws` | kind is `conjugate.gcr` |
| `runs[].prior_std` | every latent in the block declares `prior: null` |
| `runs[].beta1/beta2/eps` | `optimizer: adam` |
| `checks.<name>.reason` | `mode: skip` |
| `acknowledge_double_count` | `beam_spill` and `ground_pickup` both lit |
| `acknowledge_float32_sky` | x64 off with a sky engine |
| `acknowledge_unconverged_covariance` | `conjugate.gls` did not converge |

---

## 11. Open questions for the package author

Each is a choice between named alternatives, with a recommendation. The first three are `PACKAGE CHANGE`s that v1 assumes; the rest are decisions the schema can live with either way.

**11.1 `MapSky` — ship it, and with which constructor?**
(a) Ship `rheplicant.radio.sky.model.MapSky` taking `maps` **and** the frequency grid it was built on, with two constructors — a value node and a `format: healpix` file. (b) Ship it file-only. (c) Leave it out and route every real sky through `python:`.
**Recommend (a).** It is re-declared by hand in nine places across `examples/` and `docs/` and is in none of `src/` (verified: `radio/sky/model.py` holds only `AbstractSkyModel`, `UniformSkyModel`, `PowerLawSkyModel`). (b) fails the two ported scripts whose maps are drawn, not read. (c) makes the most common configuration in the project also the least reproducible, and in one case requires importing an example module with side effects. Ship it with a docstring stating that `__call__` ignores its `freq` argument, so the "built on another grid" failure has a named place.

**11.2 What unit is the post-gain trunk in?**
(a) Declare the post-`gain` trunk `adc_count` and require `gain`/`adc.scale` to carry `adc_count/K`. (b) Declare it `dimensionless` and treat the absolute level as a fitted nuisance. (c) Require a top-level `observation.data_unit` and refuse a run whose declared data unit and model output unit disagree.
**Recommend (a), enabled by the compound-unit grammar (§4.9).** It makes `adc_count/K × K = adc_count` a checkable identity, which is what turns check A9 from a spelling rule into a dimensional one, and it is the only option under which A43 (tone protection vs flagging threshold) can compare two numbers. (b) throws away the check. (c) adds a second place to state the same fact. Note that `gain` sits *before* `adc` on the trunk (`bandpass → gain → noise → emi → adc`), so `gain` is genuinely `dimensionless` and only `adc.scale` carries `adc_count/K`; v0's guidance pointed the wrong way here.

**11.3 A fractional-noise realisation seam.**
(a) Add `RadiometerNoiseOperator` at the `noise` graph node (or widen `NoiseOperator` with `mode: additive|fractional`). (b) Let the config layer apply the scatter outside the model, as `observed.realise` (what v1 assumes). (c) Do neither and require the user to supply `observed` from a file.
**Recommend (b), with (a) as a later addition.** The scatter belongs to the *observation*, not to the instrument: `examples/three_ways_to_a_posterior.py:75-76` and `examples/sky_to_noise_wave.py:278` both add it outside the twin, and putting it in `model.noise` forces a sigma that must be kept equal to `inference.noise.sigma` by hand. (b) needs no new operator and lets the config cross-check the generator against the likelihood, which nothing does today. (a) is still worth having for a twin that must be self-contained.

**11.4 Where does the LST bridge live?**
(a) Add a thin `rheplicant.radio.site` adapter matching the `beams.py` precedent of a units-only seam, so `site:` becomes a real section the package can validate. (b) Let the config layer call limTOD's `generate_LSTs_deg` directly, leaving rheplicant unable to check it. (c) Require `lst` as an input array always and delete `lon_deg`/`alt_m` from the schema.
**Recommend (a).** `coords.extra["lst_deg"]` is required by both real engines and produced by nothing in `src/` except `uniform_lst_grid`; a units-only pass-through is exactly the shape `radio/beams.py` already established, and it turns two inert recorded-only keys into consumed ones. (c) is honest but gives up the one thing a user with a real observing log needs.

**11.5 Are `phi0_deg` / `phi_sense` per-horn or per-export-directory?**
(a) A preset may supply them but must mark them `provisional: true`, which the validator turns into a loud warning. (b) They stay required-in-the-user's-file forever. (c) They move into a per-beam-directory sidecar YAML (the pyuvdata CST-settings precedent) that ships with the beam and is hashed with it.
**Recommend (c), falling back to (b).** `beams.py` calls them "a fact about the as-built horn, not the file", which means they travel with the *beam*, not with the run — a sidecar hashed alongside the export is the only place they can live without being restated per config or guessed per preset. Until a sidecar exists, (b). (a) is the worst of the three: `provisional: true` is a warning a user will click past, and a mirrored beam passes every integral, peak and azimuthally-symmetric diagnostic unchanged.

**11.6 Should the config refuse `beam_spill` + `ground_pickup` together?**
(a) Keep `acknowledge_double_count: true`. (b) Refuse outright. (c) Warn only, matching the code.
**Recommend (a).** The code's own position is that "it is the caller's call to make deliberately"; an acknowledgement key preserves that while making the deliberateness textual and hashable. (b) overrides a physics judgement the package deliberately left open; (c) is a warning in a stream of warnings.

**11.7 Should the projector engine be selected or inferred?**
(a) Infer from `pointing.mode`. (b) Expose `engine:` explicitly and derive nothing. (c) Infer, but allow an `engine:` override that is cross-checked.
**Recommend (b), which is what v1 does.** `MatrixProjector` has no geometry to infer from, a run can legitimately hold several projectors of different engines over one pointing (which is `examples/driftscan_mmode.py`'s whole subject), and the code already cross-checks disagreeing coordinates to 1e-3 deg. Inference here buys one saved key and costs the ability to compare engines.

**11.8 Should `basis_matrix` design matrices ever come from a file?**
(a) Keep the refusal. (b) Allow a file plus a required `built_for: {n_time:, n_freq:}` provenance block checked against the grid. (c) Allow a file only behind `python:`.
**Recommend (a).** The failure `radio/t_sys.py` describes — "a smooth, plausible, wrong temperature" — is not detectable from the output, and `{kind, n_basis}` with `n` from the grid makes it structurally impossible. (b) makes the check possible but voluntary in practice (a copied `built_for` block is exactly what a copied basis comes with). (c) is available anyway and carries a stated cost.

**11.9 Two small plumbing gaps: `blocks[].learning_rate` and `SnapshotOperator`'s node.**
For the first: (a) plumb `learning_rate` through `Block` and `SamplingPlan.estimate` and make it a real key; (b) leave it unreachable deliberately and delete the reservation.
For the second: (a) add a `snapshot` node to `RADIO_GRAPH` at the head of the processing segment (`adc → snapshot → flagging`), skipped-as-identity when absent; (b) leave `snapshot_before` compiling to `At((node,), Pipeline(SnapshotOperator(name), op))`.
**Recommend (a) for `learning_rate`** — `engines.py` already accepts it, `plan.py` simply never passes it, and a gradient block whose step size is unreachable is the same gap that made `run.kind: optimize` unusable. **Recommend (a) for the snapshot node** — the `At` route works but addresses the snapshot by the node it precedes, which is confusing, and a first-class node makes it visible in the diagram and in `lit`.

**11.10 Two export inconsistencies.**
`GeneralPointingProjector` is in `radio.sky.__all__` and not in `radio.__all__` (verified at `radio/__init__.py:158`, which exports `DriftScanProjector` and `MatrixProjector`). `SnapshotOperator` and `LambdaOperator` are exported from the top level but have no `graph_node`.
**Recommend: export `GeneralPointingProjector` from `rheplicant.radio`.** A class the docs present as one of two real sky engines should be importable from the package's advertised entry point. The two operators' lack of a `graph_node` is deliberate for `LambdaOperator` and an oversight for `SnapshotOperator` (see 11.9).

**11.11 What should output containers be?**
(a) `npz` + `json` only in v1, defer the rest. (b) Default to HDF5 (`h5py` is already an optional extra) so inputs and outputs round-trip in one format. (c) Default to ArviZ NetCDF, which matches `Draws.samples`' name-keyed shape and gives every downstream diagnostic for free.
**Recommend (a) now, (c) as an added format for chains specifically.** `npz` matches what `docs/_generate_*.py` already caches and has no dependency; making NetCDF the default would put an optional dependency on the critical path of every run. But `draws` and `posterior_predictive` are exactly ArviZ's shape, and offering `format: netcdf` there is nearly free.

**11.12 Should `run.kind: benchmark` and `run.kind: compare` exist, or is that a test harness?**
(a) Ship both as run kinds (what v1 does). (b) Ship `compare` only, and leave timing to an external harness. (c) Ship neither and say so.
**Recommend (a).** Three of the package's own configuration knobs — `uniform_sampling`, `cache_beam_rotation`, `freq_chunk` — have *no* justification other than timing, and the docstrings substantiate them with measured numbers ("14.6 ms against 1.79 ms, 8.2×"; `freq_chunk` is "a pure loss below the memory ceiling"). A user cannot find their memory ceiling without measuring, and asking them to leave the config to do it defeats the config. Both kinds are thin wrappers over `time.perf_counter` and one reduction.

**11.13 Should a user-defined `SignalGraph` be expressible at all?**
(a) v1 supports `RADIO_GRAPH` and `kind: pipeline`; a custom graph is a `python:` matter. (b) Support a `graph:` section but **refuse** any graph with an aliased node outright, rather than reproduce a warning the package can only issue in `repr()`.
**Recommend (a).** `Assembly.aliased` is empty for every shipped graph and is the documented hazard for user-defined ones, and nothing intercepts a hand-rolled `eqx.tree_at` through an aliased node. Adding `kind: pipeline` covers the eight `examples/` files that build a linear chain; a genuinely new topology is a modelling decision that deserves Python and a test, not YAML.

**11.14 `to_mermaid` and themes.**
(a) SVG and HTML in the config, mermaid light-only (what v1 does). (b) Add `theme` to `to_mermaid` and offer all three themed.
**Recommend (b) as a one-line package change**, then make it (a)-plus-mermaid in the schema. `to_svg` and `to_html` already take `theme='light'|'dark'` (`core/graph.py:565,572`); `to_mermaid()` hard-codes the light palette and takes no arguments (`:559`). The asymmetry is invisible until someone embeds a mermaid block in a dark page.

**11.15 Should `checks.linearity` be mandatory at validation?**
(a) Always run it, matching `linear_operator(check=True)`'s own default and its comment ("leave it on; turning it off buys a class of silent, confident errors"). (b) Default it on but auto-skip blocks above a declared size.
**Recommend (a), with `mode: skip` + a required reason as the escape.** It costs three forward evaluations per declared-linear latent — negligible for a noise-wave block, non-negligible for a 10⁶-coefficient sky block — but (b) reintroduces exactly the size heuristic the package refuses to make elsewhere. A user with a 10⁶ block can write one sentence.

**11.16 Does the config compute input hashes, or demand them?**
(a) The config layer computes SHA-256 for every file reference (what v1 does), which makes `campaign.inputs` derivable later at zero user cost. (b) The user declares them, which is honest about what the package does today.
**Recommend (a).** `hashlib` appears nowhere in `src/`, and capability 4's entire correctness argument rests on two epochs not silently sharing an unmodelled input product — measured at 52.6σ at N=640 with every diagnostic clean. Computing them now means capability 4 lands on a populated foundation rather than an empty one, and the cost is one file read per input.