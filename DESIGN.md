# RHEPLICANT Architecture

The name is **REPLIC**a + **ANT**enna (a digital twin *is* a replica, and this
one is of a radio antenna) with the **RH** of **RH**INO in front — the horn
antenna the framework was first built for. A replica differentiable enough that
the same model which simulates an instrument can also be fit to it. Everything
below is in service of making that replica faithful, composable, and cheap to
differentiate.

The design record: **why** the framework is shaped the way it is, as
numbered decisions (D1–D13), each with the constraint that forced it. New
here? Read the [README](https://github.com/RHINO-Experiment/rheplicant#readme) for
the philosophy and the [guided tour](https://github.com/RHINO-Experiment/rheplicant/blob/main/docs/tour.md)
for the API — this document is for contributors
and for future-us wondering "why is it like this".

The goal throughout is a **reusable framework**, not a one-off simulator:
the radio telescope digital twin is the first application, not the design
center.

Contents: [Layering](#layering) ·
[D1–D5 core](#d1--state-is-an-eqxmodule-with-a-statictraced-split) ·
[D6 SumOperator](#d6--sumoperator-parallel-additive-composition-source-only-semantics) ·
[D7 inference seam](#d7--inference-treats-the-pipeline-as-data) ·
[D8 sky engine](#d8--modular-sky-skymodel--skyprojector) ·
[D9 filters](#d9--filters-are-linear-projections-raw-data-survives-via-snapshots) ·
[D10 callbacks](#d10--host-callback-boundary-policy) ·
[D11 graph assembly](#d11--composition-is-implicit-in-the-signal-path-graph-guided-assembly) ·
[D12 inference layer](#d12--bayesian-bridge-uncertainty-propagation-neural-surrogates) ·
[D13 atmosphere entries](#d13--atmosphere-is-an-equivalent-entry-pair-not-a-trunk-stage) ·
[D14 parameter spaces](#d14--parameter-spaces-what-is-inferred-vs-how-it-enters) ·
[Element taxonomy → modules](#element-taxonomy--module-map) ·
[Physics roadmap](#roadmap-physics-to-port-into-the-placeholder-contracts)

## Layering

```
rheplicant.core        State / Operator / Pipeline / SumOperator   (domain-agnostic)
rheplicant.radio       single-antenna operators, organized by element  (placeholder physics)
                   taxonomy: sky / environment / instrument / backend
rheplicant.inference   likelihood / calibration        (treats pipelines as data)
```

**Hard rule:** `rheplicant.core` never imports from `radio` or `inference`. If the
framework proves reusable beyond radio astronomy, `core` graduates to its own
package by moving one directory.

## Decisions

### D1 — State is an `eqx.Module` with a static/traced split

Traced leaves (differentiable, vmappable): `data`, `coords`, `env`, `aux`,
`key`. Static (treedef, jit-cache key): `meta` as a hashable `FrozenMapping`.

The user-facing rule is one sentence: *strings and labels in `meta`
(recompiles on change); numbers and arrays in `aux`/`env`/`coords` (traced).*
Two explicit channels instead of one ambiguous one — and `FrozenMapping`
rejects unhashable values at construction, so jit-cache corruption is
impossible by construction.

### D2 — Functional updates via `dataclasses.replace`

`state.replace(...)` re-runs converters and `__check_init__` on every update,
so validation is preserved along the whole pipeline. `eqx.tree_at` remains the
tool for surgical deep edits (e.g. one parameter inside a nested pipeline).
Validation is *structural only* (types, dtypes, ndim) — never traced values —
so it is jit-safe.

### D3 — PRNG protocol: `subkey, state = state.next_key()`

Randomness is data flowing through the state. Operators that draw randomness
must return the advanced state. Consequences: one seed reproduces an entire
run; keys are never reused; `vmap` over a batch of keys gives independent
realisations of the whole instrument.

### D4 — One abstract base, no hierarchy

`AbstractOperator` has exactly one abstract method (`__call__`). There are no
intermediate base classes; shared behaviour lives in helpers and composition.
Differentiable parameters need zero registration machinery — Equinox already
makes every array field a leaf that `eqx.partition` / `eqx.filter_grad` can
select.

`requires` / `provides` ClassVar tuples are a declarative contract
(documentation today, the hook for a future `pipeline.validate()`).

### D5 — Pipeline is an Operator (composite pattern)

Same `State -> State` signature, so pipelines nest freely. Execution is a
Python loop that unrolls under jit — correct for heterogeneous stages. A
`lax.scan` over homogeneous operator stacks is a complementary future pattern.
`run_with_intermediates()` is a separate diagnostics method, not a flag, so
the operator contract stays uniform.

### D6 — SumOperator: parallel additive composition, source-only semantics

Physical models are sums of independent components; `SumOperator` makes that
a first-class combinator alongside sequential `Pipeline`. Semantics chosen
deliberately narrow: branches are *source-type* operators producing
contributions on the shared coordinate grid; input `data` is stripped to
`None` before entering each branch (enforced, not merely documented — a
branch that tries to read caller data fails loudly), and branch writes to
`coords`/`env`/`meta`/`aux` are discarded (parallel writes have no
well-defined merge). Each branch receives its own PRNG subkey split
off the main chain, so stochastic branches draw independent randomness and
one seed reproduces the whole sum. Accumulation is leafwise
(`jax.tree.map`), with loud trace-time errors on shape or pytree-structure
mismatches and on branches producing no data — silent tuple concatenation
and NumPy broadcasting were real failure modes caught in review.

### D7 — Inference treats the Pipeline as data

`build_forward_fn(pipeline, state_template, filter_spec)` partitions the
pipeline into (trainable params, static skeleton) and closes over the
template, exposing `f(params) -> prediction`. Gradient calibrators, NumPyro,
and future neural surrogates all connect through this one seam; calibration
never contaminates the instrument description.

The seam has a second entry point, `ParameterSpace.forward_fn` (D14), over
*named* parameters rather than a partitioned subtree. Both survive: partition
answers "train everything under here" (surrogate weights), naming answers
"these specific quantities, possibly transformed or shared" (physical fits).

### D8 — Modular sky: SkyModel × SkyProjector

The sky term factorizes into *what the sky is* (`AbstractSkyModel`:
params → `(n_freq, n_pix)` maps; differentiable amplitudes / spectral
indices / moment coefficients) and *how it is seen* (`AbstractSkyProjector`:
maps → `(n_time, n_freq)`; `forward` + `adjoint` for linear engines),
composed by `SkySourceOperator`. Either half swaps independently, so the
same sky can be observed through limTOD beam convolution, a precomputed
projection matrix, or drift-scan m-modes — and the same engine serves
different skies. Three engines, no placeholders left. Two compute the
physics and are named for the observation geometry they serve; the third
takes the projection as data:
`MatrixProjector` (offline `generate_sky2sys_projection` matrix — fully
differentiable for fixed pointing/beam, no optional dependency) →
`GeneralPointingProjector` (**delivered**: pure JAX via the `limtod_jax`
package in the limTOD repo, general pointing, differentiable w.r.t. both
sky and beam alms, exact adjoint; the oracle-equivalence and adjoint
acceptance tests live in the `limtod_jax` test suite) →
`DriftScanProjector` (**delivered**: the specialization for RHINO's actual
geometry — fixed pointing, only LST advancing — via `limtod_jax.driftscan`.
One Wigner rotation for the whole scan instead of one per sample,
`O(lmax³ + n_time·lmax)` against `O(n_time·lmax³)`, equal to the general
engine to float64 roundoff. It is an OPTIMIZATION, not a physical
approximation, which is what makes the general engine usable as its test
oracle. Two further opt-ins — `to_reference_frame()` (pay the rotation once,
outside the inference loop) and `uniform_sampling=True` (FFT synthesis on a
uniform full-turn LST grid) — trade flexibility for speed explicitly rather
than silently. Because the pointing is projector *configuration* here rather
than per-sample data, coords carrying a disagreeing pointing are rejected:
silently substituting the projector's own would produce a finite,
correctly-shaped, wrong observation (D-principle 7)).
Linear projectors expose `adjoint` (verified by dot-product tests) because
map-making reuses it (D9).

### D9 — Filters are linear projections; raw data survives via snapshots

Sidereal-repeat extraction, sky-space (Wiener/map-making) filtering, and
fringe-rate/delay filtering are all projections `P d`; `AbstractLinearFilter`
fixes the shared semantics (`mode="extract"` → `P d`, `"remove"` → `d − P d`)
and concrete filters supply `P`. `SkySpaceFilter` solves the regularised
normal equations with matrix-free CG (`lax.custom_linear_solve` under the
hood), reusing the forward model's projector adjoint — so filters are
differentiable and their transfer functions can be marginalised in inference.
Filters run on calibrated data (`ApplyCalibrationOperator`) in ordinary
analysis Pipelines; `State.checkpoint(name)` / `SnapshotOperator` preserve
raw data beforehand (zero-copy — JAX arrays are immutable).

### D10 — Host-callback boundary policy

`jax.pure_callback` into numpy packages is reserved for inherently
non-differentiable steps — RFI flagging via MomentRFI
(`MomentRFIFlaggingOperator`), where the output is boolean and a gradient is
meaningless. It is NOT a way to borrow numpy physics: a callback bridge to
numpy limTOD shipped through 0.1.4 and was removed once both sky engines were
native, because a bridge that cannot be differentiated, vmapped or transposed
is dead weight in a framework whose point is all three — and it made a poor
reference besides, since XLA runs callback threads with FTZ/DAZ set, putting
healpy inside one ~1e-7 away from the same numpy code on the main thread. Use
numpy limTOD directly when a numpy reference is what you want. Callbacks must
never sit inside a gradient path; the flags they produce flow to inference
through
`MaskedGaussianLikelihood` (zero weight on flagged samples) and to
`SkySpaceFilter` noise weighting. Existing `aux["flags"]` are always passed
as MomentRFI's `prior_mask` so flaggers compose instead of clobbering.

### D11 — Composition is implicit in the signal path: graph-guided assembly

The canonical signal-path graph (`rheplicant/radio/graph.py`, rendered by
`Assembly.to_mermaid`) makes explicit composition unnecessary:
`assemble(*operators)` compiles a *set* of operator instances into the
Pipeline/SumOperator nesting induced on the graph — absent sources are
pruned, absent transforms contract to identity, junctions materialize as
SumOperator when two or more live branches converge (the upstream trunk
becomes branch 0). The folder is a *compiler*: the result is an ordinary
composite wrapped in `Assembly` (an operator carrying static lit/skipped
metadata), so jit/grad/`build_forward_fn`/`tree_at` are untouched.

Rules hardened by adversarial review:

- **Determinism**: junction branch order is the graph's edge declaration
  order, never call-site order — same provided set ⇒ identical tree, PRNG
  stream, and jit cache entry (regression-tested bitwise).
- **Source provenance**: every materialized Sum branch must contain a live
  source; a transform-rooted branch is an assembly-time `AssemblyError`, not
  a NoneType crash inside physics code.
- **Caller-data regimes**: a sourced assembly rejects caller `state.data`
  (it would be silently discarded); a source-free assembly is a transform
  chain that *requires* caller data. Both checks are structural (jit-safe).
- **Junctions are never operator slots**; multi-instance is allowed on
  `many=True` nodes (sibling Sum branches for sources, call-order chaining
  for the `filters` node).
- **Placement** is declared on the operator class (`graph_node` ClassVar,
  MRO-inherited so subclasses keep their base's slot), with `At(node, op)`
  as the per-instance escape hatch. Assembly metadata is hashable
  (`lit: tuple[str, ...]` + graph name); graph objects never enter the
  pytree.

**Equivalent-entry leaves**: the same physical effect may enter the chain at
different stages in different forms, so the graph reserves placeholder
leaves for each form even when no operator ships yet — ground spill as a
pre-beam *field* (`ground_field`, convolved by the shared beam node) or as a
post-beam *effective temperature* (`ground_pickup`, generic `t_sys_extra`);
the astro path as component fields through `beam` or pre-convolved via
`observed_astro_sky` (`SkySourceOperator`). The shared `beam` node stays a
single differentiable object (the #1 marginalisation target) rather than
fragmenting into per-operator copies.

**Selector nodes (switched signal paths)**: the fourth node kind. A
`selector` combines its live branches per time sample instead of summing —
`data[t] = branch[switch[t]][t]` — with the switching cycle read from
`coords.extra[<node_id>]` (observation configuration, not an operator
parameter; values index the branches in edge declaration order). Materializes
as `SelectOperator` (same branch semantics as Sum: data stripped, per-branch
subkeys, context discarded); with one live branch it passes through, so the
canonical `receiver_input` selector costs nothing when no `CalLoadOperator`
is provided. This models the elements taxonomy's switched calibration loads:
the load REPLACES the antenna signal on the cycle.

**Region coverage**: an operator may claim a *contiguous path* of template
nodes (`graph_node`/`At` with a node tuple) and implement all of those stages
at once. Regions are atomic — endpoints must not be junctions/selectors,
claims are pairwise disjoint, and a live branch feeding a covered interior
node is an assembly error. A region is addressed by its LAST covered node id.
`observed_astro_sky` remains the preferred equivalent-entry route where one
exists; regions serve genuinely fused implementations.

**HTML rendering**: `SignalGraph.to_html()` / `Assembly.to_html()`
(`core/render.py`) generate a standalone page of the full template with lit /
half-lit ("wire") / dim styling and dashed reserved leaves — the signal-path
view of exactly what an assembly simulates, produced from Python so it always
reflects the actual graph (`examples/render_signal_path.py`).

### D12 — Bayesian bridge, uncertainty propagation, neural surrogates

All three complete the inference layer through the D7 seam, adding no new
runtime concepts:

- **NumPyro bridge** (`inference/numpyro_bridge.py`): priors ride on a
  `ParameterSpace` (D14), so sample sites are named by their latents. The
  earlier scheme attached priors to pipeline leaves *positionally* and
  reconstructed site names by walking pytree paths through
  Assembly/Pipeline/SumOperator containers; D14 deleted both the scheme and
  the 50-line walker. The likelihood is masked Gaussian (RFI flags →
  zero weight); `noise_std` may itself be a distribution.
  Cardinal rule, documented loudly: in a Bayesian model the noise lives in
  the LIKELIHOOD — hand the bridge a pipeline *without* stochastic operators.
  `predict_from_samples` runs the posterior through the pipeline
  (`eqx.filter_vmap` over sampled leaves only).
- **Uncertainty propagation** (`inference/uncertainty.py`). Design choice
  among (a) Fisher/delta-method, (b) MC pushforward, (c) Laplace: implement
  (a) as the primary API — the domain-standard forecast, exact for linear
  models, uniquely cheap here because `jax.jacfwd` gives exact Jacobians —
  plus (b) as `push_forward` (pairs with the NumPyro posterior); (c) is the
  composition of the two (MAP fit + `parameter_covariance` + sampling) and
  is documented rather than wrapped.
- **Neural surrogates** (`radio/surrogate.py`): `NeuralOperator` wraps an
  `eqx.nn.MLP` as a positive spectral response `data * exp(MLP(freq))` —
  network weights are ordinary differentiable leaves, so training,
  Fisher forecasts and NumPyro sampling need zero ML-specific machinery.
  Deliberately no default graph node: surrogate placement is a modelling
  decision (`At("bandpass", NeuralOperator.create(...))` replaces the
  physical bandpass with a learned one). `AdamCalibrator` (pure JAX, no
  optax) joins `GradientCalibrator` because fixed-step GD demonstrably
  stalls/collapses on MLP weights (the `exp` parametrization has a
  vanishing-gradient region) while Adam recovers a rippled bandpass to <1%.

### D14 — Parameter spaces: what is inferred vs how it enters

The positional-prior scheme (D12) could only infer a quantity the pipeline
already held, with a prior matching that leaf's shape exactly. Every
re-parameterization worth having therefore required *editing the instrument
description*: two scalars determining a beam's harmonic expansion, one gain
tied across three stages, a positive quantity explored in its logarithm. That
is precisely what D7 exists to prevent, so the fix belongs in the inference
layer.

Two objects, each with one job:

- **`Latent`** — a named quantity you infer: name, initial value (which fixes
  shape and dtype), optional prior, optional `linear=True`. Knows nothing
  about the pipeline.
- **`Bind`** — a rule turning latents into pipeline leaf values: which latents,
  which leaves (`eqx.tree_at` selectors), and the function between. Knows
  nothing about priors.

`ParameterSpace` holds both and compiles the bindings into **one** `tree_at`
call, so two bindings targeting a leaf raise instead of one silently winning.
`ParameterSpace.raw` takes a bind function outright for what the blocks cannot
express. Design choice among (a) a free-form bind function, (b) declarative
blocks, (c) blocks compiling to a function: (c), because a free-form function
is opaque to validation and to the linear-block export — and validation is
most of the value here, since every failure mode in this area yields a finite,
correctly-shaped, *wrong* inference rather than an exception. The checks: unique
names, every binding names a declared latent, **every latent is bound by
something** (an unbound latent samples happily and returns the prior), no leaf
written twice, every selector reaches a real array leaf, produced shape and
dtype-kind match the target, prior shape matches init, and binding preserves
the pipeline's treedef — the invariant `filter_vmap`, `jit` and `ravel_pytree`
all rest on. All of it runs on `jax.eval_shape`, so validation is free and
never optional.

Binding preserving the treedef is what let every consumer stay unchanged: the
calibrators needed no edit at all (a dict is a pytree), and posterior
predictive still `filter_vmap`s a stack of structurally identical pipelines.

**Linear blocks** (`inference/linear.py`). `linear=True` asserts the prediction
is affine in one latent — the case that matters for sky `alm`s and noise-wave
amplitudes, ~10⁶ real degrees of freedom where gradient samplers are hopeless
and a conjugate-Gaussian solve is exactly right. The assertion is *checked*
before it is exploited (`check_linearity` compares the model against its own
linearization), because a false declaration would otherwise produce a
confident, wrong posterior. Probes span 10⁻³–10³ times the latent's magnitude:
`x + εx²` is indistinguishable from linear near the origin, so a
moderate-probe suite signs off on exactly the blocks that fail in a sampler's
tails. A block fails only if it exceeds a relative tolerance **and** an
absolute roundoff floor — without the floor the relative measure explodes at
small probes and rejects correct blocks, and a spuriously-failing check gets
switched off, which is worse than no check.

`linear_operator` exports `A`, `Aᵀ` and the offset without forming a matrix
(`jax.linearize` + `jax.vjp`). `wiener_solve` gives the posterior mean by CG;
GCR sampling adds a fluctuation term to the same right-hand side and is the
next thing this operator is for. Two details that were measured rather than
assumed: `jax.vjp` returns the *conjugate* gradient for complex latents, so
the identity that holds is `Re Σ x·adjoint(y) == Σ (Ax)·y` (the real inner
product — the pairing a Gaussian likelihood forms); and `wiener_solve` carries
complex latents as `(real, imag)`, because a real prediction makes the map
ℝ-linear but not ℂ-linear and a Krylov method over ℂ would solve a different
problem. Its normal operator comes from gradients of the objective itself, so
it is SPD by construction with no adjoint-convention arithmetic left to
mis-handle.

Fisher matrices over a space carry their rows' names (`cov.sigma("fwhm")`,
`cov.block("fwhm", "log_gain")`), with spans derived from the actual
flattening rather than an assumed dict ordering. `FlatMatrix.kind`
distinguishes Fisher from covariance so `sigma()` refuses on the former:
`sqrt(diag(F))` looks exactly like an error bar and ignores every degeneracy.

### D13 — Atmosphere is an equivalent-entry pair, not a trunk stage

Graph v1.1 placed `atmosphere` as a trunk transform between `t_ant_sum` and
the `receiver_input` switch. The position satisfied the two hard boundary
constraints (calibration loads must not see the sky; atmospheric emission
arrives through the antenna and suffers the `(1-|Gamma|^2)` reflection loss)
— but the argument for *trunk transform* over *sum branch* was an intended
upgrade to radiative transfer `T' = e^(-τ) T_ant + T_atm (1 - e^(-τ))`, and
that upgrade is wrong physics at that position: opacity applied to the whole
antenna-temperature sum would attenuate `ground_pickup`, which never crosses
the atmosphere. (Spotted by Zheng: "atmosphere 应该跟 t_sys_extra 平行".)

Graph v1.2 therefore treats the atmosphere exactly like ground spill — one
physical effect, two equivalent entrances:

- `atmosphere` — **source leaf** into `t_ant_sum`, parallel to
  `ground_pickup`/`t_sys_extra`: the beam-averaged emission as an additive
  effective temperature (`AtmosphericEmissionOperator`, in
  `radio/environment/atmosphere.py`). Both boundary constraints still hold —
  every `t_ant_sum` branch sits before the switch and the noise-wave stage.
- `atmosphere_field` — **reserved transform** on the astro branch between
  `ionosphere` and `field_sum`: strict radiative transfer
  (`e^(-τ sec z) T_sky + T_atm (1 - e^(-τ sec z))` inside the beam integral),
  acting only on the signal that actually crosses the atmosphere — not on
  `rfi_field`, `ground_field`, or `ground_pickup`.

For a purely additive emission term the leaf form is mathematically identical
to the old trunk form (sum commutativity), so this is a semantic fix with no
numerical change to existing twins.

## Element taxonomy → module map

`rheplicant.radio` mirrors the element taxonomy of a single-antenna global-signal
experiment (source: `assets/elements.rtf`, local reference material — the
`assets/` folder is gitignored because it contains an unpublished draft).

```
Raw data elements                          Module
─────────────────────────────────────────  ─────────────────────────────────
Astrophysical
  21cm global signal (const LST, smooth ν) radio/sky/global_signal.py
  diffuse foregrounds (LST & ν variable)   radio/sky/foregrounds.py
  bright point sources (beam-diluted)      radio/sky/point_sources.py
Environmental
  ionosphere (distorts astro signal)       radio/environment/ionosphere.py
  atmosphere (emission; RT reserved, D13)  radio/environment/atmosphere.py
  ground pickup (sidelobes, T_ambient)     radio/environment/ground.py
  RFI (narrow+wideband, stochastic)        radio/environment/rfi.py
Instrumental
  beam (convolution, chromatic)            radio/instrument/beam.py
  DI gains (1/f + slower drifts)           radio/instrument/gain.py
  reflections + bandpass                   radio/instrument/receiver.py
  noise-wave T/Γ terms (GCR draft Eq. 1)   radio/instrument/noise_wave.py
  calibration signals (CW tone, loads)     radio/instrument/calibration.py
  self-generated EMI (comb-like)           radio/instrument/emi.py
  thermal noise (radiometer, T_sys)        radio/instrument/noise.py
  digitisation artifacts                   radio/instrument/adc.py
Processing
  flagging (MomentRFI)                     radio/backend/flagging.py
  averaging / integration                  radio/backend/averaging.py
  calibration application                  radio/instrument/calibration.py
  sidereal / sky-space / Fourier filters   radio/filters/
Modular sky machinery (D8)
  sky models (params -> maps)              radio/sky/model.py
  projection engines (maps -> TOD)         radio/sky/projection.py
  composed sky slot                        radio/sky/source.py
Graph-guided assembly (D11)
  SignalGraph template + folder            core/graph.py
  canonical single-antenna graph              radio/graph.py
```

Composition follows the physics, per the canonical signal-path graph
(`rheplicant/radio/graph.py`, D11): astrophysical components sum
(`SumOperator`), the ionosphere distorts that sum, RFI joins as a *pre-beam
field* (it enters through the sidelobes and is convolved by the shared beam
node), ground pickup joins as a *post-beam effective temperature*, and the
instrument chain is sequential (`Pipeline`). The chain order mirrors RHINO
paper Eq. 6, `P_rec = g (T_ant + T_nw + T_cw) + T_n`: sky-side temperatures
enter before the reflection/noise-wave terms, the CW tone joins *before*
bandpass and gain (it tracks gain drift only if it passes through the gain),
and thermal noise is added after the gain:

    astro = Pipeline(SumOperator(signal, foregrounds, point_sources), ionosphere)
    field = Pipeline(SumOperator(astro, rfi_field), beam)
    t_ant = SumOperator(field, ground_pickup, atmosphere)
    twin  = Pipeline(t_ant, noise_wave, cw_tone, bandpass, gain,
                     noise, emi, adc, flagging, averaging)

Identified pain points (beam uncertainties, foreground spectra, low-level
unflagged RFI, ground spill) are exactly where differentiable parameters +
marginalisation will matter most — each placeholder docstring records the
intended modelling strategy (beam-null degrees of freedom, moment expansion,
stochastic RFI variance, modulated topographic template).

## Roadmap (physics to port into the placeholder contracts)

The radio operators model a **generic single-antenna radio telescope**. The
primary source for real physics is **limTOD** (the in-house single-antenna TOD
simulator), which will itself be rewritten in JAX + Equinox; until then the
bodies stay placeholders. Instrument-specific parameters (e.g. RHINO's band,
horn beam, receiver noise-wave / reflection specs) enter later as concrete
operator *configurations*, never as framework assumptions.

Note: argosim was considered as a base but not used — it targets
interferometric arrays, while RHINO is single-antenna; limTOD is the right
upstream.

| Operator | Real model (generic single dish) | Source |
|---|---|---|
| GlobalSignalOperator | physical 21 cm models (troughs, physical params) | — |
| ForegroundOperator | uncertain spectral-index maps, moment expansion | limTOD, MERS |
| PointSourceOperator | source catalogue through sidelobes | limTOD |
| IonosphereOperator | chromatic absorption/refraction, time-variable | — |
| GroundPickupOperator | topographic template, alt/az modulation, beam-coupled | EM sims |
| RFIOperator | stochastic process model (night-to-night variance) | MomentRFI |
| BeamOperator | primary-beam convolution (harmonic alm rotation, ZYZ) | limTOD (TIBEC for full-Stokes); |
| AtmosphericEmissionOperator | opacity x ambient temperature, beam-weighted airmass (receiver temp lives in noise-wave T_0 / post-gain noise); strict RT reserved at `atmosphere_field` (D13) | instrument configs |
| ReceiverOperator | bandpass; reflection/impedance effects | instrument configs |
| NoiseWaveOperator | full Eq. 1 with F factor; T/Γ per frequency | noise-wave GCR draft |
| CWCalibrationOperator | tone shape/stability, switched reference loads | RHINO paper Sect. 4 |
| GainOperator | g(t) with 1/f flicker fluctuations | limTOD, hydra-tod |
| NoiseOperator | radiometer equation, 1/f covariance | limTOD, hydra-tod |
| EMIOperator | characterised switching-harmonic combs | lab measurements |
| ADCOperator | true quantization + straight-through estimator | — |
| FlaggingOperator | MomentRFI flags informing the noise covariance | MomentRFI |
| BackendOperator | integration, waterfall products | — |

Plus (delivered): NumPyro bridge, Fisher/delta-method uncertainty
propagation, Monte Carlo pushforward, neural surrogate operators, Adam
calibrator. Still ahead: GCR sampling of noise-wave parameters (draft
Eq. 28), optax integration for exotic optimizers, multi-experiment configs.

## Known deferred issues

- `data` is any pytree; the radio convention is a single `(n_time, n_freq)`
  array. Multi-stream data (TOD + CW tone) will adopt a dict-of-arrays
  convention — no State change needed.
- No enforced `data`↔`coords` consistency invariant (shape-changing operators
  must update coords manually; `BackendOperator` demonstrates the contract).
  Future `pipeline.validate()` can check `requires`/`provides`.
- `run_with_intermediates` keeps every stage's state in memory — diagnostics
  only.
- `SumOperator` discards branch writes to `coords`/`env`/`meta`/`aux` (by
  design, see D6) — an operator that must publish auxiliary output (e.g.
  flags) belongs in the sequential chain, not in a Sum branch.
- Typed PRNG keys (`jax.random.key`) are assumed throughout (jax ≥ 0.5).
