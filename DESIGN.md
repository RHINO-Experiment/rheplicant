# RHEPLICANT Architecture

The name is **REPLIC**a + **ANT**enna (a digital twin *is* a replica, and this
one is of a radio antenna) with the **RH** of **RH**INO in front — the horn
antenna the framework was first built for. A replica differentiable enough that
the same model which simulates an instrument can also be fit to it. Everything
below is in service of making that replica faithful, composable, and cheap to
differentiate.

The design record: **why** the framework is shaped the way it is, as
numbered decisions (D1–D54), each with the constraint that forced it. New
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
[D37–D52 config entry and scientific products](#d37--audit-facts-and-scientific-products-have-separate-schemas) ·
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

**The cost that rule buys, stated once.** "Recompiles on change" is the
correct behaviour for a label that changes what the program *is* (`telescope`,
`band`, a mode switch): one entry per distinct program is what a cache is for.
It is the wrong behaviour for a label that merely *identifies a run*, because
the compiled program is identical and the cache grows by one entry per
observation — monotonically, for the lifetime of the process, and every call
pays a full compile. The sharper test than "string or number?" is: **would the
emitted XLA differ if this field differed?** If no, the field does not belong
in `meta`, whatever its type.

One field in this package fails that test on purpose. `to_state` writes
`meta["time_epoch_unix_s"]` (`radio/rhino.py`), a different float for every
recording, because the epoch must survive in float64 and `Coordinates` stores
through `jnp.asarray`, which is float32 — the argument is in
`docs/ingestion.md`.
That is a deliberate trade, and it is safe only as long as no hot loop passes
such a `State` into a jitted function as an argument. Today none does: `src/`
contains no `jax.jit` at all, and the tests jit a fixed template state. The
obligation therefore falls on whoever writes the first per-observation loop,
which is the streaming-evidence campaign of D29. **D30 states what that loop
must satisfy**, why the remedy belongs at its boundary rather than in
`to_state`, and what it is measured to cost.

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
call. `eqx.tree_at` itself lets the last write to a leaf win silently, so the
no-double-write guarantee comes from `validate()`, which every entry point runs
first — not from the `tree_at` call.
`ParameterSpace.raw` takes a bind function outright for what the blocks cannot
express. Design choice among (a) a free-form bind function, (b) declarative
blocks, (c) blocks compiling to a function: (c), because a free-form function
is opaque to validation and to the linear-block export — and validation is
most of the value here, since every failure mode in this area yields a finite,
correctly-shaped, *wrong* inference rather than an exception. The checks: unique
names, every binding names a declared latent, **every latent reaches the
model** (declaratively by being named in a binding; for a raw bind, by probing
that perturbing it changes the bound pipeline — an unreached latent samples
happily and returns the prior), a raw bind and declarative bindings are never
both supplied, no leaf written twice, every selector reaches a real array leaf, produced shape and
dtype-kind match the target, prior shape matches init, and binding preserves
the pipeline's treedef *and* every leaf's shape and dtype kind — a treedef
alone encodes neither, so the structure check by itself would let a scalar be
broadcast into an array leaf. Almost all of it runs on `jax.eval_shape`, so
validation computes nothing and happens once per build rather than per
evaluation, which is why it is not made skippable.

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
(`jax.linearize` + `jax.vjp`). `wiener_solve` gives the posterior mean by CG, and `gcr_sample`
an exact posterior draw, by adding two white-noise terms to that same
right-hand side: its covariance is then the normal operator itself, so the
solve is distributed as the posterior rather than merely centred on it. One
CG solve either way — the fluctuation never touches the operator. Both take
`at=` to rebuild the block at the other latents' current values, because
linearity is a claim *given* them; that is the operation a Gibbs sweep is made
of, and omitting it would silently keep describing the model at its declared
starting point. Two details that were measured rather than
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

### D15 — The noise-wave data model is an imported package, not a rheplicant module

`NoiseWaveOperator` is an adapter over `rhino_cal_jax`, the JAX/Equinox
implementation of the noise-wave system temperature that lives in the
`RHINO-Experiment/rhino-cal` repository beside the numpy pipeline it was
verified against — 256 parameter cells agreeing to `1e-13` relative. The
dependency runs one way (`rhino_cal_jax` knows nothing about `State`,
`Pipeline` or operators) for the same reason `limtod_jax` does: the calibration
model has a life outside this framework, and a physics change should be
reviewable next to the reference implementation it must keep agreeing with.

The one thing the adapter adds is placement. Reflection coefficients belong to
*sources*, and the `receiver_input` selector discards source identity before the
`noise_wave` node sees the data. The operator therefore carries `Γ` per source
and re-reads the switch array — which the previous placeholder, holding one
scalar `Γ` for every sample, could not do.

Why that placement is the whole game: count equations **per frequency channel**,
since *while the temperatures are free per channel* nothing ties channels
together a priori. Each switch position contributes exactly one equation per
channel, so the design matrix has rank `min(n_src, k) × n_freq`, where `k` is
the number of **free temperature families**. That is why EDGES and REACH switch
between four or five calibrators.

Two sharp edges worth recording, because the loose version of this claim is
false in both directions, and a team picks a physical switching cadence off it.

**`k` is four, not three, whenever `T_rx` is fitted.** `t_rx` is a leaf of
`NoiseWaveOperator` exactly like `t_unc, t_cos, t_sin`; its coupling is 1 rather
than absent. Three distinct loads therefore make a three-family per-channel fit
square and leave a four-family one deficient by exactly `n_freq`.

**The count is per-channel and does not survive a basis.** Frequency structure
in `Γ` **does** identify *scalar*, frequency-independent noise-wave temperatures
from a single load, and basis matrices `U_unc`, `U_cos`, `U_sin` over
frequency are the general case of that: they tie channels together, and the
per-channel counting then stops applying in *both* directions with no counting
rule to replace it. Two loads and a three-coefficient basis identify all
`k · n_basis = 12` coefficients at `k = 4` where the per-channel count would say
6; a single load whose `Γ` is itself low-order in frequency falls *below* the
counting bound `min(n_src · n_freq, k · n_basis)`, because a basis function
times a low-order coupling is another low-order function.

So the rule for a basis parameterization is: measure it, with
`rheplicant.inference.identifiability`. Every number above is a measurement from
`tests/radio/test_noise_wave.py`, and `NoiseWaveOperator`'s module docstring is
where the statement lives in full.

**A decision worth flagging rather than burying in a code comment**: `Γ` is
stored on the operator as two real leaves per source/receiver
(`gamma_src_re`/`gamma_src_im`, `gamma_rec_re`/`gamma_rec_im`) rather than one
complex leaf. The reason is `jax.jacfwd`:
`rheplicant.inference.uncertainty.fisher_information` flattens the parameter
pytree and explicitly rejects any complex leaf before differentiating,
because a real-valued prediction makes the map R-linear but not C-linear, so
the Jacobian is only defined over the real degrees of freedom — confirmed by
reading `_flat_forward` and reproducing the raise on a toy complex parameter.
That is in tension with D14's whole point: D14 exists precisely so a
re-parameterization like "split a complex latent into real and imaginary
parts" is a `Bind` in the inference layer (`ParameterSpace`), not a shape
baked into the instrument description. The natural D14-respecting
alternative is one complex `gamma` leaf plus
`Bind(("gamma_re", "gamma_im"), into=..., fn=lambda re, im: re + 1j * im)`.
The split is also **preemptive**: nothing in this codebase currently calls
`fisher_information` on `NoiseWaveOperator`'s `Γ` — the only thing exercised
against it is `jax.grad` (`tests/radio/test_noise_wave.py`), which handles
complex leaves without issue. The cheapest time to undo the split, if D14's
principle is judged to matter more here than the two-line jacfwd guard, is
now, while the operator is still marked BREAKING (see CHANGELOG) — every
release after this one adds a migration cost.

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
  simplest constant sky (the SkyOperator)  radio/sky/uniform.py
Environmental
  ionosphere (distorts astro signal)       radio/environment/ionosphere.py
  atmosphere (emission; RT reserved, D13)  radio/environment/atmosphere.py
  ground pickup (sidelobes, T_ambient)     radio/environment/ground.py
  RFI (narrow+wideband, stochastic)        radio/environment/rfi.py
Instrumental
  beam (convolution, chromatic)            radio/instrument/beam.py
  horizon spill / ground mixing (D17)      radio/instrument/beam_spill.py
  antenna ohmic loss + emission (D16)      radio/instrument/antenna_loss.py
  extra T_sys on a separable basis (D28)   radio/t_sys.py
  DI gains (1/f + slower drifts)           radio/instrument/gain.py
  reflections + bandpass                   radio/instrument/receiver.py
  noise-wave T/Γ terms                     radio/instrument/noise_wave.py
  calibration signals (CW tone, loads)     radio/instrument/calibration.py
  self-generated EMI (comb-like)           radio/instrument/emi.py
  thermal noise (radiometer, T_sys)        radio/instrument/noise.py
  digitisation artifacts                   radio/instrument/adc.py
Processing
  flagging (MomentRFI)                     radio/backend/flagging.py
  averaging / integration                  radio/backend/averaging.py
  calibration application                  radio/instrument/calibration.py
  known-calibrator protection contract     radio/protection.py
  sidereal / sky-space / Fourier filters   radio/filters/
Modular sky machinery (D8)
  sky models (params -> maps)              radio/sky/model.py
  projection engines (maps -> TOD)         radio/sky/projection.py
  drift-scan m-mode engine (D20)           radio/sky/driftscan.py
  general-pointing engine (D20)            radio/sky/general_pointing.py
  CST far-field -> HEALPix seam (D25)      radio/beams.py
  composed sky slot                        radio/sky/source.py
Ingestion (files -> State)
  RHINO spectrometer HDF5 observations     radio/rhino.py
  Touchstone .sNp reflection sweeps        radio/touchstone.py
Learned stages
  MLP as an operator (D12)                 radio/surrogate.py
Graph-guided assembly (D11)
  canonical single-antenna graph           radio/graph.py
```

`rheplicant.core` is not part of that taxonomy — it is the layer the taxonomy
is written in, and nothing in it names a radio element. Its own map, complete
as of this listing (thirteen modules, `__init__.py` aside):

```
Core layer (never imports radio / inference)  Module
────────────────────────────────────────────  ───────────────────────────────
The state
  State: the scientific context               core/state.py
  time / frequency axes (+ their guards)      core/coordinates.py
  ambient conditions riding along             core/environment.py
  hashable mapping for the static channel     core/frozen.py
The operator contract
  AbstractOperator, Lambda-, Snapshot-        core/operator.py
  cascade (Pipeline)                          core/pipeline.py
  sum and switch (Sum-, SelectOperator)       core/combinators.py
Graph-guided assembly (D11)
  SignalGraph template + assembly             core/graph.py
  the fold (template + set -> tree)           core/fold.py
  lit / dim rendering of an assembly          core/render.py
Contracts
  reading a tree's own declarations           core/contract.py
  the DirtError hierarchy                     core/errors.py
Shared numerics
  separable (t, ν) design matrices (D28)      core/basis.py
```

The last two are the ones with no home in the element taxonomy and no obvious
one anywhere else: `contract.py` consumes the `requires`/`provides` ClassVars
(`walk_operators`, `stages_requiring`, `RANDOMNESS`) so a caller can refuse a
composition on what its stages say about themselves, and `errors.py` holds the
exception hierarchy every layer raises into. Both are narrated in
[`docs/contracts.md`](https://github.com/RHINO-Experiment/rheplicant/blob/main/docs/contracts.md),
alongside the one contract that lives on the radio side rather than here —
`radio/protection.py`.

Composition follows the physics, per the canonical signal-path graph
(`rheplicant/radio/graph.py`, D11): astrophysical components sum
(`SumOperator`), the ionosphere distorts that sum, RFI joins as a *pre-beam
field* (it enters through the sidelobes and is convolved by the shared beam
node), ground pickup joins as a *post-beam effective temperature*, and the
instrument chain is sequential (`Pipeline`). The chain order mirrors RHINO
`P_rec = g (T_ant + T_nw + T_cw) + T_n`: sky-side temperatures
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
| *(the `beam` node)* | primary-beam convolution (harmonic alm rotation, ZYZ) — **delivered**, but by the sky projector at `observed_astro_sky`, not by an operator on this node. `BeamOperator` was a scalar stand-in and was removed: applying it downstream of a projector double-counts the beam | limTOD (TIBEC for full-Stokes); |
| AtmosphericEmissionOperator | opacity x ambient temperature, beam-weighted airmass (receiver temp lives in noise-wave T_0 / post-gain noise); strict RT reserved at `atmosphere_field` (D13) | instrument configs |
| ReceiverOperator | bandpass; reflection/impedance effects | instrument configs |
| NoiseWaveOperator | the full system temperature with the F factor; T/Γ per frequency — **delivered** via `rhino_cal_jax` (D15) | `rhino_cal_jax` |
| CWCalibrationOperator | tone shape/stability, switched reference loads | RHINO paper Sect. 4 |
| GainOperator | g(t) with 1/f flicker fluctuations | limTOD, hydra-tod |
| NoiseOperator | radiometer equation, 1/f covariance | limTOD, hydra-tod |
| EMIOperator | characterised switching-harmonic combs | lab measurements |
| ADCOperator | true quantization + straight-through estimator | — |
| FlaggingOperator | MomentRFI flags informing the noise covariance | MomentRFI |
| BackendOperator | integration, waterfall products | — |

Plus (delivered): NumPyro bridge, Fisher/delta-method uncertainty
propagation, Monte Carlo pushforward, neural surrogate operators, Adam
calibrator, GCR sampling of noise-wave parameters (D15,
`examples/noise_wave_gcr.py`). Still ahead: optax integration for exotic
optimizers, multi-experiment configs.

### D16 — Antenna ohmic loss is a trunk stage, exactly where the atmosphere was not

Graph v1.3 adds `antenna_loss` between `t_ant_sum` and the `receiver_input`
switch — the position D13 emptied. That is not a reversal: the two effects
differ in precisely the property that decided D13.

D13 moved the atmosphere off the trunk because opacity applied to the whole
antenna-temperature sum would attenuate `ground_pickup`, which never crosses the
atmosphere. Provenance matters there: the attenuation happens *during*
collection, and different branches were collected along different paths. Ohmic
loss happens in the antenna's conductors and dielectric, *after* collection.
Every photon the beam gathered — sky, ground spill, atmospheric emission alike —
passes through the same lossy structure, so provenance no longer matters and the
trunk is the correct position.

The other boundary is the switch. Calibration loads connect at the receiver
input, downstream of the antenna, so they must not see this loss;
`antenna_loss → receiver_input` puts it on the antenna side of the switch. A
loss placed after the switch would attenuate the loads too and bias every
noise-wave solution built on them.

The form is Kirchhoff, not an arbitrary blend: `T_out = eta T_in + (1 - eta)
T_phys`, so an antenna at temperature `T` looking at a sky at the same `T`
delivers `T` for any efficiency. That isothermal fixed point is what
distinguishes the correct coefficient pairing from `(eta, eta)`, `(eta, 1)` or
`(1, 1 - eta)`, and it is asserted directly in
`tests/radio/test_antenna_loss.py`.

Finally, this is a **different loss** from `NoiseWaveOperator`'s
`c_s = (1 - |Gamma|^2)|F|^2`. That one is the impedance mismatch at the
antenna–receiver interface; this one is dissipation inside the antenna. They
multiply, and only the ohmic one contributes emission of its own. Folding an
efficiency into the noise-wave couplings would be indistinguishable from a
mismatch in the fit while silently dropping the `(1 - eta) T_phys` term.

### D17 — The horizon split is the astro branch's own stage, and f_sky is measured

1–3 % of RHINO's horn response is below the horizon and sees ground, not sky, so
the antenna collects `f_sky <T_sky>_masked + (1 - f_sky) T_ground`.
`DriftScanProjector(horizon_mask=True)` supplies the masked average and
`GroundPickupOperator` could supply the ground term, but nothing applied the
`f_sky` weight to the sky branch — which made masking, on its own, no better
than not masking: at a 3000 K sky either choice is a ~200 K bias.

`BeamSpillOperator` at the `beam_spill` node (graph v1.4) applies both halves.
Three decisions, each measured against a projector run on a sky map with the
ground painted in, at latitude 90 where the local horizon coincides with the
celestial equator and stops moving with LST — so the answer can be computed
directly rather than argued about. Residual on a ~200 K effect:

- **One operator, not a weight plus a separate ground leaf.** Split across two
  objects the two numbers can drift apart, and a sky branch weighted by `f`
  against a ground branch using `1 - f'` is a bias nothing structural can see.
  Here they sum to one by construction, and `BeamSpillOperator.from_projector`
  reads `f_sky` off the same beam that will supply the sky.
- **`f_sky` is a PIXEL-space partition of the beam, not the masked beam's
  harmonic integral.** The band-limited masked beam is a Gibbs approximation to
  a discontinuous target and its solid-angle integral is off by ~0.7 % at
  nside 16 / lmax 47, because `map2alm` of a sharply cut map does not preserve
  the mean. Using it leaves −17 K; the pixel partition leaves ~0. They are
  different objects: one is how the beam's solid angle divides, the other is
  how the visible part weights the sky.
- **The horizon ring counts HALF.** `limtod_jax.horizon_weights` uses a strict
  `el > 0`, so the pixels centred exactly on the horizon — a whole ring, 64 of
  3072 at nside 16 — get weight 0. A pixel centred on the horizon is half sky
  and half ground. Counting it as neither costs −8.6 K at nside 16, as all sky
  +8.7 K, and half **+0.005 K**; the two one-sided errors are symmetric and
  halve with nside, the signature of a miscounted ring rather than of anything
  harmonic. This one was found only by measuring: the first implementation used
  the strict cut and looked entirely reasonable.

Placement is the astro branch, not the trunk: `beam | observed_astro_sky ->
astro_ant_sum -> beam_spill -> t_ant_sum`. The split applies to the thing that
genuinely is a beam integral over the celestial sphere. The other `t_ant_sum`
leaves are *effective* temperatures by D13's construction, already carrying
whatever beam weighting their author intended, and `ground_pickup` in particular
IS a below-horizon share — running them through the split would weight them
twice. For the same reason `BeamSpillOperator` and `GroundPickupOperator`
together double-count the same spill; nothing forbids it, since a second ground
term can be legitimate (a building, a ground screen), so it is a choice to make
deliberately.

`beam_spill` and `antenna_loss` share the arithmetic `a x + (1 - a) b` and are
deliberately NOT one operator. A spill is a *mixture* — sky and ground at the
same temperature give that temperature, no loss — while ohmic loss dissipates
and re-emits at the antenna's own temperature. They carry independent physical
parameters at different points on the path, and merging them would make an
efficiency and a spill fraction indistinguishable in a fit.

### D18 — `many` instances compose the way their consumer composes

`many=True` at a source folded its instances into a `SumOperator`, always. That
is right for a junction and wrong for a selector: a switch picks one source per
sample, it does not add them up. So the graph could not express multi-load
switching — three distinct sources for an identifiable per-channel noise-wave
fit with `T_rx` held known, four with it free, per the counting in D15 — and
the workaround was to hand-build the `SelectOperator`,
which is how a `Pipeline`-instead-of-`SumOperator` bug got into an example and
survived until a gradient came back exactly zero.

The rule is now: **`many` instances fold as sibling Sum branches into a
junction, and as sibling selector branches into a selector.** `cal_loads` is
`many=True`, and the whole switching cycle comes out of `assemble()`. The
existing half of the rule — a junction or selector with one live upstream is
traversed as identity, so a load-free antenna chain needs no switch array and
leaves no `SelectOperator` behind — already held and is now pinned by a test.

Two deliberate limits. A source feeding BOTH a selector and a junction keeps
the Sum: there is no single right answer, and no shipped graph has such a node.
And the fan-out is kept beside `exprs` rather than widening it, so every other
path folds bit-for-bit as before — `SumOperator` splits the PRNG key per branch,
so a flatter tree would be a different seeded run, not merely a different shape.

`Assembly.__getitem__` was fixed alongside. A branch spanning
`observed_astro_sky -> beam_spill` is labelled by its first node, so a sibling
Sum named it `observed_astro_sky` while the Pipeline inside it had a stage of
the same name — and the breadth-first lookup returned the *fold rooted at* the
node instead of the operator *at* it, contradicting the documented contract and
making `eqx.tree_at(lambda a: a["observed_astro_sky"].sky_model.maps, ...)` fail
on an attribute the caller could see in the source. The lookup now descends
while the match keeps re-naming itself.

### D19 — A fixed pointing means a constant masked beam, so mask the map

`DriftScanProjector(horizon_mask=True)` masks the ALMS on every call: Wigner
rotation into the horizontal frame, synthesis, multiply, three rounds of
re-analysis, rotation back. Measured at nside 16 / lmax 47, that is **14.6 ms
against 1.79 ms unmasked — 8.2x**, and it was documented as an unavoidable cost
of asking for the horizon.

It is not. The horizon is static in the horizontal frame and a drift scan's
pointing is fixed by definition, so the masked beam is a CONSTANT.
`rheplicant.radio.beams.horizon_truncated_beam` truncates the beam MAP once,
before analysis: **1.04x**, and the same instrument to 2.8e-5 — the residual
being the alm->map->alm round trip the masking path takes *before* it masks,
which this one does not. (Spotted by Zheng: "horizon mask 在 horizontal
coordinate system 里是静止的 ... 不过是一个被 horizon truncate 的 beam".)

At a zenith pointing it needs no rotation at all, and that is provable rather
than assumed: `limtod_jax.horizon_weights` is a pure function of elevation,
limTOD's horizontal chart puts the ZENITH at the pole, and the beam-local chart
puts the BORESIGHT there. At zenith those poles coincide, so the charts differ
only by a rotation ABOUT that shared pole — which a pure-elevation mask is
invariant under. Azimuth and self-rotation are therefore irrelevant and the mask
applies to the beam-local map unchanged. Away from zenith the poles part and the
horizon becomes a tilted great circle in the beam-local chart;
`horizon_truncated_beam` refuses rather than hand-derive that rotation, and
`horizon_mask=True` keeps its place for exactly that case.

The function returns `f_sky` with the maps because they are the same sum, on the
same principle as `BeamSpillOperator.from_projector` (D17): a weight that
disagrees with the beam it was supposed to describe is a bias nothing structural
can catch.

### D20 — Beam-weighted-sky physics belongs to limTOD, not here

D19's `horizon_truncated_beam` and D17's `horizon_fraction()` were implemented
in this package. They should not have been. How a beam weights the sky, where
the horizon falls in it and what share of its solid angle survives are limTOD's
subject — exactly as the noise-wave data model is `rhino_cal_jax`'s (D15) — and
splitting that physics across two repos means two places to keep in step, two
test suites asserting the same conventions, and a downstream package quietly
accruing a beam library. (Raised by Zheng: "尽量让 rheplicant 模块化，跟 beam
weighted sky 有关的都交给 limTOD".)

Moved to limTOD 1.9 (`limtod_jax.driftscan`):

- `horizon_partition_weights(nside)` — the solid-angle partition, horizon ring
  counted half; a *different object* from the `horizon_weights` mask, which is
  unchanged because its masking semantics were never wrong;
- `horizon_truncated_beam(beam_map, ...)` — the map-space cut, returning the
  surviving fraction with it;
- `horizon_beam_fraction(beam_alm, az, el, selfrot, ...)` — the same fraction
  for any fixed pointing, from alms.

What stays here is **placement**: `BeamSpillOperator` consumes `f_sky` and puts
it at the `beam_spill` node; it does not compute it.
`rheplicant.radio.beams.horizon_truncated_beam` and
`DriftScanProjector.horizon_fraction()` are now pass-throughs whose only added
value is inferring `nside` from maps the caller already has, and both
feature-gate on the upstream symbol so an outdated install is named at the
boundary. The `limtod` extra is floored at 1.9.

The tests moved with the physics. rheplicant's beam suite no longer re-asserts
the partition, the half-counted ring or the zenith-only exactness — those are
locked in limTOD's `tests/limtod_jax/test_horizon_partition.py`, and duplicating
them here would be two copies of a moving target. What this side tests is the
seam: that the call arrives, that `nside` is inferred, and that a stale install
says so.

Still on this side at the time of writing: `cst_beam_maps` /
`read_cst_farfield`, the CST far-field reader — beam *ingestion* rather than
beam-weighted-sky computation. Done in D25.

### D21 — The noise level is a model, not an argument

`noise_std` was a bare scalar passed independently to five places —
`GaussianLikelihood`, `to_numpyro_model`, `fisher_information`, `wiener_solve`,
`gcr_sample`. Each of them assumed the answer is *given* and *constant*. The
radiometer equation says it is neither: `sigma = |d| / sqrt(delta_nu * tau)`, so
sigma is a function of the very quantity being inferred. (Framed by Zheng:
"给定噪声模型，我们可以定义 likelihood 和 posterior，或者说 loss function。
默认的噪声模型是使用 radiometer equation（因此是 multiplicative noise）".)

`rheplicant.inference.noise` makes it one object every route asks:
`HomoscedasticNoise` (what a bare sigma always meant), `RadiometerNoise` (the
default physics), and `FlaggedNoise` wrapping either.

Three decisions inside it, each of which had a plausible wrong answer.

**Flags reach the covariance by wrapping, not by a keyword.** A flagged sample
was not observed, which is a statement about its variance, so it belongs inside
the noise model — `FlaggedNoise` sets `sigma = inf`. The alternative, threading
`flags=` through five signatures, makes every consumer responsible for
remembering a convention. `inf` is only an encoding: `inverse_variance` and
`NoiseModelLikelihood` both give it a clean zero, because the limit does *not*
work — `r^2/sigma^2 -> 0` but `log sigma^2 -> inf`, so one flagged channel would
otherwise send the whole log-density to `-inf`.

**`depends_on_prediction` is a claim, not a hint.** It is what a solver branches
on: `False` means one solve, `True` means the covariance must be found before it
can be used (D22 / `iterative_gls`). It is also what makes `noise_std=`
polymorphic without ambiguity — jax and numpy arrays both *have* a `.std`
method, so a protocol keyed on that alone would swallow a bare sigma and call it
with the prediction.

**The log-determinant is kept by default.** With `sigma = sigma(theta)` the
Gaussian normalization is no longer an additive constant, and dropping it —
exactly what generalized least squares does — is a different estimator. For the
multiplicative model both are closed-form: GLS returns `sum d^2 / sum d`, biased
high by `(1 + f^2)`, while the full density is asymptotically unbiased. So
`include_logdet=False` exists and is documented as GLS rather than being the
silent default. The same term appears in `fisher_information` as the covariance's
own contribution, `2 (d log sigma/d theta)^T (d log sigma/d theta)`; under
`RadiometerNoise` it is the factor `(1 + 2 f^2)`, and omitting it forecasts error
bars too wide by `sqrt(1 + 2 f^2)` — a plausible number, and the wrong one.

`GaussianLikelihood` and `MaskedGaussianLikelihood` stay, and tests assert they
agree with `NoiseModelLikelihood` to roundoff: the seam generalizes them rather
than replacing them.

### D22 — A bridge that has never been executed is not a bridge

`MomentRFIFlaggingOperator` had been in the tree since the rename, with tests,
and **every one of them was skipped**: MomentRFI had no `pyproject.toml` or
`setup.py`, so it could not be installed into any environment, so the guard
`skipif(find_spec("MomentRFI") is None)` was true everywhere including CI. The
bridge was written against an API nobody had called.

Packaging MomentRFI (hatchling, flat layout; `matplotlib` an extra rather than a
dependency, because the package's `__init__` deliberately does not import
`plotting`) made the tests runnable, and they passed unmodified — the assumed
signature `fit(waterfall, kernels=..., prior_mask=...)` was right. That is a
good outcome and not evidence the arrangement was sound: it was one signature
away from silently shipping a bridge to nothing.

What the first real execution then bought, none of which a skipped test could
have claimed:

- **jit gives bit-identical flags.** The docstring asserted it; now a test does.
- **`kernel_shapes` earns the matched filter's sqrt(K).** On a 3-sigma-per-pixel
  blob under the fitter's default 4-sigma cut, round 0 recovers *none* of it and
  a single 3x3 box recovers *all* of it, at a 0.15% false-positive rate. That is
  the whole argument for broad rounds, and it was previously untested here.
- **The flags matter, measurably.** A persistent narrow-band emitter on 2 of 32
  channels biases a maximum-likelihood amplitude by +5.8%; routing MomentRFI's
  flags through `FlaggedNoise` (D21) recovers the truth and agrees to six digits
  with flagging the contaminated channels by hand.

That last test is the shape the others should have had from the start: it does
not assert a mask, it asserts the bias the mask removes. A flagger can be
checked against its own output forever and still be useless.

`rheplicant[rfi]` names MomentRFI rather than resolving it — it is not on PyPI —
which is the same arrangement `limtod` already had. The operator's ImportError
now names both install routes.

### D23 — GCR takes a covariance; iterative GLS is what supplies one

`gcr_sample` is a linear sampler *given* a covariance, and `wiener_solve` the
corresponding mean. Under `RadiometerNoise` the covariance is not given: sigma
tracks the prediction, so the weights depend on the solution and the solution
depends on the weights. (Framed by Zheng: "GCR 就是给定 covariance 的线性采样器；
只不过对于默认的 radiometer equation noise，covariance 通过 iterative GLS 确定".)

That framing is also the API. **`wiener_solve` and `gcr_sample` did not
change** — they already accepted an array `noise_std` broadcastable to the
data, and their linear-Gaussian correctness was already tested. The new module
supplies only the thing that *produces* sigma, and hands it over:

```python
found = iterative_gls(block, observed, noise=RadiometerNoise(dnu, tau), prior_std=P)
draw, _ = gcr_sample(block, observed, noise_std=found.noise_std, prior_std=P, key=k)
```

A matrix-free port of hydra-tod's `hydra_tod.linear_sampler.iterative_gls`:
that implementation forms a dense `U` and `N_inv`, while this one runs the same
fixed-point iteration on the `LinearBlock`'s JVP and VJP, which is what makes
10^6 degrees of freedom possible. A transcription of the numpy original lives
in the test file as the oracle — inlined rather than imported, because hydra-tod
imports `mpi4py` at module scope.

**The convergence tolerance cannot have a fixed default.** Two independent
floors bound how small a step is measurable, and the first default tried
(`1e-8`) violated both in turn:

- the arithmetic's epsilon — float32's is `1.2e-7`, so `1e-8` is rounding
  rather than a measurement, and the loop ran to `max_reweights` reporting
  `converged=False` for a run that had settled at `delta = 7e-8`;
- **the inner CG tolerance** — consecutive solves differ by roughly their own
  residual whatever the outer iteration does. This is the binding floor in
  float64, where the worked example's `tol=1e-10` sits five orders of magnitude
  above `eps`; `8*eps = 1.8e-15` was then *too tight* and failed the same way.

The default is `max(8*eps, tol)`. Both terms are derived, neither is tuned, and
a test pins the failure mode so the derivation is not quietly replaced by a
constant later.

**The mean can be weight-independent while the width is not.** In
`examples/gls_gcr.py` three switched loads meet three per-channel noise-wave
unknowns, so the reduced system is square — one solution, weights cancelling
out of it — and reweighting moves the point estimate by nothing, exactly. The
posterior covariance `(A^T Sigma^-1 A + S^-1)^-1` depends on Sigma regardless,
and a GCR draw is precisely a draw of that width: a frozen sigma there reports
error bars wrong by -8% to +8%, in both directions, on an estimate that was
already right. "The fit came out the same" is not evidence the covariance did
not matter. Where the system is over-determined across genuinely different
noise levels the estimate moves too — a factor of ~2.3 in recovered RMS on a
prediction spanning a decade.

Freezing sigma inside each solve is what makes every step linear-Gaussian, and
is also what makes the converged answer *generalized least squares* rather than
the maximum of the full Gaussian likelihood — the log-determinant's dependence
on the solution is held fixed rather than differentiated (D21). That is the
right estimator to condition a constrained realization on, because a GCR draw
is a draw at a *given* covariance. The full likelihood's posterior is a
gradient sampler's job, not this one's.

### D24 — An approximate posterior is only trustworthy where an exact one exists

Two engines were added on top of the noise-model seam (D21): NUTS through the
NumPyro bridge, and amortized neural posterior estimation. The ordering was the
design decision, not an accident of scheduling.

`to_numpyro_model` takes a `NoiseModel` in the `noise_std` slot, like every
other consumer. Nothing special is needed to get the log-determinant: with
`RadiometerNoise` the scale is a function of the sampled parameters, so
`Normal(loc, scale).log_prob`'s `-log scale` is already in the potential. That
makes NUTS the *full* Gaussian posterior, where `iterative_gls` (D23) converges
to generalized least squares — the two are different estimators and the bridge
now says which is which. An infinite sigma is masked rather than passed as a
scale, since `Normal(loc, inf).log_prob` is `-inf` and one flagged channel
would otherwise take the whole potential with it.

**NUTS is validated against `gcr_sample`, not against plausibility.** On a
linear-Gaussian problem the constrained realization is exact — closed form,
independent draws, no burn-in — so agreement is a real check on both: mean to
0.00 sigma and width to 1.00x.

Only then NPE, because **an approximate posterior has no internal notion of
being wrong**. A badly-fitted `q` returns a smooth, confident,
correctly-centred, incorrect density and reports nothing amiss. Both of its
failure modes appeared during development and push opposite ways:

| simulations | steps | components | width / exact |
|---|---|---|---|
| 8192 | 1500 | 1 | 0.88 |
| 8192 | 4000 | 1 | 0.84 |
| 8192 | 4000 | 2 | 0.60 |
| 32768 | 1500 | 1 | 0.98 |
| 32768 | 1500 | 2 | 1.07 |

Draws come from the prior, so only a fraction `sigma_post / sigma_prior` of
them land near any given observation and too small a bank cannot resolve the
posterior at all. Independently, too many steps on a small bank over-fits — and
over-fitting an NPE makes it too NARROW, which is the failure that looks like a
better answer. The training loss falls monotonically straight through it.

So `train_posterior` holds out a validation split by default and returns the
**best validation step, not the last**. In the worked example that is step 489
of 2000. Without it the shipped default would have been quietly over-confident,
and nothing in the output would have said so.

The estimator is deliberately a conditional Gaussian mixture rather than a
normalizing flow: exact at one component for a Gaussian posterior, which is
what makes the check above sharp, and few enough moving parts that the failure
modes stay legible. Adam is hand-rolled, as in `calibrate.py` — no optax.

### D25 — Beam ingestion is on the same line as beam physics

D20 moved the horizon partition to limTOD and left the CST far-field reader
here, calling the distinction "a follow-up, not a distinction worth defending".
It was not worth defending. Reading a measured horn into a beam map is the same
kind of thing as deciding how that beam weights the sky, and limTOD already had
the slot: `limTOD.uvbeam` does exactly this job for pyuvdata.

`limTOD.cstbeam` is now its sibling — `read_cst_farfield`, `cst_frequency_table`,
`cst_beam_maps`, plus a `cst_beam_func` that satisfies `TODSim`'s
`beam_func(freq=..., nside=...)` contract. That last one is what makes this a
migration rather than a relocation: a CST horn now drops straight into limTOD's
own simulator, which it could not do while the reader lived downstream. It also
caches each file's HEALPix resampling, so a 200-channel sweep over a 61-file
directory parses each file once instead of hundreds of times — an inefficiency
this side never had reason to notice, because rheplicant only ever asked for a
handful of frequencies at once.

What stays here is the seam, and the seam has exactly one job: **units**.
`Coordinates.freq` is in Hz; limTOD is in MHz throughout. Each package keeps
its own house convention and the adapter is where they meet, which is what an
adapter is for. rheplicant's tests were rewritten to check that conversion and
nothing else — a unit slip here would turn 70 MHz into 70 Hz and refuse every
legitimate request, or accept an illegitimate one.

The conventions moved with the code, and one of them could not move intact.
`limTOD.uvbeam` locks its azimuth mapping numerically, because pyuvdata *fixes*
that convention. CST does not: which physical direction the model's `+x` axis
points is a fact about how the horn was built and mounted, and it is simply not
in the export. So `phi0_deg` and `phi_sense` remain knobs whose defaults are an
assumption to check rather than a result, and what the upstream tests lock is
that the knobs act correctly — `phi_sense` a reflection about `phi = 0` rather
than a relabelling, `phi0_deg` a rotation conserving the integral. For RHINO's
horn, which varies 30-60 % around the `theta = 30` deg ring, a wrong handedness
mirrors that structure into the wrong half of the sky while leaving every
integral, every peak and every azimuthally-symmetric diagnostic unchanged.

RHINO's own horn stays tested here, against the real export: that is this
package's subject, not limTOD's. The synthetic convention tests went upstream
with the reader.

**Why the runtime gates check symbols and not versions**, here and in D20. The
`limtod` extra is floored at 1.10 for a resolver's benefit, but
`_require_cstbeam` imports the module and `_require_limtod_jax` tests
`hasattr` — neither reads `__version__`. An editable install reports whatever
version its dist metadata was written with, and that goes stale the moment the
source moves ahead of the last `pip install -e`. This package's own development
environment was found sitting at a recorded `1.8.0` while running 1.10.0
source, with `limTOD.cstbeam` importable the whole time: a version check would
have refused a fully capable install, and refused it with a message about
upgrading something already newer than asked for. The two checks are not
redundant — the floor describes what to install, the symbol describes what is
actually there.

### D26 — A plan is one declared partition, two exits, and two guards no per-block number can replace

`wiener_solve` and `gcr_sample` already answer for **one** block, and they
already share an implementation (`_conjugate_solve`, `key=None | k`).
`SamplingPlan` promotes that to the level a whole model is inferred at: one
declared partition of the space into `Block`s, swept to a fixed point by
`estimate` or drawn from by `sample`.

**Two methods rather than a mode flag.** `key=None | k` is the right
*implementation* and the wrong *interface*: a caller's intent is "give me the
best fit" or "give me draws", not "here is a PRNG key". Two signatures make the
invalid combinations unrepresentable instead of merely validated — `key` is
required on one and absent from the other, `n_sweeps`/`warmup` belong to one and
`max_iter`/`tol` to the other. What they share is everything up to the last
step, which is exactly where the layer below already diverges.

**The engine is derived, never restated.** `Latent(..., linear=True)` already
says which machinery a latent can take, so `Block` does not ask again. One case
is genuinely ambiguous — a block mixing declared-linear and non-linear members —
and it is refused rather than guessed; `engine=` exists for that override and
for nothing else. `steps=` on a conjugate block raises, because a Wiener solve
has no inner steps and accepting the argument would silently ignore it.

**The partition is strict, and the omission is the dangerous half.** Every
latent of the space in exactly one block. A latent in two blocks would have its
second update each sweep solving a conditional the first had just invalidated. A
latent in *none* sits frozen at its declared init for the whole run while the
sweep converges and every other number looks healthy — so both are refused by
name.

**Why the joint χ² and the rank test are two guards and not one.** A CG residual
`‖Mx − b‖` and a condition number `κ(AᵀN⁻¹A + S⁻¹)` are both computed *from the
block being solved*, so neither can see a degeneracy whose two halves live in
different blocks. `check_linearity` cannot see it either, and is right not to:
each conditional of a bilinear model genuinely is affine. On the package's own
`gain × T_ant` fixture with a free antenna temperature per (time, frequency)
cell, an alternating solve lands well over a thousand kelvin from the truth
while every per-block number reports green.

The two things that can see it work on different objects and at different
cadences, which is why neither subsumes the other:

* **`identifiability()`** is a rank test on the Jacobian of the prediction with
  respect to **all** the parameters at once. It is a property of the *model at a
  point*, runs before a sweep at both exits, and names the null directions as
  combinations of latents. The point estimate needs it more, not less: a chain
  at least has `r_hat` to scream with, while CG converges quietly onto an
  arbitrary point of the null space.
* **The joint χ² across sweeps** is a property of the *run*. It is the number
  that keeps falling while every block's own residual has already settled, and
  it is tested on the **decrease**, not on `|χ²[k] − χ²[k−1]|` — consecutive
  sweeps differ by the inner solver's own noise whatever the outer iteration is
  doing, the same trap `iterative_gls` documents for its `reweight_tol`.

`check_identifiability="once" | "each_sweep" | False` is the caller's explicit
choice with the cost documented at both ends, not a size heuristic: for a small
model the per-sweep check is cheap *and* strictly more informative, since a
nonlinear model's identifiability is a property of where you are.

**A plan does not nest `iterative_gls`.** σ is re-evaluated at the current joint
prediction before every block update, so for a `RadiometerNoise` the sweep *is*
the reweighting iteration; nesting would run one fixed point inside another at
the product of their costs. `PlanDiagnostics.noise_depends_on_prediction`
records when that applied.

**NUTS-within-Gibbs is stated, not hidden.** A conjugate block's GCR draw is an
exact conditional draw, so a plan of conjugate blocks is an exact Gibbs sampler.
A finite number of NUTS steps merely leaves the conditional invariant, which
makes the scheme Metropolis-within-Gibbs — still valid, and `Block(..., steps=)`
therefore looks like a performance knob while being a statistical assumption.
The kernel adapts through warmup and is frozen afterwards, since a kernel that
keeps adapting from the states it visits is no longer a valid transition.

### D27 — Ordering is declared by the operator, in the graph's own nouns

`calibration.py` stated its ordering constraint in a module docstring: the CW
tone must sit *before* the bandpass and the gain, because it tracks `g(t)` only
by passing through it. Nothing enforced it, and `At("noise", cw)` assembled
cleanly with the tone's gain response dropping to exactly 1.0 — a calibrator
that monitors nothing, in a model that runs, differentiates and looks healthy.

`must_precede: ClassVar[tuple[str, ...]]` on `AbstractOperator` is that
constraint moved into a place `assemble()` can check, with an optional
`must_precede_because` the refusal quotes back.

**The check is reachability, not a toposort index.** A toposort totally orders a
DAG, so it also orders nodes on branches that never meet, and "sorts earlier"
would be satisfied by a placement whose output never reaches the constrained
stage at all. An **absent** stage is not a violation — there is nothing to pass
through. A node id the template does not have **is** one, because an
unenforceable declaration is prose in a ClassVar.

**It is a third declaration alongside `requires`/`provides`, not a consumer of
them.** Those speak in `State` paths, and every operator on the receiver chain
reads `"data"` and writes `"data"` — so "before the gain" is not a sentence that
vocabulary can form. Keeping them separate is what lets `requires`/`provides`
stay a single-shape contract (see *Known deferred issues*).

**Both routes enforce it, and they are not equally able to.** The degraded test
this section once recorded as "possible and not implemented" is now
`check_stage_ordering`, called from `Pipeline.__init__`: *if a named stage is
present in this sequence, it must come after the declaring stage*. So
`Pipeline(sky, bandpass, gain, cw_tone)` no longer builds the placement
`assemble` refuses.

What the sequence route still cannot do follows from having `names` where the
graph has a node list, and each item is pinned in
`tests/core/test_ordering.py`:

* it cannot refuse an unenforceable declaration. `assemble` rejects a
  `must_precede` naming a node the template does not have; a Pipeline has no
  such list, so a typo and a legitimately absent stage are one observation.
* it cannot see into a nested composite — `names` is one level deep.
* it binds through NAMES, so an auto-derived name enforces a constraint only
  where it coincides with a node id. `GainOperator` auto-names to `gain` and
  binds; `ReceiverOperator` auto-names to `receiver`, not `bandpass`, and does
  not. Measured: `Pipeline(receiver, tone)` is accepted where
  `Pipeline(receiver, tone, names=("bandpass", "cw_tone"))` is refused.

The combinators still do not apply it, and that is not an omission:
`SumOperator` and `SelectOperator` run their branches in parallel on the same
input, so "precede" is not a relation between two of them.

`assemble()` therefore remains the stronger route, no longer the only one.

### D28 — A smooth basis is the identifiability repair, so it belongs to `core`

`identifiability()` refuses a free-per-cell model and tells the caller what to do
about it — "a smooth basis in place of one free parameter per cell is the usual
repair". A diagnostic that names a repair the package does not ship is half a
feature, so `rheplicant.core.basis` (`SeparableBasis`, `basis_matrix`) is that
basis and `rheplicant.radio.BasisTemperatureOperator` puts it on the reserved
`t_sys_extra` node, parameterized by **coefficients, not cells**.

**Why the basis and the operator are one change and not two.** Measured through
the operator on the assembled graph, with a known 5000 K CW tone against a gain
free per time sample (`n_time=7`, `n_freq=5`, at a generic coefficient point):

```
free-per-cell T_ant,  tone ON  (5000 K)   n_par=42 rank=35 nullity=7
free-per-cell T_ant,  tone OFF            n_par=42 rank=35 nullity=7
(3,2)-basis T_ant,    tone ON  (5000 K)   n_par=13 rank=13 nullity=0
(3,2)-basis T_ant,    tone OFF            n_par=13 rank=12 nullity=1
```

Against a free-per-cell antenna temperature the tone buys **exactly nothing** —
nullity is `n_time` either way, because the free cells absorb the whole of
`g[t] × (tone profile)` sample by sample. The basis therefore has to reach
`T_ant` itself; smoothing the noise waves alone would leave the tone useless.

**It is the frequency axis that does the work.** A basis complete in *frequency*
makes the tone worth nothing whatever the time axis does — the tone's profile is
then inside the span and is reabsorbed. A basis complete in *time* is still
rescued by it. So "frequency-smooth" is the condition and `n_j < n_freq` is what
it means; `n_basis == n` is legal rather than refused, because the design matrix
is perfectly well conditioned and whether completeness costs anything is a joint
property of the model that only `identifiability()` can answer.

**`legendre` and `polynomial` span the same functions and are not
interchangeable**: `cond(design)` at `n=32, n_basis=16` is 7.86 against 2.81e+05,
and that number lands on the κ of the block's normal operator.

There is a third kind, and it is a different KIND of choice. `BASIS_KINDS` is
the public tuple `("legendre", "polynomial", "fourier")`, and picking between
the first two is a conditioning decision over one span, while picking `fourier`
changes the span: its condition number at the same size is a flat 1.41, because
its harmonics are orthogonal on the grid it is evaluated on (`i/n`, endpoint
**excluded** — the endpoint-included grid puts the first and last sample one
full period apart, and the coefficients come back correlated with no shape
changing). Choosing it asserts that the quantity is *periodic* on this axis — a
sidereal cycle, a standing wave in a cable — and on a non-periodic axis its
constant-plus-harmonics span forces the two ends to agree, which is a statement
about the physics rather than a numerical detail.

**Where it lives.** `rheplicant.core.basis`, not `rheplicant.inference.basis`. It
reads like an inference utility, but the design matrices are held by an operator
on the signal path, so `radio` would have had to import `inference` — which
nothing in this package does and which the inference layer's own premise
forbids. `core` is the one layer both may depend on, and it fits: no `State`, no
radio physics, and `Coordinates` already names the `time` and `freq` axes there.

### D29 — A memory holds likelihood factors and exactly one prior

A campaign that archives each night's recording must keep something, and the
tempting something is "that night's posterior". It is the wrong object twice
over. Multiply N of them and the prior appears N times: the credible interval
narrows by `sqrt((F+N P)/(F+P))`, the mean pulls toward the prior mean, and
nothing in the output says so. And a posterior is a normalised density, so it
cannot represent the normal case — an epoch that constrains a two-dimensional
subspace of a twenty-dimensional θ.

So `BayesMemory` stores `CompressedLikelihood` terms, the word is in the type
name, and `log_posterior` applies the prior exactly once. Rank deficiency is
representable because the stored form is `[R | z]` with `log L = -½‖Rθ − z‖²`:
a term constraining `r` directions is an `R` with `r` rows, `F = RᵀR` is PSD by
construction, and the working condition number is `sqrt(kappa(F))`.
Accumulation is the QR of stacked factors, so order-invariance and
associativity are properties of the representation rather than tests.

Two constants are the whole difficulty, and both are invisible in the
posterior's *shape* — they shift the log-density without touching its gradient
or curvature, so every moment-based check passes while the evidence is wrong.
The QR's corner term `rho²` must be folded into the offset: dropped, each
combined term is wrong by a positive amount that grows with the campaign
(measured at 0.16 to 5.3 nats for a single pair). And marginalising a
per-epoch nuisance contributes **two** constants, not one — `-Σlog|R_pp,ii|`
off the discarded block *and* `-Σlog(std)`, the nuisance prior's own Gaussian
normalisation. Only the `(n_φ/2)log(2π)` halves cancel. Omitting the second
costs 1.07 nats for three nuisances at `std=0.7` and 27.5 for twenty-five at
`std=3`; it is exactly zero when `std=1`, which is how it survives a probe
built on unit priors.

Three refusals, each guarding a smooth wrong answer. **The same night twice** —
terms carry the recording's data hash, and `remember` refuses a repeat unless
`duplicate=True`, the posture D17 takes on beam-spill double counting. **Two
estimators** — a GLS term and a full-likelihood term (D21/D23) are different
estimators and their sum is neither. **A tempered term** — a factor carrying a
share of the prior would make `log_posterior` apply it twice, so the streaming
path refuses one.

The archive gets its own decision, because `eqx.tree_serialise_leaves` reverts
every `eqx.field(static=True)` member to the template's value with no error:
measured on equinox 0.13.8, `include_logdet=False` reads back `True`,
`noise_frozen_at="gls"` reads back `"none"`, `n_observed=777` reads back `0`.
All of a term's provenance lives in such fields and the raw data is gone, so
nothing could contradict a reloaded memory describing itself as a set of exact,
full-likelihood factors. The manifest is therefore a **reconstruction spec** —
the arrays come from the binary, every static field and dtype and the writer's
x64 state come from the JSON — and `load_memory` builds the template from it
and refuses on any mismatch.

The layer requires float64 and the suite does not run there. A stored offset is
the time–bandwidth product (~7.2e11 for one RHINO night) against a difference
of ~1e5, which float32 annihilates rather than rounds; but float32 is this
package's production dtype, and eighteen tests assert refusals only float32
forces. `jax_enable_x64` is process-global with no scoped form in jax 0.11, so
`tests/evidence/` is a second session, and `tests/test_evidence_session.py`
runs it as a subprocess from the default suite — asserting a positive passed
count and no skips, because a child that collected nothing exits 0 exactly like
a healthy one.

### D30 — The evidence boundary takes arrays, not a `State`, and that is what keeps the campaign loop compiling once

D29's memory implies a loop: one term per night, absorbed, recording archived.
It is the first place in this package where a function runs once per
observation, and therefore the first place D1's static/traced split can cost
something. `State.meta` is part of the jit cache key and `to_state` writes
`meta["time_epoch_unix_s"]`, a different float for every recording, so a jitted
per-epoch function taking a `State` recompiles every iteration and keeps every
compiled program for the life of the process. Measured on a toy function: three
states differing only in that one meta float produced three compilations; the
same three with the epoch traced in `aux` produced one. It is not a memory
nuisance — it is jit doing nothing.

The layer as built does not have that problem, and the reason is a design
property rather than luck: **no `State` crosses into it.** `compress_linear`
takes a design mapping, an observed array, a noise scale and shapes;
`BayesMemory.remember` takes a `CompressedLikelihood`. A memory that never sees
a `State` cannot inherit its cache key. That is recorded here because it is
load-bearing and nothing enforces it: a `compress_state(state, ...)` convenience
overload is exactly the sugar this API invites, and adding one would reintroduce
per-epoch recompilation silently, since the result would be numerically correct
every time.

The same failure is already defended from the other side. `SqrtInfo.null`
returns a square factor rather than a zero-row one specifically to keep "a fixed
treedef across a whole campaign, which is what stops `jit` retracing once per
epoch". That covers the accumulator's *shape*; `meta` is the other half, and the
two only pay for themselves together.

**What still falls to the caller.** `design=` is computed caller-side, per
night, and the natural way to compute it is a jitted pipeline over that night's
`State` — which puts the loop back, one level out, where this package cannot see
it. The remedy is at that boundary: normalise `meta` down to the keys that
genuinely select a program (`telescope`, `band`) before the jitted call, and let
per-run numbers travel traced in `aux`.

**Not by moving the epoch out of `meta` in `to_state`.** It is there so the
epoch survives in float64; `Coordinates` stores through `jnp.asarray` and would
round it. That axis has already been corrupted exactly this way once — see
`MAX_TIME_RESOLUTION_IN_SAMPLES` and `docs/ingestion.md` — and the contract that
`meta[key] + coords.time` recovers `obs.time_s` exactly is load-bearing
elsewhere. The compile cost is the cheaper of the two problems and is fixable at
the consumer; the precision loss is fixable nowhere.

### D31 — The basis is seeded with the score directions, and the metric is the likelihood's

Compressing an epoch whose model is *not* linear in its latents means expanding
the prediction in a dictionary and storing the coefficients. The obvious
dictionary is the leading SVD modes of a bank of prior draws, and it is the
wrong one for the one measurement RHINO exists to make. Singular values order
modes by prior-induced amplitude; at 60–85 MHz the foreground spread (~200 K)
sits three orders above the 21 cm trough (~0.2 K), so the science direction is
the last retained and the first dropped. Measured on the four-latent fixture in
`tests/evidence/rhino_bank.py` (128 channels, 400 draws), the residual fraction
of the `t21_depth` score direction against a plain SVD basis is **0.562 at
n_S = 3 and 1.7e-4 at n_S = 4**. A richer pre-planning bank measured 0.3147 at
3, 0.0289 at 5, 0.0040 at 8 and 0.0000 only at 13. Both numbers are real: four
near-linear latents span their own tangent space once four vectors are allowed,
so where the deletion stops is a fact about the bank and not about the method.
Seeding is what makes the answer independent of which bank a campaign happens to
have — with `dmu/dtheta_t21` placed first, `r_t21 = 1.5e-16` at n_S = 3. A
complete repair rather than an improvement, and it does not depend on choosing
n_S large enough. So `build_reduced_basis` seeds `dmu/dtheta_j` for every
**named** global latent before it looks at the bank, and `basis_fidelity`
reports `r_j` per latent in D14's named-row `FlatMatrix`, so a collapsed Fisher
eigenvalue arrives with a name attached — a scalar fidelity number names no
culprit.

The size of the test that can see this is itself constrained: the whitened
bank's **numerical rank is 6**, not the 13–38 the design quotes for raw
snapshots, and `build_reduced_basis` refuses any n_S above it. A test written at
n_S = 8 does not assert a weak claim, it raises.

**The metric is a second, separate decision, and the design ran two claims
together.** Write `Pi = S_w^T (S_w S_w^T)^-1 S_w`. It is the `N^-1`-orthogonal
projector onto the retained span *whatever metric chose the rows*, so T1's score
at the truth is `((I - Pi) mu)^T N^-1 (Pi dmu/dtheta) = 0` exactly. Selecting or
truncating in the wrong metric therefore costs **sensitivity** — a smaller
Fisher — and not bias. Projecting in the wrong metric costs **bias**, because an
unweighted least-squares projector is not self-adjoint in the likelihood's own
inner product. Measured as the campaign length at which the induced bias reaches
one sigma: at the operating point (seeded, n_S = 6) an unweighted projector
needs 5.1e21 epochs and the `N^-1` projector 5.9e22 — both roundoff, and for the
reason that matters. **The wrong projector needs a truncation residual to act
on**, and a seeded basis leaves it none. Build one that genuinely deletes the
target — n_S = 3, plain SVD, unseeded, `r_t21 = 0.562` — and the split appears
cleanly: **909 epochs** for the unweighted projector against 2.9e22 for `N^-1`.
The bias is a property of the projector, conditional on the truncation. The band
this is measured in has a sigma spread of `(85/60)^2.5 = 2.38896`, not the ~6x
the design quoted, which would need a spectral index near −5.1 over the same
band; under a flat sigma the unweighted projector *is* the `N^-1` projector and
the whole measurement goes quiet, which one test asserts so that a reader can
see the file is capable of failing.

**The dictionary is the campaign's and the metric is the epoch's.** Rows are
shared and orthonormalised once, in a declared reference metric; each epoch
stores its own `G_e` and `p_e` against them. Evaluation is then `O(n_S^2)` per
epoch rather than `O(n_S n_data)`, which is the whole point of the tier. What
that buys is not assumed away: §7's bias budget stores `grad_theta delta_e` at
compression — one JVP, `n_theta` floats — and `audit()` reports
`|bias_hat| / sigma_N` per named direction, refusing above a declared ratio. A
gradient and not a magnitude, because a constant offset has exactly zero effect
on a posterior while an arbitrarily small theta-dependent tilt has unbounded
effect. Because one basis and one instrument model serve every epoch the tilt is
**coherent**, so the bias is N-independent while `sigma_N` falls as `N^-1/2` and
the ratio grows as `sqrt(N)`: measured 1.571e-11 over four epochs and 6.286e-11
over sixty-four, a factor 4.001 against the 4.000 the law predicts. A ratio that
is comfortable at N = 10 need not be at N = 1000, and that is the point of
storing it rather than checking it once.

**Freezing `N` is measured per epoch, not argued.** Under `RadiometerNoise`
sigma tracks the prediction (D21), so the stored statistics are not constants
until the covariance is frozen — and D23 records that the converged answer is
then generalized least squares. The cost is recorded on the term over
`2 n_theta + 1` probes spanning the declared support: on this fixture
**4.2188e7 nats** over the full box, dominated by `amplitude` at ±3 prior sigma,
against **0.7568** on the `t21_depth` axis alone. The remedy the refusal names
is narrowing the support or re-anchoring the basis, never a larger `n_S` — that
is §7's remedy for a different error, which is why the two numbers are stored
separately rather than summed into one.

**The stored term is a `SqrtInfo` over the coefficient vector**, so every
property Plan A proved of that form transfers unchanged: PSD by construction,
rank deficiency representable as a short `R`, accumulation by QR, the corner
term kept. A memory accumulates T1 in a second fixed-treedef accumulator rather
than in a pytree whose child count grows with N. The two constants it carries
are why this tier is tested against absolute log-densities and never against a
posterior's shape: on the RHINO epoch the masked normalisation is **+200.738**
nats and the QR corner **−51.321**, summing to an offset of **+149.418**, and
Plan A shipped both of those errors once — each is a pure offset, so gradients
and curvature stay perfect while the evidence is wrong.

**T0 exists so the rest can be validated.** `RawLikelihood` keeps the raw data
and a live forward model, which defeats every bottleneck this layer was built
for; it refuses to be remembered or archived, by name. Without it §12.12's
boundary validation cannot be written at all, because "the tiers agree" needs
something for them to agree *with*. That validation calls T0, T1 and T2
**directly** and never through `compress`: a dispatcher routes one input to one
method, so sweeping a threshold through it shows only the routed function's own
continuity, which is smooth for anything analytic. The claim being checked is
that two methods return the same number at the same point, and a dispatcher
cannot be asked that question.

### D32 — A drifting nuisance is a chain, not N independent draws, and the constants are the whole risk

A quantity whose natural interval straddles an epoch boundary — a receiver
thermal transient over midnight, night *e*'s gain partly derived from night
*e−1*'s calibrator pass — declared `per_epoch` is one physical fluctuation
marginalised N times against independent priors. That injects information which
is not there, and it is silent: the posterior comes back centred and narrower.
Condition C1b names it; `scope="linked"` is the declaration, and this decision is
what the declaration buys.

**The recursion.** Carry a joint square-root factor over `(theta, zeta_e)`, fold
an epoch in by stacking rows and re-triangularising, advance by appending the
transition's rows and marginalising `zeta_e` — which *is* the Schur complement,
exactly. `theta` is never marginalised, so what comes back is
`log p(d_1:N | theta)`. Measured against a dense oracle that integrates `zeta`
analytically and shares no code with the recursion: agreement of **1.1e-13,
5.7e-14, 9.1e-13, 9.1e-13 nats** across four theta probes, against a density of
about −2506 nats.

**The constants are the whole risk, and there are six.** The recursion's shape,
gradient and curvature are all correct with any of them missing, which is this
codebase's recurring defect in its purest form. Measured cost of dropping one, on
the six-epoch fixture in `tests/evidence/chain_bank.py` at `theta = (0.4, −1.1)`:
the initial `zeta` prior normalisation **+0.9189**; the five per-transition
`−½ logdet(2πQ)` **+2.8618**; the five in-chain marginalisation constants
**+6.2764**; the final marginalisation's own **+0.9855** (the two sites together
are the **+7.2619** the module's table quotes); the six fold corners
**+45.9502**; the masked data normalisation **−6.8408**. Four of the six are
somebody else's arithmetic reaching one layer further than its owner expected —
the corner is D29's and the data normalisation is `compress`'s — and a reader
will assume the remaining two are handled elsewhere too. They are not: the
initial `zeta` prior normalisation and the *final* marginalisation exist only
here, and the final one is nearly a nat on its own. And
the design's shorthand is wrong by half a term: §6 names the augmentation's
`½ logdet Q⁻¹`, but a transition is a *density*, so it carries `−½ logdet(2πQ)`;
the `2π` cancels against the marginalisation's `+½ n log 2π`, and keeping only
the log-determinant leaves `+0.9189` per epoch — **+4.5947 over this fixture's
five transitions, +918 nats over a thousand-epoch campaign** — with no effect on
any posterior mean, width or gradient. One test per constant, each carrying its
measured cost, so that "the constants matter" cannot pass vacuously. One of them
is measured at exactly zero and pinned as such: the *marginalisation's* corner is
a length-zero slice, because that QR is square, so deleting it moves the answer
by 0.0 nats bit for bit. The corner worth +45.95 is the **fold's**. One phrase,
two sites, and a plan that wrote it once would have shipped half a test.

**Two sub-scopes, distinguished by the transition's type.** An OU with an
inferred correlation time is still linear-Gaussian, so a caveat phrased that way
is satisfied while its claim fails — `Q(theta)` and the Schur complement become
functions of theta and a filter run at compression time pins them, Trap 3 one
scope along. A `LinearGaussianTransition` holds numbers; a `HyperTransition`
holds a builder and is resolved *inside* the theta likelihood. One `lax.scan`
serves both, differentiable in the transition's own parameters — the analytic
gradient in `tau` and a central difference agree to **2.0e-9 relative**, which is
the step's own truncation error — which required extracting `marginalise_arrays`
from `marginalise`: the checked path concretises twice and raises
`ConcretizationTypeError` under `grad` as well as under `jit`. The refusal it
carries is not weakened but **moved**, to a strictly-positive `process_std`
checked once at declaration, which is what makes every `zeta_e` block
constrained by construction rather than by inspection at each of a thousand
steps. A chain is also *ordered* where a bag is not: `BayesMemory` sums terms in
any order and refuses one carrying a linked latent's columns, while `ChainMemory`
requires one and keeps the epochs in the sequence they arrived, because the
transition connects epoch *e* to epoch *e+1* and nothing else.

**§9's diagnostics, and the one the design promoted that cannot see the case it
was promoted for.** `sigma_N = (Σ F_e)^{-1/2}` does not read the data, so
`sigma ∝ N^{-1/2}` holds by construction — measured, the fitted power is
`−0.49991034` on a clean campaign and `−0.49991034` on one biased by 52.568
sigma, the same number, from per-N sigma arrays equal element for element. A
deterministic common mode contributes no variance at all. Measured on a repeated
nightly design, which is the realistic case: the **in-span** half of such an
error biases theta identically every night, leaves no residual, and is invisible
to per-epoch chi-square scatter, to split-half, to leave-one-out **and to the
held-out posterior-predictive z the design promotes to primary** — the clean and
the biased campaign return the same held-out scores, the largest disagreement
over all 640 epochs being **4.93e-05**, which itself falls as 1/N. The
**out-of-span** half does show, in the per-epoch chi-square (`z = +31.9231`
against `+0.4457` clean, at N = 640) and in a named systematic template's
projection (`z = +52.5498` against `+1.7163`, at unchanged scatter 1.00200),
which is why §9.3's per-epoch summary is computed at compression, stored in
~100 bytes, and read as a *mean* at √N. The chi-square's own scatter is **not**
preserved — measured 5.5467 against `sqrt(2 dof) = 3.4641`, which is
noncentrality and not injected randomness, `sqrt(2(dof + 2 lambda)) = 5.3057` at
the measured `lambda = 4.0376` — so the scatter that is quoted as unchanged is
the template projection's, not the chi-square's. Because the in-span half is
undetectable in principle, §9.4 and §9.5 are **refusals over a declaration**
rather than reports
over the data: a campaign refuses to quote an error bar below its declared
systematic floor, and a memory refuses to sum two epochs that share an
input-product hash unless the product is represented among the global latents.

### D33 — The post-`gain` trunk is in `adc_count`, and only `adc.scale` carries the conversion

Five operator fields carry no unit anywhere in this package — `adc.scale`,
`gain.gain`, `apply_cal.gain`, `flagging.threshold`, `filters[].regularization` —
while `RhinoObservation.waterfall` is documented as "raw power, arbitrary scale"
and every operator upstream of it produces kelvin. Somewhere between the two the
signal stops being a temperature, and nothing says where.

**It is declared to happen at `adc`.** The trunk downstream of `gain` is
`adc_count`, and `adc.scale` carries the compound unit `adc_count/K`. That makes
`adc_count/K × K = adc_count` a *dimensional identity a loader can check*, which
is what promotes the unit rule from a spelling convention into a real one. It is
also the only arrangement under which the tone-protection threshold and the
flagging threshold are two numbers in the same unit, so they can be compared at
all.

**`gain` sits before `adc` on the trunk** (`bandpass → gain → noise → emi →
adc`), so `gain.gain` is genuinely dimensionless and only `adc.scale` converts.
An earlier draft of the config schema pointed the other way.

**Rejected.** Calling the trunk dimensionless and letting the gain absorb the
absolute level throws the check away entirely. A top-level `observation.data_unit`
adds a second place to state one fact, and two places to state one fact is how
they come to disagree.

**Revisit if** either RHINO HDF5 producer starts recording a physical unit for
the waterfall — the conversion then stops being a declaration and becomes a
measurement, and should be read rather than declared — or the absolute level
turns out to be degenerate with another fitted quantity, in which case
"dimensionless" stops throwing away a check and starts describing the truth.

### D34 — The LST bridge is a units-only adapter in this package, not a call from the layer above

`coords.extra["lst_deg"]` is required by both real sky engines and is produced by
nothing in `src/` except `DriftScanProjector.uniform_lst_grid`, which only makes a
synthetic uniform turn. `limTOD.simulator.generate_LSTs_deg(lat, lon, height_m,
time_list, start_time_utc)` exists upstream and this package never calls it. The
consequence is that `site.lon_deg` and `site.alt_m` are recorded and never
consumed — two keys a user fills in that change nothing, which is worse than not
offering them.

**A thin `rheplicant.radio.site` adapter closes it**, matching the seam
`radio/beams.py` already established for CST ingestion: units in, units out, no
physics of its own. That makes `observation.site` a section this package can
validate rather than one it passes through.

**Rejected.** Letting the config layer call limTOD directly leaves `rheplicant`
unable to check the one quantity both engines depend on. Requiring an `lst` array
always and deleting `lon`/`alt` is honest about today, but gives up precisely what
a user with a real observing log needs.

**Revisit if** limTOD changes `generate_LSTs_deg`'s signature or semantics — the
adapter is a pass-through and inherits any drift — or if users turn out to arrive
with LST arrays already computed by their own pipeline, in which case the declared
array is the honest primary and the adapter becomes the convenience.

### D35 — `phi0_deg` and `phi_sense` are facts about the horn, so they travel with the beam

`radio/beams.py` calls them "a fact about the as-built horn, not the file". A fact
about the horn cannot live in a run config without being restated for every run,
and cannot live in a preset without the preset guessing on the user's behalf — in
exactly the place this package otherwise refuses to guess.

**They belong in a per-beam sidecar, hashed alongside the export** (the pyuvdata
CST-settings precedent). Until that format ships they stay **required in the
user's own file for `format: cst`, and no preset may supply them** — and they are
refused for every other beam format, because they describe how a CST export's
azimuth maps onto the beam-local chart and are meaningless for a raw HEALPix array
or a synthesised map. For those, `frame:` is the genuinely unverifiable fact and is
required instead.

**Rejected, and the reason is measurement-shaped.** Letting a preset supply them
marked `provisional: true` is the worst of the options available: **a mirrored beam
passes every integral, every peak and every azimuthally-symmetric diagnostic
unchanged.** There is no numerical symptom at all, so the only protection is that
the value was stated by someone who knew it — and `provisional: true` is a warning
a user clicks past.

**Revisit if** CST (or whichever exporter is in use) gains a standard metadata
block carrying orientation and sense, making the sidecar redundant; or if
pyuvdata's CST-settings convention is adopted wholesale, in which case use theirs
rather than inventing a second one.

### D36 — `check_linearity` runs by default, and the only escape is a written reason

`linear_operator(check=True)`'s own comment is the argument: "leave it on; turning
it off buys a class of silent, confident errors". A declared-linear latent that is
not linear gives a fit that converges, a residual that looks right, and an answer
that is wrong — the same shape as every other failure this file records.

**So it runs at config validation, always**, and `checks.linearity.mode: skip`
plus a **required written reason** is the only way past it. The reason lands in the
run log, so a skipped check is a sentence someone wrote rather than a flag someone
set. The cost is three forward evaluations per declared-linear latent — negligible
for a noise-wave block, non-negligible for a 10⁶-coefficient sky block.

**Rejected.** Defaulting it on but auto-skipping blocks above a declared size
reintroduces exactly the size heuristic this package refuses to make anywhere else
— notably for `identifiability`, where the absence of a size threshold is
deliberate and documented. A user with a 10⁶ block can write one sentence; a
threshold cannot know which 10⁶ block was safe.

**Revisit if** a real block is found where the three forward evaluations are
prohibitive *and* the skip-with-a-reason escape proves too coarse to express the
situation. That would be evidence for a cheaper check, not for a size threshold.

### The pattern in D33–D36

Three of these four chose the option that **keeps a check possible** over the one
that is cheaper or has fewer keys. That is the tie-breaker for the config layer's
remaining questions:

> Where two options differ in whether a wrong configuration can be *detected*,
> prefer the detectable one — because every failure mode named in this package's
> own docstrings is one that returns a finite, correctly-shaped, plausible, wrong
> number.

D35 is the sharpest instance: it rejects the convenient option specifically
because the failure it guards against has no numerical symptom at all.

### D37 — Audit facts and scientific products have separate schemas

Plan 4A owns document entry, validation, execution status, and the mandatory
input/resolved/provenance/diagnostics audit tree. Plan 4B adds scientific
products without widening either fixed audit schema: its files are described
by `products.json` and `products-v1.schema.json`. Keeping the tables separate
means adding a scientific selector cannot silently change what an old audit
consumer thinks was materialized.

### D38 — The console bootstrap is physically JAX-free until runtime is chosen

`_rheplicant_bootstrap` reads exact bytes, layers presets, preflights output,
and records audit facts without importing `rheplicant`, JAX, or jaxlib. The
sole first main-package import is the callback passed to `establish_runtime`
after process-global x64/platform environment state is installed. A convenient
top-level import would make the document's runtime declaration arrive too late
to be true.

### D39 — YAML and presets are exact-byte inputs, not reconstructed mappings

The source is consumed once with bounded safe YAML loading and duplicate-key
refusal. A selected package preset keeps its original bytes, digest, resource,
and expanded-node count; generated programs embed those snapshots rather than
adopting whatever preset a later installation happens to provide. This makes
the recorded input the input that actually ran.

### D40 — Named plugin and Python targets are trusted, but never called pure

Executable extensions are an explicit trust boundary. Validation guarantees
that RHEPLICANT/bootstrap/output code performs no output mutation, while one
fixed warning and `unobserved_io: true` state that named code may do private
I/O the trace cannot observe. A sandbox claim would be false; silently
omitting the warning would be worse.

### D41 — Every exit has one parser before any exit executes

Raw run options become immutable parsed execution/resolved views exactly once.
All selected layers and all declared schedules parse before execution begins,
so an error in the last variant cannot arrive after earlier runs already spent
time or produced state. Compatibility registries remain, but built-in
executors do not reparse their raw mappings.

### D42 — A9 dimensional compatibility is a registry-wide delivery invariant

Every production value delivery names its destination and consults the one
dimension registry. Units are checked at the model/resource field that will
consume the value, not guessed from the spelling at its source. A missing
registry row is itself a refusal: accepting one new field without a dimension
would reopen the exact route A9 exists to close.

### D43 — Declarative files are captured once before readers consume them

A regular input is streamed into a private mode-0600 snapshot while hashing;
directories use a verified manifest/tree snapshot. Readers consume only that
capture. Hashing an original and reopening it would let provenance describe
one inode while computation reads another, even when both reads individually
succeed.

### D44 — Resolved YAML is deterministic evidence, not a debug dump

Each completed layer records its effective document, parallel origin tree,
deletions, defaults, parsed schedule, and decisions. A dedicated serializer
uses the closed name codec and canonical scalar rules; it never calls
`yaml.dump` on live scientific objects. A later variant refusal therefore
still preserves the complete base document it actually earned.

### D45 — Partial audit envelopes say only what completed

The append-only trace marks a boundary only after it succeeds. Refused/error
envelopes retain earlier findings, parsed rows, run outcomes, and resolved
layers while leaving future artefacts explicitly unwritten with a closed
reason. Serialization consumes a detached candidate and contributes exactly
one final boundary only after final staged metadata is durable.

### D46 — Output publication is a descriptor-safe recoverable transaction

The output manager walks without following symlinks, proves ancestor rename
protection and access/default ACLs, checks atomic no-replace capability without
writing, and budgets the real leased filesystem `NAME_MAX` before mutation.
Recovery runs before A34 clobber authorization. Journalled staging, metadata
replacement, parent fsyncs, and no-replace publication make interruption
recoverable; ambiguity preserves every path and permits no second transaction.

### D47 — Per-target locks persist because lock identity is the exclusion

The mode-0600 sibling lock is never unlinked on close. Unlinking would allow a
second process to create and lock a new inode while the first still holds the
old one, giving both writers apparent exclusivity. One adapter and one owning
lease cover recovery, A34, staging, publication, and terminal recovery; an
ordinary terminal I/O failure is recovered before the one permitted error
sibling attempt.

### D48 — A product request names a semantic product, not an object key

The 22 `outputs.write` selectors compile to immutable requests and a typed
`(run kind, selector)` extractor registry. Extractors read known result types;
they never guess from whatever dictionary keys happen to be present. An
explicitly named incompatible run is refused. An unfiltered request records
truthful omissions and succeeds only if at least one compatible run emits a
file. This keeps broad requests useful without turning a spelling error into
an empty success.

### D49 — Scientific bytes are deterministic, finite, and self-describing

Numeric products use deterministic, pickle-free NPZ; records use canonical
finite JSON; text and signal-path renderings use UTF-8. Optional NetCDF chains
exist only when the writer imports, with no format fallback. Every emitted file
has a portable encoded path, byte count, SHA-256 digest, selector, format, and
metadata row in canonical `products.json`, which is checked against the strict
packaged schema before publication.

### D50 — Product extraction never reruns the science

`aux` and `taps` read the `State.aux` already returned by a forward run;
reports read prior execution results and recorded timings; recovery combines
recorded truth with the estimator or posterior moments already present.
Assembly and signal-path products use prepared layer assemblies. Missing
truth, zero uncertainty, or an unavailable statistic becomes an explicit
omission or refusal, never a second execution or non-finite JSON.

### D51 — Comparison failures and benchmark measurements remain data

`compare` requires two earlier successful, structurally identical numeric
products and checks mapping keys, shapes, and dtype classes before reducing
`max_rel_diff`, `rms`, or `max_abs`. Missing tolerance is not an exception:
the serializable result says `passed: false`. `benchmark` runs named prepared
variants, blocks JAX results before stopping the clock, excludes warmups, and
retains raw samples; its memory metric is explicitly Python-traced peak bytes,
not device memory.

### D52 — Scientific files share the one recoverable publication transaction

Product and report bytes are built only after successful execution and before
the first staging write. They are inserted between resolved configuration and
the two final audit metadata files in the existing `AuditBundle.files`; the
same descriptor-relative staging, modes, fsyncs, journal, clobber proof, and
recovery state machine publishes all of them. A materialization refusal emits
only the ordinary failure audit sibling and never a partial success target.
There is no second product transaction and no direct destination write.

### D53 — The GUI projects exact YAML and is a local trusted execution surface

The browser owns selection, hover, focus, open tabs and an invalid in-progress
textarea draft. It owns no scientific configuration. Every accepted form,
graph or output edit crosses the framework-free document engine and returns a
new YAML string; validation, previews and jobs are content-bound projections of
those bytes. This keeps the CLI, generated program and editor on one scientific
state rather than asking three serializers to agree.

The selected FastAPI + React application ships as one same-origin process with
immutable in-memory sessions and explicit load/save/job boundaries. Its wheel
contains the production assets, so running an installed editor requires no
Node toolchain and cannot silently serve a developer checkout. Loopback is the
default and a non-loopback bind requires an explicit acknowledgement because
the process has no authentication, tenant isolation or sandbox. Bounded safe
YAML parsing protects the parser; it does not make `python:` targets, plugins,
resource paths, output paths or expensive jobs untrusted. A configuration is
trusted input to the server account, exactly as it is at the CLI boundary.

### D54 — Workbench drafts are view state; accepted edits return exact YAML

Workbench navigation, selection and raw drafts are browser view state.
Exact accepted YAML remains the sole scientific state; every accepted control
edit returns complete YAML through a revision-checked Python transformation.

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
