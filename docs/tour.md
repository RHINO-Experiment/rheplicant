# The guided tour

Everything RHEPLICANT can do, in order, with runnable snippets. Each block builds
on the previous ones; pasted top to bottom they form a working script.

- [1. State — the scientific context](#1-state)
- [2. Operators — the one contract](#2-operators)
- [3. Composition — chain, sum, switch](#3-composition)
- [4. Graph assembly — the signal path knows](#4-graph-assembly)
- [5. Seeing the signal path](#5-rendering)
- [6. The modular sky engine](#6-sky-engine)
- [7. Analysis: snapshots, calibration, filters](#7-analysis)
- [8. Inference — one seam, many engines](#8-inference)
- [9. Writing your own operator](#9-extending)
- [10. Conventions cheat sheet](#10-conventions)

## 1. State

A `State` is the complete scientific context flowing through a twin — an
immutable JAX pytree, so the whole thing jits, vmaps, and differentiates.

```python
import jax, jax.numpy as jnp, equinox as eqx
from rheplicant import State, Coordinates, Environment

state = State(
    coords=Coordinates(time=jnp.linspace(0.0, 60.0, 8),      # seconds
                       freq=jnp.linspace(60e6, 85e6, 4)),    # Hz
    env=Environment(temperature=jnp.array(280.0)),           # rides along, traced
    key=jax.random.key(0),                                   # randomness is data
    meta={"telescope": "my-antenna", "obs_id": "tour-001"},  # static, hashable
)
```

Two channels, one rule: **strings and labels go in `meta`** (static — part of
the jit cache key, changing them recompiles); **numbers and arrays go in
`coords` / `env` / `aux`** (traced — differentiable, vmappable). Fields that
never enter the forward model (`env`, `aux`) still ride along for
diagnostics and reproducibility — and can be promoted into the model later
without restructuring anything.

States never mutate; updates are functional and re-validated:

```python
s2 = state.replace(meta={"telescope": "other"})   # new object, original untouched
s3 = state.with_data(jnp.zeros((8, 4)))           # shorthand for the common case
subkey, s4 = state.next_key()                     # the PRNG protocol: split, advance
raw_kept = s3.checkpoint("raw")                   # zero-copy snapshot into aux
```

## 2. Operators

Every transformation satisfies one contract — a pure `State -> State`
callable, implemented as an `equinox.Module`. Array-valued fields are
automatically differentiable parameters; there is no registration machinery.

```python
from rheplicant import LambdaOperator
from rheplicant.radio import GainOperator

gain = GainOperator(gain=jnp.array(1.1))          # gain is a differentiable leaf
clip = LambdaOperator.on_data(lambda d: jnp.clip(d, 0.0, jnp.inf))

out = gain(state.with_data(jnp.ones((8, 4))))
assert jnp.allclose(out.data, 1.1)
```

Operators declare what they touch (`requires` / `provides` — documentation
today, a validation hook tomorrow) and follow three rules: never mutate the
input state, draw randomness only via `state.next_key()` (returning the
advanced state), and validate structure only (shapes/dtypes — value checks
would break under jit).

## 3. Composition

Three combinators mirror how physics composes. All three are themselves
operators, so they nest arbitrarily.

**`Pipeline` — sequential effects:**

```python
from rheplicant import Pipeline
from rheplicant.radio import SkyOperator, NoiseOperator

pipe = Pipeline(
    SkyOperator(amplitude=jnp.array(100.0)),
    GainOperator(gain=jnp.array(1.1)),
    names=("sky", "gain"),
)
assert jnp.allclose(pipe(state).data, 110.0)
final, stages = pipe.run_with_intermediates(state)   # per-stage diagnostics
pipe2 = pipe.replace_stage("gain", GainOperator(gain=jnp.array(2.0)))
```

**`SumOperator` — independent additive contributions.** Branches are
*sources*: each receives the input context with `data` stripped and its own
PRNG subkey; outputs sum leafwise (structure/shape mismatches fail loudly):

```python
from rheplicant import SumOperator
from rheplicant.radio import GlobalSignalOperator, ForegroundOperator

sky = SumOperator(
    GlobalSignalOperator(depth=jnp.array(0.2), centre=jnp.array(72e6),
                         width=jnp.array(5e6)),
    ForegroundOperator(amplitude=jnp.array(1e3), spectral_index=jnp.array(2.5),
                       ref_freq=70e6),
    names=("signal", "foregrounds"),
)
```

**`SelectOperator` — switched paths.** One branch per time sample, chosen by
an integer cycle in `coords.extra` (observation configuration, not a model
parameter) — how switched calibration loads replace the antenna signal.

## 4. Graph assembly

Explicit composition is always available — but composition is *implicit in
the signal path*. The canonical single-antenna graph
(`rheplicant.radio.RADIO_GRAPH`, 29 nodes) knows how every element connects, so
you provide a **set** of operators and `assemble` compiles the sub-path they
induce:

```python
from rheplicant.radio import assemble, IonosphereOperator, BeamOperator

part = assemble(
    SkyOperator(amplitude=jnp.array(1e3)),
    IonosphereOperator(delta=jnp.array(0.01), ref_freq=70e6),
    BeamOperator(solid_angle=jnp.array(0.8)),
)
# ≡ Pipeline(sky, ionosphere, beam): the beam-convolved sky, nothing else
print(part)          # lit nodes + nodes traversed as identity
```

The rules: absent sources are pruned; absent transforms pass through as
identity; a junction with two or more live branches becomes a `SumOperator`
(branch order fixed by the graph, never by your argument order — same set,
same tree, same PRNG stream, same jit cache entry); a selector becomes a
`SelectOperator`. The result is an `Assembly` — an ordinary operator with
node-id ergonomics:

```python
twin = assemble(
    SkyOperator(amplitude=jnp.array(100.0)),
    GainOperator(gain=jnp.array(1.1)),
    NoiseOperator(sigma=jnp.array(0.5)),
)
twin["gain"]                                        # node-id access, any nesting
twin2 = twin.replace_node("gain", GainOperator(gain=jnp.array(1.0)))
observation = eqx.filter_jit(twin)(state)
```

Guard rails you will meet: a sourced assembly rejects caller data (it would
be silently discarded); a transform feeding a sum with no live source
upstream is an assembly-time error; junctions are never operator slots.
Escape hatches: `At(node_id, op)` places anything anywhere (that is how a
`NeuralOperator` takes over the bandpass node), `At((n1, n2, ...), op)` lets
one operator cover a contiguous region atomically, and *equivalent-entry
leaves* let the same physics enter in different forms — ground spill as a
pre-beam field (`ground_field`) or a post-beam effective temperature
(`ground_pickup`, generic `t_sys_extra`).

Switched calibration is one more provided operator:

```python
from rheplicant.radio import CalLoadOperator

cal_state = state.replace(coords=state.coords.replace(
    extra={"receiver_input": jnp.array([0, 1, 0, 0, 1, 0, 0, 1])}))
switched = assemble(
    SkyOperator(amplitude=jnp.array(100.0)),
    CalLoadOperator(t_load=jnp.array(300.0)),
)
# sample t: antenna signal where cycle==0, load where cycle==1
```

## 5. Rendering

Every assembly can draw itself on the full graph — lit nodes are what you
provided, half-lit nodes are traversed as identity, everything else is dim:

```python
mermaid_src = twin.to_mermaid()                     # for notebooks / docs
html_page = twin.to_html()                          # standalone page
svg = twin.to_svg()                                 # embeddable figure
# pathlib.Path("signal_path.html").write_text(html_page)
```

Rendered examples live on [the signal-path page](signal-path.md#lit-and-dim-what-an-assembly-simulates).

## 6. Sky engine

The sky term factorizes: **what the sky is** (`AbstractSkyModel`:
parameters → maps) × **how it is seen** (`AbstractSkyProjector`:
maps → time-ordered data, with an exact `adjoint` for linear engines).
Either half swaps independently.

```python
from rheplicant.radio import SkySourceOperator, PowerLawSkyModel, MatrixProjector

projector = MatrixProjector(                        # e.g. from limTOD, offline
    matrix=jax.random.normal(jax.random.key(1), (8, 6)))
source = SkySourceOperator(
    sky_model=PowerLawSkyModel(amplitude=jnp.ones(6),
                               spectral_index=jnp.array(2.5),
                               ref_freq=70e6, n_pix=6),
    projector=projector,
)
tod = source(state).data                            # (n_time, n_freq)
```

Two engines do real work. `NativeLimTODProjector` is the general one — pure
JAX, arbitrary pointing, differentiable in sky *and* beam. For a drift scan
(fixed pointing, only LST advancing) `DriftScanProjector` returns the same
numbers to float64 roundoff while rotating the beam once for the whole scan
instead of once per sample:

```python
from rheplicant.radio.sky import DriftScanProjector

projector = DriftScanProjector.from_beam_maps(
    beam_maps, lat_deg=53.2, az_deg=0.0, el_deg=90.0, lmax=191)
```

Both need `pip install -e '<limTOD>[jax]'`. Three supporting engines round
out the set: `MatrixProjector` (precomputed projection matrix — pure einsum),
`LimTODProjector` (numpy-limTOD oracle via `pure_callback` — validation only,
not differentiable), and `MModeProjector` (a placeholder kept as the minimal
m-mode contract; `DriftScanProjector` supersedes it). See
[sky engines](sky-engines.md) for the comparison, benchmarks, and figures.

`SkySourceOperator` enters the graph at `observed_astro_sky` (post-beam: its
output is already convolved).

## 7. Analysis

Data processing is the same formalism. Preserve raw data, apply a
calibration solution, project onto physically meaningful subspaces — every
filter is a linear projection with `mode="extract"` (keep it) or
`mode="remove"` (subtract it):

```python
from rheplicant import SnapshotOperator
from rheplicant.radio import ApplyCalibrationOperator, SiderealFilter, SkySpaceFilter

analysis = Pipeline(
    SnapshotOperator(name="raw"),                   # aux["snapshot/raw"]
    ApplyCalibrationOperator(gain=jnp.array(1.1)),
    SiderealFilter(n_days=2, mode="extract"),       # the day-repeating structure
    names=("snap", "cal", "sidereal"),
)
sky_locked = analysis(observation)
```

`SkySpaceFilter` map-makes through **the same projector** the forward model
uses (matrix-free conjugate gradients on the projector's adjoint) and
re-projects — Wiener-style sky separation, differentiable end to end, with
`aux["flags"]` automatically down-weighting flagged samples.
`FourierBandFilter(axis=0, ...)` is a fringe-rate filter; `axis=1` a delay
filter. RFI flagging: `FlaggingOperator` (threshold placeholder) or
`MomentRFIFlaggingOperator` (the real flagger, host-callback to MomentRFI;
existing flags compose via `prior_mask`).

## 8. Inference

Everything connects through one seam. `build_forward_fn` partitions a twin
into (trainable parameters, frozen structure) and closes over the input
state:

```python
from rheplicant.inference import build_forward_fn

model = twin.replace_node("gain", GainOperator(gain=jnp.array(1.0)))
spec = jax.tree.map(lambda _: False, model)         # freeze everything...
spec = eqx.tree_at(lambda p: p["gain"].gain, spec, replace=True)  # ...except this
forward, params0 = build_forward_fn(model, state, filter_spec=spec)
```

**Point estimates** — fixed-step gradient descent for well-conditioned
parameters, Adam for the rest (neural surrogates especially):

```python
from rheplicant.inference import GradientCalibrator, AdamCalibrator

params_fit, losses = GradientCalibrator(learning_rate=2e-7, n_steps=200).fit(
    forward, params0, observation.data)
```

**Bayesian posteriors** — priors attach to pipeline leaves positionally;
sample sites get semantic names (`"gain.gain"`); noise lives in the
likelihood (hand the bridge a twin *without* stochastic operators):

```python
import numpyro, numpyro.distributions as dist
from rheplicant.inference import prior_template, set_prior, to_numpyro_model, predict_from_samples

bayes_twin = assemble(SkyOperator(amplitude=jnp.array(100.0)),
                      GainOperator(gain=jnp.array(1.0)))
priors = set_prior(prior_template(bayes_twin),
                   lambda p: p["gain"].gain, dist.Normal(1.0, 0.3))
numpyro_model = to_numpyro_model(bayes_twin, state, priors, noise_std=0.5)

mcmc = numpyro.infer.MCMC(numpyro.infer.NUTS(numpyro_model),
                          num_warmup=100, num_samples=100, progress_bar=False)
mcmc.run(jax.random.key(0), observed=observation.data)
posterior_predictive = predict_from_samples(bayes_twin, state, priors,
                                            mcmc.get_samples())
```

**Uncertainty propagation** — Fisher forecasts from exact Jacobians, and
Monte Carlo pushforward. Matrices carry the parameter structure they were
flattened against, so a covariance from the wrong parameterization is
rejected instead of silently misused:

```python
from rheplicant.inference import (fisher_information, parameter_covariance,
                            propagate_covariance, push_forward)

F = fisher_information(forward, params0, noise_std=0.5)     # J^T N^-1 J
cov = parameter_covariance(F)                               # Cramér-Rao
band = propagate_covariance(forward, params0, cov)          # prediction std map
```

**Neural surrogates** — an `eqx.nn.MLP` is just another operator; place it
at any graph node and train only its weights through the same seam:

```python
from rheplicant import At
from rheplicant.radio import NeuralOperator

hybrid = assemble(
    SkyOperator(amplitude=jnp.array(1.0)),
    At("bandpass", NeuralOperator.create(jax.random.key(2),
                                         f_min=60e6, f_max=85e6)),
    GainOperator(gain=jnp.array(1.0)),
)
nn_spec = jax.tree.map(lambda _: False, hybrid)
nn_spec = eqx.tree_at(lambda p: p["bandpass"], nn_spec,
                      jax.tree.map(eqx.is_inexact_array, hybrid["bandpass"]))
# then: build_forward_fn(hybrid, state, filter_spec=nn_spec) + AdamCalibrator
```

See `examples/neural_surrogate.py` for the full recovery of a rippled
bandpass (< 1 % error) and `examples/bayesian_and_uncertainty.py` for the
NUTS-vs-Fisher cross-check.

## 9. Extending

A new physical effect is one small class:

```python
from typing import ClassVar
from rheplicant import AbstractOperator, State

class CableReflectionOperator(AbstractOperator):
    """Sinusoidal ripple from a cable standing wave (example)."""

    requires: ClassVar[tuple[str, ...]] = ("data", "coords.freq")
    provides: ClassVar[tuple[str, ...]] = ("data",)
    graph_node: ClassVar[str] = "bandpass"          # its home on the graph

    amplitude: jax.Array                            # differentiable leaves
    delay: jax.Array

    def __call__(self, state: State) -> State:
        phase = 2 * jnp.pi * state.coords.freq * self.delay
        return state.with_data(state.data * (1 + self.amplitude * jnp.cos(phase)))
```

That is the entire integration: `graph_node` makes it assemblable
(subclasses inherit their base's slot), its array fields are trainable,
`build_forward_fn` / NumPyro / Fisher see them automatically. For operators
whose placement is a modelling decision (surrogates), omit `graph_node` and
place with `At(...)`. To port real physics into a shipped placeholder,
replace the function body and keep the contract — the docstring of every
placeholder says what the real model should be.

## 10. Conventions

| Topic | Rule |
|---|---|
| Angles | degrees in public APIs, radians internally |
| Metadata | strings → `meta` (static); numbers/arrays → `coords`/`env`/`aux` (traced) |
| Randomness | `subkey, state = state.next_key()`; return the advanced state |
| Data grid | radio convention: `data` is `(n_time, n_freq)`; `State` itself takes any pytree |
| Updates | functional only — `replace`, `with_data`, `replace_stage/branch/node`, `eqx.tree_at` |
| Validation | structural only inside `__call__` (jit-safe); loud errors over silent breakage |
| Noise in Bayes | stochastic operators stay OUT of twins handed to `to_numpyro_model` |
| Layering | `rheplicant.core` never imports `rheplicant.radio` / `rheplicant.inference` (enforced by test) |
