# The guided tour

RHEPLICANT builds **differentiable digital twins** of radio experiments. A twin
is a *graph of operators* acting on a *state*, and because the whole thing is a
JAX pytree, the same object that simulates a night of data can be differentiated,
jitted, and handed to a sampler.

The tour is in two parts, and one worked example runs through both:

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item-card} Part 1 — Forward modelling
:class-header: sd-font-weight-bold

Assemble the RHINO signal path and record a raw waterfall, exactly as the
instrument would.
^^^
`State` · `Operator` · the graph
+++
The twin as a simulator.
:::

:::{grid-item-card} Part 2 — Bayesian inference
:class-header: sd-font-weight-bold

Hold the sky fixed and get the receiver's four noise-wave temperatures back out
of that waterfall.
^^^
`Latent` · `Bind` · the exact route
+++
The same twin as a model.
:::
::::

Snippets build on each other; pasted top to bottom they form a working script,
and `tests/test_tour_runs.py` runs it. This is an orientation, not a census —
[the operator catalog](operators.md), [inferring anything](inference.md) and
[the API reference](api.md) are the complete surfaces.

---

# Part 1 — Forward modelling

A twin is a **graph of operators**. Each node of the graph is one step of signal
transmission or of signal processing; each operator is a pure `State -> State`
function; and the graph says how they connect. The default graph is RHINO's, but
it is a *default*, not the framework — supplying your own is supported.

```{mermaid}
flowchart LR
    S["State in"] --> OP["Operator"] --> S2["State out"]
    OP -.-> G["one node of the graph"]
```

Two nouns carry everything:

`State`
: **The complete scientific context** of an experiment — data, coordinates,
  environment, metadata, randomness. Immutable, and a JAX pytree.

`Operator`
: **One step**, `State` in and `State` out. Sky models, instrument effects,
  calibration, filtering and neural networks are all the same kind of thing.

## The state

```python
import jax

jax.config.update("jax_enable_x64", True)   # this tour solves in float64

import jax.numpy as jnp
import equinox as eqx
from rheplicant import Coordinates, Environment, State

N_TIME, N_FREQ = 64, 8
freq = jnp.linspace(60e6, 85e6, N_FREQ)              # Hz
time_s = jnp.arange(float(N_TIME)) * 2.0             # seconds from the start

# The switching cycle: antenna, then three calibration loads, round and round.
switch = jnp.arange(N_TIME) % 4

state = State(
    coords=Coordinates(time=time_s, freq=freq,
                       extra={"receiver_input": switch}),
    env=Environment(temperature=jnp.array(280.0)),   # rides along, traced
    key=jax.random.key(20260806),                    # randomness is data
    meta={"telescope": "RHINO", "obs_id": "tour-001"},
)
```

Every field is optional, and there are exactly two channels:

:::{list-table}
:header-rows: 1
:widths: 24 12 64

* - Field
  - Kind
  - What goes in it
* - `data`
  - traced
  - the payload — `(n_time, n_freq)` by radio convention, any pytree in general
* - `coords`
  - traced
  - `time`, `freq`, `pointing`, plus an `extra` dict (the switch cycle above)
* - `env`
  - traced
  - numeric telemetry: temperature, humidity. Rides along for diagnostics, and
    can be promoted into the forward model later with no restructuring
* - `aux`
  - traced
  - your arrays: weights, masks, flags, snapshots
* - `key`
  - traced
  - a typed PRNG key, `jax.random.key(seed)`
* - `meta`
  - **static**
  - strings and labels only — it is part of the jit cache key, so changing it
    recompiles
:::

States never mutate. Updates are functional and re-validated:

```python
s2 = state.replace(meta={"telescope": "other"})   # new object, original untouched
s3 = state.with_data(jnp.zeros((N_TIME, N_FREQ)))  # shorthand for the common case
subkey, s4 = state.next_key()                     # the PRNG protocol: split, advance
raw_kept = s3.checkpoint("raw")                   # zero-copy snapshot into aux
```

:::{dropdown} A new State on every update — doesn't that cost memory?
:color: secondary
:icon: question

**No, because a `State` does not hold your data — it holds *references* to it.**
`replace` builds a new collection of pointers; the buffers are the same ones. The
only allocation is the outer shell, 48 bytes, so `s2.coords is state.coords`. A
16 MB array is never duplicated by an update that did not name it, and sharing is
safe because JAX arrays are immutable. That is also why `checkpoint("raw")` is
free: the snapshot *is* the same buffer under a second name.

**Freeing** happens when no collection lists a buffer any more — not when a
variable is reassigned. `checkpoint` keeps one on purpose; `history.append(state)`
keeps one by accident. If memory grows through a long run, look for the list,
dict or closure collecting states, never at `replace`.

**The one real cost is not memory.** `meta` is static, so it is part of the jit
cache key: a different `meta` is a different compiled program, kept for the life
of the process. Right for a label that changes what the program *is*
(`telescope`, `band`), wrong for one that merely names a run. The test is not
"string or number?" but **would the compiled program differ?**
:::

## Operators

One contract — a pure `State -> State` callable implemented as an
`equinox.Module`. Array-valued fields are automatically differentiable
parameters; there is no registration machinery.

```python
from rheplicant import LambdaOperator
from rheplicant.radio import GainOperator

gain = GainOperator(gain=jnp.array(1.1))     # `gain` is a differentiable leaf
clip = LambdaOperator.on_data(lambda d: jnp.clip(d, 0.0, jnp.inf))

out = gain(state.with_data(jnp.ones((N_TIME, N_FREQ))))
assert jnp.allclose(out.data, 1.1)
```

**Processing is not a different formalism.** `SnapshotOperator` preserves raw
data, `SiderealFilter` and `FourierBandFilter` are linear projections,
`MomentRFIFlaggingOperator` writes flags into `aux` — all of them are operators,
and a pipeline of them composes with the forward chain in the same way. A twin
can therefore be end-to-end (sky through to a calibrated spectrum) or any
sub-path you like: the tour's example stops at the ADC, because a raw waterfall
is what the instrument actually records.

**Writing your own** is one small class:

```python
from typing import ClassVar
from rheplicant import AbstractOperator

class CableReflectionOperator(AbstractOperator):
    """Sinusoidal ripple from a cable standing wave (example)."""

    requires: ClassVar[tuple[str, ...]] = ("data", "coords.freq")
    provides: ClassVar[tuple[str, ...]] = ("data",)
    graph_node: ClassVar[str] = "bandpass"      # its home on the graph

    amplitude: jax.Array                         # differentiable leaves
    delay: jax.Array

    def __call__(self, state: State) -> State:
        phase = 2 * jnp.pi * state.coords.freq * self.delay
        return state.with_data(state.data * (1 + self.amplitude * jnp.cos(phase)))
```

That is the whole integration: `graph_node` makes it assemblable, its array
fields are trainable, and every inference exit sees them automatically. Three
rules for implementors — never mutate the input state; draw randomness only via
`state.next_key()`, returning the advanced state; validate structure only
(shapes and dtypes — value checks break under jit).

### Three ways to compose, and only three

::::{grid} 1 1 3 3
:gutter: 2

:::{grid-item-card} Cascade
:class-header: sd-font-weight-bold

One after another — each stage transforms what the last produced.
^^^
```python
# sketch
Pipeline(sky, beam, gain)
```
+++
A chain of `transform` nodes.
:::

:::{grid-item-card} Sum
:class-header: sd-font-weight-bold

Independent contributions that **add**. Each branch gets the input context with
`data` stripped, and its own PRNG subkey.
^^^
```python
# sketch
SumOperator(signal, foregrounds)
```
+++
A `junction` node.
:::

:::{grid-item-card} Switch
:class-header: sd-font-weight-bold

Alternatives, one **selected** per time sample by an integer cycle in
`coords.extra`. They replace, they do not add.
^^^
```python
# sketch
SelectOperator(antenna, cal_load)
```
+++
A `selector` node.
:::
::::

Each is itself an operator, so they nest arbitrarily. Nothing else composes
anything.

```python
from rheplicant import Pipeline, SumOperator
from rheplicant.radio import ForegroundOperator, GlobalSignalOperator

sky = SumOperator(
    GlobalSignalOperator(depth=jnp.array(0.5), centre=jnp.array(75e6),
                         width=jnp.array(5e6)),
    ForegroundOperator(amplitude=jnp.array(2500.0),
                       spectral_index=jnp.array(2.55), ref_freq=70e6),
    names=("signal", "foregrounds"),
)
observed_sky = Pipeline(sky, gain, names=("sky", "gain"))(state).data
```

:::{dropdown} My gain isn't a constant — do I need a different operator?
:color: secondary
:icon: question

**Almost never.** `GainOperator.gain` is an array field, so it already takes
whatever shape the operator accepts: a scalar for a constant, `(n_time,)` for a
per-sample drift. Frequency structure lives at the `bandpass` node instead;
inferred jointly and freely those two share one *exactly* null direction, which
is why the bandpass is declared through `unit_mean_bandpass`.

**An arbitrary parameterisation** — a polynomial, `exp` of one — is still the
same operator. *What* is inferred and *how it enters* are separate declarations
(Part 2): the operator keeps multiplying by an array, and the parameterisation is
a `Bind`.

```python
# sketch
Latent("g_coeff", init=jnp.zeros(4))                     # inferred: 4 coefficients
Bind("g_coeff", into=lambda p: p["gain"].gain,           # the leaf it drives
     fn=lambda c: jnp.exp(basis_matrix("legendre", n=N_TIME, n_basis=4) @ c))
```

A `PolynomialGainOperator` would make every choice of family, order and link
function its own class — and its own graph slot and jit cache entry — for a
forward model whose structure never changed. It also costs you the payoff:
`g = B @ c` is *linear* in the coefficients, so `Latent(..., linear=True)` sends
that block to the exact conjugate draw rather than to a gradient sampler.

**A new operator is right when the *algebra* changes**, not the parameterisation
— a complex gain, or a 2×2 Jones matrix over two polarisations. "It varies with
something" is a shape; "it multiplies differently" is an operator.
:::

:::{dropdown} Which of an operator's declarations are actually enforced?
:color: secondary
:icon: question

Operators declare `requires` / `provides` (State paths read and written),
`graph_node` (home on a template) and `must_precede` (what the contribution must
flow through). **Two are enforced.**

`"key"` in `requires` is a **contract**: it says this operator draws randomness,
and every inference exit refuses a model containing one — a frozen draw from the
template key would be added to every prediction alike, a bias that is exactly
affine and full rank, so no shape check, no linearity check and no rank test can
see it.

`must_precede` is enforced by `assemble` — see the warning in the next section.

The rest is descriptive **by decision, not omission**: `provides` is `("data",)`
on 26 of 31 declaring classes, so enforcing it would distinguish nothing, and an
operator that reads a field *if present* would be wrongly refused.
:::

## Graph assembly

The canonical path does the composing. Composition is **implicit in the
signal path**: a graph is a template of operator
slots plus the structure joining them; you provide a *set* of operators and
`assemble` compiles the sub-path they induce, folding it into exactly the three
structures above.

:::{list-table} What `assemble` does with each node kind
:header-rows: 1
:widths: 16 42 42

* - Node kind
  - You provide none
  - You provide one or more
* - `source`
  - pruned
  - it creates data
* - `transform`
  - passes through as identity
  - it chains, in graph order
* - `junction`
  - —
  - one branch: identity. Two or more: a `SumOperator`
* - `selector`
  - —
  - a `SelectOperator`, branch order fixed by the graph
:::

Branch order comes from the graph, never from your argument order — so the same
set of operators always folds to the same tree, with the same names, the same
PRNG stream and the same jit cache entry.

Now the worked example. This is the RHINO forward segment, ending at the ADC:

```python
# needs-extra: rhino_cal_jax
import rhino_cal_jax as rcj
from rheplicant.radio import (
    ADCOperator, AntennaLossOperator, BeamSpillOperator, CalLoadOperator,
    NoiseOperator, NoiseWaveOperator, ReceiverOperator, assemble,
)

F_SKY, T_GROUND = 0.97, 290.0            # horizon split
ETA, T_PHYS = 0.97, 293.0                # horn ohmic loss
ADC_SCALE, N_BITS = 0.25, 12             # counts per kelvin at unit gain
SIGMA_POST_GAIN = 2.0                    # thermal noise, post-gain units

gamma_rec = rcj.termination_gamma("resistive", N_FREQ, impedance=45.0)
gamma_src = jnp.stack([                  # ROW ORDER = the selector's branch order
    rcj.cable_gamma(rcj.termination_gamma("open", N_FREQ), freq, length=2.0, loss=0.92),
    rcj.termination_gamma("resistive", N_FREQ, impedance=10.0),
    rcj.cable_gamma(rcj.termination_gamma("short", N_FREQ), freq, length=0.4, loss=0.98),
    rcj.cable_gamma(rcj.termination_gamma("resistive", N_FREQ, impedance=150.0),
                    freq, length=1.1, loss=0.95),
])

TRUE = {                                 # the four noise-wave temperatures, per channel
    "t_unc": 250.0 + 20.0 * jnp.linspace(-1.0, 1.0, N_FREQ),
    "t_cos": 30.0 * jnp.cos(jnp.linspace(0.0, 3.0, N_FREQ)),
    "t_sin": -40.0 + 8.0 * jnp.linspace(-1.0, 1.0, N_FREQ) ** 2,
    "t_rx": 290.0 + 5.0 * jnp.linspace(-1.0, 1.0, N_FREQ) ** 3,
}

# The bandpass carries SHAPE (mean 1), the gain carries the level.
bandpass = 1.0 + 0.10 * jnp.cos(2 * jnp.pi * (freq - freq[0]) / (freq[-1] - freq[0]))
bandpass = bandpass / jnp.mean(bandpass)
gain_t = 1.0 + 0.02 * jnp.sin(2 * jnp.pi * time_s / 60.0)

twin = assemble(
    GlobalSignalOperator(depth=jnp.array(0.5), centre=jnp.array(75e6),
                         width=jnp.array(5e6)),
    ForegroundOperator(amplitude=jnp.array(2500.0),
                       spectral_index=jnp.array(2.55), ref_freq=70e6),
    BeamSpillOperator(sky_fraction=jnp.array(F_SKY), t_ground=jnp.array(T_GROUND)),
    AntennaLossOperator(efficiency=jnp.array(ETA), t_physical=jnp.array(T_PHYS)),
    CalLoadOperator(t_load=jnp.array(300.0)),        # ambient
    CalLoadOperator(t_load=jnp.array(400.0)),        # hot
    CalLoadOperator(t_load=jnp.array(1200.0)),       # noise source
    NoiseWaveOperator(**TRUE,
                      gamma_src_re=gamma_src.real, gamma_src_im=gamma_src.imag,
                      gamma_rec_re=gamma_rec.real, gamma_rec_im=gamma_rec.imag),
    ReceiverOperator(bandpass=bandpass),
    GainOperator(gain=gain_t),
    NoiseOperator(sigma=jnp.array(SIGMA_POST_GAIN)),
    ADCOperator(scale=jnp.array(ADC_SCALE), n_bits=N_BITS),
)
observed = twin(state).data                          # the raw waterfall
```

Nothing in that call says what connects to what. The graph does: the two sky
terms **sum**, the antenna stages **chain**, and `receiver_input` is a
**selector**, so each `CalLoadOperator` *replaces* the antenna on its own switch
position instead of adding to it.

```{mermaid}
flowchart LR
    GS["global_signal"] --> AS(("astro_sum"))
    FG["foregrounds"] --> AS
    AS --> BSP["beam_spill"] --> AL["antenna_loss"] --> SW{{"receiver_input"}}
    L1["cal load 300 K"] --> SW
    L2["cal load 400 K"] --> SW
    L3["cal load 1200 K"] --> SW
    SW --> NW["noise_wave"] --> BP["bandpass"] --> G["gain"] --> N["noise"] --> ADC["adc"]
    ADC --> W["waterfall (64, 8)"]
```

The result is an `Assembly` — an ordinary operator, with node-id ergonomics:

```python
print(twin)                                   # lit nodes + nodes traversed as identity
print("switch order:", twin["receiver_input"].names)
twin["gain"]                                  # node-id access, at any nesting
twin2 = twin.replace_node("gain", GainOperator(gain=jnp.array(1.0)))
svg = twin.to_svg()                           # also .to_mermaid() / .to_html()
```

```text
Assembly(graph='single-antenna', lit=['global_signal', 'foregrounds', 'beam_spill',
'antenna_loss', 'cal_loads x3', 'noise_wave', 'bandpass', 'gain', 'noise', 'adc'],
skipped-as-identity=['ionosphere', 'atmosphere_field', 'field_sum', 'beam',
'astro_ant_sum', 't_ant_sum', 'cw_tone', 'emi'])
switch order: ('astro_sum', 'cal_loads_1', 'cal_loads_2', 'cal_loads_3')
```

Two things to read off that output. The **skipped** nodes are the template
traversed as identity — nothing was provided for them, and no `SumOperator`
wrapping a single branch was materialised. And the **switch order** is a fact you
must read, never assume: it is the order `gamma_src`'s rows have to be stacked
in, and its labels depend on which sibling leaves you supplied.

:::{warning}
**Placement can be silently wrong, so state the constraint.** `At(node, op)`
puts any operator anywhere, so an ordering rule written only in prose is one
nothing checks: a CW calibration tone assembled *after* the gain builds cleanly,
every shape correct, and its gain response is exactly 1.0 — it monitors nothing.

Declaring `must_precede = ("bandpass", "gain")` makes `assemble` refuse that
placement instead. The test is **reachability** — does my contribution flow
*through* that node — not sort order, which is why it needs the graph's node ids
rather than `State` paths.
:::

Two more things `assemble` refuses, and three escape hatches:

- **Refuses:** caller data handed to a sourced assembly (it would be silently
  discarded); a transform feeding a sum with no live source upstream; an operator
  placed at a node of the wrong kind.
- **Escapes:** `At(node, op)` places anything anywhere; `At((n1, n2), op)` lets
  one operator cover a contiguous region atomically; *equivalent-entry leaves*
  let the same physics enter in different forms (ground spill as a pre-beam
  field, or as a post-beam effective temperature).

The default template `RADIO_GRAPH` has 32 nodes and is **RHINO's** structure, not
the framework's — `SignalGraph`, `register_graph` and `get_graph` are public and
domain-agnostic. See [the canonical signal path](signal-path.md) for the rendered
graph and the node table, and [the operator catalog](operators.md) for what lives
at each node.

---

# Part 2 — Bayesian inference

The twin is now a *model*: `forward(params) -> prediction`, with everything else
closed over. The question this part answers is the one the worked example set up
— **the sky and the beam are given; what were the receiver's four noise-wave
temperatures?**

Inference is declared in three layers, and it is worth keeping them apart:

:::{list-table}
:header-rows: 1
:widths: 22 78

* - Layer
  - The question it answers
* - **The model**
  - which quantities are free, and how they enter the twin
* - **The likelihood**
  - what the noise is — a noise model *is* a likelihood
* - **The engine**
  - how to get the posterior, given the shape the first two produced
:::

## The model: what is free, and how it enters

Two words carry it. A `Latent` is **a named quantity you infer**; a `Bind` is
**a rule turning latent values into pipeline leaf values**. Keeping them separate
is what lets one latent drive several stages, or a leaf be a transform of several
latents, without a new operator for each combination.

```python
from rheplicant.inference import Bind, Latent, ParameterSpace

NAMES = ("t_unc", "t_cos", "t_sin", "t_rx")

space = ParameterSpace(
    latents=[Latent(n, init=jnp.zeros((N_FREQ,)), linear=True) for n in NAMES],
    bindings=[
        Bind("t_unc", into=lambda p: p["noise_wave"].t_unc),
        Bind("t_cos", into=lambda p: p["noise_wave"].t_cos),
        Bind("t_sin", into=lambda p: p["noise_wave"].t_sin),
        Bind("t_rx", into=lambda p: p["noise_wave"].t_rx),
    ],
)
```

Four latents, one per temperature family, each free **per channel** — 32 numbers.
`linear=True` is a claim about how they enter, and it is checked before it is
used.

## The likelihood: a noise model is a likelihood

Giving the noise is giving the likelihood — `RadiometerNoise(...)` for the
radiometer equation, `HomoscedasticNoise(...)` for a single σ,
`FlaggedNoise(inner, flags)` to down-weight flagged samples. Nothing else about
the twin changes.

Which is why a twin that draws its *own* randomness is not a model, and every
inference exit refuses one:

```python
from rheplicant.inference import check_linearity, linear_operator

try:
    linear_operator(space, twin, state, names=NAMES, check=False)
except Exception as exc:
    print(f"{type(exc).__name__}: {str(exc)[:70]}...")

fit_twin = twin.without("noise")          # the supported repair, one line
```

:::{danger}
A frozen draw from the template key would be added to **every** prediction
alike. The corruption is exactly affine and full rank, so no shape check, no
linearity check and no rank test can see it — which is why this is a refusal at
the door rather than a diagnostic afterwards.
:::

The noise still exists; it has just moved to where it belongs. Here it entered
before the ADC's scaling, so the σ the likelihood needs is `ADC_SCALE *
SIGMA_POST_GAIN`. Get that factor wrong and nothing complains: shapes are fine,
the solve converges, and only the posterior *width* is wrong.

## The engine: exact where the model is linear

The four temperatures enter the system temperature additively, and every stage
after them here — bandpass, gain, ADC scaling below saturation — is a multiply.
So the prediction is **exactly affine** in them, and that is not a matter of
taste about which sampler to use:

```python
errors = check_linearity(space, fit_twin, state, names=NAMES)
print(f"worst relative departure from affine: {max(errors.values()):.1e}")
```

```text
worst relative departure from affine: 9.4e-11
```

:::{list-table}
:header-rows: 1
:widths: 26 34 40

* - If the block is…
  - Use
  - Because
* - exactly linear
  - `wiener_solve` (mean), `gcr_sample` (exact draws)
  - the posterior is a Gaussian available in closed form
* - anything else
  - NUTS, via `to_numpyro_model`
  - a gradient sampler is what an unknown shape needs
* - a mix
  - `SamplingPlan` with `Block`s
  - each block's engine is *derived* from `linear=True`, never restated
* - no likelihood at all
  - `NeuralPosterior` (simulation-based)
  - you can simulate but not evaluate
:::

For this example the top row applies, so NUTS would be theatre — hundreds of
gradient evaluations per draw to explore a Gaussian we can write down:

```python
from rheplicant.inference import gcr_sample, wiener_solve

NOISE_STD = ADC_SCALE * SIGMA_POST_GAIN
PRIOR_STD = dict.fromkeys(NAMES, 100.0)
PRIOR_MEAN = dict.fromkeys(NAMES, 0.0)

block = linear_operator(space, fit_twin, state, names=NAMES, check=False)
solved, residual = wiener_solve(block, observed, noise_std=NOISE_STD,
                                prior_std=PRIOR_STD, prior_mean=PRIOR_MEAN,
                                tol=1e-12, maxiter=4000)

keys = jax.random.split(jax.random.key(7), 200)
draws = jax.vmap(lambda k: gcr_sample(
    block, observed, noise_std=NOISE_STD, prior_std=PRIOR_STD,
    prior_mean=PRIOR_MEAN, key=k, tol=1e-12, maxiter=4000)[0])(keys)
```

`linear_operator` exports `A`, `Aᵀ` and the offset **without ever forming a
matrix**; `wiener_solve` gives the posterior mean by conjugate gradients and
`gcr_sample` gives exact draws, one solve each.

## Reading the answer honestly

```python
for name in NAMES:
    err = solved[name] - TRUE[name]
    sig = draws[name].std(axis=0)
    print(f"{name:5s} RMS err {float(jnp.sqrt(jnp.mean(err ** 2))):6.3f} K"
          f" | posterior sigma {float(sig.min()):5.2f}..{float(sig.max()):5.2f} K"
          f" | worst pull {float(jnp.max(jnp.abs(err / sig))):.2f}")
```

```text
t_unc RMS err  2.345 K | posterior sigma  1.18..10.25 K | worst pull 3.03
t_cos RMS err  0.734 K | posterior sigma  0.48.. 3.31 K | worst pull 2.00
t_sin RMS err  1.395 K | posterior sigma  0.57.. 7.24 K | worst pull 2.45
t_rx  RMS err  1.136 K | posterior sigma  0.72.. 2.39 K | worst pull 3.05
```

**The claim to take away is not "1 K accuracy" — it is that the errors sit inside
the error bars the same machinery reports.** Over twelve noise realisations the
pulls give χ²/dof = **1.00** (range 0.49–1.77), and the largest of 32 pulls has
median 2.3 and range 1.7–3.3 — which is what thirty-two draws from a normal look
like. One realisation's worst pull is not the diagnostic; the χ² across
realisations is.

Posterior σ runs from 0.5 to 10 K against a per-sample scatter of 2 K, because
the per-channel 4×4 system over the four coupling coefficients is square but not
orthogonal: four sources separate the columns only moderately. That amplification
is the physics of noise-wave calibration, not a defect of the solve.

:::{tip}
**Four free temperature families need four switch positions.** The antenna counts
as one, so three calibration loads are the minimum — with fewer, `t_rx` has to be
held fixed. Collapse the four Γ's to one value and the design matrix drops from
rank 32 to rank 8, with the posterior falling back onto the prior. The switching
cycle *is* the calibration design.
:::

One diagnostic no per-block residual can replace: **is the model identified at
all?** `identifiability()` is a rank test on the Jacobian with respect to every
latent at once — a degeneracy whose two halves live in different blocks leaves
each conditional looking perfectly well posed.

```python
from rheplicant.inference import identifiability

report = identifiability(space, fit_twin, state)
print(f"rank {report.rank} of {report.n_par} parameters, nullity {report.nullity}")
```

## Where to go next

:::{list-table}
:header-rows: 1
:widths: 34 66

* - You want to…
  - Read
* - declare something more elaborate than one latent per leaf
  - [Inferring anything](inference.md) — tied and derived bindings, `fan=`
* - fit a model that is *not* linear
  - [Tutorial: a gradient posterior](tutorial-nuts.md), and how to tell it is wrong
* - see the exact route worked end to end
  - [Tutorial: an exact posterior for a big linear block](tutorial-gcr.md)
* - forecast rather than fit
  - `fisher_information`, `parameter_covariance`, `propagate_covariance`
* - replace a stage with a neural surrogate
  - `NeuralOperator` at any node, trained through the same seam
* - keep a campaign after the recordings are gone
  - `BayesMemory` — accumulate likelihood factors, discard the data
:::

---

## Conventions

| Topic | Rule |
|---|---|
| Angles | degrees in public APIs, radians internally |
| Data grid | radio convention: `data` is `(n_time, n_freq)`; `State` itself takes any pytree |
| Randomness | `subkey, state = state.next_key()`, return the advanced state — and declare `"key"` in `requires`, which is what makes the stage findable |
| Errors | every refusal derives from `DirtError` *and* from its closest builtin — `except ValueError` catches all of them ([contracts](contracts.md#one-base-class-for-every-refusal)) |
| Protected channels | the operator injecting a calibrator writes the channels it wet to `aux['protected']`; flaggers clear them ([contracts](contracts.md#protected-channels-keeping-a-known-calibrator-out-of-the-flags)) |
| Layering | `rheplicant.core` never imports `rheplicant.radio` / `rheplicant.inference` (enforced by test) |
