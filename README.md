<p align="center">
  <img src="https://raw.githubusercontent.com/RHINO-Experiment/rheplicant/main/docs/_static/rheplicant-banner.png"
       alt="rheplicant — digital twin for the RHINO experiment" width="640">
</p>

# RHEPLICANT

[![Documentation Status](https://readthedocs.org/projects/rheplicant/badge/?version=latest)](https://rheplicant.readthedocs.io/en/latest/)

A **REPLIC**a of an **ANT**enna — a JAX + [Equinox](https://github.com/patrick-kidger/equinox)
framework for building *differentiable replicas* of single-antenna radio
telescopes: horns, dipoles, and dishes alike.
**Documentation: [rheplicant.readthedocs.io](https://rheplicant.readthedocs.io/en/latest/)**

A RHEPLICANT twin is one pure function from sky and instrument parameters to raw
data. Because every stage — foregrounds, ionosphere, beam, receiver
reflections, gain drifts, digitisation — is differentiable, the same twin
that *simulates* an observation also *calibrates* it: gradients, Bayesian
posteriors, Fisher forecasts, and neural surrogates all run through the
instrument model itself, with no re-implementation.

```python
from rheplicant.radio import assemble, GlobalSignalOperator, ForegroundOperator, GainOperator

twin = assemble(GlobalSignalOperator(...), ForegroundOperator(...), GainOperator(...))
observation = twin(state)          # simulate — and differentiate, fit, sample
```

First deployed for RHINO (a horn antenna targeting the 21 cm global signal);
the core is domain-agnostic by construction.

## The name

**RHEPLICANT** is **REPLICANT** wearing RHINO's horn. **REPLICANT** is itself a
portmanteau of **REPLIC**a and **ANT**enna — a digital twin *is* a replica, and
this one is of a radio antenna — the two words overlapping on their shared `A`.
Slip an `H` in behind the first letter and `R…` becomes `RH…`, the mark of
**RH**INO, the horn antenna the framework was first built for:

```
R E P L I C A            replica
            A N T         antenna
─────────────────
R E P L I C A N T        replicant
  + H  →  RH…            (for RHINO)
─────────────────
R H E P L I C A N T      rheplicant
```

One-line gloss: *a differentiable replica of a radio antenna — first, of RHINO.*

## Philosophy

1. **Everything is an operator acting on a state.** One contract —
   `State in, State out` — covers sky models, instrument effects, data
   processing, filters, even neural networks. If it transforms the
   scientific context, it is an operator; there is nothing else to learn.

2. **The twin is a differentiable function.** Every physical parameter is a
   pytree leaf, so `jit`, `grad`, and `vmap` apply to the *entire
   instrument*. Systematics stop being nuisances you correct for and become
   parameters you infer, forecast, and marginalise.

3. **Composition is physics — and it is implicit in the signal path.**
   Sequential effects chain (`Pipeline`), independent contributions add
   (`SumOperator`), switched paths select (`SelectOperator`). The canonical
   signal-path graph knows how elements connect, so `assemble(*operators)`
   builds the right composition from a *set*: provide only a sky and a beam,
   get exactly the beam-convolved sky — partial models come free.

4. **Purity everywhere.** States are immutable (functional updates only),
   randomness is data flowing through the state (one seed reproduces an
   entire run), and operators have no hidden side effects. This is what
   makes the whole twin safe to transform.

5. **Forward models never contain inference.** A single seam —
   `build_forward_fn` — turns any twin into `f(params) -> prediction`.
   Gradient and Adam calibrators, NumPyro posteriors, Fisher forecasts, and
   surrogate training all connect there; calibration never contaminates the
   instrument description.

6. **Interfaces first, physics second.** Every operator ships as a
   trivial-but-runnable placeholder whose *contract* (shapes, PRNG
   consumption, linearity in calibration parameters) is real and tested.
   Real physics replaces function bodies, never interfaces — the native
   differentiable limTOD sky engine arrived exactly this way.

7. **Loud failure over silent wrongness.** Structural validation at every
   boundary, trace-time (jit-safe) shape checks, provenance-tagged
   covariance matrices, assembly-time graph errors. In a framework built to
   chase 0.1 % systematics, a wrong number is worse than an exception.

8. **The core is domain-agnostic.** `rheplicant.core` never imports the radio
   layer (a test enforces it). Radio astronomy is the first application,
   not the design center.

## Install

```bash
pip install rheplicant
# or, for development:
git clone https://github.com/RHINO-Experiment/rheplicant
cd rheplicant && uv sync          # extras: uv sync --extra numpyro
```

Requires Python ≥ 3.11, `jax ≥ 0.5`, `equinox ≥ 0.13`. Distribution and import
name are the same: `rheplicant`.

## Sixty seconds of RHEPLICANT

```python
import jax, jax.numpy as jnp, equinox as eqx
from rheplicant import State, Coordinates
from rheplicant.radio import assemble, SkyOperator, GainOperator, NoiseOperator
from rheplicant.inference import build_forward_fn, GradientCalibrator

state = State(
    coords=Coordinates(time=jnp.linspace(0, 60, 128),
                       freq=jnp.linspace(60e6, 85e6, 32)),
    key=jax.random.key(0),
    meta={"telescope": "my-antenna"},
)

# 1. Simulate: provide operators; the signal-path graph composes them.
twin = assemble(
    SkyOperator(amplitude=jnp.array(1e3)),
    GainOperator(gain=jnp.array(1.1)),          # the truth to recover
    NoiseOperator(sigma=jnp.array(0.5)),
)
observed = eqx.filter_jit(twin)(state)

# 2. Calibrate: freeze everything except the gain, descend the gradient.
model = twin.replace_node("gain", GainOperator(gain=jnp.array(1.0)))
spec = jax.tree.map(lambda _: False, model)
spec = eqx.tree_at(lambda p: p["gain"].gain, spec, replace=True)
forward, params0 = build_forward_fn(model, state, filter_spec=spec)
params_fit, losses = GradientCalibrator(learning_rate=2e-7, n_steps=200).fit(
    forward, params0, observed.data
)
print(jax.tree.leaves(params_fit)[0])           # ~1.1
```

The same `forward` plugs into NUTS posteriors (`to_numpyro_model`), Fisher
forecasts (`fisher_information`), and neural-surrogate training — see the
[guided tour](docs/tour.md).

## What is in the box

- **Core** — `State` (immutable pytree context), `Pipeline` / `SumOperator` /
  `SelectOperator` composition, `SignalGraph` + `assemble` (graph-guided
  auto-composition with lit/dim mermaid & HTML rendering).
- **Radio** — a 29-node canonical signal-path graph covering every element of
  a single-antenna experiment: sky components, ionosphere, RFI, shared
  chromatic beam, noise-wave/reflection terms, CW tone and switched
  calibration loads, gain, thermal noise, EMI, ADC, flagging, averaging —
  plus a modular sky engine — a general differentiable limTOD port and a
  drift-scan m-mode fast path that returns the same numbers orders of
  magnitude cheaper, alongside projection matrices and a numpy-limTOD
  validation bridge — and linear analysis filters (sidereal, sky-space
  map-making, fringe-rate/delay).
- **Inference** — gradient & Adam calibrators, NumPyro bridge with pytree
  priors and posterior predictive, Fisher / Cramér-Rao / delta-method
  uncertainty propagation, Monte Carlo pushforward, `NeuralOperator`
  surrogate stages, MomentRFI flagging bridge, masked likelihoods.

## Documentation

Rendered docs: **[rheplicant.readthedocs.io](https://rheplicant.readthedocs.io)**
(Sphinx + furo; build locally with
`uv run sphinx-build -b html docs docs/_build/html`).

| Document | What it covers |
|---|---|
| [Guided tour](docs/tour.md) | The complete API, top to bottom, with runnable snippets |
| [Operator catalog](docs/operators.md) | Every operator: graph node, role, parameters |
| [Architecture](DESIGN.md) | Design decisions D1–D13, element taxonomy, physics roadmap |
| [Changelog](CHANGELOG.md) | What arrived when |
| `examples/` | Four end-to-end runnable demos |

## Status

The architecture and inference layer are complete and tested end-to-end
(384 tests, ~96 % coverage, jit+grad+vmap through the full twin; assembly
is regression-tested bitwise against hand-built composition). Radio operator
*physics* is deliberately placeholder pending ports from limTOD and friends
— except the sky engines, which are real: a general differentiable limTOD
port and a drift-scan m-mode fast path that agrees with it to float64
roundoff while running ~1000x faster on RHINO's geometry (see
[sky engines](https://rheplicant.readthedocs.io/en/latest/sky-engines.html)).
Conventions:
degrees in public APIs, radians internally; strings in `meta` (static),
numbers in `coords`/`env`/`aux` (traced); one seed reproduces a run.

No CI yet — run `uv run pytest` and `uv run ruff check` before pushing.

## Developers and maintainers

- Zheng Zhang
- Phil Bull
- Jordan Norris
- Rashi Srivastava

## License

MIT
