<p align="center">
  <img src="https://raw.githubusercontent.com/RHINO-Experiment/rheplicant/main/docs/_static/rheplicant-banner.png"
       alt="rheplicant — digital twin for the RHINO experiment" width="640">
</p>

# RHEPLICANT

[![Documentation Status](https://readthedocs.org/projects/rheplicant/badge/?version=latest)](https://rheplicant.readthedocs.io/en/latest/)

A **REPLIC**a of an **ANT**enna — a **JAX model of a radio telescope, run as a
digital twin**. Built for **RHINO**, a horn antenna measuring the 21 cm global
signal, and domain-agnostic underneath: horns, dipoles and dishes alike.
(JAX + [Equinox](https://github.com/patrick-kidger/equinox).)
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

## Four things it is built to do

| | |
|---|---|
| **1 · Forward modelling** | Simulate what any stage of the experiment would produce — a sky, a receiver output, a processed product. Where you stop is a property of the graph. |
| **2 · Bayesian inference** | Read the same twin backwards. Free any subset of what it contains; the noise model *is* the likelihood; the engine follows from the model's structure. |
| **3 · Neural surrogates** | Replace an expensive stage with a trained network and leave the graph's shape untouched — or amortize the posterior itself. |
| **4 · Streaming evidence** | Keep a campaign after its recordings are archived: compress each night to a fixed-size likelihood factor, then discard the data. |

None of the four is a separate mode. They all read the **same twin object**,
which is what makes the calibration you fit the simulator you trust.

## Two nouns

**`State`** — the complete scientific context: data, coordinates, environment,
randomness, metadata. It organises *references* to buffers rather than the
buffers themselves, so a derived state allocates the shell and nothing else
(48 bytes, with a 16 MB array shared rather than copied). JAX arrays are
immutable, which is what makes sharing safe.

**`Operator`** — one step, `State` in and `State` out. Sky models, instrument
effects, calibration, filtering and neural networks are all the same kind of
thing, each carrying its physical parameters as differentiable leaves.

**`state.data` always references what the instrument has produced so far.**
For example: the sky engine produces the `(n_time, n_freq)` antenna
temperature, the antenna's ohmic loss produces that array after loss, the
receiver produces a system temperature. Nothing is written in place — each
stage hands back a *new* `State` whose `data` points at its own result, while
the fields it did not touch go on pointing where they already did.

The sky map itself is not in `state.data` — it is a **parameter of the sky
model**, differentiable like every other, which is why a map can be inferred
rather than merely assumed.

## Three ways to join them — and you rarely write any

**Cascade** (`Pipeline`) for sequential effects, **sum** (`SumOperator`) for
independent contributions that add, **switch** (`SelectOperator`) for
alternatives with one selected per time sample. There is no fourth.

You normally write none of them. Declare the operators you want and `assemble`
reads the canonical signal path — the template that says which operators exist
and what joins them — to decide the composition. Reach for the three
combinators directly only when building a structure the template does not
describe.

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

1. **Everything is an operator acting on a state.** One contract, `State in,
   State out`, covers sky models, instrument effects, processing, filters and
   neural networks alike.
2. **The twin is a differentiable function.** Every physical parameter is a
   pytree leaf, so `jit`/`grad`/`vmap` apply to the whole instrument, and a
   systematic becomes something you infer rather than correct for.
3. **Composition is physics, implicit in the signal path.** Chains, sums and
   switches are read off the canonical graph, so `assemble` builds the right
   structure from a *set* of operators and partial models come free.
4. **Purity everywhere.** Immutable states, randomness as data, no hidden side
   effects — which is what makes the twin safe to transform at all.
5. **Forward models never contain inference.** One seam turns any twin into
   `f(params) -> prediction`, and a `ParameterSpace` re-parameterizes without
   ever editing an operator.
6. **Interfaces first, physics second.** A placeholder body may ship; its
   contract may not be one. Real physics replaces bodies, never interfaces.
7. **Loud failure over silent wrongness.** Chasing 0.1 % systematics, a wrong
   number is worse than an exception.
8. **The core is domain-agnostic.** `rheplicant.core` never imports the radio
   layer, and a test enforces it.

Each is argued at length in
**[the documentation](https://rheplicant.readthedocs.io/en/latest/)**;
[Status](#status) says which operators are still placeholders.

## Install

```bash
# limTOD carries the sky engines and is a dependency, not an extra. It is on
# PyPI as of 1.10.0, so it comes with the install.
pip install rheplicant
pip install "rheplicant[cal]"     # + the noise-wave model (rhino-cal-jax)

# or, for development:
git clone https://github.com/RHINO-Experiment/rheplicant
cd rheplicant
uv venv                          # NOT `uv sync`, which cannot work here
uv pip install -e . --group dev
```

Requires Python ≥ 3.11, `jax ≥ 0.5`, `equinox ≥ 0.13`. Distribution and import
name are the same: `rheplicant`. Full instructions, the four extras and the
two-session test split are on the
[install page](https://rheplicant.readthedocs.io/en/latest/install.html).

## Seeing it work

The snippet above is the whole shape of it: provide operators, let the graph
compose them, call the result. What that `twin` then plugs into — gradients,
NUTS posteriors, Fisher forecasts, exact conjugate draws, neural surrogates —
is one worked example carried end to end in
**[the guided tour](https://rheplicant.readthedocs.io/en/latest/tour.html)**,
and fourteen runnable scripts with measured wall clocks in
[`examples/`](https://github.com/RHINO-Experiment/rheplicant/tree/main/examples).

## What is in the box

| Layer | What lives there |
|---|---|
| **Core** | `State`, the three combinators, `SignalGraph` + `assemble`, and `Assembly.replace_node` / `.without` to swap or drop a stage by node id. Domain-agnostic — a test enforces the layering. |
| **Radio** | A 33-node canonical signal path for a single-antenna experiment, a modular sky engine (a differentiable limTOD port plus a drift-scan m-mode fast path agreeing with it to roundoff), and linear analysis filters. |
| **Inference** | One noise model read by the likelihood, the weights, the Fisher matrix and the NumPyro scale alike; gradient and conjugate engines; `SamplingPlan` to partition a model into blocks whose engine is *derived* rather than restated; streaming evidence for campaigns whose data is gone. |

Per-operator detail is the
[operator catalog](https://rheplicant.readthedocs.io/en/latest/operators.html);
every signature is the
[API reference](https://rheplicant.readthedocs.io/en/latest/api.html).

## Documentation

**[rheplicant.readthedocs.io](https://rheplicant.readthedocs.io)** — the guided
tour, the operator catalog, the inference rules, the tutorials, the API, the
architecture decisions and the changelog, with a sidebar that lists them.
Build it locally with
`cd docs && ../.venv/bin/python -m sphinx -n -b html . _build/html`.

Design decisions D1–D36 and the physics roadmap are in
[`DESIGN.md`](https://rheplicant.readthedocs.io/en/latest/design.html); what
arrived when is in
[`CHANGELOG.md`](https://rheplicant.readthedocs.io/en/latest/changelog.html).

## Status

The architecture and inference layer are complete and tested end-to-end
(7638 tests, 90.2 % coverage, jit+grad+vmap through the full twin; assembly
is regression-tested bitwise against hand-built composition). Radio operator
*physics* is deliberately placeholder where the docstring says so — 15 of the
29 concrete `rheplicant.radio` operator classes — pending ports from limTOD
and friends. The other twelve no longer carry that wording: the sky engines
are real (a general differentiable limTOD port and a drift-scan m-mode fast
path that agrees with it to float64 roundoff while running ~1000x faster on
RHINO's geometry — see
[sky engines](https://rheplicant.readthedocs.io/en/latest/sky-engines.html)),
and so are the horizon split, the horn's ohmic loss, the noise-wave reflection
terms of the noise-wave data model, the CW calibration tone, and the
separable-basis antenna temperature; the Touchstone and RHINO-HDF5 readers are
an ingestion layer, not a stand-in for one.

Three of the fifteen are load-bearing even so. `ReceiverOperator`,
`GainOperator` and `CalLoadOperator` have placeholder *bodies* — no flicker,
no measured band shape, no load reflection or telemetry — but real shape and
contract: the receiver module's `unit_mean_bandpass` / `unit_mean_free` are the
bandpass/gain identifiability convention, the gain's exact linearity in `gain`
is what `Latent(..., linear=True)` claims about it, and
`CWCalibrationOperator.must_precede == ('bandpass', 'gain')` names the two
nodes the first two occupy. Conventions:
degrees in public APIs, radians internally; strings in `meta` (static),
numbers in `coords`/`env`/`aux` (traced); one seed reproduces a run.

No CI yet, and the suite is two pytest sessions rather than one — the evidence
layer needs float64 while eighteen tests elsewhere assert refusals that only
float32 forces, and `jax_enable_x64` is process-global. Plain `pytest` runs both
for you. That split is also why the reported coverage is what it is rather than
the 99.7 % it was before the evidence layer landed: the second session runs
`--no-cov` in its own process, so its passing tests contribute nothing to the
default report, and most of the default report's uncovered statements are the
evidence-layer files that session covers. The
[install page](https://rheplicant.readthedocs.io/en/latest/install.html#running-the-tests)
makes the same argument at length.

Neither `uv sync` nor `uv run` works here, with or without `--frozen`: locking
resolves every declared extra and two of them name packages that are not on
PyPI by design, so no lockfile exists or can be made. Use `uv venv` plus
`uv pip install`, and call the venv's interpreter directly. Two optional
datasets that cannot be published — the CST beam exports and a `rhino-cal`
checkout — are named by `RHEPLICANT_RHINO_BEAMS` and `RHEPLICANT_RHINO_CAL`
rather than guessed at a path; without them the work that needs them stands
down and says so. Commands, reasons and the rest of the setup are in
**[Install](https://rheplicant.readthedocs.io/en/latest/install.html)**.

## Developers and maintainers

- Zheng Zhang
- Phil Bull
- Jordan Norris
- Rashi Srivastava

## License

MIT
