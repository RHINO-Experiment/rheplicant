<p align="center">
  <img src="https://raw.githubusercontent.com/RHINO-Experiment/rheplicant/main/docs/_static/rheplicant-banner.png"
       alt="rheplicant: digital twin for the RHINO experiment" width="640">
</p>

# RHEPLICANT

[![Documentation Status](https://readthedocs.org/projects/rheplicant/badge/?version=latest)](https://rheplicant.readthedocs.io/en/latest/)

A **REPLIC**a of an **ANT**enna: a **JAX model of a radio telescope that runs as
a digital twin**. RHEPLICANT was built for **RHINO**, a horn antenna measuring
the 21 cm global signal, but its underlying framework supports multiple antenna
types, including horns, dipoles and dishes.
(JAX + [Equinox](https://github.com/patrick-kidger/equinox).)
**Documentation: [rheplicant.readthedocs.io](https://rheplicant.readthedocs.io/en/latest/)**

A RHEPLICANT twin is one pure function from sky and instrument parameters to raw
data. Every stage (foregrounds, ionosphere, beam, receiver reflections, gain
drifts, digitisation) is differentiable, so the twin that simulates an
observation also calibrates it: gradients, Bayesian posteriors, Fisher
forecasts, and neural surrogates all run through the instrument model itself,
with no re-implementation.

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
| **1 · Forward modelling** | Simulate what any stage of the experiment would produce: a sky, a receiver output, a processed product. Where you stop is a property of the graph. |
| **2 · Bayesian inference** | Read the same twin backwards. Free any subset of what it contains; the noise model is the likelihood; the engine follows from the model's structure. |
| **3 · Neural surrogates** | Replace an expensive stage with a trained network and leave the graph's shape untouched, or amortize the posterior itself. |
| **4 · Streaming evidence** | Keep a campaign after its recordings are archived: compress each night to a fixed-size likelihood factor, then discard the data. |

None of the four is a separate mode. All four read the same twin object, so
the twin you calibrate is the twin you simulate with.

**2 and 4 are built on [bayesmith](https://pypi.org/project/bayesmith/).**
bayesmith does Bayesian inference over an explicit graph and has no radio
astronomy in it. It is a required dependency of this package
(`bayesmith>=0.10,<0.11`). `rheplicant.inference` holds what depends on the
instrument (the parameter space, the noise models, the plan and the
accumulation of a campaign), builds a bayesmith graph from a twin, and calls
bayesmith for the block partition, the exact linear-Gaussian solves, the
Fisher matrix, the diagnostics and the convergence certificate. If you have a
model that is not an instrument twin, use bayesmith directly. The
[bayesmith page](https://rheplicant.readthedocs.io/en/latest/bayesmith.html)
lists each delegation and the accepted versions, and the
[inference pages](https://rheplicant.readthedocs.io/en/latest/inference.html)
carry the detail.

## Two nouns

**`State`** is the complete scientific context: data, coordinates, environment,
randomness, metadata. It organises references to buffers rather than the
buffers themselves, so a derived state allocates only the shell (48 bytes,
with a 16 MB array shared rather than copied). JAX arrays are immutable, so
sharing is safe.

**`Operator`** is one step, `State` in and `State` out. Sky models, instrument
effects, calibration, filtering and neural networks are all the same kind of
thing, each carrying its physical parameters as differentiable leaves.

`state.data` always references what the instrument has produced so far.
For example: the sky engine produces the `(n_time, n_freq)` antenna
temperature, the antenna's ohmic loss produces that array after loss, the
receiver produces a system temperature. Nothing is written in place: each
stage returns a new `State` whose `data` points at its own result, and the
fields it did not touch point where they already did.

The sky map is not in `state.data`. It is a parameter of the sky model,
differentiable like every other, so a map can be inferred rather than
assumed.

## Three ways to join them

**Cascade** (`Pipeline`) for sequential effects, **sum** (`SumOperator`) for
independent contributions that add, **switch** (`SelectOperator`) for
alternatives with one selected per time sample. There is no fourth.

You normally write none of them. Declare the operators you want and `assemble`
reads the canonical signal path (the template that says which operators exist
and what joins them) to decide the composition. Use the three combinators
directly only when building a structure the template does not describe.

## The name

**RHEPLICANT** is **REPLICANT** wearing RHINO's horn. **REPLICANT** is a
portmanteau of **REPLIC**a and **ANT**enna, the two words overlapping on their
shared `A`: a digital twin is a replica, and this one is of a radio antenna.
An `H` behind the first letter turns `R…` into `RH…`, for **RH**INO, the horn
antenna the framework was first built for:

```
R E P L I C A            replica
            A N T         antenna
─────────────────
R E P L I C A N T        replicant
  + H  →  RH…            (for RHINO)
─────────────────
R H E P L I C A N T      rheplicant
```

One-line gloss: *a differentiable replica of a radio antenna, first of RHINO.*

## Philosophy

1. **Everything is an operator acting on a state.** One contract, `State in,
   State out`, covers sky models, instrument effects, processing, filters and
   neural networks alike.
2. **The twin is a differentiable function.** Every physical parameter is a
   pytree leaf, so `jit`/`grad`/`vmap` apply to the whole instrument, and a
   systematic becomes something you infer rather than correct for.
3. **Composition is physics, implicit in the signal path.** Chains, sums and
   switches are read off the canonical graph, so `assemble` builds the right
   structure from a set of operators and partial models come free.
4. **Purity everywhere.** Immutable states, randomness as data and no hidden
   side effects make the twin safe to transform.
5. **Forward models never contain inference.** One seam turns any twin into
   `f(params) -> prediction`, and a `ParameterSpace` re-parameterizes without
   editing an operator.
6. **Interfaces first, physics second.** A placeholder body may ship; its
   contract may not be one. Real physics replaces bodies, never interfaces.
7. **Loud failure over silent wrongness.** Chasing 0.1 % systematics, a wrong
   number is worse than an exception.
8. **The core is domain-agnostic.** `rheplicant.core` never imports the radio
   layer, and a test enforces it.

Each is explained in
**[the documentation](https://rheplicant.readthedocs.io/en/latest/)**;
[Status](#status) says which operators are still placeholders.

## Install

```bash
# limTOD (the sky engines) and bayesmith (the inference arithmetic) are
# dependencies, not extras, and both are on PyPI, so they come with the install.
pip install rheplicant

# the release this page describes, from its tag:
pip install "rheplicant @ git+https://github.com/RHINO-Experiment/rheplicant@v0.9.1"

# the noise-wave model is not on PyPI; it installs from git:
pip install "rhino-cal-jax @ git+https://github.com/RHINO-Experiment/rhino-cal@feat/rhino-cal-jax"

# or, for development:
git clone https://github.com/RHINO-Experiment/rheplicant
cd rheplicant
uv venv                          # NOT `uv sync`, which cannot work here
uv pip install -e . --group dev
```

**Which version `pip install rheplicant` gives.** As of 2026-10-02 the latest
upload on PyPI is 0.2.0, which has no `rheplicant` command, no configuration
layer and no browser editor. The second line above installs 0.9.1 from its
tag, and so does the development install. Both resolve every dependency from
PyPI, including `bayesmith>=0.10,<0.11`; [the bayesmith page](https://rheplicant.readthedocs.io/en/latest/bayesmith.html)
says what that range is for.

Requires Python ≥ 3.11, `jax ≥ 0.5`, `equinox ≥ 0.13`. Distribution and import
name are the same: `rheplicant`. Full instructions, the optional integrations
and the two-session test split are on the
[install page](https://rheplicant.readthedocs.io/en/latest/install.html).

Validate and run a configuration through the JAX-safe command boundary:

```bash
rheplicant validate observation.yaml
rheplicant run observation.yaml
rheplicant script observation.yaml -o run-observation.py
```

The CLI validates every variant and every run parser before execution, embeds
exact source/preset bytes in generated programs, and publishes a recoverable
input/resolved/provenance/diagnostics audit tree with optional deterministic
scientific products, reports, and a hashed `products.json` manifest. See the
[configuration CLI](https://rheplicant.readthedocs.io/en/latest/config-cli.html)
for stdin, output, clobber, and trusted-plugin rules.

Edit the same YAML-as-truth document in the optional browser workbench:

```bash
pip install "rheplicant[gui]"
rheplicant-gui                    # http://127.0.0.1:8000/
```

Model, Config, Execute and Results are four responsive views over the exact
accepted YAML. Safe control edits return complete YAML through Python;
unsubmitted raw YAML/field drafts remain browser view state. Quick and Full
validation, progressive output setup, explicit actions, automatic job polling,
current/stale results, Re-run and identity-checked audit links share that one
revision/digest boundary.

The launcher serves packaged React assets and the FastAPI boundary from one
origin and binds to loopback by default. It has no authentication, tenant
isolation or sandbox and refuses a non-loopback bind unless `--allow-remote`
is given together with at least one `--allowed-host NAME`. It answers only to
loopback host names and the listed names, and refuses a state-changing request
whose `Origin` is not its own, so a DNS-rebinding page under any other name
cannot drive it. Those flags are acknowledgement, not protection: plugins,
`python:` targets, server paths and jobs retain the server account's authority.
Read the
[GUI security and trust boundaries](https://rheplicant.readthedocs.io/en/latest/config-gui.html)
before exposing it beyond the local machine or executing a document received
from someone else.

## Seeing it work

The snippet above shows the pattern: provide operators, let the graph
compose them, call the result. What that `twin` then plugs into (gradients,
NUTS posteriors, Fisher forecasts, exact conjugate draws, neural surrogates)
is one worked example carried end to end in
**[the guided tour](https://rheplicant.readthedocs.io/en/latest/tour.html)**,
and fifteen runnable scripts with measured wall clocks in
[`examples/`](https://github.com/RHINO-Experiment/rheplicant/tree/main/examples),
beside one configured example,
[`examples/global21cm/`](https://github.com/RHINO-Experiment/rheplicant/tree/main/examples/global21cm).

## What is in the box

| Layer | What lives there |
|---|---|
| **Core** | `State`, the three combinators, `SignalGraph` + `assemble`, and `Assembly.replace_node` / `.without` to swap or drop a stage by node id. Domain-agnostic: a test enforces the layering. |
| **Radio** | A 33-node canonical signal path for a single-antenna experiment, a modular sky engine (a differentiable limTOD port plus a drift-scan m-mode fast path agreeing with it to roundoff), and linear analysis filters. |
| **Inference** | One noise model read by the likelihood, the weights, the Fisher matrix and the NumPyro scale alike; gradient and conjugate engines; `SamplingPlan` to partition a model into blocks whose engine is derived rather than restated; streaming evidence for campaigns whose data is gone. |

Per-operator detail is the
[operator catalog](https://rheplicant.readthedocs.io/en/latest/operators.html);
every signature is the
[API reference](https://rheplicant.readthedocs.io/en/latest/api.html).

## Documentation

**[rheplicant.readthedocs.io](https://rheplicant.readthedocs.io)** holds the
guided tour, the operator catalog, the inference rules, the tutorials, the API,
the architecture decisions and the changelog, with a sidebar that lists them.
Build it locally with
`cd docs && ../.venv/bin/python -m sphinx -n -b html . _build/html`.

Design decisions D1–D54 and the physics roadmap are in
[`DESIGN.md`](https://rheplicant.readthedocs.io/en/latest/design.html); what
arrived when is in
[`CHANGELOG.md`](https://rheplicant.readthedocs.io/en/latest/changelog.html).

## Status

The architecture and inference layer are complete and tested end-to-end
(12200 tests, 90.5 % coverage, jit+grad+vmap through the full twin; assembly
is regression-tested bitwise against hand-built composition). Radio operator
physics is a placeholder where the docstring says so (17 of the
29 concrete `rheplicant.radio` operator classes), pending ports from limTOD
and friends. The other twelve do not carry that wording: the sky engines
are real (a general differentiable limTOD port and a drift-scan m-mode fast
path that agrees with it to float64 roundoff while running ~1000x faster on
RHINO's geometry; see
[sky engines](https://rheplicant.readthedocs.io/en/latest/sky-engines.html)),
and so are the horizon split, the horn's ohmic loss, the noise-wave reflection
terms of the noise-wave data model, the CW calibration tone, and the
separable-basis antenna temperature. The Touchstone and RHINO-HDF5 readers are
a real ingestion layer.

Three of the seventeen are load-bearing even so. `ReceiverOperator`,
`GainOperator` and `CalLoadOperator` have placeholder bodies (no flicker,
no measured band shape, no load reflection or telemetry) but real shape and
contract: the receiver module's `unit_mean_bandpass` / `unit_mean_free` are the
bandpass/gain identifiability convention, the gain's exact linearity in `gain`
is what `Latent(..., linear=True)` claims about it, and
`CWCalibrationOperator.must_precede == ('bandpass', 'gain')` names the two
nodes the first two occupy. Conventions:
degrees in public APIs, radians internally; strings in `meta` (static),
numbers in `coords`/`env`/`aux` (traced); one seed reproduces a run.

CI runs the suite on every push to main and on pull requests, and measures
coverage in a separate serial job
([`.github/workflows/test.yml`](https://github.com/RHINO-Experiment/rheplicant/blob/main/.github/workflows/test.yml)).
It prints what the environment collects rather than asserting on it, because a
public runner cannot hold the `RHEPLICANT_RHINO_*` datasets and so collects
fewer tests than a complete machine. The suite is three pytest sessions: the
main one, and two that need float64 (`tests/evidence` and `tests/seam`), each
run as a subprocess by its own driver because `jax_enable_x64` is
process-global and tests in the main session assert refusals that only
float32 forces. Plain `pytest` runs all three. The float64 sessions run with
`--no-cov`, so the modules only they exercise show as uncovered in the default
report. The
[install page](https://rheplicant.readthedocs.io/en/latest/install.html#running-the-tests)
has the commands.

Neither `uv sync` nor `uv run` works here, with or without `--frozen`: locking
resolves every declared extra and two of them name packages that are not on
PyPI by design, so no lockfile exists or can be made. Use `uv venv` plus
`uv pip install`, and call the venv's interpreter directly. Two optional
datasets that cannot be published (the CST beam exports and a `rhino-cal`
checkout) are named by `RHEPLICANT_RHINO_BEAMS` and `RHEPLICANT_RHINO_CAL`,
with no default path; without them the work that needs them stands down and
says so. Commands, reasons and the rest of the setup are in
**[Install](https://rheplicant.readthedocs.io/en/latest/install.html)**.

## Developers and maintainers

- Zheng Zhang
- Phil Bull
- Jordan Norris
- Rashi Srivastava

## License

MIT
