# Examples

Fifteen runnable scripts in
[`examples/`](https://github.com/RHINO-Experiment/rheplicant/tree/main/examples),
and one configured example in a directory of its own, described at the end of
this page. Each script prints its own results; none needs a real recording.
Every wall clock below was measured by running the script, on CPU. The scripts
are in the repository and not in the wheel, so run them from a clone.

```bash
.venv/bin/python examples/radio_digital_twin.py
```

:::{warning}
Four scripts take 20 s or more. `tutorial_nuts.py` runs NUTS twice and takes
**185 s**. `driftscan_mmode.py` (**61 s**) and `sky_to_noise_wave.py`
(**59 s**) are dominated by JIT compilation. `three_ways_to_a_posterior.py`
takes 26 s. Ten finish in under ten seconds. `gibbs_plan.py` does not run to
its end at 0.9.1; its row says where it stops.
:::

## Forward modelling

:::{list-table}
:header-rows: 1
:widths: 26 44 14 16

* - Script
  - What it does
  - Time
  - Needs
* - `radio_digital_twin.py`
  - Hands 17 unordered operators to `assemble()` and lets the graph compose
    them, then drops the two stochastic stages and fits the gain back
    (1.000 → 1.1198 against a truth of 1.100)
  - 2.3 s
  - `cal`
* - `sky_to_noise_wave.py`
  - RHINO's horn end to end: CST beam → HEALPix → drift-scan `T_src` → horizon
    spill, ohmic loss, mismatch → a three-position switch cycle → noise waves
    solved back out. Cross-checked against the same sum written by hand. The
    CST exports are not redistributable: name a directory with `--beam-dir` or
    `RHEPLICANT_RHINO_BEAMS`, otherwise a Gaussian beam stands in
  - **59 s**
  - `cal`
* - `driftscan_mmode.py`
  - Shows the m-mode engine reproduces the general one to 5e-15, then times
    it at ~230× and differentiates through the beam
  - **61 s**
  - —
* - `diy_global_signal.py`
  - Replaces the placeholder global signal with the EDGES flattened Gaussian:
    `graph_node` is all `assemble()` needs, a document reaches it through
    `python:` rather than `type:`, and the two dimension registrations that
    route asks for are answered beside the refusal each one lifts
  - 1.4 s
  - —
:::

## Inference

:::{list-table}
:header-rows: 1
:widths: 26 44 14 16

* - Script
  - What it does
  - Time
  - Needs
* - `inferring_anything.py`
  - One pipeline, three parameter spaces: two scalars driving 6144 matrix
    entries, a gain tied across two stages in log space, and a sky map declared
    linear and solved exactly. The pipeline's tree is never edited
  - 8.1 s
  - —
* - `three_ways_to_a_posterior.py`
  - The same gain posterior by exact solve, by NUTS and by neural posterior
    estimation, with each one's width and wall time side by side
  - 26 s
  - numpyro
* - `bayesian_and_uncertainty.py`
  - A NUTS posterior compared with a Fisher forecast on the same model, as a
    check on the forecast
  - 2.7 s
  - numpyro
* - `neural_surrogate.py`
  - An `eqx.nn.MLP` placed at the `bandpass` node with `At()`, trained through
    the ordinary seam, recovering a rippled bandpass to ~0.8 %
  - 1.6 s
  - —
* - `gls_gcr.py`
  - Why a frozen noise σ leaves the point estimate exactly unmoved but moves
    the error bars by −8 % to +8 %
  - 5.9 s
  - `cal`
* - `noise_wave_gcr.py`
  - The noise-wave model as a checked linear block: Wiener mean, exact GCR
    draws, and κ jumping from 27 to ~4e6 when one source is dropped
  - 6.1 s
  - `cal`
* - `tutorial_gcr.py`
  - The seven steps of [the exact-posterior tutorial](tutorial-gcr.md)
  - 4.6 s
  - —
* - `tutorial_nuts.py`
  - The failing-then-fixed NUTS run of
    [the gradient-posterior tutorial](tutorial-nuts.md), with `r_hat = 846`
    first
  - **185 s**
  - numpyro
* - `gibbs_plan.py`
  - One `SamplingPlan` over six latents, conjugate blocks and a gradient
    block, over the guided tour's twin and then over a seven-position repair
    of it. At 0.9.1 the script stops at its estimate over the repaired twin
    with `LinearityRefused`: the linearity check probes each latent at 1000
    times its prior width, and the twin's 12-bit ADC clips there
  - 9 s, to the refusal
  - `cal`, numpyro
:::

## Analysis and rendering

:::{list-table}
:header-rows: 1
:widths: 26 44 14 16

* - Script
  - What it does
  - Time
  - Needs
* - `sky_projection_and_filters.py`
  - A sidereal filter in both `extract` and `remove` modes, then a
    `SkySpaceFilter` map-making through the same projector's adjoint,
    recovering the sky component to ~0.06 %
  - 1.4 s
  - —
* - `render_signal_path.py`
  - Writes `signal_path.html` (12 KB, self-contained) showing the full template
    with the assembly's nodes lit, identity-traversed nodes half-lit, and the
    rest dimmed
  - 0.4 s
  - —
:::

## What "Needs" means

`—` is a default install. limTOD and bayesmith come with it: both are
dependencies rather than extras, and both are on PyPI. **`cal`** is
`rhino-cal-jax`, which is not on PyPI, so it installs from git; **numpyro** is
`pip install "rheplicant[numpyro]"`. See [Install](install.md) for the
commands.

limTOD and `rhino_cal_jax` are imported lazily: importing
`rheplicant.radio` does not pull either. A script needs them only when it
calls a sky engine or constructs a `NoiseWaveOperator`.

## A configured example: global 21 cm separation

[`examples/global21cm/`](https://github.com/RHINO-Experiment/rheplicant/tree/main/examples/global21cm)
is a directory, not a script. It fits two foreground strategies and an oracle
to one simulated drift scan and scores each recovered 21 cm curve against the
injected one. Everything is declared in six YAML documents (`oracle`,
`beamconv`, `physical` and a `_quick` form of each) that `rheplicant run`
executes; the Python beside them is the plugin and the hooks the documents
load, and the analysis of the published run trees.

Its [README](https://github.com/RHINO-Experiment/rheplicant/blob/main/examples/global21cm/README.md)
gives the commands, the time each step takes, and the results. It needs
`global21cm_jax`, from the 21cmVAE-jax repository, and `pygdsm`.

A fresh checkout cannot run the pipeline. The simulation step reads one data
file that is not public and RHINO beam files that are not redistributable, and
the simulated arrays the six documents read are not kept in git. What a fresh
checkout can do is read the kept analysis products under `results/analysis*/`
and run the example's own tests, which the repository's suite does not
collect.
