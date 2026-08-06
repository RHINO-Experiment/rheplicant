# Examples

Thirteen runnable scripts in
[`examples/`](https://github.com/RHINO-Experiment/rheplicant/tree/main/examples).
Each prints its own results; none needs a real recording. Every wall clock below
was measured by running the script, on CPU.

```bash
uv run --frozen python examples/radio_digital_twin.py
```

:::{warning}
Two are slow enough to plan around — `driftscan_mmode.py` at **61 s** and
`sky_to_noise_wave.py` at **59 s**, both dominated by JIT compilation rather
than by the work — and `tutorial_nuts.py` runs NUTS twice for **185 s**. Every
other script finishes in under ten seconds.
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
    solved back out. Cross-checked against Eq. 1 written by hand
  - **59 s**
  - limTOD, `cal`
* - `driftscan_mmode.py`
  - Shows the m-mode engine reproduces the general one to 5e-15 — an
    optimisation, not an approximation — then times it at ~230× and
    differentiates through the beam
  - **61 s**
  - limTOD
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
  - A NUTS posterior checked against a Fisher forecast on the same model —
    the cross-check that says whether the forecast was honest
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
  - —
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
    [the gradient-posterior tutorial](tutorial-nuts.md) — `r_hat = 840` first
  - **185 s**
  - numpyro
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
    `SkySpaceFilter` map-making through the *same* projector's adjoint —
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

`—` is a default install. **limTOD** is a dependency rather than an extra, but it
is not on PyPI at the required floor, so it installs from source; **`cal`** is
`rhino-cal-jax`, also from git; **numpyro** is `pip install "rheplicant[numpyro]"`.
See [Install](install.md) for the commands.

Note that limTOD and `rhino_cal_jax` are imported **lazily** — importing
`rheplicant.radio` does not pull either. A script needs them only when it
actually calls a sky engine or constructs a `NoiseWaveOperator`.
