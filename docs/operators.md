# Operator catalog

Every shipped operator, its home on the canonical single-antenna graph, and
its differentiable parameters. Placeholder physics is marked *(P)* — the
contract is real and tested; the docstring of each class records the physics
that will replace the body. Graph topology and assembly rules: see
[the tour](tour.md#4-graph-assembly) and `rheplicant/radio/graph.py`.

## Sky (astrophysical sources → `astro_sum`)

| Operator | Node | Role | Differentiable parameters |
|---|---|---|---|
| `GlobalSignalOperator` *(P)* | `global_signal` | 21 cm Gaussian absorption trough, constant in time | `depth`, `centre`, `width` |
| `ForegroundOperator` *(P)* | `foregrounds` (multi-instance) | power-law diffuse foreground | `amplitude`, `spectral_index` |
| `PointSourceOperator` *(P)* | `point_sources` | beam-diluted point-source level | `level` |
| `SkyOperator` *(P)* | `uniform_sky` | uniform sky (simplest source) | `amplitude` |

## Modular sky engine (→ `observed_astro_sky`, post-beam)

| Component | Kind | Role |
|---|---|---|
| `SkySourceOperator` | operator | composes `sky_model × projector`; output is already beam-convolved |
| `UniformSkyModel`, `PowerLawSkyModel` *(P)* | `AbstractSkyModel` | parameters → `(n_freq, n_pix)` maps |
| `MatrixProjector` | `AbstractSkyProjector` | precomputed sky→TOD matrix (limTOD `generate_sky2sys_projection`); differentiable, exact adjoint |
| `GeneralPointingProjector` | `AbstractSkyProjector` | pure-JAX (`limtod_jax`): any pointing, one rotation per sample, differentiable in sky and beam, exact adjoint |
| `DriftScanProjector` | `AbstractSkyProjector` | m-mode fast path for drift scans (`limtod_jax.driftscan`): one Wigner rotation per scan (cacheable via `to_reference_frame()`), optional FFT synthesis on uniform LST grids, optional horizon mask, exact adjoint |

## Environment

| Operator | Node | Role | Differentiable parameters |
|---|---|---|---|
| `IonosphereOperator` *(P)* | `ionosphere` | chromatic ~ν⁻² distortion of the astro sum | `delta` |
| `RFIOperator` *(P)* | `rfi_field` | sparse random spikes (pre-beam field), PRNG-driven | `amplitude` |
| `GroundPickupOperator` *(P)* | `ground_pickup` | effective ground-spill temperature, coupled to `env.temperature` | `coupling`, `t_ground` |
| `AtmosphericEmissionOperator` *(P)* | `atmosphere` | beam-averaged atmospheric emission (`t_ant_sum` branch) | `t_atm` |
| — | `ground_field` | *reserved leaf*: ground as a pre-beam field to convolve | — |
| — | `atmosphere_field` | *reserved transform*: radiative transfer on the astro sky, pre-beam | — |
| — | `t_sys_extra` | *reserved leaf (multi-instance)*: generic effective T_sys entry | — |

## Instrument (trunk order = graph order)

| Operator | Node | Role | Differentiable parameters |
|---|---|---|---|
| `BeamOperator` *(P)* | `beam` | shared chromatic beam — the single marginalisation target | `solid_angle` |
| `AntennaLossOperator` | `antenna_loss` | antenna ohmic loss: `η T + (1−η) T_phys`, on the whole `t_ant_sum`, before the switch | `efficiency`, `t_physical` |
| `CalLoadOperator` *(P)* | `cal_loads` | switched calibration load (via `receiver_input` selector) | `t_load` |
| `NoiseWaveOperator` | `noise_wave` | full Eq. 1 (noise-wave GCR note) via `rhino_cal_jax`: reflection couplings + noise-wave temperatures, linear in `t_nw = (t_unc, t_cos, t_sin)` — the GCR structure | `t_unc`, `t_cos`, `t_sin`, `t_rx`, `gamma_src_re`, `gamma_src_im`, `gamma_rec_re`, `gamma_rec_im` |
| `CWCalibrationOperator` *(P)* | `cw_tone` | CW tone injected before bandpass/gain (tracks gain drift) | `amplitude` |
| `ReceiverOperator` *(P)* | `bandpass` | frequency-dependent bandpass | `bandpass` |
| `GainOperator` *(P)* | `gain` | multiplicative gain, scalar or per-time | `gain` |
| `NoiseOperator` *(P)* | `noise` | post-gain thermal noise (PRNG protocol) | `sigma` |
| `EMIOperator` *(P)* | `emi` | self-generated EMI frequency comb | `amplitude` |
| `ADCOperator` *(P)* | `adc` | scale + clip digitisation | `scale` |
| `NeuralOperator` | *(explicit `At(...)`)* | learned positive spectral response `exp(MLP(freq))` — hybrid physics+ML | MLP weights |

`NoiseWaveOperator` requires the optional `rhino_cal_jax` package — see
`noise_wave.py`'s import guard for the install command, since it is not yet on
PyPI. It carries `Γ` **per source** (`gamma_src_re`/`gamma_src_im`, shape
`(n_source, n_freq)`) rather than one `Γ` for the whole TOD, and reads which
source is connected sample-by-sample from
`coords.extra["receiver_input"]`; with more than one source that array is
required, since defaulting to the first would return a finite,
correctly-shaped, wrong answer. See `examples/noise_wave_gcr.py` for the model
exercised as a checked linear block (Wiener mean and exact GCR draws) and D15
in `DESIGN.md` for why the per-source placement is what makes per-channel
noise-wave temperatures identifiable at all.

**The sky as `T_src`.** On the antenna branch `T_src` *is* the beam-convolved
sky, so a `SkySourceOperator` upstream feeds the receiver directly. The full
walkthrough is [From the sky to the receiver](sky-to-receiver.md); the three
joins that carry no structural guard, in short:

- the projector must return a **temperature**. Both sky engines default to
  `normalize_beam=False` (numpy limTOD's convention), which returns `∫BT`, not
  `∫BT/∫B`. Pass `normalize_beam=True` when the output feeds `T_src`. A beam
  normalized by hand to unit pixel sum is *still* biased — 0.2 % at
  nside 16 / lmax 47, ~4 % at nside 8 with a 20° beam — because the band-limit
  truncates the denominator as well.
- `gamma_src`'s **row order must match the selector's branch order**, which is
  the graph's in-edge declaration (antenna, then `cal_loads`). Both objects are
  `(n_source, n_freq)`, so swapping them is shape-legal; it moves the answer by
  tens of kelvin. Read the order off the assembly:
  `twin["receiver_input"].names`.
- a hand-wired antenna branch needs `SumOperator`, not `Pipeline`: a `Pipeline`
  of *source-type* operators replaces the data at each stage, so only the last
  source survives. Check any hand-wiring against `assemble()`.

An out-of-range switch value used to be a fourth: the eager range check in
`SwitchCycle` is skipped under tracing, and JAX's gather would clamp the coupling
lookup to a neighbouring source while the selector selected nothing.
`SwitchCycle.gather` now fills those samples with NaN instead, so the two
consumers of the switch array can no longer disagree in silence.

Multi-load switching (three sources — the minimum for an identifiable
per-channel fit) still bypasses `assemble()`, since the `cal_loads` node has no
`many=True`; build the `SelectOperator` directly, as the example does.

`AntennaLossOperator` is a *different* loss from `NoiseWaveOperator`'s
`c_s = (1−|Γ|²)|F|²`: ohmic dissipation inside the antenna versus impedance
mismatch at the receiver input. They multiply, and only the ohmic one adds
emission of its own (`(1−η) T_phys`). The node sits on the trunk between
`t_ant_sum` and `receiver_input`, so it acts on everything the beam collected
and on nothing that connects downstream — the calibration loads arrive
unattenuated. See D16 in `DESIGN.md`.

## Beam data

| Function | Role |
|---|---|
| `read_cst_farfield` | one CST Studio far-field ASCII export → `(theta_deg, phi_deg, directivity)` on its regular grid, dBi converted to linear power |
| `cst_frequency_table` | a directory of per-frequency exports → `{frequency [Hz]: path}` |
| `cst_beam_maps` | those exports → `(n_freq, npix)` HEALPix beam maps in limTOD's beam-local convention, linearly interpolated in frequency (extrapolation refused) |

Nothing is normalized on the way out: pass `normalize_beam=True` to the
projector and let it divide by its own quadrature, which is the only way the
band-limit cancels exactly. CST azimuth is measured from the model's `+x` axis,
and which physical direction that is is a fact about the as-built horn rather
than about the file — `phi0_deg` and `phi_sense` expose the offset and the
handedness, and their defaults are an assumption to check, not a result. Needs
`healpy` and `scipy`, both already required by `limTOD`.

## Processing segment

| Operator | Node | Role | Notes |
|---|---|---|---|
| `FlaggingOperator` *(P)* | `flagging` | threshold mask → `aux["flags"]` | data untouched |
| `MomentRFIFlaggingOperator` | `flagging` | MomentRFI flagger via `pure_callback` | prior flags compose |
| `BackendOperator` *(P)* | `averaging` | time-chunk integration; updates `coords.time` | shape-changing |
| `ApplyCalibrationOperator` *(P)* | `apply_cal` | apply a gain solution (`data / gain`) | inference → analysis bridge |
| `SiderealFilter` | `filters` (multi-instance) | day-repeating (sky-locked) subspace | `mode` extract/remove |
| `SkySpaceFilter` | `filters` | CG map-make/re-project through any linear projector | flags-weighted; `regularization` differentiable |
| `FourierBandFilter` | `filters` | fringe-rate (`axis=0`) / delay (`axis=1`) band | `mode` extract/remove |

## Core combinators & utilities

| Component | Role |
|---|---|
| `Pipeline` | sequential composition; `replace_stage`, `run_with_intermediates`, name access |
| `SumOperator` | parallel additive; branches are sources (data stripped, per-branch subkeys); `replace_branch` |
| `SelectOperator` | per-time-sample branch selection via `coords.extra[switch_key]` |
| `LambdaOperator` | wrap a pure function (`on_data` lifts array→array) |
| `SnapshotOperator` | zero-copy raw-data snapshot into `aux` |
| `Assembly` | graph-assembled operator: node-id access, `replace_node`, `to_mermaid`/`to_html`/`to_svg` |

## Inference layer

| Component | Role |
|---|---|
| `Latent`, `Bind`, `ParameterSpace` | what is inferred, and how it reaches the model — named, validated, re-parameterizable |
| `ParameterSpace.forward_fn` | the seam over NAMED parameters: `f(dict) -> prediction` |
| `build_forward_fn` | the seam over a whole subtree: `f(params) -> prediction` (filter_spec selects trainables) |
| `GradientCalibrator` / `AdamCalibrator` | fixed-step GD / Adam (pure JAX), `lax.scan`-driven |
| `GaussianLikelihood` / `MaskedGaussianLikelihood` | (masked) independent Gaussian log-density |
| `to_numpyro_model` | Bayesian bridge; sample sites named by their latents |
| `predict_from_samples` | posterior predictive over MCMC samples |
| `fisher_information`, `parameter_covariance` | Fisher matrix (exact Jacobians), Cramér-Rao — provenance-tagged (`FlatMatrix`), rows named (`cov.sigma("fwhm")`) |
| `check_linearity`, `linear_operator`, `wiener_solve` | verify a `linear=True` claim, export `A`/`Aᵀ` without forming a matrix, solve in closed form |
| `propagate_covariance`, `push_forward` | delta-method prediction bands; Monte Carlo pushforward |
