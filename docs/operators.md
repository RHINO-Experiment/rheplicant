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
| `MModeProjector` *(P)* | `AbstractSkyProjector` | drift-scan m-mode transfer; differentiable, exact adjoint |
| `LimTODProjector` | `AbstractSkyProjector` | numpy-limTOD oracle via `pure_callback` (not differentiable) |
| `NativeLimTODProjector` | `AbstractSkyProjector` | pure-JAX limTOD port (`limtod_jax`): general pointing, differentiable in sky and beam, exact adjoint |
| `DriftScanProjector` | `AbstractSkyProjector` | m-mode fast path for drift scans (`limtod_jax.driftscan`): one Wigner rotation per scan, optional horizon mask, exact adjoint |

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
| `CalLoadOperator` *(P)* | `cal_loads` | switched calibration load (via `receiver_input` selector) | `t_load` |
| `NoiseWaveOperator` *(P)* | `noise_wave` | reflection loss + noise-wave T terms (linear in `t_nw` — the GCR structure) | `t_unc`, `t_cos`, `t_sin`, `t_zero`, `gamma_re`, `gamma_im` |
| `CWCalibrationOperator` *(P)* | `cw_tone` | CW tone injected before bandpass/gain (tracks gain drift) | `amplitude` |
| `ReceiverOperator` *(P)* | `bandpass` | frequency-dependent bandpass | `bandpass` |
| `GainOperator` *(P)* | `gain` | multiplicative gain, scalar or per-time | `gain` |
| `NoiseOperator` *(P)* | `noise` | post-gain thermal noise (PRNG protocol) | `sigma` |
| `EMIOperator` *(P)* | `emi` | self-generated EMI frequency comb | `amplitude` |
| `ADCOperator` *(P)* | `adc` | scale + clip digitisation | `scale` |
| `NeuralOperator` | *(explicit `At(...)`)* | learned positive spectral response `exp(MLP(freq))` — hybrid physics+ML | MLP weights |

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
| `build_forward_fn` | the seam: twin → `f(params) -> prediction` (filter_spec selects trainables) |
| `GradientCalibrator` / `AdamCalibrator` | fixed-step GD / Adam (pure JAX), `lax.scan`-driven |
| `GaussianLikelihood` / `MaskedGaussianLikelihood` | (masked) independent Gaussian log-density |
| `to_numpyro_model`, `prior_template`, `set_prior` | Bayesian bridge with positional pytree priors, semantic site names |
| `predict_from_samples` | posterior predictive over MCMC samples |
| `fisher_information`, `parameter_covariance` | Fisher matrix (exact Jacobians), Cramér-Rao — provenance-tagged (`FlatMatrix`) |
| `propagate_covariance`, `push_forward` | delta-method prediction bands; Monte Carlo pushforward |
