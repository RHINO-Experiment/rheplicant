# Operator catalog

Every shipped operator, its home on the canonical single-antenna graph, and
its differentiable parameters. Placeholder physics is marked *(P)* — the
contract is real and tested; the docstring of each class records the physics
that will replace the body. Graph topology and assembly rules: see
[the tour](tour.md#graph-assembly) and `rheplicant/radio/graph.py`.

- [Sky (astrophysical sources → `astro_sum`)](#sky-astrophysical-sources--astro_sum)
- [Modular sky engine (→ `observed_astro_sky`, post-beam)](#modular-sky-engine--observed_astro_sky-post-beam)
- [Environment](#environment)
- [Instrument (trunk order = graph order)](#instrument-trunk-order--graph-order)
- [Beam data](#beam-data)
- [Processing segment](#processing-segment)
- [Core combinators & utilities](#core-combinators--utilities)
- [Inference layer](#inference-layer)

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
| `BasisTemperatureOperator` | `t_sys_extra` (multi-instance) | effective T_sys smooth in (time, frequency) by construction: `time_basis @ coeff @ freq_basis.T`. Parameterized by COEFFICIENTS, not cells — which is what makes the CW tone worth anything at all (see `rheplicant.core.basis`) | `coeff` |

## Instrument (trunk order = graph order)

| Operator | Node | Role | Differentiable parameters |
|---|---|---|---|
| `BeamSpillOperator` | `beam_spill` | horizon split of the astro branch: `f_sky T + (1−f_sky) T_ground` | `sky_fraction`, `t_ground` |
| `AntennaLossOperator` | `antenna_loss` | antenna ohmic loss: `η T + (1−η) T_phys`, on the whole `t_ant_sum`, before the switch | `efficiency`, `t_physical` |
| `CalLoadOperator` *(P)* | `cal_loads` | switched calibration load (via `receiver_input` selector) | `t_load` |
| `NoiseWaveOperator` | `noise_wave` | the full noise-wave system temperature via `rhino_cal_jax`: reflection couplings + noise-wave temperatures, linear in `t_nw = (t_unc, t_cos, t_sin)` — the GCR structure | `t_unc`, `t_cos`, `t_sin`, `t_rx`, `gamma_src_re`, `gamma_src_im`, `gamma_rec_re`, `gamma_rec_im` |
| `CWCalibrationOperator` | `cw_tone` | CW tone with a lineshape (`sinc2` / `gaussian`), a width, and linear frequency and level drift; injected before bandpass/gain (`must_precede`, enforced by `assemble`); protects every channel it wets, as a waterfall mask when it drifts | — (all settings are KNOWN static configuration) |
| `ReceiverOperator` *(P)* | `bandpass` | frequency-dependent bandpass — declare it with `unit_mean_bandpass` when the gain is free too | `bandpass` |
| `GainOperator` *(P)* | `gain` | multiplicative gain, scalar or per-time; carries the absolute level by convention | `gain` |
| `NoiseOperator` *(P)* | `noise` | post-gain thermal noise (PRNG protocol) | `sigma` |
| `EMIOperator` *(P)* | `emi` | self-generated EMI frequency comb | `amplitude` |
| `ADCOperator` *(P)* | `adc` | scale + clip digitisation | `scale` |
| `NeuralOperator` | *(explicit `At(...)`)* | learned positive spectral response `exp(MLP(freq))` — hybrid physics+ML | MLP weights |

`AntennaLossOperator` sits on the trunk between `t_ant_sum` and
`receiver_input`, so it acts on everything the beam collected and on nothing
that connects downstream — the calibration loads arrive unattenuated.
`beam_spill` instead sits on the ASTRO branch
(`beam | observed_astro_sky → astro_ant_sum → beam_spill → t_ant_sum`), because
the split applies to the thing that genuinely is a beam integral over the
celestial sphere and not to the effective temperatures that join at `t_ant_sum`.
See [D16](design.md) and [D17](design.md).

It is a *different* loss from `NoiseWaveOperator`'s `c_s = (1−|Γ|²)|F|²` —
ohmic dissipation inside the antenna versus impedance mismatch at the receiver
input. They multiply, and only the ohmic one emits `(1−η) T_phys`. The two are
worked against `BeamSpillOperator`'s mixing-without-loss in
[Step 3](sky-to-receiver.md#step-3--three-effects-none-standing-in-for-another).

Get `f_sky` from `DriftScanProjector.horizon_fraction()`, or let
`BeamSpillOperator.from_projector(projector, t_ground=...)` read it off the same
beam that supplies the sky — that is the one call the weight and the sky average
cannot get out of step. `BeamSpillOperator` already supplies the below-horizon
ground term, so a `GroundPickupOperator` alongside it is a *second*, additional
one.

### The noise-wave model, and what it needs from the graph

`NoiseWaveOperator` requires the optional `rhino_cal_jax` package — see
`noise_wave.py`'s import guard for the install command, since it is not yet on
PyPI. It carries `Γ` **per source** (`gamma_src_re`/`gamma_src_im`, shape
`(n_source, n_freq)`) rather than one `Γ` for the whole TOD, and reads which
source is connected sample-by-sample from
`coords.extra["receiver_input"]`; with more than one source that array is
required, since defaulting to the first would return a finite,
correctly-shaped, wrong answer. See `examples/noise_wave_gcr.py` for the model
exercised as a checked linear block (Wiener mean and exact GCR draws) and [D15](design.md)
in `DESIGN.md` for why the per-source placement is what makes per-channel
noise-wave temperatures identifiable at all.

Read the row order of `gamma_src` off `twin["receiver_input"].names`. The
selector's branch order and the order the loads were provided are independent,
and a transposition is shape-legal — it is Join 2 of
[From the sky to the receiver](sky-to-receiver.md#the-one-identification),
measured there at 46 K peak on a 545 K signal.

An out-of-range switch value used to be a fourth: the eager range check in
`SwitchCycle` is skipped under tracing, and JAX's gather would clamp the coupling
lookup to a neighbouring source while the selector selected nothing.
`SwitchCycle.gather` now fills those samples with NaN instead, so the two
consumers of the switch array can no longer disagree in silence.

### `line_width` is boxed on both sides, and the window is narrow

The CW tone has two hard limits, and both used to be stated nowhere but the
source comments beside the constants. This is the first; the axis it drifts
along is the second, below.

`line_width` is boxed on both sides, as a multiple of the channel spacing.
Below `MIN_WIDTH_IN_CHANNELS` (1 channel for `"sinc2"`, 0.25 for `"gaussian"`)
the sampled channels land on the lineshape's own nulls or overflow its
exponent. Above `MAX_WIDTH_IN_BAND_FRACTION` (0.25 of the band, floored at
`MIN_CEILING_IN_CHANNELS` = 2 channel spacings so a coarse grid cannot call a
critically-sampled line "too wide") the line stops being a line and becomes a
pedestal across the whole band — every channel then sits above
`protect_floor` of the peak, the protection mask covers the band, and the RFI
flagger is switched off for the entire run: genuine RFI surviving into the
data. Both refuse by name, and the legal window between them is narrow on a
realistic band: an 8-channel 50–100 MHz grid (7.14 MHz channel spacing) only admits
`line_width` in about `[7.14e6, 1.43e7]` Hz — a factor of two. Remedy: pick
`line_width` from the spectrometer's own channel response, inside that window
for the band in hand; there is no default because guessing it silently
mis-sizes the protection mask, which is the one thing this operator exists to
avoid.

### `coords.time` is checked where it is stored, and twice

`coords.time` goes through `jnp.asarray`, float32 unless x64 is on, and a
unix-second axis (~1.75e9) has ~128 s of float32 resolution — so at RHINO's
~100 s cadence two samples land on the same stored value before any operator
runs. That is a property of how the axis is STORED, not of what the tone does
with it, so `Coordinates.__check_init__` refuses such an axis at construction
(`MAX_TIME_RESOLUTION_IN_SAMPLES` = 1e-2 of the smallest **distinct** gap). The
refusal quotes the two resolutions and is reproduced, with the reading of it, in
[ingestion](ingestion.md#coordstime-is-relative) — which is also where you find
why a freshly ingested RHINO recording no longer produces such an axis.

The tone keeps a second, stricter check of its own: the smallest gap
**including zero**. The container cannot tell a genuinely repeated timestamp
from a collision and has no business refusing the first; this operator can,
because it subtracts times, so two samples sharing an elapsed value means the
tone silently stops drifting across them — which is precisely its named failure.

**The guard is unit-agnostic, which means MJD is not exempt.** It compares
stored resolution against the axis's own smallest distinct gap, so it judges a
cadence rather than a convention: MJD 60000 at daily samples is accepted, and
MJD 60000 at a 100 s cadence is refused for exactly the same reason unix
seconds are — the float32 grid there is 3.9e-3 d, or 337 s.

Making the axis relative buys about five decimal orders, not unlimited range.
A float32 relative axis carries of order 1e5 uniform samples (exactly
2¹⁷ = 131072 at 1 s cadence; the exact count depends on where `n · cadence`
falls inside its binade). A four-hour run at 1 s is 1.4e4 samples and an order
of magnitude clear; the same run at 0.05 s is 2.9e5 and is refused. **That** is
when to reach for `JAX_ENABLE_X64=1` — a long or fast run, not merely an
absolute epoch. A static tone (`drift_rate = amplitude_drift_rate = 0.0`)
never reads `coords.time` at all, but the container check still applies,
because the axis is wrong for every other consumer too — `BackendOperator`'s
chunk timestamps were measured off by up to 78 s out of a 100 s cadence.

### A `many` node with two instances loses its bare id

**Multi-load switching comes out of `assemble()`.** `cal_loads` is `many=True`
and feeds only the `receiver_input` selector, so its instances compose the way
that consumer composes — one switch position each, not a sum:

```python
twin = assemble(SkySourceOperator(...), CalLoadOperator(t_load=...),
                CalLoadOperator(t_load=...), NoiseWaveOperator(...))
twin["receiver_input"].names   # ('observed_astro_sky', 'cal_loads_1', 'cal_loads_2')
```

A switching cycle of any length comes out of `assemble()` this way, so nothing
about the configuration needs hand-wiring. How long it has to be for an
identifiable fit is `NoiseWaveOperator`'s module docstring to say, and is worked
with the rank arithmetic in
[Step 4](sky-to-receiver.md#step-4--a-real-switching-cycle).

Reach any parameter by its graph node, wherever the fold put it:
`eqx.tree_at(lambda t: t["observed_astro_sky"].sky_model.maps, twin, new_maps)`.

**A `many` node with several instances is addressed per instance.** One
instance keeps the bare node id — `twin["cal_loads"]`, and every space written
against it — but the moment a second operator is placed there the instances
become `cal_loads_1`, `cal_loads_2`, … and the bare id raises
`AmbiguousNodeError` naming them. It has to: the bare id would otherwise
resolve to the fold over the instances, so `replace_node("cal_loads", ...)`
would overwrite *both* loads with one operator — the same forward shape, one
component of the instrument gone. `assembly.instances` reports the multiplicity,
and the signal-path renderings label such a node `(x2)`.


## Beam data

**Every one of these is a pass-through.** What a beam IS, how a measured one is
read, how it weights the sky and where the horizon falls in it all belong to
limTOD ([D20](design.md), [D25](design.md)). What stays here is *placement* — `BeamSpillOperator`
consumes `f_sky` but does not compute it — and the seam's own arguments.

| Function | Delegates to | What this side adds |
|---|---|---|
| `read_cst_farfield` | `limTOD.cstbeam` | nothing but the name |
| `cst_frequency_table` | `limTOD.cstbeam` | keys in **Hz**, not limTOD's MHz |
| `cst_beam_maps` | `limTOD.cstbeam` | `freq_hz` in **Hz**, to match `Coordinates.freq` |
| `horizon_truncated_beam` | `limtod_jax` | `nside` inferred from maps that carry it |

Frequencies cross that boundary in Hz because that is what `Coordinates.freq`
carries; limTOD is in MHz throughout, as it is everywhere in that package. The
conversion is the adapter's whole contribution, so it is exactly what
rheplicant's tests check — the conventions themselves are locked upstream, in
`limTOD/tests/test_cstbeam.py`, rather than duplicated here.

Two things worth knowing before trusting a beam, both documented in full by
[`limTOD.cstbeam`](https://limtod.readthedocs.io/en/latest/cstbeam.html):
nothing is normalized on the way out (pass `normalize_beam=True` to the
projector and let it divide by its own quadrature, the only way the band limit
cancels exactly); and CST azimuth is measured from the model's `+x` axis, which
physical direction being a fact about the as-built horn rather than about the
file — `phi0_deg` and `phi_sense` expose the offset and the handedness, and
their defaults are **an assumption to check, not a result**. Needs `healpy` and
`scipy`, both already required by `limTOD`.

## Processing segment

| Operator | Node | Role | Notes |
|---|---|---|---|
| `FlaggingOperator` *(P)* | `flagging` | threshold mask → `aux["flags"]` | data untouched |
| `MomentRFIFlaggingOperator` | `flagging` | MomentRFI flagger via `pure_callback` | prior flags compose; `kernel_shapes` runs the broad rounds |
| `BackendOperator` *(P)* | `averaging` | time-chunk integration; updates `coords.time`, reduces `aux['flags']` and a 2-D `aux['protected']` over the chunk (`any`), and refuses an unknown per-time `aux` array rather than carrying it at the old length | shape-changing |
| `ApplyCalibrationOperator` *(P)* | `apply_cal` | apply a gain solution (`data / gain`) | inference → analysis bridge |
| `SiderealFilter` | `filters` (multi-instance) | day-repeating (sky-locked) subspace | `mode` extract/remove |
| `SkySpaceFilter` | `filters` | CG map-make/re-project through any linear projector | flags-weighted; `regularization` differentiable |
| `FourierBandFilter` | `filters` | fringe-rate (`axis=0`) / delay (`axis=1`) band | `mode` extract/remove |

### Flagging, and where the flags go

Flagging is a boolean decision, so it has no gradient — and that is why
`jax.pure_callback` into the numpy [MomentRFI](https://github.com/zzhang0123/MomentRFI)
package is the *permanent* integration rather than a stopgap. The operator is
jittable (and tested to give bit-identical flags under `jit`); it is not
vmappable or differentiable, by nature.

```python
op = MomentRFIFlaggingOperator(config={"sigma_threshold": 4.0},
                               kernel_shapes=((3, 3),))
flags = op(state).aux["flags"]
```

`kernel_shapes` runs MomentRFI's broad rounds, and they buy the matched
filter's $\sqrt{K}$ rather than a looser cut: a spatially continuous emitter
adds roughly linearly under a box kernel while thermal noise adds in
quadrature, so averaging $K$ pixels lifts it by $\sqrt{K}$. On a 3σ-per-pixel
blob under the fitter's default 4σ threshold, round 0 alone recovers **none**
of it and a single 3×3 box recovers **all** of it — the package's own test
asserts both numbers.

The flags then reach inference by **wrapping the noise model**, which is where
a sample that was not observed belongs:

```python
noise = FlaggedNoise(RadiometerNoise(channel_width, integration_time), flags)
```

That one object carries them into the likelihood, the Fisher matrix, the
weights of a Wiener solve or GCR draw, and a NumPyro observation scale — see
[the noise model](inference-linear.md#the-noise-model). The route is tested by the
bias it removes rather than by the mask it produces: a persistent narrow-band
emitter on 2 of 32 channels pulls a maximum-likelihood amplitude **+5.8 %**
high, and wrapping MomentRFI's own flags around the noise model recovers the
truth — matching, to six digits, what flagging the contaminated channels by
hand would have given.

Install: MomentRFI is not on PyPI, so the `rheplicant[rfi]` extra names the
requirement rather than resolving it (the same arrangement as `cal`).

```bash
pip install "MomentRFI @ git+https://github.com/zzhang0123/MomentRFI"
```

## Core combinators & utilities

The first three are the only ways to compose, and none of them is an operator —
they act *on* operators, which is why every rendering draws a cascade as an
**arrow**, a sum as an **⊕** on the wire and a switch as a **◇**, never as
another box. See [the canonical signal path](signal-path.md).

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
