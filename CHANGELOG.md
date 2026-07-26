# Changelog

## Unreleased

- **Docs: [sky engines](https://rheplicant.readthedocs.io/en/latest/sky-engines.html)**
  — a new page comparing the general and drift-scan engines side by side,
  with figures generated from live code by `docs/_generate_engine_figures.py`
  (one sidereal day of GSM sky through a zenith beam at latitude 53.2°):
  the waterfall the twin produces, the engine-to-engine agreement, the
  wall-clock scaling, and the m-mode spectrum. Every figure ships in light
  and dark variants and is theme-switched by furo. Measured at nside 64
  (lmax 191), 512 samples, 32 channels: the engines agree to **1.4e-15**
  relative, and one forward evaluation costs **68.0 s** on the general
  engine against **61 ms** with the cached beam and FFT synthesis — a
  **1111x** speed-up. `sphinx-design` is a new docs dependency.
- `examples/driftscan_mmode.py`: the end-to-end drift-scan demo — build a
  twin from a beam map, cross-check it against the general engine, time all
  three configurations, read off the m-modes, differentiate w.r.t. the beam.
- `DriftScanProjector.from_beam_maps(...)`: build the projector from HEALPix
  beam **maps** rather than alms, with `nside` inferred from the map length.
  The analysis runs in JAX (`limtod_jax.map2alm_iter`), so gradients reach
  the beam map — and it removes a genuine footgun: the quadrature transform
  the *sky* uses (`map2alm_quad`, the one visible in `forward`) silently
  rescales a beam by `npix/4pi`, and nothing previously said which transform
  the beam wanted.
- `DriftScanProjector.uniform_lst_grid(n_time, lst0_deg)`: the LST grid
  `uniform_sampling=True` requires. The natural `jnp.linspace(0, 360,
  n_time)` includes the endpoint and is a turn *plus one step* — a mistake
  this package has already been bitten by — so the correct grid is now
  provided rather than described in an error message.
- `DriftScanProjector` now REJECTS coords whose `pointing` or
  `selfrot_deg` disagrees with its own fixed pointing. Those entries are
  configuration here, not data, so a scan handed to the drift engine used to
  be silently discarded and a different observation simulated: finite,
  correctly shaped, and wrong. Constant pointing that agrees still passes —
  reusing a `NativeLimTODProjector`'s coords is the expected way to switch
  engines — and traced values, which carry nothing to compare, are left
  alone. Uniform-grid violations are also re-raised as `StateValidationError`
  so the whole boundary is catchable as one family.
- Documented the m-mode formalism's source (*M-mode RIME explicit in beam,
  fringe and sky modes*) in the module and the docs; the last line of its
  Eq. (13) is the identity the fast path implements.
- Projector cross-references: the `projection` module ladder, the `adjoint`
  error message, `MatrixProjector`, and `NativeLimTODProjector` now all point
  at the drift-scan engine, and `rheplicant.radio.sky.driftscan` was added to
  the API reference — its ~200 lines of contract documentation were missing
  from the rendered docs entirely. The stale four-engine ladders in
  `docs/tour.md`, `README.md`, and `DESIGN.md` (D8), which offered the
  `MModeProjector` placeholder as the drift-scan answer, are corrected.
- `DriftScanProjector` (`rheplicant.radio.sky.driftscan`): the m-mode fast
  path for drift scans — the "real version" the `MModeProjector` placeholder
  promised. Derives the m-mode projection from the beam alms on the fly via
  `limtod_jax.driftscan` (one Wigner rotation for the whole scan plus per-m
  phases): equal to `NativeLimTODProjector` with constant pointing to float64
  roundoff at O(lmax^3 + n_time*lmax) instead of O(n_time*lmax^3). The drift
  pointing (az/el/selfrot) is static projector configuration; `coords` only
  supplies `lst_deg`. Ships the exact sky-slot adjoint, an `mmodes` accessor
  (per-frequency Fourier coefficients of the sidereal TOD), and the optional
  horizon mask with cosine apodization (see the limTOD ringing study).
  Requires limTOD >= 1.6 (`limtod_jax.driftscan`); guarded lazy import,
  with a second feature level for `uniform_sampling=True`, which needs
  the FFT fast path and public grid check added in limTOD 1.7 — an
  outdated install now fails at the boundary with a clear message
  instead of an AttributeError inside a traced call.
- `DriftScanProjector.to_reference_frame()` pays the O(lmax^3) Wigner
  rotation once and returns an equivalent projector (new static
  `beam_frame="reference"`) that skips it on every later
  forward/adjoint/mmodes — the difference between rotating once and
  rotating per likelihood evaluation (measured 73-878x fewer per-evaluation
  FLOPs depending on lmax and n_time). A configured horizon mask is folded
  into the cached alms and its flag cleared, so it can be neither lost nor
  applied twice. The precompute is itself pure JAX, so a caller who wants
  beam-LOCAL gradients can still differentiate through it; keeping the
  `"local"` projector remains the way to get gradients w.r.t. pointing.
- `DriftScanProjector(uniform_sampling=True)` routes the time synthesis and
  its adjoint through real FFTs (limTOD's new `uniform=` fast path):
  O(n_time*log n_time) independent of lmax, 19-51x faster than the direct
  phase sum, identical to roundoff. Static by design — dispatching on the
  values of the LST grid is impossible under jit. The projector validates
  the RAW `lst_deg` per call, which stays concrete even inside a jit trace
  (deriving `dphi` there would not), so a bad grid is a clear ValueError at
  trace time rather than limTOD's NaN-poisoned fallback.
- `DriftScanProjector` gained a `beam_ref_lst_deg` invariant: it records the
  LST the cached beam was actually rotated to and only `to_reference_frame()`
  sets it, so `dataclasses.replace(cached, lst_ref_deg=...)` — which would
  measure the phases from a reference the baked-in rotation does not
  correspond to, silently — now fails validation instead. The uniform-grid
  check also stopped upcasting the LST grid to float64 before validating:
  the cast hid the grid's real precision, and limTOD's dtype-scaled
  tolerance rejects a legitimate float32 grid when checked at the f64 bound.
- Test coverage strengthened after adversarial review: every cached-beam and
  FFT test now also runs with a reference LST far from `lst_deg[0]` (they
  were all degenerate at 12.0, so a "use lst[0] regardless" bug was
  invisible — mutation-verified as killed now), compared against the
  general `NativeLimTODProjector` as an independent oracle; and the
  `to_reference_frame` gradient is compared for equality with the uncached
  projector's rather than only checked finite.
- `DriftScanProjector.mmodes()` now rejects `normalize_beam=True`: the
  normalization divides by the ones-map denominator, which is not part of
  the m-mode expansion, so the returned coefficients would silently not be
  the spectrum of `forward()` (~18x off).

## 0.1.4 (2026-07-25)

- Add project logos (a rhino dissolving into digital pixels — the
  differentiable digital-twin motif): a banner heads the README and the docs
  landing page, and the single-rhino mark is the docs sidebar logo.

## 0.1.3 (2026-07-25)

- Repository moved to the [`RHINO-Experiment`](https://github.com/RHINO-Experiment)
  GitHub organization; package metadata URLs (Homepage, Repository, Changelog)
  now point there. Publishing continues via Trusted Publishing under the new
  owner. No code or API changes.

## 0.1.2 (2026-07-25)

- `__version__` is now read from the installed distribution metadata
  (`importlib.metadata.version`) instead of a hardcoded string, so
  `pyproject.toml` is the single source of truth for the version. Falls back
  to `0.0.0+unknown` when run from an uninstalled source tree.

## 0.1.1 (2026-07-25)

- Credit the developers and maintainers (Zheng Zhang, Phil Bull, Jordan
  Norris, Rashi Srivastava) in the package metadata (`authors`) and README.
  First release carrying the full author list on the PyPI page.

## 0.1.0 (2026-07-25)

First PyPI release (`pip install rheplicant`), published via Trusted
Publishing (OIDC). Highlights of the 0.1.0 development line:

### Renamed: REPLICANT -> RHEPLICANT (RHino + REPLICa + ANTenna)

Same portmanteau (REPLICa + ANTenna = a differentiable replica of a radio
antenna), now with the **RH** of RHINO in front — the horn antenna the
framework was first built for. Distribution *and* import name are now both
`rheplicant` (the bare name is free on PyPI, so the earlier `-telescope`
suffix is dropped); import path `replicant.*` -> `rheplicant.*`; source dir
`src/replicant` -> `src/rheplicant`. GitHub repo and RTD project renamed to
`rheplicant`; old URLs redirect.

### Renamed: DIRT -> REPLICANT (a portmanteau of REPLICa + ANTenna)

A digital twin *is* a replica, and this one is of a radio antenna — so the
package is now **REPLICANT** (`REPLIC`a ⊕ `ANT`enna, overlapping the shared
`A`). Distribution name: `replicant-telescope`; import name: `replicant`
(was `dirt` / `dirt-telescope`). A PyPA packaging sample owns the bare
`replicant` name on PyPI, hence the `-telescope` suffix. Old `dirt-telescope`
GitHub/RTD URLs redirect after the rename.

### Rendering: embeddable SVG + documented lit/dim examples

`Assembly.to_svg()` / `SignalGraph.to_svg()` return a self-contained
`<svg>` (opacity classes styled inside the figure), so lit/dim signal-path
renders embed anywhere a plain image does. The docs signal-path page now
shows two real example renders, generated from live assemblies at build
time.

### Graph v1.2: atmosphere as an equivalent-entry pair (D13)

The `atmosphere` node moved from a trunk transform (between `t_ant_sum` and
the receiver-input switch) to a **source leaf** of `t_ant_sum`, parallel to
`ground_pickup`/`t_sys_extra`: `SystemTemperatureOperator` (transform,
`t_sys`) is replaced by `AtmosphericEmissionOperator` (source, `t_atm`, in
`replicant.radio.environment`). A reserved `atmosphere_field` transform on the
astro branch (between `ionosphere` and `field_sum`) marks the strict
radiative-transfer entrance — opacity acts on the astro sky alone, never on
ground pickup. Numerically identical for the additive placeholder; see
DESIGN.md D13 for the rationale.

### Renamed: e-RHINO -> DIRT (Differentiable Instrument Response Twin)

The framework applies to any single-antenna radio telescope (horns, dipoles,
dishes), so the RHINO-specific name was retired. Distribution name:
`dirt-telescope`; import name: `dirt` (was `erhino`). The GitHub repository
moved to `zzhang0123/dirt-telescope` (old URLs redirect). The canonical graph
template is now named "single-antenna".

Initial architecture of the differentiable scientific pipeline framework.

### Inference layer completed (D12)

- **NumPyro bridge** (`to_numpyro_model` — the last stub is gone): pytree
  priors via `prior_template`/`set_prior`, semantic sample-site names from
  stage names, masked Gaussian likelihood (flags -> zero weight), optional
  noise-std inference, `predict_from_samples` posterior predictive.
- **Uncertainty propagation** (`dirt.inference.uncertainty`):
  `fisher_information` (exact Jacobians via jacfwd), `parameter_covariance`
  (Cramer-Rao), `propagate_covariance` (delta-method prediction bands),
  `push_forward` (Monte Carlo). Fisher matches NUTS posterior widths on the
  demo problem.
- **Neural surrogates**: `NeuralOperator` (eqx.nn.MLP as a positive spectral
  response) — hybrid physics+ML with zero special machinery; placed
  explicitly (e.g. `At("bandpass", ...)`). `AdamCalibrator` (pure JAX)
  added; it recovers a rippled bandpass to <1% where fixed-step GD diverges.
- Examples: `bayesian_and_uncertainty.py`, `neural_surrogate.py`.

### Graph-guided assembly (D11)

- `dirt.core.graph`: `SignalGraph` declarative signal-path templates
  (validated DAG, single sink, typed nodes) and `assemble` — compiles a set
  of operator instances into the induced `Pipeline`/`SumOperator` nesting
  (absent sources pruned, absent transforms skipped as identity, junctions
  materialized as sums; deterministic branch order = graph declaration
  order). Result is an `Assembly` operator with lit/skipped metadata,
  node-id access (`assembly["gain"]`, `replace_node`), caller-data guards,
  and lit/dim `to_mermaid` rendering.
- `dirt.radio.graph`: the canonical single-antenna graph (26 nodes) with
  equivalent-entry leaves (`observed_astro_sky` — served by
  `SkySourceOperator`; reserved placeholders `ground_field`, `t_sys_extra`)
  and `graph_node` slots on every radio operator;
  `assemble(*ops)` convenience. Full-set assembly is regression-tested
  bitwise against the hand-built twin.
- `SumOperator`: branch input data now stripped to `None` (D6 enforced);
  added `replace_branch`.
- **Selector nodes** (`SelectOperator` + the `"selector"` NodeSpec kind):
  switched signal paths — one branch selected per time sample via
  `coords.extra[<node_id>]`. The canonical graph gains `cal_loads`
  (`CalLoadOperator` placeholder) and the `receiver_input` antenna/load
  switch, modeling the elements taxonomy's switched calibration signals;
  pass-through (zero cost) when no load is provided.
- **Region coverage**: `graph_node`/`At` accept a tuple of node ids — one
  operator implementing a contiguous template path atomically (disjointness
  and interior-feed validation; addressed by its last covered node).
- **HTML rendering**: `SignalGraph.to_html()` / `Assembly.to_html()` produce
  a standalone lit/dim signal-path page (`examples/render_signal_path.py`).

### Integration seams (added after initial architecture)

- **Modular sky** (`dirt.radio.sky`): `AbstractSkyModel` (params → maps) ×
  `AbstractSkyProjector` (maps → TOD, with `adjoint` for linear engines),
  composed by `SkySourceOperator`. Engines: `MatrixProjector` (precomputed
  `generate_sky2sys_projection` matrix — differentiable today),
  `LimTODProjector` (pure_callback oracle into numpy limTOD),
  `MModeProjector` (m-mode transfer, drift scans). Port task book for the
  native JAX limTOD rewrite: `docs/limtod-port-contract.md`.
- **Native limTOD projector** (`NativeLimTODProjector`): the port contract
  delivered — pure-JAX sky→TOD chain (Wigner rotation + harmonic beam sum
  from the `limtod_jax` package in the limTOD repo), general pointing,
  jit/vmap-safe, differentiable w.r.t. both sky maps and beam alms, exact
  adjoint for `SkySpaceFilter` map-making. Matches numpy
  `generate_TOD_sky(..., truncate_frac_thres=0.0)`; enable x64 for
  quantitative accuracy. Optional dependency: `pip install -e '<limTOD>[jax]'`.
- **MomentRFI** (`dirt.radio.backend`): `MomentRFIFlaggingOperator`
  (host-callback into `IterativeSurfaceFitter`; existing flags become
  `prior_mask`) + `MaskedGaussianLikelihood` (flags → noise covariance).
- **Filters** (`dirt.radio.filters`): `AbstractLinearFilter`
  (extract/remove projection semantics) with `SiderealFilter` (day-repeating
  subspace), `SkySpaceFilter` (CG map-make/reproject through any linear sky
  projector), `FourierBandFilter` (fringe-rate/delay bands); plus
  `ApplyCalibrationOperator` and raw-data preservation via
  `State.checkpoint` / `SnapshotOperator`.

### Core (`dirt.core`)

- `State`: immutable pytree container (traced `data`/`coords`/`env`/`aux`/`key`,
  static hashable `meta` via `FrozenMapping`); functional updates
  (`replace`/`with_data`) and the PRNG protocol
  (`subkey, state = state.next_key()`).
- `AbstractOperator` / `LambdaOperator`: the universal `State -> State`
  contract with declarative `requires`/`provides`.
- `Pipeline`: sequential named composition (composite pattern — nests freely);
  `run_with_intermediates`, `replace_stage`, name/index access.
- `SumOperator`: parallel additive composition for source-type branches;
  per-branch PRNG subkeys; leafwise pytree accumulation with loud trace-time
  errors on shape/structure mismatch and dataless branches.

### Radio (`dirt.radio`) — placeholder physics, real contracts

- Reorganized by the single-antenna element taxonomy:
  `sky/` (uniform, global signal, foregrounds, point sources),
  `environment/` (ionosphere, ground pickup, RFI),
  `instrument/` (beam, sky-side system temperature, noise-wave/reflection
  terms, CW calibration tone, bandpass, gain, thermal noise, EMI, ADC),
  `backend/` (flagging, averaging). Flat `dirt.radio` API preserved.
- Chain ordering follows the RHINO system equation
  `P_rec = g (T_ant + T_nw + T_cw) + T_n`: CW tone before bandpass/gain
  (it tracks gain drift only through the gain); sky-side temperatures before
  the reflection/noise-wave terms.
- `NoiseWaveOperator` preserves linearity in `t_nw = (T_unc, T_cos, T_sin)` —
  the `d = H t_nw` structure GCR sampling relies on.

### Inference (`dirt.inference`)

- `build_forward_fn(pipeline, state_template, filter_spec)`: the single seam
  between forward models and inference (Equinox partition/combine).
- `Likelihood` protocol + `GaussianLikelihood`; minimal working
  `GradientCalibrator`; `to_numpyro_model` stub (NumPyro optional extra).

### Project

- src layout, hatchling, uv-native; pytest with 80% coverage floor
  (currently ~97%); ruff clean; runnable end-to-end demo
  (`examples/radio_digital_twin.py`) including gradient recovery of a known
  gain.
