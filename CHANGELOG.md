# Changelog

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
