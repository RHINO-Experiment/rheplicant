# Changelog

## Unreleased

### `identifiability()` — the diagnostic that can see across Gibbs blocks

New **`rheplicant.inference.identifiability`**. Every convergence guard in
`inference.linear` is computed *from one block*: a CG residual and
`condition_estimate`'s κ are both properties of the operator for the block
being solved, and `check_linearity` asks about one latent at a time. None of
them can see a degeneracy whose two halves live in different blocks — and they
are not wrong to pass, because each conditional of a bilinear model genuinely
*is* affine.

The failure that leaves behind is silent and large. An alternating solve over
gain × antenna temperature reports **κ = 1** and a CG residual of **1.7e-7**
while sitting **2288 K** from the truth.

`identifiability(space, pipeline, state)` column-normalises the Jacobian of the
prediction with respect to *all* the declared latents and takes its SVD. On the
same two parameterizations, with and without a known 5000 K calibration tone:

```
free-per-cell T_ant,  tone ON    n_par=72 rank=64 nullity=8   weakest identified 7.071e-01
free-per-cell T_ant,  tone OFF   n_par=72 rank=64 nullity=8   weakest identified 7.071e-01
(3,3)-basis T_ant,    tone ON    n_par=17 rank=17 nullity=0   weakest identified 6.790e-02
(3,3)-basis T_ant,    tone OFF   n_par=17 rank=16 nullity=1   null direction at  6.647e-17
```

The tone buys **exactly nothing** against a free-per-cell antenna temperature —
the free cell at the tone's channel absorbs the gain sample by sample — and
**everything** against a frequency-smooth one. And the same partition, asked one
block at a time, reports `gain` rank 8 of 8 and `t_coeff` rank 9 of 9: two clean
bills of health on a model that is degenerate.

The null space comes back **named**. `report.participation(0)` returns
`{'gain': 0.5014, 't_coeff': 0.4986}` — the bilinear degeneracy, read as the
combination it actually is — and `report.direction(0)` returns arrays shaped
like the latents, in raw units, such that adding a small multiple of them to
the parameters leaves the prediction where it was.

Three parts of the method are load-bearing, and all three are pinned by tests
that fail when they are removed:

- **Column normalisation.** Two perfectly identified latents whose scales
  differ by 1e10 have a raw singular-value ratio of 1e-10; without the scaling
  the rank test reports the choice of units and calls the model degenerate.
- **float64, whatever the caller has configured.** In single precision the
  degenerate model's null direction surfaces at **3.1e-8** — above any usable
  tolerance — and is reported as identified. The function forces
  `jax_enable_x64` for the duration and restores it afterwards, including on
  the way out of an exception, and hands back numpy arrays so the precision
  survives leaving the context.
- **A threshold that is exposed and argued for.** `DEFAULT_RANK_RTOL = 1e-8`
  sits in an 8.7-decade window (1e-13 to 4.8e-5) where every choice returns the
  same verdict on the measured model. `report.weakest_identified` is there to
  be read before the verdict is trusted.

`names=` asks the conditional question a Gibbs block faces, `at=` moves the
evaluation point the way `linear_operator`'s `at=` does, and the function
refuses rather than guesses on an empty or repeated selection, a complex or
integer latent, and a model that pins its own prediction to float32.

### Two inference tutorials, and a sampler bug they found

New [`docs/tutorial-gcr.md`](docs/tutorial-gcr.md) and
[`docs/tutorial-nuts.md`](docs/tutorial-nuts.md), backed by
`examples/tutorial_gcr.py` and `examples/tutorial_nuts.py`. One ring toy, two
opposite questions: 256 sky pixels by exact conjugate solve, three beam
parameters by gradient MCMC. Every number in both pages is the script's real
output.

**The GCR tutorial ends by breaking itself.** With a beam that resolves the sky
the RMS error equals the posterior σ and coverage is exactly 68 %. Widen the
beam past the structure and κ rises 13×, the posterior keeps 63 % of the prior
instead of 4 %, and coverage climbs to 100 % — nothing fails, and every
diagnostic says the answer is now mostly prior. An estimator without error bars
would return a similar map and report none of it.

**The NUTS tutorial fails first, on purpose, because it really did.** Written
the obvious way it produces `r_hat = 840` and an effective sample size of **2**
out of 8000 draws, with no exception, no NaN and means that look like numbers.
Two hypotheses were tested: the likelihood is multimodal (**wrong** — a scan
finds one maximum), and the posterior is a needle inside its prior (**right** —
500× narrower).

That produced a real fix in the library. New
**`rheplicant.inference.init_to_declared(space)`**: `ParameterSpace` already
declares where to start, the calibrators and `check_linearity` both use it, and
`to_numpyro_model` was dropping it on the floor. One line takes `r_hat` from 840
to **1.002** and `n_eff` from 2 to **1327** — and the run gets *faster*.
Measured on the same problem, tightening the priors and tripling the warmup
each change nothing whatever (`r_hat` 1123 and 1124).

The NUTS page also documents what the reference pages never did: `r_hat`,
`n_eff` and divergences, what each one means, and a six-point checklist for any
run.

### The CST beam reader moved to limTOD

D20 moved the horizon partition upstream and left the CST far-field reader
behind, calling the distinction "a follow-up, not a distinction worth
defending". It was not worth defending: reading a measured horn into a beam map
is the same kind of thing as deciding how that beam weights the sky, and limTOD
already had the slot — `limTOD.uvbeam` does exactly this job for pyuvdata.

`limTOD.cstbeam` is now its sibling, and gained something it could not have
while living downstream: a **`cst_beam_func`** satisfying `TODSim`'s
`beam_func(freq=..., nside=...)` contract, so a CST horn drops straight into
limTOD's own simulator. It caches each file's HEALPix resampling too — a
200-channel sweep over a 61-file directory now parses each file once rather
than hundreds of times.

`read_cst_farfield`, `cst_frequency_table` and `cst_beam_maps` remain importable
from `rheplicant.radio` and are unchanged in signature. They are pass-throughs
now, and the seam has exactly one job: **units**. `Coordinates.freq` is in Hz;
limTOD is in MHz throughout. Each package keeps its house convention and the
adapter is where they meet.

Consequently a bad export raises `ValueError` (limTOD's) rather than
`StateValidationError`, as already happened for `horizon_truncated_beam` in
D20. rheplicant's beam tests now cover the seam and RHINO's real horn; the
synthetic convention tests — the theta-fastest reshape, the `phi_sense`
reflection, the frequency interpolation — went upstream with the reader rather
than being duplicated across two repositories.

The `limtod` extra is floored at **1.10**, which is the release the reader
landed in. The runtime gates still check the *symbol* rather than the version,
and that is not redundancy: an editable install reports whatever version its
dist metadata was written with, and this repository's own environment was found
sitting at a recorded `1.8.0` while running 1.10.0 source, `limTOD.cstbeam`
importable throughout. A version check would have refused a fully capable
install. The floor says what to install; the symbol says what is there.

### NUTS through the noise model, then inference without a likelihood at all

**`to_numpyro_model` accepts a `NoiseModel`.** With `RadiometerNoise` the
observation scale is a function of the sampled parameters, so
`Normal(loc, scale).log_prob`'s `-log scale` puts the log-determinant in the
potential automatically — this is the *full* Gaussian density, not the
generalized least squares `iterative_gls` converges to. An unobserved sample
(infinite σ) is masked rather than given an infinite scale, which would send
the whole potential to `-inf`; `flags=` and wrapping in `FlaggedNoise` are now
the same code path.

*Validated against an exact sampler.* On a linear-Gaussian problem
`gcr_sample` draws the posterior in closed form with no chain and nothing to
diagnose, and NUTS must reproduce it. It does: **mean to 0.00σ, width to
1.00×**.

**New `rheplicant.inference.npe` — amortized neural posterior estimation.**
`simulate_pairs` draws (θ, x) from the prior and the simulator;
`NeuralPosterior` is a conditional Gaussian mixture; `train_posterior` fits it.
No likelihood is written anywhere, and a second observation costs a forward
pass — measured at **3.5 ms against 66 ms** for a fresh exact solve.

**An approximate posterior has no internal notion of being wrong**, and both
failure modes appeared while building this, pushing in opposite directions:

| simulations | steps | components | width / exact |
|---|---|---|---|
| 8 192 | 1 500 | 1 | 0.88 |
| 8 192 | 4 000 | 2 | **0.60** |
| 32 768 | 1 500 | 1 | **0.98** |

Too few simulations and the width is wrong; too many steps on a small bank and
it over-fits, which makes `q` too **narrow** — the failure that presents as a
better answer. So `train_posterior` holds out a validation split by default and
**returns the best validation step, not the last**: in the worked example that
is step 489 of 2000, with the training loss still falling throughout.

New `examples/three_ways_to_a_posterior.py` puts all three engines on one
problem, in the order that makes them checkable.

### `iterative_gls`: finding the covariance a GCR draw is conditioned on

`gcr_sample` is a linear sampler *given* a covariance. Under `RadiometerNoise`
the covariance is not given — σ tracks the prediction, so the weights depend on
the solution and the solution depends on the weights.

**`wiener_solve` and `gcr_sample` did not change.** They already accepted an
array `noise_std`, and their linear-Gaussian correctness was already tested.
The new module supplies only the thing that produces σ:

```python
found = iterative_gls(block, observed, noise=RadiometerNoise(dnu, tau), prior_std=P)
draw, _ = gcr_sample(block, observed, noise_std=found.noise_std, prior_std=P, key=k)
```

A **matrix-free** port of hydra-tod's `hydra_tod.linear_sampler.iterative_gls`
— that one forms a dense `U` and `N_inv`; this runs the same fixed point on the
block's JVP and VJP. A transcription of the numpy original is the test oracle.

**The convergence tolerance cannot have a fixed default**, and the first one
tried (`1e-8`) violated both bounds in turn. Two independent floors limit how
small a step is measurable: the arithmetic's epsilon (float32's is `1.2e-7`, so
`1e-8` is rounding — the loop ran to its cap reporting `converged=False` for a
run settled at `delta = 7e-8`), and the **inner CG tolerance**, since
consecutive solves differ by their own residual whatever the outer iteration
does. The latter binds in float64, where a tight `tol=1e-10` made `8·eps` far
*too tight* and it failed the same way. The default is `max(8·eps, tol)` — both
derived, neither tuned — and a test pins the failure mode.

**The mean can be weight-independent while the width is not.** New
`examples/gls_gcr.py`: three switched loads against three per-channel
noise-wave unknowns is a **square** system, so reweighting moves the point
estimate by nothing, exactly. The posterior covariance depends on Σ regardless,
and a GCR draw is a draw of that width — a frozen σ there reports error bars
wrong by −8 % to +8 %, in both directions, on an estimate that was already
right. Where the system is over-determined across genuinely different noise
levels the estimate moves too: a factor of ≈2.3 in recovered RMS on a
prediction spanning a decade.

### MomentRFI flagging: the bridge now actually runs

`MomentRFIFlaggingOperator` had been in the tree with tests since the rename,
and **every one of those tests was skipped** — MomentRFI had no packaging, so
it could not be installed anywhere, so `skipif(find_spec(...) is None)` was
true in every environment including CI. The bridge was written against an API
nobody had called.

MomentRFI is packaged now (upstream), and the tests pass unmodified: the
assumed `fit(waterfall, kernels=..., prior_mask=...)` signature was right.
What first execution added:

- **jit gives bit-identical flags** — asserted by the docstring before, by a
  test now. `jax.pure_callback` is the permanent integration for a boolean
  decision, not a stopgap.
- **`kernel_shapes` earns the matched filter's √K.** On a 3σ-per-pixel blob
  under the fitter's default 4σ cut, round 0 alone recovers *none* of it and a
  single 3×3 box recovers *all* of it, at a 0.15 % false-positive rate.
- **The flags reach the noise covariance, and it matters.** A persistent
  narrow-band emitter on 2 of 32 channels biases a maximum-likelihood amplitude
  **+5.8 %**; wrapping MomentRFI's flags in `FlaggedNoise` recovers the truth
  and agrees to six digits with flagging those channels by hand. The test
  asserts the bias removed, not the mask produced.

New `rheplicant[rfi]` extra. Like `limtod`, it names the requirement rather
than resolving it — MomentRFI is not on PyPI — and the operator's ImportError
now gives both install routes.

### The noise level became a model instead of an argument

`noise_std` was a bare scalar handed separately to five places —
`GaussianLikelihood`, `to_numpyro_model`, `fisher_information`, `wiener_solve`
and `gcr_sample` — each assuming the answer was given and constant. **The
radiometer equation says it is neither**: `sigma = |d| / sqrt(delta_nu * tau)`,
so the noise level is a function of the very thing being inferred.

New `rheplicant.inference.noise`:

- `HomoscedasticNoise(sigma)` — what a bare `noise_std` always meant, now named.
- `RadiometerNoise(channel_width, integration_time)` — the multiplicative
  default, `sigma = |prediction| / sqrt(delta_nu * tau)`. A test asserts its
  sigma matches the one `rhino_cal_jax.power.add_radiometer_noise` actually
  draws with, so the side that scores the data and the side that generates it
  cannot drift apart.
- `FlaggedNoise(base, flags)` — RFI flags reach the covariance by *wrapping a
  noise model*, not by threading a `flags=` keyword through five signatures.
- `NoiseModelLikelihood`, `inverse_variance`, `as_noise_model`.

Every `noise_std=` argument accepts a noise model in place of a number, so no
downstream signature changed. `GaussianLikelihood` and
`MaskedGaussianLikelihood` are unchanged and tested to agree with the general
form to roundoff.

**The log-determinant is not a constant when sigma depends on the prediction,
and dropping it is a different estimator.** For the multiplicative model both
are solvable in closed form: generalized least squares returns
`sum d^2 / sum d`, biased high by `(1 + f^2)`, while the full Gaussian density
is asymptotically unbiased. `NoiseModelLikelihood` keeps the term by default;
`include_logdet=False` is the explicit GLS variant. The tests assert the
implemented log-density has a vanishing gradient at each closed form, so the
claim is pinned rather than measured.

**`fisher_information` was reporting the wrong matrix for prediction-dependent
noise** — `J^T N^-1 J` is not the Fisher information when the covariance
carries parameter dependence. The variance term
`2 (d log sigma/d theta)^T (d log sigma/d theta)` is now included whenever the
noise model reports `depends_on_prediction`. Under `RadiometerNoise` it is
exactly the factor `(1 + 2 f^2)`; without it a forecast reports error bars too
wide by `sqrt(1 + 2 f^2)`.

### Docs: the equations were never rendering, and the framing was in the wrong place

**Every equation in the physics pages shipped as literal `$$...$$` source
text.** MyST's `dollarmath` extension was not enabled and MathJax was not
loaded, so `sky-to-receiver.md` and `sky-engines.md` — both written in TeX —
rendered their maths as dollar signs. A docs build cannot warn about that; only
a reader can. `dollarmath`, `amsmath` and `sphinx.ext.mathjax` are on now, and
Eq. 1, the four coupling spectra, the rank counting and the horizon split all
display as equations.

**"Operators plus three structures" now appears where it belongs.** Cascade
(`Pipeline`), sum (`SumOperator`) and switch (`SelectOperator`) are the entire
composition vocabulary, and the four node kinds map onto them one-to-one. That
framing is now stated in the docs landing page and in the tour's composition and
graph-assembly sections, and developed in full on
[the canonical signal path](https://rheplicant.readthedocs.io/en/latest/signal-path.html)
— rather than being introduced incidentally on a physics page. Two rules fall
out of it rather than being extra: a junction or selector with one live input is
traversed as identity, and `many` instances compose the way their consumer
composes.

It also makes explicit something the code always allowed and the docs never
said: **`RADIO_GRAPH` is RHINO's template, not the framework.** `SignalGraph`
and `register_graph` are public and domain-agnostic; another instrument is
another template. A documented end-to-end path for supplying one is flagged as
planned.

**`docs/sky-to-receiver.md` rebuilt** around three figures generated by running
the documented path (`docs/_generate_receiver_figures.py`, same contract as the
engine and inference generators — transparent, theme-paired SVGs from live
code):

- *the horn against the horizon* — directivity versus zenith angle with the
  azimuthal spread as a band, and the below-horizon share per frequency;
- *the cascade* — 3026 K collected → 2975 K after the horizon split → 2895 K
  after the horn's ohmic loss → 768 K delivered, each labelled with what makes
  it different from the others;
- *the recovery* — truth against the Wiener mean for all three noise-wave
  spectra, with residuals on a ±0.45 K scale.

Plus a mermaid diagram of the assembled path showing all three structures at
once, and the three unguarded joins kept as call-outs where a reader meets them.

The diagram labels each box with what its operator *does*, and does **not**
label the edges: a cascade is a run of arrows, while a sum and a switch are
properties of the node that collects them, so writing all three as edge labels
implied a relationship between two boxes that does not exist. It read as though
`beam_spill` were a contribution added to the sky rather than a transform of it
— which it is not, and a call-out now says why (a mixture whose weights add to
one, tested by the isothermal invariant, versus the atmosphere's independent
contribution).

### Changed: the horizon physics moved to limTOD, where it belongs

`horizon_truncated_beam` and `DriftScanProjector.horizon_fraction()` were
implemented here. They should not have been: how a beam weights the sky, where
the horizon falls in it and what share of its solid angle survives are limTOD's
subject — exactly as the noise-wave data model is `rhino_cal_jax`'s (D15).

Now in **limTOD 1.9** (`limtod_jax.driftscan`): `horizon_partition_weights`,
`horizon_truncated_beam`, `horizon_beam_fraction`, with 25 tests including the
painted-ground closure that decides the conventions. `horizon_weights` is
unchanged — its masking semantics were never wrong; the partition is a
different object and now says so.

What stays here is **placement**. `BeamSpillOperator` consumes `f_sky` and puts
it at the `beam_spill` node; it does not compute it.
`rheplicant.radio.beams.horizon_truncated_beam` and
`DriftScanProjector.horizon_fraction()` are pass-throughs whose only added value
is inferring `nside` from maps the caller already has, and both feature-gate on
the upstream symbol so an outdated install is named at the boundary rather than
raising `AttributeError` midway. The `limtod` extra is floored at **1.9**.

The tests moved with the physics: this side no longer re-asserts the partition,
the half-counted horizon ring or the zenith-only exactness. Duplicating them
would be two copies of a moving target. What is tested here is the seam. See
D20.

### Changed: the graph knows how switches compose, and a fixed mask is a constant

Two claims in the previous entry were wrong, and both were wrong in the same
direction — treating a limitation as a fact about the physics.

**`many` instances now compose the way their CONSUMER composes.** `many=True`
at a source folded its instances into a `SumOperator`, always. That is right for
a junction and wrong for a selector: a switch picks one source per sample, it
does not add them up. So multi-load switching — three distinct sources, the
minimum for an identifiable per-channel noise-wave fit — could not be expressed
through `assemble()`, and the documented workaround was to hand-build the
`SelectOperator`. That workaround is how a `Pipeline`-instead-of-`SumOperator`
bug got into an example and survived until a gradient came back exactly zero.

`cal_loads` is `many=True` now, and the whole switching cycle comes out of
`assemble()`:

```python
twin = assemble(SkySourceOperator(...), BeamSpillOperator(...),
                AtmosphericEmissionOperator(...), AntennaLossOperator(...),
                CalLoadOperator(t_load=...), CalLoadOperator(t_load=...),
                NoiseWaveOperator(...))
twin["receiver_input"].names   # ('observed_astro_sky', 'cal_loads', 'cal_loads_2')
```

Six operators, four different composition relationships (sum, chain, trunk,
switch), and not one line saying what connects to what. The other half of the
rule already held and is now pinned: a selector with one live branch is
traversed as identity — no switch array required, no `SelectOperator` left
behind. A source feeding both a selector and a junction keeps the Sum (no
single right answer; no shipped graph has one), and the fan-out is kept beside
the fold state so every other path folds bit-for-bit as before — `SumOperator`
splits the PRNG key per branch, so a flatter tree would be a different seeded
run, not just a different shape. See D18.

`Assembly.__getitem__` was fixed alongside: a branch spanning
`observed_astro_sky → beam_spill` is labelled by its first node, so the lookup
returned the *fold rooted at* the node instead of the operator *at* it —
contradicting its own documented contract and making
`eqx.tree_at(lambda a: a["observed_astro_sky"].sky_model.maps, ...)` fail on an
attribute the caller could see in the source. Reaching any parameter by its
graph node now works wherever the fold put it.

**The horizon mask does not cost 8×; masking the alms per call does.**
`DriftScanProjector(horizon_mask=True)` rotates into the horizontal frame,
synthesizes, multiplies, re-analyzes three times and rotates back — on every
call. Measured at nside 16 / lmax 47: **14.6 ms against 1.79 ms unmasked, 8.2×**
(the previous entry called it 7× and treated it as the price of asking for the
horizon).

It is not. The horizon is static in the horizontal frame and a drift scan's
pointing is fixed by definition, so the masked beam is a **constant**.
`rheplicant.radio.beams.horizon_truncated_beam` truncates the beam MAP once,
before analysis:

```python
beam_maps, f_sky = horizon_truncated_beam(beam_maps, el_deg=90.0, apod_deg=3.0)
```

**1.04×** — free — and the same instrument to 2.8e-5, the residual being the
alm→map→alm round trip the masking path takes *before* it masks. At a zenith
pointing no rotation is needed at all, and that is provable rather than assumed:
`horizon_weights` is a pure function of elevation, limTOD's horizontal chart
puts the zenith at the pole and the beam-local chart the boresight, so at zenith
they share a pole and a pure-elevation mask is invariant under the rotation that
still separates them. Away from zenith the function refuses rather than
hand-derive the rotation, and `horizon_mask=True` keeps its place for that case.
It returns `f_sky` with the maps because they are the same sum. See D19.

`examples/sky_to_noise_wave.py` uses both, and lost its entire hand-wired
selector as a result.

### Added: RHINO's horn on the sky, feeding the noise-wave receiver

The two halves were already there — limTOD says what the antenna sees, the
Noise-Wave GCR draft's Eq. 1 says what the receiver does with it — and they meet
at one quantity, `T_src`. `SkySourceOperator` upstream of the `receiver_input`
selector now feeds `NoiseWaveOperator` directly, and the path is tested end to
end against Eq. 1 written out by hand (4.0e-16 relative).

**`AntennaLossOperator`** and the `antenna_loss` graph node (graph v1.3) close
the physics gap that separated them. A real antenna dissipates part of what it
collects and re-emits it at its own temperature, `T = eta T_collected +
(1 - eta) T_phys`. This is a *different* loss from the noise-wave stage's
`c_s = (1 - |Gamma|^2)|F|^2`: ohmic dissipation inside the antenna versus
impedance mismatch at the receiver input. They multiply, and only the ohmic one
adds emission of its own — an efficiency folded into the noise-wave couplings
would be indistinguishable from a mismatch in the fit while silently dropping
that term.

The node sits on the trunk between `t_ant_sum` and the switch: it acts on
everything the beam collected (unlike atmospheric opacity — D13 moved the
atmosphere *off* the trunk for exactly the reason that does not apply here) and
on nothing that connects downstream, so the calibration loads arrive
unattenuated. The pairing is pinned by its thermodynamic fixed point — an
antenna at `T` looking at a sky at `T` delivers `T` for any efficiency — which
`(eta, eta)`, `(eta, 1)` and `(1, 1 - eta)` all fail. See D16 in `DESIGN.md`.

**`rheplicant.radio.beams`** reads RHINO's horn as it is actually shipped: CST
Studio far-field ASCII exports, one file per frequency, total directivity in dBi
on a regular (theta, phi) grid. `cst_beam_maps` samples them onto HEALPix in
limTOD's beam-local convention as linear power and interpolates in frequency
(extrapolation beyond the simulated band is refused). Validated against the real
horn: the directivity still integrates to 4*pi after resampling, the boresight
lands on the pole, and the below-horizon fraction survives. CST azimuth is
measured from the model's `+x` axis, which is a fact about the as-built horn and
not about the file — `phi0_deg`/`phi_sense` expose the offset and the handedness
as assumptions to check, not results.

**`examples/sky_to_noise_wave.py`** runs the whole path on the real horn:
CST → HEALPix → drift-scan m-mode sky with `normalize_beam=True`, ground spill
and atmospheric emission, ohmic loss, a three-source switching cycle
(antenna + ambient + hot load), Eq. 8 fractional radiometer noise, a closed-form
recovery of the per-channel noise-wave temperatures with the sky treated as
known data (0.11–0.14 K RMS, kappa ~ 40), and one gradient from the HEALPix sky
map through the beam convolution, the switch and Eq. 1.
`docs/sky-to-receiver.md` walks through it.

**Fixed — an out-of-range switch value now refuses instead of clamping.**
`SwitchCycle`'s range check needs concrete values and is skipped under tracing,
which is the production path. There the two consumers of one switch array
disagreed: `SelectOperator` selected no branch (`T_src = 0`) while
`SwitchCycle.gather` clamped to a neighbouring coupling row, leaving a sample
with a receiver contribution, no source, and another load's `Gamma` — finite and
correctly shaped throughout. `gather` now fills out-of-range samples with NaN.
(In `rhino_cal_jax`; `one_hot` needs no change, since an unmatched sample
already gets an all-zero row.)

**Three joins that carry no structural guard** are documented and pinned by
`tests/radio/test_sky_noise_wave_integration.py` (21 tests), alongside the
physics — a matched antenna passes the sky through untouched, a mismatched one
attenuates it by exactly `c_s`, the receiver terms do not scale with the sky,
load samples never see it and antenna samples never see the load:

- **`normalize_beam` decides whether `T_src` is a temperature at all.** Both sky
  engines default to `False` (numpy limTOD's convention), returning `∫BT` rather
  than `∫BT/∫B`. Against a uniform 200 K sky: `True` gives 200.0000 K exactly, a
  raw beam gives 32838 K, and a beam normalized by hand to unit pixel sum still
  gives 200.4113 K — **+0.21 %**, growing to ~4 % at nside 8 with a 20° beam,
  because the band-limit truncates the denominator too. The hand-normalized case
  is the one that hides. Now documented in `docs/sky-engines.md`, which had no
  mention of `normalize_beam` at all.
- **`gamma_src`'s row order must match the selector's branch order.** Both are
  `(n_source, n_freq)`, so a transposition is shape-legal; measured cost, 46 K
  peak / 28 K mean on a 545 K signal. Read the order off the assembly
  (`twin["receiver_input"].names`) rather than assuming it.
- **A hand-wired antenna branch needs `SumOperator`, not `Pipeline`.** A
  `Pipeline` of *source-type* operators replaces the data at each stage, so
  `Pipeline(sky, ground, atmosphere)` returns the atmosphere alone — the sky
  silently gone, the result finite and correctly shaped. This was a live bug in
  the first draft of the example, caught only because the gradient with respect
  to the sky map came back exactly zero. Whenever `assemble()` could have built
  the branch, check the hand-wiring against it.

**The horizon split (`BeamSpillOperator`, `beam_spill`, graph v1.4).** 1–3 % of
RHINO's horn response is below the horizon and sees ground, not sky, so the
antenna collects `f_sky <T_sky>_masked + (1 - f_sky) T_ground`.
`DriftScanProjector(horizon_mask=True)` supplied the masked average and
`GroundPickupOperator` could supply the ground term, but nothing applied the
`f_sky` weight to the sky branch — which made masking, on its own, no better
than not masking: at a 3000 K sky either choice is a ~200 K bias.

`BeamSpillOperator` applies both halves, so the weights sum to one by
construction, and `BeamSpillOperator.from_projector(projector, t_ground=...)`
reads `f_sky` off the same beam that will supply the sky —  the one call that
cannot get the two out of step. `DriftScanProjector.horizon_fraction()` computes
it; call it before `to_reference_frame()`, which folds the mask into the cached
alms and leaves no unmasked denominator (it raises if asked afterwards).

Every decision here was settled by measurement, against a projector run on a sky
map with the ground painted in at latitude 90, where the local horizon coincides
with the celestial equator and stops moving with LST. Residual on the ~200 K
effect, at nside 16:

| `f_sky` from | residual |
|---|---|
| the masked beam's harmonic integral | −17 K |
| pixel partition, horizon ring dropped (`horizon_weights` as shipped) | −8.6 K |
| pixel partition, horizon ring counted as all sky | +8.7 K |
| **pixel partition, horizon ring counted half** | **+0.005 K** |

Two findings in that table. The band-limited masked beam's solid-angle integral
is not `f_sky` — `map2alm` of a sharply cut map does not preserve the mean, so
it is off by ~0.7 %. And `limtod_jax.horizon_weights` uses a strict `el > 0`,
dropping the whole ring of pixels centred exactly on the horizon (64 of 3072 at
nside 16); a pixel centred on the horizon is half sky and half ground, and the
two one-sided alternatives are symmetric and halve with nside — a miscounted
ring, not anything harmonic. The first implementation used the strict cut and
looked entirely reasonable.

`beam_spill` sits on the ASTRO branch (`beam | observed_astro_sky →
astro_ant_sum → beam_spill → t_ant_sum`), not the trunk: the split applies to
the thing that genuinely is a beam integral over the celestial sphere, while the
other `t_ant_sum` leaves are effective temperatures by D13's construction and
`ground_pickup` in particular IS a below-horizon share. It shares the arithmetic
`a x + (1-a) b` with `AntennaLossOperator` and is deliberately not the same
operator: a spill is a *mixture* (sky and ground at one temperature give that
temperature — no loss), ohmic loss dissipates and re-emits. See D17.

### Added: parameter spaces — infer anything, however it is parameterized

The inference seam could already fit any pipeline leaf. It could only fit a
*leaf*: priors attached positionally and had to match that leaf's shape
exactly. So the models it could express were the models whose parameters
happened to already be stored numbers. A beam described by a width and a
pointing offset, a gain tied across three stages, a positive quantity explored
in its logarithm — each of those required writing a new operator whose leaves
were your parameters, which is exactly the "calibration contaminates the
instrument description" the design exists to prevent.

Two objects now carry the two ideas that were conflated:

- **`Latent`** — a named quantity you infer. Name, initial value (which fixes
  shape and dtype), optional prior, optional `linear=True`. Knows nothing
  about the pipeline.
- **`Bind`** — a rule turning latents into pipeline leaf values. Which latents,
  which leaves, and the function between them. Knows nothing about priors.

`ParameterSpace` holds both. It covers the three shapes a parameterization
takes — direct, tied (one latent to several leaves), derived (several latents
to one leaf) — and `ParameterSpace.raw` takes a bind function outright for
anything they cannot express.

```python
space = ParameterSpace(
    latents=[Latent("fwhm", init=0.5, prior=dist.Uniform(0.15, 0.70)),
             Latent("offset", init=0.0, prior=dist.Normal(0.0, 0.4))],
    bindings=[Bind(("fwhm", "offset"),
                   into=lambda p: p["sky"].projector.matrix, fn=beam_matrix)],
)
forward, start = space.forward_fn(twin, state)   # {"fwhm": ..., "offset": ...}
```

Neither calibrator needed a single change — a dict is a pytree — and posterior
predictive still `filter_vmap`s over the posterior, because bindings are
required to preserve the pipeline's treedef.

**Validation, because every failure mode here is silent.** Duplicate names, a
binding naming an undeclared latent, a latent nothing binds (it samples
happily and returns the prior), two bindings on one leaf, a selector landing
on static configuration, a produced shape or dtype-kind that does not fit its
target, a prior sized differently from its latent, a bind function that
changes the treedef. Each of those otherwise yields a finite,
correctly-shaped, wrong inference. All are caught, almost all on
`jax.eval_shape` — so validation computes nothing and happens once per build
rather than per evaluation, which is why it is not made skippable.

### Added: declared-linear blocks, checked and then exploited

`linear=True` asserts the prediction is affine in one latent — the case that
matters for sky alms and noise-wave amplitudes, ~1e6 real degrees of freedom
where gradient samplers are hopeless and a conjugate-Gaussian solve is exactly
right.

The assertion is checked before anything uses it. `check_linearity` compares
the model against its own linearization at probes spanning 1e-3 to 1e3 times
the latent's magnitude; the span is the point, since `x + eps*x^2` is
indistinguishable from linear near the origin and grossly nonlinear far from
it. A block fails only if it exceeds a relative tolerance AND an absolute
roundoff floor — without the floor the relative measure explodes at small
probes and rejects correct blocks, and the cure a user reaches for on a
spuriously-failing check is to switch it off.

`linear_operator` exports `A`, `A^T` and the offset without forming a matrix
(`jax.linearize` + `jax.vjp`), so applying a 1e6-dimensional block costs one
forward evaluation. `wiener_solve` gives the posterior mean by CG. GCR
sampling adds a fluctuation term to the same right-hand side and is what this
operator is for next.

Two facts measured rather than assumed. `jax.vjp` returns the conjugate
gradient for complex latents, so the identity that holds is
`Re sum(x * adjoint(y)) == sum(forward(x) * y)` — the adjoint of the REAL
inner product, which is the pairing a Gaussian likelihood forms. And
`wiener_solve` carries complex latents as `(real, imag)`, because a real
prediction makes the map R-linear but not C-linear and a Krylov method over C
would be solving a different problem.

### Added: exact posterior draws from a linear block (GCR)

`gcr_sample` turns the Wiener solve into a constrained realization by adding
two white-noise terms to the same right-hand side:

```
(A^T N^-1 A + S^-1) x = A^T N^-1 (d - offset) + A^T N^-1/2 w1 + S^-1/2 w2
```

The right-hand side then has covariance equal to the normal operator itself, so
`x = M^-1 b` carries the posterior mean AND covariance `M^-1 M M^-1 = M^-1`
exactly. Every call is an independent draw — no chain, no burn-in, nothing to
diagnose — for the same single CG solve the mean costs, because the fluctuation
enters the right-hand side and never the operator. That is what makes a
1e6-dimensional block samplable at all.

Both moments are tested against a densely-constructed posterior: 6000 draws
reproduce the covariance to 12% in Frobenius norm, and with uninformative data
the draws reproduce the PRIOR width to 5%. Testing the mean alone would not
have been enough — a sampler that gets the mean right and the covariance wrong
looks entirely healthy.

`linear_operator` and `check_linearity` also gained `at=`, which fixes the
OTHER latents. A block is linear only *given* them, so a Gibbs sweep must
rebuild it wherever they currently are; without `at=` the block silently kept
describing the model at its declared starting point, which is right for exactly
one sweep.

### Changed: Fisher and covariance matrices carry named rows

`cov.sigma("fwhm")` instead of `cov.matrix[0, 0]`; `cov.block("fwhm",
"log_gain")` for a cross-covariance. Spans come from the actual flattening,
not from an assumption about dict ordering. `FlatMatrix.kind` also
distinguishes Fisher from covariance, and `sigma()` refuses on a Fisher
matrix: `sqrt(diag(F))` looks exactly like an error bar and ignores every
parameter degeneracy.

### Removed: `prior_template` / `set_prior` (breaking)

`to_numpyro_model` and `predict_from_samples` take a `ParameterSpace` where
they took a positional prior pytree. Sample sites are named by their latents,
which deleted `_site_name` outright — 50 lines that walked pytree paths and
recognised Assembly/Pipeline/SumOperator containers to reconstruct a readable
name for a parameter that never had one. Once parameters have their own
namespace, the problem does not exist. Net ~90 lines removed from the bridge.

`build_forward_fn` is NOT removed and is not deprecated: it answers "train
everything under this subtree", which is what a neural surrogate's weights
want, while `ParameterSpace.forward_fn` answers "these specific quantities,
possibly transformed or shared".

New: [`docs/inference.md`](https://rheplicant.readthedocs.io/en/latest/inference.html),
[`examples/inferring_anything.py`](https://github.com/RHINO-Experiment/rheplicant/blob/main/examples/inferring_anything.py), and
`docs/_generate_inference_figures.py`, which produces the page's figures by
running the code rather than illustrating it.

### Added: a `limtod` extra — the sky engines install from PyPI

limTOD is [published on PyPI](https://pypi.org/project/limTOD/), so the
engine it backs is now a declared optional dependency rather than a
build-it-yourself instruction:

```bash
pip install "rheplicant[limtod]"
```

The extra floors at `limTOD[jax]>=1.8`, so a fresh install gets every fast
path including the hoisted Wigner plane. The runtime feature check stays, so
an environment already holding 1.7 keeps working and only forgoes that
speed-up — the floor expresses what a new install *should* take, not the
bare minimum the code tolerates. Every `pip install -e '<limTOD>[jax]'` in the
docs, examples and error messages is updated accordingly.

### Performance: the drift-scan engine stops repeating itself

Three hoists, none of which change a number. Measured at nside 64 / lmax 191
/ 512 samples / 32 channels in x64, on the cached-beam projector.

- **`forward_alms()` / `sky_to_alms()` / `mmodes_alms()`** — the sky may now
  enter as pre-analysed quadrature alms. `AbstractSkyProjector`'s contract is
  maps-in, so `forward()` re-ran the analysis on every call; once the beam
  rotation is cached that IS the engine — **176 ms of a 193 ms forward and
  114 MB of its 114 MB peak**. Analysed once instead, a forward drops from
  **163 ms / 122 MB to 1.1 ms / 11 MB** (153x), agreeing to 5e-16.
  `forward()` and `mmodes()` are now thin wrappers, so the map-based contract
  is unchanged. Note `sky_to_alms` uses the QUADRATURE transform, not the
  true-alm one `from_beam_maps` uses for the beam — the two differ by
  `npix/4pi`, which is why this is a method rather than a docstring.
- **The Wigner-d plane is built once per projector.** It depends on the
  pointing alone — LST enters the zyz composition in the first-applied slot,
  so it moves `psi` and never the plane — but `limtod_jax` rebuilt it on every
  call. Hoisted through limTOD's new `dl_array` parameter: **44.0 ms -> 2.8 ms
  (15.6x)** on the rotation at lmax 127, bit-for-bit identical. This is the
  only amortization available when the BEAM is the fitted parameter, where
  `to_reference_frame()` cannot be used because gradients must reach the
  beam-local alms. Skipped gracefully on a limTOD without the parameter.
- **`freq_chunk`** (static, default `None`) walks the frequency axis in
  batches instead of all at once. Peak memory is linear in `n_freq` — 3.4 MB
  per channel, so 114 MB at 32 channels but ~1.8 GB at nside 256. Chunk 8 cut
  the peak 3.2x for 1.7x the time; chunk 1, 9.4x for 7.9x. Off by default
  because below the memory ceiling it is a pure loss.

Requires the matching limTOD (`dl_array` / `dl_plane_for_pointing`) for the
second item only; the others are self-contained.

The docs benchmark was re-timed afterwards, and the split is exactly where the
optimizations predict. The **general** engine is untouched (its per-sample
rotation cannot hoist a plane, and it never enters the m-mode synthesis):
176.4 / 1354.8 / 10150.6 / 65569 ms across lmax 23 / 47 / 95 / 191, all within
the ±7 % run-to-run scatter of the previous run. The **un-cached m-mode** path
gained 1.2–1.8x (115.2 -> 62.6 ms at lmax 191), and the **fully optimized**
path barely moved (63.7 -> 60.4 ms) because its rotation was already cached
and it was already on the FFT synthesis.

One consequence worth knowing: plain m-mode now runs within **4 %** of the
cached-beam + FFT path at lmax 191, where it used to be 81 % slower. What is
left in those 60 ms is the sky analysis, not the rotation — which is precisely
what `forward_alms()` removes.

### Fixed: the normalization denominator was recomputed on every call

`_ones_alm` — the quadrature alms of the ones map, used when
`normalize_beam=True` — is a pure function of the static `(nside, lmax)`, and
a comment in both engines asserted XLA constant-folds it. **It does not.** It
is a full s2fft analysis (63 unrolled ring FFTs plus a loop), which exceeds
XLA's constant-folding budget, so it was traced into every call. Measured at
nside 64 / lmax 191 in x64: **64 ms and 10 MB per call, 7700 lines of HLO**.

Hoisted into JAX's compile-time constant-evaluation context, which computes it
eagerly at trace time: **0.04 ms, 0.15 MB, 48 lines of HLO** — a 1600x
reduction, with bitwise-identical output. End to end on a 32-channel /
512-sample forward the normalization overhead drops from ~51 ms to ~11 ms (the
remainder being the genuine second synthesis for the denominator). Both
comments now state the measured behaviour instead of the assumed one.

Also documented: `horizon_mask=True` on the `"local"` beam frame reads like a
physics switch but is the most expensive path in the projector — it adds a map
synthesis, `mask_iterations` analysis+synthesis rounds and a second Wigner
rotation to every call, roughly 7x the cached unmasked cost. The remedy
already existed (`to_reference_frame()` folds the mask in once); nothing said
so.

### Removed: `MModeProjector` and `LimTODProjector`

**Breaking.** Three sky engines remain, no placeholders: two that compute the
physics (`GeneralPointingProjector`, `DriftScanProjector`, named for the
observation geometry) and `MatrixProjector`, which takes the projection as
data. `rheplicant.radio.sky.projection` is down from five classes to two.

`MModeProjector` was a placeholder that took m-mode transfer matrices as
given. `DriftScanProjector` computes them from the beam, so the placeholder's
only remaining job was explaining that it was not the one to use — which four
separate docs had to say. It could not have grown into the real thing either:
its `(n_freq, n_m, n_pix)` layout is ~6.4 GB at nside 64 / 32 channels / 512
samples, and it required `n_m == n_time`, whereas m-mode pipelines store
`(n_freq, n_m, n_alm)` with `n_m = 2*lmax+1` independent of the sampling.

`LimTODProjector` bridged to numpy limTOD through `jax.pure_callback`: not
differentiable, not vmappable, no adjoint — so it could feed neither
calibration, nor Fisher forecasts, nor sky-space map-making, which is most of
what this framework is for. Now that both real engines are native JAX, anyone
wanting a numpy reference should call `limTOD.simulator.generate_TOD_sky`
directly, which is also the more trustworthy comparison: XLA runs callback
threads with FTZ/DAZ set, so healpy inside one lands ~1e-7 relative from the
same numpy code on the main thread. (This also retires the
`truncate_frac_thres` knob added moments earlier in this same Unreleased
cycle.) D10's host-callback policy is narrowed to match: callbacks are for
inherently non-differentiable steps such as RFI flagging, not for borrowing
numpy physics.

### Renamed: `NativeLimTODProjector` -> `GeneralPointingProjector`

**Breaking, no alias.** The two real sky engines were named on different
axes: one for its provenance (`NativeLimTOD`), one for its applicability
(`DriftScan`). Since BOTH are ports of `limtod_jax`, provenance does not
distinguish them — it just spent the name. Engines are now named for the
observation geometry they serve, which is the question a user actually asks:

| engine | named for | when |
|---|---|---|
| `GeneralPointingProjector` | observation geometry | pointing varies per sample |
| `DriftScanProjector` | observation geometry | pointing is fixed; the Earth scans |
| `MatrixProjector` | the data you supply | the projection is already built |

Module `rheplicant.radio.sky.native` -> `rheplicant.radio.sky.general_pointing`.
No compatibility alias: the old name is gone, so a stale import fails loudly
at import time rather than silently working until it does not.

- **Docs: [sky engines](https://rheplicant.readthedocs.io/en/latest/sky-engines.html)**
  — a new page comparing the general and drift-scan engines side by side,
  with figures generated from live code by `docs/_generate_engine_figures.py`
  (one sidereal day of GSM sky through a zenith beam at latitude 53.2°):
  the waterfall the twin produces, the engine-to-engine agreement, the
  wall-clock scaling, and the m-mode spectrum. Every figure ships in light
  and dark variants and is theme-switched by furo. Measured at nside 64
  (lmax 191), 512 samples, 32 channels: the engines agree to **1.4e-15**
  relative, and one forward evaluation costs **66 s** on the general engine
  against **60 ms** with the cached beam and FFT synthesis — about a
  thousandfold. (Both are `forward()` on sky maps, so they include the sky
  analysis; the general engine's timing carries a few per cent of
  run-to-run scatter, so the ratio is not four significant figures.)
  `sphinx-design` is a new docs dependency.
  `--replot` redraws every figure from a cached run, so a purely visual
  change never costs another 20-minute generic-engine sweep.
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
  reusing a `GeneralPointingProjector`'s coords is the expected way to switch
  engines — and traced values, which carry nothing to compare, are left
  alone. Uniform-grid violations are also re-raised as `StateValidationError`
  so the whole boundary is catchable as one family.
- Documented the m-mode formalism's source (*M-mode RIME explicit in beam,
  fringe and sky modes*) in the module and the docs; the last line of its
  Eq. (13) is the identity the fast path implements.
- Projector cross-references: the `projection` module ladder, the `adjoint`
  error message, `MatrixProjector`, and `GeneralPointingProjector` now all point
  at the drift-scan engine, and `rheplicant.radio.sky.driftscan` was added to
  the API reference — its ~200 lines of contract documentation were missing
  from the rendered docs entirely. The stale engine ladders in
  `docs/tour.md`, `README.md`, and `DESIGN.md` (D8), which pointed at a
  placeholder as the drift-scan answer, are corrected.
- `DriftScanProjector` (`rheplicant.radio.sky.driftscan`): the m-mode fast
  path for drift scans. Derives the m-mode projection from the beam alms via
  `limtod_jax.driftscan` (one Wigner rotation for the whole scan plus per-m
  phases): equal to `GeneralPointingProjector` with constant pointing to float64
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
  general `GeneralPointingProjector` as an independent oracle; and the
  `to_reference_frame` gradient is compared for equality with the uncached
  projector's rather than only checked finite.
- `DriftScanProjector.mmodes()` now rejects `normalize_beam=True`: the
  normalization divides by the ones-map denominator, which is not part of
  the m-mode expansion, so the returned coefficients would silently not be
  the spectrum of `forward()` (~18x off).

### Changed: `NoiseWaveOperator` implements the full Eq. 1 (breaking)

`NoiseWaveOperator` now runs Eq. 1 of the noise-wave GCR note through
`rhino_cal_jax` (see D15 in `DESIGN.md`) instead of the `F -> 1` placeholder
that summed one scalar `Gamma` straight into the sky-side temperature.
`t_zero` is renamed `t_rx`; the scalar `gamma_re`/`gamma_im` pair is replaced
by per-source `gamma_src_re`/`gamma_src_im` of shape `(n_source, n_freq)` plus
`gamma_rec_re`/`gamma_rec_im` of shape `(n_freq,)` — reflection coefficients
belong to *sources*, and the `receiver_input` selector discards source
identity before this node sees the data, which the old single-`Gamma` shape
could not represent. The operator now reads which source is connected from
`coords.extra["receiver_input"]` and raises if it carries more than one source
and that array is absent, rather than silently defaulting to the first.

### Added: the noise-wave temperatures as a checked linear block (worked example)

`examples/noise_wave_gcr.py`: the per-channel noise-wave temperatures
(`t_unc`, `t_cos`, `t_sin`) declared as ONE `linear=True` latent, solved in
closed form (`wiener_solve`, GCR note Eq. 30) and sampled exactly
(`gcr_sample`, Eq. 31). A `--one-source` mode shows what switching buys: with
one load the per-channel design matrix is deficient by a factor of three and
two of every three directions fall back to the prior; with three genuinely
different loads it is square and every direction is data-constrained.

### Fixed: `wiener_solve`/`gcr_sample`'s convergence guard now certifies the ERROR, not the residual

`require_convergence` bounded the relative *residual* of the CG solve, but
what a caller needs bounded is the *solution* error, and jax's `cg` gives no
other convergence signal to check. The two differ by the condition number of
the normal operator `M = AᵀN⁻¹A + S⁻¹`, and κ is large by *design* in exactly
the case these solvers exist for: whenever the data does not fully identify
the block, `λ_min(M)` is exactly `1/prior_std²` and κ runs past 1e6. On a
48-dimensional calibration block (two of every three directions
prior-dominated, `cond(M) ~ 4e8`) CG stopped on a residual dominated by the
one well-constrained direction, having left the prior-dominated directions at
their starting value of zero — `gcr_sample` reported a posterior sigma of
0.026 K where 81 K was correct, understating the width by a factor of ~3000,
always toward false confidence, while the residual sat at ~1e-7 and
`require_convergence`'s old default of `1e-3` never fired.

Tightening `tol` alone would only have moved the threshold at which this same
failure mode recurs — it does not detect anything, so a block conditioned a
few orders of magnitude worse would again pass silently. `require_convergence`
now bounds `κ · relative_residual` instead, with κ estimated by two power
iterations on the operator the solver already applies (`λ_min` measured, not
assumed worst-case, since the rigorous bound `λ_min ≥ 1/prior_std²` would flag
every healthy full-rank block as ill-conditioned). A second, separate error
covers the case where `κ · eps` already exceeds the target: no tolerance or
iteration count helps there, only precision, and the natural response to the
first message — tighten `tol`, raise `maxiter` — would burn iterations to
arrive at an equally wrong answer.

- **`condition_estimate(block, noise_std=..., prior_std=...)`** is new and
  public: it reports κ, which is what a caller needs to choose `tol`
  (`tol ≈ require_convergence / κ`) — matrix-free, like everything else here.
- **`rheplicant.inference.conditioning`** is new: `tree_norm`,
  `largest_eigenvalue`, and `extreme_eigenvalues`, matrix-free spectral
  diagnostics over pytrees with no dependency on the block machinery above
  them.
- **`require_convergence` changed meaning**, from a bound on the relative
  residual to a bound on the relative error. For a well-conditioned block
  κ≈1 and the two coincide, so healthy solves are unaffected; a block the
  data does not identify, which used to return silently wrong, now raises
  and names the remedy (tighten `tol`, or strengthen the prior). `tol`'s
  default is unchanged at `1e-6` — it is the guard that changed, not the
  starting point it checks. `require_convergence=None` still disables the
  guard entirely, and also skips its cost.

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
