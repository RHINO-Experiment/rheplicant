# From the sky to the receiver

The [sky engines](sky-engines.md) produce an antenna temperature. The
[noise-wave model](operators.md#instrument-trunk-order--graph-order) consumes
one. This page joins them end to end on RHINO's actual horn, and is a
walkthrough of
[`examples/sky_to_noise_wave.py`](https://github.com/RHINO-Experiment/rheplicant/blob/main/examples/sky_to_noise_wave.py),
which runs everything shown below. Every number and every figure here comes
from that path being executed, not from illustrating it.

```bash
uv run python examples/sky_to_noise_wave.py
```

## The one identification

The Noise-Wave GCR draft's Eq. 1 says what the receiver measures:

$$
T_\mathrm{sys}(\nu, t) \;=\; T_\mathrm{src}\,c_s
  \;+\; T_\mathrm{unc}\,\kappa_\mathrm{unc}
  \;+\; T_\mathrm{cos}\,\kappa_\mathrm{cos}
  \;+\; T_\mathrm{sin}\,\kappa_\mathrm{sin}
  \;+\; T_\mathrm{rx}
$$

with the four coupling spectra fixed by the reflection coefficients of the
source and of the receiver,

$$
F = \frac{\sqrt{1 - |\Gamma_\mathrm{rec}|^2}}
         {1 - \Gamma_\mathrm{src}\Gamma_\mathrm{rec}},
\qquad
\begin{aligned}
c_s &= \bigl(1 - |\Gamma_\mathrm{src}|^2\bigr)\,|F|^2, &\qquad
\kappa_\mathrm{cos} &= \mathrm{Re}\bigl(\Gamma_\mathrm{src} F\bigr), \\[2pt]
\kappa_\mathrm{unc} &= |\Gamma_\mathrm{src}|^2\,|F|^2, &\qquad
\kappa_\mathrm{sin} &= \mathrm{Im}\bigl(\Gamma_\mathrm{src} F\bigr).
\end{aligned}
$$

$T_\mathrm{src}$ is whatever the receiver input is connected to. On antenna
samples that is the beam-convolved sky; on load samples the switch replaces it
and takes the load's $\Gamma$ with it. **That is the entire interface**, and the
graph already encodes the ordering, so there is no glue code — but three joins
carry no structural guard, and each returns a finite, correctly-shaped, wrong
answer when you get it wrong. They are called out as you reach them.

## The path, as the graph builds it

Six operators, and one call. All three composition structures appear here at
once, and the picture shows each of them differently because they *are*
different kinds of thing — a **cascade** is a run of arrows, a **sum** and a
**switch** are properties of the node that collects them:

```{mermaid}
flowchart LR
  sky["SkySourceOperator<br/>the beam-convolved sky<br/>observed_astro_sky"]
  spill["BeamSpillOperator<br/>mixes ground into it<br/>beam_spill"]
  atm["AtmosphericEmissionOperator<br/>its own emission<br/>atmosphere"]
  tsum(("+"))
  loss["AntennaLossOperator<br/>attenuates, and emits<br/>antenna_loss"]
  sw(("sw"))
  amb["CalLoadOperator<br/>ambient load, 300 K<br/>cal_loads_1"]
  hot["CalLoadOperator<br/>hot load, 400 K<br/>cal_loads_2"]
  nw["NoiseWaveOperator<br/>Eq. 1<br/>noise_wave"]
  out["T_sys"]
  sky --> spill
  spill --> tsum
  atm --> tsum
  tsum --> loss
  loss --> sw
  amb --> sw
  hot --> sw
  sw --> nw
  nw --> out
  classDef lit fill:#FAC775,stroke:#854F0B,color:#412402;
  classDef wire fill:#F1EFE8,stroke:#854F0B,color:#444441;
  class sky,spill,atm,loss,amb,hot,nw lit;
  class tsum,sw,out wire;
```

Read it structure by structure:

- **cascade** — `sky → beam_spill`, and again `t_ant_sum → antenna_loss →
  receiver_input`. Each stage transforms what the last produced.
- **sum** at the `(+)`: the sky branch and the atmosphere are *independent
  contributions*, and they add.
- **switch** at the `(sw)`: the antenna and the two loads are *alternatives*.
  One is connected per sample; the loads **replace** the antenna rather than
  adding to it.

:::{admonition} Why the spill cascades and the atmosphere sums
:class: tip
`BeamSpillOperator` looks like it should be a sum — sky *plus* ground — and it
is not. It computes a **mixture**,
$f_\mathrm{sky} T_\mathrm{sky} + (1 - f_\mathrm{sky}) T_\mathrm{ground}$,
whose two weights add to one by construction. The test is the isothermal one: a
sky and a ground at the same $T$ must give $T$, and addition cannot do that
($T + T = 2T$). So the spill *transforms* the sky — one input, one output, a
`transform` node — while `AtmosphericEmissionOperator` contributes something
independent that nothing constrains against the sky, and is a `source` leaf into
the junction.

The mixture could have been split into a scaling and a separate ground leaf.
That is exactly what it must not be: two objects holding two numbers that have
to satisfy $f + (1-f) = 1$, with nothing enforcing it. See D17.
:::

Not one line of the `assemble()` call says any of this — the
[canonical path](signal-path.md#rhinos-template) does.

---

## 1. The horn

RHINO ships its horn as CST Studio far-field ASCII exports, one file per
frequency, holding total directivity in dBi on a regular $(\theta, \phi)$ grid.
{func}`~rheplicant.radio.beams.cst_beam_maps` reads them onto HEALPix in
limTOD's beam-local convention (boresight at the pole), as linear power, and
interpolates between the bracketing files in frequency.

A real beam does not stop at the horizon, and what is below it sees ground:

```{figure} _static/receiver-horizon-light.svg
:figclass: only-light
:alt: RHINO horn directivity versus zenith angle, and the below-horizon share
:width: 100%

Left: the horn's directivity against zenith angle at the two ends of the band.
The line is the azimuthal mean per ring, the band its 10–90 % spread — this horn
varies by tens of percent around a ring, which is why the azimuth convention is
not a detail. Right: the share of solid angle below the horizon, peaking near
3.4 % at 70 MHz.
```

```{figure} _static/receiver-horizon-dark.svg
:figclass: only-dark
:alt: RHINO horn directivity versus zenith angle, and the below-horizon share
:width: 100%

Left: the horn's directivity against zenith angle at the two ends of the band.
The line is the azimuthal mean per ring, the band its 10–90 % spread — this horn
varies by tens of percent around a ring, which is why the azimuth convention is
not a detail. Right: the share of solid angle below the horizon, peaking near
3.4 % at 70 MHz.
```

For a drift scan the pointing is fixed, so the horizon is fixed and the
truncated beam is a **constant** — one multiply, done once, which also yields
the surviving fraction $f_\mathrm{sky}$:

```python
from rheplicant.radio import cst_beam_maps, horizon_truncated_beam
from rheplicant.radio.sky import DriftScanProjector

freq = jnp.linspace(60e6, 85e6, 8)
beam_maps = cst_beam_maps("~/Dataspace/RHINO/CST_beams/HornDryGround",
                          freq, nside=16)
beam_maps, f_sky = horizon_truncated_beam(beam_maps, el_deg=90.0, apod_deg=3.0)

projector = DriftScanProjector.from_beam_maps(
    beam_maps, lat_deg=53.2, az_deg=0.0, el_deg=90.0,
    lmax=47, normalize_beam=True,
).to_reference_frame(lst_ref_deg=0.0)          # pay the Wigner rotation once
```

:::{admonition} Join 1 — `normalize_beam` decides whether `T_src` is a temperature
:class: warning
Both sky engines default to `normalize_beam=False`, matching numpy limTOD: the
forward model then returns $\int B\,T \,d\Omega$, not
$\int B\,T \,d\Omega \big/ \int B \,d\Omega$. The first is not a temperature.
Use limTOD's own switch rather than normalizing the beam map by hand — a
hand-normalized beam is *still* biased at the percent level, because the
band-limit truncates the denominator too. The numbers are in
[sky engines](sky-engines.md#is-the-output-a-temperature).
:::

The example checks this rather than asserting it, against the one sky whose beam
average is known by definition — a uniform one:

```text
beam: RHINO horn, HornDryGround
      peak 12.14..14.38 dBi over 60-85 MHz
      f_sky = 0.9661..0.9896  -> 1.0-3.4% of the response is below the horizon
      uniform 200 K sky -> 200.0000 K  (this is what makes it a temperature)
drift-scan sky: (96, 8)  1869 .. 4682 K
```

Four thousand kelvin is not a typo: at 60–85 MHz the Galactic synchrotron sky
really is that bright, which is why the calibration problem is hard.

---

## 2. The antenna chain, assembled from the graph

```python
twin = assemble(
    SkySourceOperator(sky_model=MapSky(sky_maps), projector=projector),
    BeamSpillOperator(sky_fraction=f_sky, t_ground=jnp.array(290.0)),
    AtmosphericEmissionOperator(t_atm=jnp.array(3.0)),
    AntennaLossOperator(efficiency=jnp.array(0.97), t_physical=jnp.array(293.0)),
    CalLoadOperator(t_load=jnp.array(300.0)),
    CalLoadOperator(t_load=jnp.array(400.0)),
    NoiseWaveOperator(
        t_unc=..., t_cos=..., t_sin=..., t_rx=jnp.array(290.0),
        gamma_src_re=gamma_src.real, gamma_src_im=gamma_src.imag,
        gamma_rec_re=gamma_rec.real, gamma_rec_im=gamma_rec.imag,
    ),
)
```

```text
Assembly(graph='single-antenna', lit=['observed_astro_sky', 'atmosphere',
  'beam_spill', 'antenna_loss', 'cal_loads', 'noise_wave'],
  skipped-as-identity=['astro_ant_sum'])
switch order: ('t_ant_sum', 'cal_loads') <- gamma_src rows stack in THIS order
```

:::{admonition} Join 2 — `gamma_src`'s row order is the selector's branch order
:class: warning
`NoiseWaveOperator` carries $\Gamma$ per source; the selector orders its
branches by the graph's in-edge declaration, then by the order the loads were
provided. The two orderings are independent and both objects are
`(n_source, n_freq)`, so transposing them is shape-legal. Measured cost of
getting it backwards: **46 K peak, 28 K mean** on a 545 K signal. Read the order
off the assembly — `twin["receiver_input"].names` — rather than assuming it.
:::

The example then checks the assembled twin against Eq. 1 written out by hand,
with the antenna chain in $T_\mathrm{src}$'s place:

```python
t_visible   = projector.forward(sky_maps, coords)          # masked sky average
t_collected = f_sky * t_visible + (1 - f_sky) * T_GROUND + T_ATM
t_ant       = ETA * t_collected + (1 - ETA) * T_PHYS
```

```text
assembled twin vs Eq. 1 by hand: 2.7e-16 relative — roundoff
```

---

## 3. Three effects, none standing in for another

The sky is modified three times on its way to the receiver. They are easy to
confuse, they compose, and each has a distinct signature:

```{figure} _static/receiver-cascade-light.svg
:figclass: only-light
:alt: Sky temperature through the horizon split, ohmic loss and mismatch loss
:width: 100%

Band-averaged, in order. The horizon split is a *mixture*; the ohmic loss both
attenuates and emits; the mismatch loss only attenuates. None of it is a detail
— the first two are ~4 % between them, the third a factor of four.
```

```{figure} _static/receiver-cascade-dark.svg
:figclass: only-dark
:alt: Sky temperature through the horizon split, ohmic loss and mismatch loss
:width: 100%

Band-averaged, in order. The horizon split is a *mixture*; the ohmic loss both
attenuates and emits; the mismatch loss only attenuates. None of it is a detail
— the first two are ~4 % between them, the third a factor of four.
```

:::{list-table}
:header-rows: 1
:widths: 27 37 36

* - Stage
  - What it is
  - Signature
* - $f_\mathrm{sky}$ — `BeamSpillOperator`
  - part of the beam is looking at **ground**, not sky
  - **mixing, no loss** — sky and ground at the same $T$ give $T$
* - $\eta$ — `AntennaLossOperator`
  - ohmic dissipation **inside** the horn
  - loss **and** its own emission, $(1-\eta)\,T_\mathrm{phys}$
* - $c_s$ — inside `NoiseWaveOperator`
  - impedance **mismatch** at the receiver input
  - loss, nothing added
:::

The first two share the arithmetic $a\,x + (1-a)\,b$ and are deliberately not
one operator: merging them would make an efficiency and a spill fraction
indistinguishable in a fit, and would silently drop whichever additive term the
survivor does not carry. Each pairing is pinned by an invariant rather than by
inspection — for the ohmic loss, an antenna at $T$ looking at a sky at $T$
delivers $T$ for any efficiency; for the spill, the same statement with ground
in place of the antenna.

---

## 4. A real switching cycle

An identifiable per-channel noise-wave fit needs several sources with genuinely
different $\Gamma$. Each switch position contributes exactly **one equation per
frequency channel**, so *while every temperature is free per channel* the design
matrix has rank

$$
\mathrm{rank} = \min\!\left(n_\mathrm{src},\, k\right) \times n_\mathrm{freq}
$$

where $k$ is the number of **free temperature families**: four when $T_{rx}$ is
fitted alongside $T_{unc}, T_{cos}, T_{sin}$, three only when it is held known.
So the three loads assembled below make a three-family fit square and leave a
four-family one deficient by exactly $n_\mathrm{freq}$.

:::{warning}
That count is per-channel and nothing more. The moment the temperatures become
coefficients of a frequency basis, the basis ties channels together and the
counting stops applying — in **both** directions, with no counting rule to
replace it. Measure that case with
{func}`~rheplicant.inference.identifiability.identifiability` instead of
counting loads. See [D15](design.md) and `NoiseWaveOperator`'s module docstring
for the measured numbers.
:::

`cal_loads` is `many=True` and feeds only the selector, so each
`CalLoadOperator` becomes its own switch position rather than being summed with
its sibling. The `assemble()` call in §2 *is* the whole switching cycle.

:::{admonition} If you hand-wire a branch anyway
:class: warning
A `Pipeline` of *source-type* operators **replaces** the data at each stage; only
the last one survives. `Pipeline(sky, ground, atmosphere)` therefore returns the
atmosphere alone — the sky silently gone, the result finite and correctly shaped.
Summing is what the `t_ant_sum` junction does, and `SumOperator` is how you say
it by hand. This bug was in an earlier draft of this example, back when
multi-load switching still required hand-wiring, and it was caught only because
the gradient with respect to the sky map came back exactly zero. The graph
knowing the composition rules is what removed the opportunity.
:::

Noise is the draft's Eq. 8 — fractional, $d \to d\,(1 + w)$ with
$w \sim \mathcal{N}(0, \sigma_w)$ and $\sigma_w = 1/\sqrt{\Delta\nu\,\tau}$:

```python
observed  = rcj.add_radiometer_noise(truth, key, t_int=T_INT, delta_nu=DELTA_NU)
noise_std = observed / (DELTA_NU * T_INT) ** 0.5   # per sample, not a scalar
```

Because the noise is multiplicative it is ~2× larger on antenna samples than on
the loads. A scalar $\sigma$ would weight them equally and throw that away.

```text
simulated waterfall: (96, 8), 776.4 K mean, sigma 0.201..0.668 K (Eq. 8)
   antenna   32 samples    1220.02 K mean   232.68 K rms
   ambient   32 samples     586.66 K mean    21.42 K rms
   hot       32 samples     522.62 K mean    14.20 K rms
```

---

## 5. Closing the loop

The sky is **data** here, not a parameter: limTOD supplies it and its only job
is to be right. That is exactly why the block stays linear in the noise-wave
temperatures and needs no gradient sampler — the same
[linear-block machinery](inference.md#linear-blocks) as
`examples/noise_wave_gcr.py`, now with a real sky in the $T_\mathrm{src}$ column.

```python
space = ParameterSpace(
    latents=[Latent("t_nw", init=jnp.full((3, N_FREQ), 100.0), linear=True)],
    bindings=[
        Bind("t_nw", into=(lambda p: p["noise_wave"].t_unc,
                           lambda p: p["noise_wave"].t_cos,
                           lambda p: p["noise_wave"].t_sin),
             fn=lambda v: (v[0], v[1], v[2])),
    ],
)
block = linear_operator(space, twin(jnp.zeros((3, N_FREQ))), state)
solved, residual = wiener_solve(block, observed, noise_std=noise_std,
                                prior_std=100.0, tol=1e-10, maxiter=4000)
```

```{figure} _static/receiver-recovery-light.svg
:figclass: only-light
:alt: Truth against the Wiener mean for the three noise-wave temperatures
:width: 100%

Truth against the Wiener mean, per frequency channel, with the residual
underneath on a ±0.45 K scale. The spectra come back to ~0.1 K out of spreads of
8–60 K, from a waterfall whose own scatter is 0.2–0.7 K.
```

```{figure} _static/receiver-recovery-dark.svg
:figclass: only-dark
:alt: Truth against the Wiener mean for the three noise-wave temperatures
:width: 100%

Truth against the Wiener mean, per frequency channel, with the residual
underneath on a ±0.45 K scale. The spectra come back to ~0.1 K out of spreads of
8–60 K, from a waterfall whose own scatter is 0.2–0.7 K.
```

```text
linearity check: worst relative departure 6.7e-13
condition_estimate: kappa = 4.00e+01

Wiener mean (Eq. 30), CG residual 5.1e-11:
   T_unc  RMS error   0.134 K   ( 0.33% of its 40 K spread)
   T_cos  RMS error   0.113 K   ( 0.19% of its 60 K spread)
   T_sin  RMS error   0.110 K   ( 1.40% of its  8 K spread)
```

$\kappa \approx 40$ is a well-conditioned system: with $T_{rx}$ held known,
three distinct $\Gamma$ make the per-channel $3\times 3$ square, and the antenna
counts as a source like any other because its $T_\mathrm{src}$ is known. Fit
$T_{rx}$ as well and the per-channel system is $4\times 4$ and wants a fourth
load. Drop to one source and $\kappa$
jumps to $\sim 4\times 10^{6}$, at which point `wiener_solve`'s guard
[refuses to answer](inference.md#conditioning-why-a-residual-is-not-an-accuracy)
rather than returning a prior-driven posterior that looks converged.

---

## 6. One differentiable object

From the HEALPix sky map, through the beam convolution, the horizon split, the
ohmic loss, the switch and Eq. 1 — one gradient, no finite differences anywhere.
Parameters are reached by their **graph node**, wherever `assemble()` folded
them:

```python
pipeline = eqx.tree_at(lambda t: t["observed_astro_sky"].sky_model.maps,
                       twin, new_maps)
pipeline = eqx.tree_at(lambda t: t["antenna_loss"].efficiency, pipeline, eta)
```

```text
d(sum P^2)/d(sky pixel):  4.042e+01  (nonzero on 24576/24576 pixels)
d(sum P^2)/d(eta):        4.752e+08
```

---

## The horizon split, measured

$f_\mathrm{sky}$ gets its own section because it was got wrong twice before it
was got right, and only measurement settled it. With a truncated beam (or
`horizon_mask=True`) the projector gives the beam average over the *visible*
sky, and the rest of the antenna temperature is ground:

$$
T_\mathrm{collected} \;=\; f_\mathrm{sky}\,
  \bigl\langle T_\mathrm{sky} \bigr\rangle_\mathrm{masked}
  \;+\; \bigl(1 - f_\mathrm{sky}\bigr)\, T_\mathrm{ground},
\qquad
f_\mathrm{sky} = \frac{\int_\mathrm{above} B \,d\Omega}
                      {\int_{4\pi} B \,d\Omega}.
$$

`BeamSpillOperator` applies **both** halves, so the weights sum to one by
construction — split across a weight here and a `GroundPickupOperator` there,
the two numbers can drift apart and nothing structural would notice.

The reference that settled it: a projector run on a sky map with the ground
painted in, at latitude 90° where the local horizon coincides with the celestial
equator and stops moving with LST — so the right answer is *computed*, not
argued. Residual on a ~200 K effect at nside 16:

:::{list-table}
:header-rows: 1
:widths: 64 36

* - $f_\mathrm{sky}$ taken from
  - Residual
* - the masked beam's harmonic integral
  - −17 K
* - pixel partition, horizon ring dropped (`horizon_weights` as shipped)
  - −8.6 K
* - pixel partition, horizon ring counted as all sky
  - +8.7 K
* - **pixel partition, horizon ring counted half**
  - **+0.005 K**
:::

Two findings in that table. The band-limited masked beam's solid-angle integral
is *not* $f_\mathrm{sky}$: `map2alm` of a sharply cut map does not preserve the
mean, so it is off by ~0.7 %. And `horizon_weights` uses a strict `el > 0`,
which drops the whole ring of pixels centred exactly on the horizon — 64 of 3072
at nside 16, at *exactly* zero elevation, not nearly. A pixel centred on the
horizon is half sky and half ground; the two one-sided alternatives are
symmetric and halve with nside, which is the signature of a miscounted ring
rather than of anything harmonic.

The first implementation used the strict cut, and looked entirely reasonable.

:::{admonition} Where this lives
:class: note
All of it is **limTOD's** — `horizon_partition_weights`,
`horizon_truncated_beam` and `horizon_beam_fraction`, from 1.9 — and the locks
are in its `tests/limtod_jax/test_horizon_partition.py`. How a beam weights the
sky is limTOD's subject, the same way the noise-wave data model is
`rhino_cal_jax`'s; this package supplies the *placement*, and
`BeamSpillOperator` consumes $f_\mathrm{sky}$ without computing it. See D20.
:::

---

## What is tested

:::{list-table}
:header-rows: 1
:widths: 40 60

* - File
  - Covers
* - `tests/radio/test_sky_noise_wave_integration.py`
  - the sky really is `T_src`: a matched antenna passes it through untouched, a mismatched one attenuates it by exactly $c_s$, the receiver terms do not scale with it, load samples never see it, each extra load is its own switch position, and a hand-wired branch reproduces `assemble()`
* - `tests/radio/test_antenna_loss.py`
  - the isothermal fixed point, the $\eta = 0$ and $\eta = 1$ limits, that the loss reaches the receiver as a changed `T_src`, and that the calibration loads are downstream of it
* - `tests/radio/test_beam_spill.py`
  - the painted-ground closure, the horizon-ring convention, that a spill mixes without losing, and that the split reaches the astro branch and nothing else
* - `tests/radio/test_beams.py`
  - the CST reader against synthetic files with a known answer, RHINO's own horn ($4\pi$ directivity, boresight at the pole, below-horizon share), and the limTOD seam
:::
