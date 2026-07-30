# From the sky to the receiver

The [sky engines](sky-engines.md) produce an antenna temperature. The
[noise-wave model](operators.md#instrument-trunk-order--graph-order) consumes
one. This page joins them, end to end, on RHINO's actual horn — and is a
walkthrough of
[`examples/sky_to_noise_wave.py`](https://github.com/RHINO-Experiment/rheplicant/blob/main/examples/sky_to_noise_wave.py),
which runs everything shown below.

```bash
uv run python examples/sky_to_noise_wave.py
```

The whole thing rests on one identification. The Noise-Wave GCR draft's Eq. 1
is

$$T_\mathrm{sys} = T_\mathrm{src}\,c_s + T_\mathrm{unc}\,\kappa_\mathrm{unc}
  + T_\cos\,\kappa_\cos + T_\sin\,\kappa_\sin + T_\mathrm{rx}$$

and $T_\mathrm{src}$ is whatever the receiver input is connected to. On antenna
samples that is the beam-convolved sky; on load samples the switch replaces it,
and takes the load's reflection coefficient $\Gamma$ with it. The graph already
encodes that ordering — `observed_astro_sky → t_ant_sum → antenna_loss →
receiver_input → noise_wave` — so no glue code is needed. What *is* needed is
care at three joins, each of which returns a finite, correctly-shaped, wrong
answer when you get it wrong. They are called out as you reach them.

---

## 1. The horn

RHINO ships its horn as CST Studio far-field ASCII exports, one file per
frequency, holding total directivity in dBi on a regular $(\theta, \phi)$ grid.
{func}`~rheplicant.radio.beams.cst_beam_maps` reads them onto HEALPix in
limTOD's beam-local convention (boresight at the pole), as linear power, and
interpolates between the bracketing files in frequency:

```python
from rheplicant.radio import cst_beam_maps
from rheplicant.radio.sky import DriftScanProjector

NSIDE, N_FREQ, LMAX = 16, 8, 47
freq = jnp.linspace(60e6, 85e6, N_FREQ)

beam_maps = jnp.asarray(cst_beam_maps(
    "~/Dataspace/RHINO/CST_beams/HornDryGround", freq, nside=NSIDE,
))

local = DriftScanProjector.from_beam_maps(
    beam_maps, lat_deg=53.2, az_deg=0.0, el_deg=90.0,
    lmax=LMAX, normalize_beam=True, horizon_mask=True, apod_deg=3.0,
)
f_sky = local.horizon_fraction()               # read it BEFORE caching
projector = local.to_reference_frame(lst_ref_deg=0.0)   # Wigner rotation once
```

:::{admonition} Join 1 — `normalize_beam` decides whether `T_src` is a temperature
:class: warning
Both sky engines default to `normalize_beam=False`, matching numpy limTOD: the
forward model then returns $\int B T\,d\Omega$, not $\int B T\,d\Omega / \int B\,d\Omega$.
The first is not a temperature. Use limTOD's own switch rather than normalizing
the beam map by hand — a hand-normalized beam is *still* biased at the percent
level, because the band-limit truncates the denominator too. The numbers, and
why `normalize_beam=True` cancels exactly, are in
[sky engines](sky-engines.md#is-the-output-a-temperature).
:::

The example checks this rather than asserting it, against the one sky whose
beam average is known by definition — a uniform one:

```text
beam: RHINO horn, HornDryGround
      peak 12.14..14.38 dBi over 60-85 MHz, 1.0-3.1% of the response below the horizon
      f_sky = 0.9661..0.9896 (above-horizon share of the beam; the rest sees ground)
      uniform 200 K sky -> 200.0000 K  (this is what makes it a temperature)
drift-scan sky: (96, 8)  1869 .. 4682 K
```

Four thousand kelvin is not a typo: at 60–85 MHz the Galactic synchrotron sky
really is that bright, which is why the calibration problem is hard.

---

## 2. The antenna chain, assembled from the graph

Three things reach the antenna terminals — the beam-convolved sky, ground spill,
atmospheric emission — and then the antenna's own conductor loss acts on all
three at once:

```python
from rheplicant.radio import (
    AntennaLossOperator, AtmosphericEmissionOperator, BeamSpillOperator,
    CalLoadOperator, NoiseWaveOperator, SkySourceOperator, assemble,
)

twin = assemble(
    SkySourceOperator(sky_model=MapSky(sky_maps), projector=projector),
    BeamSpillOperator(sky_fraction=f_sky, t_ground=jnp.array(290.0)),
    AtmosphericEmissionOperator(t_atm=jnp.array(3.0)),
    AntennaLossOperator(efficiency=jnp.array(0.97), t_physical=jnp.array(293.0)),
    CalLoadOperator(t_load=jnp.array(300.0)),
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
switch order: ('t_ant_sum', 'cal_loads') <- gamma_src rows must be stacked in THIS order
```

Nothing above says what connects to what — the graph does, and the relationships
are not all the same. `beam_spill` **chains after** the sky, on the astro branch;
`atmosphere` is a separate leaf that **sums** at `t_ant_sum`; `antenna_loss` is
the trunk stage after that sum; `receiver_input` is a *selector*, which is why
the load **replaces** the antenna rather than adding to it.

:::{admonition} Join 2 — `gamma_src`'s row order is the selector's branch order
:class: warning
`NoiseWaveOperator` carries $\Gamma$ per source; the selector orders its branches
by the graph's in-edge declaration. The two orderings are independent and both
objects are `(n_source, n_freq)`, so transposing them is shape-legal. Measured
cost of getting it backwards on this configuration: **46 K peak, 28 K mean** on a
545 K signal. Read the order off the assembly — `twin["receiver_input"].names` —
rather than assuming it.
:::

The example then checks the assembled twin against Eq. 1 written out by hand,
with the antenna chain in $T_\mathrm{src}$'s place:

```python
t_visible = projector.forward(sky_maps, coords)              # masked sky average
t_collected = f_sky * t_visible + (1.0 - f_sky) * T_GROUND + T_ATM
t_ant = ETA * t_collected + (1.0 - ETA) * T_PHYS
by_hand = rcj.system_temperature(
    rcj.Couplings.from_stacked(rcj.couplings(gamma_src, gamma_rec).stacked[switch]),
    t_src=jnp.where((switch == 0)[:, None], t_ant, T_AMBIENT),
    t_unc=TRUE_T[0], t_cos=TRUE_T[1], t_sin=TRUE_T[2], t_rx=jnp.array(T_RX),
)
```

```text
assembled twin vs Eq. 1 by hand: 2.7e-16 relative — roundoff
```

---

## 3. Three effects, none standing in for another

The sky is modified three times on its way to the receiver. They are easy to
confuse, they compose, and each has a distinct signature:

| | what it is | signature |
|---|---|---|
| $f_\mathrm{sky}$ (`BeamSpillOperator`) | part of the beam is looking at **ground**, not sky | **mixing, no loss** — sky and ground at the same $T$ give $T$ |
| $\eta$ (`AntennaLossOperator`) | ohmic dissipation **inside** the horn | loss **and** its own emission $(1-\eta)T_\mathrm{phys}$ |
| $c_s = (1-\lvert\Gamma_\mathrm{src}\rvert^2)\lvert F\rvert^2$ | impedance **mismatch** at the receiver input | loss, nothing added |

The first two share the arithmetic $a x + (1-a) b$ and are deliberately not one
operator: merging them would make an efficiency and a spill fraction
indistinguishable in a fit. Each pairing is pinned by an invariant rather than
by inspection — for the ohmic loss, an antenna at $T$ looking at a sky at $T$
delivers $T$ for any efficiency; for the spill, the same statement with ground
in place of the antenna.

```text
what happens to the sky, in order:
   visible-sky beam average       3026.0 K
   after horizon spill            2975.1 K   (f_sky ~ 0.981; the rest sees 290 K ground)
   after ohmic loss (eta=0.97)    2894.7 K   (+9 K of the horn's own emission)
   after mismatch (c_s=0.264)      765.3 K
   c_s per source: antenna 0.264   ambient 0.595   hot 0.080
```

---

## 4. A real switching cycle

An identifiable per-channel noise-wave fit needs three sources with genuinely
different $\Gamma$ — see [D15](design.md) for the equation counting. That is
one more calibration load than a two-position switch, and `assemble()` expresses
it directly: `cal_loads` is `many=True` and feeds only the selector, so each
`CalLoadOperator` becomes its own switch position rather than being summed with
its sibling.

```python
twin = assemble(
    SkySourceOperator(sky_model=MapSky(sky_maps), projector=projector),
    BeamSpillOperator(sky_fraction=f_sky, t_ground=jnp.array(T_GROUND)),
    AtmosphericEmissionOperator(t_atm=jnp.array(T_ATM)),
    AntennaLossOperator(efficiency=jnp.array(ETA), t_physical=jnp.array(T_PHYS)),
    CalLoadOperator(t_load=jnp.array(T_AMBIENT)),
    CalLoadOperator(t_load=jnp.array(T_HOT)),
    receiver(t_nw, jnp.stack([gamma_ant, gamma_ambient, gamma_hot])),
)
```

That is the entire wiring. Six operators and a switch cycle, and not one line
saying what connects to what — the graph holds four different relationships at
once here, and gets each right:

| | |
|---|---|
| sky, atmosphere | leaves that **sum** at `t_ant_sum` |
| `beam_spill` | **chains** after the sky, on the astro branch |
| `antenna_loss` | the **trunk** stage after the sum |
| the two loads | sibling **selector** branches — they *replace* the antenna |

:::{admonition} If you hand-wire it anyway
:class: warning
A `Pipeline` of *source-type* operators **replaces** the data at each stage;
only the last one survives. `Pipeline(sky, ground, atmosphere)` therefore
returns the atmosphere alone — the sky silently gone, the result finite and
correctly shaped. Summing is what the `t_ant_sum` junction does, and
`SumOperator` is how you say it by hand. This bug was in an earlier draft of
this example, back when multi-load switching *did* require hand-wiring, and it
was caught only because the gradient with respect to the sky map came back
exactly zero. The graph knowing the composition rules is what removed the
opportunity.
:::

Noise is the draft's Eq. 8 — fractional, $\sigma = T_\mathrm{sys}/\sqrt{\Delta\nu\,\tau}$:

```python
observed = rcj.add_radiometer_noise(truth, jax.random.key(1),
                                    t_int=T_INT, delta_nu=DELTA_NU)
noise_std = observed / (DELTA_NU * T_INT) ** 0.5   # per sample, not a scalar
```

Because the noise is multiplicative it is ~2× larger on antenna samples than on
the loads. A scalar $\sigma$ would weight them equally and throw that away.

```text
simulated waterfall: (96, 8), 776.4 K mean, sigma 0.201..0.668 K (Eq. 8, fractional)
   antenna   32 samples    1220.02 K mean   232.68 K rms
   ambient   32 samples     586.66 K mean    21.42 K rms
   hot       32 samples     522.62 K mean    14.20 K rms
```

---

## 5. Closing the loop

The sky is **data** here, not a parameter: limTOD supplies it and its only job is
to be right. That is exactly why the block stays linear in the noise-wave
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

```text
linearity check: worst relative departure 6.7e-13
condition_estimate: kappa = 4.00e+01

Wiener mean (Eq. 30), CG residual 5.1e-11:
   T_unc  RMS error   0.134 K   ( 0.33% of its 40 K spread)
   T_cos  RMS error   0.113 K   ( 0.19% of its 60 K spread)
   T_sin  RMS error   0.110 K   ( 1.40% of its  8 K spread)
```

$\kappa \approx 40$ is a well-conditioned system: three distinct $\Gamma$ make the
per-channel $3\times 3$ square, and the antenna counts as a source like any other
because its $T_\mathrm{src}$ is known. Drop to one source and $\kappa$ jumps to
$\sim 4\times 10^6$, at which point `wiener_solve`'s guard
[refuses to answer](inference.md#conditioning-why-a-residual-is-not-an-accuracy)
rather than returning a prior-driven posterior that looks converged.

---

## 6. One differentiable object

From the HEALPix sky map, through the beam convolution, the ohmic loss, the
switch and Eq. 1 — one gradient, no finite differences anywhere:

```python
d_maps, d_eta = jax.grad(total_power, argnums=(0, 1))(sky_maps, jnp.array(ETA))
```

```text
d(sum P^2)/d(sky pixel):  4.042e+01  (nonzero on 24576/24576 pixels)
d(sum P^2)/d(eta):        4.752e+08
```

---

## The horizon split, measured

1–3 % of the horn's response (frequency-dependent) is below the horizon and sees
ground, not sky. `horizon_mask=True` gives the beam average over the *visible*
sky, and the rest of the antenna temperature is ground:

$$T_\mathrm{collected} = f_\mathrm{sky}\,\langle T_\mathrm{sky}\rangle_\mathrm{masked}
  + (1 - f_\mathrm{sky})\,T_\mathrm{ground}$$

`BeamSpillOperator` applies **both** halves, so the weights sum to one by
construction — split across a weight here and a `GroundPickupOperator` there,
the two numbers can drift apart and nothing structural would notice.

$f_\mathrm{sky}$ *is* the horizon cut, measured on the beam. Three choices go
into it, and each was decided by measurement rather than argument, against a
projector run on a sky map with the ground painted in at latitude 90 — where the
local horizon coincides with the celestial equator and stops moving with LST, so
the right answer can simply be computed. Residual on a ~200 K effect:

| choice | residual at nside 16 |
|---|---|
| the masked beam's harmonic integral | −17 K |
| pixel partition, horizon ring dropped (`horizon_weights` as shipped) | −8.6 K |
| pixel partition, horizon ring counted as all sky | +8.7 K |
| **pixel partition, horizon ring counted half** | **+0.005 K** |

Two things fall out of that table. The band-limited masked beam's solid-angle
integral is *not* $f_\mathrm{sky}$: `map2alm` of a sharply cut map does not
preserve the mean, so it is off by ~0.7 %. And `limtod_jax.horizon_weights` uses
a strict `el > 0`, which drops the whole ring of pixels centred exactly on the
horizon — 64 of 3072 at nside 16. A pixel centred on the horizon is half sky and
half ground; the two one-sided alternatives are symmetric and halve with nside,
which is the signature of a miscounted ring rather than of anything harmonic.

The first implementation used the strict cut and looked entirely reasonable.

```python
local  = DriftScanProjector.from_beam_maps(..., horizon_mask=True, apod_deg=3.0)
spill  = BeamSpillOperator.from_projector(local, t_ground=jnp.array(290.0))
cached = local.to_reference_frame(lst_ref_deg=0.0)
```

`from_projector` is the one call that cannot get the weight and the sky average
out of step, because it reads the fraction off the same beam. Read it *before*
`to_reference_frame()`, which folds the mask into the cached alms and leaves no
unmasked denominator to divide by (it raises if you ask afterwards).

Note that `apod_deg` does not move $f_\mathrm{sky}$ at all, and should not: it
is a mitigation for the ringing in the masked beam used for the sky *average*,
and a tapered region does not partition a sphere.

---

## What is tested

| File | Covers |
|---|---|
| `tests/radio/test_sky_noise_wave_integration.py` | the sky really is `T_src`: a matched antenna passes it through untouched, a mismatched one attenuates it by exactly $c_s$, the receiver terms do not scale with it, load samples never see it, and the hand-wired branch reproduces `assemble()` |
| `tests/radio/test_antenna_loss.py` | the isothermal fixed point, the $\eta=0$ and $\eta=1$ limits, and that the calibration loads are downstream of the loss |
| `tests/radio/test_beam_spill.py` | the painted-ground closure above, the horizon-ring convention (including the RING-ordering identity the half-weight rests on), that a spill mixes without losing, and that the split reaches the astro branch and nothing else |
| `tests/radio/test_beams.py` | the CST reader against synthetic files with a known answer, plus RHINO's own horn: a directivity still integrates to $4\pi$, its boresight lands on the pole, and the below-horizon fraction survives resampling |
