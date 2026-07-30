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

projector = DriftScanProjector.from_beam_maps(
    beam_maps, lat_deg=53.2, az_deg=0.0, el_deg=90.0,
    lmax=LMAX, normalize_beam=True,
).to_reference_frame(lst_ref_deg=0.0)          # pay the Wigner rotation once
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
    AntennaLossOperator, AtmosphericEmissionOperator, CalLoadOperator,
    GroundPickupOperator, NoiseWaveOperator, SkySourceOperator, assemble,
)

twin = assemble(
    SkySourceOperator(sky_model=MapSky(sky_maps), projector=projector),
    GroundPickupOperator(coupling=jnp.array(0.02), t_ground=jnp.array(290.0)),
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
Assembly(graph='single-antenna', lit=['observed_astro_sky', 'ground_pickup',
  'atmosphere', 'antenna_loss', 'cal_loads', 'noise_wave'],
  skipped-as-identity=[])
switch order: ('t_ant_sum', 'cal_loads') <- gamma_src rows must be stacked in THIS order
```

Nothing above says what connects to what. The graph does: the three sources are
leaves into the `t_ant_sum` junction, `antenna_loss` is the trunk stage after
it, and `receiver_input` is a *selector*, which is why the load **replaces** the
antenna rather than adding to it.

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
t_collected = projector.forward(sky_maps, coords) + 0.02 * 290.0 + 3.0
t_ant = ETA * t_collected + (1.0 - ETA) * T_PHYS
by_hand = rcj.system_temperature(
    rcj.Couplings.from_stacked(rcj.couplings(gamma_src, gamma_rec).stacked[switch]),
    t_src=jnp.where((switch == 0)[:, None], t_ant, T_AMBIENT),
    t_unc=TRUE_T[0], t_cos=TRUE_T[1], t_sin=TRUE_T[2], t_rx=jnp.array(T_RX),
)
```

```text
assembled twin vs Eq. 1 by hand: 4.0e-16 relative — roundoff
```

---

## 3. Two different losses

The sky is attenuated twice on its way to the receiver, by two effects that are
easy to confuse and must not be merged:

| | what it is | what it does |
|---|---|---|
| $\eta$ (`AntennaLossOperator`) | ohmic dissipation **inside** the horn | attenuates **and adds** $(1-\eta)T_\mathrm{phys}$ |
| $c_s = (1-\lvert\Gamma_\mathrm{src}\rvert^2)\lvert F\rvert^2$ | impedance **mismatch** at the receiver input | attenuates, adds nothing |

They multiply. Folding an efficiency into the noise-wave couplings would be
indistinguishable from a mismatch *in the fit* while silently dropping the
additive term — the antenna's own thermal emission, which a mismatch does not
produce. The invariant that pins the pairing is thermodynamic: an antenna at
temperature $T$ looking at a sky at the same $T$ must deliver $T$, whatever its
efficiency. `AntennaLossOperator` is tested against exactly that.

```text
what happens to the sky, in order:
   collected by the beam          3026.0 K
   after ohmic loss (eta=0.97)    2944.0 K   (-82 K, and +9 K of the horn's own emission)
   after mismatch (c_s=0.264)      778.4 K
   c_s per source: antenna 0.264   ambient 0.595   hot 0.080
```

---

## 4. A real switching cycle

An identifiable per-channel noise-wave fit needs three sources with genuinely
different $\Gamma$ — see [D15](design.md) for the equation counting. That is one
more calibration load than `assemble()` can currently express (the `cal_loads`
node has no `many=True`), so the selector is built directly. It is the same
operator `assemble()` would have produced:

```python
from rheplicant import Pipeline, SelectOperator, SumOperator

twin = Pipeline(
    SelectOperator(
        Pipeline(
            SumOperator(*antenna_sources(), names=("sky", "ground", "atmosphere")),
            antenna_loss(),
            names=("t_ant_sum", "antenna_loss"),
        ),
        CalLoadOperator(t_load=jnp.array(T_AMBIENT)),
        CalLoadOperator(t_load=jnp.array(T_HOT)),
        names=("antenna", "ambient", "hot"),
        switch_key="receiver_input",
    ),
    receiver(t_nw, jnp.stack([gamma_ant, gamma_ambient, gamma_hot])),
    names=("receiver_input", "noise_wave"),
)
```

:::{admonition} Join 3 — `SumOperator`, not `Pipeline`, for the sources
:class: warning
A `Pipeline` of *source-type* operators **replaces** the data at each stage; only
the last one survives. `Pipeline(sky, ground, atmosphere)` therefore returns the
atmosphere alone — the sky silently gone, the result finite and correctly
shaped. Summing is what the `t_ant_sum` junction does, and `SumOperator` is how
you say it by hand. This bug was in the first draft of the example and was caught
only because the gradient with respect to the sky map came back exactly zero.

Whenever you hand-wire a branch that `assemble()` could otherwise have built,
check it against `assemble()`. The example does, in one assertion.
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
simulated waterfall: (96, 8), 781.3 K mean, sigma 0.201..0.674 K (Eq. 8, fractional)
   antenna   32 samples    1234.54 K mean   235.21 K rms
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
condition_estimate: kappa = 4.04e+01

Wiener mean (Eq. 30), CG residual 5.6e-11:
   T_unc  RMS error   0.135 K   ( 0.34% of its 40 K spread)
   T_cos  RMS error   0.114 K   ( 0.19% of its 60 K spread)
   T_sin  RMS error   0.110 K   ( 1.41% of its  8 K spread)
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
d(sum P^2)/d(sky pixel):  4.074e+01  (nonzero on 24576/24576 pixels)
d(sum P^2)/d(eta):        4.906e+08
```

---

## Known gap

1–3 % of the horn's response (frequency-dependent) is below the horizon, and this
configuration lets that fraction see celestial sky rather than ground. The
correct split is

$$T_\mathrm{ant} = f_\mathrm{sky}\,\langle T_\mathrm{sky}\rangle_\mathrm{masked}
  + (1 - f_\mathrm{sky})\,T_\mathrm{ground}$$

with $f_\mathrm{sky}$ the above-horizon beam fraction. `GroundPickupOperator`
supplies the second term, and `DriftScanProjector(horizon_mask=True)` supplies
$\langle T_\mathrm{sky}\rangle_\mathrm{masked}$, but **no node applies the
$f_\mathrm{sky}$ weight to the sky branch** — so using the mask on its own is no
better than not using it, and at a 3000 K sky either choice is a ~90 K bias. The
missing piece is a beam-spill split on the antenna-temperature sum; it is not
the ohmic loss wearing a different hat, and should not be expressed as one.

---

## What is tested

| File | Covers |
|---|---|
| `tests/radio/test_sky_noise_wave_integration.py` | the sky really is `T_src`: a matched antenna passes it through untouched, a mismatched one attenuates it by exactly $c_s$, the receiver terms do not scale with it, load samples never see it, and the hand-wired branch reproduces `assemble()` |
| `tests/radio/test_antenna_loss.py` | the isothermal fixed point, the $\eta=0$ and $\eta=1$ limits, and that the calibration loads are downstream of the loss |
| `tests/radio/test_beams.py` | the CST reader against synthetic files with a known answer, plus RHINO's own horn: a directivity still integrates to $4\pi$, its boresight lands on the pole, and the below-horizon fraction survives resampling |
