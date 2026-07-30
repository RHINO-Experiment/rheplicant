"""RHINO's horn on the sky, feeding the noise-wave receiver — end to end.

limTOD says what the antenna sees. The Noise-Wave GCR draft's Eq. 1 says what
the receiver does with it::

    T_sys = T_src c_s + T_unc k_unc + T_cos k_cos + T_sin k_sin + T_rx

They meet at one quantity, ``T_src``. On antenna samples it is the drift-scan
TOD of the real RHINO horn — a CST far-field export, beam-convolved through the
m-mode engine; on load samples the switch replaces it, and takes the load's
reflection coefficient with it. In between sits the antenna's own ohmic loss,
which is a different loss from the receiver's mismatch and must not be confused
with it.

Six steps:

1. the RHINO horn: CST -> HEALPix -> drift-scan sky, checked to BE a temperature;
2. the antenna chain, assembled from the graph, cross-checked against Eq. 1;
3. what the receiver did to the sky (mismatch loss vs ohmic loss);
4. a real switching cycle: antenna + two loads + Eq. 8 radiometer noise;
5. close the loop — solve for the noise waves with the sky known;
6. differentiate the whole thing, sky map included.

Needs both wings:  pip install "rheplicant[limtod]" \
                       "rhino-cal-jax @ git+https://github.com/RHINO-Experiment/rhino-cal.git"
Run:  uv run python examples/sky_to_noise_wave.py
      uv run python examples/sky_to_noise_wave.py --beam-dir /path/to/CST_beams/HornDryGround
"""

import argparse
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)  # limTOD's map<->alm steps want it

import equinox as eqx  # noqa: E402
import healpy as hp  # noqa: E402  (a limTOD dependency; here only for pix2ang)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import rhino_cal_jax as rcj  # noqa: E402

from rheplicant import (  # noqa: E402
    Coordinates,
    Pipeline,
    SelectOperator,
    State,
    SumOperator,
)
from rheplicant.inference import (  # noqa: E402
    Bind,
    Latent,
    ParameterSpace,
    check_linearity,
    condition_estimate,
    linear_operator,
    wiener_solve,
)
from rheplicant.radio import (  # noqa: E402
    AntennaLossOperator,
    AtmosphericEmissionOperator,
    CalLoadOperator,
    GroundPickupOperator,
    NoiseWaveOperator,
    SkySourceOperator,
    assemble,
    cst_beam_maps,
)
from rheplicant.radio.sky import DriftScanProjector  # noqa: E402
from rheplicant.radio.sky.model import AbstractSkyModel  # noqa: E402

RHINO_BEAMS = Path("~/Dataspace/RHINO/CST_beams/HornDryGround").expanduser()

NSIDE, N_FREQ = 16, 8
LMAX = 3 * NSIDE - 1
N_PIX = 12 * NSIDE**2
N_TIME = 96
LAT_DEG, AZ_DEG, EL_DEG = 53.2, 0.0, 90.0      # RHINO's latitude, zenith drift
ETA, T_PHYS = 0.97, 293.0                      # radiation efficiency, horn temperature
T_RX, T_AMBIENT, T_HOT = 290.0, 300.0, 400.0
T_GROUND, GROUND_COUPLING, T_ATM = 290.0, 0.02, 3.0
DELTA_NU, T_INT = 25e6 / N_FREQ, 2.0           # channel bandwidth [Hz], dump [s]

parser = argparse.ArgumentParser()
parser.add_argument("--beam-dir", type=Path, default=RHINO_BEAMS,
                    help="directory of per-frequency CST far-field exports")
args = parser.parse_args()

freq = jnp.linspace(60e6, 85e6, N_FREQ)


class MapSky(AbstractSkyModel):
    """Fixed brightness maps — stand-in for a GSM/pyGDSM realisation."""

    maps: jax.Array

    def __call__(self, freq: jax.Array) -> jax.Array:
        return self.maps


# ============================================== 1. the horn, and the sky ----
# CST exports directivity in dBi on a (theta, phi) grid, one file per frequency.
# cst_beam_maps reads them onto HEALPix in limTOD's beam-local convention
# (boresight at the pole) as linear power, and interpolates in frequency.
if args.beam_dir.is_dir():
    beam_maps = jnp.asarray(cst_beam_maps(args.beam_dir, freq, nside=NSIDE))
    beam_label = f"RHINO horn, {args.beam_dir.name}"
else:  # keeps the script runnable without the (unpublished) CST dataset
    theta = jnp.arccos(1.0 - 2.0 * (jnp.arange(N_PIX) + 0.5) / N_PIX)
    beam_maps = jnp.stack([jnp.exp(-0.5 * (theta / 0.40) ** 2)] * N_FREQ)
    beam_label = f"Gaussian stand-in ({args.beam_dir} not found)"

structure = jax.random.normal(jax.random.key(0), (N_PIX,))
sky_maps = jnp.stack([
    rcj.synchrotron_temperature(nu) * (1.0 + 0.15 * structure) for nu in freq
])

# normalize_beam=True is limTOD's own switch and the whole reason the projector
# output is a TEMPERATURE: it returns int(B T)/int(B), not int(B T). Normalizing
# the beam map by hand instead leaves a percent-level bias, because the
# band-limit truncates the denominator too — see docs/sky-engines.md.
projector = DriftScanProjector.from_beam_maps(
    beam_maps, lat_deg=LAT_DEG, az_deg=AZ_DEG, el_deg=EL_DEG,
    lmax=LMAX, normalize_beam=True,
).to_reference_frame(lst_ref_deg=0.0)          # pay the Wigner rotation once

lst_deg = 360.0 * jnp.arange(N_TIME) / N_TIME
switch = jnp.arange(N_TIME) % 3                # 0 antenna, 1 ambient, 2 hot
coords = Coordinates(time=jnp.arange(float(N_TIME)) * T_INT, freq=freq,
                     extra={"lst_deg": lst_deg, "receiver_input": switch})

peak_dbi = 10.0 * np.log10(np.asarray(beam_maps).max(axis=1))
theta_pix, _ = hp.pix2ang(NSIDE, np.arange(N_PIX))
maps_np = np.asarray(beam_maps)
below = maps_np[:, theta_pix > np.pi / 2].sum(axis=1) / maps_np.sum(axis=1)
print(f"beam: {beam_label}")
print(f"      peak {peak_dbi.min():.2f}..{peak_dbi.max():.2f} dBi over "
      f"{float(freq[0]) / 1e6:.0f}-{float(freq[-1]) / 1e6:.0f} MHz, "
      f"{100 * below.min():.1f}-{100 * below.max():.1f}% of the response "
      "below the horizon")

# A uniform sky is the one case whose beam average is known by definition.
uniform = jnp.full((N_FREQ, N_PIX), 200.0)
print(f"      uniform 200 K sky -> {float(projector.forward(uniform, coords).mean()):.4f} K"
      "  (this is what makes it a temperature)")

t_sky = projector.forward(sky_maps, coords)
print(f"drift-scan sky: {t_sky.shape}  "
      f"{float(t_sky.min()):.0f} .. {float(t_sky.max()):.0f} K\n")

# ================================ 2. the antenna chain, from the graph ------
# Reflection coefficients: the antenna's is structured (a cable to an open),
# the loads' are not. They must be stacked in the SAME order the switch indexes
# the selector's branches — printed below, never assumed.
gamma_ant = rcj.cable_gamma(
    rcj.termination_gamma("open", N_FREQ), freq, length=2.0, loss=0.92
)
gamma_ambient = rcj.termination_gamma("resistive", N_FREQ, impedance=10.0)
gamma_hot = rcj.cable_gamma(
    rcj.termination_gamma("short", N_FREQ), freq, length=0.4, loss=0.98
)
gamma_rec = rcj.termination_gamma("resistive", N_FREQ, impedance=45.0)

TRUE_T = jnp.stack([
    250.0 + 20.0 * jnp.linspace(-1.0, 1.0, N_FREQ),      # T_unc(nu)
    30.0 * jnp.cos(jnp.linspace(0.0, 3.0, N_FREQ)),      # T_cos(nu)
    -40.0 + 8.0 * jnp.linspace(-1.0, 1.0, N_FREQ) ** 2,  # T_sin(nu)
])


def antenna_sources():
    """The three things that reach the antenna terminals, before its own loss."""
    return (
        SkySourceOperator(sky_model=MapSky(sky_maps), projector=projector),
        GroundPickupOperator(coupling=jnp.array(GROUND_COUPLING),
                             t_ground=jnp.array(T_GROUND)),
        AtmosphericEmissionOperator(t_atm=jnp.array(T_ATM)),
    )


def antenna_loss():
    return AntennaLossOperator(efficiency=jnp.array(ETA),
                               t_physical=jnp.array(T_PHYS))


def receiver(t_nw, gamma_src):
    return NoiseWaveOperator(
        t_unc=t_nw[0], t_cos=t_nw[1], t_sin=t_nw[2], t_rx=jnp.array(T_RX),
        gamma_src_re=gamma_src.real, gamma_src_im=gamma_src.imag,
        gamma_rec_re=gamma_rec.real, gamma_rec_im=gamma_rec.imag,
    )


# The graph knows where each piece goes. antenna_loss sits on the trunk between
# t_ant_sum and the switch: it acts on everything the beam collected, and on
# nothing that connects downstream of the antenna — which is why the loads,
# entering at receiver_input, arrive unattenuated.
two_branch = assemble(
    *antenna_sources(),
    antenna_loss(),
    CalLoadOperator(t_load=jnp.array(T_AMBIENT)),
    receiver(TRUE_T, jnp.stack([gamma_ant, gamma_ambient])),
)
print(two_branch)
print("switch order:", two_branch["receiver_input"].names,
      "<- gamma_src rows must be stacked in THIS order\n")

# Cross-check: Eq. 1 spelled out by hand, with the antenna chain in T_src's place.
ab_switch = jnp.arange(N_TIME) % 2
ab_coords = coords.replace(extra={"lst_deg": lst_deg, "receiver_input": ab_switch})
assembled = eqx.filter_jit(two_branch)(State(coords=ab_coords)).data

t_collected = (projector.forward(sky_maps, ab_coords)
               + GROUND_COUPLING * T_GROUND + T_ATM)
t_ant = ETA * t_collected + (1.0 - ETA) * T_PHYS
by_hand = rcj.system_temperature(
    rcj.Couplings.from_stacked(
        rcj.couplings(jnp.stack([gamma_ant, gamma_ambient]), gamma_rec)
        .stacked[ab_switch]
    ),
    t_src=jnp.where((ab_switch == 0)[:, None], t_ant, T_AMBIENT),
    t_unc=TRUE_T[0], t_cos=TRUE_T[1], t_sin=TRUE_T[2], t_rx=jnp.array(T_RX),
)
worst = float(jnp.max(jnp.abs(assembled - by_hand)) / jnp.max(jnp.abs(by_hand)))
print(f"assembled twin vs Eq. 1 by hand: {worst:.1e} relative — roundoff\n")

# =============================== 3. two different losses, not one ----------
# c_s = (1 - |G_src|^2)|F|^2 is the impedance MISMATCH at the receiver input: it
# attenuates and adds nothing. eta is ohmic dissipation INSIDE the horn: it
# attenuates AND adds (1 - eta) T_phys. They multiply; neither substitutes for
# the other, and folding one into the other loses the additive term.
coup = rcj.couplings(jnp.stack([gamma_ant, gamma_ambient, gamma_hot]), gamma_rec)
print("what happens to the sky, in order:")
print(f"   collected by the beam        {float(t_sky.mean()):8.1f} K")
print(f"   after ohmic loss (eta={ETA})  "
      f"{float(ETA * t_sky.mean() + (1 - ETA) * T_PHYS):8.1f} K   "
      f"(-{float((1 - ETA) * (t_sky.mean() - T_PHYS)):.0f} K, and "
      f"+{(1 - ETA) * T_PHYS:.0f} K of the horn's own emission)")
print(f"   after mismatch (c_s={float(coup.c_src[0].mean()):.3f})     "
      f"{float(coup.c_src[0].mean() * (ETA * t_sky.mean() + (1 - ETA) * T_PHYS)):8.1f} K")
print(f"   c_s per source: antenna {float(coup.c_src[0].mean()):.3f}   "
      f"ambient {float(coup.c_src[1].mean()):.3f}   "
      f"hot {float(coup.c_src[2].mean()):.3f}\n")

# ======================================= 4. a real switching cycle ---------
# Three sources: the antenna and two loads. assemble() cannot express this yet
# (the cal_loads node has no many=True), so the selector is built directly —
# the documented workaround, and the same operator assemble() would have made.
# SumOperator, not Pipeline, for the three sources: a Pipeline of source-type
# operators REPLACES the data at each stage, so the sky and the ground would be
# silently overwritten by the atmosphere. That is what the graph's t_ant_sum
# junction is for, and it is why the assertion below compares the hand-built
# branch against the assembled one rather than trusting it.
def twin(t_nw):
    return Pipeline(
        SelectOperator(
            Pipeline(
                SumOperator(*antenna_sources(),
                            names=("sky", "ground", "atmosphere")),
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


state = State(coords=coords, meta={"telescope": "RHINO"})

# The hand-built selector must reproduce what assemble() builds, on the samples
# the two configurations share (source 0, the antenna). Cheap, and it is the
# only thing standing between a hand-wired branch and a silent mis-assembly.
shared = eqx.filter_jit(twin(TRUE_T))(
    State(coords=coords.replace(extra={"lst_deg": lst_deg,
                                       "receiver_input": jnp.zeros(N_TIME, int)}))
).data
antenna_only = eqx.filter_jit(two_branch)(
    State(coords=coords.replace(extra={"lst_deg": lst_deg,
                                       "receiver_input": jnp.zeros(N_TIME, int)}))
).data
assert jnp.allclose(shared, antenna_only, rtol=1e-12), (
    "hand-built antenna branch disagrees with assemble()"
)

truth = eqx.filter_jit(twin(TRUE_T))(state).data
observed = rcj.add_radiometer_noise(truth, jax.random.key(1),
                                    t_int=T_INT, delta_nu=DELTA_NU)

# Eq. 8's noise is FRACTIONAL: sigma = T_sys / sqrt(delta_nu * t_int), so it is
# ~2x larger on antenna samples than on the loads here. A scalar sigma would
# weight them equally and throw that away; the observed power is the standard
# estimator of the per-sample sigma (slightly biased by the noise it estimates,
# which is the price of not knowing T_sys in advance).
noise_std = observed / (DELTA_NU * T_INT) ** 0.5
print(f"simulated waterfall: {observed.shape}, {float(observed.mean()):.1f} K "
      f"mean, sigma {float(noise_std.min()):.3f}..{float(noise_std.max()):.3f} K "
      "(Eq. 8, fractional)")
for index, name in enumerate(("antenna", "ambient", "hot")):
    rows = observed[switch == index]
    print(f"   {name:8s} {rows.shape[0]:3d} samples   "
          f"{float(rows.mean()):8.2f} K mean   {float(rows.std()):6.2f} K rms")
print()

# ============================ 5. close the loop: solve for the noise waves --
# The sky is DATA here, not a parameter — limTOD supplies it and its only job is
# to be right. Which is why the block stays exactly linear in the noise-wave
# temperatures and needs no gradient sampler.
#
# The switch is what makes it solvable: each position gives one equation per
# channel, so three distinct Gamma make the per-channel 3x3 system square. The
# antenna counts as a source like any other because its T_src is known.
space = ParameterSpace(
    latents=[Latent("t_nw", init=jnp.full((3, N_FREQ), 100.0), linear=True)],
    bindings=[
        Bind("t_nw", into=(lambda p: p["noise_wave"].t_unc,
                           lambda p: p["noise_wave"].t_cos,
                           lambda p: p["noise_wave"].t_sin),
             fn=lambda v: (v[0], v[1], v[2])),
    ],
)
start = twin(jnp.zeros((3, N_FREQ)))
errors = check_linearity(space, start, state)
print(f"linearity check: worst relative departure {max(errors.values()):.1e}")

block = linear_operator(space, start, state)
kappa = condition_estimate(block, noise_std=noise_std, prior_std=100.0)
print(f"condition_estimate: kappa = {float(kappa):.2e}")

solved, residual = wiener_solve(block, observed, noise_std=noise_std,
                                prior_std=100.0, tol=1e-10, maxiter=4000)
print(f"\nWiener mean (Eq. 30), CG residual {float(residual):.1e}:")
for i, name in enumerate(("T_unc", "T_cos", "T_sin")):
    err = float(jnp.sqrt(jnp.mean((solved[i] - TRUE_T[i]) ** 2)))
    span = float(TRUE_T[i].max() - TRUE_T[i].min())
    print(f"   {name:6s} RMS error {err:7.3f} K   "
          f"({100 * err / max(span, 1e-9):5.2f}% of its {span:.0f} K spread)")

# ======================================= 6. one differentiable object ------
# From the HEALPix sky map through the beam convolution, the ohmic loss, the
# switch and Eq. 1 — one gradient, no finite differences anywhere.
def total_power(maps, efficiency):
    pipeline = twin(TRUE_T)
    pipeline = eqx.tree_at(
        lambda p: p["receiver_input"].branches[0]["t_ant_sum"]["sky"].sky_model.maps,
        pipeline, maps,
    )
    pipeline = eqx.tree_at(
        lambda p: p["receiver_input"].branches[0]["antenna_loss"].efficiency,
        pipeline, efficiency,
    )
    return jnp.sum(pipeline(state).data ** 2)


d_maps, d_eta = jax.grad(total_power, argnums=(0, 1))(sky_maps, jnp.array(ETA))
print(f"\nd(sum P^2)/d(sky pixel):  {float(jnp.abs(d_maps).max()):.3e}  "
      f"(nonzero on {int(jnp.sum(jnp.abs(d_maps) > 0))}/{d_maps.size} pixels)")
print(f"d(sum P^2)/d(eta):        {float(d_eta):.3e}")

print(f"\nKNOWN GAP: {100 * below.min():.1f}-{100 * below.max():.1f}% of the "
      "horn's response is below the horizon and this\nrun lets it see "
      "celestial sky rather than ground. The "
      "correct split needs the\nsky branch weighted by the above-horizon beam "
      "fraction, which no graph node\napplies yet — ground_pickup adds the "
      "ground term but nothing scales the sky.")
print("\nOne object: RHINO's horn, limTOD's sky, rhino-cal-jax's Eq. 1, "
      "rheplicant's graph.")
