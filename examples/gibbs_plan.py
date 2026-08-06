"""One SamplingPlan over six latents: conjugate blocks and a gradient block.

The guided tour solves the four noise-wave temperature families in closed form
(they are exactly affine) and samples the foreground amplitude and spectral
index with NUTS (they are not). A real fit wants both at once, and that is what
``SamplingPlan`` declares: a partition into blocks, with each block's engine
DERIVED from ``Latent(..., linear=True)`` rather than restated.

What this script is really about is the refusal in the middle. Over the tour's
own twin -- antenna plus three calibration loads -- the six latents are exactly
degenerate: per channel the four temperature families map bijectively onto the
four switch positions' levels, so they can reproduce ANY antenna-position
spectrum, which is precisely what the foreground's two parameters produce.
``identifiability`` reports nullity 2 of 34, singular values 1.6e-16 and 1.0e-16
against 1.82, and ``plan.estimate`` refuses by name.

The repair is design, not tolerance: three more calibration loads, seven switch
positions, nullity 0. The tour's own tip -- the switching cycle IS the
calibration design -- one step further. Two more unknowns need more design.

Run:  uv run --frozen python examples/gibbs_plan.py    (~45 s)
Needs: rhino-cal-jax (the `cal` extra) and numpyro.
"""

import jax

jax.config.update("jax_enable_x64", True)   # this tour solves in float64

import equinox as eqx  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from rheplicant import Coordinates, Environment, State  # noqa: E402

N_TIME, N_FREQ = 64, 8
freq = jnp.linspace(60e6, 85e6, N_FREQ)              # Hz
time_s = jnp.arange(float(N_TIME)) * 2.0             # seconds from the start

# The switching cycle: antenna, then three calibration loads, round and round.
switch = jnp.arange(N_TIME) % 4

state = State(
    coords=Coordinates(time=time_s, freq=freq,
                       extra={"receiver_input": switch}),
    env=Environment(temperature=jnp.array(280.0)),   # rides along, traced
    key=jax.random.key(20260806),                    # randomness is data
    meta={"telescope": "RHINO", "obs_id": "tour-001"},
)

s2 = state.replace(meta={"telescope": "other"})   # new object, original untouched
s3 = state.with_data(jnp.zeros((N_TIME, N_FREQ)))  # shorthand for the common case
subkey, s4 = state.next_key()                     # the PRNG protocol: split, advance
raw_kept = s3.checkpoint("raw")                   # zero-copy snapshot into aux

from rheplicant import LambdaOperator  # noqa: E402
from rheplicant.radio import GainOperator  # noqa: E402

gain = GainOperator(gain=jnp.array(1.1))     # `gain` is a differentiable leaf
clip = LambdaOperator.on_data(lambda d: jnp.clip(d, 0.0, jnp.inf))

out = gain(state.with_data(jnp.ones((N_TIME, N_FREQ))))
assert jnp.allclose(out.data, 1.1)

from typing import ClassVar  # noqa: E402

from rheplicant import AbstractOperator  # noqa: E402


class CableReflectionOperator(AbstractOperator):
    """Sinusoidal ripple from a cable standing wave (example)."""

    requires: ClassVar[tuple[str, ...]] = ("data", "coords.freq")
    provides: ClassVar[tuple[str, ...]] = ("data",)
    graph_node: ClassVar[str] = "bandpass"      # its home on the graph

    amplitude: jax.Array                         # differentiable leaves
    delay: jax.Array

    def __call__(self, state: State) -> State:
        phase = 2 * jnp.pi * state.coords.freq * self.delay
        return state.with_data(state.data * (1 + self.amplitude * jnp.cos(phase)))

from rheplicant import Pipeline, SumOperator  # noqa: E402
from rheplicant.radio import ForegroundOperator, GlobalSignalOperator  # noqa: E402

sky = SumOperator(
    GlobalSignalOperator(depth=jnp.array(0.5), centre=jnp.array(75e6),
                         width=jnp.array(5e6)),
    ForegroundOperator(amplitude=jnp.array(2500.0),
                       spectral_index=jnp.array(2.55), ref_freq=70e6),
    names=("signal", "foregrounds"),
)
observed_sky = Pipeline(sky, gain, names=("sky", "gain"))(state).data

import rhino_cal_jax as rcj  # noqa: E402

from rheplicant.radio import (  # noqa: E402
    ADCOperator,
    AntennaLossOperator,
    BeamSpillOperator,
    CalLoadOperator,
    NoiseOperator,
    NoiseWaveOperator,
    ReceiverOperator,
    assemble,
)

F_SKY, T_GROUND = 0.97, 290.0            # horizon split
ETA, T_PHYS = 0.97, 293.0                # horn ohmic loss
ADC_SCALE, N_BITS = 0.25, 12             # counts per kelvin at unit gain
SIGMA_POST_GAIN = 2.0                    # thermal noise, post-gain units

gamma_rec = rcj.termination_gamma("resistive", N_FREQ, impedance=45.0)
gamma_src = jnp.stack([                  # ROW ORDER = the selector's branch order
    rcj.cable_gamma(rcj.termination_gamma("open", N_FREQ), freq, length=2.0, loss=0.92),
    rcj.termination_gamma("resistive", N_FREQ, impedance=10.0),
    rcj.cable_gamma(rcj.termination_gamma("short", N_FREQ), freq, length=0.4, loss=0.98),
    rcj.cable_gamma(rcj.termination_gamma("resistive", N_FREQ, impedance=150.0),
                    freq, length=1.1, loss=0.95),
])

TRUE = {                                 # the four noise-wave temperatures, per channel
    "t_unc": 250.0 + 20.0 * jnp.linspace(-1.0, 1.0, N_FREQ),
    "t_cos": 30.0 * jnp.cos(jnp.linspace(0.0, 3.0, N_FREQ)),
    "t_sin": -40.0 + 8.0 * jnp.linspace(-1.0, 1.0, N_FREQ) ** 2,
    "t_rx": 290.0 + 5.0 * jnp.linspace(-1.0, 1.0, N_FREQ) ** 3,
}

# The bandpass carries SHAPE (mean 1), the gain carries the level.
bandpass = 1.0 + 0.10 * jnp.cos(2 * jnp.pi * (freq - freq[0]) / (freq[-1] - freq[0]))
bandpass = bandpass / jnp.mean(bandpass)
gain_t = 1.0 + 0.02 * jnp.sin(2 * jnp.pi * time_s / 60.0)

twin = assemble(
    GlobalSignalOperator(depth=jnp.array(0.5), centre=jnp.array(75e6),
                         width=jnp.array(5e6)),
    ForegroundOperator(amplitude=jnp.array(2500.0),
                       spectral_index=jnp.array(2.55), ref_freq=70e6),
    BeamSpillOperator(sky_fraction=jnp.array(F_SKY), t_ground=jnp.array(T_GROUND)),
    AntennaLossOperator(efficiency=jnp.array(ETA), t_physical=jnp.array(T_PHYS)),
    CalLoadOperator(t_load=jnp.array(300.0)),        # ambient
    CalLoadOperator(t_load=jnp.array(400.0)),        # hot
    CalLoadOperator(t_load=jnp.array(1200.0)),       # noise source
    NoiseWaveOperator(**TRUE,
                      gamma_src_re=gamma_src.real, gamma_src_im=gamma_src.imag,
                      gamma_rec_re=gamma_rec.real, gamma_rec_im=gamma_rec.imag),
    ReceiverOperator(bandpass=bandpass),
    GainOperator(gain=gain_t),
    NoiseOperator(sigma=jnp.array(SIGMA_POST_GAIN)),
    ADCOperator(scale=jnp.array(ADC_SCALE), n_bits=N_BITS),
)
observed = twin(state).data                          # the raw waterfall

print(twin)                                   # lit nodes + nodes traversed as identity
print("switch order:", twin["receiver_input"].names)
twin["gain"]                                  # node-id access, at any nesting
twin2 = twin.replace_node("gain", GainOperator(gain=jnp.array(1.0)))
svg = twin.to_svg()                           # also .to_mermaid() / .to_html()

from rheplicant.inference import Bind, Latent, ParameterSpace  # noqa: E402

NAMES = ("t_unc", "t_cos", "t_sin", "t_rx")

space = ParameterSpace(
    latents=[Latent(n, init=jnp.zeros((N_FREQ,)), linear=True) for n in NAMES],
    bindings=[
        Bind("t_unc", into=lambda p: p["noise_wave"].t_unc),
        Bind("t_cos", into=lambda p: p["noise_wave"].t_cos),
        Bind("t_sin", into=lambda p: p["noise_wave"].t_sin),
        Bind("t_rx", into=lambda p: p["noise_wave"].t_rx),
    ],
)

from rheplicant.inference import check_linearity, linear_operator  # noqa: E402

try:
    linear_operator(space, twin, state, names=NAMES, check=False)
except Exception as exc:
    print(f"{type(exc).__name__}: {str(exc)[:70]}...")

fit_twin = twin.without("noise")          # the supported repair, one line

errors = check_linearity(space, fit_twin, state, names=NAMES)
print(f"worst relative departure from affine: {max(errors.values()):.1e}")

from rheplicant.inference import gcr_sample, wiener_solve  # noqa: E402

NOISE_STD = ADC_SCALE * SIGMA_POST_GAIN
PRIOR_STD = dict.fromkeys(NAMES, 100.0)
PRIOR_MEAN = dict.fromkeys(NAMES, 0.0)

block = linear_operator(space, fit_twin, state, names=NAMES, check=False)
solved, residual = wiener_solve(block, observed, noise_std=NOISE_STD,
                                prior_std=PRIOR_STD, prior_mean=PRIOR_MEAN,
                                tol=1e-12, maxiter=4000)

keys = jax.random.split(jax.random.key(7), 500)
draws = jax.vmap(lambda k: gcr_sample(
    block, observed, noise_std=NOISE_STD, prior_std=PRIOR_STD,
    prior_mean=PRIOR_MEAN, key=k, tol=1e-12, maxiter=4000)[0])(keys)

for name in NAMES:
    err = solved[name] - TRUE[name]
    sig = draws[name].std(axis=0)
    print(f"{name:5s} RMS err {float(jnp.sqrt(jnp.mean(err ** 2))):6.3f} K"
          f" | posterior sigma {float(sig.min()):5.2f}..{float(sig.max()):5.2f} K"
          f" | worst pull {float(jnp.max(jnp.abs(err / sig))):.2f}")

from rheplicant.inference import identifiability  # noqa: E402

report = identifiability(space, fit_twin, state)
print(f"rank {report.rank} of {report.n_par} parameters, nullity {report.nullity}")


import numpyro.distributions as dist  # noqa: E402

from rheplicant.inference import Block, SamplingPlan, split_rhat  # noqa: E402

# One space, all six latents. Only the temperatures claim to be linear.
joint = ParameterSpace(
    latents=[
        *[Latent(n, init=jnp.zeros((N_FREQ,)), linear=True,
                 prior=dist.Normal(jnp.zeros(N_FREQ), 400.0)) for n in NAMES],
        Latent("fg_log_amp", init=jnp.log(jnp.array(2000.0)),
               prior=dist.Normal(jnp.log(2000.0), 0.5)),
        Latent("fg_beta", init=jnp.array(2.30), prior=dist.Normal(2.3, 0.3)),
    ],
    bindings=[
        Bind("t_unc", into=lambda p: p["noise_wave"].t_unc),
        Bind("t_cos", into=lambda p: p["noise_wave"].t_cos),
        Bind("t_sin", into=lambda p: p["noise_wave"].t_sin),
        Bind("t_rx", into=lambda p: p["noise_wave"].t_rx),
        Bind("fg_log_amp", into=lambda p: p["foregrounds"].amplitude, fn=jnp.exp),
        Bind("fg_beta", into=lambda p: p["foregrounds"].spectral_index),
    ],
)
FG = ("fg_log_amp", "fg_beta")
plan = SamplingPlan(joint, Block(*NAMES), Block(*FG, steps=200))
print(plan)

# The tour's own four switch positions cannot carry six latents.
try:
    plan.estimate(fit_twin, state, observed, noise=NOISE_STD, max_iter=3)
except Exception as exc:
    lines = str(exc).splitlines()
    print(lines[0][:104] + " ...")
    print("\n".join(x for x in lines if x.startswith("  direction")))

# The repair is design, not tolerance: three more calibration loads.
EXTRA = ((600.0, 75.0, 0.7, 0.94), (900.0, 25.0, 1.5, 0.90),
         (150.0, 200.0, 0.25, 0.99))
gamma_7 = jnp.concatenate([gamma_src] + [
    rcj.cable_gamma(rcj.termination_gamma("resistive", N_FREQ, impedance=z),
                    freq, length=ln, loss=ls)[None] for (_, z, ln, ls) in EXTRA])
state7 = state.replace(coords=Coordinates(
    time=time_s, freq=freq, extra={"receiver_input": jnp.arange(N_TIME) % 7}))
twin7 = assemble(
    GlobalSignalOperator(depth=jnp.array(0.5), centre=jnp.array(75e6),
                         width=jnp.array(5e6)),
    ForegroundOperator(amplitude=jnp.array(2500.0),
                       spectral_index=jnp.array(2.55), ref_freq=70e6),
    BeamSpillOperator(sky_fraction=jnp.array(F_SKY), t_ground=jnp.array(T_GROUND)),
    AntennaLossOperator(efficiency=jnp.array(ETA), t_physical=jnp.array(T_PHYS)),
    CalLoadOperator(t_load=jnp.array(300.0)),
    CalLoadOperator(t_load=jnp.array(400.0)),
    CalLoadOperator(t_load=jnp.array(1200.0)),
    *[CalLoadOperator(t_load=jnp.array(t)) for (t, _, _, _) in EXTRA],
    NoiseWaveOperator(**TRUE, gamma_src_re=gamma_7.real, gamma_src_im=gamma_7.imag,
                      gamma_rec_re=gamma_rec.real, gamma_rec_im=gamma_rec.imag),
    ReceiverOperator(bandpass=bandpass),
    GainOperator(gain=gain_t),
    NoiseOperator(sigma=jnp.array(SIGMA_POST_GAIN)),
    ADCOperator(scale=jnp.array(ADC_SCALE), n_bits=N_BITS),
)
observed7 = twin7(state7).data
fit7 = twin7.without("noise")
TRUTH6 = {**TRUE, "fg_log_amp": jnp.log(jnp.array(2500.0)),
          "fg_beta": jnp.array(2.55)}

# Exit 1 -- the best fit.
est = plan.estimate(fit7, state7, observed7, noise=NOISE_STD, max_iter=45, tol=1e-3)
d = est.diagnostics
print(f"\nestimate  sweeps {d.sweeps}  converged {d.converged}  "
      f"chi2 {d.chi2[0]:.3g} -> {d.chi2[-1]:.4f}   (512 data, 34 parameters)")
print(f"  engines {dict(d.engines)}")
print(f"  per-block last number "
      f"{ {k: float(f'{v:.3g}') for k, v in d.block_residuals.items()} }")
print(f"  rank {d.identifiability.rank} of {d.identifiability.n_par}, "
      f"nullity {d.identifiability.nullity}")

# Exit 2 -- draws, started from that best fit. 26 sweeps is what fits in a
# tour; n_sweeps=120, warmup=20 gives r_hat 0.990 and sigma(fg_beta) 0.0069,
# in 81 s.
warm = eqx.tree_at(lambda s: [s.latent(n).init for n in FG], joint,
                   [est.values[n] for n in FG])
draws = SamplingPlan(warm, Block(*NAMES), Block(*FG, steps=25)).sample(
    fit7, state7, observed7, noise=NOISE_STD, key=jax.random.key(11),
    n_sweeps=26, warmup=8)
dd = draws.diagnostics
print(f"\nsample    sweeps {dd.sweeps}  warmup {dd.warmup}  kept {draws.n_draw}  "
      f"r_hat(joint chi2) {dd.rhat:.3f}  converged {dd.converged}")

print(f"\n{'latent':11s} {'engine':10s} {'max|est-truth|':>14s} {'posterior sigma':>17s}"
      f" {'worst pull':>11s} {'r_hat':>6s}")
for name in joint.names:
    chain = draws.samples[name].reshape(draws.n_draw, -1)
    sig = draws.std[name]
    err_e = float(jnp.max(jnp.abs(est.values[name] - TRUTH6[name])))
    pull = float(jnp.max(jnp.abs((draws.mean[name] - TRUTH6[name]) / sig)))
    print(f"{name:11s} {d.engines[NAMES if name in NAMES else FG]:10s} {err_e:14.4f} "
          f"{float(jnp.min(sig)):7.4f}..{float(jnp.max(sig)):6.4f} {pull:11.2f} "
          f"{max(split_rhat(t) for t in chain.T):6.3f}")
