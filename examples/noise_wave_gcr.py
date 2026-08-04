"""Noise waves as a linear block: Wiener mean and exact posterior draws.

The data model (Noise-Wave GCR note, Eq. 1) is exactly linear in the noise-wave
temperatures, so they never need a gradient sampler. This script declares that
linearity, has it CHECKED, then solves in closed form and draws exact posterior
samples -- the note's Eqs. 30 and 31 respectively.

The switching is what makes it work, and the demonstration is deliberately
per-channel. Each switch position contributes one equation per frequency
channel, so while the temperatures are free per channel the design matrix has
rank min(n_src, k) x n_freq over the k FREE temperature families. Here k = 3:
T_rx is held at its known value and only T_unc, T_cos, T_sin are fitted, so
three loads make the system square and one leaves it deficient threefold. Fit
T_rx as well and k becomes 4 and a fourth load is needed. Scalar noise waves
would be identified by a single load, which is why a scalar version of this
example would prove nothing -- and a BASIS parameterization is not covered by
this counting at all; measure that one with rheplicant.inference.
identifiability. See NoiseWaveOperator's module docstring for the full rule.

Run:  uv run python examples/noise_wave_gcr.py
      uv run python examples/noise_wave_gcr.py --one-source
"""

import argparse

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import rhino_cal_jax as rcj  # noqa: E402

from rheplicant import Coordinates, State  # noqa: E402
from rheplicant.core.pipeline import Pipeline  # noqa: E402
from rheplicant.inference import (  # noqa: E402
    Bind,
    Latent,
    ParameterSpace,
    check_linearity,
    condition_estimate,
    gcr_sample,
    linear_operator,
    wiener_solve,
)
from rheplicant.radio import NoiseWaveOperator  # noqa: E402

N_TIME, N_FREQ = 96, 16
T_RX, T_SRC, NOISE, PRIOR = 290.0, 300.0, 0.5, 100.0

# --one-source deliberately makes the per-channel normal operator (AtN^-1A +
# S^-1) severely ill-conditioned: 2 of 3 directions carry no data at all, and
# condition_estimate() below reports kappa ~ 4e6 for this run (kappa ~ 27 for
# the default, well-conditioned three-load run). wiener_solve/gcr_sample's
# guard (require_convergence, default 1e-3) bounds kappa * relative_residual,
# not the residual alone -- because a residual that LOOKS converged (CG
# settles on the one well-constrained direction, which dominates the
# aggregate residual, while the other two sit unresolved at their starting
# value) is not evidence the solution is right. At the library's own default
# tol=1e-6, kappa * residual is order-unity here, nowhere near 1e-3, so the
# guard correctly RAISES rather than silently handing back the badly-wrong
# posterior this script used to print for --one-source (~0.03 K reported
# instead of the ~75-100 K the physics demands). Tightening tol here -- to
# roughly require_convergence / kappa -- is what makes --one-source's numbers
# trustworthy without disabling the guard; see condition_estimate()'s
# docstring in rheplicant/inference/linear.py for how to choose it for a new
# block rather than guessing.
CG_TOL, CG_MAXITER = 1e-10, 4000

parser = argparse.ArgumentParser()
parser.add_argument("--one-source", action="store_true",
                    help="use a single load, to show what switching buys")
args = parser.parse_args()

freq = jnp.linspace(60e6, 85e6, N_FREQ)
gamma_rec = rcj.termination_gamma("resistive", N_FREQ, impedance=45.0)

# Truth: three smooth spectra, one per noise-wave term.
TRUE_T = jnp.stack([
    250.0 + 20.0 * jnp.linspace(-1.0, 1.0, N_FREQ),      # T_unc(nu)
    30.0 * jnp.cos(jnp.linspace(0.0, 3.0, N_FREQ)),      # T_cos(nu)
    -40.0 + 8.0 * jnp.linspace(-1.0, 1.0, N_FREQ) ** 2,  # T_sin(nu)
])

# Three loads whose reflection coefficients differ in shape, not just in level.
# `ambient` needs a real mismatch: an impedance close to the 50 ohm system
# (e.g. 52 ohm) gives |Gamma| ~ 0.02, which makes that load's row of the
# per-channel 3x3 system numerically indistinguishable from zero and leaves
# T_unc/T_sin poorly conditioned even with three "independent" loads -- caught
# by comparing this script's Wiener-mean RMS against the truth's own dynamic
# range, per boundary-validation.md. impedance=10.0 (|Gamma| ~ 0.67) gives all
# three loads real separation in (k_unc, k_cos, k_sin) space.
antenna = rcj.cable_gamma(
    rcj.termination_gamma("open", N_FREQ), freq, length=2.0, loss=0.92
)
ambient = rcj.termination_gamma("resistive", N_FREQ, impedance=10.0)
short = rcj.cable_gamma(
    rcj.termination_gamma("short", N_FREQ), freq, length=0.4, loss=0.98
)

if args.one_source:
    gamma_src, n_source = jnp.stack([antenna]), 1
else:
    gamma_src, n_source = jnp.stack([antenna, ambient, short]), 3
switch = jnp.arange(N_TIME) % n_source

state = State(
    data=jnp.full((N_TIME, N_FREQ), T_SRC),
    coords=Coordinates(time=jnp.arange(float(N_TIME)), freq=freq,
                       extra={"receiver_input": switch}),
    meta={"telescope": "RHINO"},
)


def twin(t_nw):
    return Pipeline(
        NoiseWaveOperator(
            t_unc=t_nw[0], t_cos=t_nw[1], t_sin=t_nw[2], t_rx=jnp.array(T_RX),
            gamma_src_re=gamma_src.real, gamma_src_im=gamma_src.imag,
            gamma_rec_re=gamma_rec.real, gamma_rec_im=gamma_rec.imag,
        ),
        names=("noise_wave",),
    )


truth = twin(TRUE_T)(state).data
observed = truth + NOISE * jax.random.normal(jax.random.key(0), truth.shape)
print(f"observation: {observed.shape}, {float(observed.mean()):.2f} K mean")
# k = 3 free temperature families here: T_rx is pinned at T_RX and is not a
# latent. Fitting it too would make k = 4 and the denominators below 4 x N_FREQ.
K_FREE = 3
print(f"loads: {n_source}   unknowns: {TRUE_T.size} "
      f"({K_FREE} families x {N_FREQ} channels)")
print(f"equations per channel: {n_source}   -> expected rank "
      f"{min(n_source, K_FREE) * N_FREQ}/{K_FREE * N_FREQ}\n")

# All three spectra are ONE latent of shape (3, n_freq) feeding three leaves.
# Declaring linear=True is a claim; check_linearity is what turns it into a fact.
space = ParameterSpace(
    latents=[Latent("t_nw", init=jnp.zeros((3, N_FREQ)), linear=True)],
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
kappa = condition_estimate(block, noise_std=NOISE, prior_std=PRIOR)
print(f"condition_estimate: kappa = {float(kappa):.2e} "
      f"(what require_convergence's guard multiplies the residual by)")

solved, residual = wiener_solve(
    block, observed, noise_std=NOISE, prior_std=PRIOR, tol=CG_TOL, maxiter=CG_MAXITER
)
print(f"\nWiener mean (Eq. 30), CG residual {float(residual):.1e}")
for i, name in enumerate(("T_unc", "T_cos", "T_sin")):
    err = jnp.sqrt(jnp.mean((solved[i] - TRUE_T[i]) ** 2))
    print(f"   {name:6s} RMS error {float(err):8.3f} K   "
          f"(truth spans {float(TRUE_T[i].min()):7.1f} .. {float(TRUE_T[i].max()):7.1f})")

# gcr_sample adds the two fluctuation terms of Eq. 31 to the same right-hand
# side, so every solve is an independent, exact posterior draw.
keys = jax.random.split(jax.random.key(9), 300)
draws = jax.vmap(
    lambda k: gcr_sample(
        block, observed, noise_std=NOISE, prior_std=PRIOR, key=k,
        tol=CG_TOL, maxiter=CG_MAXITER,
    )[0]
)(keys)
print("\n300 exact posterior draws (Eq. 31): per-channel sigma, and how much of")
print(f"the {PRIOR:.0f} K prior width the data bought back:")
for i, name in enumerate(("T_unc", "T_cos", "T_sin")):
    sig = draws[:, i].std(axis=0)
    print(f"   {name:6s} sigma {float(sig.min()):7.3f} .. {float(sig.max()):7.3f} K"
          f"   ({100 * float(sig.mean()) / PRIOR:5.1f}% of prior)")

if args.one_source:
    print(f"\nOne load: {N_FREQ} equations per channel-triple against "
          f"{3 * N_FREQ} unknowns.")
    print("Two of every three directions are unconstrained by the data, so the")
    print("prior sets their width. Re-run without --one-source to see it close.")
else:
    print("\nThree loads with genuinely different Gamma: the per-channel system")
    print("is square, so every direction is constrained by data and the sigmas")
    print("drop far below the prior. That is what the switch buys.")
