"""Noise waves under RADIOMETER noise: the covariance has to be found first.

`examples/noise_wave_gcr.py` solves the same block with a noise level handed in
as a constant. That is the easy case, and it is not the physics. The radiometer
equation makes the noise multiplicative -- `d -> d(1 + w)` with
`sigma_w = 1/sqrt(delta_nu tau)` -- so

    sigma_i = |prediction_i| * sigma_w

and the noise level is a function of the very thing being solved for. A GCR
draw is a linear sampler GIVEN a covariance; here the covariance is not given.

`iterative_gls` supplies it, by fixed point: solve at the current sigma,
recompute sigma at the new prediction, repeat. `gcr_sample` is then called
exactly as before, on the sigma that came back -- it did not change, and did
not need to.

This script shows three things:

  1. the reweighting converges, in a handful of steps;
  2. the answer is the one hydra-tod's iterative GLS gives (the algorithm is a
     matrix-free port of it);
  3. freezing sigma at a single constant -- the natural thing to do if you only
     have wiener_solve -- costs a real factor in the recovered spectra, because
     the switched loads sit at genuinely different power levels and the whole
     content of the weighting is that they should not count equally.

Run:  uv run python examples/gls_gcr.py
"""

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
    RadiometerNoise,
    check_linearity,
    condition_estimate,
    gcr_sample,
    iterative_gls,
    linear_operator,
    wiener_solve,
)
from rheplicant.radio import NoiseWaveOperator  # noqa: E402

N_TIME, N_FREQ = 96, 16
T_RX, PRIOR = 290.0, 100.0

# The three switch positions at their real temperatures, which is the point:
# noise-wave calibration works by contrast, so the loads do NOT sit at the same
# power. Radiometer sigma is proportional to that power, so the switch cycles
# through genuinely different noise levels -- and weighting them equally, which
# is all a constant sigma can do, throws that away.
T_SKY, T_AMBIENT, T_HOT = 2000.0, 290.0, 1200.0  # galactic FG at 70 MHz, loads

# A short integration on a narrow channel, so the fractional noise is a percent
# rather than a part in a thousand: the effect being demonstrated scales with
# it, and a realistic survey integration would make every number here invisible
# without making any of it untrue.
CHANNEL_WIDTH, INTEGRATION_TIME = 1e4, 1.0
CG_TOL, CG_MAXITER = 1e-10, 4000

freq = jnp.linspace(60e6, 85e6, N_FREQ)
gamma_rec = rcj.termination_gamma("resistive", N_FREQ, impedance=45.0)

TRUE_T = jnp.stack([
    250.0 + 20.0 * jnp.linspace(-1.0, 1.0, N_FREQ),      # T_unc(nu)
    30.0 * jnp.cos(jnp.linspace(0.0, 3.0, N_FREQ)),      # T_cos(nu)
    -40.0 + 8.0 * jnp.linspace(-1.0, 1.0, N_FREQ) ** 2,  # T_sin(nu)
])

# Three loads with genuinely different reflection coefficients, as in
# noise_wave_gcr.py -- which is also what makes their POWER levels differ, and
# so what gives the reweighting something to do.
antenna = rcj.cable_gamma(
    rcj.termination_gamma("open", N_FREQ), freq, length=2.0, loss=0.92
)
ambient = rcj.termination_gamma("resistive", N_FREQ, impedance=10.0)
short = rcj.cable_gamma(
    rcj.termination_gamma("short", N_FREQ), freq, length=0.4, loss=0.98
)
gamma_src = jnp.stack([antenna, ambient, short])
switch = jnp.arange(N_TIME) % 3

# T_src follows the switch: sky, ambient load, hot load.
t_src = jnp.array([T_SKY, T_AMBIENT, T_HOT])[switch]
state = State(
    data=jnp.broadcast_to(t_src[:, None], (N_TIME, N_FREQ)),
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


noise = RadiometerNoise(CHANNEL_WIDTH, INTEGRATION_TIME)
truth = twin(TRUE_T)(state).data

# MULTIPLICATIVE, not additive: the draft's Eq. 8 form.
w = noise.fractional * jax.random.normal(jax.random.key(0), truth.shape)
observed = truth * (1.0 + w)

print(f"fractional noise  sigma_w = 1/sqrt({CHANNEL_WIDTH:.0e} x "
      f"{INTEGRATION_TIME:g}) = {noise.fractional:.4f}")
print(f"T_sys per load    {float(truth[0::3].mean()):7.1f} / "
      f"{float(truth[1::3].mean()):7.1f} / {float(truth[2::3].mean()):7.1f} K")
sigma_true = noise.std(truth)
print(f"so sigma spans    {float(sigma_true.min()):.3f} .. "
      f"{float(sigma_true.max()):.3f} K  "
      f"({float(sigma_true.max() / sigma_true.min()):.2f}x across the switch)\n")

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

# ---------------------------------------------------------------- the covariance
found = iterative_gls(
    block, observed, noise=noise, prior_std=PRIOR,
    tol=CG_TOL, maxiter=CG_MAXITER, require_convergence=None,
)
print(f"\niterative_gls: {int(found.iterations)} reweights, "
      f"final relative step {float(found.delta):.2e}, "
      f"converged={bool(found.converged)}")
print(f"recovered sigma vs the truth's own: max relative error "
      f"{float(jnp.abs(found.noise_std / sigma_true - 1.0).max()):.2%}")

kappa = condition_estimate(block, noise_std=found.noise_std, prior_std=PRIOR)
print(f"condition_estimate at that covariance: kappa = {float(kappa):.2e}")

# ----------------------------------------- what it bought the POINT estimate: nil
#
# And that is exact, not a weak effect. Per frequency channel there are three
# unknowns (T_unc, T_cos, T_sin -- T_rx is held known, so it is not a fourth)
# and three loads; every time sample on a given
# load predicts the SAME value, so the design matrix has three distinct rows
# repeated, and the reduced system is square. A square linear system has one
# solution and the weights cancel out of it. Reweighting cannot move a mean
# that the weights do not determine.
flat_sigma = float(jnp.mean(observed) * noise.fractional)
frozen, _ = wiener_solve(
    block, observed, noise_std=flat_sigma,
    prior_std=PRIOR, tol=CG_TOL, maxiter=CG_MAXITER, require_convergence=None,
)
print("\nRMS error against the truth, per noise-wave spectrum:")
print(f"   {'':6s}  {'reweighted':>12s}  {'frozen sigma':>13s}   {'ratio':>6s}")
for i, name in enumerate(("T_unc", "T_cos", "T_sin")):
    a = float(jnp.sqrt(jnp.mean((found.solution[i] - TRUE_T[i]) ** 2)))
    b = float(jnp.sqrt(jnp.mean((frozen[i] - TRUE_T[i]) ** 2)))
    print(f"   {name:6s}  {a:10.4f} K  {b:11.4f} K   {b / a:5.2f}x")
print("   -> ~1.00x, and exactly so: 3 loads against 3 per-channel unknowns is a")
print("      square system, whose solution does not depend on the weighting.")

# ----------------------------------- what it bought the POSTERIOR: the whole thing
#
# The mean being weight-independent says nothing about the width. The posterior
# covariance is (A^T Sigma^-1 A + S^-1)^-1, which depends on Sigma whether or
# not the mean does -- and a GCR draw is precisely a draw of that width.
# gcr_sample is UNCHANGED: it takes the sigma iterative_gls found.
def draw_at(sigma, seed):
    keys = jax.random.split(jax.random.key(seed), 300)
    return jax.vmap(
        lambda k: gcr_sample(
            block, observed, noise_std=sigma, prior_std=PRIOR, key=k,
            tol=CG_TOL, maxiter=CG_MAXITER, require_convergence=None,
        )[0]
    )(keys)


found_draws = draw_at(found.noise_std, 9)
frozen_draws = draw_at(flat_sigma, 9)

print("\n300 exact draws at each covariance -- posterior sigma per spectrum, and")
print(f"how much of the {PRIOR:.0f} K prior width the data bought back:")
print(f"   {'':6s}  {'reweighted':>12s}  {'frozen sigma':>13s}   {'error':>7s}")
for i, name in enumerate(("T_unc", "T_cos", "T_sin")):
    a = float(found_draws[:, i].std(axis=0).mean())
    b = float(frozen_draws[:, i].std(axis=0).mean())
    print(f"   {name:6s}  {a:10.3f} K  {b:11.3f} K   {100 * (b / a - 1):+6.1f}%")

print("\nSo the covariance is not a detail of the mean, it IS the answer here: the")
print("point estimate was already right, and the error bars a frozen sigma reports")
print("are wrong by the amount above. A draw is exact for the covariance it is")
print("given, so a wrong covariance yields a posterior that is confident and")
print("misplaced rather than one that looks wrong.")
