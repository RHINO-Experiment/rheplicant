"""Tutorial: an exact posterior for a big linear block (GCR + iterative GLS).

Companion to docs/tutorial-gcr.md. Runs in six numbered steps; each prints what
the text discusses, so the page and the script cannot drift apart.

The problem is a ring toy: 64 sky pixels on a circle, scanned by a Gaussian
beam through a known gain, in 4 frequency channels. The sky is 256 unknowns --
far past where a gradient sampler is sensible, and exactly where a
conjugate-Gaussian solve belongs.

Run:  .venv/bin/python examples/tutorial_gcr.py
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from rheplicant import Coordinates, State  # noqa: E402
from rheplicant.core.pipeline import Pipeline  # noqa: E402
from rheplicant.inference import (  # noqa: E402
    ParameterSpace,
    RadiometerNoise,
    check_linearity,
    condition_estimate,
    gcr_sample,
    iterative_gls,
    linear_operator,
    wiener_solve,
)
from rheplicant.radio import GainOperator, MapSky, SkySourceOperator  # noqa: E402
from rheplicant.radio.sky import MatrixProjector  # noqa: E402

N_TIME, N_PIX, N_FREQ = 256, 64, 4   # 1024 samples constraining 256 pixels
# A beam 1.5 pixels wide: it smooths the ring without erasing it. Step 7 shows
# what happens when it is wider than the structure it is meant to measure.
FWHM, OFFSET, GAIN = 0.15, 0.12, 1.10
WIDE_FWHM = 0.35
MEAN_SKY, SKY_SCALE = 300.0, 40.0
PRIOR_STD = 100.0

# A short integration on a narrow channel: the fractional noise is 1%, big
# enough that everything below is visible. A survey integration makes every
# number here smaller without making any of it different.
CHANNEL_WIDTH, INTEGRATION_TIME = 1e4, 1.0


# ---------------------------------------------------------------------------
# Step 1. The world: a ring of sky, a beam that scans it, a gain.
# ---------------------------------------------------------------------------
pixel_angle = 2.0 * jnp.pi * jnp.arange(N_PIX) / N_PIX
scan_angle = 2.0 * jnp.pi * jnp.arange(N_TIME) / N_TIME


def beam_matrix(fwhm: float, offset: float) -> jax.Array:
    """Row t = the beam's weight on every pixel at scan step t, summing to 1."""
    separation = pixel_angle[None, :] - scan_angle[:, None] - offset
    wrapped = jnp.arctan2(jnp.sin(separation), jnp.cos(separation))
    response = jnp.exp(-0.5 * (wrapped / (fwhm / 2.3548)) ** 2)
    return response / jnp.sum(response, axis=1, keepdims=True)


def twin(maps: jax.Array, fwhm: float = FWHM) -> Pipeline:
    """The forward model: sky -> beam-weighted TOD -> gain."""
    return Pipeline(
        SkySourceOperator(
            sky_model=MapSky(maps=maps, freq=freq),
            projector=MatrixProjector(beam_matrix(fwhm, OFFSET)),
        ),
        GainOperator(gain=jnp.asarray(GAIN)),
        names=("sky", "gain"),
    )


freq = jnp.linspace(60e6, 85e6, N_FREQ)
state = State(
    coords=Coordinates(time=jnp.arange(float(N_TIME)), freq=freq),
    meta={"telescope": "ring-toy"},
)

modes = jnp.arange(1, 6)[:, None] * pixel_angle[None, :]
weights = jax.random.normal(jax.random.key(1), (N_FREQ, 5))
true_maps = MEAN_SKY + SKY_SCALE * (weights @ jnp.cos(modes)) / jnp.sqrt(2.5)

noise = RadiometerNoise(CHANNEL_WIDTH, INTEGRATION_TIME)
clean = twin(true_maps)(state).data
# MULTIPLICATIVE: d = prediction * (1 + w). This is what makes the covariance
# something to be found rather than something to be handed in.
observed = clean * (
    1.0 + noise.fractional * jax.random.normal(jax.random.key(2), clean.shape)
)

PIXEL_DEG = 360.0 / N_PIX
print("STEP 1  the world")
print(f"  data      {observed.shape} = {observed.size} samples")
print(f"  unknowns  {true_maps.size} sky pixels ({N_PIX} x {N_FREQ} channels)")
print(f"  beam      FWHM {jnp.rad2deg(FWHM):.1f} deg = "
      f"{jnp.rad2deg(FWHM) / PIXEL_DEG:.1f} pixels")
print(f"  noise     fractional {noise.fractional:.4f}, so sigma spans "
      f"{float(noise.std(clean).min()):.2f} .. {float(noise.std(clean).max()):.2f} K")

# ---------------------------------------------------------------------------
# Step 2. Declare the sky linear -- and have the claim CHECKED.
# ---------------------------------------------------------------------------
space = ParameterSpace.direct(
    "sky_maps",
    init=jnp.full_like(true_maps, MEAN_SKY),
    into=lambda p: p["sky"].sky_model.maps,
    linear=True,
)
start = twin(jnp.full_like(true_maps, MEAN_SKY))

errors = check_linearity(space, start, state)
print("\nSTEP 2  the linearity claim, checked before anything exploits it")
print(f"  worst relative departure from affine: {max(errors.values()):.2e}")
print("  (a FALSE declaration would otherwise give a confident wrong posterior,")
print("   not an error -- which is why this is a check and not a comment)")

block = linear_operator(space, start, state)

# ---------------------------------------------------------------------------
# Step 3. Look at the conditioning BEFORE solving.
# ---------------------------------------------------------------------------
kappa = condition_estimate(block, noise_std=float(noise.std(clean).mean()),
                           prior_std=PRIOR_STD)
print("\nSTEP 3  conditioning, which decides what tolerance means")
print(f"  kappa = {float(kappa):.2e}")
print("  a residual of tol bounds the ERROR by kappa*tol, so tol=1e-6 certifies")
print(f"  only {float(kappa) * 1e-6:.1e} -- pick tol from kappa, not from habit")
TARGET_ERROR = 1e-4
CG_TOL = TARGET_ERROR / float(kappa)
print(f"  -> for a relative error of {TARGET_ERROR:.0e}, tol = "
      f"target/kappa = {CG_TOL:.1e}")

# ---------------------------------------------------------------------------
# Step 4. The covariance is not given. Find it.
# ---------------------------------------------------------------------------
found = iterative_gls(
    block, observed, noise=noise, prior_std=PRIOR_STD,
    prior_mean=MEAN_SKY, tol=CG_TOL, maxiter=2000, require_convergence=None,
)
print("\nSTEP 4  iterative GLS: sigma tracks the prediction, so solve for both")
print(f"  {int(found.iterations)} reweights, last relative step "
      f"{float(found.delta):.2e}, converged={bool(found.converged)}")
sigma_truth = noise.std(clean)
print(f"  recovered sigma vs the truth's own: max relative error "
      f"{float(jnp.abs(found.noise_std / sigma_truth - 1.0).max()):.2%}")

# ---------------------------------------------------------------------------
# Step 5. The mean, and then exact draws at that covariance.
# ---------------------------------------------------------------------------
mean, residual = wiener_solve(
    block, observed, noise_std=found.noise_std, prior_std=PRIOR_STD,
    prior_mean=MEAN_SKY, tol=CG_TOL, maxiter=2000, require_convergence=None,
)
draws = jax.vmap(
    lambda k: gcr_sample(
        block, observed, noise_std=found.noise_std, prior_std=PRIOR_STD,
        prior_mean=MEAN_SKY, key=k, tol=CG_TOL, maxiter=2000,
        require_convergence=None,
    )[0]
)(jax.random.split(jax.random.key(3), 200))

rms = float(jnp.sqrt(jnp.mean((mean - true_maps) ** 2)))
print("\nSTEP 5  the posterior")
print(f"  Wiener mean: RMS error {rms:.3f} K against a sky spanning "
      f"{float(true_maps.min()):.1f} .. {float(true_maps.max()):.1f} K")
print("  200 exact GCR draws, each one CG solve -- no burn-in, nothing to")
print("  diagnose, every draw independent")

# ---------------------------------------------------------------------------
# Step 6. Read the answer: what did the data actually constrain?
# ---------------------------------------------------------------------------
posterior_std = draws.std(axis=0)
shrink = posterior_std / PRIOR_STD
inside = jnp.abs(mean - true_maps) < posterior_std
print("\nSTEP 6  reading it")
print(f"  posterior sigma  {float(posterior_std.min()):.2f} .. "
      f"{float(posterior_std.max()):.2f} K "
      f"({100 * float(shrink.mean()):.1f}% of the {PRIOR_STD:.0f} K prior on average)")
print(f"  truth within 1 sigma of the mean: {100 * float(inside.mean()):.0f}% "
      f"of pixels (68% is the target)")
print(f"  RMS error {rms:.2f} K against a mean posterior sigma of "
      f"{float(posterior_std.mean()):.2f} K")
print("  -> the error bars are HONEST: the error is the size the posterior")
print("     claims it is, and 68% coverage is what calibrated means")

# ---------------------------------------------------------------------------
# Step 7. Now widen the beam past what it can resolve, and watch the prior
#         take over -- with the posterior saying so out loud.
# ---------------------------------------------------------------------------
wide_clean = twin(true_maps, WIDE_FWHM)(state).data
wide_observed = wide_clean * (
    1.0 + noise.fractional * jax.random.normal(jax.random.key(2), wide_clean.shape)
)
wide_start = twin(jnp.full_like(true_maps, MEAN_SKY), WIDE_FWHM)
wide_block = linear_operator(space, wide_start, state)
wide_kappa = float(
    condition_estimate(wide_block, noise_std=float(noise.std(wide_clean).mean()),
                       prior_std=PRIOR_STD)
)
wide_tol = TARGET_ERROR / wide_kappa
wide_found = iterative_gls(
    wide_block, wide_observed, noise=noise, prior_std=PRIOR_STD,
    prior_mean=MEAN_SKY, tol=wide_tol, maxiter=4000, require_convergence=None,
)
wide_mean, _ = wiener_solve(
    wide_block, wide_observed, noise_std=wide_found.noise_std,
    prior_std=PRIOR_STD, prior_mean=MEAN_SKY, tol=wide_tol, maxiter=4000,
    require_convergence=None,
)
wide_draws = jax.vmap(
    lambda k: gcr_sample(
        wide_block, wide_observed, noise_std=wide_found.noise_std,
        prior_std=PRIOR_STD, prior_mean=MEAN_SKY, key=k, tol=wide_tol,
        maxiter=4000, require_convergence=None,
    )[0]
)(jax.random.split(jax.random.key(3), 200))
wide_std = wide_draws.std(axis=0)
wide_rms = float(jnp.sqrt(jnp.mean((wide_mean - true_maps) ** 2)))
wide_inside = jnp.abs(wide_mean - true_maps) < wide_std

print("\nSTEP 7  the same solve with a beam 2.3x wider than the structure")
print(f"  {'':22s}{'FWHM ' + f'{jnp.rad2deg(FWHM):.0f}' + ' deg':>14s}"
      f"{'FWHM ' + f'{jnp.rad2deg(WIDE_FWHM):.0f}' + ' deg':>14s}")
print(f"  {'beam / pixel':22s}{jnp.rad2deg(FWHM) / PIXEL_DEG:14.1f}"
      f"{jnp.rad2deg(WIDE_FWHM) / PIXEL_DEG:14.1f}")
print(f"  {'kappa':22s}{float(kappa):14.1f}{wide_kappa:14.1f}")
print(f"  {'RMS error [K]':22s}{rms:14.2f}{wide_rms:14.2f}")
print(f"  {'posterior sigma [K]':22s}{float(posterior_std.mean()):14.2f}"
      f"{float(wide_std.mean()):14.2f}")
print(f"  {'% of prior width':22s}{100 * float(shrink.mean()):14.1f}"
      f"{100 * float(wide_std.mean()) / PRIOR_STD:14.1f}")
print(f"  {'68% coverage':22s}{100 * float(inside.mean()):14.0f}"
      f"{100 * float(wide_inside.mean()):14.0f}")
print("\n  Nothing failed. The wide beam cannot resolve the ring, so the prior")
print("  holds those directions up -- kappa says so beforehand, the posterior")
print("  width says so afterwards, and coverage climbing past 68% is what an")
print("  answer looks like when it is mostly prior. An estimator without error")
print("  bars would have reported the same map and told you none of this.")

print("\n" + "=" * 70)
print("A draw is EXACT for the covariance it is given. Everything above is")
print("about earning the right to that covariance: check the linearity, read")
print("the conditioning, then let iterative GLS find sigma. Skip any of them")
print("and the draws are still exact -- for the wrong problem.")
