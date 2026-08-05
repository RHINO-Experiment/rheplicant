"""Infer anything: three parameterizations of ONE twin — end-to-end demo.

The instrument description is written once and never touched again. What
changes below is only the *parameter space*: which quantities are inferred,
and how they reach the model.

  1. **A beam from two numbers.** Beam width and pointing offset are the
     latents; the whole (time x pixel) response matrix is DERIVED from them.
     Two scalars drive several thousand numbers.
  2. **Add a gain, tied across two stages, in log space.** One latent lands in
     two leaves through a positivity transform — and joining it to the beam
     block exposes a real degeneracy, which is the honest reason to fit them
     together rather than in sequence.
  3. **A sky map as a linear block.** Declared `linear=True`, checked, then
     both solved in closed form AND sampled exactly by conjugate gradients —
     the pattern that scales to sky alms, where a gradient sampler is hopeless.

None of this required a new operator class. That is the point: a
re-parameterization is an inference decision, so it lives in the inference
layer, not in the instrument description.

Run:  uv run python examples/inferring_anything.py
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from rheplicant import Coordinates, State  # noqa: E402
from rheplicant.core.pipeline import Pipeline  # noqa: E402
from rheplicant.inference import (  # noqa: E402
    AdamCalibrator,
    Bind,
    Latent,
    ParameterSpace,
    check_linearity,
    fisher_information,
    gcr_sample,
    linear_operator,
    parameter_covariance,
    wiener_solve,
)
from rheplicant.radio import GainOperator, SkySourceOperator  # noqa: E402
from rheplicant.radio.sky import MatrixProjector  # noqa: E402
from rheplicant.radio.sky.model import AbstractSkyModel  # noqa: E402

N_TIME, N_FREQ, N_PIX = 64, 4, 96
TRUE_FWHM, TRUE_OFFSET, TRUE_GAIN = 0.35, 0.12, 1.10
MEAN_SKY, SKY_SCALE, NOISE = 10.0, 1.0, 0.02


class MapSky(AbstractSkyModel):
    """A fixed brightness map per channel — what fit 3 infers."""

    maps: jax.Array

    def __call__(self, freq: jax.Array) -> jax.Array:
        return self.maps


# --------------------------------------------------------------- the model ---
# A ring of sky pixels drifting past a Gaussian beam. Deliberately small, and
# deliberately built from ordinary operators: nothing here knows about
# inference.
pixel_angle = 2.0 * jnp.pi * jnp.arange(N_PIX) / N_PIX
scan_angle = 2.0 * jnp.pi * jnp.arange(N_TIME) / N_TIME


def beam_matrix(fwhm: jax.Array, offset: jax.Array) -> jax.Array:
    """(n_time, n_pix) response of a Gaussian beam of width `fwhm`, mis-pointed
    by `offset`. THIS is what the two beam latents drive."""
    separation = pixel_angle[None, :] - scan_angle[:, None] - offset
    wrapped = jnp.arctan2(jnp.sin(separation), jnp.cos(separation))
    sigma = fwhm / 2.3548
    response = jnp.exp(-0.5 * (wrapped / sigma) ** 2)
    return response / jnp.sum(response, axis=1, keepdims=True)


key = jax.random.key(0)
# A smooth sky: the beam is ~20 deg wide, so structure finer than that is not
# in the data at all and no estimator can invent it.
modes = jnp.arange(1, 5)[:, None] * pixel_angle[None, :]
weights = jax.random.normal(jax.random.fold_in(key, 1), (N_FREQ, 4))
true_maps = MEAN_SKY + SKY_SCALE * (weights @ jnp.cos(modes)) / jnp.sqrt(2.0)

state = State(
    coords=Coordinates(
        time=jnp.arange(float(N_TIME)), freq=jnp.linspace(60e6, 85e6, N_FREQ)
    ),
    meta={"telescope": "ring-toy"},
)


def build_twin(maps, fwhm, offset, gain):
    return Pipeline(
        SkySourceOperator(
            sky_model=MapSky(maps=maps),
            projector=MatrixProjector(beam_matrix(fwhm, offset)),
        ),
        GainOperator(gain=jnp.asarray(gain)),
        GainOperator(gain=jnp.asarray(gain)),
        names=("sky", "gain_lna", "gain_backend"),
    )


truth = build_twin(true_maps, TRUE_FWHM, TRUE_OFFSET, TRUE_GAIN)
observed = truth(state).data + NOISE * jax.random.normal(
    jax.random.fold_in(key, 2), (N_TIME, N_FREQ)
)
print(f"observation: {observed.shape}, {float(observed.mean()):.3f} K mean\n")


def fit(space, pipeline, steps=3000, lr=0.01):
    forward, start = space.forward_fn(pipeline, state)
    fitted, losses = AdamCalibrator(learning_rate=lr, n_steps=steps).fit(
        forward, start, observed
    )
    return fitted, forward, float(losses[0]), float(losses[-1])


# ----------------------------------------------- 1. a beam from two numbers ---
# Two scalar latents; `fn` builds the full (n_time x n_pix) response from them.
# The beam matrix is never a parameter — it is DERIVED from parameters, which
# is exactly why this needs no new operator class.
#
# The gain is held at truth here, so this measures the beam alone. Fit 2 shows
# what happens when it is not.
beam_space = ParameterSpace(
    latents=[Latent("fwhm", init=0.50), Latent("offset", init=0.0)],
    bindings=[
        Bind(("fwhm", "offset"), into=lambda p: p["sky"].projector.matrix, fn=beam_matrix)
    ],
)
known_gain = build_twin(true_maps, 0.50, 0.0, TRUE_GAIN)
beam_fit, beam_forward, loss0, loss1 = fit(beam_space, known_gain)
print(f"1. a beam from two numbers  (2 latents -> {beam_matrix(0.5, 0.0).size} "
      "matrix entries)")
print(f"   fwhm   {float(beam_fit['fwhm']):.4f}   (truth {TRUE_FWHM:.4f})")
print(f"   offset {float(beam_fit['offset']):.4f}   (truth {TRUE_OFFSET:.4f})")
print(f"   loss   {loss0:.3g} -> {loss1:.3g}")

# Error bars come back by NAME, not by position in a flattened vector.
cov = parameter_covariance(fisher_information(beam_forward, beam_fit, noise_std=NOISE))
print(f"   sigma(fwhm) {float(cov.sigma('fwhm')):.4f}, "
      f"sigma(offset) {float(cov.sigma('offset')):.4f}\n")

# ---------------------------------------- 2. add a tied gain, in log space ---
# One latent -> TWO leaves, through exp() so the gain cannot go negative.
# Blocks compose: this space is the beam space plus one more latent and one
# more binding.
joint_space = ParameterSpace(
    latents=[*beam_space.latents, Latent("log_gain", init=0.0)],
    bindings=[
        *beam_space.bindings,
        Bind(
            "log_gain",
            into=(lambda p: p["gain_lna"].gain, lambda p: p["gain_backend"].gain),
            fn=jnp.exp,
        ),
    ],
)
twin = build_twin(true_maps, 0.50, 0.0, 1.0)  # every number wrong
joint_fit, joint_forward, _, _ = fit(joint_space, twin, steps=6000, lr=0.005)
per_stage = float(jnp.exp(joint_fit["log_gain"]))
bound = joint_space.bind(twin, joint_fit)
print("2. the same beam, plus one gain tied across two stages, in log space")
print(f"   fwhm   {float(joint_fit['fwhm']):.4f}   (truth {TRUE_FWHM:.4f})")
print(f"   gain   {per_stage:.4f} per stage   (truth {TRUE_GAIN:.4f})")
print("   both leaves carry it: "
      f"{float(bound['gain_lna'].gain):.4f} == {float(bound['gain_backend'].gain):.4f}")

joint_cov = parameter_covariance(
    fisher_information(joint_forward, joint_fit, noise_std=NOISE)
)
# block() always returns a matrix, so a scalar-vs-scalar cross term is (1, 1).
correlation = float(
    jnp.squeeze(
        joint_cov.block("fwhm", "log_gain")
        / jnp.sqrt(joint_cov.block("fwhm") * joint_cov.block("log_gain"))
    )
)

# What happens if you DON'T free the gain: the same beam space, fitted against
# data whose gain the model has wrong.
stuck, _, _, _ = fit(beam_space, twin)
bias = float((stuck["fwhm"] - TRUE_FWHM) / cov.sigma("fwhm"))
print(f"   fwhm/log_gain correlation {correlation:+.2f} — nearly independent")
print(f"   but holding the gain at the WRONG value drives fwhm to "
      f"{float(stuck['fwhm']):.4f} ({bias:+.0f} sigma).")
print("   The beam matrix is row-normalised, so NO fwhm changes the level at")
print("   all; the model is short 21 % of signal and its only lever is contrast,")
print("   which narrowing buys about 7 % of. So the fit runs to the boundary")
print("   rather than finding a compensating optimum — model mis-specification,")
print("   not degeneracy, which the near-zero cross-block is how you tell.\n")

# ------------------------------------------- 3. the sky as a linear block ---
# The map enters linearly, so it needs no sampler at all: declare it, let the
# declaration be CHECKED, then solve in closed form. `fn` adds a known mean, so
# the latent is the deviation and the block is genuinely affine — that constant
# part is what LinearBlock.offset holds.
calibrated = joint_space.bind(twin, joint_fit)
sky_space = ParameterSpace.direct(
    "sky_delta",
    init=jnp.zeros((N_FREQ, N_PIX)),
    into=lambda p: p["sky"].sky_model.maps,
    fn=lambda delta: MEAN_SKY + delta,
    linear=True,
)
# `names=` rather than `name=` throughout, including for this block of ONE
# latent: the answer then comes back as {"sky_delta": array} and can be handed
# straight to sky_space.forward_fn / bind / identifiability. The singular
# spelling returns a bare array, which all of those reject -- see
# docs/inference.md, "One latent, or a group". A grouped block also takes its
# prior PER MEMBER, since S is block-diagonal over the group.
SKY_BLOCK = ("sky_delta",)
SKY_PRIOR = {"sky_delta": SKY_SCALE}

errors = check_linearity(sky_space, calibrated, state, names=SKY_BLOCK)
print("3. the sky map as a declared-linear block")
print(f"   linearity check passed: worst relative departure {max(errors.values()):.1e} "
      f"over probes\n      spanning {min(errors):g}x - {max(errors):g}x "
      "(the small-probe figure is roundoff, below the absolute floor)")

block = linear_operator(sky_space, calibrated, state, names=SKY_BLOCK)
answer, residual = wiener_solve(block, observed, noise_std=NOISE, prior_std=SKY_PRIOR)
solved = answer["sky_delta"]
recovered = MEAN_SKY + solved
before = jnp.sqrt(jnp.mean((MEAN_SKY - true_maps) ** 2))
after = jnp.sqrt(jnp.mean((recovered - true_maps) ** 2))
print(f"   CG solved {solved.size} degrees of freedom, residual {float(residual):.1e}")
print(f"   RMS vs truth: {float(before):.3f} K (prior mean) -> {float(after):.3f} K")
# The answer is keyed by latent, so it goes back into the model with no
# unpacking: forward(answer) is the prediction at the posterior mean.
sky_forward, _ = sky_space.forward_fn(calibrated, state)
print(f"   and it feeds straight back: forward(answer) -> "
      f"{jnp.shape(sky_forward(answer))}, chi2/N "
      f"{float(jnp.mean(((observed - sky_forward(answer)) / NOISE) ** 2)):.3f}")

# ...and the posterior is not just its mean. gcr_sample adds a fluctuation term
# to the SAME right-hand side, which makes each solve an exact, independent
# posterior draw — no chain, no burn-in, same cost as the mean.
keys = jax.random.split(jax.random.key(9), 200)
draws = jax.vmap(
    lambda k: gcr_sample(
        block, observed, noise_std=NOISE, prior_std=SKY_PRIOR, key=k
    )[0]["sky_delta"]
)(keys)
spread = draws.std(axis=0)
print(f"   {keys.shape[0]} exact posterior draws: per-pixel sigma spans "
      f"{float(spread.min()):.3f}-{float(spread.max()):.3f} K")
# Compare like with like: the RMS of the mean's error against the standard
# error of the mean. (Comparing the WORST element to a per-element sigma would
# fail for a perfectly correct sampler -- the max over 384 draws sits near 3
# sigma by construction.)
mean_error = jnp.sqrt(jnp.mean((draws.mean(axis=0) - solved) ** 2))
sem = jnp.sqrt(jnp.mean(spread**2) / keys.shape[0])
print(f"   their mean matches the Wiener mean: RMS error {float(mean_error):.4f} K "
      f"vs expected {float(sem):.4f} K")
print("   -> closed form and exactly samplable. The same calls take sky alms,\n"
      "      where a gradient sampler on 1e6 coefficients is not an option.\n")

# ----------------------------------------------------- the model, unchanged ---
print("instrument description edited across all three fits:",
      jax.tree_util.tree_structure(twin) != jax.tree_util.tree_structure(calibrated))
