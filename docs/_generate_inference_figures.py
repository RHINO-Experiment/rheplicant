"""Generate the figures for docs/inference.md from live code.

Same contract as ``_generate_engine_figures.py``: transparent, theme-paired
SVGs written into ``docs/_static``, produced by actually running the thing
being documented rather than by illustrating it.

Two figures:

* ``inference-posterior`` — the beam fit. A NUTS posterior over the two beam
  latents, with the truth marked and the Fisher forecast overlaid, so the
  claim "the posterior recovers the truth" is a picture of real numbers.
* ``inference-linear`` — the sky block. Truth, prior mean, the CG posterior
  mean and a 68% band from exact GCR draws on one axis, plus the per-channel
  shrinkage that shows what the data actually constrained.

Run:  .venv/bin/python docs/_generate_inference_figures.py
      .venv/bin/python docs/_generate_inference_figures.py --replot   (cached)
"""

import argparse
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402
import numpy as np  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from rheplicant import Coordinates, State  # noqa: E402
from rheplicant.core.pipeline import Pipeline  # noqa: E402
from rheplicant.inference import (  # noqa: E402
    Bind,
    Latent,
    ParameterSpace,
    fisher_information,
    gcr_sample,
    linear_operator,
    parameter_covariance,
    wiener_solve,
)
from rheplicant.radio import GainOperator, SkySourceOperator  # noqa: E402
from rheplicant.radio.sky import MatrixProjector  # noqa: E402
from rheplicant.radio.sky.model import AbstractSkyModel  # noqa: E402

STATIC = Path(__file__).parent / "_static"
CACHE = Path(__file__).parent / "_inference-figure-data.npz"  # git-ignored

THEMES = {
    "light": {"fg": "#24292f", "muted": "#57606a", "grid": "#d0d7de",
              "accent": "#0969da", "warm": "#bc4c00", "good": "#1a7f37"},
    "dark": {"fg": "#e6edf3", "muted": "#9198a1", "grid": "#30363d",
             "accent": "#58a6ff", "warm": "#ff9b57", "good": "#3fb950"},
}

N_TIME, N_FREQ, N_PIX = 128, 4, 128
TRUE_FWHM, TRUE_OFFSET, TRUE_GAIN = 0.35, 0.12, 1.10
MEAN_SKY, SKY_SCALE, NOISE = 10.0, 1.0, 0.02


def styled(theme: str):
    """rcParams for a transparent figure legible on that theme's background."""
    c = THEMES[theme]
    return {
        "figure.facecolor": "none", "axes.facecolor": "none",
        "savefig.facecolor": "none", "savefig.transparent": True,
        "text.color": c["fg"], "axes.labelcolor": c["fg"],
        "axes.edgecolor": c["grid"], "xtick.color": c["muted"],
        "ytick.color": c["muted"], "grid.color": c["grid"],
        "axes.titlecolor": c["fg"], "legend.frameon": False,
        "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 130, "grid.alpha": 0.55, "grid.linewidth": 0.6,
    }


def save(fig, name: str, theme: str, ext: str = "svg") -> None:
    out = STATIC / f"{name}-{theme}.{ext}"
    fig.savefig(out, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"  wrote {out.relative_to(STATIC.parent.parent)}")


# ---------------------------------------------------------------- the model --
pixel_angle = 2.0 * jnp.pi * jnp.arange(N_PIX) / N_PIX
scan_angle = 2.0 * jnp.pi * jnp.arange(N_TIME) / N_TIME


class MapSky(AbstractSkyModel):
    maps: jax.Array

    def __call__(self, freq: jax.Array) -> jax.Array:
        return self.maps


def beam_matrix(fwhm, offset):
    separation = pixel_angle[None, :] - scan_angle[:, None] - offset
    wrapped = jnp.arctan2(jnp.sin(separation), jnp.cos(separation))
    response = jnp.exp(-0.5 * (wrapped / (fwhm / 2.3548)) ** 2)
    return response / jnp.sum(response, axis=1, keepdims=True)


def build_twin(maps, fwhm, offset, gain):
    return Pipeline(
        SkySourceOperator(
            sky_model=MapSky(maps=maps),
            projector=MatrixProjector(beam_matrix(fwhm, offset)),
        ),
        GainOperator(gain=jnp.asarray(gain)),
        names=("sky", "gain"),
    )


def make_world():
    key = jax.random.key(0)
    modes = jnp.arange(1, 6)[:, None] * pixel_angle[None, :]
    weights = jax.random.normal(jax.random.fold_in(key, 1), (N_FREQ, 5))
    true_maps = MEAN_SKY + SKY_SCALE * (weights @ jnp.cos(modes)) / jnp.sqrt(2.5)
    state = State(
        coords=Coordinates(
            time=jnp.arange(float(N_TIME)), freq=jnp.linspace(60e6, 85e6, N_FREQ)
        ),
        meta={"telescope": "ring-toy"},
    )
    truth = build_twin(true_maps, TRUE_FWHM, TRUE_OFFSET, TRUE_GAIN)
    observed = truth(state).data + NOISE * jax.random.normal(
        jax.random.fold_in(key, 2), (N_TIME, N_FREQ)
    )
    return true_maps, state, observed


def run_beam_posterior(state, observed, true_maps):
    """NUTS over the two beam latents, plus the Fisher forecast at the truth."""
    import numpyro
    import numpyro.distributions as dist

    from rheplicant.inference import to_numpyro_model

    space = ParameterSpace(
        latents=[
            Latent("fwhm", init=TRUE_FWHM, prior=dist.Uniform(0.15, 0.70)),
            Latent("offset", init=TRUE_OFFSET, prior=dist.Normal(0.0, 0.4)),
        ],
        bindings=[
            Bind(("fwhm", "offset"), into=lambda p: p["sky"].projector.matrix,
                 fn=beam_matrix)
        ],
    )
    twin = build_twin(true_maps, 0.5, 0.0, TRUE_GAIN)
    model = to_numpyro_model(twin, state, space, noise_std=NOISE)
    mcmc = numpyro.infer.MCMC(
        numpyro.infer.NUTS(model), num_warmup=500, num_samples=1500,
        progress_bar=False,
    )
    mcmc.run(jax.random.key(0), observed=observed)
    samples = mcmc.get_samples()

    forward, _ = space.forward_fn(twin, state)
    at_truth = {"fwhm": jnp.array(TRUE_FWHM), "offset": jnp.array(TRUE_OFFSET)}
    cov = parameter_covariance(fisher_information(forward, at_truth, noise_std=NOISE))
    return (
        np.asarray(samples["fwhm"]),
        np.asarray(samples["offset"]),
        float(cov.sigma("fwhm")),
        float(cov.sigma("offset")),
    )


def run_linear_block(state, observed, true_maps, n_draws=400):
    """Declare the sky linear, check it, solve it AND sample it exactly."""
    calibrated = build_twin(true_maps, TRUE_FWHM, TRUE_OFFSET, TRUE_GAIN)
    space = ParameterSpace.direct(
        "sky_delta", init=jnp.zeros((N_FREQ, N_PIX)),
        into=lambda p: p["sky"].sky_model.maps,
        fn=lambda delta: MEAN_SKY + delta, linear=True,
    )
    block = linear_operator(space, calibrated, state)
    solved, _ = wiener_solve(block, observed, noise_std=NOISE, prior_std=SKY_SCALE)
    keys = jax.random.split(jax.random.key(21), n_draws)
    draws = jax.vmap(
        lambda k: gcr_sample(block, observed, noise_std=NOISE,
                             prior_std=SKY_SCALE, key=k)[0]
    )(keys)
    return (
        np.asarray(MEAN_SKY + solved),
        np.asarray(true_maps),
        np.asarray(MEAN_SKY + draws),
    )


# ------------------------------------------------------------------- plots --
def plot_posterior(fwhm, offset, sigma_fwhm, sigma_offset):
    for theme in THEMES:
        c = THEMES[theme]
        with plt.rc_context(styled(theme)):
            fig, (ax, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.2))

            ax.scatter(fwhm, offset, s=3, alpha=0.25, color=c["accent"],
                       edgecolors="none")
            ax.axvline(TRUE_FWHM, color=c["warm"], lw=1.1, ls="--")
            ax.axhline(TRUE_OFFSET, color=c["warm"], lw=1.1, ls="--")
            ax.plot([TRUE_FWHM], [TRUE_OFFSET], marker="+", ms=13, mew=2.0,
                    color=c["warm"], ls="none")
            # Annotated rather than put in a legend: a frameless legend entry
            # for a "+" marker reads as a SECOND truth cross floating in the
            # data, which is exactly the wrong thing for this figure to imply.
            ax.annotate("truth", xy=(TRUE_FWHM, TRUE_OFFSET), xytext=(7, 7),
                        textcoords="offset points", color=c["warm"],
                        fontweight="bold", fontsize=9)
            ax.annotate("NUTS samples", xy=(0.03, 0.05), xycoords="axes fraction",
                        color=c["accent"], fontsize=9)
            ax.set_xlabel("beam FWHM  [rad]")
            ax.set_ylabel("pointing offset  [rad]")
            ax.set_title("two latents, ~10⁴ derived matrix entries")
            ax.grid(True)

            for values, sigma, colour, label in (
                (fwhm, sigma_fwhm, c["accent"], "fwhm"),
                (offset, sigma_offset, c["good"], "offset"),
            ):
                centred = (values - values.mean()) / values.std()
                ax2.hist(centred, bins=45, density=True, histtype="step", lw=1.4,
                         color=colour,
                         label=f"{label}:  σ_post {values.std():.4f}  "
                               f"σ_Fisher {sigma:.4f}")
            grid = np.linspace(-4, 4, 200)
            ax2.plot(grid, np.exp(-0.5 * grid**2) / np.sqrt(2 * np.pi),
                     color=c["muted"], lw=1.0, ls=":", label="unit normal")
            ax2.set_xlabel("(sample − mean) / posterior σ")
            ax2.set_yticks([])
            ax2.set_title("posterior width vs Fisher forecast")
            ax2.legend(loc="upper left", fontsize=7.5)
            ax2.grid(True, axis="x")

            fig.tight_layout()
            save(fig, "inference-posterior", theme)


def plot_linear(recovered, truth, draws):
    for theme in THEMES:
        c = THEMES[theme]
        with plt.rc_context(styled(theme)):
            fig, (ax, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.2))

            angle = np.degrees(np.asarray(pixel_angle))
            lo, hi = np.percentile(draws[:, 0, :], [16, 84], axis=0)
            ax.fill_between(angle, lo, hi, color=c["accent"], alpha=0.22, lw=0,
                            label=f"posterior 68% ({draws.shape[0]} exact draws)")
            ax.plot(angle, truth[0], color=c["muted"], lw=2.4, alpha=0.6,
                    label="true sky")
            ax.plot(angle, recovered[0], color=c["accent"], lw=1.3,
                    label="posterior mean (CG)")
            ax.axhline(MEAN_SKY, color=c["warm"], lw=1.0, ls="--",
                       label="prior mean")
            ax.set_xlabel("sky pixel  [deg]")
            ax.set_ylabel("brightness  [K]")
            ax.set_title(f"{truth.size} degrees of freedom, solved AND sampled")
            # headroom so the legend does not sit on the posterior band
            low, high = ax.get_ylim()
            ax.set_ylim(low, high + 0.30 * (high - low))
            ax.legend(loc="upper right", fontsize=7.5)
            ax.grid(True)

            before = np.sqrt(((truth - MEAN_SKY) ** 2).mean(axis=1))
            after = np.sqrt(((truth - recovered) ** 2).mean(axis=1))
            index = np.arange(len(before))
            ax2.bar(index - 0.2, before, width=0.4, color=c["warm"],
                    label="prior mean")
            ax2.bar(index + 0.2, after, width=0.4, color=c["good"],
                    label="after the solve")
            ax2.set_xticks(index)
            ax2.set_xticklabels([f"ch {i}" for i in index])
            ax2.set_ylabel("RMS error vs truth  [K]")
            ax2.set_title("what the data actually constrained")
            ax2.legend(loc="upper right", fontsize=8)
            ax2.grid(True, axis="y")

            fig.tight_layout()
            save(fig, "inference-linear", theme)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replot", action="store_true",
                        help="redraw from the cached run (seconds, not minutes)")
    args = parser.parse_args()

    if args.replot:
        if not CACHE.exists():
            raise SystemExit(f"no cache at {CACHE}; run without --replot first")
        cache = np.load(CACHE)
        plot_posterior(cache["fwhm"], cache["offset"],
                       float(cache["sigma_fwhm"]), float(cache["sigma_offset"]))
        plot_linear(cache["recovered"], cache["truth"], cache["draws"])
        return

    true_maps, state, observed = make_world()
    print("running NUTS over the beam latents ...")
    fwhm, offset, sigma_fwhm, sigma_offset = run_beam_posterior(
        state, observed, true_maps
    )
    print(f"  fwhm   {fwhm.mean():.4f} +/- {fwhm.std():.4f}  (truth {TRUE_FWHM})")
    print(f"  offset {offset.mean():.4f} +/- {offset.std():.4f}  "
          f"(truth {TRUE_OFFSET})")

    print("solving the linear sky block ...")
    recovered, truth, draws = run_linear_block(state, observed, true_maps)
    print(f"  RMS vs truth: {np.sqrt(((truth - MEAN_SKY)**2).mean()):.3f} K -> "
          f"{np.sqrt(((truth - recovered)**2).mean()):.3f} K")

    plot_posterior(fwhm, offset, sigma_fwhm, sigma_offset)
    plot_linear(recovered, truth, draws)
    np.savez_compressed(
        CACHE, fwhm=fwhm, offset=offset, sigma_fwhm=sigma_fwhm,
        sigma_offset=sigma_offset, recovered=recovered, truth=truth, draws=draws,
    )
    print(f"  wrote {CACHE.name} (redraw with --replot)")


if __name__ == "__main__":
    main()
