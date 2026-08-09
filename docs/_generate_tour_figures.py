"""Generate the figures for docs/tour.md from the tour's own worked example.

Same contract as the other generators: transparent, theme-paired SVGs written
into ``docs/_static``, produced by running the thing being documented. What is
run here is *literally* the tour's example -- the same 64x8 grid, the same
four-position switch cycle, the same four noise-wave temperatures, the same
seed -- so the numbers under the figures are the numbers on the page.

Two figures:

* ``tour-waterfall`` -- Part 1's output. The raw waterfall the ADC records,
  with the switch cycle beside it, so the four sources are visible as four
  levels rather than described as four levels.
* ``tour-recovery`` -- Part 2's answer, in one figure: truth against the
  posterior mean with its 1-sigma band, the 32x32 correlation the same draws
  give, and the pull histogram over many noise realisations, because "the errors
  sit inside the error bars" is the claim and a pull distribution shows it.

Run:  .venv/bin/python docs/_generate_tour_figures.py
      (~60 s: the GCR sampler and the 500 re-solves dominate)
"""

from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402
import numpy as np  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import rhino_cal_jax as rcj  # noqa: E402

from rheplicant import Coordinates, Environment, State  # noqa: E402
from rheplicant.inference import (  # noqa: E402
    Bind,
    Latent,
    ParameterSpace,
    gcr_sample,
    linear_operator,
    wiener_solve,
)
from rheplicant.radio import (  # noqa: E402
    ADCOperator,
    AntennaLossOperator,
    BeamSpillOperator,
    CalLoadOperator,
    ForegroundOperator,
    GainOperator,
    GlobalSignalOperator,
    NoiseOperator,
    NoiseWaveOperator,
    ReceiverOperator,
    assemble,
)

STATIC = Path(__file__).parent / "_static"

THEMES = {
    "light": {"fg": "#24292f", "muted": "#57606a", "grid": "#d0d7de",
              "accent": "#0969da", "warm": "#bc4c00", "good": "#1a7f37",
              "band": "#0969da",
              # One colour per switch position, used in every panel. Three
              # alphas of one hue is not an encoding -- it was unreadable.
              "sources": ("#0969da", "#bc4c00", "#8250df", "#1a7f37")},
    "dark": {"fg": "#e6edf3", "muted": "#9198a1", "grid": "#30363d",
             "accent": "#58a6ff", "warm": "#ff9b57", "good": "#3fb950",
             "band": "#58a6ff",
             "sources": ("#58a6ff", "#ff9b57", "#d2a8ff", "#3fb950")},
}


def styled(theme: str):
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


def save(fig, name: str, theme: str) -> None:
    out = STATIC / f"{name}-{theme}.svg"
    fig.savefig(out, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"  wrote {out.relative_to(STATIC.parent.parent)}")


# ============================================================ the tour's run ==
N_TIME, N_FREQ = 64, 8
freq = jnp.linspace(60e6, 85e6, N_FREQ)
time_s = jnp.arange(float(N_TIME)) * 2.0
switch = jnp.arange(N_TIME) % 4
SOURCES = ("antenna", "ambient 300 K", "hot 400 K", "noise source 1200 K")

state = State(
    coords=Coordinates(time=time_s, freq=freq, extra={"receiver_input": switch}),
    env=Environment(temperature=jnp.array(280.0)),
    key=jax.random.key(20260806),
    meta={"telescope": "RHINO", "obs_id": "tour-001"},
)

F_SKY, T_GROUND = 0.97, 290.0
ETA, T_PHYS = 0.97, 293.0
ADC_SCALE, N_BITS = 0.25, 12
SIGMA_POST_GAIN = 2.0

gamma_rec = rcj.termination_gamma("resistive", N_FREQ, impedance=45.0)
gamma_src = jnp.stack([
    rcj.cable_gamma(rcj.termination_gamma("open", N_FREQ), freq, length=2.0, loss=0.92),
    rcj.termination_gamma("resistive", N_FREQ, impedance=10.0),
    rcj.cable_gamma(rcj.termination_gamma("short", N_FREQ), freq, length=0.4, loss=0.98),
    rcj.cable_gamma(rcj.termination_gamma("resistive", N_FREQ, impedance=150.0),
                    freq, length=1.1, loss=0.95),
])

TRUE = {
    "t_unc": 250.0 + 20.0 * jnp.linspace(-1.0, 1.0, N_FREQ),
    "t_cos": 30.0 * jnp.cos(jnp.linspace(0.0, 3.0, N_FREQ)),
    "t_sin": -40.0 + 8.0 * jnp.linspace(-1.0, 1.0, N_FREQ) ** 2,
    "t_rx": 290.0 + 5.0 * jnp.linspace(-1.0, 1.0, N_FREQ) ** 3,
}

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
    CalLoadOperator(t_load=jnp.array(300.0)),
    CalLoadOperator(t_load=jnp.array(400.0)),
    CalLoadOperator(t_load=jnp.array(1200.0)),
    NoiseWaveOperator(**TRUE,
                      gamma_src_re=gamma_src.real, gamma_src_im=gamma_src.imag,
                      gamma_rec_re=gamma_rec.real, gamma_rec_im=gamma_rec.imag),
    ReceiverOperator(bandpass=bandpass),
    GainOperator(gain=gain_t),
    NoiseOperator(sigma=jnp.array(SIGMA_POST_GAIN)),
    ADCOperator(scale=jnp.array(ADC_SCALE), n_bits=N_BITS),
)
observed = twin(state).data
print(f"waterfall {observed.shape}: {float(observed.min()):.1f}.."
      f"{float(observed.max()):.1f} counts")

NAMES = ("t_unc", "t_cos", "t_sin", "t_rx")
LABEL = {"t_unc": r"$T_\mathrm{unc}$", "t_cos": r"$T_\mathrm{cos}$",
         "t_sin": r"$T_\mathrm{sin}$", "t_rx": r"$T_\mathrm{rx}$"}
NOISE_STD = ADC_SCALE * SIGMA_POST_GAIN
PRIOR_STD = dict.fromkeys(NAMES, 100.0)
PRIOR_MEAN = dict.fromkeys(NAMES, 0.0)

space = ParameterSpace(
    latents=[Latent(n, init=jnp.zeros((N_FREQ,)), linear=True) for n in NAMES],
    bindings=[
        Bind("t_unc", into=lambda p: p["noise_wave"].t_unc),
        Bind("t_cos", into=lambda p: p["noise_wave"].t_cos),
        Bind("t_sin", into=lambda p: p["noise_wave"].t_sin),
        Bind("t_rx", into=lambda p: p["noise_wave"].t_rx),
    ],
)
fit_twin = twin.without("noise")
block = linear_operator(space, fit_twin, state, names=NAMES, check=False)
solved, residual = wiener_solve(block, observed, noise_std=NOISE_STD,
                                prior_std=PRIOR_STD, prior_mean=PRIOR_MEAN,
                                tol=1e-12, maxiter=4000)
N_DRAW = N_REAL = 500   # one number: draws validate, realisations calibrate
keys = jax.random.split(jax.random.key(7), N_DRAW)
draws = jax.vmap(lambda k: gcr_sample(
    block, observed, noise_std=NOISE_STD, prior_std=PRIOR_STD,
    prior_mean=PRIOR_MEAN, key=k, tol=1e-12, maxiter=4000)[0])(keys)
print(f"CG relative residual {float(residual):.1e}")

f_mhz = np.asarray(freq) / 1e6
wf = np.asarray(observed)
sw = np.asarray(switch)
truth = {n: np.asarray(TRUE[n]) for n in NAMES}
mean = {n: np.asarray(solved[n]) for n in NAMES}

# --------------------------------------------- the covariance, from draws --
# The draws are a sample of the posterior, so their covariance IS the posterior
# covariance -- no separate machinery needed.
_stack = np.concatenate([np.asarray(draws[n]) for n in NAMES], axis=1)   # (N_DRAW, 32)
SIGMA_MATRIX = np.cov(_stack, rowvar=False)
sd = np.sqrt(np.diag(SIGMA_MATRIX))
CORR = SIGMA_MATRIX / np.outer(sd, sd)
sigma = {n: sd[i * N_FREQ:(i + 1) * N_FREQ] for i, n in enumerate(NAMES)}

# --------------------------------------------------------------- the pulls --
# There are 32 recovered numbers, so one run gives 32 pulls however many draws
# are taken: draws estimate sigma, they do not add data. What adds pulls is
# another noise realisation -- and since the covariance is data-independent, the
# same sigma serves every one of them.
real_keys = jax.random.split(jax.random.key(4242), N_REAL)


def _pulls_for(k):
    obs = twin(state.replace(key=k)).data
    sol, _ = wiener_solve(block, obs, noise_std=NOISE_STD, prior_std=PRIOR_STD,
                          prior_mean=PRIOR_MEAN, tol=1e-12, maxiter=4000)
    return jnp.concatenate([(sol[n] - TRUE[n]) / jnp.asarray(sigma[n]) for n in NAMES])


_pull_rows = np.asarray(jax.vmap(_pulls_for)(real_keys))
all_pulls = _pull_rows.ravel()
chi2_per = (_pull_rows ** 2).mean(axis=1)
print(f"{N_REAL} realisations x 32 = {all_pulls.size} pulls: "
      f"chi2/dof {chi2_per.mean():.3f} (per-realisation range "
      f"{chi2_per.min():.2f}-{chi2_per.max():.2f})")


# ------------------------------------------------------------------- NUTS --
# The tour's nonlinear pair, run twice: as written, and with a prior-aware init.
import numpyro  # noqa: E402
import numpyro.distributions as dist  # noqa: E402

from rheplicant.inference import init_to_declared, to_numpyro_model  # noqa: E402

FG_NAMES = ("fg_log_amp", "fg_beta")
FG_TRUE = {"fg_log_amp": float(np.log(2500.0)), "fg_beta": 2.55}
FG_LABEL = {"fg_log_amp": r"$\log A$", "fg_beta": r"$\beta$"}

nuts_space = ParameterSpace(
    latents=[
        Latent("fg_log_amp", init=jnp.log(jnp.array(2000.0)),
               prior=dist.Normal(jnp.log(2000.0), 0.5)),
        Latent("fg_beta", init=jnp.array(2.30), prior=dist.Normal(2.3, 0.3)),
    ],
    bindings=[
        Bind("fg_log_amp", into=lambda p: p["foregrounds"].amplitude, fn=jnp.exp),
        Bind("fg_beta", into=lambda p: p["foregrounds"].spectral_index),
    ],
)
_nuts_model = to_numpyro_model(fit_twin, state, nuts_space, noise_std=NOISE_STD)


def _run_nuts(**kw):
    mcmc = numpyro.infer.MCMC(
        numpyro.infer.NUTS(_nuts_model, dense_mass=True, **kw),
        num_warmup=1000, num_samples=1000, num_chains=4,
        chain_method="vectorized", progress_bar=False)
    mcmc.run(jax.random.key(3), observed=observed, extra_fields=("diverging",))
    chained = mcmc.get_samples(group_by_chain=True)
    s = numpyro.diagnostics.summary({n: chained[n] for n in FG_NAMES}, prob=0.9)
    return chained, s, int(mcmc.get_extra_fields()["diverging"].sum())


CHAIN_BAD, STAT_BAD, DIV_BAD = _run_nuts()
CHAIN_OK, STAT_OK, DIV_OK = _run_nuts(init_strategy=init_to_declared(nuts_space))
print(f"NUTS as written    r_hat {max(float(STAT_BAD[n]['r_hat']) for n in FG_NAMES):.3f}"
      f"  n_eff {min(float(STAT_BAD[n]['n_eff']) for n in FG_NAMES):.0f}"
      f"  div {DIV_BAD}")
print(f"NUTS prior-aware   r_hat {max(float(STAT_OK[n]['r_hat']) for n in FG_NAMES):.3f}"
      f"  n_eff {min(float(STAT_OK[n]['n_eff']) for n in FG_NAMES):.0f}"
      f"  div {DIV_OK}")


# ============================================================== figure four ==
def nuts_figure(theme: str):
    """The failing chain, the healthy chain, and the posterior it finds."""
    c = THEMES[theme]
    fig, (a0, a1, a2) = plt.subplots(1, 3, figsize=(10.6, 3.2),
                                     gridspec_kw={"width_ratios": [1, 1, 1.05],
                                                  "wspace": 0.34})
    chain_c = (c["accent"], c["warm"], "#8250df", c["good"])

    for ax, chained, stat, div, title in (
        (a0, CHAIN_BAD, STAT_BAD, DIV_BAD, "as written"),
        (a1, CHAIN_OK, STAT_OK, DIV_OK, "prior-aware init"),
    ):
        for i in range(4):
            ax.plot(np.asarray(chained["fg_beta"][i]), lw=0.7,
                    color=chain_c[i], alpha=0.85)
        ax.axhline(FG_TRUE["fg_beta"], ls="--", lw=1.2, color=c["fg"], alpha=0.8)
        ax.set_xlabel("draw")
        ax.set_ylabel(r"$\beta$")
        r = max(float(stat[n]["r_hat"]) for n in FG_NAMES)
        n = min(float(stat[n]["n_eff"]) for n in FG_NAMES)
        ax.set_title(f"{title}\n$\\hat{{r}}$={r:.2f}  $n_{{eff}}$={n:.0f}  div={div}",
                     loc="left", fontsize=9)
        ax.grid(True)
    a0.set_ylim(a0.get_ylim())

    la = np.asarray(CHAIN_OK["fg_log_amp"]).ravel()
    be = np.asarray(CHAIN_OK["fg_beta"]).ravel()
    a2.hexbin(la, be, gridsize=34, cmap="Blues", mincnt=1, linewidths=0)
    # Annotated in place, not in a legend: a legend key for a star marker draws
    # a SECOND star, and readers took it for a second data point.
    a2.plot(FG_TRUE["fg_log_amp"], FG_TRUE["fg_beta"], "*", ms=14,
            color=c["good"], mec=c["fg"], mew=0.7, zorder=5)
    a2.annotate("truth", xy=(FG_TRUE["fg_log_amp"], FG_TRUE["fg_beta"]),
                xytext=(14, -14), textcoords="offset points", fontsize=8,
                color=c["fg"],
                arrowprops={"arrowstyle": "-", "lw": 0.8, "color": c["muted"]})
    a2.set_xlabel(FG_LABEL["fg_log_amp"])
    a2.set_ylabel(FG_LABEL["fg_beta"])
    a2.set_title(f"posterior, correlation {np.corrcoef(la, be)[0, 1]:+.2f}",
                 loc="left")
    return fig


# =============================================================== figure one ==
def waterfall_figure(theme: str):
    """The waterfall, with the switch cycle aligned under it.

    The cycle strip sits directly beneath the image and shares its x axis, so
    the stripes explain themselves: every fourth sample is the antenna. A bar
    chart of "16 samples" four times, which this panel used to be, says nothing
    the caption does not.
    """
    c = THEMES[theme]
    src_c = c["sources"]
    fig = plt.figure(figsize=(9.6, 3.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[2.1, 1.25],
                          height_ratios=[6, 1], hspace=0.12, wspace=0.30)

    axw = fig.add_subplot(gs[0, 0])
    im = axw.imshow(wf.T, aspect="auto", origin="lower", cmap="magma",
                    extent=(0, N_TIME, f_mhz[0], f_mhz[-1]))
    axw.set_ylabel("frequency  [MHz]")
    axw.set_xticklabels([])
    axw.set_title(f"the raw waterfall the ADC records  ({N_TIME} × {N_FREQ})",
                  loc="left")
    cb = fig.colorbar(im, ax=axw, pad=0.015)
    cb.set_label("ADC counts", color=c["fg"], fontsize=8)
    cb.ax.tick_params(colors=c["muted"], labelsize=7)
    cb.outline.set_edgecolor(c["grid"])

    # imshow, not fill_between: on a strip this flat the fills came out as
    # hairlines and the panel read as empty.
    axs = fig.add_subplot(gs[1, 0], sharex=axw)
    axs.imshow(sw[None, :], aspect="auto", origin="lower", interpolation="nearest",
               cmap=matplotlib.colors.ListedColormap(src_c),
               vmin=-0.5, vmax=3.5, extent=(0, N_TIME, 0, 1))
    axs.set_yticks([])
    axs.set_xlabel("sample   →   the switch cycle repeats every 4")
    for s in ("left", "bottom"):
        axs.spines[s].set_visible(False)

    axm = fig.add_subplot(gs[:, 1])
    for i, src in enumerate(SOURCES):
        rows = wf[sw == i]
        axm.plot(f_mhz, rows.mean(axis=0), marker="o", ms=3.5, lw=1.6,
                 color=src_c[i], label=src)
    axm.set_xlabel("frequency  [MHz]")
    axm.set_ylabel("mean counts")
    axm.set_title("one spectrum per source", loc="left")
    axm.grid(True)
    axm.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.20),
           ncol=2, columnspacing=1.0, handlelength=1.4)

    return fig


# =============================================================== figure two ==
def recovery_figure(theme: str):
    """Truth against the posterior, and the pulls that say whether to believe it."""
    c = THEMES[theme]
    fig = plt.figure(figsize=(10.6, 4.2))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.9], hspace=0.55, wspace=0.34)

    for k, n in enumerate(NAMES):
        ax = fig.add_subplot(gs[k // 2, k % 2])
        ax.fill_between(f_mhz, mean[n] - sigma[n], mean[n] + sigma[n],
                        color=c["band"], alpha=0.22, lw=0,
                        label="posterior ±1σ" if k == 0 else None)
        ax.plot(f_mhz, mean[n], color=c["accent"], lw=1.6,
                label="posterior mean" if k == 0 else None)
        ax.plot(f_mhz, truth[n], ls="--", lw=1.4, color=c["good"],
                label="truth" if k == 0 else None)
        ax.set_title(LABEL[n], loc="left")
        ax.grid(True)
        if k // 2 == 1:
            ax.set_xlabel("frequency  [MHz]")
        if k % 2 == 0:
            ax.set_ylabel("K")
        if k == 0:
            ax.legend(fontsize=7, loc="lower right")

    axp = fig.add_subplot(gs[:, 2])
    axp.hist(all_pulls, bins=np.linspace(-4, 4, 41), color=c["accent"],
             alpha=0.55, density=True, edgecolor="none")
    x = np.linspace(-4, 4, 200)
    axp.plot(x, np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi), color=c["good"], lw=1.7)
    axp.set_xlabel(r"pull  $(\hat\theta - \theta)/\sigma$")
    axp.set_yticks([])
    axp.set_title(f"32 × {N_REAL} pulls   "
                  f"$\\chi^2/\\mathrm{{dof}}$ = {np.mean(all_pulls ** 2):.2f}",
                  loc="left")
    axp.grid(True, axis="x")
    return fig


# ============================================================= figure three ==
def covariance_figure(theme: str):
    """The 32x32 posterior correlation, and the 4x4 it is eight copies of."""
    c = THEMES[theme]
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(9.0, 4.1),
                                   gridspec_kw={"width_ratios": [1.25, 1.0],
                                                "wspace": 0.32})

    im = axl.imshow(CORR, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    for k in (1, 2, 3):
        axl.axhline(k * N_FREQ - 0.5, color=c["fg"], lw=0.9, alpha=0.75)
        axl.axvline(k * N_FREQ - 0.5, color=c["fg"], lw=0.9, alpha=0.75)
    ticks = [i * N_FREQ + N_FREQ / 2 - 0.5 for i in range(4)]
    short = [LABEL[n] for n in NAMES]
    axl.set_xticks(ticks, short)
    axl.set_yticks(ticks, short)
    axl.tick_params(length=0)
    axl.set_title("posterior correlation, 32 × 32", loc="left")
    cb = fig.colorbar(im, ax=axl, pad=0.025, shrink=0.82)
    cb.set_ticks([-1, 0, 1])
    cb.ax.tick_params(colors=c["muted"], labelsize=7)
    cb.outline.set_edgecolor(c["grid"])

    j = N_FREQ // 2
    idx = [i * N_FREQ + j for i in range(4)]
    sub = CORR[np.ix_(idx, idx)]
    axr.imshow(sub, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    for a in range(4):
        for b in range(4):
            axr.text(b, a, f"{sub[a, b]:+.2f}", ha="center", va="center", fontsize=9,
                     color="#ffffff" if abs(sub[a, b]) > 0.55 else c["fg"])
    axr.set_xticks(range(4), short)
    axr.set_yticks(range(4), short)
    axr.tick_params(length=0)
    axr.set_title(f"one channel, {f_mhz[j]:.0f} MHz", loc="left")
    return fig


if __name__ == "__main__":
    STATIC.mkdir(exist_ok=True)
    for theme in THEMES:
        with plt.rc_context(styled(theme)):
            save(waterfall_figure(theme), "tour-waterfall", theme)
            save(recovery_figure(theme), "tour-recovery", theme)
            save(covariance_figure(theme), "tour-covariance", theme)
            save(nuts_figure(theme), "tour-nuts", theme)
