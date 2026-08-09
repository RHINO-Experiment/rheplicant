"""Generate the figures for docs/sky-to-receiver.md from live code.

Same contract as the other two generators: transparent, theme-paired SVGs
written into ``docs/_static``, produced by actually running the thing being
documented rather than by illustrating it. Every number on these axes comes
from the same calls the example makes.

Three figures:

* ``receiver-horizon`` -- RHINO's horn against the horizon. The beam profile in
  dBi versus zenith angle with the below-horizon region shaded, and the share
  that falls there, per frequency. Makes "1-3% of the response sees ground" a
  picture rather than a sentence.
* ``receiver-cascade`` -- what happens to the sky, in order: the horizon split
  (mixing), the horn's ohmic loss (loss plus its own emission), the receiver
  mismatch (loss, nothing added). Three effects that are easy to conflate,
  drawn to scale.
* ``receiver-recovery`` -- the payoff. Truth against the Wiener mean for the
  three per-channel noise-wave temperatures, with the residual underneath.

Falls back to an analytic horn when the (unpublished) RHINO CST export is
absent, so the figures regenerate anywhere; the caption says which was used.

Run:  .venv/bin/python docs/_generate_receiver_figures.py
      .venv/bin/python docs/_generate_receiver_figures.py --replot   (cached)
"""

import argparse
import os
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import equinox as eqx  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import rhino_cal_jax as rcj  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from rheplicant import Coordinates, State  # noqa: E402
from rheplicant.inference import (  # noqa: E402
    Bind,
    Latent,
    ParameterSpace,
    linear_operator,
    wiener_solve,
)
from rheplicant.radio import (  # noqa: E402
    AntennaLossOperator,
    AtmosphericEmissionOperator,
    BeamSpillOperator,
    CalLoadOperator,
    NoiseWaveOperator,
    SkySourceOperator,
    assemble,
    cst_beam_maps,
    horizon_truncated_beam,
)
from rheplicant.radio.sky import DriftScanProjector  # noqa: E402
from rheplicant.radio.sky.model import AbstractSkyModel  # noqa: E402

STATIC = Path(__file__).parent / "_static"
CACHE = Path(__file__).parent / "_receiver-figure-data.npz"  # git-ignored
# Named rather than guessed: the CST exports are not redistributable, so a
# hard-coded home directory here described one machine and quietly produced the
# Gaussian-beam figure everywhere else.
_NAMED_BEAMS = os.environ.get("RHEPLICANT_RHINO_BEAMS")
RHINO_BEAMS = Path(_NAMED_BEAMS).expanduser() if _NAMED_BEAMS else None

THEMES = {
    "light": {"fg": "#24292f", "muted": "#57606a", "grid": "#d0d7de",
              "accent": "#0969da", "warm": "#bc4c00", "good": "#1a7f37",
              "ground": "#8250df"},
    "dark": {"fg": "#e6edf3", "muted": "#9198a1", "grid": "#30363d",
             "accent": "#58a6ff", "warm": "#ff9b57", "good": "#3fb950",
             "ground": "#bc8cff"},
}

NSIDE, N_FREQ, N_TIME = 16, 8, 96
LMAX, N_PIX = 3 * NSIDE - 1, 12 * NSIDE**2
LAT_DEG, AZ_DEG, EL_DEG = 53.2, 0.0, 90.0
ETA, T_PHYS = 0.97, 293.0
T_RX, T_AMBIENT, T_HOT, T_GROUND, T_ATM = 290.0, 300.0, 400.0, 290.0, 3.0
DELTA_NU, T_INT = 25e6 / N_FREQ, 2.0


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


class MapSky(AbstractSkyModel):
    maps: jax.Array

    def __call__(self, freq: jax.Array) -> jax.Array:
        return self.maps


# ------------------------------------------------------------- the world ----
def compute():
    """Run the documented path and return everything the figures plot."""
    freq = jnp.linspace(60e6, 85e6, N_FREQ)
    theta = jnp.arccos(1.0 - 2.0 * (jnp.arange(N_PIX) + 0.5) / N_PIX)
    if RHINO_BEAMS is not None and RHINO_BEAMS.is_dir():
        raw = jnp.asarray(cst_beam_maps(RHINO_BEAMS, freq, nside=NSIDE))
        source = f"RHINO horn (CST export, {RHINO_BEAMS.name})"
    else:
        raw = jnp.stack([jnp.exp(-0.5 * (theta / 0.40) ** 2)] * N_FREQ)
        source = "Gaussian stand-in (CST export unavailable)"

    truncated, f_sky = horizon_truncated_beam(raw, el_deg=EL_DEG, apod_deg=3.0)
    projector = DriftScanProjector.from_beam_maps(
        jnp.asarray(truncated), lat_deg=LAT_DEG, az_deg=AZ_DEG, el_deg=EL_DEG,
        lmax=LMAX, normalize_beam=True,
    ).to_reference_frame(lst_ref_deg=0.0)

    structure = jax.random.normal(jax.random.key(0), (N_PIX,))
    sky_maps = jnp.stack([
        rcj.synchrotron_temperature(nu) * (1.0 + 0.15 * structure) for nu in freq
    ])
    switch = jnp.arange(N_TIME) % 3
    coords = Coordinates(
        time=jnp.arange(float(N_TIME)) * T_INT, freq=freq,
        extra={"lst_deg": 360.0 * jnp.arange(N_TIME) / N_TIME,
               "receiver_input": switch},
    )

    gamma_ant = rcj.cable_gamma(
        rcj.termination_gamma("open", N_FREQ), freq, length=2.0, loss=0.92
    )
    gamma_ambient = rcj.termination_gamma("resistive", N_FREQ, impedance=10.0)
    gamma_hot = rcj.cable_gamma(
        rcj.termination_gamma("short", N_FREQ), freq, length=0.4, loss=0.98
    )
    gamma_rec = rcj.termination_gamma("resistive", N_FREQ, impedance=45.0)
    gamma_src = jnp.stack([gamma_ant, gamma_ambient, gamma_hot])

    true_t = jnp.stack([
        250.0 + 20.0 * jnp.linspace(-1.0, 1.0, N_FREQ),
        30.0 * jnp.cos(jnp.linspace(0.0, 3.0, N_FREQ)),
        -40.0 + 8.0 * jnp.linspace(-1.0, 1.0, N_FREQ) ** 2,
    ])

    def twin(t_nw):
        return assemble(
            SkySourceOperator(sky_model=MapSky(sky_maps), projector=projector),
            BeamSpillOperator(sky_fraction=jnp.asarray(f_sky),
                              t_ground=jnp.array(T_GROUND)),
            AtmosphericEmissionOperator(t_atm=jnp.array(T_ATM)),
            AntennaLossOperator(efficiency=jnp.array(ETA),
                                t_physical=jnp.array(T_PHYS)),
            CalLoadOperator(t_load=jnp.array(T_AMBIENT)),
            CalLoadOperator(t_load=jnp.array(T_HOT)),
            NoiseWaveOperator(
                t_unc=t_nw[0], t_cos=t_nw[1], t_sin=t_nw[2], t_rx=jnp.array(T_RX),
                gamma_src_re=gamma_src.real, gamma_src_im=gamma_src.imag,
                gamma_rec_re=gamma_rec.real, gamma_rec_im=gamma_rec.imag,
            ),
        )

    state = State(coords=coords, meta={"telescope": "RHINO"})
    clean = eqx.filter_jit(twin(true_t))(state).data
    observed = rcj.add_radiometer_noise(clean, jax.random.key(1),
                                        t_int=T_INT, delta_nu=DELTA_NU)
    noise_std = observed / (DELTA_NU * T_INT) ** 0.5

    space = ParameterSpace(
        latents=[Latent("t_nw", init=jnp.full((3, N_FREQ), 100.0), linear=True)],
        bindings=[Bind("t_nw", into=(lambda p: p["noise_wave"].t_unc,
                                     lambda p: p["noise_wave"].t_cos,
                                     lambda p: p["noise_wave"].t_sin),
                       fn=lambda v: (v[0], v[1], v[2]))],
    )
    block = linear_operator(space, twin(jnp.zeros((3, N_FREQ))), state)
    solved, _ = wiener_solve(block, observed, noise_std=noise_std,
                             prior_std=100.0, tol=1e-10, maxiter=4000)

    # The cascade, per frequency: what the sky is worth at each stage.
    visible = np.asarray(projector.forward(sky_maps, coords).mean(axis=0))
    f = np.asarray(f_sky)
    spilled = f * visible + (1.0 - f) * T_GROUND
    lossy = ETA * spilled + (1.0 - ETA) * T_PHYS
    c_src = np.asarray(rcj.couplings(gamma_src, gamma_rec).c_src)

    return dict(
        source=np.array(source), freq=np.asarray(freq),
        theta=np.asarray(theta), beam=np.asarray(raw), f_sky=f,
        visible=visible, spilled=spilled, lossy=lossy,
        delivered=c_src[0] * lossy, c_src=c_src,
        observed=np.asarray(observed), switch=np.asarray(switch),
        truth=np.asarray(true_t), solved=np.asarray(solved),
    )


# ----------------------------------------------------------- the figures ----
def figure_horizon(d, theme):
    c = THEMES[theme]
    zenith = np.rad2deg(d["theta"])
    edges = np.arange(0.0, 181.0, 3.0)
    index = np.clip(np.digitize(zenith, edges) - 1, 0, edges.size - 2)
    centres = 0.5 * (edges[:-1] + edges[1:])

    def profile(beam):
        """Azimuthal mean per zenith ring, with the spread it hides.

        Plotting every pixel would draw the horn's 30-60% azimuthal structure
        as a vertical spray and read as noise; the band says the same thing and
        stays legible.
        """
        lo, mid, hi = [], [], []
        for b in range(centres.size):
            vals = beam[index == b]
            if vals.size == 0:
                for bucket in (lo, mid, hi):
                    bucket.append(np.nan)
                continue
            lo.append(np.percentile(vals, 10))
            mid.append(vals.mean())
            hi.append(np.percentile(vals, 90))
        return (10.0 * np.log10(np.maximum(np.array(x), 1e-6))
                for x in (lo, mid, hi))

    with plt.rc_context(styled(theme)):
        fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.2, 3.4),
                                     gridspec_kw={"width_ratios": [1.55, 1]})
        for i, nu in enumerate([0, N_FREQ - 1]):
            lo, mid, hi = profile(d["beam"][nu])
            ax.fill_between(centres, lo, hi, color=c["accent"], alpha=0.16 - 0.05 * i,
                            lw=0, zorder=2)
            ax.plot(centres, mid, "-" if i == 0 else "--", lw=1.6,
                    color=c["accent"], alpha=1.0 - 0.3 * i, zorder=3,
                    label=f"{d['freq'][nu] / 1e6:.0f} MHz")
        ax.axvspan(90.0, 180.0, color=c["ground"], alpha=0.13, lw=0, zorder=1)
        ax.axvline(90.0, color=c["ground"], lw=1.2, zorder=4)
        ax.text(136, -34, "below the horizon\n(sees ground)", ha="center",
                va="center", color=c["ground"], fontsize=8.5, zorder=5)
        ax.set_xlim(0, 180)
        ax.set_ylim(-40, 20)
        ax.set_xticks([0, 45, 90, 135, 180])
        ax.set_xlabel("zenith angle  [deg]")
        ax.set_ylabel("directivity  [dBi]")
        ax.set_title("The horn does not stop at the horizon")
        ax.legend(loc="lower left", fontsize=8, title="band: 10-90% in azimuth",
                  title_fontsize=7.5)
        ax.grid(True, zorder=0)

        bx.plot(d["freq"] / 1e6, 100.0 * (1.0 - d["f_sky"]), "o-", lw=1.6,
                ms=5, color=c["ground"])
        bx.set_xlabel("frequency  [MHz]")
        bx.set_ylabel(r"$1-f_{\rm sky}$   [% of solid angle]")
        bx.set_title("...and how much of it does not")
        bx.set_ylim(0, None)
        bx.grid(True)
        fig.tight_layout()
        save(fig, "receiver-horizon", theme)


def figure_cascade(d, theme):
    c = THEMES[theme]
    stages = ["collected\nby the beam", "after the\nhorizon split",
              "after the\nhorn's loss", "delivered to\nthe receiver"]
    values = [d["visible"].mean(), d["spilled"].mean(),
              d["lossy"].mean(), d["delivered"].mean()]
    notes = ["", "mixing, no loss:\nground replaces sky",
             "loss + the horn's\nown emission",
             "mismatch loss:\nnothing added"]
    colors = [c["accent"], c["ground"], c["warm"], c["good"]]
    ceiling = max(values) * 1.28
    with plt.rc_context(styled(theme)):
        fig, ax = plt.subplots(figsize=(8.6, 3.7))
        ax.grid(True, axis="y", zorder=0)
        ax.bar(range(4), values, color=colors, alpha=0.9, width=0.58, zorder=2)
        for i, (v, note) in enumerate(zip(values, notes, strict=True)):
            ax.text(i, v + 0.030 * ceiling, f"{v:,.0f} K", ha="center",
                    fontsize=10.5, fontweight="bold", color=c["fg"], zorder=3)
            if not note:
                continue
            # A note only fits inside a tall bar; under a short one it goes above.
            inside = v > 0.32 * ceiling
            ax.text(i, v / 2 if inside else v + 0.150 * ceiling, note,
                    ha="center", va="center", fontsize=8.5, zorder=3,
                    color="#ffffff" if inside else c["fg"])
        for i in range(3):
            ax.annotate("", xy=(i + 0.73, values[i + 1]),
                        xytext=(i + 0.30, values[i]),
                        arrowprops=dict(arrowstyle="-|>", color=c["muted"], lw=1.2))
        ax.set_xticks(range(4))
        ax.set_xticklabels(stages)
        ax.set_ylabel("sky temperature  [K]")
        ax.set_ylim(0, ceiling)
        ax.set_title("Three effects on the way in, none standing in for another")
        fig.tight_layout()
        save(fig, "receiver-cascade", theme)


def figure_recovery(d, theme):
    c = THEMES[theme]
    names = [r"$T_{\rm unc}$", r"$T_{\rm cos}$", r"$T_{\rm sin}$"]
    with plt.rc_context(styled(theme)):
        fig, axes = plt.subplots(2, 3, figsize=(9.2, 4.2), sharex=True,
                                 gridspec_kw={"height_ratios": [2.4, 1]})
        mhz = d["freq"] / 1e6
        for i in range(3):
            top, bot = axes[0, i], axes[1, i]
            top.plot(mhz, d["truth"][i], "-", lw=2.6, color=c["grid"],
                     label="truth", zorder=1)
            top.plot(mhz, d["solved"][i], "o", ms=4.5, color=c["accent"],
                     label="Wiener mean", zorder=2)
            top.set_title(names[i])
            top.grid(True)
            if i == 0:
                top.set_ylabel("temperature  [K]")
                top.legend(loc="best", fontsize=8)
            bot.axhline(0.0, color=c["grid"], lw=0.9)
            bot.plot(mhz, d["solved"][i] - d["truth"][i], "o-", ms=3.5, lw=1.1,
                     color=c["warm"])
            bot.set_xlabel("frequency  [MHz]")
            bot.set_ylim(-0.45, 0.45)
            bot.grid(True)
            if i == 0:
                bot.set_ylabel("residual  [K]")
        fig.suptitle(
            "The noise waves, recovered with the sky treated as known data",
            fontsize=10, fontweight="bold", color=c["fg"], y=1.03,
        )
        fig.tight_layout()
        save(fig, "receiver-recovery", theme)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replot", action="store_true",
                        help="re-draw from the cached run instead of recomputing")
    args = parser.parse_args()

    if args.replot and CACHE.exists():
        data = dict(np.load(CACHE))   # numpy refuses object arrays by default
        print(f"replotting from {CACHE.name}")
    else:
        print("running the documented path...")
        data = compute()
        np.savez_compressed(CACHE, **data)
    print(f"beam: {data['source']}")

    for theme in THEMES:
        figure_horizon(data, theme)
        figure_cascade(data, theme)
        figure_recovery(data, theme)


if __name__ == "__main__":
    main()
