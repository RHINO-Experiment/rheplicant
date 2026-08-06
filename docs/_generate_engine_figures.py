"""Generate the sky-engine figures shipped with the documentation.

Run MANUALLY (not at documentation build time) — the drift-scan engine needs
``limTOD[jax]`` plus ``healpy``/``pygdsm``, none of which the Read the Docs
environment installs. The outputs are committed under ``docs/_static``::

    uv run --frozen python docs/_generate_engine_figures.py           # full, ~10 min
    uv run --frozen python docs/_generate_engine_figures.py --smoke   # fast check

Every figure is written twice, ``-light`` and ``-dark``, and embedded in
``docs/sky-engines.md`` through furo's ``only-light`` / ``only-dark`` classes,
so the page reads correctly in either theme.

The physics: a zenith-pointing drift scan from the RHINO site (Jodrell Bank
latitude), a chromatic Gaussian beam, and the GSM16 sky. Over one sidereal day
the beam sweeps the whole ``dec = +53.2`` circle, so the data runs from the
bright Galactic-plane crossings (around LST 20h-4h) down to the cold minimum
near LST 12.5h, where the north Galactic pole passes overhead.
"""

import argparse
import dataclasses
import json
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import equinox as eqx  # noqa: E402
import healpy as hp  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402
import numpy as np  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from rheplicant import Coordinates  # noqa: E402
from rheplicant.radio.sky import DriftScanProjector, GeneralPointingProjector  # noqa: E402

STATIC = Path(__file__).parent / "_static"
CACHE = Path(__file__).parent / "_engine-figure-data.npz"  # git-ignored
LAT_DEG = 53.2  # Jodrell Bank — the RHINO site
AZ_DEG, EL_DEG = 0.0, 90.0  # zenith drift scan: traces the delta = LAT circle
FWHM_REF_DEG, FREQ_REF_MHZ = 12.0, 70.0  # chromatic beam: FWHM ~ lambda

# --------------------------------------------------------------- theming ---
THEMES = {
    "light": {"fg": "#24292f", "muted": "#57606a", "grid": "#d0d7de",
              "accent": "#0969da", "warm": "#bc4c00", "good": "#1a7f37"},
    "dark": {"fg": "#e6edf3", "muted": "#9198a1", "grid": "#30363d",
             "accent": "#58a6ff", "warm": "#ff9b57", "good": "#3fb950"},
}


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


def save(fig, name: str, theme: str, ext: str) -> None:
    out = STATIC / f"{name}-{theme}.{ext}"
    fig.savefig(out, bbox_inches="tight", transparent=True,
                dpi=200 if ext == "png" else None)
    plt.close(fig)
    print(f"  wrote {out.relative_to(STATIC.parent.parent)}")


# ------------------------------------------------------------ ingredients ---
def beam_alms(nside: int, lmax: int, freqs_mhz: np.ndarray) -> jnp.ndarray:
    """Chromatic Gaussian beam, beam-local alms — as numpy limTOD builds them.

    Normalized to unit pixel sum (limTOD's ``example_beam_map`` convention), so
    the un-normalized beam-weighted sum this projector returns is already a
    beam-weighted AVERAGE in kelvin rather than an arbitrary scale.
    """
    theta, _ = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    rows = []
    for f in freqs_mhz:
        fwhm = np.deg2rad(FWHM_REF_DEG * FREQ_REF_MHZ / f)
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        beam = np.exp(-0.5 * (theta / sigma) ** 2)
        rows.append(hp.map2alm(beam / beam.sum(), lmax=lmax))
    return jnp.asarray(np.array(rows))


def gsm_sky(nside: int, freqs_mhz: np.ndarray) -> tuple[jnp.ndarray, str]:
    try:
        from pygdsm import GlobalSkyModel16

        gsm = GlobalSkyModel16()
        maps = np.array([hp.ud_grade(gsm.generate(f), nside) for f in freqs_mhz])
        return jnp.asarray(maps), "GSM16"
    except Exception as exc:  # noqa: BLE001 — any pygdsm failure falls back
        print(f"  ! GSM unavailable ({exc!r}); using a synthetic sky")
        rng = np.random.default_rng(0)
        base = 200.0 * np.abs(rng.normal(size=hp.nside2npix(nside)))
        base = hp.smoothing(base, fwhm=np.deg2rad(10.0))
        spec = (freqs_mhz / FREQ_REF_MHZ) ** -2.6
        return jnp.asarray(spec[:, None] * base[None, :]), "synthetic"


def projectors(alms, nside, lmax, *, uniform: bool):
    common = dict(lat_deg=LAT_DEG, lmax=lmax, nside=nside)
    generic = GeneralPointingProjector(beam_alms=alms, **common)
    mmode = DriftScanProjector(beam_alms=alms, az_deg=AZ_DEG, el_deg=EL_DEG, **common)
    cached = mmode.to_reference_frame(lst_ref_deg=0.0)
    fast = dataclasses.replace(cached, uniform_sampling=True) if uniform else cached
    return generic, mmode, cached, fast


def coords_pair(lst_deg, freqs_hz):
    n_time = lst_deg.shape[0]
    drift = Coordinates(time=jnp.arange(float(n_time)), freq=freqs_hz,
                        extra={"lst_deg": lst_deg})
    generic = drift.replace(
        pointing=jnp.stack([jnp.full(n_time, AZ_DEG), jnp.full(n_time, EL_DEG)], -1),
        extra={"lst_deg": lst_deg, "selfrot_deg": jnp.zeros(n_time)},
    )
    return drift, generic


def best_of(fn, *args, repeats: int = 3) -> float:
    jax.block_until_ready(fn(*args))  # compile
    return min(
        (lambda t0: (jax.block_until_ready(fn(*args)), time.perf_counter() - t0)[1])(
            time.perf_counter()
        )
        for _ in range(repeats)
    )


# ---------------------------------------------------------------- figures ---
def figure_waterfall(lst, freqs_mhz, tod, sky_kind: str) -> None:
    """The product: one sidereal day of drift-scan data, 2-panel.

    The colour scale is logarithmic because the sky is a steep power law: on a
    linear scale the lowest channel saturates the map and the Galactic transit
    — the actual signal — is invisible everywhere else.
    """
    from matplotlib.colors import LogNorm

    lst_h = lst / 15.0
    for theme in THEMES:
        c = THEMES[theme]
        with plt.rc_context(styled(theme)):
            fig, (ax0, ax1) = plt.subplots(
                2, 1, figsize=(7.6, 5.6), sharex=True, height_ratios=[2.1, 1],
                constrained_layout=True,
            )
            im = ax0.pcolormesh(lst_h, freqs_mhz, tod.T, cmap="magma",
                                shading="auto",
                                norm=LogNorm(vmin=tod.min(), vmax=tod.max()))
            ax0.set_ylabel("frequency  [MHz]")
            ax0.set_title(f"One sidereal day of drift-scan data — {sky_kind} sky, "
                          f"zenith beam at latitude {LAT_DEG:g}°", loc="left")
            cb = fig.colorbar(im, ax=ax0, pad=0.015)
            cb.set_label("$T_{\\rm ant}$  [K]")
            cb.outline.set_edgecolor(c["grid"])
            cb.ax.tick_params(color=c["grid"])

            picks = [0, len(freqs_mhz) // 2, len(freqs_mhz) - 1]
            for i, idx in enumerate(picks):
                ax1.semilogy(lst_h, tod[:, idx], lw=1.5,
                             color=[c["accent"], c["good"], c["warm"]][i],
                             label=f"{freqs_mhz[idx]:.0f} MHz")
            ax1.set_xlabel("local sidereal time  [hours]")
            ax1.set_ylabel("$T_{\\rm ant}$  [K]")
            ax1.set_xlim(lst_h[0], lst_h[-1])
            ax1.grid(True, which="both")
            ax1.legend(ncol=3, loc="lower left", fontsize=8)
            ax1.set_title(f"Every channel sees the same sky drift past the "
                          f"$\\delta={LAT_DEG:g}°$ circle", loc="left", fontsize=9)
            save(fig, "engine-waterfall", theme, "png")


def figure_agreement(lst, freqs_mhz, tod_generic, tod_mmode) -> None:
    """Same physics, two engines: overlay + residual at float64 roundoff."""
    lst_h = lst / 15.0
    scale = np.max(np.abs(tod_generic))
    resid = np.abs(tod_mmode - tod_generic) / scale
    worst = float(resid.max())
    for theme in THEMES:
        c = THEMES[theme]
        with plt.rc_context(styled(theme)):
            fig, (ax0, ax1) = plt.subplots(
                2, 1, figsize=(7.6, 4.6), sharex=True, height_ratios=[2, 1],
                constrained_layout=True,
            )
            step = max(1, len(lst_h) // 26)  # sparse enough to read as markers
            for i in range(tod_generic.shape[1]):
                ax0.plot(lst_h, tod_generic[:, i], lw=1.6, color=c["muted"],
                         label="general-pointing engine" if i == 0 else None, zorder=1)
                ax0.plot(lst_h[::step], tod_mmode[::step, i], ls="none", marker="o",
                         ms=4.5, mfc="none", mew=1.3, color=c["accent"],
                         label="m-mode engine" if i == 0 else None, zorder=2)
            ax0.set_ylabel("$T_{\\rm ant}$  [K]")
            ax0.set_title("The m-mode engine is not an approximation", loc="left")
            ax0.grid(True)
            ax0.legend(loc="upper right")

            for i in range(resid.shape[1]):
                ax1.semilogy(lst_h, np.maximum(resid[:, i], 1e-18), lw=0.9,
                             color=c["accent"], alpha=0.75)
            # A legend entry, not floating text: the residual band spans the
            # whole panel width, so any in-plot label lands on top of data.
            ax1.axhline(2.22e-16, ls="--", lw=1.0, color=c["warm"],
                        label="float64 eps")
            ax1.legend(loc="upper right", fontsize=8)
            ax1.set_ylim(1e-18, 1e-11)
            ax1.set_ylabel("$|\\Delta| / \\max|T|$")
            ax1.set_xlabel("local sidereal time  [hours]")
            ax1.set_xlim(lst_h[0], lst_h[-1])
            ax1.grid(True)
            ax1.set_title(f"worst disagreement {worst:.1e} — float64 roundoff",
                          loc="left", fontsize=9)
            save(fig, "engine-agreement", theme, "svg")


def figure_scaling(bench: list[dict]) -> None:
    """Wall-clock vs band-limit: O(n_t·lmax³) against O(lmax³ + n_t·lmax)."""
    lmax = np.array([b["lmax"] for b in bench])
    gen = np.array([b["generic"] for b in bench]) * 1e3
    mm = np.array([b["mmode"] for b in bench]) * 1e3
    fast = np.array([b["fast"] for b in bench]) * 1e3
    for theme in THEMES:
        c = THEMES[theme]
        with plt.rc_context(styled(theme)):
            fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8.2, 3.5),
                                           constrained_layout=True)
            ax0.loglog(lmax, gen, "o-", color=c["warm"], lw=1.8, ms=5,
                       label="general pointing  $O(n_t\\,\\ell_{\\max}^3)$")
            ax0.loglog(lmax, mm, "s-", color=c["accent"], lw=1.8, ms=5,
                       label="m-mode  $O(\\ell_{\\max}^3 + n_t\\ell_{\\max})$")
            ax0.loglog(lmax, fast, "^--", color=c["good"], lw=1.6, ms=5,
                       label="m-mode + cached beam + FFT")
            ax0.set_xlabel("harmonic band-limit  $\\ell_{\\max}$")
            ax0.set_ylabel("one forward evaluation  [ms]")
            ax0.set_title("Cost of one sidereal day", loc="left")
            ax0.grid(True, which="both")
            ax0.legend(fontsize=8, loc="upper left")

            x = np.arange(len(lmax))
            ax1.bar(x - 0.19, gen / mm, width=0.36, color=c["accent"],
                    label="m-mode")
            ax1.bar(x + 0.19, gen / fast, width=0.36, color=c["good"],
                    label="+ cached beam + FFT")
            for xi, (a, b) in enumerate(zip(gen / mm, gen / fast, strict=True)):
                ax1.text(xi - 0.19, a, f"{a:.0f}×", ha="center", va="bottom",
                         fontsize=8, color=c["fg"])
                ax1.text(xi + 0.19, b, f"{b:.0f}×", ha="center", va="bottom",
                         fontsize=8, color=c["fg"])
            ax1.set_xticks(x, [str(v) for v in lmax])
            ax1.set_xlabel("harmonic band-limit  $\\ell_{\\max}$")
            ax1.set_ylabel("speed-up over the general engine")
            ax1.set_title("Same numbers, a fraction of the work", loc="left")
            ax1.margins(y=0.18)
            ax1.grid(True, axis="y")
            ax1.legend(fontsize=8, loc="upper left")
            save(fig, "engine-scaling", theme, "svg")


def figure_mmodes(mmodes: np.ndarray, freqs_mhz: np.ndarray) -> None:
    """|V_m| — what the drift scan actually measures."""
    m = np.arange(mmodes.shape[1])
    for theme in THEMES:
        c = THEMES[theme]
        with plt.rc_context(styled(theme)):
            fig, ax = plt.subplots(figsize=(7.0, 3.4), constrained_layout=True)
            colors = [c["accent"], c["good"], c["warm"]]
            picks = [0, mmodes.shape[0] // 2, mmodes.shape[0] - 1]
            for i, idx in enumerate(picks):
                amp = np.abs(mmodes[idx])
                ax.semilogy(m, np.maximum(amp / amp[0], 1e-12), lw=1.4,
                            color=colors[i], label=f"{freqs_mhz[idx]:.0f} MHz")
            ax.set_xlabel("m-mode index  $m$")
            ax.set_ylabel("$|\\tilde V_m| / |\\tilde V_0|$")
            ax.set_title("The drift scan measures a handful of m-modes",
                         loc="left")
            ax.set_xlim(0, m[-1])
            ax.grid(True, which="both")
            ax.legend(ncol=3)
            save(fig, "engine-mmodes", theme, "svg")


# ------------------------------------------------------------------- main ---
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="tiny configuration: check the plumbing, not the physics")
    ap.add_argument("--replot", action="store_true",
                    help="redraw from the cached run — no physics, no waiting. "
                         "Use for any change that is purely visual.")
    args = ap.parse_args()

    if args.replot:
        cache = np.load(CACHE)
        figure_waterfall(cache["lst"], cache["freqs_mhz"], cache["tod_mmode"],
                         str(cache["sky_kind"]))
        figure_agreement(cache["lst"], cache["freqs_shown"],
                         cache["tod_generic_shown"], cache["tod_mmode_shown"])
        figure_mmodes(cache["mmodes"], cache["freqs_mhz"])
        figure_scaling(json.loads(str(cache["bench"])))
        print("  redrawn from cache")
        return

    if args.smoke:
        nside, n_time, n_freq = 8, 64, 4
        bench_nsides = (4, 8)
    else:
        nside, n_time, n_freq = 64, 512, 32
        bench_nsides = (8, 16, 32, 64)

    lmax = 3 * nside - 1
    freqs_mhz = np.linspace(50.0, 100.0, n_freq)
    freqs_hz = jnp.asarray(freqs_mhz * 1e6)
    lst = jnp.linspace(0.0, 360.0, n_time, endpoint=False)
    drift_coords, generic_coords = coords_pair(lst, freqs_hz)

    print(f"main run: nside={nside} lmax={lmax} n_time={n_time} n_freq={n_freq}")
    alms = beam_alms(nside, lmax, freqs_mhz)
    sky, sky_kind = gsm_sky(nside, freqs_mhz)
    generic, mmode, cached, fast = projectors(alms, nside, lmax,
                                              uniform=2 * lmax < n_time)

    fwd_drift = eqx.filter_jit(lambda p, s: p.forward(s, drift_coords))
    fwd_generic = eqx.filter_jit(lambda p, s: p.forward(s, generic_coords))

    print("  running the m-mode engine ...")
    tod_mmode = np.asarray(fwd_drift(fast, sky))
    print("  running the general-pointing engine (the slow one) ...")
    tod_generic = np.asarray(fwd_generic(generic, sky))
    worst = np.max(np.abs(tod_mmode - tod_generic)) / np.max(np.abs(tod_generic))
    print(f"  agreement: {worst:.3e}")

    mmodes = np.asarray(eqx.filter_jit(lambda p, s: p.mmodes(s, drift_coords))(
        cached, sky))

    lst_np = np.asarray(lst)
    show = slice(None, None, max(1, n_freq // 3))  # 3-4 channels read clearly
    figure_waterfall(lst_np, freqs_mhz, tod_mmode, sky_kind)
    figure_agreement(lst_np, freqs_mhz[show], tod_generic[:, show],
                     tod_mmode[:, show])
    figure_mmodes(mmodes, freqs_mhz)

    # ---- scaling benchmark: one frequency, the band-limit is the variable ---
    bench = []
    for ns in bench_nsides:
        lm = 3 * ns - 1
        if 2 * lm >= n_time:
            print(f"  ! skipping nside={ns}: 2*lmax={2 * lm} >= n_time={n_time}")
            continue
        a1 = beam_alms(ns, lm, freqs_mhz[:1])
        s1, _ = gsm_sky(ns, freqs_mhz[:1])
        g1, m1, _, f1 = projectors(a1, ns, lm, uniform=True)
        fd = eqx.filter_jit(lambda p, s: p.forward(s, drift_coords))
        fg = eqx.filter_jit(lambda p, s: p.forward(s, generic_coords))
        row = {"lmax": lm, "nside": ns,
               "generic": best_of(fg, g1, s1),
               "mmode": best_of(fd, m1, s1),
               "fast": best_of(fd, f1, s1)}
        bench.append(row)
        print(f"  lmax={lm:4d}  generic {row['generic'] * 1e3:9.1f} ms   "
              f"m-mode {row['mmode'] * 1e3:7.2f} ms  "
              f"({row['generic'] / row['mmode']:.0f}x)   "
              f"fast {row['fast'] * 1e3:7.2f} ms "
              f"({row['generic'] / row['fast']:.0f}x)")
    figure_scaling(bench)

    (STATIC / "engine-benchmark.json").write_text(json.dumps(
        {"config": {"nside": nside, "lmax": lmax, "n_time": n_time,
                    "n_freq": n_freq, "sky": sky_kind, "lat_deg": LAT_DEG,
                    "az_deg": AZ_DEG, "el_deg": EL_DEG},
         "agreement": float(worst), "scaling": bench}, indent=2) + "\n")
    print("  wrote docs/_static/engine-benchmark.json")

    # Cache everything the figures need: a purely visual change should never
    # cost another full generic-engine run (68 s per evaluation at lmax=191).
    np.savez_compressed(
        CACHE, lst=lst_np, freqs_mhz=freqs_mhz, tod_mmode=tod_mmode,
        freqs_shown=freqs_mhz[show], tod_generic_shown=tod_generic[:, show],
        tod_mmode_shown=tod_mmode[:, show], mmodes=mmodes,
        sky_kind=sky_kind, bench=json.dumps(bench),
    )
    print(f"  wrote {CACHE.name} (redraw with --replot)")


if __name__ == "__main__":
    main()
